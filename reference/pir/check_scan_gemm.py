#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Validate the batched-GEMM formulation of the PIR first dimension, and the int8 radix-128 limb
scheme it lowers to.  Host-only (no device) -- this settles the formulation before any kernel work.

WHY THIS EXISTS
The first dimension in the eval domain is
    acc[r][c][i] = sum_k  DB[k*rest + r][i] * sel[k][c][i]   (mod q)
with the coefficient slot i appearing on BOTH operands.  Forcing i onto the two free axes of a single
GEMM produces every slot-PAIR and leaves only the diagonal useful (32x waste).  Putting i on the BATCH
axis instead gives, per slot i and RNS channel:
    C_i (rest x ncomp) = A_i (rest x N0) @ B_i (N0 x ncomp)
i.e. a batch of n GEMMs, both operands slot-local, no diagonal.  Check 1 confirms that bookkeeping.

Check 2 is the one that can bite silently at scale: lowering a 28-bit modular MAC onto the int8 matrix
engine via radix-128 limbs accumulates in int32.  With a 512-wide contraction, does it overflow, and is
the radix recombine exact?  A wrong answer here is a wrong DB entry, not a crash.

    python3 check_scan_gemm.py [--N0 512] [--rest 128] [--slots 8] [--trials 3]
"""
import argparse, random

RADIX_BITS = 7                      # radix-128 int8 limbs, as kernels/shared/compute/fused_matmul_a.cpp
NLIMB = 4
POW = [1 << (RADIX_BITS * p) for p in range(NLIMB)]      # 1, 128, 16384, 2097152
INT32_MAX = (1 << 31) - 1

# a 28-bit NTT-friendly prime, as used by the engine (q = 268369921 = 2^28 - ...)
Q = 268369921


def limb(v, p):
    return (v >> (RADIX_BITS * p)) & 0x7F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N0", type=int, default=512, help="first-dimension width (contraction)")
    ap.add_argument("--rest", type=int, default=128, help="output cells (GEMM m)")
    ap.add_argument("--slots", type=int, default=8, help="coefficient slots to check (GEMM batch)")
    ap.add_argument("--ncomp", type=int, default=2, help="ciphertext components (GEMM n)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    q = Q

    print(f"batched-GEMM check: q={q} ({q.bit_length()}b)  N0={a.N0} rest={a.rest} "
          f"ncomp={a.ncomp} slots={a.slots}")

    bad1 = bad2 = 0
    worst_acc = 0
    for t in range(a.trials):
        # NON-CONSTANT plaintexts: each cell/slot independent. This is the 128 MB regime -- the
        # constant-poly case (PIR_DEMO) is what let a shared query operand look correct.
        DB = [[rng.randrange(q) for _ in range(a.slots)] for _ in range(a.N0 * a.rest)]
        SEL = [[[rng.randrange(q) for _ in range(a.slots)] for _ in range(a.ncomp)] for _ in range(a.N0)]

        # ---- reference: per-slot dot product, exact modular arithmetic ----
        ref = [[[0] * a.slots for _ in range(a.ncomp)] for _ in range(a.rest)]
        for r in range(a.rest):
            for c in range(a.ncomp):
                for i in range(a.slots):
                    s = 0
                    for k in range(a.N0):
                        s += DB[k * a.rest + r][i] * SEL[k][c][i]
                    ref[r][c][i] = s % q

        # ---- check 1: batched GEMM (slot on the BATCH axis) ----
        for i in range(a.slots):
            A = [[DB[k * a.rest + r][i] for k in range(a.N0)] for r in range(a.rest)]   # rest x N0
            B = [[SEL[k][c][i] for c in range(a.ncomp)] for k in range(a.N0)]           # N0  x ncomp
            for r in range(a.rest):
                for c in range(a.ncomp):
                    got = sum(A[r][k] * B[k][c] for k in range(a.N0)) % q
                    if got != ref[r][c][i]:
                        bad1 += 1

        # ---- check 2: int8 radix-128 lowering, int32 accumulate, radix recombine ----
        # The modular product decomposes as
        #   x*y mod q == sum_{p,s} 128^(p+s) * limb(x,p)*limb(y,s)  (mod q)
        # so each (p,s) plane accumulates the contraction over k in int32, and the planes recombine by
        # their radix weight. This is the arithmetic fused_matmul_a performs (it folds the radix powers
        # into one operand so 4 planes suffice rather than 16 pairwise products; the accumulator bound
        # measured here is the same either way, since it is set by limb*limb summed over k).
        for i in range(a.slots):
            for r in range(min(a.rest, 4)):          # a few rows is enough; cost is O(rest*N0*16)
                for c in range(a.ncomp):
                    plane = {}
                    for p in range(NLIMB):
                        for s in range(NLIMB):
                            acc = 0
                            for k in range(a.N0):
                                acc += limb(DB[k * a.rest + r][i], p) * limb(SEL[k][c][i], s)
                            plane[(p, s)] = acc
                            worst_acc = max(worst_acc, acc)
                    tot = 0
                    for (p, s), v in plane.items():
                        tot += (POW[p] * POW[s] % q) * v
                    if tot % q != ref[r][c][i]:
                        bad2 += 1

    # ---- check 4: 32x32 TILED form -- pins the index bookkeeping the kernel will implement ----
    # C[mt][nt] = sum_kt  A[mt][kt] @ B[kt][nt], each block 32x32, accumulating mod q across kt.
    # (Accumulating the per-kt mod-q results is exact, since mod-q addition is exact -- so a kernel may
    # either accumulate int32 planes before one recombine, or recombine per kt and add mod q.)
    T = 32
    bad4 = 0
    rng4 = random.Random(a.seed + 1)
    m, kk, ncomp, slot = a.rest, a.N0, a.ncomp, 0
    if m % T == 0 and kk % T == 0:
        DBx = [[rng4.randrange(q) for _ in range(1)] for _ in range(kk * m)]
        SELx = [[[rng4.randrange(q)] for _ in range(ncomp)] for _ in range(kk)]
        refx = [[0] * ncomp for _ in range(m)]
        for r in range(m):
            for c in range(ncomp):
                refx[r][c] = sum(DBx[k * m + r][0] * SELx[k][c][0] for k in range(kk)) % q
        A = [[DBx[k * m + r][0] for k in range(kk)] for r in range(m)]      # m x k
        B = [[SELx[k][c][0] for c in range(ncomp)] for k in range(kk)]      # k x n (n padded to T)
        for mt in range(m // T):
            for nt in range(1):                                            # ncomp <= T -> one n-tile
                acc = [[0] * T for _ in range(T)]
                for kt in range(kk // T):
                    for i in range(T):
                        for j in range(ncomp):
                            s = 0
                            for l in range(T):
                                s += A[mt*T + i][kt*T + l] * B[kt*T + l][j]
                            acc[i][j] = (acc[i][j] + s) % q
                for i in range(T):
                    for j in range(ncomp):
                        if acc[i][j] != refx[mt*T + i][j]:
                            bad4 += 1
    print(f"  [4] 32x32 tiled GEMM (m/k tiled, n padded) == reference       : mismatches={bad4}")

    print(f"  [1] batched GEMM (slot on batch axis) == per-slot dot product : mismatches={bad1}")
    print(f"  [2] int8 radix-128 limbs + int32 accumulate + recombine       : mismatches={bad2}")
    print(f"  [3] largest int32 plane accumulator observed = {worst_acc:,}  "
          f"({100.0*worst_acc/INT32_MAX:.4f}% of int32 max)")
    hdr = 127 * 127 * a.N0
    print(f"      theoretical worst case for N0={a.N0}: 127*127*N0 = {hdr:,} "
          f"({100.0*hdr/INT32_MAX:.4f}% of int32 max) -> "
          f"{'SAFE' if hdr <= INT32_MAX else 'OVERFLOW'}")
    n0_max = INT32_MAX // (127 * 127)
    print(f"      int32 headroom: contraction width is safe up to N0 = {n0_max:,}")
    ok = (bad1 == 0 and bad2 == 0 and hdr <= INT32_MAX)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
