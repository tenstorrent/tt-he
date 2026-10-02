# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""TT-faithful 32x32 modular matmul  C = (A @ B) mod q  via signed-int8 tile matmul.

This is the numpy oracle for the tt-metal kernel planned in
docs/matmul/history/PLAN.md, written for the hardware we actually target:

  * TT-metal exposes a SIGNED int8 tile matmul (32x32 int8 -> 32x32 int32).
    Signed int8 on Tensix is sign-magnitude, but every limb we produce is
    non-negative and <= 127, so it is identical in sign-magnitude, two's
    complement, or plain unsigned -> the signedness mismatch disappears.

  * To keep every limb in [0, 127] we decompose in RADIX 128 (7-bit limbs),
    NOT radix 256.  Since 128**4 == 2**28, a value < 2**28 splits into exactly
    4 int8 limbs.  Hence the constraint: q is a prime with q < 2**28.

It also reproduces the EXACT tile/lane layout the kernel uses, so passing here
means the hardware index mapping is correct, not merely the abstract algorithm.

Layout (see PLAN.md section 4):
    k = bk*8 + kappa     bk in 0..3, kappa in 0..7      (contraction)
    j = bj*8 + jsig      bj in 0..3, jsig  in 0..7       (output column)
    lane   m   = kappa*4 + p     in [0,32)               (p = limb of A)
    outcol     = jsig*4  + r     in [0,32)               (r = limb of B)
"""

import numpy as np

BASE = 128          # radix: 7-bit limbs -> each fits a non-negative int8
NLIMB = 4           # 128**4 == 2**28  -> 4 limbs cover any value < 2**28
TILE = 32
BLK = 8             # 8 logical values per 32-wide tile (8 * 4 limbs == 32)
NB = TILE // BLK    # = 4 block groups along K and J


# ---------------------------------------------------------------------------
# Little-endian radix-128 decomposition.  Result limbs are int8 in [0, 127].
# ---------------------------------------------------------------------------
def decompose_base128(x: np.ndarray, nlimb: int = NLIMB) -> np.ndarray:
    t = x.astype(np.int64)
    limbs = []
    for _ in range(nlimb):
        limbs.append((t % BASE).astype(np.int8))
        t //= BASE
    out = np.stack(limbs, axis=-1)          # [..., nlimb]
    assert (out >= 0).all() and (out <= 127).all(), "limb outside non-negative int8 range"
    return out


# ---------------------------------------------------------------------------
# Offline precompute of the parameter B:
#   Bp[k,p,j] = (128**p * B[k,j]) mod q          (< q < 2**28)
#   bp[k,p,j,r] = limb_r(Bp[k,p,j])              (int8 in [0,127])
# ---------------------------------------------------------------------------
def precompute_B(B: np.ndarray, q: int) -> np.ndarray:
    K, J = B.shape
    weights_p = (BASE ** np.arange(NLIMB, dtype=np.int64))          # [NLIMB]
    Bp = (B.astype(np.int64)[:, None, :] * weights_p[None, :, None]) % q   # [K,p,J]
    bp = decompose_base128(Bp)                                       # [K,p,J,r]
    return bp


# ---------------------------------------------------------------------------
# Online decomposition of the dynamic operand A:
#   a[i,k,p] = limb_p(A[i,k])                     (int8 in [0,127])
# ---------------------------------------------------------------------------
def decompose_A(A: np.ndarray) -> np.ndarray:
    return decompose_base128(A)                                      # [I,K,p]


# ---------------------------------------------------------------------------
# The "hardware" op: 32x32 signed-int8 matmul accumulating into int32.
# Inputs are non-negative (<=127) so this faithfully models the Tensix FPU.
# ---------------------------------------------------------------------------
def hw_int8_matmul(L: np.ndarray, R: np.ndarray) -> np.ndarray:
    assert L.dtype == np.int8 and R.dtype == np.int8
    assert L.shape == (TILE, TILE) and R.shape == (TILE, TILE)
    return L.astype(np.int32) @ R.astype(np.int32)                  # 32x32 int32


def modular_matmul_via_tt_int8(A: np.ndarray, B: np.ndarray, q: int) -> np.ndarray:
    """C = (A @ B) mod q for 32x32 inputs, through the int8 tile kernel only."""
    assert A.shape == (TILE, TILE) and B.shape == (TILE, TILE)
    assert q < 2**28, "q must be < 2**28 so values split into 4 base-128 limbs"

    a = decompose_A(A)            # [32, 32, 4]  a[i,k,p]
    bp = precompute_B(B, q)       # [32, 4, 32, 4]  bp[k,p,j,r]

    # ---- build the 4 LHS byte-tiles  L_bk[i, kappa*4+p] = a[i, bk*8+kappa, p]
    # a[:, bk*8:bk*8+8, :] is [32, 8(kappa), 4(p)] -> reshape [32,32]: col = kappa*4+p
    L = [a[:, bk * BLK:(bk + 1) * BLK, :].reshape(TILE, TILE).astype(np.int8)
         for bk in range(NB)]

    C = np.zeros((TILE, TILE), dtype=np.int64)
    weights_r = (BASE ** np.arange(NLIMB, dtype=np.int64))           # [4]

    for bj in range(NB):
        # ---- accumulate the 4 contraction blocks into one int32 product tile
        P = np.zeros((TILE, TILE), dtype=np.int64)
        for bk in range(NB):
            # R[kappa*4+p, jsig*4+r] = bp[bk*8+kappa, p, bj*8+jsig, r]
            sub = bp[bk * BLK:(bk + 1) * BLK, :, bj * BLK:(bj + 1) * BLK, :]  # [8,4,8,4]
            R = sub.reshape(TILE, TILE).astype(np.int8)              # row=kappa*4+p, col=jsig*4+r
            P += hw_int8_matmul(L[bk], R)
        assert P.max() < 2**31, "int32 accumulator overflow"

        # ---- recombine the r limbs:  C[i, bj*8+jsig] = sum_r 128**r * P[i, jsig*4+r]
        Pr = P.reshape(TILE, BLK, NLIMB)                             # [i, jsig, r]
        block = (Pr * weights_r[None, None, :]).sum(axis=-1) % q     # [32, 8]
        C[:, bj * BLK:(bj + 1) * BLK] = block

    return C.astype(np.uint64)


def reference_modular_matmul(A: np.ndarray, B: np.ndarray, q: int) -> np.ndarray:
    """Ground truth via exact Python big-int arithmetic."""
    C = A.astype(object) @ B.astype(object)
    return (C % q).astype(np.uint64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _test(q, seed):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, q, size=(TILE, TILE), dtype=np.uint64).astype(np.uint64)
    B = rng.integers(0, q, size=(TILE, TILE), dtype=np.uint64).astype(np.uint64)

    got = modular_matmul_via_tt_int8(A.astype(np.int64), B.astype(np.int64), q)
    exp = reference_modular_matmul(A, B, q)

    ok = np.array_equal(got, exp)
    print(f"[{'PASS' if ok else 'FAIL'}] 32x32  q={q:>11d} (~2^{q.bit_length()-1})  seed={seed}")
    if not ok:
        bad = np.argwhere(got != exp)[:5]
        for i, j in bad:
            print(f"   ({i},{j}) got={got[i, j]} expected={exp[i, j]}")
    return ok


if __name__ == "__main__":
    cases = [
        (12289,     0),   # NTT prime, 14-bit
        (40961,     1),   # 16-bit
        (786433,    2),   # 20-bit
        (134217689, 3),   # 27-bit prime
        (268435399, 4),   # 2^28 - 57, the largest practical 28-bit prime
        (268435399, 5),
        (167772161, 6),   # 5*2^25 + 1, NTT-friendly 28-bit prime
    ]
    all_ok = all(_test(q, s) for q, s in cases)
    print("\nALL PASS" if all_ok else "\nSOME FAILED")
