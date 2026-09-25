#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Emit a device test case (binary) for the gadget_decompose_signed kernel (#9) and the
CRT-combine bridge (#13) — validated against the CPU golden in onionpir_ref.py.

Mirrors reference/he/gen_ctmul_case.py: the golden is computed in exact big-int here, dumped
little-endian, and the device kernel is checked bit-exact against it.

K-parameterized (K = RNS limb count): K=2 (~56-bit q), K=3 (~84-bit q). One coefficient's data
flow on device (plan §5):
    (x mod q_0, .. x mod q_{K-1}) --CRT-combine (#13)--> X (28K-bit, radix-2^28 limbs) --#9--> digits
This case file carries all three representations so #9 (gadget) or #13 (CRT) can be tested
standalone, or the #13->#9 pipeline together:

  * RNS residues  (x mod q_j)               -> input to CRT-combine (#13)
  * radix-2^28    (X >> 28j) & (2^28-1)      -> input to gadget_decompose (#9)
  * digits        (ell signed int32 polys, Σ_k digit_k·B^k ≡ X mod q)  -> golden output

Binary layout (VERSION 2, all little-endian; coefficients in NATURAL order — host tilizes):
    u32 MAGIC=0x47414447('GADG'), u32 VERSION=2
    u32 n, u32 ell, u32 nvec, u32 B, u32 K, u32 LOG2B
    K * u32   primes   (q = Π primes; consumer computes)
    then nvec blocks, each:
        K   * (n u32)   residues  (x mod q_j,   j=0..K-1)
        K   * (n u32)   limbs     ((X>>28j)&M,  j=0..K-1)
        ell * (n i32)   digits    (digit k contiguous; two's-complement in u32)

Usage:
    python3 gen_gadget_case.py                       # default n=2048,B=256,ell=8,K=2 -> gadget_k2_l8.bin
    python3 gen_gadget_case.py --K 3 --n 2048 --B 2048 --ell 8 --out gadget_k3.bin
    python3 gen_gadget_case.py --verify gadget_k2_l8.bin   # read back + reconstruct-check
"""

import argparse
import random
import struct
import sys

import onionpir_ref as ref

MAGIC = 0x47414447
VERSION = 2
MASK28 = (1 << 28) - 1


def _log2_exact(x):
    b = x.bit_length() - 1
    assert (1 << b) == x, f"B={x} not a power of two"
    return b


def edge_values(q, B, ell, n, rng):
    """A poly of interesting coefficients: 0, q-1, base boundaries, sign-flip cases, random."""
    vals = [0, 1, q - 1, q // 2, B - 1, B, B // 2, B // 2 - 1]
    # values whose top digit(s) go negative (coeff just under a B^k boundary)
    for k in range(1, ell):
        vals.append((pow(B, k) - 1) % q)
        vals.append((pow(B, k) + B // 2) % q)
    while len(vals) < n:
        vals.append(rng.randrange(q))
    return [v % q for v in vals[:n]]


def gen_case(n, B, ell, nvec, K, seed):
    rng = random.Random(seed)
    pp = ref.Params(n=n, B=B, ell=ell, K=K)
    q, primes = pp.q, pp.primes
    g = ref.gadget_vector(B, ell, q)

    vecs = []
    for v in range(nvec):
        x = edge_values(q, B, ell, n, rng) if v == 0 else ref.rand_poly(n, q, rng)
        digits = ref.gadget_decompose_signed(x, B, ell, q)         # ell polys, [-B/2, B/2)
        # self-consistency: reconstruction and range
        recon = [sum(digits[k][i] * g[k] for k in range(ell)) % q for i in range(n)]
        assert recon == x, f"vec {v}: reconstruction mismatch"
        assert all(-(B // 2) <= digits[k][i] < B // 2 for k in range(ell) for i in range(n)), \
            f"vec {v}: digit out of [-B/2, B/2)"
        vecs.append({
            "res":   [[xi % p for xi in x] for p in primes],
            "limbs": [[(xi >> (28 * j)) & MASK28 for xi in x] for j in range(K)],
            "digits": digits,
        })
    return {"n": n, "ell": ell, "nvec": nvec, "B": B, "K": K,
            "primes": list(primes), "q": q, "vecs": vecs}


def write_case(path, case):
    K, n, ell = case["K"], case["n"], case["ell"]
    with open(path, "wb") as f:
        f.write(struct.pack("<8I", MAGIC, VERSION, n, ell, case["nvec"], case["B"],
                            K, _log2_exact(case["B"])))
        f.write(struct.pack(f"<{K}I", *case["primes"]))
        for vec in case["vecs"]:
            for j in range(K):
                f.write(struct.pack(f"<{n}I", *vec["res"][j]))
            for j in range(K):
                f.write(struct.pack(f"<{n}I", *vec["limbs"][j]))
            for k in range(ell):
                f.write(struct.pack(f"<{n}i", *vec["digits"][k]))   # signed int32


def read_case(path):
    with open(path, "rb") as f:
        magic, version, n, ell, nvec, B, K, log2b = struct.unpack("<8I", f.read(32))
        assert magic == MAGIC and version == VERSION, f"bad magic/version ({magic:x}/{version})"
        primes = list(struct.unpack(f"<{K}I", f.read(4 * K)))
        q = 1
        for p in primes:
            q *= p
        vecs = []
        for _ in range(nvec):
            vec = {"res": [], "limbs": []}
            for _ in range(K):
                vec["res"].append(list(struct.unpack(f"<{n}I", f.read(4 * n))))
            for _ in range(K):
                vec["limbs"].append(list(struct.unpack(f"<{n}I", f.read(4 * n))))
            vec["digits"] = [list(struct.unpack(f"<{n}i", f.read(4 * n))) for _ in range(ell)]
            vecs.append(vec)
    return {"n": n, "ell": ell, "nvec": nvec, "B": B, "K": K,
            "primes": primes, "q": q, "vecs": vecs}


def verify(path):
    case = read_case(path)
    n, ell, B, K, q, primes = (case["n"], case["ell"], case["B"], case["K"],
                               case["q"], case["primes"])
    g = ref.gadget_vector(B, ell, q)
    ok = True
    for v, vec in enumerate(case["vecs"]):
        for i in range(n):
            X = sum(vec["limbs"][j][i] << (28 * j) for j in range(K))
            # radix-2^28 limbs reconstruct the 28K-bit value, consistent with RNS residues
            if any(X % primes[j] != vec["res"][j][i] for j in range(K)):
                ok = False; print(f"  vec{v} i{i}: RNS/limb mismatch"); break
            recon = sum(vec["digits"][k][i] * g[k] for k in range(ell)) % q
            if recon != X:
                ok = False; print(f"  vec{v} i{i}: digit reconstruction mismatch"); break
            if not all(-(B // 2) <= vec["digits"][k][i] < B // 2 for k in range(ell)):
                ok = False; print(f"  vec{v} i{i}: digit range"); break
    print(f"verify {path}: n={n} ell={ell} nvec={case['nvec']} B={B} K={K} "
          f"q=~{q.bit_length()}b -> {'OK' if ok else 'FAILED'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="gadget_decompose_signed device test-case generator")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--ell", type=int, default=8)
    ap.add_argument("--nvec", type=int, default=8)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gadget_k2_l8.bin")
    ap.add_argument("--verify", metavar="FILE", help="read back a case file and re-check it")
    args = ap.parse_args()

    if args.verify:
        sys.exit(0 if verify(args.verify) else 1)

    case = gen_case(args.n, args.B, args.ell, args.nvec, args.K, args.seed)
    write_case(args.out, case)
    print(f"wrote {args.out}: n={case['n']} ell={case['ell']} nvec={case['nvec']} "
          f"B={case['B']} K={case['K']} primes={case['primes']} "
          f"q=~{case['q'].bit_length()}b")
    verify(args.out)


if __name__ == "__main__":
    main()
