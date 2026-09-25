#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""External-product MAC-core test case for the device (#16 back-half).

Validates the risky integration: CRT-combine -> gadget -> digit_to_rns -> NTT(digit) -> pointwise
MAC vs RGSW(NTT) -> INTT -> out.  The RGSW rows are NTT'd ON-DEVICE (so no Python/device NTT-convention
matching); this file only provides coefficient-domain inputs and the coefficient-domain golden output.

Operation tested (the external product's arithmetic):  given 2ℓ signed gadget digits and a 2ℓ×2 RGSW
matrix C (coeff domain),
    out_col = Σ_r  digit_r ⊛ C[r][col]   (negacyclic, mod q),  col ∈ {0,1}
per RNS channel (mod q_j, j=0..K-1).

K-parameterized (K = RNS limb count): K=2 (~56-bit q), K=3 (~84-bit q).

Binary (VERSION 2, little-endian):
    u32 MAGIC=0x45585052('EXPR'), u32 VERSION=2, u32 n, u32 twoL(=2ℓ), u32 B, u32 K
    K * u32   primes   (q = Π primes)
    d0res : K * n u32         (d0 mod q_j)
    d1res : K * n u32         (d1 mod q_j)
    Cres  : K * (twoL*2) * n u32   (C[r][col] mod q_j, channel-major then r then col)
    out   : 2(cols) * K * n u32    (out_col mod q_j)
    digits: twoL * n i32      (signed, [-B/2,B/2); host sanity check)

    python3 gen_extprod_case.py --n 2048 --ell 5 --B 4096            # K=2 default
    python3 gen_extprod_case.py --K 3 --n 2048 --ell 8 --B 2048 --out extprod_k3.bin
"""
import argparse, random, struct
import onionpir_ref as ref

MAGIC = 0x45585052
VERSION = 2

def gen(n, ell, B, K, seed):
    rng = random.Random(seed)
    pp = ref.Params(n=n, B=B, ell=ell, K=K)
    q, primes = pp.q, pp.primes
    twoL = 2 * ell
    # a real BFV ct d=(d0,d1) (coeff, mod q); the RGSW rows C[r][col] (coeff, mod q)
    d0 = [rng.randrange(q) for _ in range(n)]
    d1 = [rng.randrange(q) for _ in range(n)]
    C = [[[rng.randrange(q) for _ in range(n)] for _ in range(2)] for _ in range(twoL)]
    # digits = G^{-1}(d0) ++ G^{-1}(d1)  (the canonical balanced gadget decomposition)
    digits = ref.gadget_decompose_signed(d0, B, ell, q) + ref.gadget_decompose_signed(d1, B, ell, q)
    # golden: out_col = Σ_r digit_r ⊛ C[r][col] mod q  (== external_product operation)
    out = [[0] * n for _ in range(2)]
    for col in range(2):
        acc = [0] * n
        for r in range(twoL):
            dr = [dv % q for dv in digits[r]]
            acc = ref.poly_add(acc, ref.negacyclic_mul(dr, C[r][col], q), q)
        out[col] = acc
    return {"n": n, "ell": ell, "twoL": twoL, "B": B, "K": K, "q": q,
            "primes": list(primes), "d0": d0, "d1": d1, "digits": digits, "C": C, "out": out}

def write(path, c):
    n, twoL, K = c["n"], c["twoL"], c["K"]
    primes = c["primes"]
    with open(path, "wb") as f:
        f.write(struct.pack("<6I", MAGIC, VERSION, n, twoL, c["B"], K))
        f.write(struct.pack(f"<{K}I", *primes))
        for d in (c["d0"], c["d1"]):
            for qi in primes:
                f.write(struct.pack(f"<{n}I", *[x % qi for x in d]))
        for qi in primes:
            for r in range(twoL):
                for col in range(2):
                    f.write(struct.pack(f"<{n}I", *[x % qi for x in c["C"][r][col]]))
        for col in range(2):
            for qi in primes:
                f.write(struct.pack(f"<{n}I", *[x % qi for x in c["out"][col]]))
        for r in range(twoL):
            f.write(struct.pack(f"<{n}i", *c["digits"][r]))   # digits last (host sanity check)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--ell", type=int, default=5)
    ap.add_argument("--B", type=int, default=4096)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="extprod_case.bin")
    a = ap.parse_args()
    c = gen(a.n, a.ell, a.B, a.K, a.seed)
    write(a.out, c)
    print(f"wrote {a.out}: n={c['n']} ell={c['ell']} twoL={c['twoL']} B={c['B']} K={c['K']} "
          f"primes={c['primes']} q=~{c['q'].bit_length()}b")

if __name__ == "__main__":
    main()
