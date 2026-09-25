#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Galois-automorphism (#14) + monomial-multiply (#15) golden cases for the device host reindex.

Both are coefficient permutations with a negacyclic sign flip on wrap (x^n = -1):
    automorph σ_j:  out[(i·j) mod 2n < n ? i·j : i·j-n] = ± in[i]
    monomial x^k :  out[(i+k) mod 2n < n ? i+k : i+k-n] = ± in[i]
On TT these are host-side reindex (like the NTT output reindex), applied during the inter-stage
transfers — no SFPU kernel. This dumps a golden so the C++ host reindex is checked bit-exact vs the
validated CPU reference (onionpir_ref.automorph_poly / mono_mul_poly).

Binary (little-endian):
    u32 MAGIC=0x504d5554('PMUT'), u32 n, u32 q, u32 nj, u32 nk
    in       : n u32
    per j:   u32 j, then n u32  (automorph σ_j of in)
    per k:   i32 k, then n u32  (monomial x^k of in)

    python3 gen_permute_case.py            # n=2048, q=268369921 -> permute_case.bin
"""
import argparse, random, struct
import onionpir_ref as ref

MAGIC = 0x504D5554

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--q", type=int, default=268369921)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="permute_case.bin")
    a = ap.parse_args()
    n, q = a.n, a.q
    rng = random.Random(a.seed)
    inp = [rng.randrange(q) for _ in range(n)]
    # expansion Galois elements j = n/2^d + 1 (d = 0..3), all odd/coprime to 2n
    js = [ (n >> d) + 1 for d in range(4) ]
    ks = [1, 5, -3, -(n // 2)]                       # a few monomial shifts (incl. negative)

    with open(a.out, "wb") as f:
        f.write(struct.pack("<5I", MAGIC, n, q, len(js), len(ks)))
        f.write(struct.pack(f"<{n}I", *inp))
        for j in js:
            g = ref.automorph_poly(inp, j, q)
            f.write(struct.pack("<I", j)); f.write(struct.pack(f"<{n}I", *g))
        for k in ks:
            g = ref.mono_mul_poly(inp, k, q)
            f.write(struct.pack("<i", k)); f.write(struct.pack(f"<{n}I", *g))
    print(f"wrote {a.out}: n={n} q={q} js={js} ks={ks}")

if __name__ == "__main__":
    main()
