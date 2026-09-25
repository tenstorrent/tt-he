#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Subs(ct, j) (#3) = Galois automorphism (#14) + BV key-switch (#10) — a WIRING of two done blocks.

Subs(ct,j) = keyswitch( automorph(ct0,j), automorph(ct1,j), galois_key_j ).  Since the automorph is a
host reindex (validated: test_permute) and the key-switch is the validated device pipeline
(test_bv_keyswitch), we emit a KEY-SWITCH case (ks_case format) whose inputs are ALREADY automorphed;
then `./build/test_bv_keyswitch subs_case.bin` computes keyswitch(c0j,c1j,gk) == Subs, validated vs
onionpir_ref.subs.  This exercises the composition end-to-end with a real Galois key.

    python3 gen_subs_case.py            # n=1024, ell=3, B=2^20, j=n/2+1 -> subs_case.bin
"""
import argparse, random
import onionpir_ref as ref
import gen_ks_case

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--ell", type=int, default=6)
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--jshift", type=int, default=2, help="j = n/jshift + 1 (jshift=1 -> n+1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="subs_case.bin")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pp = ref.Params(n=a.n, B=a.B, ell=a.ell, K=a.K)
    q = pp.q
    s = ref.keygen(pp, rng)
    j = a.n // a.jshift + 1
    gk = ref.gen_galois_key(pp, s, j, rng)                       # KSK: s(x^j) under s (ell rows [col0,col1])
    ct = ref.encrypt(pp, s, [rng.randrange(pp.t) for _ in range(a.n)], rng)
    c0j = ref.automorph_poly(ct[0], j, q)                        # the Galois-permuted components (host reindex)
    c1j = ref.automorph_poly(ct[1], j, q)
    d0, d1 = ref.subs(pp, ct, j, gk)                             # golden Subs == keyswitch(c0j,c1j,gk)
    digits = ref.gadget_decompose_signed(c1j, a.B, a.ell, q)
    case = {"n": a.n, "ell": a.ell, "B": a.B, "K": a.K, "primes": list(pp.primes),
            "c0": c0j, "c1": c1j, "ksk": gk, "d0": d0, "d1": d1, "digits": digits}
    gen_ks_case.write(a.out, case)
    print(f"wrote {a.out}: n={a.n} ell={a.ell} B={a.B} j={j} (Subs = automorph_{j} + BV key-switch)")

if __name__ == "__main__":
    main()
