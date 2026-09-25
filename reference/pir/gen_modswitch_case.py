#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Modulus-switch (#11) test case, n=1024.  c'_i = round(q'·c_i/q) mod q'  (runs once on one ct).
Device path: CRT-combine c_i (validated device kernel) -> 56-bit C -> scale-round finalize.
Validated vs onionpir_ref.mod_switch.

Binary (LE): u32 MAGIC=0x4d535754('MSWT'), n, q1, q2, qprime
    c0q1,c0q2,c1q1,c1q2 : n u32   (input ct residues mod q1,q2)
    c0p, c1p            : n u32    (golden output mod q')
    python3 gen_modswitch_case.py
"""
import argparse, random, struct
import onionpir_ref as ref
MAGIC = 0x4D535754

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1024); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="modswitch_case.bin"); a = ap.parse_args()
    rng = random.Random(a.seed); pp = ref.Params(n=a.n); q, q1, q2 = pp.q, pp.q1, pp.q2
    qprime = ref.ntt_primes(1, a.n, bits=22)[0]
    c0 = [rng.randrange(q) for _ in range(a.n)]; c1 = [rng.randrange(q) for _ in range(a.n)]
    out = ref.mod_switch(pp, [c0, c1], qprime)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<5I", MAGIC, a.n, q1, q2, qprime))
        for d in (c0, c1):
            for qi in (q1, q2): f.write(struct.pack(f"<{a.n}I", *[x % qi for x in d]))
        for d in out: f.write(struct.pack(f"<{a.n}I", *d))
    print(f"wrote {a.out}: n={a.n} q1={q1} q2={q2} qprime={qprime}")

if __name__ == "__main__":
    main()
