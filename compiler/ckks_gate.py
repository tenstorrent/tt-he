# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ckks_gate.py — the CKKS front-end gate. Two independent checks, both exact.

  1. DECRYPT. Build the CKKS ct_mul as an L1 program, run it on the reference interpreter, CRT the
     output limbs back, decrypt, divide out the scale -> must equal the EXACT negacyclic integer
     product m1*m2. Not "close in 40 bits" -- equal.
  2. MEANING-PRESERVING LOWERING. run(L1) == run(L2) == run(L3), bit-identical, so the tile lowering
     and the int8 lowering are proven refinements of the same function.

Run:  python3 ckks_gate.py
"""
from __future__ import annotations
import random
import sys
from math import prod

import _paths                                     # noqa: F401  (repo layout)
import ir
import reference as ref
from builders import build_ct_mul_ckks
from lower import lower_l1_to_l2, lower_l2_to_l3
from ckks_mul_ref import CKKSParams, encrypt, decrypt, crt_poly, _exact_product
from bfv_relin_dnum import keygen_relin_dnum
from bfv_relin_ref import small_poly


def gate(N=32, LQ=4, LP=2, dnum=2, ndrop=1, seed=0, mbound=3,
         factors=None, T=32, do_l3=True, verbose=True):
    pp = CKKSParams(N=N, LQ=LQ, LP=LP, ndrop=ndrop)
    rng = random.Random(seed)
    s = small_poly(N, rng)
    evks, groups = keygen_relin_dnum(pp, s, rng, dnum)
    m1 = [rng.randrange(-mbound, mbound + 1) for _ in range(N)]
    m2 = [rng.randrange(-mbound, mbound + 1) for _ in range(N)]
    ct1, ct2 = encrypt(pp, m1, s, rng), encrypt(pp, m2, s, rng)

    l1, inputs, o0, o1, base = build_ct_mul_ckks(pp, ct1, ct2, (evks, groups), ndrop=ndrop)
    r1 = ir.run(l1, inputs)

    # --- check 1: decrypt ---
    res0 = [[int(v) for v in r1[n]] for n in o0]
    res1 = [[int(v) for v in r1[n]] for n in o1]
    ctout = (crt_poly(res0, base, N), crt_poly(res1, base, N))
    got = decrypt(pp, ctout, s, Qprod=prod(base), scale=(pp.Delta ** 2, prod(pp.Q[len(base):])))
    want = _exact_product(m1, m2, N)
    dec_ok = got == want

    # --- check 2: meaning-preserving lowering ---
    l2 = lower_l1_to_l2(l1, factors=factors, T=T)
    r2 = ir.run(l2, inputs)
    l2_ok = all(ref._eq(r1[k], r2[k]) for k in r1)
    l3 = l3_ok = None
    if do_l3:
        l3 = lower_l2_to_l3(l2)
        r3 = ir.run(l3, inputs)
        l3_ok = all(ref._eq(r1[k], r3[k]) for k in r1)

    if verbose:
        err = 0 if dec_ok else max(abs(g - w) for g, w in zip(got, want))
        print(f"  N={N:<4} LQ={LQ} LP={LP} dnum={dnum} ndrop={ndrop} seed={seed} "
              f"ntt={'4step'+str(factors) if factors else 'single'} T={T}")
        print(f"      L1 {len(l1.ops):>5} ops   L2 {len(l2.ops):>5} ops"
              + (f"   L3 {len(l3.ops):>6} ops" if l3 else ""))
        print(f"      decrypt {'OK' if dec_ok else f'FAIL (max err {err})'}"
              f"   ·   run(L1)==run(L2) {'OK' if l2_ok else 'FAIL'}"
              + (f"   ·   ==run(L3) {'OK' if l3_ok else 'FAIL'}" if l3 else ""))
    return dec_ok and l2_ok and (l3_ok is not False)


if __name__ == "__main__":
    print("=== CKKS front end: decrypt + meaning-preserving lowering ===")
    ok = True
    ok &= gate(N=16, LQ=4, LP=2, dnum=2, seed=0)
    ok &= gate(N=32, LQ=4, LP=2, dnum=2, seed=3)
    ok &= gate(N=32, LQ=5, LP=2, dnum=3, seed=1)
    print("\n=== four-step NTT (mandatory past N=2^11) ===")
    ok &= gate(N=16, LQ=4, LP=2, dnum=2, seed=0, factors=(4, 4))
    ok &= gate(N=64, LQ=4, LP=2, dnum=2, seed=0, factors=(8, 8), do_l3=False)
    print("\n=== multi-prime rescale ===")
    ok &= gate(N=32, LQ=6, LP=2, dnum=3, ndrop=2, seed=1)
    print("\nCKKS GATE OK" if ok else "\nCKKS GATE FAIL")
    sys.exit(0 if ok else 1)
