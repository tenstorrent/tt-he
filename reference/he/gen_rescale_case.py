#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Emit a device rescale test case (binary) for host/tests/he/bfv_rescale.cpp.

Primes: 8 distinct primes just below 2^28 (same selection the device uses) — Q=3, S(aux)=4, m_sk=1.
Rescale is coefficient-domain modular arithmetic, so NTT-friendliness is NOT required here.

Dumps little-endian uint32, TILIZED, in this order (N=1024, ELEMS=1024/tile):
    eQ[3]   eS[4]   emsk[1]        (plain tensor residues — device INPUT)
    tildeQ[3]                      (golden ẽ = round(t·e/Q) mod q_i — device CHECKS against this)
Header (uint32): N, t, then the 8 primes (q0..q2, s0..s3, msk).
"""
import sys, struct, random
from math import prod
from types import SimpleNamespace
import bfv_mul_ref as m

T = 32; ELEMS = 1024
def is_prime(n):
    if n < 2: return False
    d = 2
    while d*d <= n:
        if n % d == 0: return False
        d += 1
    return True
def tilize(v):
    o = [0]*ELEMS
    for r in range(T):
        for c in range(T):
            o[((r//16)*2 + c//16)*256 + (r%16)*16 + (c%16)] = v[r*T + c]
    return o

def main():
    N = 1024; t = 12289; seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    pr = []
    c = (1 << 28) - 1
    while len(pr) < 8:
        if is_prime(c): pr.append(c)
        c -= 2
    Q = pr[0:3]; S = pr[3:7]; msk = pr[7]
    pp = SimpleNamespace(N=N, t=t, Q=Q, Qprod=prod(Q))

    # random tensor value per coeff < N·Q^2 (real tensor range); its residues + golden ẽ
    bound = 1 << 170
    e_val = [rng.randrange(bound) for _ in range(N)]
    def rows(base): return [[v % b for v in e_val] for b in base]
    e_res = {'Q': rows(Q), 'S': rows(S), 'msk': rows([msk])}
    tildeQ = m.rns_scale(pp, e_res, S, msk)          # ẽ in base Q (3 rows), device-faithful

    out = open(sys.argv[2] if len(sys.argv) > 2 else "rescale_case.bin", "wb")
    out.write(struct.pack("<2I", N, t))
    out.write(struct.pack("<8I", *Q, *S, msk))
    def dump(rowset):
        for row in rowset:
            til = tilize(row)
            out.write(struct.pack("<%dI" % ELEMS, *til))
    dump(e_res['Q']); dump(e_res['S']); dump(e_res['msk'])
    dump(tildeQ)
    out.close()
    print(f"wrote rescale case: N={N} t={t} Q={Q} S={S} msk={msk}  seed={seed}")

if __name__ == "__main__":
    main()
