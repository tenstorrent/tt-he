#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Emit a device FULL-MULTIPLY test case (binary) for host/tests/he/bfv_ct_mul.cpp.

Uses the exact e2e_verify params (N=1024, Q=3/P=3 NTT-friendly, S=aux 4, m_sk, t=12289) —
the config whose decrypt(ct1*ct2)==m1*m2 is already validated in bfv_mul_ref.e2e_verify.

The device driver chains (correctness mode, golden tensor threaded in):
    e (tensor, host golden)  ->  device rescale x3  ->  device relin(e~2)  ->  device add
and checks its (out0,out1) bit-exact against the golden out below (which decrypts to m1*m2).

Dumps little-endian uint32, TILIZED. Header (uint32): N, t, then 11 primes q0..q2, p0..p2, s0..s3, msk.
Then, all base-limb rows (each 1024 tilized uint32):
    evk0[6 QP limbs], evk1[6 QP limbs]                 (coeff domain, QP=Q|P)   [relin key]
    c0[8], c1[8], d0[8], d1[8]                          (ct residues Q3,S4,msk1) [tensor IN]
    e0[Q3,S4,msk1], e1[...], e2[...]                    (tensor residues)        [tensor CHK / rescale IN]
    tildeQ0[3], tildeQ1[3], tildeQ2[3]                  (golden e~ base Q)       [rescale CHK]
    k0[3], k1[3]                                        (golden relin base Q)    [relin  CHK]
    out0[3], out1[3]                                    (golden final base Q)    [final  CHK]
"""
import sys, struct, random
from math import prod
import bfv_mul_ref as m
from bfv_relin_ref import ntt_primes, crt_reconstruct, rns_of

T = 32; ELEMS = 1024
def tilize(v):
    o = [0] * ELEMS
    for r in range(T):
        for c in range(T):
            o[((r // 16) * 2 + c // 16) * 256 + (r % 16) * 16 + (c % 16)] = v[r * T + c]
    return o

def main():
    N = 1024; t = 12289; seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    pp = m.MulParams(N=N, LQ=3, LP=3, LB=4, t=t)
    Q, P, S = pp.Q, pp.P, pp.B
    msk = [p for p in ntt_primes(80, N) if p not in set(Q + P + S)][0]
    QP = pp.QP                                              # Q|P, 6 limbs (relin domain)

    s = m.small_poly(N, rng); evk = m.keygen_relin(pp, s, rng)   # evk=(evk0,evk1) over QP, coeff
    m1 = [rng.randrange(t) for _ in range(N)]; m2 = [rng.randrange(t) for _ in range(N)]
    ct1, ct2 = m.encrypt(pp, m1, s, rng), m.encrypt(pp, m2, s, rng)

    # ---- golden stages (device-faithful RNS), mirrors rns_ct_mul ----
    e = m.tensor_unsigned(pp, ct1, ct2)                    # (e0,e1,e2) over Z, unsigned
    def rows(poly, base): return [[c % b for c in poly] for b in base]
    e_res = [ {'Q': rows(ek, Q), 'S': rows(ek, S), 'msk': rows(ek, [msk])} for ek in e ]
    tilde = [ m.rns_scale(pp, er, S, msk) for er in e_res ] # e~_k in base Q (3 rows each)
    tilde_int = [ [crt_reconstruct([tk[i][n] for i in range(3)], Q) for n in range(N)] for tk in tilde ]
    from bfv_relin_ref import relinearize
    k0, k1 = relinearize(pp, rns_of(tilde_int[2], Q), evk[0], evk[1], rng)   # base Q rows
    out0 = [[(tilde[0][i][n] + k0[i][n]) % Q[i] for n in range(N)] for i in range(3)]
    out1 = [[(tilde[1][i][n] + k1[i][n]) % Q[i] for n in range(N)] for i in range(3)]

    # sanity: decrypt(out) == m1*m2 (mod t)
    out0i = [crt_reconstruct([out0[i][n] for i in range(3)], Q) for n in range(N)]
    out1i = [crt_reconstruct([out1[i][n] for i in range(3)], Q) for n in range(N)]
    dec = m.decrypt(pp, [out0i, out1i], s)
    want = m.negacyclic_mul(m1, m2, t)
    mism = sum(a != b for a, b in zip(dec, want))
    print(f"seed={seed} golden decrypt(ct1*ct2) vs m1*m2 mism={mism}/{N}  [{'OK' if mism==0 else 'BAD'}]")

    path = sys.argv[2] if len(sys.argv) > 2 else "ctmul_case.bin"
    f = open(path, "wb")
    f.write(struct.pack("<2I", N, t))
    f.write(struct.pack("<11I", *Q, *P, *S, msk))
    def dump_rows(rowset):
        for row in rowset:
            f.write(struct.pack("<%dI" % ELEMS, *tilize(row)))
    dump_rows(evk[0]); dump_rows(evk[1])                   # 6 + 6 QP limbs
    base8 = Q + S + [msk]                                   # tensor base = Q(3)|S(4)|msk(1)
    def rows8(poly): return [[c % p for c in poly] for p in base8]
    for poly in [ct1[0], ct1[1], ct2[0], ct2[1]]:          # c0,c1,d0,d1 residues (tensor input)
        dump_rows(rows8(poly))
    for k in range(3):
        dump_rows(e_res[k]['Q']); dump_rows(e_res[k]['S']); dump_rows(e_res[k]['msk'])
    for k in range(3): dump_rows(tilde[k])
    dump_rows(k0); dump_rows(k1)
    dump_rows(out0); dump_rows(out1)
    f.close()
    print(f"wrote {path}: N={N} t={t} Q={Q} P={P} S={S} msk={msk}")

if __name__ == "__main__":
    main()
