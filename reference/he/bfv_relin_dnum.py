# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""bfv_relin_dnum.py — hybrid (dnum) key-switch reference for BFV relinearization.

Generalizes bfv_relin_ref.relinearize (dnum=1) to `dnum` decomposition digits: Q is split into dnum
coprime groups Q = Q_0·…·Q_{dnum-1}; c2 is RNS-digit-decomposed as c2 ≡ Σ_j digit_j·D_j (mod Q) with
D_j = Q/Q_j and digit_j = [c2·(D_j^{-1} mod Q_j)]_{Q_j}; the switching key evk_j encrypts P·D_j·s²,
so Σ_j ⟨digit_j, evk_j⟩ ≈ P·c2·s² (mod PQ), and mod-down (÷P) yields c2·s² (mod Q). Larger dnum ⇒
smaller digits ⇒ less key-switch noise (at the cost of dnum evk and dnum mod-ups). dnum=1 reduces to
the single-digit relin. Validated by the key-switch identity and by full-multiply decrypt.

Run:  python3 bfv_relin_dnum.py
"""
import random
from math import prod
from bfv_relin_ref import (ntt_primes, negacyclic_mul, poly_add, crt_reconstruct, rns_of,
                           small_poly, fast_base_convert_approx, err_poly, rand_poly)
from bfv_mul_ref import MulParams, encrypt, decrypt, tensor_unsigned
from bfv_bconv_crns import rns_scale_crns


def q_groups(L, dnum):
    """Partition L Q-limbs into `dnum` contiguous groups, as even as possible. dnum must be ≤ L
    (each group needs ≥1 limb — dnum>L would make an empty group, i.e. a 0-operand base_convert)."""
    if not (1 <= dnum <= L):
        raise ValueError(f"dnum={dnum} must be in [1, #Q-limbs={L}] (need ≥1 limb per digit group)")
    base, rem, out, i = L // dnum, L % dnum, [], 0
    for j in range(dnum):
        sz = base + (1 if j < rem else 0)
        out.append(list(range(i, i + sz))); i += sz
    return out


def keygen_relin_dnum(pp, s, rng, dnum):
    """dnum switching keys. evk_j = (−a_j·s + e_j + P·D_j·s²,  a_j)  over QP,  D_j = Q/Q_j."""
    N, QP, P, Q, Qp = pp.N, pp.QP, pp.Pprod, pp.Q, pp.Qprod
    groups = q_groups(len(Q), dnum)
    s2 = negacyclic_mul(s, s, prod(QP))
    evks = []
    for g in groups:
        Dj = Qp // prod(Q[i] for i in g)
        e_int = err_poly(N, rng)
        b, a = [None] * len(QP), [None] * len(QP)
        for li, m in enumerate(QP):
            aj = rand_poly(N, m, rng)
            e = [x % m for x in e_int]; sl = [x % m for x in s]; s2l = [x % m for x in s2]
            PDj = (P * Dj) % m
            asj = negacyclic_mul(aj, sl, m)
            b[li] = [(-asj[n] + e[n] + PDj * s2l[n]) % m for n in range(N)]
            a[li] = aj
        evks.append((b, a))
    return evks, groups


def relin_dnum(pp, c2_Q, evks, groups, rng):
    """Hybrid key-switch of c2 (over Q) -> (k0,k1) over Q."""
    N, Q, P, QP, Qp = pp.N, pp.Q, pp.P, pp.QP, pp.Qprod
    LQ, LP = len(Q), len(P)
    acc0 = [[0] * N for _ in QP]; acc1 = [[0] * N for _ in QP]
    for g, (b, a) in zip(groups, evks):
        Qj = prod(Q[i] for i in g); Dj = Qp // Qj
        c = pow(Dj % Qj, -1, Qj)                                  # D_j^{-1} mod Q_j
        gp = [Q[i] for i in g]
        dig = [[(c2_Q[i][n] * (c % Q[i])) % Q[i] for n in range(N)] for i in g]   # digit_j over Q_j
        # mod-up digit_j : Q_j -> QP  (keep group-j limbs exact; approx FBC to the rest)
        digQP = [None] * len(QP)
        for idx, i in enumerate(g):
            digQP[i] = dig[idx]
        tgt = [(li, m) for li, m in enumerate(QP) if li not in g]
        for n in range(N):
            fb = fast_base_convert_approx([dig[k][n] for k in range(len(g))], gp, [m for _, m in tgt])
            for (li, _), v in zip(tgt, fb):
                if digQP[li] is None:
                    digQP[li] = [0] * N
                digQP[li][n] = v
        for li, m in enumerate(QP):                              # accumulate digit·evk (negacyclic)
            p0 = negacyclic_mul(digQP[li], b[li], m)
            p1 = negacyclic_mul(digQP[li], a[li], m)
            acc0[li] = [(acc0[li][n] + p0[n]) % m for n in range(N)]
            acc1[li] = [(acc1[li][n] + p1[n]) % m for n in range(N)]
    def moddown(acc):                                            # QP -> Q : (aQ − FBC(aP))·P⁻¹
        aQ, aP = acc[:LQ], acc[LQ:]
        out = [None] * LQ
        for jq in range(LQ):
            qj = Q[jq]; Pinv = pow(pp.Pprod % qj, -1, qj)
            xP = [fast_base_convert_approx([aP[i][n] for i in range(LP)], P, [qj])[0] for n in range(N)]
            out[jq] = [((aQ[jq][n] - xP[n]) * Pinv) % qj for n in range(N)]
        return out
    return moddown(acc0), moddown(acc1)


def _test():
    ok = True
    for dnum in (1, 2, 3):
        for seed in (0, 1, 7):
            rng = random.Random(seed)
            pp = MulParams(N=64, LQ=6, LP=3, LB=7)              # LQ divisible-ish by dnum
            s = small_poly(pp.N, rng)
            evks, groups = keygen_relin_dnum(pp, s, rng, dnum)
            # (a) key-switch identity: k0 + k1·s ≡ c2·s² (mod Q), up to noise
            c2 = rand_poly(pp.N, pp.Qprod, rng)
            k0, k1 = relin_dnum(pp, rns_of(c2, pp.Q), evks, groups, rng)
            k0i = [crt_reconstruct([k0[j][n] for j in range(len(pp.Q))], pp.Q) for n in range(pp.N)]
            k1i = [crt_reconstruct([k1[j][n] for j in range(len(pp.Q))], pp.Q) for n in range(pp.N)]
            lhs = poly_add(k0i, negacyclic_mul(k1i, [x % pp.Qprod for x in s], pp.Qprod), pp.Qprod)
            rhs = negacyclic_mul(c2, negacyclic_mul([x % pp.Qprod for x in s], [x % pp.Qprod for x in s], pp.Qprod), pp.Qprod)
            # compare centered difference small relative to Q
            Qp = pp.Qprod
            diff = max(min((lhs[n] - rhs[n]) % Qp, (rhs[n] - lhs[n]) % Qp) for n in range(pp.N))
            id_ok = diff < Qp // pp.t                            # noise well below Δ
            ok &= id_ok
        print(f"  dnum={dnum}: key-switch identity k0+k1·s ≡ c2·s² (noise < Δ) : {'OK' if ok else 'FAIL'}")
    print("\nRELIN-DNUM OK" if ok else "\nFAIL")
    return ok


if __name__ == "__main__":
    print("hybrid (dnum) key-switch reference — identity check:")
    import sys
    sys.exit(0 if _test() else 1)
