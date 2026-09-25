# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""RNS-BFV relinearization / hybrid key-switch with dnum>1 digit decomposition — CPU golden.

Generalizes `bfv_relin_ref.relinearize` (dnum=1) to `dnum` digits, and swaps the approximate FBC
for the verified m_sk-free **CRNS** exact base conversion (`bfv_bconv_crns._fbc_exact_crns`) in
both mod-up and mod-down. This is the spec for the device dnum-loop key-switch.

Hybrid (special-modulus P) key-switch of the degree-2 term c2 back to (k0,k1) under s:

    k0 + k1·s  ≡  c2·s²   (mod Q)      up to key-switch noise.

Digit decomposition (dnum groups, alpha = LQ/dnum limbs each):
  - group j owns Q-limbs [j·alpha : (j+1)·alpha];  Q_j = ∏ of that group's primes.
  - gadget idempotent  ĝ_j = (Q/Q_j)·[(Q/Q_j)^{-1} mod Q_j]  (mod Q):  ĝ_j ≡ 1 (mod Q_j),
    ≡ 0 (mod Q/Q_j), so  Σ_j v_j·ĝ_j ≡ d (mod Q)  where v_j = [d]_{Q_j}.  This is why the
    per-digit products recombine to d·s² after mod-down.
  - switching key for digit j:  evk_j = (−a_j·s + e_j + P·ĝ_j·s²,  a_j)  (mod QP).

Algorithm  relinearize_dnum(c2 in base Q):
  for each digit j:
    1. mod-up   v_j = [c2]_{Q_j} raised Q_j -> QP   (own limbs kept, rest via CRNS exact FBC)
    2. acc += v_j ⊛ evk_j      (per-limb negacyclic poly-mult, accumulate over digits)
  3. mod-down  acc : QP -> Q   (CRNS exact FBC(P->Q) + subtract + ×P^-1)  -> (k0,k1)

dnum=1 collapses to the original single-group relin (regression-checked).

Run:  python3 bfv_relin_dnum_ref.py
"""

import math
import random
from math import prod

from bfv_relin_ref import (
    ntt_primes, negacyclic_mul, poly_add, poly_sub,
    crt_reconstruct, small_poly, err_poly, rand_poly, rns_of,
)
from bfv_bconv_crns import _fbc_exact_crns, crns_precompute, crns_safe_u


# --------------------------------------------------------------------------- #
# parameters: N, LQ limbs of Q, LP limbs of P, dnum digits  (LQ % dnum == 0)
# --------------------------------------------------------------------------- #
class DnumParams:
    def __init__(self, N=1024, LQ=4, LP=2, dnum=2):
        assert LQ % dnum == 0, "dnum must divide LQ"
        pr = ntt_primes(LQ + LP, N)
        self.N, self.dnum, self.alpha = N, dnum, LQ // dnum
        self.Q = pr[:LQ]
        self.P = pr[LQ:LQ + LP]
        self.QP = self.Q + self.P
        self.Qprod = prod(self.Q)
        self.Pprod = prod(self.P)
        # digit groups (Q-limb indices) and per-group products Q_j
        self.groups = [list(range(j * self.alpha, (j + 1) * self.alpha)) for j in range(dnum)]
        self.Qj = [prod(self.Q[i] for i in g) for g in self.groups]
        # gadget idempotents ĝ_j (mod Q):  ≡ 1 mod Q_j, ≡ 0 mod (Q/Q_j)
        self.ghat = []
        for j in range(dnum):
            Qj = self.Qj[j]
            comp = self.Qprod // Qj                       # Q / Q_j
            self.ghat.append((comp * pow(comp % Qj, -1, Qj)) % self.Qprod)


# --------------------------------------------------------------------------- #
# keygen: one evk pair per digit
# --------------------------------------------------------------------------- #
def keygen_relin_dnum(pp, s, rng):
    """evk_j = (−a_j·s + e_j + P·ĝ_j·s²,  a_j)  (mod QP), one per digit.
       e is ONE small integer poly reduced per limb (per bfv_relin_ref note); a is per-limb."""
    N, QP, P = pp.N, pp.QP, pp.Pprod
    s2 = negacyclic_mul(s, s, prod(QP))                   # s² as one small integer poly
    evks = []
    for j in range(pp.dnum):
        ghat = pp.ghat[j]
        e_int = err_poly(N, rng)
        evk0 = [None] * len(QP)
        evk1 = [None] * len(QP)
        for li, m in enumerate(QP):
            a = rand_poly(N, m, rng)
            e = [x % m for x in e_int]
            sl = [x % m for x in s]
            s2l = [x % m for x in s2]
            Pghat = (P % m) * (ghat % m) % m
            as_ = negacyclic_mul(a, sl, m)
            evk0[li] = [(-as_[n] + e[n] + Pghat * s2l[n]) % m for n in range(N)]
            evk1[li] = a
        evks.append((evk0, evk1))
    return evks


# --------------------------------------------------------------------------- #
# mod-up one digit: [c2]_{Q_j} (own limbs) -> full QP (rest via CRNS exact FBC)
# --------------------------------------------------------------------------- #
def _modup_digit(pp, c2_Q, j, u=None):
    N, Q, QP = pp.N, pp.Q, pp.QP
    g = pp.groups[j]
    src = [Q[i] for i in g]                               # this digit's primes
    dst_idx = [li for li in range(len(QP)) if li not in g]
    dst = [QP[li] for li in dst_idx]
    uu = crns_safe_u(src) if u is None else u
    pre = crns_precompute(src, dst, uu)
    vj = [None] * len(QP)
    for li in g:
        vj[li] = c2_Q[li][:]                              # own limbs kept as-is
    conv = [_fbc_exact_crns([c2_Q[i][n] for i in g], src, dst, u=uu, pre=pre) for n in range(N)]
    for kk, li in enumerate(dst_idx):
        vj[li] = [conv[n][kk] for n in range(N)]
    return vj


# --------------------------------------------------------------------------- #
# mod-down acc : QP -> Q   (CRNS exact FBC(P->Q) + subtract + ×P^-1)
# --------------------------------------------------------------------------- #
def _moddown(pp, acc, u=None):
    N, Q, P = pp.N, pp.Q, pp.P
    LQ, LP = len(Q), len(P)
    aQ, aP = acc[:LQ], acc[LQ:]
    uu = crns_safe_u(list(P)) if u is None else u
    pre = crns_precompute(list(P), Q, uu)
    Pinv = [pow(pp.Pprod % Q[j], -1, Q[j]) for j in range(LQ)]
    conv = [_fbc_exact_crns([aP[i][n] for i in range(LP)], list(P), Q, u=uu, pre=pre) for n in range(N)]
    out = [None] * LQ
    for j in range(LQ):
        out[j] = [((aQ[j][n] - conv[n][j]) * Pinv[j]) % Q[j] for n in range(N)]
    return out


# --------------------------------------------------------------------------- #
# hybrid key-switch, dnum digits
# --------------------------------------------------------------------------- #
def relinearize_dnum(pp, c2_Q, evks, u_up=None, u_down=None):
    """c2 given in base Q (per-limb). Returns (k0,k1) in base Q."""
    N, QP = pp.N, pp.QP
    acc0 = [[0] * N for _ in QP]
    acc1 = [[0] * N for _ in QP]
    for j in range(pp.dnum):
        vj = _modup_digit(pp, c2_Q, j, u=u_up)
        evk0, evk1 = evks[j]
        for li in range(len(QP)):
            m = QP[li]
            p0 = negacyclic_mul(vj[li], evk0[li], m)
            p1 = negacyclic_mul(vj[li], evk1[li], m)
            acc0[li] = [(acc0[li][n] + p0[n]) % m for n in range(N)]
            acc1[li] = [(acc1[li][n] + p1[n]) % m for n in range(N)]
    return _moddown(pp, acc0, u=u_down), _moddown(pp, acc1, u=u_down)


# --------------------------------------------------------------------------- #
# validation: key-switch identity  k0 + k1·s ≡ c2·s²  (mod Q)  up to noise
# --------------------------------------------------------------------------- #
def _run_one(N, LQ, LP, dnum, seed):
    rng = random.Random(seed)
    pp = DnumParams(N=N, LQ=LQ, LP=LP, dnum=dnum)
    s = small_poly(N, rng)
    evks = keygen_relin_dnum(pp, s, rng)

    c2 = rand_poly(N, pp.Qprod, rng)                     # random degree-2 term (big-int poly)
    c2_Q = rns_of(c2, pp.Q)

    k0, k1 = relinearize_dnum(pp, c2_Q, evks)

    k0i = [crt_reconstruct([k0[j][n] for j in range(LQ)], pp.Q) for n in range(N)]
    k1i = [crt_reconstruct([k1[j][n] for j in range(LQ)], pp.Q) for n in range(N)]
    sQ = [x % pp.Qprod for x in s]
    lhs = poly_add(k0i, negacyclic_mul(k1i, sQ, pp.Qprod), pp.Qprod)
    rhs = negacyclic_mul(c2, negacyclic_mul(sQ, sQ, pp.Qprod), pp.Qprod)

    diff = poly_sub(lhs, rhs, pp.Qprod)
    def centered(v, q): return v - q if v > q // 2 else v
    ninf = max(abs(centered(d, pp.Qprod)) for d in diff)
    logQ = math.log2(pp.Qprod)
    lognoise = math.log2(ninf) if ninf else 0.0
    margin = logQ - lognoise
    ok = margin > 20
    return ok, logQ, lognoise, margin, pp


def _test():
    print("=== dnum>1 hybrid key-switch (CRNS bconv) — key-switch identity ===\n")
    all_ok = True
    for dnum in (1, 2, 4):
        row_ok = True
        for seed in (0, 1, 7, 42):
            ok, logQ, lognoise, margin, pp = _run_one(1024, 4, 2, dnum, seed)
            row_ok &= ok
            if seed == 0:
                print(f"dnum={dnum}  alpha={pp.alpha}  Q={len(pp.Q)}limb P={len(pp.P)}limb  "
                      f"log2Q={logQ:.1f}")
            print(f"    seed={seed:>2}: log2||noise||={lognoise:5.1f}  margin={margin:5.1f}b  "
                  f"[{'PASS' if ok else 'FAIL'}]")
        all_ok &= row_ok
        print()
    print(f"RESULT: {'ALL PASS — dnum-decomposed key-switch preserves decryption' if all_ok else 'FAILURES ABOVE'}")
    return all_ok


if __name__ == "__main__":
    _test()
