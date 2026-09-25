# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""RNS fast base conversion (the core primitive under mod-up / mod-down / key-switch).

Convert the residues of an integer (or a vector of integers, e.g. polynomial
coefficients) from a source RNS base  Q = {q_0,...,q_{L-1}}  to a destination
base  P = {p_0,...,p_{k-1}}, WITHOUT reconstructing the big integer.

Math (per coefficient).  With  q̂_i = Q / q_i  and  y_i = [x_i · q̂_i^{-1}]_{q_i} ∈ [0,q_i):

    x  =  ( Σ_i y_i · q̂_i )  −  ε·Q ,      ε = ⌊ Σ_i y_i / q_i ⌋ ∈ {0,...,L-1}

so for each destination modulus p_j:

    x mod p_j  =  ( Σ_i y_i·(q̂_i mod p_j)  −  ε·(Q mod p_j) )  mod p_j

  * fast_base_convert  drops the −ε·(Q mod p_j) term  → returns (x + ε·Q) mod p_j
    (the "approximate FBC": pure vec_mul_const + vec_modadd; error is a multiple of Q,
    which folds into noise for key-switch mod-up/down).
  * base_convert(exact=True)  also computes ε and subtracts the correction
    → bit-exact (needed for the BFV t/Q scaling). ε is a floor of a sum of fractions:
    a floating-point op (HPS), or an integer extra-modulus op (Shenoy-Kumaresan) on HW.

See docs/he/bfv_build_plan.md §1.  Everything here is a reference / oracle; on TT the
linear combination is `vec_mul_const`+`vec_modadd` (matmul-shaped), the ε correction is the
one piece outside the modular-vector-op set.
"""

from math import floor, prod
from typing import List, Sequence


# --------------------------------------------------------------------------------------
# vector kernels (mirror the hardware: element-wise modular ops over a coefficient vector)
# --------------------------------------------------------------------------------------
def vec_mul_const(a: Sequence[int], c: int, q: int) -> List[int]:
    return [(x * c) % q for x in a]


def vec_modadd(a: Sequence[int], b: Sequence[int], q: int) -> List[int]:
    return [(x + y) % q for x, y in zip(a, b)]


# --------------------------------------------------------------------------------------
# precomputed constants (host side, once per (src, dst) pair)
# --------------------------------------------------------------------------------------
class BConvParams:
    """All constants needed to convert from src base to dst base."""

    def __init__(self, src: Sequence[int], dst: Sequence[int]):
        self.src = list(src)
        self.dst = list(dst)
        self.Q = prod(self.src)                              # full source modulus
        self.qhat = [self.Q // qi for qi in self.src]        # q̂_i = Q/q_i
        # q̂_i^{-1} mod q_i
        self.qhat_inv = [pow(self.qhat[i], -1, self.src[i]) for i in range(len(self.src))]
        # q̂_i mod p_j   (the CRT mixing matrix, [L_src][L_dst])
        self.qhat_mod_dst = [[self.qhat[i] % pj for pj in self.dst] for i in range(len(self.src))]
        # Q mod p_j      (for the exact ε correction)
        self.Q_mod_dst = [self.Q % pj for pj in self.dst]


# --------------------------------------------------------------------------------------
# the conversions
# --------------------------------------------------------------------------------------
def _alphas(X: Sequence[Sequence[int]], p: BConvParams) -> List[List[int]]:
    """y_i = [x_i · q̂_i^{-1}]_{q_i}  for each source limb  (one vec_mul_const each)."""
    return [vec_mul_const(X[i], p.qhat_inv[i], p.src[i]) for i in range(len(p.src))]


def fast_base_convert(X: Sequence[Sequence[int]], p: BConvParams) -> List[List[int]]:
    """Approximate FBC: returns (x + ε·Q) mod p_j per dst limb. Pure vector ops."""
    y = _alphas(X, p)
    n = len(X[0])
    out = []
    for j, pj in enumerate(p.dst):
        acc = [0] * n
        for i in range(len(p.src)):
            acc = vec_modadd(acc, vec_mul_const(y[i], p.qhat_mod_dst[i][j], pj), pj)
        out.append(acc)
    return out


def overflow_eps(X: Sequence[Sequence[int]], p: BConvParams, use_float: bool = True) -> List[int]:
    """ε = ⌊ Σ_i y_i / q_i ⌋ ∈ {0,...,L-1}, per coefficient.

    use_float=True  : the HPS-style float sum (what fp32 SFPU would do; double here is exact).
    use_float=False : exact integer (reference; on HW this is the Shenoy-Kumaresan path).
    """
    y = _alphas(X, p)
    n = len(X[0])
    eps = [0] * n
    if use_float:
        for k in range(n):
            s = sum(y[i][k] / p.src[i] for i in range(len(p.src)))
            eps[k] = floor(s + 1e-9)       # tiny guard against fp ties near an integer
    else:
        for k in range(n):
            s = sum(y[i][k] * p.qhat[i] for i in range(len(p.src)))  # = x + εQ (big int)
            eps[k] = s // p.Q
    return eps


def base_convert(X: Sequence[Sequence[int]], p: BConvParams,
                 exact: bool = True, use_float: bool = True) -> List[List[int]]:
    """Base convert. exact=True subtracts the ε·Q correction → bit-exact x mod p_j."""
    approx = fast_base_convert(X, p)
    if not exact:
        return approx
    eps = overflow_eps(X, p, use_float=use_float)
    out = []
    for j, pj in enumerate(p.dst):
        corr = (p.Q_mod_dst[j]) % pj
        out.append([(approx[j][k] - eps[k] * corr) % pj for k in range(len(X[0]))])
    return out


# --------------------------------------------------------------------------------------
# reference helpers (big-int) for validation
# --------------------------------------------------------------------------------------
def to_rns(vals: Sequence[int], base: Sequence[int]) -> List[List[int]]:
    return [[v % q for v in vals] for q in base]


def base_convert_ref(vals: Sequence[int], dst: Sequence[int]) -> List[List[int]]:
    """Exact reference: reduce the true integers mod each dst modulus."""
    return [[v % pj for v in vals] for pj in dst]


# --------------------------------------------------------------------------------------
# test
# --------------------------------------------------------------------------------------
def _test():
    import random
    random.seed(0)

    # a few RNS bases (28-bit-ish NTT-friendly primes for Q; arbitrary primes for P)
    cases = [
        # (src Q, dst P)
        ([167772161, 469762049, 1004535809], [377487361, 1224736769]),
        ([12289, 40961, 65537], [786433, 5767169, 7340033]),
        ([167772161, 201326593], [268435009, 1004535809, 377487361]),
    ]
    NCOEF = 64
    all_ok = True
    for src, dst in cases:
        Q = prod(src)
        p = BConvParams(src, dst)
        vals = [random.randrange(Q) for _ in range(NCOEF)]   # integers < Q
        X = to_rns(vals, src)

        ref = base_convert_ref(vals, dst)
        exact_f = base_convert(X, p, exact=True, use_float=True)
        exact_i = base_convert(X, p, exact=True, use_float=False)
        approx = base_convert(X, p, exact=False)

        ok_f = exact_f == ref
        ok_i = exact_i == ref
        # approximate result equals ref shifted by ε·Q (mod p_j)
        eps = overflow_eps(X, p)
        approx_ok = all(
            approx[j][k] == (ref[j][k] + eps[k] * (Q % dst[j])) % dst[j]
            for j in range(len(dst)) for k in range(NCOEF)
        )
        eps_hist = {}
        for e in eps:
            eps_hist[e] = eps_hist.get(e, 0) + 1

        status = "PASS" if (ok_f and ok_i and approx_ok) else "FAIL"
        all_ok &= ok_f and ok_i and approx_ok
        print(f"[{status}] L_src={len(src)} L_dst={len(dst)}  "
              f"exact(float)={ok_f} exact(int)={ok_i} approx_consistent={approx_ok}  "
              f"ε histogram={eps_hist}")

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return all_ok


if __name__ == "__main__":
    _test()
