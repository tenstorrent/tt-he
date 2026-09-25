# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ckks_mul_ref.py — CKKS ct_mul reference, COEFFICIENT DOMAIN. Oracle for the compiler front end.

WHY COEFFICIENT DOMAIN. The compiler never sees the canonical embedding: encode/decode sit entirely
outside the op graph, and the RNS graph the compiler lowers and schedules is identical whether the
plaintext arrived through a complex FFT or was handed over as an integer polynomial. So this reference
drops the encoder and validates against an EXACT negacyclic integer product, which is a stronger gate
than an approximate slot comparison -- `round(dec/Delta) == m1*m2` exactly, not "close in 40 bits".

WHAT DIFFERS FROM BFV (`bfv_mul_ref.py`), and it is only the order and the rescale:

    BFV    tensor over Q u S  ->  rescale (t/Q, BEHZ two-base floor)  ->  relin       ->  add
    CKKS   tensor over Q      ->  relin                               ->  add         ->  rescale (drop a prime)

CKKS's rescale is the cheap one: dropping prime q_L is
    out_i = (x_i - (x_L mod q_i)) * q_L^-1   (mod q_i),  i < L
i.e. one modulus change plus a subtract and a scalar multiply per surviving limb -- no second base, no
FBC over a wide accumulator. That is why the hardware pipeline's rescale is a handful of kernels
(`bconv_one`, `modsub_reduce`, `modmul_scalar`) against the base conversions' hundreds.

This is the UNCENTRED rescale: it computes floor(x/q_L) from the unsigned representative rather than
the centred one. For a value representing a negative number the two differ by exactly Q/q_L, which IS
the new modulus, so the result is correct in the new RNS base either way. The hardware bench carries
the same choice as `TG18NTT_NOCENTRE`.

SCALE BOOKKEEPING. Delta is set to the LAST Q prime, so multiply (scale -> Delta^2) followed by one
rescale (drop that prime) returns the scale to Delta exactly. That is what makes the exact gate work.

Self-check:  python3 ckks_mul_ref.py
"""
from __future__ import annotations
import random
from math import prod

from bfv_relin_ref import (ntt_primes, negacyclic_mul, crt_reconstruct, rns_of,
                           small_poly, err_poly, rand_poly, keygen_relin, relinearize)
from bfv_relin_dnum import keygen_relin_dnum, relin_dnum, q_groups


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
class CKKSParams:
    """CKKS RNS parameters. Attribute names match MulParams where they overlap, so every
    scheme-agnostic helper (keygen_relin_dnum, relin_dnum, builders._emit_relin*) works unchanged."""

    def __init__(self, N=64, LQ=4, LP=2, ndrop=1):
        self.N = N
        pr = ntt_primes(LQ + LP, N)
        self.Q = pr[:LQ]
        self.P = pr[LQ:LQ + LP]
        self.QP = self.Q + self.P
        self.Qprod, self.Pprod = prod(self.Q), prod(self.P)
        # Delta must equal EXACTLY what the rescale removes, or the scale does not come back to Delta
        # after multiply+rescale (scale Delta^2 -> Delta^2/prod(dropped)). Tying the two together is
        # what keeps the exact gate `round(dec/scale) == m1*m2` available at any ndrop.
        self.ndrop = ndrop
        self.Delta = prod(self.Q[LQ - ndrop:])
        # BFV-compat attributes some shared helpers read. CKKS has no plaintext modulus; t is unused
        # by anything on the CKKS path and is present only so a MulParams-shaped consumer can bind.
        self.B, self.Bprod, self.t = [], 1, 0
        self.msk = None

    def __repr__(self):
        return (f"CKKSParams(N={self.N}, LQ={len(self.Q)}, LP={len(self.P)}, "
                f"log2Q={self.Qprod.bit_length()}, log2Delta={self.Delta.bit_length()})")


def crt_poly(res, base, N):
    """crt_reconstruct is PER COEFFICIENT; this is the polynomial form."""
    return [crt_reconstruct([res[i][j] for i in range(len(base))], base) for j in range(N)]


# --------------------------------------------------------------------------- #
# encrypt / decrypt  (symmetric key; only to make test ciphertexts)
# --------------------------------------------------------------------------- #
def encrypt(pp, m, s, rng):
    """c0 + c1*s = Delta*m + e  (mod Q).  m is a small INTEGER polynomial. Returns big-int polys."""
    N, Q = pp.N, pp.Qprod
    a = rand_poly(N, Q, rng)
    e = err_poly(N, rng)
    as_ = negacyclic_mul(a, s, Q)
    c0 = [(pp.Delta * m[i] + e[i] - as_[i]) % Q for i in range(N)]
    return c0, a


def decrypt_raw(pp, ct, s, Qprod=None):
    """c0 + c1*s (mod Qprod), CENTRED to (-Q/2, Q/2]. Works for any ciphertext length >= 2."""
    Q = Qprod if Qprod is not None else pp.Qprod
    N = pp.N
    acc = list(ct[0])
    pw = list(s)
    for k in range(1, len(ct)):
        term = negacyclic_mul(ct[k], pw, Q)
        acc = [(acc[i] + term[i]) % Q for i in range(N)]
        if k + 1 < len(ct):
            pw = negacyclic_mul(pw, s, Q)
    return [(v - Q if v > Q // 2 else v) for v in acc]


def decrypt(pp, ct, s, Qprod=None, scale=None):
    """Decrypt and divide out the scale -> the integer plaintext polynomial (rounded).

    `scale` is an int, or a (num, den) pair for the rational scale a multi-prime rescale leaves."""
    raw = decrypt_raw(pp, ct, s, Qprod)
    sc = scale if scale is not None else pp.Delta
    if isinstance(sc, tuple):
        num, den = sc
        return [_round_div(v * den, num) for v in raw]
    return [_round_div(v, sc) for v in raw]


def _round_div(n, d):
    q, r = divmod(n, d)
    return q + 1 if 2 * r >= d else q


# --------------------------------------------------------------------------- #
# the multiply, stage by stage (RNS, per-limb residues)
# --------------------------------------------------------------------------- #
def tensor(pp, ct1, ct2, Qprod=None):
    """(c0,c1) x (d0,d1) -> (e0,e1,e2), degree-2 ciphertext at scale Delta^2."""
    Q = Qprod if Qprod is not None else pp.Qprod
    c0, c1 = ct1
    d0, d1 = ct2
    e0 = negacyclic_mul(c0, d0, Q)
    e2 = negacyclic_mul(c1, d1, Q)
    t0 = negacyclic_mul(c0, d1, Q)
    t1 = negacyclic_mul(c1, d0, Q)
    e1 = [(t0[i] + t1[i]) % Q for i in range(pp.N)]
    return e0, e1, e2


def rescale_res(pp, res, base, ndrop=1):
    """Drop the last `ndrop` primes of `base`. res = per-limb residues (list of lists).

    out_i = (x_i - (x_L mod q_i)) * q_L^-1 mod q_i, applied once per dropped prime, high prime first.
    Returns (residues over the surviving base, surviving base)."""
    res = [list(r) for r in res]
    base = list(base)
    for _ in range(ndrop):
        qL = base[-1]
        xL = res[-1]
        keep, kb = [], base[:-1]
        for i, qi in enumerate(kb):
            inv = pow(qL % qi, -1, qi)
            keep.append([((res[i][j] - xL[j]) * inv) % qi for j in range(pp.N)])
        res, base = keep, kb
    return res, base


def ct_mul(pp, ct1, ct2, s, rng, ks=None, ndrop=1, want_stages=False):
    """Full CKKS multiply: tensor -> relin -> add -> rescale. Returns (ct_out, Qprod_out, scale_out).

    ct1/ct2 are big-int (c0,c1) over Q. `ks=(evks,groups)` selects hybrid dnum key-switch;
    `evk=(evk0,evk1)` the dnum=1 path. Output scale is Delta^(2)/prod(dropped) = Delta when ndrop=1."""
    Q, N = pp.Q, pp.N
    e0, e1, e2 = tensor(pp, ct1, ct2)
    e2_res = rns_of(e2, Q)
    if ks is None:
        raise ValueError("ct_mul needs ks=(evks, groups) from keygen_relin_dnum -- the CKKS path "
                         "always uses the hybrid key-switch")
    if len(ks[1]) < 2:
        raise ValueError(
            "dnum must be >= 2. At dnum=1 the single digit's mod-up is Q->P over the WHOLE Q, so the "
            "approximate FBC's dropped eps*Q_j term is eps*Q -- far outside the noise budget this "
            "reference leaves, and decrypt misses by tens. Reducing that term is exactly what hybrid "
            "key-switching's dnum>1 buys; the hardware pipeline runs dnum=3.")
    k0_res, k1_res = relin_dnum(pp, e2_res, ks[0], ks[1], rng)
    # add the relin result back onto (e0, e1), per limb
    a0 = [[(x + y) % Q[i] for x, y in zip(rns_of(e0, Q)[i], k0_res[i])] for i in range(len(Q))]
    a1 = [[(x + y) % Q[i] for x, y in zip(rns_of(e1, Q)[i], k1_res[i])] for i in range(len(Q))]
    r0, base = rescale_res(pp, a0, Q, ndrop)
    r1, _ = rescale_res(pp, a1, Q, ndrop)
    Qout = prod(base)
    out = (crt_poly(r0, base, N), crt_poly(r1, base, N))
    scale_out = (pp.Delta ** 2, prod(Q[len(base):]))            # (num, den); == Delta when ndrop=1
    if want_stages:
        return out, Qout, scale_out, dict(e0=e0, e1=e1, e2=e2, k0=k0_res, k1=k1_res)
    return out, Qout, scale_out


# --------------------------------------------------------------------------- #
# self-check
# --------------------------------------------------------------------------- #
def _exact_product(m1, m2, N):
    """Negacyclic product over the integers -- what round(dec/Delta) must equal exactly."""
    out = [0] * N
    for i, a in enumerate(m1):
        for j, b in enumerate(m2):
            k = i + j
            if k < N:
                out[k] += a * b
            else:
                out[k - N] -= a * b
    return out


def _check(N=32, LQ=4, LP=2, dnum=2, seed=0, mbound=3, verbose=True):
    pp = CKKSParams(N=N, LQ=LQ, LP=LP)
    rng = random.Random(seed)
    s = small_poly(N, rng)
    evks, groups = keygen_relin_dnum(pp, s, rng, dnum)
    m1 = [rng.randrange(-mbound, mbound + 1) for _ in range(N)]
    m2 = [rng.randrange(-mbound, mbound + 1) for _ in range(N)]
    ct1 = encrypt(pp, m1, s, rng)
    ct2 = encrypt(pp, m2, s, rng)
    out, Qout, scale = ct_mul(pp, ct1, ct2, s, rng, ks=(evks, groups), ndrop=1)
    got = decrypt(pp, out, s, Qprod=Qout, scale=scale)
    want = _exact_product(m1, m2, N)
    ok = got == want
    if verbose:
        err = max(abs(g - w) for g, w in zip(got, want)) if not ok else 0
        print(f"  N={N:<5} LQ={LQ} LP={LP} dnum={dnum} seed={seed}  "
              f"log2Q={pp.Qprod.bit_length():<4} scale=2^{scale[0].bit_length()-scale[1].bit_length():<3} "
              f"-> {'OK' if ok else f'FAIL (max err {err})'}")
    return ok


if __name__ == "__main__":
    print("=== CKKS ct_mul reference: round(dec/scale) == exact negacyclic m1*m2 ===")
    allok = True
    for N in (16, 32, 64):
        for dnum in (2, 3):
            for seed in (0, 3):
                allok &= _check(N=N, LQ=4, LP=2, dnum=dnum, seed=seed)
    print("\n=== deeper Q / more digits (dnum=3 is the hardware shape) ===")
    for LQ, dnum in ((6, 2), (6, 3), (8, 4)):
        allok &= _check(N=32, LQ=LQ, LP=2, dnum=dnum, seed=1)
    print("\n=== multi-prime rescale (ndrop=2) ===")
    pp = CKKSParams(N=32, LQ=6, LP=2, ndrop=2)      # Delta = the TWO primes the rescale drops
    rng = random.Random(5)
    s = small_poly(32, rng)
    evks, groups = keygen_relin_dnum(pp, s, rng, 3)
    m1 = [rng.randrange(-3, 4) for _ in range(32)]
    m2 = [rng.randrange(-3, 4) for _ in range(32)]
    out, Qo, sc = ct_mul(pp, encrypt(pp, m1, s, rng), encrypt(pp, m2, s, rng), s, rng,
                         ks=(evks, groups), ndrop=2)
    got = decrypt(pp, out, s, Qprod=Qo, scale=sc)
    want = _exact_product(m1, m2, 32)
    # ndrop=2 divides by an extra prime, so the scale no longer lands on Delta -- allow rounding slack
    err = max(abs(g - w) for g, w in zip(got, want))
    print(f"  ndrop=2 (Delta = 2 primes)  scale=2^{sc[0].bit_length()-sc[1].bit_length()}  "
          f"max|got-want| = {err}  -> {'OK' if err == 0 else 'FAIL'}")
    allok &= err == 0
    print("\nCKKS REFERENCE OK" if allok else "\nCKKS REFERENCE FAIL")
