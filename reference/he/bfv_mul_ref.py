# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Full BFV multiply — CPU big-int golden (ground truth for the device ct_mul assembly).

Exact-integer reference for the whole multiply pipeline the device must reproduce in RNS:

    ct_mul(a, b):
      1. tensor    (c0,c1)×(d0,d1) -> (e0,e1,e2)      [over Z, negacyclic]   [bfv_mul_raw]
      2. scale     ẽ_k = round(t·e_k / Q)  mod Q                             [scale kernel  ← NEW]
      3. relin     ẽ2 -> (k0,k1) ;  out = (ẽ0+k0, ẽ1+k1)                     [bfv_relin]
      decrypt(out) ?= m1·m2  (in R_t = Z_t[x]/(x^N+1))

Everything here is EXACT big-int (Python arbitrary precision) so it is the ground truth: the
device does the same in RNS (base Q, aux base B, sentinel m_sk) and is checked against this.

Two uses (see the harness plan):
  * full-system check:  decrypt(ct_mul) == m1·m2 mod t, and noise well below Q.
  * stage compare / injection: `ct_mul(..., want_stages=True)` returns the exact per-stage
    values, reducible to any RNS base via `rns_of`, to validate or feed each device stage.

Reuses the validated relin reference (keygen_relin / relinearize) unchanged.
"""

import random, math
from math import prod
from bfv_relin_ref import (
    is_prime, ntt_primes, negacyclic_mul, poly_add, poly_sub,
    crt_reconstruct, small_poly, err_poly, rand_poly, rns_of,
    keygen_relin, relinearize,
)

# --------------------------------------------------------------------------- #
# exact integer helpers (no modular reduction — the tensor lives over Z)
# --------------------------------------------------------------------------- #
def negacyclic_mul_Z(a, b):
    """a·b mod (x^N+1) over the integers (coefficients are exact big-ints)."""
    N = len(a); c = [0] * N
    for i in range(N):
        ai = a[i]
        if ai == 0: continue
        for j in range(N):
            k = i + j; v = ai * b[j]
            if k < N: c[k] += v
            else:     c[k - N] -= v
    return c

def poly_add_Z(a, b): return [x + y for x, y in zip(a, b)]

def round_div(n, d):
    """round(n/d) to nearest, half toward +inf, exact for integers (n may be negative)."""
    return (2 * n + d) // (2 * d)

def center(c, Q):
    """balanced representative in (-Q/2, Q/2]."""
    return [x - Q if x > Q // 2 else x for x in c]

# --------------------------------------------------------------------------- #
# parameters:  Q (ct modulus), P (relin special base), B (scale aux base), t
# --------------------------------------------------------------------------- #
class MulParams:
    def __init__(self, N=1024, LQ=4, LP=4, LB=4, t=12289):
        self.N, self.t = N, t
        pr = ntt_primes(LQ + LP + LB + 1, N)      # distinct NTT-friendly 28-bit primes
        self.Q = pr[:LQ]
        self.P = pr[LQ:LQ + LP]
        self.B = pr[LQ + LP:LQ + LP + LB]
        self.msk = pr[LQ + LP + LB]               # sentinel m_sk (ε recovery)
        self.Qprod, self.Pprod, self.Bprod = prod(self.Q), prod(self.P), prod(self.B)
        self.QP = self.Q + self.P                 # relin domain
        self.Delta = self.Qprod // t              # scale Δ = ⌊Q/t⌋

# --------------------------------------------------------------------------- #
# encrypt / decrypt  (symmetric-key BFV; encryption is only for test inputs)
# --------------------------------------------------------------------------- #
def encrypt(pp, m, s, rng):
    """c0 + c1·s = Δ·m + e  (mod Q).  Returns (c0,c1) as big-int polys in [0,Q)."""
    N, Q = pp.N, pp.Qprod
    a = rand_poly(N, Q, rng)
    e = err_poly(N, rng)
    a_s = negacyclic_mul(a, [x % Q for x in s], Q)
    c0 = [(pp.Delta * (m[n] % pp.t) - a_s[n] + e[n]) % Q for n in range(N)]
    return [c0, a]

def decrypt(pp, ct, s):
    """degree-1 ct=(c0,c1):  m = round(t/Q · [c0 + c1·s]_Q)  mod t."""
    N, Q, t = pp.N, pp.Qprod, pp.t
    v = poly_add(ct[0], negacyclic_mul(ct[1], [x % Q for x in s], Q), Q)
    return [round_div(t * vc, Q) % t for vc in center(v, Q)]

# --------------------------------------------------------------------------- #
# the multiply pipeline
# --------------------------------------------------------------------------- #
def tensor(pp, ct1, ct2):
    """(c0,c1)×(d0,d1) -> (e0,e1,e2) over Z (centered inputs; exact products)."""
    Q = pp.Qprod
    c0, c1 = center(ct1[0], Q), center(ct1[1], Q)
    d0, d1 = center(ct2[0], Q), center(ct2[1], Q)
    e0 = negacyclic_mul_Z(c0, d0)
    e1 = poly_add_Z(negacyclic_mul_Z(c0, d1), negacyclic_mul_Z(c1, d0))
    e2 = negacyclic_mul_Z(c1, d1)
    return [e0, e1, e2]

def scale(pp, e):
    """ẽ_k = round(t·e_k / Q)  mod Q   — the BFV rescale (exact big-int, ground truth)."""
    Q, t, N = pp.Qprod, pp.t, pp.N
    return [[round_div(t * ek[n], Q) % Q for n in range(N)] for ek in e]

# --------------------------------------------------------------------------- #
# device-faithful RNS scale (the 4->4 spec the device kernel implements).
# Works ONLY from residues + m_sk, exactly like the device: t·e -> exact FBC(Q->B)
# -> (t·e_B - FBC)·Q^{-1} in base B.  NB: at 4->4 this is NOT bit-exact to scale()
# above (ẽ can exceed base B); it is the reference the device kernel is checked to.
# --------------------------------------------------------------------------- #
def _fbc_exact_sk(res_src, src, dst, msk, x_msk):
    """exact FBC {x mod src_i} (+ x mod msk) -> {x mod dst_j}, ε recovered via m_sk."""
    S = prod(src); k = len(src)
    y = [res_src[i] * pow(S // src[i], -1, src[i]) % src[i] for i in range(k)]   # canonical y_i
    A_sk = sum(y[i] * ((S // src[i]) % msk) for i in range(k)) % msk              # (x+εS) mod msk
    eps = ((A_sk - x_msk) * pow(S % msk, -1, msk)) % msk                          # exact ε in [0,k)
    out = []
    for dj in dst:
        approx = sum(y[i] * ((S // src[i]) % dj) for i in range(k)) % dj          # (x+εS) mod dj
        out.append((approx - eps * (S % dj)) % dj)                               # − ε·S  -> exact
    return out

def _fbc_approx(res_src, src, dst):
    """approximate FBC {x mod src_i} -> {(x+εS) mod dst_j}  (drops ε; ε<len(src))."""
    S = prod(src); k = len(src)
    y = [res_src[i] * pow(S // src[i], -1, src[i]) % src[i] for i in range(k)]
    return [sum(y[i] * ((S // src[i]) % dj) for i in range(k)) % dj for dj in dst]

def rns_scale(pp, e_res, S, msk, round_bias=False):
    """DEVICE-FAITHFUL rescale of ONE poly (residues only; unsigned/canonical — no RNS centering,
       exactly what the device does with uint residues). S = aux base (holds ẽ), msk = SEPARATE
       sentinel (not in S). e_res = {'Q':[|Q| rows], 'S':[|S| rows], 'msk':[1 row]} = tensor residues.
         1. f = t·e (+⌊Q/2⌋ if round_bias) everywhere.
         2. Q -> S,msk  APPROXIMATE FBC of [f]_Q  (drops ε<|Q|; noise-absorbed); ẽ_x=(f_x-fbc)·Q^{-1}.
         3. S -> Q       EXACT FBC (m_sk) of ẽ (ẽ<∏S unsigned, so exact, no centering).
       Returns ẽ in base Q (|Q| rows)."""
    N, Q, t, Qp = pp.N, pp.Q, pp.t, pp.Qprod
    bias = (Qp // 2) if round_bias else 0
    dst = list(S) + [msk]
    Qinv = {x: pow(Qp % x, -1, x) for x in dst}
    tilde = {x: [0] * N for x in dst}                         # ẽ residues over S and msk
    for n in range(N):
        fQ = [((t * e_res['Q'][i][n] + bias) % Q[i]) for i in range(len(Q))]      # [f]_Q residues
        y  = [fQ[i] * pow(Qp // Q[i], -1, Q[i]) % Q[i] for i in range(len(Q))]
        fS = {S[js]: (t * e_res['S'][js][n] + bias) for js in range(len(S))}
        fS[msk] = (t * e_res['msk'][0][n] + bias)
        for x in dst:
            fbc = sum(y[i] * ((Qp // Q[i]) % x) for i in range(len(Q))) % x        # approx [f]_Q mod x
            tilde[x][n] = ((fS[x] % x - fbc) * Qinv[x]) % x                        # (f_x - fbc)·Q^{-1}
    # S -> Q, exact via m_sk sentinel (ẽ < ∏S since unsigned)
    outQ = [[0] * N for _ in range(len(Q))]
    for n in range(N):
        ex = _fbc_exact_sk([tilde[s][n] for s in S], list(S), Q, msk, tilde[msk][n])
        for i in range(len(Q)):
            outQ[i][n] = ex[i]
    return outQ

def tensor_unsigned(pp, ct1, ct2):
    """(c0,c1)×(d0,d1) -> (e0,e1,e2) over Z with UNSIGNED [0,Q) coeffs (device-faithful: the device
       works with canonical residues). e ∈ [0, N·Q²) — its residues mod any base are canonical."""
    c0, c1 = ct1[0], ct1[1]; d0, d1 = ct2[0], ct2[1]
    e0 = negacyclic_mul_Z(c0, d0)
    e1 = poly_add_Z(negacyclic_mul_Z(c0, d1), negacyclic_mul_Z(c1, d0))
    e2 = negacyclic_mul_Z(c1, d1)
    return [e0, e1, e2]

def rns_ct_mul(pp, ct1, ct2, evk, rng, S, msk):
    """DEVICE-FAITHFUL full multiply: unsigned tensor -> per-poly rns_scale (Q=3, S aux, m_sk)
       -> relin. Every step is what the device does from residues. Returns (out0,out1) in [0,Q)."""
    e = tensor_unsigned(pp, ct1, ct2)
    def rows(poly, base): return [[c % b for c in poly] for b in base]
    es = []
    for ek in e:
        er = {'Q': rows(ek, pp.Q), 'S': rows(ek, S), 'msk': rows(ek, [msk])}
        outQ = rns_scale(pp, er, S, msk)                     # ẽ_k in base Q (|Q| rows)
        es.append([crt_reconstruct([outQ[i][n] for i in range(len(pp.Q))], pp.Q) for n in range(pp.N)])
    k0, k1 = relinearize(pp, rns_of(es[2], pp.Q), evk[0], evk[1], rng)
    k0i = [crt_reconstruct([k0[j][n] for j in range(len(pp.Q))], pp.Q) for n in range(pp.N)]
    k1i = [crt_reconstruct([k1[j][n] for j in range(len(pp.Q))], pp.Q) for n in range(pp.N)]
    return [poly_add(es[0], k0i, pp.Qprod), poly_add(es[1], k1i, pp.Qprod)]

def rns_ct_add(pp, ct1, ct2):
    """DEVICE-FAITHFUL HE add: per-limb modadd in base Q (what bfv_add does)."""
    return [ [ (ct1[k][n] + ct2[k][n]) % pp.Qprod for n in range(pp.N) ] for k in range(2) ]

def ct_mul(pp, ct1, ct2, evk, rng, want_stages=False):
    """full BFV multiply: tensor -> scale -> relin.  Returns degree-1 (out0,out1) in [0,Q)."""
    e  = tensor(pp, ct1, ct2)                     # (e0,e1,e2) over Z
    es = scale(pp, e)                             # (ẽ0,ẽ1,ẽ2) mod Q
    e2_Q = rns_of(es[2], pp.Q)
    k0, k1 = relinearize(pp, e2_Q, evk[0], evk[1], rng)
    k0i = [crt_reconstruct([k0[j][n] for j in range(len(pp.Q))], pp.Q) for n in range(pp.N)]
    k1i = [crt_reconstruct([k1[j][n] for j in range(len(pp.Q))], pp.Q) for n in range(pp.N)]
    out = [poly_add(es[0], k0i, pp.Qprod), poly_add(es[1], k1i, pp.Qprod)]
    if want_stages:
        return out, {"tensor": e, "scaled": es, "k": (k0i, k1i), "out": out}
    return out

# --------------------------------------------------------------------------- #
# validation: decrypt(ct1·ct2) == m1·m2 in R_t, and noise well below Q
# --------------------------------------------------------------------------- #
def _test(N=256, t=12289, seed=0):
    rng = random.Random(seed)
    pp = MulParams(N=N, t=t)
    print(f"N={N}  t={t}  log2(Q)={math.log2(pp.Qprod):.1f}  Δ={math.log2(pp.Delta):.1f} bits  "
          f"Q={pp.Q}\n  B={pp.B}  P={pp.P}")

    s = small_poly(N, rng)
    evk = keygen_relin(pp, s, rng)

    m1 = [rng.randrange(t) for _ in range(N)]
    m2 = [rng.randrange(t) for _ in range(N)]
    ct1, ct2 = encrypt(pp, m1, s, rng), encrypt(pp, m2, s, rng)

    # sanity: fresh encryptions decrypt
    assert decrypt(pp, ct1, s) == m1 and decrypt(pp, ct2, s) == m2, "fresh decrypt failed"

    out = ct_mul(pp, ct1, ct2, evk, rng)
    got = decrypt(pp, out, s)
    want = negacyclic_mul(m1, m2, t)              # m1·m2 in R_t

    mism = sum(g != w for g, w in zip(got, want))
    print(f"decrypt(ct1·ct2) vs m1·m2 mod t:  mismatches = {mism}/{N}")

    # noise margin: v = [c0+c1·s]_Q = Δ·m + noise (mod Q); measure ||noise||∞ centered.
    Q = pp.Qprod
    v = poly_add(out[0], negacyclic_mul(out[1], [x % Q for x in s], Q), Q)
    noise = 0
    for n in range(N):
        r = (v[n] - pp.Delta * want[n]) % Q       # residual off the clean Δ·(m1·m2)
        noise = max(noise, min(r, Q - r))         # centered magnitude
    ln = math.log2(noise) if noise else 0
    print(f"  log2(||noise||_inf)={ln:.1f}  margin to Q = {math.log2(Q) - ln:.1f} bits")
    ok = (mism == 0)
    print(f"  [{'PASS' if ok else 'FAIL'}] full BFV multiply {'decrypts to m1·m2' if ok else 'BROKEN'}")
    return ok

def e2e_verify(N=1024, t=12289, seeds=(0, 1, 7, 42), LQ=3):
    """END-TO-END HE verification (device-faithful RNS, Q=3):
       encrypt m1,m2 -> HE add & HE mul -> decrypt -> check == m1+m2 / m1·m2 (mod t)."""
    from bfv_relin_ref import ntt_primes
    ok = True
    for seed in seeds:
        rng = random.Random(seed)
        pp = MulParams(N=N, LQ=LQ, LP=LQ, LB=4, t=t)
        S = pp.B                                             # aux base (4 limbs, holds ẽ)
        msk = [p for p in ntt_primes(80, N) if p not in set(pp.Q + pp.P + pp.B)][0]
        s = small_poly(N, rng); evk = keygen_relin(pp, s, rng)
        m1 = [rng.randrange(t) for _ in range(N)]; m2 = [rng.randrange(t) for _ in range(N)]
        ct1, ct2 = encrypt(pp, m1, s, rng), encrypt(pp, m2, s, rng)

        add = decrypt(pp, rns_ct_add(pp, ct1, ct2), s)
        want_add = [(a + b) % t for a, b in zip(m1, m2)]
        mul = decrypt(pp, rns_ct_mul(pp, ct1, ct2, evk, rng, S, msk), s)
        want_mul = negacyclic_mul(m1, m2, t)

        ma = sum(x != y for x, y in zip(add, want_add))
        mm = sum(x != y for x, y in zip(mul, want_mul))
        ok &= (ma == 0 and mm == 0)
        print(f"seed={seed:>2}: HE add mism={ma}/{N}   HE mul mism={mm}/{N}   "
              f"[{'PASS' if ma==0 and mm==0 else 'FAIL'}]")
    print(f"log2(Q)={math.log2(pp.Qprod):.0f}  N={N}  t={t}  "
          f"=> encrypt→(add,mul)→decrypt {'ALL PASS' if ok else 'FAILED'}")
    return ok

if __name__ == "__main__":
    print("=== per-op golden ==="); _test()
    print("\n=== end-to-end HE verification (encrypt → add & mul → decrypt) ===")
    e2e_verify()
