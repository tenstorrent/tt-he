# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""RNS-BFV relinearization / key-switch — CPU reference (golden for the device assembly).

  VALIDATED: the key-switch identity  k0+k1·s ≡ c2·s² (mod Q)  holds with ~104-bit noise margin
  (log2 Q = 112) across seeds. Subtle correctness point learned here: the key-switch key's error
  e must be ONE small integer polynomial reduced to each RNS limb — sampling e independently per
  limb makes its CRT reconstruction ≈ QP (huge), which destroys the noise bound. The mask a, by
  contrast, IS sampled independently per limb (it cancels in k0+k1·s). dnum=1 with P≈Q suffices
  for the identity at these params.



Proves the ALGORITHM the device blocks compose into: hybrid (special-modulus) key-switch of
the degree-2 term c2 back to (k0,k1) under s, validated by the key-switch identity

    k0 + k1·s  ≡  c2·s²   (mod Q)      up to key-switch noise.

Every step here is exactly a device-validated primitive (base conversion, negacyclic
poly-mult, mod-down), so this reference doubles as the assembly spec:

  relinearize(c0,c1,c2):
    1. mod-up   c2 : Q -> QP           (approximate FBC(Q->P), concat)          [baseconv_fbc]
    2. c2_qp · evk (poly-mult in QP)                                            [nNTT / ⊙ / niNTT]
    3. mod-down each accumulator : QP -> Q                                      [bfv_moddown]
    4. (c0,c1) += (k0,k1)                                                       [ct_add]

Scope matches docs/he/bfv_prototype_plan.md: N=1024, Q=4 / P=4 limbs, dnum=1 (mod-up all of Q
to QP at once, one evk pair). All arithmetic is exact big-int here (the device does it in RNS).
"""

import random
from math import prod

# ---------------------------------------------------------------------------
# ring / modulus helpers (negacyclic  Z_q[x]/(x^N+1))
# ---------------------------------------------------------------------------
def is_prime(n):
    if n < 2: return False
    d = 2
    while d * d <= n:
        if n % d == 0: return False
        d += 1
    return True

def ntt_primes(cnt, N):
    """cnt primes p = k·2N + 1 (2N | p−1, so both NTT and negacyclic ψ exist), p < 2^28."""
    out, k = [], ((1 << 28) - 1) // (2 * N)
    while len(out) < cnt and k > 0:
        p = k * 2 * N + 1
        if p < (1 << 28) and is_prime(p): out.append(p)
        k -= 1
    return out

def negacyclic_mul(a, b, q):
    """a·b mod (x^N+1, q)  — the ring multiplication."""
    N = len(a); c = [0] * N
    for i in range(N):
        if a[i] == 0: continue
        for j in range(N):
            k = i + j; v = a[i] * b[j] % q
            if k < N: c[k] = (c[k] + v) % q
            else:     c[k - N] = (c[k - N] - v) % q
    return c

def poly_add(a, b, q): return [(x + y) % q for x, y in zip(a, b)]
def poly_sub(a, b, q): return [(x - y) % q for x, y in zip(a, b)]
def poly_smul(a, s, q): return [x * s % q for x in a]

# ---------------------------------------------------------------------------
# RNS <-> integer (per coefficient), and fast base conversion (approximate)
# ---------------------------------------------------------------------------
def crt_reconstruct(residues, base):
    """CRT: residues[i] mod base[i]  ->  integer in [0, prod(base))."""
    B = prod(base); x = 0
    for bi, ri in zip(base, residues):
        Bi = B // bi
        x += ri * Bi * pow(Bi, -1, bi)
    return x % B

def fast_base_convert_approx(x_res, src, dst):
    """Approximate FBC: {x mod src_i} -> {(x+εS) mod dst_j}  (drops the ε overflow)."""
    S = prod(src)
    y = [x_res[i] * pow(S // src[i], -1, src[i]) % src[i] for i in range(len(src))]
    return [sum(y[i] * ((S // src[i]) % dj) for i in range(len(src))) % dj for dj in dst]

# ---------------------------------------------------------------------------
# BFV key-switch (hybrid / special modulus P), dnum = 1
# ---------------------------------------------------------------------------
class Params:
    def __init__(self, N=1024, LQ=4, LP=4, seed=0):
        self.N = N
        pr = ntt_primes(LQ + LP, N)
        self.Q = pr[:LQ]; self.P = pr[LQ:LQ + LP]
        self.Qprod = prod(self.Q); self.Pprod = prod(self.P)
        self.QP = self.Q + self.P

def rand_poly(N, q, rng): return [rng.randrange(q) for _ in range(N)]
def small_poly(N, rng):   return [rng.choice((-1, 0, 1)) for _ in range(N)]           # ternary
def err_poly(N, rng):     return [rng.choice((-2, -1, 0, 1, 2)) for _ in range(N)]     # tiny noise

def rns_of(poly, base):   return [[c % b for c in poly] for b in base]                 # per-limb residues

def keygen_relin(pp, s, rng):
    """evk that key-switches s^2 -> s, in base QP:  evk = (−a·s + e + P·s², a) mod QP.
       Decrypt(evk) = P·s²  (mod QP)  so ⟨c2_qp, evk⟩ ≈ P·c2·s², removed by mod-down (÷P)."""
    N, QP, P = pp.N, pp.QP, pp.Pprod
    s2 = negacyclic_mul(s, s, prod(QP))                # s²: ONE small integer poly, reduced per limb
    e_int = err_poly(N, rng)                           # error: ONE small integer poly (NOT per-limb!)
    evk0 = [None] * len(QP); evk1 = [None] * len(QP)
    for li, m in enumerate(QP):
        a  = rand_poly(N, m, rng)                      # mask: independent per limb (cancels) — correct
        e  = [x % m for x in e_int]                    # same small e reduced to this limb
        sl = [x % m for x in s]
        s2l = [x % m for x in s2]
        Pm = P % m
        as_ = negacyclic_mul(a, sl, m)
        evk0[li] = [(-as_[n] + e[n] + Pm * s2l[n]) % m for n in range(N)]   # −a·s + e + P·s²
        evk1[li] = a
    return evk0, evk1

def relinearize(pp, c2_Q, evk0, evk1, rng):
    """c2 given in base Q (per-limb). Returns (k0,k1) in base Q via hybrid key-switch."""
    N, Q, P, QP = pp.N, pp.Q, pp.P, pp.QP
    LQ, LP = len(Q), len(P)

    # 1. mod-up c2 : Q -> QP   (Q-part kept, P-part via approximate FBC(Q->P))
    c2_QP = [None] * len(QP)
    for i in range(LQ): c2_QP[i] = c2_Q[i][:]
    # per coefficient, base-convert the Q-residues to the P moduli
    fbc = [fast_base_convert_approx([c2_Q[i][n] for i in range(LQ)], Q, P) for n in range(N)]
    for j in range(LP): c2_QP[LQ + j] = [fbc[n][j] for n in range(N)]

    # 2. poly-mult in QP:  acc = c2_qp · evk   (per limb, negacyclic)   [dnum=1: single group]
    acc0 = [negacyclic_mul(c2_QP[li], evk0[li], QP[li]) for li in range(len(QP))]
    acc1 = [negacyclic_mul(c2_QP[li], evk1[li], QP[li]) for li in range(len(QP))]

    # 3. mod-down each accumulator : QP -> Q     y_i = (x_i − FBC(x_P)_i)·P⁻¹ mod q_i
    def moddown(acc):
        aQ = acc[:LQ]; aP = acc[LQ:]
        out = [None] * LQ
        for j in range(LQ):
            qj = Q[j]; Pinv = pow(P and pp.Pprod % qj or 0, -1, qj) if False else pow(pp.Pprod % qj, -1, qj)
            xPtoQ = [fast_base_convert_approx([aP[i][n] for i in range(LP)], P, [qj])[0] for n in range(N)]
            out[j] = [((aQ[j][n] - xPtoQ[n]) * Pinv) % qj for n in range(N)]
        return out
    return moddown(acc0), moddown(acc1)

# ---------------------------------------------------------------------------
# validation: the key-switch identity  k0 + k1·s ≡ c2·s²  (mod Q)  up to noise
# ---------------------------------------------------------------------------
def _test():
    rng = random.Random(0)
    pp = Params(N=1024, LQ=4, LP=4)
    N, Q = pp.N, pp.Q
    print(f"N={N}  Q={pp.Q}  P={pp.P}")

    s = small_poly(N, rng)
    evk0, evk1 = keygen_relin(pp, s, rng)

    # a random degree-2 term c2 (in base Q)
    c2 = rand_poly(N, pp.Qprod, rng)              # as a big-int poly < Qprod
    c2_Q = rns_of(c2, Q)

    k0, k1 = relinearize(pp, c2_Q, evk0, evk1, rng)

    # reconstruct k0,k1 to big-int (base Q), and s; check  k0 + k1·s ≡ c2·s²  (mod Q)
    k0i = [crt_reconstruct([k0[j][n] for j in range(len(Q))], Q) for n in range(N)]
    k1i = [crt_reconstruct([k1[j][n] for j in range(len(Q))], Q) for n in range(N)]
    lhs = poly_add(k0i, negacyclic_mul(k1i, [x % pp.Qprod for x in s], pp.Qprod), pp.Qprod)
    rhs = negacyclic_mul(c2, negacyclic_mul([x % pp.Qprod for x in s], [x % pp.Qprod for x in s], pp.Qprod), pp.Qprod)

    # key-switch is approximate: lhs − rhs is small NOISE (not exactly 0). Measure ||·||∞ vs Q.
    diff = poly_sub(lhs, rhs, pp.Qprod)
    def centered(v, q): return v - q if v > q // 2 else v
    ninf = max(abs(centered(d, pp.Qprod)) for d in diff)
    import math
    logQ = math.log2(pp.Qprod); lognoise = math.log2(ninf) if ninf else 0
    ok = lognoise < logQ - 20          # noise must be far below Q for a correct key-switch
    print(f"key-switch identity  k0+k1·s ≡ c2·s² (mod Q):")
    print(f"  log2(Q) = {logQ:.1f}   log2(||noise||_inf) = {lognoise:.1f}   margin = {logQ - lognoise:.1f} bits")
    print(f"  [{'PASS' if ok else 'FAIL'}] noise is {'well below' if ok else 'NOT below'} Q → relinearization {'preserves decryption' if ok else 'BROKEN'}")
    return ok

if __name__ == "__main__":
    _test()
