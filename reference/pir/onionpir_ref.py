#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""OnionPIR v2 — CPU big-int reference (the golden for the on-device build).

Ground-truth, exact-integer implementation of every NEW primitive the PIR device path needs
(see docs/pir/history/pir_kernel_plan.md).  The device reproduces these in RNS / NTT and is validated
bit-exact against the per-stage values here, exactly like reference/he/bfv_mul_ref.py is the
golden for the BFV multiply.

Scope = the buckets B/C kernels of the plan, as CPU goldens:
    * signed gadget decomposition            (#9  gadget_decompose_signed)
    * RGSW type + external product           (#16, #17)
    * Galois automorphism + BV key-switch    (#14, #10)  -> Subs (#3)
    * monomial multiply x^{±k}               (#15)
    * query expansion / ExpandBFV            (#18)
    * hypercube database fold                 (#19, initial dim #12 + folds)
    * end-to-end retrieve                      (the whole pipeline)

Arithmetic is plain big-int mod a COMPOSITE q = q1*q2 (two ~28-bit NTT-friendly primes,
q1,q2 ≡ 1 mod 2n) — matching the paper's composite modulus and the device's 28-bit RNS limbs.
No NTT here: negacyclic multiply is schoolbook (this is the golden, not the fast path).

    python3 onionpir_ref.py            # run the self-test suite (prints PASS/FAIL per stage)
    python3 onionpir_ref.py -v         # verbose (noise budgets)
"""

import random
import sys

try:
    import numpy as _np                       # optional accel; falls back to pure python if absent
except ImportError:
    _np = None

_QFACT = {}   # q -> [prime list]: lets negacyclic_mul run per-prime (RNS) + numpy (see Params)

# --------------------------------------------------------------------------- #
# number theory helpers
# --------------------------------------------------------------------------- #
def is_prime(x):
    if x < 2:
        return False
    for d in range(2, int(x ** 0.5) + 1):
        if x % d == 0:
            return False
    return True

def ntt_primes(count, n, bits=28):
    """`count` primes p ≡ 1 (mod 2n), just below 2**bits, descending (NTT-friendly)."""
    twon = 2 * n
    k = ((1 << bits) - 1) // twon
    out = []
    while k > 0 and len(out) < count:
        p = k * twon + 1
        if p < (1 << bits) and is_prime(p):
            out.append(p)
        k -= 1
    if len(out) < count:
        raise RuntimeError("not enough NTT primes")
    return out

def modinv(a, m):
    return pow(a % m, -1, m)

# --------------------------------------------------------------------------- #
# ring R_q = Z_q[x]/(x^n + 1)  (schoolbook, big-int)
# --------------------------------------------------------------------------- #
def poly_add(a, b, q): return [(x + y) % q for x, y in zip(a, b)]
def poly_sub(a, b, q): return [(x - y) % q for x, y in zip(a, b)]

def _negconv_mod_np(a, b, p):
    """Negacyclic convolution mod prime p (p < 2^31), exact int64 via 14-bit-split np.convolve.
    Splitting keeps every partial product < 2^28 so the length-n sums never overflow int64."""
    n = a.shape[0]
    HALF = 1 << 14
    a = a % p
    b = b % p
    alo = a & (HALF - 1); ahi = a >> 14
    blo = b & (HALF - 1); bhi = b >> 14
    cll = _np.convolve(alo, blo) % p                              # 14x14 products, sum<2^39
    clh = (_np.convolve(alo, bhi) + _np.convolve(ahi, blo)) % p
    chh = _np.convolve(ahi, bhi) % p
    w14 = HALF % p; w28 = (HALF * HALF) % p
    full = (cll + (clh * w14) % p + (chh * w28) % p) % p          # each term reduced -> no overflow
    c = full[:n].copy()
    c[:n - 1] = (c[:n - 1] - full[n:2 * n - 1]) % p               # negacyclic fold: c[k]=full[k]-full[k+n]
    return c % p

def negacyclic_mul(a, b, q):
    # Fast path: RNS per prime (composite q) or single-modulus (small q), vectorized with numpy.
    if _np is not None and q in _QFACT:
        primes = _QFACT[q]
        if len(primes) == 2 and q < (1 << 62):                    # K=2 (~56-bit): CRT fits int64 (fast path)
            aw = _np.asarray(a, dtype=_np.int64); bw = _np.asarray(b, dtype=_np.int64)
            q1, q2 = primes; q1inv = modinv(q1, q2)
            c1 = _negconv_mod_np(aw, bw, q1); c2 = _negconv_mod_np(aw, bw, q2)
            tt = ((c2 - c1) * q1inv) % q2                         # c ≡ c1 (q1), c2 (q2)
            return ((c1 + q1 * tt) % q).tolist()
        # K>=3: q exceeds int64. Reduce operands mod each prime in Python FIRST (coeffs are >int64),
        # per-prime conv in int64, then Garner CRT in Python bigints (O(n·K)).
        def red(vec, p): return _np.fromiter((int(x) % p for x in vec), dtype=_np.int64, count=len(vec))
        x = None; radix = 1
        for p in primes:
            ci = _negconv_mod_np(red(a, p), red(b, p), p).astype(object)
            if x is None:
                x = ci
            else:
                inv = modinv(radix % p, p)
                x = x + radix * (((ci - x) * inv) % p)
            radix *= p
        return (x % q).tolist()
    if _np is not None and q < (1 << 31):
        aw = _np.asarray(a, dtype=_np.int64); bw = _np.asarray(b, dtype=_np.int64)
        return _negconv_mod_np(aw, bw, q).tolist()
    # Pure-python fallback (exact, slow) — the original schoolbook reference.
    n = len(a)
    c = [0] * n
    for i in range(n):
        ai = a[i]
        if ai == 0:
            continue
        for j in range(n):
            k = i + j
            v = ai * b[j]
            if k < n:
                c[k] += v
            else:
                c[k - n] -= v
    return [x % q for x in c]

def center(c, q):
    """balanced representative in (-q/2, q/2]."""
    return [(x % q) - q if (x % q) > q // 2 else (x % q) for x in c]

def rand_poly(n, q, rng):  return [rng.randrange(q) for _ in range(n)]
def small_poly(n, rng):    return [rng.choice((-1, 0, 1)) for _ in range(n)]        # ternary secret
def err_poly(n, rng):      return [rng.choice((-1, 0, 1)) for _ in range(n)]        # tiny noise

def inf_norm(c, q):
    return max(abs(v) for v in center(c, q))

# --------------------------------------------------------------------------- #
# #9  signed gadget decomposition  G^{-1}(v) = [d_0..d_{ell-1}],  Σ d_k B^k ≡ v (mod q)
#     digits balanced in [-B/2, B/2)
# --------------------------------------------------------------------------- #
def gadget_decompose_signed(poly, B, ell, q):
    """Return ell digit-polys; each coeff in [-B/2, B/2); Σ_k digit_k · B^k ≡ poly (mod q)."""
    halfB = B // 2
    digits = [[0] * len(poly) for _ in range(ell)]
    for i, v in enumerate(poly):
        x = v % q
        for k in range(ell):
            d = x % B
            if d >= halfB:
                d -= B
            digits[k][i] = d
            x = (x - d) // B
    return digits

def gadget_vector(B, ell, q):
    return [pow(B, k, q) for k in range(ell)]

# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
class Params:
    def __init__(self, n=256, t=257, B=256, ell=8, K=2, seed=0):
        self.n, self.t = n, t
        self.B, self.ell = B, ell
        self.K = K                                 # RNS limb count (2 -> ~56-bit, 3 -> ~84-bit, ...)
        primes = ntt_primes(K, n)                  # K distinct 28-bit NTT-friendly primes
        self.primes = primes
        self.q1, self.q2 = primes[0], primes[1]    # kept for the device path (2-limb gen scripts)
        self.q = 1
        for p in primes:
            self.q *= p                            # composite ct modulus = Π primes
        self.delta = self.q // t                   # BFV scaling Δ = ⌊q/t⌋
        _QFACT[self.q] = list(primes)              # enable RNS+numpy fast mul (any K)
        # balanced base-B digits span [-B^ell/2, B^ell/2)  ->  need B^ell >= 2q
        assert B ** ell >= 2 * self.q, "gadget too short for q (need B^ell >= 2q for balanced digits)"

    def __repr__(self):
        return (f"Params(n={self.n}, t={self.t}, q=Π{self.K}primes={self.q} (~{self.q.bit_length()}b), "
                f"B={self.B}, ell={self.ell})")

# --------------------------------------------------------------------------- #
# BFV (RLWE) — the EASY ops PIR uses (keygen, encrypt, decrypt, add, plaintext×ct)
#   ct = (c0, c1),   decrypt: c0 + c1·s = Δ·m + e   (mod q)
# --------------------------------------------------------------------------- #
def keygen(pp, rng):
    return small_poly(pp.n, rng)                    # secret key s (ternary)

def encrypt(pp, s, m, rng):
    n, q = pp.n, pp.q
    a = rand_poly(n, q, rng)
    e = err_poly(n, rng)
    c0 = poly_add(poly_sub([0] * n, negacyclic_mul(a, s, q), q),
                  poly_add(e, [pp.delta * (mi % pp.t) for mi in m], q), q)
    return [c0, a]                                  # c1 = a

def decrypt(pp, s, ct):
    q, t = pp.q, pp.t
    phase = poly_add(ct[0], negacyclic_mul(ct[1], s, q), q)    # Δm + e
    out = []
    for v in center(phase, q):
        out.append(round(v / pp.delta) % t)
    return out

def encrypt_scaleless(pp, s, m, rng):
    """RLWE with the message in the RAW phase (no Δ):  c0 + c1·s = m + e  (mod q).
    RGSW rows live in this un-scaled domain (rgsw_encrypt adds m·g_k without Δ), so the
    query's GSW-destined coefficients are encoded here, not Δ-scaled like a BFV plaintext."""
    n, q = pp.n, pp.q
    a = rand_poly(n, q, rng)
    e = err_poly(n, rng)
    c0 = poly_add(poly_sub([0] * n, negacyclic_mul(a, s, q), q),
                  poly_add(e, [mi % q for mi in m], q), q)
    return [c0, a]                                  # c1 = a

def ct_add(pp, a, b): return [poly_add(a[0], b[0], pp.q), poly_add(a[1], b[1], pp.q)]
def ct_sub(pp, a, b): return [poly_sub(a[0], b[0], pp.q), poly_sub(a[1], b[1], pp.q)]

def pt_ct_mul(pp, pt, ct):
    """plaintext × ciphertext (the initial-dimension dot-product op). pt is a plaintext poly."""
    q = pp.q
    return [negacyclic_mul(pt, ct[0], q), negacyclic_mul(pt, ct[1], q)]

def noise_budget(pp, s, ct):
    """log2(q / |e|) headroom; decrypt fails when this hits ~log2(2t)."""
    q = pp.q
    phase = poly_add(ct[0], negacyclic_mul(ct[1], s, q), q)
    m = decrypt(pp, s, ct)
    exact = [pp.delta * mi for mi in m]
    e = inf_norm(poly_sub(phase, exact, q), q)
    e = max(e, 1)
    import math
    return math.log2(q / (2 * e))

# --------------------------------------------------------------------------- #
# #11 modulus switch — shrink the response modulus q -> q' (runs once, on one ct)
#   c'_i = round(q'·c_i / q) mod q'    (scales the phase Δm+e by q'/q -> Δ'm+e', Δ'=q'/t)
# --------------------------------------------------------------------------- #
def mod_switch(pp, ct, qprime):
    q = pp.q
    def rs(poly): return [round(qprime * (c % q) / q) % qprime for c in poly]
    return [rs(ct[0]), rs(ct[1])]

def decrypt_modswitched(pp, s, ct, qprime):
    """decrypt a mod-switched ct (modulus q', scaling Δ'=q'/t)."""
    phase = poly_add(ct[0], negacyclic_mul(ct[1], s, qprime), qprime)
    return [round(v * pp.t / qprime) % pp.t for v in center(phase, qprime)]

# --------------------------------------------------------------------------- #
# #17 RGSW type + #16 external product
#   RGSW(m) = 2ℓ×2 grid; each row an RLWE(0) plus m·g_k in one column.
#     rows 0..ℓ-1 : (−a s + e + m g_k , a)          (message in col 0)
#     rows ℓ..2ℓ-1: (−a s + e         , a + m g_k)  (message in col 1)
# --------------------------------------------------------------------------- #
def rgsw_encrypt(pp, s, m, rng):
    n, q, B, ell = pp.n, pp.q, pp.B, pp.ell
    g = gadget_vector(B, ell, q)
    rows = []
    for col in (0, 1):
        for k in range(ell):
            a = rand_poly(n, q, rng)
            e = err_poly(n, rng)
            b = poly_add(poly_sub([0] * n, negacyclic_mul(a, s, q), q), e, q)
            mg = [(mi * g[k]) % q for mi in m]
            if col == 0:
                rows.append([poly_add(b, mg, q), a])
            else:
                rows.append([b, poly_add(a, mg, q)])
    return rows                                     # 2ℓ rows, each [col0, col1]

def external_product(pp, C, d):
    """RGSW(m) ⊡ RLWE(μ) -> RLWE(m·μ).  d=(d0,d1); returns (out0,out1)."""
    n, q, B, ell = pp.n, pp.q, pp.B, pp.ell
    u = gadget_decompose_signed(d[0], B, ell, q)    # G^{-1}(d0), ℓ digit polys
    w = gadget_decompose_signed(d[1], B, ell, q)    # G^{-1}(d1)
    digits = u + w                                   # 2ℓ digits, aligns with rows
    out0 = [0] * n
    out1 = [0] * n
    for r in range(2 * ell):
        out0 = poly_add(out0, negacyclic_mul(digits[r], C[r][0], q), q)
        out1 = poly_add(out1, negacyclic_mul(digits[r], C[r][1], q), q)
    return [out0, out1]

def mux_1of2(pp, C_bit, x, y):
    """b·(y−x)+x via one external product: returns x if bit=0, y if bit=1."""
    return ct_add(pp, external_product(pp, C_bit, ct_sub(pp, y, x)), x)

# --------------------------------------------------------------------------- #
# #15 monomial multiply  ct · x^k   (negacyclic coefficient shift with sign)
# --------------------------------------------------------------------------- #
def mono_mul_poly(poly, k, q):
    n = len(poly)
    twon = 2 * n
    r = [0] * n
    for i in range(n):
        p = (i + k) % twon
        if p < n:
            r[p] = (r[p] + poly[i]) % q
        else:
            r[p - n] = (r[p - n] - poly[i]) % q
    return r

def mono_mul(pp, ct, k):
    return [mono_mul_poly(ct[0], k, pp.q), mono_mul_poly(ct[1], k, pp.q)]

# --------------------------------------------------------------------------- #
# #14 Galois automorphism  x -> x^j   (index permute i↦i·j mod 2n, negacyclic sign)
# --------------------------------------------------------------------------- #
def automorph_poly(poly, j, q):
    n = len(poly)
    twon = 2 * n
    r = [0] * n
    for i in range(n):
        p = (i * j) % twon
        if p < n:
            r[p] = (r[p] + poly[i]) % q
        else:
            r[p - n] = (r[p - n] - poly[i]) % q
    return r

# --------------------------------------------------------------------------- #
# #10 BV-style key-switch  +  #3 Subs = automorphism then key-switch back to s
#   KSK_j encrypts s(x^j) under s, in gadget form.
# --------------------------------------------------------------------------- #
def gen_ksk(pp, s, s_from, rng):
    """gadget encryption of s_from under s: rows k = (−a s + e + s_from·B^k , a)."""
    n, q, B, ell = pp.n, pp.q, pp.B, pp.ell
    g = gadget_vector(B, ell, q)
    ksk = []
    for k in range(ell):
        a = rand_poly(n, q, rng)
        e = err_poly(n, rng)
        b = poly_add(poly_sub([0] * n, negacyclic_mul(a, s, q), q), e, q)
        sg = [(si * g[k]) % q for si in s_from]
        ksk.append([poly_add(b, sg, q), a])
    return ksk

def keyswitch(pp, c0, c1, ksk):
    """(c0,c1) valid under s_from -> ct under s (same message)."""
    n, q, B, ell = pp.n, pp.q, pp.B, pp.ell
    w = gadget_decompose_signed(c1, B, ell, q)
    d0 = list(c0)
    d1 = [0] * n
    for k in range(ell):
        d0 = poly_add(d0, negacyclic_mul(w[k], ksk[k][0], q), q)
        d1 = poly_add(d1, negacyclic_mul(w[k], ksk[k][1], q), q)
    return [d0, d1]

def gen_galois_key(pp, s, j, rng):
    return gen_ksk(pp, s, automorph_poly(s, j, pp.q), rng)

def subs(pp, ct, j, galois_key):
    """apply x->x^j to the message, staying under key s."""
    q = pp.q
    c0j = automorph_poly(ct[0], j, q)
    c1j = automorph_poly(ct[1], j, q)
    return keyswitch(pp, c0j, c1j, galois_key)

# --------------------------------------------------------------------------- #
# #18 query expansion (ExpandBFV / Angel-Chen-Laine-Setty oblivious expansion)
#   one packed ct Enc(Σ a_i x^i)  ->  size = 2^m_exp cts, cts[i] = Enc(a_i)
#   client pre-scales the packed coeffs by (2^m_exp)^{-1} mod t to undo the ×2^m_exp.
# --------------------------------------------------------------------------- #
def expand_query(pp, ct, m_exp, galois_keys):
    """galois_keys: dict j -> key, for j = n/2^a + 1, a=0..m_exp-1."""
    n = pp.n
    cts = [ct]
    for a in range(m_exp):
        j = (n >> a) + 1
        gk = galois_keys[j]
        new = [None] * (2 * len(cts))
        for b in range(len(cts)):
            c = cts[b]
            c_sub = subs(pp, c, j, gk)
            new[b] = ct_add(pp, c, c_sub)                       # keeps even-indexed half
            c_sh = mono_mul(pp, c, -(1 << a))                   # ct · x^{-2^a}
            c_sh_sub = subs(pp, c_sh, j, gk)
            new[b + len(cts)] = ct_add(pp, c_sh, c_sh_sub)      # keeps odd-indexed half
        cts = new
    return cts[:1 << m_exp]

def galois_keys_for_expand(pp, s, m_exp, rng):
    keys = {}
    for a in range(m_exp):
        j = (pp.n >> a) + 1
        keys[j] = gen_galois_key(pp, s, j, rng)
    return keys

# --------------------------------------------------------------------------- #
# #18b QueryUnpack — RLWE -> RGSW conversion  (OnionPIRv2 GSWEval::query_to_gsw)
#   A fold bit, after expansion, is ℓ RLWE "message rows" with phase m·B^k.
#   RGSW(m) top ℓ rows = those message rows as-is;
#            bottom ℓ rows = RGSW(s) ⊡ (message row)  -> phase m·s·B^k.
#   RGSW(s) is one-time client key material (uploaded once, amortized over all queries).
# --------------------------------------------------------------------------- #
def gen_gsw_key(pp, s, rng):
    """RGSW(s): RGSW encryption of the secret key — the bridge that supplies the s·(·) rows."""
    return rgsw_encrypt(pp, s, s, rng)

def rlwe_to_rgsw(pp, msg_rows, gsw_key):
    """ℓ message rows (phase m·B^k)  ->  full RGSW(m) (2ℓ rows), one external product per row."""
    rows = [[list(r[0]), list(r[1])] for r in msg_rows]          # top ℓ: message rows as-is
    for r in msg_rows:
        rows.append(external_product(pp, gsw_key, r))            # bottom ℓ: s-scaled rows
    return rows

# --------------------------------------------------------------------------- #
# database as a hypercube + fold to the answer
#   dims = [N0, 2, 2, ...] ;  index i -> coords (c0, c1, c2, ...)
#   (b) initial dim  : plaintext×ct dot product over the N0 axis
#   (c) higher dims  : one external product per size-2 fold  (mux_1of2)
# --------------------------------------------------------------------------- #
def make_database(pp, dims, rng):
    N = 1
    for d in dims:
        N *= d
    return [[rng.randrange(pp.t) for _ in range(pp.n)] for _ in range(N)]   # N plaintext polys

def onehot_cts(pp, s, N0, target0, rng):
    """the N0 initial-dimension selector ciphertexts: Enc(1) at target0, Enc(0) else."""
    return [encrypt(pp, s, [1 if k == target0 else 0] + [0] * (pp.n - 1), rng)
            for k in range(N0)]

def pir_answer(pp, db, dims, sel0_cts, sel_bits_rgsw):
    """Server compute: initial-dim dot product then fold higher dims. Returns one BFV ct."""
    N0 = dims[0]
    rest = 1
    for d in dims[1:]:
        rest *= d
    # (b) initial dimension: acc[r] = Σ_k sel0[k] · DB[k*rest + r]   (plaintext×ct)
    acc = []
    for r in range(rest):
        s_acc = pt_ct_mul(pp, db[0 * rest + r], sel0_cts[0])
        for k in range(1, N0):
            s_acc = ct_add(pp, s_acc, pt_ct_mul(pp, db[k * rest + r], sel0_cts[k]))
        acc.append(s_acc)
    # (c) higher dimensions: fold each size-2 axis with its RGSW selection bit
    for dim_idx in range(1, len(dims)):
        C_bit = sel_bits_rgsw[dim_idx - 1]
        half = len(acc) // 2
        acc = [mux_1of2(pp, C_bit, acc[r], acc[half + r]) for r in range(half)]
    assert len(acc) == 1
    return acc[0]

# --------------------------------------------------------------------------- #
# Fully chained server pipeline:  EXPAND -> UNPACK -> ANSWER
#   client sends ONE packed ct per part; the server expands the init-dim one-hot into
#   selectors and expands+unpacks each fold bit into an RGSW, then runs the answer.
# --------------------------------------------------------------------------- #
def _log2_exact(x):
    m = 0
    while (1 << m) < x:
        m += 1
    return m

def _coords(dims, idx):
    """hypercube coords: c0 (initial dim) + one bit per fold dim, matching pir_answer's fold order."""
    rest = 1
    for d in dims[1:]:
        rest *= d
    c0 = idx // rest
    r = idx % rest
    bits = []
    blocks = dims[1:]
    for j in range(len(blocks)):
        sub = 1
        for k in range(j + 1, len(blocks)):
            sub *= blocks[k]
        bits.append((r // sub) % blocks[j])
    return c0, bits

def gsw_msg_rows(pp, s, bit, gkeys, rng, expand_ladder=True):
    """The ℓ RGSW message rows (phase bit·B^k) for one fold bit.
    expand_ladder: client packs the scaleless gadget ladder into one ct and the SERVER expands it
    (the faithful chained path); else inject the rows directly (skips expansion, faster budget probe)."""
    g = gadget_vector(pp.B, pp.ell, pp.q)
    if not expand_ladder:
        return [encrypt_scaleless(pp, s, [(bit * g[k]) % pp.q] + [0] * (pp.n - 1), rng)
                for k in range(pp.ell)]
    m_lad = _log2_exact(pp.ell)
    inv = modinv(1 << m_lad, pp.q)                       # undo the ×2^m the Subs tree introduces
    packed = [0] * pp.n
    for k in range(pp.ell):
        packed[k] = (inv * bit * g[k]) % pp.q
    ctq = encrypt_scaleless(pp, s, packed, rng)
    return expand_query(pp, ctq, m_lad, gkeys)[:pp.ell]

def pir_query_and_answer(pp, db, dims, idx, s, rng, expand_ladder=True, report=None):
    """End-to-end: (a) EXPAND init-dim one-hot -> selectors, (b) EXPAND+UNPACK each fold bit ->
    RGSW, (c) ANSWER. Returns the response ct. `report(name, budget)` gets per-stage noise budgets."""
    N0 = dims[0]
    c0, fold_bits = _coords(dims, idx)
    m_exp = _log2_exact(N0)
    m_lad = _log2_exact(pp.ell)
    gkeys = galois_keys_for_expand(pp, s, max(m_exp, m_lad), rng)
    # (a) EXPAND: packed one-hot (Δ-scaled) -> N0 BFV selectors
    inv = modinv(1 << m_exp, pp.t)
    packed = [0] * pp.n
    packed[c0] = inv                                     # client pre-scales by (2^m)^{-1} mod t
    sel0 = expand_query(pp, encrypt(pp, s, packed, rng), m_exp, gkeys)[:N0]
    if report is not None:
        report("expand_selector", noise_budget(pp, s, sel0[c0]))
    # (b) UNPACK: each fold bit -> RGSW via RGSW(s)
    gsw_key = gen_gsw_key(pp, s, rng)
    sel_bits = [rlwe_to_rgsw(pp, gsw_msg_rows(pp, s, b, gkeys, rng, expand_ladder), gsw_key)
                for b in fold_bits]
    # (c) ANSWER: initial-dim scan + folds
    ans = pir_answer(pp, db, dims, sel0, sel_bits)
    if report is not None:
        report("final_answer", noise_budget(pp, s, ans))
    return ans

def run_pipeline_report():
    """Step-0 check: full expand->unpack->answer, correctness + per-stage noise budget."""
    import time
    configs = [
        # (n,    dims,     ell, B,        expand_ladder, indices)
        (2048, [4, 2],    3,  1 << 20,  True,  [0, 3]),    # n=2048, device fold params -> expect FAIL
        (2048, [4, 2],    6,  1 << 10,  True,  None),      # n=2048, RECOMMENDED (ell=6), all 8 idx
        (2048, [8, 2, 2], 6,  1 << 10,  True,  [0, 17, 31]),  # n=2048, bigger hypercube (32 entries)
    ]
    allok = True
    for (n, dims, ell, B, ladder, indices) in configs:
        rng = random.Random(7)
        pp = Params(n=n, t=257, B=B, ell=ell)
        N = 1
        for d in dims:
            N *= d
        idxs = indices if indices is not None else list(range(N))
        print(f"\n=== pipeline @ {pp}  dims={dims} (N={N})  expand_ladder={ladder}  m_exp={_log2_exact(dims[0])} ===")
        s = keygen(pp, rng)
        db = make_database(pp, dims, rng)
        for idx in idxs:
            t0 = time.time()
            rep = {}
            ans = pir_query_and_answer(pp, db, dims, idx, s, rng, expand_ladder=ladder,
                                       report=lambda k, v: rep.__setitem__(k, v))
            got = decrypt(pp, s, ans)
            ok = (got == db[idx])
            allok &= ok
            extra = "  ".join(f"{k}={v:.1f}b" for k, v in rep.items())
            print(f"  [{'OK ' if ok else 'FAIL'}] idx={idx:2d} got={got[0]:3d} want={db[idx][0]:3d} | "
                  f"{extra}  ({time.time() - t0:.1f}s)")
    print(f"\n{'PIPELINE ALL PASS' if allok else 'PIPELINE SOME FAILED'}")
    return allok

# --------------------------------------------------------------------------- #
# self-test suite
# --------------------------------------------------------------------------- #
def _check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    return ok

def run_tests(verbose=False):
    rng = random.Random(1234)
    pp = Params(n=256, t=257, B=256, ell=8)
    print(pp)
    s = keygen(pp, rng)
    allok = True

    # --- gadget decompose reconstruction ---
    v = rand_poly(pp.n, pp.q, rng)
    dg = gadget_decompose_signed(v, pp.B, pp.ell, pp.q)
    g = gadget_vector(pp.B, pp.ell, pp.q)
    recon = [sum(dg[k][i] * g[k] for k in range(pp.ell)) % pp.q for i in range(pp.n)]
    maxdig = max(abs(dg[k][i]) for k in range(pp.ell) for i in range(pp.n))
    allok &= _check("gadget_decompose_signed reconstructs & digits balanced",
                    recon == v and maxdig < pp.B // 2 + 1, f"max|digit|={maxdig} (B/2={pp.B//2})")

    # --- BFV basics ---
    m1 = [rng.randrange(pp.t) for _ in range(pp.n)]
    ct1 = encrypt(pp, s, m1, rng)
    allok &= _check("encrypt/decrypt roundtrip", decrypt(pp, s, ct1) == m1,
                    f"budget={noise_budget(pp, s, ct1):.1f} bits")
    m2 = [rng.randrange(pp.t) for _ in range(pp.n)]
    ct2 = encrypt(pp, s, m2, rng)
    add_ok = decrypt(pp, s, ct_add(pp, ct1, ct2)) == [(a + b) % pp.t for a, b in zip(m1, m2)]
    allok &= _check("ct_add", add_ok)
    pt = [rng.randrange(pp.t) for _ in range(pp.n)]
    want = [x % pp.t for x in negacyclic_mul(pt, m1, pp.t)]
    ptct_ok = decrypt(pp, s, pt_ct_mul(pp, pt, ct1)) == want
    allok &= _check("plaintext×ct (initial-dim op)", ptct_ok)

    # --- external product: RGSW(bit) ⊡ RLWE(μ) = RLWE(bit·μ) ---
    for bit in (0, 1):
        C = rgsw_encrypt(pp, s, [bit] + [0] * (pp.n - 1), rng)   # RGSW of the scalar `bit`
        mu = [rng.randrange(pp.t) for _ in range(pp.n)]
        ctmu = encrypt(pp, s, mu, rng)
        ep = external_product(pp, C, ctmu)
        want = mu if bit else [0] * pp.n
        allok &= _check(f"external_product (bit={bit})", decrypt(pp, s, ep) == want,
                        f"budget={noise_budget(pp, s, ep):.1f} bits")

    # --- 1-of-2 mux b·(y−x)+x ---
    x = encrypt(pp, s, [rng.randrange(pp.t) for _ in range(pp.n)], rng)
    y = encrypt(pp, s, [rng.randrange(pp.t) for _ in range(pp.n)], rng)
    for bit in (0, 1):
        C = rgsw_encrypt(pp, s, [bit] + [0] * (pp.n - 1), rng)
        got = decrypt(pp, s, mux_1of2(pp, C, x, y))
        allok &= _check(f"mux_1of2 (bit={bit})",
                        got == decrypt(pp, s, y if bit else x))

    # --- monomial multiply ---
    k = 5
    mm = decrypt(pp, s, mono_mul(pp, ct1, k))
    want = [v % pp.t for v in mono_mul_poly(m1, k, pp.t)]
    allok &= _check("mono_mul x^k", mm == want)

    # --- Subs (automorphism + key-switch) ---
    j = pp.n + 1
    gk = gen_galois_key(pp, s, j, rng)
    sb = decrypt(pp, s, subs(pp, ct1, j, gk))
    want = [v % pp.t for v in automorph_poly(m1, j, pp.t)]
    allok &= _check(f"subs (j={j})", sb == want)

    # --- modulus switch preserves the message (q -> q'≈22-bit) ---
    qprime = ntt_primes(1, pp.n, bits=22)[0]
    mm5 = [rng.randrange(pp.t) for _ in range(pp.n)]
    ctm = encrypt(pp, s, mm5, rng)
    ms = mod_switch(pp, ctm, qprime)
    allok &= _check("mod_switch q->q' preserves message",
                    decrypt_modswitched(pp, s, ms, qprime) == mm5, f"q'={qprime}(~{qprime.bit_length()}b)")

    # --- query expansion: packed one-hot -> per-slot cts ---
    m_exp = 3                                        # expand to 8 slots
    N0 = 1 << m_exp
    target = 5
    inv = modinv(1 << m_exp, pp.t)
    packed = [0] * pp.n
    packed[target] = inv                             # client pre-scales by (2^m)^{-1}
    ctq = encrypt(pp, s, packed, rng)
    gkeys = galois_keys_for_expand(pp, s, m_exp, rng)
    expanded = expand_query(pp, ctq, m_exp, gkeys)
    dec = [decrypt(pp, s, c)[0] for c in expanded]
    allok &= _check("expand_query -> one-hot",
                    dec == [1 if i == target else 0 for i in range(N0)], f"slots={dec}")

    # --- RLWE -> RGSW conversion (QueryUnpack): converted RGSW acts like a native RGSW ---
    gsw_key = gen_gsw_key(pp, s, rng)
    g = gadget_vector(pp.B, pp.ell, pp.q)
    for bit in (0, 1):
        rows = [encrypt_scaleless(pp, s, [(bit * g[k]) % pp.q] + [0] * (pp.n - 1), rng)
                for k in range(pp.ell)]                     # ℓ message rows (phase bit·B^k)
        C_conv = rlwe_to_rgsw(pp, rows, gsw_key)
        mu = [rng.randrange(pp.t) for _ in range(pp.n)]
        ep = external_product(pp, C_conv, encrypt(pp, s, mu, rng))
        want = mu if bit else [0] * pp.n
        allok &= _check(f"rlwe_to_rgsw -> RGSW (bit={bit})", decrypt(pp, s, ep) == want,
                        f"budget={noise_budget(pp, s, ep):.1f} bits")

    # --- FULL chained pipeline: expand -> unpack -> answer (all indices) ---
    dims_p = [4, 2]                                          # 8 entries, N0=4, 1 fold
    db_p = make_database(pp, dims_p, rng)
    last_b = 0.0
    okp = True
    for idx in range(8):
        rep = {}
        ans = pir_query_and_answer(pp, db_p, dims_p, idx, s, rng, expand_ladder=True,
                                   report=lambda k, v: rep.__setitem__(k, v))
        okp &= (decrypt(pp, s, ans) == db_p[idx])
        last_b = rep.get("final_answer", 0.0)
    allok &= _check("full pipeline expand->unpack->answer (8 idx)", okp,
                    f"final budget={last_b:.1f} bits")

    # --- end-to-end PIR retrieve ---
    dims = [4, 2, 2]                                  # N = 16 entries
    N = dims[0] * dims[1] * dims[2]
    db = make_database(pp, dims, rng)
    want_index = 11                                  # retrieve DB[11]
    c0 = want_index // (dims[1] * dims[2])
    c1 = (want_index // dims[2]) % dims[1]
    c2 = want_index % dims[2]
    sel0 = onehot_cts(pp, s, dims[0], c0, rng)
    sel_bits = [rgsw_encrypt(pp, s, [b] + [0] * (pp.n - 1), rng) for b in (c1, c2)]
    ans = pir_answer(pp, db, dims, sel0, sel_bits)
    got = decrypt(pp, s, ans)
    allok &= _check(f"end-to-end PIR retrieve DB[{want_index}]  (coords {c0},{c1},{c2})",
                    got == db[want_index], f"budget={noise_budget(pp, s, ans):.1f} bits")

    # retrieve a few more indices to be sure the fold indexing is right
    ok_all = True
    for idx in (0, 1, 7, 15):
        c0 = idx // (dims[1] * dims[2]); c1 = (idx // dims[2]) % dims[1]; c2 = idx % dims[2]
        sel0 = onehot_cts(pp, s, dims[0], c0, rng)
        sel_bits = [rgsw_encrypt(pp, s, [b] + [0] * (pp.n - 1), rng) for b in (c1, c2)]
        got = decrypt(pp, s, pir_answer(pp, db, dims, sel0, sel_bits))
        ok_all &= (got == db[idx])
    allok &= _check("end-to-end PIR retrieve {0,1,7,15}", ok_all)

    print(f"\n{'ALL PASS' if allok else 'SOME FAILED'}")
    return allok


if __name__ == "__main__":
    if "--pipeline" in sys.argv:
        ok = run_pipeline_report()
    else:
        ok = run_tests(verbose="-v" in sys.argv)
    sys.exit(0 if ok else 1)
