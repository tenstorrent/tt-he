# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""reference.py — unified bit-exact golden for the HE IR (L1 / L2 / L3).

The oracle for every op in docs/ir/ops.md. All arithmetic is EXACT: Python ints
(object ndarrays), reduced mod q, no float anywhere. Because lowering
L1 -> L2 -> L3 is meaning-preserving, the SAME values come out at every level, so
a lowering pass is validated by running both programs through this reference and
diffing. Device kernels are in turn validated bit-exact against these functions.

Layout:
  §L1  RNS-poly ops     : const ntt intt mul add sub base_convert crt_combine
                          gadget_decompose automorph
  §L2  tile / kernel ops: matmul_modq mul_modq add_modq transpose reshape slice
                          split concat all_to_all_tiles
  §L3  int8 ops         : decompose recombine_modq
  §X   cross-level identities (the lowering equivalences), exercised in __main__.

Names in OP_TABLE match ops.md exactly (rns.* / tile.*).  Run: python3 reference.py
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# helpers (exact, object dtype)
# --------------------------------------------------------------------------- #
def A(x):
    """As a Python-int (object) ndarray."""
    return np.asarray(x, dtype=object)

def modq(x, q):
    return A(x) % q

def crt(residues, primes):
    """Exact CRT: unique X in [0, prod(primes)) with X % primes[i] == residues[i]."""
    M = 1
    for p in primes:
        M *= p
    X = 0
    for r, p in zip(residues, primes):
        Mi = M // p
        X += int(r) * Mi * pow(Mi, -1, p)
    return X % M

def find_2nth_root(q, N):
    """A primitive 2N-th root of unity psi mod q: psi^N == -1, psi^(2N) == 1.
    Fast: raise small bases to (q-1)/(2N) (works for real 28-bit NTT primes, q == 1 mod 2N)."""
    e = (q - 1) // (2 * N)
    for g in range(2, q):
        cand = pow(g, e, q)
        if pow(cand, N, q) == q - 1:      # order exactly 2N  <=>  psi^N == -1
            return cand
    raise ValueError(f"no 2N-th root for q={q}, N={N} (need q == 1 mod 2N)")


# =========================================================================== #
# §L1 — RNS-poly ops
# =========================================================================== #
def const(value, q):
    return modq(value, q) if q is not None else A(value)   # q=None: raw/wide baked constant

def ntt(x, q, psi):
    """Forward negacyclic NTT (coeff -> eval).  X[k] = Σ_j x[j]·psi^{j(2k+1)} mod q."""
    x = [int(v) % q for v in x]
    N = len(x)
    return A([sum(x[j] * pow(psi, (j * (2 * k + 1)) % (2 * N), q) for j in range(N)) % q
              for k in range(N)])

def intt(X, q, psi):
    """Inverse negacyclic NTT (eval -> coeff)."""
    X = [int(v) % q for v in X]
    N = len(X)
    omega_inv = pow(psi * psi % q, -1, q)
    psi_inv = pow(psi, -1, q)
    Ninv = pow(N, -1, q)
    out = []
    for j in range(N):
        s = sum(X[k] * pow(omega_inv, (j * k) % N, q) for k in range(N)) % q
        out.append(Ninv * pow(psi_inv, j, q) % q * s % q)
    return A(out)

def mul(a, b, q):          # pointwise / Hadamard (eval domain)
    return modq(A(a) * A(b), q)

def add(*xs, q):
    acc = A(xs[0])
    for x in xs[1:]:
        acc = acc + A(x)
    return modq(acc, q)

def sub(a, b, q):
    return modq(A(a) - A(b), q)

def base_convert(residues, src_base, dst_base, signed=False):
    """Exact RNS base change: residues (mod src) -> residues (mod dst), per coeff.
    signed=True interprets the CRT value > prod/2 as negative (rescale/centered case)."""
    residues = [A(r) for r in residues]
    N = len(residues[0])
    M = 1
    for p in src_base:
        M *= p
    out = [[0] * N for _ in dst_base]
    for i in range(N):
        x = crt([int(residues[s][i]) for s in range(len(src_base))], src_base)
        if signed and x > M // 2:
            x -= M
        for d, p in enumerate(dst_base):
            out[d][i] = x % p
    return [A(o) for o in out]

def crt_combine(residues, src_base):
    """Reconstruct the integer coeff-wise from its RNS residues (exact)."""
    residues = [A(r) for r in residues]
    N = len(residues[0])
    return A([crt([int(residues[s][i]) for s in range(len(src_base))], src_base) for i in range(N)])

def gadget_decompose(x, base_B, ell, signed=True):
    """Signed base-B digits: x = Σ_{j<ell} d_j·B^j, d_j in [-B/2, B/2)."""
    x = [int(v) for v in x]
    digits = [[0] * len(x) for _ in range(ell)]
    for i, v in enumerate(x):
        for j in range(ell):
            d = v % base_B
            if signed and d >= base_B // 2:
                d -= base_B
            digits[j][i] = d
            v = (v - d) // base_B
    return [A(d) for d in digits]

def automorph(a, k, q, N):
    """Galois automorphism sigma_k: a(X) -> a(X^k) in Z_q[X]/(X^N+1). k odd, coprime to 2N."""
    a = [int(v) % q for v in a]
    out = [0] * N
    for j in range(N):
        e = (j * k) % (2 * N)
        sign = 1 if e < N else -1
        out[e % N] = (out[e % N] + sign * a[j]) % q
    return A(out)


# =========================================================================== #
# §L2 — tile / kernel ops
# =========================================================================== #
def matmul_modq(a, b, q, transpose_a=False, transpose_b=False):
    a, b = A(a), A(b)
    if transpose_a:
        a = a.T
    if transpose_b:
        b = b.T
    return modq(a @ b, q)

def matmul_wide(a, b, transpose_a=False, transpose_b=False):
    """Integer matmul with NO modular reduction (exact bignum). The wide contraction the
    CRNS base_convert needs for its fixed-point accumulator (there is no prime to reduce by)."""
    a, b = A(a), A(b)
    if transpose_a:
        a = a.T
    if transpose_b:
        b = b.T
    return A(a @ b)

def fixed_slice(x, shift, mask=None):
    """Arithmetic right-shift (+ optional low-bit mask) of a wide value — the CRNS slice that
    recovers the overflow k (shift=u) and the sign bit (shift=u-1, mask=1) from the accumulator."""
    y = A(x) >> shift
    return (y & mask) if mask is not None else y

def mul_modq(a, b, q):
    return modq(A(a) * A(b), q)          # numpy broadcasting

def add_modq(*xs, q):
    acc = A(xs[0])
    for x in xs[1:]:
        acc = acc + A(x)
    return modq(acc, q)

def transpose(x):
    return A(x).T

def reshape(x, shape):
    return A(x).reshape(tuple(shape))

def slice_(x, axis, start, size):
    idx = [slice(None)] * A(x).ndim
    idx[axis] = slice(start, start + size)
    return A(x)[tuple(idx)]

def split(x, axis, parts):
    return list(np.split(A(x), parts, axis=axis))

def concat(xs, axis):
    return np.concatenate([A(x) for x in xs], axis=axis)

def all_to_all_tiles(inputs, perm, transpose=None):
    transpose = transpose or [False] * len(perm)
    outs = []
    for p, t in zip(perm, transpose):
        src = A(inputs[p])
        outs.append(src.T if t else src)
    return outs


# =========================================================================== #
# §L3 — int8 decomposition ops (matmul-only)
# =========================================================================== #
def decompose(x, radix=128, nlimbs=4):
    x = A(x)
    return [ (x // (radix ** p)) % radix for p in range(nlimbs) ]

def recombine_modq(limbs, q, radix=128):
    acc = A(0)
    for r, c in enumerate(limbs):
        acc = acc + (radix ** r) * A(c)
    return modq(acc, q)

def recombine_wide(planes, radix=128, exps=None):
    """Place-value recombine with NO modular reduction — the wide (bignum) combine the CRNS
    f-accumulator needs (there is no prime to reduce by). `exps` gives each plane's radix
    exponent (default 0..len-1); result = Σ_i radix^exps[i] · planes[i], exact."""
    if exps is None:
        exps = list(range(len(planes)))
    acc = A(0)
    for e, c in zip(exps, planes):
        acc = acc + (radix ** int(e)) * A(c)
    return acc


# =========================================================================== #
# §X — cross-level realizations (the lowering equivalences)
# =========================================================================== #
def matmul_modq_via_int8(a, b, q, radix=128, nlimbs=4):
    """L2 matmul_modq realized by the L3 int8 decomposition (both operands, general form):
       A@B = Σ_{p,r} radix^{p+r} (A_p @ B_r), A_p/B_r the radix limbs. Exact."""
    ap = decompose(a, radix, nlimbs)
    br = decompose(b, radix, nlimbs)
    acc = A(0)
    for p in range(nlimbs):
        for r in range(nlimbs):
            acc = acc + (radix ** (p + r)) * (A(ap[p]) @ A(br[r]))
    return modq(acc, q)


# --- CRNS (Cox-Rower / Kawamura) realization of base_convert --------------- #
# Ported from reference/he/bfv_bconv_crns.py (verified m_sk-free, decrypts all seeds).
# This is the *primitive-op* form of rns.base_convert: two contractions (E, f) + a
# fixed-point shift-extract (k / sign) + a per-dst correction. lower.py emits exactly
# these ops; here we prove the decomposition reproduces the exact-CRT base_convert.
def crns_safe_u(src):
    """Smallest u guaranteeing k is exact for EVERY x in [0, ∏src):  2^u > n·max(src)·M."""
    M = 1
    for p in src:
        M *= p
    return (len(src) * max(src) * M).bit_length() + 1

def crns_precompute(src, dst, u):
    """The three baked constant tables (E matrix, f vector, g vector); y=z=1 (no scaling)."""
    M = 1
    for p in src:
        M *= p
    I = [(pow((M // s) % s, -1, s) * (M // s)) % M for s in src]   # ICRT factors mod M
    E = [[Ii % d for d in dst] for Ii in I]                        # E[i][j] = I_i mod dst_j
    f = [((Ii << u) + M - 1) // M for Ii in I]                     # f_i = ⌈I_i·2^u / M⌉
    g = [(-(M % d)) % d for d in dst]                              # g_j = −M mod dst_j
    return E, f, g

def base_convert_via_crns(residues, src_base, dst_base, signed=False, u=None):
    """base_convert realized by CRNS — the exact decomposition lower.py targets.
       residues: |src| arrays (x mod src_i).  Returns |dst| arrays (x mod dst_j).
       signed=True: source encodes a balanced value (centered), sign recovered from bit u-1."""
    res = [A(r).astype(object) for r in residues]
    ns = len(src_base)
    if u is None:
        u = crns_safe_u(src_base)
    E, f, g = crns_precompute(src_base, dst_base, u)
    M = 1
    for p in src_base:
        M *= p
    s = A(0)                                                       # f-accumulator (wide): Σ_i r_i·f_i
    for i in range(ns):
        s = s + res[i] * f[i]
    k = s >> u                                                     # overflow (integer part)
    neg = (s >> (u - 1)) & 1 if signed else A(0)                   # sign bit (top fractional bit)
    out = []
    for j, d in enumerate(dst_base):
        acc = A(0)                                                 # E base-extension: Σ_i r_i·E_ij
        for i in range(ns):
            acc = acc + res[i] * E[i][j]
        out.append((acc + k * g[j] - neg * (M % d)) % d)          # + k·g_j − neg·(M mod d_j)
    return [A(o) for o in out]


def fast_base_convert(residues, src_base, dst_base):
    """APPROXIMATE fast base conversion {x mod src_i} -> {(x+εM) mod dst_j}, dropping the ε overflow
    (ε < len(src); absorbed into HE noise). The BEHZ mod-up/mod-down primitive. Structure = pre-scale
    (elementwise) + base-extension matmul, NO overflow correction (that is what makes it the cheap
    approximate variant, vs base_convert_via_crns's exact k-recovery):
        y_i   = r_i · (M/s_i)^{-1} mod s_i                 (elementwise pre-scale, per src prime)
        out_j = Σ_i y_i · ((M/s_i) mod d_j)  mod d_j       (base-extension matmul, per dst prime)"""
    residues = [A(r).astype(object) for r in residues]
    M = 1
    for p in src_base:
        M *= p
    y = [modq(residues[i] * pow((M // src_base[i]) % src_base[i], -1, src_base[i]), src_base[i])
         for i in range(len(src_base))]
    out = []
    for d in dst_base:
        acc = A(0)
        for i in range(len(src_base)):
            acc = acc + y[i] * ((M // src_base[i]) % d)
        out.append(modq(acc, d))
    return [A(o) for o in out]


# --------------------------------------------------------------------------- #
# unified op table (names == ops.md)
# --------------------------------------------------------------------------- #
OP_TABLE = {
    "rns.const": const, "rns.ntt": ntt, "rns.intt": intt, "rns.mul": mul,
    "rns.add": add, "rns.sub": sub, "rns.base_convert": base_convert,
    "rns.crt_combine": crt_combine, "rns.gadget_decompose": gadget_decompose,
    "rns.automorph": automorph, "rns.fast_base_convert": fast_base_convert,
    "tile.matmul_modq": matmul_modq, "tile.matmul_wide": matmul_wide, "tile.fixed_slice": fixed_slice,
    "tile.mul_modq": mul_modq, "tile.add_modq": add_modq,
    "tile.transpose": transpose, "tile.reshape": reshape, "tile.slice": slice_,
    "tile.split": split, "tile.concat": concat, "tile.all_to_all_tiles": all_to_all_tiles,
    "tile.decompose": decompose, "tile.recombine_modq": recombine_modq,
    "tile.recombine_wide": recombine_wide,
}


# --------------------------------------------------------------------------- #
# self-tests: per-op correctness + the cross-level identities
# --------------------------------------------------------------------------- #
def _eq(a, b):
    return np.array_equal(A(a), A(b))

def _run_tests():
    import random
    rng = random.Random(7)
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, f"FAIL: {name}"
        print(f"  ok  {name}")
        ok += 1

    # ---- L1: NTT round-trip + negacyclic convolution property ----
    N, q = 8, 17
    psi = find_2nth_root(q, N)
    x = [rng.randrange(q) for _ in range(N)]
    check("ntt/intt round-trip", _eq(intt(ntt(x, q, psi), q, psi), modq(x, q)))

    a = [rng.randrange(q) for _ in range(N)]
    b = [rng.randrange(q) for _ in range(N)]
    # schoolbook negacyclic convolution a·b mod (X^N+1)
    full = [0] * (2 * N - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            full[i + j] += ai * bj
    neg = [0] * N
    for i in range(2 * N - 1):
        neg[i % N] += full[i] if i < N else -full[i]
    neg = modq(neg, q)
    via_ntt = intt(mul(ntt(a, q, psi), ntt(b, q, psi), q), q, psi)
    check("negacyclic mul == intt(ntt(a)⊙ntt(b))", _eq(neg, via_ntt))

    # ---- L1: mul/add/sub ----
    check("add/sub inverse", _eq(sub(add(a, b, q=q), b, q), modq(a, q)))

    # ---- L1: base_convert (exact CRT) unsigned + signed ----
    src, dst = [17, 97], [193, 241]
    Msrc = 17 * 97
    v = [rng.randrange(Msrc) for _ in range(5)]
    res = [[vi % p for vi in v] for p in src]
    y = base_convert(res, src, dst)
    check("base_convert unsigned", _eq(y[0], modq(v, 193)) and _eq(y[1], modq(v, 241)))
    vs = [rng.randrange(-Msrc // 2, Msrc // 2) for _ in range(5)]
    ress = [[vi % p for vi in vs] for p in src]
    ys = base_convert(ress, src, dst, signed=True)
    check("base_convert signed/centered", _eq(ys[0], modq(vs, 193)) and _eq(ys[1], modq(vs, 241)))

    # ---- L1: crt_combine ----
    check("crt_combine", _eq(crt_combine(res, src), modq(v, Msrc)))

    # ---- L1: gadget_decompose — unsigned exact over full range; signed over its range ----
    B, ell = 4, 4
    def _recon(dig):
        r = A(0)
        for j, dj in enumerate(dig):
            r = r + (B ** j) * A(dj)
        return r
    gu = [rng.randrange(B ** ell) for _ in range(6)]                    # full [0, B^ell)
    check("gadget unsigned reconstruct", _eq(_recon(gadget_decompose(gu, B, ell, signed=False)), gu))
    gs = [rng.randrange(-(B ** ell) // 8, (B ** ell) // 8) for _ in range(6)]  # representable balanced range
    digs = gadget_decompose(gs, B, ell, signed=True)
    check("gadget signed reconstruct", _eq(_recon(digs), gs))
    check("gadget signed digit range", all(-B // 2 <= int(d) < B // 2 for dj in digs for d in dj))

    # ---- L1: automorph is a ring homomorphism over negacyclic mul ----
    k = 3
    lhs = automorph(neg, k, q, N)
    rhs_full = [0] * (2 * N - 1)
    aa, bb = automorph(a, k, q, N), automorph(b, k, q, N)
    for i, ai in enumerate(aa):
        for j, bj in enumerate(bb):
            rhs_full[i + j] += int(ai) * int(bj)
    rhs = [0] * N
    for i in range(2 * N - 1):
        rhs[i % N] += rhs_full[i] if i < N else -rhs_full[i]
    check("automorph homomorphism", _eq(lhs, modq(rhs, q)))
    check("automorph k=1 identity", _eq(automorph(a, 1, q, N), modq(a, q)))

    # ---- L2/L3: decompose∘recombine round-trip ----
    Q = 251658241
    xs = [rng.randrange(Q) for _ in range(10)]
    check("decompose/recombine round-trip", _eq(recombine_modq(decompose(A(xs)), Q), modq(xs, Q)))

    # ---- CROSS-LEVEL: L2 matmul_modq == its L3 int8 realization ----
    M = A([[rng.randrange(Q) for _ in range(4)] for _ in range(4)])
    Nm = A([[rng.randrange(Q) for _ in range(4)] for _ in range(4)])
    check("matmul_modq == int8 decomposition (L2==L3)",
          _eq(matmul_modq(M, Nm, Q), matmul_modq_via_int8(M, Nm, Q)))

    # ---- CROSS-LEVEL: base_convert == its CRNS realization (unsigned + signed) ----
    src2, dst2 = [40961, 65537, 114689], [147457, 163841]         # real 17/18-bit NTT primes
    Ms = 1
    for p in src2:
        Ms *= p
    vb = [rng.randrange(Ms) for _ in range(7)]
    resb = [[vi % p for vi in vb] for p in src2]
    check("base_convert == CRNS (unsigned)",
          all(_eq(a_, b_) for a_, b_ in zip(base_convert(resb, src2, dst2),
                                            base_convert_via_crns(resb, src2, dst2))))
    vbs = [rng.randrange(-Ms // 2, Ms // 2) for _ in range(7)]     # balanced / signed source
    resbs = [[vi % p for vi in vbs] for p in src2]
    check("base_convert == CRNS (signed/centered)",
          all(_eq(a_, b_) for a_, b_ in zip(base_convert(resbs, src2, dst2, signed=True),
                                            base_convert_via_crns(resbs, src2, dst2, signed=True))))

    # ---- L2: layout ops sanity ----
    t = A([[1, 2], [3, 4]])
    check("transpose", _eq(transpose(t), [[1, 3], [2, 4]]))
    check("all_to_all_tiles perm+transpose",
          _eq(all_to_all_tiles([t, t + 10], [1, 0], [False, True])[1], t.T))

    print(f"\nALL {ok} CHECKS PASSED — L1/L2/L3 reference is internally consistent + cross-level exact.")


if __name__ == "__main__":
    _run_tests()
