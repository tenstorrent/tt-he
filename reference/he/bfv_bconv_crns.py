# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""CRNS (Cox-Rower / Kawamura) exact base conversion — m_sk-free alternative to the SK path.

Companion to `bfv_mul_ref.py`. The BFV rescale needs ONE *exact* base conversion (S -> Q of the
rescaled ẽ). `bfv_mul_ref._fbc_exact_sk` does it by recovering the CRT overflow ε via a redundant
sentinel modulus m_sk (Shenoy-Kumaresan). CRNS gets the SAME overflow term a different way — a
fixed-point floor with `u` bits of precision (integer dot-product + shift, no float, no sentinel).
The algorithm is the CRNS construction of Langowski–Devadas, ePrint 2025/1068. This is an
independent implementation; its precompute / convert split follows the structure of Google jaxite's
JAX version (`jaxite_ec/finite_field_context.py`, Apache-2.0), from which no code is taken.

Why it is exact without m_sk.  ICRT:  x = Σ_i r_i·I_i − k·M,  I_i = (M/s_i)^{-1}(M/s_i),  M = ∏S,
where the overflow k = ⌊(Σ_i r_i·I_i)/M⌋.  Since 0 ≤ x < M we have Σ r_i·I_i/M = k + x/M with
x/M ∈ [0,1), so ⌊·⌋ = k exactly.  CRNS estimates k with f_i = ⌈I_i·2^u/M⌉ (round up):
    k_est = ⌊(Σ_i r_i·f_i)/2^u⌋ = ⌊ k + x/M + D/2^u ⌋,   0 ≤ D < Σ_i r_i < n·max(S).
So k_est = k  iff  x/M + D/2^u < 1  ⇔  2^u > n·max(S)·M/(M−x).  Worst case M−x = 1 needs
u > log2(n·max(S)·M); with any headroom below M (BFV: ẽ ≪ ∏S, a full spare limb) far less suffices.
m_sk trades this precision for one extra prime; CRNS trades the prime for a wide (u-bit) accumulator.

Run:  python3 bfv_bconv_crns.py           # unit FBC check + u-sweep + full ct_mul cross-check
"""

import random, math
from math import prod
from bfv_relin_ref import (
    ntt_primes, negacyclic_mul, poly_add, crt_reconstruct, small_poly, rns_of,
    keygen_relin, relinearize,
)
import bfv_mul_ref as ref
from bfv_mul_ref import MulParams, encrypt, decrypt, tensor_unsigned, _fbc_exact_sk, _fbc_approx


# --------------------------------------------------------------------------- #
# CRNS exact base extension  {x mod src_i} -> {x mod dst_j}   (no sentinel)
# --------------------------------------------------------------------------- #
def crns_safe_u(src):
    """smallest u that guarantees exactness for EVERY x in [0, ∏src):  2^u > n·max(src)·M."""
    M = prod(src)
    return (len(src) * max(src) * M).bit_length() + 1

def crns_precompute(src, dst, u):
    """E, f, g exactly as the CRNS precompute of ePrint 2025/1068 (y=z=1)."""
    M = prod(src)
    I = [(pow((M // s) % s, -1, s) * (M // s)) % M for s in src]     # ICRT factors (mod M)
    E = [[Ii % d for d in dst] for Ii in I]                          # matrix_E     (E_ij = I_i mod d_j)
    f = [((Ii << u) + M - 1) // M for Ii in I]                      # vector_f_T   (⌈I_i·2^u/M⌉, integer)
    g = [(-(M % d)) % d for d in dst]                                # vector_g     (−M mod d_j)
    return E, f, g

def _fbc_exact_crns(res_src, src, dst, u=None, pre=None, center=False):
    """CRNS exact FBC (drop-in for _fbc_exact_sk, minus the m_sk args).  RECONSTRUCTION-FREE.
       res_src = {x mod src_i}.  Returns {x mod dst_j}.  `pre`=(E,f,g) to reuse constants.
       center=False: treats the value as the unsigned [0,∏src) representative (plain base extension).
       center=True : treats it as the BALANCED representative in (−∏src/2, ∏src/2] — needed when the
                     source encodes a SIGNED value (e.g. the BFV rescaled ẽ, |ẽ| ≪ ∏src/2).  This is
                     the job m_sk does in the SK rescale: recover which side of ∏src the value is on.

       Both the overflow k and the sign bit come from ONE fixed-point accumulator s = Σ r_i·f_i:
         s/2^u ≈ Σ r_i·I_i/M = k + V/M,  so  k = s>>u  (integer part)  and  the fractional bits are V/M.
         Bit (u−1) of s is set iff V/M ≥ 1/2 ⇔ V ≥ ∏src/2 — the centering bit, no big-int reconstruct.
         Robust because |ẽ| ≪ ∏src/2 ⇒ V sits near 0 or near ∏src, never near the ∏src/2 boundary."""
    if u is None:
        u = crns_safe_u(src)
    E, f, g = pre if pre is not None else crns_precompute(src, dst, u)
    s   = sum(res_src[i] * f[i] for i in range(len(src)))           # step 11: fixed-point accumulator
    k   = s >> u                                                    #          overflow  (integer part)
    neg = (s >> (u - 1)) & 1 if center else 0                       #          sign bit  (top frac bit)
    out = []
    for j in range(len(dst)):
        acc = sum(res_src[i] * E[i][j] for i in range(len(src)))    # step 12: x_M·E
        out.append((acc + k * g[j] - neg * (prod(src) % dst[j])) % dst[j])   # + k·g − neg·∏src
    return out


# --------------------------------------------------------------------------- #
# rescale + full multiply using CRNS (no m_sk anywhere)
# --------------------------------------------------------------------------- #
def rns_scale_crns(pp, e_res, S, round_bias=False, u=None):
    """DEVICE-FAITHFUL rescale of ONE poly, m_sk-free.  Same as ref.rns_scale but:
         step 2  Q -> S      APPROXIMATE FBC (drop ε; noise-absorbed)   [no msk row]
         step 3  S -> Q      EXACT FBC via CRNS                          [no sentinel]
       e_res = {'Q':[|Q| rows], 'S':[|S| rows]}  (msk row not needed).  Returns ẽ in base Q."""
    N, Q, t, Qp = pp.N, pp.Q, pp.t, pp.Qprod
    bias = (Qp // 2) if round_bias else 0
    Qinv = {s: pow(Qp % s, -1, s) for s in S}
    tilde = {s: [0] * N for s in S}                                  # ẽ residues over S only
    for n in range(N):
        fQ = [((t * e_res['Q'][i][n] + bias) % Q[i]) for i in range(len(Q))]
        y  = [fQ[i] * pow(Qp // Q[i], -1, Q[i]) % Q[i] for i in range(len(Q))]
        for s in S:
            fbc = sum(y[i] * ((Qp // Q[i]) % s) for i in range(len(Q))) % s
            fS  = (t * e_res['S'][S.index(s)][n] + bias) % s
            tilde[s][n] = ((fS - fbc) * Qinv[s]) % s
    if u is None:
        u = crns_safe_u(list(S))
    pre = crns_precompute(list(S), Q, u)                             # constants reused across coeffs
    outQ = [[0] * N for _ in range(len(Q))]
    for n in range(N):
        ex = _fbc_exact_crns([tilde[s][n] for s in S], list(S), Q, u=u, pre=pre, center=True)  # ẽ is signed
        for i in range(len(Q)):
            outQ[i][n] = ex[i]
    return outQ

def rns_ct_mul_crns(pp, ct1, ct2, evk, rng, S, u=None):
    """Full multiply with the CRNS rescale (no m_sk).  Mirrors ref.rns_ct_mul."""
    e = tensor_unsigned(pp, ct1, ct2)
    def rows(poly, base): return [[c % b for c in poly] for b in base]
    es = []
    for ek in e:
        er = {'Q': rows(ek, pp.Q), 'S': rows(ek, S)}                 # no 'msk'
        outQ = rns_scale_crns(pp, er, S, u=u)
        es.append([crt_reconstruct([outQ[i][n] for i in range(len(pp.Q))], pp.Q) for n in range(pp.N)])
    k0, k1 = relinearize(pp, rns_of(es[2], pp.Q), evk[0], evk[1], rng)
    k0i = [crt_reconstruct([k0[j][n] for j in range(len(pp.Q))], pp.Q) for n in range(pp.N)]
    k1i = [crt_reconstruct([k1[j][n] for j in range(len(pp.Q))], pp.Q) for n in range(pp.N)]
    return [poly_add(es[0], k0i, pp.Qprod), poly_add(es[1], k1i, pp.Qprod)]


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def unit_fbc_check(trials=200000, seed=0):
    """CRNS S->Q exact FBC vs (a) ground truth x mod q_j and (b) the m_sk path, random x < ∏S."""
    rng = random.Random(seed)
    pp = MulParams(LQ=3, LP=3, LB=4)
    S, Q, msk = list(pp.B), pp.Q, pp.msk
    M = prod(S)
    u = crns_safe_u(S)
    pre = crns_precompute(S, Q, u)
    bad_truth = bad_sk = 0
    for _ in range(trials):
        x = rng.randrange(0, M)
        rs = [x % s for s in S]
        want = [x % q for q in Q]
        got  = _fbc_exact_crns(rs, S, Q, u=u, pre=pre)
        sk   = _fbc_exact_sk(rs, S, Q, msk, x % msk)
        bad_truth += (got != want)
        bad_sk    += (got != sk)
    print(f"[unit] S->Q exact FBC   ∏S={M.bit_length()}b  |S|={len(S)}  safe u={u}")
    print(f"       CRNS vs truth  : {bad_truth}/{trials} mismatches   {'EXACT' if bad_truth==0 else 'FAIL'}")
    print(f"       CRNS vs m_sk   : {bad_sk}/{trials} mismatches   {'IDENTICAL' if bad_sk==0 else 'FAIL'}")
    return bad_truth == 0 and bad_sk == 0

def u_sweep(trials=40000, seed=1):
    """empirical minimum precision u for exactness, full-range x∈[0,∏S) vs BFV-bounded x<∏S/max(S)."""
    rng = random.Random(seed)
    pp = MulParams(LQ=3, LP=3, LB=4)
    S, Q = list(pp.B), pp.Q
    M = prod(S); ms = max(S)
    safe = crns_safe_u(S)
    print(f"[sweep] ∏S={M.bit_length()}b  worst-case-safe u={safe}   (BFV input ẽ < ∏S/max(S) ⇒ far less)")
    for u in range(28, safe + 4, 4):
        E, f, g = crns_precompute(S, Q, u)
        full = bnd = 0
        for _ in range(trials):
            x = rng.randrange(0, M)
            if _fbc_exact_crns([x % s for s in S], S, Q, u=u, pre=(E, f, g)) != [x % q for q in Q]:
                full += 1
            xb = rng.randrange(0, M // ms)                          # BFV: a full spare limb of headroom
            if _fbc_exact_crns([xb % s for s in S], S, Q, u=u, pre=(E, f, g)) != [xb % q for q in Q]:
                bnd += 1
        tag = "  <- guaranteed" if u >= safe else ""
        print(f"        u={u:3d}: full[0,∏S) mism={full:5d}/{trials}   BFV-bounded mism={bnd:5d}/{trials}{tag}")

def e2e_cross_check(N=1024, t=12289, seeds=(0, 1, 7, 42), LQ=3):
    """(a) CRNS rescale bit-identical to m_sk rescale on the ACTUAL tensor (rng-free isolation), and
       (b) full CRNS multiply decrypts to m1·m2."""
    ok = True
    for seed in seeds:
        rng = random.Random(seed)
        pp = MulParams(N=N, LQ=LQ, LP=LQ, LB=4, t=t)
        S = pp.B
        msk = [p for p in ntt_primes(80, N) if p not in set(pp.Q + pp.P + pp.B)][0]
        s = small_poly(N, rng); evk = keygen_relin(pp, s, rng)
        m1 = [rng.randrange(t) for _ in range(N)]; m2 = [rng.randrange(t) for _ in range(N)]
        ct1, ct2 = encrypt(pp, m1, s, rng), encrypt(pp, m2, s, rng)

        # (a) isolate the rescale: same tensor residues into both bconv paths, compare ẽ over base Q.
        e = tensor_unsigned(pp, ct1, ct2)
        def rows(poly, base): return [[c % b for c in poly] for b in base]
        scale_id = True
        for ek in e:
            er_full = {'Q': rows(ek, pp.Q), 'S': rows(ek, S), 'msk': rows(ek, [msk])}
            er_crns = {'Q': rows(ek, pp.Q), 'S': rows(ek, S)}
            sk   = ref.rns_scale(pp, er_full, S, msk)
            crns = rns_scale_crns(pp, er_crns, S)
            scale_id &= (sk == crns)

        # (b) full CRNS multiply decrypts correctly (single rng; relin randomness cancels on decrypt).
        out_crns = rns_ct_mul_crns(pp, ct1, ct2, evk, rng, S)
        want = negacyclic_mul(m1, m2, t)
        mm = sum(x != y for x, y in zip(decrypt(pp, out_crns, s), want))
        good = scale_id and mm == 0
        ok &= good
        print(f"seed={seed:>2}: CRNS rescale ≡ m_sk rescale {'YES' if scale_id else 'NO '}   "
              f"CRNS multiply decrypt mism={mm}/{N}   [{'PASS' if good else 'FAIL'}]")
    print(f"=> CRNS rescale bit-identical to m_sk and CRNS multiply decrypts to m1·m2  "
          f"{'ALL PASS' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    print("=== CRNS exact base conversion — verification vs m_sk reference ===\n")
    print("1) unit S->Q FBC (random x < ∏S)"); a = unit_fbc_check()
    print("\n2) precision sweep");             u_sweep()
    print("\n3) end-to-end HE multiply cross-check (CRNS rescale vs m_sk rescale)")
    b = e2e_cross_check()
    print(f"\nRESULT: {'ALL VERIFIED' if (a and b) else 'FAILURES ABOVE'}")
