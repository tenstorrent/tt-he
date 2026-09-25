# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""builders.py — L1 program builders for real HE workloads.

Constructs actual computations (he.add, tensor) as L1 `Program`s using ir.py, then validates
them END TO END against the crypto reference (reference/he/bfv_*_ref.py): build -> run through
the golden -> reconstruct big-int -> decrypt, and check it equals the plaintext result. This is
the proof that the L1 op set *composes* into correct homomorphic evaluation, not just passes in
isolation.

This is the Python analogue of host/lib/he/he_api.cpp, but it EMITS an L1 Program (data) instead
of dispatching kernels — the "host builds the L1 program" role (FORMALIZATION scope decision).

Run: python3 builders.py    (needs algorithm/he on the path; done below.)

Status: he.add (full, decrypts correct) + tensor (NTT-domain multiply, exact mod Q). The
scale (BEHZ rescale) and relin subgraphs are the next builder increments — same pattern, more
ops (base_convert / crt).
"""
from __future__ import annotations
import os, sys, random

import _paths                                     # noqa: F401  (repo layout; see _paths.py)
from bfv_relin_ref import (ntt_primes, crt_reconstruct, rns_of, small_poly,   # noqa: E402
                           negacyclic_mul, poly_add, keygen_relin, relinearize)
from bfv_mul_ref import MulParams, encrypt, decrypt, tensor as ref_tensor, scale  # noqa: E402

import reference as ref            # noqa: E402
import ir                          # noqa: E402
from ir import Value, Op, Program  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def recon_poly(res_limbs, base):
    """[residue-poly per limb] -> big-int poly via CRT (per coefficient)."""
    N = len(res_limbs[0])
    return [crt_reconstruct([int(res_limbs[i][n]) for i in range(len(base))], base) for n in range(N)]


# --------------------------------------------------------------------------- #
# he.add:  per-limb rns.add over the ciphertext RNS residues
# --------------------------------------------------------------------------- #
def build_he_add(pp, ct1, ct2):
    """L1 Program for (c0,c1)+(d0,d1). Returns (program, inputs_dict, out_names)."""
    Q = pp.Q
    a0, a1 = rns_of(ct1[0], Q), rns_of(ct1[1], Q)
    b0, b1 = rns_of(ct2[0], Q), rns_of(ct2[1], Q)
    N = pp.N
    inputs, in_vals, ops, o0v, o1v = {}, [], [], [], []
    for i, q in enumerate(Q):
        def mkin(tag, data):
            v = Value(f"{tag}{i}", (N,), q=q, domain="coeff")
            in_vals.append(v); inputs[v.name] = data; return v
        va0, va1 = mkin("a0_", a0[i]), mkin("a1_", a1[i])
        vb0, vb1 = mkin("b0_", b0[i]), mkin("b1_", b1[i])
        o0 = Value(f"o0_{i}", (N,), q=q, domain="coeff")
        o1 = Value(f"o1_{i}", (N,), q=q, domain="coeff")
        ops.append(Op("rns.add", [va0, vb0], o0))
        ops.append(Op("rns.add", [va1, vb1], o1))
        o0v.append(o0); o1v.append(o1)
    prog = Program(inputs=in_vals, ops=ops, outputs=o0v + o1v)
    return prog, inputs, ([v.name for v in o0v], [v.name for v in o1v])


# --------------------------------------------------------------------------- #
# tensor (ct_mul_raw):  per-limb ntt -> pointwise (e0,e1,e2) -> intt
# --------------------------------------------------------------------------- #
def build_tensor(pp, ct1, ct2):
    """L1 Program for (c0,c1)x(d0,d1) -> (e0,e1,e2), NTT-domain, over base Q. Returns
    (program, inputs, {k: [name per limb]})."""
    Q, N = pp.Q, pp.N
    c0, c1 = rns_of(ct1[0], Q), rns_of(ct1[1], Q)
    d0, d1 = rns_of(ct2[0], Q), rns_of(ct2[1], Q)
    inputs, in_vals, ops = {}, [], []
    outs = {0: [], 1: [], 2: []}
    for i, q in enumerate(Q):
        psi = ref.find_2nth_root(q, N)
        def mkin(tag, data):
            v = Value(f"{tag}{i}", (N,), q=q, domain="coeff")
            in_vals.append(v); inputs[v.name] = data; return v
        vc0, vc1 = mkin("c0_", c0[i]), mkin("c1_", c1[i])
        vd0, vd1 = mkin("d0_", d0[i]), mkin("d1_", d1[i])

        def ntt(v, tag):
            r = Value(f"{tag}{i}", (N,), q=q, domain="eval")
            ops.append(Op("rns.ntt", [v], r, {"psi": psi, "N": N})); return r
        C0, C1, D0, D1 = ntt(vc0, "C0_"), ntt(vc1, "C1_"), ntt(vd0, "D0_"), ntt(vd1, "D1_")

        def mul(x, y, tag):
            r = Value(f"{tag}{i}", (N,), q=q, domain="eval")
            ops.append(Op("rns.mul", [x, y], r)); return r
        E0 = mul(C0, D0, "E0_"); E2 = mul(C1, D1, "E2_")
        T0 = mul(C0, D1, "T0_"); T1 = mul(C1, D0, "T1_")
        E1 = Value(f"E1_{i}", (N,), q=q, domain="eval")
        ops.append(Op("rns.add", [T0, T1], E1))

        for k, Ek in ((0, E0), (1, E1), (2, E2)):
            e = Value(f"e{k}_{i}", (N,), q=q, domain="coeff")
            ops.append(Op("rns.intt", [Ek], e, {"psi": psi, "N": N}))
            outs[k].append(e.name)
    allout = [Value(n, (N,), q=Q[0]) for k in (0, 1, 2) for n in outs[k]]
    prog = Program(inputs=in_vals, ops=ops, outputs=allout)
    return prog, inputs, outs


# --------------------------------------------------------------------------- #
# full ct_mul:  tensor -> rescale -> relin -> add   (composite-level L1 program)
#   tensor is decomposed to primitives separately (build_tensor); here the stages are
#   `macro` nodes backed by the exact reference (rescale/relin ARE composites — ops.md §1b).
# --------------------------------------------------------------------------- #
def _L(arr):
    return [int(v) for v in arr]

def _relin(pp, e2_poly, evk, rng):
    k0, k1 = relinearize(pp, rns_of(_L(e2_poly), pp.Q), evk[0], evk[1], rng)
    ki = lambda k: [crt_reconstruct([k[j][n] for j in range(len(pp.Q))], pp.Q) for n in range(pp.N)]
    return [ki(k0), ki(k1)]

def build_ct_mul(pp, ct1, ct2, evk, rng):
    """L1 Program for the full BFV multiply. Values are ciphertext polys (big-int / mod Q)."""
    N = pp.N
    def P(n): return Value(n, (N,), q=None, domain="coeff")
    c0, c1, d0, d1 = P("c0"), P("c1"), P("d0"), P("d1")
    e0, e1, e2 = P("e0"), P("e1"), P("e2")
    t0, t1, t2 = P("et0"), P("et1"), P("et2")
    k0, k1 = P("k0"), P("k1")
    out0, out1 = P("out0"), P("out1")
    ops = [
        Op("macro", [c0, c1, d0, d1], [e0, e1, e2],
           {"name": "tensor", "fn": lambda a, b, cc, dd: ref_tensor(pp, [_L(a), _L(b)], [_L(cc), _L(dd)])}),
        Op("macro", [e0, e1, e2], [t0, t1, t2],
           {"name": "rescale", "fn": lambda x, y, z: scale(pp, [_L(x), _L(y), _L(z)])}),
        Op("macro", [t2], [k0, k1],
           {"name": "relin", "fn": lambda z: _relin(pp, z, evk, rng)}),
        Op("macro", [t0, k0], out0, {"name": "add", "fn": lambda x, y: poly_add(_L(x), _L(y), pp.Qprod)}),
        Op("macro", [t1, k1], out1, {"name": "add", "fn": lambda x, y: poly_add(_L(x), _L(y), pp.Qprod)}),
    ]
    prog = Program(inputs=[c0, c1, d0, d1], ops=ops, outputs=[out0, out1])
    inputs = {"c0": ct1[0], "c1": ct1[1], "d0": ct2[0], "d1": ct2[1]}
    return prog, inputs


# --------------------------------------------------------------------------- #
# full ct_mul as PRIMITIVE L1 ops:  tensor over Q∪S -> rescale -> relin -> add.
# Unlike build_ct_mul (macro stand-ins), rescale/relin are decomposed into rns.* primitives
# (base_convert = CRNS exact + approximate), so the whole multiply lowers L1->L2->L3 and schedules.
# Validated by decrypt (oracle: bfv_bconv_crns.rns_ct_mul_crns). docs/ir/crns_bconv_lowering.md.
# --------------------------------------------------------------------------- #
def _const(ops, consts, name, arr, shape, q):
    v = Value(name, shape, q=q)
    if name not in consts:
        consts[name] = ref.A(arr); ops.append(Op("rns.const", [], v, {"name": name}))
    return v


def _emit_tensor_base(pp, base, c0, c1, d0, d1, ops):
    """NTT-domain tensor (c0,c1)×(d0,d1) over `base` (per limb) -> {k:[Value per limb]} e0,e1,e2."""
    N = pp.N
    outs = {0: [], 1: [], 2: []}
    for i, q in enumerate(base):
        psi = ref.find_2nth_root(q, N)
        def ntt(v, nm):
            r = Value(f"{nm}{i}", (N,), q=q, domain="eval"); ops.append(Op("rns.ntt", [v], r, {"psi": psi, "N": N})); return r
        C0, C1, D0, D1 = ntt(c0[i], "tC0"), ntt(c1[i], "tC1"), ntt(d0[i], "tD0"), ntt(d1[i], "tD1")
        def mul(x, y, nm):
            r = Value(f"{nm}{i}", (N,), q=q, domain="eval"); ops.append(Op("rns.mul", [x, y], r)); return r
        E0, E2, T0, T1 = mul(C0, D0, "tE0"), mul(C1, D1, "tE2"), mul(C0, D1, "tT0"), mul(C1, D0, "tT1")
        E1 = Value(f"tE1{i}", (N,), q=q, domain="eval"); ops.append(Op("rns.add", [T0, T1], E1))
        for k, Ek in ((0, E0), (1, E1), (2, E2)):
            e = Value(f"e{k}_{i}", (N,), q=q, domain="coeff"); ops.append(Op("rns.intt", [Ek], e, {"psi": psi, "N": N}))
            outs[k].append(e)
    return outs


def _emit_rescale(pp, tag, eQ, eS, ops, consts):
    """ẽ = ⌊t·e/Q⌉ over base Q (BEHZ fast-RNS-floor via CRNS). eQ/eS = e residues over Q / S."""
    Q, S, t, Qp, N = pp.Q, list(pp.B), pp.t, pp.Qprod, pp.N
    tC = _const(ops, consts, "scal_t", [t], (1,), None)
    fQ = [Value(f"{tag}_fQ{i}", (N,), q=q) for i, q in enumerate(Q)]
    for i in range(len(Q)): ops.append(Op("rns.mul", [eQ[i], tC], fQ[i]))          # f = t·e in base Q
    fS = [Value(f"{tag}_fS{i}", (N,), q=s) for i, s in enumerate(S)]
    for i in range(len(S)): ops.append(Op("rns.mul", [eS[i], tC], fS[i]))          # f = t·e in base S
    fbc = [Value(f"{tag}_fbc{i}", (N,), q=s) for i, s in enumerate(S)]
    ops.append(Op("rns.base_convert", fQ, fbc, {"exact": False}))                   # approx FBC Q->S
    etS = [Value(f"{tag}_etS{i}", (N,), q=s) for i, s in enumerate(S)]
    for i, s in enumerate(S):
        d = Value(f"{tag}_d{i}", (N,), q=s); ops.append(Op("rns.sub", [fS[i], fbc[i]], d))
        qc = _const(ops, consts, f"Qinv_{s}", [pow(Qp % s, -1, s)], (1,), None)
        ops.append(Op("rns.mul", [d, qc], etS[i]))                                  # ẽ_s = (f_s − fbc)·Q⁻¹
    etQ = [Value(f"{tag}_etQ{i}", (N,), q=q) for i, q in enumerate(Q)]
    ubnd = (len(S) * max(S)).bit_length() + 8               # BFV-bounded ẽ ⇒ small exact-FBC precision
    ops.append(Op("rns.base_convert", etS, etQ, {"exact": True, "signed": True, "u": ubnd}))  # exact CRNS S->Q
    return etQ


def _emit_relin(pp, tag, c2Q, evk0, evk1, ops, consts):
    """relin c2 (over Q) -> (k0,k1) over Q: mod-up (approx FBC Q->P) -> ×evk (negacyclic) -> mod-down."""
    Q, P, QP, N, Pp = pp.Q, pp.P, pp.QP, pp.N, pp.Pprod
    LQ = len(Q)
    c2P = [Value(f"{tag}_c2P{i}", (N,), q=p) for i, p in enumerate(P)]
    ops.append(Op("rns.base_convert", list(c2Q), c2P, {"exact": False}))            # mod-up Q->P (approx)
    c2QP = list(c2Q) + c2P
    # CSE: NTT each c2QP limb ONCE (the same transform feeds both evk0 and evk1) — reused below.
    X = []
    for li, q in enumerate(QP):
        psi = ref.find_2nth_root(q, N)
        xv = Value(f"{tag}_X{li}", (N,), q=q, domain="eval")
        ops.append(Op("rns.ntt", [c2QP[li]], xv, {"psi": psi, "N": N}))
        X.append((xv, psi))
    def keymul(evk, kk):                                                            # acc = c2·evk (negacyclic)
        acc = []
        for li, q in enumerate(QP):
            xv, psi = X[li]
            # NTT the evk through the SAME lowering (bake coeff-domain, emit rns.ntt) so it lands in
            # the same eval-domain order as X — otherwise a four-step NTT's permuted order mismatches
            # a pre-baked ref.ntt const.
            evc = _const(ops, consts, f"{tag}_evk{kk}_{li}c", [int(x) % q for x in evk[li]], (N,), q)
            ev = Value(f"{tag}_evk{kk}_{li}", (N,), q=q, domain="eval"); ops.append(Op("rns.ntt", [evc], ev, {"psi": psi, "N": N}))
            Y = Value(f"{tag}_{kk}Y{li}", (N,), q=q, domain="eval"); ops.append(Op("rns.mul", [xv, ev], Y))
            a = Value(f"{tag}_{kk}a{li}", (N,), q=q, domain="coeff"); ops.append(Op("rns.intt", [Y], a, {"psi": psi, "N": N}))
            acc.append(a)
        return acc
    acc0, acc1 = keymul(evk0, "A"), keymul(evk1, "B")
    def moddown(acc, kk):                                                           # QP -> Q: (aQ − FBC(aP))·P⁻¹
        aQ, aP = acc[:LQ], acc[LQ:]
        xP = [Value(f"{tag}_{kk}xP{i}", (N,), q=q) for i, q in enumerate(Q)]
        ops.append(Op("rns.base_convert", aP, xP, {"exact": False}))                # mod-down P->Q (approx)
        out = []
        for j, qj in enumerate(Q):
            d = Value(f"{tag}_{kk}d{j}", (N,), q=qj); ops.append(Op("rns.sub", [aQ[j], xP[j]], d))
            pc = _const(ops, consts, f"Pinv_{qj}", [pow(Pp % qj, -1, qj)], (1,), None)
            o = Value(f"{tag}_{kk}o{j}", (N,), q=qj); ops.append(Op("rns.mul", [d, pc], o)); out.append(o)
        return out
    return moddown(acc0, "A"), moddown(acc1, "B")


def _emit_relin_dnum(pp, tag, c2Q, evks, groups, ops, consts):
    """Hybrid (dnum) key-switch of c2 (over Q) -> (k0,k1) over Q. Q split into `dnum` coprime groups;
       per group: RNS digit -> mod-up (approx FBC Q_j -> rest of QP) -> ×evk_j (negacyclic) -> ACCUM;
       then mod-down QP->Q. dnum=1 coincides with _emit_relin. Ref: he/bfv_relin_dnum.py."""
    Q, P, QP, N, Pp, Qp = pp.Q, pp.P, pp.QP, pp.N, pp.Pprod, pp.Qprod
    LQ, LP = len(Q), len(P)
    acc0, acc1 = [None] * len(QP), [None] * len(QP)
    for j, (g, (evkb, evka)) in enumerate(zip(groups, evks)):
        Qj = 1
        for i in g:
            Qj *= Q[i]
        cinv = pow((Qp // Qj) % Qj, -1, Qj)                              # D_j^{-1} mod Q_j, D_j=Q/Q_j
        dig = []                                                          # digit_j residues over Q_j
        for i in g:
            cc = _const(ops, consts, f"{tag}_dc{j}_{Q[i]}", [cinv % Q[i]], (1,), None)
            d = Value(f"{tag}_dig{j}_{i}", (N,), q=Q[i]); ops.append(Op("rns.mul", [c2Q[i], cc], d)); dig.append(d)
        tgt = [li for li in range(len(QP)) if li not in g]                # mod-up: Q_j -> the other QP limbs
        outv = [Value(f"{tag}_du{j}_{li}", (N,), q=QP[li]) for li in tgt]
        ops.append(Op("rns.base_convert", dig, outv, {"exact": False}))
        digQP = [None] * len(QP)
        for idx, i in enumerate(g):
            digQP[i] = dig[idx]
        for li, v in zip(tgt, outv):
            digQP[li] = v
        for li, q in enumerate(QP):                                      # ×evk_j (negacyclic) + accumulate
            psi = ref.find_2nth_root(q, N)
            X = Value(f"{tag}_X{j}_{li}", (N,), q=q, domain="eval"); ops.append(Op("rns.ntt", [digQP[li]], X, {"psi": psi, "N": N}))
            evbc = _const(ops, consts, f"{tag}_evkb{j}_{li}c", [int(x) % q for x in evkb[li]], (N,), q)
            evac = _const(ops, consts, f"{tag}_evka{j}_{li}c", [int(x) % q for x in evka[li]], (N,), q)
            evb = Value(f"{tag}_evkb{j}_{li}", (N,), q=q, domain="eval"); ops.append(Op("rns.ntt", [evbc], evb, {"psi": psi, "N": N}))
            eva = Value(f"{tag}_evka{j}_{li}", (N,), q=q, domain="eval"); ops.append(Op("rns.ntt", [evac], eva, {"psi": psi, "N": N}))
            Y0 = Value(f"{tag}_Y0_{j}_{li}", (N,), q=q, domain="eval"); ops.append(Op("rns.mul", [X, evb], Y0))
            Y1 = Value(f"{tag}_Y1_{j}_{li}", (N,), q=q, domain="eval"); ops.append(Op("rns.mul", [X, eva], Y1))
            a0 = Value(f"{tag}_a0_{j}_{li}", (N,), q=q, domain="coeff"); ops.append(Op("rns.intt", [Y0], a0, {"psi": psi, "N": N}))
            a1 = Value(f"{tag}_a1_{j}_{li}", (N,), q=q, domain="coeff"); ops.append(Op("rns.intt", [Y1], a1, {"psi": psi, "N": N}))
            if acc0[li] is None:
                acc0[li], acc1[li] = a0, a1
            else:
                n0 = Value(f"{tag}_acc0_{j}_{li}", (N,), q=q); ops.append(Op("rns.add", [acc0[li], a0], n0)); acc0[li] = n0
                n1 = Value(f"{tag}_acc1_{j}_{li}", (N,), q=q); ops.append(Op("rns.add", [acc1[li], a1], n1)); acc1[li] = n1
    def moddown(acc, kk):                                                # QP -> Q : (aQ − FBC(aP))·P⁻¹
        aQ, aP = acc[:LQ], acc[LQ:]
        xP = [Value(f"{tag}_{kk}xP{i}", (N,), q=q) for i, q in enumerate(Q)]
        ops.append(Op("rns.base_convert", aP, xP, {"exact": False}))
        out = []
        for jj, qj in enumerate(Q):
            d = Value(f"{tag}_{kk}d{jj}", (N,), q=qj); ops.append(Op("rns.sub", [aQ[jj], xP[jj]], d))
            pc = _const(ops, consts, f"Pinv_{qj}", [pow(Pp % qj, -1, qj)], (1,), None)
            o = Value(f"{tag}_{kk}o{jj}", (N,), q=qj); ops.append(Op("rns.mul", [d, pc], o)); out.append(o)
        return out
    return moddown(acc0, "A"), moddown(acc1, "B")


def build_ct_mul_rns(pp, ct1, ct2, evk0=None, evk1=None, ks=None):
    """Full BFV multiply as a PRIMITIVE L1 program (tensor over Q∪S -> rescale -> relin -> add).
       relin: dnum=1 via (evk0,evk1); hybrid dnum>1 via ks=(evks, groups) (he/bfv_relin_dnum.py).
       Returns (program, inputs, out0_names, out1_names); decrypts to m1·m2 (validated below)."""
    Q, S, N = pp.Q, list(pp.B), pp.N
    base = list(Q) + S; LQ = len(Q)
    ops, consts, inputs, in_vals = [], {}, {}, []
    def mkin(tag, poly):
        vs = []
        for i, q in enumerate(base):
            v = Value(f"{tag}{i}", (N,), q=q, domain="coeff")
            inputs[v.name] = [c % q for c in poly]; in_vals.append(v); vs.append(v)
        return vs
    c0, c1, d0, d1 = mkin("c0", ct1[0]), mkin("c1", ct1[1]), mkin("d0", ct2[0]), mkin("d1", ct2[1])
    e = _emit_tensor_base(pp, base, c0, c1, d0, d1, ops)
    et = {k: _emit_rescale(pp, f"rs{k}", e[k][:LQ], e[k][LQ:LQ + len(S)], ops, consts) for k in (0, 1, 2)}
    if ks is not None:
        k0, k1 = _emit_relin_dnum(pp, "rl", et[2], ks[0], ks[1], ops, consts)
    else:
        k0, k1 = _emit_relin(pp, "rl", et[2], evk0, evk1, ops, consts)
    out0 = [Value(f"out0_{j}", (N,), q=q) for j, q in enumerate(Q)]
    out1 = [Value(f"out1_{j}", (N,), q=q) for j, q in enumerate(Q)]
    for j in range(LQ):
        ops.append(Op("rns.add", [et[0][j], k0[j]], out0[j]))
        ops.append(Op("rns.add", [et[1][j], k1[j]], out1[j]))
    prog = Program(inputs=in_vals, ops=ops, outputs=out0 + out1, consts=consts)
    return prog, inputs, [v.name for v in out0], [v.name for v in out1]


# --------------------------------------------------------------------------- #
# CKKS ct_mul as PRIMITIVE L1 ops:  tensor over Q -> relin -> add -> rescale (drop primes).
#
# The ORDER is what differs from BFV, and it is the whole difference:
#     BFV    tensor over Q u S -> rescale (BEHZ two-base floor) -> relin -> add
#     CKKS   tensor over Q     -> relin -> add -> rescale (drop a prime)
# so CKKS needs NO aux base S, and its rescale is a modulus change + subtract + scalar multiply per
# surviving limb instead of a two-base FBC over a wide accumulator. Everything else -- the hybrid
# dnum key-switch, the approximate FBC, the NTT/iNTT structure -- is shared with BFV verbatim:
# _emit_relin_dnum reads only pp.{Q,P,QP,N,Pprod,Qprod}, which CKKSParams provides, so it is reused
# unchanged rather than forked.
#
# Oracle: reference/he/ckks_mul_ref.py (exact -- round(dec/scale) == negacyclic m1*m2).
# --------------------------------------------------------------------------- #
def _emit_rescale_ckks(pp, tag, xQ, base, ndrop, ops, consts):
    """Drop the last `ndrop` primes of `base`:  out_i = (x_i - (x_L mod q_i)) * q_L^-1  (mod q_i).

    The modulus change x_L -> (x_L mod q_i) is emitted as a ONE-SOURCE rns.base_convert. With a single
    source the approximate FBC degenerates to an exact modulus change (M = src[0], the pre-scale factor
    is 1, and the extension coefficient (M/src[0]) mod d_j is 1), so `exact=False` costs nothing here
    and lowers to the cheap kernel -- this is the `bconv_one` of the hardware pipeline.

    UNCENTRED: computes floor(x/q_L) from the unsigned representative. For a value standing for a
    negative number the two differ by exactly Q/q_L, which IS the surviving modulus, so the result is
    correct in the new base either way (the bench carries the same choice as TG18NTT_NOCENTRE)."""
    N = pp.N
    cur, cb = list(xQ), list(base)
    for step in range(ndrop):
        qL, xL = cb[-1], cur[-1]
        keep_b = cb[:-1]
        lifted = [Value(f"{tag}_lift{step}_{i}", (N,), q=qi) for i, qi in enumerate(keep_b)]
        ops.append(Op("rns.base_convert", [xL], lifted, {"exact": False}))
        out = []
        for i, qi in enumerate(keep_b):
            d = Value(f"{tag}_rsd{step}_{i}", (N,), q=qi)
            ops.append(Op("rns.sub", [cur[i], lifted[i]], d))
            ic = _const(ops, consts, f"qLinv_{qL}_{qi}", [pow(qL % qi, -1, qi)], (1,), None)
            o = Value(f"{tag}_rs{step}_{i}", (N,), q=qi)
            ops.append(Op("rns.mul", [d, ic], o))
            out.append(o)
        cur, cb = out, keep_b
    return cur, cb


def build_ct_mul_ckks(pp, ct1, ct2, ks, ndrop=1):
    """Full CKKS multiply as a PRIMITIVE L1 program. ks=(evks, groups) from keygen_relin_dnum.

    Returns (program, inputs, out0_names, out1_names, surviving_base)."""
    Q, N = list(pp.Q), pp.N
    ops, consts, inputs, in_vals = [], {}, {}, []

    def mkin(tag, poly):
        vs = []
        for i, q in enumerate(Q):
            v = Value(f"{tag}{i}", (N,), q=q, domain="coeff")
            inputs[v.name] = [c % q for c in poly]
            in_vals.append(v)
            vs.append(v)
        return vs

    c0, c1, d0, d1 = mkin("c0", ct1[0]), mkin("c1", ct1[1]), mkin("d0", ct2[0]), mkin("d1", ct2[1])
    e = _emit_tensor_base(pp, Q, c0, c1, d0, d1, ops)          # -> e[0], e[1], e[2] over Q
    k0, k1 = _emit_relin_dnum(pp, "rl", e[2], ks[0], ks[1], ops, consts)
    a0 = [Value(f"acc0_{j}", (N,), q=q) for j, q in enumerate(Q)]
    a1 = [Value(f"acc1_{j}", (N,), q=q) for j, q in enumerate(Q)]
    for j in range(len(Q)):
        ops.append(Op("rns.add", [e[0][j], k0[j]], a0[j]))
        ops.append(Op("rns.add", [e[1][j], k1[j]], a1[j]))
    r0, base = _emit_rescale_ckks(pp, "rsA", a0, Q, ndrop, ops, consts)
    r1, _ = _emit_rescale_ckks(pp, "rsB", a1, Q, ndrop, ops, consts)
    prog = Program(inputs=in_vals, ops=ops, outputs=r0 + r1, consts=consts)
    return prog, inputs, [v.name for v in r0], [v.name for v in r1], base


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _validate_he_add():
    pp = MulParams(N=16)
    rng = random.Random(1)
    s = small_poly(pp.N, rng)
    m1 = [rng.randrange(pp.t) for _ in range(pp.N)]
    m2 = [rng.randrange(pp.t) for _ in range(pp.N)]
    ct1, ct2 = encrypt(pp, m1, s, rng), encrypt(pp, m2, s, rng)

    prog, inputs, (o0n, o1n) = build_he_add(pp, ct1, ct2)
    env = ir.run(prog, inputs)
    out_c0 = recon_poly([env[n] for n in o0n], pp.Q)
    out_c1 = recon_poly([env[n] for n in o1n], pp.Q)
    dec = decrypt(pp, [out_c0, out_c1], s)
    exp = [(a + b) % pp.t for a, b in zip(m1, m2)]
    ok = dec == exp
    print(f"  he.add : decrypt == m1+m2 (mod t) : {'OK' if ok else 'FAIL'}   ({len(prog.ops)} ops)")
    return ok


def _validate_tensor():
    pp = MulParams(N=16)
    rng = random.Random(2)
    s = small_poly(pp.N, rng)
    m1 = [rng.randrange(pp.t) for _ in range(pp.N)]
    m2 = [rng.randrange(pp.t) for _ in range(pp.N)]
    ct1, ct2 = encrypt(pp, m1, s, rng), encrypt(pp, m2, s, rng)

    prog, inputs, outs = build_tensor(pp, ct1, ct2)
    env = ir.run(prog, inputs)
    e_ref = ref_tensor(pp, ct1, ct2)                       # exact over Z
    ok = True
    for k in (0, 1, 2):
        got = recon_poly([env[n] for n in outs[k]], pp.Q)  # via CRT over Q
        want = [v % pp.Qprod for v in e_ref[k]]            # reference e_k reduced mod Q
        ok &= (got == want)
    print(f"  tensor : NTT-domain (e0,e1,e2) == ref mod Q : {'OK' if ok else 'FAIL'}   ({len(prog.ops)} ops)")
    return ok


def _validate_ct_mul():
    pp = MulParams(N=16)
    rng = random.Random(4)
    s = small_poly(pp.N, rng)
    evk = keygen_relin(pp, s, rng)
    m1 = [rng.randrange(pp.t) for _ in range(pp.N)]
    m2 = [rng.randrange(pp.t) for _ in range(pp.N)]
    ct1, ct2 = encrypt(pp, m1, s, rng), encrypt(pp, m2, s, rng)

    prog, inputs = build_ct_mul(pp, ct1, ct2, evk, rng)
    env = ir.run(prog, inputs)
    dec = decrypt(pp, [env["out0"], env["out1"]], s)
    exp = negacyclic_mul(m1, m2, pp.t)                     # m1·m2 in R_t
    ok = dec == exp
    print(f"  ct_mul : decrypt == m1*m2 (mod t) : {'OK' if ok else 'FAIL'}   ({len(prog.ops)} stage-ops)")
    return ok


def _validate_ct_mul_rns():
    from lower import lower_l1_to_l2, lower_l2_to_l3, fuse_a, FUSE_A_ALL   # noqa: E402
    pp = MulParams(N=16, LQ=3, LP=3, LB=4)
    ok, nops, mmr = True, 0, ""
    def _mm(p): return sum(1 for op in p.ops if op.opcode == "tile.matmul_modq")
    FUSE_CFGS = [("ntt",), ("baseconv",), FUSE_A_ALL]      # each strategy on its own, then together
    for seed in (0, 1, 4, 7, 42):
        rng = random.Random(seed)
        s = small_poly(pp.N, rng)
        evk0, evk1 = keygen_relin(pp, s, rng)
        m1 = [rng.randrange(pp.t) for _ in range(pp.N)]
        m2 = [rng.randrange(pp.t) for _ in range(pp.N)]
        ct1, ct2 = encrypt(pp, m1, s, rng), encrypt(pp, m2, s, rng)
        prog, inputs, o0n, o1n = build_ct_mul_rns(pp, ct1, ct2, evk0, evk1)
        nops = len(prog.ops)
        want = negacyclic_mul(m1, m2, pp.t)
        l2 = lower_l1_to_l2(prog)
        checks = [prog, l2, lower_l2_to_l3(l2)]            # L1 == L2 == L3
        counts = {}
        for cfg in FUSE_CFGS:                              # + each Fusion-A strategy config
            l2f = fuse_a(l2, cfg)
            counts[cfg] = _mm(l2f)
            checks += [l2f, lower_l2_to_l3(l2f)]
        for p in checks:
            env = ir.run(p, inputs)
            o0 = recon_poly([env[n] for n in o0n], pp.Q)
            o1 = recon_poly([env[n] for n in o1n], pp.Q)
            ok &= (decrypt(pp, [o0, o1], s) == want)
        mmr = (f"none {_mm(l2)}, ntt {counts[('ntt',)]}, baseconv {counts[('baseconv',)]}, "
               f"both {counts[FUSE_A_ALL]}")
    print(f"  ct_mul_rns : decrypt == m1*m2, L1==L2==L3 (Fusion-A matmuls: {mmr}), all seeds : "
          f"{'OK' if ok else 'FAIL'}   ({nops} L1 ops)")
    return ok


if __name__ == "__main__":
    print("L1 program builders — end-to-end vs crypto reference (bfv_*_ref.py):")
    a = _validate_he_add()
    t = _validate_tensor()
    c = _validate_ct_mul()
    cr = _validate_ct_mul_rns()
    print("\n--- he.add L1 program (first few ops) ---")
    prog, _, _ = build_he_add(MulParams(N=16), encrypt(MulParams(N=16), [0]*16, small_poly(16, random.Random(0)), random.Random(0)),
                              encrypt(MulParams(N=16), [0]*16, small_poly(16, random.Random(0)), random.Random(0)))
    print("\n".join(ir.print_program(prog).splitlines()[:8]) + "\n  ...")
    ok = a and t and c and cr
    print("\nBUILDERS OK" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)
