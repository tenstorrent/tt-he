# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""lower.py — stage 3: L1 (RNS-poly) -> L2 (tile / kernel) lowering pass.

Rewrites each L1 op into equivalent L2 tile ops (docs/ir/FORMALIZATION §1.5 stage 3). At tile
scale the NTT IS a matmul by the (baked) DFT matrix, so:

    rns.ntt / rns.intt  ->  tile.const(W) ; tile.matmul_modq(W, x)     [W baked at plan time]
    rns.mul             ->  tile.mul_modq
    rns.add             ->  tile.add_modq
    rns.sub             ->  tile.sub_modq

macro / already-L2 ops pass through. The lowering is **meaning-preserving**: run() the L1 and
L2 programs and they produce bit-identical results (checked in __main__ against the tensor
program that decrypts correctly). A large NTT matmul (N>32) is further tiled to 32x32 by the
ntt_matmul K/M/N-tiling (a follow-up); here N<=32 stays one matmul.

(Stage 4, L2->L3: each tile.matmul_modq expands to decompose + int8 matmul + recombine — already
proven equivalent in reference.matmul_modq_via_int8; that pass reuses ntt_matmul.lower_matmul_int8.)
"""
from __future__ import annotations
import reference as ref
from ir import Value, Op, Program


def ntt_matrix(N, q, psi, inverse=False):
    """The negacyclic NTT/INTT as an NxN matrix W so that (I)NTT(x) = W @ x mod q."""
    if not inverse:
        # ntt(x)[k] = Σ_j x[j] psi^{j(2k+1)}
        return [[pow(psi, (j * (2 * k + 1)) % (2 * N), q) for j in range(N)] for k in range(N)]
    omega_inv = pow(psi * psi % q, -1, q)
    psi_inv = pow(psi, -1, q)
    Ninv = pow(N, -1, q)
    # intt(X)[j] = Ninv · psi^{-j} · Σ_k X[k] omega^{-jk}
    return [[Ninv * pow(psi_inv, j, q) % q * pow(omega_inv, (j * k) % N, q) % q for k in range(N)]
            for j in range(N)]


def _emit_ntt_fourstep(x, r, N, q, psi, factors, ops, consts, emitted):
    """Multi-factor negacyclic NTT/INTT as a flat sweep — generalizes the 2-factor four-step to ANY
    k factors with ∏factors = N (e.g. (32,32,32) for N=2^15, so every matmul stays f×f). Structure
    (adapted from ntt_matmul/planner.build_program, made negacyclic + invertible):
      forward:  y = psi^j·x ; then per factor f_t:  cur -> DFT_f(cur) -> ·twiddle_t -> transpose ->
                reshape to the next factor's (f_{t+1}, N/f_{t+1}).
      inverse:  the EXACT reverse of the forward op sequence (transpose ; ·twiddle_t^{-1} ; DFT_f^{-1},
                which folds in 1/f so ∏ = 1/N) ; then post-weight psi^{-j}.
    Building the INTT as the literal inverse guarantees intt(ntt(x))==x for any factorization (the
    eval-domain order is internal; the ct_mul only needs forward->pointwise->inverse). Verified below."""
    inv = (r.domain == "coeff")                       # intt: eval -> coeff
    omega = psi * psi % q                             # primitive N-th root (psi is a 2N-th root)
    k = len(factors)
    Ps, acc = [], 1                                   # Ps[t] = product of factors[:t]
    for f in factors:
        Ps.append(acc); acc *= f

    def const(name, arr, shape):
        v = Value(name, shape, q=q)
        if name not in emitted:
            consts[name] = ref.A(arr); emitted.add(name)
            ops.append(Op("tile.const", [], v, {"name": name}))
        return v
    ctr = [0]
    def op(oc, ins, shape, dom=None, at=None):
        ctr[0] += 1
        res = r if False else Value(f"{r.name}#s{ctr[0]}", tuple(shape), q=q, domain=dom)
        ops.append(Op(oc, ins, res, at or {})); return res
    def Wf(t, f, invert):                             # f×f DFT matrix (or its inverse) for factor t
        wf = pow(omega, N // f, q); wf = pow(wf, -1, q) if invert else wf
        sc = pow(f, -1, q) if invert else 1
        nm = f"W{'i' if invert else 'f'}_{q}_{N}_{t}"
        return const(nm, [[(sc * pow(wf, (a * b) % f, q)) % q for b in range(f)] for a in range(f)], (f, f))
    def Tw(t, f, invert):                             # twiddle T[i,col] = omega^{±P·i·(col//P)}
        nc, P = N // f, Ps[t]
        nm = f"T{'i' if invert else 'f'}_{q}_{N}_{t}"
        e = (lambda i, c: (-(P * i * (c // P))) % N) if invert else (lambda i, c: (P * i * (c // P)) % N)
        return const(nm, [[pow(omega, e(i, c), q) for c in range(nc)] for i in range(f)], (f, nc))

    if not inv:                                       # ---------- forward ----------
        psv = const(f"psi_{q}_{N}", [pow(psi, j, q) for j in range(N)], (N,))
        y = op("rns.mul", [x, psv], (N,))                                    # pre-weight psi^j
        f0 = factors[0]
        cur = op("tile.reshape", [y], (f0, N // f0), at={"shape": (f0, N // f0)})
        for t, f in enumerate(factors):
            nc = N // f
            a = op("tile.matmul_modq", [Wf(t, f, False), cur], (f, nc), at={"q": q})
            b = op("tile.mul_modq", [a, Tw(t, f, False)], (f, nc), at={"q": q})
            bt = op("tile.transpose", [b], (nc, f))
            if t < k - 1:
                nf = factors[t + 1]; cur = op("tile.reshape", [bt], (nf, N // nf), at={"shape": (nf, N // nf)})
            else:
                cur = bt
        ops.append(Op("tile.reshape", [cur], r, {"shape": (N,)}))
    else:                                             # ---------- inverse (exact reverse) ----------
        lf = factors[-1]
        cur = op("tile.reshape", [x], (N // lf, lf), at={"shape": (N // lf, lf)})
        for t in reversed(range(k)):
            f = factors[t]; nc = N // f
            b = op("tile.transpose", [cur], (f, nc))                         # undo transpose
            a = op("tile.mul_modq", [b, Tw(t, f, True)], (f, nc), at={"q": q})   # undo twiddle
            prev = op("tile.matmul_modq", [Wf(t, f, True), a], (f, nc), at={"q": q})  # undo DFT (folds 1/f)
            if t > 0:
                pf = factors[t - 1]; cur = op("tile.reshape", [prev], (N // pf, pf), at={"shape": (N // pf, pf)})
            else:
                y = op("tile.reshape", [prev], (N,), at={"shape": (N,)})
        psinv = pow(psi, -1, q)
        pv = const(f"psinv_{q}_{N}", [pow(psinv, j, q) for j in range(N)], (N,))
        ops.append(Op("rns.mul", [y, pv], r))                                # post-weight psi^{-j}


def _emit_base_convert_crns(op, ops, consts, emitted):
    """Lower one rns.base_convert to CRNS primitive tile ops (docs: reference.base_convert_via_crns,
    docs/ir/crns_bconv_lowering.md). The common CRNS form — the base extension is the ONLY matmul,
    the overflow correction is elementwise:
       s   = matmul_wide(F, RES)                      # wide f-accumulator (matmul)
       k   = s>>u ,  neg = (s>>u-1)&1                 # fixed-point slice (elementwise)
       xE_j = matmul_modq(E_.j, RES, d_j)             # base extension Σ_i r_i·E_ij (matmul)
       out_j = (xE_j + k·g_j − neg·(M mod d_j)) mod d_j   # k·g / neg·M corrections are ELEMENTWISE
    Keeping k·g elementwise means the (wide, multi-chunk) k never enters a matmul, so it is never
    int8-decomposed — the base-extension matmul stays 4-limb."""
    src = [o.q for o in op.operands]
    dst = [v.q for v in (op.result if isinstance(op.result, list) else [op.result])]
    outs = op.result if isinstance(op.result, list) else [op.result]
    signed = op.attrs.get("signed", False)
    N = op.operands[0].shape[0]
    ns = len(src)
    # u: fixed-point precision for the overflow k. Default = worst-case-safe (exact for ANY x in
    # [0,∏src)). A caller that knows its input is BFV-bounded (x ≪ ∏src, e.g. the rescaled ẽ) may pass
    # attrs["u"] = a much smaller value (~log2(ns·max(src))), which keeps the f-vector — and thus the
    # int8 matmul_wide — tiny at large limb counts (docs/ir §8; validated by the u-sweep).
    u = op.attrs.get("u") or ref.crns_safe_u(src)
    E, f, g = ref.crns_precompute(src, dst, u)
    M = 1
    for p in src:
        M *= p
    tag = outs[0].name

    def const(name, arr, shape, q=None):
        v = Value(name, shape, q=q)
        if name not in emitted:
            consts[name] = ref.A(arr); emitted.add(name)
            ops.append(Op("tile.const", [], v, {"name": name}))
        return v

    # stack the |src| residues into RES : (ns, N)   (reshape each (N,) -> (1,N), concat axis 0)
    rows = []
    for o in op.operands:
        rv = Value(f"{tag}#r_{o.name}", (1, N), q=None)
        ops.append(Op("tile.reshape", [o], rv, {"shape": (1, N)}))
        rows.append(rv)
    RES = Value(f"{tag}#RES", (ns, N), q=None)
    ops.append(Op("tile.concat", rows, RES, {"axis": 0}))

    # wide f-accumulator  s = F @ RES : (1,N) -> (N,).  f_i ~ 2^u (needs ceil((u+1)/7) i8 limbs at
    # radix 128), residues < max(src): size both operands' int8 decomposition from their ranges.
    F = const(f"crnsF_{tuple(src)}_{u}", [f], (1, ns))
    sw = Value(f"{tag}#s", (1, N), q=None)
    ops.append(Op("tile.matmul_wide", [F, RES], sw,
                  {"radix": 128, "nlimbs_a": _nlimbs_for(max(f)), "nlimbs_b": _nlimbs_for(max(src))}))

    # slice the wide accumulator: k = s>>u  and (signed) sign bit neg = (s>>u-1)&1. k is a RAW wide
    # value (q=None) — for large source-modulus counts it spans >32 bits, which is fine because it is
    # only used ELEMENTWISE below (never a matmul operand, so never int8-decomposed).
    k = Value(f"{tag}#k", (1, N), q=None)
    ops.append(Op("tile.fixed_slice", [sw], k, {"shift": u}))
    if signed:
        neg = Value(f"{tag}#neg", (1, N), q=None)
        ops.append(Op("tile.fixed_slice", [sw], neg, {"shift": u - 1, "mask": 1}))

    # per dst limb: out_j = (Σ_i r_i·E_ij + k·g_j − neg·(M mod d_j)) mod d_j.  The base extension
    # Σ_i r_i·E_ij is the ONLY matmul (residues × baked E, both < 2^28 → 4 int8 limbs); k·g_j and
    # −neg·(M mod d_j) are ELEMENTWISE (pointwise), keeping the wide k off the matrix engine.
    nlr = _nlimbs_for(max(src))
    for j, (d, ov) in enumerate(zip(dst, outs)):
        Ej = const(f"crnsE_{tuple(src)}_{d}", [[E[i][j] for i in range(ns)]], (1, ns))
        xE = Value(f"{tag}#xE_{d}", (1, N), q=d)
        ops.append(Op("tile.matmul_modq", [Ej, RES], xE,
                      {"q": d, "radix": 128, "nlimbs_a": _nlimbs_for(d), "nlimbs_b": nlr}))
        gj = const(f"crnsg_{tuple(src)}_{d}", [[g[j]]], (1, 1), q=d)
        kg = Value(f"{tag}#kg_{d}", (1, N), q=d)
        ops.append(Op("tile.mul_modq", [k, gj], kg, {"q": d}))         # elementwise k·g_j (k raw → mod d_j)
        acc = Value(f"{tag}#acc_{d}", (1, N), q=d)
        ops.append(Op("tile.add_modq", [xE, kg], acc, {"q": d}))
        if signed:
            Mj = const(f"crnsMmod_{tuple(src)}_{d}", [[M % d]], (1, 1), q=d)
            negM = Value(f"{tag}#negM_{d}", (1, N), q=d)
            ops.append(Op("tile.mul_modq", [neg, Mj], negM, {"q": d}))  # elementwise neg·(M mod d_j)
            acc2 = Value(f"{tag}#accs_{d}", (1, N), q=d)
            ops.append(Op("tile.sub_modq", [acc, negM], acc2, {"q": d}))
            acc = acc2
        ops.append(Op("tile.reshape", [acc], ov, {"shape": (N,)}))


def _nlimbs_for(maxv, radix=128):
    """Smallest nlimbs with radix^nlimbs > maxv — enough radix-`radix` limbs to hold maxv exactly.
    The CRNS f-vector (~2^u) and the k-augmented residue stack exceed one 28-bit limb, so the int8
    decomposition must size limbs from the actual value range, not the default 4 (docs/ir §5)."""
    nl, m = 1, max(1, int(maxv))
    while radix ** nl <= m:
        nl += 1
    return nl


def _emit_base_convert_approx(op, ops, consts, emitted):
    """Lower rns.base_convert(exact=False) to the APPROXIMATE FBC (reference.fast_base_convert):
       y_i = r_i·(M/s_i)^{-1} mod s_i  (elementwise pre-scale) ; out_j = Σ_i y_i·((M/s_i) mod d_j) mod d_j
       (base-extension matmul). No overflow correction — all values < 2^28 → standard 4-limb int8."""
    src = [o.q for o in op.operands]
    outs = op.result if isinstance(op.result, list) else [op.result]
    dst = [v.q for v in outs]
    N, ns = op.operands[0].shape[0], len(src)
    M = 1
    for p in src:
        M *= p
    tag = outs[0].name

    def const(name, arr, shape, q=None):
        v = Value(name, shape, q=q)
        if name not in emitted:
            consts[name] = ref.A(arr); emitted.add(name)
            ops.append(Op("tile.const", [], v, {"name": name}))
        return v

    # pre-scale  y_i = r_i · (M/s_i)^{-1} mod s_i   (elementwise, per src prime)
    yrows = []
    for i, o in enumerate(op.operands):
        ci = pow((M // src[i]) % src[i], -1, src[i])
        cc = const(f"fbcC_{tuple(src)}_{src[i]}", [[ci]], (1, 1), q=src[i])
        o1 = Value(f"{tag}#o_{o.name}", (1, N), q=src[i])
        ops.append(Op("tile.reshape", [o], o1, {"shape": (1, N)}))
        yi = Value(f"{tag}#y_{o.name}", (1, N), q=src[i])
        ops.append(Op("tile.mul_modq", [o1, cc], yi, {"q": src[i]}))
        yrows.append(yi)
    Y = Value(f"{tag}#Y", (ns, N), q=None)
    ops.append(Op("tile.concat", yrows, Y, {"axis": 0}))

    # base extension  out_j = matmul_modq(D_.j, Y, d_j),  D_ij = (M/s_i) mod d_j  (the only matmul)
    nlr = _nlimbs_for(max(src))
    for j, (d, ov) in enumerate(zip(dst, outs)):
        Dj = const(f"fbcD_{tuple(src)}_{d}", [[(M // src[i]) % d for i in range(ns)]], (1, ns))
        yj = Value(f"{tag}#fbc_{d}", (1, N), q=d)
        ops.append(Op("tile.matmul_modq", [Dj, Y], yj,
                      {"q": d, "radix": 128, "nlimbs_a": _nlimbs_for(d), "nlimbs_b": nlr}))
        ops.append(Op("tile.reshape", [yj], ov, {"shape": (N,)}))


def lower_l1_to_l2(prog: Program, factors=None, T=None) -> Program:
    """Stage 3. If `factors` given, lower each NTT via the four-step (parametrized by the
    factorization plan); else one full DFT matmul. If `T` given, tile every matmul to <=T."""
    consts = dict(prog.consts)
    ops, emitted = [], set()
    for op in prog.ops:
        oc = op.opcode
        if oc in ("rns.ntt", "rns.intt"):
            x, r = op.operands[0], op.result
            N, q, psi = x.shape[0], r.q, op.attrs["psi"]
            inv = (oc == "rns.intt")
            if factors:
                _emit_ntt_fourstep(x, r, N, q, psi, factors, ops, consts, emitted)
                continue
            wname = f"W_{'i' if inv else 'f'}_{q}"     # content-keyed: identical NTTs share one const
            wv = Value(wname, (N, N), q=q)
            if wname not in emitted:
                consts[wname] = ref.A(ntt_matrix(N, q, psi, inverse=inv))
                ops.append(Op("tile.const", [], wv, {"name": wname}))
                emitted.add(wname)
            ops.append(Op("tile.matmul_modq", [wv, x], r, {"q": q}))
        elif oc == "rns.mul":
            ops.append(Op("tile.mul_modq", op.operands, op.result, {"q": op.result.q}))
        elif oc == "rns.add":
            ops.append(Op("tile.add_modq", op.operands, op.result, {"q": op.result.q}))
        elif oc == "rns.sub":
            ops.append(Op("tile.sub_modq", op.operands, op.result, {"q": op.result.q}))
        elif oc == "rns.base_convert":
            if op.attrs.get("exact", True):
                _emit_base_convert_crns(op, ops, consts, emitted)
            else:
                _emit_base_convert_approx(op, ops, consts, emitted)
        else:
            ops.append(op)                    # macro / structural / already L2
    out = Program(prog.inputs, ops, prog.outputs, consts, dict(prog.meta))
    return tile_matmul(out, T) if T else out


def tile_matmul(prog: Program, T: int) -> Program:
    """Tile every tile.matmul_modq into <=T x T blocks (M/N/K), K-tiles accumulated with
    add_modq — makes the tile size `T` a real parameter, so a matmul with any dim > T is
    lowered to hardware-sized tiles. (Non-transposed 2D operands; the factored-NTT and
    fuse_a paths produce these.)"""
    ops, c = [], [0]
    def nm(b):
        c[0] += 1; return f"{b}#{c[0]}"
    def chunks(n):
        return [(i, min(T, n - i)) for i in range(0, n, T)]
    def slice2d(X, rs, rl, cs, cl, q):
        t1 = Value(nm("s"), (rl, X.shape[1]), q=q); ops.append(Op("tile.slice", [X], t1, {"axis": 0, "start": rs, "size": rl}))
        t2 = Value(nm("s"), (rl, cl), q=q);         ops.append(Op("tile.slice", [t1], t2, {"axis": 1, "start": cs, "size": cl}))
        return t2

    for op in prog.ops:
        if op.opcode != "tile.matmul_modq" or op.attrs.get("transpose_a") or op.attrs.get("transpose_b"):
            ops.append(op); continue
        A, B, R, q = op.operands[0], op.operands[1], op.result, op.result.q
        onedim = (len(B.shape) == 1)
        if onedim:                                   # (N,) RHS -> treat as (N,1)
            B2 = Value(nm("b2"), (B.shape[0], 1), q=q); ops.append(Op("tile.reshape", [B], B2, {"shape": (B.shape[0], 1)})); B = B2
        M, K = A.shape; Nf = B.shape[1]
        if M <= T and K <= T and Nf <= T and not onedim:
            ops.append(op); continue
        if M <= T and K <= T and Nf <= T and onedim:
            Y = Value(nm("y"), (M, 1), q=q); ops.append(Op("tile.matmul_modq", [A, B], Y, {"q": q}))
            ops.append(Op("tile.reshape", [Y], R, {"shape": R.shape})); continue
        rows = []
        for ms, ml in chunks(M):
            cols = []
            for ns, nl in chunks(Nf):
                parts = []
                for ks, kl in chunks(K):
                    at = slice2d(A, ms, ml, ks, kl, q)
                    bt = slice2d(B, ks, kl, ns, nl, q)
                    p = Value(nm("p"), (ml, nl), q=q); ops.append(Op("tile.matmul_modq", [at, bt], p, {"q": q})); parts.append(p)
                cell = parts[0] if len(parts) == 1 else Value(nm("acc"), (ml, nl), q=q)
                if len(parts) > 1: ops.append(Op("tile.add_modq", parts, cell, {"q": q}))
                cols.append(cell)
            row = cols[0] if len(cols) == 1 else Value(nm("row"), (ml, Nf), q=q)
            if len(cols) > 1: ops.append(Op("tile.concat", cols, row, {"axis": 1}))
            rows.append(row)
        full = rows[0] if len(rows) == 1 else Value(nm("full"), (M, Nf), q=q)
        if len(rows) > 1: ops.append(Op("tile.concat", rows, full, {"axis": 0}))
        ops.append(Op("tile.reshape", [full], R, {"shape": R.shape}))   # to original name/shape
    return Program(prog.inputs, ops, prog.outputs, dict(prog.consts), dict(prog.meta))


# Fusion A strategies — each SELECTS which shared-weight matmul_modq ops (const LHS + prime) it will
# batch. They share the same batching mechanism (temporal subsets); they only differ in eligibility,
# so you can enable/disable each independently to isolate a bug, e.g. fuse_a(prog, ("ntt",)).
FUSE_A_STRATEGIES = {
    "ntt":      lambda op: len(op.operands[1].shape) == 1,   # DFT-matrix NTT   W @ x   (1-D RHS)
    "baseconv": lambda op: len(op.operands[1].shape) == 2,   # base-extension   W @ X   (2-D RHS)
}
FUSE_A_ALL = ("ntt", "baseconv")


def fuse_a(prog: Program, strategies=FUSE_A_ALL) -> Program:
    """Stage 5, Fusion A — tile packing. Batch independent matmul_modq ops that share a const weight
    (LHS) + prime into ONE matmul over a concatenated RHS, filling a tile's columns instead of many
    under-full matmuls:  W@x0, W@x1, ... -> W @ [x0|x1|...] -> split.

    `strategies` (subset of FUSE_A_STRATEGIES) selects which classes of matmul get packed — toggle
    them independently to isolate a bug (e.g. strategies=("ntt",) keeps only the proven NTT packing
    and disables base-extension packing). The batching mechanism is shared:
      * concat/split along the last axis (so 1-D DFT-matrix NTT and 2-D base-extension both work);
      * each shared-weight group is partitioned into TEMPORALLY-COHERENT batches (placed just after
        all their RHS are ready, only if every member's result is consumed later), so an early cluster
        (the tensor's 4 NTTs) fuses even when a later op (a relin NTT) shares the same weight.
    Meaning-preserving. (Cross-stream / mixed-prime packing still deferred — FORMALIZATION §8.)"""
    from collections import OrderedDict, defaultdict
    sel = [FUSE_A_STRATEGIES[s] for s in strategies]
    ops_in = prog.ops
    const_names = {op.result.name for op in ops_in if op.opcode == "tile.const"}
    def_pos = {v.name: -1 for v in prog.inputs}
    for i, op in enumerate(ops_in):
        for o in (op.result if isinstance(op.result, list) else [op.result]):
            def_pos[o.name] = i
    min_use = {}                                          # value -> earliest op index that consumes it
    for i, op in enumerate(ops_in):
        for o in op.operands:
            min_use[o.name] = min(min_use.get(o.name, i), i)

    groups = OrderedDict()
    for i, op in enumerate(ops_in):
        if (op.opcode == "tile.matmul_modq" and op.operands[0].name in const_names
                and any(f(op) for f in sel)):             # eligible under an ENABLED strategy
            groups.setdefault((op.operands[0].name, op.result.q), []).append((i, op))

    INF = len(ops_in) + 1
    plans, handled = defaultdict(list), set()             # insertion index -> [(W, q, [member ops])]
    for (wname, q), members in groups.items():
        wpos = def_pos[wname]                             # the shared weight const must also be defined
        members.sort(key=lambda im: def_pos[im[1].operands[1].name])   # by RHS availability
        cur, pstar = [], wpos
        def close(cur, pstar):
            if len(cur) >= 2:
                plans[pstar + 1].append((cur[0].operands[0], q, cur))
                handled.update(m.result.name for m in cur)
        for _i, op in members:
            newp = max(pstar, wpos, def_pos[op.operands[1].name])
            # valid to extend the batch iff every member's result is still consumed strictly after the
            # (raised) insertion point — otherwise this member belongs to a later temporal batch.
            if cur and min_use.get(op.result.name, INF) > newp and \
                    all(min_use.get(m.result.name, INF) > newp for m in cur):
                cur.append(op); pstar = newp
            else:
                close(cur, pstar); cur, pstar = [op], max(wpos, def_pos[op.operands[1].name])
        close(cur, pstar)

    ctr = [0]
    def nm(b): ctr[0] += 1; return f"{b}#fa{ctr[0]}"

    def emit(out, W, q, grp):
        cols, seen, meta = [], {}, []
        for m in grp:                                     # normalize each RHS to 2-D, once per unique RHS
            r = m.operands[1]; oned = (len(r.shape) == 1)
            meta.append((m, r.shape))
            if r.name not in seen:
                if oned:
                    xc = Value(nm("col"), (r.shape[0], 1), q=r.q)
                    out.append(Op("tile.reshape", [r], xc, {"shape": (r.shape[0], 1)}))
                    seen[r.name] = xc
                else:
                    seen[r.name] = r
            cols.append(seen[r.name])
        g, K = len(cols), cols[0].shape[0]
        ncol = cols[0].shape[-1] if len(cols[0].shape) > 1 else 1
        Xb = Value(nm("rhs"), (K, g * ncol), q=q); out.append(Op("tile.concat", cols, Xb, {"axis": 1}))
        M = W.shape[0]
        Yb = Value(nm("out"), (M, g * ncol), q=q); out.append(Op("tile.matmul_modq", [W, Xb], Yb, {"q": q}))
        ycols = [Value(nm("yc"), (M, ncol), q=q) for _ in grp]
        out.append(Op("tile.split", [Yb], ycols, {"axis": 1, "parts": g}))
        for (m, shp), yc in zip(meta, ycols):
            out.append(Op("tile.reshape", [yc], m.result, {"shape": shp}))

    out = []
    for i, op in enumerate(ops_in):
        for (W, q, grp) in plans.get(i, []):
            emit(out, W, q, grp)
        if not (op.opcode == "tile.matmul_modq" and op.result.name in handled):
            out.append(op)                                # keep non-batched ops; drop folded members
    for (W, q, grp) in plans.get(len(ops_in), []):
        emit(out, W, q, grp)
    return Program(prog.inputs, out, prog.outputs, dict(prog.consts), dict(prog.meta))


def lower_l2_to_l3(prog: Program, radix=128, nlimbs=4) -> Program:
    """Stage 4: expand each tile.matmul_modq / tile.matmul_wide into the int8 (L3) realization —
       decompose both operands to radix-128 limbs (per-operand counts from attrs nlimbs_a/nlimbs_b,
       default `nlimbs`), na·nb int8 matmuls, group by p+r, recombine. The mod-q matmul reduces via
       recombine_modq; the wide matmul (CRNS f-accumulator) combines by place value with NO reduction
       (recombine_wide). Bit-identical to matmul_modq / matmul_wide (reference.*)."""
    ops, ctr = [], [0]
    def nm(b):
        ctr[0] += 1; return f"{b}_{ctr[0]}"
    for op in prog.ops:
        if op.opcode in ("tile.matmul_modq", "tile.matmul_wide"):
            wide = (op.opcode == "tile.matmul_wide")
            A, B, R = op.operands[0], op.operands[1], op.result
            q = None if wide else R.q
            rad = op.attrs.get("radix", radix)
            na = op.attrs.get("nlimbs_a", nlimbs)
            nb = op.attrs.get("nlimbs_b", nlimbs)
            aL = [Value(nm("a"), A.shape, dtype="i8") for _ in range(na)]
            bL = [Value(nm("b"), B.shape, dtype="i8") for _ in range(nb)]
            ops.append(Op("tile.decompose", [A], aL, {"radix": rad, "nlimbs": na}))
            ops.append(Op("tile.decompose", [B], bL, {"radix": rad, "nlimbs": nb}))
            if wide:                                              # place-value combine, no reduction
                planes, exps = [], []
                for p in range(na):
                    for r in range(nb):
                        pv = Value(nm("wP"), R.shape)
                        ops.append(Op("tile.matmul_wide", [aL[p], bL[r]], pv, {}))
                        planes.append(pv); exps.append(p + r)
                ops.append(Op("tile.recombine_wide", planes, R, {"radix": rad, "exps": exps}))
            else:                                                 # group planes by p+r, reduce mod q
                P = {}
                for p in range(na):
                    for r in range(nb):
                        pv = Value(nm("P"), R.shape, q=q)
                        ops.append(Op("tile.matmul_modq", [aL[p], bL[r]], pv, {"q": q}))
                        P[(p, r)] = pv
                C = []
                for s in range(na + nb - 1):
                    terms = [P[(p, r)] for p in range(na) for r in range(nb) if p + r == s]
                    cv = Value(nm("C"), R.shape, q=q)
                    ops.append(Op("tile.add_modq", terms, cv, {"q": q}))
                    C.append(cv)
                ops.append(Op("tile.recombine_modq", C, R, {"radix": rad, "q": q}))
        else:
            ops.append(op)
    return Program(prog.inputs, ops, prog.outputs, dict(prog.consts), dict(prog.meta))


# --------------------------------------------------------------------------- #
# self-test: lower the tensor program L1 -> L2 -> L3, check each stays == L1
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os, sys, random
    import _paths                                     # noqa: F401  (repo layout; see _paths.py)
    from bfv_mul_ref import MulParams, encrypt              # noqa: E402
    from bfv_relin_ref import small_poly                    # noqa: E402
    import ir                                               # noqa: E402
    from builders import build_tensor                       # noqa: E402

    pp = MulParams(N=16)
    rng = random.Random(7)
    s = small_poly(pp.N, rng)
    ct1 = encrypt(pp, [rng.randrange(pp.t) for _ in range(pp.N)], s, rng)
    ct2 = encrypt(pp, [rng.randrange(pp.t) for _ in range(pp.N)], s, rng)

    l1, inputs, outs = build_tensor(pp, ct1, ct2)
    l2 = lower_l1_to_l2(l1)          # stage 3
    l2f = fuse_a(l2)                 # stage 5 (Fusion A) — applied at L2, before int8
    l3 = lower_l2_to_l3(l2)          # stage 4 (no packing)
    l3f = lower_l2_to_l3(l2f)        # stage 4 after packing

    def mm(p): return sum(1 for op in p.ops if op.opcode == "tile.matmul_modq")
    R = {k: ir.run(p, inputs) for k, p in [("l1", l1), ("l2", l2), ("l2f", l2f), ("l3", l3), ("l3f", l3f)]}
    same = lambda k: all(ref._eq(R["l1"][o], R[k][o]) for o in R["l1"])

    print(f"lowering + Fusion A on the tensor program (N={pp.N}), meaning-preserving throughout:")
    print(f"  L1 (poly)         {len(l1.ops):3d} ops")
    print(f"  L2 (tile)         {len(l2.ops):3d} ops   {mm(l2):3d} matmuls           run==L1 : {'OK' if same('l2') else 'FAIL'}")
    print(f"  L2 + Fusion A     {len(l2f.ops):3d} ops   {mm(l2f):3d} matmuls (batched)  run==L1 : {'OK' if same('l2f') else 'FAIL'}")
    print(f"  L3 (int8)         {len(l3.ops):3d} ops   {mm(l3):3d} int8 matmuls      run==L1 : {'OK' if same('l3') else 'FAIL'}")
    print(f"  L3 after Fusion A {len(l3f.ops):3d} ops   {mm(l3f):3d} int8 matmuls      run==L1 : {'OK' if same('l3f') else 'FAIL'}")
    ok = all(same(k) for k in R)
    print(f"\n  Fusion A: {mm(l2)} -> {mm(l2f)} matmuls at L2  =>  {mm(l3)} -> {mm(l3f)} int8 matmuls at L3")
    print("\nLOWER OK (stages 3 + 4 + 5 validated)" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)
