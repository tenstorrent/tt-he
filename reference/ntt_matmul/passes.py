# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ntt_matmul passes (see ntt_matmul/PLAN.md).

These run AFTER the standard ntt optimizations (optimize_fold + tile_dataflow +
fuse_all_to_all from ntt_passes.py), i.e. on the transpose-free, M/N-tiled,
all_to_all-cross-stage form. They add the remaining two pieces:

  tile_k(prog, T)            split any matmul_modq with K > T into K-tiles + add_modq
  lower_matmul_int8(prog)    rewrite each matmul_modq into the int8 decomposition:
                             decompose (dynamic operand) + i8 matmul_modq + add_modq +
                             recombine_modq, with the CONSTANT operand's folded+decomposed
                             limbs baked offline (never computed at runtime).

Both are transpose-aware: transpose is linear, so it distributes over the radix
decomposition and the per-limb matmuls simply carry the original transpose_a/transpose_b
flags. The numpy interpreter in ir.py stays the bit-exact golden.
"""
from __future__ import annotations

import numpy as np

from ir import OpCode, Op, Value, Program, outputs, verify


def _producers(prog: Program) -> dict:
    return {v.name: op for op in prog.ops for v in outputs(op)}


def _uses(prog: Program) -> dict:
    """value name -> list of consumer op indices (for finding the twiddle that follows a matmul)."""
    u = {}
    for i, op in enumerate(prog.ops):
        for v in op.operands:
            u.setdefault(v.name, []).append(i)
    return u


def _const_array(prog, producer, name):
    """If `name` is produced by a CONST op, return (baked array, const-name), else (None, None)."""
    op = producer.get(name)
    if op is not None and op.opcode is OpCode.CONST:
        return np.asarray(prog.consts[op.attrs["name"]], dtype=object), op.attrs["name"]
    return None, None


def _largest_divisor_leq(dim: int, T: int) -> int:
    for d in range(min(T, dim), 0, -1):
        if dim % d == 0:
            return d
    return 1


def _slice_axis(arr, axis, lo, size):
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(lo, lo + size)
    return arr[tuple(sl)]


# --------------------------------------------------------------------------- #
# K-tiling:  contract over K in tiles of <= T, accumulate with add_modq.
# Handles transpose flags: the contraction axis of a transposed operand is the
# other stored axis.
# --------------------------------------------------------------------------- #
def tile_k(prog: Program, T: int) -> Program:
    prod = _producers(prog)
    q = prog.q
    consts = dict(prog.consts)
    new_ops: list = []
    ctr = [0]

    def fresh(shape, dtype="i32"):
        v = Value(f"%k{ctr[0]}", tuple(shape), dtype)
        ctr[0] += 1
        return v

    for op in prog.ops:
        if op.opcode is not OpCode.MATMUL_MODQ:
            new_ops.append(op)
            continue
        A, B = op.operands
        res = op.result
        ta = op.attrs.get("transpose_a", False)
        tb = op.attrs.get("transpose_b", False)
        a_axis = 0 if ta else 1            # contraction axis inside stored A
        b_axis = 1 if tb else 0            # contraction axis inside stored B
        K = A.shape[a_axis]
        if K <= T:
            new_ops.append(op)
            continue

        c = _largest_divisor_leq(K, T)
        parts = K // c
        Amat, Aname = _const_array(prog, prod, A.name)
        Bmat, Bname = _const_array(prog, prod, B.name)

        def cut(val, mat, name, axis, lo):
            shp = list(val.shape); shp[axis] = c
            if mat is not None:
                nm = f"{name}__k{lo // c}"
                consts[nm] = _slice_axis(mat, axis, lo, c)
                v = fresh(shp); new_ops.append(Op(OpCode.CONST, [], v, {"name": nm, "shape": tuple(shp)}))
            else:
                v = fresh(shp); new_ops.append(Op(OpCode.SLICE, [val], v, {"axis": axis, "start": lo, "size": c}))
            return v

        partials = []
        for kt in range(parts):
            av = cut(A, Amat, Aname, a_axis, kt * c)
            bv = cut(B, Bmat, Bname, b_axis, kt * c)
            pv = fresh(res.shape)
            new_ops.append(Op(OpCode.MATMUL_MODQ, [av, bv], pv,
                              {"q": q, "transpose_a": ta, "transpose_b": tb}))
            partials.append(pv)
        new_ops.append(Op(OpCode.ADD_MODQ, partials, Value(res.name, res.shape), {"q": q}))

    out = Program(q, prog.inputs, new_ops, prog.output, consts, prog.meta)
    verify(out)
    return out


# --------------------------------------------------------------------------- #
# Full M/N/K matmul tiling: block every matmul into <= T tiles in ALL THREE dims
# (no new NTT stage). K-tiles accumulate (add_modq); M/N-tiles reassemble (concat).
# Transpose-aware. Subsumes tile_k.
# --------------------------------------------------------------------------- #
def tile_matmul(prog: Program, T: int, prefix: str = "%m") -> Program:
    prod = _producers(prog)
    uses = _uses(prog)
    cdef = {o.result.name: o for o in prog.ops
            if not isinstance(o.result, list) and o.opcode is OpCode.CONST}
    q = prog.q
    consts = dict(prog.consts)
    new_ops: list = []
    ctr = [0]
    slice_cache: dict = {}                       # (name, cuts) -> Value  (dedup data slices)
    cur_prov = [None]                            # provenance: which source op these new ops expand

    def emit(op):
        if cur_prov[0] is not None:
            op.attrs.setdefault("prov", cur_prov[0])
        new_ops.append(op)

    def fresh(shape, dtype="i32"):
        v = Value(f"{prefix}{ctr[0]}", tuple(shape), dtype)
        ctr[0] += 1
        return v

    def block(val, mat, name, cuts):
        """Slice operand `val` by `cuts`=[(axis,lo,size),...] (skip no-op cuts). Constant
        operands -> baked CONST (parameter selection, free); dynamic operands -> SLICE,
        deduplicated so each data tile is sliced exactly once."""
        cuts = [(ax, lo, sz) for (ax, lo, sz) in cuts if sz != val.shape[ax]]
        if not cuts:
            return val
        if mat is not None:
            arr = mat
            for ax, lo, sz in cuts:
                arr = _slice_axis(arr, ax, lo, sz)
            nm = f"{name}__" + "_".join(f"{ax}.{lo}" for ax, lo, sz in cuts)
            consts[nm] = arr
            v = fresh(arr.shape, "i8" if val.dtype == "i8" else "i32")
            emit(Op(OpCode.CONST, [], v, {"name": nm, "shape": tuple(arr.shape)}))
            return v
        key = (val.name, tuple(cuts))
        if key in slice_cache:
            return slice_cache[key]
        cur = val
        for ax, lo, sz in cuts:
            shp = list(cur.shape); shp[ax] = sz
            nv = fresh(shp)
            emit(Op(OpCode.SLICE, [cur], nv, {"axis": ax, "start": lo, "size": sz}))
            cur = nv
        slice_cache[key] = cur
        return cur

    skip = set()                                 # twiddle mul_modq ops folded per-tile
    for idx, op in enumerate(prog.ops):
        if idx in skip:
            continue
        if op.opcode is not OpCode.MATMUL_MODQ:
            new_ops.append(op)
            continue
        A, B = op.operands
        res = op.result
        ta = op.attrs.get("transpose_a", False)
        tb = op.attrs.get("transpose_b", False)
        Ma, Ka = (1, 0) if ta else (0, 1)
        Kb, Nb = (1, 0) if tb else (0, 1)
        M, K, N = A.shape[Ma], A.shape[Ka], B.shape[Nb]
        cM, cN, cK = (_largest_divisor_leq(d, T) for d in (M, N, K))
        pM, pN, pK = M // cM, N // cN, K // cK
        if pM == pN == pK == 1:
            new_ops.append(op)
            continue

        # does this matmul feed a single elementwise twiddle (mul_modq by a constant)?
        tw = twmat = None
        us = uses.get(res.name, [])
        if len(us) == 1:
            cand = prog.ops[us[0]]
            if (cand.opcode is OpCode.MUL_MODQ and cand.operands[0].name == res.name
                    and cand.operands[1].name in cdef):
                tw, twname = cand, cand.operands[1].name
                twmat = np.asarray(consts[cdef[twname].attrs["name"]], dtype=object)
                tw_const = cdef[twname].attrs["name"]
                skip.add(us[0])

        Amat, Aname = _const_array(prog, prod, A.name)
        Bmat, Bname = _const_array(prog, prod, B.name)
        mm = {"q": q, "transpose_a": ta, "transpose_b": tb}
        mm_prov = res.name                                # matmul block = the matmul's result
        out_prov = tw.result.name if tw is not None else res.name   # twiddle/reassembly block

        mrows = []
        for mi in range(pM):
            ncols = []
            for ni in range(pN):
                ks = []
                cur_prov[0] = mm_prov
                for ki in range(pK):
                    a_blk = block(A, Amat, Aname, [(Ma, mi * cM, cM), (Ka, ki * cK, cK)])
                    b_blk = block(B, Bmat, Bname, [(Kb, ki * cK, cK), (Nb, ni * cN, cN)])
                    pv = fresh((cM, cN))
                    emit(Op(OpCode.MATMUL_MODQ, [a_blk, b_blk], pv, dict(mm)))
                    ks.append(pv)
                cv = ks[0] if pK == 1 else None
                if pK > 1:
                    cv = fresh((cM, cN))
                    emit(Op(OpCode.ADD_MODQ, ks, cv, {"q": q}))                # K-accumulate
                if tw is not None:                                            # PER-TILE twiddle
                    cur_prov[0] = out_prov
                    blk = _slice_axis(_slice_axis(twmat, 0, mi * cM, cM), 1, ni * cN, cN)
                    nm = f"{tw_const}__{mi}.{ni}"
                    consts[nm] = blk
                    tvv = fresh((cM, cN))
                    emit(Op(OpCode.CONST, [], tvv, {"name": nm, "shape": (cM, cN)}))
                    tw_out = fresh((cM, cN))
                    emit(Op(OpCode.MUL_MODQ, [cv, tvv], tw_out, {"q": q}))
                    cv = tw_out
                ncols.append(cv)
            cur_prov[0] = out_prov
            if pN > 1:
                rv = fresh((cM, N))
                emit(Op(OpCode.CONCAT, ncols, rv, {"axis": 1}))               # stage-boundary reassembly
                mrows.append(rv)
            else:
                mrows.append(ncols[0])
        cur_prov[0] = out_prov
        target = tw.result if tw is not None else res
        if pM > 1:
            emit(Op(OpCode.CONCAT, mrows, Value(target.name, target.shape), {"axis": 0}))
        else:
            new_ops[-1].result = Value(target.name, new_ops[-1].result.shape)
        cur_prov[0] = None

    out = Program(q, prog.inputs, new_ops, prog.output, consts, prog.meta)
    verify(out)
    return out


# --------------------------------------------------------------------------- #
# Tile-granular corner-turn: collapse the cross-stage concat/slice/all_to_all
# layout ops into direct tile routing. In fold mode the corner-turn is a pure
# regrouping (each consumer T-tile == exactly one producer T-tile, unchanged), so
# we tag-trace element provenance, rewire each consumer tile to its producer tile,
# and dead-code-eliminate the layout ops. The only split/concat left are the
# input tiling (begin) and output reassembly (end).
# --------------------------------------------------------------------------- #
_COMPUTE = {OpCode.MATMUL_MODQ, OpCode.MUL_MODQ, OpCode.ADD_MODQ,
            OpCode.DECOMPOSE, OpCode.RECOMBINE_MODQ}
_LAYOUT = {OpCode.RESHAPE, OpCode.TRANSPOSE, OpCode.SLICE, OpCode.SPLIT,
           OpCode.CONCAT, OpCode.ALL_TO_ALL, OpCode.ALL_TO_ALL_TILES}


def _trace_tags(prog: Program, T: int) -> dict:
    """Element-provenance tags: layout ops permute tags; compute ops mint fresh ids."""
    import numpy as _np
    tags = {}
    N = int(_np.prod(prog.inputs[0].shape))
    nxt = [N]

    def fresh(shape):
        n = int(_np.prod(shape))
        a = _np.arange(nxt[0], nxt[0] + n, dtype=_np.int64).reshape(shape)
        nxt[0] += n
        return a

    for v in prog.inputs:
        tags[v.name] = _np.arange(int(_np.prod(v.shape)), dtype=_np.int64).reshape(v.shape)
    for op in prog.ops:
        oc = op.opcode
        if oc is OpCode.CONST:
            tags[op.result.name] = _np.full(tuple(op.attrs["shape"]), -1, dtype=_np.int64)
        elif oc is OpCode.RESHAPE:
            tags[op.result.name] = tags[op.operands[0].name].reshape(tuple(op.attrs["shape"]))
        elif oc is OpCode.TRANSPOSE:
            tags[op.result.name] = _np.ascontiguousarray(tags[op.operands[0].name].T)
        elif oc is OpCode.SLICE:
            s = tags[op.operands[0].name]
            idx = [slice(None)] * s.ndim
            idx[op.attrs["axis"]] = slice(op.attrs["start"], op.attrs["start"] + op.attrs["size"])
            tags[op.result.name] = _np.ascontiguousarray(s[tuple(idx)])
        elif oc is OpCode.SPLIT:
            for v, c in zip(outputs(op), _np.split(tags[op.operands[0].name], op.attrs["parts"], op.attrs["axis"])):
                tags[v.name] = _np.ascontiguousarray(c)
        elif oc is OpCode.CONCAT:
            tags[op.result.name] = _np.concatenate([tags[o.name] for o in op.operands], axis=op.attrs["axis"])
        elif oc is OpCode.ALL_TO_ALL_TILES:
            tr = op.attrs.get("transpose", [False] * len(op.attrs["perm"]))
            for v, p, t in zip(outputs(op), op.attrs["perm"], tr):
                s = tags[op.operands[p].name]
                tags[v.name] = _np.ascontiguousarray(s.T) if t else s
        elif oc is OpCode.ALL_TO_ALL:
            full = _np.concatenate([tags[o.name] for o in op.operands], axis=op.attrs["concat_axis"])
            r = full.reshape(tuple(op.attrs["reshape_to"]))
            for v, c in zip(outputs(op), _np.split(r, op.attrs["parts"], op.attrs["split_axis"])):
                tags[v.name] = _np.ascontiguousarray(c)
        elif oc in _COMPUTE:
            for v in outputs(op):
                tags[v.name] = fresh(v.shape)
        else:
            raise ValueError(f"trace: unknown opcode {oc}")
    return tags


def tilize_corner_turns(prog: Program, T: int) -> Program:
    """Replace each mid-computation layout region (the cross-stage corner-turn) with a single
    explicit tile-granular `all_to_all` (a tile permutation). Begin (input->tiles) and end
    (tiles->output) layout regions are left as-is. Communication stays visible as one op."""
    tags = _trace_tags(prog, T)
    n_ops = len(prog.ops)
    in_name = prog.inputs[0].name
    out_name = prog.output.name

    # global consumers (op index) per value
    uses: dict = {}
    for i, op in enumerate(prog.ops):
        for v in op.operands:
            uses.setdefault(v.name, []).append(i)
    defop = {v.name: op for op in prog.ops for v in outputs(op)}
    def_idx = {v.name: i for i, op in enumerate(prog.ops) for v in outputs(op)}

    # union-find over LAYOUT op indices that share any value -> layout "regions"
    parent = {i: i for i, op in enumerate(prog.ops) if op.opcode in _LAYOUT}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    val_layout: dict = {}
    for i in parent:
        op = prog.ops[i]
        for v in list(op.operands) + outputs(op):
            val_layout.setdefault(v.name, []).append(i)
    for idxs in val_layout.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    regions: dict = {}
    for i in parent:
        regions.setdefault(find(i), []).append(i)

    ctr = [0]

    def fresh_name():
        ctr[0] += 1
        return f"%a2a{ctr[0]}"

    insert_after: dict = {}  # op index -> [new all_to_all Op] (emit right after this op)
    drop: set = set()        # op indices removed (region members folded into the all_to_all)

    for region in regions.values():
        members = set(region)
        produced = {v.name for i in region for v in outputs(prog.ops[i])}
        # in_tiles: operands entering the region from outside (skip consts/params)
        in_tiles, seen = [], set()
        for i in region:
            for v in prog.ops[i].operands:
                if v.name not in produced and v.name not in seen:
                    d = defop.get(v.name)
                    if d is not None and d.opcode is OpCode.CONST:
                        continue
                    in_tiles.append(v); seen.add(v.name)
        # out_tiles: region-produced values consumed by a NON-region op (i.e. by compute)
        out_tiles = []
        for i in region:
            for v in outputs(prog.ops[i]):
                if any(c not in members for c in uses.get(v.name, [])):
                    out_tiles.append(v)
        touches_input = any(v.name == in_name for v in in_tiles)
        produces_output = out_name in produced
        if touches_input or produces_output or not in_tiles or not out_tiles:
            continue                                  # begin / end region: keep as-is
        # only a genuine tile-granular corner-turn: every in/out value must be a <=T tile
        if any(max(v.shape) > T for v in in_tiles + out_tiles):
            continue

        # try to express the region as a pure tile permutation (+ optional per-tile transpose)
        idx = {np.ascontiguousarray(tags[v.name]).tobytes(): k for k, v in enumerate(in_tiles)}
        idxT = {np.ascontiguousarray(tags[v.name].T).tobytes(): k for k, v in enumerate(in_tiles)}
        perm, trans, clean = [], [], True
        for v in out_tiles:
            key = np.ascontiguousarray(tags[v.name]).tobytes()
            if key in idx:
                perm.append(idx[key]); trans.append(False)
            elif key in idxT:
                perm.append(idxT[key]); trans.append(True)
            else:
                clean = False; break
        if not clean:
            continue                                  # not a clean tile permutation: keep region

        # place the all_to_all right after the last input-tile producer (all inputs defined,
        # before any output-tile consumer, which is the next stage's compute)
        pos = max(def_idx[v.name] for v in in_tiles)
        first_use = min((i for v in out_tiles for i in uses.get(v.name, [])), default=n_ops)
        if pos >= first_use:
            continue                                  # would violate ordering; keep region
        attrs = {"perm": perm}
        if any(trans):
            attrs["transpose"] = trans
        # provenance: if this region refines an existing all_to_all (re-tilizing at a finer tile),
        # link the new one to it so a viewer can show the coarse->fine corner turn.
        for i in region:
            if prog.ops[i].opcode in (OpCode.ALL_TO_ALL, OpCode.ALL_TO_ALL_TILES):
                attrs["prov"] = outputs(prog.ops[i])[0].name
                break
        insert_after.setdefault(pos, []).append(
            Op(OpCode.ALL_TO_ALL_TILES, list(in_tiles), list(out_tiles), attrs))
        drop.update(region)

    if not drop:
        verify(prog)
        return prog

    new_ops = []
    for i, op in enumerate(prog.ops):
        if i not in drop:
            new_ops.append(op)
        for a2a in insert_after.get(i, []):
            new_ops.append(a2a)

    out = Program(prog.q, prog.inputs, new_ops, prog.output, dict(prog.consts), prog.meta)
    verify(out)
    return out


# --------------------------------------------------------------------------- #
# int8 precision lowering of one modular matmul (transpose-aware).
# --------------------------------------------------------------------------- #
def lower_matmul_int8(prog: Program, radix: int = 128, nlimbs: int = 4) -> Program:
    """Replace each matmul_modq by its radix-`radix`, `nlimbs`-limb int8 expansion.
    The constant (parameter) operand is folded + decomposed at PLAN TIME and baked as
    CONSTs; only the dynamic operand gets a runtime `decompose`. transpose_a/transpose_b
    flags are carried onto the per-limb i8 matmuls (transpose is linear in the limbs).
    """
    prod = _producers(prog)
    q = prog.q
    consts = dict(prog.consts)
    new_ops: list = []
    ctr = [0]
    cur_prov = [None]                            # provenance: the matmul being int8-expanded

    def emit(op):
        if cur_prov[0] is not None:
            op.attrs.setdefault("prov", cur_prov[0])
        new_ops.append(op)

    def fresh(shape, dtype="i32"):
        v = Value(f"%d{ctr[0]}", tuple(shape), dtype)
        ctr[0] += 1
        return v

    def bake(name, arr):
        consts[name] = np.asarray(arr, dtype=object)
        v = fresh(arr.shape, "i8")
        emit(Op(OpCode.CONST, [], v, {"name": name, "shape": tuple(arr.shape)}))
        return v

    for op in prog.ops:
        if op.opcode is not OpCode.MATMUL_MODQ:
            new_ops.append(op)
            continue
        A, B = op.operands
        res = op.result
        ta = op.attrs.get("transpose_a", False)
        tb = op.attrs.get("transpose_b", False)
        mm = {"q": q, "transpose_a": ta, "transpose_b": tb}
        Amat, Aname = _const_array(prog, prod, A.name)
        Bmat, Bname = _const_array(prog, prod, B.name)
        cur_prov[0] = res.name                   # all int8-expansion ops belong to this matmul

        if Amat is not None and Bmat is not None:
            # both constant -> fold the whole matmul to one CONST offline
            a_ = Amat.T if ta else Amat
            b_ = Bmat.T if tb else Bmat
            nm = f"{res.name}__const"
            consts[nm] = (a_ @ b_) % q
            new_ops.append(Op(OpCode.CONST, [], Value(res.name, res.shape),
                              {"name": nm, "shape": tuple(res.shape)}))
            cur_prov[0] = None
            continue
        if Amat is None and Bmat is None:
            raise NotImplementedError("lower_matmul_int8 needs >=1 constant operand (the NTT parameter)")

        if Bmat is not None:
            # fold B (parameter): b_{p,r} = limb_r(radix^p * B_stored mod q); plain-decompose A
            a_limbs = [fresh(A.shape, "i8") for _ in range(nlimbs)]
            emit(Op(OpCode.DECOMPOSE, [A], a_limbs, {"radix": radix, "nlimbs": nlimbs}))
            C = []
            for r in range(nlimbs):
                prods = []
                for p in range(nlimbs):
                    Vp = (Bmat * (radix ** p)) % q
                    b_pr = bake(f"{Bname}__i8_p{p}_r{r}", (Vp // (radix ** r)) % radix)
                    pv = fresh(res.shape)
                    emit(Op(OpCode.MATMUL_MODQ, [a_limbs[p], b_pr], pv, dict(mm)))
                    prods.append(pv)
                cv = fresh(res.shape)
                emit(Op(OpCode.ADD_MODQ, prods, cv, {"q": q}))               # C_r = Σ_p a_p @ b_{p,r}
                C.append(cv)
            emit(Op(OpCode.RECOMBINE_MODQ, C, Value(res.name, res.shape),
                    {"radix": radix, "q": q}))                               # Σ_r radix^r·C_r mod q
        else:
            # fold A (parameter): a_{r,p} = limb_p(radix^r * A_stored mod q); plain-decompose B
            b_limbs = [fresh(B.shape, "i8") for _ in range(nlimbs)]
            emit(Op(OpCode.DECOMPOSE, [B], b_limbs, {"radix": radix, "nlimbs": nlimbs}))
            C = []
            for p in range(nlimbs):
                prods = []
                for r in range(nlimbs):
                    Vr = (Amat * (radix ** r)) % q
                    a_rp = bake(f"{Aname}__i8_r{r}_p{p}", (Vr // (radix ** p)) % radix)
                    pv = fresh(res.shape)
                    emit(Op(OpCode.MATMUL_MODQ, [a_rp, b_limbs[r]], pv, dict(mm)))
                    prods.append(pv)
                cv = fresh(res.shape)
                emit(Op(OpCode.ADD_MODQ, prods, cv, {"q": q}))               # C_p = Σ_r a_{r,p} @ b_r
                C.append(cv)
            emit(Op(OpCode.RECOMBINE_MODQ, C, Value(res.name, res.shape),
                    {"radix": radix, "q": q}))                               # Σ_p radix^p·C_p mod q
        cur_prov[0] = None

    out = Program(q, prog.inputs, new_ops, prog.output, consts, prog.meta)
    verify(out)
    return out
