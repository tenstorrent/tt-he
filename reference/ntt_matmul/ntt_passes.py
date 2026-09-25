# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""IR -> IR optimization passes (PLAN.md §6, §7).

Both passes are pure rewrites that preserve the program's output exactly (they apply
algebraic identities), so the numpy golden reference validates them for free. The op
set never grows — passes only toggle attributes, fold constants, and rewire operands.

  eliminate_transposes  (§6): fold a TRANSPOSE into a consuming MATMUL_MODQ via
        transpose_a/transpose_b when they are separated only by identity reshapes.
        Removes the 2-factor inter-stage corner-turns; the ≥3-factor residual
        (transpose feeding a *non-identity* reshape) correctly stays — matching the
        "a 2D matmul exposes only 2 of ≥3 axes" limit.

  fuse_diagonal_twiddles (§7): fold a row-diagonal MUL_MODQ constant into the
        weight of the MATMUL_MODQ that produced its other operand. Only diagonal
        twiddles fuse; the four-step inter-stage twiddle is NOT diagonal and stays.
        (In the flat sweep this removes the trivial all-ones last-stage twiddle; the
        general win is the negacyclic ψ pre/post weights added later.)
"""
from __future__ import annotations

from ir import OpCode, Op, Value, Program, verify


def _clone(prog: Program) -> Program:
    ops = [Op(o.opcode, list(o.operands), o.result, dict(o.attrs)) for o in prog.ops]
    return Program(prog.q, list(prog.inputs), ops, prog.output,
                   dict(prog.consts), dict(prog.meta))


def _use_index(prog: Program):
    """name -> list of (op_index | 'out', operand_index)."""
    uses: dict = {}
    for i, o in enumerate(prog.ops):
        for j, v in enumerate(o.operands):
            uses.setdefault(v.name, []).append((i, j))
    uses.setdefault(prog.output.name, []).append(("out", 0))
    return uses


def _def_index(prog: Program):
    return {o.result.name: i for i, o in enumerate(prog.ops)}


def _rewire(prog: Program, old_name: str, new_value):
    """Replace every use of `old_name` with `new_value` (a Value)."""
    for o in prog.ops:
        o.operands = [new_value if v.name == old_name else v for v in o.operands]
    if prog.output.name == old_name:
        prog.output = new_value


# --------------------------------------------------------------------------- #
def eliminate_transposes(prog: Program) -> Program:
    prog = _clone(prog)
    changed = True
    while changed:
        changed = False
        uses = _use_index(prog)
        for i, op in enumerate(prog.ops):
            if op.opcode is not OpCode.TRANSPOSE:
                continue
            src = op.operands[0]
            chain = [i]
            cur = op.result
            # walk through identity reshapes (in.shape == out.shape)
            while True:
                u = uses.get(cur.name, [])
                if len(u) != 1 or u[0][0] == "out":
                    break
                nxt = prog.ops[u[0][0]]
                if (nxt.opcode is OpCode.RESHAPE
                        and tuple(nxt.operands[0].shape) == tuple(nxt.result.shape)):
                    chain.append(u[0][0])
                    cur = nxt.result
                    continue
                break
            u = uses.get(cur.name, [])
            if len(u) != 1 or u[0][0] == "out":
                continue
            ci, cj = u[0]
            M = prog.ops[ci]
            if M.opcode is not OpCode.MATMUL_MODQ:
                continue
            flag = "transpose_a" if cj == 0 else "transpose_b"
            M.attrs[flag] = not M.attrs.get(flag, False)
            M.operands[cj] = src                       # skip transpose + identity reshapes
            for idx in sorted(chain, reverse=True):
                del prog.ops[idx]
            changed = True
            break
    verify(prog)
    return prog


# --------------------------------------------------------------------------- #
def _is_row_diagonal(arr):
    return arr.ndim == 2 and all(len(set(arr[r, :].tolist())) == 1 for r in range(arr.shape[0]))


def fuse_diagonal_twiddles(prog: Program) -> Program:
    prog = _clone(prog)
    changed = True
    while changed:
        changed = False
        uses = _use_index(prog)
        defs = _def_index(prog)
        for i, op in enumerate(prog.ops):
            if op.opcode is not OpCode.MUL_MODQ:
                continue
            a, b = op.operands
            if a.name not in defs or b.name not in defs:
                continue
            aop, bop = prog.ops[defs[a.name]], prog.ops[defs[b.name]]
            if not (aop.opcode is OpCode.MATMUL_MODQ and bop.opcode is OpCode.CONST):
                continue
            if len(uses.get(a.name, [])) != 1:           # matmul result used only here
                continue
            tw = prog.consts[bop.attrs["name"]]
            if not _is_row_diagonal(tw):
                continue
            wop = prog.ops[defs[aop.operands[0].name]]
            if wop.opcode is not OpCode.CONST:
                continue
            # W'[r,:] = t_r * W[r,:]
            W = prog.consts[wop.attrs["name"]].copy()
            for r in range(W.shape[0]):
                W[r, :] = (W[r, :] * tw[r, 0]) % prog.q
            new_name = wop.attrs["name"] + "_f"
            prog.consts[new_name] = W
            wop.attrs = {**wop.attrs, "name": new_name}
            # drop the mul: route its consumers to the matmul result
            _rewire(prog, op.result.name, a)
            del prog.ops[i]
            # drop the now-dead twiddle const if unused
            if not _use_index(prog).get(b.name):
                del prog.ops[_def_index(prog)[b.name]]
            changed = True
            break
    verify(prog)
    return prog


# --------------------------------------------------------------------------- #
def _transpose_const(prog: Program, const_op: Op):
    """Pre-transpose a CONST's baked data (free at plan time); update its op."""
    old = const_op.attrs["name"]
    new = old + "_T"
    if new not in prog.consts:
        prog.consts[new] = prog.consts[old].T.copy()
    sh = tuple(reversed(const_op.result.shape))
    const_op.attrs = {**const_op.attrs, "name": new, "shape": sh}
    const_op.result = Value(const_op.result.name, sh)


def fold_result_transposes(prog: Program) -> Program:
    """Merge a TRANSPOSE of a matmul result into the matmul by reordering operands
    and pre-transposing the constants (PLAN §6, the transpose(A·B)=Bᵀ·Aᵀ identity):

        transpose(matmul(W, M) ⊙ T)  ->  matmul(M, Wᵀ, transpose_a) ⊙ Tᵀ
        transpose(matmul(W, M))      ->  matmul(M, Wᵀ, transpose_a)

    Removes the explicit transpose at every stage (not just 2-factor), at the cost of
    a `transpose_a` on the data operand — the §6 lever. Pure algebraic identity, so
    the output is bit-identical. Assumes the matmul has no existing transpose flags
    (true for planner output) and that W (and the twiddle) are constants.
    """
    prog = _clone(prog)
    changed = True
    while changed:
        changed = False
        uses = _use_index(prog)
        defs = _def_index(prog)
        for ti, top in enumerate(prog.ops):
            if top.opcode is not OpCode.TRANSPOSE:
                continue
            S = tuple(top.result.shape)
            src = top.operands[0]
            if src.name not in defs or len(uses.get(src.name, [])) != 1:
                continue
            prod = prog.ops[defs[src.name]]

            # transpose(mul(matmul(W,M), Tc))  -> mul(matmul(M,Wᵀ,ta), Tcᵀ)
            if prod.opcode is OpCode.MUL_MODQ:
                a, tc = prod.operands
                if a.name not in defs or tc.name not in defs:
                    continue
                mm, tcop = prog.ops[defs[a.name]], prog.ops[defs[tc.name]]
                if not (mm.opcode is OpCode.MATMUL_MODQ and tcop.opcode is OpCode.CONST):
                    continue
                if len(uses.get(a.name, [])) != 1 or mm.attrs.get("transpose_a") or mm.attrs.get("transpose_b"):
                    continue
                w, m = mm.operands
                if w.name not in defs or prog.ops[defs[w.name]].opcode is not OpCode.CONST:
                    continue
                _transpose_const(prog, prog.ops[defs[w.name]])     # W -> Wᵀ
                _transpose_const(prog, tcop)                        # Tc -> Tcᵀ
                mm.operands = [m, prog.ops[defs[w.name]].result]
                mm.attrs = {**mm.attrs, "transpose_a": True}
                mm.result = Value(mm.result.name, S)
                prod.operands = [mm.result, tcop.result]
                prod.result = Value(prod.result.name, S)
                _rewire(prog, top.result.name, prod.result)
                del prog.ops[ti]
                changed = True
                break

            # transpose(matmul(W,M))  -> matmul(M, Wᵀ, ta)
            if prod.opcode is OpCode.MATMUL_MODQ:
                if prod.attrs.get("transpose_a") or prod.attrs.get("transpose_b"):
                    continue
                w, m = prod.operands
                if w.name not in defs or prog.ops[defs[w.name]].opcode is not OpCode.CONST:
                    continue
                _transpose_const(prog, prog.ops[defs[w.name]])
                prod.operands = [m, prog.ops[defs[w.name]].result]
                prod.attrs = {**prod.attrs, "transpose_a": True}
                prod.result = Value(prod.result.name, S)
                _rewire(prog, top.result.name, prod.result)
                del prog.ops[ti]
                changed = True
                break
    verify(prog)
    return prog


# --------------------------------------------------------------------------- #
def _chunk_size(dim, T):
    """Largest divisor of `dim` that is <= T (so tiles split evenly, no remainder)."""
    for c in range(min(T, dim), 0, -1):
        if dim % c == 0:
            return c
    return 1


def tile_dataflow(prog: Program, T: int) -> Program:
    """Tile each matmul's NTT-data operand along its non-reduction dim into equal
    tiles of size <= T (PLAN §13). Runs on a fold-mode (transpose-free) program.

    Per stage `matmul(data, Wᵀ, transpose_a) [⊙ Twᵀ]`:
      * `split` the data operand on its free dim into `parts` equal tiles (one op,
        many outputs); chunk = largest divisor of the dim that is <= T,
      * one small matmul per tile (reduction k never split; weight shared),
      * the twiddle is pre-split into per-tile constants and `mul`'d on each tile
        directly (no concat first),
      * `concat` the tile results before the stage-boundary reshape (the only layout
        op; no transpose, since transposes were folded).

    Pure partition of the same computation -> output is bit-identical.
    """
    prog = _clone(prog)
    cdef = {o.result.name: o for o in prog.ops
            if not isinstance(o.result, list) and o.opcode is OpCode.CONST}
    uses = _use_index(prog)
    uid = [0]

    def fresh(base):
        uid[0] += 1
        return f"{base}__t{uid[0]}"

    new_ops, skip = [], set()
    for idx, op in enumerate(prog.ops):
        if idx in skip:
            continue
        if op.opcode is not OpCode.MATMUL_MODQ:
            new_ops.append(op)
            continue

        data, W = op.operands[0], op.operands[1]
        ta, tb = op.attrs.get("transpose_a", False), op.attrs.get("transpose_b", False)
        q = op.attrs["q"]
        if data.name in cdef:                       # operand 0 must be the NTT data
            new_ops.append(op)
            continue

        tile_axis = 1 if ta else 0                  # the data's free (non-reduction) dim
        dim = data.shape[tile_axis]
        c = _chunk_size(dim, T)
        parts = dim // c

        # is this matmul followed by its twiddle mul?
        mul_op = mul_idx = tw = None
        us = uses.get(op.result.name, [])
        if len(us) == 1 and us[0][0] != "out":
            cand = prog.ops[us[0][0]]
            if cand.opcode is OpCode.MUL_MODQ and cand.operands[0].name == op.result.name \
                    and cand.operands[1].name in cdef:
                mul_op, mul_idx, tw = cand, us[0][0], cand.operands[1]

        if parts == 1:                              # dim <= T (or no proper divisor): leave as-is
            new_ops.append(op)
            continue

        # split data -> `parts` equal tiles (one op, many outputs)
        ts = list(data.shape); ts[tile_axis] = c
        tiles = [Value(fresh(data.name), tuple(ts)) for _ in range(parts)]
        new_ops.append(Op(OpCode.SPLIT, [data], tiles, {"axis": tile_axis, "parts": parts}))

        out_n = W.shape[0] if tb else W.shape[1]
        outs = []
        for i, tile in enumerate(tiles):
            a = Value(fresh(op.result.name), (c, out_n))
            new_ops.append(Op(OpCode.MATMUL_MODQ, [tile, W], a,
                              {"q": q, "transpose_a": ta, "transpose_b": tb}))
            if mul_op is not None:                  # pre-split twiddle const along its rows
                arr = prog.consts[cdef[tw.name].attrs["name"]]
                nm = f"{cdef[tw.name].attrs['name']}__r{i}"
                prog.consts[nm] = arr[i * c:(i + 1) * c, :].copy()
                tv = Value(fresh(tw.name), (c, out_n))
                new_ops.append(Op(OpCode.CONST, [], tv, {"name": nm, "shape": (c, out_n)}))
                b = Value(fresh(op.result.name), (c, out_n))
                new_ops.append(Op(OpCode.MUL_MODQ, [a, tv], b, {"q": q}))
                outs.append(b)
            else:
                outs.append(a)

        repl = mul_op.result if mul_op is not None else op.result
        new_ops.append(Op(OpCode.CONCAT, outs, Value(repl.name, repl.shape), {"axis": 0}))
        if mul_idx is not None:
            skip.add(mul_idx)

    # drop CONST ops left unused after twiddle pre-splitting
    used = {v.name for o in new_ops for v in o.operands} | {prog.output.name}
    new_ops = [o for o in new_ops
               if not (o.opcode is OpCode.CONST and o.result.name not in used)]
    prog.ops = new_ops
    verify(prog)
    return prog


# --------------------------------------------------------------------------- #
def fuse_all_to_all(prog: Program) -> Program:
    """Replace each inter-stage `concat -> reshape -> split` chain with a single
    `all_to_all` op (PLAN §13). The re-tile between stages is a block transpose =
    all-to-all communication; this makes that one step explicit in the IR.
    """
    prog = _clone(prog)
    uses = _use_index(prog)
    consumed, replace = set(), {}
    for i, op in enumerate(prog.ops):
        if op.opcode is not OpCode.CONCAT:
            continue
        u = uses.get(op.result.name, [])
        if len(u) != 1 or u[0][0] == "out":
            continue
        ri = u[0][0]
        R = prog.ops[ri]
        if R.opcode is not OpCode.RESHAPE:
            continue
        ur = uses.get(R.result.name, [])
        if len(ur) != 1 or ur[0][0] == "out":
            continue
        si = ur[0][0]
        S = prog.ops[si]
        if S.opcode is not OpCode.SPLIT:
            continue
        replace[i] = Op(OpCode.ALL_TO_ALL, list(op.operands), S.result, {
            "concat_axis": op.attrs["axis"],
            "reshape_to": tuple(R.attrs["shape"]),
            "split_axis": S.attrs["axis"],
            "parts": S.attrs["parts"],
        })
        consumed.update((ri, si))

    prog.ops = [replace.get(i, op) for i, op in enumerate(prog.ops) if i not in consumed]
    verify(prog)
    return prog


# --------------------------------------------------------------------------- #
def optimize(prog: Program) -> Program:
    """Operand-side transpose folding (§6 conservative): only the 2-factor
    corner-turn folds; ≥3-factor residual stays."""
    return eliminate_transposes(fuse_diagonal_twiddles(prog))


def optimize_fold(prog: Program) -> Program:
    """Result-side transpose folding (§6 with the transpose_a lever): removes the
    transpose at *every* stage via operand reorder + pre-transposed constants."""
    return fold_result_transposes(fuse_diagonal_twiddles(prog))


def op_counts(prog: Program) -> dict:
    c: dict = {}
    for o in prog.ops:
        c[o.opcode.value] = c.get(o.opcode.value, 0) + 1
    return c


if __name__ == "__main__":
    import numpy as np
    from ir import print_program, run
    from ntt_planner import forward_program, DEFAULT_PRIME

    q = DEFAULT_PRIME
    for plan in [(32, 32), (2, 4, 8, 16)]:
        p0 = forward_program(plan, q)
        p1 = optimize(p0)
        x = np.random.default_rng(0).integers(0, q, size=1024, dtype=np.int64)
        same = np.array_equal(np.asarray(run(p0, x), dtype=object),
                              np.asarray(run(p1, x), dtype=object))
        print(f"plan {plan}: {op_counts(p0)} -> {op_counts(p1)}   identical={same}")
    print("\n=== optimized IR for (32, 32) ===")
    print(print_program(optimize(forward_program((32, 32), q))))
