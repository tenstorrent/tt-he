# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the int8-decomposition lowering (ntt_matmul/PLAN.md).

Run:  python3 test.py        # from the ntt_matmul/ directory

Checks (numpy golden, no device):
  1. decompose -> recombine_modq round-trips an i32 matrix.
  2. add_modq sums tiles mod q.
  3. lower_matmul_int8 of a single W@data matmul == the matmul it replaces.
  4. tile_k of a K>T matmul == the un-tiled matmul.
  5. end-to-end NTT: tile_k + lower_matmul_int8 == the base mod-q IR, bit-exact, and
     == the O(N^2) golden (ntt_matrix) in natural order.
Also emits example_matmul_highlevel.mlir / example_matmul_int8.mlir (the fold/expand pair).
"""
import os
import sys

import numpy as np

import ir
import passes
import planner

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ntt"))
from ntt_matrix import DEFAULT_PRIME, nth_root_of_unity, ntt as golden_ntt  # noqa: E402

Q = DEFAULT_PRIME
RADIX = 128


def _ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return bool(cond)


def matmul_program(W, K, N, q=Q):
    """Single op:  out = (W @ data) mod q,  W a baked (M,K) constant, data the (K,N) input."""
    M = W.shape[0]
    data = ir.Value("%x", (K, N))
    wv = ir.Value("%w", (M, K))
    out = ir.Value("%o", (M, N))
    ops = [
        ir.Op(ir.OpCode.CONST, [], wv, {"name": "W", "shape": (M, K)}),
        ir.Op(ir.OpCode.MATMUL_MODQ, [wv, data], out, {"q": q}),
    ]
    return ir.Program(q, [data], ops, out, {"W": np.asarray(W, dtype=object)})


# --------------------------------------------------------------------------- #
def test_decompose_recombine():
    rng = np.random.default_rng(0)
    M, N = 4, 5
    X = rng.integers(0, Q, size=(M, N), dtype=np.int64)
    x = ir.Value("%x", (M, N))
    limbs = [ir.Value(f"%l{p}", (M, N), "i8") for p in range(4)]
    out = ir.Value("%o", (M, N))
    ops = [
        ir.Op(ir.OpCode.DECOMPOSE, [x], limbs, {"radix": RADIX, "nlimbs": 4}),
        ir.Op(ir.OpCode.RECOMBINE_MODQ, limbs, out, {"radix": RADIX, "q": Q}),
    ]
    prog = ir.Program(Q, [x], ops, out)
    got = ir.run(prog, X)
    return _ok("decompose∘recombine round-trip", np.array_equal(got % Q, X % Q))


def test_add_modq():
    rng = np.random.default_rng(1)
    M, N = 3, 3
    a = ir.Value("%x", (M, N))
    c1 = ir.Value("%c1", (M, N)); c2 = ir.Value("%c2", (M, N))
    s = ir.Value("%s", (M, N))
    # build 2 consts and add them to the input
    A1 = rng.integers(0, Q, size=(M, N), dtype=np.int64)
    A2 = rng.integers(0, Q, size=(M, N), dtype=np.int64)
    ops = [
        ir.Op(ir.OpCode.CONST, [], c1, {"name": "A1", "shape": (M, N)}),
        ir.Op(ir.OpCode.CONST, [], c2, {"name": "A2", "shape": (M, N)}),
        ir.Op(ir.OpCode.ADD_MODQ, [a, c1, c2], s, {"q": Q}),
    ]
    prog = ir.Program(Q, [a], ops, s,
                      {"A1": np.asarray(A1, dtype=object), "A2": np.asarray(A2, dtype=object)})
    X = rng.integers(0, Q, size=(M, N), dtype=np.int64)
    got = ir.run(prog, X)
    exp = (X.astype(object) + A1 + A2) % Q
    return _ok("add_modq (3-way)", np.array_equal(got, exp))


def test_lower_single_matmul():
    rng = np.random.default_rng(2)
    M, K, N = 6, 8, 5
    W = rng.integers(0, Q, size=(M, K), dtype=np.int64)
    prog = matmul_program(W, K, N)
    data = rng.integers(0, Q, size=(K, N), dtype=np.int64)
    base = ir.run(prog, data)
    low = passes.lower_matmul_int8(prog, radix=RADIX, nlimbs=4)
    got = ir.run(low, data)
    nmm = sum(1 for o in low.ops if o.opcode is ir.OpCode.MATMUL_MODQ)
    return _ok(f"lower_matmul_int8 single matmul (16 i8 matmuls: {nmm})",
               np.array_equal(got, base) and nmm == 16)


def test_tile_matmul():
    # M, N, K all > T  -> full 3D blocking; verify == base and every block <= T
    rng = np.random.default_rng(3)
    M, K, N, T = 64, 64, 48, 32
    W = rng.integers(0, Q, size=(M, K), dtype=np.int64)
    prog = matmul_program(W, K, N)
    data = rng.integers(0, Q, size=(K, N), dtype=np.int64)
    base = ir.run(prog, data)
    tiled = passes.tile_matmul(prog, T)
    got = ir.run(tiled, data)
    low = passes.lower_matmul_int8(tiled, radix=RADIX, nlimbs=4)
    got2 = ir.run(low, data)
    mx = _max_matmul_dim(tiled)
    return _ok(f"tile_matmul (M,N,K=64,64,48; T=32) maxdim={mx} + lower == base",
               np.array_equal(got, base) and np.array_equal(got2, base) and mx <= T)


def _max_matmul_dim(prog):
    m = 0
    for op in prog.ops:
        if op.opcode is ir.OpCode.MATMUL_MODQ:
            for v in op.operands + [op.result]:
                m = max(m, *v.shape)
    return m


def _count(prog, oc):
    return sum(1 for o in prog.ops if o.opcode is oc)


def test_ntt_end_to_end():
    """Full pipeline = ntt optimizations (fold/no-transpose + M/N tile + all_to_all) + K-tile + int8.

    Multi-tile stages (free dim > T) -> all_to_all cross-stage; single-tile -> reshape (no comm)."""
    cases = [((8, 8, 8), 16), ((4, 4, 4, 4), 8), ((64, 64), 32), ((64, 32), 32), ((32, 64), 32)]
    COMPUTE = {ir.OpCode.MATMUL_MODQ, ir.OpCode.MUL_MODQ, ir.OpCode.ADD_MODQ,
               ir.OpCode.DECOMPOSE, ir.OpCode.RECOMBINE_MODQ}
    allok = True
    for plan, T in cases:
        factors = planner.flatten(plan)
        N = int(np.prod(factors))
        ref = planner.forward_program(plan, Q, transpose="keep")
        opt = passes.tilize_corner_turns(
            passes.tile_matmul(planner.forward_program(plan, Q, transpose="fold", tile=T), T), T)
        low = passes.lower_matmul_int8(opt, radix=RADIX, nlimbs=4)
        rng = np.random.default_rng(7)
        x = rng.integers(0, Q, size=N, dtype=np.int64)
        base = ir.run(ref, x)
        got = np.asarray(ir.run(low, x), dtype=object)
        bit_exact = np.array_equal(got, base)
        perm = planner.digitrev_perm(factors)
        gold = golden_ntt(x, Q, nth_root_of_unity(N, Q))
        natural_ok = np.array_equal(got[perm] % Q, gold % Q)
        # structure: no transpose, all matmul dims <= T, and NO mid-computation slice/concat:
        #   no slice cuts a compute result, no concat feeds a compute op (begin/end only).
        defop = {v.name: o for o in low.ops for v in ir.outputs(o)}
        uses = {}
        for o in low.ops:
            for v in o.operands:
                uses.setdefault(v.name, []).append(o)
        bad_sl = sum(1 for o in low.ops if o.opcode is ir.OpCode.SLICE
                     and defop.get(o.operands[0].name) and defop[o.operands[0].name].opcode in COMPUTE)
        bad_cc = sum(1 for o in low.ops if o.opcode is ir.OpCode.CONCAT
                     and any(c.opcode in COMPUTE for c in uses.get(o.result.name, [])))
        n_tr = _count(low, ir.OpCode.TRANSPOSE)
        mx = _max_matmul_dim(low)
        # cross-stage comm (either all_to_all flavor) must be tile-granular: every operand/result <= T
        a2a = [o for o in low.ops if o.opcode in (ir.OpCode.ALL_TO_ALL, ir.OpCode.ALL_TO_ALL_TILES)]
        a2a_n = len(a2a)
        a2a_tile = all(max(v.shape) <= T for o in a2a for v in o.operands + ir.outputs(o))
        struct = (n_tr == 0 and mx <= T and bad_sl == 0 and bad_cc == 0 and a2a_tile)
        allok &= _ok(f"NTT {plan} T={T}: exact={bit_exact} golden={natural_ok} | transpose={n_tr} "
                     f"maxdim={mx} slice/concat_in_compute={bad_sl}/{bad_cc} all_to_all={a2a_n}(tile={a2a_tile})",
                     bit_exact and natural_ok and struct)
    return allok


def test_two_level_tiling():
    """Second (matmul/hardware) tile level: T_mm < T_ntt blocks each NTT matmul to <=T_mm,
    stays bit-exact, and matmul shapes drop to T_mm while the all_to_all stays NTT-granular."""
    COMPUTE = {ir.OpCode.MATMUL_MODQ, ir.OpCode.MUL_MODQ, ir.OpCode.ADD_MODQ,
               ir.OpCode.DECOMPOSE, ir.OpCode.RECOMBINE_MODQ}
    allok = True
    for plan, T_ntt, T_mm in [((128, 256), 1024, 32), ((8, 8, 8), 16, 8), ((64, 64), 64, 32)]:
        N = int(np.prod(planner.flatten(plan)))
        ref = planner.forward_program(plan, Q, transpose="keep")
        opt = passes.tilize_corner_turns(passes.tile_matmul(
            planner.forward_program(plan, Q, transpose="fold", tile=T_ntt), T_ntt), T_ntt)
        two = passes.tile_matmul(opt, T_mm, prefix="%h") if T_mm < T_ntt else opt
        low = passes.lower_matmul_int8(two, radix=RADIX, nlimbs=4)
        x = np.random.default_rng(3).integers(0, Q, size=N, dtype=np.int64)
        bit_exact = np.array_equal(ir.run(low, x), ir.run(ref, x))
        mx = _max_matmul_dim(two)                     # every matmul now <= T_mm
        # cross-stage comm stays NTT-granular (<= T_ntt), unaffected by the matmul tiling
        a2a_ok = all(max(v.shape) <= T_ntt for o in two.ops
                     if o.opcode in (ir.OpCode.ALL_TO_ALL, ir.OpCode.ALL_TO_ALL_TILES)
                     for v in o.operands + ir.outputs(o))
        allok &= _ok(f"two-level {plan} T_ntt={T_ntt} T_mm={T_mm}: exact={bit_exact} matmul_maxdim={mx}",
                     bit_exact and mx <= T_mm and a2a_ok)
    return allok


def test_roundtrip():
    plan, T = (8, 8), 16
    factors = planner.flatten(plan)
    N = int(np.prod(factors))
    perm = planner.digitrev_perm(factors)
    fwd = passes.lower_matmul_int8(passes.tilize_corner_turns(passes.tile_matmul(
        planner.forward_program(plan, Q, transpose="fold", tile=T), T), T))
    inv = passes.lower_matmul_int8(passes.tilize_corner_turns(passes.tile_matmul(
        planner.inverse_program(plan, Q, transpose="fold", tile=T), T), T))
    rng = np.random.default_rng(11)
    x = rng.integers(0, Q, size=N, dtype=np.int64)
    X = np.asarray(ir.run(fwd, x), dtype=object)[perm]        # forward, natural order
    xr = np.asarray(ir.run(inv, X), dtype=object)[perm]        # inverse, natural order
    return _ok("intt(ntt(x)) == x  (both int8-lowered)", np.array_equal(xr % Q, x % Q))


def emit_examples():
    """Write the fold/expand MLIR pair for the viewer (high-level vs int8-lowered)."""
    W = np.random.default_rng(0).integers(0, Q, size=(8, 8), dtype=np.int64)
    prog = matmul_program(W, 8, 8)
    low = passes.lower_matmul_int8(prog, radix=RADIX, nlimbs=4)
    with open("example_matmul_highlevel.mlir", "w") as f:
        f.write(ir.print_mlir(prog, func_name="matmul_modq"))
    with open("example_matmul_int8.mlir", "w") as f:
        f.write(ir.print_mlir(low, func_name="matmul_modq_i8"))
    print("  wrote example_matmul_highlevel.mlir / example_matmul_int8.mlir")


if __name__ == "__main__":
    print("ntt_matmul tests:")
    results = [
        test_decompose_recombine(),
        test_add_modq(),
        test_lower_single_matmul(),
        test_tile_matmul(),
        test_ntt_end_to_end(),
        test_two_level_tiling(),
        test_roundtrip(),
    ]
    emit_examples()
    print("\nALL PASS" if all(results) else "\nSOME FAILED")
    sys.exit(0 if all(results) else 1)
