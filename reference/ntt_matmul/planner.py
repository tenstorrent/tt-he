# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""NTT planner: Strategy -> IR (PLAN.md §4, §5).

Takes a decomposition `plan` (a nested-group expression or flat factor list),
flattens it to an ordered factor list (the 2D-only backend ignores grouping, §5),
and emits the verified multidimensional flat-sweep as IR:

    for each factor f_t (P = prod of factors before t):
        reshape   -> (f_t, N/f_t)
        matmul    -> W_{f_t} @ M          # W_{f_t}[a,b] = w^{(N/f_t)·(ab mod f_t)}
        mul       -> ⊙ twiddle            # T[i,col]   = w^{P·i·(col//P)}
        transpose -> full corner-turn
    final reshape -> (N,)

The transform is computed in **natural-input, bit-reversed-output** order (standard
for FHE NTT). `digitrev_perm` maps bit-reversed -> natural; the forward/inverse pair
cancels the permutation so a round-trip needs none.

Math verified bit-exact against a naive mod-q DFT and the recursive four-step
(see _proto.py) for many factor lists.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

from ir import OpCode, Op, Value, Program, run, verify

# reuse number-theory helpers + golden from the sibling ntt package.
# append (don't insert at 0): local modules must shadow same-named ones in ../ntt (e.g. viz.py).
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ntt"))
from ntt_matrix import DEFAULT_PRIME, nth_root_of_unity  # noqa: E402


# --------------------------------------------------------------------------- #
# strategy / plan handling
# --------------------------------------------------------------------------- #
def flatten(plan) -> list:
    """Flatten a nested-group plan to an ordered list of leaf factors.

    int -> [int];  (A, B, ...) / [A, B, ...] -> concatenation of flattened children.
    Grouping is ignored for the 2D-only backend (PLAN §5).
    """
    if isinstance(plan, int):
        return [plan]
    out = []
    for child in plan:
        out.extend(flatten(child))
    return out


def digitrev_perm(factors) -> np.ndarray:
    """Index map so that  natural[k] = bitrev_output[perm[k]]  (mixed-radix digit reversal)."""
    N = math.prod(factors)
    weight_hi = [math.prod(factors[i + 1:]) for i in range(len(factors))]
    perm = np.empty(N, dtype=np.int64)
    for k in range(N):
        rem, pos = k, 0
        for i, f in enumerate(factors):
            d = rem % f
            rem //= f
            pos += d * weight_hi[i]
        perm[k] = pos
    return perm


# --------------------------------------------------------------------------- #
# baked constants
# --------------------------------------------------------------------------- #
def _ntt_matrix_const(f, q, wf):
    """f×f NTT matrix W[a,b] = wf^{(a·b) mod f}, wf a primitive f-th root."""
    pw = [pow(int(wf), e, q) for e in range(f)]
    M = np.empty((f, f), dtype=object)
    for a in range(f):
        for b in range(f):
            M[a, b] = pw[(a * b) % f]
    return M


def _twiddle_const(f, ncols, P, q, root, N):
    """T[i,col] = root^{P·i·(col//P)} (mod q); col//P selects the in-block remainder."""
    T = np.empty((f, ncols), dtype=object)
    for i in range(f):
        for col in range(ncols):
            n2 = col // P
            T[i, col] = pow(int(root), (P * i * n2) % N, q)
    return T


# --------------------------------------------------------------------------- #
# IR emission
# --------------------------------------------------------------------------- #
def build_program(factors, q, root, final_scale=None) -> Program:
    """Emit the flat-sweep IR for `factors` (product = N) using `root` (primitive
    N-th root). `final_scale` appends a scalar mul (used by the inverse: 1/N)."""
    N = math.prod(factors)
    consts: dict = {}
    ops: list = []
    _ctr = [0]

    def nv(shape):
        v = Value(f"%{_ctr[0]}", tuple(shape))
        _ctr[0] += 1
        return v

    x = Value("%x", (N,))

    f0 = factors[0]
    cur = nv((f0, N // f0))
    ops.append(Op(OpCode.RESHAPE, [x], cur, {"shape": (f0, N // f0)}))

    P = 1
    for t, f in enumerate(factors):
        ncols = N // f
        wf = pow(int(root), N // f, q)                       # primitive f-th root
        wname, tname = f"W{t}", f"T{t}"
        consts[wname] = _ntt_matrix_const(f, q, wf)
        consts[tname] = _twiddle_const(f, ncols, P, q, root, N)

        wv = nv((f, f))
        ops.append(Op(OpCode.CONST, [], wv, {"name": wname, "shape": (f, f)}))
        a = nv((f, ncols))
        ops.append(Op(OpCode.MATMUL_MODQ, [wv, cur], a, {"q": q}))

        tv = nv((f, ncols))
        ops.append(Op(OpCode.CONST, [], tv, {"name": tname, "shape": (f, ncols)}))
        b = nv((f, ncols))
        ops.append(Op(OpCode.MUL_MODQ, [a, tv], b, {"q": q}))

        bt = nv((ncols, f))
        ops.append(Op(OpCode.TRANSPOSE, [b], bt, {}))

        P *= f
        if t < len(factors) - 1:
            nf = factors[t + 1]
            cur = nv((nf, N // nf))
            ops.append(Op(OpCode.RESHAPE, [bt], cur, {"shape": (nf, N // nf)}))
        else:
            cur = bt

    out = nv((N,))
    ops.append(Op(OpCode.RESHAPE, [cur], out, {"shape": (N,)}))

    if final_scale is not None:
        consts["SCALE"] = np.array([int(final_scale) % q], dtype=object)
        sv = nv((1,))
        ops.append(Op(OpCode.CONST, [], sv, {"name": "SCALE", "shape": (1,)}))
        scaled = nv((N,))
        ops.append(Op(OpCode.MUL_MODQ, [out, sv], scaled, {"q": q}))
        out = scaled

    prog = Program(q, [x], ops, out, consts,
                   meta={"factors": list(factors), "N": N, "layout": "bitrev"})
    verify(prog)
    return prog


# --------------------------------------------------------------------------- #
# high-level API
# --------------------------------------------------------------------------- #
# transpose config:
#   "keep" -> emit every stage's explicit transpose (the literal flat sweep)
#   "fold" -> merge all transposes into the matmuls (transpose_a + pre-transposed
#             constants); the program becomes transpose-free
TRANSPOSE_MODES = ("keep", "fold")


def _apply_passes(prog: Program, transpose: str, tile) -> Program:
    if transpose == "fold":
        from ntt_passes import optimize_fold      # local import: ntt_passes uses ntt_ir only
        prog = optimize_fold(prog)
    elif transpose != "keep":
        raise ValueError(f"transpose must be one of {TRANSPOSE_MODES}, got {transpose!r}")
    if tile is not None:
        if transpose != "fold":
            raise ValueError("tile requires transpose='fold' (tiling assumes the transpose-free form)")
        from ntt_passes import tile_dataflow, fuse_all_to_all
        prog = fuse_all_to_all(tile_dataflow(prog, tile))   # inter-stage retile -> all_to_all
    return prog


def forward_program(plan, q=DEFAULT_PRIME, transpose="keep", tile=None) -> Program:
    factors = flatten(plan)
    N = math.prod(factors)
    w = nth_root_of_unity(N, q)
    return _apply_passes(build_program(factors, q, w), transpose, tile)


def inverse_program(plan, q=DEFAULT_PRIME, transpose="keep", tile=None) -> Program:
    factors = flatten(plan)
    N = math.prod(factors)
    w = nth_root_of_unity(N, q)
    w_inv = pow(int(w), q - 2, q)
    n_inv = pow(N, q - 2, q)
    return _apply_passes(build_program(factors, q, w_inv, final_scale=n_inv), transpose, tile)


def ntt(x, plan, q=DEFAULT_PRIME, transpose="keep", tile=None):
    """Forward NTT, returning the raw **bit-reversed** output (no permutation)."""
    return run(forward_program(plan, q, transpose, tile), x)


def ntt_natural(x, plan, q=DEFAULT_PRIME, transpose="keep", tile=None):
    """Forward NTT in **natural** output order (applies the digit-reversal)."""
    factors = flatten(plan)
    return ntt(x, plan, q, transpose, tile)[digitrev_perm(factors)]


def intt(x_natural, plan, q=DEFAULT_PRIME, transpose="keep", tile=None):
    """Inverse NTT: natural-order input -> natural-order output."""
    factors = flatten(plan)
    raw = run(inverse_program(plan, q, transpose, tile), x_natural)
    return raw[digitrev_perm(factors)]


if __name__ == "__main__":
    from ir import print_program

    q = DEFAULT_PRIME
    print("=== IR for plan ((4,4),8)  ->  flattened [4,4,8], N=128 ===\n")
    print(print_program(forward_program(((4, 4), 8), q)))

    rng = np.random.default_rng(0)
    x = rng.integers(0, q, size=128, dtype=np.int64)
    X = ntt_natural(x, ((4, 4), 8), q)
    xr = intt(X, ((4, 4), 8), q)
    print("\nround-trip intt(ntt(x)) == x:", np.array_equal(xr % q, x % q))
