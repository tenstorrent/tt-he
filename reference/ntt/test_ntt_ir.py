# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the decomposed-NTT IR + planner.

Verifies, for many decomposition plans:
  - forward NTT (natural order) == ntt_matrix (schoolbook) and ntt_butterfly references
  - inverse round-trip  intt(ntt(x)) == x
  - grouped plan ≡ flattened plan  (identical IR and identical result, PLAN §5)
  - the static shape verifier accepts every emitted program

Run:  python3 test_ntt_ir.py
"""
import math

import numpy as np

import ntt_matrix
import ntt_butterfly
from ntt_ir import print_program, verify, run
from ntt_planner import (
    DEFAULT_PRIME, flatten, forward_program, ntt_natural, intt,
)
from ntt_passes import optimize, optimize_fold, tile_dataflow, op_counts

Q = DEFAULT_PRIME


def _eq(a, b):
    return np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object))


def _rand(n, seed):
    return np.random.default_rng(seed).integers(0, Q, size=n, dtype=np.int64)


PLANS_1024 = [
    1024,
    (2, 512),
    (32, 32),
    (2, 4, 8, 16),
    (16, 4, 8, 2),
    (4, 4, 4, 4, 4),
    ((4, 4), (4, 4), 4),       # nested
    (16, (8, 8)),
]
PLANS_32768 = [
    (32, 32, 32),
    ((32, 32), 32),
    (8, 8, 8, 8, 8),
    (2, 4, 8, 16, 32),
]


def test_forward_against_references_1024():
    x = _rand(1024, 1)
    gold_matrix = ntt_matrix.ntt(x, Q)        # O(N^2) schoolbook, natural order
    gold_butter = ntt_butterfly.ntt(x, Q)     # O(N log N) radix-2, natural order
    assert _eq(gold_matrix, gold_butter), "the two references disagree!"
    for plan in PLANS_1024:
        assert math.prod(flatten(plan)) == 1024
        got = ntt_natural(x, plan, Q)
        assert _eq(got, gold_matrix), f"forward mismatch vs matrix for plan {plan}"
    print(f"  [ok] forward == matrix & butterfly references, {len(PLANS_1024)} plans @ N=1024")


def test_forward_against_butterfly_32768():
    # matrix reference is O(N^2) -> too slow at 32768; use the butterfly reference.
    x = _rand(32768, 2)
    gold = ntt_butterfly.ntt(x, Q)
    for plan in PLANS_32768:
        assert math.prod(flatten(plan)) == 32768
        got = ntt_natural(x, plan, Q)
        assert _eq(got, gold), f"forward mismatch vs butterfly for plan {plan}"
    print(f"  [ok] forward == butterfly reference, {len(PLANS_32768)} plans @ N=32768")


def test_roundtrip():
    for n, plans, seed in ((1024, PLANS_1024, 3), (32768, PLANS_32768, 4)):
        x = _rand(n, seed)
        for plan in plans:
            X = ntt_natural(x, plan, Q)
            xr = intt(X, plan, Q)
            assert _eq(xr, x % Q), f"round-trip failed for plan {plan} @ N={n}"
    print("  [ok] intt(ntt(x)) == x for all plans @ N=1024 and N=32768")


def test_grouping_equiv_flatten():
    # A grouped plan and its flattened factor list must produce identical IR + output.
    pairs = [
        (((4, 4), (4, 4), 4), (4, 4, 4, 4, 4)),
        (((32, 32), 32),      (32, 32, 32)),
        ((16, (8, 8)),        (16, 8, 8)),
    ]
    x1024 = _rand(1024, 5)
    for grouped, flat in pairs:
        assert flatten(grouped) == list(flat)
        ir_g = print_program(forward_program(grouped, Q))
        ir_f = print_program(forward_program(flat, Q))
        assert ir_g == ir_f, f"IR differs: {grouped} vs {flat}"
        if math.prod(flat) == 1024:
            assert _eq(ntt_natural(x1024, grouped, Q), ntt_natural(x1024, flat, Q))
    print(f"  [ok] grouped plan ≡ flattened plan (identical IR), {len(pairs)} pairs")


def test_verifier_accepts_all():
    for plan in PLANS_1024 + PLANS_32768:
        verify(forward_program(plan, Q))   # raises on any shape/SSA problem
    print(f"  [ok] shape verifier accepts all {len(PLANS_1024)+len(PLANS_32768)} programs")


def test_passes_preserve_output():
    x = _rand(1024, 6)
    for plan in PLANS_1024:
        p0 = forward_program(plan, Q)
        p1 = optimize(p0)                  # verify() runs inside optimize
        assert _eq(run(p0, x), run(p1, x)), f"optimize changed output for {plan}"
    # 2-factor: the inter-stage corner-turn must be eliminated (transposes 2 -> 1)
    c = op_counts(optimize(forward_program((32, 32), Q)))
    assert c["transpose"] == 1, f"expected 1 residual transpose for 2-factor, got {c}"
    # >=3-factor: corner-turns feed non-identity reshapes -> residual transposes stay
    c4 = op_counts(optimize(forward_program((2, 4, 8, 16), Q)))
    assert c4["transpose"] == 4, f"expected 4 residual transposes for 4-factor, got {c4}"
    print("  [ok] optimize() preserves output; transpose-elim matches §6 (2-factor→1, ≥3→residual)")


def test_result_fold_removes_all_transposes():
    x = _rand(1024, 7)
    for plan in PLANS_1024:
        p0 = forward_program(plan, Q)
        pf = optimize_fold(p0)                          # result-side fold (transpose_a lever)
        assert _eq(run(p0, x), run(pf, x)), f"optimize_fold changed output for {plan}"
        assert op_counts(pf).get("transpose", 0) == 0, \
            f"result-side fold should remove all transposes for {plan}, got {op_counts(pf)}"
    print("  [ok] optimize_fold() preserves output and removes ALL transposes (transpose_a lever)")


def test_tiling_preserves_output():
    from ntt_planner import ntt_natural
    from ntt_ir import OpCode, outputs
    x = _rand(1024, 8)
    gold = ntt_matrix.ntt(x, Q)
    for plan in [(32, 32), (16, 64), (2, 4, 8, 16), (4, 4, 4, 4, 4)]:   # all N=1024
        for T in (8, 16, 32):
            got = ntt_natural(x, plan, Q, transpose="fold", tile=T)         # tiling requires fold
            assert _eq(got, gold), f"tiled mismatch: {plan} T={T}"
    # the full planner path fuses inter-stage concat+reshape+split into all_to_all
    pp = forward_program((2, 4, 8, 16), Q, transpose="fold", tile=16)
    c = op_counts(pp)
    assert c.get("all_to_all", 0) == 3, f"expected 3 inter-stage all_to_all, got {c}"
    assert c.get("concat", 0) == 1, f"only the final output concat should remain, got {c}"
    assert "transpose" not in c, "fold+tiled must be transpose-free"
    # split-based, no transpose; every tile-matmul's tiled (row) dim <= T; even splits
    p = tile_dataflow(forward_program((2, 4, 8, 16), Q, transpose="fold"), 16)
    assert "transpose" not in op_counts(p), "fold+tiled must be transpose-free"
    assert op_counts(p).get("split", 0) >= 1, "expected split ops"
    for op in p.ops:
        if op.opcode is OpCode.SPLIT:
            src = op.operands[0].shape[op.attrs["axis"]]
            assert src % op.attrs["parts"] == 0, "split must divide evenly"
        if op.opcode is OpCode.MATMUL_MODQ:
            assert op.result.shape[0] <= 16, f"tile-matmul row dim exceeds T: {op.result.shape}"
    print("  [ok] tiling preserves output (fold, T∈{8,16,32}); split-based, transpose-free, even tiles")


if __name__ == "__main__":
    tests = [
        test_verifier_accepts_all,
        test_grouping_equiv_flatten,
        test_forward_against_references_1024,
        test_forward_against_butterfly_32768,
        test_roundtrip,
        test_passes_preserve_output,
        test_result_fold_removes_all_transposes,
        test_tiling_preserves_output,
    ]
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print("\nOK: all tests pass")
