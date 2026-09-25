# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ntt_matmul entry point.

Edit the PARAMETERS block below, then:

    python3 main.py             # generate MLIR + launch the DAG viewer (fold/expand)
    python3 main.py --no-viz    # just generate the two .mlir files
    python3 main.py --port 8060 # serve on a different port

It builds the NTT for `PLAN` with TWO levels of tiling, lowers every modular matmul
to the int8 decomposition, checks bit-exactness vs the base IR, and writes
ntt_fold.mlir / ntt_expand.mlir (+ viewer with a fold/expand toggle).

Two tile parameters:
  TILE_NTT  NTT-algorithm-level tile -- how the NTT stages/batches are tiled and the
            granularity of the cross-stage all_to_all (corner turn). Sets the SHAPES
            of the matmuls the NTT emits (up to TILE_NTT).
  TILE_MM   matmul HARDWARE tile -- each NTT matmul is then blocked into <= TILE_MM
            tiles in M/N/K (K accumulates), i.e. the on-device 32x32 tiling. Must
            divide TILE_NTT; TILE_MM == TILE_NTT reproduces the single-level form.

Examples (the two device designs, from one IR):
  PLAN=(32,32,32), TILE_NTT=32,   TILE_MM=32  -> 3-factor (32x32 matmuls)
  PLAN=(128,256),  TILE_NTT=1024, TILE_MM=32  -> 2-factor (big matmuls, hw-tiled to 32)
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

import ir
import passes
import planner

# ============================== PARAMETERS (edit me) ==============================
PLAN     = (32, 64)    # NTT decomposition: factors (nested tuples allowed), product = N
TILE_NTT = 32     # NTT-level tile: stages/batches + all_to_all granularity
TILE_MM  = 32              # matmul hardware tile: each matmul blocked to <= TILE_MM (M/N/K)

Q        = 167772161       # NTT-friendly prime, must be < 2**28
RADIX    = 128             # 32-bit -> 8-bit limbs (fixed 4 limbs)
NLIMBS   = 4
FOLD_FILE   = "ntt_fold.mlir"     # high-level matmul_modq (NTT-level tiles)
EXPAND_FILE = "ntt_expand.mlir"   # int8-decomposed (hardware tiles)
# =================================================================================


def generate():
    factors = planner.flatten(PLAN)
    N = int(np.prod(factors))
    assert Q < (1 << 28), "Q must be < 2**28 (radix-128, 4 limbs)"
    assert TILE_NTT % TILE_MM == 0, "TILE_MM must divide TILE_NTT"

    # --- NTT-algorithm level: fold (no transpose) + tile M/N to <=TILE_NTT + all_to_all ---
    ref = planner.forward_program(PLAN, Q, transpose="keep")                  # semantics reference
    opt = planner.forward_program(PLAN, Q, transpose="fold", tile=TILE_NTT)   # no transpose, M/N tiled
    opt = passes.tile_matmul(opt, TILE_NTT)                                   # matmul shapes <= TILE_NTT
    opt = passes.tilize_corner_turns(opt, TILE_NTT)                           # corner turn at NTT-tile granularity

    # --- matmul hardware level: block each matmul to <= TILE_MM (the on-device 32x32 tiling) ---
    if TILE_MM < TILE_NTT:
        opt = passes.tile_matmul(opt, TILE_MM, prefix="%h")                   # M/N/K -> hardware tiles, K accumulates
        opt = passes.tilize_corner_turns(opt, TILE_MM)                        # corner turn at TILE_MM: the MM tiles
                                                                             # communicate directly (no concat-up to TILE_NTT)

    # --- int8 precision lowering (on the hardware-tiled matmuls) ---
    low = passes.lower_matmul_int8(opt, radix=RADIX, nlimbs=NLIMBS)

    # correctness: the optimized + lowered IR must equal the reference bit-exactly
    x = np.random.default_rng(0).integers(0, Q, size=N, dtype=np.int64)
    assert np.array_equal(ir.run(opt, x), ir.run(ref, x)), "optimization changed the result!"
    assert np.array_equal(ir.run(low, x), ir.run(ref, x)), "int8 lowering changed the result!"

    with open(FOLD_FILE, "w") as f:
        f.write(ir.print_mlir(opt, func_name="ntt"))
    with open(EXPAND_FILE, "w") as f:
        f.write(ir.print_mlir(low, func_name="ntt_int8"))

    def counts(p):
        return dict(Counter(o.opcode.value for o in p.ops))

    print(f"plan={PLAN}  N={N}  q={Q}  TILE_NTT={TILE_NTT}  TILE_MM={TILE_MM}  radix={RADIX}")
    print(f"  fold   -> {FOLD_FILE:18s} {counts(opt)}")
    print(f"  expand -> {EXPAND_FILE:18s} {counts(low)}")
    print("  two-level tiling (NTT + matmul/hardware); bit-exact vs reference.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="generate NTT int8 MLIR and launch the viewer")
    ap.add_argument("--no-viz", action="store_true", help="only write the .mlir files")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()

    generate()
    if not args.no_viz:
        import viz
        viz.serve(PLAN, Q, TILE_NTT, TILE_MM, radix=RADIX, nlimbs=NLIMBS, port=args.port)
