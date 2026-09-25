# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Example: generate, export, and run a decomposed NTT.

Generates BOTH program versions:
  - "keep" : every stage keeps its explicit transpose (the literal flat sweep)
  - "fold" : all transposes merged into the matmuls (transpose_a + pre-transposed
             constants) -> a transpose-free program

The TRANSPOSE config below selects which one is actually executed (round-trip), and
the chosen program is exported to JSON, re-loaded with load_json, and run from the
loaded form to show the export is self-contained.

  python3 example.py
"""
import numpy as np

from ntt_ir import print_mlir, export_json, load_json, run
from ntt_planner import (
    forward_program, inverse_program, ntt_natural, intt, flatten,
    digitrev_perm, DEFAULT_PRIME, TRANSPOSE_MODES,
)
from ntt_passes import op_counts

# ============================ config ============================
PLAN      = (8, 8, 8, 8)  # decomposition; product = N
TRANSPOSE = "fold"          # "keep" or "fold" — transpose handling
TILE      = 16             # tile m/n parallel dims to <= this (None = no tiling)
# ================================================================

Q = DEFAULT_PRIME
N = int(np.prod(flatten(PLAN)))
assert TRANSPOSE in TRANSPOSE_MODES
print(f"plan = {PLAN}  (factors {flatten(PLAN)}, N={N}),  "
      f"transpose = {TRANSPOSE!r},  tile = {TILE}\n")

# --- generate the untiled versions (transpose handling only) -------------------
prog_keep = forward_program(PLAN, Q, transpose="keep")
prog_fold = forward_program(PLAN, Q, transpose="fold")
print("op counts (untiled):")
print("  keep :", op_counts(prog_keep))
print("  fold :", op_counts(prog_fold))

# --- run, forward + inverse round-trip (tiling requires fold) ------------------
x = np.arange(1, N + 1, dtype=np.int64)
run_mode = "fold" if TILE else TRANSPOSE
X = ntt_natural(x, PLAN, Q, transpose=run_mode, tile=TILE)
x_rt = intt(X, PLAN, Q, transpose=run_mode, tile=TILE)
print(f"\nran tile={TILE} ('{run_mode}'):  round-trip intt(ntt(x)) == x : "
      f"{np.array_equal(np.asarray(x_rt, dtype=object), x % Q)}")

# every variant agrees (all algebraically identical)
ref = np.asarray(ntt_natural(x, PLAN, Q, transpose='keep'), dtype=object)
variants = [("keep", None), ("fold", None), ("fold", TILE)]
for mode, t in variants:
    assert np.array_equal(np.asarray(ntt_natural(x, PLAN, Q, transpose=mode, tile=t),
                                     dtype=object), ref), f"variant {mode}/{t} disagrees!"
print("keep, fold, and fold+tiled all produce identical results: True")

# --- tiled version (split-based; fold mode only) -------------------------------
prog_tiled = forward_program(PLAN, Q, transpose="fold", tile=TILE)
print(f"tiled (T={TILE}, split-based): op count = {op_counts(prog_tiled)}")

# --- export all versions, and the chosen one as re-executable JSON -------------
with open("example_ntt_keep.mlir", "w") as f:
    f.write(print_mlir(prog_keep, "ntt_keep") + "\n")
with open("example_ntt_fold.mlir", "w") as f:
    f.write(print_mlir(prog_fold, "ntt_fold") + "\n")
with open("example_ntt_tiled.mlir", "w") as f:
    f.write(print_mlir(prog_tiled, "ntt_tiled") + "\n")
chosen = prog_tiled if TILE else (prog_fold if TRANSPOSE == "fold" else prog_keep)
export_json(chosen, "example_ntt.json", include_const_data=True)

# --- re-load the exported program and run it standalone ------------------------
loaded = load_json("example_ntt.json")
X_loaded = np.asarray(run(loaded, x), dtype=object)[digitrev_perm(flatten(PLAN))]
print(f"\nloaded example_ntt.json and ran it: matches in-memory run : "
      f"{np.array_equal(X_loaded, np.asarray(X, dtype=object))}")

print("\nexported:")
print("  example_ntt_keep.mlir   (all transposes)")
print("  example_ntt_fold.mlir   (transpose-free, transpose_a matmuls)")
print(f"  example_ntt_tiled.mlir  (fold + tiled to T={TILE}: split -> per-tile matmul/mul -> concat)")
print("  example_ntt.json        (the chosen version, re-executable via load_json)")

label = f"tiled_T{TILE}" if TILE else TRANSPOSE
print(f"\n================ chosen ('{label}') program ================")
print(print_mlir(chosen, f"ntt_{label}"))
