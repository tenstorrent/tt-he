# ntt_matmul — int8 lowering of the NTT's modular matmuls

Same pipeline as `../ntt` (fold / **no transpose**, M/N tiled to ≤ `T`, **all_to_all** cross-stage)
**plus** two additions: K-tiling (so M, N, **and** K are all ≤ `T`) and lowering every
high-precision modular matmul `R = (A @ B) mod q` (`q < 2^28`) onto the **int8 tile matmul**.

Pipeline: `forward_program(plan, q, transpose="fold", tile=T)` → `tile_matmul(T)` →
`tilize_corner_turns(T)` → `lower_matmul_int8`.

### Two tile levels (`TILE_NTT` and `TILE_MM`)

Tiling has **two independent levels** (`main.py`):
- **`TILE_NTT`** — NTT-algorithm level: how stages/batches tile and the granularity of the
  cross-stage `all_to_all` (corner turn). Sets the **shapes** of the matmuls the NTT emits.
- **`TILE_MM`** — matmul **hardware** level: a second `tile_matmul(TILE_MM, prefix="%h")` blocks
  each NTT matmul into ≤ `TILE_MM` tiles (M/N/K, K accumulates with `add_modq`) — the on-device
  32×32 tiling. Must divide `TILE_NTT`; `TILE_MM == TILE_NTT` is the old single-level behaviour.

After the `TILE_MM` matmul tiling, a second `tilize_corner_turns(TILE_MM)` re-tilizes the corner
turn at the hardware granularity, so the cross-stage `all_to_all` drops from `TILE_NTT` to `TILE_MM`
tiles — **the MM tiles communicate directly**. The coarse `all_to_all` plus the matmul's concat-up
to `TILE_NTT` collapse into one `TILE_MM`-granular `all_to_all` (no mid-computation concat beyond the
begin/end boundaries), matching the device which works at 32×32 throughout.

One IR now expresses the device design spectrum: `PLAN=(32,32,32),TILE_NTT=32,TILE_MM=32` → the
3-factor (1536 i8 matmuls); `PLAN=(128,256),TILE_NTT=1024,TILE_MM=32` → the 2-factor big-matmul
hardware-tiled (6144 i8 matmuls, 1 all_to_all) — matching the measured device kernels. (Note: the
second level currently emits one `recombine_modq` per K-block then `add_modq`; the device instead
accumulates int32 planes across K and recombines once with a wider finalize — a future
"recombine-hoist" pass would close that gap.)

The result is **fully tile-resident**: every value in the computation is a ≤T tile, the twiddle
`mul_modq` runs per-tile, K-accumulation is `add_modq`, and the cross-stage corner-turn is an
**explicit tile-granular `all_to_all`** — a permutation over ≤T tiles, e.g. `(64,64)` emits
`all_to_all %g00,%g01,%g10,%g11 {perm=[0,2,1,3]}` (the grid transpose), all operands 32×32. The
**only** `slice`/`concat` are at the very begin (input → tiles) and end (tiles → output): no `slice`
ever cuts a compute result and no `concat` ever feeds a compute op. A factor larger than `T`
(e.g. `(64,64)`/`(32,64)` with T=32) is handled with no extra NTT stage.

## Files
- `ir.py` — the NTT IR (copied from `../ntt/ntt_ir.py`) **extended** with three opcodes and a
  per-value `dtype` (`i32` field / `i8` limb):
  - `DECOMPOSE` — bitcast one i32 matrix → 4 i8 radix-128 limbs (multi-output).
  - `RECOMBINE_MODQ` — merge 4 limb results → i32: `(Σ_r 128^r·C_r) mod q`.
  - `ADD_MODQ` — sum ≥2 tiles mod q (K-tile / per-limb accumulate).
  Verifier, numpy interpreter (bit-exact golden), `print_program`, and `print_mlir`
  (dtype-aware: emits `tensor<…xi8>` / `…xi32>`) all handle the new ops.
- `planner.py`, `ntt_passes.py` — copied from `../ntt` (imports repointed to the local `ir`);
  provide `forward_program` and the standard optimizations: `optimize_fold` (no transpose),
  `tile_dataflow` (M/N tiling), `fuse_all_to_all` (cross-stage comm).
- `passes.py`
  - `tile_matmul(prog, T)` — block every `matmul_modq` into ≤ T tiles in M, N, **and** K
    (K-tiles → `add_modq`, M/N-tiles → `concat`); transpose-aware; supports factor > T with no
    new NTT stage. (`tile_k` is the K-only special case, kept for convenience.)
  - `tilize_corner_turns(prog, T)` — replace each mid-computation layout region (the cross-stage
    corner-turn) with one explicit **`all_to_all_tiles`** op (a `perm` over ≤T tiles, optional
    per-tile `transpose`), computed by tag-tracing element provenance. Begin/end layout regions stay.
    Leaves only begin/end `slice`/`concat`; per-tile twiddle and deduped data slices come from
    `tile_matmul`.

Two distinct corner-turn opcodes (see `ir.py`):
  - `ALL_TO_ALL` — reshape-based retile (`concat→reshape→split`); appears when tiles aren't square
    `T×T` (e.g. `(8,8,8)`'s 8×16 tiles).
  - `ALL_TO_ALL_TILES` — pure tile permutation/gather; the device-tile NoC routing (e.g. `(64,64)`'s
    `perm=[0,2,1,3]` grid transpose). Both keep all operands ≤ T.
  - `lower_matmul_int8(prog, radix=128, nlimbs=4)` — replace each `matmul_modq` with
    `decompose` (dynamic operand) + 16 i8 `matmul_modq` + 4 `add_modq` + `recombine_modq`.
    The **constant** (parameter) operand is folded + decomposed **at plan time** and baked as
    CONSTs — no runtime computation on constants, no runtime `mul_modq`.
- `test.py` — numpy golden tests (run `python3 test.py`). Also emits the fold/expand MLIR pair.
- `viz.py` — **hierarchical multi-level** web viewer. Instead of unfolding the whole (unreadable)
  graph, it shows the NTT collapsed and lets you **drill into one block at a time**. Each
  decomposition pass is a level: **L1** raw (untiled — one matmul per stage, **reshape** corner
  turns) → **L2** NTT-tile (`TILE_NTT`; the cross-stage corner turn becomes an explicit
  **all_to_all** — expand an L1 cross-stage reshape to see it) → **L3** hardware-tile (`TILE_MM`)
  → **L4** int8. (L1↔L2 is a regeneration, linked by matching op output values; L2→L3→L4 chain via
  the `prov` attr.) **Click a block** to inspect it in
  the right panel; use the panel's **Expand / Collapse** buttons to drill that block into its next
  *real* decomposition (levels that don't change it are skipped) or fold it back. "Collapse all"
  resets. Double-bordered blocks are the ones that can be expanded. The panel shows the op, the
  **device kernel** that implements it (e.g. `matmul_compute_mtk.cpp`, `decompose_limbs.cpp`,
  `recombine_bigK.cpp`), and the **I/O shapes + datatypes** of every operand and result (e.g.
  `32x32 i8` → `32x32 i32`). **Mouse**: left-drag empty space = box-select multiple blocks;
  left-drag a node = move it (selected blocks move together); **right-drag = pan**; **scroll = zoom**;
  "Re-layout" re-runs the auto-layout. Powered by the `prov` op
  attribute each pass stamps (source op name); the viewer builds the provenance forest in-process
  from `planner`/`passes` (no MLIR parsing).

## Run
```
cd algorithm/ntt_matmul
python3 main.py            # edit PARAMETERS (PLAN/Q/TILE_NTT/TILE_MM), then run:
                           #   builds NTT, two-level tiling, int8 lowering, writes
                           #   ntt_fold.mlir + ntt_expand.mlir, launches the hierarchical viewer
python3 main.py --no-viz   # just generate the two .mlir files
python3 test.py            # numpy golden tests (no device)
python3 viz.py [port]      # viewer standalone (uses the default PLAN/tiles)
```
`main.py` is the entry point — change `PLAN`, `Q`, `TILE_NTT`, `TILE_MM` at the top.

## Status vs PLAN
- §3 K-tiling (`tile_k`), §4–§6 int8 lowering (`lower_matmul_int8`), §5 ops: **done, verified**
  (lowered IR is bit-exact vs the base mod-q IR and vs the O(N²) golden; `intt(ntt(x))==x`).
- §11b: `print_mlir` emits the decomposed matmul as a standalone `func.func @matmul_modq_i8`
  (see `example_matmul_int8.mlir`). Emitting it as a shared `func.func` **called** from `@ntt`
  (one callee reused by every stage) is the remaining refinement.
- §11c fold/expand viewer: **done** (global toggle). Per-node compound-node expand is the
  remaining nicety.

## Files — two things that look like clutter and are not

**The four `.mlir` are committed on purpose.** `main.py` writes `ntt_fold.mlir` / `ntt_expand.mlir`
and `test.py` writes `example_matmul_{highlevel,int8}.mlir`, so they are generated — but they are the
readable output of this prototype, `viz.py` loads them, and committing them means reading the lowered
IR needs no Python. Same rule as the committed `.bin` fixtures (see `reference/pir/.gitignore`).
Regenerate with `python3 main.py --no-viz` and `python3 test.py`; both are deterministic, so a clean
`git status` after running them is the check.

**`ntt_passes.py` and `planner.py` are deliberate copies of `../ntt/`'s, and must not be deduped.**
They differ from the originals only in imports — `from ntt_ir` becomes `from ir` — and that is the
whole point: this directory's `ir.py` is `../ntt/ntt_ir.py` plus ~190 lines (K-tiling, the int8
lowering). Importing the sibling's copies instead would silently run the passes against the *other*
IR. `planner.py` additionally appends `../ntt` to `sys.path` to reuse `ntt_matrix`'s number theory,
which is the pattern to follow when something genuinely is shared.

`PLAN.md` is the original design plan, kept for its derivations and because `docs/ir/ops.md` §4
cites `PLAN.md` §4. This README is the entry point.
