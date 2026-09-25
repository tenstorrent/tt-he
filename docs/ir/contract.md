# kernels_ir — the uniform kernel library for the HE IR

One kernel per L2/L3 op (docs/ir/ops.md), with a **uniform, tile-batched, DRAM-in/DRAM-out,
prime-agnostic** contract so codegen is a lookup table. See docs/ir/formalization.md (part 2:
orchestrate existing kernels) and ops.md (per-value `q`).

## Contract

- **Tile-batched.** Every kernel processes a run of `n_tiles` independent 32×32 tiles
  (`n_tiles` is runtime arg 0 of the compute kernel). One 32×32 tile = one RNS limb chunk.
- **DRAM-in / DRAM-out.** Inputs and outputs are DRAM buffers; no cross-kernel SRAM handoff
  (no Fusion B in v1). A reader streams DRAM→CB, the compute kernel CB→CB, a writer CB→DRAM.
- **Prime-agnostic.** No kernel hardcodes `q`. `q` (+ derived Montgomery scalars) are **runtime
  args**; per-prime data (twiddles, folded matrix, evk) are **selectable input buffers**. One
  binary serves every prime. v1: **one prime per dispatch** (scalar `q`); mixed-prime per-tile
  table is deferred (FORMALIZATION §8).
- **Fixed CB indices.** Inputs on `c_0` (and `c_1` for binary ops); output on `c_16`. UInt32,
  double-buffered (depth 2), page = one 32×32×4B tile = 4096 B.

## Standard runtime-arg layout

| kernel role | args |
|---|---|
| reader (binary) | `{a_addr, b_addr, n_tiles, start_tile}` — reads a,b at tile `start+i` → c_0,c_1 |
| reader (unary)  | `{in_addr, n_tiles, start_tile}` — reads in → c_0 |
| reader (matmul) | `{dyn_addr, wfold_addr}` + TensorAccessorArgs compile-args — dyn int32 → c_0, folded int8 W → c_1 |
| writer          | `{out_addr, n_tiles, start_tile}` — writes c_16 → tile `start+i` |
| compute         | `{n_tiles, q, <op-specific scalars…>}` — `n_tiles` (or op args) first |

Compute-kernel op-specific scalars:
- `mul_modq`: `{n_tiles, q, q_inv_32, q_low16, q_high16}` (Montgomery)
- `add_modq`, `sub_modq`, `transpose`: `{n_tiles}` (+ `q` for add/sub)
- `matmul_modq`: `{q, q_inv_32, q_low16, q_high16, mont(2^14)}` — one 32×32; param W folded to
  16 int8 tiles offline, dynamic operand decomposed on device (L3 internal)

## Current kernels (validated bit-exact vs golden on n150)

| IR op | compute kernel | status |
|---|---|---|
| `tile.mul_modq` | `compute/mul_modq.cpp` (Montgomery; from `modmul_compute`) | ✅ device-tested |
| `tile.add_modq` (binary) | `compute/add_modq.cpp` | ✅ device-tested |
| `rns.sub` / `sub_modq` | `compute/sub_modq.cpp` | ✅ device-tested |
| `tile.matmul_modq` | `compute/matmul_modq.cpp` (from `fused_matmul_a`: decompose+int8+recombine) | ✅ device-tested |
| `tile.transpose` | `compute/transpose.cpp` (from `transpose_tile`) | ✅ device-tested |
| `rns.gadget_decompose` | `compute/gadget_decompose.cpp` (from `gadget_decompose_signed`; 56-bit 2-limb input, compile-time B/ELL) | ✅ device-tested |
| `rns.crt_combine` | **composite:** `garner_diff.cpp` → `mul_modq` → `crt_mul_split.cpp` (3 dispatches) | ✅ device-tested |
| `rns.base_convert` | **composite of `mul_modq` + `add_modq`** (approx FBC = per-dst-limb modmul-by-CRT-const + modadd; exact adds `sub_modq` for the `m_sk`/CRNS correction) — no new kernel | ✅ primitives done |

**Multi-Q (per-element prime) — for limb-major packing / base conversion.**
`compute/mul_modq_multiq.cpp` + `dataflow/reader_multiq.cpp`: same Montgomery modmul, but `q` and
its constants (`q_inv`, `q_low`, `q_high`) are passed as **tiles** (CBs `c_2..c_5`, loaded once)
instead of scalar args — so **each element reduces mod its own prime**. The prime tile shares the
data tile's layout, so it covers per-row (limb-major), per-tile, and uniform with no row/lane
bookkeeping. Validated on device: 32 distinct primes, one per tile row (`test_ir_multiq`, ✅
0/1024). This is the capability that unblocks **limb-major Fusion A** (pack several base-conversion
ops' limb-tiles to fill the 32 rows) — see FORMALIZATION §1.5/§8.

**All primitive compute kernels are built and validated.** The higher-level HE/PIR ops are
*compositions* of these (ops.md §1b): `base_convert`, `rescale`, `relin`, `external_product`,
`mux`, `mod_switch` all orchestrate `mul_modq`/`add_modq`/`sub_modq`/`matmul_modq`/`transpose`/
`crt_*`/`gadget_decompose` — no further kernels needed. (Optional future coarse kernels — fused
`baseconv_fbc`, recombine-hoist — are dispatch-count optimizations, not new capability.)

With mul/add/sub/matmul/transpose an **NTT is fully buildable from the library**. Shared dataflow:
`reader_binop`, `reader_unop`, `reader_matmul`, `reader_crt`, `reader_gadget`, `writer_out`,
`writer_i32`. Tests: `test_ir_kernels`, `test_ir_matmul`, `test_ir_rns` (all bit-exact on n150).

## Notes

- `add_modq` here is **binary** (2 inputs). The IR `add_modq` is variadic (K-accumulate);
  compose binary adds, or add a variadic reducer later.
- Montgomery domain: `mul_modq` expects Montgomery-form inputs and returns Montgomery form
  (lazy `[0,2q)`; canonicalize with a finalize). `add`/`sub` are domain-agnostic (linear).
- Test: `./build/test_ir_kernels` — dispatches each op, checks bit-exact vs the C++ golden
  (which matches `reference.py`).
