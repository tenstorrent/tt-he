# Plan: High-Performance Vector Modular Multiplication on TT-Metal (Wormhole B0)

> **Status: Historical.** Superseded by [`../modular_multiplication.md`](../modular_multiplication.md).
> The original plan, written before the kernel existed.
> Numbers are a snapshot, measured on Wormhole n150.

**Goal.** Compute, for an input vector of length `N = 2^12 = 4096`, the element-wise
modular product under a fixed 28-bit prime modulus `q`, using the *same Montgomery
algorithm* as `reference/cpu/montgomery.hpp` (`mulmod` = `reduce_optimized(mul_hi_lo(a,b))`,
operands in lazy form `[0, 2q]`, output `(a*b) mod q`). Target device: **n150 /
Wormhole B0** (confirmed via `tt-smi`).

Status: design plan. No device code written yet. Reference CPU implementation and
its test live in `reference/cpu/` (repo-relative).

---

## 1. The hardware reality that shapes everything

The Tensix **SFPU** (the per-element SIMD unit, 32 lanes/face, `vInt`/`vUInt`/`vFloat`)
is the right engine for this op (FPU/matrix engine does dot-products, not element-wise
modmul). Key facts established by reading the toolchain on this machine:

- **No integer multiply instruction, no integer `operator*`.** SFPI exposes int
  add/sub (`SFPIADD`), AND/OR/XOR (`SFPAND`...), logical shifts (`SFPSHFT`),
  compares, and `v_if/v_else` predication — but `operator*` exists **only for
  `vFloat`** (`runtime/sfpi/include/sfpi_funcs.h:391`). `fractional_mul`/`sfpmul24`
  is gated to Blackhole (`sfpi_lib.h:158`, `#if __riscv_xtttensixbh`) → **unavailable here**.
- **Integer multiply is provided, but emulated.** Wormhole B0 ships:
  - `mul_int_tile<DataFormat::{Int32,UInt32}>` → **32×32 → low 32 bits**
    (`tt_metal/hw/inc/api/compute/mul_int_sfpu.h`, impl
    `hw/ckernels/wormhole_b0/.../ckernel_sfpu_mul_int32.h`).
  - `mul_int_tile<DataFormat::UInt16>` → **16×16 → low 16 bits**
    (`.../ckernel_sfpu_mul_int.h`, throughput-optimized, ~12 cyc/row).
  - The emulation splits operands into small limbs (11-bit for the 32-bit version,
    8-bit for the 16-bit), casts limbs to **fp32** (the only real multiplier),
    accumulates partial products with `SFPMAD`, and reconstructs integers via the
    `x + 2^23` magic-constant mantissa-extraction trick (`SFPEXMAN`).

**Why this matters for Montgomery.** `mul_int32` returns only the **low 32 bits** of
a product. Montgomery reduction needs the **high 32 bits** of `t·q`
(`t_final = floor(t*q / 2^32)`). This is precisely why `montgomery.hpp` already
decomposes into 16-bit limbs and assembles the high half by hand — that same
decomposition is what we map onto the SFPU. The CPU algorithm and the device's
native int-mul emulation use the *same* limb+fp32 idea, so the port is natural.

---

## 2. Algorithm recap (from `reference/cpu/montgomery.hpp`) and its op-level shape

Constants (compile-time, baked into the kernel): `q` (28-bit prime), `q_inv_32 = q^-1
mod 2^32`, `q_low = q & 0xFFFF`, `q_high = q >> 16`.

`mulmod(a, b)`, with `a,b ∈ [0, 2q]`, `2q < 2^29`:

1. **Full product** `a*b` (≤ `4q^2 < 2^58`) as two 32-bit words `(hi, lo)` — the
   `mul_hi_lo` step. Needs limb decomposition (operands ~29-bit → three 11-bit limbs).
2. **Montgomery reduce** `(hi, lo)` → lazy `[0, 2q)`:
   - `t = lo * q_inv_32  (mod 2^32)`   — **low 32 bits**
   - `t_low = t & 0xFFFF`, `t_high = t >> 16`
   - `pHH=t_high*q_high`, `pMH=t_high*q_low`, `pML=t_low*q_high`, `pLL=t_low*q_low`
     (each 16×16, exact `< 2^32`)
   - `mid_low  = (pMH & 0xFFFF) + (pML & 0xFFFF) + (pLL >> 16)`
   - `mid_high = (pMH >> 16) + (pML >> 16) + (mid_low >> 16)`
   - `t_final  = pHH + mid_high`              — this is `floor(t*q / 2^32)`
   - `b = hi + q - t_final`                   — in `[0, 2q)`
3. `finalize(b) = b >= q ? b - q : b`         — optional, gives canonical `[0, q)`.

Operation inventory per element: a handful of 16×16→32 multiplies + one 32×32→32
(`t`), plus masks, shifts, adds/subs, and one compare/select. **Every one of these
maps to an available SFPI op.** Subtraction uses `SFPIADD` with a negated operand /
two's complement; the `>= q` select uses `v_if`.

Because `q` and its derived constants are **compile-time**, all the "× constant"
multiplies (`t_*·q_*`, `lo·q_inv_32`) become (variable limb)×(constant limb) fp32
`SFPMAD`s with the constant pre-cast — cheaper than a general int×int.

---

## 3. Data layout

- Element type: **`uint32`**, `DataFormat::UInt32`, 4 bytes/elem.
- Tile = 32×32 = **1024 elements** = **4096 bytes** (page size = tile size).
- `N = 4096` → **4 tiles** per input vector. Two input vectors A, B → one output C.
- Host buffers: 3 DRAM interleaved `MeshBuffer`s (A, B, C), page_size = 4096 B.
- CBs (L1): `c_0` (A), `c_1` (B), `c_16` (out), each `DataFormat::UInt32`,
  double-buffered (2 tiles) for reader/compute overlap.
- Inputs must already be in **Montgomery form** (`to_montgomery` on host, as in
  `montgomery.hpp`); output is Montgomery-form product. Conversion to/from normal
  domain is host-side (matches the CPU test harness).

---

## 4. Implementation strategy — two phases

### Phase 1 — Correctness via building-block `mul_int_tile` (compose, don't fuse)

Implement `mulmod` by composing existing, already-validated LLK int-multiplies plus
SFPI glue. This de-risks correctness before optimizing.

- Use `mul_int_tile<DataFormat::UInt32>(idst0, idst1, odst)` for the low-32 products
  (`t = lo*q_inv_32`, and the 16×16 limb products with masked operands, which are
  `< 2^32` so the low-32 result is exact).
- Constant operands (`q_inv_32`, `q_low`, `q_high`, `q`) provided as constant-filled
  tiles, or via a small SFPI helper that materializes them into DST.
- Masks/shifts/adds/subtracts and the high-half assembly: a small **custom SFPI
  compute kernel** (model on `programming_examples/custom_sfpi_add/kernels/shared/compute/`
  and `ttnn/.../ternary_addcmul_int_sfpu.cpp` for the `mul_int_tile` call pattern +
  `mul_int_tile_init`).
- Validate against the CPU `mulmod` bit-for-bit (host cross-check, see §6).

Cost: several DST round-trips and full 32-bit multiplies where a 16×16 would do.
Correct but not optimal — establishes the reference.

### Phase 2 — Fused custom SFPI Montgomery kernel (the high-performance target)

Write **one** custom SFPU function that does the entire `mulmod` per 32-lane vector
in registers, with **no intermediate DST/CB round-trips**, reusing the limb+fp32+
`2^23` trick directly (model implementation: `ckernel_sfpu_mul_int32.h`). Fusion wins:

- The constants `q_*`/`q_inv_32` limbs are pre-cast to fp32 **once** in an `_init_`
  (cf. `mul_int32_init` programming `vConstFloatPrgm*`), not per-element.
- `mul_hi_lo` and `reduce` share intermediate limbs of `t`; computing `hi`/`lo` and
  immediately reducing avoids re-materializing the product.
- Many products are ×constant → single `SFPMAD` each, and `SFPMAD` fuses the
  `+2^23` so the integer-add to combine partials is often free (as the LLK comment notes).
- All 4 faces × 8 row-slots looped (`ITERATIONS=8`), `#pragma GCC unroll 8`.

Per-element target: ~1 low-32 multiply (`t`) + 4 (variable×const) 16-limb MADs for
the high half + the `a*b` limb product, all in fp32 MADs with mantissa extraction —
i.e. on the order of ~15–25 SFPU instructions per element, fully vectorized over
32 lanes, no reloads.

---

## 5. Files to create

```
tt-he/reference/cpu/montgomery.hpp                 # (exists) golden reference
tt-he/device/
  vector_modmul.cpp                          # host program (single + multi-core)
  kernels/
    dataflow/reader_modmul.cpp               # DRAM->CB, A and B, n_tiles loop
    dataflow/writer_modmul.cpp               # CB->DRAM, C, n_tiles loop
    compute/modmul_compute.cpp               # Phase 1: mul_int_tile + SFPI glue
    compute/modmul_sfpi.cpp                   # Phase 2: fused custom SFPI kernel
  CMakeLists.txt                             # find_package(TT-Metalium); OVERRIDE_KERNEL_PREFIX
```

Host skeleton (mirrors `scratch/add-2-int` / `scratch/multicore_add`):
`MeshDevice::create_unit_mesh` → `CreateProgram` → 3 DRAM `MeshBuffer`s →
3 `CircularBuffer`s (UInt32) → `CreateKernel` reader(RISCV_1)/writer(RISCV_0)/
compute → `SetRuntimeArgs`(addresses, n_tiles, per-core start_tile) →
`EnqueueWriteMeshBuffer` A,B → `EnqueueMeshWorkload` → `Finish` →
`EnqueueReadMeshBuffer` C → `close`.

Reader/writer = standard `InterleavedAddrGenFast<true>` + `noc_async_read_tile`/
`noc_async_write_tile` with `cb_reserve_back`/`cb_push_back` (resp. `cb_wait_front`/
`cb_pop_front`), `DataFormat::UInt32`, looping `start_tile .. start_tile+n_tiles`.

Multi-core: `split_work_to_cores` over the 4 tiles (e.g. 4 cores × 1 tile, or scale
to the full grid for larger N later); same kernels, per-core `(start_tile, n_tiles)`.

---

## 6. Validation

1. **Host golden**: generate random `x,y ∈ [0,q)`; `a=to_montgomery(x)`,
   `b=to_montgomery(y)` (optionally + q to exercise lazy `[0,2q)` operands, as in
   `reference/cpu/test_montgomery.cpp`). Expected device output (Montgomery form) ==
   `cpu` `mulmod(a,b)` exactly; and `from_montgomery(result) == (x*y) mod q`.
2. **Bit-exact** check over all 4096 elements for several 28-bit primes
   (`167772161`, `201326593`, `251658241` — reuse the CPU test's primes).
3. Edge tiles: operands at `0`, `q-1`, `2q-1`, and the `b >= q` finalize boundary.
4. Phase 2 output must equal Phase 1 output element-for-element.

Numeric-safety note: the LLK fp32 trick is **exact** only because every partial
product stays ≤ 23 bits (11-bit limbs) / ≤ 16 bits (8-bit limbs). When we hand-roll
limbs we must preserve that invariant — with `q < 2^28` and lazy operands `< 2^29`,
11-bit limbs (3 per 32-bit word) keep all partials `< 2^23`. Verify no fp rounding
creeps in by the bit-exact test above.

---

## 7. Performance — grounded in real ttnn / programming-example patterns

**The dominant truth for this op: it is SFPU-instruction-bound, not NoC-bound.**
The data is tiny (3 × 4 tiles × 4 KB = 48 KB total), but Montgomery modmul is tens of
SFPU instructions per element (every integer multiply is itself ~a dozen fp32
MAD/cast/exman ops — see `ckernel_sfpu_mul_int32.h`). So wall-clock ≈
(guide/element) × (elements/core) / SFPU-rate. The two levers that matter,
in priority order:

### Lever 1 (biggest) — minimize SFPU instructions per element
- **Fuse (Phase 2).** Composing `mul_int_tile` calls (Phase 1) pays a full
  generic 32×32 emulation + DST round-trip *per multiply*. The fused custom kernel
  keeps limbs of `t` in LREGs and reuses them, and folds the `+2^23` into the
  `SFPMAD` so the recombine-add is free (the LLK's own comment notes this:
  `ckernel_sfpu_mul_int.h`, "use of FMA means we can avoid an integer add later").
- **Constant-fold every ×q multiply.** `q`, `q_low`, `q_high`, `q_inv_32` are
  compile-time. Pass them as `get_compile_time_arg_val` / kernel defines (pattern:
  `binary_ng_no_bcast.cpp:15` `constexpr ... = get_compile_time_arg_val(0)`;
  accumulation/ternary pass `DATA_FORMAT` as a template/define). Pre-cast their
  limbs to fp32 **once** in an `_init_` (exactly like `mul_int32_init` programming
  `vConstFloatPrgm2 = 2^23`). Then `t_high*q_high` etc. are single (variable
  limb)×(constant limb) `SFPMAD`s — no generic int-mul machinery.
- **Use the cheapest limb width.** The four `t_*·q_*` products have 16-bit
  operands → 8-bit/2-limb path (`_mul_int_`, ~12 cyc/row) not the 11-bit/3-limb
  32-bit path. Only `t = lo*q_inv_32` and the `a*b` product genuinely need the
  wider decomposition. Don't call `mul_int_tile<UInt32>` where a 16×16 suffices.

### Lever 2 — parallelize across cores, but know the ceiling
- Split with `tt::tt_metal::split_work_to_cores(grid, n_tiles, row_major)`
  (`binary_ng_program_factory.cpp:1018`, `vecadd_multi_core.cpp:136-162`); it returns
  `core_group_1/2` with different per-core tile counts to absorb remainders. Set
  per-core `(start_tile, n_tiles)` runtime args in the standard two-group loop.
- **Honest ceiling: N=4096 = only 4 tiles → at most 4 cores at 1 tile each**, leaving
  ~60 cores of an 8×8 Wormhole grid idle. For a compute-bound op this is the real
  limiter, *not* DRAM. So:
  - To **measure/showcase throughput**, parameterize N and let it scale across the
    full grid — the op is embarrassingly parallel, perf should scale ~linearly with
    cores until grid-saturated. Treat N=4096/4-tiles as the correctness target and a
    *latency* (not throughput) point.
  - A tile is the indivisible work unit for element-wise; you cannot usefully split
    one tile across cores. 4 tiles ⇒ 4 cores is the cap at this N.

### Lever 3 — pipelining & buffering (modest here, do it anyway)
- Double-buffer CBs (depth 2): reader fills tile N+1 while compute does tile N
  (`vecadd_multi_core.cpp:143`). Going deeper than 2 buys nothing at 1–2 tiles/core.
- Reader on `RISCV_0`/`NOC::RISCV_0_default`, writer on `RISCV_1`/`NOC::RISCV_1_default`
  — separate RISCs + NoCs so read/compute/write overlap (`vecadd_multi_core.cpp:195-223`).
- Per-`acquire/commit` block: amortize `tile_regs_acquire` over multiple tiles
  (`binary_ng_no_bcast.cpp:34-62` loops `n` tiles inside one acquire). **But** int32
  + required `fp32_dest_acc_en` cuts DST to **2 tiles/operand** (`mul_int_sfpu.h` doc;
  `binary_ng_program_factory.cpp:717` forces `fp32_dest_acc_en` for UInt32/Int32) —
  so block size is 1–2 tiles for our op. Production int paths use
  `num_tiles_per_cycle = 1` (`ternary_addcmul_int_sfpu.cpp:16`); start there.

### Sharding — small win here, but cheap; reconsider if chained
- `set_globally_allocated_address(*buf->get_backing_buffer())` binds a CB straight to
  an L1-resident shard, so the reader becomes a no-op (`cb_reserve_back`+`cb_push_back`
  with data already present; **must not** `cb_wait_front`) — see
  `vecadd_sharding/kernels/add_sharding.cpp` and `vecadd_sharding.cpp:41-70`.
- For *this* op the DRAM round-trip (48 KB) is noise next to compute, so sharding's
  direct benefit is small. It becomes worthwhile only if this modmul is **one stage in
  a chain** (e.g. NTT butterflies) where keeping operands L1-resident across stages
  avoids repeated DRAM traffic — likely the real downstream use. Flag for later.

### Knobs that do NOT matter for us (avoid the trap)
- `math_fidelity` / `math_approx_mode` / HiFi-vs-LoFi and any "pack uint32 into bf16"
  scheme are **irrelevant** — those concern the FPU float path. We use the integer
  SFPU `mul_int` path, which is exact and unaffected by fidelity. `fp32_dest_acc_en`
  is the one config flag that *is* mandatory (for int32 DST correctness).

### Measuring
- Quick: host wall-clock around `EnqueueMeshWorkload` + `Finish(cq)`
  (`profiler/test_custom_cycle_count`). For per-kernel device cycles, build with the
  profiler and call `ReadMeshDeviceProfilerResults(*mesh_device)`. Report both
  latency (4-tile case) and throughput (scaled-N case); confirm scaling vs core count.

---

## 8. Risks / open items

- **High-half correctness** is the crux: `mul_int32` gives only low 32 bits, so the
  `t_final` high-half assembly must be hand-built (Phase 2) or composed from masked
  16×16 products (Phase 1). Phase 1 first precisely to pin correctness.
- **Signed vs unsigned** in SFPI: use `vUInt` and logical shifts; confirm `SFPIADD`
  borrow/`>=` semantics for the `b - q` finalize via the boundary tests.
- **`mul_int_tile` DST operand layout** (`dst_tile_size`, `idst` indexing) — confirm
  against `ternary_addcmul_int_sfpu.cpp` usage before wiring Phase 1.
- Constant materialization in-kernel (filling a DST tile with `q_inv_32` etc.) vs.
  passing constant tiles from host — decide in Phase 1; Phase 2 bakes them in `_init_`.

---

## Status — IMPLEMENTED & VALIDATED (on n150 Wormhole B0)

Implemented in `tt-he/device/` (host `vector_modmul.cpp`, kernels under
`kernels/`). Bit-exact vs the `montgomery.hpp` golden for all elements, swept
across 28-bit primes (167772161, 201326593, 251658241, 268435399, 268435455, …).

Approach taken: **building-block kernel** composing proven Wormhole int SFPU
tile-ops (`mul_int_tile` 32×32→lo32, `add/sub_int_tile`, `binary_logical_right_shift`,
`binary_left_shift`, `bitwise_or_binary`, `binary_min_uint32`, `copy_dest_values`,
`fill_tile_int`) — chosen over a hand-rolled SFPI kernel because every op is
individually validated and it fit the DST budget. Config: `fp32_dest_acc_en=true`
+ `dst_full_sync_en=true` ⇒ 8 int32 DST tiles; the kernel schedule peaks at exactly 8.

Key gotchas hit & resolved (see [[wormhole-sfpu-int-mul]]):
- Unary `right_shift_tile` is **arithmetic** (sign-extends) → used
  `binary_logical_right_shift_tile` with a shift-amount tile (=16).
- Unary `bitwise_and` header has an ambiguous-overload compile bug → computed
  `x & 0xFFFF` as `(x<<16)>>16` (logical), avoiding both the bug and an extra DST tile.
- Branchless finalize: `min_uint32(b, b−q)` maps lazy `[0,2q)` → canonical `[0,q)`.

Measured (50-iter steady-state, full modmul):

| N | tiles | cores | µs/iter | throughput |
|---|------:|------:|--------:|-----------:|
| 4,096 | 4 | 4 | ~38 | ~0.11 G/s (dispatch-bound) |
| 65,536 | 64 | 64 | ~47 | ~1.4 G/s |
| 1,048,576 | 1024 | 64 | ~511 | ~2.05 G/s |

Confirms §7: at the 4096/4-tile target only 4 of 64 cores are used and time is
host-dispatch-bound (a latency point); throughput scales to ~2 G modmul/s once the
grid is saturated and per-launch cost amortizes. Future perf: the fused custom-SFPI
kernel (Phase 2) to cut guide/element, and sharding when chained into NTT.

## Task 1 — Hybrid kernel: fuse all non-multiply ops into SFPI, keep multiply on the SFPU API

**Goal.** Cut the building-block kernel's ~75 SFPU tile-op passes per tile (measured:
18 `DUP` + 12 `SHR` + 11 `ADD` + 16 shift-from-`LOW16` + 10 `MUL` + 6 `FILL` + …)
down to ~10 multiplies + a handful of fused passes — **without** hand-rolling the
risky part. The split follows the hardware exactly:

- **Everything except multiply → one custom SFPI kernel, computed in registers.**
  SFPI has direct `vUInt` support for add, sub, `&`, `|`, `>>`/`<<` (logical), and
  `v_if`-select. So the masks, shifts, limb assembly, `hi + q − t_final`, and the
  `min(b, b−q)` finalize all become register arithmetic — **no DST round-trips, no
  per-op `*_init()`, and all 18 `DUP`s disappear** (duplicating an LREG is free).
- **Multiply → still the SFPU API** (`mul_int_tile<UInt32>` / `mul_int32`). SFPI has
  **no integer multiply**; rather than re-implement and debug the fp32-limb +`2^23`
  routine, reuse the already-validated LLK multiply. This keeps the one error-prone
  primitive on a proven path (directly addresses the correctness risk that motivated
  the building-block approach in the first place).

**Why this is the right next step.** The op is SFPU-instruction-bound, so
guide/element is the dominant lever (§7). This hybrid removes the bulk of the
overhead (DST traffic, DUPs, re-inits) while leaving the multiply — the part SFPI
can't do natively — untouched and trusted.

**Structure (dataflow between SFPI passes and `mul_int_tile` goes through DST tiles).**
A custom SFPI function is a multi-in/multi-out pass over DST tiles (wrapped via
`_llk_math_eltwise_*_sfpu_params_`, reading/writing `dst_reg[...]`), executing many
register ops per element in a single pass. Group the algorithm so each contiguous run
of non-multiply ops collapses into one SFPI pass:

1. **SFPI S1 — limb extract.** in: `a`,`b` → out: `aL,aH,bL,bH`. (replaces ~8 DUP/shift/mask ops)
2. **MUL ×4 (SFPU):** `ll=aL·bL`, `hh=aH·bH`, `aL·bH`, `aH·bL`.
3. **SFPI S2 — assemble lo/hi.** in: those 4 → `cross`, `mid`, `lo`, `hi`. (replaces ~12 ops)
4. **MUL ×1 (SFPU):** `t = lo·q_inv_32` (needs `q_inv_32` in a DST tile).
5. **SFPI S3 — split t.** in: `t` → `tL,tH`.
6. **MUL ×4 (SFPU):** `pHH,pMH,pML,pLL` (needs `q_low,q_high` DST tiles).
7. **SFPI S4 — reduce + finalize.** in: `pHH,pMH,pML,pLL,hi` → `mid_low`, `mid_high`,
   `t_final`, `b_out = hi+q−t_final`, `min(b_out, b_out−q)`. (replaces ~20 ops)

**Constants.** `q_inv_32`, `q_low`, `q_high` must be **DST tiles** (the LLK multiply
reads operands from DST) — `FILL` them once into dedicated tiles and don't overwrite
them. `q` (only used inside SFPI S4) can instead live in a `vConstIntPrgm` register set
at init from the runtime arg — no DST tile needed. SFPI freeing the working tiles makes
the constant tiles fit within the 8-tile budget.

**Op-count target.** ~9 `mul_int_tile` + 4 SFPI passes + 3 const `FILL`s ≈ **~16 SFPU
passes vs ~75** today; zero `DUP`s; re-init only at the SFPI↔multiply boundaries (the
multiply reprograms SFPU const regs, so re-establish SFPI/`vConst` state after each
`mul_int_tile` group).

**Risks / notes.**
- Use `vUInt` and logical `>>` inside SFPI (the `vInt` arithmetic shift would
  sign-extend intermediates with bit 31 set — same trap as the unary `right_shift_tile`).
- DST-tile bookkeeping across the mul boundaries must be explicit (which index holds
  what); document the allocation like the building-block kernel does.
- `_llk_math_eltwise_*_sfpu_params_` only iterates faces of a tile — the multi-tile
  inputs/outputs of each pass are addressed by `dst_index * 32` face-base offsets
  (cf. `custom_sfpi_add` and `ckernel_sfpu_mul_int32.h`).

**Validation.** New kernel `kernels/shared/compute/modmul_sfpi_hybrid.cpp`, selected via the
existing `MODMUL_KERNEL` env var. Must be **bit-exact vs the building-block kernel and
the CPU golden** element-for-element (`MODMUL_MODE=modmul`, swept primes), then report
the µs/iter and throughput delta at N≥64 tiles (grid-saturated regime).

### Task 1 — DONE ✅ (implemented & validated on n150)

`kernels/shared/compute/modmul_sfpi_hybrid.cpp`. 4 SFPI passes (`face_limbs`, `face_lohi`,
`face_split`, `face_reduce`) interleaved with 9 `mul_int_tile` calls; finalize is a
branchless `v_if(reinterpret<vInt>(b−q) < 0)` in-register. Bit-exact vs building-block
and CPU golden across all swept primes. ~6 DST tiles peak, no shift-amount/DUP tiles.

Implementation notes that proved out:
- Multi-tile addressing works: the `_llk_math_eltwise_binary_sfpu_params_` wrapper does
  `start_(0)` and forwards all tile indices, so a face fn reads/writes any tile via
  `dst_reg[idx*32 + i]`, `i<8`, and the wrapper iterates the 4 faces uniformly (RC mode).
  Pass `VectorMode::RC` explicitly when forwarding >3 indices.
- `vUInt` has native `&`(imm), `|`, `<<`/`>>` (logical), `+`, `-` — so the building-block
  kernel's two workarounds (broken unary AND, arithmetic unary shift) simply vanish.
- Multiply stays on `mul_int_tile` (SFPI has no int multiply) — the one risky primitive
  remains on the validated path; the hybrid was correct on the first run.

Measured speedup vs building-block (same harness, 50-iter):

| N | building-block | hybrid-SFPI | speedup |
|---:|---:|---:|---:|
| 4,096 | 37.9 µs / 0.11 G/s | 21.8 µs / 0.19 G/s | 1.74× |
| 65,536 | 46.9 µs / 1.40 G/s | 28.1 µs / 2.33 G/s | 1.67× |
| 1,048,576 | 511 µs / 2.05 G/s | 297 µs / **3.53 G/s** | 1.72× |

Next lever (Task 2, future): constant-fold the `×q` multiplies into the SFPI passes too
(fp32-limb MADs with `q_*` pre-cast at init) to drop the `mul_int_tile` round-trips for
the 16×16 products — pushes past the ~3.5 G/s the `mul_int_tile`-bound hybrid saturates at.

## 9. Milestones

1. **M0** — Host harness + reader/writer + identity/passthrough compute kernel;
   verify 4 tiles of UInt32 round-trip DRAM→L1→DRAM unchanged.
2. **M1** — Phase 1 compute (`mul_int_tile` + SFPI glue) for a single tile, single
   core; bit-exact vs CPU `mulmod` for one prime.
3. **M2** — All 4 tiles, all test primes, lazy operands; full validation suite.
4. **M3** — Phase 2 fused custom SFPI kernel; equals Phase 1 output; measure
   guide/element and wall-clock.
5. **M4** — Multi-core scaling via `split_work_to_cores`; (stretch) parameterize N.

---

## Appendix: key reference files

- CPU golden: `tt-he/reference/cpu/montgomery.hpp`, `tt-he/reference/cpu/test_montgomery.cpp`
- Int-mul compute API: `tt_metal/hw/inc/api/compute/mul_int_sfpu.h`
- WH int-mul impl (the fp32-limb trick to emulate): `hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h` (32→32), `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_mul_int.h` (16→16)
- `mul_int_tile` usage pattern: `ttnn/cpp/ttnn/operations/eltwise/ternary/kernels/shared/compute/ternary_addcmul_int_sfpu.cpp`
- Custom SFPI kernel skeleton: `tt_metal/programming_examples/custom_sfpi_add/kernels/shared/compute/`
- SFPI ops: `runtime/sfpi/include/sfpi.h`, `sfpi_lib.h` (casts: `int32_to_float`, `SFPEXMAN`, `SFPMAD`)
- Host + build + run patterns: `scratch/add-2-int/`, `scratch/multicore_add/`, `scratch/eltwise_binary/`
- Build/run env (from memory + `docs/manual-install-guide.md`): `find_package(TT-Metalium)` + link `TT::Metalium`; `OVERRIDE_KERNEL_PREFIX` bakes kernel dir; export `TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT=/usr/local/libexec/tt-metalium`, `CMAKE_PREFIX_PATH=/usr/local`, `LD_LIBRARY_PATH=/usr/local/lib`; build `cmake -B build -G Ninja && cmake --build build`.
```
