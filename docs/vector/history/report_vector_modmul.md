# Report: Vector Modular Multiplication on Tenstorrent (Wormhole B0)

> **Status: Historical.** Consolidated into [`../modular_multiplication.md`](../modular_multiplication.md).
> The full results report; its headline numbers and bottleneck analysis are carried forward.
> Numbers are a snapshot, measured on Wormhole n150.

**Date:** 2026-06-02 · **Device:** n150 (Wormhole B0)
**Status:** ✅ Implemented, validated on real silicon, multi-core, timed.

---

## 1. Objective

Implement a high-performance element-wise **modular multiplication** of uint32
vectors on tt-metal, using the **same Montgomery algorithm** as the CPU reference
`tt-he/reference/cpu/montgomery.hpp`:

```
C[i] = (A[i] * B[i]) mod q,   i in [0, N)
```

- `N = 2^12 = 4096` (correctness target); parameterized to scale further.
- `q` = 28-bit prime modulus.
- `A`, `B` are **Montgomery-domain** residues in lazy form `[0, 2q)`; output is the
  canonical Montgomery product in `[0, q)`.

Deliverables live in `tt-he/`:
```
reference/cpu/montgomery.hpp            golden reference (CPU), + test_montgomery.cpp
host/tests/vector/vector_modmul.cpp      host program (multi-core, timing, golden check)
kernels/shared/compute/modmul_compute.cpp   full Montgomery modmul kernel
kernels/shared/compute/modmul_mullo.cpp     stage-1 bring-up kernel
kernels/shared/dataflow/{reader,writer}_modmul.cpp
device/CMakeLists.txt, device/README.md
docs/plan_vector_modmul.md    design + performance plan
docs/report_vector_modmul.md  this report
```

---

## 2. The central hardware finding

The Tensix **SFPU** (32-lane SIMD vector engine) is the right engine for an
element-wise op (the FPU/matrix engine does dot-products, not element-wise).

The pivotal question was whether the SFPU can multiply integers. The answer is
nuanced and shaped the entire design:

- There is **no integer multiply instruction** and **no `vInt`/`vUInt operator*`** in
  SFPI. The only true multiplier is **fp32** (24-bit mantissa). `fractional_mul`/
  `sfpmul24` is **Blackhole-only** (`#if __riscv_xtttensixbh`) — absent on this n150.
- BUT Wormhole B0 ships an **emulated** integer multiply in the LLK:
  `mul_int_tile<DataFormat::{Int32,UInt32}>` (impl `ckernel_sfpu_mul_int32.h`) and a
  16-bit variant. It splits each operand into small limbs (11-bit / 8-bit), casts
  limbs to fp32, accumulates partial products with `SFPMAD`, and reconstructs
  integers via the `x + 2^23` magic-constant mantissa-extraction trick.
- **Consequence for Montgomery:** `mul_int32` returns only the **low 32 bits** of a
  product. Montgomery reduction needs the **high 32 bits** of `t·q`
  (`t_final = floor(t·q / 2^32)`). That is exactly why the CPU "optimized" routine
  decomposes into 16-bit limbs and assembles the high half by hand — and that same
  decomposition is what we map onto the device.

This is recorded in memory (`wormhole-sfpu-int-mul`) for future integer-arithmetic work.

---

## 3. Algorithm mapping (CPU → device)

`mulmod(a, b)` with `a, b ∈ [0, 2q]`, `q < 2^28` (so `2q < 2^29`, `a·b < 2^58`):

1. **`mul_hi_lo`** — full product as two 32-bit words `(hi, lo)` via 16-bit limbs:
   `ll=aL·bL`, `cross=aL·bH+aH·bL`, `hh=aH·bH`, then
   `mid=(ll>>16)+(cross&0xFFFF)`, `lo=(ll&0xFFFF)|(mid<<16)`,
   `hi=hh+(cross>>16)+(mid>>16)`.
2. **`reduce_optimized`** — `t = lo·q_inv_32 (mod 2^32)`; four `t*q` limb products;
   assemble `t_final = floor(t·q/2^32)`; `b_out = hi + q − t_final ∈ [0, 2q)`.
3. **finalize** — canonical `[0, q)`.

Every operation maps to a **proven Wormhole int SFPU tile-op**:

| algorithm op | device tile-op |
|---|---|
| 16×16 / 32×32 → lo32 multiply | `mul_int_tile<UInt32>` |
| add / subtract | `add_int_tile`, `sub_int_tile` |
| `x >> 16` (logical) | `binary_logical_right_shift_tile` (+ shift-amount tile) |
| `x << 16` | `binary_left_shift_tile` |
| `x & 0xFFFF` | `(x<<16)>>16` (see §5) |
| bitwise OR | `bitwise_or_binary_tile` |
| DST→DST copy | `copy_dest_values<UInt32>` |
| materialize constant | `fill_tile_int<UInt32>` |
| finalize `min(b, b−q)` | `binary_min_uint32_tile` |

---

## 4. Design decisions

- **Building-block kernel over hand-rolled SFPI.** Both were viable. I chose to
  compose the validated LLK tile-ops (lower correctness risk: each op is individually
  tested by Tenstorrent) rather than re-derive the fp32-limb multiply in raw SFPI.
- **DST budget.** `DEST_REGISTER_FULL_SIZE_BYTES = 32 KB` ⇒ 16 tiles @16-bit ⇒ **8
  tiles @32-bit** only with `fp32_dest_acc_en=true` **and** `dst_full_sync_en=true`
  (otherwise 4). `fp32_dest_acc_en` is mandatory for int32 correctness anyway. The
  kernel's DST schedule was hand-allocated to peak at **exactly 8** tiles (d0–d6
  working + d7 holding the shift-amount constant 16).
- **Branchless finalize.** `finalize(b) = min_uint32(b, b−q)`: if `b≥q`, `b−q<b` wins;
  if `b<q`, `b−q` underflows to a huge uint32 and `b` wins. No predication needed.
- **Incremental on-device validation.** Brought up in stages — raw `mul_int_tile`
  first (validates uint32 datapath + multiply primitive end-to-end), then the full
  modmul — each checked bit-exact against the CPU golden via an env-selectable MODE.

---

## 5. Gotchas hit & resolved

1. **Unary `right_shift_tile` is arithmetic** (sign-extends). Several intermediates
   (`ll`, `t`, `pLL`, `pMH`, `pML`) can have bit 31 set, so an arithmetic shift would
   corrupt them. Fixed by using `binary_logical_right_shift_tile` with a tile filled
   with the shift amount (16).
2. **Unary `bitwise_and` header fails to compile** — ambiguous `operator&`
   (`vInt & uint`) in `ckernel_sfpu_bitwise_and.h`. Avoided entirely by computing
   `x & 0xFFFF` as `(x<<16)>>16` (logical shifts) — also saves a DST tile vs. a mask tile.
3. **Per-op init.** Each SFPU tile-op must call its `*_init()` before use, and re-init
   when switching op type within one `tile_regs_acquire` (the ops reprogram SFPU
   constant registers). Wrapped each op as `init();op();` helpers.
4. **Kernel entry point.** Compute kernels here use plain `void kernel_main()` (as the
   programming examples do), not the `NAMESPACE/MAIN` macro. Paths resolved via
   `OVERRIDE_KERNEL_PREFIX` baked by CMake.

---

## 6. Results

Bit-exact vs the `montgomery.hpp` golden for **all** elements, swept across 28-bit
moduli (167772161, 201326593, 251658241, 268435399, 268435455, 65537, …). The host
correctly rejects ≥2^28 moduli.

Steady-state timing (50 iters, host `EnqueueMeshWorkload`+`Finish`, full modmul):

| N | tiles | cores | µs/iter | throughput |
|---:|---:|---:|---:|---:|
| 4,096 | 4 | 4 | ~38 | ~0.11 G modmul/s |
| 65,536 | 64 | 64 | ~47 | ~1.4 G modmul/s |
| 1,048,576 | 1,024 | 64 | ~511 | **~2.05 G modmul/s** |

**Hybrid SFPI kernel (Task 1, `modmul_sfpi_hybrid.cpp`)** — folds all non-multiply ops
into 4 custom SFPI register passes, keeps multiply on `mul_int_tile`. Bit-exact, ~1.7×
faster everywhere:

| N | building-block | hybrid-SFPI | speedup |
|---:|---:|---:|---:|
| 4,096 | 37.9 µs / 0.11 G/s | 21.8 µs / 0.19 G/s | 1.74× |
| 65,536 | 46.9 µs / 1.40 G/s | 28.1 µs / 2.33 G/s | 1.67× |
| 1,048,576 | 511 µs / 2.05 G/s | 297 µs / **3.53 G/s** | 1.72× |

**Single-prime size sweep (2^12 … 2^18 elements), both kernels:**

| vector size | tiles | cores | building-block | hybrid-SFPI | speedup |
|---|--:|--:|---|---|--:|
| 2^12 = 4,096 | 4 | 4 | 37.4 µs / 0.11 G/s | 21.7 µs / 0.19 G/s | 1.72× |
| 2^14 = 16,384 | 16 | 16 | 39.7 µs / 0.41 G/s | 23.0 µs / 0.71 G/s | 1.73× |
| 2^16 = 65,536 | 64 | 64 | 46.8 µs / 1.40 G/s | 28.1 µs / 2.33 G/s | 1.67× |
| 2^18 = 262,144 | 256 | 64 | 146.8 µs / 1.79 G/s | 88.9 µs / 2.95 G/s | 1.65× |

Two regimes are visible. **2^12→2^16 (4→64 cores):** the work is spread over more cores
as it grows, so latency barely moves (37→47 µs building-block) — it is dominated by the
fixed ~per-launch dispatch overhead, and throughput scales ~linearly with core count
(0.11→1.40 G/s). **2^16→2^18 (grid saturated at 64 cores):** 4× the work now means 4
tiles/core instead of 1, so latency grows ~3× (compute-bound) and throughput climbs
toward the asymptote (~3.5 G/s hybrid at 1024 tiles). The hybrid SFPI kernel is
~1.65–1.73× faster across the whole range.

**Interpretation (matches the plan's §7 analysis):**
- The op is **SFPU-instruction-bound**, not data-bound (each modmul is tens of SFPU
  ops; the data is tiny).
- At the **N=4096 target there are only 4 tiles ⇒ at most 4 of 64 cores**; the ~38µs
  is dominated by fixed host-dispatch overhead. This is a **latency** point, not a
  throughput one — a hard ceiling at this N because a tile is the indivisible work unit.
- Throughput scales ~linearly with cores and **saturates near ~2 G modmul/s** once the
  grid is full and per-launch cost amortizes (N≥64 tiles).

---

## 7. How to build & run

```bash
cd tt-he/device
cmake -B build -G Ninja && cmake --build build
./build/vector_modmul                                   # default: building-block, N=4096
# fastest kernel (hybrid SFPI), at throughput scale (~3.5 G modmul/s):
MODMUL_KERNEL=compute/modmul_sfpi_hybrid.cpp MODMUL_NTILES=1024 ./build/vector_modmul
MODMUL_KERNEL=compute/modmul_mullo.cpp MODMUL_MODE=mullo ./build/vector_modmul   # stage-1 check
MODMUL_PRIME=201326593 ./build/vector_modmul
```
Requires tt-metalium installed + env (`TT_METAL_HOME`, `TT_METAL_RUNTIME_ROOT`,
`CMAKE_PREFIX_PATH`, `LD_LIBRARY_PATH`). CPU reference test:
`cd cpu && g++ -O2 -std=c++17 test_montgomery.cpp -o t && ./t`.

---

## 8. Task 1 — Hybrid SFPI kernel ✅ (implemented, validated, ~1.7× faster)

**File:** `kernels/shared/compute/modmul_sfpi_hybrid.cpp`, selectable via
`MODMUL_KERNEL=compute/modmul_sfpi_hybrid.cpp`.

**Idea.** The building-block kernel pays for ~75 SFPU tile-op passes per tile — most of
them tiny (18 `DUP` register-copies, per-op `*_init()`, masks/shifts each a full pass)
purely because a tile-op can't hold an intermediate in a register. The hybrid keeps the
split that matches the hardware:
- **Everything except multiply → custom SFPI passes, in registers.** SFPI has native
  `vUInt` add/sub/`&`(imm)/`|`/logical `<<`/`>>` and `v_if` select, so masks, shifts,
  limb assembly, `hi+q−t_final`, and finalize are register arithmetic — no DST
  round-trips, no DUPs, no per-op init, no shift-amount tile.
- **Multiply → still `mul_int_tile` (SFPU API).** SFPI has no integer multiply; reusing
  the validated LLK multiply keeps the one error-prone primitive on a proven path.

**Structure.** 4 SFPI face functions interleaved with 9 `mul_int_tile` calls, dataflow
through DST tiles, peak ≈6 DST tiles:
1. `face_limbs`: `a,b → aL,aH,bL,bH`
2. MUL×4: `ll, hh, aL·bH, aH·bL`
3. `face_lohi`: `→ lo, hi`
4. MUL×1: `t = lo·q_inv_32`; `face_split`: `t → tL,tH`
5. MUL×4: `pHH,pMH,pML,pLL`
6. `face_reduce`: mid_low/mid_high/t_final/`b=hi+q−t_final`, finalize
   `v_if(reinterpret<vInt>(b−q) < 0){ b } v_endif` (underflow ⇒ b<q ⇒ keep b).

**SFPI mechanics that made it work (verified in the LLK source):**
- `_llk_math_eltwise_binary_sfpu_params_(func, i0,i1,i2, VectorMode::RC, i3,...)` calls
  `_llk_math_eltwise_sfpu_start_(0)` (base = tile 0) and **forwards all indices** to the
  face fn. So the fn addresses any tile via `dst_reg[idx*32 + i]`, loops `i<8` (one
  16×16 face), and the wrapper iterates the 4 faces (RC mode) — applying the same base
  increment to every `dst_reg` access. Multi-in/multi-out works; pass `VectorMode::RC`
  explicitly when forwarding more than 3 indices.
- `vUInt` (unsigned) gives logical `>>` and `&`(imm) directly — so the building-block
  kernel's two workarounds (the ambiguous unary `bitwise_and`, the arithmetic unary
  `right_shift`) simply don't arise in SFPI.

**Outcome.** Bit-exact vs both the building-block kernel and the CPU golden across all
swept primes; correct on the first device run. Measured (50-iter steady-state):

| N | building-block | hybrid-SFPI | speedup |
|---:|---:|---:|---:|
| 4,096 | 37.9 µs / 0.11 G/s | 21.8 µs / 0.19 G/s | 1.74× |
| 65,536 | 46.9 µs / 1.40 G/s | 28.1 µs / 2.33 G/s | 1.67× |
| 1,048,576 | 511 µs / 2.05 G/s | 297 µs / **3.53 G/s** | 1.72× |

---

## 9. Next steps (performance)

1. **Task 2 — constant-fold the multiplies into SFPI too.** The hybrid is now bounded by
   the 9 `mul_int_tile` round-trips. Fold the `×q` products into the SFPI passes as
   fp32-limb `SFPMAD`s with `q_low/high/inv` **pre-cast to fp32 once in an `_init_`** (q
   is compile-time), and use the cheaper 8-bit/2-limb path for the four 16×16 `t·q`
   products. Removes the remaining DST round-trips — the main lever past ~3.5 G/s.
2. **Sharding** when this modmul becomes a stage in a chain (e.g. NTT butterflies): bind
   CBs to L1-resident shards (`set_globally_allocated_address`) to keep operands on-core
   across stages and avoid repeated DRAM traffic.
3. **Profiling** for per-kernel device cycles via `ReadMeshDeviceProfilerResults` to
   confirm the multiply-bound hypothesis and guide Task 2.

---

## 10. RNS / multi-modulus version ✅ (up to 16 vectors, each its own prime)

**Files:** host `tests/vector/vector_modmul_rns.cpp`, kernel
`kernels/shared/compute/modmul_rns_sfpi.cpp` (reuses the same reader/writer).
This is the residue-number-system layout used in HE: up to **16 vectors**, each up
to **2^16 elements**, **each with its own 28-bit prime**.

**Layout & modulus selection.** Data is **vector-major**: vector `v` occupies tiles
`[v*Lt, (v+1)*Lt)` where `Lt = tiles_per_vector`. The prime is constant *within* a
vector and changes *between* vectors, so the kernel picks the modulus per tile by
`vec = (start_tile + local_tile) / tiles_per_vector` and indexes per-vector constant
arrays passed as runtime args (`q[v]`, `q_inv_32[v]`, `q_low[v]`, `q_high[v]` for
`v < V`). It re-`FILL`s the four constants each tile; the SFPI compute body is
identical to the single-prime hybrid (§8).

**Why this shape.** Tiles within a vector share a prime, so the per-tile lookup is a
few scalar ops amortized over 1024 elements — measured throughput is unchanged from
the single-prime hybrid. Work is split across the grid by total tiles
(`V * Lt`, up to 16×64 = 1024 tiles = 2^20 elements); a core may span vectors, and the
per-tile `vec` lookup handles that transparently. Primes are generated host-side
(spread, trial-division-verified, all `< 2^28`).

**Validated on device** (bit-exact vs per-vector CPU golden):

| V | tiles/vec | N | result | throughput |
|--:|--:|--:|:--|--:|
| 16 | 64 | 1,048,576 | OK (16 primes) | 3,517 M/s |
| 1 | 64 | 65,536 | OK | 2,311 M/s |
| 4 | 64 | 262,144 | OK | 2,948 M/s |
| 8 | 16 | 131,072 | OK | 2,651 M/s |
| 16 | 1 | 16,384 | OK | 697 M/s |

The `V=16, tiles/vec=1` case is the stress test — every tile switches prime — and it
passes, confirming the per-tile modulus lookup + `FILL` is correct across moduli.

**Multi-modulus vs single-prime: no performance loss.** Apples-to-apples at identical
total work (1,048,576 elements = 1024 tiles, 64 cores), 3 runs each:

| workload | µs/iter | throughput |
|---|--:|--:|
| single-prime hybrid (1024 tiles) | 297.0 / 297.1 / 297.1 | ~3,530 M/s |
| RNS, 16 primes (16 vec × 64 tiles) | 297.8 / 298.2 / 298.0 | ~3,518 M/s |

Δ ≈ 0.3% (~0.9 µs) — within run-to-run noise. Multi-modulus is effectively free because:
- The per-tile overhead is only an integer divide (`tile_id / tiles_per_vector`) plus 4
  indexed `get_arg_val` reads to fetch that vector's constants — a handful of scalar ops
  **per tile, amortized over 1024 elements**, dominated by the SFPI passes + 9 multiplies.
- The compute body and the constant `FILL`s are **identical** to the single-prime hybrid
  (which also re-FILLs constants every tile), so there is no extra SFPU work.
- The larger runtime-args buffer (68 vs 5 args) is one-time program setup, not per-iter.

Performance tracks **total tile count**, not vector count: 16×64 tiles ≡ 1×1024 tiles in
speed. Small-N cases (e.g. V=16×1 = 16 tiles → 697 M/s) are dispatch-bound exactly as the
single-prime kernel is at 16 tiles — that fixed launch cost is identical for both, not an
RNS penalty. A per-tile prime switch is free; only a hypothetical *per-element* modulus
change (which RNS never needs) would cost anything.

Run:
```bash
./build/vector_modmul_rns                                   # 16 vectors x 2^16, 16 primes
MODMUL_NVEC=8 MODMUL_VEC_TILES=64 ./build/vector_modmul_rns  # 8 vectors x 2^16
```

---

## 11. Device profiling traces (single-prime, 2^12 … 2^18)

The installed library is a profiler build, so we capture real per-core/per-RISC device
timestamps. Enabled by `TT_METAL_DEVICE_PROFILER=1` + a `ReadMeshDeviceProfilerResults`
call in the host (guarded; `MODMUL_ITERS=1` for a clean single-execution trace). Two trace formats are archived under `device/generated/profiler/traces/`:
- **CSV** `trace_2p{12,14,16,18}.csv` — raw timestamps (columns: core_x, core_y, RISC,
  timer, cycles, …, zone, ZONE_START/END). Written directly by the profiler env var.
- **Tracy** `modmul_2p{12,14,16,18}.tracy` — open in the Tracy profiler GUI for the
  per-core/per-RISC zone timeline (1.7k–6.7k zones each). Produced via the separate
  Tracy capture step (env var alone only emits the CSV).

Reproduce the CSV:
```bash
TT_METAL_DEVICE_PROFILER=1 MODMUL_ITERS=1 MODMUL_NTILES=64 \
  MODMUL_KERNEL=compute/modmul_sfpi_hybrid.cpp ./build/vector_modmul
# -> $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
```
Reproduce the Tracy timeline (capture server + instrumented run):
```bash
CAP=$TT_METAL_HOME/../../tools/profiler/bin/capture-release   # or <tt-metal build>/tools/profiler/bin/capture-release
"$CAP" -o modmul_2p16.tracy -f -a 127.0.0.1 &     # writes .tracy on client disconnect
TT_METAL_DEVICE_PROFILER=1 MODMUL_ITERS=3 MODMUL_NTILES=64 \
  MODMUL_KERNEL=compute/modmul_sfpi_hybrid.cpp ./build/vector_modmul
```

**Per-RISC kernel duration** (median across cores, cycles @ 1 GHz = ns; hybrid kernel):

| size | tiles/core | TRISC_1 MATH (compute) | NCRISC reader | TRISC_0 unpack | TRISC_2 pack | BRISC writer* |
|---|--:|--:|--:|--:|--:|--:|
| 2^12 (4 tiles) | 1 | 18,700 (~18.7 µs) | 1,283 | 1,838 | 338 | 19,105 |
| 2^14 (16) | 1 | 19,643 | 2,230 | 2,796 | 336 | 20,161 |
| 2^16 (64) | 1 | 22,053 | 4,657 | 5,211 | 337 | 22,541 |
| 2^18 (256) | 4 | 76,292 | 27,690 | 59,408 | 42,519 | 76,713 |

\* BRISC (writer) zone ≈ end-to-end because it is pipeline-blocked on `cb_wait_front`
until compute produces tiles; it is not doing that much work itself.

**What the trace shows:**
- **MATH (TRISC_1) is the bottleneck at every size** — it is the SFPU modmul and dwarfs
  the reader and unpack/pack stages. This is direct device confirmation of the
  SFPU-instruction-bound thesis (the same conclusion §7 reached from first principles).
- **Linear in tiles/core:** 2^12→2^16 are all 1 tile/core, so MATH is ~flat (18.7k→22k
  cycles; the mild rise is NoC/DRAM contention as 4→64 cores light up). 2^18 is 4
  tiles/core → MATH ≈ 4× (~76k). So per-tile MATH cost ≈ 18–19k cycles, independent of N.
- **Unpack/pack grow at high tiles/core:** at 2^18, TRISC_0 (unpack, the `copy_tile`s) and
  TRISC_2 (pack) become non-trivial (59k / 42k) — a secondary target after the multiplies.
- Device MATH (~76k cyc = 76 µs at 2^18) tracks the host steady-state latency (88.9 µs)
  once dispatch overhead is added — the two measurements corroborate.

Focused 2^16 real-vs-theory component analysis lives in
`docs/report_modmul_hybrid_real_vs_theory.md`.

Caveat: profiling re-JITs kernels with markers (`-DPROFILE_KERNEL=1`), which perturbs
absolute cycles slightly and inflates wall-clock; use §6 for clean throughput, this
section for the relative per-stage breakdown.

---

## Appendix: key reference files
- CPU golden: `tt-he/reference/cpu/montgomery.hpp`
- WH int-mul emulation (the fp32-limb trick): `hw/ckernels/wormhole_b0/.../ckernel_sfpu_mul_int32.h`
- int tile-op APIs: `tt_metal/hw/inc/api/compute/{mul_int_sfpu,add_int_sfpu,sub_int_sfpu,binary_shift,binary_bitwise_sfpu,binary_max_min,copy_dest_values}.h`
- `mul_int_tile` usage pattern: `ttnn/.../eltwise/ternary/kernels/shared/compute/ternary_addcmul_int_sfpu.cpp`
- multi-core split: `tt_metal/programming_examples/vecadd_multi_core/vecadd_multi_core.cpp`
- DST size constant: `tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/tensix_types.h`
