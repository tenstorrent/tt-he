# Optimization — kernel side

**Status:** Current · One of three, split by where the time goes. Start at
[`optimization.md`](optimization.md) for the order of attack; measure with
[`profiling.md`](profiling.md) before and after.

| | |
|---|---|
| [`optimization_host.md`](optimization_host.md) | dispatch count, occupancy, batching, trace |
| [`optimization_dram.md`](optimization_dram.md) | read depth, bank spreading, NoC, the ceilings |
| [`optimization_kernel.md`](optimization_kernel.md) | SFPI passes, `sfpmul24`, DST traffic, registers |

---

**Diagnosis — and first, check the kernel is worth touching.** A whole-tile move costs ~**0.1 µs**, so
DST/CB traffic is rarely the dominant term; and see [`optimization.md`](optimization.md) on the 3.3× pipeline inflation. Per-kernel work
has poor leverage on a dispatch-bound workload.

### What actually pays

| lever | measured | where |
|---|---|---|
| **`sfpi::fractional_mul` (SFPMUL24) instead of the fp32 byte-split** — Blackhole has an integer multiply; a 16×16 product is 2 instructions, not ~21 | `modmul` 12.430 → **5.376 µs/tile (2.31×)** with *nothing else changed* | [`../he/history/exp_tg018s.md`](../he/history/exp_tg018s.md) |
| **Fuse SFPI passes** — collapse ops whose only purpose is shuttling intermediates through DST | `modmul`: 12.432 → **3.086 µs/tile (4.03×)**, 14 passes → 1, DST traffic 608 → 96 per tile | [`../he/exp_modmul001.md`](../he/exp_modmul001.md) |
| **Fuse adjacent dispatches** that exist only to hand one tile over | `modadd_twiddle` (reduce+twiddle): removed a full 8-tile DRAM round trip; 1219 → 1130 dispatches | [`../he/blackhole_port.md`](../he/blackhole_port.md) |
| **One barrier, not one per tile** — independent reads should not serialize on DRAM latency | `ph_matmul` **6.82 → 1.25 µs (−82%)**; `fused_matmul_a` 16.44 → 11.45 µs. The reader had been pacing the matmul at ~2 GB/s | same |
| **Keep results in DST across phases** — `matmul_tiles`' last argument is the DST index; it was hardcoded to 0 | copy_tile 5 → 1, acquire/release 5 → 2; −0.69 ms | [`../he/exp_tg018m.md`](../he/exp_tg018m.md) |
| **Scalar broadcast instead of tile fills** — `vUInt(x)`, not `fill_tile_int` | removes 21% of DST accesses | [`../he/exp_modmul001.md`](../he/exp_modmul001.md) |
| **Constants into L1** when shared across a dispatch and re-read | 95 KB/core (6.6% of L1): matmuls 17.2 → 16.3 µs, **5%** | [`../he/dram_guide.md`](../he/dram_guide.md) §3.6 |
| **Use the matrix engine** where an op is pure compute | FBC via the matrix-engine route: **1.98×** at this exact shape — *built and measured, never ported* | [`../he/history/exp_tg018r.md`](../he/history/exp_tg018r.md) |

Cumulative on the large parameter set: **238.663 → 50.920 ms (4.69×)**, `0/6029312` throughout.

**Rows 1 and 2 are not independent, and the order matters.** The SFPU has **8 general LREGs**
(`p_sfpu::LREG0..7`). The fp32 byte-split needs 8 registers per product, so `modmul_sfpi_full`'s own
comment records the constraint: *"One product per pass keeps SFPI register pressure low enough to
compile."* `fractional_mul` needs **2**. The fusion is therefore only expressible *after* the
multiply is swapped — the control kernel with the fp32 multiply restored does not compile:
`internal compiler error: cannot store sfpu register (register spill)`. Ready-to-use kernels and the
three techniques: [`../../kernels/OPTIMISED.md`](../../kernels/OPTIMISED.md).

**`SFPMUL24` semantics, measured over 532 probed operand pairs** — an exact **23×23 → 46-bit**
product, `Low` = bits [22:0], `High` = bits [45:23], and **both operands silently masked to 23 bits**
(`a = 2^23, b = 1` returns 0). Not 24 despite the name. `mul_int_tile` wraps the same instruction but
its LLK only ever issues `MOD1_LOWER`, so the upper half is unreachable through the tile-op API.

### The floors

* **One SFPI pass, 2 loads + 1 store** is the floor for a two-operand elementwise op, and `modmul` is
  there: ~2 cycles/op, near the SFPU issue rate — **arithmetic-bound, not DST-bound**.
* **6 tile moves/dispatch** is the floor for the NTT matmul shape: `matmul_tiles` reads its operands
  through the **unpacker, not DST**, so the decompose must pack its limbs to a CB.
* **Cores are stateless; every stage boundary is DRAM.** Nothing is handed core-to-core. This is why
  changing the tile→core mapping is free — and it is the structural ceiling: of the multiply's ~5.7 GB,
  most exists only because stages talk through memory. Removing it means stateful cores, which
  collides with the workload cache and trace (both assume a dispatch is a pure function of its runtime
  args). A next real ceiling, not a tuning knob.

### Config

* **`fp32_dest_acc_en = true` is mandatory** for int32 results.
* **The float-fidelity knobs are irrelevant on the integer SFPU path** — and **`MathFidelity::LoFi` on
  the NTT matmuls is REFUTED**: not exact, `6029312/6029312` mismatches. Its 1.1% delta also showed
  `matmul_tiles` is only ~4% of those kernels.

### Three lessons that generalise

1. **The biggest win came from a stale comment, not a new instruction.** *"One product per pass keeps
   register pressure low enough to compile"* was true of the old fp32 code and false after the
   multiply changed. The new instruction's value was dropping 8 registers per product to 2 — which is
   what made the fusion **compile at all**.
2. **An optimisation's value is set by the bottleneck, not by what it removes.** The same change that
   deletes 21% of DST accesses is worth 2.9% on an arithmetic-bound kernel and materially more on a
   DST-bound one.
3. **Measure against the best available option, not the one that is running.** A 2.31× headline
   evaporated when a faster kernel already in the tree was found.

### Occupancy

A dispatch takes `min(G, jobs remaining)` cores, so **a stage whose job list is shorter than G
silently runs on part of the grid** — that reasoning found the two biggest wins here (the FBC on 64 of
130 cores; the corner turn likewise). But it is a way to *think*, not a formula to *solve*: filling
the tightest stage is only half the problem, because the same parameter decides how every other stage
chunks, and the value the formula gives measured **worst** of the four tried (78.091 ms vs 69.374 at
the measured optimum). **Occupancy reasoning finds bugs; it does not pick constants.**

---
