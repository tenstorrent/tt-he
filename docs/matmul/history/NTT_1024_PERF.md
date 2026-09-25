# Performance report: size-1024 NTT on one Tensix core

> **Status: Historical.** Superseded by [`../ntt_1024_perf.md`](../ntt_1024_perf.md).
> Measures the original host-orchestrated `ntt1024.cpp` (1167 µs). Kept for its host-cost attribution, which is why the design moved on-device.
> Numbers are a snapshot of the code at the time and were measured on Wormhole n150.

**Program:** `tests/ntt/ntt1024.cpp`  ·  **Kernels:** `kernels/{compute/matmul_compute.cpp, compute/modmul_compute.cpp, dataflow/*_matmul.cpp, dataflow/*_modmul.cpp}`
**Device:** Wormhole (n150), single Tensix core `{0,0}`, all three compute stages.
**Field:** `q = 167772161 = 5·2²⁵+1` (NTT-friendly 28-bit prime).
**Status:** correct — `mismatches = 0/1024` vs the direct `O(N²)` NTT golden, across seeds 0,1,2,3,42.

## What it computes

A 1024-point forward NTT factored as `N = 32·32` (Cooley–Tukey four-step), built entirely
from the repo's two basic kernels plus free host reshapes:

| stage | op | kernel | on core {0,0} |
|---|---|---|---|
| A | `G  = W @ Xin`  (32×32 modular matmul) | `matmul_compute.cpp` (int8) | ✓ |
| B | `G' = G ⊙ T`    (1024-elem twiddle)     | `modmul_compute.cpp` (Montgomery) | ✓ |
| C | `H  = G' @ W`   (32×32 modular matmul) | `matmul_compute.cpp` (int8) | ✓ |
| — | `out[32·j1+j2] = H[j2,j1]` (output reindex) | host (free; 2-factor ⇒ no transpose) | — |

Twiddle trick: the Montgomery kernel returns `REDC(a·b)=a·b·R⁻¹`. Feeding `G` in the normal
domain and `T` pre-converted to Montgomery (`T·R`) yields `G·T mod q` in the normal domain,
so no domain conversion of the live data is needed between stages.

## Measured timing (single core {0,0})

Host wall-clock, `std::chrono`. "Device only" = steady-state mean over 200 back-to-back
`EnqueueMeshWorkload` calls bracketed by one `Finish` (after a warm-up enqueue+Finish to
absorb JIT/first-dispatch); it includes dispatch overhead and kernel execution but **excludes**
host data prep and the blocking read-back. End-to-end is the full single-NTT latency including
all host work. Two representative runs:

| quantity | per call | calls / NTT | run 1 | run 2 |
|---|---|---:|---:|---:|
| matmul (device only) | one 32×32 modular matmul | 2 | 13.51 µs | 13.80 µs |
| modmul (device only) | one 1024-elem tile | 1 | 36.01 µs | 35.95 µs |
| recombine (host) | per matmul (128ʳ fold + mod q) | 2 | 66.38 µs | 65.50 µs |
| **device compute / NTT** | `2·matmul + modmul` | — | **63.03 µs** | **63.55 µs** |
| **end-to-end / NTT** | full pipeline incl. host prep+glue+reads | — | **1195 µs** | **1167 µs** |

## Analysis

**Device compute is ~63 µs/NTT.** If the pipeline were device-bound, that's ≈ **16k NTT/s**
on a single core. Breakdown of the on-device time: `2 × 13.7 = 27.4 µs` matmul + `36 µs` modmul.

**The modmul (one 1024-element tile, 36 µs) costs more than both matmuls combined (27 µs).**
That is expected: each int8 matmul is 16 hardware `matmul_tiles` calls on the fast FPU, whereas
the Montgomery modmul is a long straight-line SFPU sequence (~50 int32 tile-ops: limb splits,
4 partial products, carries, reduction, finalize) over the tile. The matrix engine is simply
much faster per useful operation than the SFPU integer path here.

**End-to-end is host-bound: ~1160 µs vs ~66 µs of device compute (~18×).** The end-to-end number
includes all host pre- and post-processing — and that overhead *is* essentially the whole cost.
Measured per-NTT attribution (instrumented in the program):

| component | where | µs/NTT | share |
|---|---|--:|--:|
| operand prep (radix-128 limb decompose + tilize) | host | 2 × 343 = 686 | ~59% |
| recombine + mod q (`128ʳ` fold) | host | 2 × 65 = 131 | ~11% |
| buffer writes + 3 blocking reads + dispatch | host↔device | ~230 | ~20% |
| device compute (2 matmul + modmul) | core {0,0} | 66 | ~6% |

The full host-orchestrated matmul call measures **~513 µs**, of which only **~15 µs** is device:
`513 = prep 343 + recombine 65 + device 15 + writes/read/dispatch ~89`.

- **Operand prep dominates (343 µs/matmul).** `build_lhs`/`build_rhs` decompose each 28-bit value
  into 4 radix-128 int8 limbs and tilize them (16 RHS + 4 LHS tiles, ~16k `% q` ops), on the CPU,
  **rebuilt every call** — including the constant `W`, which never changes.
- **Recombine + mod q: 131 µs/NTT**, the matmul's host-side `128ʳ` reduction (PLAN.md M5).
- **Transfer/dispatch serialization:** 6 `EnqueueWriteMeshBuffer` + 3 `EnqueueMeshWorkload` +
  3 **blocking** `EnqueueReadMeshBuffer`; each blocking read drains the queue, so host and device
  never overlap.

Both big host costs are artifacts of the "three separate programs glued on the host" design, not
intrinsic work — see optimizations 1–3 below.

## Bottlenecks & optimization opportunities (ranked)

1. **Keep data resident on-device; stop bouncing through the host.** The dominant cost is host
   glue + blocking reads between stages. Chaining the three kernels in one program (or at least
   avoiding the read-back + re-write of the intermediate `G`/`G'`) would remove most of the
   ~1100 µs gap and approach the ~63 µs device floor.
2. **Move recombine + `mod q` on-device (PLAN.md M5).** Removes the 131 µs/NTT host reduction and
   a host round-trip; also a prerequisite for (1) since the matmul output must be reduced before
   it feeds the next stage.
3. **Avoid re-decomposing constants.** `W` is a fixed constant used in both matmuls; its RHS limb
   tiles and the twiddle `T` can be built once and reused, not rebuilt every NTT.
4. **Speed up / amortize the modmul.** It is the single largest device cost. Options: batch more
   tiles per dispatch (the kernel already loops `n_tiles`), or fold the twiddle into a matmul
   where the algebra allows (PLAN §7 — not foldable for the inter-stage twiddle, but the
   negacyclic ψ weights are).
5. **Batch many NTTs.** The 32×32 matmul wastes rows when the batch is 1; batching independent
   transforms amortizes dispatch and fills the tile.

## Methodology caveats

- Numbers are **host wall-clock**, not device-profiler cycle counts. For exact per-RISC kernel
  cycles, rerun with `TT_METAL_DEVICE_PROFILER=1` (the installed lib is a profiler build);
  `vector_modmul.cpp` shows the `ReadMeshDeviceProfilerResults` hook.
- "Device only" includes host→device dispatch overhead (it is an enqueue-loop steady state), so
  it slightly overstates pure kernel time for these tiny single-tile workloads.
- Single core, single NTT, batch = 1. No multi-core scaling measured (out of scope — the ask was
  one Tensix core).
- Correctness is randomized (5 seeds) against the direct `O(N²)` definition; `intt(ntt(x))==x`
  round-trip is not exercised by this program (the inverse NTT isn't built yet).

## Reproduce

```
cd device && cmake -B build -G Ninja && ninja -C build ntt1024
./build/ntt1024 [q] [seed]      # defaults: 167772161 0
```
