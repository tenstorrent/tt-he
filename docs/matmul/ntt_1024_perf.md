# NTT-1024 — performance

**Status:** Current — the single owner of NTT-1024 measurements.
*Consolidated from `NTT_1024_PERF.md`, `NTT_1024_ONDEVICE_PERF.md` and the multi-core results in
`NTT_1024_MULTICORE.md` (2026-08). Those originals are kept verbatim in [`history/`](history/) — each
was correct for the code at the time, and their per-version analysis is why the design is what it is.*

Design and algorithm: [`ntt_1024.md`](ntt_1024.md).

> **Hardware caveat.** Every number here was measured on **Wormhole n150**. The current hardware is
> Blackhole p150, where the standalone NTT-8192 runs about 2× faster. These are not comparable to
> p150 results — see [`../he/blackhole_port.md`](../he/blackhole_port.md) §3.

All versions are bit-exact (`0/1024`) against the direct O(N²) NTT golden, multiple seeds,
`q = 167772161 = 5·2²⁵+1`, single Tensix core unless stated.

## The progression

| version | µs / NTT | vs previous | what changed |
|---|--:|--:|---|
| host-orchestrated (`ntt1024.cpp`) | **1167** | — | 3 kernels glued on the host, blocking read-back between stages |
| pure on-device | 465 | 2.5× | intermediates stay in DRAM; no host arithmetic in the transform |
| single-kernel recombine | 287 | 1.6× | one Montgomery reduction instead of 4 modmuls + modadd |
| radix fold baked into `W` | **147** | 2.0× | the fold was 4 runtime modmuls on `Xin` (127 µs); `W` is constant, so it moved offline |
| + SFPI recombine | 112 | 1.31× | reduction rewritten as SFPI register passes |
| **+ fused matmul stages (A, C)** | **95.5** | 1.17× | decompose+matmul+recombine in one kernel per stage, L1-resident |

**~12× overall**, entirely by removing host work and modular-reduction cost — never by touching the
matmul.

## Where the time goes now (95.5 µs, 3 chained programs)

| stage | µs | what runs |
|---|--:|---|
| A — fused matmul | 29.4 | decompose → int8 matmul(`WfoldA`) → recombine |
| B — twiddle modmul | 36.0 | `G' = G ⊙ T`, still the building-block `modmul_compute.cpp` |
| C — fused matmul | 30.4 | decompose → int8 matmul(`Wrhs`) → recombine |

**The modular reduction dominates, not the matmul** — the finding that drove every optimization
above. From the 147 µs version, where stages were separately measurable:

| operation | µs | engine |
|---|--:|---|
| Montgomery reduction (recombine A, recombine C, twiddle B) | ~37 each, **~112 total ≈ 75%** | SFPU, ~50 int32 tile-ops |
| int8 matmul (×2) | ~14 each | matrix engine — the "heavy" O(n³) part, and it is the cheap one |
| decompose (×2) | ~14 each | mask & shift |

The matrix engine is far faster per useful operation than the SFPU integer path, so the transform is
bound by `mod q`, not by multiply-accumulate. **Next lever:** stage B still uses the slow
building-block modmul; swapping in the verified `modmul_sfpi_hybrid.cpp` (drop-in, ~1.7×) would take
B from ~36 → ~21 µs, i.e. the NTT to **~80 µs**.

## Multi-core matmul variant (`ntt1024_dev_mc.cpp`)

Splitting the plane-matmul across 4 cores, measured at the 147 µs stage:

| | single core | 4 cores |
|---|--:|--:|
| A2 matmul | 14.2 | **12.6** |
| C2 matmul | 14.2 | **12.1** |
| **NTT total** | **147** | **135** (~8%) |

Small, and consistent with the finding above: at single-tile sizes a matmul stage is dominated by
dispatch + CB/NOC overhead, and the three reductions (~75%) remain single-core. The layout matters for
larger M/N/K, where the matmul does dominate.

## Why the original was 1167 µs (the host-bound baseline)

Worth keeping because it shows what "device compute" was never the problem:

| component | where | µs / NTT | share |
|---|---|--:|--:|
| operand prep (radix-128 decompose + tilize, rebuilt every call — including constant `W`) | host | 686 | ~59% |
| recombine + `mod q` (`128ʳ` fold) | host | 131 | ~11% |
| buffer writes + 3 blocking reads + dispatch | host↔device | ~230 | ~20% |
| **device compute** (2 matmul + modmul) | core {0,0} | **66** | **~6%** |

Device compute was ~63–66 µs even then; the other ~1100 µs was host glue and blocking reads that
prevented host/device overlap. Everything since has been removing that, then attacking the reductions.

## Methodology

Host wall-clock (`std::chrono`), steady state: a warm-up enqueue + `Finish` to absorb JIT, then
100–200 back-to-back `EnqueueMeshWorkload` bracketed by one `Finish`. This **includes dispatch
overhead** — significant at single-tile sizes — so it overstates pure kernel time. Batch = 1, forward
NTT only. For true per-RISC cycles, rerun with `TT_METAL_DEVICE_PROFILER=1`.

## Reproduce

```bash
. env.sh && cmake -B build -G Ninja
cmake --build build --target ntt1024_dev
./build/ntt1024_dev [q] [seed]        # prints total + per-stage timing
./build/ntt1024_dev_mc [q] [seed]     # 4-core matmul variant
./build/ntt1024 [q] [seed]            # the original host-orchestrated version
```
