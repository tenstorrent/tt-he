# NTT-32768 — performance

**Status:** Current — the single owner of NTT-32768 measurements.
*Consolidated from `NTT_32768_OPTIMIZED.md` and `NTT_32768_PERF.md` (2026-08); both kept verbatim in
[`history/`](history/), along with the superseded `NTT_32768_COMPARISON.md` whose conclusion these
numbers reversed.*

Design: [`ntt_32768.md`](ntt_32768.md).

> **Hardware caveat.** Measured on **Wormhole n150**, host wall-clock, 20 iterations, one `Finish`.
> The current hardware is Blackhole p150 — not comparable, see
> [`../he/blackhole_port.md`](../he/blackhole_port.md) §3.
>
> **The p150 re-measurement is [`exp_ntt001.md`](exp_ntt001.md), and it changes the conclusion below.**
> On pure kernel time (trace-derived critical-path busy) the 3-factor leads 121 vs 146 µs — but on
> host wall, the metric this document uses, the **2-factor** leads 153.6 vs 184.0 µs, because the
> 3-factor issues 12 dispatches to the 2-factor's 7 at ~8.3 µs of launch each. The §4 head-to-head
> below is host wall-clock and therefore carries that bias; it ranks the pipelines as scheduled, not
> the factorizations as algorithms.

**Headline: 3-factor 319.5 µs, 2-factor 334.7 µs at C=32**, both `0/32768` bit-exact at every core
count. Optimization reversed the earlier conclusion — see §3.

## 2. What was optimized

The same three kernel-level wins established on NTT-1024, applied here:

| optimization | what it does | applied to |
|---|---|---|
| **SFPI recombine** | Montgomery `mod q` reduction in fused register passes (no DST round-trips) instead of ~50 building-block tile-ops | 3-factor (`recombine_compute.cpp`) |
| **SFPI bigK recombine** | same, large-K variant (5 conditional subtracts for `low < ~4q`) | 2-factor (`recombine_bigK.cpp`) |
| **hybrid SFPI twiddle** | drop-in `modmul_compute.cpp → modmul_sfpi_hybrid.cpp` (~75 → ~16 passes) | both (twiddle stages) |

Not yet applied: **fusing** decompose+matmul+recombine into one program (done for NTT-1024; both
32768 variants still run those as separate chained stages), and multi-coring the 3-factor's
all-to-all input staging.

## 3. Before → after (C=32, µs / NTT-32768)

| variant | baseline | optimized | speedup | what moved |
|---|---:|---:|---:|---|
| 3-factor | 407 | **319.5** | 1.27× | recombine 143→96 (SFPI), twiddle 87→55 (hybrid) |
| 2-factor | 386 | **334.7** | 1.15× | recombine 98→65 (SFPI bigK), twiddle 43→26 (hybrid) |

The 3-factor gained more because it has **more reduction stages** (3 recombines + 2 twiddles vs the
2-factor's 2 + 1) — and reduction is exactly what these optimizations attack.

## 4. Updated head-to-head sweep

| cores C | 3-factor µs | 2-factor µs | winner |
|--:|--:|--:|---|
| 1  | 4574 | 5550 | 3-factor |
| 2  | 2318 | 2820 | 3-factor |
| 4  | 1201 | 1461 | 3-factor |
| 8  |  666 |  757 | 3-factor |
| 16 |  407 |  428 | 3-factor |
| 32 | **319.5** | 334.7 | 3-factor |

**Conclusion reversed.** Before optimization the naive 2-factor was marginally *ahead* (386 vs 407 at
C=32). After applying the known reductions to both, the **3-factor is now ahead at every C** — its
extra reduction passes, once cheap, no longer offset its matmul savings. The gap is modest at C=32
(~5%) but consistent.

## Core-count sweep

Each stage runs on `C` cores; core `c` handles tiles `{c, c+C, …}` = `32/C` tiles (via the
`(start, stride)` args in the matmul kernels, the strided `_g` reader/writers, and the multi-core
corner turn). Steady-state host wall-clock, 20 iters, one `Finish`.

| cores C | waves ⌈32/C⌉ | mism | µs / NTT-32768 | speedup vs C=1 | efficiency |
|--:|--:|--:|--:|--:|--:|
| 1  | 32 | 0 | 7154 | 1.00×  | 100% |
| 2  | 16 | 0 | 3610 | 1.98×  | 99% |
| 4  | 8  | 0 | 1850 | 3.87×  | 97% |
| 8  | 4  | 0 |  988 | 7.24×  | 90% |
| 16 | 2  | 0 |  573 | 12.48× | 78% |
| 32 | 1  | 0 |  407 | 17.59× | 55% |

(Stable across seeds; seed 0 and seed 7 agree to <1%.)

These are **after optimizing the all-to-all** (below). The earlier version — single-core,
per-element corner turn — flattened hard (C=32: 1285 µs, 5.87×, 18% efficiency), because the
972 µs all-to-all was a fixed serial cost. Optimizing it cut C=32 from **1285 → 407 µs (3.2×)** and
lifted efficiency **18% → 55%**.

## Optimizing the all-to-all (the original bottleneck)

In the first version the efficiency drop was traced (per-stage timing at C=32) to ONE stage: the
**single-core, per-element corner turn at 971.6 µs — 76% of the C=32 runtime.** It ran on one core
regardless of C (an Amdahl serial fraction; the naive curve fit `T(C)=S+(T(1)−S)/C`, S≈1083 µs),
and was slow because `all_to_all_corner.cpp` did 32768 per-element L1 copies, recomputing face-index
arithmetic per element (~30 cycles/elt × 32768 ≈ 1 ms). It was **not** DRAM bandwidth or dispatch.

`all_to_all_corner_mc.cpp` fixes it with two changes (informed by the tt-metal NOC/data-movement
APIs and existing transpose/permute/reshard kernels):
1. **Multi-core** — the 32 output tiles are independent, so core `c` builds `{c, c+C, …}`; the
   corner-turn work divides by C like every other stage.
2. **Row-chunk copy** — a tile row is two contiguous 16-element face-chunks, so we copy 16 words at
   a time with the face-index math hoisted out of the inner column loop (~16× fewer index ops).

Result: the corner turn went **972 µs → 94.8 µs at C=32 (10×)**, and now scales.

## 5. Where the time goes now (C=32 per-stage, µs)

| cost class | 3-factor | 2-factor |
|---|--:|--:|
| matmul | 75.6 (A2 25.3 + C2 23.8 + D2 26.5) | **239** (M1 K=8: 158, M2 K=4: 81) |
| recombine | 95.9 (3×32) | 64.9 (2×32) |
| twiddle | 54.7 (B 29 + TB 25) | 25.6 (1×) |
| decompose | 50.0 (3×~16) | 32.4 (2×16) |
| **all-to-all (corner turn)** | **95.8** | **0** |
| **total (chained)** | **319.5** | **334.7** |

After optimization the **structural costs are exposed**, and they differ sharply:

- **2-factor is matmul-bound**: M1+M2 = 239 µs (71% of the op). This is the fundamental 4× MAC cost
  of the coarse factorization — the optimizations can't touch it (the matmul is already int8 + K-loop).
- **3-factor is movement/reduction-bound**: the single-core **all-to-all corner turn (95.8 µs)** is now
  its largest stage, tied with its 3 recombines (95.9 µs); its matmul is cheap (75.6 µs).

So the two designs still trade the same budget, but the optimizations sharpened the picture: the
2-factor's wall is matmul FLOPs; the 3-factor's wall is the corner turn + having one more reduction
and twiddle.

## 6. Remaining levers (ranked, per variant)

**3-factor** (319.5 µs):
1. **Multi-core the all-to-all input staging** — the corner turn is the top single cost (95.8 µs) and
   each core still reads all 32 input tiles; a permuted/strided read makes the read divide by C
   (`NTT_32768_PERF.md`). Biggest win.
2. **Fuse** A1–A3 and C1–C3 / D1–D3 like NTT-1024 — eliminates per-stage dispatch + DRAM round-trips
   for the on-tile NTT-1024 blocks.

**2-factor** (334.7 µs):
1. **Matmul operand delivery** — M1 (K=8) at 158 µs is the bottleneck and is operand-feed bound
   (output-stationary K-loop re-reads K-tiles per output tile). Cache/coalesce the K-column reads.
2. Little reduction headroom left — recombine + twiddle are already SFPI.

## 7. Scaling outlook (unchanged by these optimizations)

At N=32768 the two are close (319.5 vs 334.7). The crossover still favors the 3-factor as N grows:
its matmul scales `O(N·N^{1/3})` vs the 2-factor's `O(N·√N)`, so the 2-factor's matmul wall (already
71% of its op) blows up faster. The optimizations don't change that asymptote — they just made the
3-factor win at N=32768 too, by cutting the reduction overhead that had been hiding its matmul
advantage.

## 8. Validation

Both variants: `mismatches = 0/32768` vs the direct O(N²) DFT golden at **every** C ∈ {1,2,4,8,16,32}
(seed 0; methodology matches the prior docs). The SFPI recombine / bigK / hybrid-twiddle kernels are
bit-identical to the building-block versions they replace.
