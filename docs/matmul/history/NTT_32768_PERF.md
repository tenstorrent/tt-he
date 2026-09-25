# Performance report: on-device NTT-32768 (32,32,32)

> **Status: Historical.** Consolidated into [`../ntt_32768_perf.md`](../ntt_32768_perf.md).
> Core-count sweep and the all-to-all corner-turn fix (972 → 94.8 µs). Both carried forward.
> Numbers are a snapshot, measured on Wormhole n150.

**Program:** `tests/ntt/ntt32768_dev.cpp` (Wormhole n150), `q = 167772161`. Fully on-device — host
writes input + baked constants once, runs 12 chained device programs through DRAM, reads once. No
host arithmetic in the transform, no host↔device traffic between stages. Correct: `0/32768` vs the
3-pass golden (which is itself `0/32768` vs the direct O(N²) DFT, spot-checked in-binary), every
seed and **every core count**. Design: `docs/matmul/NTT_32768_3FACTOR_PLAN.md`. Golden:
`reference/ntt_matmul/oracle_32768.py`.

## Pipeline (12 stages, 1 all-to-all)

32768 values = 32 tiles of 32×32 (cube `n = n1 + 32·n2 + 1024·n3`). Block 1 is the verified NTT-1024
run on each tile; one corner turn couples the 32 tiles before Block 2 (Pass 3).

```
Block 1 (per tile):  A1 decompose → A2 matmul(W@) → A3 recombine = G
                     B twiddle_a → C1 decompose → C2 matmul(@W) → C3 recombine = H
TB twiddle_b → A2A all-to-all corner turn (n1↔s) → Block 2: D1 decompose → D2 matmul(W@) → D3 recombine = E
out[s + 32t + 1024v] = E[s][v][t]
```

All matmuls are int8-plane modular (decompose → 16 int8 matmuls → recombine, `W` folded offline).

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

## Per-stage timing at C=32 (optimized)

| stage @ C=32 | µs |
|---|--:|
| all-to-all (32 cores) | 94.8 |
| each recombine (A3/C3/D3) | ~47.7 |
| each twiddle (B/TB) | ~43 |
| each matmul (A2/C2/D2) | ~25 |
| each decompose (A1/C1/D1) | ~15 |

- **Near-linear to 8 cores** (90% efficiency), good to 16 (78%), 55% at 32 — vs 18% before.
- **The remaining limiters at high C** are now (a) the **3 Montgomery reductions** in the recombines
  (~143 µs combined) plus the 2 twiddles (~86 µs) — the modular reduction is the real compute cost,
  as in the 1024 case; and (b) the corner turn's redundant input staging (each core still reads all
  32 input tiles into L1, ~95 µs) and the fixed 12-enqueue dispatch. No single stage dominates now.
- **Best operating point.** For throughput-per-core, **4 cores (71%, 2.66 ms)** or **8 cores (51%,
  1.85 ms)** is the sweet spot; 32 cores minimizes latency (1.28 ms) at low efficiency. If running
  many NTTs, pack several transforms across the 64-core grid at ~4–8 cores each rather than one
  transform on 32.

## Next levers (ranked) — after the all-to-all fix

1. **Corner-turn input staging**: each core still reads all 32 input tiles into L1 (~95 µs floor at
   C=32). Read only the needed rows via sub-tile DRAM `noc_async_read` at computed byte offsets (the
   `reader_permute_interleaved_tiled_generic.cpp` technique), so the read also divides by C.
2. **Fuse stages** (decompose+matmul+recombine into one program) to cut the 12-enqueue dispatch — at
   55% efficiency on 32 cores the fixed dispatch now matters.
3. **Cheaper modular reduction**: the 3 recombines + 2 twiddles (~230 µs combined at C=32) are the
   dominant compute now — same Montgomery-reduction wall as the 1024 case.
4. **Throughput packing**: at 4–8 cores efficiency is 90–97%, so for many NTTs, pack several
   transforms across the 64-core grid at ~8 cores each rather than one transform on 32.

## Reproduce
```
cd device && cmake -B build -G Ninja && ninja -C build ntt32768_dev
./build/ntt32768_dev [q] [seed]      # prints the core-count sweep table
```

## Caveats
- Host wall-clock (includes dispatch); single n150; forward NTT only; N=32768 = 32×32×32, T=32.
- The all-to-all is single-core in this version (noted above).
