# exp_Tg018dram — the DRAM sweep for exp_Tg018's two dominant kernels

**Status: Historical** — moved to `history/` with the rest of the exp_Tg018 lineage. Not superseded, though: its content is **negative results**, and those do not expire. Five of the six DRAM levers it tried did nothing or hurt, and the sixth was already what `dram_guide.md` does. Read it before trying to make DRAM faster, so the five are not re-tried.

**Status: DONE.** Every number below is measured on this p150, every configuration was verified
against a host golden, and the device-side join was checked 1-1 on every run. Nothing here is an
estimate unless it says so, and where a later measurement superseded an earlier one the earlier one
is marked rather than deleted.

`host/experiments/exp_tg018dram.cpp` · `kernels/experiments/exp_tb001/` · parent: `docs/he/exp_tg018.md`

> **Looking for the answer rather than the audit trail? → `docs/he/dram_guide.md`.** That file is the
> distilled reference: the measured hardware ladder, the cost model, the recipe that reaches 433
> GB/s, the null results not worth retrying, and the code. This document is the chronological
> record — it keeps the wrong turns and what killed them, which is why it is long.

## Why

exp_Tg018's CKKS `ct_mul` costs **236.6 ms** of critical-path busy, and two kernels own 69% of it:
`modmul_sfpi_full` (48.4 ms, 20.5%) and the fused int8 matmul family (43.8 ms across `_a`, `_a_acc`,
`_c`). The investigation recorded in `exp_tg018.md` concluded that the limiter was **outstanding
request depth** — each core holding 8 KB in flight against a machine that wanted ~256 KB per core.

That conclusion was assembled from three measurements taken on different workloads: a NOC benchmark,
a per-RISC breakdown of `modadd`, and a per-core table spanning two experiments. This experiment
measures the two kernels *directly*, with the two things that were confounded there pulled apart
into independent axes:

* **T** — tiles of work per core. The tree fixes this at 1.
* **G** — cores occupied. The tree varies this, and it is what shares the DRAM path.

Every point is **one program dispatch**: G cores in a single `CoreRangeSet`, T tiles looped inside
each core's kernel. Dispatch cost is therefore O(1) across the whole sweep and cannot be what moves
any number — the same structural choice `exp_Bw001` makes.

## What is measured

| column | what it is |
|---|---|
| `host ms` | wall of `EnqueueMeshWorkload` + `Finish`, min over ITERS. What a caller pays. |
| `kern us` | the dispatch's device span: max over cores of (last marker − first marker). `critical path busy` for a single dispatch. |
| `read / unpk / math / pack / write` | the same span per RISC — NCRISC is the reader, BRISC the writer, TRISC_0/1/2 the unpack/math/pack pipeline. |
| `GB/s` | bytes moved to and from DRAM ÷ the **kernel** span. Dividing by a host wall would measure the dispatch path, not the memory system. |
| `us/tile` | `kern us / T` — the per-core per-tile latency exp_Tg018's per-core table used. |

**The per-RISC columns are envelopes, not busy time.** A RISC's `*-KERNEL` zone runs from kernel
entry to kernel exit, so it includes every `cb_wait_front` / `cb_reserve_back` stall. `read` ≈ `kern`
means the reader was resident for the whole dispatch, not that it was transferring for the whole
dispatch. Two places where the envelope *is* clean are used below and nowhere else:

* **`read` at T = 1** — one group, on an empty CB, so no stall is possible. This is a pure DRAM
  round trip, and it is what pins the bandwidth ceiling.
* **`us/tile` at G = 1** — one core, nothing to contend with. This is the pure compute floor.

## The three knobs

| knob | values | what it isolates |
|---|---|---|
| depth mode | `seq` (D=1) / `batch` (D = min(T, L1 cap)) | request depth, holding everything else fixed. **`seq` is the production kernel**: `reader_ab_depth` at D=1 *is* `reader_modmul.cpp`, and `reader_fused_depth` at D=1 *is* `reader_fused_batch.cpp`. The `seq` rows at T>1 are "today's kernel, run T times". |
| `w_share` | 0 / 1 | fused matmul only. A work tile is 4 KB of data and **16 KB of folded weights**, and every tile of a dispatch runs against one prime, hence one W. `w_share=1` reads W once per core instead of once per tile. |
| G, T | 1…130 / 1…64 | the two axes above |

Correctness is checked at **every** point, not sampled: the output buffer is poisoned with
`0xDEADBEEF` before each configuration and the full G·T tile range is compared against a host golden
after it, so a timing row can never be a measurement of a kernel that quietly did nothing. All 84 +
168 configurations passed with 0 mismatches.

## Result 1 — the read path reaches ~225 GB/s at one tile in flight

> **Partly superseded.** The ~220 GB/s figure below is what these kernels reached at the arithmetic
> intensity they happen to have; it is not the machine's ceiling. The `extra` knob added later drives
> intensity directly and measures **253.8 GB/s**. See *Reaching the memory-bound regime on purpose*.
> The measurements here stand — they just answer "what did this configuration achieve", not "what can
> the card do".

`read` at T = 1 is a pure round trip. Three kernels, two different transfer sizes, one answer:

| | G=1 | G=8 | G=32 | G=64 | G=104 | G=130 |
|---|---|---|---|---|---|---|
| `modmul`, 8 KB/core | 12.4 | 53.0 | 135.3 | 168.6 | 208.9 | **225.6** |
| fused `ws=0`, 20 KB/core | 20.4 | 70.6 | 141.8 | 151.7 | 191.8 | **224.3** |
| fused `ws=1`, 20 KB/core | 14.1 | 62.0 | 146.2 | 162.0 | 205.7 | **234.9** |

*(GB/s aggregated over all G cores.)*

The highest **sustained** figure over a whole dispatch is **217.0 GB/s** (fused, `ws=0`, `seq`,
G=104, T=64) with 211–212 GB/s at G=130. So the usable ceiling on this card, on real HE kernels, is
**~215–225 GB/s** at one tile in flight — against 287 GB/s from the repo's NoC benchmark and 305 GB/s
from its naive DRAM benchmark. Raising arithmetic intensity later closed most of that gap: 253.8 GB/s
measured, which is the figure to design against.

### This corrects a recorded finding

`exp_tg018.md` records "at G=64, 2 tiles in flight → 47.6 GB/s, 16% of achievable". That number came
from **bytes ÷ whole-dispatch time**. Reproduced here exactly: `modmul` at G=64, T=1 moves 786 KB in
a 16.16 µs dispatch = 48.6 GB/s. But 13.2 µs of that dispatch is SFPU math. The reader moved its
512 KB in **3.11 µs = 164 GB/s**.

So the 16% was never a statement about the memory system — it was the *arithmetic intensity of a
compute-bound kernel*. The depth conclusion still holds for the kernels it was drawn on (`modadd`,
the corner turn, which have almost no compute). It does **not** hold for these two, and the section
in `exp_tg018.md` has been narrowed accordingly.

## Result 2 — one kernel is compute-bound, the other is memory-bound, and a one-line model says which

Per-tile compute floor (G=1, T=64, `seq` — one core, no contention, latency fully amortised):

| kernel | floor | bytes/tile | demand per core | at G=130 | vs 250 GB/s |
|---|---|---|---|---|---|
| `modmul_sfpi_full` | **12.45 µs** | 12 KB | 0.96 GB/s | 125 GB/s | **compute-bound everywhere** |
| fused matmul `ws=0` | **7.99 µs** | 24 KB | 3.01 GB/s | 391 GB/s | **memory-bound above G ≈ 83** |
| fused matmul `ws=1` | **7.99 µs** | 8 KB | 1.00 GB/s | 130 GB/s | **compute-bound everywhere** |

All three predictions are confirmed by the sweep:

| kernel | `us/tile` G=1 → G=130 (T=64) | GB/s at G=130 | predicted |
|---|---|---|---|
| `modmul` | 12.45 → 12.93 (**+4%**) | 123.5 | 125 ✓ |
| fused `ws=0` | 7.99 → 15.08 (**+89%**) | 211.9 | ceiling ✓ |
| fused `ws=1` | 7.99 → 8.38 (**+5%**) | 131.0 | 130 ✓ |

`modmul` delivers 123.5 GB/s at G=130 and its compute *asks for* 125 — it is getting everything it
wants. Adding bandwidth to it would buy nothing. The fused matmul at `ws=0` asks for 391 and gets
217, so it runs at 53% of its own compute rate.

**The knee is located.** Fused `ws=0`, T=64: G=1…64 sits flat at 8.0–8.2 µs/tile (compute-bound),
then G=104 → 11.78 and G=130 → 15.08. 250 ÷ 3.01 = 83 cores, and the measured break is between 64
and 104. At G=64 the dispatch runs at 191 GB/s — right at the edge.

## Result 3 — `w_share` is worth 1.80× and it is free

Reading W once per core instead of once per work tile, `seq`:

| G | T | `ws=0` | `ws=1` | speedup | DRAM 0 → 1 |
|---|---|---|---|---|---|
| 64 | 64 | 526.3 µs | 522.6 µs | 1.01× | 96 → 33 MB |
| 104 | 64 | 753.7 µs | 531.7 µs | **1.42×** | 156 → 54 MB |
| 130 | 8 | 137.2 µs | 90.5 µs | **1.52×** | 24 → 10 MB |
| 130 | 32 | 483.7 µs | 281.9 µs | **1.72×** | 98 → 35 MB |
| 130 | 64 | 964.9 µs | 536.4 µs | **1.80×** | 195 → 67 MB |

It does nothing below the knee (1.01× at G=64, exactly as the model says — there is spare bandwidth,
so wasting it costs nothing) and everything above it. At G=130, T=64 it moves the kernel from
memory-bound back to compute-bound: 15.08 → 8.38 µs/tile against a floor of 7.99.

It is free because the sharing is already legal in the tree — a transform pass dispatches one limb,
so every tile on a core already runs against the same prime and therefore the same folded W. The
only reason the tree re-reads it is that a dispatch currently carries exactly one tile.

## Result 4 — deeper requests make both kernels SLOWER

`batch` (D = min(T, L1 cap), the whole group in flight against one barrier) against `seq` (D = 1,
today's pattern), same G, same T, same bytes:

| kernel | G=64, T=16 | G=130, T=64 |
|---|---|---|
| `modmul` | 1.22× slower | 1.29× slower |
| fused `ws=0` | 1.69× slower | 1.17× slower |
| fused `ws=1` | 1.22× slower | 1.22× slower |

This is the opposite of what the depth hypothesis predicted, and it is consistent across all 126
`seq`/`batch` pairs. The mechanism: with 8–12 µs of compute per tile, `seq` overlaps the reader with
the compute *tile by tile* — the reader fetches tile n+1 during the ~10 µs the SFPU spends on tile n,
which is far more time than a round trip needs. `batch` replaces that smooth stream with bursts: 130
cores each issue 16–32 reads simultaneously, the DRAM queue depth spikes, and the **first** tile of
each group — the one compute is waiting on — arrives later than it would have. Depth helps when
there is nothing to overlap with. Here there is.

The reader envelope shows the mechanism directly. `modmul` at G=130, T=64: `read` is 791 µs under
`seq` (resident nearly the whole 828 µs dispatch, quietly staying one tile ahead) and 287 µs under
`batch` (bursts of work separated by idle) — yet the dispatch is 1071 µs. The reader finished its
transfers *sooner* and the kernel still took 29% longer.

## What this changes for exp_Tg018

1. **`modmul_sfpi_full` (48.4 ms, 20.5% of the multiply) will not respond to any memory
   optimisation.** It is compute-bound at every core count, running at 96% of its single-core floor
   even with all 130 cores loaded. The only lever is the SFPI Montgomery multiply itself — 12.45 µs
   per 32×32 tile is ~16 µs of SFPU work per 1024 elements. Tier 1 (batched feed/drain) was recorded
   in `exp_tg018.md` as a null result; this explains why it *had* to be, and closes it.

2. **`w_share` is the single highest-value change available**, worth up to 1.80× on the fused matmul
   family — 43.8 ms of the multiply — and it needs no new arithmetic, only more than one tile per
   dispatch so there is something to share across. It requires the same "T tiles per core" change
   that this experiment already implements in `fused_matmul_a_loop.cpp`.

3. **More tiles per core is worth having on its own**, before any sharing: `modmul` at G=64 goes
   from 16.16 µs/tile at T=1 to 12.64 at T=64 (**−22%**), because the first tile's round trip stops
   being paid once per tile. This is exp_Tg018's deferred "Tier 2", and it is now quantified.

4. **Do not batch the readers.** Tier 3's 16 KB-packet direction inherits the same risk: it deepens
   requests on kernels that do not need depth. If it is tried, it should be measured against `seq`
   at the same (G, T) here first.

5. **The design ceiling is ~250 GB/s, not 305 or 512.** Measured directly with the arithmetic-
   intensity knob; flat between G=128 and G=130, so this data does not support extrapolating past it.
   Any future estimate of "how much bandwidth does this pass need" should be compared against that.

## The full grid — G = 2ⁿ up to 64, workload = 1 … 128 tiles per core

The sweep above used a coarse G ladder to reach 130 cores. This is the dense power-of-two grid over
both axes, `seq` mode (D=1, the production reader), `w_share=0` for the fused matmul (the production
W policy). 56 configurations per kernel, all verified, 224/224 dispatch groups joined 1-1 each.
Data: `data/tg018dram_grid_modmul.csv`, `data/tg018dram_grid_fused.csv` (every row carries all five
per-RISC columns).

Each cell is **one dispatch**, so the profiler cost is amortised over the whole T loop and is
negligible at the long end — at T=128 a single dispatch runs 1.0–1.6 ms.

### Kernel span, µs — trace, max over cores of the dispatch envelope

`modmul_sfpi_full`

| T \ G | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 1 | 13.71 | 13.68 | 13.91 | 14.33 | 14.43 | 14.93 | 16.26 |
| 2 | 26.03 | 26.11 | 26.30 | 26.76 | 27.10 | 27.91 | 29.36 |
| 4 | 50.96 | 51.02 | 51.32 | 51.93 | 52.02 | 53.69 | 57.21 |
| 8 | 100.73 | 100.99 | 101.48 | 102.27 | 103.05 | 105.95 | 112.52 |
| 16 | 200.21 | 200.44 | 200.46 | 201.28 | 202.37 | 205.38 | 212.87 |
| 32 | 399.19 | 399.01 | 399.34 | 400.13 | 401.30 | 404.23 | 410.37 |
| 64 | 796.96 | 796.70 | 797.12 | 798.05 | 799.32 | 802.16 | 808.89 |
| 128 | 1592.59 | 1592.25 | 1593.14 | 1594.00 | 1595.55 | 1598.67 | 1605.74 |

`fused_matmul_a` (`w_share=0`)

| T \ G | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 1 | 9.35 | 9.39 | 9.98 | 10.94 | 11.51 | 13.40 | 17.08 |
| 2 | 17.29 | 17.40 | 17.90 | 18.93 | 19.63 | 21.92 | 28.96 |
| 4 | 33.36 | 33.41 | 33.87 | 34.94 | 35.71 | 37.64 | 43.99 |
| 8 | 65.22 | 65.27 | 65.81 | 66.85 | 67.97 | 70.57 | 80.25 |
| 16 | 128.94 | 128.99 | 129.47 | 130.56 | 131.69 | 134.25 | 143.95 |
| 32 | 256.40 | 256.42 | 256.90 | 257.91 | 259.06 | 261.87 | 271.61 |
| 64 | 511.44 | 511.16 | 511.76 | 512.75 | 513.93 | 517.05 | 526.05 |
| 128 | 1021.28 | 1021.13 | 1021.21 | 1022.71 | 1023.52 | 1027.53 | 1036.87 |

### µs per tile per core — `kern_us / T`, the cost-model number

`modmul_sfpi_full` — floor **12.44 µs**

| T \ G | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 1 | 13.708 | 13.676 | 13.914 | 14.333 | 14.428 | 14.926 | 16.259 |
| 2 | 13.013 | 13.056 | 13.148 | 13.378 | 13.549 | 13.955 | 14.682 |
| 4 | 12.741 | 12.755 | 12.830 | 12.983 | 13.004 | 13.422 | 14.303 |
| 8 | 12.592 | 12.623 | 12.685 | 12.784 | 12.881 | 13.243 | 14.065 |
| 16 | 12.513 | 12.527 | 12.528 | 12.580 | 12.648 | 12.836 | 13.304 |
| 32 | 12.475 | 12.469 | 12.479 | 12.504 | 12.541 | 12.632 | 12.824 |
| 64 | 12.453 | 12.448 | 12.455 | 12.470 | 12.489 | 12.534 | 12.639 |
| 128 | **12.442** | 12.440 | 12.446 | 12.453 | 12.465 | 12.490 | **12.545** |

`fused_matmul_a` (`w_share=0`) — floor **7.98 µs**

| T \ G | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 1 | 9.352 | 9.390 | 9.984 | 10.944 | 11.513 | 13.397 | 17.082 |
| 2 | 8.646 | 8.700 | 8.950 | 9.466 | 9.817 | 10.961 | 14.479 |
| 4 | 8.339 | 8.353 | 8.467 | 8.736 | 8.926 | 9.409 | 10.998 |
| 8 | 8.152 | 8.159 | 8.226 | 8.357 | 8.497 | 8.821 | 10.031 |
| 16 | 8.059 | 8.062 | 8.092 | 8.160 | 8.231 | 8.391 | 8.997 |
| 32 | 8.012 | 8.013 | 8.028 | 8.060 | 8.096 | 8.183 | 8.488 |
| 64 | 7.991 | 7.987 | 7.996 | 8.012 | 8.030 | 8.079 | 8.220 |
| 128 | **7.979** | 7.978 | 7.978 | 7.990 | 7.996 | 8.028 | **8.101** |

Both surfaces are **flat in G and monotone in T**, which is the whole story on this grid: up to 64
cores neither kernel pays a meaningful contention penalty, and everything the workload axis buys is
the amortisation of the first tile's round trip. The G=64 column costs +0.8% (modmul) and +1.5%
(fused) over the G=1 column at T=128; the T=1 → T=128 move is worth **−23%** and **−15%**.

### DRAM GB/s — bytes ÷ kernel span

`modmul_sfpi_full` (12 KB/tile) · `fused_matmul_a` `w_share=0` (24 KB/tile)

| T \ G | 8 | 16 | 32 | 64 | | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.9 | 13.6 | 26.3 | 48.4 | | 18.0 | 34.2 | 58.7 | 92.1 |
| 8 | 7.7 | 15.3 | 29.7 | 55.9 | | 23.5 | 46.3 | 89.2 | 156.8 |
| 32 | 7.9 | 15.7 | 31.1 | 61.3 | | 24.4 | 48.6 | 96.1 | 185.3 |
| 128 | 7.9 | 15.8 | 31.5 | **62.7** | | 24.6 | 49.2 | 98.0 | **194.2** |

Both scale essentially linearly in G across this whole grid — no knee, because 64 cores is below the
~83 where the fused matmul first hits the ~250 GB/s ceiling. `modmul` tops out at 62.7 GB/s not
because anything is limiting it but because that is all its compute asks for.

### Read (NCRISC) envelope, µs — and where it stops meaning "DRAM"

| T \ G | 1 | 8 | 32 | 64 | | 1 | 8 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| | **modmul** | | | | | **fused** | | | |
| 1 | 0.75 | 1.22 | 1.96 | 3.28 | | 1.00 | 2.31 | 4.73 | 8.58 |
| 2 | 1.50 | 2.14 | 3.60 | 6.11 | | 2.09 | 4.49 | 9.16 | 15.38 |
| 4 | 14.22 | 15.24 | 17.97 | 20.71 | | 11.50 | 13.79 | 17.43 | 29.40 |
| 128 | 1555.87 | 1557.16 | 1565.55 | 1572.56 | | 999.46 | 1000.91 | 1006.97 | 1018.25 |

The T=1 and T=2 rows are real transfer time — the CB is empty, so no stall is possible. From **T=4
onward the envelope is dominated by `cb_reserve_back` waiting on compute**, which is why it tracks
the kernel span rather than the byte count. Read these two rows as measurement and the rest as
"the reader was resident", nothing more.

The clean rows say the same thing the coarse sweep did: at G=64 the modmul reader moves 8 KB/core in
3.28 µs = 2.5 GB/s per core, 160 GB/s aggregate — three quarters of the way to the ceiling on the
*reads alone*, while the dispatch as a whole reports 48.4 GB/s because 13.2 µs of it is SFPU math.

### Per-RISC at T=128, the longest dispatch

| kernel | G | kern | read | unpack | math | pack | write |
|---|---|---|---|---|---|---|---|
| `modmul` | 1 | 1592.6 | 1555.9 | 1580.0 | 1592.2 | 1579.7 | 1592.5 |
| `modmul` | 64 | 1605.7 | 1572.6 | 1593.2 | 1605.4 | 1592.9 | 1605.6 |
| `fused` | 1 | 1021.3 | 999.5 | 1020.1 | 1020.9 | 1014.6 | 1021.2 |
| `fused` | 64 | 1036.9 | 1018.2 | 1034.3 | 1035.1 | 1028.8 | 1036.9 |

Every RISC is resident for essentially the whole dispatch — the pipeline is full end to end, and the
G=1 → G=64 delta is under 1.6% on every one of them. At steady state with this much compute per
tile, all five RISCs are simply following the slowest, which is `math`.

## Read bandwidth from the NCRISC envelope — and when that number means anything

### Which RISC carries DRAM

This is an assignment the **host program makes**, not a hardware fact — both RISCs can drive either
NOC and both NOCs reach DRAM. In every kernel in this tree:

| RISC | `DataMovementProcessor` | job | share of DRAM traffic |
|---|---|---|---|
| **NCRISC** | `RISCV_1`, NOC_1 | **every DRAM read** | modmul 8 of 12 KB/tile (67%) · fused `ws=0` 20 of 24 KB/tile (83%) |
| **BRISC** | `RISCV_0`, NOC_0 | **every DRAM write** | the remaining 4 KB/tile |

So NCRISC is the right place to look for reads, and it is where nearly all the traffic is — but it
is only the read half. **The write half cannot be measured this way at all**: BRISC's envelope opens
at kernel entry and immediately blocks on `cb_wait_front` for the compute output, so it runs ≈ the
whole dispatch no matter how many bytes it moves. Every `write` column in the T=128 table above is
within 0.1% of `kern` for exactly that reason.

### The table

Bytes **read** ÷ **read envelope**, aggregated over all G cores. Read bytes per core are
`T × 8 KB` (modmul), `T × 20 KB` (fused `ws=0`), `16 KB + T × 4 KB` (fused `ws=1`).

`modmul_sfpi_full`

| T \ G | 1 | 2 | 4 | 8 | 16 | 32 | 64 | |
|---|---|---|---|---|---|---|---|---|
| **1** | 10.9 | 21.7 | 37.1 | 53.5 | 89.0 | 134.0 | **160.1** | ← clean |
| **2** | 10.9 | 22.3 | 38.8 | 61.3 | 95.1 | 145.6 | **171.6** | ← clean |
| 4 | 2.3 | 4.6 | 8.9 | 17.2 | 33.4 | 58.4 | 101.2 | |
| 8 | 1.0 | 2.0 | 4.0 | 7.9 | 15.6 | 29.7 | 54.6 | |
| 16 | 0.8 | 1.6 | 3.2 | 6.4 | 12.6 | 24.7 | 47.5 | |
| 32 | 0.7 | 1.4 | 2.9 | 5.8 | 11.5 | 22.8 | 44.8 | |
| 64 | 0.7 | 1.4 | 2.8 | 5.5 | 11.0 | 21.9 | 43.4 | |
| 128 | 0.7 | 1.3 | 2.7 | 5.4 | 10.8 | 21.4 | 42.7 | |

`fused_matmul_a`, `w_share=0`

| T \ G | 1 | 2 | 4 | 8 | 16 | 32 | 64 | |
|---|---|---|---|---|---|---|---|---|
| **1** | 20.4 | 39.1 | 55.4 | 71.0 | 106.0 | 138.5 | **152.8** | ← clean |
| **2** | 19.6 | 40.2 | 58.2 | 73.0 | 110.6 | 143.2 | **170.4** | ← clean |
| 4 | 7.1 | 14.2 | 26.9 | 47.5 | 87.4 | 150.4 | 178.3 | |
| 8 | 3.8 | 7.5 | 14.8 | 28.7 | 55.7 | 102.7 | 164.2 | |
| 16 | 3.1 | 6.1 | 12.2 | 24.0 | 47.3 | 91.6 | 166.9 | |
| 32 | 2.8 | 5.6 | 11.1 | 22.1 | 44.0 | 86.7 | 166.0 | |
| 64 | 2.7 | 5.4 | 10.7 | 21.3 | 42.5 | 84.4 | 165.2 | |
| 128 | 2.6 | 5.2 | 10.5 | 21.0 | 41.8 | 83.3 | 164.8 | |

`fused_matmul_a`, `w_share=1`

| T \ G | 1 | 2 | 4 | 8 | 16 | 32 | 64 | |
|---|---|---|---|---|---|---|---|---|
| **1** | 14.0 | 27.1 | 44.8 | 61.9 | 101.2 | 143.1 | **159.0** | ← clean |
| **2** | 12.3 | 24.2 | 41.3 | 61.0 | 97.9 | 148.1 | **167.0** | ← clean |
| 4 | 3.2 | 6.4 | 12.4 | 22.9 | 42.8 | 76.0 | 116.5 | |
| 8 | 1.2 | 2.3 | 4.6 | 9.0 | 17.6 | 33.5 | 59.6 | |
| 16 | 0.8 | 1.5 | 3.1 | 6.1 | 12.1 | 23.7 | 44.9 | |
| 32 | 0.6 | 1.3 | 2.5 | 5.0 | 10.0 | 19.8 | 38.6 | |
| 64 | 0.6 | 1.1 | 2.3 | 4.6 | 9.1 | 18.1 | 35.8 | |
| 128 | 0.5 | 1.1 | 2.2 | 4.3 | 8.7 | 17.3 | 34.3 | |

### How to read it — the falling columns are the point, not a bug

Only **T ≤ 2 is bandwidth.** There the CB is empty when the reader starts, so no stall is possible
and the envelope is transfer time. All three kernels agree at G=64: **160–172 GB/s** with one tile in
flight, on `seq`, which is the production reader.

From T=4 on the envelope is dominated by `cb_reserve_back` waiting for compute to free a slot, so
`bytes ÷ envelope` stops measuring the memory system and starts measuring **the rate at which the
kernel consumes bytes** — its demand. That makes the columns diagnostic rather than useless:

* **`modmul` collapses** 171.6 → 42.7 GB/s as T grows. The reader spends the dispatch idle-waiting;
  its demand is 8 KB per 12.5 µs of SFPU work. **Compute-bound.**
* **fused `ws=0` holds flat** at 164–178 GB/s across every T at G=64. The reader never gets ahead —
  demand and capability are the same number. **Memory-bound**, and this is the cleanest single
  signature of it anywhere in the experiment.
* **fused `ws=1` collapses** 167.0 → 34.3 GB/s, exactly like `modmul`. Removing the repeated W read
  moved it across the line into compute-bound. That is the same conclusion the `us/tile` floor gives,
  reached from the opposite direction.

A flat column means the reader is saturated; a falling column means it is waiting.

### `w_share`, concretely, on the same grid

`kern us`, `w_share=0` / `w_share=1`, and the DRAM read volume at G=64:

| T | G=8 | G=32 | G=64 | read MB @ G=64, 0 → 1 |
|---|---|---|---|---|
| 1 | 10.9 / 11.2 | 13.4 / 13.1 | 17.1 / 16.8 | 1.2 → 1.2 (1.00×) |
| 4 | 34.9 / 35.2 | 37.6 / 37.4 | 44.0 / 41.6 | 5.0 → 2.0 (2.50×) |
| 16 | 130.6 / 130.8 | 134.3 / 134.4 | 143.9 / 140.2 | 20.0 → 5.0 (4.00×) |
| 64 | 512.8 / 512.6 | 517.0 / 516.4 | 526.1 / 521.9 | 80.0 → 17.0 (4.71×) |
| 128 | 1022.7 / 1022.1 | 1027.5 / 1026.0 | 1036.9 / 1031.8 | 160.0 → 33.0 (4.85×) |

**Up to 64 cores `w_share` cuts DRAM reads by up to 4.9× and buys almost no time** — 0.5% at G=64,
T=128. That is not a contradiction: at G ≤ 64 there is spare bandwidth, so not using it costs
nothing. The traffic reduction only converts into time above the knee, where the coarse sweep
measured **1.42× at G=104 and 1.80× at G=130**. Same change, worth nothing at 64 cores and 1.8× at
130 — which is exactly what a bandwidth ceiling looks like from below and from above.

## A two-term latency model for these kernels

Everything measured above collapses into

$$\text{latency}(G,T) \;=\; M(G,s) \;+\; T\cdot C(G,s)$$

| term | what it is | depends on |
|---|---|---|
| **C** | steady-state cost of one more tile | the kernel, and G *only above the bandwidth knee* |
| **M** | the un-hideable transient at the start of a dispatch | G and the traffic per tile; **saturates by T ≈ 8** |

### C — the per-tile term

$$C(G,s) \;=\; \max\!\Big(\underbrace{C_{\text{compute}}}_{\text{kernel}},\;\; \underbrace{\tfrac{G\cdot s}{B}}_{\text{memory}}\Big), \qquad B \approx 250\ \text{GB/s},\; s = \text{bytes per tile, read + written}$$

Fitting C independently at each core count on the 2ⁿ grid gives a **spread of 0.11–0.17%** across all
seven core counts — it is flat in G to within measurement noise, which is what identifies it as
compute. Then validated against the coarse sweep, which reaches past the knee:

| kernel | s | G=8 | G=32 | G=64 | G=104 | G=130 |
|---|---|---|---|---|---|---|
| `modmul` predicted | 12 KB | 12.44 | 12.44 | 12.44 | 12.44 | 12.44 |
| `modmul` measured | | 12.43 | 12.43 | 12.43 | 12.42 | 12.44 |
| fused `ws=0` predicted | 24 KB | 7.97 | 7.97 | 7.97 | **11.62** | **14.52** |
| fused `ws=0` measured | | 7.96 | 7.97 | 7.97 | **10.97** | **15.04** |
| fused `ws=1` predicted | 8 KB | 7.96 | 7.96 | 7.96 | 7.96 | 7.96 |
| fused `ws=1` measured | | 7.96 | 7.95 | 7.98 | 7.96 | 7.95 |

15 points: **≤0.3% error wherever the compute term wins, 3–6% where the memory term takes over.**
The `max` switches at exactly the core count the knee was observed at, and it explains `w_share` in
one line — it changes `s` from 24 KB to 8 KB, which moves the crossover from G≈83 to G≈243, i.e. off
this machine.

### M — the transient

Empirically `M = kern − C·T`, and it is **flat in T from T=8 onward** and **linear in G above G≈4**:

| kernel | M(1) | M(8) | M(32) | M(64) | fit for G ≥ 4 |
|---|---|---|---|---|---|
| `modmul` | 1.11 | 2.32 | 6.45 | 13.25 | `0.56 + 0.196·G` |
| fused `ws=0` | 1.49 | 2.99 | 7.09 | 16.58 | `0.50 + 0.242·G` |
| fused `ws=1` | 2.02 | 3.36 | 7.02 | 12.86 | `1.85 + 0.170·G` |

The slope is 0.17–0.24 µs **per core**, ordered by traffic per tile (4 KB → 0.170, 8 KB → 0.196,
20 KB → 0.242), so it is the DRAM path being paid for, not the launch path.

The mechanism is a **synchronised start**. At kernel entry all G cores issue their first reads in the
same instant, so the queue is at its worst exactly when nothing is yet pipelined. Over the following
few tiles the cores drift apart — each is gated by its own compute — the request stream smooths, and
from then on every additional tile costs exactly `C`. The table says that drift completes in about
**8 tiles**; note that M is ~5× larger than a single drain of the first burst would be, so the
"~8 tiles to desynchronise" reading is a description of the shape, not an independently verified
mechanism.

### Accuracy, and where it does not apply

With `M(G)` taken from the table above and `C` from the formula:

| | max error | mean |
|---|---|---|
| **T ≥ 8** | **0.34 – 0.50%** | < 0.2% |
| T = 4 | ~2% | |
| T = 1 | not modelled | |

**Below T ≈ 8 the model does not hold**, because the transient is still accumulating: treating M as
fully paid over-predicts (up to +58% at G=64, T=1), and ramping it linearly under-predicts (−41%).
Anything in that regime has to come from the measured cells.

That limitation is itself the finding worth carrying: **the tree runs at T = 1**, so every dispatch
in exp_Tg018 lives entirely inside the transient and never reaches the regime this model describes.
The cost of a production dispatch is not `C` — it is dominated by `M`, the one term that scales with
core count. Which is the same conclusion the per-tile tables reached, arrived at from the algebra.

## Reaching the memory-bound regime on purpose — the `extra` knob

Both kernels are compute-bound at every core count this card has (`modmul` would need G ≈ 250), so
sweeping G alone cannot show the memory-bound half of the roofline. `TGD_EXTRA=E` adds **E further
4 KB tile reads per work tile**, scattered across the same buffer by a prime stride and dumped into a
scratch CB nothing consumes (`reader_ab_depth.cpp`, `reader_fused_depth.cpp`).

Compute is untouched and the result is unchanged — **the golden check still has to pass at every
point, and did** — so E moves *only* `s`, the bytes per tile. It is the arithmetic-intensity axis,
and it walks the knee across the core grid. The reads scatter rather than repeat because re-reading
one page would hit DRAM row buffers and measure locality instead of the memory path. They share the
real read's barrier, so compute genuinely waits for them; unbarriered they would be nearly free.

The grid below is G = 1…130 × T = 8…64 × E = 0, 8, 16, 24, `seq` mode, `w_share=0`, both kernels —
576 configurations, every one verified, 576/576 dispatch groups joined 1-1 on each run.

### On the core ladder: 128 is fine, a rectangle is not required

The p150 grid is 13×10 = 130, so earlier sweeps used 104 (= 13×8, the largest full rectangle below
130) out of habit. That was a presentation choice, not a constraint — `crs_of()` builds a
`CoreRangeSet` one core at a time, so G = 128 is simply nine full rows plus 11 cores of the tenth,
the same ragged shape `exp_Bw001` documents for 112. It costs nothing measurable: at T=64, G=128 and
G=130 agree to 1–2% on every E, and peak bandwidth is 253.8 vs 252.6 GB/s. Use a clean 2ⁿ ladder and
add 130 only when the full grid is wanted.

### `fused_matmul_a` `ws=0` — µs per tile, cell = E 0 / 8 / 16 / 24 (compute floor 7.97)

| T \ G | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 130 |
|---|---|---|---|---|---|---|---|---|---|
| **8** |8.16 / 8.20 / 8.24 / 8.29|8.17 / 8.26 / 8.35 / 8.45|8.23 / 8.42 / 8.62 / 8.81|8.36 / 8.72 / 9.08 / 11.83|8.50 / 9.67 / 13.34 / 17.69|8.83 / 12.03 / 17.83 / 25.14|10.24 / 20.48 / 31.82 / 43.28|16.78 / 31.37 / 47.26 / 63.92|17.14 / 31.91 / 47.73 / 64.58|
| **16** |8.06 / 8.09 / 8.11 / 8.13|8.06 / 8.12 / 8.16 / 8.21|8.10 / 8.20 / 8.29 / 8.39|8.16 / 8.34 / 8.52 / 11.24|8.23 / 8.81 / 12.61 / 16.87|8.40 / 11.39 / 18.16 / 24.49|9.01 / 20.83 / 31.89 / 42.53|15.39 / 30.58 / 46.81 / 63.33|15.47 / 31.08 / 47.15 / 64.10|
| **32** |8.02 / 8.03 / 8.04 / 8.05|8.01 / 8.04 / 8.07 / 8.09|8.03 / 8.08 / 8.13 / 8.18|8.06 / 8.15 / 8.28 / 10.89|8.10 / 8.44 / 12.24 / 16.44|8.18 / 11.56 / 18.11 / 24.34|8.50 / 19.54 / 31.45 / 42.65|15.14 / 29.90 / 46.23 / 63.00|15.03 / 30.27 / 46.69 / 63.52|
| **64** |8.00 / 8.00 / 8.01 / 8.02|7.99 / 8.00 / 8.02 / 8.03|8.00 / 8.02 / 8.05 / 8.07|8.01 / 8.06 / 8.10 / 10.84|8.03 / 8.20 / 12.08 / 16.32|8.07 / 11.22 / 17.74 / 24.34|8.23 / 19.02 / 30.41 / 43.03|14.73 / 29.70 / 45.91 / 62.96|14.46 / 30.05 / 46.51 / 63.62|

`s` = 24 / 56 / 88 / 120 KB per tile · predicted knee `G* = B·C/s` = 81 / 35 / 22 / 16

### `fused_matmul_a` `ws=0` — achieved GB/s

| T \ G | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 130 |
|---|---|---|---|---|---|---|---|---|---|
| **8** |3 / 7 / 11 / 15|6 / 14 / 22 / 29|12 / 27 / 42 / 56|24 / 53 / 79 / 83|46 / 95 / 108 / 111|89 / 153 / 162 / 156|154 / 179 / 181 / 182|188 / 234 / 244 / 246|186 / 234 / 245 / 247|
| **16** |3 / 7 / 11 / 15|6 / 14 / 22 / 30|12 / 28 / 43 / 59|24 / 55 / 85 / 87|48 / 104 / 114 / 117|94 / 161 / 159 / 161|175 / 176 / 181 / 185|204 / 240 / 246 / 248|206 / 240 / 248 / 249|
| **32** |3 / 7 / 11 / 15|6 / 14 / 22 / 30|12 / 28 / 44 / 60|24 / 56 / 87 / 90|49 / 109 / 118 / 120|96 / 159 / 159 / 162|185 / 188 / 183 / 184|208 / 245 / 249 / 250|213 / 246 / 251 / 251|
| **64** |3 / 7 / 11 / 15|6 / 14 / 22 / 31|12 / 29 / 45 / 61|25 / 57 / 89 / 91|49 / 112 / 119 / 120|97 / 164 / 163 / 162|191 / 193 / 190 / 183|214 / 247 / 251 / 250|221 / 248 / 252 / 251|

### `modmul_sfpi_full` — µs per tile, cell = E 0 / 8 / 16 / 24 (compute floor 12.44)

| T \ G | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 130 |
|---|---|---|---|---|---|---|---|---|---|
| **8** |12.58 / 12.62 / 12.67 / 12.72|12.60 / 12.68 / 12.78 / 12.87|12.64 / 12.80 / 12.99 / 13.18|12.73 / 13.03 / 13.56 / 13.74|12.88 / 13.38 / 13.97 / 17.10|13.22 / 14.02 / 17.63 / 23.13|14.07 / 18.48 / 28.37 / 38.63|16.41 / 27.71 / 43.72 / 59.64|16.48 / 28.11 / 44.33 / 60.65|
| **16** |12.50 / 12.53 / 12.55 / 12.58|12.51 / 12.55 / 12.60 / 12.65|12.52 / 12.62 / 12.71 / 12.81|12.58 / 12.73 / 12.98 / 13.12|12.65 / 12.88 / 13.19 / 15.97|12.83 / 13.19 / 16.86 / 23.22|13.30 / 16.26 / 29.32 / 38.98|14.40 / 27.25 / 43.17 / 58.52|14.44 / 27.53 / 43.54 / 59.54|
| **32** |12.47 / 12.48 / 12.49 / 12.50|12.47 / 12.49 / 12.51 / 12.54|12.48 / 12.52 / 12.57 / 12.62|12.50 / 12.58 / 12.72 / 12.80|12.54 / 12.66 / 12.80 / 15.52|12.63 / 12.83 / 16.18 / 22.54|12.82 / 15.26 / 26.95 / 40.64|13.42 / 25.00 / 41.89 / 57.50|13.41 / 26.00 / 42.42 / 58.29|
| **64** |12.45 / 12.45 / 12.46 / 12.46|12.45 / 12.46 / 12.47 / 12.48|12.45 / 12.47 / 12.50 / 12.52|12.47 / 12.51 / 12.57 / 12.60|12.49 / 12.55 / 12.62 / 15.21|12.53 / 12.63 / 15.95 / 21.85|12.64 / 15.06 / 26.21 / 38.42|12.93 / 22.72 / 41.47 / 56.96|12.93 / 23.19 / 41.60 / 58.07|

`s` = 12 / 44 / 76 / 108 KB per tile · predicted knee `G*` = 253 / 69 / 40 / 28

### `modmul_sfpi_full` — achieved GB/s

| T \ G | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 130 |
|---|---|---|---|---|---|---|---|---|---|
| **8** |1 / 4 / 6 / 9|2 / 7 / 12 / 17|4 / 14 / 24 / 34|8 / 28 / 46 / 64|15 / 54 / 89 / 104|30 / 103 / 141 / 153|56 / 156 / 176 / 183|96 / 208 / 228 / 237|97 / 208 / 228 / 237|
| **16** |1 / 4 / 6 / 9|2 / 7 / 12 / 17|4 / 14 / 24 / 35|8 / 28 / 48 / 67|16 / 56 / 94 / 111|31 / 109 / 148 / 152|59 / 177 / 170 / 182|109 / 212 / 231 / 242|111 / 213 / 232 / 241|
| **32** |1 / 4 / 6 / 9|2 / 7 / 12 / 18|4 / 14 / 25 / 35|8 / 29 / 49 / 69|16 / 57 / 97 / 114|31 / 112 / 154 / 157|61 / 189 / 185 / 174|117 / 231 / 238 / 246|119 / 225 / 238 / 247|
| **64** |1 / 4 / 6 / 9|2 / 7 / 12 / 18|4 / 14 / 25 / 35|8 / 29 / 50 / 70|16 / 57 / 99 / 116|31 / 114 / 156 / 162|62 / 192 / 190 / 184|122 / 254 / 240 / 249|124 / 253 / 243 / 248|

### How to read it

**Across a cell** (E rising) — the same kernel as arithmetic intensity falls. At small G the four
numbers are nearly identical: the extra traffic is free. At large G they fan out hard — 14.46 →
63.62 µs/tile at G=130, T=64.

**Along a row** (G rising) — flat, then the knee, then unbounded growth. The knee walks left as E
grows, near the predicted `G* = B·C/s` in every row of both kernels.

**Down a column** (T rising) — the transient `M(G)` amortising. Worth ~2% at E=0 and essentially
nothing once memory-bound, because past the knee `C` dwarfs `M`.

The deepest point is **64.58 µs/tile against a 7.97 floor — 8.1×, and it is pure queueing**: the
kernel does exactly the same arithmetic it does at 8.02. Peak aggregate bandwidth is **253.8 GB/s**
(modmul, T=64, G=128, E=8), with the fused matmul reaching 252 — two different kernels topping out in
the same place.

**What memory-bound costs:** fused, E=24, G=1 → 130 is **×7.9 per-tile latency** for **×16.6
throughput** (15.3 → 253.8 GB/s). Past the knee, cores buy throughput and are paid for in latency,
roughly one for one. Below it they are free.

### B(G), and an important restriction on it

`bytes ÷ kernel span` only measures the memory system where the kernel is actually memory-bound.
Restricting to cells at more than 1.3× the compute floor:

| G | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 130 |
|---|---|---|---|---|---|---|---|---|---|
| memory-bound cells | 0 | 0 | 0 | 4 | 9 | 19 | 29 | 29 | 29 |
| **B(G), GB/s** | ≥15.3 | ≥30.6 | ≥60.9 | 90.7 | 120.5 | 163.6 | 192.9 | **253.8** | 252.6 |
| per core | ≥15.32 | ≥15.30 | ≥15.22 | 11.34 | 7.53 | 5.11 | 3.01 | 1.98 | 1.94 |

**G ≤ 4 is a lower bound, not a measurement.** Even at E=24 — 120 KB per tile — those core counts
never leave the compute-bound regime, so all we learn is that one core sustains *at least* 15.3 GB/s
while busy computing. The genuine measurements start at G=8.

B is flat between 128 and 130 (253.8 / 252.6), so **~253 GB/s is the measured ceiling** and this data
does not support extrapolating past it.

#### Two retractions

An earlier revision of this section fitted `B(G) = 278·G/(G + 22.9)` and reported a linear region
ending at exactly G=8, which it tied to Blackhole's 8 DRAM banks. Both are withdrawn:

* the fit was built on points that included **compute-limited** low-G cells, and re-fitting on the
  E=24 data drops the asymptote to 234 GB/s with 6.8% mean error while under-predicting G=128 by 16%.
  A two-parameter hyperbolic does not describe this curve; the measured table above is the result.
* the "linear to 8 cores" region was an artefact of the same problem. With compute-limited cells
  excluded there is no measurement below G=8 at all, so the coincidence with the bank count is not
  evidence of anything.

### Why latency rises long before the bandwidth ceiling — it is the NoC route, not DRAM

The grid above has cells that are clearly memory-bound while the aggregate is nowhere near 250 GB/s.
`fused`, T=32, E=16, G=16: 12.3 µs/tile against a 7.97 floor, at **112 GB/s** — 44% of the ceiling.
Something is limiting it that is not the memory system's capacity.

**Step 1: what can one core do?** Pushing E to 96 (404 KB read per work tile) finally makes even a
single core memory-bound:

| E | 8 | 16 | 32 | 64 | 96 |
|---|---|---|---|---|---|
| **G=1 per-core GB/s** | 6.63 | 10.69 | 18.79 | 34.82 | **50.67** |

One core sustains **≥50.7 GB/s** — and it is still climbing at E=96, so that is a floor on the solo
rate, not a ceiling. Now compare what a core gets when it has company (max over E, row-major
placement — the sweep's default, `CC(i) = {i%13, i/13}`):

| G | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| per-core GB/s | 50.67 | 39.55 | 20.93 | **11.13** | 7.21 | 4.82 | 2.83 | 1.90 |
| fraction of solo | 1.00 | 0.78 | 0.41 | **0.22** | 0.14 | 0.10 | 0.06 | 0.04 |
| aggregate | 50.7 | 79.1 | 83.7 | **89.1** | 115.3 | 154.3 | 181.3 | 243.3 |

**The aggregate plateaus at ~85 GB/s from G=4 to G=8** — a third of what DRAM can deliver — and only
resumes climbing after that. Eight cores each lose 78% of their solo rate to reach a total the memory
system could have served four times over.

**Step 2: the placement test.** `CC(i) = {i%13, i/13}` packs the first G cores into as few grid ROWS
as possible: G ≤ 13 is a single row. `TGD_PLACE=col` switches to `{i/10, i%10}`, which spreads the
same G cores down columns so they span as many rows as possible. Same cores, same count, same work,
same kernel — only which NoC links they share on the way to DRAM.

| G | rows used, row / col | aggregate row-major | aggregate col-major | **gain** | µs/tile at E=96 |
|---|---|---|---|---|---|
| 1 | 1 / 1 | 50.7 | 50.7 | 1.00× | 8.2 → 8.2 |
| 2 | 1 / 2 | 79.1 | 100.9 | 1.28× | 10.5 → 8.2 |
| 4 | 1 / 4 | 83.7 | 184.6 | **2.20×** | 19.8 → 9.0 |
| 8 | 1 / 8 | 89.1 | **215.2** | **2.42×** | 37.2 → 15.4 |
| 16 | 2 / 10 | 115.3 | 237.1 | 2.06× | 58.1 → 28.0 |
| 32 | 3 / 10 | 154.3 | 247.1 | 1.60× | 87.8 → 53.6 |
| 64 | 5 / 10 | 181.3 | 246.1 | 1.36× | 146.1 → 107.6 |
| 128 | 10 / 10 | 243.3 | 242.5 | 1.00× | 217.6 → 218.4 |

**Eight cores in a column reach 215 GB/s. Eight cores in a row reach 89.** Same eight cores, 2.42× the
bandwidth and 2.4× less latency, from nothing but where they sit. The two placements converge exactly
where they must — at G=128 both occupy all ten rows and both measure ~243.

**So the answer to "why does latency rise before the bandwidth ceiling":** because the first resource
to saturate is not DRAM's aggregate bandwidth, it is **the route the chosen cores share**. Pack cores
into one grid row and their reads funnel through that row's links, which top out around 85 GB/s. The
cores then queue on each other, the round trip stretches, per-core rate collapses from 50.7 to 11.1
GB/s, and per-tile latency rises — all while the memory system sits at ~35% utilisation. Utilisation
of the *bottleneck* is 100%; utilisation of the thing being measured is not.

This also explains the shape of `B(G)`: it is not a property of the memory system, it is the
row-major placement's route capacity growing as G spills into more rows. The measured B(G) table
above is therefore specific to that placement, and 8 well-placed cores beat 64 badly-placed ones
(215 vs 181 GB/s).

#### Rows are the unit — a shape test at fixed core count

`TGD_BLOCKW=W` lays the same G cores out as a W-wide block, so the core *count* is fixed and only the
*shape* changes. G=8, E=64 (276 KB read per work tile), T=32:

| block | rows spanned | µs/tile | aggregate GB/s | per core |
|---|---|---|---|---|
| 8 × 1 | 1 | 25.41 | 89.0 | 11.12 |
| 4 × 2 | 2 | 16.17 | 139.8 | 17.48 |
| 2 × 4 | 4 | 12.39 | 182.5 | 22.81 |
| 1 × 8 | 8 | **10.60** | **213.4** | 26.67 |

Monotone in rows spanned, 2.4× end to end, with the core count never changing. **The contended
resource is per-row.**

#### It is the interconnect, not the destination

The buffers are DRAM-interleaved over all 8 banks (`NUM_DRAM_BANKS=8`), so *every* core reads from
*every* bank regardless of where it sits. The set of destinations is therefore identical across all
four rows of that table — the only thing that differs is the route. That is what makes this an
on-chip interconnect effect rather than a DRAM-endpoint effect, and it is a deduction from the
interleaving rather than a direct observation of a link.

#### The layered picture — confirmed against the hardware specs

The per-row figure was measured before the spec was known, and it lands on it:

| level | measured | spec | ratio |
|---|---|---|---|
| one core, alone | **≥50.7 GB/s** | — | still climbing at E=96, so a floor |
| **one grid row, one direction** | **85–89 GB/s** | **86.4 GB/s** (64 B/cycle × 1.35 GHz) | **0.99–1.03** |
| whole chip | **245–253 GB/s** | 520 GB/s (8 channels × 65) | 0.47–0.49 |

Row-major saturation by core count: G=2 → 78.8 (91% of spec), G=4 → 83.6 (97%), G=8 → 89.0 (103%).
That agreement turns the earlier *deduction* — "it must be the route, because interleaving makes the
destinations identical" — into an identification: **the resource that saturates first is the NoC row
bandwidth in the return direction**, which is where a read's 4 KB payload travels. The ~3% overshoot
at G=8 is within the timebase assumption (§ *Confirm the timebase*) and the byte accounting.

#### The design rule this gives

A single grid row can never carry more than ~86 GB/s of read return traffic, no matter how many cores
are in it or how deep their requests. So:

$$\text{rows needed before DRAM can even become the limit} \;=\; \frac{520}{86.4} \;=\; \textbf{6 rows}$$

Below 6 rows the DRAM specification is irrelevant — the row is the constraint, and adding cores
*within* a row buys nothing. The full 10-row grid offers 864 GB/s of row capacity, comfortably above
DRAM's 520, which is why the two placements converge at G=128.

This is why the sweep's default row-major placement is the worst case: it packs the first 13 cores
into one row, so every configuration with G ≤ 13 was pinned at ~86 GB/s regardless of E.

#### One core saturates a whole row — the port and the row are the same link

Pushing E far enough finally saturates a *single* core (T=32, G=1):

| E | KB read per tile | µs/tile | GB/s | % of the 86.4 spec |
|---|---|---|---|---|
| 96 | 404 | 8.2 | 50.66 | 59% |
| 128 | 532 | 8.2 | 66.29 | 77% |
| 192 | 788 | 10.3 | 78.32 | 91% |
| 256 | 1044 | 13.4 | 79.90 | 92% |
| **384** | 1556 | 19.6 | **81.50** | **94%** |

**A single Tensix core reaches 81.5 GB/s — 94% of its 64 B/cycle inbound port.** Now put that beside
what a whole row achieves, at the same E:

| E | G=1 | G=2 | G=4 | G=8 (all one row) | G=128 |
|---|---|---|---|---|---|
| 96 | 50.7 | 79.1 | 83.7 | 89.1 | 242.5 |
| 192 | 78.3 | 82.8 | 84.3 | 89.6 | 242.1 |
| 384 | **81.5** | 84.3 | 84.7 | **89.4** | 240.1 |

**One core gets 81.5; eight cores in that row get 89.4.** Adding seven more cores to a row buys **10%**.

That is the sharpest form of the result: **a core's inbound port and a grid row's bandwidth are both
64 B/cycle, so one core is enough to saturate its own row.** Packing cores along a row is not merely
inefficient for DRAM traffic — it is very nearly pointless. Every core after the first in a row is
competing for a link its neighbour has already filled.

#### The practical rule: one reader per row

With column placement (one core per row until the rows run out):

| G, col-major | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| aggregate GB/s | 50.6 | 101.0 | 184.8 | 214.0 | 238.9 | **246.7** | 245.3 | 242.0 |

Linear to G=2, then rolling off, and **essentially flat from G=16**. Sixteen well-placed cores get
238.9 GB/s — 99% of what all 128 cores get (242.0), and more than the 181 GB/s that 64 row-major
cores manage. For a DRAM-bound stage the scheduling advice is therefore *spread over rows first, and
stop adding cores once the rows are covered*; cores beyond that contribute compute, not bandwidth.

#### A null result: the L1-destination and page-order fixes changed nothing

The reader was rewritten to follow the recommended access pattern — **32 distinct L1 landing slots**
instead of one shared address, and **consecutive page ids** instead of a prime stride, so tt-metal's
round-robin puts successive pages on successive channels. Both kernels, both placements, 80
configurations re-measured:

| G | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| row-major after / before | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.02 | 0.99 | 1.00 |
| col-major after / before | 1.00 | 1.00 | 1.00 | 0.99 | 1.01 | 1.00 | 1.00 | 1.00 |

**No effect anywhere.** Two reasons, in hindsight: the prime stride 613 is coprime to the channel
count, so it already visited every channel — just not in index order; and the single L1 destination
never throttled, because the reads were already to *distinct DRAM pages*, which is the half of the
rule that matters. The corrected reader is kept, since it is the pattern to copy, but **every number
recorded earlier in this document stands unchanged** — the harness was not measuring itself here.

#### Both NoCs — the recipe that reaches 429 GB/s

The ~245 GB/s plateau survived every placement (row, column, diagonal), every depth, and every core
count. It had one cause left: **the kernel was only driving one of the chip's two NoCs.** The reader
runs on NCRISC/NOC_1 and the writer on BRISC/NOC_0, so a kernel whose only reads live in the reader
leaves half the fabric idle no matter how well the cores are placed.

`TGD_EXTRA_W` gives the writer read traffic of its own, on NOC_0. Diagonal placement, T=32, reader
E=192 throughout:

| G | writer EW=0 | EW=96 | EW=192 |
|---|---|---|---|
| 8 | 215.6 | 312.4 | **395.8** |
| 16 | 234.8 | 333.3 | **433.0** |
| 32 | 227.8 | 333.7 | 423.3 |
| 64 | 236.8 | 346.9 | 419.4 |
| 128 | 241.9 | 354.7 | 394.7 |

**1.79×, from using the second NoC.** That was the whole missing factor — the single-NoC plateau of
~245 was almost exactly half of what the DRAM path can deliver.

The recipe, swept cleanly (diagonal placement, E=192 on NOC_1 + EW=192 on NOC_0, T=32, 1556 KB read
per work tile per core):

| G | 1 | 2 | 4 | 8 | 10 | **16** | 24 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|
| **GB/s** | 143.2 | 264.8 | 328.0 | 399.8 | 422.4 | **429.2** | 417.4 | 423.2 | 417.6 | 396.4 |
| per core | 143.2 | 132.4 | 82.0 | 50.0 | 42.2 | 26.8 | 17.4 | 13.2 | 6.5 | 3.1 |
| % of 520 spec | 28 | 51 | 63 | 77 | 81 | **83** | 80 | 81 | 80 | 76 |

**Peak 433 GB/s — 83% of the 520 GB/s specification, or 95% of 455 if the part runs 7 active
channels.** Two further things the table says:

* **A single core reaches 143 GB/s** using both its NoC ports — well past the 86.4 GB/s one port
  allows, which is the direct confirmation that the two NoCs are independent paths.
* **16 cores is the peak, and more cores make it worse** (128 → 396). Once DRAM is saturated,
  additional cores add contention and nothing else. This is the opposite of the intuition the
  row-major sweep gave, where bandwidth climbed monotonically to G=128 — that climb was rows being
  unlocked, not DRAM being fed.

#### One core per channel — what a channel actually delivers, and why 433 is the end of the road

tt-metal maps interleaved page *p* to DRAM bank `p % NUM_DRAM_BANKS`, so walking pages with **stride
equal to the bank count** keeps every read on one channel. `TGD_NBANK=8` pins each core to a channel;
`TGD_CHANMOD=1` puts *every* core on the same one, which is what saturates a single channel.

All cores on channel 0, reader only, E=192:

| cores | µs/tile | GB/s from that one channel | % of the 65 GB/s spec |
|---|---|---|---|
| 1 | 18.4 | 42.6 | 66% |
| 2 | 42.9 | 36.7 | 56% |
| 4 | 63.5 | 49.5 | 76% |
| 8 | 124.3 | 50.6 | 78% |
| 16 | 243.0 | 51.8 | 80% |
| 32 | 468.8 | 53.7 | 83% |
| **64** | 933.5 | **53.9** | **83%** |

And every channel, saturated the same way:

| channel | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **sum** |
|---|---|---|---|---|---|---|---|---|---|
| GB/s (G=64) | 53.9 | 53.9 | 53.8 | 54.1 | 55.1 | 55.3 | 54.8 | 55.1 | **436.0** |

Two conclusions, and they end the investigation:

* **All 8 channels are present and identical** — 53.8 to 55.3 GB/s, a 2.8% spread. Nothing is
  harvested on this part, so the "7 active channels" configuration does not apply here and `520` is
  the right nominal figure.
* **The sum of the eight channels' own saturated rates is 436.0 GB/s, and the two-NoC recipe
  measured 433.** That is **99.3%**. The recipe is not 83% of the machine — it is at the machine.

So the residual "17% gap to spec" was never ours to close. A channel's *nominal* 65 GB/s is a pin
rate; what it delivers for a scattered 4 KB-page read stream is ~54, and 8 × 54 is exactly what the
recipe achieves. **The right denominator for any future bandwidth claim on this part is ~436 GB/s,
not 520.**

Two smaller observations. A single core pinned to one channel gets 42.6 GB/s, so **one core cannot
saturate one channel** — it takes about four. And **two** cores on one channel measure *worse* than
one (36.7 against 42.6): a single core walks that channel's pages sequentially and keeps the row
buffer, while two interleaving streams thrash it. Not chased further.

Finally, channel pinning is a *measurement* device, not an optimisation: at G=16 over 8 channels it
delivers 189 GB/s against the 235 that ordinary round-robin interleaving gives, because pinning
concentrates each core's traffic on one endpoint instead of spreading it over eight.

#### Why bandwidth *falls* past G≈16 — it is a tail, not a slowdown

`GB/s` is bytes ÷ the dispatch span, and the span is **max over cores**: the dispatch is not finished
until the slowest core is. Adding the per-core spread to the parse shows what actually happens. Every
core reads the same 48.6 MB over the dispatch, so its own span implies its own service rate:

| G | fastest core | slowest core | unfairness | max/mean span | aggregate GB/s |
|---|---|---|---|---|---|
| 1 | 143.3 GB/s | 143.3 GB/s | 1.0× | 1.000 | 143.7 |
| 4 | 87.1 | 82.1 | 1.1× | 1.022 | 329.3 |
| 8 | 71.1 | 49.5 | 1.4× | 1.087 | 396.8 |
| **16** | 44.5 | 26.8 | 1.7× | 1.170 | **430.3** |
| 32 | 57.7 | 13.2 | 4.4× | 1.501 | 423.9 |
| 64 | 50.9 | 6.6 | 7.8× | 1.570 | 421.0 |
| 128 | 40.6 | **3.1** | **13.2×** | 1.628 | 396.3 |

**Identical work, 13× different service rates.** At G=128 the first core finishes its 48.6 MB in
1.25 ms and the last takes 16.5 ms. So the machine is fully loaded only at the start; through the
long tail progressively fewer cores are still reading, aggregate demand falls below what DRAM can
serve, and the average over the whole span comes out lower.

Two things this rules out. The memory system did **not** get slower — the *fastest* core at G=128
still runs at 40.6 GB/s, comparable to the 44.5 it manages at G=16. And it is not saturation
degrading: peak instantaneous bandwidth during the loaded phase must be **at or above** the 430 GB/s
that G=16 sustains uniformly; the high-G figures understate it.

What is *not* separated here is why the service is unfair — NoC/DRAM arbitration favouring some
requesters, or simply distance from the DRAM endpoints. The parse records only per-dispatch
aggregates, so attributing the spread to particular cores would need per-core output.

The practical consequence sharpens rule 5: for a DRAM-bound stage, fewer well-placed cores are better
for **two independent reasons** — DRAM is already saturated by ~16, and a wide dispatch develops a
tail that leaves the memory system idle while it drains.

#### The recipe, as a checklist

1. **Drive both NoCs.** Reader on NCRISC/NOC_1 *and* a second stream on BRISC/NOC_0. Worth 1.79×,
   the single largest factor found anywhere in this experiment.
2. **Spread over rows.** One reader per grid row; a row carries 86.4 GB/s and a single core already
   saturates it. Diagonal (distinct row *and* column) is marginally better than column-only at small
   G and identical past G≈4.
3. **Keep requests deep.** ~192 pages in flight per core per NoC. Below that the round trip is
   exposed; above it there is no further gain.
4. **Interleaved buffer, consecutive page ids**, so tt-metal's round-robin spreads successive pages
   over successive channels. Distinct L1 landing slots per outstanding page — though measured here,
   that last part changed nothing (see the null result above).
5. **Stop at ~16 cores** for the DRAM-bound part of a workload. Cores past that contribute compute,
   not bandwidth, and eventually cost bandwidth — both because DRAM is already saturated and because
   a wide dispatch finishes with a long, badly-balanced tail (13× spread in per-core service rate at
   G=128) during which the memory system is under-used.

#### What is still open

The 429–433 GB/s peak is **99.3% of the 436 GB/s that the eight channels deliver when each is
saturated individually** — see the per-channel section above, which settles this. The gap to the 520
GB/s nominal figure is DRAM read efficiency for a scattered 4 KB-page stream (a channel gives ~54 of
its nominal 65), not anything in the kernels or the schedule. All 8 channels are active and within
2.8% of each other.

Column contention turned out **not** to be the missing factor: diagonal placement (distinct row and
column) matches column placement from G≈4 onward, so the second NoC — not the column links — was
what the plateau was hiding.

Two caveats. Which physical links saturate was not measured — the experiment shows same-row cores
contend and cross-row cores do not, which is a statement about the grid, not about a named link.
And a real kernel cannot always choose its placement freely; it interacts with whatever else is
resident. But the lever is host-side, needs no kernel change, and this is the largest single effect
anywhere in the experiment.

### The model, refit

With the measured `B(G)` above rather than a constant, `C = max(C_compute, G·s/B(G))` predicts the
roofline cells to **1.5% median and 2.5–2.7% mean** for both kernels. The largest errors are all *at*
the knee, where neither term dominates and `max()` is too sharp a corner — the true transition is
smooth, as a queueing system's would be. Away from the corner the model holds across a 8× range of
memory pressure.

### Data files

| file | grid |
|---|---|
| `data/tg018dram_{modmul,fused}.csv` | the coarse sweep: G to 130, T to 64, `seq`+`batch`, both W policies |
| `data/tg018dram_grid_{modmul,fused,fused_ws1}.csv` | 2ⁿ grid, G to 64, T to 128, `seq` |
| `data/tg018dram_roof_{modmul,fused}.csv` | first roofline pass, T=32, E = 0…16 |
| `data/tg018dram_roof2_{modmul,fused}.csv` | T × E roofline with G = 104 |
| `data/tg018dram_pow2_{modmul,fused}.csv` | **the grid above** — 2ⁿ ladder with G = 128 |

Every row of every file carries all five per-RISC columns, the achieved GB/s and the mismatch count.

## Limits of this experiment, stated plainly

* **The per-RISC columns are envelopes.** Only `read` at T=1 and `us/tile` at G=1 are used as clean
  measurements; everywhere else they are read as "was this RISC resident" and nothing more.
* **The profiler adds a little to every dispatch.** `host ms` under `TT_METAL_DEVICE_PROFILER=1`
  carries the marker cost. It is small against 8–1000 µs kernels, but the honest host numbers come
  from a run without the profiler (the sweep still runs; the device columns go blank).
* **One prime, one direction.** The arithmetic is the same at every prime, so a second would only
  cost time — but this is a performance experiment and does not re-verify the scheme.
* **`bytes` counts what the kernel asks DRAM for**, including W tiles that 130 cores request from
  the same 16 pages. That broadcast has its own locality, so the fused `ws=0` GB/s figures flatter
  the memory system slightly relative to a fully scattered stream.
* **The knee is bracketed, not resolved.** G was swept at 1, 8, 32, 64, 104, 130; the fused `ws=0`
  transition sits between 64 and 104. The model puts it at 73. It was not measured at 73.

## Running it

```
. ./env.sh
TT_METAL_DEVICE_PROFILER=1 ./build/exp_tg018dram          # full sweep, both kernels
TT_METAL_DEVICE_PROFILER=1 TGD_KERNEL=modmul TGD_ITERS=3 ./build/exp_tg018dram
```

| env | default | meaning |
|---|---|---|
| `TGD_G`, `TGD_T` | `1,2,4,8,16,32,64,104,130` / `1,2,4,8,16,32,64` | the two sweep axes |
| `TGD_KERNEL` | `both` | `modmul` \| `fused` \| `both` |
| `TGD_MODE` | `both` | `seq` \| `batch` \| `both` |
| `TGD_WSHARE` | `both` | `0` \| `1` \| `both` (fused only) |
| `TGD_ITERS` | 5 | timed repetitions; the reported host wall is the min |
| `TGD_CSV` | `docs/he/data/tg018dram_sweep.csv` | row-per-configuration output |
| `TGD_NOVERIFY` | off | skip the golden check |

Run the two kernels as separate invocations. The join between profiler groups and configurations is
verified 1-1 and a single dropped marker invalidates the device columns for the **whole** run, so a
shorter run is a cheaper failure.

### How the device timing actually gets out

tt-metal clears its profiler log directory at device init and only writes the CSV at device
**close** — `ReadMeshDeviceProfilerResults` pulls markers off the device into host memory, it does
not flush them to disk. The sweep therefore calls it after every configuration (that drain is what
stops the on-device per-RISC buffer from overflowing and silently dropping markers, the failure that
forced exp_Tg018 onto a census estimate) and parses the whole log **once, after `mesh->close()`,
still inside the same process**. One binary, complete tables, no post-processing step.

Groups appear in the log in issue order and the sweep records one label per `EnqueueMeshWorkload` in
that same order, so group *i* is label *i*. If the counts disagree the device columns are dropped
with a diagnostic rather than misattributed — a shifted join would hand every configuration its
neighbour's numbers, which is worse than no numbers.

## New code

| file | what |
|---|---|
| `host/experiments/exp_tg018dram.cpp` | the sweep, the goldens, the profiler join, the tables |
| `kernels/experiments/exp_tb001/compute/fused_matmul_a_loop.cpp` | `fused_matmul_a` over n tiles. The one non-mechanical part: `unary_op_init_common` re-runs the DEST-sync handshake, which deadlocks MATH against PACK on Blackhole if re-run mid-kernel, so it is hoisted out of the loop and every per-iteration engine switch is light-weight only. |
| `kernels/experiments/exp_tb001/dataflow/reader_ab_depth.cpp` | two-operand reader, D tile pairs in flight. D=1 is `reader_modmul.cpp`. |
| `kernels/experiments/exp_tb001/dataflow/reader_fused_depth.cpp` | matmul feed, D work tiles in flight, `w_share` policy. D=1 with `w_share=0` is `reader_fused_batch.cpp`. |
| `kernels/experiments/exp_tb001/dataflow/writer_i32_depth.cpp` | int32 writer, D writes in flight. D=1 is `writer_i32.cpp`. |

The D=1 identities are deliberate: the baseline row of a sweep has to be the production kernel, not
an approximation of it. Groups never straddle a CB wrap because the host sizes every CB at 2D — the
rule `reader_gather_batch.cpp` already documents.

There is deliberately **no** `DeviceZoneScopedN` in the new compute kernel. The measurement uses the
per-RISC `*-KERNEL` envelopes tt-metal emits for every dispatch, so a named zone would buy nothing —
and it costs: tt-metal hashes zone source locations into 16 bits, and adding one more to this tree
collided ("Source location hashes are colliding") and aborted the device at close.
