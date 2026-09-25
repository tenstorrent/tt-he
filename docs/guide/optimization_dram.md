# Optimization — DRAM and NoC

**Status:** Current · One of three, split by where the time goes. Start at
[`optimization.md`](optimization.md) for the order of attack; measure with
[`profiling.md`](profiling.md) before and after.

| | |
|---|---|
| [`optimization_host.md`](optimization_host.md) | dispatch count, occupancy, batching, trace |
| [`optimization_dram.md`](optimization_dram.md) | read depth, bank spreading, NoC, the ceilings |
| [`optimization_kernel.md`](optimization_kernel.md) | SFPI passes, `sfpmul24`, DST traffic, registers |

---

**Diagnosis:** compute the kernel's achieved GB/s and compare against the right ceiling. Everything
here is measured by `host/experiments/exp_tg018dram.cpp`; the full guide is
[`../he/dram_guide.md`](../he/dram_guide.md).

### Check this first

**Interleaved page `p` maps to DRAM bank `p % NUM_DRAM_BANKS`.** A buffer holding **one** page lives
on **one** bank, so every core reading it hammers a single channel. Recognise it by achieved GB/s
sitting near a single-channel figure while the kernel "should" be chip-wide.

### The ceilings — a kernel is bound by whichever it hits first

| level | measured | nominal |
|---|--:|--:|
| one core, one NoC port | 81.5 GB/s | 86.4 |
| one core, **both** ports | 143.2 GB/s | 172.8 |
| one grid row, one direction | 89.0 GB/s | 86.4 |
| one DRAM channel | 53.9 GB/s | 65 |
| all 8 channels summed | **436.0 GB/s** | 520 |
| whole chip, achieved by the recipe | **433 GB/s** | — |

**The right denominator for any bandwidth claim on this part is 436 GB/s, not the 520 pin rate.**
Measured 433 is the machine, not 83% of it.

Two non-obvious consequences: a core's inbound port **and a grid row are the same 64 B/cycle link**
(one core reaches 81.5 of a row's 89.0 — the other seven cores buy 10%); and one core cannot saturate
one channel (42.6 of 53.9, it takes about four).

### The recipe — 433 GB/s vs 245 for the same kernel done the obvious way

| # | rule | worth |
|---|---|--:|
| 1 | **Drive both NoCs.** A reader on `RISCV_1` uses NOC_1, a writer on `RISCV_0` uses NOC_0. Reads only in the reader leave half the fabric idle | **1.79×** |
| 2 | **Spread over rows** — one reader per grid row; rows bind, columns do not | **2.42×** |
| 3 | **~192 pages in flight** per core per NoC (42.6 GB/s at 96, 78.3 at 192, 81.5 at 384) | — |
| 4 | **Interleaved buffer, consecutive page ids.** Never hard-code a channel address | — |
| 5 | **Stop at ~16 cores** for the DRAM-bound part | — |
| 6 | **Materialise constants, don't stream them** (scalar → runtime arg; shared → L1) | — |

> **Rule 2 inverts when the work is latency-bound.** It was measured at 32–64 tiles per core, where
> row capacity binds. Applied to a stage dispatching **one tile per core**, spreading is a **2.5×
> regression** (133.6 ms row-major vs 333.5 ms diagonal): with nothing to saturate, spreading only
> buys NoC distance, and a dispatch's span is a *max over cores* — the worst-placed core sets it.
>
> **Spread for bandwidth; pack for latency.**

Rule 5 has two independent causes: DRAM is already saturated at 16 cores, and a wide dispatch develops
a badly balanced tail — identical work, per-core service rates differing **13.2×** at G=128. Cores past
16 contribute compute, not bandwidth.

| G | 1 | 2 | 4 | 8 | **16** | 32 | 64 | 128 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| GB/s | 143.2 | 264.8 | 328.0 | 399.8 | **429.2** | 423.2 | 417.6 | 396.4 |

### Null results — each cost a run to disprove

| idea | result |
|---|---|
| Batching reads deeper behind one barrier (D=1 → 16–32) | **1.17–1.69× SLOWER** on both kernels. With 8–12 µs compute/tile a one-at-a-time reader already overlaps; batching converts a smooth request stream into bursts that queue against each other |
| Distinct L1 landing slots (1 → 32) | **No effect** over 80 configurations. The half of the rule that matters is distinct *DRAM pages* |
| Consecutive page ids instead of a prime stride | **No effect** — 613 is coprime to the channel count, so the stride already visited every channel |
| **Pinning cores to DRAM channels** | **Worse** — 189 vs 235 GB/s. It concentrates traffic on one endpoint. Pinning is a *measurement* device, not an optimisation |
| More cores past the knee | **Worse** — 433 → 396 from G=16 to G=128 |
| Two cores sharing one channel | **Worse than one** (36.7 vs 42.6) — two interleaved streams thrash the row buffer |
| Scalar constants as runtime args | Correct and free, but **0.3% end to end**. Take it for the 40 MB of DRAM it returns, not for speed |

---
