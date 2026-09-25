# exp_Bw001 — int32 elementwise multiply: one dispatch, 112 cores, traced

Status: Current. Measured 2026-08-07 on the **Blackhole p150b**, SDK
`v0.76.0-dev20260728`. Every number below is `min` of 5 blocking traced replays, all bit-exact
against a CPU golden (`0` mismatches in every row of every table).

Run it: `./build/exp_bw001 [NMAX=12] [NCORES=112] [W=64] [ITERS=5] [SWEEP_CORES=0]`

## What this measures

`C = A * B` over a `(2^n, NCORES, 32, 32)` int32 tensor, as **one dispatch**:

* `NCORES` cores, each owning one full 32×32 tile per iteration;
* the `2^n` axis is a sequential loop **inside** each core's kernel;
* the whole thing is a single program launch, then captured as a Metal Trace and replayed.

The multiply is the bare SFPU `mul_int32` — the raw wrapping low-32-bit product, with **no modular
reduction**. That makes it the only multiply kernel in this repo that is not Montgomery, and it is
deliberate: this is a machine-limit measurement, not an HE primitive.

## Yes, it is one dispatch

This was the open question. It is, and two things make it so:

1. **The `2^n` axis is a runtime arg**, so the sequential loop lives in the kernel. The host does not
   re-dispatch per iteration, and dispatch cost is therefore O(1) in the tensor size.
2. **The 112-core axis is a `CoreRangeSet` in ONE program** with per-core runtime args — not 112
   programs. This is not a stylistic choice: separate programs *serialize on the single command
   queue* no matter which core they target (`host/experiments/test_dispatch_micro.cpp` regime D measures exactly
   that), so co-scheduling requires one program with disjoint tile assignments.

112 is not a rectangle in a 13×10 grid, so the core set is a full 13×8 block plus a partial row of 8.

The measurement confirms the O(1) claim directly. Dispatch cost is a **constant ~15–25 µs**, so at
small `n` the traced replay is visibly cheaper than the untraced enqueue, and by `n=16` the two have
converged into the noise — which is the check that the traced wall really is the kernel time:

| n | untraced enqueue | traced replay |
|--:|--:|--:|
| 2 | 0.0848 ms | 0.0618 ms |
| 16 | 279.92 ms | 279.87 ms |

## Results at 112 cores

### A — streaming (every iteration reads a fresh tile pair from DRAM, writes C)

| n | 2^n | tiles | bytes moved | trace ms | GB/s | Gmul/s |
|--:|--:|--:|--:|--:|--:|--:|
| 4 | 16 | 1,792 | 21 MB | 0.1071 | 205.7 | 17.14 |
| 6 | 64 | 7,168 | 84 MB | 0.3157 | 279.0 | 23.25 |
| 8 | 256 | 28,672 | 336 MB | 1.1434 | 308.1 | 25.68 |
| 10 | 1,024 | 114,688 | 1.3 GB | 4.3846 | 321.4 | 26.78 |
| 12 | 4,096 | 458,752 | 5.2 GB | 17.543 | 321.3 | 26.78 |
| 14 | 16,384 | 1,835,008 | 21 GB | 70.156 | 321.4 | 26.78 |
| 16 | 65,536 | 7,340,032 | 84 GB | 279.87 | **322.3** | **26.86** |

Flat from `n≥10` at **~321 GB/s**. Time is linear in `2^n` across six doublings (70.16 → 140.23 →
279.87 ms), so the kernel-internal loop is genuinely executing and nothing is being elided.

### B — L1-resident (one tile pair in DST, 2^n multiplies, zero memory traffic)

| n | 2^n | trace ms | Gmul/s | mul/cycle/core |
|--:|--:|--:|--:|--:|
| 8 | 256 | 0.0954 | 307.9 | 2.0 |
| 10 | 1,024 | 0.2772 | 423.7 | 2.8 |
| 12 | 4,096 | 0.9682 | 485.2 | 3.2 |
| 14 | 16,384 | 3.6616 | 513.2 | 3.4 |
| 16 | 65,536 | 14.524 | **517.5** | **3.4** |

Asymptote **~3.4 int32 multiplies per cycle per core** (at 1.35 GHz). Below `n≈8` the numbers are
depressed by the ~40 µs trace-launch floor, not by the SFPU.

### The roofline

```
SFPU ceiling (L1-resident)      517.5 Gmul/s
achieved streaming from DRAM     26.9 Gmul/s  =  5.2% of ceiling
DRAM traffic sustaining it      322.3 GB/s
```

**Elementwise int32 multiply is hard memory-bound, by a factor of ~19.** It moves 12 bytes per
multiply (read A, read B, write C), which is about the worst arithmetic intensity an op can have, so
this is the expected shape rather than a defect — but it does say plainly that on this workload the
Tensix compute is idle 95% of the time and *any* optimization that is not about moving fewer bytes is
wasted effort. Fusing more arithmetic between the load and the store is the only lever that matters.

## Core scaling — where the bandwidth knee is

`SWEEP_CORES=1`, at `n=10`, 130-core grid, **constant DRAM footprint at every point** (see the
caveat below):

| cores | GB/s | GB/s/core | scaling | ideal |
|--:|--:|--:|--:|--:|
| 1 | 25.2 | 25.17 | 1.0x | 1x |
| 2 | 50.4 | 25.19 | 2.0x | 2x |
| 4 | 100.0 | 25.00 | 4.0x | 4x |
| 8 | 115.7 | 14.47 | 4.6x | 8x |
| 16 | 180.0 | 11.25 | 7.2x | 16x |
| 32 | 226.5 | 7.08 | 9.0x | 32x |
| 56 | 265.8 | 4.75 | 10.6x | 56x |
| 112 | 325.9 | 2.91 | 12.9x | 112x |
| 130 | 354.7 | 2.73 | 14.1x | 130x |

Two clean facts:

* **A single core sustains 25 GB/s** and scaling is *perfectly* linear to 4 cores (100 GB/s, 4.0x).
  That is a per-core NOC limit, not a memory-system one.
* **The knee is between 4 and 8 cores.** Past it, aggregate bandwidth keeps climbing but per-core
  throughput collapses: 130 cores deliver 14.1x one core, not 130x.

So for *this* op, 112 cores buys 12.9x over one core and 130 buys 14.1x — the last 118 cores
contribute about as much as the first 4 did. If the cores are wanted for anything else, an
elementwise int32 multiply will not miss them.

**Measurement caveat that changed the answer.** The first version of this sweep held `W` (the working
set in rows) fixed while varying core count, so the small-core points streamed a tiny footprint — at
`NC=1, W=64` only 256 KB, re-read 1,024 times. That re-hits the same DRAM pages and reported a
per-core bandwidth no real stream reaches, which faked a much sharper knee. `W` now scales inversely
with `NC` so `W*NC` is constant across the sweep. The numbers above are the corrected ones.

## Optimizations applied

All present in the measured configuration:

* batched NOC reads/writes — `BATCH` iterations of A and B in flight, retired by one barrier, instead
  of the read-barrier-push-per-tile pattern that pays full DRAM latency in series;
* `BATCH` tiles per DST acquire, amortizing the `tile_regs` handshake over 4 tiles;
* CBs `4*BATCH` slots deep so reader / compute / writer pipeline;
* reader and writer on separate RISCs and separate NOCs;
* `fp32_dest_acc_en` + `dst_full_sync_en` → 8 int32 DST tiles → `BATCH=4` pairs (the hard ceiling);
* one program over all cores; persistent `MeshWorkload`, never rebuilt between measurements;
* Metal Trace capture/replay.

## Trace discipline (the two traps hit while building this)

Both are recorded in `docs/pir/latency_optimization.md` and both bit here too:

* **Capture records but does not execute.** The warm enqueue must come *first* — the first enqueue
  writes the kernel binaries, and trace capture forbids writes.
* **Do not allocate while a trace is alive.** A program's circular buffers *are* L1 allocations, so
  building the next program while the previous trace is still captured triggers
  `Allocating device buffers is unsafe due to the existence of an active trace`. The benchmark builds,
  captures, replays and then **releases** each trace before the next build, which also keeps exactly
  one program's CBs resident — holding all 2·(NMAX+1) programs at once would ask ~2.4 MB/core of a
  1.46 MB L1.

## Correctness

`gold[i] = hA[i] * hB[i]` in wrapping uint32 on the host, compared elementwise. Elementwise ops are
layout-agnostic (A, B and C share one tile layout), so no tilize/untilize step is involved and the
golden is a flat product. Every row of every table above reports `0` mismatches, over both regimes,
every `n` from 0 to 16 and every core count from 1 to 130.
