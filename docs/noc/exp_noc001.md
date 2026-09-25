# exp_noc001 — DRAM against the NoC for an all-to-all, and what each costs as it scales

`Status: Current.` `[M-p150]` unless tagged otherwise. Bench `host/experiments/exp_noc001.cpp`, kernels
`kernels/experiments/exp_noc001/dataflow/`, driver `tools/bench/exp_noc001.sh`, data
[`data/noc001.csv`](data/noc001.csv). Every cell in every table is bit-exact against the same host
golden (`mism 0`); the barrier probe is the one deliberate exception and must fail its gate.

## 0. What it answers, and the headline

The question the project actually had was "should a collective go through DRAM or across the
fabric?" — prompted by the NTT corner turn, where moving it core-to-core took the stage from 23.97
to 4.44 µs ([`../matmul/exp_ntt001.md`](../matmul/exp_ntt001.md) §2b-sexies). Reading that as
*"the NoC is 5.4× faster than DRAM"* is the wrong lesson, and this experiment exists to show why.

Three results, in decreasing order of how much they change what we build:

1. **Most of that 5.4× was never about the medium.** The DRAM implementation read whole 4 KB pages
   to use 128 bytes of each. Fetching only the needed bytes — same medium, same memory controller —
   is **10.6–15.8× faster** and closes most of the gap on its own.
2. **At the corner turn's own parameters, strided DRAM beats the core-to-core exchange**: 2.70 µs
   against 4.23 µs, with no semaphores and no cross-core dependency at all. The NoC only wins beyond
   `G ≈ 45` participants.
3. **The NoC exchange is barrier-bound and issue-bound — never fabric-bound.** At `G=64` the payload
   takes 2.52 µs and the barrier 2.75 µs. Separately, §3b measures a single core sustaining
   **39.5 GB/s** across the fabric (~92% of the port rate) — so the exchange's 5.9 GB/s per core is
   not a fabric limit. Every NoC write costs a fixed **~20 cycles** against `MSG/32` cycles of
   transfer, so a 128 B message runs at **17% wire efficiency**. An all-to-all shrinks its own
   message as `G` grows, which penalises it twice over.

## 1. The application and the three routes

**Personalised all-to-all (block transpose).** `G` cores, core *n* owning `P` tiles. Conceptually a
`G × G` matrix of blocks with block `(n,s)` on core *n*; afterwards core *s* holds block `(n,s)` for
every *n*. Each core sends exactly its own data once, sliced `G` ways, so the *useful* traffic is
`G·P` tiles no matter how it is routed.

This is the communication skeleton of the corner turn `D[s][n1][t] = Cb[n1][s][t]`, with the tile
ROW generalised to a BLOCK. That generalisation is the point: in the real corner turn `G` is pinned
at 32 by tile geometry (32 rows) and the message is pinned at 128 B. Here

    block size = P·4096/G bytes

so `G` and the message size move independently and both become sweepable. It is *not* the corner
turn's exact face-level permutation — it is its traffic pattern, which is what a communication
benchmark should isolate.

| route | kernel | DRAM read traffic | requests per core |
|---|---|---|---|
| **A** `dram_bulk` | `a2a_dram_bulk.cpp` | `G·(G·P)` tiles — **`O(G²)`** | `G·P` × 4096 B |
| **B** `dram_strided` | `a2a_dram_strided.cpp` | `G·P` tiles — `O(G)` | `G` × `BLK` B |
| **C** `noc` | `a2a_noc.cpp` | `G·P` tiles — `O(G)` | `G` × `BLK` B, L1→L1 |

**There are three routes rather than two because a two-route comparison conflates two independent
effects.** Route A is the shape `all_to_all_corner_mc.cpp` has, and its `O(G²)` is not a property of
DRAM — it is a property of `noc_async_read_tile`, whose unit is a page. Route B uses
`addrgen.get_noc_addr(page, offset)` to fetch exactly the needed bytes and so moves **the same bytes
as route C**. Therefore:

- **A vs B** — same medium, different access pattern → isolates the `O(G²)` waste
- **B vs C** — **same bytes**, different medium → isolates DRAM against fabric

### 1.1 Fairness, stated explicitly

All three compute the identical permutation and every cell is gated against the same golden before
its timing is reported. All three end by writing the result to DRAM (`a2a_wb`), so the accounting is
identical. All three run one dataflow kernel on RISCV_1/NOC_1 — none of them is given both NoCs.

The one asymmetry is deliberate and is reported **both** ways. Route C must first have its own `P`
tiles in L1 (`a2a_ld`); routes A and B have no equivalent because their read *is* the exchange. In
the pipeline this models, that data is already resident from the previous stage — which is the
actual source of the win — so `a2a` alone is the number that transfers to a fused pipeline, and
`kernel_us` (`ld` + `a2a` + `wb`) is the like-for-like number. §2 quotes `a2a`; §4 quotes both.

## 2. Axis A — scaling participants (`P=1`, message `1024 → 64` B)

`a2a` zone, µs, critical-path convention (max over cores). `barr` is the barrier priced alone (§3).

| G | msg | bulk | strided | noc | ├ put | ├ sync | barr | bulk/strided | strided/noc |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 1024 B | 9.03 | **0.63** | 0.76 | 0.39 | 0.33 | 0.29 | 14.3× | 0.83 |
| 8 | 512 B | 11.18 | **0.87** | 1.06 | 0.53 | 0.52 | 0.51 | 12.9× | 0.82 |
| 16 | 256 B | 16.23 | **1.03** | 1.99 | 0.87 | 1.18 | 0.99 | 15.8× | 0.52 |
| 32 | 128 B | 31.85 | **2.18** | 3.13 | 1.42 | 1.65 | 1.57 | 14.6× | 0.70 |
| 64 | 64 B | 86.81 | 8.15 | **5.52** | 2.52 | 3.07 | 2.75 | 10.6× | **1.48** |

**Route A is 10.6–15.8× worse throughout, and that is the single biggest number in this experiment.**
It is also the one most likely to be misattributed: nothing about DRAM is being exercised badly here,
only the decision to fetch a page in order to use `1/G` of it.

**Route A's time grows far slower than its traffic, and the reason matters.** From `G=4` to `G=64`
its traffic grows 256× (`G²`) while its time grows only 9.6×. Its *achieved* bandwidth is what
closes the gap:

| G | 4 | 8 | 16 | 32 | 64 |
|---|--:|--:|--:|--:|--:|
| route A achieved GB/s | 7.3 | 23.4 | 64.6 | 131.7 | **193.3** |

At small `G` it is latency-bound and wasting bandwidth it was not using anyway; by `G=64` it reaches
44% of the 436 GB/s measured ceiling ([`../he/dram_guide.md`](../he/dram_guide.md)) and is genuinely
bandwidth-bound. **So the quadratic term only starts costing real time once the machine is busy
enough to feel it** — which is exactly when a workload is worth optimising, and exactly when a
small-scale test would have shown nothing.

**The crossover between B and C is at `G ≈ 45`.** Below it, strided DRAM wins; at `G=64` the fabric
wins by 1.48×. Route B degrades because its request shrinks to 64 B — the shape
[`dram_guide.md`](../he/dram_guide.md) §3.4 says DRAM likes least — while route C's cost is
dominated by a barrier that grows only linearly.

## 3. What the NoC route actually spends its time on

`a2a` is split into `a2a_put` (issue my `G` writes, flush) and `a2a_sync` (the `2G`-message barrier).
`a2a_sync` is not pure barrier overhead — `a2a_put` ends when *my* writes are flushed while peers'
may still be in flight, so `a2a_sync` also absorbs the slowest peer's tail. To separate them,
`NOC1_ROUTE=syncprobe` compiles the kernel with `-DA2A_NO_PAYLOAD`, deleting the payload writes and
keeping the barrier.

| G | `a2a_put` | `a2a_sync` | barrier alone | barrier as % of `a2a` |
|--:|--:|--:|--:|--:|
| 4 | 0.39 | 0.33 | 0.29 | 38% |
| 8 | 0.53 | 0.52 | 0.51 | 48% |
| 16 | 0.87 | 1.18 | 0.99 | 50% |
| 32 | 1.42 | 1.65 | 1.57 | 50% |
| 64 | 2.52 | 3.07 | 2.75 | **50%** |

The barrier alone accounts for essentially all of `a2a_sync` at every `G`, so **peers' tails are not
the cost — the barrier is.** It fits

    barrier µs = 0.224 + 0.0402·G          ⇒  ~20 ns per semaphore message

linearly in participants, and it is **half the exchange from `G=16` upward**.

**Consequence: the fabric wins the medium comparison and loses the route comparison.** Useful
bandwidth, payload only:

| | 128 B | 256 B | 512 B |
|---|--:|--:|--:|
| strided DRAM (`G=32`) | 60.1 GB/s | 121.9 | 121.1 |
| fabric `a2a_put` (`G=32`) | **92.3** | **183.3** | **190.0** |
| ratio | 1.54× | 1.50× | 1.57× |

At matched transfer size the fabric moves **~1.5× the bytes per second** the memory controller does.
Route C loses anyway below `G≈45` because the barrier costs more than that advantage buys.

**That 190 GB/s is not a fabric ceiling, and an earlier revision of this section wrongly called it
one.** It is 5.9 GB/s per core against a ~43 GB/s port, i.e. 14% — a property of the message size,
measured in §3b. Both media are far from their limits at these transfer sizes; the ratio above
compares the two routes as implemented, not the two fabrics.

### 3b. The fabric's actual limit, and why the exchange never approaches it

The 190 GB/s above invites the obvious objection: *the NoC should be much faster than that.* It is.
`noc_bw_probe.cpp` measures it directly — each core writes a fixed 16 KB into one peer's L1 in a
ring (no DRAM, no barrier, no permutation), sweeping only the message size, so the request count is
the only thing that changes. Data: [`data/noc001_bwprobe.csv`](data/noc001_bwprobe.csv). `[M-p150]`

| message | requests | cycles/req | of which wire | **overhead** | GB/s, best core | wire efficiency |
|--:|--:|--:|--:|--:|--:|--:|
| 64 B | 256 | 22.0 | 2 | 20.0 | 3.92 | **9%** |
| 128 B | 128 | 23.5 | 4 | 19.5 | 7.34 | **17%** |
| 256 B | 64 | 25.8 | 8 | 17.8 | 13.39 | 31% |
| 512 B | 32 | 30.9 | 16 | 14.9 | 22.39 | 52% |
| 4096 B | 4 | 208.0 | 128 | 80.0 | 26.58 | 62% |
| 16384 B | 1 | 560.0 | 512 | 48.0 | **39.50** | **91%** |

**The fabric is not the problem.** At 16 KB a single core sustains 39.5 GB/s — ~92% of the ~43 GB/s
a 32 B/clk port gives at 1.35 GHz. The wire does what it should.

**The currency is requests, not bytes.** Every write costs a fixed **~20 cycles** regardless of size,
against `MSG/32` cycles of actual transfer. So

    cycles per request ≈ 20 + MSG/32       ⇒  efficiency = (MSG/32) / (20 + MSG/32)

At 128 B — *the corner turn's message size* — you spend 20 cycles to move 4 cycles' worth of data.
**17% efficiency, and it is structural.** (Above ~1 KB the overhead grows again in steps, consistent
with the write being split into packets that each pay the fixed cost; inferred from the cycle counts,
not measured directly.)

**`noc_async_write_one_packet_set_state` does not help.** Programming the destination and size into
the command buffer once and reducing each write to two address stores changed nothing measurable
(0–3%, both directions, within run-to-run spread). So the ~20 cycles is a hardware/command-buffer
floor, not software overhead waiting to be optimised away. `[M-p150]`

**Why this bites an all-to-all specifically.** Slicing your data `G` ways *is* the algorithm, so the
message size is `P·4096/G` — it shrinks as participants grow. An all-to-all is therefore intrinsically
a small-message pattern, and this is a **second, independent penalty on top of the `O(G)` barrier**:
more participants means both more barrier and worse wire efficiency. At `G=64` the message is 64 B
and the fabric runs at 9%. The exchange still beat strided DRAM there only because DRAM at 64 B
requests is worse still.

**The measured per-core spread is 5.5×** (best 39.5, worst 6.8 GB/s at 16 KB). The ring's wrap hops
traverse a whole row, so the slow cores are the long-path ones — and a barrier-synchronised collective
pays the *worst* core, not the average. Placement matters as much as message size once messages are
large; at 128 B the spread collapses to 1.03× because everyone is issue-bound instead.

## 4. Axis B — scaling data at fixed core count (`G=32`)

`a2a` µs, then whole-kernel µs (`ld` + `a2a` + `wb`, the like-for-like figure).

| P | data | msg | bulk | strided | noc | | bulk | strided | noc |
|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|
| 1 | 128 KB | 128 B | 31.85 | **2.18** | 3.13 | | 32.32 | **2.70** | 4.23 |
| 2 | 256 KB | 256 B | 61.49 | **2.15** | 3.79 | | 62.01 | **3.04** | 5.16 |
| 4 | 512 KB | 512 B | 120.54 | **4.33** | 6.08 | | 121.16 | **5.01** | 8.00 |

**The `P=1 → P=2` cell is the most informative one here: twice the data in the same time** (2.18 →
2.15 µs for 128 → 256 KB). Doubling `P` doubles the volume *and* doubles the message, and the
message wins — DRAM's achieved rate goes 60.1 → 121.9 GB/s. Beyond 256 B it saturates (121.1 GB/s at
512 B), so the free doubling happens once.

Route A scales cleanly linearly in `P` (31.85 → 61.49 → 120.54) because its traffic is `G²·P` and it
is already bandwidth-bound at `G=32`.

## 5. Cross-validation, and a live consequence for the NTT

`G=32, P=1, 128 B` **is** the corner turn's parameter point — 32 tiles, one 128-byte row each. That
makes the two experiments checkable against each other:

| | exp_ntt001 §2b | exp_noc001 `G=32,P=1` |
|---|--:|--:|
| DRAM, whole pages, one NoC | 32.24 µs | 31.85 µs (route A) |
| core-to-core exchange | 4.44 µs | 4.23 µs (route C, `kernel_us`) |

Both agree to ~3% on independently written kernels, which is the strongest evidence available that
this benchmark reproduces the real stage rather than a simplification of it. `[M-p150]`

**And it makes route B a live optimisation we have not taken.** `[X]` The corner turn currently uses
the exchange at 4.44 µs. Route B at the same parameters is **2.70 µs — 1.64× better than the
exchange**, and it needs no semaphore, no full-grid launch constraint, and no core-to-core
dependency at all (the exchange requires exactly `NT` cores or peers hang;
`all_to_all_corner_x.cpp`'s header says so). The catch is real and must be checked before believing
the transfer: exp_ntt001's exchange writes into the receiver's *final face layout* so the receiver
assembles nothing, whereas a strided DRAM read returns rows that still need placing into face order —
the 8.74 µs of assembly the exchange eliminated. Whether a strided reader can address face offsets
directly on the DRAM side decides it. **Not a recommendation yet; a measurement worth taking.**

## 6. Traps this run hit

1. **A probe that "computes nothing" computes whatever was in L1 last time.** The barrier probe
   initially *passed* its correctness gate at `(G=32, P=4)`: CBs land at identical L1 addresses in
   every run, so the receive buffer still held the correct answer from the previous real run and the
   writeback shipped it to DRAM. Fixed by storing one `0xDEADBEEF` sentinel — after which the same
   cell reports `mism 32`, exactly 32 cores × 1 word, confirming the diagnosis. This is the second
   time in this repo a deliberately-wrong kernel passed its gate (`all_to_all_corner_probe.cpp` was
   the first, for the opposite reason — it wrote nothing at all). **A probe must be made to fail;
   absence of correct output is not failure.**
2. **`set -e` killed the sweep on the probe's exit code**, which is 1 *by design*. The driver now
   uses `|| true` on that one invocation and lets the explicit gate decide. Silent, because the
   script died mid-cell and printed nothing.
3. The profiler CSV lives under `$TT_METAL_HOME`, not the repo. A `find` rooted at the repo returns
   nothing and, without a check, the table sums to `0.00` — the fourth instance of this class in this
   session. `tools/bench/exp_noc001.sh` aborts on a missing or empty CSV.

## 7. Reproducing

```sh
sh tools/bench/exp_noc001.sh                    # full gated sweep -> docs/noc/data/noc001.csv
NOC1_G=32 NOC1_P=1 ./build/exp_noc001           # one cell, all routes, host wall
env TT_METAL_DEVICE_PROFILER=1 NOC1_PROF=1 NOC1_G=64 NOC1_ROUTE=noc ./build/exp_noc001
python3 tools/traces/zone_phases.py "$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
```

`NOC1_ROUTE=bulk|strided|noc|syncprobe|all`, `NOC1_G` (≤64, 8×8 grid), `NOC1_P`, `NOC1_ITERS`,
`NOC1_PROF`, `NOC1_CSV`. `G` must divide `P·1024` and the block must divide the tile; the bench
rejects other combinations rather than measuring a permutation it cannot express.

**Host wall is not usable here** and is printed only for completeness: each route is a single
dispatch, so wall (15–36 µs) is dominated by the dispatch floor and ranks the routes differently
from kernel time. Quote the zones.
