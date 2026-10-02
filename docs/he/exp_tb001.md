# exp_Tb001 — CROSS Set-B HE-multiply with evaluation form as the default ciphertext I/O

> **Hardware.** All numbers here were measured on **Blackhole p150**, 2026-08-04, on the code state
> that includes the batched reader and the fused reduce+twiddle
> ([`blackhole_port.md`](blackhole_port.md) §5). Not comparable with the Wormhole n150 figures in
> `history/`.

**What it is.** `host/experiments/exp_tb001.cpp` — the same algorithm and the same parameters as
[`he_mul_cross`](cross_comparison.md): CROSS's HE-Mult sub-kernel DAG at **N=8192, L=8 (LQ=8, 28-bit),
dnum=2 (alpha=4), P=4 → QP=12**. Tensor is pure pointwise, rescale is a real drop-prime done first,
key-switch is dnum digit-decomposed eval-in/eval-out.

**The one thing it changes.** The ciphertext I/O format is **evaluation form by default**.
`he_mul_cross` is untouched and remains the reference implementation.

## Workload terminology

A run of this experiment is described by three numbers:

| term | symbol | definition |
|---|---|---|
| **program core number** | `G` | the **hard core budget** for one program dispatch. It is given, and the program must be built to fit and *use* it. `NTT_CORES` sets it. |
| **batch group number** | `B` | how many copies of the same program run **at the same time on different cores** |
| **sequential number** | `S` | how many times a program runs **sequentially on the same cores** |

Identities:

```
ct_muls completed   = B * S
cores occupied      = G * B          (must be <= the grid: 130 on p150)
```

Worked from your example: `G=4, B=3` → 3 ct_muls in parallel on 12 cores, 4 cores each. Add `S=5` →
each group runs 5 in sequence, so **15 ct_muls total**, 3 groups × 5, each group on its own 4 cores.

### What a "program dispatch" is

**One program launch, counted as one unit.** One `emit(...)` in `host/experiments/exp_tb001.cpp` builds a
`Program`, wraps it in a `MeshWorkload` and enqueues it with `EnqueueMeshWorkload` (or, traced,
records it into a segment for replay). Two things inside it are deliberately **grouped away**:

- **the 3 kernels count as one.** Every dispatch always carries reader + compute + writer — reader on
  `RISCV_1` (logged `NCRISC`), writer on `RISCV_0` (logged `BRISC`), compute as the TRISC triplet
  (`TRISC_0` unpack / `TRISC_1` math / `TRISC_2` pack). They are one Program, one launch; the reader
  and writer exist only to feed and drain the compute.
- **the N cores count as one.** A dispatch names a `CoreRangeSet`; every core runs the *same* three
  kernels with *different runtime args* (different DRAM addresses), so core *i* handles job *i*. They
  start together and run in parallel with no inter-core dependency.

Measured, one ct_mul at the workload:

| level | count | |
|---|--:|---|
| program dispatches | **552** | the unit — all 8 cores wide |
| (dispatch, core) pairs | **4,416** | = 552 × 8 |
| RISC kernel envelopes | **22,080** | = 5 per pair; every dispatch carries all 5 RISC types |

The dispatch count cross-checks against the transform census: **57 forward + 19 inverse** transforms.
After the contraction fusion (§5) a forward NTT is 4 dispatches and an inverse 5, giving
`57×4 + 19×5 = 323`, plus 4 `bconv_one` + 4 FBC + 190 pointwise + 65 add + 42 sub − overlap = **552**.
Before the optimisations below it was 1266, then 1098.

#### The program uses 98% of its core budget

`G` is a hard constraint: given `G` cores, every dispatch may use up to `G` and the program should be
built to use them. Measured at the workload:

```
budget = G x critical path busy = 8 x 7.494 ms = 59.952 core-ms
UTILISATION                                    =   ~98 %

  552 dispatches use 8 of 8 cores  ->  0 idle
```

(The 97.7% figure was measured against core-time on the 1098-dispatch build; every dispatch is still
exactly 8 cores wide, so the property holds, but the ratio has not been re-measured since.)

**Every dispatch fills its budget**, because `G=8` matches the narrowest stage the decomposition
exposes. The four-step NTT at N=8192 = 8 blocks × 1024 has a 64-job stage and an 8-job stage, and
forward/inverse swap which is which:

| stage | jobs | dispatches at `G=8` | kernel |
|---|--:|--:|---|
| forward stage-1 | 64 (the 8×8 block contraction) | 8 | `fused_matmul_a` |
| forward stage-2 | 8 (one per block, shared 32×32 W) | 1 | `fused_matmul_c` |
| inverse stage-1 | 8 | 1 | `fused_matmul_a` |
| inverse stage-2 | 64 | 8 | `fused_matmul_c` |

The 8-job stages have 8 jobs because the decomposition exposes 8 tile-blocks (`NBLK = N2/T = 8`).
Setting `G` to that is what holds utilisation at 100%: a wider `G` cannot help those stages, it just
idles cores through them. **Throughput therefore comes from `B`, not from a wider `G`** — `G=8, B=4`
occupies 32 cores with all of them busy, which is why the workload is specified this way rather than as
one wide multiply.

#### Why grouping is correct, not just convenient

Each level has a different **reduction operator**, and the grouping is what turns *work* into
*latency*:

```
critical path busy  =   Σ           max          union
                     dispatches   over cores   over a core's RISC lanes
                     (serial)     (parallel)   (overlapped via circular buffers)
```

- **within a core** the 5 RISC lanes *overlap* — reader, compute and writer pipeline through circular
  buffers — so the core's busy interval is their **envelope**, never their sum;
- **across the N cores** of a dispatch there is no dependency and they overlap, so the dispatch costs
  the **max**, not the sum. Summing here is what produces *core-time*, a work total, not a latency;
- **across dispatches** execution is serial and non-overlapping (verified: 0 of 635 gaps ≤ 0), so the
  total is the **sum**.

So the two groupings you would apply are exactly levels 1 and 2, and applying them is what makes
`Σ max` the critical path rather than an over-count.

**Grouping is right for accounting, wrong for diagnosis.** At the grouped level a dispatch is just
"16 µs"; the reason is invisible. The batched-reader win was only findable *inside* level 1 — seeing
that the reader delivered tiles at 0.5 µs while the compute consumed them at 0.43 µs, i.e. that the
matmul phase was pacing off the reader rather than the matrix engine. Group to account, un-group to
explain.

**The grouping also rests on properties that could change.** If dispatches ever overlapped — a second
command queue, say — level 3's `Σ` would have to become a union, and the formula would be wrong.
`tools/traces/busy_metrics.py` checks the preconditions (group widths match the program's real widths;
no non-positive gaps) rather than assuming them.

**A program dispatch is not** a per-core kernel invocation (10,128 per ct_mul), not a RISC envelope
(50,640), not a trace segment (15 of those; one segment holds many dispatches, and the stalls
*between* segments are the 11.86 ms), not a buffer transfer (host↔device copies appear as
BRISC/NCRISC-only groups with no compute), and not a ct_mul (that is 552 dispatches).

### The workload: `G=8`, `B=4`, `S=1` — fixed

**Every latency figure in this document is `G=8`.** `G` is not a tuning knob here, it is the workload:
a number produced at any other `G` is not comparable, and none appear below.

`G=8` and `S=1` are set and measured. **`B=4` cannot be run today** — `B>1` needs the batching
implementation (per-group buffer sets, job lists interleaved so group *m* lands on cores
`m*G … m*G+G-1`, and a group index threaded through `dev_hemul_res`). It is not a flag. The measured
configuration is therefore **`G=8, B=1, S=1`** — 8 of the 130 cores occupied, where `B=4` would occupy
32 — and every table says so.

One ct_mul is **552 program dispatches**, **every one exactly 8 cores wide**. This is after all six
optimisations recorded below; it was 1266 originally and 1098 before the contraction fusion:

| kernel | dispatches | |
|---|--:|---|
| `modmul_sfpi_full` | 190 | pointwise, evk MAC **and both twiddles** (§6) |
| `transpose_tile` | 76 | one per transform |
| `modadd_map` | 65 | |
| `fusedAcc_total` | 57 | forward stage-1, all 8 partials fused (§5) |
| `fusedC_total` | 57 | forward stage-2 |
| `modsub_reduce` | 42 | |
| `fused_total` / `fusedCacc_total` / `modadd_reduce` | 19 each | inverse chain |
| `baseconv_fbc` | 4 | 2 mod-up + 2 mod-down |
| `bconv_one` | 4 | rescale (single-source) |
| **total** | **552** | |

### Two sub-metrics: host-side and trace-side kernel latency

Both are *per kernel program*, and neither measures "sending" the program — both measure it **running**.

| sub-metric | how | source |
|---|---|---|
| **host-side kernel latency** | host stopwatch around **one** program: enqueue it, `Finish`, stop. `TB001_KLAT=1` | host |
| **trace-side kernel latency** | that dispatch's `max over its cores`, from the trace | trace |

Measured at `G=8`, side by side. The dispatch counts are produced by two fully independent
measurements and agree exactly, which is the cross-check. **Taken before the two optimizations recorded
below** (hence 1266 dispatches); the per-kernel µs figures are unaffected, only counts and totals moved:

| kernel | dispatches | host-side ms | trace-side ms | ratio |
|---|--:|--:|--:|--:|
| `baseconv_fbc` / `fbc_total` | 8 | 0.4416 | **0.3860** | 1.14× |
| `modadd_twiddle` | 57 | 0.0719 | 0.0226 | 3.2× |
| `modmul_compute` | 33 | 0.0718 | 0.0165 | 4.4× |
| `modmul_sfpi_full` | 128 | 0.0734 | 0.0147 | 5.0× |
| `fused_matmul_c` / `fusedC_total` | 321 | 0.0714 | 0.0109 | 6.5× |
| `fused_matmul_a` / `fused_total` | 489 | 0.0733 | 0.0108 | 6.8× |
| `modadd_reduce` | 33 | 0.0719 | 0.0091 | 7.9× |
| `modadd_map` | 65 | 0.0745 | 0.0029 | 26× |
| `modsub_reduce` | 42 | 0.0742 | 0.0026 | 29× |
| `transpose_tile` | 90 | 0.0730 | 0.0017 | **43×** |
| **total** | **1266** | **95.076** | **16.330** | |

### Which numbers to compare — read this before quoting either

| question | use | at `G=8` |
|---|---|---|
| how much of the device time is actual kernel work? | **trace-side 11.99 µs** vs **host-measured 13.20 µs** per dispatch | differ by **1.21 µs** |
| which kernel owns the latency? | the per-kernel table below | FBC 18.8%, `fused_total` 32.4%, … |
| what does forcing a dispatch to be serial cost? | `TB001_KLAT` **~72 µs** | — |

- **13.58 µs** = critical path busy 7.494 ms / 552. **16.47 µs** = host device busy 9.089 ms / 552.
  The 1.21 µs between them is intra-dispatch stagger + intra-segment device idle + the 15
  blocking-call overheads amortised.
- **13.20 µs is an average, not a per-kernel figure.** On the normal path the host times **15
  segments**, never individual dispatches, so it cannot be attributed per kernel. Only the trace can.
- **`TB001_KLAT`'s ~72 µs and 93.8 ms total are not wrong — they measure a different thing:** the cost
  of a dispatch when it is *forced* to be serial. Useful for judging where a required `Finish` would
  hurt. Never quote them as the workload's latency; the shipped path pipelines ~98% of that away.

### Why host-side and trace-side differ so much

Splitting the host stopwatch into its two parts (`TB001_KLAT` reports both):

| kernel | trace-side | enqueue | Finish | host-side |
|---|--:|--:|--:|--:|
| `fused_matmul_a` | 0.0108 | 0.0276 | 0.0443 | 0.0719 |
| `fused_matmul_c` | 0.0109 | 0.0271 | 0.0433 | 0.0705 |
| `transpose_tile` | 0.0017 | 0.0275 | 0.0457 | 0.0733 |
| `baseconv_fbc` | **0.3860** | 0.0314 | **0.4068** | 0.4382 |

**Two constant costs the trace cannot see:**

1. **enqueue ≈ 28 µs** — the host writing the dispatch command over PCIe into the command queue. Dead
   constant (0.0259–0.0314 ms) across every kernel, independent of kernel and of core count.
2. **completion ≈ 25–33 µs** — worker cores finish → dispatch core writes a completion → host polls
   the completion queue and notices.

The trace-side figure is the **worker-core kernel envelope only**. The command's journey in and the
completion's journey out happen on the **dispatch cores and the host**, and profiler markers are
written by worker-core RISCs — so the trace is blind to both by construction.

The completion cost **overlaps** the kernel rather than adding to it: for fast kernels `Finish` is a
flat ~44 µs (the kernel ends long before the host notices), but for FBC `Finish` = 407 µs ≈ its 386 µs
kernel + ~21 µs. The 44 µs is *detection latency*, not work. So
`host-side ≈ enqueue (~28 µs) + max(kernel, poll floor ~44 µs)`.

**And nearly all of it is pipelined away in normal operation.** At `G=8`:

| | per dispatch |
|---|--:|
| trace-side kernel latency | **11.99 µs** |
| normal path (host device busy 9.089 ms / 552) | **16.47 µs** |
| serialised (`TB001_KLAT`) | **~72 µs** |

The normal path exposes only **2.66 µs** per dispatch, because the host enqueues all 552 programs
without waiting, so command writes and completion polling for dispatch *i* overlap kernel execution of
*i−1, i−2, …*. Pipelining hides **~98%** of the round trip.

**Consequence for optimisation — do not read the host/trace ratio as waste.** `transpose_tile`'s 43×
ratio is *serialised* overhead the normal path already hides; merging it recovers its **trace-side**
1.7 µs × 90 = **0.15 ms**, about 1% of the critical path, not 43× of anything. **Merge value = share
of critical path busy (trace-side).** The ratio only tells you how small a kernel is relative to a
serial round trip, which matters where you are *forced* to serialise — a data dependency requiring a
`Finish`, say.

`TB001_KLAT`'s 93.8 ms total is likewise **not** a latency: it serialises every dispatch and so runs
~5× the 17.9 ms host device busy of the same workload. Use it for per-kernel attribution only.

### `S` is a workload axis, `HE_ITERS` is a measurement knob — they are not the same thing

`HE_ITERS` repeats the whole multiply to get a stable average; it is a property of the *measurement*,
and the reported per-mul figures are divided by it. `S` is a property of the *workload*: how many
ct_muls a group is asked to do in sequence, which you are free to **optimise across** — pipeline them,
overlap one multiply's host work with the previous one's device work, hoist anything invariant out of
the loop.

Today they happen to coincide numerically and `S` buys nothing, because **nothing overlaps**:

- each multiply is self-contained — `hold3.clear(); WI=0; PI=0; seg_idx=0` at entry, no state carried;
- its tail runs **14 blocking `t_fin(cq)` drains and 112 blocking tile readbacks**
  (`rd_natE` reads with `blocking=true`, 7 limbs × 2 polynomials × 8 tiles);
- so multiply *i* is fully drained and read back before *i+1* starts.

Per-mul cost is therefore independent of how many you run, and **throughput is exactly 1/latency**.
Running `HE_ITERS=20` measures the same number twenty times; it does not exercise `S`.

### What the code supports today

| term | how it is set | today |
|---|---|---|
| `G` | `NTT_CORES` → `NCORES` | 4 / 8 / 16 / 32 / **64** (default); unset = whole grid |
| `S` | no workload knob — `HE_ITERS` only repeats the measurement | **effectively 1: no cross-multiply optimisation exists** |
| `B` | — | **always 1. Not implemented.** |

`B > 1` is the batching that [`cross_comparison.md`](cross_comparison.md) §7.4 planned and never
built: several independent multiplies interleaved into the **same** programs so one dispatch advances
all `B` groups. Launching `B` separate programs achieves nothing — dispatches serialise on the one
command queue.

### Latency vs throughput: which axis moves which

| goal | axis | what it attacks |
|---|---|---|
| **latency** (one multiply, start to finish) | fewer/cheaper dispatches | **critical path busy** — 7.494 ms of the 9.089 ms the host measures |
| **throughput** (multiplies per second) | `B` — spatial | dispatch count is set by `G` alone, so `B` groups share the same 552 and amortised cost per mul falls ~`1/B` |
| **throughput** | `S` — temporal, pipelined | the idle the device spends waiting on the host: the inter-segment stalls plus the blocking tail above |

Both throughput axes are unbuilt and they attack *different* parts of the metric split, so they
compose: `B` shrinks the amortised critical path busy per ct_mul, `S`-pipelining recovers host-stall
time that neither `G` nor `B` touches.

**Reporting rule when `B` or `S` is not 1:** the four latency metrics below are always quoted **per
ct_mul**, i.e. the measured value divided by the number of multiplies completed. State `G`, `B` and
`S` alongside any number — the same 7.494 ms means completely different things at `B=1` and `B=4`.

## Latency terminology

Four metrics, and only these four. **Two are measured on the host, two are derived from the trace** —
that pairing is the point: switching the profiler on perturbs wall time and busy span, but barely
touches host device busy (11.18–11.23 ms un-profiled vs 11.27–11.49 ms profiled, ~0.6%). So the host
pair anchors the trace pair, and the four together let you check a measurement at several levels
instead of trusting one number.

| term | source | definition |
|---|---|---|
| **wall time** | host | host end-to-end |
| **busy span** | trace | first cycle the device starts running → last cycle it stops. A window, so the idle between dispatches is still inside it. |
| **host device busy** | host | sum of the host's stopwatch around each **blocking** segment replay. Excludes host stalls *between* segments; includes device idle *inside* a segment plus ~52 µs/call of blocking-call overhead. |
| **critical path busy** | trace | busy span **minus the idle** = `Σ over dispatches of ( max over that dispatch's cores )` |

Guaranteed orderings, which the tool checks:

```
wall time  >=  busy span
wall time  >=  host device busy  >=  critical path busy
```

`busy span` vs `host device busy` is **not** ordered in general — the host figure drops the
inter-segment stalls but adds its own per-call overhead. Which one is larger tells you which of the
two dominates, so the comparison is itself informative.

Only host-side instruments can give wall time and host device busy; only the trace can give busy span
and critical path busy. All four must come from **one execution** or the orderings can appear
violated, which is why `HE_PROF2` measures its own profiled replay.

### What each metric does and does not include

Verified against the code, not assumed:

| | wall time | busy span | host device busy | critical path busy |
|---|:-:|:-:|:-:|:-:|
| kernel execution | ✓ | ✓ | ✓ | ✓ |
| on-device program launch/teardown | ✓ | ✓ | ✓ | ✓ |
| per-mul ciphertext I/O (32 `up_natE` uploads, 14 blocking `rd_natE` readbacks) + their host `tilize`/`untilize` | ✓ | partly¹ | ✗ | **✗** |
| host stalls between trace segments | ✓ | ✓ | ✗ | ✗ |
| **parameter / twiddle / folded-W / evk setup** | ✗ | ✗ | ✗ | ✗ |
| **CPU oracle (`cpu_hemul`)** | ✗ | ✗ | ✗ | ✗ |
| **the correctness comparison** | ✗ | ✗ | ✗ | ✗ |

¹ the readbacks are interleaved between dispatches, so the device idles during them and that idle
falls inside the span; the trailing readback is after the last kernel and so outside it.

**So nothing in any of the four metrics contains preprocessing, the oracle, or the correctness check.**
Constants (`bigWres`/`smallWres`/`twres`, `evk0f`, the eval-form inputs via `fwd8192`) are all built
once at setup, outside `dev_hemul_res()`; `cpu_hemul()` and the comparison run in the mode blocks after
it returns.

**Critical path busy is pure kernel time** — it counts only the 552 kernel dispatches, so the
ciphertext I/O is excluded entirely. **Wall time does include** that per-mul I/O and its host-side
format conversion, which is legitimate for an end-to-end figure but is also an implementation artifact
worth hoisting: the `tilize`/`untilize` per multiply is not part of the algorithm.

### Measured, at the workload

One execution, `G=8, B=1, S=1`, **552 dispatches**, verified 1:1 against the trace:

| term | source | value |
|---|---|--:|
| wall time | host | **21.6 – 24.1 ms** |
| busy span | trace | **not obtainable at `G=8`** — see below |
| host device busy | host | **8.65 – 9.09 ms** |
| critical path busy | trace | **7.494 ms** |

```sh
TT_METAL_DEVICE_PROFILER=1 HE_TRACE2=1 HE_PROF2=1 ./build/exp_tb001 0     # G=8 is the default
# the program prints the follow-up command with its own numbers filled in:
python3 tools/traces/busy_metrics.py <csv> --expect 552 --wall-ms 27.947 --host-busy-ms 8.703
```

The wall and host-busy in that command are from the **profiled** replay, which `HE_PROF2` makes
blocking in order to measure it. The table quotes the non-blocking replays at `HE_ITERS=20`, bit-exact
`0/114688`, and gives them as **ranges**: two runs of the same binary measured 24.096 / 9.089 and
21.554 / 8.649 ms. Host-side wall varies ~10% run to run, so quote it as a range or say which run.
**Critical path busy does not** — it is a sum of per-dispatch durations, immune to host-side scheduling,
which is why it is the number to optimise against.

Per dispatch: critical path busy **13.81 µs**, host-measured **16.47 µs** — 2.66 µs apart. The gap per
dispatch grew as the count fell (1084 → 552), which is what it should do: the fixed per-launch cost is
now spread over half as many, larger dispatches.

These five numbers replace the 1098-dispatch set this section used to quote; the changes that
moved them are in *Optimizations landed at this workload* §3–§5.

**Why the busy span is unavailable at `G=8`.** A span needs an undistorted timeline, and one is not
capturable at this workload. All dispatches land on the same 8 cores, so each core's profiler DRAM
buffer must hold 552 × (4 guaranteed + 2 named-zone) markers per TRISC — past the overflow point, and
tt-metal drops markers ("Profiler DRAM buffers were full"). `HE_PROF_DRAIN` keeps every marker but
inserts ~23 ms stalls that inflate a span by 40×. Critical path busy and core-time are **sums**, so the
drain does not corrupt them; a span is idle-sensitive, so it does. Getting one would need a build with
the outer `DeviceZoneScopedN` zones compiled out too, which would cost the per-kernel attribution.
Open item.

### Device timebase — every trace-derived number was 1.35× too high

The CSV column is **`time[cycles since reset]`: device cycles, not nanoseconds.** Cycles tick at
AICLK, measured by `tt-smi` on this p150 at **800 MHz idle and a sustained 1350 MHz under load**. Every tool in `tools/traces/` divided by 1000 and called the
result microseconds — i.e. assumed 1 GHz — overstating every device-side duration by **1.35×**.

The assumption is not just imprecise, it is provably impossible:

| assumed clock | trace critical path | host stopwatch | verdict |
|---|--:|--:|---|
| 1000 MHz | 17.766 ms | 14.492 ms | **impossible — exceeds the stopwatch that bracketed it** |
| 1350 MHz (measured) | **13.160 ms** | 14.492 ms | consistent (+1.33 ms of overhead) |

(That table is the 1098-dispatch measurement that *established* the clock; it is evidence for 1350 MHz,
not a current figure.)

A host stopwatch bracketing device work cannot come out shorter than the work. Both tools now take
`--clock-mhz` (default 1350).

**What this does and does not affect:**

- **Unaffected: every host-measured number.** Wall time, and the programs' `DEVICE-BUSY span`, are
  host stopwatches. The headline latencies in the README and `cross_comparison.md` are
  host-measured and stand as published.
- **Unaffected: every ratio and share.** "86% of compute is NTT/INTT matmul", "`ph_matmul` fell 82%",
  the core-count sweeps — all are ratios of trace numbers, so the scale factor cancels. The
  optimization conclusions in [`blackhole_port.md`](blackhole_port.md) §5 stand.
- **Affected: every absolute µs/ms taken from a trace.** Divide by 1.35. Corrected examples:

| quantity | as published (1 GHz) | corrected (1350 MHz) |
|---|--:|--:|
| `matmul-A` per call | 19.40 µs | **14.37 µs** |
| `matmul-C` per call | 18.39 µs | **13.62 µs** |
| FBC per call | 511.6 µs | **379.0 µs** |
| `reduce+twiddle` per call | 13.48 µs | **9.98 µs** |
| `ph_matmul` (after the reader fix) | 1.25 µs | **0.93 µs** |

### Two instruments, and what each can measure

| | host stopwatch | trace |
|---|---|---|
| what | `std::chrono` around each of the 15 blocking `replay_mesh_trace()` calls, summed | per-RISC cycle-stamped markers in the profiler CSV |
| clock | host CPU — immune to the cycles/ns issue above | device AICLK, needs the `--clock-mhz` conversion |
| where | `prof::g_replay`, printed as `DEVICE-BUSY span` | `tools/traces/busy_metrics.py` |

**Only the trace can give a busy span.** The host cannot see when the device started or stopped
running, only when it issued a call and when that call returned.

The host stopwatch is **not** the same thing as critical path busy — it sits above it. Decomposition
of one execution at the workload. **These figures are from the 1098-dispatch build** — kept because
they illustrate the decomposition, whose shape is unchanged; the current totals are 552 dispatches /
7.494 ms / 9.089 ms:

```
  critical path busy   (Σ per-dispatch max-over-cores)     13.160 ms
+ intra-dispatch stagger (cores not perfectly aligned)      0.119 ms
= device-side time attributable to the dispatches          13.279 ms

  host stopwatch (sum of 15 blocking replay walls)         14.492 ms
- device-side time above                                   13.279 ms
= host per-call overhead + intra-segment idle               1.213 ms   = 81 us/call over 15 calls
```

So the host figure **excludes** the host stalls *between* segments (hence it is not a span) but
**includes** intra-segment device idle plus its own per-call overhead. Good as a headline to track;
not the thing to optimise against.

**What the split tells you to work on.** Of the 14.492 ms the host measures, **13.160 ms is kernels** —
per-dispatch overhead is only 1.26 µs of 14.23 µs, so the dispatch path is no longer the problem. The
remaining host cost is the stalls **between** the 15 trace segments, which the host stopwatch does not
even count and which a `G=8` capture cannot size (no span available). Segment count is controllable:
segments break on every `t_wr`/`t_rd` through `prof::seg_hook`. `he_mul_cross` runs 46 segments;
eval-form I/O cut this experiment to 15. Hoisting the 32 per-multiply `up_natE` uploads out of the
multiply would cut it further — the same lever, not yet pulled.

Read the metrics as: **critical path busy** = kernel work to speed up; **host device busy − critical
path busy** = per-dispatch and intra-segment overhead; **wall − host device busy** = host work and
inter-segment stalls.

### Why critical path busy = Σ max-over-cores

The host dispatches **sequentially**; within a dispatch each participating core runs **one** kernel,
concurrently with the others; **dispatches do not overlap**. So the critical path is one term per
dispatch, and that term is its slowest core. Verified against the capture:

| check | result |
|---|---|
| do dispatches overlap? | **no** — 0 of 1,575 consecutive-group gaps ≤ 0; smallest gap 649 ns |
| group widths | **only ever exactly 8 cores** — the program's dispatch width at `G=8`. Overlapping dispatches would merge into mixed widths (16, 24, …); none exist. |
| are a dispatch's cores simultaneous? | yes — `group length − max-over-cores`: median **36 ns**, mean 159 ns, against ~20 µs programs (**<1%**) |

### The 1:1 requirement

Every host dispatch must appear in the trace exactly once with all 5 RISCs. The program prints its own
dispatch count (`prof::n_enq` / `WI`) so the check is against ground truth, not an estimate. Two
problems had to be solved to make it hold:

1. **Contamination.** `ReadMeshDeviceProfilerResults` rewrites the CSV from the host's *full
   accumulated* marker history on every call, so deleting the file beforehand does nothing — a capture
   holds **2016** dispatches for a 1266-dispatch multiply, measured before the optimizations below
     (setup + warm-up + the multiply).
   The profiled multiply runs last, so `busy_metrics.py` windows to the final N dispatches.
2. **Dropped markers.** Markers go to a per-RISC DRAM buffer that overflows; tt-metal then warns
   *"Profiler DRAM buffers were full, markers were dropped! … decrease the number of ops being
   profiled or run read device profiler more often"* (`profiler.cpp:1595`). A dispatch that lost its
   TRISC markers still shows BRISC/NCRISC, so it masquerades as a dispatch with no compute — that is
   what the 258 "no TRISC" groups were. Remedies tried: draining more often (`HE_DRAIN_K=20`) did
   **not** clear the warnings, and `TT_METAL_PROFILER_ACCUMULATE=1` made them **worse**. What holds is
   that the drops land in the setup/warm-up phase, so the final N come out complete — verified
   **1266/1266** then and **552/552** now, all with 5 RISCs. `busy_metrics.py` refuses to report if any windowed dispatch is
   incomplete, so a lossy capture can never be mistaken for a good one.

The relevant SDK limits are in `hostdevcommon/profiler_common.h`: `PROFILER_L1_OPTIONAL_MARKER_COUNT
= 250` per RISC plus 4 guaranteed markers, and a `DROPPED_ZONES` control slot. The `-FW`/`-KERNEL`
envelopes are the *guaranteed* markers; `DeviceZoneScopedN` zones are the *optional* ones, which is
why the inner `ph_*` zones are the first thing to cut if a capture needs to be smaller.

## Host-side optimisations — the four that landed

Quick reference. The ladder numbers below are **owned by**
[`history/OPTIMIZATION_REPORT.md`](history/OPTIMIZATION_REPORT.md) and were measured on **Wormhole
n150** at an earlier code state (102 trace segments / 2065 cached workloads, against 15 / 552 here).
They are quoted for the **mechanism and the relative attribution**, not as current figures — today's
number is 14.09 ms on p150.

| # | method | what it removes | effect | where it lives now |
|---|---|---|--:|---|
| 1 | **paramgen hoisted** | twiddle/fold **generation** per call — was **~85%** of the per-op host gap | 5044 → **771 ms** (6.5×) | `bigWres`/`smallWres`/`twres` built once at setup |
| 2 | **constant residency** | per-call constant **uploads**; each prime's folded constants sit in DRAM and transforms reference them | 771 → **727 ms** | the same resident DRAM buffers |
| 3 | **buffer residency** | the ~150 synchronous readbacks; intermediates stay in DRAM, transpose and elementwise move on-device | alone: **1881 ms — slower** | `dev_hemul_res`, `transpose8_dev`, `Buf8` intermediates |
| 4 | **program reuse + MeshTrace capture/replay** | the **build** (17–19%) and the **enqueue serialisation** (53–65%) | → **38 ms** device-busy | `CACHE_ON`/`WC` (reuse), `TR_PHASE`/`SEG`/`replay_mesh_trace` (trace) |

**Three things worth remembering, because none is obvious:**

- **Step 1 dwarfed step 2.** Regenerating twiddles was the cost, not uploading them — 6.5× versus a
  further 1.06×. The intuitive target (transfers) was the small one.
- **Step 3 alone made things *worse* (727 → 1881 ms), and was still essential.** Its real payoff is
  not readback avoidance: **MeshTrace bakes buffer addresses into the recorded command stream, so
  every buffer touched inside a trace must be allocated once and reused.** Residency is what makes
  step 4 legal. 3 and 4 are complementary, never alternatives.
- **Step 4 is what closes the host/trace gap.** It removes the host from the per-dispatch path
  entirely — hence per-dispatch host latency stops existing as a measurable quantity, and
  `host device busy` is measured over 15 *segments* rather than 552 dispatches. It is also why the
  ~72 µs serialised figure is an artifact of `TB001_KLAT` and not a cost the shipped path pays.

**All four are in the current code** — that is the difference between 5044 ms and 14.09 ms. What is
left is not per-dispatch overhead (only 1.21 µs of the 13.20 µs per dispatch), but the
**inter-segment host stalls**: a per-*segment* problem. Each host touch — an upload, a readback, host-side
FBC — splits the trace into another segment, so the levers are fewer host touches (hoist the 32
per-multiply `up_natE` uploads out of the multiply) and the two unbuilt throughput axes, `B` and
`S`-pipelining.

## Why this experiment exists

`he_mul_cross`'s resident path opens every multiply with `NTTin()`: it uploads the four input
polynomials in **coefficient** form and converts them on device — `4 polys × |qin|=8 limbs = 32
forward NTTs` — *inside* the traced, timed region. Two consequences:

1. It describes itself as "eval-resident, tensor = pointwise (0 NTT)" while paying a coeff→eval
   conversion on every call.
2. It is not a like-for-like comparison against CROSS, whose HE-Mult consumes eval-form ciphertexts;
   the coeff→eval step is encryption-side there.

Transform census per multiply (resident path):

| stage | fwd NTT | INTT |
|---|--:|--:|
| **`NTTin` — coeff→eval of the 4 inputs** | **32** | — |
| rescale, ×4 polys | 28 | 4 |
| `e2c` before key-switch | — | 7 |
| key-switch mod-up | 15 | — |
| mod-down, ×2 | 14 | 22 |
| **total** | **89** | **33** |

`NTTin` is **32 of 122 transforms — 26%**, and it is pure format conversion, not algebra.

## What exp_Tb001 does instead

The device-side canonical ciphertext is **natural-eval in the E layout**, uploaded directly with
`up_natE(x, /*mont=*/false, li)` — which lands exactly what `NTT_res(up_natC(x))` produced, same
layout, same non-Montgomery domain. The coeff→eval conversion happens **once at setup** on the host
path (`fwd8192`), outside the measured multiply, which is where CROSS puts it.

Correctness is unchanged and still absolute: the CPU oracle is computed from the same coefficient
polynomials, so `device-vs-oracle` validates the eval-form path end to end against exact big-int RNS
arithmetic. A mismatch between `up_natE` and the device transform would fail it.

## Results — the A/B, one binary, one card

`TB001_COEFF_IN=1` restores the coeff-input behaviour in the same binary, so the two differ in
nothing else. Traced-resident, `HE_ITERS=10`, at the workload (`G=8`):

Measured **before** the two optimizations recorded below, hence 1266 rather than 1098 dispatches; the
eval-vs-coeff conclusion is independent of them.

| ciphertext I/O | input NTTs | trace segments | dispatches | host device busy | correctness |
|---|--:|--:|--:|--:|---|
| **EVALUATION (default)** | **0** | **15** | **1266** | **18.174 ms** | `0/114688` |
| COEFFICIENT (`he_mul_cross` behaviour) | 32 | 46 | 1618 | 24.158 ms | `0/114688` |

**1618 − 1266 = 352 dispatches removed = exactly 32 NTTs × 11 dispatches each** — a forward NTT at
`G=8` is 11 dispatches (transpose + 8×stage-1 + fused reduce+twiddle + stage-2). The trace-segment
count falls 46 → 15 because the 32 uploads now run back to back with no program between them to force
a boundary. **1.33× on host device busy**, bit-exact both ways.

Canonical run, `sh tools/bench/exp_tb001.sh` (`HE_ITERS=20`, profiler on):

| metric | value |
|---|--:|
| wall time | **37.485 ms/mul** |
| host device busy | **18.603 ms/mul** (15 segment replays) |
| `device-vs-oracle` | **0/114688** |

## Where the time goes — per kernel, on the critical path

One ct_mul at the workload (`G=8, B=1, S=1`). All 552 program dispatches labelled by their compute
kernel; the `crit ms` column sums to the 7.494 ms critical path busy.

**"crit ms per dispatch"** is that kernel's critical-path contribution divided by its dispatch count —
the mean over its dispatches of `max over cores`. **Trace-derived** (device cycles at 1350 MHz), not
host device busy.

| compute kernel | what it is | dispatches | crit ms | **% crit** | µs per dispatch |
|---|---|--:|--:|--:|--:|
| **`modmul_sfpi_full`** | **Montgomery modmul — pointwise, evk MAC AND both twiddles** | **190** | **2.771** | **37.0%** | **14.6** |
| **`fusedAcc_total`** | **NTT stage-1, all 8 partials in one dispatch** | **57** | **2.038** | **27.2%** | **35.8** |
| **`fusedCacc_total`** | **inverse stage-2, all 8 partials in one dispatch** | **19** | **0.679** | **9.1%** | **35.7** |
| **`fbc_total`** | **base conversion, Ls=3/4** | **4** | 0.669 | **8.9%** | **167.3** |
| `fusedC_total` | forward stage-2 (never was a contraction) | 57 | 0.617 | 8.2% | 10.8 |
| `fused_total` | inverse stage-1 (never was a contraction) | 19 | 0.210 | 2.8% | 11.1 |
| `modadd_map` | vec add | 65 | 0.194 | 2.6% | 3.0 |
| `transpose_tile` | reindex | 76 | 0.135 | 1.8% | 1.8 |
| `modsub_reduce` | vec sub | 42 | 0.120 | 1.6% | 2.9 |
| **`bconv_one`** | **base conversion, Ls=1** | **4** | **0.031** | **0.4%** | **7.8** |
| `modadd_reduce` | vec add (inverse chain) | 19 | 0.029 | 0.4% | 1.5 |
| | | **552** | **7.494** | | |

`modadd_twiddle` and `modmul_compute` are absent: both were the same Montgomery multiply in LLK form
and now run on `modmul_sfpi_full` (§6), which is why it holds 190 dispatches and 37% of the path.

The four `fused*` rows are the same two NTT stages in both directions; `_acc` is the fused block
contraction (§5) and the plain names are the stages that were always one dispatch per output block.

### And by HE poly op — the view that matters

Joining the per-dispatch labels (`TB001_LABELS` → `dispatch_csv.py --labels`):

| he_op | dispatches | ms | % crit |
|---|--:|--:|--:|
| **keyswitch** | 311 | **4.472** | **59.7%** |
| rescale | 192 | 2.551 | 34.0% |
| tensor | 35 | 0.430 | 5.7% |
| combine | 14 | 0.041 | 0.5% |

The split barely moved — keyswitch 60.8% → 59.8%, rescale 34.1% → 34.0% — because the transform
fusion applies to every transform wherever it sits, so it took ~42% out of all three in proportion.

| he_phase | dispatches | ms | % crit | µs/disp |
|---|--:|--:|--:|--:|
| `rescale.ntt` | 112 | 1.765 | **23.6%** | 15.76 |
| `ks.moddown_tail` | 84 | 1.116 | 14.9% | 13.29 |
| `ks.modup_ntt` | 60 | 0.930 | 12.4% | 15.51 |
| `ks.evk_mac` | 88 | 0.784 | 10.5% | 8.91 |
| `ks.moddown_intt` | 40 | 0.519 | 6.9% | 12.96 |
| `ks.to_coeff` | 35 | 0.454 | 6.1% | 12.96 |
| `tensor.pointwise` | 35 | 0.430 | 5.7% | 12.29 |
| `rescale.scale` | 28 | 0.415 | 5.5% | 14.82 |
| **`ks.moddown_fbc`** | **2** | 0.347 | 4.6% | **173.74** |
| **`ks.modup_fbc`** | **2** | 0.321 | 4.3% | **160.51** |
| `rescale.intt` | 20 | 0.260 | 3.5% | 12.99 |
| `rescale.sub` | 28 | 0.080 | 1.1% | 2.87 |
| `combine.add` | 14 | 0.041 | 0.5% | 2.90 |
| **`rescale.fbc`** | **4** | **0.031** | **0.4%** | **7.83** |

**`tensor.montify` is absent from this table** — its 14 dispatches and 0.206 ms were removed outright
by §3, not made faster.

**Keyswitch is 60% of the critical path.** Transforms are still the bulk of it, but they are no longer
a long tail of ~11 µs dispatches: the contraction fusion turned 8 dispatches per output block into 1,
so a forward NTT is now 4 dispatches at ~16 µs where it was 11 at ~11 µs. The two remaining FBC phases
are **4 dispatches at ~167 µs = 0.668 ms, 8.8%** — down from 386 µs each (§4) but still 11× the cost of
any other single dispatch, so the kernel itself is still the lever there.

## ct_add at the same config

`TB001_OP=add` runs a **ct_add** through the same binary, the same kernels, the same trace path and the
same 1:1-verified metrics, so the two operations are directly comparable. Verified `0/131072` against
its own CPU oracle (16 output rows × 8192, vs the multiply's 14).

**What a ct_add is, and what it therefore costs.** Adding two ciphertexts at the same level is
`r_i = c_i + d_i mod q`, and that is all of it:

- **no rescale** — both operands are at the same level and the plaintext scale is unchanged;
- **no keyswitch** — the ciphertext degree does not grow, so there is nothing to relinearise;
- **no transform** — addition is `Z_q`-linear, so it commutes with the NTT and eval-form operands
  need no conversion. This is the one HE op where the eval-form default I/O costs nothing *and* saves
  nothing, because a coefficient-form add would have been just as free.

So the whole operation is **2 polynomials × 8 limbs = 16 modular adds** of 8192 coefficients. At `G=8`
one polynomial limb is 8 tiles on 8 cores = one program dispatch, so it is **16 dispatches**, all of
them `modadd_map`.

| term | source | ct_add | ct_mul | ratio |
|---|---|--:|--:|--:|
| program dispatches | host | **16** | 552 | 34.5× |
| **critical path busy** | trace | **0.047 ms** | 7.622 ms | **162×** |
| host device busy | host | 0.15 – 0.46 ms | 8.65 – 9.09 ms | — |
| wall time | host | 14.8 ms | 21.6 – 24.1 ms | — |
| transforms (NTT+INTT) | — | **0** | 76 | — |

The critical path figures are the comparable pair: **a ct_mul costs 162 ct_adds of device time** at this
parameter set. That ratio is the interesting number, and it is where the cost of the multiply's
machinery shows up — 76 transforms, 8 base conversions and the whole dnum=2 keyswitch, none of which an
add touches.

**Read the other two rows with care.** At 47 µs of device work the operation is far below the noise
floor of anything host-side:

- **host device busy is 3–10× the critical path busy**, against 1.15× for the multiply. This is not the
  add being slower than the trace says; it is that a blocking segment replay costs tens of µs of
  round-trip regardless of what it replays, and here that overhead is larger than the work. Its
  run-to-run spread (0.15 vs 0.46 ms across two runs of the same binary) is the same effect.
- **wall time is ~14.8 ms, 300× the device work.** The op is entirely host-bound: 16 enqueues, the
  operand uploads and 16 readbacks. Quoting 14.8 ms as "the latency of a ct_add" would be reporting the
  harness.

**The obvious optimisation, not taken.** `G=8` constrains *cores*, not tiles per core, and each of these
16 dispatches gives its 8 cores one tile each. All 16 limbs — 128 tiles — could go in **one** dispatch at
16 tiles per core. On the critical path that is worth only ~9 µs (15 launches × 0.60 µs, so 0.047 →
~0.038 ms), because the arithmetic is unchanged. What it would really cut is the **16 host enqueues → 1**,
which is where the wall actually goes. It needs a reader that takes several operand pairs per core plus a
loop in `modadd_map`, so it is a real change rather than an argument tweak, and it has not been made.

## Optimizations landed at this workload

Each was verified bit-exact `0/114688` against the CPU oracle, and each has an env flag that restores
the previous behaviour in the same binary for an A/B on one card.

| # | change | dispatches | critical path busy | |
|---|---|--:|--:|--:|
| 0 | baseline (eval-form I/O, `G=8`) | 1266 | 16.425 ms | — |
| 1 | mod-down keeps the Q limbs in eval | 1098 | 14.618 ms | 1.12× |
| 2 | single-source base conversion for rescale | 1098 | **13.160 ms** | **1.25×** |

### 1. Mod-down: only the P limbs leave the eval domain

`ks.moddown_intt` was 264 dispatches / 2.785 ms — 22 INTTs (2 accumulators × 11 QP limbs). But the FBC
only needs the **4 P limbs** as sources; the other 7 were converted purely so the subtract-and-scale
could happen in coeff before one NTT back.

`PinvC[li] = fill8(constant)` is a **scalar**, and the NTT is `Z_q`-linear, so

```
NTT( (ac − f) · Pinv )  =  Pinv · ( NTT(ac) − NTT(f) )  =  Pinv · ( acc_eval − NTT(f) )
```

and `acc_eval` is exactly what the evk accumulator already holds. So NTT the 7 corrections instead of
the 7 accumulator limbs, and subtract and scale in eval: `11 INTT + 7 NTT` → `4 INTT + 7 NTT`.

**Removed 14 INTTs = 168 dispatches = 1.807 ms** (predicted 1.77). `ks.moddown_intt` 264 → 96
dispatches, 2.785 → 1.014 ms; `ks.moddown_tail` unchanged at 182 dispatches, confirming the NTT moved
rather than the work being displaced. Host-side only, no kernel touched.

### 2. The rescale's base conversion had Ls=1 and was paying for the general case

FBC costs **~386 µs per dispatch regardless of source count** — Ls = 1, 3 and 4 measured
386.1 / 386.3 / 386.1 µs. The full CRNS machinery (per-source pre-scale, base-extension accumulation,
fixed-point overflow term, `k·g` correction) runs whether or not there is anything to accumulate.

Rescale converts from a **single** prime. There `S = q_L`, so `S/q_0 = 1` and the constants collapse to
`C1[0] = C2[0][j] = mont(1)` — the whole conversion is `out_j = x mod q_j`, an integer reduction into
each destination modulus. `kernels/experiments/exp_tb001/compute/bconv_one.cpp` does it as two branchless
conditional subtracts (`min(x, x−q)`, twice so the prime ordering does not matter).

| | dispatches | ms | µs/dispatch |
|---|--:|--:|--:|
| `bconv_one` (Ls=1) | 4 | **0.031** | **7.8** |
| `fbc_total` (Ls=3/4) | 4 | 1.545 | 386.3 |

**386 → 7.8 µs, 50× on the same conversion.** `rescale.fbc` fell from 10.6% of the critical path to
0.2%. `TB001_GENERIC_FBC=1` forces the general path back — both bit-exact.

Knock-on across both changes: host device busy **18.174 → 14.469 ms**, wall **38.822 → 27.229 ms** at
matched `HE_ITERS=10`. The wall gained more than the device did, because 168 fewer dispatches is also
168 fewer host enqueues.


### 3. The tensor's Montgomery pre-scale was a modmul that the rescale could absorb

The tensor product multiplies two rescaled polynomials, and one operand has to be pre-scaled by `R`
first (`mont_mul(a,b) = a·b·R⁻¹`, so one side must carry an extra `R`). That was 14 dispatches of
`tensor.montify`, 0.206 ms.

But the rescale already ends in a modmul by a constant — `pw_buf(diff, qLinvC[li], ...)`, the
`q_L⁻¹` scale. Multiplying by `mont(mont(q_L⁻¹))` instead of `mont(q_L⁻¹)` produces `mont(diff·q_L⁻¹)`
in the very same multiply:

```cpp
qLinvC[k]  = fill8(ctx[k].to_montgomery(qLinv));                          // scale
qLinvMC[k] = fill8(ctx[k].to_montgomery(ctx[k].to_montgomery(qLinv)));    // scale AND montify
...
pw_buf(diff, (poly>=2 ? qLinvMC[li] : qLinvC[li]), o, li);   // polys 2,3 = d0,d1 = the tensor's rhs
```

Only the two polynomials that become the tensor's right-hand side get the doubled constant, so the
left-hand side stays natural and the product is correct. **14 dispatches and 0.206 ms removed for one
extra host-side constant.** There is no flag for this one: it is strictly free, and reverting it would
mean re-adding a kernel that computes something a constant already encodes.

### 4. The base conversion looped over a padded width, not its actual one

`baseconv_fbc.cpp` took a single `n` and the host always passed `FBCW = 8`. The conversions it actually
performs are `Ls=4 → Ld=7` (mod-down), `Ls=4 → Ld=8` and `Ls=3 → Ld=8` (mod-up) — and the kernel's two
phases are `O(Ls)` and `O(Ls·Ld)`, so padding both to 8 computed **72 modmuls where 32 were needed.**
The padding was arithmetically harmless (`C1[i] = C2[i][j] = 0` past `Ls`, so the extra terms
contributed zero) which is exactly why it survived: nothing was wrong, only slow.

The kernel now takes `Ls` and `Ld` as separate args and loops each over its own bound, and the host
emits a compact constant block (`fbc_consts`) in the matching order.

| | dispatches | crit ms | µs/dispatch |
|---|--:|--:|--:|
| padded to 8×8 | 4 | 1.545 | 386.2 |
| actual widths | 4 | 0.668 | 167.0 |

**2.3× on the same conversion, −0.877 ms.** Ratio 0.43, against 32/72 = 0.44 of the modmuls — the cost
really is the `Ls×Ld` accumulation, so the saving lands where the operation count says it should.

**One trap this exposed, worth remembering.** `c_3` (the per-destination product buffer) is reserved
*inside* the phase-2 loop, `Ls` pages at a time, on a CB with 8 pages. While `Ls` was always 8 every
reserve began at page 0. At `Ls=4` the second reserve begins at page 4 and the third wraps — and
`pack_tile(dst, cb, i)` offsets from the write pointer **without wrapping**, so the tail tiles landed
outside the buffer. It failed as 4 wrong limbs out of 14, not as a crash. The fix is to size the CB to
the actual width so every reserve is page-aligned, which is what `build_mm_acc` and both FBC sites now
do. Any CB reserved repeatedly at a width that does not divide its depth has this bug.

### 5. The NTT block contraction: 8 dispatches per output block became 1

This is the big one. The four-step NTT at `N=8192` contracts 8 input blocks into each of 8 output
blocks. That ran as **64 independent dispatches**: each decomposed its input tile into four 7-bit
limbs, did 16 int8 tile-matmuls into four int32 planes, and then ran the **full recombine** — build the
64-bit value from the planes, Montgomery-reduce it, reduce mod q — to produce one partial residue. A
modadd tree then summed the 8 partials mod q.

The recombine is the expensive part (~30 SFPU ops against 16 int8 matmuls), and it was running 8 times
per output block to produce 8 values that were then immediately added together. But it is **Z-linear in
the planes**:

```
sum_p 128^p * plane_p  =  sum_a sum_n W_{a,n} * X_{a,n}       exactly, before any mod
```

because `sum_p 128^p · limb_p(y) = y`. So the planes can accumulate across partials — which the matrix
engine does for free, since `matmul_tiles` already accumulates into DEST — and the recombine runs
**once**, on the accumulated planes. The result is bit-identical, not an approximation.

`compute/fused_matmul_{a,c}_acc.cpp` take `NACC` inputs and `NACC×16` folded-W tiles from
`dataflow/reader_fused_acc.cpp`, and accumulate over both the limb index and the partial index:

```cpp
for (uint32_t p = 0; p < NB; ++p) {                     // one DEST accumulation per plane
    tile_regs_acquire();
    for (uint32_t a = 0; a < NACC; ++a)
        for (uint32_t r = 0; r < NB; ++r)
            matmul_tiles(cb_w, cb_limbs, a*NB*NB + p*NB + r, a*NB + r, 0);
    ...
}
```

**What does not ride along** is the final reduction. `low = plane0 + 128·plane1` scales with `NACC`, and
the single-partial kernel reduces it with exactly **two** conditional subtracts, because at `NACC=1`
`low < 2q`. The chain length is now computed on the host — `nsub = ⌊r_max/q⌋ + 1` with
`r_max = NACC·129·(4·32·127²) + q` — and its comparison had to be made valid over the full unsigned
range, because at `NACC=8` `r_max ≈ 2.40e9`, past `int32`. Two levels do that: a value with its top bit
set is unconditionally larger than a 28-bit `q`, so only below `2³¹` does the borrow test decide.

Everything else already tolerated a 32-bit `hi`: the peak intermediate is
`hi = plane2 + 128·plane3 ≈ 2.13e9`, and the Montgomery step needs only `hi·mont(2¹⁴) < q·2³²`, which
holds with 4× margin. **`NACC = 8` is therefore the ceiling and also exactly the block count**, so the
contraction collapses to one dispatch per output block and the reducer is left with a single partial.

| `TB001_ACC` | dispatches | crit ms | stage-1 µs/dispatch |
|--:|--:|--:|--:|
| 1 (one partial per dispatch) | 1084 | 12.148 | 11.0 |
| 4 | 628 | 8.286 | 21.6 |
| **8 (default)** | **552** | **7.622** | **35.7** |

**−4.53 ms, −37%, and 532 fewer dispatches.** Where it came from:

| | before | after |
|---|--:|--:|
| forward stage-1 | 475 disp / 5.224 ms | 57 disp / 2.037 ms |
| inverse stage-2 | 152 disp / ~1.68 ms | 19 disp / 0.679 ms |
| `modadd_twiddle` (8-wide → 1-wide reduce) | 1.301 ms | 0.931 ms |
| `modadd_reduce` (8-wide → 1-wide reduce) | 0.174 ms | 0.028 ms |

Note the per-dispatch cost went *up* 3.2× while the total fell 2.6× — 8 partials' worth of matmul in
one dispatch is the point. The residual 35.7 µs is now dominated by the 16 int8 matmuls × 8, which is
real arithmetic, not overhead.

`TB001_ACC=1` restores the per-partial path in the same binary, and `TB001_ACC=2/4` bisect it; all four
settings verify `0/114688`. This fusion had been attempted and abandoned once before; what
made it work here was recognising that the overflow constraint lives in the *final reduction's compare*,
not in the plane accumulation.

### 6. LLK tile API vs hand-written SFPI — measured, and what it is worth

Some compute kernels are written against the **LLK tile API** (`add_int_tile`, `mul_int_tile`,
`binary_min_uint32_tile`, …) and some against **SFPI** directly, as face functions over
`sfpi::dst_reg`. Six of the kernels the multiply dispatches were LLK-only, worth **1.633 ms, 21.4%** of
the critical path, so the question of what that style costs is a real one rather than a matter of taste.

There was already a controlled A/B in the profile: `modmul_compute` (LLK) and `modmul_sfpi_full` (SFPI)
compute the **same** one-tile Montgomery multiply, from the same reader, to the same writer, with
identical runtime args.

| | tile-wide passes | `_init` calls | µs/dispatch |
|---|--:|--:|--:|
| `modmul_compute` (LLK) | ~72 | one per helper call | **16.5** |
| `modmul_sfpi_full` (SFPI) | ~21 | 4 | **14.7** |

**LLK costs 11% on identical arithmetic.** Two reasons, and they are structural rather than incidental:

1. **Every helper re-inits.** `DUP`, `SHR`, `MUL`, `ADD`, `MIN` each call their `*_tile_init()` before
   the op, so a kernel with 72 ops does 72 reconfigurations.
2. **Every intermediate has to land in a DST tile.** LLK ops read and write whole tiles, so a value
   that must survive the next op needs an explicit copy — `modmul_compute` contains **17 `DUP`s whose
   only purpose is that**. An SFPI face function keeps intermediates in vector registers across the
   whole computation and touches DST once at each end.

It is only 11% and not 3× because what dominates both is the **emulated 32-bit multiply** — the SFPU has
no 32×32 primitive, so a 28-bit modmul is built from 16-bit partial products either way. Neither style
avoids that, and it is why `modmul_sfpi_full` costs 14.6 µs for one tile while `modadd_map`, the same
dataflow with no multiply, costs 3.0 µs.

**What was done with it.** After §5 the fused reduce+twiddle reduces a *single* partial, so
`modadd_twiddle` was no longer reducing anything — it was a Montgomery multiply wearing a reducer's
name, in LLK. `rtwiddle` (the inverse chain's twiddle) was the same multiply on `modmul_compute`. Both
now route to `modmul_sfpi_full`, which is byte-for-byte the operation they were emulating:

| | before | after |
|---|--:|--:|
| `modadd_twiddle` (forward twiddle) | 57 disp / 0.931 ms / 16.33 µs | — |
| `modmul_compute` (inverse twiddle) | 19 disp / 0.313 ms / 16.47 µs | — |
| `modmul_sfpi_full` | 114 disp / 1.678 ms / 14.72 µs | **190 disp / 2.771 ms / 14.58 µs** |

**−0.128 ms (−1.7%), 7.622 → 7.494 ms**, bit-exact, and two kernels dropped from the multiply's
footprint. The predicted saving from the 11% figure was
`(16.33−14.58)×57 + (16.47−14.58)×19 = 0.136 ms` against 0.128 measured — the model held.

**What is left of this lever is small.** The remaining LLK-only kernels are `modadd_map` (0.194 ms),
`transpose_tile` (0.135 ms), `bconv_one` (0.031 ms) and `modadd_reduce` (0.029 ms) — 0.389 ms, 5.2%.
None contains a multiply, and the per-dispatch floor is ~1.5 µs (`modadd_reduce`, essentially a copy and
a pack), so most of what they cost is dataflow and launch, not arithmetic. Collapsing `modadd_map`'s
add + conditional-subtract into one SFPI pass is worth an estimated **0.03–0.045 ms (~0.5%)**;
`transpose_tile` has no arithmetic to fuse. **The style is not where the remaining time is.**

**Where it actually is: `modmul_sfpi_full` is now 37.0% of the critical path**, 190 dispatches at
14.6 µs, of which ~11.6 µs is the emulated 32-bit multiply (by difference against `modadd_map`'s 3.0 µs
on the same dataflow). Halving the multiply count or finding a cheaper 28-bit reduction is worth more
than every remaining LLK conversion combined, by an order of magnitude.


### The six together

| after | dispatches | critical path busy | host device busy |
|---|--:|--:|--:|
| §1 mod-down reorder + §2 `bconv_one` | 1098 | 13.160 ms | 14.492 ms |
| §3 folded montify | 1084 | — | 14.203 ms |
| §4 FBC actual widths | 1084 | 12.148 ms | — |
| §5 contraction fusion (`ACC=8`) | 552 | 7.622 ms | 9.089 ms |
| §6 twiddles onto the SFPI modmul | **552** | **7.494 ms** | — |

**13.160 → 7.494 ms critical path busy, −43%, at 552 dispatches instead of 1098.** Every step is
bit-exact `0/114688` against the CPU oracle, and §4 and §5 each keep a flag that restores the previous
behaviour in the same binary.

### What is left

- **`modmul_sfpi_full` is now the #1 cost: 37.0%, 190 dispatches at 14.6 µs.** Two independent levers
  on it: **merge runs of it** (it is the same shape the reduce+twiddle fusion had before it was
  merged), and **make the multiply itself cheaper** — ~11.6 µs of those 14.6 is the emulated 32-bit
  multiply, and it is already near-minimal as schoolbook 16-bit partial products, so this means finding
  a different reduction rather than tightening the current one.
- **The 4 remaining FBC dispatches are 8.8% at 167 µs** and are genuine multi-source conversions, so
  neither the `Ls=1` collapse (§2) nor the width fix (§4) has anything left to give. Splitting
  `fbc_phase1` / `fbc_phase2` with `TB001_PHASE_ZONES` on a short capture would say which half to
  attack.
- **The forward twiddle is still 57 separate dispatches** even though §6 made each one cheaper. Folding
  it into the contraction kernel's pack stage would remove all 57.
- **The remaining LLK-only kernels are worth ~0.5% together** (§6) — not a lever.
- `B=4` batching is still unimplemented; it is now a much better bet than it was, because the dispatch
  count it would amortise is half what it was.

### Core-time and critical path agree at this workload

Every dispatch is 8 cores wide, so core-time is a **uniform ~7.8×** multiple of critical path busy and
the two rank kernels identically — measured on an earlier capture of this workload, shares agreed to
within 0.4 pp across the four biggest kernels, at ratios of 7.76–7.99. That relationship is structural
(one width, no partial-grid dispatches) and the contraction fusion did not change it, but the ratio
itself has not been re-measured since; the numbers to quote are the critical-path ones above.

It is a property of **this** workload, not a general truth. With mixed dispatch widths the two
diverge sharply — a kernel on 64 cores accumulates 8× the core-time of the same latency on 8 cores, so
core-time over-weights wide kernels and mis-ranks them. Fixing `G` removes that trap, which is one more
reason the workload is pinned. `tools/traces/trace_hemul.py` still prints a caveat, and it still applies
to any capture taken at a different `G`.

### Two checks that validate the table

The **8 FBC dispatches** are exactly the expected 4 (`bconv_one`, one per polynomial's rescale) + 2
(mod-up, one per digit) + 2 (mod-down, one per accumulator). And the **transform census reconciles**:
57 forward + 19 inverse transforms per multiply, which is what the four `fused*` dispatch counts say
independently — forward contributes 57 `fusedAcc_total` (stage 1, fused) and 57 `fusedC_total`
(stage 2), inverse contributes 19 `fused_total` (stage 1) and 19 `fusedCacc_total` (stage 2, fused).
Solving `8F + I = 475` and `F + 8I = 209` on the *pre-fusion* counts gives the same `F=57, I=19`, which
is a genuinely independent confirmation: the two censuses come from different captures with different
kernel counts.

The FBC figure is the "FBC wall" asserted in [`cross_comparison.md`](cross_comparison.md) §7.4
("baseconv caps at ~8 tile-positions") — quantified on the critical path rather than assumed. It cannot be widened within this
workload: `G=8` is the budget, so the lever is the FBC kernel itself, which is what §4 did.

## Running it

The per-kernel breakdown the bench script prints (via `trace_hemul.py`) is a **coarse orientation
view only**, and at `G=8` its zone counts are distorted by dropped markers — it can show ten times as
many instances of one kernel as another that ran more often. The per-kernel table above comes from a
`HE_PROF_DRAIN` capture put through `dispatch_csv.py`, which keeps every marker and verifies the 1:1
host↔trace map. Trust that one.

```bash
sh tools/bench/exp_tb001.sh 0            # traced-resident + trace breakdown (the canonical number)

./build/exp_tb001 0                      # plain run, eval input
TB001_COEFF_IN=1 ./build/exp_tb001 0     # plain run, coeff input (the A/B)
```

Every run prints the input domain and the resulting input-NTT count, so the property the experiment
is named for is visible in the output rather than assumed. All other env flags are `he_mul_cross`'s
and are documented in [`cross_comparison.md`](cross_comparison.md) §9.

## Independence

exp_Tb001 is self-contained: its own host source (`host/experiments/exp_tb001.cpp`), its own driver
(`tools/bench/exp_tb001.sh`), and **its own kernel tree** (`kernels/experiments/exp_tb001/`, 23 kernels, see the
README there). The target wires it up with the same mechanism `kernels/ir/` uses:

```
he_device_target(exp_tb001 SOURCES host/experiments/exp_tb001.cpp KERNELS kernels/experiments/exp_tb001 TEST hw)
```

**Why it matters:** `kernels/shared/` is shared by 76 device binaries — `he_mul_cross`, the whole NTT suite,
`lib/he`, `lib/pir`. Two of the kernels this experiment uses, `fused_matmul_a.cpp` and
`fused_matmul_c.cpp`, carry the mid-kernel re-init deadlock fix ([§2](blackhole_port.md)). A private
copy means a kernel idea tried here cannot take that out from under every NTT binary at once.

**Verified load-bearing, not a silent fallback.** With one private kernel temporarily moved aside the
run fails outright:

```
TT_THROW: Failed to generate binaries for modadd_twiddle
Cannot open kernel source file: .../kernels/experiments/exp_tb001/compute/modadd_twiddle.cpp
```

It looks only in the private tree. The 23 copies start byte-identical to `kernels/shared/`, so the first run
reproduces `he_mul_cross`'s arithmetic exactly and any later divergence is visible in a `cmp`; the
README there gives the one-liner.

## Relationship to the other Set-B programs

| binary | params | ciphertext domain | role |
|---|---|---|---|
| `he_mul_setb` | N=8192, L=8, dnum=2 | coeff-resident (tensor NTTs its inputs) | superseded; heavier than CROSS |
| `he_mul_cross` | N=8192, L=8, dnum=2 | eval internally, **coeff I/O** | reference implementation, unchanged |
| **`exp_tb001`** | N=8192, L=8, dnum=2 | **eval I/O (default)** | this experiment |

`bfv_ct_mul` is a different experiment — N=1024, Q=3, fixture-driven correctness — see
[`ct_mul_perf.md`](ct_mul_perf.md).
