# exp_roofline001 — a roofline for modmul, mul and add on Blackhole p150

**Status: DONE (measurement).** The six points asked for — three ops at both ends of the roofline —
plus a fourth op (`addp`) added for cross-hardware comparability, and the curve joining them all. Every one of the **36 points** bit-exact against a host
golden (`0` mismatches). **[M-p150]**

**The headline: the Montgomery modmul sits almost exactly on this machine's balance point.** Its ridge
is at `R* = 1.7` — so with **no data reuse it is memory-bound** (86% of the memory roof), and with *any*
reuse at all it becomes arithmetic-bound. That single number is why fusion, not kernel micro-optimisation,
has been the lever in every recent experiment.

`host/experiments/exp_roofline001.cpp` · `kernels/experiments/exp_roofline001/` · machine limits
[`exp_bw001.md`](exp_bw001.md) · the kernel measured [`../he/exp_modmul001.md`](../he/exp_modmul001.md)

---

## 1. The points

`q = 268435399` (28-bit), 32768 tiles = 33.55 M elements, 403 MB moved per pass, all 130 cores, one
dispatch, Metal Trace, **min of 5 replays**. Quote `min`
([`../guide/profiling.md`](../guide/profiling.md) §0).

| op | arithmetic bound *(full reuse)* | no reuse (`R=1`) | % of memory roof | % of its own roof | ridge `R*` |
|---|--:|--:|--:|--:|--:|
| **modmul** — Montgomery | **46.83 Gop/s** | **24.09 Gop/s** | 86.1 | **51.4** | **1.7** |
| **mul** — 32×32 → low 32 | **158.85 Gop/s** | **27.59 Gop/s** | 98.6 | 17.4 | 5.7 |
| **add** — modular | **530.04 Gop/s** | **27.88 Gop/s** | 99.6 | 5.3 | 18.9 |
| **addp** — plain 32-bit add | **1097.57 Gop/s** | **27.99 Gop/s** | **100.0** | 2.6 | 39.2 |

**Why four ops and not three.** `add` is a *modular* add (add + conditional subtract + select) because
that is what the HE pipeline issues. Elementwise rooflines on other hardware conventionally measure a
**plain** add, so `addp` was added to make that row like-for-like — and it matters: **2.07×**
(1097.57 against 530.04).

**Memory ceiling measured here: 335.9 GB/s** — 77% of the machine's 436, because both operand reads ride one NoC (§5b). That is **not** a lower bound: rebalancing the reads was tried and came out worse, since a read+write kernel has no spare dataflow RISC to make a second pure reader. At `R=1` the arithmetic intensity is `1/12` op/byte, so
the memory roof allows **27.99 Gop/s** — and `addp` achieves exactly that. A non-multiply op streaming
from DRAM is *perfectly* memory-bound, which is the check that the ops are being measured against the
feed rather than against each other.

## 2. How the two regimes are produced — one kernel, one argument

Each op is one SFPI pass, run `R` times per tile on operands already in DST:

```
R = 1        read A, read B, ONE op, write C          no reuse
R large      read A, read B, R ops, write C           memory amortised to nothing
intensity    = R * 1024 ops / (3 * 4096 bytes) = R/12 ops/byte
```

**`R` is literally the x-axis**, and nothing else changes between the two ends — same kernel, same
CBs, same feed. That is what makes the ends comparable.

The memory side applies `dram_guide.md`: `page = tile` so a core's run walks all **eight** banks
(`bank = page % 8`); `CHUNK=8` tile-pairs in flight behind one barrier (the depth rule); reads on
NCRISC/NOC_1 and writes on BRISC/NOC_0; one dispatch over 130 cores.

**It is not, however, at the machine's roof — §5b diagnoses why.** Putting *both* operand reads on one
NoC caps them at half the fabric. So the memory line drawn here is this feed's, ~77% of the part's.

## 3. The sweep

`Gop/s` counts **elements**, not instructions. `GB/s` is the 403 MB actually moved.

| R | AI op/B | modmul | GB/s | mul | GB/s | add | GB/s | addp | GB/s |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.083 | 24.09 | 289.0 | 27.59 | 331.1 | 27.88 | 334.5 | 27.99 | 335.9 |
| 2 | 0.167 | 41.87 | 251.2 | 54.37 | 326.2 | 55.70 | 334.2 | 55.08 | 330.5 |
| 4 | 0.333 | 44.17 | 132.5 | 96.55 | 289.6 | 110.22 | 330.7 | 109.73 | 329.2 |
| 8 | 0.667 | 45.41 | 68.1 | 146.34 | 219.5 | 213.65 | 320.5 | 218.68 | 328.0 |
| 16 | 1.333 | 46.14 | 34.6 | 152.16 | 114.1 | 395.88 | 296.9 | 407.48 | 305.6 |
| 32 | 2.667 | 46.50 | 17.4 | 155.53 | 58.3 | 492.29 | 184.6 | 785.93 | 294.7 |
| 64 | 5.333 | 46.61 | 8.7 | 157.15 | 29.5 | 510.57 | 95.7 | 1023.99 | 192.0 |
| 128 | 10.667 | 46.75 | 4.4 | 157.98 | 14.8 | 519.71 | 48.7 | 1058.67 | 99.2 |
| 256 | 21.333 | **46.79** | 2.2 | **158.41** | 7.4 | **524.82** | 24.6 | **1077.77** | 50.5 |

Read down a column and you see the roofline: `Gop/s` climbing while the op is memory-bound, then
flattening once it is arithmetic-bound. `modmul` flattens by `R=4`; `addp` is still
climbing at `R=128`, which is exactly what a ridge of 39 predicts. The four ops' ridges (1.7, 5.7,
18.9, 39.2) order the columns by how much reuse each needs before arithmetic binds.

## 4. The measurement discipline, because two things here are easy to get wrong

**The arithmetic bound is the MARGINAL cost between the two largest `R`, not a fit over the sweep.**
Both endpoints are deep in the compute-bound regime, so differencing them cancels the fixed term *and*
any residual memory time. A least-squares fit over all nine points is **wrong**, because at low `R` the
time is `max(memory, compute)` — not a line. The first version of this bench did fit all nine and
reported **`add` at 706 Gop/s against a true ~530**, with an 8.3% residual that was the tell. Recorded
rather than quietly amended.

**The linearity guard — is the compiler collapsing R identical passes?** The R loop is deliberately
*not* data-dependent (each iteration recomputes `d2 = op(d0,d1)`, as `exp_bw001`'s L1 kernel does) so
that it measures throughput rather than a dependent chain's latency, and so that the golden is the same
at every `R`. The risk is dead-code elimination. It does not happen, and it is **checked, not assumed**:
the marginal slope over `R=128..256` must agree with the slope over `R=64..128`.

| op | slope `R=128..256` | slope `R=64..128` | agreement |
|---|--:|--:|--:|
| modmul | 0.71680 | 0.71648 | **+0.04%** |
| mul | 0.21127 | 0.21122 | **+0.03%** |
| add | 0.06334 | 0.06339 | **−0.07%** |
| addp | 0.03057 | 0.03063 | **−0.20%** |

Four independent ops agreeing to a fifth of a percent or better. A collapsed loop could not do that.

**Spread.** Points at large `R` — the ones the ceilings come from — have spread **≤0.16%** (modmul 0.01,
mul 0.01, add 0.16, addp 0.08). The `R=1..4` points are ~1.2 ms and show host scheduling: 0.6–3.5% in
this run, and an earlier run recorded a **348%** outlier on one of five `mul` replays at `R=4`. `min` is
quoted throughout, which is what rule 3 requires, and the outlier is named rather than hidden — it does
not touch any ceiling.

## 5. What the ceilings mean

**The compute ceilings track the multiply count, and two independent kernels agree on the machine's
`sfpmul24` throughput:**

| op | `sfpmul24` per element | Gop/s | ⇒ G·sfpmul24/s |
|---|--:|--:|--:|
| modmul | 22 | 46.83 | **1030** |
| mul | 6 | 158.85 | **953** |
| add *(modular)* | 0 | 530.04 | — |
| addp *(plain)* | 0 | 1097.57 | — |

**≈1.0 T sfpmul24 lane-ops/s**, from two kernels that share nothing but the primitive. The
`mul : modmul` ratio is **3.39×** measured against **3.67×** predicted from 22/6 — the gap is the
non-multiply work (shifts, masks, the conditional subtract) that the modmul also carries.

**The two adds price the modular reduction on its own: 1097.57 → 530.04 is 2.07×.** A conditional
subtract plus a select costs as much as the add it guards, which is worth knowing wherever a kernel
can defer or batch its reductions — and it is the same "one conditional subtract is ~7%" effect
`exp_modmul002` §3 found inside the modmul, here isolated.

**Cross-checks against the repo's existing numbers:**

* `exp_modmul001` measured the same kernel at **3.086 µs/tile on one core**. Scaled to 130 cores that
  is 43.1 Gop/s against **46.83** here — 8.6% apart, in the right direction (here `a` and `b` stay in
  DST across the R loop, so the per-op DST traffic is lower).
* `exp_bw001` measured **322.3 GB/s**; here **335.9**. The agreement is not a coincidence and it is
  not the machine's ceiling either: `exp_bw001` is also a 2-read-1-write stream with both reads on one
  RISC, so it hits the same one-NoC plateau §5b identifies. That is why both land ~75% of
  `dram_guide.md`'s 433, which was measured with *both* RISCs reading.
* `exp_bw001`'s SFPU ceiling of **517.5 Gmul/s** is `mul_int_tile`, the *tile-op* API, which
  `kernels/OPTIMISED.md` records as issuing only `MOD1_LOWER` — one `sfpmul24`. This experiment's
  SFPI path sustains ~1.0 T/s of the same instruction, i.e. **about 2× what the tile-op API extracts
  from it.** That is the same gap `exp_modmul001` found and is the mechanical reason the one-pass
  SFPI kernels win.

**The core sweep shows this feed's roof is reached** (`R=1`, `add`) — reached, but see §5b for why it
is 334.6 rather than 436:

| cores | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 130 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| GB/s | 25.0 | 49.7 | 98.8 | 133.6 | 181.5 | 230.9 | 249.5 | **334.6** |
| GB/s/core | 24.97 | 24.85 | 24.71 | 16.71 | 11.34 | 7.22 | 3.90 | 2.57 |

Perfectly linear to 4 cores at ~25 GB/s each, then bandwidth-limited — matching `exp_bw001` §"cores".

## 5b. The memory roof here is ~335 GB/s, not the machine's 436 — diagnosed

> **On the two numbers.** §1 quotes **335.9 GB/s** and this section's tables **334.6** — the same
> measurement in two different runs, 0.4% apart. Each table below is an internally paired comparison
> (default vs split, default vs probe) taken in one session, which is what makes those deltas valid;
> they are not mixed with §1's run.

**This is a conservatism in the harness, not a property of the ops, and it moves the roofline's memory
line.** `dram_guide.md` §1 is explicit that **436 GB/s is the right denominator** for this part (520 is
a pin rate) and records **433 GB/s achieved**. This experiment reaches **334.6 = 77%**. Two hypotheses
were tested; the first was wrong. **[M-p150]**

**Not read depth.** `dram_guide.md` §3.3 measures a single core needing ~192-384 pages outstanding;
this reader had 16. Sweeping it changed nothing:

| CHUNK (tile-pairs) | 4 | 8 | 16 | 32 |
|---|--:|--:|--:|--:|
| pages outstanding/core | 8 | 16 | 32 | 64 |
| GB/s | 335.5 | **338.1** | 326.4 | 328.8 |

Flat, and mildly *worse* deeper. **The depth rule is a few-core rule**: at 130 cores the aggregate is
already ~2000 pages in flight, so per-core depth stops mattering. Recorded because the guide states
that curve per core and it is easy to over-apply.

**It is the reads being on ONE NoC.** `writer_probe.cpp` deletes the DRAM write (gate must fail, §5c)
so only the two reads remain:

| | read GB/s | write GB/s | total |
|---|--:|--:|--:|
| full kernel (2 reads NOC_1, 1 write NOC_0) | 223.1 | 111.5 | **334.6** |
| reads only (probe) | **237.5** | 0 | 237.5 |

**The read stream saturates at ~237 GB/s whether or not writes are present** — so this is not
read/write turnaround. And `dram_guide.md` §3.1's table has the confirming row measured independently
two experiments ago: **"none (NOC_1 only) = 234.8 GB/s"**, against 433.0 once the writer's RISC is
given read work too. **234.8 there, 237.5 here — 1.2% apart.** Two experiments, different kernels,
different core counts, same plateau: one NoC is worth about half the fabric, "the single largest factor
found anywhere". The reader puts both operand streams on NCRISC/NOC_1; NOC_0 carries only the write
and idles at 111 GB/s.

**Is it too many cores?  No — tested, and the curve is still climbing.** `dram_guide.md` §3.5 records
a real knee: the DRAM-bound part peaks at **G=16 (429.2 GB/s)** and *declines* to 396.4 at G=128, from
DRAM saturation plus a badly balanced tail. So G=130 is a natural suspect. It is not the cause here:

| G | 16 | 32 | 130 |
|---|--:|--:|--:|
| this experiment, GB/s | 181.4 | 239.6 | **335.5** |
| GB/s **per core** | 11.3 | 7.5 | 2.6 |
| `dram_guide` §3.5 at the same G | 429.2 | 423.2 | ~396 (G=128) |

**This feed rises monotonically to 130 cores** — it never reaches the regime where more cores hurt.
`dram_guide`'s knee appears only once DRAM is genuinely saturated at ~429 GB/s; at 11.3 GB/s per core
this feed is starved, so every extra core still buys throughput. **Too many cores is a real effect on
this part and it is not what is happening here.**

**And more read depth makes it WORSE, at every core count** — the opposite of §3.3:

| CHUNK (tile-pairs) | 4 | 8 | 16 | 32 |
|---|--:|--:|--:|--:|
| G=16 | **181.4** | 180.3 | 169.7 | 165.1 |
| G=32 | **239.6** | 228.4 | 219.0 | 210.1 |
| G=130 | 335.5 | **338.1** | 326.4 | 328.8 |

**Because "depth" here is a synchronous BATCH, not a pipeline.** The reader reserves CHUNK pairs,
issues 2·CHUNK reads, barriers, then pushes — so the compute kernel cannot start until the whole batch
lands, and a bigger batch trades reader/compute overlap for outstanding pages. §3.3's curve was
measured on a pure-read kernel with no consumer, where outstanding pages are free. **The depth rule
transfers only if the reads are genuinely in flight against a running consumer**; as a batch it can be
net negative. That is a correction to how this repo has been applying §3.3, not just to this bench.

**The obvious fix was tried and it made things WORSE.** `ROOF_SPLIT=1` moves operand B's reads to
BRISC/NOC_0 alongside the write (`reader_a_deep.cpp` + `rw_b_deep.cpp`), so each NoC carries 4 B/op
inbound instead of 8 and 0. All points `0` mismatches; the arithmetic bounds are unchanged
(46.83 / 158.88 / 529.10 against 46.81 / 158.82 / 529.72), so only the feed moved. **[M-p150]**

| `R=1` | default: 1 reader + 1 writer | split: BRISC reads B *and* writes |
|---|--:|--:|
| add | **334.6 GB/s** | 314.9 |
| mul | **327.0** | 300.9 |
| modmul | **288.3** | 252.3 |

**Why it loses.** The constraint is not NoC bandwidth — it is that **there are only two dataflow RISCs**.
The default already has both busy: NCRISC reading, BRISC writing, concurrently. The split turns that
into one RISC doing half the reads and one RISC alternating read-barrier / write-barrier, and the
serialisation on BRISC costs more than the NoC balance buys.

`exp_dram001` reached 445.7 GB/s with reads on both NoCs, but **both RISCs were pure readers there** —
no writes to interleave. A read+write kernel cannot have that: it needs a writer, and there is no third
dataflow RISC. So for a streaming elementwise op on this part:

* **1 reader + 1 writer** — reads capped by one RISC on one NoC, total **334.6 GB/s**
* **split** — better NoC balance, but a serialised BRISC, **314.9**

**334.6 GB/s is therefore close to the practical ceiling for a 2-read-1-write kernel**, not a defect to
be engineered away, and the `R=1` figures in §1 and §3 stand as measured rather than as lower bounds.

> **This retracts a projection made earlier in this same document.** §5b originally concluded that
> ~1.30x was "available from the feed alone" and that §6's "only fusion can reach it" was wrong. That
> was a cost model built from a correct diagnosis (the one-NoC wall is real) plus an untested inference
> about the fix — **exactly the failure `docs/matmul/exp_ntt001.md` records three times.** Measuring it
> refuted it in one run. §6's original claim was right.

## 5c. A gate that failed to fail

`writer_probe.cpp` must produce a wrong answer, and for `modmul` and `mul` it did — `33421312` of
`33554432`, the shortfall being exactly `130 x 1024`, the one tile per core the probe still writes.
That is the mechanism working.

For `add` the gate reported **`0` mismatches**, which is impossible for a kernel that writes 130 of
32768 tiles. Cause: the *previous run* of this bench left `add`'s correct golden in the output buffer,
the allocator returned the same DRAM address, and nothing zeroes DRAM between processes. **A stale
buffer can make a probe look correct.** Timings are unaffected (they do not depend on the gate) and the
other two ops confirm the probe really does corrupt output — but a probe validated only on `add` would
have been believed wrongly. Zero the output buffer before a probe run.

## 6. What this says for the pipeline

**The modmul's ridge is `R* = 1.7`.** A streaming modmul — read two operands, one multiply, write —
sits at intensity `1/12`, just under the ridge. Three consequences:

1. **At `R=1` the modmul reaches 86% of the memory roof but only 51% of its own compute roof.** So a
   streaming modmul has **~1.94x left in it** (46.83/24.09), and that headroom is reachable **only by
   raising arithmetic intensity — i.e. by fusion.** §5b tried the obvious alternative (rebalance the
   reads across both NoCs) and it came out *worse*: with only two dataflow RISCs, a read+write kernel
   cannot have two pure readers, and the serialisation costs more than the balance buys. No amount of
   further work on the multiply itself can be worth more than 14% while the memory roof is 86% occupied.
2. **This is why the fusion experiments paid and the kernel experiments plateaued.** `exp_tg018ntt`
   fusing six NTT passes into one dispatch (50.863 → 32.448 ms) is exactly the move this roofline
   prescribes: it raises R without touching the arithmetic. The same document's §4e then took it to
   **23.991 ms** by moving the base conversion onto the matrix engine — which is the *other* move this
   roofline prescribes: it takes work out from under the vector unit's modmul roof, the binding one.
3. **`add` and `mul` are not worth optimising as kernels at all in a streaming context** — at `R=1`
   they already sit at 99.6% and 98.6% of the memory roof (`addp` at 100.0%). Their compute ceilings
   (23×, 11× and 3.4× the modmul's) are unreachable without reuse.

**What this does not say.** These are single-op kernels; a real pipeline stage mixes ops and has its
own reuse pattern, so `R` there is an effective quantity, not a knob. And the arithmetic bound is a
*throughput* ceiling measured on an independent-op loop — a dependent chain would be latency-bound and
slower. Both are deliberate: the roofline's y-axis is throughput.

## 7. Running it

```
. ./env.sh
./build/exp_roofline001
ROOF_SWEEP_CORES=1 ROOF_CSV=roof.csv ./build/exp_roofline001
```

| env | meaning |
|---|---|
| `ROOF_TILES` | tiles per pass (default 32768 = 403 MB moved; big enough that `R=1` is ~1.2 ms) |
| `ROOF_CORES` | cores (default the full grid, 130) |
| `ROOF_ITERS` | traced replays per point (default 5); `min` is quoted, spread printed |
| `ROOF_SWEEP_CORES` | add the core sweep at `R=1` |
| `ROOF_SPLIT` | 1 = operand B's read moves to BRISC/NOC_0 beside the write (§5b). **Measured worse**; kept so the refutation is reproducible |
| `ROOF_NOWRITE` | 1 = reads-only probe (§5b). **Wrong answer by design** — the gate must fail |
| `ROOF_CSV` | write every point |

**The four ops**, all one SFPI pass, all gated against a host golden:

| kernel | op | multiplies |
|---|---|--:|
| `roof_modmul.cpp` | Montgomery modular multiply | 11 `mul16` = 22 `sfpmul24` |
| `roof_mul.cpp` | `(a*b) mod 2^32` — full 32×32 → low 32 | 3 `mul16` = 6 `sfpmul24` |
| `roof_add.cpp` | `(a+b) mod q` — modular | 0 |
| `roof_addp.cpp` | `(a+b) mod 2^32` — plain | 0 |

**Discard the first run** — it JIT-compiles ~9 kernel variants. The bench already discards the first
execution of each point internally, but the first *process* also pays compile time in wall clock.
