# exp_ntt001 — NTT-32768, 3-factor vs 2-factor, on Blackhole

**Status: DONE (measurement).** First **p150** measurement of the two factorizations head to head, in
one binary, at equal effort. Bit-exact `0/32768` on **every** run — 44 pipeline executions across five
kernel configurations.

**Two results, and the second is the more useful one.**

**1. The metric decides the winner, and the two metrics disagree.** On pure kernel time the 3-factor
wins, **121 vs 146 µs (1.21×)**; on host wall the 2-factor wins, **153.6 vs 184.0 µs (1.20×)**. Both
are true. The 3-factor does less arithmetic and pays for it in launches — **12 dispatches against 7**,
at ~8.3 µs each. Quote the trace figure for "NTT latency"; quote the wall figure only with the
dispatch count beside it.

**2. Both variants were communication-bound, and fixing that moved each of them more than the choice
of factorization does.** Two hypotheses were raised from the decomposition and then tested:

| | fix | result |
|---|---|---|
| matmul feed (78% of the 2-factor) | `NTT1_DEEP=1` — depth rule | matmul stages **1.53× / 1.42×**; hypothesis **confirmed** |
| corner turn (26% of the 3-factor) | `NTT1_A2A=bothnoc` — both NoCs | that stage **1.31×**; hypothesis **confirmed** |
| per-stage DRAM handoff | `NTT1_FUSE=1` — dec+mm+rec in one kernel | the three triples **2.14–2.27× each**; 12 dispatches → **6** |
| corner turn through DRAM | `NTT1_A2A=exchange` — core-to-core NoC | that stage **5.40×** (23.97 → 4.44 µs) |
| all remaining dispatch boundaries | `NTT1_MEGA=1` — the whole NTT in one kernel | 6 dispatches → **1**; host wall collapses onto kernel time |
| the constant load sitting in front of compute | staged reader | 3-factor 34.0 → **29.6**, 2-factor 62.1 → **56.1** |

Ladder, critical-path busy: **3-factor 121 → 110.0 → 101.6 → 65.1 → 45.5 → 34.0 → 29.6 (4.09×)**,
**2-factor 146 → 114.0 → 98.6 → 62.1 → 56.1 (2.60×)**. The ranking narrowed to a near-tie and then
widened; the 3-factor led at every rung, which is better evidence than the single stock measurement was.
**At one dispatch the two metrics stop disagreeing** — host wall 35.9 against critical path 34.0 at that
rung, where at stock they were 184.0 against 121.

> **Read §2b-undecies for the comparison, not §1.** §1 measures both variants unoptimised and is kept
> because it is where the metric disagreement is visible. The like-for-like result, both fully
> SRAM-resident at one dispatch each, is **29.6 vs 56.1 µs — 1.89×**.

**The largest win needed no new compute kernel.** `compute/fused_matmul_{a,c}_1pass.cpp` already did
dec+mm+rec in one kernel for `exp_Tg018m` and had simply never been wired into either NTT-32768
variant; the only missing piece was a reader with a tile index. See §2b-quater — and note it beat this
document's own cost-model projection by a wide margin, for a reason the model could not see.

`host/experiments/exp_ntt001.cpp` · design [`ntt_32768.md`](ntt_32768.md) · prior n150 numbers
[`ntt_32768_perf.md`](ntt_32768_perf.md) · new kernels in
[`../../kernels/OPTIMISED.md`](../../kernels/OPTIMISED.md)

---

## 1. Numbers

`q=167772161`, seed 0, `C=32`, `modmul=1pass`, **[M-p150]**. Quote `min`
([`../guide/profiling.md`](../guide/profiling.md) §0).

| metric | source | 3-factor (32,32,32) | 2-factor (128,256) | winner |
|---|---|--:|--:|---|
| **critical-path busy** | trace | **121 µs** | 146 µs | **3-factor, 1.21×** |
| host wall | host | 184.0 µs | **153.6 µs** | 2-factor, 1.20× |
| stage-sum | host | 216.2 µs | **207.7 µs** | 2-factor, 1.04× *(unresolved)* |
| dispatches / NTT | — | 12 | 7 | — |
| correctness | — | 0/32768 | 0/32768 | — |

Repetitions, so the spread is visible rather than asserted:

| | rep 1 | rep 2 | rep 3 | min | spread |
|---|--:|--:|--:|--:|--:|
| 3-factor critical-path busy | 121 | 122 | 124 | **121** | 2.5% |
| 2-factor critical-path busy | 148 | 147 | 146 | **146** | 1.4% |
| 3-factor host wall (ITERS=50) | 202.9 | 200.6 | 184.0 | 184.0 | 10.3% |
| 2-factor host wall (ITERS=50) | 233.6¹ | 153.7 | 153.6 | 153.6 | 52%¹ |

`data/ntt001.csv` is a **separate** later run (3f wall 179.8, 2f 153.6, stage-sum 215.4 / 200.4) and
its 3-factor wall is below the 184.0 min above — as expected of a metric with 10% spread, and the
reason the conclusion rests on the trace figure rather than on either wall number.

¹ a host-interference outlier against 153.7 / 153.6 — on its own enough to show why a single
wall-clock pass cannot rank these (`profiling.md` rule 3, earned again). **Critical-path busy is a sum
of per-dispatch durations and is therefore immune to host scheduling**, which is exactly what its 2.5%
against wall's 52% demonstrates.

Every profiled capture passed the 1:1 gate — *"all 12 dispatches present, RISCs complete, contiguous"*
— with `--dataflow-ok`, needed because the corner turn is a dataflow-only pair (`a2a_corner_*`) and
would otherwise be silently dropped, exactly as in the `exp_tg018m` census correction.

**Do not compare any of this with the 319.5 / 334.7 µs in
[`ntt_32768_perf.md`](ntt_32768_perf.md).** Those are **[M-n150]** — different architecture *and*
older code, as that doc's own header says. This is not a speedup; it is a different machine.

### On `stage-sum`, a metric introduced here

Each stage run **alone**, `ITERS` times back to back, then drained; per-stage costs summed. It is an
**attribution, not a latency** — not the "host device busy" of `profiling.md` §1, and it does not obey
that taxonomy's `wall >= host device busy` ordering (here it comes out *larger* than wall, because in
the real chain consecutive *different* programs overlap while isolated re-runs of one program do not).
Use it to localise cost per stage; never quote it as an NTT latency.

---

## 2. Kernel decomposition of the critical path — the authoritative breakdown

`tools/traces/dispatch_csv.py --dataflow-ok`, one row per dispatch, `trace_kernel_latency_us` =
that dispatch's **max over cores**. Here one dispatch is one stage, so this decomposes the critical
path exactly. **[M-p150]**

| 3-factor | µs | % | | 2-factor | µs | % |
|---|--:|--:|---|---|--:|--:|
| A1 dec | 3.13 | 2.5 | | dec A | 3.18 | 2.1 |
| A2 mm | 11.97 | 9.7 | | **M1 mm** (K=256) | **72.96** | **49.3** |
| A3 rec | 10.13 | 8.2 | | rec1 | 10.48 | 7.1 |
| B tw | 5.51 | 4.5 | | twiddle | 5.45 | 3.7 |
| C1 dec | 3.00 | 2.4 | | dec S' | 2.98 | 2.0 |
| C2 mm | 14.85 | 12.1 | | **M2 mm** (K=128) | **42.49** | **28.7** |
| C3 rec | 10.46 | 8.5 | | rec2 | 10.34 | 7.0 |
| TB tw | 5.46 | 4.4 | | | | |
| **A2A** (corner turn) | **32.24** | **26.2** | | | | |
| D1 dec | 2.93 | 2.4 | | | | |
| D2 mm | 13.24 | 10.7 | | | | |
| D3 rec | 10.28 | 8.3 | | | | |
| **total** | **123.2** | | | **total** | **147.9** | |

Both totals reconcile with `busy_metrics.py` (121–124 and 146–148 µs), and the stage mapping is
**independently confirmed**: `modmul_fused3` is the only kernel carrying a profiler zone, and it
appears at exactly `seq` 3 and 7 in the 3-factor and `seq` 3 in the 2-factor — precisely the twiddle
stages. The rest read `unlabelled` because those kernels have no `DeviceZoneScopedN`; the join to
stage names is by dispatch order, which the host controls.

Grouped by kernel class — this is where the two factorizations actually differ:

| class | 3-factor | 2-factor | |
|---|--:|--:|---|
| **matmul** | 40.06 | **115.44** | **2.88× more** in the 2-factor |
| recombine | 30.87 | 20.82 | 3 stages vs 2 |
| corner turn | **32.24** | — | the 3-factor's price for its factorization |
| twiddle | 10.97 | 5.45 | 2 stages vs 1 |
| decompose | 9.06 | 6.15 | 3 vs 2 |
| **total** | **123.2** | **147.9** | 1.20× |

**The `O(N·N^{1/3})` vs `O(N·√N)` MAC argument is worth 2.88× — and it is spent almost entirely on
the corner turn.** The 3-factor's matmul saving is 75.4 µs; it hands back 32.2 µs to `A2A`, 10.1 µs to
a third recombine and 5.5 µs to a second twiddle, leaving only 24.7 µs of net advantage. That is why
the total ratio is 1.20× and not 2.88×.

Two consequences that were not visible before this decomposition:

* **The 2-factor is 78% matmul** (115.4 of 147.9). `M1` alone at K=256 is 49.3% of its whole NTT.
  Nothing but a finer factorization touches that — it is the fundamental cost the design doc predicted.
* **The 3-factor's single largest item is not compute at all.** `A2A` is 26.2%, a dataflow-only
  reshuffle (`all_to_all_corner_mc.cpp`), and it is the item to attack next: its three matmuls
  *together* (40.1 µs) cost only 24% more than the corner turn alone. Note `ntt_32768_perf.md`
  already took this kernel from 972 → 94.8 µs on n150; the 32.2 µs here is what is left after that
  work, on faster hardware.

### 2a. Why the 2-factor's matmul is 2.88× — it is TRAFFIC, not MACs

The obvious explanation is the MAC count, and it is the wrong magnitude. Both variants run the *same*
compute-kernel structure; only the accumulation depth per `pack_tile` differs
(`matmul_compute_mt` inner loop `bk < NB` = 4; `matmul_compute_mtk` inner loop `a < K_acc = NB·Kt`):

| per core, per stage (`npc`=1, `NB`=4) | matmul_tiles |
|---|--:|
| 3-factor `A2/C2/D2` (K=1) | 4 planes × 4 = **16** |
| 2-factor `M1` (Kt=8 → K_acc=32) | 4 × 32 = **128** |
| 2-factor `M2` (Kt=4 → K_acc=16) | 4 × 16 = **64** |

48 against 192 — **exactly 4.000×**, the `O(N·√N)` vs `O(N·N^{1/3})` cost confirmed at instruction
level. But each `matmul_tiles` reads 2 int8 tiles and each output tile writes 4 int32 planes:

| | matmul_tiles | DRAM traffic | measured |
|---|--:|--:|--:|
| 3-factor | 48 | 144 KB | 40.06 µs |
| 2-factor | 192 | 416 KB | 115.44 µs |
| **ratio** | **4.000×** | **2.889×** | **2.882×** |

**Time tracks traffic to 0.2%, not MACs.** The gap between the two ratios is the output write: 4 int32
planes (16 KB) per stage regardless of K, which is 33% of the 3-factor's traffic but 7.7% of the
2-factor's — the 3-factor saves 4× on reads and writes 50% more output (3 stages vs 2).

**These stages are feed-bound.** Per-core bandwidth is constant across all five, two different kernels
and three different K depths:

| | 3f A2 | 3f C2 | 3f D2 | 2f M1 | 2f M2 |
|---|--:|--:|--:|--:|--:|
| GB/s/core | 4.11 | 3.31 | 3.71 | 3.82 | 3.47 |

≈115 GB/s aggregate over 32 cores — **26% of the 436 GB/s ceiling**
([`../he/dram_guide.md`](../he/dram_guide.md) §1). `reader_matmul_mt` / `_mtk` do `read; barrier; push`
per tile, i.e. **one read in flight**, where §3.3 measures 42.6 GB/s at 96 pages outstanding and 81.5
at 384, and §3.1 measures **1.79×** for driving both NoCs. This is the defect
`reader_fused_batch.cpp` already fixed elsewhere (17 KB in 8.77 µs ≈ 2 GB/s → one barrier instead of
17, −82%, [`../he/blackhole_port.md`](../he/blackhole_port.md) §5).

**This contradicts [`ntt_32768_perf.md`](ntt_32768_perf.md) §5**, which concluded of the 2-factor's
matmul that *"the optimizations can't touch it — the matmul is already int8 + K-loop"*. It is int8 and
K-looped and still running at a quarter of the machine, because the constraint is the feed, not the
matrix engine. `dataflow/reader_seq_batch.cpp` ([`../../kernels/OPTIMISED.md`](../../kernels/OPTIMISED.md),
"up to ~5× on feed-bound kernels") is the drop-in.

### 2a-bis. CONFIRMED by experiment — deep readers, `NTT1_DEEP=1`

The falsifying test named above was run. `dataflow/reader_matmul_mt_deep.cpp` and
`reader_matmul_mtk_deep.cpp` are siblings of the stock readers with the barrier hoisted out of the
inner loop, so one whole output plane is in flight at once — **8 tiles (3-factor, NB=4 pairs) and 64
tiles (2-factor, K_acc = NB·Kt = 32 pairs)** against 2. Index math, argument list and push order are
identical; only the CB depth changes on the host side. All runs bit-exact `0/32768`. **[M-p150]**

**Only the matmul stages moved**, which is what makes this a controlled result rather than a
coincidence:

| 2-factor | stock | deep | | 3-factor | stock | deep |
|---|--:|--:|---|---|--:|--:|
| dec A | 3.179 | 3.110 | | A1 dec | 3.133 | 3.138 |
| **M1 mm** | **72.956** | **54.594** | | **A2 mm** | **11.970** | **8.039** |
| rec1 | 10.477 | 10.446 | | A3 rec | 10.129 | 10.050 |
| twiddle | 5.454 | 5.585 | | B tw | 5.507 | 5.532 |
| dec S' | 2.976 | 3.011 | | C1 dec | 2.999 | 3.044 |
| **M2 mm** | **42.488** | **26.939** | | **C2 mm** | **14.850** | **9.687** |
| rec2 | 10.340 | 10.357 | | C3 rec | 10.461 | 10.384 |
| | | | | TB tw | 5.461 | 5.522 |
| | | | | **A2A** | **32.236** | **32.255** |
| | | | | D1 dec | 2.927 | 2.948 |
| | | | | **D2 mm** | **13.241** | **8.458** |
| | | | | D3 rec | 10.281 | 10.379 |
| **total** | **147.9** | **114.0** | | **total** | **123.2** | **109.8** |

Every non-matmul stage is within ±0.13 µs, including `A2A` at 32.24 → 32.26 — untouched, because its
reader was not changed. **Matmul totals: 3-factor 40.06 → 26.18 (1.53×), 2-factor 115.44 → 81.53
(1.42×).** The §2a claim is therefore no longer [P]: these stages *were* feed-bound.

Critical-path busy, min of 3 reps each:

| | stock | deep | gain |
|---|--:|--:|--:|
| 3-factor | 121 | **110.0** (110.73/110.49/109.99, spread 0.7%) | 1.10×, −11 µs |
| 2-factor | 146 | **114.0** (114.02/114.10/114.25, spread 0.2%) | **1.28×, −32 µs** |

**They are now nearly tied.** The 3-factor's lead falls from **1.21× to 1.036×** — 4.0 µs on 110.
Still resolved (the ranges do not overlap: 3-factor max 110.73 < 2-factor min 114.02), but the margin
is now a twentieth of what §1 reports for stock dataflow.

> **§2c's projection was directionally right and quantitatively too aggressive.** It predicted the
> 2-factor would gain more (correct: −32 vs −11 µs) and that a 2× feed improvement would take it to
> ~90 µs and *reverse* the ranking. The measured feed improvement was 1.42× on the matmuls, not 2×,
> and the ranking narrowed to near-parity without reversing. Recorded rather than quietly amended.

**Headroom remains.** At 5.1–5.5 GB/s per core (≈165–176 GB/s over 32 cores) the matmul stages are
still at **38–40% of the 436 GB/s ceiling**, because the deep readers still drive **one NoC**. That is
`dram_guide.md` §3.1, the largest single factor measured anywhere in this repo at **1.79×**, and it is
untaken on both variants. **[X]**

### 2b. The corner turn drives ONE NoC, and a both-NoC version already exists

`exp_ntt001` runs the corner turn as `dataflow/all_to_all_corner_mc.cpp`, on a **single kernel,
`RISCV_1` / `NOC_1`**. That kernel *is* optimised relative to `all_to_all_corner.cpp` — multi-core
plus row-chunk copy, the work that took it 972 → 94.8 µs on n150 — but both of those optimisations
are about *index arithmetic and core count*, not about the fabric.

**All of its reads therefore ride NOC_1 while NOC_0 sits idle**, and the corner turn is
read-dominated (128 KB in against 4 KB out per core). `kernels/experiments/exp_tb001/dataflow/a2a_corner_rd.cpp`
+ `a2a_corner_wr.cpp` already split exactly this across NCRISC/NOC_1 and BRISC/NOC_0 for
`exp_Tg018m`, whose own header states the reason: *"driving only one is the single largest factor
measured anywhere in `dram_guide.md` (§3.1, 1.79×)."*

It was not in [`../../kernels/OPTIMISED.md`](../../kernels/OPTIMISED.md), which has no corner-turn
entry at all. **It has now been ported** as `dataflow/all_to_all_corner_{rd,wr}.cpp`
(`NTT1_A2A=bothnoc`); `all_to_all_corner_mc.cpp` is retained unchanged.

### 2b-bis. CONFIRMED — the corner turn on both NoCs

The port keeps the row-chunk assembly loop verbatim and changes only the fabric: NCRISC/NOC_1 reads
tiles `[0, NT/2)`, BRISC/NOC_0 reads `[NT/2, NT)`, then BRISC waits on `c_0`, assembles and writes.
3 reps each, deep readers on in both arms, all `0/32768`. **[M-p150]**

| | one NoC | both NoCs | gain |
|---|--:|--:|--:|
| `A2A` stage | 31.92 (32.45/31.92/32.15) | **24.29** (25.38/24.29/25.43) | **1.31×**, −7.6 µs |
| 3-factor critical path | 108.69 | **101.63** | 1.07×, −7.1 µs |

**1.31× overall, not the 1.79× of `dram_guide.md` §3.1.** An earlier version of this section explained
the shortfall by solving two equations (one-NoC vs both-NoC) and concluded the stage was
**assembly-bound, ~68% L1 copying**. **That was wrong, and it was wrong in the direction that would
have discouraged the next step.** The split has since been measured directly — see §2b-quinquies.

### 2b-quinquies. The corner turn is READ-bound, measured not modelled

`dataflow/all_to_all_corner_probe.cpp` is `all_to_all_corner_wr.cpp` with the assembly loop deleted and
a single sentinel word written in its place, so the stage cost becomes read + write + launch and
`(real − probe)` is the assembly. It computes a deliberately wrong answer, and **the correctness gate
must fail for the measurement to be valid** — 3 reps, paired, same session **[M-p150]**:

| | stage | gate |
|---|--:|---|
| probe (no assembly) | **15.48 µs** (16.02 / 15.48 / 15.48) | `32768/32768` ✗ — as required |
| real (`bothnoc`) | **24.22 µs** (24.22 / 24.84 / 24.30) | `0/32768` ✓ |
| **difference = assembly** | **8.74 µs, 36%** | |

**So reads + write + launch are 64% of the stage, not 32%.** The two-equation model had it backwards.
Re-deriving the both-NoC gain on the term it actually applies to: with ~3 µs of write+launch, reads are
~12.5 µs on both NoCs against ~20.2 µs on one — **~1.6×, close to §3.1's 1.79×** rather than a rule
that failed to transfer. The 1.31× headline is diluted by the 36% that is not reads, nothing more.

> **Why the first version of this probe was thrown away.** It wrote `get_write_ptr(cb_o)` without
> putting anything there, and the correctness gate came back **`0/32768` — a pass, for a kernel that
> cannot possibly be correct.** Its 15.5 µs was therefore not quotable, whatever it happened to say.
> Writing one sentinel word made the gate fail as it must, and only then was the timing usable. **A
> probe that corrupts the output and still passes is a broken instrument, and the gate is what
> catches it** — the same rule as `profiling.md` §3's "a counter that reads 0 calls is a broken
> instrument", applied to a correctness check instead of a timer.

**What this licenses.** The 15.48 µs is dominated by each core reading **all 32 input tiles (128 KB)**
to take one row from each. A core-to-core exchange would have each core read only **its own** tile
(4 KB) and receive the 32 rows it needs (4 KB) over the NoC — 16× less DRAM traffic for the stage. That
is the T3 premise, and it is now supported rather than assumed. Bounded, though: the assembly's 8.74 µs
does not move (the same 1024 word copies happen, just distributed differently), and 32 × 128 B NoC
messages per core work against this repo's own findings on transfer size, so the ceiling on the whole
stage is roughly 24.2 → 13–17 µs, i.e. **~1.1–1.2× on the 65.1 µs path.** **[P]**

### 2b-quater. Fused dec+mm+rec triples — `NTT1_FUSE=1`, the largest single win

The three `decompose → matmul → recombine` triples were collapsed into **one compute kernel each**, so
the int8 limbs and the int32 planes never leave the core. This needed **no new compute kernel**:
`compute/fused_matmul_{a,c}_1pass.cpp` already do exactly this for `exp_Tg018m` and were simply never
wired into either NTT-32768 variant. The one missing piece was a reader —
`dataflow/reader_fused_batch_g.cpp`, `reader_fused_batch.cpp` plus a tile index, because that kernel
hardcodes `read_tile(0, …)` while the NTT keeps all 32 cube tiles in one buffer and gives core `c`
tile `c`. 12 dispatches → **6**. `0/32768`. **[M-p150]**

| | unfused | fused | |
|---|--:|--:|--:|
| A triple (`A1 dec`+`A2 mm`+`A3 rec`) | 21.23 | **9.91** | **2.14×** |
| C triple (`C1`+`C2`+`C3`) | 23.12 | **10.18** | **2.27×** |
| D triple (`D1`+`D2`+`D3`) | 21.79 | **9.83** | **2.22×** |
| twiddles `B`+`TB` | 11.05 | 11.26 | unchanged ✓ |
| `A2A` | 24.29 | 24.55 | unchanged ✓ |
| **total** | **101.5** | **65.1** | **1.56×** |

Again controlled: only the fused stages moved. Reps 65.14 / 65.39 / 65.72, **spread 0.9%**.

**This beat the §2c projection (~76 µs, 1.34×) because the projection missed two things.** It assumed
fusion only removes DRAM round trips and that the recombines would stay ~28 µs. But
`fused_matmul_a_1pass` also keeps the four int32 planes **in DST** across the matmul→recombine boundary
and runs the recombine as **one SFPI pass** — so the recombine largely disappears into the matmul
rather than merely losing its round trip. Three optimisations arrived together, and a cost model built
from stage times could not see the last two.

**Two things this changes about the rest of the document.**

1. **`A2A` is now 37.4% of the critical path** (was 23.9%), and after §2b-bis ~68% of *it* is serial L1
   assembly. Splitting that assembly across the two RISCs is now the single largest remaining item on
   the 3-factor side. **[X]**
2. **The §3 modmul A/B is now resolvable, and should be re-run.** The twiddles are **17.1%** of the
   path (was 10.9%), so the ~3.6 µs 1pass↔hybrid difference is **5.5% of 65.1 µs against a 0.9%
   spread** — comfortably above the floor, where at stock dataflow it was 2.9% against 2.5% and could
   not be separated. The same experiment that was correctly reported as unresolved becomes decidable
   purely because everything around it got faster. **[X — not re-run.]**

### 2b-sexies. T3 — the corner turn as a core-to-core exchange. **5.40×**

`dataflow/all_to_all_corner_x.cpp`. Each core reads only **its own** tile (4 KB, not 128) and pushes
each row straight into the peer that needs it, over the NoC. Paired against the DRAM version, 3 reps,
same session, `NTT1_DEEP=1 NTT1_FUSE=1` in both arms **[M-p150]**:

| | `A2A` stage | 3-factor total | correctness |
|---|--:|--:|---|
| `bothnoc` (DRAM) | 23.97 (24.48/23.97/24.24) | 65.11 | `0/32768` |
| **`exchange`** | **4.44** (4.44/4.50/4.49) | **45.52** | `0/32768` |
| | **5.40×** | **1.43×** | |

Validated harder than end-to-end: the exchange's intermediate `D` buffer checksums
**`30e9f1c188e2768f`, byte-identical to the DRAM version's** — it is not merely producing an answer
that survives the rest of the pipeline, it is producing the same corner turn.

**It removed the assembly as well as the read, which is why it beat its own bound.** The DRAM versions
must gather 32 scattered rows out of 32 staged tiles into face-tiled order — 1024 word copies. Here the
**sender** knows where its row lands, so it writes into the receiver's final face-layout offsets
directly (two 64-byte writes per peer, destination offsets loop-invariant because the destination row
is always the sender's own index) and **the receiver assembles nothing**.

This is the first core-to-core synchronisation in the repo — everything else goes through DRAM. The
protocol: scatter, flush, then increment every peer's semaphore (so an announcement always follows the
data it announces), wait for `NT`, then write out. **It requires the full grid** — a core that never
sends hangs every other core's wait — so the host refuses `C != NTILE`. It did not deadlock on the
first run.

> **Third cost-model failure, and the pattern is now clear.** §2c predicted this at 24.2 → 13–17 µs,
> ~1.1–1.2× on the path, on two assumptions: that the assembly's 8.74 µs "does not move", and that
> 64 small NoC writes would underperform bulk DRAM page reads. **Both were wrong** — the assembly was
> designed out entirely, and the writes pipeline fine. Earlier the fusion beat its projection for a
> similar reason, and the read/assembly split was backwards. **Cost models built from stage-level
> timings systematically miss what a rewrite can restructure**; they bound the *current* shape, not the
> achievable one. They are still worth writing — they are what made these three the ones to try — but
> the number they produce should not be trusted as a ceiling.

### 2b-septies. T3 — the whole NTT as ONE dispatch. **34.0 µs** *(3-factor only; superseded by 2b-undecies)*

`compute/ntt32768_3f_fused.cpp` + `dataflow/ntt32768_3f_fused_{rd,wr}.cpp`, `NTT1_MEGA=1`. The compute
kernel holds the entire chain — A triple, B twiddle, C triple, TB twiddle, **a mid-kernel park**, D
triple — and BRISC runs the 32-core corner turn while compute is blocked on `cb_d`. Every intermediate
(limbs, the four int32 planes, G/G'/H/Cb/D) stays in L1; the kernel reads Xin and the folded constants
once and writes E once. **[M-p150]**

| | dispatches | critical-path busy | host wall |
|---|--:|--:|--:|
| 6 dispatches (fused + exchange) | 6 | 45.5 | 95.3 |
| **1 dispatch (mega)** | **1** | **34.0** (34.32/34.01/34.15, 0.9%) | **35.9** (35.9/36.3) |
| | | **1.34×** | **2.65×** |

**Correctness is verified byte-for-byte, not merely end-to-end.** `Cb`, `D` and `E` all checksum
identically to the six-dispatch version (`E = 0e80b747e53594c6`), so the mega kernel reproduces every
intermediate exactly. `0/32768` on every run.

**The headline is that wall time collapsed onto kernel time.** At stock dataflow this document opened
with a 63 µs gap between what the host sees (184.0) and what the device computes (121) — the gap that
made the two metrics disagree about which factorization wins. At one dispatch there is no launch
overhead and no inter-stage host stall, so **host wall 35.9 vs critical path 34.0 — a gap of ~2 µs.**
The metric ambiguity that motivated the whole experiment disappears when there is only one dispatch to
account for.

It did not deadlock, which was the live risk: eight SFPU↔matrix-engine switches in one kernel against
the re-init hazard that "hung every NTT-based binary on Blackhole", plus a compute kernel parked on a
global barrier. The discipline that made it work is stated at the top of the kernel — one full init at
entry, then only `matmul_init` / `reconfig_data_format*` / `copy_tile_init` / `pack_reconfig_data_format`.

**Cost model, fourth attempt, finally about right.** §2b-sexies predicted "~8 µs of critical path plus
launch overhead"; measured 11.5 µs of critical path and 59 µs of wall. Close on the first, low on the
second — the estimate missed that the mega kernel also reads the 16-tile W folds **once** instead of
once per triple.

### 2b-octies. The SAME fusion applied to the 2-factor — a like-for-like rung

Everything after §2b-quater had been applied to the 3-factor only, so the widening lead was partly an
artefact of where the effort went. `compute/fused_matmul_k_1pass.cpp` fixes that: the K-looped
matmul with the **bigK** recombine fused in (five conditional subtracts, not two — at K=256 the plane
sums reach ~2²², per `recombine_bigK.cpp`), planes kept in DST. 7 dispatches → **5**, `0/32768`.
**[M-p150]**

| 2-factor | unfused | fused | |
|---|--:|--:|--:|
| `dec A` | 3.18 | 3.12 | — |
| `M1 mm` + `rec1` | 54.59 + 10.45 = 65.04 | **56.58** | **1.15×** |
| `twiddle` | 5.59 | 5.55 | — |
| `dec S'` | 3.01 | 3.07 | — |
| `M2 mm` + `rec2` | 26.94 + 10.36 = 37.30 | **30.29** | **1.23×** |
| **total** | **114.0** | **98.6** | **1.16×** |

**1.16× against the 3-factor's 1.56× from the same technique**, and the asymmetry is structural rather
than unfair: fusion removes *handoffs*, and the 2-factor's time is real matmul work. It is 78% matmul
(§2), and §2a showed that matmul is 4× the MACs of the 3-factor's.

**Why the later rungs do not transfer at all.** The 2-factor's matmuls are **not tile-local**: M1's
output tile `(mt,nt)` needs `limb_p(A)[mt][kt]` for `kt=0..7` — **8 input tiles** — and M2's needs
`limb_r(S')[kt][nt]` for `kt=0..3`, **4 tiles owned by 4 other cores**. So its cross-tile traffic is
*smeared through both matmul readers*, where the 3-factor's is isolated in one named corner turn.
There is nothing to point `NTT1_A2A=exchange` at. That is also why the decompose could not be folded
in: 8 cores share each `mt`, so a fused decompose would have all 8 decompose the same 8 tiles.

**The honest comparison, at equal treatment:**

| both variants, same techniques | 3-factor | 2-factor | lead |
|---|--:|--:|---|
| stock | 121 | 146 | 1.21× |
| + deep readers | 110.0 | 114.0 | 1.036× |
| + fused matmul/recombine | **65.1** | **98.6** | **1.51×** |
| + corner-turn work + mega *(3-factor only)* | 34.0 | 98.6 | 2.90× |
| **+ both fully SRAM-resident, staged reader** | **29.6** | **56.1** | **1.89× — the fair figure** |

**1.51× is the like-for-like number.** The step from there to 2.90× is real but rests on optimisations
that have no 2-factor counterpart — which is itself a finding about the factorizations, not a gap in
the experiment. A 2-factor mega kernel would need a cross-core step between the twiddle and M2 plus
the decompose duplication above; **not built [X]**, and its expected value is low because the term it
would attack is MAC work, not movement.

### 2b-undecies. THE FAIR COMPARISON — both variants fully SRAM-resident, one dispatch each

Every earlier table compared variants at different levels of optimisation. This one does not. **Both
transforms run as a single dispatch with NO DRAM ACCESS DURING THE NTT**: inputs and folded constants
are read once at kernel entry by a staged reader, the result is written once at exit, and every dynamic
operand in between reaches its consumer over the NoC. Same rule, same core count, same treatment of
both readers. `0/32768` on every run. **[M-p150]**

| | 3-factor (32,32,32) | 2-factor (128,256) | |
|---|--:|--:|---|
| **total latency** | **29.6 µs** (29.74/29.84/29.64) | **56.1 µs** (56.13/57.17/56.64) | **1.89×** |
| spread | 0.7% | 1.9% | |
| dispatches | 1 | 1 | |
| **constant load** (NCRISC span) | **8.61** | **35.69** | 4.1× — the `f²·NB²` law, in time |
| decompose | 10.70 (3 inst) ¹ | 2.51 (2 inst) | |
| matmul | 1.32 (3) | 53.80 (2) ¹ | |
| recombine | 11.29 (3) → 3.76 each | 8.20 (2) → 4.10 each | bigK: 5 subtracts vs 2, +9% |
| **modmatmul** (dec+mm+rec) | **23.31** | **64.51** | |
| **modmul** (twiddle) | 8.25 (2) → **4.13** each | 4.50 (1) → **4.50** | *same kernel — agrees to 9%* |
| **on-chip exchange** | **4.44** (1: corner turn) | **4.67** (exch 1) + 43.65 ¹ (exch 2) | |
| constants resident / core | 40 KB | 196 KB | 4.9× |
| L1 working set / core | ~80 KB (5.6%) | ~264 KB (18.3%) | |

¹ **wait-contaminated — do not sum these rows.** Zones overlap across the three TRISCs and charge CB
waits to whichever phase holds them. Specifically: the 3-factor's `decompose` includes contention on
the single `cb_limbs` its three triples share; the 2-factor's `matmul` includes waiting for the reader's
W256 stage; `exch 2` includes waiting for M1. Rows sum to 31.6 (3-factor) and 69.0 (2-factor) against
totals of 29.6 and 56.1.

**The exchanges cost the same — 4.44 vs 4.67 µs.** This corrects an earlier reading in this document.
Before the reader was staged, the 2-factor's first exchange zone measured 38.6 µs and looked like an
enormous cost of its denser communication; **36.6 of that was the constant load it was blocked behind.**
With the load staged, the exchange proper shows up at 4.67 µs — within 5% of the 3-factor's corner turn
**despite moving 8× the bytes** (32 KB broadcast against 4 KB transpose), because 1 KB messages are far
more efficient than the corner turn's 64 B ones. *A zone that begins with `cb_wait_front` measures the
wait, not the work.*

**So the 1.89× is not communication.** With movement equalised, what remains is:
* **the constant load, 8.61 vs 35.69 µs** — 196 KB against 40 KB per core, still 64% of the 2-factor's
  runtime and only partly hidden;
* **the MAC count** — 4× more, though its K-loop recovers some of that (§2a);
* **the bigK recombine**, +9% per instance.

The twiddle row is the control: the *same* `modmul_fused3` kernel measures 4.13 and 4.50 µs per
instance in the two pipelines. A phase split that disagreed there would not be worth reading elsewhere.

**Staging the reader was worth 1.15× / 1.11×** (34.01 → 29.64, 62.14 → 56.13) and **left the ratio
essentially unchanged, 1.83× → 1.89×**. The prediction that it would favour the 2-factor was wrong:
staging costs issue depth, since a stage's reads cannot be issued until the previous barrier retires,
and for the variant with the most data that loss nearly cancels the overlap won. Its reader span barely
moved (36.63 → 35.96); the gain came from compute starting sooner, not from a faster load.

**Still untaken on the 2-factor's load [X]:** the W256 slice is shared by the 4 cores with equal `nt`,
so a multicast would cut chip-wide constant traffic ~4×; and `dram_guide.md` §3.1's both-NoC rule is
unapplied on these readers. Either could move the number that now dominates it.

`compute/ntt32768_{2f,3f}_fused.cpp` · `dataflow/ntt32768_{2f,3f}_fused_*.cpp`. Neither mega kernel
deadlocked on its first run — the 2-factor's carries two semaphores and two group barriers.

### 2b-decies. Per-stage comparison table — per-stage dispatches, DRAM-minimal, on-chip a2a

**Superseded as the headline by §2b-undecies**, which puts both variants fully on chip. This remains the canonical *per-stage* comparison — the only table whose rows are separately attributable and additive.

**Per-stage comparison.** Each stage type is its own dispatch, so every row is
separately attributable; each dispatch is internally DRAM-free (fused dec+mm+rec for the 3-factor,
fused mm+rec for the 2-factor); the corner turn is the on-chip exchange; every kernel is the best
variant available. `dispatch_csv.py` for the stage rows, `zone_phases.py` for the split inside the
fused kernels (`-DNTT_PHASE_ZONES` / `-DTB001_PHASE_ZONES`). `0/32768` throughout. **[M-p150]**

| | 3-factor (32,32,32) | 2-factor (128,256) | |
|---|--:|--:|---|
| **decompose** | 15.02 | 6.14 | 3 stages vs 2 |
| **matmul** | **1.85** | **79.02** | operands resident vs streamed |
| **recombine** | 11.29 | 8.22 | 3 vs 2 |
| **twiddle** | 10.09 | 5.53 | 2 vs 1 |
| **communication** | **4.39** (on-chip a2a) | **0 — inside matmul** ¹ | |
| **total latency** | **45.6** | **99.4** | 3-factor **2.18×** |
| dispatches | 6 | 5 | |
| **constants / core** | **40 KB** | **196 KB** | 2-factor **4.9×** ² |
| *% of 1440 KB L1* | 2.8% | 13.6% | |
| **constants / chip, distinct** | **164 KB** | **1408 KB** | 2-factor **8.6×** ² |

**The rows are additive here, which is why this table supersedes §2b-nonies.** 3-factor:
15.02 + 1.85 + 11.29 = **28.16** against the three fused triple dispatches' **27.97** — 0.7%.
2-factor: 6.14 + 79.02 + 8.22 + 5.53 = **98.91** against **99.31** — 0.4%. (Rows sum ~6% under the
3-factor total; the remainder is per-dispatch launch, which no row owns.) In the mega kernel the same
phases overlapped across TRISCs and summed to 43.6 against a 33.9 µs kernel — not additive, and not a
table to quote.

**`matmul` 1.85 against 79.02 µs is the single most informative pair.** It is the same `matmul_tiles`
loop in both. The 3-factor's operands are already resident in on-core CBs; the 2-factor's stream 256
(M1) and 128 (M2) tiles through the CBs while the loop waits, and that waiting is inside the zone. The
2-factor does **4×** the MACs (§2a) — it does not do 43× the work.

¹ "Not separable", not "none": M1's output tile needs 8 input tiles and M2's needs 4 owned by other
cores, so the 2-factor's cross-tile traffic lives inside the matmul row. The 3-factor's is one stage,
which is exactly why it could be attacked and went 32.24 → 4.39 µs.

² `[P]` — derived from the fold layouts, not measured. A folded `f × f` DFT costs `f² · NB²` bytes:
f=32 → 16 KB, f=128 → 256 KB, f=256 → **1 MB**. The coarse factorization pays *quadratically* in
twiddle storage, and unlike latency that is not something scheduling can recover.

**Regression on the shared kernels.** `ph_matmul`/`ph_recombine` were added to `fused_matmul_a_1pass`
and all three zones to `fused_matmul_c_1pass`, behind the pre-existing default-off `TB001_PHASE_ZONES`.
With zones off: 3-factor **45.57 µs**, 2-factor **99.31 / 99.50** — unchanged from before the edit, so
no existing caller moves.

**A tooling note worth keeping.** `dispatch_csv.py` intermittently returns an empty window (~1 run in
3 here) and a measuring script that pipes it through `awk` will silently report **0.00 µs**. That is
the probe lesson (§2b-quinquies) in a second costume: **an analysis script must fail loudly rather than
sum nothing to zero.**

### 2b-nonies. Stage comparison, earlier configurations — SUPERSEDED by 2b-decies

**Superseded by §2b-decies**, which measures the configuration these two bracket and whose rows are
additive. Retained because table (a) is the only place the *unfused* stage costs are recorded, and
table (b) is the only measurement of the mega kernel's internal phases.

Two versions, because the rows mean different things in each. `tools/traces/zone_phases.py` produces
the fused split from inner phase zones (`-DNTT_PHASE_ZONES`); `dispatch_csv.py` produces the separable
one. **[M-p150]** except the constants, which are `[P]` — derived from the fold layouts, not measured.

**(a) Stages separable — deep readers on both, one dispatch per stage. The rows are additive.**

| µs | 3-factor | 2-factor | |
|---|--:|--:|---|
| decompose | 9.13 | 6.12 | 3 stages vs 2 |
| matmul | 26.18 | **81.53** | 2-factor 3.11× |
| recombine | 30.81 | 20.80 | 3 stages vs 2 |
| twiddle | 11.05 | 5.58 | 2 stages vs 1 |
| communication | **32.26** | 0 ¹ | |
| **total** | **109.4** | **114.0** | |
| dispatches | 12 | 7 | |

**(b) Optimised — 3-factor mega (1 dispatch), 2-factor fused (5). The rows are NOT additive.**

| µs | 3-factor | 2-factor | |
|---|--:|--:|---|
| decompose | 22.71 ² | 6.19 | |
| matmul | **1.32** ² | **79.42** | resident vs streamed operands |
| recombine | 11.29 ² | 8.22 | |
| twiddle | 8.25 ² | 5.55 | |
| communication | 4.44 | 0 ¹ | |
| **total** | **34.0** | **98.6** | 3-factor **2.90×** |
| dispatches | **1** | 5 | |

¹ **"Not separable", not "none".** The 2-factor's cross-tile traffic is inside its matmul row — M1's
output tile needs 8 input tiles, M2's needs 4 owned by other cores — which is why `ph_matmul` is 79.42
of its 87.14 µs fused kernel: that zone contains the `cb_wait_front` for every streamed operand pair.
The 3-factor's is isolated in one stage, which is precisely why it could be attacked (32.26 → 4.44).

² **These four do not sum to 34.0 (they total 43.6).** In the mega kernel the phases pipeline across
the three TRISCs and the zones include CB waits, so they overlap — `ph_decompose`'s 22.71 is mostly
*waiting* on the reader and on the 32-core exchange, not decomposing. Read them as "which phase held
the zone", not as a budget. Table (a)'s rows are additive; these are not.

**The most informative pair of cells is matmul: 1.32 against 79.42 µs.** Both are the same
`matmul_tiles` issue loop. The 3-factor's operands are already resident in on-core CBs; the 2-factor's
stream 256 (M1) and 128 (M2) tiles through the CBs while the loop waits. The 2-factor does 4× the MACs
(§2a) — it does not do 60× the MACs. The gap is *where the operands live*, and it is the clearest
single statement of what the fusion bought.

### Constant parameters that must be on chip

| | 3-factor | KB | | 2-factor | KB |
|---|---|--:|---|---|--:|
| | WfoldA (32×32 DFT, fold-LHS) | 16 | | W256 slice (`nt` fixed) | 128 |
| | Wrhs (32×32 DFT, fold-RHS) | 16 | | W128 slice (`mt` fixed) | 64 |
| | Ta, Tb twiddles | 8 | | Tb twiddle | 4 |
| **per core** | | **40** | | | **196** |
| *% of 1440 KB L1* | | 2.8% | | | 13.6% |
| **whole chip, distinct** | | **164** | | | **1408** |

**A folded `f × f` DFT costs `f² · NB²` bytes** (NB = 4 radix-128 limbs): f=32 → 16 KB, f=128 → 256 KB,
f=256 → **1 MB**. So the coarse factorization pays **quadratically** in twiddle storage — 4.9× per core
and 8.6× chip-wide. This is the hidden price of `O(N·√N)`, and unlike the latency it is not something
scheduling can recover. **[P — derived from the fold layouts in `host/experiments/exp_ntt001.cpp`, not measured.]**

### 2b-ter. The full ladder

Critical-path busy, min of 3 reps at each rung, every rung bit-exact `0/32768` **[M-p150]**:

| rung | dispatches | 3-factor | 2-factor | who leads |
|---|--:|--:|--:|---|
| stock dataflow | 12 / 7 | 121 | 146 | 3-factor, **1.21×** |
| `+ NTT1_DEEP=1` (deep matmul readers) | 12 / 7 | 110.0 | **114.0** | 3-factor, 1.036× — near-tie |
| `+ NTT1_A2A=bothnoc` (both-NoC corner turn) | 12 / 7 | 101.6 | 114.0 | 3-factor, **1.12×** |
| `+ NTT1_FUSE=1` (fused dec+mm+rec triples) | 6 / 7 | 65.1 | 114.0 | 3-factor, 1.75× |
| `+ NTT1_A2A=exchange` (core-to-core corner turn) | 6 / 7 | 45.5 | 114.0 | 3-factor, 2.50× |
| `+ NTT1_MEGA=1` (whole NTT in one kernel) | 1 / 5 | 34.0 | 98.6 | 3-factor, 2.90× |
| `+ NTT1_MEGA=1` on BOTH, staged readers | **1 / 1** | **29.6** | **56.1** | 3-factor, **1.89×** |
| | | **4.09× total** | **2.60× total** | |

The 2-factor column now carries the same fused matmul/recombine (§2b-octies); the rungs it skips —
both-NoC, exchange, mega — are all corner-turn work, and it has no corner turn. **The like-for-like
lead is 1.51× (65.1 vs 98.6); 2.90× includes 3-factor-only optimisations.**

Host wall over the same ladder: **184.0 → 95.3 → 35.9 µs, 5.12×** — and at one dispatch it sits ~2 µs
above the critical path instead of 63 µs above it.

The corner turn across all three implementations: **32.24 → 23.97 → 4.44 µs, 7.26×**, and from 26.2%
of the critical path down to **9.8%**. What now dominates is the three fused triples (29.9 µs, 66%)
and the two twiddles (11.25 µs, 25%).

**The ranking oscillated but never flipped, and then widened decisively.** The deep readers helped the
2-factor far more (2.9× the matmul traffic), collapsing the lead to near-parity; the corner-turn fix
helped *only* the 3-factor, because a 2-factor pipeline has no corner turn; the triple fusion then
helped the 3-factor again, because it has **three** triples to the 2-factor's two and its triples are
dominated by the fixed per-stage cost that fusion removes rather than by matmul work that it does not.
That the winner survived optimisations pulling in opposite directions is better evidence for §1 than
the original stock measurement was.

**The 2-factor's 114.0 µs is not its floor** — the same triple fusion applies to it (2 triples), and
its `M1`/`M2` are real matmul work that fusion does not remove, so it would gain less. Not run. **[X]**

### 2c. So the ranking rests on both variants being equally un-optimised

Stated plainly, because it bounds what §1 is entitled to claim:

| variant | dominant term | share of its critical path | state | fix that exists |
|---|---|--:|---|---|
| 3-factor | corner turn | 26.2% | single-NoC | `a2a_corner_{rd,wr}.cpp` (≈1.79× ceiling) |
| 2-factor | matmul feed | 78.0% | one read in flight, 26% of DRAM ceiling | `reader_seq_batch.cpp` (≤5× on feed-bound) |

The only `OPTIMISED.md` kernel actually in play here is the modmul, and §3 shows it is 8.9% of the
path and unresolvable. **The comparison is like-for-like and therefore a fair ranking, but it is not
a measurement of either factorization at its best.**

**And the two fixes are asymmetric, in the direction that threatens the §1 conclusion.** The
3-factor's corner turn could give back at most ~14 µs (1.79× on 32.2). The 2-factor is feed-bound
across 78% of its path; even a 2× improvement in its matmul feed returns ~58 µs, which would take it
from 146 µs to ~90 and reverse the kernel-time ranking outright. **So "the 3-factor is the faster
algorithm" is conditional on the current dataflow, and the more communication-bound variant has the
most to gain from fixing it.** **[P — scaled from the measured shares above, not run.]**

---

## 2d. Where the wall-clock cost is

Per-stage `stage-sum`, min of 3 reps, `C=32` **[M-p150]**:

| 3-factor | µs | | 2-factor | µs |
|---|--:|---|---|--:|
| A1 dec | 14.3 | | dec A | 14.6 |
| A2 mm | 15.6 | | **M1 mm** (K=256) | **74.3** |
| A3 rec | 15.7 | | rec1 | 16.8 |
| B tw | 15.9 | | twiddle | 17.1 |
| C1 dec | 15.4 | | dec S' | 19.1 |
| C2 mm | 16.7 | | **M2 mm** (K=128) | **43.9** |
| C3 rec | 16.4 | | rec2 | 16.1 |
| TB tw | 16.6 | | | |
| **A2A** | **34.0** | | | |
| D1 dec | 16.5 | | | |
| D2 mm | 17.1 | | | |
| D3 rec | 14.9 | | | |

**The per-dispatch launch cost falls out of the two metrics together**, and this is the load-bearing
arithmetic of the whole experiment:

| | stage-sum | − critical-path busy | = launch | ÷ dispatches |
|---|--:|--:|--:|--:|
| 3-factor | 216.2 | 121 | 95.2 µs | **7.9 µs** |
| 2-factor | 207.7 | 146 | 61.7 µs | **8.8 µs** |

Two independent pipelines agree on **~8.3 µs of launch per dispatch**, which is what makes the split
trustworthy. From it:

* **The 3-factor's advantage is arithmetic**, 121 vs 146 µs of kernel time — the `O(N·N^{1/3})` vs
  `O(N·√N)` MAC argument of [`ntt_32768.md`](ntt_32768.md), now measured on p150 rather than asserted.
* **Its deficit is launches**: 5 extra dispatches × ~8.3 µs ≈ **41 µs**, which more than cancels its
  25 µs kernel-time lead. That, and nothing else, is why it loses on host wall.
* Eleven of its twelve stages sit in a 14.3–17.1 µs band — roughly 8 µs launch on ~6–9 µs of kernel.
  **Only `A2A` (34.0) carries substantial work.** On the 2-factor side only the two matmuls do, and
  `M1` alone (74.3) exceeds any single 3-factor stage by 2.2×.

> **A correction, recorded because the wrong version was written first.** An earlier draft put the
> floor at 14.5 µs by taking the cheapest observed *stage* as pure overhead, and concluded the
> 3-factor did **2.7×** less compute. That stage contains real kernel time; the floor is ~8.3 µs and
> the true ratio is **1.21×**, measured directly by the trace rather than modelled. The qualitative
> conclusion survived, the magnitude did not — which is the argument for taking the trace figure
> instead of inferring compute from host-side numbers.

---

## 3. The optimised modmul: **UNRESOLVED [X]**, and why that is the interesting part

`compute/modmul_sfpi_1pass.cpp` is a drop-in for `modmul_sfpi_hybrid.cpp` (identical runtime-arg
interface) and measures **4.03×** standalone — 12.430 → 3.086 µs/tile,
[`../he/exp_modmul001.md`](../he/exp_modmul001.md). Swapped into the twiddle stages of both variants,
two warm reps each, `ITERS=20`, `stage-sum`:

| | 1pass | hybrid |
|---|--:|--:|
| 3-factor | 257.2, 256.9 | 265.4, 222.2 |
| 2-factor | 229.4, 217.2 | 219.3, 202.6 |

**Not resolved.** The hybrid arm's own two runs differ by 19% — larger than any effect being looked
for — and on these numbers `hybrid` would appear *faster*, which is noise, not a result. No claim
either way is supportable and none is made.

**The mechanism is understood; this is not a measurement failure — and §2 now puts a ceiling on it.**
The twiddle stages are **10.97 µs of the 3-factor's 123.2 µs critical path, 8.9%**, and 5.45 of the
2-factor's 147.9, 3.7%. So *even an infinitely fast modmul* saves at most 8.9%, and the 1pass↔hybrid
difference is a fraction of that: at `C=32` with `NTILE=32` each core handles **exactly one tile**, so
the stage is ~5.5 µs (1pass, measured) against ~7.3 µs (hybrid, projected from
`exp_modmul001`'s 3.086 vs 4.930 µs/tile) — **~3.6 µs over two stages, 2.9% of the critical path**,
which is inside the 2.5% run-to-run spread of the metric itself. The experiment could not have
resolved it, and now that is a measured statement rather than an inference.

> Carried forward from [`../he/exp_modmul001.md`](../he/exp_modmul001.md): **an optimisation's value
> is set by the bottleneck, not by what it removes.** A kernel measured at 4.03× on one core, one tile
> at a time, contributes nothing detectable to a pipeline whose stages are launch-bound. Neither the
> kernel nor the 4.03× is wrong — this pipeline is not where it pays.

To resolve it: run at `C=8` or lower so each core gets 4+ tiles, or measure the twiddle stage alone
with a `T` sweep as `exp_modmul002` does. Neither was done here.

---

## 3b. The corner turn may have a cheaper answer than the exchange **[X]**

`docs/noc/exp_noc001.md` re-measured this stage's traffic pattern at exactly its parameters (32
participants, 128 B per message) with a third route neither this experiment nor
`all_to_all_corner_*.cpp` tried: **a strided DRAM read that fetches only the bytes it needs**, via
`addrgen.get_noc_addr(page, offset)` instead of `noc_async_read_tile`.

| route at `G=32`, 128 B | here | exp_noc001 |
|---|--:|--:|
| DRAM, whole pages, one NoC | 32.24 µs | 31.85 µs |
| core-to-core exchange | **4.44 µs** | 4.23 µs |
| DRAM, strided | not tried | **2.70 µs** |

The two experiments agree to ~3% on the routes they share, so the third number is worth taking
seriously: **1.64× better than the exchange**, with no semaphore, no `C == NT` launch constraint, and
no core-to-core dependency. That reframes §2b-sexies — most of that stage's 7.26× was never about
the medium, it was about reading a 4 KB page to use 128 bytes of it.

**The catch, which decides it and has not been measured.** The exchange also removed the receive-side
assembly (8.74 µs of the 24) by having the *sender* write into the receiver's final face layout. A
strided DRAM read returns rows that still need placing into face order. So the swap only pays if a
strided reader can address face offsets on the DRAM side directly. Until that is measured this is a
candidate, not a regression against the exchange.

## 4. What this says, and does not say

**Does say.** On p150 at `C=32`, both bit-exact, **on stock dataflow in both variants** (§2c): the
**3-factor is the faster algorithm** — 121 vs 146 µs of kernel time, 1.21×, resolved well outside the
spread. The **2-factor finishes sooner in
wall-clock** — 153.6 vs 184.0 µs — **entirely because it launches 7 programs instead of 12.**

**Does not say** which to use in a pipeline. That turns on whether the 12 dispatches can be collapsed,
which is scheduling, not arithmetic, and this bench does not answer it.

**The actionable consequence.** [`ntt_32768.md`](ntt_32768.md) already records the missing work:
*"Not yet applied: fusing decompose+matmul+recombine into one program (done for NTT-1024; both 32768
variants still run those as separate chained stages)."* This measurement prices it. Fusing each
`dec → mm → rec` triple takes the 3-factor from 12 dispatches to **4** (three fused blocks + the
corner turn), removing ~8 × 8.3 ≈ **66 µs of launch** — and additionally removing two full DRAM round
trips per block, which the corner-turn analysis in `exp_tg018m` prices separately. The same fusion
buys the 2-factor only 4 dispatches (~33 µs), because its cost is real matmul work that fusion does
not remove. Expected result: the 3-factor wins on *both* metrics.
**[P — cost model from the launch cost measured in §2, not a run.]**

**Not measured [X]:** any core count other than 32; the fusion above; the modmul A/B at an occupancy
where it could resolve (§3).

---

## 5. Running it

```
. ./env.sh
./build/exp_ntt001 [q] [seed]

# the pure figure — one variant, one NTT, nothing else in the capture
TT_METAL_DEVICE_PROFILER=1 NTT1_PROF=1 NTT1_ONLY=3f ./build/exp_ntt001
python3 tools/traces/busy_metrics.py \
  "$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv" --expect 12 --dataflow-ok
```

| env | meaning |
|---|---|
| `NTT1_CORES` | cores per stage (default 32; at 32 each core gets 1 tile — see §3) |
| `NTT1_ITERS` | repetitions inside each timing loop (default 20; 50 used here) |
| `NTT1_MODMUL` | `1pass` (default) or `hybrid` — the §3 A/B |
| `NTT1_ONLY` | `3f` \| `2f` \| `both`; defaults to `3f` under `NTT1_PROF` |
| `NTT1_PROF` | profile-only: one clean NTT, timing loops skipped, prints the `busy_metrics.py` line |
| `NTT1_CSV` | write the table |

`NTT1_ONLY` exists because `busy_metrics.py` gates on `--expect N` matching the host's dispatch count:
a capture holding both variants *and* the timing loops has 228 dispatches, not 12, and the gate
correctly refuses it. Under `NTT1_PROF` the skipped variant prints **`SKIPPED`**, never `0/32768` —
its mismatch counter is initialised to zero, so an unconditional print would report a pass for a check
that never ran.

**Discard the first run** — it JIT-compiles ~19 programs. If a run is killed mid-flight, or another
process holds the card, the next run **blocks on `CHIP_IN_USE_0_PCIe` rather than failing**, which
looks exactly like a hang in whatever you changed last; check `ps` for the holder before reaching for
`tt-smi -r`.
