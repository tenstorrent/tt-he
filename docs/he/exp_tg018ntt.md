# exp_Tg018ntt — the exp_ntt001 communication findings applied to ct_mul's NTT

**Status: DONE (measurement).** Headline: **50.863 → 23.991 ms/op (2.120×)**, `0/6029312`, in three
steps — taking the corner turn on-chip (§2), fusing the whole NTT into ONE dispatch around it (§4c),
and moving the base conversion off the SFPU onto the matrix engine (§4e). 2884 dispatches → 1278,
host wall 155.1 → 42.0 ms.

| step | ms/op | cumulative |
|---|--:|--:|
| baseline (`exp_Tg018m`) | 50.863 | |
| §2 core-to-core corner turn | 43.932 | 1.158× |
| §4c whole NTT in one dispatch | 32.448 | 1.568× |
| **§4e base conversion on the matrix engine** | **23.991** | **2.120×** |

Of the five communication rungs in
[`../matmul/exp_ntt001.md`](../matmul/exp_ntt001.md)'s 3.56× ladder, exp_Tg018m already had **four**.
Applying the fifth — the **core-to-core corner turn** — takes the whole ct_mul from **50.863 to 43.932
ms/op (1.158×)**, bit-exact `0/6029312` in both arms, with the corner-turn stage itself going
**9.108 → 2.122 ms (4.29×)** and every other kernel unchanged to ±0.03 ms.

**Read §1 before the number.** The headline of the source experiment was 3.56×; what was actually
transferable here was one rung, because the other four were already applied — and two of them had been
exported *from* this pipeline. A ladder's total is a property of the pipeline it was measured on.

`host/experiments/exp_tg018ntt.cpp` · `kernels/experiments/exp_tb001/dataflow/a2a_corner_x.cpp` · baseline
[`exp_tg018m.md`](exp_tg018m.md) · source of the idea [`../matmul/exp_ntt001.md`](../matmul/exp_ntt001.md)

---

## 1. What was left to transfer — the audit came first

exp_ntt001 raised five communication rungs. Before writing anything, each was checked against what
exp_Tg018m already does, because **four of the five were already there** — and two of them were
exported *from* this pipeline in the first place:

| rung (exp_ntt001) | already in exp_Tg018m? | evidence |
|---|---|---|
| deep matmul readers (`NTT1_DEEP`) | **yes** | `reader_fused_batch_pg.cpp` already issues all 17 reads before one barrier |
| fused dec+mm+rec triples (`NTT1_FUSE`) | **yes** | `kernels/experiments/exp_tb001/compute/fused_matmul_{a,c}.cpp` *are* the fused 1pass kernels; exp_ntt001 borrowed them from here |
| both-NoC corner turn (`A2A=bothnoc`) | **yes** | `a2a_corner_{rd,wr}.cpp`, split NCRISC/BRISC |
| **core-to-core corner turn (`A2A=exchange`)** | **NO** | ← what this bench adds |
| whole NTT in one kernel (`NTT1_MEGA`) | no | not attempted, §5 |

That the remaining rung aimed at the right term was luck worth checking: exp_Tg018m's verified 50.920 ms
census puts **`a2a_corner` at 9.103 ms, 17.9%** — its largest non-arithmetic item.

**This is why the audit is the first section.** Reading exp_ntt001's headline (3.56×) and applying its
whole ladder would have produced three no-ops and one real change, and the no-ops would have been
reported as part of the win.

---

## 2. The result

`TG18NTT_DEV=1 TG18NTT_TRACE=1`, `G=130`, `LPD=4`, `K=2`, traced resident replay. **host device busy**
is the mean over the bench's 3 internal iterations; runs are whole-binary repetitions. Quote `min`
([`../guide/profiling.md`](../guide/profiling.md) §0). **[M-p150]**

| arm | run 1 | run 2 | run 3 | min | spread | correctness |
|---|--:|--:|--:|--:|--:|---|
| `bothnoc` *(= exp_Tg018m)* | 50.863 | 50.884 | 50.897 | **50.863** | 0.07% | `0/6029312` |
| **`exchange`** | 43.932 | 44.018 | 43.955 | **43.932** | 0.20% | `0/6029312` |
| | | | | **1.158×** | | **−6.931 ms** |

At 0.07% and 0.20% spread against a 13.6% effect, this is resolved by three orders of magnitude — the
opposite of the situation in `exp_modmul001`'s host-wall A/B.

**The control reproduces `exp_tg018m.md`'s 50.920 ms to 0.1%**, which is what licenses comparing this
bench's exchange arm against that document's census.

**Both arms cache 2884 workloads** — the dispatch count is *identical*, so nothing here is a launch-count
effect. The two binaries differ in one stage and in nothing else.

### 2a. The mechanism, confirmed by a knob that stops mattering

`K` — outputs per core — exists in the DRAM arm *only* to amortise redundant reads: every core reads all
32 input tiles whether it produces one output or two, so `K=2` halves the reads per output.
`exp_tg018m.md` measured that and chose `K=2`. **If the exchange really removes the redundant reads,
then `K` should stop mattering in the exchange arm** — there is nothing left to amortise. It does:

| | `K=1` | `K=2` | sensitivity to `K` |
|---|--:|--:|--:|
| `bothnoc` | 60.040 | **50.863** | **18.0%** |
| `exchange` | 44.113 | **43.932** | **0.4%** |

**A 45× collapse in the knob's importance.** This is better evidence for *why* the exchange wins than
the 1.158× itself: the parameter that the DRAM arm needs in order to be tolerable is nearly inert once
each core reads only the tiles it owns. It also means the exchange arm did not simply inherit a
well-tuned constant — it would have been within 0.4% at either setting.

Incidentally this re-confirms `exp_tg018m.md`'s `K=2` choice on the DRAM arm (18% is a large effect
that a single run could not have been trusted for), and it means **the exchange's win is not sensitive
to a tuning decision made for a different implementation.**

---

## 3. Why it works, and what it cost to make it work

`a2a_corner_{rd,wr}` have **every** core read **all 32** input tiles (128 KB) in order to take `K=2`
rows out of each. exp_ntt001 §2b-quinquies measured that shape directly with a probe kernel and found
it **64% read+write+launch, 36% L1 assembly** — read-bound. But the rows every core wants are already
on-chip in some other core's L1, so the DRAM traffic is redundant.

`a2a_corner_x.cpp` has each core read only the `K` tiles it **owns** (8 KB, not 128) and push each row
straight into the peer that needs it. **It removes the assembly as well as the read**, which the DRAM
version cannot: the sender knows where its row lands, so it writes into the receiver's final
face-tiled offsets directly and the receiver assembles nothing.

Two things differ from `all_to_all_corner_x.cpp`, and neither is cosmetic:

* **Page stride.** exp_Tg018 interleaves the two half-transforms in one 64-page buffer, so input tile
  `n1` is page `n1*2+kh`, not page `n1`. (Aside, and it is a finding of its own: `bank = page % 8`, so a
  stride of 2 touches only **4 of the 8 DRAM banks**. That costs the DRAM arm, which reads 32 such
  pages per core, and costs the exchange almost nothing, which reads 2.)
* **`K > 1` tiles per core.** `all_to_all_corner_x` assumes one tile per core and therefore 32 cores per
  transform. exp_Tg018m gives each core `K=2` outputs — its own measured optimum. Generalising the
  kernel to `NPC = NT/P` tiles per core keeps the row assignment identical to `a2a_corner_wr`'s
  (`out_start=me`, `out_stride=P`), which is what makes this a like-for-like swap rather than a
  reshaped stage, and is why the dispatch count did not move.

**The host change that was actually required.** The DRAM arm chunks jobs with `base += NC`, which can
put part of a 16-core block in one dispatch and the rest in the next. That is harmless when cores are
independent and it **deadlocks an exchange**, because every core waits for all `P` peers of its block.
The exchange arm therefore chunks in **whole blocks**: `NC/P = 8` blocks of 16 = 128 of 130 cores, so
two cores idle. That is the price of the barrier, and it is charged in the numbers above.

It did not deadlock on the first run. The protocol (in the kernel header) is: scatter, **flush**, then
increment every peer's semaphore — so an announcement always follows the data it announces — then wait
for `P`, then write out.

---

## 4. Per-kernel census — the win is entirely in the stage it was aimed at

`TG18NTT_DRAIN_ONLY=1` + `dispatch_csv.py --expect 2884 --drain --dataflow-ok --labels`. Both captures
passed the gate — *"join verified: all 2884 rows agree on the kernel between host and trace"* —
which is required before a per-kernel breakdown is quotable
([`../guide/profiling.md`](../guide/profiling.md) §4). `--dataflow-ok` is not optional
here: the corner turn is a dataflow-only stage and is silently dropped without it. **[M-p150]**

| kernel | dispatches | `bothnoc` ms | `exchange` ms | Δ |
|---|--:|--:|--:|--:|
| `fbc_combine` | 15 | 10.667 | 10.669 | +0.002 |
| `modmul_fused3` | 1088 | 10.481 | 10.490 | +0.009 |
| `fusedAcc_total` | 304 | 5.382 | 5.353 | −0.029 |
| `fusedC_total` | 304 | 3.757 | 3.754 | −0.003 |
| `fused_total` | 304 | 3.742 | 3.759 | +0.017 |
| **`a2a_corner`** | **152** | **9.108** | **2.122** | **−6.986** |
| `modmul_scalar` | 142 | 2.338 | 2.342 | +0.004 |
| `modadd_map` | 328 | 1.622 | 1.617 | −0.005 |
| `modsub_reduce` | 142 | 0.939 | 0.934 | −0.005 |
| `fbc_scale` | 5 | 0.569 | 0.570 | +0.001 |
| `modadd_scalar` | 96 | 0.452 | 0.452 | 0.000 |
| `bconv_one` | 4 | 0.391 | 0.398 | +0.007 |
| **critical path busy** | 2884 | **49.448** | **42.460** | **−6.988** |

**Every other kernel is within ±0.03 ms.** The corner turn went **9.108 → 2.122 ms, 4.29×**, and it
accounts for **6.986 of the 6.988 ms** the critical path lost. This is a controlled result in the same
sense as exp_ntt001 §2a-bis, where only the matmul stages moved.

Three independent numbers agree, which is the real check:

* the corner turn's `bothnoc` cost, **9.108 ms**, reproduces `exp_tg018m.md`'s census of **9.103 ms**
  — a different binary, a different session;
* the critical-path delta, **−6.988 ms**, matches the host-measured device-busy delta, **−6.931 ms**,
  to 0.06 ms — two metrics from different instruments;
* the ratio on the stage, **4.29×**, is the same mechanism exp_ntt001 measured at **5.40×** on
  NTT-32768. Lower here, and §3 says why: this corner turn already carries `K=2`, so its DRAM arm was
  the *better* of the two baselines exp_ntt001 compared against.

**What now dominates.** With the corner turn at 5.0%, the profile has inverted onto arithmetic:
`fbc_combine` **25.1%** and `modmul_fused3` **24.7%** are together half the multiply, and the three
fused NTT matmuls another 30.3%. There is no large communication term left in this pipeline —
which is the honest reading of why the remaining exp_ntt001 rung (§5) is the only one worth trying.

---

## 4b. Why the exchange still touches DRAM, and what removing it would be worth

The exchange is *core-to-core for the shuffle* but it still does **one DRAM read and one DRAM write per
core** — it reads the `K` tiles it owns and writes the `K` it assembles. The obvious question is why
that cannot go too.

**What forces it: the corner turn is its own dispatch, and its neighbours map work to cores
differently.** L1 is addressable only within a program; anything crossing a dispatch boundary has to
land somewhere both programs can name, and that is DRAM. Worse, even a persistent-L1 scheme would not
work here as the passes are currently written:

| pass | how it assigns work | core `c` holds |
|---|---|---|
| `twpass` / `mmpass` (producer, consumer) | `for (base = 0; base < B*NTILE; base += NC)` | job `base+c` → tile `(base+c) % 64` of transform `(base+c)/64` |
| `a2apass` | blocks of `P=16` cores per (transform, half) | input tiles `{c, c+16}` of half `kh` |

Two mismatches, not one: the **mapping** differs (a2a core `c` wants tiles a different core produced),
and the **chunk boundaries** differ (130 vs the exchange's `8×16=128`), so the producer of a tile is
often in a different *dispatch* as well as on a different core.

**What it is worth: measured, not modelled.** `a2a_corner_x_probe.cpp` is the exchange with both DRAM
accesses deleted — the scatter, the semaphore barrier and the launch are byte-identical, so
`(real − probe)` is exactly the DRAM half. It computes a wrong answer by construction and **the gate
confirmed that** (`6029312/6029312` FAIL), which is what makes the timing usable. **[M-p150]**

| corner turn | `a2a_corner` | critical path | host device busy |
|---|--:|--:|--:|
| `bothnoc` — DRAM round trip, 32 reads/core | 9.108 ms | 49.448 | 50.863 |
| `exchange` — NoC shuffle, `K` reads + `K` writes/core | **2.122 ms** | 42.460 | 43.932 |
| `xprobe` — NoC shuffle, **no DRAM at all** *(wrong answer)* | **0.858 ms** | 41.210 | 42.780 |
| **so the gap left by the DRAM is** | **−1.264 ms** | **−1.250** | **−1.152** |
| | **2.47× on the stage** | 1.030× | **1.027×** |

**The remaining DRAM is 60% of the stage** — more than the intuition that "it is only two pages per core
now" suggests — but only **2.6% of the multiply**, against the 13.6% the first exchange bought. All
three metrics agree on the gap to within 0.11 ms.

### The residual DRAM is already at the bandwidth ceiling

This is what decides whether the gap is worth chasing *by tuning* rather than by fusion. The exchange's
residual traffic is exactly countable — 128 cores × (`K` tiles in + `K` tiles out) × 4 KB × 152
dispatches:

| | traffic per multiply | time | rate |
|---|--:|--:|--:|
| `exchange` residual DRAM | **318.8 MB** | 1.264 ms | **252 GB/s** |

**252 GB/s is the practical ceiling** in [`dram_guide.md`](dram_guide.md) §1. So there is **no
access-pattern fix left** in this stage: the two pages per core are not latency-bound, not
bank-starved, not depth-starved — they are moving at the speed the card can move bytes. (That also
settles the §3 aside: the stride-2 page mapping touches 4 of 8 banks *within a buffer*, but eight
different jobs are in flight per dispatch on eight different buffers, so the aggregate is not
bank-limited.)

The corollary is the useful part: **the only way to close this gap is to not do the traffic at all**,
which means fusion. Note also what this says about the win already banked — the `bothnoc` arm was not
slow because it accessed DRAM *badly*; it requested **8.5× more bytes** (2.71 GB against 0.319 GB).
The exchange won by moving less data, not by moving it better.

**So a fully DRAM-free corner turn is worth ~1.15-1.26 ms, and it is reachable — this is exactly
exp_ntt001's `NTT1_MEGA` rung**, whose newest measurement is "whole NTT in one kernel, no DRAM during
it". The path here is narrower than a full mega-kernel, though: the a2a's neighbours (`twpass` →
`a2apass` → `mmpass`) are all *per-tile independent* apart from the shuffle itself, so fusing just
those three needs
1. the twiddle pass re-mapped so core `c` produces tiles `{c, c+16}` of half `kh` — a host-side job
   reordering, no kernel change; and
2. one program where BRISC runs the exchange while the compute kernel is parked, which is the
   structure `ntt32768_3f_fused_wr.cpp` already uses.

Not attempted here. **[X]** The reason to record the 0.858 ms is that it bounds the prize: fusing the
corner turn's neighbours cannot buy more than 2.9%, so it should be judged against fusing the passes
that are now actually large (`fbc_combine` 25.1%, `modmul_fused3` 24.7%).

---

## 4c. The whole NTT as ONE dispatch, corner turn on-chip — `TG18NTT_FUSE=1`

§4b said the only way to close the remaining gap was fusion. It was done. `ntt64_fused.cpp` runs all
six NTT passes in one compute kernel with the corner turn core-to-core in the middle, so **no
intermediate reaches DRAM at all**: the kernel reads its input tiles and the folded constants once and
writes its outputs once.

**The decomposition that makes it fit.** Every pass except the corner turn is tile-local, and the
corner turn's ownership is a *fixed point*: it maps input tile `n1` to output row `n1` within a half,
so a core's tile indices are unchanged across it. P1 pairs pages `2n1` and `2n1+1` — the two halves of
the same `n1` — which forces a core to own both halves and then hands it all four tiles for free:

* a block is **P=16 cores and carries one transform**; core `me` owns `n1 ∈ {me, me+16}`
* that is pages `{2me, 2me+1, 2me+32, 2me+33}` — **4 tiles/core × 16 cores = 64 = one transform**
* 8 blocks = 128 cores = **8 transforms per dispatch**

BRISC runs the 16-core exchange while the compute kernel is parked on `cb_d`, then writes the results.

| rung | dispatches | host device busy | host wall | correctness |
|---|--:|--:|--:|---|
| `bothnoc` *(= exp_Tg018m)* | 2884 | 50.863 | 155.1 | `0/6029312` |
| `exchange` (§2) | 2884 | 43.932 | 155.4 | `0/6029312` |
| **`FUSE=1`** (this) | **1288** | **32.448** | **41.3** | `0/6029312` |
| | **2.24× fewer** | **1.568×** | **3.76×** | |

Reps 32.448 / 32.452 — 0.01% spread. **The host wall collapses onto kernel time**, 155.1 → 41.3 ms,
exactly as exp_ntt001 §2b-septies saw when it reached one dispatch: there is no longer a per-stage
launch and inter-stage host stall for every one of six passes.

### How it was debugged, because the method is the transferable part

The first run was **wrong on every word of every page** — and it did *not* deadlock, so the barrier and
the park were never in question. Two staged bisects found it without a single guess surviving:

1. **`FUSE_DBG=1` — the fused kernel does ONLY the corner turn.** Result: `0/65536` against the DRAM
   corner turn. That cleared the plumbing, the page mapping, the local-slot layout, the exchange, the
   semaphore barrier and the writer in one measurement, and localised the bug to an arithmetic phase.
2. **`FUSE_DBG=k` — run the first `k−1` phases, then hand the result to the writer**, compared against
   the unfused chain's own scratch buffer for that pass. **Phase 1 (P1) already differed**, so nothing
   downstream needed looking at.

**The bug: `mkbufW` allocates the W folds in L1, not DRAM** (`BufferType::DRAM` is behind the
`TG18NTT_W_DRAM` A/B). The new reader addressed them with a hardcoded `InterleavedAddrGenFast<true>`
— a *DRAM* address generator — and read garbage, silently. The existing readers use `TensorAccessor`
with `TensorAccessorArgs` appended to the compile args precisely because that carries the buffer's real
type. Fixed by doing the same; the input and twiddle buffers come from `mkbuf` and really are DRAM,
which is why `FUSE_DBG=1` passed and why only the constant-reading phases were wrong.

> **The lesson is about the instrument, not the bug.** A hardcoded address generator that disagrees
> with the buffer's type fails *silently and totally* — it cannot be caught by inspection of the
> arithmetic, which is where the search would naturally go. What caught it was a bisect whose first
> step deliberately used **none** of the constants.

**Still on the table [X]:** the fused kernel reads all six folded constant sets per dispatch (~112 KB
per core) where the unfused passes each read only what they needed; at 8 transforms per dispatch that
is amortised, but neither `G` nor the transforms-per-dispatch were swept. And `fbc_combine` (now the
largest single term) is untouched.

---

## 4d. How much of `ct_mul` is SFPU work? **86.9%** — superseded by §4e, now **73–82%**

> **This section measures the arm WITHOUT §4e.** Its census and its 86.9% are correct for the 32.448 ms
> `fbc_combine` build and are kept because §4e's controlled A/B is against exactly this table. For the
> current best `ct_mul` the figure is **73–82%** — see §4e. The qualitative conclusion is unchanged.

Asked of the then-best `ct_mul` (32.448 ms, `TG18NTT_FUSE=1`). Two measurements, because after the fusion
one kernel does both kinds of work and a per-kernel census can no longer separate them. **[M-p150]**

**Step 1 — the fused arm's census**, `--expect 1288 --drain --dataflow-ok --labels`, join verified,
`0/6029312`:

| kernel | dispatches | ms | % | engine |
|---|--:|--:|--:|---|
| **`ntt64_fused`** | 76 | **11.065** | 35.2 | **MIXED** — matrix + SFPU |
| `fbc_combine` | 15 | 10.666 | 33.9 | SFPU |
| `modmul_fused3` | 480 | 3.422 | 10.9 | SFPU |
| `modmul_scalar` | 142 | 2.334 | 7.4 | SFPU |
| `modadd_map` | 328 | 1.610 | 5.1 | SFPU |
| `modsub_reduce` | 142 | 0.926 | 2.9 | SFPU |
| `fbc_scale` | 5 | 0.571 | 1.8 | SFPU |
| `modadd_scalar` | 96 | 0.452 | 1.4 | SFPU |
| `bconv_one` | 4 | 0.397 | 1.3 | SFPU |
| **critical path busy** | **1288** | **31.443** | | |

Every kernel on the path was classified by grepping for `matmul_tiles`: **only `ntt64_fused` uses the
matrix engine at all.** Note `a2a_corner` has vanished as a line item — the corner turn is now inside
the fused kernel, on BRISC.

**Step 2 — split the mixed kernel** with `NTT64_PHASE_ZONES` (`TG18NTT_ZONES=1`, drained at K=20).
`ph_sfpu` wraps the decompose / recombine / twiddle passes, `ph_matmul` the `matmul_tiles` loops:

| zone | share of `ntt64_fused` |
|---|--:|
| `ph_sfpu` | **62.7%** |
| `ph_matmul` | **5.9%** |
| remainder (a2a park, CB sync, engine reconfig) | 31.4% |

The zone *counts* validate the split independently: 11136 `ph_sfpu` to 4248 `ph_matmul` = **2.62**,
against **2.67** predicted from the kernel's structure (32 SFPI passes to 12 matmul blocks per
dispatch). If the zones had been misplaced this ratio would not land.

### The answer

| | ms | share of `ct_mul` |
|---|--:|--:|
| **SFPU** | **27.320** | **86.9%** |
| matrix engine | 0.648 | **2.1%** |
| other — a2a park, CB sync, engine reconfig | 3.474 | 11.0% |
| **total (critical path busy)** | **31.443** | |

**CKKS `ct_mul` is a vector workload, not a matrix workload.** The matrix engine — the unit Blackhole
is built around, and the one carrying its 664 TFLOP/s headline — accounts for **2.1%** of the
multiply. Everything else is the SFPU.

**Why the conclusion is robust to how the mixed kernel is attributed.** Even charging the *entire*
31.4% remainder to the matrix engine puts it at 13%, and the true figure cannot be that high because
the remainder is dominated by the corner-turn park, which is BRISC waiting on the NoC. The direction
is not in doubt.

**One thing this figure is not.** The 86.9% is time in *SFPU-class work*, which includes the DST↔L1
`copy_tile`/`pack_tile` that surrounds every SFPI pass — `ph_sfpu` brackets the whole
acquire→release, while `ph_matmul` brackets only the matmul issue. So it is "the fraction of the
multiply spent doing vector work and moving its operands", not "the fraction of cycles the SFPU ALU
is retiring instructions". The former is the right number for deciding what hardware matters; the
latter would need instruction-level counters this part does not expose.

### What follows

* **It bounds what the matrix engine can rescue.** Moving more of the multiply onto it attacks
  `fbc_combine`, which is **33.9%** here and pure SFPU. That is the largest single lever left, and this
  measurement is what identifies it. **[DONE — §4e]**, worth 1.355× on the whole multiply.

  This bullet originally read *"**[X — not applied to this pipeline.]**"*, which was wrong: the kernel
  (`fbc_fpu_tr.cpp`) was already in this tree, already measured by `exp_tg018r` at 1.19× (8→8) to 10.3×
  (32→32). A census that identifies a lever is not evidence the lever is unbuilt — **grep for the
  kernel before writing it up as future work.**

---

## 4e. The base conversion moved to the MATRIX ENGINE — `TG18NTT_FBC=fpu`: 32.498 → 23.991 ms

**Status: DONE (measurement).** §4d's closing paragraph named `fbc_combine` as "the largest single
lever left" and marked it **[X — not applied]**. That was wrong twice over: the kernel that applies it
(`kernels/experiments/exp_tb001/compute/fbc_fpu_tr.cpp`) **already existed in this tree**, built and measured by
`exp_tg018r`, and had merely never been wired into the multiply. §4d re-derived a lever that was
sitting finished one directory away. It is now wired in. **[M-p150]**

| arm | host device busy | critical path busy | dispatches | correctness |
|---|--:|--:|--:|---|
| `fbc_combine` (SFPU) | 32.498 ms | 31.443 ms | 1288 | `0/6029312` |
| **`TG18NTT_FBC=fpu`** | **23.991 ms** | **22.917 ms** | 1278 | `0/6029312` |
| | **1.355×** | **1.372×** | | |

Matched pair: same binary, same session, `TG18NTT_ITERS=10`, only the flag differs. **−8.507 ms.**

### The win is entirely in the stage it was aimed at

Re-running the §4d census against the FPU arm, every other kernel reproduces to **±0.9%**:

| kernel | SFPU arm | FPU arm | Δ |
|---|--:|--:|--:|
| `ntt64_fused` | 11.065 | 11.16 | +0.9% |
| `modmul_fused3` | 3.422 | 3.41 | −0.4% |
| `modmul_scalar` | 2.334 | 2.30 | −1.5% |
| `modadd_map` | 1.610 | 1.60 | −0.6% |
| `modsub_reduce` | 0.926 | 0.91 | −1.7% |
| `fbc_scale` | 0.571 | 0.57 | −0.2% |
| `modadd_scalar` | 0.452 | 0.43 | −4.9% |
| `bconv_one` | 0.397 | 0.40 | +0.8% |
| **base conversion** | **10.666** `fbc_combine` | **≈2.14** `fbc_fpu_tr` | **≈5.0×** |
| **critical path busy** | **31.443** | **22.917** | **1.372×** |

**The 2.14 ms is a residual, not a sample.** `fbc_fpu_tr` is only 5 dispatches of 1278 and they landed
in the drained capture's dropped-marker region, so `kernel_census.py` reported *"no trace sample for
['fbc_fpu_tr'] — their contribution is MISSING from the total"* and summed the rest to 20.79 ms. The
2.14 is 22.917 (measured, 1-1 map verified over all 1278 dispatches) minus that 20.79. Quote it as
"about 2 ms", and treat the ±0.9% agreement on the other eight rows — not the residual — as the
evidence that nothing outside the FBC moved.

### Two things kept it out for this long, and neither was the arithmetic

`exp_tg018r` shows both routes bit-exact against one golden, so there was never a correctness
question. What blocked it was plumbing:

* **Layout.** `reader_fbc_tr.cpp` addresses the scaled operand as `y.get_noc_addr(ot*L + i)` — ONE
  interleaved buffer holding every limb. This pipeline keeps `L` separate 64-page buffers, so that
  page does not exist. `reader_fbc_tr_pg.cpp` / `writer_fbc_tr_pg.cpp` take the `L` base addresses as
  runtime args and read page `ot` from each; same transpose, same CB contract, nothing else changed.
* **The constants are a BUFFER, not runtime args.** The folded int8 matrix is 16 tiles, so unlike the
  SFPU route's `C2` it cannot ride in the arg list. It is built **once per (source set, destination
  chunk)** and cached: the multiply has 5 FBC call sites with fixed prime sets, so the cache fills on
  the record pass and **no DRAM write ever happens inside a traced replay**. The cache key must include
  the *source* set as well as the destination chunk — the matrix is built from `shat`, which is derived
  from the sources.

Then two follow-ups, both mechanical:

* **Fill the grid.** One dispatch per destination chunk uses 64 of 130 cores. The chunk loop and the
  tile loop are independent, so they flatten into one job list exactly as the SFPU route already does:
  26.003 → **25.162 ms**. Unlike the SFPU route this program's CB depths do not depend on `Ld`, so jobs
  of different `Ld` may share a dispatch and no grouping-by-`Ld` is needed.
* **Widen the chunk.** `Ld` ≤ 32 is an *algorithmic* limit — the matrix engine contracts within a tile
  and the contracted axis is the source limb, so one tile holds at most 32 destinations. Widening
  amortises the operand decompose, which is redone per chunk.

### The width sweep, and one width that is wrong

| `TG18NTT_FBC_CH` | chunks of Ldt=48 | correctness | host device busy |
|---|---|---|--:|
| 8 | 6 | `0/6029312` | — |
| 16 | 3 | `0/6029312` | 25.162 |
| **24** | **2** | **`0/6029312`** | **23.991** |
| 32 | 2 (32+16) | **`262144/262144`** | — |

**`Ld = 32` comes back wrong on every word, and I did not root-cause it.** What is established:

* `Ld = 24` is green, so it is **not** a "wider than 16" problem — the upper-face path (`j ≥ 16`) works.
* `exp_tg018r` is green at exactly `Ls=16, Ld=32` (`mism 0`), so it is **not** the compute kernel.
* Therefore it is in `reader_fbc_tr_pg` / `writer_fbc_tr_pg` at exactly the full-tile width, i.e. in
  the code this port added. The most likely candidate is the writer's 64 outstanding
  `noc_async_write`s per group behind a single barrier, but that is a hypothesis, not a finding.

24 splits `Ldt = 48` into two even chunks, which is all the width was wanted for, so it is the
**default** and the bench now **refuses** `TG18NTT_FBC_CH > 24` rather than leaving a silent trap.
Recovering 32 is worth ~0.3 ms — the extra decompose — and is not worth a wrong answer.

### What this does to §4d's 86.9%

**It invalidates the exact figure, in the expected direction.** `fbc_combine`'s 10.666 ms was *pure*
SFPU; `fbc_fpu_tr` is mixed, and its ~2.14 ms cannot be split without phase zones this port did not
add. So the SFPU share can only be bounded:

| | ms | share of 22.917 |
|---|--:|--:|
| SFPU, lower bound (charge all of `fbc_fpu_tr` to the matrix engine) | 16.654 | **72.7%** |
| SFPU, upper bound (charge all of it to the SFPU) | 18.794 | **82.0%** |

**So `ct_mul` is now 73–82% SFPU, down from 86.9% — and §4d's qualitative conclusion survives
unchanged.** The multiply is still a vector workload; the largest lever available to the matrix engine
has now been pulled, and it moved the SFPU share by at most 14 points while leaving the SFPU as the
overwhelming majority of the work. The next-largest pure-SFPU line item is `modmul_fused3` at 3.41 ms
(14.9%), an eighth the size of the one just removed.

---

## 5. What this does and does not say

**Does say.** On p150, at `L_Q=48`, `dnum=3`, `N=65536`, `G=130`, bit-exact: replacing the corner
turn's DRAM round trip with a core-to-core NoC exchange is worth **1.158× on the whole multiply**. The
idea transferred from a different pipeline (NTT-32768) at a different size, which is the useful part —
`docs/matmul/exp_ntt001.md` §2b-sexies measured 5.40× *on that stage*, and this reproduces the
mechanism, not the headline.

**Does not say** that exp_ntt001's 3.56× transfers. It does not, and it was never going to: four of
its five rungs were already applied here, two of them having originated here. **A ladder's total is a
property of the pipeline it was measured on**; only the rungs a target lacks are available to it.

**Not attempted [X]:**

* **`NTT1_MEGA` — the whole NTT in one kernel.** exp_ntt001's largest remaining rung (1.34× there, and
  it collapsed host wall onto kernel time). It is a much bigger lift here: exp_Tg018's NTT is a
  three-level (32,32,64) transform over 64 tiles with limb batching across dispatches (`LPD=4`), not
  32768 over 32 tiles with one tile per core, so the mega-kernel's "every intermediate stays in L1"
  premise has to be re-established at a different working-set size.
* The bank-stride observation in §3: splitting the interleaved 64-page buffer so the corner turn's
  pages are consecutive would give the **DRAM** arm all 8 banks. Only interesting if the exchange were
  rejected, which §2a says it should not be.
* **`G` other than 130 and `LPD` other than 4.** Both inherited from exp_Tg018m. The exchange's
  whole-block chunking changes the occupancy arithmetic — it uses `⌊G/16⌋·16` cores, so `G=128` would
  waste none where `G=130` wastes two — and neither was re-swept.

---

## 6. Running it

```
. ./env.sh
# the four rungs, same binary
TG18NTT_DEV=1 TG18NTT_TRACE=1 TG18NTT_A2A=bothnoc  ./build/exp_tg018ntt   # 50.863 -- the control
TG18NTT_DEV=1 TG18NTT_TRACE=1 TG18NTT_A2A=exchange ./build/exp_tg018ntt   # 43.932 -- a2a on-chip
TG18NTT_DEV=1 TG18NTT_TRACE=1 TG18NTT_FUSE=1       ./build/exp_tg018ntt   # 32.448 -- one dispatch
TG18NTT_DEV=1 TG18NTT_TRACE=1 TG18NTT_FUSE=1 TG18NTT_FBC=fpu \
                                                   ./build/exp_tg018ntt   # 23.991 -- FBC on the FPU

# the NTT-only gate, ~10x faster to iterate on than the whole multiply
TG18NTT_DEV=1 TG18NTT_FUSE=1 TG18NTT_NTTCHK=1 TG18NTT_NTTCHK_ONLY=1 ./build/exp_tg018ntt
```

| env | meaning |
|---|---|
| **`TG18NTT_FUSE`** | 1 = the whole NTT in ONE dispatch, corner turn on-chip (§4c). Subsumes `TG18NTT_A2A`: there is no separate corner-turn stage left to select |
| `TG18NTT_A2A` | `bothnoc` (default, the control), `exchange`, or `xprobe` (§4b — DRAM deleted, **wrong answer by design**, the gate must fail) |
| `TG18NTT_NTTCHK` | add **[D0]**: the fused NTT against the unfused chain on device, tile for tile, both directions. `TG18NTT_NTTCHK_ONLY` stops there |
| `TG18NTT_FUSE_DBG` | the §4c bisect. `1` = corner turn only (reference: the DRAM corner turn); `2..5` = the first k−1 pre-exchange phases (reference: the unfused chain's scratch buffer for that pass). Both compute wrong answers for the full NTT — that is the point |
| `TG18NTT_A2A_OUT` | `K`, outputs per core (default 2); the exchange uses `NHALF/K` cores per block |
| `TG18NTT_G` | cores (default 130) |
| `TG18NTT_LPD` | limbs per dispatch (default 4) |
| `TG18NTT_DEV` / `TG18NTT_CHAIN` / `TG18NTT_TRACE` | device ladder / whole multiply / traced resident |
| `TG18NTT_DRAIN` | drained capture for the per-kernel census (§4) |
| `TG18NTT_ZONES` | split `ntt64_fused` into its SFPU and matrix-engine phases (§4d); needs `TG18NTT_DRAIN_ONLY` |
| **`TG18NTT_FBC`** | `fpu` = run the base conversion on the MATRIX ENGINE (§4e), 1.355× on the whole multiply. Anything else keeps the SFPU `fbc_combine` |
| `TG18NTT_FBC_CH` | destination primes per FBC chunk, default **24**. `>24` is **refused**: 32 is the algorithmic limit but returns a wrong answer through this port's readers (§4e) |

Every other `TG18NTT_*` variable is exp_Tg018m's, renamed; see [`exp_tg018m.md`](exp_tg018m.md).

**The exchange requires whole blocks of `NHALF/K` cores in a dispatch.** The host enforces this by
chunking in blocks; if you change `G` to something below `NHALF/K` the block cannot be formed and the
stage would hang rather than run slowly.
