# exp_dram001 — six proposed DRAM levers, tested: five refuted, one already known

**Status: DONE (measurement).** Six levers were proposed against
[`dram_guide.md`](dram_guide.md)'s DRAM figures. **Five do nothing or hurt; the sixth was already
known and already applied by the guide.** The value here is therefore mostly *negative* — five
plausible ideas refuted — plus one refinement to an existing null result. Every point checksum-gated.
**[M-p150]**

> **READ THIS BEFORE THE NUMBER.** `dram_guide.md`'s **433 GB/s already drives both NoCs for reads**:
> its §3.1 rows are *"writer read traffic: none / EW=96 / EW=192"*, and `EW=192` means the writer's
> RISC was given its own read work. §3.4 states it directly — *"the kernel that only reads in its
> reader gets half the fabric; give the writer's RISC read work of its own."* So lever #6 is **not a
> new technique**, and **445.7 against 433 is +2.9% at a different configuration (130 cores vs G=16)
> — a reproduction, not an improvement.** An earlier version of this document framed it as beating the
> ceiling. That was wrong.

| | pure-read GB/s | vs `dram_guide` |
|---|--:|--:|
| one NoC, best of every other lever | 240.9 | ≈ its 234.8 |
| **both NoCs reading** | **445.7** | ≈ its **433** (+2.9%) |

`host/experiments/exp_dram001.cpp` · `kernels/experiments/exp_dram001/` · baseline [`dram_guide.md`](dram_guide.md) ·
motivation [`../vector/exp_roofline001.md`](../vector/exp_roofline001.md) §5b

---

## 1. The six levers, scored

Pure DRAM reads, 256 MB, 130 cores, min of 5 traced replays, every point verified by a per-core XOR
checksum against a host golden.

| # | lever | verdict | evidence |
|---|---|---|---|
| 1 | **16 KB packets** (`NOC_MAX_BURST_SIZE`) instead of 4 KB | **No — slightly worse** | 4 KB **240.9**, 16 KB 235.4 |
| 2 | **In-flight depth as a trid sliding window** | **No — ~2%** | TRID 1/2/4 → 236.4 / 239.7 / **240.9** |
| 3 | **~4 requesters per channel** | **Not enough** | 4 adjacent readers give a bank **28.0–31.3 GB/s**, about half its 53.9 saturated figure |
| 4 | **Readers adjacent to their bank** | **Helps pinning; pinning still loses** | at 32 cores pin 162.4 → pinadj **193.5** (+19%); at 64, 179.3 → **221.4** (+23%). But plain interleaving beats both: **239.1** |
| 5 | **Balance across 8 banks, verify each** | **Already balanced** | all 8 within 12% (28.0–31.3) — no concentration to fix |
| 6 | **Both NoCs reading** | **Yes, 1.86× — but already known** | 239.0 → **445.7 GB/s**, against the guide's own 433 |

**Only #6 moves it, and the guide had already found it.** The 1.86× is a valid controlled A/B *inside
this experiment* (same kernel, same cores, only the NoC split changes), and it is the right way to see
the size of the effect — but it is re-deriving `dram_guide.md` §3.1's 1.79×, not adding to it.

What is genuinely new is small: 445.7 is **2.2% above the 436 denominator** and **2.9% above the 433
achieved**, i.e. that denominator is *mildly* conservative — 55.7 GB/s/channel, 86% of the 65 nominal
rather than 83%. At a 2–3% margin between different configurations, treat 436 and 445.7 as **the same
number**; the useful content of this experiment is §§2–4, not this row.

## 2. What the one-NoC plateau really is

The striking result is the *flatness*: on one NoC, **nothing helps**. Packet size, window depth, bank
affinity, adjacency and core count all land between 235 and 241 GB/s. That plateau is not an access
pattern that can be tuned — it is the NoC itself, and three independent measurements now agree on it:

| source | one-NoC read bandwidth |
|---|--:|
| `dram_guide.md` §3.1 ("none (NOC_1 only)") | 234.8 |
| `exp_roofline001` §5b (reads-only probe) | 237.5 |
| this experiment (best of four levers) | **240.9** |

**~238 ± 3 GB/s across three unrelated kernels.** Any single-NoC reader is at that wall regardless of
how it is written, and the only way off it is to put reads on both.

## 3. Bank pinning and adjacency — the guide's null result, refined

`dram_guide.md`'s null table says pinning cores to channels is **worse** (189 vs 235 GB/s at G=16).
That is confirmed, but it was conflating two things, and separating them changes the advice:

| cores | interleaved | pinned | pinned **+ adjacent** |
|---|--:|--:|--:|
| 32 | 168.1 | 162.4 | **193.5** |
| 64 | 185.8 | 179.3 | **221.4** |
| 128/130 | **239.1** | 223.0 | 217.3 |

**Adjacency is a real and large effect on a pinned layout — +19% and +23% at 32 and 64 cores** — and at
those core counts pinned-and-adjacent *beats* interleaving. What it cannot do is beat interleaving once
the grid is full, because interleaving already spreads every core's traffic over all eight channels and
needs no locality to do it. So: **"pinning is worse" stands as advice, but the reason is not proximity —
it is that pinning gives up the free spreading that interleaving provides.**

## 4. The checksum earned its place twice

Both bugs made the bandwidth look **better**, which is the failure mode a bandwidth benchmark cannot
self-detect:

* **Two banks claimed the same worker.** `use[i] = near[b][i/NB]` is not a bijection — two banks can
  have the same nearest core, `SetRuntimeArgs` then overwrote one slot, and those reads never happened.
  Reported as `BAD`, fixed with a greedy unique assignment.
* **Both NoC readers shared one landing buffer.** With `nocs=2` the NCRISC and BRISC readers both used
  `c_0`, overwriting each other's slots. The reads all still happened, so it reported **434.7 GB/s —
  99.7% of 436, a headline-shaped number** — with corrupt data. Fixed by giving each NoC its own CB;
  the honest figure is 445.7.

> **A bandwidth kernel that reads wrong, reads short, or whose barrier does not wait always reports a
> HIGHER number.** An ungated sweep rewards its own bugs. Two words XOR-folded per packet cost nothing
> and caught both.

A third bug — bank-pinned modes with a core count not divisible by 8 — addressed past the end of the
buffer and segfaulted the host. Now rejected explicitly rather than read out of bounds.

## 5. What this changes

* **436 stands as the denominator.** 445.7 is 2.2% above it and 2.9% above the guide's achieved 433,
  from a configuration that differs in core count — not a resolved improvement. The guide's recipe had
  already given the writer's RISC read work; this is the same idea on a pure-read workload.
* **The one-NoC wall is ~238 GB/s and is not tunable.** For any kernel whose reads all sit in the
  reader, that is the number to compare against — not 436 and not 445.7.
* **`exp_roofline001`'s memory roof should be re-derived.** It measured 334.6 GB/s with two reads on
  NOC_1 and one write on NOC_0. With reads split across both NoCs the roof should approach 445.7,
  taking the `R=1` points from 27.9 to ~37 Gop/s. **[X — not re-run.]**

**Not measured:** mixed read/write at this bandwidth (splitting reads across both NoCs means reads and
writes no longer have a NoC each, which is the trade the proposal flagged); packet sizes above 16 KB
(that is the burst limit); more than 130 cores (there are no more).

## 6. Running it

```
. ./env.sh
./build/exp_dram001                     # the sweeps
DR1_PERBANK=1 ./build/exp_dram001       # + per-bank verification
```

| env | meaning |
|---|---|
| `DR1_MB` | source size, MB (default 512; 256 used here) |
| `DR1_PERBANK` | add the per-bank check, 4 adjacent readers on one bank at a time |
| `DR1_ITERS` | traced replays per point (default 5); `min` quoted |
| `DR1_CSV` | write every point |

Bank-pinned modes require a core count divisible by 8; the bench rejects anything else rather than
addressing past the end of the buffer.
