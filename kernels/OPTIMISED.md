# The optimised kernel set

These are the kernels that carry the measured wins from `docs/he/exp_modmul001.md`,
`docs/he/exp_tg018m.md`, `docs/he/exp_tg018r.md` and `docs/he/dram_guide.md`, promoted out of the
experiment trees so anything in the repo can use them.

**Nothing was replaced.** Every kernel these derive from is still present and unchanged, so no
existing caller changes behaviour. Pick the optimised name when you want the optimised behaviour.

| use this | instead of | what it buys |
|---|---|---|
| `compute/modmul_sfpi_1pass.cpp` | `compute/modmul_sfpi_full.cpp` | **4.03×** — 12.430 → 3.086 µs/tile, canonical, bit-exact |
| `compute/fused_matmul_a_1pass.cpp` | `compute/fused_matmul_a.cpp` | NTT matmul block 17.501 → 12.943 ms |
| `compute/fused_matmul_a_acc_1pass.cpp` | `compute/fused_matmul_a_acc.cpp` *(exp tree)* | as above, accumulating variant |
| `compute/fused_matmul_c_1pass.cpp` | `compute/fused_matmul_c.cpp` | as above, stage C |
| `compute/baseconv_fbc_wide.cpp` | `compute/baseconv_fbc.cpp` | lifts the `Ld ≤ 8` cap |
| `compute/fbc_scale_wide.cpp` | — | FBC stage 1 with the same cap lifted |
| `compute/baseconv_fbc_matmul_tr.cpp` + `dataflow/{reader,writer}_fbc_matmul_tr.cpp` | `compute/baseconv_fbc.cpp` | **1.19× at 8→8, 10.3× at 32→32** |
| `dataflow/{reader,writer}_seq_batch.cpp` | `dataflow/{reader_i32,writer_i32}.cpp` | the DRAM depth rule; up to ~5× on feed-bound kernels |
| `dataflow/reader_matmul_mtk_deep.cpp` | `dataflow/reader_matmul_mtk.cpp` | the depth rule on the K-looped matmul feed: **1.42×** on the NTT-32768 2-factor matmul stages (115.44 → 81.53 µs) |
| `dataflow/reader_matmul_mt_deep.cpp` | `dataflow/reader_matmul_mt.cpp` | same, K=1 variant: **1.53×** on the 3-factor matmul stages (40.06 → 26.18 µs) |
| `dataflow/all_to_all_corner_{rd,wr}.cpp` | `dataflow/all_to_all_corner_mc.cpp` | corner-turn reads split over **both NoCs** instead of one RISC doing everything: **1.31×** (31.92 → 24.29 µs) |
| `dataflow/reader_fused_batch_g.cpp` | `dataflow/reader_fused_batch.cpp` | the same batched 17-read feed, plus a **tile index**, so a grid of cores can each drive a `fused_matmul_*_1pass` kernel on a different tile |
| `compute/fused_matmul_k_1pass.cpp` | `compute/matmul_compute_mtk.cpp` + `compute/recombine_bigK.cpp` | K-looped matmul with the bigK recombine fused in, planes in DST: **1.15–1.23×** on the NTT-32768 2-factor matmul stages |
| `dataflow/all_to_all_corner_x.cpp` | `dataflow/all_to_all_corner_{mc,rd,wr}.cpp` | corner turn as a **core-to-core NoC exchange** — each core reads only its own tile and pushes rows to peers, landing them in final face layout so no assembly remains: **5.40×** (23.97 → 4.44 µs). The repo's first `noc_semaphore_*` barrier; needs the full grid |
| `exp_tb001/dataflow/a2a_corner_x.cpp` | `exp_tb001/dataflow/a2a_corner_{rd,wr}.cpp` | corner turn as a core-to-core exchange in ct_mul: that stage **9.108 → 2.122 ms (4.29×)**, whole multiply **50.863 → 43.932 ms** |
| `exp_tb001/compute/ntt64_fused.cpp` + `dataflow/ntt64_fused_{rd,wr}.cpp` | the six separate NTT passes | the WHOLE ct_mul NTT in **one dispatch**, corner turn on-chip, no intermediate in DRAM: **50.863 → 32.448 ms (1.568×)**, 2884 → 1288 dispatches, host wall 155.1 → 41.3 ms |
| `ir/compute/matmul_modq_mbatch_1pass.cpp` | `ir/compute/matmul_modq_mbatch.cpp` | PIR scan **180 → 128 ms/query** (with the reader below) |
| `ir/dataflow/reader_mbatch_deep.cpp` | `ir/dataflow/reader_mbatch.cpp` | the depth rule on the 6.0 GB database feed |
| `ir/compute/add_modq_1pass.cpp` | `ir/compute/add_modq.cpp` | one pass, no `fill_tile_int` (no measured gain — see below) |
| `ir/dataflow/reader_binop_deep.cpp` | `ir/dataflow/reader_binop.cpp` | depth on the scan accumulator (no measured gain) |
| `ir/dataflow/writer_i32_deep.cpp` | `ir/dataflow/writer_i32.cpp` | write depth (no measured gain) |
| `pirclient::ntt_shoup` (`reference/pir/client/pir_client.hpp`) | `pirclient::ntt` | host NTT **2.04×** standalone, 0/2048 mismatches |

### Applied to the OnionPIR demo — `examples/pir_demo65536.cpp`

`docs/pir/` records the demo itself; this is only what the kernel set above bought it. One query over
65 536 rows on a p150, steady state, min of 6 runs each, `--batch 777`, answer byte-identical to
`db65536.txt` throughout and all 16 `--checks` isolation checks green:

| stage | before | after |
|---|--:|--:|
| expand | 93 | 83 |
| convert | 33 | 33 |
| **scan** | **180** | **128** |
| folds | 31 | 26 |
| fold RGSWs *(host)* | 68 | 67 |
| decrypt *(host)* | 24 | 24 |
| **wall / query** | **439 ms** | **372 ms** |

The scan is the term that matters: it streams the whole 6.0 GB folded database past the cores on
*every* query, so it is 41% of the baseline. Its 1.41× came almost entirely from the compute kernel
(166 → 130 ms single-run), not from the feed (180 → 166) — see the null results below.

**Null results, so nobody repeats them.** All measured on this demo, device stages reproducing to
±0 ms across runs:

| idea | result |
|---|---|
| **`MathFidelity::HiFi2`** on the scan matmul | **Wrong answer** (garbled row) *and* scan unchanged at 128 ms. The int8 matmul needs HiFi4 for exactness, and fidelity is not what the scan is paying for. LoFi was already refuted in `exp_tg018m` |
| **Read/write depth + one-pass add on the accumulator** (~4.6 GB/query) | 129 → 128 ms. Correct, kept, but the accumulate is not the constraint |
| **Chunked `cb_wait_front`** in the scan matmul (512 wait/pop pairs → 32) | No change. CB synchronisation is not the constraint either |
| **Shoup host NTT** for the fold RGSWs | 2.04× standalone (101 → 49 ms for 672 transforms, single-thread) but **not resolved end-to-end**: that stage runs 7 threads on a 4-core host and its run-to-run spread (68–111 ms) is wider than the ~14 ms the change should save |

Not attempted: **Metal Trace**. The engine implements it (`TR_PHASE`, `begin/end/replay_mesh_trace`)
and the demo already reserves a 64 MB trace region, but the demo's per-stage `distributed::Finish`
calls sit inside what would be an open capture, so enabling it needs the stage boundaries reworked
first. Host `ENQ` is ~64 ms of the remaining 372, which bounds what it could pay.

---

### Applied to NTT-32768 — `host/experiments/exp_ntt001.cpp`

The three dataflow entries above came out of `docs/matmul/exp_ntt001.md`, which measured both
NTT-32768 factorizations by trace-derived critical-path busy and found each of them bound by
communication rather than arithmetic. Both fixes are behind flags so the stock arm stays measurable
(`NTT1_DEEP=1`, `NTT1_A2A=bothnoc`); all rungs bit-exact `0/32768`:

| rung | dispatches | 3-factor | 2-factor |
|---|--:|--:|--:|
| stock dataflow | 12 / 7 | 121 µs | 146 µs |
| + deep matmul readers | 12 / 7 | 110.0 | **114.0** |
| + both-NoC corner turn | 12 / 7 | 101.6 | 114.0 *(no corner turn)* |
| + **fused dec+mm+rec triples** | 6 / 7 | 65.1 | 114.0 *(not applied)* |
| + **core-to-core corner turn** | 6 / 7 | 45.5 | 114.0 *(no corner turn)* |
| + fused matmul/recombine (2-factor) | 6 / 5 | 45.5 | 98.6 |
| + whole NTT in one kernel, no DRAM during it | 1 / 1 | 34.0 | 62.1 |
| + **staged reader (load overlaps compute)** | **1 / 1** | **29.6** | **56.1** |

**The fair end state: both fully SRAM-resident, one dispatch each, 29.6 vs 56.1 µs — 1.89×.** The
2-factor needs TWO on-chip exchanges to get there (an 8-core `mt` group for M1, a 4-core `nt` group
for M2) where the 3-factor needs one, because both of its matmuls are non-tile-local.

**The biggest rung reused kernels already in this file.** `fused_matmul_{a,c}_1pass.cpp` were listed
here for `exp_Tg018m` and had never been wired into an NTT; doing so gave **2.14–2.27× per triple**
(21–23 µs → ~10 µs). It works because the fused kernel removes three things at once — the two DRAM
round trips, the plane round trip through L1 (planes stay in DST), and the multi-pass recombine.

**Two rules of this file were confirmed and one was bounded.** The depth rule held (matmul stages
1.42–1.53×, and *only* those stages moved — every other stage within ±0.13 µs). The both-NoC rule held
but at **1.31×, not §3.1's 1.79×**: the corner turn has a serial L1-assembly half that splitting the
reads does not touch, and after the fix ~68% of what remains is that assembly. **A rule measured on a
pure streaming kernel does not transfer at full value to a kernel with a serial half.**

**The corner turn went 32.24 → 23.97 → 4.44 µs (7.26×) across three implementations** — both NoCs,
then eliminating the DRAM round trip entirely. It is now 9.8% of the 3-factor path, down from 26.2%.

Still untaken: the matmul readers drive **one NoC** — all reads on NOC_1, writers issue none — so at
5.1–5.5 GB/s/core they remain at ~38% of the 436 GB/s ceiling. That is the largest remaining item and
it would favour the 2-factor, which has 2.9× the matmul traffic.

**A caution carried out of this work.** Three times a cost model built from stage timings was used to
predict what a rewrite would buy, and three times it was wrong — twice far too pessimistic (the fusion,
the exchange), once backwards (which half of the corner turn dominated). Such models are good for
*choosing what to try*; they are not ceilings, because they price the current shape rather than the
achievable one.

---

### Transferred back to ct_mul — `host/experiments/exp_tg018ntt.cpp`

`docs/he/exp_tg018ntt.md` audited exp_ntt001's five rungs against exp_Tg018m and found **four already
applied** — two of them (the fused triples, the batched matmul feed) having been exported *from* that
pipeline in the first place. Only the **core-to-core corner turn** was left, and it happened to aim at
exp_Tg018m's largest non-arithmetic term. Ported as `exp_tb001/dataflow/a2a_corner_x.cpp`;
`a2a_corner_{rd,wr}.cpp` retained. Both arms `0/6029312`, 2884 dispatches each:

| | `bothnoc` (= exp_Tg018m) | `exchange` | |
|---|--:|--:|--:|
| `a2a_corner` stage | 9.108 ms | **2.122 ms** | **4.29×** |
| critical path busy | 49.448 ms | **42.460 ms** | 1.165× |
| host device busy | 50.863 ms | **43.932 ms** | 1.158× |

Every other kernel moved by ≤0.03 ms, so the census attributes **6.986 of the 6.988 ms** to the one
stage. Two porting differences mattered: exp_Tg018 pages tiles as `n1*2+kh` (not `n1`), and it gives
each core `K=2` tiles rather than one, so the kernel was generalised to `NPC = NT/P` tiles per core —
which is what kept the dispatch count identical and made this a like-for-like swap.

**The best evidence is a knob that stopped mattering.** `K` exists in the DRAM arm only to amortise the
32 redundant reads every core makes. In the exchange there are none left to amortise, and `K`'s
influence collapses **18.0% → 0.4%** — confirming the mechanism rather than just the timing.

**Two lessons for reusing a ladder.** First, *audit before applying*: four of five rungs here were
no-ops, and applying them blind would have banked them as part of the win. Second, **a ladder's total
is a property of the pipeline it was measured on** — 3.56× on NTT-32768 became 1.158× here, and that is
not a failure to reproduce, it is the four rungs already being in place.

## The three techniques, and when each applies

Every compute win above is one or more of these. They are worth knowing separately because they
apply to any SFPU kernel in this repo, not just the ones already converted.

### 1. `sfpmul24` instead of the fp32 byte-split

`modmul_sfpi_full`'s header says *"Wormhole SFPI has no integer multiply (and no sfpmul24), so each
product is the fp32 byte-split trick"* — 4 fp32 multiplies plus 8 int↔float conversions per 16×16
product. **Blackhole has the instruction**, exposed as `sfpi::fractional_mul(a, b, Low|High)` behind
`#if __riscv_xtttensixbh`:

```cpp
vUInt lo = vUInt(fractional_mul(x, y, FractionalHalf::Low));    // product bits [22:0]
vUInt hi = vUInt(fractional_mul(x, y, FractionalHalf::High));   // product bits [45:23]
return lo + (hi << 23u);                                        // exact for x,y < 2^23
```

**Measured semantics** (`docs/he/exp_tg018s.md`, 532 probed operand pairs): an exact **23×23 → 46-bit**
product, and **both operands are silently masked to 23 bits** — `a = 2²³, b = 1` returns `0`. Not 24
bits despite the name. Feeding it 32-bit operands gives a wrong answer with no indication.

`mul_int_tile` wraps the same instruction but its LLK only ever issues `MOD1_LOWER`, so the upper half
— the thing that makes a wide multiply cheap — is unreachable through the tile-op API.

### 2. Collapse SFPI passes; keep intermediates in registers

Each `_llk_math_eltwise_*_sfpu_params_` call walks the tile and pays **SFPLOAD/SFPSTORE against DST
per face-row**. A pass that computes one product costs 2 loads + 1 store to do ~4 instructions of
work, and the result is reloaded immediately by the next pass.

`modmul_sfpi_full` ran **14 passes / 53 DST accesses**; `modmul_sfpi_1pass` runs **1 / 3**.

**This only became possible because of technique 1.** `modmul_sfpi_full`'s own comment states the old
constraint — *"One product per pass keeps SFPI register pressure low enough to compile"* — because the
fp32 byte-split needs 8 registers per product. `fractional_mul` needs 2. The control kernel
`kernels/experiments/exp_modmul001/compute/modmul_fused3_fp32.cpp` is the same fusion with the fp32 multiply put
back: it **does not compile**, `internal compiler error: cannot store sfpu register (register spill)`.

### 3. Broadcast scalars in register, not through `fill_tile_int`

`fill_tile_int` writes a **whole DST tile** to broadcast one scalar every lane needs identically — 21%
of a modmul's DST traffic. SFPI can materialise a runtime `uint32_t` directly: pass it through the
variadic tail of `_llk_math_eltwise_binary_sfpu_params_` and write `vUInt(QINV & 0xFFFFu)`.

Worth **2.9%** on the arithmetic-bound fp32 kernel and materially more once the arithmetic is cheap —
*an optimisation's value is set by the bottleneck, not by what it removes.*

### Bonus, for matmul kernels: `matmul_tiles`' last argument is a DST index

It is easy to leave at `0` and pack each result out separately. Accumulating result `p` into DST tile
`p` inside one `tile_regs_acquire` lets a following SFPU pass read them all in place — worth 4
`pack_tile` + 4 `copy_tile` + 3 acquire cycles per dispatch in the NTT matmuls.

---

## The DRAM rules

Full derivation and the null results in `docs/he/dram_guide.md`. The two that are baked into
`reader_seq_batch` / `writer_seq_batch`:

* **Depth.** Issue every read of a batch before the barrier. One outstanding read is the worst case;
  read depth is the difference between ~50 and ~250 GB/s.
* **Consecutive page ids.** TT-Metal maps `bank = page_id % NUM_DRAM_BANKS`, so a buffer laid out
  `page = tile * L + limb` puts the `L` pages one core touches on consecutive banks. The obvious
  layout `page = limb * NTILE + tile` puts them all on **one** bank whenever `NTILE` is a multiple of
  8 — and a single-page buffer is always bank 0, which is what cost exp_Tg018 a 1.79×.

Also worth knowing, from the same guide: drive **both NoCs** (1.79×), spread over grid **rows**
(2.42×), and stop at ~16 cores per stream — past that, more cores is worse.

---

## What is NOT converted

Honest list, so nobody assumes coverage:

* `compute/modmul_scalar.cpp` and `compute/tensor2_fused.cpp` still carry the fp32 byte-split. The
  same three techniques should apply; **not measured**, so no number is quoted.
* In the PIR tree, `ir/compute/matmul_modq_mbatch_{c,k2,c_k2}.cpp` and `ir/compute/matmul_modq.cpp`
  are the same shape as the converted `matmul_modq_mbatch` and are **not** converted — only the one
  on the scan's hot path was. `ir/dataflow/reader_mbatch_c*.cpp` likewise keep depth 1.
* `compute/modmul_sfpi_half.cpp` — half-sync variant, also unconverted.
* The `_1pass` matmuls are promoted copies of the exp-tree kernels, which are what
  `host/experiments/exp_tg018m.cpp` actually runs. The shared-tree `fused_matmul_a2/c2` variants are untouched.
