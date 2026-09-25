# DRAM and NoC on Blackhole p150 — a measured guide for kernel optimisation

**Every number here was measured on this card**, by `host/experiments/exp_tg018dram.cpp`, with each
configuration verified against a host golden and each device timing taken from the profiler's
per-RISC `*-KERNEL` cycle counters. The chronological record — including the wrong turns and what
killed them — is `docs/he/history/exp_tg018dram.md`. This file is the distilled version: what the machine
does, how to get it, and what not to bother trying.

---

## 0. Check this first: the single-page buffer trap

**tt-metal maps interleaved page *p* to DRAM bank `p % NUM_DRAM_BANKS`.** A buffer holding **one**
page holds it on **page 0**, and page 0 is **always bank 0**.

So a data structure allocated as *N separate one-page buffers* — which is the natural way to write it
when each tile is conceptually its own object — puts **every one of those tiles on one of the eight
DRAM channels**. Not spread badly: *not spread at all*.

```cpp
for (i) b[i] = mkbuf(1, I32);      // WRONG: N buffers x 1 page -> every tile on bank 0
auto b     = mkbuf(N,  I32);       // RIGHT: 1 buffer x N pages -> tile t on bank t % 8
```

This was live in `exp_Tg018` and cost **1.79× on the whole CKKS multiply**. Fixing it alone —
no change to the scheme, the arithmetic, or any compute kernel — took 238.7 ms to 133.6 ms at
bit-identical output. The kernels with almost no compute per tile went **3.7–4.1×**
(`docs/he/exp_tg018m.md`).

**How to recognise it in a profile.** Compute a kernel's achieved GB/s and compare it against the
*single-channel* figure in §1 (53.9 GB/s), not the chip figure. exp_Tg018's corner turn moved 4 MB in
170.4 µs = **24.6 GB/s** — about half of one channel. Anything hovering at or below ~54 GB/s
aggregate, regardless of core count, is a candidate.

The fix costs a `(base address, page)` pair where a per-tile address used to go. Kernels that already
take a `start_tile` (`reader_ab`, `writer_i32`, `writer_modmul`) need **no change at all** — the host
just stops passing 0.

---

## 1. The ladder

Six independent limits, each measured in isolation. A kernel is bounded by whichever it hits first,
and they are *not* in the order most people assume.

| level | measured | nominal | how it was measured |
|---|---|---|---|
| one core, **one** NoC port | **81.5 GB/s** | 86.4 (64 B/cy × 1.35 GHz) | G=1, 384 pages in flight |
| one core, **both** ports | **143.2 GB/s** | 172.8 | G=1, reads on NCRISC *and* BRISC |
| **one grid row**, one direction | **89.0 GB/s** | 86.4 | 8 cores packed into row 0 |
| **one DRAM channel** | **53.9 GB/s** | 65 | 64 cores all pinned to one channel |
| all 8 channels, summed | **436.0 GB/s** | 520 | each channel saturated separately |
| **whole chip, achieved** | **433 GB/s** | — | the recipe in §3 — **99.3% of 436** |

Three consequences that are not obvious:

* **A core's inbound port and a grid row are the same 64 B/cycle link.** One core saturates its own
  row. Eight cores in a row reach 89.0 GB/s; *one* core in that row reaches 81.5. The other seven buy
  **10%**. Packing cores along a row is not inefficient for DRAM traffic — it is nearly pointless.
* **One core cannot saturate one channel** (42.6 of 53.9). It takes about four.
* **520 GB/s is a pin rate, not a target.** A channel delivers ~54 for a scattered 4 KB-page read
  stream. **The right denominator for any bandwidth claim on this part is 436 GB/s.** Measured 433 is
  the machine, not 83% of it.
  > **Corroborated, and 436 confirmed as the right denominator.** [`exp_dram001.md`](exp_dram001.md)
  > measures **445.7 GB/s** of pure reads with both NoCs reading — the same technique §3.1 already
  > uses for its 433 — so 2–3% apart at a different core count, i.e. the same number. Its real content
  > is that the **one-NoC wall (~238 GB/s, matching §3.1's 234.8) is NOT tunable**: 16 KB bursts,
  > transaction-id in-flight depth, bank affinity and core count all leave it between 235 and 241.

All 8 channels are present and within 2.8% of each other (53.8 – 55.3 GB/s), so nothing is harvested
here and the "7 active channel" configuration does not apply to this card.

---

## 2. The cost model

$$\text{latency}(G,T) \;=\; \underbrace{M(G,s)}_{\text{transient}} \;+\; T\cdot \underbrace{C(G,s)}_{\text{per tile}}
\qquad
C = \max\Big(C_{\text{compute}},\ \frac{G\cdot s}{B}\Big)$$

with `G` = cores, `T` = tiles per core, `s` = bytes per tile (read **and** written), `B` = the
binding bandwidth from §1 — **which is the row's 86.4 GB/s unless the cores are spread**.

Accurate to **1.5% median / 2.7% mean** for T ≥ 8 across an 8× range of memory pressure. Below T ≈ 8
the transient has not finished accumulating and the model does not apply.

**`C` — the per-tile term.** Flat in G to 0.11–0.17% while compute-bound, which is what identifies it
as compute. The `max` switches at exactly the measured knee.

| kernel | `C_compute` | `s` | knee `G* = B·C/s` |
|---|---|---|---|
| `modmul_sfpi_full` | 12.44 µs | 12 KB | ~250 (never, on this card) |
| `fused_matmul_a` `w_share=0` | 7.97 µs | 24 KB | ~83 |
| `fused_matmul_a` `w_share=1` | 7.97 µs | 8 KB | ~243 |

**`M` — the transient.** Flat in T from T ≈ 8, linear in G above G ≈ 4, slope 0.17–0.24 µs **per
core** ordered by traffic per tile — so it is the DRAM path, not the launch path. It is the
pipeline fill: at kernel entry all G cores issue their first reads simultaneously, then drift apart
as each is gated by its own compute.

| kernel | M(1) | M(8) | M(32) | M(64) | fit, G ≥ 4 |
|---|---|---|---|---|---|
| `modmul` | 1.11 | 2.32 | 6.45 | 13.25 | `0.56 + 0.196·G` |
| fused `ws=0` | 1.49 | 2.99 | 7.09 | 16.58 | `0.50 + 0.242·G` |

Per tile this reads `kern/T = C + M(G)/T`, so **everything more work-per-core buys is the second term
shrinking**. The tree runs at T=1, which is entirely inside the transient — a production dispatch's
cost is dominated by `M`, the only G-scaling term.

---

## 3. The recipe — how to actually get the bandwidth

Measured end state: **433 GB/s**, against **245** for the same kernel done the obvious way.

### 3.1 Drive both NoCs — worth **1.79×**

The single largest factor found anywhere. A reader on `RISCV_1` uses **NOC_1**; a writer on
`RISCV_0` uses **NOC_0**. A kernel whose reads all live in the reader leaves *half the fabric idle*
no matter how well its cores are placed. Measured at G=16, diagonal placement, reader E=192:

| writer read traffic | GB/s |
|---|---|
| none (NOC_1 only) | 234.8 |
| EW=96 | 333.3 |
| EW=192 | **433.0** |

The 245 GB/s plateau that survived every placement, depth and core count was almost exactly *half*
of what the path delivers.

### 3.2 Spread over rows — worth **2.42×** at G=8

Same 8 cores, only the shape changes (E=64, T=32):

| block | rows spanned | GB/s | µs/tile |
|---|---|---|---|
| 8 × 1 | 1 | 89.0 | 25.41 |
| 4 × 2 | 2 | 139.8 | 16.17 |
| 2 × 4 | 4 | 182.5 | 12.39 |
| **1 × 8** | 8 | **213.4** | **10.60** |

**One reader per grid row.** Diagonal placement (distinct row *and* column) beats column-only at
G ≤ 2 and is identical past G ≈ 4 — columns are not a bottleneck, rows are.

> **This rule inverts when the work is latency-bound.** It was measured at 32–64 tiles per core,
> where row capacity is what binds. Applied to `exp_Tg018`, which dispatches **one tile per core** and
> therefore lives entirely inside the transient `M(G)` of §2, spreading is a **2.5× regression**:
> 133.6 ms row-major against 333.5 ms diagonal. With nothing to saturate, spreading only buys NoC
> distance — and since a dispatch's span is a *max over cores*, the worst-placed core sets it.
>
> **Spread for bandwidth; pack for latency.** The test is whether the stage is past its knee
> (§2): if `G·s/B < C_compute`, do not spread.

### 3.3 Keep ~192 pages in flight per core per NoC

A single core's rate against pages outstanding: 42.6 GB/s at 96, 66.3 at 128, 78.3 at 192, 81.5 at
384. Past ~192 there is little left; below it the round trip is exposed.

### 3.4 Interleaved buffer, consecutive page ids

tt-metal maps page *p* to bank `p % NUM_DRAM_BANKS`, so consecutive ids round-robin the channels.
Never hard-code a channel address. **Do not pin cores to channels** — at G=16 pinning gives 189 GB/s
against 235 for ordinary interleaving, because it concentrates each core's traffic on one endpoint
instead of spreading it over eight. Pinning is a *measurement* device (§1), not an optimisation.

### 3.5 Stop at ~16 cores for the DRAM-bound part

| G | 1 | 2 | 4 | 8 | **16** | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| GB/s | 143.2 | 264.8 | 328.0 | 399.8 | **429.2** | 423.2 | 417.6 | 396.4 |

More cores make it **worse**, for two independent reasons. DRAM is already saturated at 16; and a
wide dispatch develops a badly balanced tail — identical work, but per-core service rates differ by
**13.2×** at G=128 (fastest core finishes its 48.6 MB in 1.25 ms, slowest takes 16.5 ms). The machine
is fully loaded only at the start and idles while the stragglers drain. Cores past 16 contribute
compute, not bandwidth.

### 3.6 Constants: do not stream what you can materialise

Three questions, in order of what they are worth:

**Core count matters through the work it lets a dispatch carry, not by itself.** exp_Tg018m derives
`limbs per dispatch = G / tiles-per-limb`, so G=64 and G=96 both give 1 while G=128 gives 2 — and the
measurement follows the *step*, not the core count: 120.6 ms, 120.8 ms, **84.1 ms**. Adding cores
without crossing a threshold that changes what a dispatch contains only makes the chunking ragged.

**Batch across whatever axis has the most independent work, not the most obvious one.** A stage
called once per limb on a 64-tile polynomial dispatches 64 jobs however large G is. The instinct is
to run several *operations* concurrently to fill the grid; the cheaper answer is usually that the
operation already contains far more independent work than that. In exp_Tg018m, running two ct_muls
side by side would have offered 2 units of extra work where batching the multiply's own 46–48 limbs
offered 24× more — and since cores are stateless and a dispatch just takes `min(G, jobs)` from a flat
list, the hardware cannot tell the difference.

**Check the grid is full before optimising anything else.** A stage whose work list is shorter than
G silently runs on part of the grid, and no memory optimisation will show up through that. In
exp_Tg018m the FBC chunked a 64-long tile axis by G and so used 64 of 130 cores through 29% of the
multiply; flattening its two independent loops into one job list was **1.17× on the whole workload**,
more than any single memory change after the bank fix. The tell is a stage whose dispatch count does
not fall when G rises.

**Occupancy reasoning finds bugs; it does not pick constants.** "How many batched items cover G for
the tightest stage" is the right question to ask, and asking it found two stages running on half the
grid. But the answer it gives is not the right *setting*, because the same constant also decides how
every other stage chunks, and the tail matters: on exp_Tg018m the formula's LPD=5 measured **78.1 ms
against 69.4 at LPD=4**, because 5 limbs give the matmul a 130+130+60 split. Use the reasoning to
find idle cores; use measurement to choose the number.

**Occupancy is the other half of every one of these trades.** Amortising work over more tiles per
core — whether that is `w_share`, `T > 1`, or reusing one read for several outputs — shrinks the
number of cores a given amount of work can fill. Measured on the corner turn: giving each core 2
outputs instead of 1 gains 1.5%, and giving it 4 **loses 5%**, because at that point the batch no
longer has enough independent transforms to keep 64 cores busy. Choose `(G, T)` from how much
independent work the *stage* has, not from the grid size.

**Is it a scalar?** Then it must not be a buffer. A compute kernel can put any 32-bit value into DST
for free with `fill_tile_int` — the modular kernels already do it for `q`, `q_inv_32` and the `q`
halves. Passing it as a runtime arg instead of reading a tile removes one of the operand streams
outright.

`exp_Tg018` held 160 scalars as 160 *64-tile DRAM buffers* (`fill64`) — **40 MB of DRAM to carry 160
distinct numbers**, and a 4 KB read per tile at every use. Measured after the fix
(`docs/he/exp_tg018m.md`):

| kernel | two operands | scalar operand | |
|---|---|---|---|
| `modadd` | 4.4 µs | **3.2 µs** | −27% |
| `modmul` | 17.0 µs | **15.0 µs** | −12% |

The relative saving tracks how compute-bound the kernel is: `modadd` has almost no compute so losing
a read is most of its time, while `modmul` sits on a 12.44 µs floor and a read is worth ~2 µs.
End to end it was only **0.3%**, because those ops are 9% of the dispatches — the win here is 40 MB
of DRAM and 160 pointless uploads, not wall time. Do it because it is free and correct, not because
it is fast.

**Is it shared across a dispatch, and re-read every time?** Then it belongs in L1. `BufferType::L1`
buffers persist across programs, so a constant can be uploaded once at setup — and the change is the
*allocation only*: `TensorAccessor` takes the buffer type from its compile args, so readers follow it
with no edit. Interleaving is usually good enough on its own: the 16 tiles of one W set land on 16
different L1 banks, so 64 readers spread over 16 sources rather than piling on one.

Measured on `exp_Tg018m`: all 128 NTT matrix sets are 12.0 MB, which as an L1-interleaved buffer is
**95 KB per core, 6.6% of L1**. The matmul kernels went 17.2 → 16.3 µs — **5%**, against a traffic
reduction of 70%. See the caveat under the results table below: at one tile per core, latency is the
cost, not bytes.

The prize is large where the constant is bigger than the data. In `exp_Tg018` each matmul dispatch
re-reads a **16 KB** folded W per core to process a **4 KB** tile:

| | per multiply |
|---|---|
| W read by the three matmul passes | **2,432 MB** |
| polynomial data those passes move | 1,064 MB |
| **constant share of matmul DRAM traffic** | **70%** |

**Is the whole set small enough to be resident?** Count per core, not in total. One core needs 20 KB
for one matmul dispatch (16 KB W + 4 KB tile) against 1440 KB of L1 — trivially. A core that *owned*
a prime would need 224 KB for both directions. It fits; what does not fit is the schedule, which
rotates limbs across dispatches so no core keeps the same constants twice. **Residency is a
scheduling property, not a capacity one.**

---

## 4. Things that do NOT work — measured null results

Each of these looked reasonable and cost a run to disprove. Do not re-litigate them without new
evidence.

| idea | result |
|---|---|
| **Batching reads deeper behind one barrier** (`seq` D=1 → `batch` D=16-32) | **1.17–1.69× SLOWER**, consistently, on both kernels. With 8–12 µs of compute per tile a one-tile-at-a-time reader already overlaps perfectly; batching converts a smooth request stream into bursts that queue against each other. |
| **Distinct L1 landing slots** (1 → 32) for outstanding reads | **No effect** (0.99–1.02× over 80 configurations). The half of the rule that matters is distinct *DRAM pages*, which was already true. |
| **Consecutive page ids instead of a prime stride** | **No effect.** 613 is coprime to the channel count, so the stride already visited every channel — just not in index order. |
| **Pinning cores to DRAM channels** | **Worse** — 189 vs 235 GB/s at G=16. Concentrates traffic on one endpoint. **Refined by [`exp_dram001.md`](exp_dram001.md) §3:** placing the pinned readers *adjacent* to their bank recovers +19–23% (162→194 at 32 cores, 179→221 at 64) and beats interleaving there — but interleaving still wins on a full grid (239 vs 217), because it spreads over all eight channels for free. Pinning is still the wrong default; proximity was not the reason. |
| **More cores** past the DRAM knee | **Worse** — 433 → 396 from G=16 to G=128. |
| **Two cores sharing one channel** | **Worse than one** (36.7 vs 42.6 GB/s). One core walks pages sequentially and keeps the row buffer; two interleaved streams thrash it. |
| **Scalar constants as runtime args** (§3.6) | **Correct and free, but worth 0.3% end to end** — the affected ops are 9% of dispatches. Take it for the 40 MB of DRAM it returns, not for speed. |

---

## 5. Code

### 5.1 A reader that keeps N pages in flight

Depth `D` with one barrier per group. `D=1` is the production pattern (`reader_modmul.cpp`); the
group size must divide the CB depth so a group never straddles the wrap.

```cpp
// kernels/experiments/exp_tb001/dataflow/reader_ab_depth.cpp
for (uint32_t i = 0; i < n; i += D) {
    const uint32_t k = (n - i < D) ? (n - i) : D;
    cb_reserve_back(cb_a, k);
    cb_reserve_back(cb_b, k);
    const uint32_t pa = get_write_ptr(cb_a), pb = get_write_ptr(cb_b);
    for (uint32_t j = 0; j < k; ++j) {             // 2k reads in flight...
        const uint32_t id = start + i + j;
        noc_async_read_tile(id, a, pa + j * ts);
        noc_async_read_tile(id, b, pb + j * ts);
    }
    noc_async_read_barrier();                      // ...then ONE barrier
    cb_push_back(cb_a, k);
    cb_push_back(cb_b, k);
}
```

**The host must size the CB at a multiple of `D`** (it uses `2D`, for double buffering), so every
group starts on a `D`-aligned slot. `get_write_ptr()` returns a cursor and tile *j* is written at
`cursor + j*tile_size` with no wrap handling.

### 5.2 Using the second NoC

The kernel that only reads in its reader gets half the fabric. Give the writer's RISC read work of
its own — it is on the other NOC:

```cpp
// kernels/experiments/exp_tb001/dataflow/writer_i32_depth.cpp — BRISC, NOC_0
for (uint32_t j = 0; j < k; ++j) {
    noc_async_write_tile(start + i + j, s, base + j * ts);
    for (uint32_t e = 0; e < extra; ++e)                  // second-NoC read traffic
        noc_async_read_tile((start + i + j + 1u + e) % n_total,
                            src, junk2 + (e % JUNKP) * ts);
}
if (extra) noc_async_read_barrier();
noc_async_write_barrier();
```

In a real kernel this means **splitting the input stream between the two dataflow RISCs** rather than
inventing traffic: half the pages fetched by NCRISC, half by BRISC, each into its own CB region.

### 5.3 Placement

Row-major — the natural `{i % grid.x, i / grid.x}` — is the **worst case**: it packs the first 13
cores into one grid row, pinning them at 86.4 GB/s regardless of anything else.

```cpp
// host/experiments/exp_tg018dram.cpp
auto CC = [&](uint32_t i) -> CoreCoord {
    if (PL == "diag") return CoreCoord{i % GX, i % GY};   // distinct row AND column
    if (PL == "col")  return CoreCoord{i / GY, i % GY};   // distinct row
    return CoreCoord{i % BW, i / BW};                     // row-major: DO NOT for DRAM-bound work
};
```

`GX = 13` and `GY = 10` are coprime, so by CRT the 130 diagonal pairs are all distinct and the first
10 cores land on 10 distinct rows *and* 10 distinct columns. A `CoreRangeSet` is built one core at a
time, so **a rectangular core set is not required** — G=128 (nine full rows plus 11 of the tenth)
measures within 1–2% of G=130.

### 5.4 Buffer

```cpp
distributed::DeviceLocalBufferConfig d{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
auto buf = distributed::MeshBuffer::create(
    distributed::ReplicatedBufferConfig{.size = page_bytes * n_pages}, d, mesh.get());
```

Interleaved by default; page *p* lands on bank `p % 8`. Read consecutive ids and let the round-robin
do the work.

---

## 6. What this says about the HE kernels

Applied to `exp_Tg018`'s CKKS `ct_mul` (236.6 ms, of which these two kernels are 69%):

* **`modmul_sfpi_full` (48.4 ms) is compute-bound at every core count** — 12.44 µs of SFPU work per
  12 KB tile. At G=130 it delivers 123.5 GB/s and its compute asks for 125. **No memory optimisation
  can help it.** This is why the batched-feed change was a measured null result, and it closes that
  line. The only lever is the multiply itself.
* **The fused matmul family (43.8 ms) is memory-bound above G ≈ 83** at `w_share=0`. `w_share=1` —
  reading the 16 KB folded W once per core instead of once per work tile — changes `s` from 24 KB to
  8 KB, moves the knee to G ≈ 243, and is worth **1.80× at G=130**. It needs no new arithmetic, only
  more than one tile per dispatch.
* **The tree runs at T=1 and row-major**, which is the worst corner of everything above: every
  dispatch lives entirely inside the transient `M`, and every G ≤ 13 configuration is pinned at
  86.4 GB/s by the row. More tiles per core is worth **−22%** on `modmul` at G=64 (16.16 → 12.64
  µs/tile) before any of the bandwidth work.

### What actually happened when this was applied

`exp_Tg018m` (`docs/he/exp_tg018m.md`) applied it and the ranking above turned out to be wrong for
this workload — usefully so:

| change | predicted | measured on the multiply |
|---|---|---|
| **bank spreading** (§0) | not on the list | **1.79×** on the whole multiply |
| second NoC (§3.1) on the corner turn | 1.8× | **1.37×** on that kernel, **1.07×** overall |
| scalar constants (§3.6) | — | 1.27× / 1.12× per kernel, **1.003×** overall |
| NTT matrices into L1 (§3.6) | — | 1.05–1.06× per kernel, **1.016×** overall |
| more outputs per core (corner turn) | — | **1.015×** at K=2; K=4 **regresses** |
| G = 64 → 128 (limbs per dispatch 1 → 2) | — | **1.43×** — G=96 buys *nothing* |
| LPD 2 → 4 (more limbs batched) | — | 1.04×, plateaus at 8 |
| flatten the FBC job list to fill the grid | — | **1.17×** — it ran 64 cores of 130 |
| batch the per-limb elementwise ops | — | 1.04× |
| batch the tensor and combine stages | — | 1.03× — 336 more dispatches at 64 of 130 |
| row spreading (§3.2) | 2.4× | **0.40×** — a regression, see the caveat there |
| more tiles per core | −22% | not applied yet |
| **cumulative** | | **238.663 → 50.920 ms, 4.69×** (the last steps are arithmetic, not memory -- `history/exp_tg018s.md`, `exp_modmul001.md`) |

Two of those four returned far less than their traffic reduction implies, and for the same reason:
**at one tile per core a dispatch is inside the transient `M(G)`, where cost is un-hidden latency,
not bandwidth.** Removing bytes cannot pay until there is a second tile to amortise the first one's
round trip over. Traffic reductions are worth taking — they are free and they compound — but the
model in §2 says which ones will show up on the clock, and it was right both times.

The lesson for the next kernel: **§0 before anything else**, then establish whether the stage is
bandwidth- or latency-bound (§2) *before* choosing a placement, because the two want opposite things.

Ordering, by measured value: **§0 bank spreading first**, then the second NoC, then more tiles per
core, then `w_share`, then placement — *only* for stages past their knee — and **not** deeper
request batching.

---

## 7. Reproducing

```
. ./env.sh
TT_METAL_DEVICE_PROFILER=1 ./build/exp_tg018dram          # default sweep
```

| env | meaning |
|---|---|
| `TGD_G`, `TGD_T` | cores, tiles per core |
| `TGD_KERNEL` | `modmul` \| `fused` \| `both` |
| `TGD_MODE` | `seq` (D=1, production) \| `batch` \| `both` |
| `TGD_WSHARE` | fused matmul W policy |
| `TGD_EXTRA`, `TGD_EXTRA_W` | extra pages per tile on NOC_1 / NOC_0 — the arithmetic-intensity axis |
| `TGD_PLACE` | `row` (default) \| `col` \| `diag` |
| `TGD_BLOCKW` | lay G cores out as a W-wide block |
| `TGD_NBANK`, `TGD_CHANMOD`, `TGD_CHANOFF` | channel pinning, for characterisation only |
| `TGD_ITERS`, `TGD_CSV`, `TGD_NOVERIFY` | repetitions, output, skip the golden check |

Device timings come from tt-metal's own `*-KERNEL` zones (no zone is added to any kernel), parsed
after `mesh->close()` — which is when tt-metal writes the log — with the group↔configuration join
verified 1-1 or the device columns dropped. Timebase is 1350 MHz, the `CHIP_FREQ` the profiler itself
declares; every ratio here is cycle arithmetic and independent of it, while absolute GB/s scales with
it.

Run one kernel per invocation: a single dropped marker invalidates the device columns for the whole
run, so a shorter run is a cheaper failure.
