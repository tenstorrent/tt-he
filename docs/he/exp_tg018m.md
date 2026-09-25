# exp_Tg018m — exp_Tg018 with the DRAM findings applied

**Status: DONE.** `238.663 ms → 50.920 ms` host device busy, **4.69×**, at bit-identical output
(`0/6029312` against the host reference, 39.0 bits of slot precision — the same numbers exp_Tg018
produces). One change to how buffers are allocated; no change to the scheme, the parameters, the
arithmetic, or any compute kernel.

`host/experiments/exp_tg018m.cpp` · baseline `docs/he/exp_tg018.md` · method `docs/he/dram_guide.md`

> **SUPERSEDED AS THE BEST FIGURE.** 50.920 ms is still the correct number for *this* bench and the
> census below is still the reference breakdown, but [`exp_tg018ntt.md`](exp_tg018ntt.md) has since
> taken the same multiply to **23.991 ms (2.123× on top of this)** by making the corner turn a
> core-to-core exchange, fusing the whole NTT into ONE dispatch, and moving the base conversion onto
> the matrix engine. Its control arm reproduces 50.863 ms here, so the two are directly comparable.
> Quote **23.991** for "the current ct_mul latency".

## The bug

`exp_Tg018` allocates every polynomial tile as **its own single-page buffer** — `mkbuf(1, I32)`, 64
of them per 64-tile polynomial. tt-metal maps interleaved page *p* to DRAM bank `p % NUM_DRAM_BANKS`
(`dataflow_api_addrgen.h`: `bank_index = id - (id/NUM_DRAM_BANKS)*NUM_DRAM_BANKS`). A one-page
buffer's only page is page 0, and **page 0 is always bank 0**.

So the entire working set of the multiply — every ciphertext tile, every twiddle set — sat on **one
of the card's eight DRAM channels.**

It was visible in the profile the whole time without being recognisable. The corner turn is 43.8% of
the multiply at 170.4 µs per dispatch, moving 4 MB: **24.6 GB/s**, against a measured 53.9 GB/s for
one saturated channel and 433 GB/s for the chip. That is not a corner-turn problem. It is one
channel, running at about half of what one channel can do.

## The fix

A 64-tile polynomial becomes **one buffer of 64 pages** instead of 64 buffers of one page, so tile
*t* lands on bank `t % 8` and a 64-tile read set is spread over all eight channels. The same for the
per-tile twiddle sets (`tb`: 64 pages; `ta`: 2 pages).

Mechanically that means every dispatch site passes `(base address, page)` where it used to pass a
per-tile address. Most kernels needed no change at all — `reader_ab`, `writer_i32`, `writer_modmul`
already take a `start_tile`, so the host just stops passing 0. Five needed a page argument and got
`_pg` variants, leaving the originals untouched for exp_Tg018:

| new kernel | change |
|---|---|
| `a2a_corner_pg.cpp` | one in/out buffer + `kh`; computes page `n1*2+kh` itself, which also drops the argument count from `4+NT+n_out` to 7 |
| `reader_fused_batch_pg.cpp` | `x_page` argument |
| `reader_fused_acc_pg.cpp` | `x_page`; partial *a* is page `x_page + a` (the two P1 partials are consecutive tiles) |
| `reader_gather_batch_pg.cpp` | one `page` for all `Ls` sources — they are tile *t* of different polynomials |
| `writer_scatter_batch_pg.cpp` | same, for the `Ld` destinations |

One host-side subtlety: the twiddle operand does **not** share a page with the data. `x` and the
output are page *t*, while the twiddle is page `t&1` of `ta` or page *t* of `tb`. `reader_modmul`
uses one tile id for both operands, so every elementwise op moved to `reader_ab`, which takes
independent `a`/`b` starts and is otherwise identical.

## Result

Host device busy, G=64, traced, 3 iterations — both measured in the same session:

| | exp_Tg018 | exp_Tg018m | |
|---|---|---|---|
| host device busy | 238.663 ms | **133.620 ms** | **1.79×** |
| host wall | 240.120 ms | — | |
| `[D6]` vs host reference | 0/6029312 | 0/6029312 | identical |
| `[D7]` slot precision | 39.0 bits | 39.0 bits | identical |

Per kernel (census; exp_Tg018m estimate 125.60 ms against its 133.6 ms measured, the expected
ordering):

| kernel | exp_Tg018 µs/call | exp_Tg018m µs/call | speedup | total ms |
|---|---|---|---|---|
| **`a2a_corner`** | 170.4 | **41.5** | **4.11×** | 103.62 → 25.20 |
| `modadd_map` | 16.5 | **4.4** | **3.74×** | 9.75 → 2.61 |
| `modsub_reduce` | 16.5 | **4.3** | **3.87×** | 4.64 → 1.20 |
| `bconv_one` | 212.0 | 98.3 | 2.16× | 0.85 → 0.39 |
| `modmul_sfpi_full` | 23.4 | 16.9 | 1.38× | 48.43 → 35.07 |
| `fused_matmul_a` | 20.5 | 17.2 | 1.19× | 12.47 → 10.47 |
| `fused_matmul_c` | 20.2 | 17.2 | 1.18× | 12.29 → 10.45 |
| `fused_matmul_a_acc` | 31.2 | 27.8 | 1.13× | 19.00 → 16.87 |
| `baseconv_fbc` | 852.4 | 777.4 | 1.10× | 25.57 → 23.32 |
| **total** | | | **1.88×** | 236.62 → 125.60 |

With the scalar-operand change below, the census total is **125.00 ms** and the measured host device
busy **133.324 ms**.

The gain lands exactly where the diagnosis said it would. **The three kernels with almost no compute
per tile — the corner turn and the two elementwise maps — go 3.7–4.1×**, because they were doing
nothing but waiting on one DRAM channel. The compute-bound kernels move much less: `modmul` has a
12.44 µs/tile compute floor (`dram_guide.md` §2) and goes 23.4 → 16.9, from 88% above that floor to
36% above it.

This also re-explains an earlier finding. `exp_tg018.md` identified `modadd_map` and `modsub_reduce`
as latency-bound and attributed it to outstanding-request depth. They were latency-bound — but the
cause was single-banking, not depth, which is why deepening the requests measured as a null result.

## The one guide rule that did NOT transfer

`dram_guide.md` §3.2 measures **2.42× from spreading cores across grid rows**, because a row carries
only 86.4 GB/s however many cores are in it. Applied here it is a large **regression**:

| placement at G=64 | host device busy |
|---|---|
| row-major (`{i%13, i/13}`) | **133.620 ms** |
| diagonal (`{i%13, i%10}`) | 333.465 ms — **2.5× worse** |

The rule was measured on a kernel carrying 32–64 tiles per core, deep in the bandwidth-bound regime,
where row capacity is the binding constraint. exp_Tg018 dispatches **one tile per core**, which puts
every dispatch inside the transient `M(G)` — latency-bound, not bandwidth-bound. Spreading then buys
nothing and costs NoC distance on every transfer, and because the dispatch span is a max over cores,
the worst-placed core sets it.

**Spread for bandwidth; pack for latency.** Row-major is the default here and `TG018M_DIAG=1` opts
into the other, so the comparison stays reproducible.

## Second change: scalar constants as runtime args

`exp_Tg018` built the second operand of every rescale and mod-down multiply with `fill64(v)` — a
64-tile, 256 KB DRAM buffer holding one repeated 32-bit number. 160 of them: `P^-1` per limb, and
`qL^-1` and `qHalf` per (dropped prime, limb). **40 MB of DRAM to carry 160 distinct numbers**, plus
a 4 KB read per tile at every use.

The compute kernels already materialise `q`, `q_inv_32` and the `q` halves in DST with
`fill_tile_int`; a scalar is no different. `modadd_scalar.cpp` and `modmul_scalar.cpp` take it as a
runtime arg, which removes `cb_in1` entirely — one operand stream instead of two — and the constants
become 160 `uint32_t`.

| kernel | before | after | per call |
|---|---|---|---|
| `modadd_map` → `modadd_scalar` | 4.4 µs | **3.2 µs** | −27% |
| `modmul_sfpi_full` → `modmul_scalar` | 17.0 µs | **15.0 µs** | −12% |

The relative saving tracks how compute-bound each kernel is: `modadd` has almost no compute, so
losing a read is most of its time; `modmul` sits on a 12.44 µs floor and a read is worth ~2 µs.

**End to end it is 0.3%** — 133.620 → 133.324 ms — because these ops are 472 of 5412 dispatches.
Census 125.60 → 125.00 ms. This is a footprint and hygiene change (40 MB of DRAM and 160 pointless
uploads removed), not a speed change, and it is recorded as such.

## Third change: the corner turn on both NoCs

The corner turn is read-dominated — 128 KB in against 4 KB out per core — and `a2a_corner_pg.cpp`
does all of it on one RISC, so every read rides NOC_1 and NOC_0 sits idle. `dram_guide.md` §3.1
measures driving only one NoC as the single largest factor anywhere (1.79×), so this is the obvious
place to spend it.

Split into a pair that share the same runtime-argument vector:

* `a2a_corner_rd.cpp` — NCRISC/NOC_1, fetches input tiles 0…15 into `c_0` and pushes.
* `a2a_corner_wr.cpp` — BRISC/NOC_0, fetches 16…31 into `c_1` on the *other* NoC, waits on `c_0`,
  assembles from both L1 regions and writes.

`c_1` is never pushed: this kernel is both its producer and its only consumer, so a reserve and a
write pointer are all the ownership it needs. `c_0` is a plain producer/consumer CB across the two
RISCs — no semaphore.

| | µs/call | total |
|---|---|---|
| one NoC (`a2a_corner_pg`) | 41.5 | 25.20 ms |
| **both NoCs** | **30.2** | **18.35 ms** |

**1.37× on the kernel, 7.1% on the multiply** — 133.324 → **124.464 ms**. Less than the 1.79× the
guide measures for this lever, which is expected: the guide's figure comes from a configuration that
was purely bandwidth-limited, while this kernel also spends time on the in-L1 face shuffle that no
amount of NoC helps.

## Fourth change: the folded W lives in L1

The `W` the matmul passes contract against is the **folded NTT matrix** — `w32a` / `w32b` (16 int8
tiles each) and `w64` (4 × 16). It is a pure function of the prime and the ring: it never changes,
and every HE operation re-reads it. Today each 16 KB set is read from DRAM **~304 times per
multiply** (64 cores × ~4.75 dispatches), and again on every operation after that, forever.

It fits in L1 without argument:

| | size | as an L1-interleaved buffer |
|---|---|---|
| W per (prime, direction) | 96 KB | |
| **all 128 sets** | **12.0 MB** | **95 KB per core — 6.6% of L1** |
| twiddles (`ta`, `tb`), all sets | 33.0 MB | +259 KB per core (24.6% with W) |

And it distributes well by accident: the 16 tiles of one set land on **16 different L1 banks**, so
64 readers spread over 16 source cores rather than piling on one.

The change is the allocation and nothing else — `BufferType::L1` instead of `BufferType::DRAM`.
**No kernel change at all**: the readers use `TensorAccessor`, which takes the buffer type from its
compile args, so they follow the allocation. `TG018M_W_DRAM=1` switches back for A/B.

| kernel | W in DRAM | **W in L1** | |
|---|---|---|---|
| `fused_matmul_a` | 17.2 µs | **16.3** | −5% |
| `fused_matmul_c` | 17.2 µs | **16.3** | −5% |
| `fused_matmul_a_acc` | 27.8 µs | **26.2** | −6% |

**124.464 → 122.467 ms, 1.6%** — real, free, and much smaller than the traffic reduction suggests.
Worth understanding why, because it is the guide's own model talking: at **one tile per core** these
dispatches live entirely inside the transient `M(G)`, where the cost is the *un-hidden latency of
the first read*, not bandwidth. Moving that read from DRAM to L1 shortens it but does not remove it,
and nothing else in the dispatch changes. Eliminating 12 MB of traffic per multiply cannot pay until
there is more than one tile per core to amortise it over.

Keep it regardless: it is a two-line change, it removes 12 MB of DRAM traffic per operation for as
long as the context lives, and it is a precondition for the T>1 work rather than a competitor to it.

## Fifth change: more outputs per core in the corner turn — and where that stops paying

An all-to-all means **every core reads all NHALF input tiles**. It then produced exactly **one**
output tile, so 32 reads were amortised over one write. Those reads are pure redundancy: give a core
K outputs and the same 32 reads serve all K. The kernel already took `(out_start, out_stride,
n_out)`, so this is a host-side change — `per = NHALF/K`, stride `per`, count `K`.

It is worth doing, and it stops paying quickly:

| K (outputs per core) | host device busy |
|---|---|
| 1 | 122.427 ms |
| **2** | **120.618 ms** |
| 4 | 129.082 ms — **regression** |

Two things push back, and the second is the interesting one:

* **The K face shuffles are real RISC work** — 32 tiles × 1024 words of L1 copy per output — and they
  do not shrink when the reads are amortised.
* **Raising K shrinks `per`, so a dispatch needs `B ≥ K` transforms in its batch to still fill the
  grid.** Below that, cores go idle and the utilisation loss outweighs the traffic saved. That is
  what K=4 is measuring: the batch is not always deep enough to keep 64 cores busy at `per = 8`.

So the trade is **DRAM traffic against occupancy**, and the optimum is small. This is the same
tension from the other side as the `T > 1` work: batching more work per core only helps while there
is enough independent work left to keep the grid full.

## Sixth: G was set to half the grid, which halved the work per dispatch

Everything above was measured at **G = 64** — inherited from exp_Tg018 — on a **13×10 = 130** grid.
66 cores idle throughout.

*(The knob for this used to be an environment variable called `NTT_CORES`, inherited from
`host/experiments/legacy/nntt8192_mc.cpp` via `he_mul_cross` and exp_Tb001. It never meant "cores for the NTT" — it was
always just G, and the workload here is described by G, B and S and nothing else. It is now
`TG018M_G`, and the default is the whole grid.)*

That is not merely lost parallelism. `LPD = max(1, NC / NTILE)` — **limbs per dispatch** — is derived
from G, and `NTILE` is 64, so `G = 64` gives `LPD = 1`: one limb per dispatch, 64 tiles, one tile per
core. Doubling G does not just add cores, it doubles the *work* in a dispatch and halves the dispatch
count.

| G | LPD | dispatches | host device busy |
|---|---|---|---|
| 32 | 1 | 10,216 | 209.179 ms |
| 64 | 1 | 5,412 | 120.612 ms |
| 96 | 1 | 5,412 | 120.797 ms — **no gain** |
| **128** | **2** | ~2,706 | **84.183 ms** |
| **130** *(default)* | 2 | ~2,706 | **84.140 ms** |

**1.43×**, and the shape of that table is the whole point. **96 cores buys nothing** — `96/64 = 1`
still, so the extra 32 only make the chunking ragged. And **G=32 costs 1.73×**, because halving the
cores doubles the dispatch count (5,412 → 10,216) while `M(G)` only falls by about half. The gain
appears exactly where `LPD` steps.

**G is a work-per-dispatch dial here, not a parallelism dial.** 130 and 128 are within noise (84.140
vs 84.183); the default is the whole grid.

## Seventh: the FBC ran on 64 of 130 cores

The question that led here was whether to run several ct_muls concurrently (`B > 1`) to raise
throughput, on the theory that we are memory bound. **We are not** — the multiply moves 5.74 GB in
84 ms, which is **73 GB/s of the 433 achievable, 17%**. Every kernel's `G·s/B` is far below its
compute floor. What is real is *latency exposure*, and the useful version of the question is
therefore: **where is the grid actually idle?**

Two experiments answered it.

**More independent work per dispatch helps, but only a little.** `LPD` is limbs batched per dispatch:

| LPD | host device busy |
|---|---|
| 2 (was the default) | 84.140 ms |
| **4** | **80.871 ms** |
| 8 | 81.021 ms — flat |

4% and it plateaus, because the transform passes were *already* filling the grid at LPD=2. So `B > 1`
would buy little there either — the transform is not where the idleness is.

**The FBC is where it is.** Its tile axis is only `NTILE = 64` long and it chunked that by `NC`, so
with `G = 130` every FBC dispatch ran **64 cores and left 66 idle** — through **29% of the multiply**.
The destination-chunk loop (`c0`, 8 limbs at a time) is independent of the tile loop, so the two
flatten into one job list: 6 chunks × 64 tiles = **384 jobs**, chunked by `NC`, which fills 130 cores.
Every argument was already per-core, so a dispatch can carry jobs from different `c0` chunks with
different constants. Jobs are grouped by `Ld` first, because the `c_16` depth must equal the writer's
group size and `Ld` is ragged on the last chunk.

| | host device busy |
|---|---|
| FBC on 64 cores, LPD=4 | 80.871 ms |
| **FBC filling the grid, LPD=4** | **69.374 ms** |

**1.17×**, and it makes `LPD = 4` the better default (69.4 vs 72.7 at LPD=2) — with the FBC no longer
starving, longer job lists leave fewer ragged tail dispatches.

**On `B > 1`:** the honest projection is now ~1.1×, not the 1.6× the core-time model suggested. That
model attributed all of G=130's inefficiency to contention, when most of it was this one stage
wasting half the grid. Filling it directly was cheaper than co-scheduling ciphertexts, and it takes
the same idleness away. `B` remains available for the stages that are genuinely limited by having
only one ciphertext's worth of work, but that list is now short.

## Eighth: the answer to "run two ct_muls at once"

The proposal was **G=64, B=2** — two multiplies side by side, 64 cores each. Testing what it could
actually buy turned out to answer it without building it.

**In this dispatch model, B=2 *is* "make the job lists twice as long".** Cores are stateless and a
dispatch takes `min(G, jobs)` from a flat list; it does not care whether two jobs come from different
limbs or different ciphertexts. So B only helps where a stage runs out of independent work **within
one multiply** — and the question becomes: which stages were short?

At that point every stage filled 130 cores except one family:

| stage | job list | cores |
|---|---|---|
| transform matmul | LPD × 64 = 256 | 130 ✓ |
| corner turn | 2 × LPD × 16 = 128 | 128 ✓ |
| FBC (after the earlier fix) | 6 × 64 = 384 | 130 ✓ |
| **`msub` / `pws` / `madds`** *(now batched)* | limbs × 64 ≈ 3008 | 130 ✓ |

Those are called **once per limb** on a single 64-tile polynomial, so each dispatch had 64 jobs and
left 66 cores idle — 754 dispatches of them. But the fix is not B: a multiply has **46–48 limbs** of
this work available, against the 2 ciphertexts B=2 would offer. Batching `(limb, tile)` into one job
list — the same thing `ew_jobs` already did for the twiddles — fills the grid **24× more** thoroughly
than B=2 could.

| | host device busy |
|---|---|
| per-limb dispatches, 64 cores | 69.453 ms |
| **batched over limbs, 130 cores** | **66.552 ms** |

**1.04×**. And it was not the last one — the **tensor** (4 pointwise multiplies + 1 add per limb) and
the **combine** (2 adds per limb) were also called per limb on single polynomials: another **336
dispatches on 64 of 130 cores**. Batching those the same way:

| | host device busy |
|---|---|
| tensor and combine per-limb | 66.552 ms |
| **batched over limbs** | **64.861 ms** |

At that point **every stage fills the grid**:

| stage | job list | cores |
|---|---|---|
| transform matmul | LPD × 64 = 256 | 130 |
| corner turn | 2 × LPD × 16 = 128 | 128 |
| FBC | 6 × 64 = 384 | 130 |
| tensor / combine | LQ × 64 = 3072 | 130 |
| rescale / mod-down elementwise | ≈ 3008 | 130 |

### What that leaves for B

**B is now worth ≈ 0, and the arithmetic is exact rather than a guess.** A stage's job list is
already several thousand entries against G=130, so its dispatch count is `jobs / G`. Multiplying the
job list by B multiplies the dispatch count by B, and each dispatch costs the same. Four multiplies
via B=4 and four multiplies run one after another are then the *same number of equally-full
dispatches* — the hardware cannot tell whether a dispatch's 130 jobs came from four ciphertexts or
from four limbs of one.

B converts **idle cores** into work. There are no idle cores left to convert. It would become
interesting again only for a stage that genuinely has fewer than G jobs of its own — every such stage
in this multiply has now been batched away.

## Ninth: `modmul` — an emulated multiply, then 14 SFPI passes collapsed to 1

Two changes, in that order, both to the tile×tile Montgomery multiply. Neither touched the algorithm.

**(a) The multiply was emulated.** `modmul_sfpi_full`'s header states its premise: *"Wormhole SFPI
has no integer multiply (and no sfpmul24), so each product is the fp32 byte-split trick"* — four fp32
multiplies and twelve int↔float conversions per 16×16 product. Blackhole has `sfpmul24`
(`docs/he/history/exp_tg018s.md` measured its semantics: an exact 23×23 → 46-bit product, silently truncating
past bit 23). Two instructions replace ~21.

**(b) That unlocked the real win.** `modmul_sfpi_full` says *"One product per pass keeps SFPI register
pressure low enough to compile"* — the fp32 code needs 8 registers per product. `fractional_mul`
needs 2, so the partial products can share a pass and stop travelling through DST. Collapsed to a
single pass, with the four `fill_tile_int` broadcasts replaced by scalar arguments,
**DST traffic per tile falls from 608 accesses to 96**. `docs/he/exp_modmul001.md` has the ladder.

| kernel | µs/tile (1 core, fitted) | ported | host device busy |
|---|---|---|---|
| `modmul_sfpi_full` | 12.430 | — | 64.159 ms |
| `modmul_mul24` — (a) only | 5.376 | yes | **56.527 ms** |
| **`modmul_fused3`** — (a)+(b) | **3.087** | yes | **54.100 ms** |

**7.63 + 2.43 = 10.06 ms, 15.7%**, at `0/6029312` and 39.0 bits throughout.

Both ports came in well under their kernel ratios — 2.31× of kernel gave 11.9% end to end, 1.74× more
gave 4.3%. `modmul` dispatches are not 100% of the critical path, so part of each win overlaps work
that did not get faster. **Kernel ratios bound the pipeline effect; they do not predict it.**

Not done: `modmul_scalar` (multiply by a runtime scalar) and `tensor2_fused` still carry the fp32
byte-split and have had none of this applied.

## The design, as it now stands

### Three parameters, three scopes

| | value | scope | what it sets |
|---|---|---|---|
| **G** | 130 (whole grid) | global | how many cores a dispatch may use — the container |
| **LPD** | 4 | the NTT stages | how many limbs are batched into one dispatch — how much work goes in the container |
| **K** | 2 | the corner turn only | how many output tiles a core produces from the inputs it has already read |

`G` is set by `TG018M_G` and defaults to the whole grid. `LPD` and `K` have overrides
(`TG018M_LPD`, `TG018M_A2A_OUT`) that exist for A/B, not for tuning.

### The occupancy rule, and why it is not a formula

A dispatch takes `min(G, jobs remaining)` cores, so **a stage whose job list is shorter than G
silently runs on part of the grid.** Each limb offers a different number of independent jobs:

| stage | jobs per limb | LPD needed to cover G=130 |
|---|---|---|
| transform matmul | `NTILE` = 64 | 3 |
| corner turn | `2·NHALF/K` = 32 | **5** ← tightest |

That reasoning is how to *think* about occupancy, and it is how the two biggest wins were found — the
FBC was running 64 of 130 cores, and at LPD=2 so was the corner turn. But as a way to *pick* LPD it
is wrong, and measurably so:

| LPD | host device busy |
|---|---|
| 2 | 72.678 ms |
| **4** | **69.374 ms** |
| 5 — what the formula gives | **78.091 ms** |
| 8 | 69.615 ms |

Filling the tightest stage is only half the problem: LPD also decides how every *other* stage chunks.
Five limbs give the matmul 320 jobs, which at G=130 is 130+130+60 — a third of a dispatch thrown away
on the tail. Four limbs give 256, which chunks 130+126. No LPD divides every stage evenly, so **LPD=4
is a measured constant with a known reason, not a derived one.**

### Cores are stateless; every stage boundary is DRAM

Nothing is handed core-to-core. A dispatch's readers pull tiles from DRAM by page index, the compute
runs, the writers put tiles back, and the next dispatch re-establishes an entirely new tile→core
mapping from its own runtime args.

```
corner-turn dispatch                 next dispatch
core 0 → out[0], out[2] ─┐        ┌→ core 0 reads whatever tile it is assigned
core 1 → out[1], out[3] ─┴ DRAM ──┴→ core 1 reads ...
```

Two consequences worth holding onto:

* **It is why K is free to change.** K only alters *which core wrote which tile*, and nothing
  downstream knows or cares. No handoff, no ordering, no semaphore.
* **It is the structural ceiling on this design.** Every stage boundary is a full round trip: 64
  tiles out, 64 tiles back. The corner turn moves 4 MB of DRAM to perform a *transpose*. Of the
  multiply's ~5.7 GB, the great majority exists only because stages talk through memory.

Removing that means making cores stateful across dispatches — the same tile staying on the same core,
or a direct core-to-core NoC handoff. Both are larger changes than anything here, and both collide
with the workload cache and the trace machinery, which assume a dispatch is a pure function of its
runtime args. It is the next real ceiling, not a tuning knob.

## Where it stands now

| kernel | exp_Tg018 | exp_Tg018m | total ms |
|---|---|---|---|
| `modmul_sfpi_full` | 23.4 | 17.0 | 48.43 → 30.47 |
| `baseconv_fbc` | 852.4 | 777.2 | 25.57 → 23.31 |
| `a2a_corner` | 170.4 | **30.1** | 103.62 → **18.29** |
| `fused_matmul_a_acc` | 31.2 | 26.2 | 19.00 → 15.95 |
| `fused_matmul_a` | 20.5 | 16.3 | 12.47 → 9.89 |
| `fused_matmul_c` | 20.2 | 16.3 | 12.29 → 9.89 |
| `modmul_scalar` | — | 15.0 | — → 4.23 |
| `modadd_map` | 16.5 | 4.4 | 9.75 → 1.78 |
| `modsub_reduce` | 16.5 | 4.3 | 4.64 → 1.20 |
| `modadd_scalar` | — | 3.2 | — → 0.61 |
| `bconv_one` | 212.0 | 99.8 | 0.85 → 0.40 |
| **census total** | | | **236.62 → 116.00** (at G=64, before K=2) |
| **measured host device busy, G=64** | **238.663** | 120.612 ms | 1.98× |
| **measured host device busy, G=130** | | **66.552 ms** | **3.59×** |

*(the per-kernel column above is the G=64 census; the G=128 figure is end-to-end)*

The profile has completely changed shape. `a2a_corner` went from the largest item at 43.8% to the
third at 15.5%; the three biggest are now `modmul` (25.8%), `baseconv_fbc` (19.7%) and the corner
turn, and **all three are compute-dominated**.

## The profile at 50.920 ms — VERIFIED 1-1, a2a included

`tools/traces/dispatch_csv.py ... --drain --dataflow-ok --labels`, 2884 dispatches,
**`join verified: all 2884 rows agree on the kernel between host and trace`**.
Data: `docs/he/data/tg018m_dispatch_51ms.csv`.

| item | dispatches | ms | % of 50.98 |
|---|---|---|---|
| **FBC** (`fbc_combine` 10.666 + `fbc_scale` 0.569) | 20 | **11.235** | **22.0%** |
| **modmul** (`modmul_fused3` 10.489 + `modmul_scalar` 2.341) | 1230 | **12.830** | **25.2%** |
| **NTT matmuls** (`fusedAcc` 5.398 + `fused` 3.799 + `fusedC` 3.746) | 912 | **12.943** | **25.4%** |
| **`a2a_corner`** | 152 | **9.103** | **17.9%** |
| `modadd_map`, `modsub_reduce`, `modadd_scalar`, `bconv_one` | 570 | 3.399 | 6.7% |
| **not inside any kernel** | — | **1.469** | **2.9%** |

### Two earlier claims here were wrong, and this is why

An earlier census in this doc reported **"dispatch overhead 8.42 ms, 15.6%"** and a per-kernel table
with `modmul_fused3` at 1149 dispatches against the host's 1088. Both came from the same defect.

`dispatch_csv.py` keeps only dispatch groups that carry a **TRISC envelope** — a compute kernel. The
corner turn is `a2a_corner_rd.cpp` + `a2a_corner_wr.cpp`, **dataflow only**, so all 152 of its
dispatches were dropped from the window. Consequences:

* **`a2a_corner`'s 9.103 ms disappeared entirely** — 17.9% of the multiply, invisible.
* the shortfall showed up as a gap between critical-path busy and device busy, which I read as
  **launch overhead**. It was not. Real overhead is **1.469 ms, 2.9%**: the device is inside a kernel
  **97.1%** of the time.
* the surviving windows absorbed the a2a *names*, inflating four kernels' counts by exactly 152 and
  their times by ~13%.

**The fix is a flag the tool already documents**, `--dataflow-ok`: *"keep dispatches that run no
compute kernel (no TRISC envelope). They carry no named zone either, so the host label is the only
name available."* Paired with `--labels` it names them from the host's own record and verifies every
row. Adding a `DeviceZoneScopedN` to the two a2a dataflow kernels does **not** fix it on its own —
the group is filtered before naming ever happens.

**Always run this census as `--drain --dataflow-ok --labels <TG018M_LABELS file>` and check for
`join verified`.** Without it the corner turn is silently missing.

### REFUTED: `MathFidelity::LoFi` on the NTT matmuls

`ComputeConfig`'s default is `MathFidelity::HiFi4` (`kernel_types.hpp:106`) and exp_Tg018m never sets
it, so `fused_matmul_a`, `_a_acc` and `_c` contract their INT8 radix-128 limbs at HiFi4 — four passes
through the multiplier where the operands are 7-bit magnitudes. That looked like a free 4× on the
matrix-engine phase.

**It is not exact.** With LoFi on all three: `786432/786432`, `262144/262144` and `6029312/6029312`
mismatches — every output wrong. So the existing default is correct, and the reference tests
(`baseconv_matmul.cpp`, `fused_modmatmul.cpp`) that set `HiFi4` explicitly were right to. Exact
int8×int8 → int32 accumulation needs the full mantissa datapath on this hardware.

**But the timing is the useful part: 54.100 → 53.526 ms, 1.1%.** Taking the standard model that LoFi
is ~4× cheaper than HiFi4 on the matmul, the matmul phase can be at most ≈0.77 ms of the 17.50 ms
those three kernels cost:

| phase of `fused_matmul_a` / `_a_acc` / `_c` | what it runs | share |
|---|---|---|
| decompose | 1 SFPI pass | |
| **matmul** | `matmul_tiles` x16 | **~4%** |
| **recombine** | **6 SFPI passes, 9 `mul_int_tile`, 5 `fill_tile_int`** | **the rest** |

**The kernels are misnamed.** They are SFPU kernels with a small matmul inside, and the matrix engine
is not where their time goes. Optimising them means optimising the recombine — which is structurally
the old modmul (`hybrid_canon` in `docs/he/exp_modmul001.md`: 4 SFPI passes + 9 `mul_int_tile` +
4 FILLs, 4.930 µs/tile) and takes the same three moves that got that to 3.086:

1. `mul_int_tile` → `sfpi::fractional_mul` — 2 registers per product instead of 8
2. collapse the 6 recombine passes into one
3. delete all 5 `fill_tile_int` broadcasts via scalar arguments

Not built. The caveat from the modmul work applies: a 1.6× on the recombine is not 1.6× on the
kernel, and the kernel ratio is not the pipeline ratio.

### Tenth: the NTT matmuls are SFPU kernels, and their recombine is the old modmul

The LoFi refutation above put the matrix engine at ~4% of `fused_matmul_a` / `_a_acc` / `_c`, so
their time is the SFPU phases. Their phase-3 recombine computes
`R = (C0 + C1<<7 + C2<<14 + C3<<21) mod q` by one Montgomery reduction — structurally the modmul,
and it ran the same way the old modmul did: **6 SFPI passes, 9 `mul_int_tile`, 5 `fill_tile_int`**.

The three moves from `docs/he/exp_modmul001.md` apply unchanged: `fractional_mul` instead of
`mul_int_tile` (2 registers per product, not 8), the six passes collapsed into one, and all five
`fill_tile_int` broadcasts replaced by scalar arguments. Per kernel:

| | SFPI passes | `mul_int_tile` | `fill_tile_int` |
|---|---|---|---|
| before | 7 | 9 | 5 |
| **after** | **2** | **0** | **0** |

(The surviving pass is phase-1 decompose.) No register spill, and the arithmetic is untouched — same
reduce, same constants, same canonical output.

| | host device busy | |
|---|---|---|
| before | 54.100 ms | |
| `fused_matmul_a` alone | 53.401 ms | −0.70 ms, isolating the change to 304 of 912 matmul dispatches |
| **all three** | **51.607 ms** | **−2.49 ms, 4.6%** |

`0/6029312` and 39.0 bits at both steps. **Cumulative: 238.663 → 51.607 ms, 4.63×.**

### The three kernels are not duplicates

Worth recording, since it decides where further work goes:

* **`fused_matmul_a` vs `fused_matmul_c`** are the two matmuls of the four-step NTT (N=1024 = 32×32),
  which is `W·X` then twiddle then `X·W`. The matrix engine computes `LHS @ RHS`, so one stage needs
  the twiddle matrix on the left and the other on the right — `matmul_tiles(cb_w, cb_limbs, ...)`
  against `matmul_tiles(cb_limbs, cb_w, ...)`. Same arithmetic, mirrored operand roles. A streams its
  W tiles one at a time; C keeps all 16 resident and indexes them.
* **`fused_matmul_a_acc`** is stage A that accumulates `NACC` inputs into ONE set of planes and
  recombines **once**, replacing `NACC` separate dispatches each doing a full recombine plus a modadd
  tree to sum the partials.

**So `_acc` already amortises its recombine over `NACC` inputs**, which means this change helps it
less per unit of work than it helps A and C — and since the matmul is ~4%, `_acc`'s remaining cost is
likely dominated by its `NACC` decompose passes. That is a different target and is not yet measured.

### Eleventh: the planes never needed to leave DST — and where the NTT matmul stops

`matmul_tiles`' last argument is the DST index, and it was hardcoded to `0`: each of the 4 planes was
accumulated, packed to `cb_planes`, released — then phase 3 copied all four straight back. Letting
plane `p` accumulate into DST tile `p` inside ONE acquire lets the recombine read them in place.

| per dispatch | before | after |
|---|---|---|
| `copy_tile` (CB→DST) | 5 | **1** |
| `pack_tile` (DST→CB) | 6 | **5** |
| acquire/release cycles | 5 | **2** |

| | host device busy | |
|---|---|---|
| after the recombine fusion | 51.607 ms | |
| `fused_matmul_a` alone | 51.335 ms | −0.272, isolating the change |
| **all three** | **50.920 ms** | **−0.687, 1.3%** |

`0/6029312` throughout. **Cumulative: 238.663 → 50.920 ms, 4.69×.**

**What is left is structural.** The decompose must `pack_tile` its 4 int8 limbs to a CB, because
`matmul_tiles` reads its operands through the UNPACKER, not DST. So 6 tile moves per dispatch (1
input copy, 4 limb packs, 1 output pack) is the floor for this kernel shape, down from 11.

### STATUS: the NTT matmul is CLOSED, and why

Three things were tried; the accounting they produced is the reason to stop.

| | result |
|---|---|
| `MathFidelity::LoFi` | **refuted** — not exact (`6029312/6029312`). Its 1.1% delta put `matmul_tiles` at ~4% of these kernels |
| recombine fused to one pass | **−2.49 ms**, the block 17.501 → 14.694 ms (−16.0%) |
| planes kept in DST | **−0.69 ms** |

**A whole-tile move costs only ~0.1 µs** — 8 of them saved 0.81 µs/dispatch. So tile traffic was
never the dominant term, and removing the last 4 would not repay the effort.

**The kernel is not the bottleneck; the pipeline is.** The same kernel costs ~3.3× more inside
exp_Tg018m than on one core:

| | one core | at G=130 | inflation |
|---|---|---|---|
| `modmul_fused3` | 3.086 µs/tile | 11.202 ms / 1088 = 10.3 µs | **3.3×** |

That inflation, not the arithmetic, is what `fused_matmul_a`'s 12.92 µs/dispatch mostly is. Its own
DRAM traffic is 8 KB/core (the folded W lives in L1, limbs and planes stay in CBs) = **82 GB/s at
G=130, 33% of the measured ceiling** — so the kernel individually has headroom, while
`modmul_fused3` at 12 KB/core is already at **155 GB/s, 62%**.

**So further per-kernel work has poor leverage.** The remaining levers are system-level:

1. **The FBC** — 10.67 ms, 15 dispatches of pure compute, and `docs/he/history/exp_tg018r.md` already
   measured the matrix-engine route at **1.98×** at this exact shape. Built, never ported.
2. **Dispatch overhead** — 8.42 ms, 15.6%, at 2.92 µs × 2884 dispatches; responds to batching.
3. **The G=130 contention itself**, which is what makes every kernel cost ~3× its single-core figure.

### What to take next

1. **The FBC.** Largest clean item, only 15 dispatches at 711 µs each — pure compute, nothing to
   schedule away — and **the fix is already built and measured**: `docs/he/history/exp_tg018r.md` puts the
   matrix-engine route at **1.98×** at this exact shape with the transpose fused in. 11.23 → ~5.7 ms,
   ~10% of the multiply. Never ported.
2. **Dispatch count.** 15.6% is launch tax, and it responds to batching rather than arithmetic.
3. **The NTT matmul block** is the largest but needs the a2a attribution separated before it can be
   sized honestly, and it is three kernels plus the corner turn rather than one change.

## What is left

* **`modmul` (tile×tile) — DONE, see "Ninth" below.** 12.430 → 3.087 µs/tile at the kernel, and
  **64.159 → 54.100 ms** end to end. It is now arithmetic-bound: one SFPI pass, 2 loads + 1 store,
  which is the floor for a two-operand elementwise op. `modmul_scalar` and `tensor2_fused` still
  carry the old fp32 byte-split and are untouched.
* **More than one tile per core is now the only structural change left, and everything is waiting
  on it — but the K sweep above is a warning about how it will behave.** Occupancy and
  work-per-core trade directly: batching helps only while enough independent work remains to keep
  the grid full. The right axis is probably not "raise T at fixed G" but "choose (G, T) per stage
  from the amount of independent work that stage actually has". At T=1 every dispatch sits inside the transient `M(G)`, which is why the W-in-L1 change
  returned 1.6% for a 12 MB traffic reduction. T>1 amortises the transient, makes `w_share` mean
  something, and is what would let the placement rule apply. Three levers, one restructuring.
* **`baseconv_fbc` barely moved** (1.10×) — at 777 µs per dispatch it is dominated by compute, not
  by its feed.
* **`baseconv_fbc` is now 19.7% and has barely moved** (852 → 777 µs). At 777 µs per dispatch it is
  compute, not feed — the only lever there is the FBC arithmetic itself. **That lever now exists and
  is measured:** `docs/he/history/exp_tg018r.md` runs the same conversion on the int8 matrix engine with the
  limb↔coefficient transpose fused into its own reader and writer, and at this exact shape
  (`Ls=16, Ld=8`, N=2^16, 64 cores) it measures **399.2 µs against 788.7 µs — 1.98×**, output
  verified identical. That is **11.897 → 6.02 ms, 9.2% of the whole multiply**, with no change to
  the limb-major layout. Porting it is the single largest remaining item that is already built.
* **The corner turn's remaining 30.2 µs is increasingly the in-L1 face shuffle**, not DRAM. Both
  NoCs are now busy; going further means moving less data (a different decomposition) rather than
  moving it faster.
* **More than one tile per core** is untouched here and is what would move the whole workload out of
  the transient regime — after which the placement rule would begin to apply and `w_share` would
  become available for the matmul passes.

## Running it

```
. ./env.sh
TG018M_DEV=1 TG018M_TRACE=1 TG018M_ITERS=3 ./build/exp_tg018m     # G defaults to the whole grid
```

Same environment variables as exp_Tg018 with the `TG018_` prefix replaced by `TG018M_`, plus:

| env | default | meaning |
|---|---|---|
| `TG018M_G` | the whole grid (130) | **G**, the core budget — the only knob for it |
| `TG018M_A2A_OUT` | 2 | K, output tiles per core in the corner turn |
| `TG018M_DIAG` | off | diagonal core placement (measures 2.5× worse — see above) |
| `TG018M_W_DRAM` | off | put the folded NTT matrices back in DRAM, for A/B |
| `TG018M_LPD` | `G / 64` | limbs per dispatch, normally derived from G |
