# PIR answer-pipeline latency — optimization record (Blackhole p150)

Working record of taking the OnionPIR **answer pipeline** from 3.45 s to 164 ms per query on a single
Blackhole **p150b**, bit-exact throughout. This is milestone **#5** of `ONION_MATCH_PLAN.md` (device
residency), which that plan lists as the sole remaining gap to an end-to-end latency number.

> **Hardware.** Numbers here were measured on **Wormhole n150**. Current hardware is Blackhole p150 — see the support matrix in the [root README](../../README.md) and [`../he/blackhole_port.md`](../he/blackhole_port.md) §3 for why they are not directly comparable.


All numbers here are **p150**. Every PIR number in the other `docs/pir/*.md` files predates this and was
measured on a Wormhole **n150** — treat those as stale.

Method follows `docs/he/history/OPTIMIZATION_REPORT.md`, whose ladder (5044 → 727 → 142.7 → 83.8 ms/mul) this
work reuses directly rather than rediscovering.

---

## 1. Headline ladder

Workload: `test_pir_answer2048`, K=3 (~84-bit, 3×28-bit RNS), n=2048, ℓ=8, B=2²⁰, dims=[8,2,2]
(32-cell hypercube, 2 folds). Single Tensix core for the answer ops. Bit-exact vs `onionpir_ref.pir_answer`
at every rung.

| step | ms/query | vs baseline | correctness |
|---|--:|--:|---|
| baseline (host-orchestrated, warm JIT) | 3450 | 1.0× | 0/12288 |
| **1 — workload cache + fixed-address buffer pool** | **241** | **14.3×** | 0/12288 |
| 2 — packed transform readbacks | 221 | 15.6× | 0/12288 |
| 3 — resident MAC in the external product | 183 | 18.9× | 0/12288 |
| **4 — resident digit/RGSW transforms (packed, offset reader)** | **164.5** | **21.0×** | 0/12288 |
| **5 — Metal Trace capture/replay** | **112.5** | **30.7×** | 0/12288 |
| 6 — decrypt-check harness (no golden needed) | 112.5 | — | 0/2048 decrypt |
| **7 — co-scheduled MAC (`mmul_batch`)** | **103.8** | **33.2×** | 0/12288 + 0/2048 decrypt |

**NEGATIVE RESULT, recorded deliberately.** The same co-scheduling applied to `digit_rns`
(`digit_rns_batch`, all `2ℓ=16` splits in one program) measured **~2–3% SLOWER** here: 107.0 ms vs
104–105 ms, both at <5% spread. Cause is scale — only 48 of 5,619 dispatches are `digit_rns`, so
consolidating them changes little, while the single large readback lands more squarely on the critical
path now that device time is exposed (readback is 92% of the replay). The call site was **reverted**; the
method is retained in the engine for EXPANSION, which has ~4,088 such calls (8 per key-switch × 511).
**Re-measure it there rather than assuming it transfers** — co-scheduling is not universally a win, it
pays in proportion to how many independent ops of that shape the workload actually has.

Step 7: `mac_ntt_buf`'s `nk` independent products became ONE program over `nk` cores instead of `nk`
sequential dispatches. Cached workloads 5,934 → 5,664 (−270 = 18 MACs × 16 products, exactly as expected).
Only −7.7% here because this workload has just 18 MACs — the mechanism matters for **expansion**, which has
511 key-switches each with a 2ℓ=16-row MAC. See §4e for the 12×-over-trace measurement behind it.

First REAL retrieval beyond a toy hypercube: **512 cells (dims=[64,2,2,2]) at 846 ms/query**, validated by
decryption alone with no Python golden — see §4c.

Remaining budget at 112.5 ms: **blocking readback 93.1 ms (94.7%)**, enqueue 1.7 ms (1.8%), write ~5 ms,
program build 0. Run-to-run spread is down to **1.3%**.

---

## 2. Measurement methodology (read this before trusting any number)

**Two traps cost real time here; both are avoidable.**

1. **Discard the first run.** A cold kernel-JIT cache reads **10.0 s** for this workload (folds 8.9 s).
   That is compilation, not dispatch. Warm runs are stable at ~3.45 s. An early conclusion that "Blackhole
   fold latency regressed 3.2× vs n150" was entirely this artifact — warm p150 (3.45 s) is *faster* than
   the recorded n150 (4.3 s).
2. **Never time a single pass.** The enqueue term alone swings ~12% run to run, which hides changes of
   ~25 ms. `test_pir_answer2048` takes `PIR_ITERS` (default 5) and reports min/mean/max/spread; **use
   `min`** (least host interference). Typical spread 3.6–9.7%, so changes above ~15 ms are detectable.

**Host-phase attribution.** `pir2048::Engine` carries a `prof::` split (`pir_engine2048.hpp`) modelled on
`he_mul_cross.cpp`: BUILD (`CreateProgram`+`CreateKernel`), ENQ (`EnqueueMeshWorkload`), WRITE, READ
(blocking). Which term dominates decides the fix — BUILD/ENQ → cache workloads and trace; READ → residency.

**What the split is NOT.** It decomposes *host wall-clock* by what the single host thread was doing. Rows
sum to the wall; there is no device-busy row. Device work overlaps these phases and the device mostly
*starves* waiting on host prep. Device-busy only surfaces at blocking drains.

---

## 3. Step 1 — workload cache + buffer pool (14.3×, the big one)

**Measured cause.** At baseline, ENQ was 83–86% of host time at **0.49–0.51 ms/call** over 5,898
dispatches; BUILD was ~11%; blocking READ only ~5%.

**The misreading to avoid.** That split invites the conclusion "caching only removes the 11% build". It is
wrong: the enqueue cost *is* fresh-program serialization. Every dispatch used a freshly built `Program`, so
`EnqueueMeshWorkload` re-serialized binaries, CB configs and runtime args from scratch. Confirmed directly
by `test_dispatch_micro` (64 single-tile modmul dispatches, all bit-exact 0/2048):

| regime | ms/dispatch | speedup |
|---|--:|--:|
| build-fresh + enqueue | 0.570 | 1.0× |
| **persistent re-enqueue** | **0.031** | **18.4×** |
| traced replay | 0.031 | 18.5× |

Trace *ties* persistent at 64 dispatches — its advantage is being O(1) in dispatch count, so it only
separates at PIR's real scale (5,898 dispatches → ~183 ms persistent vs a few ms traced).

**Implementation** (`pir_engine2048.hpp`):
- `emit(lambda)` replaces inline program construction at all dispatch sites. On replay the lambda is
  **never invoked**, so construction is skipped entirely — caching the `Program` but rebuilding around it
  would not have worked.
- `CACHE_ON` / `CACHE_REC` + `WC` (recorded workloads) + `WI` (replay cursor).
- `POOL` + `PI` + `begin_pass()`: op scratch comes from a deterministic pool that rewinds per query, so
  cached runtime args (which bake in buffer addresses) stay valid. Divergent allocation order throws
  rather than silently aliasing. Engine constants are allocated with the cache OFF so they never enter the pool.

**Result:** 3450 → 241 ms. BUILD 392 → 0.0 ms; ENQ 0.491 → **0.020 ms/call**.

### This is a compile-once (JIT) model, and it is validated as such

The record pass is a one-time cost, re-paid only when the DB **structure** changes — not per query. Proven,
not assumed: `test_pir_answer2048 caseA.bin caseB.bin` records on idx=5, then replays a **different query**
(idx=9, same dims/K/ℓ) with **zero workloads rebuilt** → **0/12288 vs its own golden**, and the two answers
differ in 12288/12288 coefficients (guard so a stale result cannot fake a pass). Selectors and DB contents
enter as untraced buffer writes; only structure is baked in.

---

## 4. Steps 2–4 — residency, in the order the data demanded

After step 1 the profile **inverted**: blocking readback went 5% → 43%. Only then was residency worth doing.

**Ordering matters, and the plan had it backwards.** `ONION_MATCH_PLAN.md` prescribes *kernel fusion* for
milestone #5. HE's experience says otherwise: its Step-3 buffer residency **alone** was 2.8× *slower*
(727 → 2047 ms) because an on-device transpose added ~720 dispatches. Residency's real payoff is that trace
bakes in buffer addresses, so buffers must be allocated once. Correct order: **cache first, then residency,
then trace** — and after caching an extra dispatch costs 0.020 ms instead of 0.57, so the transpose that
sank HE's attempt is ~28× cheaper here.

**Step 2 — packed transform output.** `madd2`/`fusedC` gained an output tile offset, so a transform's two
blocks share one buffer (`bOut2`) and are read in **one** call, not two. Readbacks 1,668 → 1,146 (−31%),
readback time 96.2 → 71.3 ms. Wall was **flat** (239.6 → 241.4 ms) — within noise, and reported as such at
the time. Real value was structural: 522 fewer host boundaries, which is what fragments a trace.

**Step 3 — resident MAC.** Replaced per-row `pmul` + `padd` (2·twoL·K + 4·K blocking readbacks per
extproduct) with `mac_ntt`: products written into one DRAM buffer at tile offsets, summed by a strided
`modadd_reduce`, one readback per (col,ch). RGSW rows uploaded once per fold instead of per call.
221 → **183 ms**; readbacks 1,146 → 840; pool *shrank* 1,824 → 1,260 (`pmul` allocated 3 buffers/call vs
`tilebuf2`'s 1). Workload count unchanged at 5,898 — the win is purely removed host round-trips.

**Step 4 — resident digit + RGSW transforms.** Removed the round trip where `fwd2048` read a freq-domain
poly back to host only for `tilebuf2` to re-upload it (2·twoL·K readbacks *and* the same in writes, per
extproduct). Chain is now `MAC → transpose → inverse transform` entirely in DRAM, one readback at the end.
183 → **164.5 ms**; readbacks 840 → **486** (−42%); readback share 31% → 20%.

### The two facts step 4 rests on (both verified, not assumed)

1. **The on-device transpose IS the host reindex, exactly.** Host fwd writes
   `X[(bk2·T+k2loc) + N2·k1] = blkOut[bk2][k2loc·T + k1]`; host inv reads
   `blkIn[bk2][k1·T + k2loc]`. So `blkIn[bk2]` is the within-tile transpose of `blkOut[bk2]` — the same
   identity `he_mul_cross.cpp` uses. Ops *between* fwd and inv are elementwise, hence layout-agnostic, so
   only the fwd→inv hand-off needs a transpose. Validated: `test_ntt2048_res` checks [1][2][4].
2. **Montgomery conversion commutes with the transform.** The pipeline did `to_mont(fwd2048(x))` —
   transform on device, Montgomery-convert **on host**, which a resident transform has nowhere to put.
   `to_mont` is scaling by R mod q and the NTT is linear mod q, so `fwd(to_mont(x)) == to_mont(fwd(x))`,
   letting `to_mont` move to the coefficient-domain input. Validated **0/2048** as check [5]. An
   off-by-a-factor here would corrupt every RGSW row while still looking like valid ciphertext, so this
   check is load-bearing.

### `reader_fused` blocked this, and why it got a sibling not an edit

`reader_fused.cpp` hardcodes `noc_async_read_tile(0, ...)`, so a transform stage can only read block 0 of
its input — blocking the packed single-buffer-per-poly layout the MAC needs. It has **20+ call sites**
across `device/ntt`, `device/he` (including the freshly Blackhole-validated `he_mul_cross`) and
`device/pir`, and TT runtime args are **positional**, so adding an argument there would make every caller
that does not pass it read garbage.

Solution: **`dataflow/reader_fused_off.cpp`**, a copy taking a tile-offset arg, used only by the PIR
resident path. Plus engine `fusedA_off`/`fusedC_off`, packed `fwd2048_buf2`/`inv2048_buf2`, and tile
offsets on `transpose_dev`. Every existing caller is provably untouched.

---

## 4b. Step 5 — Metal Trace capture/replay (1.46×, and it changes what the bottleneck IS)

Program reuse had enqueue at 0.019 ms/call, but ~5.9k dispatches still made it **77%** of the query. Trace
is O(1) in dispatch count: **837 traces** captured from 5,934 cached workloads, so replay issues 837
commands instead of 5,934 enqueues.

| | step 4 | step 5 |
|---|--:|--:|
| REPLAY min | 164.5 ms | **112.5 ms** |
| workload ENQ | 115.2 ms (77%), 0.019 ms/call | **1.7 ms (1.8%)**, 0.002 ms/call |
| buffer READ | 29.3 ms (20%), 0.060 ms/call | 93.1 ms (94.7%), **0.192 ms/call** |
| spread | 9.7% | **1.3%** |

**Implementation** (`pir_engine2048.hpp`): `TR_PHASE` (0 eager / 1 capture / 2 replay), `SEG` trace ids,
`seg_break()` called from `wrb`/`rdb`, `close_capture()`. Test reserves a 256 MB trace region at init
(`create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 256<<20)`) and runs record → capture → replay.

**Two things that will bite anyone repeating this:**
1. **Capture RECORDS, it does not EXECUTE.** The capture pass's outputs are meaningless — comparing them
   against the recorded answer gives 12288/12288 "mismatches" and a spurious FAIL. Discard them
   (`(void)ansCap`), as `he_mul_cross` does. The meaningful gates are replay-vs-record, replay-vs-golden,
   and the new-query check.
2. **Capture must come after the record pass.** The first enqueue of a fresh program writes its binary, and
   trace forbids writes during capture — so persistent programs must be warmed first.

**The readback per-call rise (0.060 → 0.192 ms) is not a regression — it is device compute becoming
visible.** With host enqueue no longer occupying 115 ms, there is nothing left to hide device work behind,
so it surfaces at the blocking drains. Per `OPTIMIZATION_REPORT.md`: "device-busy only becomes visible wall
time at the two blocking drains." So the 93.1 ms now contains real device work plus DMA, and we are near
this pipeline's device floor rather than measuring host overhead. **Next lever is therefore fewer host
boundaries** (486 readbacks → also fewer than 837 segments), which cuts DMA count *and* lets more device
work overlap — not more dispatch-cost work, which is now 1.8%.

## 4c. Step 6 — decrypt-check, and the first REAL multi-hundred-cell retrieval

`pir_decrypt()` in `test_pir_answer2048.cpp` is the client decrypt: phase = `c0 + c1·s` per RNS channel,
Garner-CRT the K residues to the ~84-bit q, then `round(v/delta) % t` on the balanced representative
(`delta = q//t`), mirroring `onionpir_ref.decrypt`. Rounding is half-away-from-zero vs Python's banker's
rounding; ties are measure-zero under noise.

**Gated before being trusted.** Run first on a case that HAS a golden (dims=[8,2,2]): golden compare
0/12288 *and* decrypt-check **0/2048**, coeff0 got=129 expected=129. Only then used where no golden exists.

**Real retrieval at 512 cells, no Python golden anywhere** (`--no-golden` case, dims=[64,2,2,2], idx=100):

```
RECORD:  22.76 s   (30,614 workloads cached, 11,328 pool buffers)
CAPTURE: 5,977 traces
REPLAY:  846 ms/query   (spread 3.9%)
DECRYPT-CHECK  decrypt(device answer) vs plaintext DB[100]: 0/2048   coeff0 got=255 expected=255
```

16× the cells of anything run before, retrieved correctly, validated end-to-end by decryption alone.
Scaling is sub-linear — 32 cells → 112.5 ms, 512 cells → 846 ms (7.5× for 16× cells) — because folds grow
logarithmically while the scan grows linearly.

### Measured consequence: the int8 scan is a PREREQUISITE for 128 MB, not an optimization

The scan dominated the record pass (16.5 s of 22.8 s), and 512 cells already needed 30,614 cached workloads
and 11,328 pool buffers. The 128 MB cell is 40,330 cells, ~79× more, which puts the **pointwise-MAC** path
at millions of cached workloads and multi-GB of pool — it will not fit the 256 MB trace region, and the
record pass would run for ~30 minutes. So `scan_mm` + `recombine_rns` must replace the pointwise MAC before
a full-size run is attempted; it is not a throughput nicety.

## 4d. Step 7 — the first dimension as a BATCHED GEMM (resolving an apparent matmul mismatch)

### The apparent gap

`scan_mm.cpp` performs a genuine matrix product, `C[i][j] = Σ_k Σ_m A_k[i][m]·B_k[m][j]` (its CPU golden
in `test_scan_mc.cpp` confirms this), while the first dimension in the eval domain is

```
acc[r][c][i] = Σ_k  DB[k·rest + r][i] · sel[k][c][i]          (mod q)
```

with the coefficient slot `i` on **both** operands. Forcing `i` onto the two free axes of one GEMM yields
`C[i][j] = Σ_k DB[…][i]·sel[k][j]` — every slot *pair* — of which only the diagonal is meaningful: 32×
redundant MACs and 32× redundant output writes. Corroborating evidence that the existing benchmark was
built for the *demo's* semantics: `reader_scan_mm.cpp` shares one `B` across all output tiles, which is
only valid when the eval-domain query is slot-independent, i.e. **constant-polynomial** plaintexts
(`pir_demo.md`: "each DB slot holds one integer"). The 128 MB cell packs a full plaintext per cell, so
that collapse does not hold.

### The resolution: put the slot on the BATCH axis

Ask not "is it a GEMM" but "which GEMM". Per slot `i` and RNS channel:

```
C_i (rest × ncomp) = A_i (rest × N0) @ B_i (N0 × ncomp)
```

a **batch of n GEMMs** (batch = slots × channels). Both operands are slot-local, so there is **no
cross-slot product and no diagonal to extract**. Axis mapping:

| GEMM axis | size (128 MB cell) | 32-tiles |
|---|--:|--:|
| m = `rest` | 128 | 4 |
| k = `N0` | 512 | 16 |
| n = ct components | 2 | 1 (padded from 2) |
| batch = slots × channels | 2048 × 3 | — |

**Validated host-side** by `reference/pir/check_scan_gemm.py` (no device needed):

| check | result |
|---|---|
| batched GEMM == per-slot dot product | **0 mismatches** |
| int8 radix-128 limbs + int32 accumulate + radix recombine == modular product | **0 mismatches** |
| largest int32 plane accumulator, N0=512 | 8.26M = **0.38%** of int32 max |
| int32-safe contraction width | up to **N0 = 133,144** |

So the lowering is exact and the accumulator has ~260× headroom at N0=512 — the silent-overflow failure
mode does not exist here. `recombine_rns` also needs **no diagonal variant**: it now reduces genuine m×n
output tiles.

### The one real inefficiency, and the fix

`n = 2` on a 32-wide output axis wastes 16×. That is not cosmetic:

- useful ≈ 805M modular MACs/query × 4 planes ≈ **3.2 G int8 MACs** → ~2 ms at 1556 GMAC/s
- with 16× padding → ~51 G → **~85 ms**, flipping the scan from bandwidth-bound (~20 ms to stream the
  991 MB limb-form DB) to compute-bound

**Fix: batch queries on the `n` axis** — `n = 2·Q`, so `Q=16` fills all 32 columns. It composes because
**`A` (the DB) is shared across queries**, so the DB streams once per *batch*:

| | per query (Q=16) |
|---|--:|
| scan compute (n fully utilized) | ~2 ms |
| DB streaming amortized | ~1.3 ms |

This makes **QPS/card** the honest headline with single-query latency reported separately — which is what
`ONION_MATCH_PLAN.md §8` asks for anyway.

### Device kernel — DONE and validated at the real shape

`kernels/shared/compute/scan_gemm.cpp` + `kernels/shared/dataflow/reader_scan_gemm.cpp` + `pir/test_scan_gemm.cpp`:

| case | shape | result |
|---|---|---|
| small | m=32, N0=64, ncomp=2 | **0/64** |
| **real 128 MB cell, per slot** | **m=128, N0=512, ncomp=2** | **0/256** |

Both bit-exact vs a host golden computing `C = A @ B mod q` with non-constant operands (the packed-DB
regime, not the constant-poly one).

**Key simplification that made this low-risk:** 4 int8 limbs per element is *exactly 4 bytes* — identical
to one uint32. So pre-decomposing both operands offline costs **zero** extra storage or bandwidth, and it
removes the on-core SFPU decompose phase entirely. That keeps the kernel **matmul-only**, avoiding the
SFPU↔matmul engine switch that caused the DEST-sync deadlock in `blackhole_port.md`. The mod-q recombine
stays a separate program (the already-validated SFPU tail).

That also retires the storage question: no 4× fold blowup on either operand (folding the DB would have
been 3.96 GB streamed/query ≈ 81 ms; folding the query, 1.57 GB of fold work per query). The DB stays at
its natural 991 MB → ~20 ms, bandwidth-bound.

Kernel structure: 16 limb-pair matmuls accumulate into **7 int32 DEST planes** (p+s ∈ 0..6, fits the
8-tile budget with `fp32_dest_acc_en`), then `Σ_d 128^d·plane_d mod q`.

**One correctness note carried forward:** `reader_scan_gemm.cpp` shares `B` only across output tiles of the
SAME slot — the fix relative to `reader_scan_mm.cpp`, which shares one `B` across *all* outputs. Callers
must launch per slot or pass a per-slot `B` base.

### Build order from here

1. ~~A/B layout~~ — done (limb tiles `[mt][kt][limb]` / `[kt][limb]`, validated).
2. ~~Per-slot `B`~~ — done (`reader_scan_gemm.cpp`).
3. ~~Driver~~ — done (`scan_gemm.cpp`, bit-exact at m=128/N0=512).
4. ~~Batch the slot axis~~ — done. Output tiles enumerate `(slot, m-tile)` as `gidx = slot*MTS + mt`, so
   `A[slot][mt][kt][limb]` indexes as `(gidx*KB + kt)*NL + l` and only `B` needs the per-slot stride
   (reader arg 5 = `MTS`). The **compute kernel needed no change** — it just processes `n_out` output
   tiles. Validated with INDEPENDENT random operands per slot (so slot-indexing bugs cannot pass by
   coincidence): m=32/N0=64 slots=1 **0/64** (regression), slots=4 **0/256**, and at the real cell shape
   m=128/N0=512 slots=4 **0/1024**.
5. ~~Shard across cores~~ — done, launch geometry only (kernel unchanged). At m=128/N0=512, 512 slots,
   bit-exact **0/131072** at every core count:

   | cores | time | A-stream | useful modmul |
   |--:|--:|--:|--:|
   | 1 | 33.04 ms | 4.06 GB/s | 2.03 G/s |
   | 8 | 10.20 ms | 13.16 GB/s | 6.58 G/s |
   | 32 | 7.88 ms | 17.03 GB/s | 8.51 G/s |

   (A-stream is below `test_scan_mc`'s 48.61 GB/s because this kernel does 16 limb-pair matmuls per
   4-tile A block — 4× the MACs per byte — so it is compute-bound, not bandwidth-bound. Also: a small
   workload measures only fixed overhead; at 32 slots the 1→8 core speedup was a meaningless 1.2×.)

6. ~~Query batching on the n axis~~ — **CONFIRMED, and it is free.** Same 512-slot workload, 32 cores,
   varying the n-axis fill:

   | ncomp | time | useful modmul | correctness |
   |--:|--:|--:|---|
   | 2 | 7.948 ms | 8.44 G/s | 0/131072 |
   | 8 | 8.035 ms | 33.41 G/s | 0/524288 |
   | **32** | **7.988 ms** | **134.42 G/s** | **0/2097152** |

   Time is **identical** while useful work scales **16×** — the n axis was pure padding. A-stream holds at
   ~16.8 GB/s, confirming the DB streams once per batch however many queries share it (`A` is
   query-independent). At `n=32` = 2 components × **Q=16 queries**.

   **Full-cell projection from measurement** (2048 slots × 3 channels = 12× the measured work):
   **95.9 ms per batch of 16 → 6.0 ms/query**, with compute (~96 ms) and DB streaming (1.6 GB padded
   capacity at 16.8 GB/s ≈ 95 ms) balanced.

   **CORRECTION — there is NO core headroom, and the reason is actionable.** 64 cores measured *slower*
   than 32 (8.495 vs 8.140 ms, 126.4 vs 131.9 G modmul/s): the kernel saturates by ~32 cores. Diagnosis:
   `reader_scan_gemm.cpp` re-reads `KB·NL = 64` B tiles **per output tile**, so at 2048 output tiles that
   is **128 MB of B traffic** on top of 134 MB of A — ~262 MB in 8.14 ms ≈ **32 GB/s**, close to the
   48.61 GB/s ceiling. So it is bandwidth-bound on self-inflicted traffic, not on the DB.

   **Fix applied: `B` L1-resident per slot.** The naive restructure (`for slot: for kt: for mt`) would need
   `MTS·NPLANE = 28` int32 DEST tiles and only 8 exist, so instead the reader loads a slot's B ONCE
   (`KB·NL = 64` int8 tiles = 64 KB, `cb_b` depth sized to `KB*NL`) and the compute kernel waits on it
   once, indexes it as `kt*NL+s`, and pops only after that slot's `MTS` output tiles.

   Result at 32 cores: **131.9 → 147.8 G modmul/s (+12%)**, bit-exact 0/2097152. Full-cell scan
   → **5.45 ms/query** at Q=16.

   **But it did NOT restore 64-core scaling** (8.782 ms / 122.3 G/s, still worse than 32). So the earlier
   "bandwidth-bound on self-inflicted traffic" diagnosis was only partly right: B traffic was a real ~12%,
   but the 32-core plateau is a **COMPUTE ceiling**. At 32 cores this kernel sustains 17.2 G int8 MACs in
   7.265 ms = **2.36 T MAC/s aggregate**, *above* `test_scan_mc`'s 1.556 T, while DRAM traffic is only
   ~166 MB / 7.265 ms ≈ 22.8 GB/s — less than half the 48.61 GB/s ceiling. Further scan gains need fewer
   MACs (the padding is already gone at n=32), not more cores or less traffic.
7. **On-device recombine** — NOT a parameter change to `recombine_rns`. That kernel computes
   `(C0 + C1<<7 + C2<<14 + C3<<21) mod p_j`: 4 planes, weights as bit SHIFTS, collapsed into one
   Montgomery reduction with a bound assumption (`C0+(C1<<7) < 1.61·p_j`). The 7-plane form needs
   `128^0…128^6` and `128^6 = 2^42` exceeds 32 bits, so it cannot be a shift — it needs a real modmul by
   `128^d mod q`. Different algorithm, not a different constant. Two routes:

   **(a) RECOMMENDED — reuse validated kernels, add none.** Per `d`, one `modmul_compute` dispatch over
   ALL output tiles against a constant tile of `128^d mod q`, then a strided `modadd_reduce` summing the
   7 planes per output position (the pattern `mac_ntt` already uses). ~8 dispatches for the whole
   recombine, since both kernels take a tile count. DB stays 1× storage.

   **(b) Fold the radix powers into one operand** → 4 planes with shift weights → `recombine_rns`
   unchanged. But folding is 4× on the folded operand: on the DB that is 6.4 GB streamed per batch, which
   by the §5/§6 measurements turns a ~95 ms compute-bound batch into ~380 ms bandwidth-bound —
   **6 ms/query → 24 ms/query**. On the query side it is ~393 MB of host fold work per 16-query batch.

   → take (a). Risk to watch: it is buffer/stride plumbing, and an off-by-one silently corrupts DB
   entries, so gate it against the host recombine that currently passes.
8. Wire into the answer pipeline, replacing the pointwise MAC, then scale to 512×2⁷.

### Status of the 48.61 GB/s figure

It remains a sound **bandwidth** result for streaming 128 MiB of int8 tiles, and DB streaming is the
dominant term, so it stays a good predictor for the batched scan. But its `GMAC/s` component was measured
with `B` shared across all outputs (the constant-poly dataflow), so it is **not yet** a measurement of the
packed-DB GEMM. The batched form above is what turns it into one.

## 4e. Step 8 — QUERY EXPANSION measured: dispatch reduction alone will NOT close it

On the CPU reference implementation expansion is **~52% of server time** (vs 34% for the first dimension),
so it — not the scan — sets the
end-to-end figure. First measurement on device (`test_query_expand2048`, instrumented, K=3, ℓ=8,
m_exp=4 → 16 selectors):

```
EXPAND (Subs tree, 4 levels -> 16 selectors): 7696.1 ms
  workload ENQ    5586.6 ms  82.0%  11550 calls (0.484 ms/call)
  program BUILD    951.1 ms  14.0%  11550 calls
  buffer READ      248.9 ms   3.7%   3180 calls
  buffer WRITE      25.9 ms   0.4%
selector mismatches 0/196608 · unpack 0/98304 · PASS
```

**Same signature as the pre-optimization answer pipeline** (82% enqueue = fresh-program serialization),
so the workload cache + trace machinery of §3/§4b transfers directly: 7.7 s → ~1.4 s for 16 selectors.

**But that is not enough, and this is the key finding.** The 128 MB cell needs 512 selectors (m_exp=9).
The Subs tree has `Σ_a 2^a` nodes, so it goes **15 → 511 key-switches (34×)**, ~393k dispatches. Even
fully traced at 0.002 ms/dispatch that is ~786 ms of enqueue alone, and an optimized total in the **tens of
seconds**.

**The structural problem is different from the answer pipeline's.** 511 key-switches × 24 NTTs each ≈
**12,000 n=2048 transforms**, and the engine runs each as a **single-tile, single-core** four-step NTT. The
scan got fast by using 32 cores and the matrix engine; expansion is still one core, one tile at a time.

### MEASURED: co-scheduling beats trace by 12× on independent work — and that rescues expansion

`test_dispatch_micro 32 20` (single-tile modmul, the engine's most-dispatched shape), all bit-exact:

| regime | ms/iter | ms/dispatch | vs A |
|---|--:|--:|--:|
| A build-fresh + enqueue | 21.204 | 0.6626 | 1.00× |
| B persistent re-enqueue | 1.017 | 0.0318 | 20.8× |
| C traced replay | 1.003 | 0.0313 | 21.1× |
| **D co-scheduled — ONE program, 32 cores** | **0.082** | **0.0025** | **260×** |

Correctness: co-scheduled **0/65536** (every core correct).

**Trace does NOT parallelise independent work.** It removes per-dispatch host cost but still issues 32
sequential dispatches (32 × 0.031 ≈ 1.0 ms). One program with disjoint core ranges runs all 32 at once
(0.082 ms) — **12.2× better than trace**. Separate programs serialize on the single command queue no
matter which core they target.

**Revised expansion outlook.** 511 key-switches × ~100 dispatches ≈ 51k dispatches:
traced-but-serial → 51k × 0.031 = **1.6 s**; co-scheduled → **~128 ms**. So the earlier
"tens of seconds even optimized" estimate was too pessimistic — it treated trace as the ceiling and missed
that the Subs tree's independence makes co-scheduling applicable.

### Cache + trace applied to expansion: 23× measured (7998 → 347.8 ms)

`test_query_expand2048` now runs the Subs tree through the same record → capture → replay ladder:

| | ms | note |
|---|--:|---|
| EXPAND record | 7998.5 | 11,550 workloads, 6,480 pool buffers |
| capture | — | 3,270 traces |
| **EXPAND replay** | **347.8** | **23.0×**, spread 0.9% |

Gates: expand 0/196608 · replay-vs-record 0/196608 · unpack 0/98304.

At 347.8 ms, **blocking readback is 90.9%** (274.3 ms over 3,180 calls) — the same inversion the answer
pipeline showed once dispatch cost was removed.

> **Trap found here: the workload cache is keyed by dispatch ORDER, not op identity.** Leaving
> `CACHE_ON`/`TR_PHASE=2` set after the expand made the *unpack* section replay the **expand's** cached
> workloads — observed as unpack 98304/98304. Scope cache/replay to exactly the recorded region
> (`eng.CACHE_ON = false; eng.TR_PHASE = 0;` after it).

### Residency applied to the key-switch: 347.8 → 178.1 ms (1.95×)

`keyswitch()` restructured exactly like the answer pipeline's resident extproduct — reuse, not new
machinery: digit transforms go straight into packed DRAM buffers via `fwd2048_buf2`, key material is
transformed once into resident buffers (`KnbL`, `to_mont` on the coefficient-domain input), and
`mac_ntt_buf → transpose_dev → inv2048_buf2` stays in DRAM with ONE readback per (col, channel).

| | cache+trace | + residency |
|---|--:|--:|
| EXPAND replay | 347.8 ms | **178.1 ms** |
| blocking readbacks | 3,180 | **660** (−79%) |
| traces | 3,270 | 1,440 |
| pool buffers | 6,480 | 3,600 |
| vs record | 23.0× | **39.1×** |

Gates: expand 0/196608 · replay-vs-record 0/196608 · unpack 0/98304 · spread 0.5%.

### `digit_rns_batch` re-measured here: 178.1 → 165.5 ms (−7%) — the opposite of the answer path

Same method that measured ~3% WORSE in the answer pipeline is ~7% BETTER in expansion, because expansion's
key-switch calls it `ℓ=8` times per node (120 calls here vs 48 there). **Co-scheduling pays in proportion
to how much independence the workload actually offers** — which is why the answer-path negative result was
worth recording but not a reason to discard the method. Measure per stage, per workload.
(First reading here was 186.5 ms at 27.9% spread — noise. Clean: 165.5 ms at 0.7%.)

### Key-switch `c0 + mac` moved to host: 165.5 → 156.6 ms (−5.4%)

For `col==0` the key-switch added `ct[ch] + mac` via `eng.padd()` — a device program plus two blocking
readbacks to add 2048 pairs of uint32. Pure overhead once readback dominates; done on host instead.
Readbacks **450 → 270 (−40%)**, traces 1230 → 1050, gates unchanged.

**Cumulative expansion: 7998 → 156.6 ms (51.1×).**

### Where the remaining 156.6 ms actually is — and why co-scheduling is the ONLY lever left

Readback is **96%** (136.5 ms / 270 calls) but at **0.506 ms/call**, up from 0.086 ms when there were 3,180
calls. That rise is **not** overhead — it is real device compute waiting at each blocking drain, now that
there is nothing else to hide it behind. With the transforms running on a **single core**, that device work
is serialized.

So the residual is *unparallelized device work*, not host overhead. Further host-boundary removal has hit
diminishing returns; **stage/node co-scheduling is the remaining lever**, and it attacks exactly this:
- stage-wise within a key-switch: 24 independent `fwd2048_buf2` calls → 10 programs instead of 240
- across nodes: level `a` has `2^a` independent key-switches → up to ~32× at deep levels

### Stage-wise co-scheduled transforms: 156.6 → 79.3 ms (1.97×)

`Engine::fwd2048_buf2_batch` runs `n` independent forward transforms in **10 programs instead of 10n**: each
of the four stages (fusedA ×2, modadd, twiddle-mmul, fusedC) becomes ONE program over `n` cores, per bk2.
Used in the key-switch for a channel's `ell` digit transforms. Compile args come from instance 0 — valid
because every buffer is `mkbuf()`-allocated with identical DRAM-interleaved config. Sequential fallback
when `n` exceeds the grid.

| | before | after |
|---|--:|--:|
| EXPAND replay | 156.6 ms | **79.3 ms** |
| cached workloads | 10,260 | **3,960** (−61%) |
| traces | 1,050 | 420 |
| record pass | 6,972 ms | 2,672 ms |
| readback per call | 0.506 ms | **0.227 ms** |

Gates unchanged (0/196608 · 0/196608 · 0/98304).

### A/B settling the figure (and a false alarm worth recording)

A run at 15 iterations initially showed **92.5% spread** (min 77.6 / max 149.4), which looked like
co-scheduling had destabilised the distribution — plausible mechanism: each program now barriers 8 cores, so
one straggler sets the pace. **That was a transient, not a property of the optimization.** Settled by A/B on
the same binary (`PIR_NOBATCH=1` forces the sequential path), 15 iterations each:

| | min | median | mean | max | spread |
|---|--:|--:|--:|--:|--:|
| **co-scheduled** | 75.6 | **77.6** | 77.3 | 78.3 | **3.6%** |
| sequential | 155.1 | 157.2 | 157.8 | 171.6 | 10.6% |

Co-scheduling is **2.03× on the median** *and* **more stable**. Lessons: (1) quote median/mean, not min —
the harness now prints min/median/mean/max, as the answer test already did; (2) an A/B switch on one binary
beats reasoning about a suspected mechanism; (3) a single noisy run is not evidence of a regression.

**This confirms the diagnosis above:** readback call COUNT was unchanged (270) while per-call time HALVED —
so that time was serialized device compute, exactly as claimed, and parallelizing the transforms recovered
it. **Cumulative expansion: 7998 → 79.3 ms (100.9×).**

### Widening the batch 8 → 24: only 3.5% — width is SATURATED, and that redirects the plan

The fold/twiddle buffer addresses and modulus params are **runtime** args, so one program can mix RNS
channels (each core gets its own). That widens a key-switch's transform batch from `ell`=8 to `ell*K`=24:

| | 8-wide | 24-wide |
|---|--:|--:|
| EXPAND replay (median) | 77.6 ms | **74.9 ms** (−3.5%) |
| cached workloads | 3,960 | 3,360 |
| traces | 420 | 360 |

**3× the width bought 3.5%** — so width is no longer the constraint. Where the 74.9 ms actually goes:
**270 blocking readbacks × 0.22 ms = 59.5 ms**, i.e. host boundaries dominate again.

**Why those readbacks exist is the actionable part:** `automorph`, `monomul` and `ctadd` are **host**
operations, so every key-switch output must return to host before the next Subs level. Moving them
on-device (the §4/§6 "on-device automorph" item, long listed as a loose end) would let the whole Subs tree
stay resident — it is now the **binding constraint**, not tidiness.

**This also revises an earlier judgement in this document.** §4 argued fusion was the wrong lever and
caching/trace the right one. That was correct *at the time* (cost was fresh-program serialization). Now that
dispatch cost is ~2% and width is saturated, the levers are (a) removing the remaining host boundaries and
(b) fewer sequential stages per transform — so **fusion becomes relevant after all**. Levers are
stage-dependent; re-measure rather than inheriting a previous conclusion.

**Node-level co-scheduling is therefore DEPRIORITISED**: it adds width, and width is saturated. Do (a) first.

Full width: 74.9 × 34 ≈ **2.5 s**.

Readback is still 94% of what remains (152.7 ms / 660 calls), but per-call has risen to 0.231 ms — that is
device compute becoming visible at the blocking drains, not new overhead. Remaining host boundaries are
`crt_and_gadget`, `digit_rns` and `padd`, which still return host vectors.

**Full-width outlook, honestly.** Scaling 178.1 ms by the 34× node count → **~6.1 s**. Both remaining
levers are needed:
1. **Residency** — readback is 91% of what is left; removing per-op host round-trips is the bigger factor.
2. **Node co-scheduling** — level `a` has `2^a` independent key-switches; deep levels (a≥5) can fill the
   grid, worth up to ~32× where applicable. But see the `digit_rns_batch` negative result above:
   co-scheduling pays in proportion to how many independent ops of that shape exist, so measure per stage.

**Lever for expansion = parallelism + batching of the NTTs**, not dispatch cost:
- multi-core NTT (the `nntt2048` analogue of `ntt1024_dev_mc`), and
- batching independent transforms into single multi-tile programs.

The Subs tree suits this exactly: level `a` has `2^a` **independent** nodes, so deep levels (a≥5 → ≥32
nodes) saturate the grid; only the thin early levels underutilize. This is what `ONION_MATCH_PLAN.md §5`
anticipated and what has never been built. Do NOT expect trace replay alone to fix expansion.

## 4f. PLAN TO A REAL END-TO-END RESULT (no projections)

**Target.** The p150 runs OnionPIRv2's *same* workload — `CONFIG_N2048_K1`-equivalent, 128 MB DB,
512×2⁷ hypercube, full query expand → unpack → scan → fold → mod-switch — **entirely on device**, with
high core utilisation, and every headline figure **measured, not extrapolated**. A number is only reportable
if it came out of a run.

**Rules this plan holds itself to.** (1) No projection in any headline; scaled estimates must be labelled as
such. (2) Every step gated bit-exact (small hypercube) or by decrypt-check (at scale). (3) Report
median/mean, never min alone. (4) Re-measure per stage — levers proved stage-dependent (§4e).

### Phase A — resident Subs tree — ✅ COMPLETE: 7998 -> 53.0 ms (151x), ZERO host boundaries in the tree
Status 2026-07-31: A1, A2, A3, A4 built and validated; A5 (buffer-valued ciphertext) is the remaining wiring.
Measured after A1-A3: expansion **74.9 -> 73.0 ms median**, readbacks **270 -> 210 (-22%)**, traces 360 -> 300,
gates still 0/196608 (selectors), 0/98304 (unpack), 0/196608 (replay-vs-record).
The 210 remaining = 7 per key-switch x 30 key-switches (the Subs tree calls `subs` twice per node):
**6 are `read_inv_out2`, 1 is the digit readback** — both removed by A5.
Validated separately (`test_permute_dev`, all 0/2048): automorph 6 indices, monomul 11 shifts, modadd_ab.

After A4 (`c0 + mac` on device via `padd_ab`): expansion **73.0 -> 64.4 ms median** (-12%), spread 2.3%,
gates 0/196608 + 0/98304 + 0/196608. **Cumulative expansion: 7998 -> 64.4 ms (124x).**
Full width: 64.4 x 34 ~ 2.19 s.

> **LAYOUT TRAP found doing A4 — cost one broken run (selectors 97896/196608).** `inv2048_buf2`'s output
> lives in the inverse-transform BLOCK layout, `blk[n1*T+n2loc] = x[n1 + 32*(bn2*T+n2loc)]`, **not** natural
> tilized order. Adding a `til2()`-packed operand to it combines mismatched coefficient positions — and it
> fails silently-plausibly (deterministic, replay-vs-record still 0, only the golden catches it). Any
> elementwise op mixing an inv output with a natural-order poly must pack via `write_inv_layout()`. General
> rule: **each buffer carries a layout; elementwise ops require both operands in the SAME one.** The fwd
> path has the analogous constraint (`write_fwd_in2`), and `transpose_dev` is what bridges fwd->inv.

### A5 DONE — the Subs tree is FULLY ON DEVICE

Ciphertexts are `vector<MeshBuf>` (2K polys, natural tilized) all the way through: `automorph`/`monomul` are
`permute_batch`, `ctadd` is `padd_ab`, `keyswitch` is buffer-in/buffer-out, and the inv->natural reindex is
absorbed into a permutation table (`perm_table_inv_to_natural`) so it costs one program instead of a readback.

| | before A5 | after A5 |
|---|--:|--:|
| median | 64.4 ms | **64.1 ms** |
| blocking readbacks | 210 | **126** (-40%) |
| trace segments | 300 | **121** (-60%) |
| host arithmetic in the tree | automorph, monomul, ctadd | **none** |

Gates: selectors 0/196608 · unpack 0/98304 · replay-vs-record 0/196608. Spread 4.5%.
Remaining 126 readbacks = 96 final selector reads (inherent — that is the result) + 30 digit reads.
**Cumulative expansion: 7998 -> 64.1 ms (125x).**

> **The permute kernel is a SCALAR single-RISC loop (2048 iterations), so it must be co-scheduled.** Run one
> at a time, A5 was a net LOSS: 64.4 -> 78.3 ms, because it added ~720 scalar permutes to save 84 readbacks.
> `permute_batch` (2K permutes per program, per-instance q so a batch may mix channels) recovered it to 64.1.
> Same lesson as §4e in a new place: on-device is not automatically faster — a scalar device loop can lose to
> a fast host, and the win comes from parallelism plus removed host boundaries together.

A1. ✅ `crtK_buf` — residue **buffers** in, packed limb buffer `bX` out, no readback. *(Layout note: `bX` is
    tile-major/limb-minor, tile index `t*K+limb`, which is exactly the order `reader_gadget` pushes, so the
    crt→gadget hand-off needs no new kernel — the §10b "interleave gotcha" applies to a different path.)*
A2. ✅ `gadgetK_buf` — `bX` in via a sequential `reader_i32` (`n = 2K`), digit buffer `bdig` out, no readback.
A3. ✅ `digit_rns_from_bdig` — digit `k` lives at tiles `k` and `ell+k`, i.e. stride `ell`, offset `k` →
    `reader_i32_stride`. Co-scheduled across `ell` (one program).
A4. ✅ `padd_ab` (`compute/modadd_ab.cpp`, validated 0/2048) — two-buffer mod-add for `c0+mac` and `ctadd` (pack operands into one 4-tile buffer + strided
    `modadd_reduce`, the `mac_ntt` pattern).
A5. ✅ Buffer-valued ciphertext through the tree: `keyswitch` returns buffers (drop `read_inv_out2`),
    `permute_dev` (done, validated) + A4 consume buffers. Read back **once**, after the last level.
    Gate: `test_query_expand2048` selectors 0/196608, unpack 0/98304. Then measure.

### Phase A final: 53.0 ms median, spread 1.5% — the tree is entirely on device

Removing the last host boundary (the digit readback + host `write_fwd_in2`) by running the fwd-input reshape
as a device permutation out of the resident digit buffer:

| | |
|---|--:|
| expansion | **53.0 ms median** (spread 1.5%), 151x from 7998 |
| host boundaries inside the tree | **zero** |
| blocking readbacks | 96 (only the final selector fetch, which is validation scaffolding) |
| trace segments | 3,270 -> **91** |

On device now: Garner CRT, gadget decompose, digit->RNS, BOTH layout reshapes, fwd/inv transforms, MAC,
transposes, automorph, monomial multiply, ct-add. Gates: 0/196608 · 0/98304 · 0/196608.
Full width: 53.0 x 34 ~ **1.8 s**.

> **TIMING TRAP — caught in the act.** Once a stage has no blocking readback left, the timed region measures
> **host enqueue only**. It read "min 1.3 ms, spread 3976%" while the real work surfaced in a later fetch at
> 4.6 ms/readback. `Engine::finish()` (queue drain) inside the timed region fixes it. Without that, removing
> the last blocking op looks like a 40x win that does not exist. Any future stage that becomes fully async
> needs the same treatment.

### Where expansion's 60.8 ms actually is (measured, after separating test scaffolding)

Reading the 16 selectors back is **validation scaffolding** — in the real pipeline they feed unpack/answer on
device. Timing it separately: median **64.1 -> 60.8 ms**, i.e. those 96 readbacks cost only ~3.3 ms total
(~0.03 ms each: the data is already computed and the device is idle).

**So a prior attribution here was wrong.** "Readback is 96% at 0.443 ms/call" invited the conclusion that all
126 readbacks were expensive. They are not: the average is dominated by the **30 in-tree digit readbacks at
~1.9 ms each**, because each blocks on that key-switch's accumulated device work. Fetches after the work is
done are nearly free. **Attribute per call-site, not by average.**

**The last host boundary in expansion is therefore one specific thing:** `digit_rns_from_bdig` reads back so
`write_fwd_in2` can do the forward-transform input reshape **on host**. That reshape
(`blk[b*T+n1] = x[n1 + 32*(bn2*T+b)]`) is a FIXED PERMUTATION, so it goes on device exactly as the
inv->natural reindex did (§A5). Needed:
1. `perm_table_natural_to_fwdin()` — same construction as `perm_table_inv_to_natural`.
2. `digit_rns_from_bdig` returning its buffer instead of reading back.
3. Input tile offset+stride args on `permute_signed.cpp`, because residue `(k, ch)` sits at tiles
   `k*2K + ch` and `k*2K + K + ch` (stride K) inside the digit_rns output buffer.
Then the Subs tree has **zero** host boundaries between levels, and ~57 of the 60.8 ms is addressable.

### Phase B1 (permute half): 53.0 -> 51.8 ms (-2.3%)

The `2^a` nodes of a level are independent, so their permutes batch across nodes: three batched permutes per
level (automorph of c, monomial shift of c, automorph of c_sh) replace `3*2^a` per-node calls. Median
**51.8 ms**, spread 0.7%, gates unchanged. **Cumulative expansion: 7998 -> 51.8 ms (154x).**
Full width: 51.8 x 34 ~ **1.76 s**.

Only -2.3%, again consistent with §4e: **width is saturated**. Workloads fell just 3885 -> 3852, because
`permute_ct` already batched 6-wide; batching across nodes changes program count, not total permute work.

**Consequence for the rest of Phase B.** With zero host boundaries and `finish()` draining the queue, the
51.8 ms is now essentially all DEVICE COMPUTE. So B1's larger half (merging every key-switch stage across
nodes) buys width that measurement says is already saturated. The remaining levers are:
- **B3 fusion** — fewer sequential stages per transform (depth, not width). §4e already rehabilitated this.
- reducing arithmetic: the scalar permute kernel is 2048 scalar RISC iterations; a vectorised or
  matmul-expressed permute would cut real work rather than re-spreading it.
Measure per stage before choosing (the `digit_rns_batch` and this result are both cases where the obvious
lever returned almost nothing).

### Phase B — core utilisation
B1. Batch each Subs level's `2^a` nodes into the co-scheduled stages (`fwd2048_buf2_batch` already takes
    per-instance channels, so it extends to per-instance nodes). Deep levels (a≥5) fill the grid.
B2. Report **achieved core count and utilisation per stage**, not just wall time. §4e showed width
    saturates, so verify utilisation rather than assuming it.
B3. Only if width is genuinely saturated: fuse the NTT's sequential stages (fewer, bigger programs). §4e
    rehabilitated fusion — depth, not width, may bound it.

### REAL 1024-cell retrieval (largest so far) — and it is the case C2 needs

`dims=[32,2,2,2,2,2]`, 1024 cells, N0=32, rest=32, K=3, `--no-golden`, validated by decrypt-check only:

```
RECORD:  50.7 s   (64,736 workloads, 26,112 pool buffers; scan 28.6 s = 56% of it)
CAPTURE: 12,913 traces
REPLAY:  1606 ms min / 1623 mean  (spread 3.1%)
DECRYPT-CHECK  decrypt(device answer) vs DB[700]: 0/2048   coeff0 got=192 expected=192
```

Linear in cells, as the pointwise scan implies: 32 cells 112 ms -> 512 cells 846 ms -> **1024 cells 1606 ms**.
`N0` and `rest` are both multiples of 32, so this is the smallest GEMM-addressable shape (§4f C2).

**This quantifies why the GEMM is MANDATORY, not an optimization.** 1024 cells costs **64,736 workloads**; the
128 MB cell is 65,536 cells, so the pointwise path would need ~4M workloads — far beyond the 256 MB trace
region, with a record pass measured in hours. Pool buffers (26,112 at 1024 cells) scale the same way, which
is also why the DB must be one large tile buffer rather than one buffer per cell (§4f D2).

### Phase C — scan into the answer pipeline
C1. ✅ **DONE — on-device recombine, validated 0/4096 (`test_permute_dev`). NO NEW COMPUTE KERNEL.** `recombine_bigK` handles 4 planes with
    SHIFT weights; beyond d=3 the weights stop being shifts (`128^4 = 2^28` vs a 28-bit q). Decompose:
    `R = Σ_{d=0..3} 128^d·plane_d + 128^4 · Σ_{e=0..2} 128^e·plane_{e+4}` — **two `recombine_bigK` calls**
    (second with its 4th plane zero) **+ one `modmul_compute`** by `mont(128^4 mod q)` **+ one `padd_ab`**.
    Both groups are 4 CONTIGUOUS tiles if `scan_gemm` packs **8** planes (7 real + 1 zero) instead of 7 —
    a one-line change to `NPLANE` and the pack loop. Supersedes the earlier route (a) sketch below, which
    needed 7 strided modmuls.
C1-alt (superseded). On-device recombine, route (a) of §4d — per `d`, one `modmul_compute` over all output tiles against a
    `128^d mod q` constant tile, then strided `modadd_reduce`. ~8 dispatches, no new kernel.
C2. Replace the answer pipeline's pointwise MAC with `scan_gemm` + C1. Gate: answer 0/12288 + decrypt.
    **BLOCKED ON CASE SHAPE, not on code.** The GEMM needs `m = rest` and `k = N0` to be multiples of 32, but
    the 32-cell case has `rest=4, N0=8` — the tiling cannot address it. Folds are binary (`mux` halves `acc`),
    so `rest = 2^(ndims-1)`: the smallest GEMM-valid shape is **`dims=[32,2,2,2,2,2]` = 1024 cells**
    (N0=32 -> k=1 tile, rest=32 -> m=1 tile, 2048 slots x 3 channels of batch). Generated with `--no-golden`
    (~20-25 min, 5 folds of `rlwe_to_rgsw`); validation is the decrypt-check. This is why C2 and D are coupled
    — both need a case bigger than the bit-exact one, and the small case stays as the primary correctness gate.

    **ARCHITECTURAL CONSTRAINT — C2 needs an OUTPUT TRANSPOSE stage, it is not just wiring.**
    The batched-per-slot GEMM produces **slot-major** output: for slot `s` the output tile holds `[r][c]`, so
    element `[r][c]` of tile `s` is **coefficient s** of the eval-domain poly `(r,c)`. But the inverse
    transform consumes a poly across slots (**slot-minor**). Assembling poly `(r,c)` therefore gathers one
    element from each of 2048 slot-tiles: a transpose of `32(r) x 2(c) x 2048(slots)` per RNS channel.
    This cannot be re-tiled away — putting slots on the tile row axis makes the selector operand depend on a
    row index of `A`, which is precisely the diagonal problem that forced slots onto the BATCH axis (§4d).
    So the scan path is: GEMM -> recombine -> **transpose/gather** -> inv2048 per (r, comp, ch).
    **RESOLVED — the transpose is cheap, and needs only existing kernels + one generalization.** View the
    scan output as `M[slot][r]` (2048x32); the inverse transform needs `M^T[r][slot]`. Tiled, `M` is 64
    row-blocks x 1 col-block of 32x32 tiles and `M^T` is 1x64 — so the transpose is **an in-tile transpose
    plus a RELABEL**, with no inter-tile data movement: the tile at block `(SB,0)` *is* block `(0,SB)` of the
    result. `transpose_dev` (validated) does the in-tile part.
    What remains is that poly `(r,c)` is **row r across those 64 tiles** — a row-gather at 32-element
    granularity — and that **folds into the inv-input layout permutation**, which is already table-driven
    (`perm_table_natural_to_fwdin` is the analogous case). So the scan tail is:
      GEMM -> `scan_recombine` -> 64x `transpose_dev` (relabel is free) -> one table-driven permute per
      (r, c, ch) that composes the row-gather with the inv-input layout -> `inv2048_buf2`.
    ✅ **DONE — `permute_signed` now takes separate in/out tile counts** (runtime arg 7 = `n_in`, 0 => same as
    `n_out`; CB depths sized accordingly). So a single permute can read 64 tiles and write 2, with the table
    sized `n_out*1024` and entries indexing the `n_in*1024` input space. All existing gates re-validated after
    the change: `test_permute_dev` total 0, expansion 0/196608 + 0/98304 + 0/196608 (51.7 ms, 0.0% spread).
    Watch L1 when using it wide: 64 input tiles = 256 KB, so either raise the CB or gather in two passes of 32.

    ✅ **`Engine::scan_gemm` added and the whole scan tail validated through the engine path:
    `scan_gemm -> scan_recombine` vs a host modular GEMM = 0/256** (`test_permute_dev`). The launcher is the
    validated inline version from `test_scan_gemm.cpp` lifted into the engine (sharded, B L1-resident, exact
    core split), so the answer pipeline can call it directly.

    **So every primitive C2 needs now exists and is validated.** What is left for C2 is assembly: build the
    A/B limb layouts from the eval-domain DB and selectors, run the GEMM, `scan_recombine`, the 64
    `transpose_dev` + composed-table permute, then `inv2048_buf2`. Gated by the 1024-cell case
    (`dims=[32,2,2,2,2,2]`, `--no-golden`) via decrypt-check, with the 32-cell bit-exact case kept as the
    primary correctness signal.

    Implementation once the case exists: build A as `[slot][mt][kt][limb]` int8 from the eval-domain DB and B
    as `[slot][kt][limb]` with components on the n axis, run `scan_gemm` (sharded, B L1-resident),
    `scan_recombine` for the mod-q tail, then `inv2048_buf2` per (r, comp, ch) to return to the coefficient
    domain. Query batching (C3) then fills the n axis at Q=16, measured free in §4d.
C3. Query batching on the n axis (Q=16 fills 32 columns; measured free in §4d).

### C2 ✅ DONE — GEMM scan wired into the answer pipeline and validated at 1024 cells

`USE_GEMM` gate: the GEMM needs `rest % 32 == 0 && N0 % 32 == 0`, else the pointwise MAC runs — so the
32-cell bit-exact case still exercises the original path as the primary correctness gate (re-verified
**0/12288** + decrypt 0/2048 at 104.6 ms).

At `dims=[32,2,2,2,2,2]` (1024 cells, N0=32, rest=32), decrypt-check **0/2048**, coeff0 192 == 192:

| | pointwise | **GEMM** |
|---|--:|--:|
| replay | 1606.3 ms | **1354.7 ms** (1.19x) |
| cached workloads | 64,736 | **58,223** |
| pool buffers | 26,112 | **6,936 (-73%)** |
| trace segments | 12,913 | **6,391 (-50%)** |

Latency gain is modest at this size (MT=KB=1 -> one tile-pair per output tile), but the **structural** result
is what matters: the scan's program count is now essentially **independent of cell count** (a few sharded
GEMM programs regardless of DB size), where the pointwise MAC grew linearly. That is what removes the
~4M-workload wall that made 128 MB impossible.

> **CORRECTNESS TRAP avoided by construction, worth stating:** `pmul` is a **Montgomery** modmul but the GEMM
> produces a **plain** modular product, so the GEMM path must NOT pre-`to_mont` the DB. Doing so would make
> every product off by a factor of R — and it would still decrypt to *something*, so only the bit-exact gate
> would catch it. The DB transform is therefore conditioned on `USE_GEMM`.

Remaining in the scan tail (optimizations, not correctness): the slot->poly gather is currently on host
(untilize per output tile); the on-device form is 64x `transpose_dev` + a composed-table permute (§4f C2), and
the per-cell DB forward transforms are offline precompute in a real deployment (they dominate the 58,223
workloads here).

### Phase D — the 128 MB workload
D1. Self-checking DB built in C++ (`DB[k] = k`), so validation needs no reference data at all.
D2. Cache the NTT'd DB to a file — deterministic and query-independent, ~1 GB, write-once.
D3. Chain expand → unpack → scan → fold → mod-switch in one resident pipeline; move mod-switch finalize
    on-device (last host arithmetic).
D4. Run at 512×2⁷ / 128 MB. Validate by decrypt == index. **This is the first real end-to-end number.**

#### Phase D blockers, identified 2026-07-31 (D is NOT just assembly)

**D-a. The case-file route does not scale.** A 128 MB case is `dims=[512,2,2,2,2,2,2,2]` -> 65,536 cells.
The DB alone would be `65,536 x 3 x 2048 x 4 B = 1.6 GB` in the case file, and Python would have to generate
400M values. So the generator needs a **query-only case format** (keys + selectors + RGSW ladders + secret +
the target index, no DB) and the device side must synthesise the DB itself. That is exactly what D1's
self-checking DB (`DB[k] = k`) enables — with slot k holding its own index, the DB needs no transfer at all
and the decrypt-check compares against the index.

**D-b. The DB forward transform cannot run per-cell on device.** 65,536 cells x 3 channels x ~10 programs per
`fwd2048` is **~2M programs** — far past the 256 MB trace region and the workload cache. Two viable routes:
  1. **Host-side C++ NTT** for the offline precompute, writing the eval-domain DB to a cached file (D2).
     The repo has no host C++ NTT today (device-side only; Python has one) — this is the main new code D needs.
  2. Batch the DB transforms with `fwd2048_buf2_batch` (24-wide+) and stream in chunks, keeping the workload
     cache bounded by processing the DB in blocks rather than all at once.
  Route 1 matches how a real server works (the DB is static, so its transform is amortised over every query)
  and is the honest configuration to benchmark; route 2 avoids new host code but must not be counted as
  per-query cost.

Today's 58,223 workloads at 1024 cells are dominated by exactly these per-cell DB transforms, which is the
measurement that makes this concrete.

### Phase E — the end-to-end measurement
E1. Measure: per-stage latency, end-to-end (host wall + device-busy span), QPS/card at Q=16.
E2. Rewrite `onion_match_perf.md` with p150 numbers and remove its stale banner.

### Known risks
- A3/A4 are stride plumbing — an off-by-one silently corrupts data; gate before building on them.
- D4's decrypt-check can pass for the wrong reason if the scan reads wrong cells → keep the small-hypercube
  bit-exact gate as the primary correctness signal.
- Trace region is 256 MB; the 128 MB pipeline's program count must be checked against it.
- Pool buffers scale with cells; the DB must be one large buffer of tiles, not one buffer per cell.

## 4g. REMAINING WORK — handoff (state as of 2026-07-31 end of session)

### Done and validated on device (all gates bit-exact or decrypt-checked)
Expansion 7998 -> **51.7 ms** (154x, zero host boundaries) · answer pipeline 3.45 s -> **104.6 ms** (33x) ·
**1024-cell retrieval with the GEMM scan: 1354.7 ms, decrypt 0/2048** · scan at full cell shape
**~5.5 ms/query**.
Plan phases: **A ✅ · B1 ✅ · C1 ✅ · C2 ✅**.

On device: Garner CRT, gadget decompose, digit->RNS, both layout reshapes, fwd/inv NTT, MAC, transposes,
automorph, monomial multiply, ct-add, batched-GEMM scan, mod-q recombine, generalised permute.

### 1. C3 — query batching on the n axis  *(small, measured free)*
`n = 2*Q`; Q=16 fills all 32 columns and removes the 16x padding waste (§4d measured the widening as free:
time unchanged 2 -> 32 while useful work rose 16x). Needs the answer harness to carry Q queries and the B
operand to hold their selectors. Makes **QPS/card** the headline alongside single-query latency.

### 2. D — the 128 MB workload  *(the gate on a real end-to-end number; two named prerequisites)*
- **D-a. Query-only case format.** A 65,536-cell case would put 1.6 GB of DB in the file. Emit keys +
  selectors + RGSW ladders + secret + target index only; the device synthesises the DB. Enabled by the
  self-checking DB (`DB[k] = k`) — no DB transfer, and decrypt-check compares against the index.
- **D-b. Offline DB forward transform.** Per-cell on device is ~2M programs (65,536 x 3 x ~10). Either a
  **host-side C++ NTT** writing a cached eval-domain DB file (the repo has none — device-side only; this is
  the main new code D needs, and it matches how a real server amortises a static DB), or chunked
  `fwd2048_buf2_batch` with a bounded workload cache. Do NOT count either as per-query cost.
- Then: chain expand -> unpack -> scan -> fold -> mod-switch resident; run at `dims=[512,2,2,2,2,2,2,2]`;
  validate by decrypt == index. **This is the first real end-to-end number.**
- Watch: DB as ONE large tile buffer (not per-cell buffers — pool scaled 26k at 1024 cells before the GEMM
  cut it 73%); trace region is 256 MB; keep the 32-cell bit-exact case as the primary correctness gate.

### 3. E — the end-to-end measurement
Per-stage latency, end-to-end (host wall + device-busy span), QPS/card at Q=16. Then rewrite
`onion_match_perf.md` with p150 numbers and drop its stale banner.

### 4. Deferred optimizations (not blockers)
- **Scan tail slot->poly gather is still on host** (untilize per output tile). On-device form is designed:
  64x `transpose_dev` (in-tile transpose + relabel, no inter-tile movement) + one composed-table permute
  using the now-generalised `permute_signed` (n_in != n_out).
- **B3 fusion / cheaper permute.** Width saturated three separate times (§4e, B1), so the levers are DEPTH
  (fuse the NTT's sequential stages) and less arithmetic (the permute kernel is a 2048-iteration scalar RISC
  loop; vectorising it or expressing it as a matmul cuts real work).
- **On-device mod-switch finalize** — the last host arithmetic outside the tree.

### 5. Measurement discipline to keep (each cost a wrong conclusion here)
1. Discard the first run (cold JIT read 10.0 s vs 3.45 s warm).
2. Never time a single pass; report **median/mean**, not min.
3. **Drain the queue** (`Engine::finish()`) once a stage has no blocking readback, or you measure host
   enqueue only (read 1.3 ms instead of 53 ms).
4. Attribute readback **per call-site**, not by average (96 fetch readbacks were ~free; 30 in-tree ones were
   ~1.9 ms each).
5. Every buffer carries a **layout**; elementwise ops need both operands in the same one (a mismatch gives
   deterministic, plausible, wrong results that only a golden catches).
6. `pmul` is Montgomery, the GEMM is a plain product — do not pre-`to_mont` for the GEMM path.
7. Co-scheduling pays in proportion to available independence; measure per stage (it was -3% in one place,
   +7% in another, and 3.5% for a 3x widening).

## 5. TT scan, measured at real scale

`test_scan_mc 8192 16` (OT=8192 output plane-tiles × KB=16 = exactly 128 MiB of int8 DB tiles), int8
matmul-accumulate validated bit-exact vs a CPU golden:

| cores | n150 (docs) | p150 |
|--:|--:|--:|
| 16 | 22.58 GB/s | 28.78 GB/s |
| 32 | 27.85 GB/s | **48.61 GB/s** (1556 GMAC/s) |
| scaling @32 | 15.03× | **22.50×** |

n150 saturated at a ~28 GB/s DRAM ceiling; **p150 is still climbing at 32 cores**, and its grid is 13×10 =
130 cores, not 8×8 — so the plan's "≤32 cores" budget leaves real headroom unmeasured.

---

## 6. Remaining work to a same-size demo

`int8 is NOT a gap` — `scan_mm` already runs the initial dimension on the int8 matrix engine, validated
bit-exact at 48.61 GB/s. What is missing is **integration**.

1. **Trace capture/replay** — enqueue is now **77%** of the 164 ms. Needs `trace_region_size` at device
   init, persistent programs **warmed before capture** (the first enqueue writes binaries and trace forbids
   writes during capture), and segments broken at the remaining host boundaries (486 readbacks left).
2. **Self-checking DB** (design: store each slot's own index as its payload, `DB[k] = k`). A correct
   retrieval then decrypts to the index requested — no reference data at all, Python drops out of DB
   generation entirely, and the NTT'd DB can be **cached to a file** (deterministic and query-independent,
   ~1 GB at 128 MB scale). Failures also become diagnosable: a wrong-but-valid index points at
   selector/fold indexing rather than yielding garbage.
3. **C++ client decrypt** — per-channel negacyclic multiply with `s` (u64), CRT to u128, scale-round by
   t/q. Gate it against the 32-cell case that HAS a golden before trusting it where none exists.
4. **`scan_mm` + `recombine_rns` as the real initial dimension**, replacing the pointwise MAC (at
   N₀=512 × rest=128 the pointwise form is ~393k dispatches).
5. **Chain query expansion at full width** — validated standalone (`test_query_expand2048`, 0/49152) but
   never run chained at height-10 / 512 selectors. It is about half of server time (§4e), so a demo
   shipping pre-expanded selectors would not be a real end-to-end result.
6. Then scale to **512×2⁷ / 128 MB**.

### Case generation at scale is no longer a blocker

`gen_pir_answer_case.py --no-golden` skips the O(cells) Python `pir_answer` golden (which is what costs
days at 128 MB) while still emitting keys, DB, selectors, the secret `s` and `dbidx`. Measured:

| case | cells | golden | time |
|---|--:|---|--:|
| dims=[8,2,2] | 32 | yes | 11m00s |
| dims=[64,2,2,2] | 512 | **no** | 13m27s |

16× the cells for ~20% more time — the residual cost is `rlwe_to_rgsw` at O(folds × ℓ²), i.e.
**cell-independent**. Extrapolated to 7 folds / 512 selectors: ~30–45 min, one time. Not a wall.

> Caution learned the hard way: the generator's final line printed `recovered==DB[idx] OK` from a
> hardcoded string even under `--no-golden`, where that check is skipped. Fixed, but the lesson generalises
> — a validation that can *appear* to pass is worse than none, which is why item 3 above must be gated
> against a real golden first.

---

## 7. Reproduce

```bash
. env.sh && cd device && cmake -B build -G Ninja && cmake --build build \
    --target test_pir_answer2048 test_ntt2048_res test_dispatch_micro test_scan_mc

# goldens (~11 min each; the 2nd is a DIFFERENT query on the same structure)
cd ../cpu/pir
python3 gen_pir_answer_case.py --K 3 --n 2048 --ell 8 --B 2048 --dims 8,2,2 --idx 5 --out a5.bin
python3 gen_pir_answer_case.py --K 3 --n 2048 --ell 8 --B 2048 --dims 8,2,2 --idx 9 --out a9.bin

cd ../../device
PIR_ITERS=8 ./build/test_pir_answer2048 ../cpu/pir/a5.bin ../cpu/pir/a9.bin   # ladder + all 3 gates
./build/test_ntt2048_res            # residency mechanism, checks [1][2][4][5]
./build/test_dispatch_micro 64 20   # dispatch-cost regimes
./build/test_scan_mc 8192 16        # 128 MiB scan, core scaling
```

Discard the first run of anything (cold JIT). A hung run wedges the card until `tt-smi -r`.

## 8. Artifacts

- Engine: `lib/pir/pir_engine2048.hpp` — `prof::` split, `emit`/`WC`/`POOL`/`begin_pass`,
  `fwd2048_buf`/`inv2048_buf`/`transpose_dev`, `fusedA_off`/`fusedC_off`,
  `fwd2048_buf2`/`inv2048_buf2`, `write_fwd_in2`/`read_inv_out2`, `mac_ntt_buf`.
- Kernel: `kernels/shared/dataflow/reader_fused_off.cpp` (new).
- Tests: `tests/pir/test_ntt2048_res.cpp`, `host/experiments/test_dispatch_micro.cpp` (new);
  `test_pir_answer2048.cpp` (record/replay, `PIR_ITERS`, optional 2nd-case new-query gate).
- Generator: `reference/pir/gen_pir_answer_case.py --no-golden`.
- Method source: `docs/he/history/OPTIMIZATION_REPORT.md`, `docs/he/blackhole_port.md`.
