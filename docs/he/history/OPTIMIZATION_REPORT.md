# HE-multiply on Tenstorrent — optimization report (CROSS Set-B)

> **Status: Historical.** Superseded by [`../ct_mul_perf.md`](../ct_mul_perf.md) and [`../blackhole_port.md`](../blackhole_port.md) §3.
> The Set-B optimization record (5044 -> 727 -> 38 ms). Its numbers are a SNAPSHOT: measured at 102 trace segments / 2065 cached workloads, where the current code produces 46 / 1219. Kept because the step-by-step attribution is why the design is what it is.
> Measured on Wormhole n150.

Progress, results, and analysis for optimizing the on-device HE-multiply at CROSS's Table VIII
parameter set, **Set B** (N=2¹³=8192, L=8 limbs, 28-bit primes, dnum=2, P=4 → QP=12). All numbers on
one Wormhole **n150**, fixed 16 Tensix cores, **bit-exact vs a CPU oracle (0/114688)** throughout.

> **Baselines here are a code snapshot, not just a machine.** The traced-resident numbers below were
> measured at 102 trace segments / 2065 cached workloads; the current code produces 46 / 1219. Do not
> diff them against newer runs (or other hardware) without accounting for that — see
> [blackhole_port.md](../blackhole_port.md) §3.

Files: `host/experiments/legacy/he_mul_cross.cpp` (CROSS-aligned HE-mul, being optimized), `he_mul_setb.cpp`
(coeff-resident reference), `bfv_relin_dnum.cpp` (dnum key-switch), `nntt8192_{dev,mc}.cpp`,
`nintt8192_dev.cpp` (NTT-8192), `reference/he/bfv_relin_dnum_ref.py` (golden). Profiling:
`generated/profiler/traces/he/{trace_hemul,trace_viz}.py` + the self-contained canvas trace.

---

## 1. Where we started & the workload

- **Workload = CROSS Set B HE-Mult** (the smallest HE-Mult point in Table VIII).
- **Same HE sub-kernel DAG as CROSS** (only the NTT *internals* differ): eval-resident ciphertexts →
  **tensor = pure pointwise (0 NTT)**, drop-prime rescale first, dnum digit-decomposed key-switch
  (eval-in/eval-out), combine. Verified §7 of `cross_comparison.md`. Residual difference vs CROSS:
  dnum=2 (ours) vs 3 (Set B).
- **Correctness bar**: bit-exact vs a CPU oracle of the identical RNS arithmetic (evk placeholder-random —
  latency is evk-independent; HE key-switch correctness proven separately by `bfv_relin_dnum_ref.py`,
  105-bit identity margin).

## 2. Baseline & profiling analysis

Initial (correctness-first) build: **5044 ms/mul** at 16 cores. On-device profiling
(`TT_METAL_DEVICE_PROFILER=1`, 1 GHz → 1 cycle = 1 ns) showed it is **~99% host-orchestration, not
device compute**:

- **Device-compute distribution** (per-call exact): matmul-A 61% + matmul-C 21% (NTT/INTT) = **82%
  matmul**, base conversion **9%** (FBC is ~1.2 ms/call vs ~24 µs for a matmul), twiddle 4.5%, add 3%,
  pointwise 1.5%. Summed device-compute ≈ 0.97 s across the 16 core/RISC lanes; **wall device-active ≈
  tens of ms**.
- **The gaps.** Timeline span ~9 s (≈2 captured muls) was **93% idle** — 170 gaps of ~50 ms each. The
  ~50 ms is **per transform helper call** (not per dispatch): each transform regenerated its twiddle
  matrix on the CPU + did a synchronous readback. Dispatches *within* a helper run back-to-back (<1 ms).
- **Core count barely matters** (4→64 cores: only 27%) and **repetition doesn't amortize** (4 sequential
  muls ≈ 1 mul each) — both confirm host-orchestration-bound, not device/dispatch-bound.

**Conclusion:** the enemy is host work between kernels (twiddle regen + device↔host readback), and the
per-dispatch launch overhead (~tens of µs, a TT-stack fixed cost) is a distant secondary floor.

## 3. Optimization steps (each bit-exact 0/114688)

| step | change | ms/mul | vs baseline |
|---|---|--:|--:|
| baseline | regen constants + readback per op | 5044 | 1.0× |
| **1 — paramgen hoisted** | precompute all (prime,dir) twiddle/fold sets once at setup; upload from a host cache (no regen between kernels) | **771** | **6.5×** |
| **2 — constant residency** | upload each prime's folded constants to resident DRAM once; transforms reference them (no per-call const upload) | **727** | 6.9× |
| **3 — buffer residency** | keep all intermediates in DRAM; on-device transpose (reindex) + device elementwise; one final readback | *in progress* | — |

**Step-1 analysis (6.5×).** Twiddle *generation* (`modpow` + radix-128 fold + tilize, recomputed every
transform call) was **~85%** of the per-op host gap. Hoisting it to a one-time precompute recovered it.

**Step-2 analysis (small).** Constant *uploads* were minor (~44 ms total). Confirms the remaining 727 ms
is **not** constants — it's the per-op **`Finish` + blocking readback**: ~150 ops × ~5 ms = the device↔
host ping-pong (each op copies its result to a host `std::vector` so the next op can read it; the readback
is slowed by the `hugepage not on NUMA node` warning).

## 4. Step 3 — buffer residency (the remaining lever)

**Goal:** during the whole multiply, **nothing returns to host**; every intermediate lives in DRAM;
read back only the final result once. Removes the ~150 device↔host round-trips (the entire remaining
727 ms) → expected **tens of ms** (dispatch-bound floor).

**Mechanism validated (Brick 3.1):** buffer-resident `fwd → on-device transpose_tile → inv` round-trip
is **bit-exact (0/8192)** with one `Finish`, no host round-trip. Key facts:
- Constants already resident (Step 2).
- The host reindex becomes an **on-device `transpose_tile`** (the project's own note: reindex = tile
  transpose). Exact because **`inv`'s input tile = transpose of `fwd`'s output tile**.
- `fwd_buf` / `inv_buf` / `transpose_dev` are in the file and validated.

**Remaining assembly (the large part):** thread residency through the full tensor + rescale + key-switch,
with three fiddly sub-parts: (a) **resident rescale** (mixes domains: INTT-last + cross-modulus reduce +
NTT-correction), (b) **unified-Montgomery** threading (one operand of each product in Montgomery form),
(c) **buffer aliasing** (no `Finish` between ops → need enough distinct resident intermediates). Template:
the project's buffer-resident `bfv_ct_mul` (N=1024). Built brick-by-brick with validation gates.

## 5. Bottom line so far

- **6.9× measured, bit-exact** (5044 → 727 ms), by removing host constant regen + upload.
- Remaining 727 ms **fully diagnosed** as removable device↔host readbacks; Step-3 mechanism proven.
- The bottleneck is **orchestration, not arithmetic** — the sub-kernel workload matches CROSS.

---

## 6. CURRENT STATE & RESUME NOTES (post-Step-3 experiment)

### Latency ladder (16 cores, bit-exact 0/114688 throughout)
| step | ms/mul | note |
|---|--:|---|
| baseline | 5044 | regen constants + readback per op |
| 1 — paramgen precomputed | 771 | 6.5× — twiddle *generation* was ~85% of the gap |
| **2 — constant residency (BEST)** | **727** | fastest validated; **this is the default `he_mul_cross` behavior** |
| 3 — buffer residency (`HE_RES`) | 1881 | **SLOWER** — bit-exact but counterproductive (see below) |

### KEY FINDING — cost model, now measured per-phase (idle-cause attribution)
The mul is **~98% device-idle**; the idle is the host thread preparing dispatches, **not** data movement.
Host-side phase timing of one **resident (Step-3)** mul, 16 cores (env-guarded `prof::` wrappers around
every `EnqueueMeshWorkload`/`EnqueueWriteMeshBuffer`/`EnqueueReadMeshBuffer`/`Finish` + the 5 program
builders; printed by the `HE_RES` timing loop). Wall = 2047 ms/mul, bit-exact:

| phase | ms/mul | % | calls | ms/call |
|---|--:|--:|--:|--:|
| **enqueue (dispatch serialization)** | **1339** | **65.4%** | 2073 | 0.646 |
| program build (`CreateProgram`/`CreateKernel`×3/`SetRuntimeArgs`) | 390 | 19.0% | 2065 | 0.189 |
| host misc (CPU `fbc8192`, tilize, vector/map) | 282 | 13.8% | — | — |
| readback (D→H) | 19.9 | 1.0% | 328 | 0.061 |
| Finish (drain; device compute hides here) | 13.1 | 0.6% | 22 | 0.597 |
| upload (H→D) | 2.8 | 0.1% | 952 | 0.003 |

**Corrects the earlier "~1 ms/dispatch = program construction" claim.** It IS ~0.84 ms/dispatch of host
work, but it splits **0.65 enqueue + 0.19 build** — the **enqueue/serialization dominates (65%), not the
build (19%)**. Cause: every dispatch uses a *freshly built* `Program`, so `EnqueueMeshWorkload` re-serializes
its binaries + CB configs + runtime args + launch into the command queue from scratch (nothing cached).
All real device↔host **data movement is ~1.7% combined** (upload+readback+Finish) — so Step-3's "keep it on
DRAM" removed a non-bottleneck. Slower purely on **dispatch count**: 2073 (Step-3) vs ~1050 (Step-2), each
~0.84 ms host. **The bottleneck is per-dispatch host prep; residency is the wrong lever.**

### Step-2 vs Step-3 phase breakdown (both 16 cores, bit-exact) — dispatch COUNT is the whole story
| phase | Step-2 768ms | Step-3 2047ms | ms/call |
|---|--:|--:|--:|
| enqueue (dispatch serialization) | 410 (53%) | 1339 (65%) | ~0.58–0.65 |
| program build | 127 (17%) | 390 (19%) | ~0.20 |
| host misc (CPU `fbc8192`, tilize) | 160 (21%) | 282 (14%) | — |
| readback (D→H) | 46 (6%) | 20 (1%) | ~0.06 |
| Finish (drain) | 20 (3%) | 13 (1%) | — |
| upload (H→D) | 3 (0.4%) | 3 (0.1%) | ~0.004 |
| **dispatches** | **~700** | **~2070** | — |

Both cost ~0.78 ms/dispatch host-prep (0.20 build + 0.58 enqueue). Step-2 wins purely on 2.7× fewer
dispatches — despite MORE readbacks (824 vs 328). **Readback is 1–6%, never the bottleneck.** build+enqueue
= 70% (Step-2) / 84% (Step-3).

### How to read these tables — host-wall decomposition, NOT a busy/idle split
The tables above decompose the **total host wall-clock** (768 ms Step-2, 2047 ms Step-3) by which
operation the **single host thread** was running. They are a *host-thread* accounting, so:
- **The rows sum to the wall.** They are NOT "idle time only," and there is **no separate row for
  device-busy (compute) time.**
- **Device compute runs asynchronously and overlaps these phases.** `enqueue` is non-blocking, so the
  device executes a queued command (~24 µs for a matmul) while the host is already building/enqueuing the
  next op (~0.6 ms). Because the host prep is ~30× slower than the device op, the device finishes and then
  **starves** — it is idle waiting for the host most of the wall.
- **Device-busy only becomes visible wall time at the two blocking drains:** `Finish` and blocking
  `readback`. Those bound the exposed compute: Step-2 `Finish` 20 ms + `readback` 46 ms ≈ **≤66 ms** contains
  all device work surfaced in the wall; the device profiler puts the real critical-path compute even lower
  (~tens of ms/mul).

What each phase MEANS, and the device's state during it:

| phase | what the HOST thread does | device during it |
|---|---|---|
| **program build** | CPU: `CreateProgram`/`CreateKernel`×3/`SetRuntimeArgs` — assemble next program | **idle** (nothing queued) |
| **enqueue (dispatch)** | serialize program binaries+CB configs+RTargs+launch into the command queue | mostly **idle** (drains 1 cmd in µs, waits ~0.6 ms) |
| **upload (H→D)** | issue (async) buffer writes | idle |
| **readback (D→H)** | *blocking* read: drain queue, then DMA copy to host | **busy** during drain, then copy |
| **Finish (drain)** | block until all queued work done | **busy** — accumulated compute exposed here |
| **host misc** | CPU `fbc8192` (base conversion), tilize/untilize, vector/map shuffles | **idle** |

**Therefore the true busy/idle split (Step-2, 768 ms):** device-busy ≈ tens of ms (hidden in Finish+readback,
no row of its own); **device-idle ≈ 700–740 ms (~91%)** = the build+enqueue+upload+misc phases, during which
the device has nothing to run. The two views are complementary: the **device profiler** timeline (FW-zone
union) says "device idle ~98% of wall"; this **host table** says "wall is ~91% build+enqueue+misc" — i.e.
that host prep is exactly *what the device is idle waiting for*. The host table shows the *cause* of the
idle; the profiler shows the idle *interval*.

### THE REAL LEVER (next step) — Program reuse + MeshTrace replay (TT stack supports preprocessing this)
The two dominant costs are exactly the two things the TT trace API preprocesses:
1. **Program reuse** → kills the **build** (17–19%): keep each `Program` alive (caches compiled binaries +
   finalized config), re-`SetRuntimeArgs` per mul, skip `CreateProgram`/`CreateKernel`.
2. **MeshTrace capture/replay** → kills the **enqueue serialization** (53–65%): run the dispatch sequence
   ONCE in capture mode (records all dispatch commands — binaries, RTargs, launch — into a DRAM trace
   buffer); each mul then issues a single `enqueue_trace(trace_id)` → device replays from DRAM, host cost
   ~µs. Confirmed API (installed headers): `distributed::BeginTraceCapture(mesh,cq_id)`→`MeshTraceId`,
   `mesh->record_end()`, `mesh->enqueue_trace(id,blocking)`/`replay_mesh_trace(...)`, `trace_region_size`
   in the MeshDevice config.

**Residency + trace are COMPLEMENTARY, not alternatives.** Trace bakes in buffer addresses → every buffer
touched inside the trace must be allocated ONCE and reused. **That fixed-address requirement is the real
payoff of Step-3 residency** (not readback avoidance). So the plan: (a) init device with `trace_region_size`;
(b) use the resident fixed-address buffer layout; (c) capture the device-op segments *between* FBC host
boundaries as traces; (d) `enqueue_trace` per mul, writing fresh inputs into fixed buffers with an
un-traced `EnqueueWriteMeshBuffer` first. Caveat: host `fbc8192` can't be inside a trace → either move FBC
on-device (whole mul = one trace) or keep it as an un-traced segment between trace replays (still removes
all 70–84% of build+enqueue). **Projected: 727 ms → ~150–250 ms/mul** (host-FBC + device-active floor),
then move FBC on-device for the residual 14–21%.

Instrumentation for the table above lives in `he_mul_cross.cpp` (`prof::` namespace + `t_wr/t_rd/t_enq/t_fin`
wrappers); the default run AND the `HE_RES` run both print the phase breakdown.

### T1 DONE — persistent programs + Metal Trace mechanism PROVEN (`HE_TRACE=1`, bit-exact)
Device init now reserves a 256 MB trace region (`create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 256MB)`).
`HE_TRACE` builds a 22-dispatch resident transform chain (fwd + 8 transpose + inv) as **persistent
workloads** (built once), warms them (loads binaries), then captures the re-enqueue as a MeshTrace
(`mesh->begin_mesh_trace(0)` → re-enqueue → `end_mesh_trace`) and replays (`replay_mesh_trace`).
**Correctness: `traced_vs_persistent = 0/8192` (replay ≡ eager).** Timing (22 dispatches, 20 iters):

| regime | ms/iter | ms/dispatch | speedup |
|---|--:|--:|--:|
| build-fresh + enqueue (status quo) | 17.745 | 0.81 | 1.0× |
| persistent re-enqueue (reuse `Program`s) | 0.592 | 0.027 | **30×** |
| **traced replay** | 0.508 | flat | **35×** |

**Key refinement of the cost model:** the dominant "enqueue" cost (0.54 ms/dispatch) was **fresh-program
serialization** — build-fresh split build 3.04 ms + enqueue 11.85 ms, but persistent re-enqueue of the SAME
objects drops the whole thing to 0.59 ms (enqueue → 0.027 ms/dispatch). So **Program reuse alone recovers
~almost all of it (30×)**; trace's added value is being **O(1) in dispatch count** (single replay command),
which only pulls ahead at scale: at the full mul's ~2000 dispatches, persistent ≈ 54 ms vs traced ≈ few ms.
Caveats learned: (1) **trace forbids writes during capture** — the first enqueue of a fresh program writes
its binary, so you MUST warm persistent programs before capture; (2) allocating buffers while a trace is
alive warns (allocate all resident buffers up front). Projection for the full mul: build+enqueue 537 ms
(70%) → ~single-digit ms ⇒ wall ≈ host-`fbc8192` (~160 ms) + io + device ≈ **~200 ms/mul**; then T3 (FBC
on-device) attacks the residual.

### T2 DONE — persistent-program cached resident path (`HE_RES_CACHE=1`, bit-exact 142.7 ms/mul)
Refactored the resident path so **buffers come from a POOL** (deterministic order, `PI` reset each mul →
stable addresses across muls) and **device workloads are cached** (`emit()` shim: build+cache on mul #1
via `CACHE_REC`, re-enqueue `WC[WI++]` after). `mkBuf8` pool-backed; `transpose_dev`/`fwd_buf`/`inv_buf`/
`ew` route through `emit`; resident constants (RConst/evk/ttmp) allocated with cache OFF so they stay
separate from the pool. Driver: record mul #1, then replay; host `fbc8192`/up/rd run every mul.
**`CACHED RESIDENT device-vs-oracle = 0/114688` PASS**; 2065 cached workloads, 468 pool buf-groups.

| version | ms/mul | note |
|---|--:|---|
| original baseline | 5044 | — |
| Step-2 const residency (prior best) | 727 | — |
| resident Step-3 | 2047 | — |
| **T2 cached resident (`HE_RES_CACHE`)** | **142.7** | **5.1× over Step-2, 35× over baseline, bit-exact** |

Phase breakdown at 142.7 ms: **build 0.0 (cached)**, **enqueue 80.55 ms (56.5%, 2073×0.039 ms/call** — was
0.646!), readback 11.1, Finish 10.8, upload 3.7, unaccounted 36.5 (host `fbc8192`+tilize). So build+fresh-
program-serialization is GONE; the remaining enqueue cost is purely **dispatch COUNT × 0.039 ms**.

### LATENCY METRIC (how to report end-to-end, per user 2026-07-24)
Two rules for the CROSS-comparable number:
1. **Tiling is NOT latency** — inputs are pre-stored in the device tiled layout; `tilize`/`untilize` is
   one-time data-prep, excluded from the measured region.
2. **End-to-end = device-busy span** — first cycle any device unit is busy → last cycle any device unit is
   busy (= all device work complete). Host wall-clock is NOT the metric.

Measured from the Step-2 profiler capture (`hemul_cross_16c.csv`): first-busy→last-busy SPAN = 9103 ms/~2
muls (host-bound, meaningless), but the **UNION of device-busy intervals (packed, host gaps removed) =
37.55 ms / ~2 muls ≈ 18.8 ms/mul** — this is pure device work (matmul/modmul/add/transpose; host `fbc8192`
is NOT counted). **So the device-work floor is ~19 ms/mul** vs the 142.7 ms host wall. The 142→19 gap =
enqueue gaps (80 ms, killed by trace) + host FBC/tiling/readback (killed by on-device FBC + pre-tiling).
Caveats: (a) ~19 ms is the Step-2 workload; the cached-resident path has ~720 extra single-tile transpose
dispatches so its packed device-busy is likely higher — must re-measure on the optimized path; (b) to
MEASURE the span directly (not estimate the union), the device must run back-to-back → a blocking
`replay_mesh_trace` wall ≈ device-busy span (host just waits). So trace is also the measurement enabler.

### T2.5 DONE — Metal Trace on the cached path (`HE_TRACE2=1`, bit-exact); device-busy MEASURED
Auto segment state machine: a "segment" = maximal run of `emit()` with no host boundary; `prof::seg_hook`
(called by `t_wr`/`t_rd`/`t_fin`) closes the open capture trace / arms the next replay segment. 3-phase
lifecycle: **REC** (warm+cache workloads) → **CAP** (`TR_PHASE=1`: re-enqueue cached workloads wrapped in
`begin/end_mesh_trace`, one trace per segment) → **REP** (`TR_PHASE=2`: first `emit()` of each segment
calls `replay_mesh_trace`; host FBC runs between). Fix vs first attempt: `t_fin` must close the open trace
before `Finish` (event-sync forbidden in capture).

**`TRACED RESIDENT device-vs-oracle = 0/114688` PASS.** 2065 cached workloads → **102 trace segments**
(20× fewer host ops). Results (16 cores):

| metric | ms/mul | note |
|---|--:|---|
| host wall | **83.8** | vs T2 142.7 — enqueue collapsed (2065 enqueues → 102 replays) |
| **DEVICE-BUSY span** | **38.0** | **CROSS-comparable end-to-end latency** — device work only, tiling + host-`fbc8192` excluded, measured as Σ of 102 blocking segment-replay walls |

**Reading the 38 ms:** 2× the ~19 ms Step-2 union estimate, because the resident path has **~720 single-tile
transpose dispatches** (on-device reindex), each using **1 of 16 cores for 1 tile** — now the dominant
device-busy inefficiency (not the matmuls). host wall − device-busy = ~46 ms host `fbc8192` + tiling +
inter-segment readbacks. (Benign warning: "allocating buffers during active trace" prints in CAP but
correctness passes bit-exact incl. timed replays.)

### Full latency ladder (16 cores, all bit-exact)
| version | host wall ms | device-busy ms |
|---|--:|--:|
| original baseline | 5044 | — |
| Step-2 const residency | 727 | — |
| T2 cached resident (`HE_RES_CACHE`) | 142.7 | — |
| **T2.5 traced (`HE_TRACE2`)** | **83.8** | **38.0** |

### Latency metric measured two ways (16c single mul, traced, bit-exact)
- **device-busy span** = Σ of 102 blocking segment-replay walls (tiling + host-FBC excluded) = **39.6 ms/mul**.
- **host end-to-end** = 85.4 ms/mul.
- **pure device busy** (compute-only ZONE union): profiler flush-isolation is broken (a single-mul capture
  dumps the whole run's buffer → 100 s span, union ~46 ms across ~2-3 muls) → **~18–23 ms/mul**. So the
  39.6 ms span is inflated by idle transpose cores + 102 per-segment replay/sync overheads; true compute is
  ~half of it.

### Baseline reconciliation: 4 muls on 16 cores (4 cores each) — the AGREED throughput baseline
All optimization above was measured as **1 mul on 16 cores** (latency config). The user's baseline is
**4 concurrent muls, 4 cores each** (throughput / spatial co-schedule). Measured single-mul points (traced,
bit-exact):

| config (1 mul) | host end-to-end | device-busy span | cached workloads |
|---|--:|--:|--:|
| 16 cores | 85.4 ms | 39.6 ms | 2065 |
| 4 cores | 127.9 ms | 80.1 ms | 3529 |

**Derived 4-mul-on-16 batch** (4 disjoint 4-core groups run concurrently → batch device-busy span ≈ one
4-core mul's span = 80 ms for 4 muls):

| metric | 1 mul / 16c | **4 muls / 16c (4 each)** |
|---|--:|--:|
| amortized device-busy / mul | 39.6 ms | **~20 ms** |
| device throughput | ~25 mul/s | **~50 mul/s (2×)** |

**The 4-mul batch is ~2× better throughput than 1-mul-16** (matches the co-schedule 2× in
[[he_compiler_coschedule_verified]]). Reason the device-busy metric surfaced: the **720 single-tile
transposes serialize on 1 core in the 16-core config** (15 idle); in the 4-mul batch each mul's transposes
run on its own group → all 16 cores stay busy. So for this transpose-heavy workload, spreading 4 muls across
the grid beats concentrating 16 cores on one mul. host-wall of the batch is **host-`fbc8192`-bound** (one
host thread, 4 muls' FBC serially) → T3 matters most there.
CAVEATS: batch numbers are DERIVED (1 mul × 4 cores × ideal concurrency), not measured — true
co-scheduling (one program across 4 groups) not yet built, so inter-group DRAM contention isn't captured;
pure-device-busy needs a clean isolated profiler capture (flush issue).

### Batched transpose DONE (`transpose8_dev`, bit-exact) — overturned the hypothesis
Replaced the 720 single-tile transposes (1 tile/1 core) with 90 batched 8-tile/8-core transposes.
Result (1 mul / 16c, traced): cached workloads **2065 → 1211 (42% fewer)**, host wall **85.4 → 81.3 ms**,
device-busy span **39.6 → 35.3 ms (only ~11%)**. **The transposes were a dispatch-COUNT problem (each ~1 µs
compute), not a device-busy problem** — trace/cache already absorbed the dispatch cost, so batching them
barely moved device-busy. 

**KEY finding: device-busy span is dominated by the ~102 per-segment replay/sync overheads**, not compute
(pure compute ≈ 20 ms via union; the rest of the 35 ms is per-segment sync). The 102 segments come from the
~8 host-`fbc8192` boundaries. **So the lever that cuts device-busy is FEWER SEGMENTS → T3 (on-device FBC)**,
which merges ~102 segments → ~1 (whole mul = one trace), cutting both per-segment sync (device-busy) AND the
host FBC (host wall, currently 81 ms, FBC-bound). This also re-frames the co-schedule 2× (attributed to
transpose utilization; it's really segment/host overhead — T3 changes that calculus).

### Ladder (1 mul / 16c, traced, bit-exact)
| step | host wall | device-busy span | workloads |
|---|--:|--:|--:|
| T2.5 traced | 85.4 ms | 39.6 ms | 2065 |
| + batched transpose | 81.3 ms | 35.3 ms | 1211 |

### T3 DONE — buffer-resident FBC (`fbc8192_res`, bit-exact) — mul is now DEVICE-BOUND
The base conversion always ran on device (`baseconv_fbc`); what was on host was the DATA MOVEMENT around it
(`rd_natC` readback → re-tilize/upload → `up_natC`). The FBC is per-tile-element (SFPU, no reindex) ⇒
layout-agnostic ⇒ can consume/produce the C'' buffers directly. Built two dataflow kernels —
`reader_gather` (1 tile from each of N source-limb buffers → c_0) and `writer_scatter` (first Ld tiles of
c_16 → Ld dest buffers) — and `fbc8192_res` (same `baseconv_fbc` kernel + constants as `fbc8192`, so
bit-identical; only the plumbing is resident, and it routes through `emit()` so it's cached+traced). Wired
the 3 FBC sites (rescale, keyswitch mod-up, moddown) to pass C'' `Buf8` directly — no host round-trip.

**`TRACED RESIDENT device-vs-oracle = 0/114688` PASS.** trace segments **102 → 46**; results (1 mul / 16c):

| step | host wall | device-busy span | segments |
|---|--:|--:|--:|
| T2.5 traced | 85.4 ms | 39.6 ms | 102 |
| + batched transpose | 81.3 ms | 35.3 ms | 102 |
| **+ T3 resident FBC** | **47.9 ms** | **43.0 ms** | **46** |

**host wall −41%** (host no longer does FBC readback/upload, and the FBC `baseconv` is now cached not rebuilt
per mul). **device-busy went UP 35→43 ms because it's now HONEST**: the old traced span EXCLUDED the FBC
`baseconv` device time (FBC ran as an untraced raw `t_enq` outside the segments); now it's counted. **43 ms
is the true all-on-device ct_mul latency** — nothing runs on host inside the mul except the final readback.
**The mul is now ~90% device-bound** (48 vs 43 ms, ~10% overhead) vs 56% overhead at T2.5 start.

### KEY FUSION INSIGHT — device-busy is 82% matmul, so fusion can't move it much
Per-op-type device-compute breakdown (profiled, current best build, bit-exact):

| op | % of device compute |
|---|--:|
| **matmul (NTT/INTT)** | **82.0%** |
| FBC (base conversion) | 11.0% |
| twiddle | 3.7% |
| vec add | 2.6% |
| pointwise | 0.5% |
| transpose | 0.1% |
| vec sub | 0.1% |

Two fusion experiments (batched transpose: ~11%; fused tensor `tensor2_fused`: neutral) are explained by
this: **fusion targets the ~7% of cheap ops and removes inter-op DRAM/reshuffle/dispatch — it does NOT
reduce the 82% matmul FLOPs.** So a *perfect full fusion* ceiling is only **~15–20%** (43 → ~35 ms), not a
step change. **The real lever is the NTT matmul itself.** The go/no-go question for a fused NTT: **is the
matmul DRAM-read-bound** (then keeping tiles in L1 across the four-step helps — build the fused NTT) **or
compute-bound** (then need a better NTT algorithm / matmul kernel, or it's the device's floor)?
Measure the compute-vs-DRAM split inside `fused_matmul` before investing in a fused NTT.

### Next
The 46 remaining segments are now **input uploads + acc-init + final readback**, NOT FBC. Levers:
1. **Pre-upload inputs** (user's "tiling excluded" rule) + **on-device acc-init** → removes those boundaries
   → fewer segments → lower device-busy + host wall.
2. **4-mul co-scheduled batch** (one program, 4 groups) for throughput — now meaningful since the mul is
   device-bound.

### (superseded) earlier T2.5/T3 sketch
- **T2.5 — Metal Trace on the cached path** (now legal: persistent programs + fixed pool buffers). Capture
  the device-op segments BETWEEN the ~8 `fbc8192` host boundaries; replay per mul. T1 proved replay is
  O(1) in dispatch count → enqueue 80 ms → ~few ms ⇒ **wall ≈ 65 ms**.
- **T3 — move `fbc8192` on-device** → removes the 36 ms host-misc + the FBC-boundary readbacks (11 ms), and
  makes the whole mul one traceable region (jax.jit whole-function). Floor ≈ device-active + upload ≈ **~25–30 ms**.

### Files / knobs (to resume)
- `host/experiments/legacy/he_mul_cross.cpp` — the optimized CROSS-aligned HE-mul. Default = Step-2 (727 ms, constant
  residency). Contains: `KC` host cache (Step 1), `bigWres/smallWres/twres` resident consts (Step 2),
  `fwd_buf`/`inv_buf`/`transpose_dev` (Brick 3.1, validated), and `dev_hemul_res` + resident helpers
  (`NTT_res`/`INTT_res`/`pw_buf`/`madd_buf`/`msub_buf`, `ew`) gated by `HE_RES` (Step-3, slower).
- Env knobs: `NTT_CORES=<n>` (fixed core count, default full grid), `HE_ITERS=<n>` (timed iters),
  `HE_PROFILE=1` (isolate one mul, dump profiler CSV), `HE_RES=1` (run resident dev_hemul_res),
  `HE_RT=1` (Brick 3.1 buffer-resident transform roundtrip).
- Build: `cmake --build build --target he_mul_cross`. Run: `NTT_CORES=16 HE_ITERS=1 ./build/he_mul_cross 0`.
- Params: N=8192, LQ=8 (qidx=0..6 after dropping prime 7), LP=4 (pidx=8..11), dnum=2, QP=11 working.
- Profiling/trace: `generated/profiler/traces/he/trace_viz.py <csv> out.html` → self-contained canvas
  trace (like `compiler/mlir_dag.html`); `trace_hemul.py` for the per-op breakdown + Perfetto JSON.
  Interactive Dash viewer `trace_app.py` (needs port-forward). Last trace: `hemul_cross_16c.{html,json}`.

### Not done (roadmap)
- **Persistent programs** (next; the corrected Step 3) → expected big drop from 727 ms.
- **Dispatch fusion** on the transform (fewer/wider dispatches).
- **32-core batch throughput** + **energy/perf-per-watt (tt-smi)**.
- Match CROSS exactly: dnum=3 (currently 2).
- HE-correctness of Set-B is validated only via the small-L golden (`bfv_relin_dnum_ref.py`, 105-bit
  margin) + bit-exact-vs-oracle here (evk placeholder-random; latency is evk-independent).
