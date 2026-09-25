# Profiling rules

**Status:** Current · **Normative** — a number produced outside these rules is not quotable in a doc,
a commit message or a slide. If you are taking a first measurement in this repo, read this before the
run, not after it.

Companion: [`optimization.md`](optimization.md) — what to *do* once a term dominates (host / DRAM / kernel, one doc each).

---

## 0. Three rules that come before any number

| # | rule | why — measured |
|---|---|---|
| 1 | **Name the metric and the shape.** Never a bare "3.4 ms". | The same 7.494 ms means different things at `B=1` and `B=4`. Always state which of §1's four metrics, plus `G`/`B`/`S` (or cores/dims). |
| 2 | **Discard the first run.** | Cold kernel-JIT: **10.0 s** for a workload that runs **3.45 s** warm. An early "Blackhole fold latency regressed 3.2× vs n150" was *entirely* this artifact — warm p150 is faster. |
| 3 | **Never time a single pass.** | The enqueue term alone swings **~12% run to run**, which hides changes of ~25 ms. Loop it (`PIR_ITERS`, `HE_ITERS`), report min/mean/max/spread, **quote `min`** (least host interference). Spread sets your detection floor: at 7% spread only changes >~15 ms are real. |

---

## 1. The four latency metrics — and only these four

Two are host-measured, two are trace-derived. That pairing is the point: switching the profiler on
perturbs wall time and busy span but moves **host device busy by ~0.6%**, so the host pair anchors the
trace pair.

| term | source | definition |
|---|---|---|
| **wall time** | host | host end-to-end |
| **busy span** | trace | first cycle the device starts → last cycle it stops. A *window*: idle between dispatches is still inside it. |
| **host device busy** | host | Σ host stopwatch around each **blocking** segment replay. Excludes host stalls *between* segments; includes device idle *inside* a segment + ~52 µs/call blocking overhead. |
| **critical path busy** | trace | busy span **minus idle** = `Σ over dispatches ( max over that dispatch's cores )` |

Guaranteed orderings — `busy_metrics.py` checks them:

```
wall time  >=  busy span
wall time  >=  host device busy  >=  critical path busy
```

`busy span` vs `host device busy` is **not** ordered; which is larger tells you whether inter-segment
host stalls or per-call overhead dominates.

**All four must come from one execution**, or the orderings can appear violated. None of the four
includes setup, the CPU oracle, or the correctness check. Critical path busy is pure kernel time —
it excludes ciphertext I/O entirely; wall time includes it.

Full definition and the inclusion matrix: [`../he/exp_tb001.md`](../he/exp_tb001.md#latency-terminology).

---

## 2. The host-phase split — the first thing to measure

Four buckets, decomposing **host wall-clock by what the single host thread was doing**. Rows sum to
the wall. There is no device-busy row: the device mostly *starves* waiting on host prep, and device
time only surfaces at blocking drains.

Which bucket dominates picks the fix, and that is the whole purpose of the split:

| dominant term | meaning | go to |
|---|---|---|
| **ENQ** (`EnqueueMeshWorkload`) | fresh-program serialization — binaries, CB configs, runtime args re-sent every dispatch | [`optimization.md` §1](optimization.md#1-host-side) — workload cache, then trace, then co-scheduling |
| **BUILD** (`CreateProgram`/`CreateKernel`) | rebuilding programs that could be cached | same — the cache attacks BUILD and ENQ together |
| **READ** (blocking) | host round-trips between device ops | [`optimization.md` §1](optimization.md#1-host-side) — buffer residency |
| **WRITE** | uploads | rarely dominant (~0.3% typical) |

> **Do not read "BUILD is only 10%" as "caching saves only 10%".** ENQ cost *is* fresh-program
> serialization, so program reuse attacks both — ~94% of host time in the measured case.

Representative real output, one workload (`test_pir_answer2048`, K=3, dims=[8,2,2]), record vs replay:

| | RECORD | REPLAY |
|---|--:|--:|
| total | 3688 ms | **248.1 ms (14.9×)** |
| program BUILD | 392 ms (11%) | 0.0 ms (**0 calls**) |
| workload ENQ | 2897 ms (83%), 0.491 ms/call | 119.6 ms (53%), **0.020 ms/call** |
| buffer READ (block) | 191 ms (5%) | 96.2 ms (**43%**) |

The profile **inverts** as you optimize: READ went 5% → 43%, which is what justified residency next.
Re-measure the split after every rung; the previous winner stops being the lever.

---

## 3. Example — the host-side instrument

The whole instrument is ~20 lines. This is `prof::` from
[`../../host/lib/pir/pir_engine2048.hpp`](../../host/lib/pir/pir_engine2048.hpp) (same shape in
`host/experiments/legacy/he_mul_cross.cpp`); copy it rather than inventing a variant.

```cpp
namespace prof {
inline double g_build = 0, g_enq = 0, g_wr = 0, g_rd = 0;
inline long   n_build = 0, n_enq = 0, n_wr = 0, n_rd = 0;
using hrc = std::chrono::high_resolution_clock;
inline double msd(hrc::time_point t) {
    return std::chrono::duration<double, std::milli>(hrc::now() - t).count(); }
inline void reset() { g_build = g_enq = g_wr = g_rd = 0; n_build = n_enq = n_wr = n_rd = 0; }
inline double total() { return g_build + g_enq + g_wr + g_rd; }
inline void report(const char* tag) {
    double t = total();
    auto row = [&](const char* nm, double g, long n) {
        fmt::print("    {:<22} {:9.1f} ms  {:5.1f}%  {:7} calls  ({:.3f} ms/call)\n",
                   nm, g, t > 0 ? 100.0 * g / t : 0.0, n, n ? g / n : 0.0); };
    fmt::print("  host-phase split [{}]  total {:.1f} ms\n", tag, t);
    row("program BUILD",       g_build, n_build);
    row("workload ENQ",        g_enq,   n_enq);
    row("buffer WRITE",        g_wr,    n_wr);
    row("buffer READ (block)", g_rd,    n_rd);
}
}  // namespace prof
```

Wrap each SDK call once, so no call site can forget to be timed:

```cpp
template<class...A> void t_enq(A&&...a) { auto t = prof::hrc::now();
    Dp::EnqueueMeshWorkload(std::forward<A>(a)...);   prof::g_enq += prof::msd(t); ++prof::n_enq; }
template<class...A> void t_rd (A&&...a) { auto t = prof::hrc::now();
    Dp::EnqueueReadMeshBuffer(std::forward<A>(a)...); prof::g_rd  += prof::msd(t); ++prof::n_rd;  }
```

Call `prof::reset()` at the start of each pass and `prof::report("pass")` at the end — otherwise a
cold and a warm pass are summed together and neither is interpretable.

### Put the marker where every path passes, not where it is convenient

A real bug in this repo's own instrument: BUILD was timed **inside `emit()`**, but ~40 sites use the
bare `if (!cached()) { Program p = ...; run(std::move(p)); }` shape and never enter `emit()`. BUILD
therefore reported **0 calls** and its cost landed in no bucket at all — the split still summed to the
wall, so nothing looked wrong.

```cpp
bool cached() {                                  // stamp where the cache DECLINES ...
    if (!(CACHE_ON && !CACHE_REC)) { t_dispatch = prof::hrc::now(); return false; }
    ...
}
void run(Program&& p) {                          // ... and attribute at the one common choke point
    prof::g_build += prof::msd(t_dispatch); ++prof::n_build;
    ...
}
```

**Rule: a counter that reads 0 calls is a broken instrument until proven otherwise, not a fast path.**

---

## 4. The device trace

```bash
# capture
TT_METAL_DEVICE_PROFILER=1 HE_TRACE2=1 HE_PROF2=1 NTT_CORES=64 ./build/he_mul_cross 0
CSV="$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"

# the four metrics of §1, and nothing else
python3 tools/traces/busy_metrics.py "$CSV" --expect 552 --wall-ms 27.947 --host-busy-ms 8.703
```

**The device-side profiler buffer overflows partway through a full `ct_mul` and silently drops
markers.** For a complete capture use `HE_PROF_DRAIN=1` (drains every `HE_DRAIN_K`, default 100)
instead of `HE_PROF2=1` — and then pass `--expect N` to the viewers, because a drained capture stalls
~23 ms periodically and the default window-picker would shatter one multiply into ~40 clusters.

A drained capture is also **unusable for a busy span** (a span is idle-sensitive; the stalls inflate it
~40×). Sums — critical path busy, core-time — survive draining. That is why the `G=8` busy span is
reported as *not obtainable* rather than estimated.

Analyzer inventory: [`../../tools/traces/README.md`](../../tools/traces/README.md).

### Verify the trace against the host before believing any breakdown

**A tool filtered out every dataflow-only dispatch group and made 17.9% of the multiply invisible.**
`dispatch_csv.py` kept only groups carrying a TRISC envelope; the corner turn is
`a2a_corner_rd.cpp` + `a2a_corner_wr.cpp`, dataflow only, so all 152 of its dispatches vanished. Three
consequences, all wrong in the same direction:

* `a2a_corner`'s 9.103 ms disappeared from the census;
* the shortfall re-appeared as a gap between critical path busy and device busy, reported as launch
  overhead — **"8.42 ms, 15.6%". The real figure is 1.469 ms, 2.9%**: the device is inside a kernel
  97.1% of the time;
* surviving windows absorbed the missing names, inflating four kernels' counts by exactly 152.

Adding a `DeviceZoneScopedN` to the dataflow kernels does **not** fix it — the group is filtered out
before naming happens. The fix is `--dataflow-ok` plus `--labels`, letting the host's own record name
what the trace cannot.

**Gate: the join must verify.** `busy_metrics.py` refuses to report on a capture with dropped markers,
and a census is only quotable once it prints *"all 2884 rows agree on the kernel between host and
trace"*. A per-kernel breakdown that does not reconcile 1:1 with the host's dispatch count is wrong,
and it will be wrong *plausibly*.

Verified census at 50.920 ms ([`../he/exp_tg018m.md`](../he/exp_tg018m.md)), as an example of the
end state:

| bucket | ms | % |
|---|--:|--:|
| NTT matmuls | 12.943 | 25.4 |
| modmul | 12.830 | 25.2 |
| FBC | 11.235 | 22.0 |
| a2a_corner | 9.103 | 17.9 |
| other | 3.399 | 6.7 |
| **not in kernel** | **1.469** | **2.9** |

---

## 4b. Comparing two kernels — the unified metric

§1's four metrics answer *"where does the pipeline spend time"*. They **cannot** answer *"is kernel A
faster than kernel B"* at the few-percent level. That needs one metric and four rules.

**The metric: the MATH (`TRISC_1`) `*-KERNEL` span, FITTED over a tile sweep.**

```
span(T) = fixed + T · per_tile        →  quote per_tile, report max residual
```

The firmware zone wraps `kernel_main()`, so it excludes host enqueue, the command queue, and
`wait_for_go_message()` (which sits *outside* it). The fit removes what remains.

| measured by | fixed cost / dispatch | resolves |
|---|---|---|
| host wall-clock ÷ T | **~27 µs** | nothing below ~5% |
| host wall-clock, fitted | ~27 µs (removed) | ~5% |
| **device `*-KERNEL`, fitted** | **0.62 µs** (removed) | **0.0065% residual → 2.3% gap** |

### The four rules

1. **Fit, never divide.** `ms/T` folds a fixed cost into every tile and it **penalises the faster
   kernel** — the same 27 µs is a larger fraction of a shorter kernel. Reported Barrett 5% slow when
   the true gap was 2.3%.
2. **Equal instrumentation.** A `DeviceZoneScopedN` inside one kernel's loop and not the other's
   compares instrumentation, not arithmetic. Compare kernels with **zero** internal zones; use the
   firmware envelope.
3. **Interleave if you must use host timing.** Position in the dispatch sequence outweighs a
   few-percent effect: removing an unrelated third variant, which only moved one kernel from third
   place to second, **flipped the sign** of the answer.
4. **Discard the run after any kernel edit.** The edit changes the JIT hash; the next run absorbs
   compilation. A *negative* fitted slope is the tell.

### Example — fit and residual

```cpp
// spans[] = min MATH span over the timed replays at each T in TL
double sx=0, sy=0, sxx=0, sxy=0; const double n = TL.size();
for (size_t i=0;i<TL.size();++i){ sx+=TL[i]; sy+=spans[i];
    sxx+=(double)TL[i]*TL[i]; sxy+=(double)TL[i]*spans[i]; }
const double per_tile = (n*sxy - sx*sy) / (n*sxx - sx*sx);      // <-- quote this
const double fixed    = (sy - per_tile*sx) / n;
double resid = 0;                                                // <-- and this
for (size_t i=0;i<TL.size();++i) resid = std::max(resid, std::abs(spans[i]-(fixed+per_tile*TL[i])));
```

**A difference smaller than ~3× the residual is not a result.** Working code:
[`../../host/experiments/exp_modmul002.cpp`](../../host/experiments/exp_modmul002.cpp) (trace fit + a paired protocol),
[`../../host/experiments/exp_modmul001.cpp`](../../host/experiments/exp_modmul001.cpp) (six-point fit, 0.006% residual).

### Cross-check before believing it

* the same kernel measured in two benches must agree — `modmul_sfpi_1pass`: **3.086** in
  `exp_modmul001`, **3.0866** in `exp_modmul002`;
* the result must be consistent with the static instruction count. Disassemble:
  `riscv-tt-elf-objdump -d <cache>/kernels/<name>/*/trisc1/trisc1.elf | grep -oiE '\bsfp[a-z0-9]+' | sort | uniq -c`.
  104 vs 97 instructions predicted 7.2%; measured 2.3% — the schedule hid two-thirds. **A timing that
  contradicts the instruction count is wrong until explained.**

Full history: [`../he/exp_modmul002.md`](../he/exp_modmul002.md) §1.

---

## 5. Reporting

Every figure carries a provenance tag:

| tag | meaning |
|---|---|
| **[M-p150]** | measured on current hardware, bit-exact where correctness applies |
| **[M-n150]** | measured on Wormhole — **historical**; different architecture *and* older code. Never placed beside a p150 figure for comparison |
| **[P]** | cost-model projection (per-kernel latency × op count). Not a run |
| **[X]** | not measured — named so it is visibly absent rather than quietly assumed |

Also required: correctness alongside latency (`0/114688`, `0/12288` — this repo quotes mismatch
counts, not "passed"); the spread, so a reader knows what is detectable; and, when `B` or `S` ≠ 1,
metrics quoted **per multiply**.

Record negative and refuted results in the doc with the same weight as wins — they are what stops the
next person paying for the same run. See `optimization.md` §4.

---

## 6. Traps, all paid for once already

| trap | symptom | rule |
|---|---|---|
| cold JIT cache | 10.0 s vs 3.45 s | discard run 1 |
| single-pass timing | ~12% swing | loop, quote `min` |
| trace **capture** | output is garbage; a spurious `12288/12288` FAIL | capture RECORDS, does not EXECUTE — discard its output. Capture must follow the record pass (first enqueue writes binaries; trace forbids writes during capture) |
| profiler buffer overflow | silently dropped markers | `HE_PROF_DRAIN=1` |
| drained capture | span inflated ~40×; window shattered | spans invalid; pass `--expect N` |
| tool-side filtering | a whole kernel missing; fake "launch overhead" | verify the host↔trace join, 1:1 |
| instrument on the wrong path | a bucket reading 0 calls | stamp at the choke point (§3) |
| device wedged from an earlier crash | *next* run hangs with no output | `tt-smi -r`, then a small sanity run. It looks exactly like a fresh failure in whatever you just changed |

Hangs and wrong answers (Watcher, tt-triage, DPRINT): [`../debugging.md`](../debugging.md). Use
**Watcher, not DPRINT**, for hangs — DPRINT serializes the TRISCs and hides exactly the class of bug
you are chasing.
