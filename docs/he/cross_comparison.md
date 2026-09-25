# Set-B HE-multiply benchmark (CROSS Table VIII parameters) — plan

Benchmark our on-device HE **multiply** at the parameter set of the CROSS paper's Table VIII
("Latency and energy efficiency of HE kernels", HPCA 2026), on **32 Tensix cores** of the
Wormhole **n150**. Goal: an amortized-latency (then perf-per-watt) number for HE-Mult at a
published, well-defined parameter set.

Status: **plan** (not yet implemented). Builds on the working single-core BFV `ct_mul`
(`docs/he/README.md`, `ct_mul_perf.md`), the verified `dnum=2` key-switch
(`tests/he/bfv_relin_dnum.cpp`), and the compiler's placement/scheduling + batched codegen
(`compiler/schedule.py`, `kernels/ir/stage7_run.cpp`).

---

> **Status note (2026-08).** This began as a plan; the benchmark it describes now exists and runs
> (`host/experiments/legacy/he_mul_cross.cpp`, `tools/bench/hemul_cross.sh`). Measured today: **14.09 ms/mul device-busy,
> 24.43 ms host wall**, bit-exact `0/114688`, at 64 cores on **Blackhole p150** —
> see [`blackhole_port.md`](blackhole_port.md) §3. The step-by-step optimization that got there is in
> [`history/OPTIMIZATION_REPORT.md`](history/OPTIMIZATION_REPORT.md).
>
> The sub-kernel DAG matches CROSS's (§7). The `TBD` cells below are *not* stale work items; they
> are metrics that still need an amortized/batched measurement.

## 0. Decisions (locked)

| axis | choice | rationale |
|---|---|---|
| **Parameter set** | **Set B**: `N=2¹³ (8192)`, `L=8` limbs, `28-bit` primes, **`dnum=2` (α=4)** | The smallest HE-Mult config in Table VIII. Reachable scale-up from current `N=1024/L=3`; every base conversion stays at width ∈ {4,8} → no general-width FBC needed (`n│8`). |
| **Parallelism** | **two-level**: parallelize one HE-mul across a **core-group (≥4 cores)** for utilization, then **batch-tile** groups across 32 cores | User guidance: never single-core; maximize core utilization; amortized batch on 32 cores is the reported metric (= CROSS's "amortized single-batch latency"). |
| **Metric** | **amortized latency (µs) first**, then energy / perf-per-watt if telemetry allows | Table VIII is "latency AND energy efficiency"; latency is always obtainable, energy is best-effort via `tt-smi`. |

Our headline metric is **HE-Mult amortized latency** at these params.

---

## 1. Scope — what "HE-Mult" means here

CROSS HE-Mult = **tensor → rescale → relinearize** (Fig. 12 breakdown: `INTT-MatMul` 38% + `NTT-MatMul` + `BConv-MatMul` dominate; `VecModOps` ~3–4%). Our benchmark measures the **same op graph**, built from the project's scheme-agnostic `Z_q` kernels:

```
HE-mul op graph (N=8192, L=8, dnum=2):
  tensor      : eval-domain pointwise  (EMul×3 + EAdd)                     — c0d0, c1d1, c0d1+c1d0
  rescale     : CKKS-style drop-prime  (per-limb sub + ×q⁻¹)              — see §1.1
  relinearize : dnum=2 key-switch      (mod-up · nNTT · ×evk · niNTT · mod-down)
```

### 1.1 Rescale = CKKS drop-prime (not BFV `⌊t·e/Q⌉`)
CROSS is **CKKS**; its rescale is a **single-prime division + limb drop** (cheap: per-limb subtract + `×q_L⁻¹`). Our validated `ct_mul` is **BFV**, whose rescale is `⌊t·e/Q⌉` = **two full base conversions** and, at `L=8`, needs a **width-9 aux base** → the general-width FBC we have *not* built (the "next task", `docs/ir` notes). 

**For this benchmark we use the CKKS-style drop-prime rescale**: it matches CROSS's scheme, is the like-for-like op at this parameter set, and keeps every base conversion at width ≤ 8. The NTT / modmul / FBC-combine kernels are `Z_q`-linear and **identical** across schemes (established: only rescale differs) — so this is a kernel-faithful HE-Mult, just with CKKS's rescale rule. We validate **bit-exact against a CPU reference of the same arithmetic** (not a full CKKS decrypt — this is a compute/latency benchmark, not a scheme bring-up).

> Open question Q1 (see §6): confirm CKKS-style rescale is acceptable, or require BFV `⌊t·e/Q⌉` (which pulls in the general-width FBC as a prerequisite).

### 1.2 Parameter budget
- Ciphertext base `Q` = 8 limbs (28-bit, NTT-friendly `k·2N+1`).
- Key-switch: `dnum=2` → α=4 limbs/digit; special base `P` = 4 limbs (dominates a 4-limb digit) → `QP` = 12 limbs.
- Base-conversion widths: mod-up `4→8`, mod-down `4→8` (both `n│8`, on the validated FBC path).
- `dnum` is a knob — `dnum=2` (fewer, wider digits → fewer key-switch NTTs) mirrors CROSS's own low-dnum preference; `dnum=4` (α=2, P=2) is the alternative.

---

## 2. Kernel + host design — 32 cores

### 2.1 Reused, already-validated kernels
- **NTT/INTT**: four-step `matmul → twiddle → matmul` + on-device reindex (`fused_matmul_{a,c}`, `modmul_compute`, `transpose_tile`). Need **N=8192** (see §2.4).
- **Pointwise**: `modmul_sfpi_full` (eval-domain ×), `modadd_reduce`, `modsub_reduce`.
- **Base conversion**: `baseconv_fbc` (CRNS-combine, width 4 & 8) — the key-switch bconv, as in `bfv_relin_dnum.cpp`.
- **Key-switch**: the `dnum=2` structure from `bfv_relin_dnum.cpp`, scaled to L=8 / QP=12.

No **new** compute kernel is required for the op graph (confirmed by the dnum analysis) — the work is (a) N=8192 NTT, (b) the 32-core host orchestration + scheduling.

### 2.2 Two-level parallelism

**Level 1 — intra-op (one HE-mul across a core-group).** Parallel axes, in priority order:
1. **Limb parallelism** — the strongest axis. Every stage is per-limb independent: tensor over `L=8` limbs, key-switch over `QP=12` limbs (× 2 digits × {evk0,evk1}). That's ≥ 8–24 independent per-limb NTT/modmul pipelines per multiply → trivially fills a core-group.
2. **Poly/accumulator parallelism** — the 4 tensor inputs and the 2 key-switch accumulators are independent.
3. **Four-step internal** — the 8192 NTT's row/col matmuls are themselves tile-parallel (secondary).

→ Map one HE-mul's DAG onto a **core-group of `G` cores** (G = 4 or 8) via the compiler scheduler (`schedule.py`, HEFT-lite: rank by critical path, place on earliest-finishing core). Aim for balanced occupancy of both the **FPU** (matmul/NTT) and **SFPU** (modmul/FBC) per core.

**Level 2 — batch (fill the 32-core grid).** Tile `⌊32/G⌋` independent HE-muls across disjoint core-ranges in **one program** (the proven co-schedule pattern → 2.00× spatial, and batched codegen 57.9× at 64 cores). Batch `B = 32/G` (e.g. G=8 → B=4; G=4 → B=8). Report **amortized latency = device-active-time / B**.

**Utilization levers** (the current single-core `ct_mul` is *dispatch-bound* — 501 dispatches, `kernels.md §3`):
- **Fuse dispatches** — collapse per-limb single-tile dispatches into wide multi-tile / multi-core dispatches (the batched `stage7_run.cpp` form). This is the main throughput lever.
- **Buffer-resident chaining** — keep NTT→pointwise→INTT in L1/DRAM across a stage (already done in relin), avoid host round-trips.
- Pick `G` empirically to maximize `cores_busy × engine_occupancy` while keeping per-op latency acceptable.

### 2.3 Host structure
Reuse the `codegen.py → codegen_run.cpp` path: build the HE-mul as a `Program`/schedule, replicate across the grid for the batch, one `EnqueueMeshWorkload`. Correctness gate: **bit-exact vs CPU reference** (per-limb RNS), as every device block in this repo is validated.

### 2.4 N=8192 NTT
`8192 = 2^13`. Four-step factor e.g. `64×128` or `32×256`, reusing the **NTT-32768** groundwork (`docs/matmul/NTT_32768_*`). This is the main net-new compute piece (current on-device NTT is 1024). Twiddles/ψ baked per prime; reindex on the matrix engine.

---

## 3. Measurement methodology

### Phase 1 — latency (primary)
- Build with `TT_METAL_DEVICE_PROFILER=1`; read device-active time from `…/profiler/.logs/profile_log_device.csv` (the project's standard flow, `README.md`, `ct_mul_perf.md`).
- Warm run (JIT excluded); ≥ 50 timed enqueues; steady-state.
- **Report `µs / HE-mul` = total device-active-time / batch B.** Also per-stage split (tensor / rescale / key-switch) and per-RISC (FPU vs SFPU) to mirror CROSS Fig. 12.

### Phase 2 — energy / perf-per-watt (best-effort)
- Sample n150 board power via `tt-smi` telemetry during a sustained multiply loop; `energy/op = avg_power × amortized_latency`.
- Report **perf-per-watt (`HE-mul/s/W`)** — scale-invariant, so it sidesteps CROSS's iso-power core-scaling (Table VIII scales TPU cores to match each baseline's power; we instead report efficiency directly).
- Caveats to note in results: `tt-smi` sampling granularity, whole-board vs core power, host-idle subtraction.

### Caveats (state explicitly in the writeup)
- **Scheme**: CKKS-style rescale (§1.1); note the BFV-rescale delta.
- **dnum**: we use 2, Table VIII's Set B uses 3 — note it.

---

## 4. Milestones

| # | milestone | gate |
|---|---|---|
| **M1** | HE-mul op graph at `N=8192, L=8, dnum=2`, **single core-group**, correct | bit-exact vs CPU ref (all seeds) |
| **M2** | N=8192 NTT on device (four-step, reindex) | bit-exact vs CPU NTT |
| **M3** | Intra-op parallel across `G` cores; single-op latency measured | valid + latency recorded, `G` swept |
| **M4** | Batch-tile to 32 cores; **amortized latency** measured | amortized latency recorded |
| **M5** | Dispatch fusion / batched codegen; re-measure | utilization ↑, latency ↓ |
| **M6** | Energy / perf-per-watt via `tt-smi` | perf-per-watt recorded |

---

## 5. Risks & open questions

**Risks**
- **N=8192 NTT** is the main net-new compute (mitigated: NTT-32768 groundwork exists).
- **Dispatch-bound**: without fusion, 32 cores won't help much — M5 is load-bearing for a good number.
- **Memory**: `N=8192 × QP=12 limbs × batch` L1/DRAM footprint; may cap batch B or G.
- **tt-smi power** granularity may make energy noisy (hence best-effort / phased).

**Open questions for you (Q1–Q3)**
- **Q1 (scheme/rescale):** OK to use **CKKS-style drop-prime rescale** (matches CROSS, avoids the width-9 general FBC)? Or require **BFV `⌊t·e/Q⌉`** (adds general-width FBC as a prerequisite — bigger scope)?
- **Q2 (core-group size G):** preference for **G=8** (B=4, lower single-op latency) vs **G=4** (B=8, more batch/throughput)? Default: sweep both, report the better amortized number.
- **Q3 (correctness bar):** bit-exact vs a **CPU reference of the op graph** (compute benchmark) sufficient, or do you want a **full CKKS encrypt→mul→decrypt** correctness wrapper too (larger, scheme bring-up)?

---

## 6. Result table skeleton (to fill in)

| metric (Set B, N=2¹³, L=8) | **Tenstorrent** |
|---|---|
| HE-Mult amortized latency (µs) | **14,090** (p150, 64 cores, device-busy — see note) |
| perf-per-watt (mul/s/W) | **TBD (M6)** |
| stage split (tensor/rescale/ks) | **TBD** |
| cores used | up to 64 |

### Measured so far (component: NTT-8192, the dominant HE-Mult sub-kernel)

| NTT-8192 (N=2¹³, single 28-bit limb) | latency | dispatches | cores | KNTT/s |
|---|---|---|---|---|
| naive single-core (`nntt8192_dev`) | 2466 µs | 88 | 1 | 0.4 |
| **multi-core fused (`nntt8192_mc`)** | **137 µs** | **4** | 64 | **7.3** |

**Read:** dispatch fusion + grid placement gave **18×** (2466→137 µs), bit-exact. Still dispatch-bound
(4 dispatches × ~34 µs). The remaining levers are residual dispatch overhead (→ fuse the 4 stages
further / kernel-fuse) and throughput batching.

### End-to-end Set-B HE-mul (M1) — `host/experiments/legacy/he_mul_setb.cpp`

Full pipeline **tensor → dnum=2 key-switch → CKKS drop-prime rescale** at N=8192, L=8, dnum=2, P=4
(QP=12), all heavy arithmetic on device via `transform8192` + `pointwise8192` + `fbc8192`.
**Validated bit-exact vs a CPU oracle: 0/114688** (all seeds). Result: **5.37 s/mul wall-clock** on
n150.

The 5.37 s is a **correctness-first, unoptimized orchestration**: every one of ~190 device ops per
mul (104 NTT/INTT + 80 pointwise + 4 FBC) does its own host-side twiddle-constant regeneration +
synchronous readback. That host + readback time — **not device compute** — dominates. Device-active
compute is ~tens of ms (≈104 transforms × ~137 µs). Optimization levers (all identified, none blocking
correctness): (1) **cache twiddle/fold constants** per prime (regenerated every call today — the
single biggest cost); (2) **buffer-resident chaining** (keep data on device across ops, kill readbacks);
(3) **cross-transform dispatch fusion** (§2.2 / the 18× NTT win); (4) batching + perf-per-watt (M6).
Correctness and the full op-graph structure are done; the number is an unoptimized upper bound.

---

## 7. Experiments & findings (measured, all bit-exact)

Two HE-mul implementations at Set B (N=8192, L=8, dnum=2, P=4), both validated **bit-exact vs a CPU
oracle (0/114688)**:
- `he_mul_setb.cpp` — **coeff-resident** (tensor NTTs its 4 inputs). Heavier than CROSS.
- `he_mul_cross.cpp` — **CROSS-aligned**: eval-resident, tensor = pointwise (**0 NTT**, like CROSS's
  `polynomial_mult`), drop-prime rescale first, dnum key-switch eval-in/eval-out. Same HE sub-kernel
  DAG as CROSS (only the NTT *internals* differ). `NTT_CORES` env pins a fixed core count.

### 7.1 Sub-kernel workload vs CROSS (per mul)
| stage | CROSS (hemul.py) | he_mul_cross | match |
|---|---|---|---|
| rescale (first) | drop-prime, ~4 INTT + 4(L−1) NTT | 4 INTT + 28 NTT | ✅ structure |
| tensor | pointwise, **0 NTT** | pointwise, **0 NTT** | ✅ |
| key-switch | INTT(e2) + dnum·[BConv→NTT-comp→×evk] + mod-down | same (eval-in/out, own limbs reuse eval e2) | ✅ structure |
| combine | ⊕ (eval) | ⊕ (eval) | ✅ |
| dnum | 3 (Set B) | 2 | ⚠️ residual |

Aligned totals ≈ 57 NTT + 33 INTT + 4 BConv + 72 ⊙ per mul. Only residual mismatch: dnum (2 vs 3).

### 7.2 CROSS-aligned HE-mul latency (bit-exact, all seeds)
| config | ms/mul (amortized) |
|---|--:|
| 1 mul, 4 cores | 5926 |
| 1 mul, 8 cores | 5193 |
| 1 mul, 16 cores | 5044 |
| 1 mul, 64 cores | 4651 |
| **4 muls sequential, 16 cores** | **5096** (≈ 1 mul) |

coeff-resident `he_mul_setb` (64 cores): 5374 ms → CROSS-aligned drops it to 4651 ms (removing the 56
tensor transforms, ~13%).

### 7.3 Findings
1. **Core count barely matters** (4→64 cores = only 27%): not device/dispatch-bound at the mul level.
2. **Repetition doesn't amortize** (4 sequential muls ≈ 1 mul/each): the single-threaded host serializes
   muls (per-op `Finish` + constant regen + readback).
3. ⇒ The mul is **host-orchestration-bound and serial**, ~90% host (twiddle-constant regeneration +
   synchronous readbacks), device-active ≈ tens of ms. The bottleneck is **orchestration**, NOT
   arithmetic (the sub-kernel workload matches CROSS's, §7.1).

### 7.4 Consequence for batch/throughput ("4 muls on 16 cores, 4 each")
Spatial concurrency alone can't beat the host serialization → amortized ≈ 1 mul. A real throughput
multiplier needs **all three**: (a) **shared constants** (same primes → generate twiddles once, not
per-instance-per-call), (b) **batched dispatch** (one workload / one `Finish` per op-step across the B
core-groups so device work overlaps), (c) **buffer-residency** (kill the ~150 synchronous readbacks).
These are the same levers that fix the single-mul latency; batching then amortizes the residual host
cost across B muls. Constant-caching is the prerequisite and likely the single biggest win.

### 7.5 NTT component
(n150. Same benchmark on Blackhole p150: 80.8 µs @32 cores / 68.9 µs @64 cores, ~2× —
see [blackhole_port.md](blackhole_port.md) §3.)

Standalone NTT-8192 (`nntt8192_mc`, bit-exact): 2466 µs single-core → **137 µs at 64 cores / 162 µs at
32 cores** (18× via grid placement + fused dispatch), still dispatch-bound (latency ≈ #dispatches ×
~34 µs). Factorization note: TT
favors one factor = tile size (32), keeping stage-2 modadd-free; CROSS's 128-wide layout suits the
TPU MXU (hardware-dependent optimum).

---

## 8. Step-by-step latency optimization

**Owned by [`history/OPTIMIZATION_REPORT.md`](history/OPTIMIZATION_REPORT.md)** — the full ladder,
per-step attribution and per-phase host timing live there, and are not restated here so the two
cannot drift apart.

Summary: **5044 → 727 ms** by hoisting twiddle generation (~85% of the per-op host gap) and making
constants resident, then **→ 38 ms device-busy** via the traced-resident path. Today on Blackhole
p150: **14.09 ms/mul device-busy, 24.43 ms host wall** at 64 cores, bit-exact `0/114688`
([`blackhole_port.md`](blackhole_port.md) §3). The last step, 20.85 → 14.09 ms, was the 2026-08-03
dispatch-bound pass — batched reads, wider matmul waves, one fused reduce+twiddle
([`blackhole_port.md`](blackhole_port.md) §5).

Those are not directly comparable — different hardware *and* a different code state (the report was
measured at 102 trace segments / 2065 cached workloads; the current code produces 46 / 1219).

---

## 9. Running it — modes and env flags

`sh tools/bench/hemul_cross.sh [seed]` is the supported entry point: it builds, runs the
traced-resident path at `NTT_CORES=16`, and prints the two headline numbers plus a device-trace
breakdown. Everything below is for driving `./build/he_mul_cross` directly.

| flag | default | what it selects |
|---|---|---|
| `NTT_CORES=N` | full grid | cores per ct_mul (4 / 8 / 16 / 64). **Unset means the whole grid** — 13×10=130 on p150, which is slower, not faster (§7.4). The bench script pins 16. |
| `HE_ITERS=N` | 20 | replay iterations to average over |
| *(none)* | — | plain run: build + dispatch per op, correctness-first. Slowest, simplest to read. |
| `HE_RES=1` | — | buffer-resident intermediates, no host round-trip per op |
| `HE_RES_CACHE=1` | — | as `HE_RES`, plus cached workloads (build once, re-enqueue) |
| `HE_TRACE2=1` | — | **the headline path**: resident + Metal Trace capture/replay (REC→CAP→REPLAY). Reports host wall *and* device-busy span. |
| `HE_RT=1` | — | fwd→transpose→inv round-trip on one prime, no host round-trip — an NTT-only sanity path |

Profiling (all additionally need `TT_METAL_DEVICE_PROFILER=1`):

| flag | what you get |
|---|---|
| `HE_PROF2=1` | one clean single-replay trace. **Incomplete by construction** — the device-side profiler buffer overflows partway through a ct_mul and drops markers. |
| `HE_PROF_DRAIN=1` | the **full** trace, no dropped markers: drains the profiler every `HE_DRAIN_K` dispatches (default 100). On the same 16-core workload this recovers ~4.7× the pointwise-modmul calls and 2× the FBC calls that `HE_PROF2` loses. Use this whenever a breakdown has to be complete. |

The analyzers that read the resulting CSV are in [`../../tools/traces/`](../../tools/traces/README.md).

**Eval-form ciphertext I/O.** The resident path here converts its four input polynomials coeff->eval
on device at the top of every multiply (32 forward NTTs, 26% of all transforms) even though the DAG
is otherwise eval-resident. [`exp_tb001.md`](exp_tb001.md) is the same algorithm and parameters with
eval as the default I/O format instead: 636 vs 764 dispatches, 11.27 vs 14.09 ms/mul device-busy,
bit-exact. This program is unchanged and remains the reference.

**Core count and the elementwise ops.** The 8-tile-block ops (mod-add-reduce, twiddle, transpose,
FBC, pointwise) run on `EWC = min(8, NTT_CORES)` cores, looping when there are more blocks than
cores. At `NTT_CORES ≥ 8` that is one full-width pass, unchanged; below 8 it is what keeps a run
from spilling onto cores it does not own — without it a 4-core profile is polluted by 8-core zones.
