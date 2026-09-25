# Documentation index

Start at the [root README](../README.md) for what this project is, prerequisites and a quick start.
This page maps the rest.

> **Working here for the first time?** [`guide/`](guide/) is the normative pair —
> [`profiling.md`](guide/profiling.md) (how a number is taken and what makes it quotable) and
> [`optimization.md`](guide/optimization.md), which is an index over three domain guides
> ([host](guide/optimization_host.md) / [DRAM](guide/optimization_dram.md) /
> [kernel](guide/optimization_kernel.md)) in order of attack. They
> carry no new measurements; every figure links the doc that owns it.

> **How these docs are organised.** One design doc plus one perf doc per subsystem, with superseded
> material moved (never deleted) into that topic's `history/` and marked `Status: Historical`. So the
> entry point for a subsystem is its `README.md` or `*_overview.md`, and anything under `history/` is
> there because something newer replaced it: `matmul/` is 7 live against 11 historical, `vector/` 3
> against 5, and the NTT-1024 cluster that used to sprawl is 2 live against 4.
>
> **Measured data lives in `<topic>/data/`.** Every CSV an experiment emits is there — 31 under
> `he/`, 4 under `noc/`, 3 under `matmul/`, 1 each under `bconv/` and `vector/` — so a topic folder
> shows its prose and nothing else. The `data_` prefix is gone from the filenames because the
> directory now says it: `docs/he/data/modmul001.csv`, not `docs/he/data_modmul001.csv`.
>
> One thing is not uniform yet: a `Status:` header. 31 of 92 docs carry one. `he/` is the least
> consolidated topic even after the exp_Tg018 lineage was archived — 19 live docs against 7 in
> `history/`.

> **Reading performance numbers.** Most figures in these docs were measured on Wormhole **n150** and on
> earlier code. The current hardware is Blackhole **p150**. They are not directly comparable — see
> [he/blackhole_port.md](he/blackhole_port.md) §3 for why, and the root README for verified p150 results.

**Naming.** Content docs are `lower_snake_case.md`; `README.md` keeps capitals as the repo-level
index. Superseded material lives in each
topic's `history/`, marked `Status: Historical` — never deleted.

## Where to start, by interest

| I want to… | read |
|---|---|
| **take a measurement** | [guide/profiling.md](guide/profiling.md) — the four metrics, the host split, the example instrument, the traps |
| **make something faster** | [guide/optimization.md](guide/optimization.md) — order of attack, the measured ladders, and the null results not to repeat |
| see HE working end to end | [he/README.md](he/README.md) — BFV evaluator: encrypt, add, multiply, decrypt |
| build / set up a machine | [he/blackhole_port.md](he/blackhole_port.md) §1 — install flow, config, env |
| debug a device hang | [he/blackhole_port.md](he/blackhole_port.md) §2 + [debugging.md](debugging.md) |
| understand the arithmetic stack | [vector/](vector/) → [matmul/](matmul/) → [bconv/](bconv/) → [he/](he/) |
| understand PIR | [pir/pir_demo.md](pir/pir_demo.md), then [pir/pir_perf.md](pir/pir_perf.md) |
| understand the HE compiler | **[compiler/he_compiler.md](compiler/he_compiler.md)** (the entry point, the four mapping passes, the p150 cost model and its ladder gate, the demo cases) + [ir/](ir/) for the IR design |
| profile a device run | [../tools/traces/README.md](../tools/traces/README.md) |

## By subsystem

**`guide/` — how to work here.** Two normative documents, distilled from the subsystem docs
and meant to be read first. **`profiling.md`**: the four latency metrics and their guaranteed
orderings, the host-phase split (BUILD / ENQ / WRITE / READ) and what each dominant term implies, a
copyable `prof::` instrument, device-trace capture including the buffer-overflow and host↔trace-join
gates, the provenance tags, and the traps that have already cost a run each. **`optimization.md`**:
the order of attack, then host side (workload cache → residency → trace → co-scheduling, 3450 →
103.8 ms), DRAM/NoC (the six measured ceilings, the 433 GB/s recipe, spread-for-bandwidth /
pack-for-latency), and kernel side (SFPI pass fusion, DST residency, the floors, and what does *not*
pay on a dispatch-bound workload). Both carry the null and refuted results explicitly.

**`vector/` — modular multiplication (the bottom of the stack).** Montgomery modmul on the SFPU.
**`modular_multiplication.md`** is the single current doc: design, measurements, and how to run it.
**`exp_roofline001.md`** is the roofline for the four primitives the repo issues — the Montgomery
modmul, a bare multiply and a non-multiply op — each measured with **no data reuse** and with **full
reuse**, and swept between the two. Its headline is that the **modmul's ridge is R*=1.7**: streaming, it
is memory-bound at 86% of the memory roof and only 51% of its own compute roof, which is why fusion
rather than kernel work has been the lever. **`exp_bw001.md`** is the machine-limit companion: a *non*-modular int32 elementwise multiply run as
one traced dispatch over 112 cores, giving the SFPU ceiling (~3.4 mul/cycle/core, 518 Gmul/s) and the
DRAM ceiling (~322 GB/s) that every kernel above it is bounded by, plus the core-scaling knee.
Superseded records (the plan, the full report, the roofline analysis, the old device README) are in
`vector/history/`.

**`matmul/` — int8 modular matmul and NTT.** `modular_matmul_overview.md` (32×32 modular matmul on the
matrix engine) and **`ntt_1024.md`** (design) + **`ntt_1024_perf.md`** (the single owner of all
NTT-1024 measurements) are the entry points. `NTT_32768_*` cover the larger transform and its variants;
`PLAN.md` is the original plan.
**`exp_ntt001.md`** is the p150 re-measurement of 3-factor vs 2-factor at N=2¹⁵ — the two metrics
disagree (3-factor wins on kernel time 121 vs 146 µs, 2-factor on host wall 153.6 vs 184.0 µs,
because of 12 dispatches against 7), and it prices the stage fusion that would settle it.
Superseded per-version write-ups are in `matmul/history/`.

**`bconv/` — RNS base conversion.** `base_conversion_overview.md`: approximate, exact
(Shenoy-Kumaresan) and CRNS fixed-point variants. **`exp_fbc001.md`** measures the SFPU against the
matrix engine with **both** routes as one fused dispatch and no DRAM during the computation, swept over
`(Ls,Ld)`: the matrix engine is flat (`435 + 0.68·Ls + 0.06·Ld`) where the SFPU pays `Ls·Ld`, and it
wins every cell but `8→8`. It also records that fusing the scale in is a *loss* below `Ls ≈ 24`.

**`noc/` — the fabric against DRAM.** **`exp_noc001.md`** measures a personalised all-to-all three
ways at matched correctness: through DRAM reading whole pages (`O(G²)` traffic, the shape the corner
turn had), through DRAM reading only the needed bytes (`O(G)`, same bytes as the fabric), and
core-to-core L1→L1. Splitting those three is what keeps the medium comparison honest — **most of the
corner turn's 5.4× was the access pattern, not the medium**: strided DRAM is 10.6–15.8× faster than
whole-page DRAM, and *beats* the core-to-core exchange (2.70 vs 4.23 µs) at the corner turn's own
parameters. The exchange is barrier-bound (half its cost is a `0.224 + 0.0402·G` µs barrier), and the
fabric wins on raw rate (~1.5× DRAM at matched transfer size) yet only wins the route beyond `G≈45`.

**`he/` — BFV ciphertext operations and the full multiply.**
`README.md` (evaluator overview + the demo), `kernels.md` (per-kernel correctness + device profile),
`ct_mul_perf.md` (latency distribution of a full `ct_mul`), `cross_comparison.md` (the Set-B
HE-multiply benchmark at CROSS Table VIII's parameters), **`exp_tb001.md`** (the Set-B cost model at a fixed
`G=8` workload, 16.4 → 7.49 ms critical path busy, plus the ct_add), **`exp_tb001b.md`** (a
**decrypt-verified** BFV multiply at the same ring — the whole multiply on device, bit-exact and
`decrypt(a ⊗ b) == a·b`), **`exp_tg000.md`** (a **decrypt+decode-verified CKKS** multiply on the same
kernels — canonical embedding, `dnum=2` relinearisation, drop-prime rescale; 7.60 ms critical path
busy at 45 bits of precision, and the measurement that found where the precision was going),
**`exp_dram001.md`** (six proposed DRAM levers tested: **five do nothing or hurt**, and the sixth —
both NoCs reading — was already what `dram_guide` does for its 433. Its content is the negative
results: the one-NoC wall sits at ~238 GB/s and is **not tunable** by packet size, in-flight depth,
bank affinity or core count), **`exp_tg018m.md`** (the same CKKS multiply scaled to `N=65536`, `L_Q=48`, `dnum=3`: `238.663 →
50.920 ms` from the DRAM bank-allocation fix, plus the verified per-kernel census), **`exp_tg018ntt.md`**
(the current best — **50.863 → 23.991 ms, 2.120×**, by taking the NTT's corner turn core-to-core,
fusing all six NTT passes into ONE dispatch with no intermediate in DRAM, and moving the base
conversion off the SFPU onto the matrix engine; 2884 → 1278 dispatches and host wall 155 → 42 ms), `roadmap.md`, and **`blackhole_port.md`** (the p150
port: build flow, the config layer, and the mid-kernel re-init deadlock). The build record and the
Set-B optimization ladder are in `he/history/`.

**`pir/` — private information retrieval.** `pir_demo.md` (end-to-end retrieval),
`pir_kernel_plan.md`, `ONION_MATCH_PLAN.md` / `onion_match_perf.md` (matching OnionPIR),
`pir_perf.md`, `latency_optimization.md` (the largest doc here, the latency work).

**`ir/` — the HE compiler.** `ops.md` (the op set), `formalization.md` (semantics),
`crns_bconv_lowering.md` (lowering base conversion), `state.md` (where the compiler stands).

## Repo-wide

| doc | contents |
|---|---|
| [debugging.md](debugging.md) | debugging device programs: watcher, profiler, common failures |
| *(provenance)* | `git log --follow <path>` — every move in this repo used `git mv`, so history is intact |
