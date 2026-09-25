# HE compiler (`compiler/`)

The Python compiler for the HE workload. Design: `docs/ir/formalization.md` + `ops.md`.
Device kernels the codegen targets: `kernels/ir/`. Everything here self-validates —
run any file with `python3 <file>.py`.

> ## Start here: `python3 he.py`
>
> **One script does everything** — compile a CKKS `ct_mul`, run the four optimisation passes, schedule
> it across the grid, emit a dispatch plan, build and dispatch a host on the p150, verify bit-exact,
> write the visual tool and serve it. Config block at the top of the file; `--case NAME` overrides it.
>
> ```
> python3 he.py --list                  # the demo cases
> python3 he.py --case mid              # 2.3 s, everything (best all-round demo)
> python3 he.py --case tile32           # 12.2 s, the biggest pass ladder (1.72x)
> python3 he.py --case mid --all-primes # 100% of the pointwise ops on device, verified
> ```
>
> Full write-up, measured timings and honest limits: **[`../docs/compiler/he_compiler.md`](../docs/compiler/he_compiler.md)**.

## What was added on top of the stage 1-7 pipeline below

| file | role | validated |
|---|---|---|
| `he.py` | **the entry point** — all six stages, serves the viz, prints artifact paths | 6/6 demo cases end-to-end ✅ |
| `hw.py` | Blackhole **p150** model with an ENGINE dimension; supersedes `cost_model.py`'s n150 seeds | **ladder gate: retrodicts the measured 4-rung hardware ladder to 0.2%** ✅ |
| `passes.py` | the four mapping passes + pass manager + mapping-aware scheduler | meaning-preserving after all four ✅ |
| `viz_app.py` | pass ladder + engine-coloured tile DAG + scheduling trace, self-contained | serves, payload validated ✅ |
| `ckks_gate.py` | CKKS front-end gate | decrypt exact + `run(L1)==run(L2)==run(L3)` ✅ |
| `_paths.py` | the one repo-layout resolver (the restructure had broken 9 of 10 modules) | — |
| `sweep_cases.py` | time candidate configs, see which passes fire | — |

The four passes, each a rung of the **measured** ladder in `../docs/he/exp_tg018ntt.md`:

| pass | decides | measured rung |
|---|---|---|
| `place_comm` | edge memory tiers; keeps the NTT corner turn core-to-core | 50.863 → 43.932 ms |
| `fuse_dispatches` | merges chains into one dispatch, marks operands resident (reuse pricing) | 43.932 → 32.448 ms |
| `select_engines` | per-op SFPU vs matrix engine | 32.448 → 23.991 ms |
| `legalize` | kernel variant + params; 256-arg budget, `Ld<=24`, L1 working set | — |

**CKKS front end** (`../reference/he/ckks_mul_ref.py` + `builders.build_ct_mul_ckks`):
tensor → relin → add → drop-prime rescale, coefficient domain, validated against an *exact* negacyclic
integer product. BFV is untouched and still works.

| file | role | validated |
|---|---|---|
| `reference.py` | exact op-semantics golden (L1/L2/L3), the bit-exact oracle | 15 checks ✅ |
| `ir.py` | `Program`/`Op`/`Value` + `verify` + `run` (+ `macro`) — the program representation | 4 checks ✅ |
| `cost_model.py` | mapping cost model (seeded from measurements) | self-check ✅ |
| `builders.py` | **stage 2**: build real workloads as L1 programs | he.add + ct_mul **decrypt** ✅ |
| `lower.py` | **stages 3 + 4 + 5**: L1→L2 (parametrized: NTT plan + tile size `T`), L2→L3 (int8), Fusion A | meaning-preserving ✅ |
| `schedule.py` | **stage 6**: placement + scheduling across cores (greedy HEFT-lite, p=1) | valid + parallel ✅ |

## Pipeline status (docs/ir/FORMALIZATION §1.5)

| Stage | What | Status |
|---|---|---|
| 2. → L1 | build workload as L1 program | ✅ **full ct_mul in primitives** — `build_ct_mul_rns` (tensor over Q∪S → rescale → relin → add), rescale/relin decomposed into `rns.*`; base_convert = CRNS (exact + approximate). Decrypts to `m₁·m₂`, `run(L1)==run(L2)==run(L3)`, all seeds. (`build_ct_mul` with `macro` nodes kept as the composite reference.) |
| 3. L1 → L2 | tile ops; NTT → matmul(s) | ✅ `lower_l1_to_l2(factors, T)` — **parametrized**: NTT factorization `plan` (four-step) + tile size `T` (`tile_matmul`), both meaning-preserving at N=64 (`plan=(8,8)`, `T=32`, and combined) |
| 4. L2 → L3 | matmul_modq → int8 (decompose + planes + recombine) | ✅ `lower_l2_to_l3`, `run==L1` |
| 5. Fusion A | tile packing | **limb-major** (the one that matters at real N): pack `32/L` independent `L`-limb reductions into one 32-row tile via the multi-Q kernel. **On hardware: 7.99× at L=4, 4.00× at L=8** (near-ideal), packed results correct — `kernels/ir/fusionA_limb.cpp`. (Coefficient-direction `fuse_a` is a toy-N mechanism demo; full tiles at N≥1024.) IR-pass wiring pending limb-major ops. |
| 6. Placement + scheduling | greedy HEFT-lite (p=1): rank by critical path, place on earliest-finishing core | ✅ `schedule.py` — tensor N=64: 3.95×/7.65× on 4/8 cores, balanced, valid, meaning-preserving |
| 7. Codegen | emit placed schedule → device dispatch | ✅ **general DAG**: `codegen.py` (Program+schedule → spec) + `kernels/ir/codegen_run.cpp` (buffer/Value, dependency waves, ops across cores, **bit-exact vs golden 0/65536**). Batched form `stage7_run.cpp`: **57.9× at 64 cores, near-linear; scheduler predicts within ~1%**. Per-op codegen is dispatch-bound (→ fuse into wide dispatches); structural-op / matmul codegen is the remaining generalization. |

**Validation method:** every lowering is checked by re-running the golden and diffing —
`run(L1) == run(L2) == run(L3)`, bit-identical. Lowering is a *proven* refinement, not a hope.

## Base conversion (CRNS) — the core RNS primitive

`base_convert` is `rns.base_convert` with two modes (`docs/ir/crns_bconv_lowering.md`), both realized
as **matmul + elementwise** (the SFPU has no bignum):

- **exact** (default) — CRNS (Cox-Rower / Kawamura), m_sk-free: base-extension matmul (`Σ rᵢ·Eᵢⱼ`) +
  a wide accumulator `s = matmul_wide(F, RES)` sliced (`k = s>>u`, sign `= s>>(u-1)`) + the `k·gⱼ −
  neg·(M mod dⱼ)` correction, done **elementwise** (k stays off the matrix engine, so its wide/
  multi-chunk value — up to ~34 bits at ~60 moduli — never needs int8 decomposition).
- **approximate** (`exact=False`) — BEHZ fast base conversion: pre-scale (elementwise) + base-extension
  matmul, dropping the ε overflow (absorbed into HE noise). Used by rescale mod-up/down and relin.

The wide `matmul_wide` lowers to int8 at L3 via `recombine_wide` (place-value combine, no reduction);
everything else reuses `decompose` / int8 `matmul_modq` / `recombine_modq`. All validated
`run(L1)==run(L2)==run(L3)`.

## Remaining work

- **Cost-model seeding (stage 6):** `schedule.op_dur` prices `tile.matmul_wide` / `tile.fixed_slice`
  at the 0.5 µs default — the full-ct_mul schedule is valid + parallel, but the makespan estimate is
  approximate until these are measured/seeded.
- **Tiling at L3 (perf):** `tile_matmul` tiles at L2 (tile-then-decompose); the decompose-then-tile
  reorder (`docs/ir/crns_bconv_lowering.md §5`) is a real-N performance change, not a correctness one.
- **Stage-7 codegen** for the CRNS/structural ops (`matmul_wide`, `fixed_slice`, per-limb `base_convert`).

## The other entry points, and which viz is which

`he.py` above is the one to start with. Six of the 23 modules are runnable, and the README only
named two of them, which is the actual source of confusion in this directory:

| run this | for |
|---|---|
| **`he.py`** | the demo: all six stages, serves the viz, prints artifact paths |
| `run_he.py` | one-shot driver — workload params in, artifacts out, no viz. What `USAGE.md` documents |
| `he_compile.py` | compile a `.he` program for a named config: `python3 compiler/he_compile.py compiler/configs/n1k.json compiler/programs/ct_mul.he`. Writes into `compiler/generated/<prog>_<cfg>/` |
| `sweep_cases.py` | time the candidate demo configs and pick one |
| `codegen_draft.py` | the DRAFT stage-7 dispatch emitter. **Not** part of the pipeline — nothing imports it, and it does not compile against tt-metalium. Turning it into a real emitter is listed as remaining work in `docs/ir/state.md` §5, which is why it is kept |

Four viz modules, and `viz.py`'s own docstring claiming it is "ONE app" predates two of them:

| | |
|---|---|
| `viz_app.py` | the app `he.py` serves — pass ladder + engine-coloured tile DAG + scheduling trace |
| `viz.py` | the older combined app; still used, via `viz_bridge.py` |
| `viz_bridge.py` | renders the new mapping through `viz.py`/`sched_viz.py`; used by `he_compile.py` |
| `mlir_viz.py`, `sched_viz.py` | the DAG viewer and the Gantt viewer — components, imported by the apps above rather than run directly |

## Layout: source at the top, data in four directories

The top level is source only — the 23 modules, three docs and `run.sh`. Everything else is in a
directory that says what it is:

| | |
|---|---|
| `artifacts/` | **committed** compiler output: the ten `example_*.mlir` (readable IR at L1/L2/L3, written by `mlir.py`) and `codegen_spec.bin` (written by `codegen.py`/`run_he.py`) |
| `configs/`, `programs/` | compiler **inputs** — hardware/workload JSON, and `.he` programs |
| `generated/` | per-run output of `he.py` / `he_compile.py`. **Gitignored** |

`codegen_spec.bin` is committed because `host/tests/ir/codegen_run.cpp` reads
`compiler/artifacts/codegen_spec.bin` as its default path — a device test depends on it. It is a
different file from `generated/codegen_spec.bin`, which `he.py` writes per run; do not "dedupe" them.
The `.mlir` are committed for the same reason `reference/ntt/`'s are: they are the readable output,
and reading the IR then needs no Python.

Not committed: the viz HTML, the draft dispatch plan, and `cases.json` (7.0 MB of it used to be
tracked). Nothing reads any of them, and `generated/` was already gitignored for the same kinds of
file. Regenerate with `viz.py` / `mlir_viz.py` / `sched_viz.py` / `codegen_draft.py` /
`sweep_cases.py`.
