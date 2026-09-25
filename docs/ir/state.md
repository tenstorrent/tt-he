# HE compiler — current state

Status snapshot of `compiler/` (+ `reference/he/` references). The pipeline compiles a **full BFV
ciphertext multiply** from RNS-poly ops down to int8 tile ops, schedules it across cores, and renders
it. Everything self-validates by re-running the golden (`run(L1)==run(L2)==run(L3)`) and/or **decrypt**.

Companion docs: `crns_bconv_lowering.md` (base-conversion design), `formalization.md` / `ops.md` (IR
spec), and `../../compiler/README.md` for the pipeline and its entry points.

---

## 1. What works end-to-end

**Full `ct_mul` as a primitive DAG** (`builders.build_ct_mul_rns`): tensor over Q∪S → rescale×3 →
relin → add, all `rns.*` primitives (no macro stand-ins). Lowers L1→L2→L3 and **decrypts to m₁·m₂**,
bit-exact, all seeds. Validated in `builders._validate_ct_mul_rns`.

Levels: **L1** RNS-poly (`rns.ntt/intt/mul/add/sub/base_convert`) → **L2** tile
(`tile.matmul_modq/matmul_wide/fixed_slice/mul_modq/add_modq/...`) → **L3** int8
(`decompose → int8 matmul → recombine_modq/recombine_wide`).

Parameter sweep (no bugs): build+lower+decrypt across limb counts L=(2,2,3)…(5,5,6), t∈{12289,40961,
65537}, N=16…512; build-only to L≈62 limbs and N=4096; single-DFT **and** multi-factor NTT.

---

## 2. Components (files + functions)

| area | where | notes |
|---|---|---|
| **L1 builder** | `builders.build_ct_mul_rns(pp, ct1, ct2, evk0, evk1 \| ks=(evks,groups))` | tensor `_emit_tensor_base` (over Q∪S), rescale `_emit_rescale`, relin `_emit_relin` / `_emit_relin_dnum`, add |
| **Lowering** | `lower.lower_l1_to_l2(prog, factors, T)` · `lower_l2_to_l3(prog)` | stage 3 (tile + NTT), stage 4 (int8). `T` tiles matmuls to ≤T×T |
| **Base conversion (CRNS)** | `lower._emit_base_convert_crns` (exact) · `_emit_base_convert_approx` (approx) | `rns.base_convert{exact}`. Exact = base-ext matmul + `matmul_wide` accumulator + `fixed_slice`(k) + **elementwise** k·g. Approx (BEHZ) = pre-scale + base-ext matmul, drops ε. See `crns_bconv_lowering.md`. Bounded-`u` via `attrs["u"]` (rescale passes ~40b, not worst-case ~600b) |
| **Multi-factor NTT** | `lower._emit_ntt_fourstep` | **any k factors** ∏=N, e.g. (32,32,32) for N=2¹⁵. Per-factor DFT_f·twiddle·transpose (à la `ntt_matmul/planner`), made negacyclic (psi pre/post-weight) + inverse = exact reverse of forward (guaranteed round-trip) |
| **dnum key-switch** | `he/bfv_relin_dnum.py` (ref) · `builders._emit_relin_dnum` | hybrid, Q split into `dnum` coprime groups; validated identity + decrypt for dnum∈{1,2,3}. `dnum ≤ #Q-limbs` (guarded) |
| **Fusion A** | `lower.fuse_a(prog, strategies)` · `FUSE_A_STRATEGIES` | tile packing: batch shared-const-weight matmuls. Strategies `"ntt"` (1-D RHS) / `"baseconv"` (2-D RHS), toggleable. Temporal-subset batching + weight-const-def ordering |
| **Scheduler** | `schedule.schedule(prog, P, reuse, cap)` | HEFT-lite greedy, p=1. Memory tiers SRAM/NoC (Fusion B). `reuse=True`: per-core LRU working-set (residency) → reused const/intermediate free, operand-affinity placement. Reports: `fusion_b_report`, `reuse_report` |
| **MLIR emitter** | `mlir.emit_mlir(prog, func_name)` | any L1/L2/L3 program → MLIR text (`rns.`/`tile.` dialect). Writes `example_*.mlir` |
| **Codegen draft** | `codegen_draft.py` | scheduled L3 → device dispatch plan (kernel per op, buffers, waves) → `ctmul_dispatch.{json,cpp.txt}`. Structural draft, not tt-metalium-compilable |

Interpreter (`ir.run`) is the exact golden (object/bignum). `Value`/`Op` are `slots=True` dataclasses
(memory/alloc win at high op counts). Verifier allows a raw (`q=None`) operand in elementwise ops
(the CRNS overflow `k`).

---

## 3. Tools (viewers)

- **`viz.py` — the unified app.** Params block at top (N, LQ/LP/LB, t, DNUM, NTT factors, TILE,
  FUSION_A, CORES, REUSE, MAX_NODES, VALIDATE, PORT). `python3 viz.py` builds + lowers + schedules +
  serves ONE page: tabs **L1 · L2 · L3 · L2+FusionA · L3+FusionA · Gantt**. DAG views (layered layout,
  click→operands/consumers, search); Gantt (cores×time bars, click→placement + SRAM/NoC handoff
  arrows, P sub-tabs). Prints per-phase **progress + timing**.
  - `MAX_NODES` — skip laying out / embedding any view above this op count; **`None` = no limit**.
  - `VALIDATE` — `None`=auto (decrypt-check only when N≤256). When off, **dummy zero keys/data**:
    the op-graph *structure* is identical, and `keygen`/`encrypt` (O(N²) `negacyclic_mul`) are
    **skipped** — the key perf lever at large N.
- Superseded single-view servers still present: `mlir_viz.py` (DAG only), `sched_viz.py` (Gantt only).

---

## 4. Key findings

- **The scheduler has no real data / arithmetic** — purely structural (opcodes, deps, cost-model
  durations → core+time). Real computation only in `ir.run` (decrypt check), skipped at large N. So
  `keygen`/`encrypt` are needed *only* for validation ⇒ dummy zeros for the scheduling/viz path.
- **Fusion A is a NO-OP under the four-step NTT** (164→164) vs single-DFT (97→56), across all
  limb/dnum. Cause: four-step matmuls are const-LHS `Wn1@Y` (temporally blocked — depth-first
  emission → cross-stream) + const-RHS `Ap@Wn2` (weight on the right, no strategy). Fix (not done):
  interleave four-step emission breadth-first + add a shared-RHS-const strategy.
- **Reuse/residency scheduling** cuts operand-load traffic ~45% (ct_mul L3, P=8: 28% of fetches
  reused from L1) but is orthogonal to Fusion B (still ~17% same-core edges — greedy favors
  parallelism). The real lever is **limb-major placement** (pin each limb's chain + consts to a core).
- **Fusion B is reported, not decided.** Co-location falls out of greedy placement; true kernel
  fusion is hardware-gated (L1 capacity, dest/CB depth, fusability) and not yet modeled.
- **Bounded-`u`**: exact CRNS FBC uses BFV-bounded precision (~40b) not worst-case (~600b at L≈62),
  keeping the `matmul_wide` f-vector ~6 int8 limbs not ~89.
- **Multi-factor NTT ordering bug (fixed):** relin's `evk` must be NTT'd through the *same* lowering
  (baked coeff-domain + `rns.ntt`), else the four-step's permuted output order mismatches a
  pre-baked `ref.ntt` const.

---

## 5. Scheduler strategy (direction)

Current: greedy HEFT-lite + reuse/residency + Fusion-B *reporting*. Recommended next: **limb-major
placement** — RNS limbs are independent through NTT/pointwise/rescale, coupling only at base
conversions, so pinning each limb's chain (and its constants) to a core makes reuse + on-core
locality fall out of the structure; NoC traffic only at the CRT-mixing corner-turns
(`all_to_all_tiles`). Then seed the cost model with real n150 measurements so makespan/traffic reflect
the device (currently NoC/DRAM/CRNS-op costs are placeholders).

Relayout among tiles = `all_to_all_tiles` = NoC communication (not a big-matrix transpose); designed,
not yet emitted at 1-tile scale.

---

## 6. Open work (none are correctness gaps)

1. **Fusion A for four-step** — interleave emission + shared-RHS strategy (so it fires at large N).
2. **Limb-major scheduler** + **cost-model seeding** (measured NoC/DRAM/CRNS costs).
3. **Fusion B decision** — hardware model (L1 tiles, dest/CB, fusability) to *pick* fusion groups.
4. **Real-N performance** — the `object`-dtype interpreter is exact but O(N)-per-op; four-step const
   generation is the build wall past ~2¹². (Op graph is N-independent in *shape*; N scales counts.)
5. **Stage-7 codegen** — turn `codegen_draft` into a real tt-metalium emitter for the CRNS/tile ops.
6. **CKKS** — the HPCA paper is CKKS; the shared RNS/CRNS core is built, the CKKS scheme (rescale,
   bootstrapping) is not.

---

## 7. Self-tests (all green)

`reference.py · ir.py · lower.py · schedule.py · builders.py · codegen_draft.py · he/bfv_relin_dnum.py`
— run any file directly. `builders.py` covers he.add / tensor / ct_mul (macro) / **ct_mul_rns**
(primitive, +Fusion A per strategy). Validation method: re-run the golden and diff — lowering is a
*proven* refinement, not a hope.
