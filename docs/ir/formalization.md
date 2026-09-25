# Formalizing the HE workload: IR levels + mapping/scheduling

**Status:** design / idea doc. Nothing here is built yet. This is the shared picture to
read, poke holes in, and turn into a step-by-step plan.

**What we want.** Take the HE workloads we already run on device (`ct_mul`, `ct_add`, the
PIR pipeline) and (1) express them in a multi-level IR that decomposes each operation down
to hardware primitives based on hardware properties, giving an **IR DAG**; then (2)
**map/schedule/bind** that DAG onto the physical Tensix grid — different stages using
different numbers of cores, and independent operations running *concurrently* on disjoint
core sets (e.g. NTT on 4 cores for data A while modmul runs on 3 cores for data B at the
same time).

The `reference/ntt_matmul/` compiler is our worked example of step (1) for **one** op (the
NTT). This doc generalizes that idea to the whole workload and adds step (2), which is new.

**Scope decision (top of the IR = RNS-Poly).** The IR starts at the **RNS-Poly** level. The
protocol/application level (`he.mul`, `pir.retrieve`) is *not* part of the IR — it is too far
from hardware, and it stays host-side (encrypt / keygen / decrypt are already host by BFV
convention). The host is the *driver*: it constructs the RNS-Poly program and reads results.
RNS-Poly is the right ceiling because it is where the **batch / RNS-limb / vector** structure
is explicit — i.e. where the parallelism we want to map actually lives.

---

## 0. The one structural decision: two orthogonal axes

Everything below hangs on keeping these two transformations **separate**. They are different
in kind, and conflating them is the main way this design goes wrong.

- **Vertical — Lowering (a.k.a. Decomposition).**
  *"This 84-bit modular matmul **is** 16 int8 matmuls + a recombine."*
  Refines *what* the computation is, into hardware-expressible ops. Meaning-preserving,
  verified bit-exact against the numpy/big-int golden. **This is exactly what
  `ntt_matmul` already does** (L1 → L2 → L3 → L4 in its viewer).

- **Horizontal — Mapping.**
  *"This NTT **runs on** cores (0,0)–(3,0) during time window [t0,t1]; that modmul runs on
  cores (4,0)–(6,0) at the same time."*
  Refines *where* and *when*, onto the 64-core grid. Also meaning-preserving — it is a
  scheduling refinement, so results stay bit-identical; only latency/throughput change.

The NTT example is *entirely* the vertical axis. Everything about "3 cores vs 5 cores,
different stages, NTT-on-A while modmul-on-B" is *entirely* the horizontal axis, and it is
the new work.

**Why the existing IR is already the right substrate for both:** `ntt_matmul` already emits
an explicit tile-granular `all_to_all_tiles` op — a permutation/gather over ≤T tiles. That
op *is* the NoC. So the tile IR is already "spatially aware": mapping becomes "assign the
other ops to cores, decide when they run, and turn `all_to_all_tiles` into physical NoC
routing."

### Naming (the question was: mapping / scheduling / bind?)

- **Lowering** — the vertical axis.
- **Mapping** — the umbrella for the horizontal axis, with three sub-steps:
  - **Placement** (op → set of cores). "Bind" is a fine synonym.
  - **Scheduling** (op → time / order).
  - **Routing** (data → NoC path).

This matches standard spatial-compiler (CGRA / HLS) vocabulary, so we don't fight
terminology later.

```
      ( Protocol — HOST driver, not in the IR )
           │  host builds the program
        L1 RNS-Poly  ───────────────►  Mapping  ───►  Physical schedule
           │  lower                    (place +        (cores × time,
        L2 Tile (mod-q) = kernel IR ◄── schedule HERE   explicit NoC)
           │                            route)
           └─(matmul_modq only)─► L3 int8 decomposition  (FPU low-precision;
                                                          internal to the matmul task)
        └────── VERTICAL (lowering) ──────┘   └──── HORIZONTAL (mapping) ────┘
```

- L1 is the top IR level; numbering keeps the `ntt_matmul` viewer's L1–L4 convention. There
  is no L0 IR — the protocol level is host-side orchestration.
- **L2 is the kernel IR and the single mapping level.** SFPU/vector ops (`mul_modq`,
  `add_modq`, `transpose`) are primitive at L2. **L3 hangs only off `matmul_modq`** (int8
  radix-128 for the low-precision FPU matrix unit) and is internal to that task, never a
  separate mapping level.

---

## 1. The IR level stack (generalizing `ntt_matmul` to the whole workload)

Today we have a great IR for *one* op. To formalize the *workload* we lift the same idea up
**one** level — to RNS-Poly — so `ct_mul` and PIR are themselves programs in an IR, reusing
the existing `Program` / `Op` / `Value` dataclasses, the shape verifier, and the numpy
interpreter in `reference/ntt_matmul/ir.py`. The protocol level above stays on the host (see
the scope decision in the intro); it is the driver that emits the L1 program, not an IR level.

| Level | Name | Values | Example ops | Golden reference |
|---|---|---|---|---|
| *(host)* | *Protocol / Application (not IR)* | *ciphertexts `(c0,c1)`, RGSW, PIR query* | *`he.add`, `he.mul`, `pir.retrieve` — host driver* | *`reference/he/bfv_mul_ref.py`, `reference/pir/onionpir_ref.py`* |
| **L1** | RNS-Poly **(top IR level; the scheduling DAG)** | RNS polynomials `poly[limb][N]` — carries batch / limb / vector structure | `poly_ntt`, `poly_intt`, `pointwise_mul`, `base_convert`, `modadd`, `rescale`, `gadget_decompose`, `extern_product` | per-stage big-int |
| **L2** | Tile (mod-q) — **the kernel IR** | ≤T tiles, `i32` | `matmul_modq` (FPU/matrix unit), `mul_modq`, `add_modq`, `transpose`, `all_to_all_tiles` (SFPU / NoC) | **the existing `ntt_matmul` IR** |
| **L3** | int8 decomposition — **matmul-only, not a universal level** | `i8` limbs | `decompose`, i8 `matmul_modq`, `recombine_modq` | int8 golden |

**L3 is op-specific, not a horizontal level.** It exists for *one* reason: the matrix unit
(FPU) is **low-precision (int8×int8→int32)**, so a 28-bit `matmul_modq` cannot run on it
directly and must be decomposed into radix-128 int8 limbs. So **only `matmul_modq` has an
L3.** The elementwise / vector ops (`mul_modq`, `add_modq`, `transpose`) run on the **SFPU**
and are *already primitive at L2* — there is nothing below them to lower to. The stack is
therefore not a clean pyramid: it is L1 → L2 for everything, with a local "int8 bump" under
matmul nodes only.

Consequence for the mapper: **L2 is the kernel IR** — the level where (almost) every op is
one primitive device kernel. The lone exception is `matmul_modq`, whose int8 expansion is
handled internally (see §2.0). L2 and L3 already exist and are verified (that's
`ntt_matmul`). The missing piece is **L1**, which matters most for scheduling because that is
where the parallelism is visible — the batch and RNS-limb axes at L1 are exactly the
independent work the mapper spreads across cores.

**Two consequences of tile-then-decompose ordering.**

1. *Every matmul the scheduler sees is tile-size and p=1.* Because `tile_matmul` runs before
   `lower_matmul_int8`, all `matmul_modq` are ≤T by the time they decompose; and the matrix
   engine only does exact matmul via int8, so every exact modular matmul takes the int8 path.
   The mapper therefore only ever sees **tile-size int8 matmuls, defaulted to p=1** (the
   `ntt1024_dev_mc` ~8% result says a single tile-matmul is best left atomic on one core).

2. *Way 1 vs Way 2 — a lowering choice, orthogonal to mapping.* There are two orders to turn a
   large 28-bit matmul into int8:
   - **Way 1 (current IR): tile → decompose.** Tile K to ≤T, then decompose each tile-matmul;
     `recombine_modq` runs *once per K-tile*, summed with `add_modq`.
   - **Way 2 (recombine-hoist): decompose → accumulate → recombine once.** int8 matmuls
     accumulate int32 planes across *all* of K; a single wider recombine at the end.

   They are **bit-exact equivalent**. Note the split state today: the IR models Way 1, but the
   **device kernels already run Way 2** (README: "accumulates int32 planes across K and
   recombines once … a future 'recombine-hoist' pass would close that gap"). When it matters:
   recombine is the costly step (~85% of base-conversion; the biggest NTT stage), and Way 2
   saves `(K_tiles − 1)` recombines per output tile — so it is a **no-op when K ≤ T** (the
   3-factor 32×32 NTT; the whole current HE workload) and a **big win only for large-K**
   matmuls (the 2-factor big-matmul plan; a future matmul-based base conversion). Precision:
   Way 1 reduces mod q per K-tile (tiny intermediates, never wrong); Way 2 defers reduction
   (planes grow with K → needs an overflow guard). **Decision: Way 1 for v1.** Way 2's benefit
   is recoverable *without* reordering — add a **recombine-hoist peephole** on the decomposed
   graph (rewrite per-K-tile recombine+`add_modq` → plane-accumulate + single recombine) only
   once a large-K matmul makes it pay. Either way this is a vertical (lowering) concern; it
   does not touch p=1 or the scheduler.

### Why L1 is the important new level

At L1, `ct_mul`'s raw-multiply stage (`ct_mul_raw`; "tensor" in BFV terms) is literally:

```
NTT each of {c0, c1, d0, d1} across 8 RNS limbs   → 32 independent poly_ntt nodes
pointwise combine (e0 = c0·d0, e1 = c0·d1 + c1·d0, e2 = c1·d1)
INTT the results                                  → 24 independent poly_intt nodes
```

Those 32 NTTs have **no dependency between them**. That is exactly the "NTT on data A while
modmul on data B" concurrency — it is sitting in the DAG the moment L1 exists. You cannot
schedule what you cannot see; L1 is what makes the available parallelism explicit and
machine-readable.

The host driver emits the L1 program directly (`he.mul` on the host expands into the
ct_mul_raw/rescale/relin subgraph of `poly_*` ops). The `prov` provenance attribute already used
by the viewer extends naturally to link the lowering chain L1 → L2 → L3.

**Grounding note:** the L1 op list should be *read off* `lib/he/he_api.cpp` and
`lib/pir/pir_demo_api.cpp` (the actual device pipelines), not invented — so the IR matches
what really runs.

---

## 1.5 The compilation pipeline (end to end)

The full flow, from application to running device code. Each stage says what it does, **what
information enters** there, and whether it **exists** (in `ntt_matmul`) or is **new**.

| # | Stage | What it does | Info injected | Status |
|---|---|---|---|---|
| 1 | **Application** | protocol op (`he.mul`, `pir.retrieve`) | — | host (exists) |
| 2 | **→ L1 (RNS-Poly)** | decompose to hardware-agnostic poly ops | **none (no HW info)** | new |
| 3 | **L1 → L2 (tile)** | tile M/N/K to ≤T, corner-turns → `all_to_all_tiles` | **tile size** (the 32×32 matrix-engine tile — 1st HW fact) + tiling strategy | exists (for NTT) |
| 4 | **L2 → L3 (int8)** | radix-128 decomposition of `matmul_modq` only | **none** (radix/limbs fixed) | exists |
| 5 | **Fusion A — tile packing** | pack under-full **limb-major** tiles (few limbs → empty rows) into full 32×32 | — | new (needs multi-Q kernel) |
| 6 | **Placement + scheduling + memory** (Fusion B folded in) | greedily assign each op a core-set *and* a time; the placement **is** the memory-tier choice — co-located producer→consumer = SRAM (this *is* Fusion B), different cores = NoC, neither fits = DRAM. (later: also choose moldable `p`) | **max cores P** (2nd & last HW fact) | new (the core of this effort) |
| 7 | **Codegen** | emit kernels + host program + baked constants | — | partial (`print_mlir`; `.cpp` hand-written) |

> **v1 = greedy, two passes.** Stage 5 = obvious same-stream packing only. Stage 6 = **HEFT**
> (comm-aware greedy list scheduling: rank by critical path, place each op where it finishes
> earliest, with the DRAM/NoC/SRAM tiers as the communication cost) — and start with allotment
> fixed at **p = 1** (co-scheduling only), adding moldable `p > 1` in v1.1.

### v1 parameter defaults (stage 3)

Stage 3 (L1 → L2) is the one place tiling parameters enter. v1 keeps them minimal:

- **Square tiles only** — `T × T`, not `T_m × T_n`. TT's matrix engine is square, so this is
  the natural assumption. Non-square tiles are a future generalization for other hardware.
- **`T = 32`, uniform** — the Wormhole matrix-engine tile. Keep the two `ntt_matmul` tile
  levels equal (`TILE_NTT = TILE_MM = 32`, single-level tiling) so there is no second tiling
  knob to reason about. This is the "3-factor 32×32" regime that already matches the device.
- **Flat NTT plan** — the factorization is a flat tuple `(a, b, c)`, **not** the recursive /
  nested form `(a, (b, c))`. Flat mixed-radix is sufficient; no Cooley-Tukey recursion tree.
  A factor larger than `T` (e.g. the `64` in N=2048's `(32, 64)`) is fine — `ntt_matmul`
  already tiles a factor > T without adding an NTT stage — so "flat tuple" and "T = 32"
  coexist.

Note this splits the stage-3 input cleanly along the two-axes principle: the **tile
(square, 32)** is the hardware fact; the **NTT plan (flat tuple)** is an algorithmic tuning
choice carried in from the host, with no hardware content.

### Two clean properties this pipeline has

1. **Exactly two hardware-info injection points:** the **tile shape + size** (stage 3;
   v1 = square 32×32) and the **core count P** (stage 6). Everything else is
   hardware-agnostic algebra (L1) or
   mechanically determined (L3, codegen). This is *why* the mapper can be HE-unaware and *why*
   L1 is portable. Treat it as a design invariant: if a stage wants a third piece of hardware
   info, question the design.
2. **Correctness is invariant across every stage.** Stages 2–7 are all meaning-preserving —
   the numpy interpreter golden runs unchanged end to end. Only stages 5–6 change *latency*,
   validated by the timing simulator (§3), not the correctness golden.

### The two "fusions", and why Fusion B is *inside* placement

- **Fusion A = tile / batch packing** (stage 5, standalone). The matrix engine bills a full
  32×32 tile regardless of fill, so pack under-full tiles.
  **Crucial refinement for real HE (`N ≥ 1024`):** the packing target is the **limb** direction,
  *not* the coefficient direction.
  - *Coefficient-major* (vector-slice): at `N ≥ 1024` a poly limb is `N/1024` **full** tiles —
    nothing to pack. (Coefficient packing only helps at toy `N`; moot in production.)
  - *Limb-major* (base conversion): the tile's **rows are limbs**. With a small base (e.g. 4
    limbs) only 4 of 32 rows are used — 7/8 empty. The real Fusion A **merges several
    independent base-conversion operations' limb-tiles to fill the 32 rows** (~8× on a 4-limb
    base). **This requires the multi-Q kernel** (§8): a packed limb-tile spans multiple primes
    (rows are different `q_i`; the per-row reduction is mod different `p_j`), so the reduction
    must apply a **per-row / per-tile prime**. Multi-Q is therefore a *prerequisite* for Fusion
    A here, not a later optimization.
  - **Split by placement:** same-stream packing is placement-independent (stage 5); cross-stream
    packing (data A + data B) requires co-location — a placement-time decision (defer to v1.1).
- **Fusion B is not a separate stage — it is a consequence of placement** (folded into stage
  6). "Keep this intermediate in SRAM" *is* "place the consumer on the producer's core." So
  placement decides the memory tier per edge: same core → SRAM (this is the fusion), different
  cores → NoC (`all_to_all_tiles`), neither fits → DRAM. This is why 6 and 7 merge: they were
  always one decision. It directly attacks today's dispatch-bound `ct_mul` and generalizes the
  hand-done buffer-resident chaining (`docs/he/history/PLAN.md` §4).

### The remaining coupling: stage 5 ↔ stage 6 (phase ordering)

With Fusion B folded into placement, only one phase-ordering seam is left: packing (5) changes
op count/sizes → what the placer (6) sees; and the placer's co-location choices could enable
*more* packing (the deferred cross-stream case). This is the classic trap, now smaller.
**v1: greedy, fixed order 5 → 6, measure, then consider one feedback iteration (5 → 6 → 5).**
Do *not* build a joint co-optimizer first. Within stage 6, greedy = HEFT with p = 1.

---

## 2. Formalizing Mapping: this is *moldable task scheduling*

This is the framing that turns "3 cores vs 5 cores" from a vibe into a solvable problem.

### 2.0 At which level do we map? — one level, and it's HE-unaware

Mapping *looks* like it could happen at every level: place the L1 poly-ops, then sub-place
the L2 tiles inside each, then further split the L3 int8 planes across cores. **Doing all
three is a mistake** — passes making overlapping decisions about the same physical cores,
which then have to be reconciled. That is pure redundancy and complexity for no gain.
**Map at exactly one level.** (And note L3 barely qualifies as a "level" anyway — per §1 it
only exists under `matmul_modq`; most ops are already primitive at L2.)

The right level is **L2, the tile / kernel IR**, and the reason is that it gives
*HE-algorithm-unaware scheduling*:

- L2 ops are generic modular-tile algebra — `matmul_modq`, `mul_modq`, `add_modq`,
  `transpose`, `all_to_all_tiles`. **No HE semantics.** The scheduler never learns whether
  it is placing an NTT, a rescale, a PIR fold, or a future CKKS op. The HE compiler and the
  mapper are cleanly decoupled: any HE algorithm lowers to the same ~5 kernel ops and shares
  one mapper.
- Each L2 op corresponds to **one dispatched program over a core range** — the natural
  schedulable / moldable unit on TT.
- The cost model is per-kernel, and L2 ops *are* the kernels, so schedule-unit and
  cost-model line up 1:1.

**What happens to "further mapping at L3"?** It collapses into a single parameter — and note
L3 only arises under `matmul_modq` in the first place (§1), so this concerns matmul nodes
alone. The only decision L3 would make is how one `matmul_modq`'s 16 int8 matmuls / 4 result
planes spread across cores — exactly the `ntt1024_dev_mc` experiment. That is not a separate
placement; it is just *how many cores that one kernel was given* = the moldable allotment
`p` for that task. So **L3 stays pure lowering** (it defines what the matmul kernel computes
and feeds its cost model); it is never independently scheduled. Non-matmul ops have no L3 to
worry about at all.

**What happens to L1?** It is not the scheduling level, but it does not disappear: it is the
human-meaningful grouping for the Gantt / viz (label a region by the poly-op it came from)
and an *optional* clustering hint (see below).

**The one tradeoff.** The batch / RNS-limb independence that is obvious at L1 becomes
*implicit DAG topology* at L2 — a wider graph, more nodes. It is not lost (a list scheduler
recovers it from the dependency structure), but the search space is bigger. Mitigation:
**clustering** — group L2 ops into regions before scheduling (by `prov`, i.e. which L1 op
they came from, or by connected component). Clustering is itself HE-unaware (pure topology),
and it is the granularity knob: coarse clusters ≈ L1-task grain (fast, coarse schedule),
fine clusters ≈ per-tile (slow, closer to optimal).

### 2.1 The moldable-task model

Take the **L2 DAG** as a task graph (optionally pre-clustered per §2.0):

- **Tasks** = L2 kernel ops (or clusters of them) — one dispatched program each.
- **Each task is *moldable*** — it can be given a variable number of cores `p`, running in
  time `T_op(p)`. This is a classic, well-studied scheduling class (*moldable / malleable
  tasks* on `P` processors, minimize makespan). It maps directly onto the intuition "this
  stage uses 4 cores, that one uses 3." The intra-kernel work-split (the L3 plane spread) is
  precisely this `p`.
- **Resources:** `P ≈ 64` cores. Wormhole n150 exposes an 8×8 usable Tensix grid
  (the `vector_modmul` throughput runs cite 64 cores).
- **Objective:** minimize makespan (latency) — or maximize throughput for a batched
  workload — subject to (a) DAG dependencies and (b) the cost of `all_to_all` transfers
  between producer/consumer tasks placed on different cores.

### The two kinds of parallelism both fall out of this one formulation

- **Data-parallel work-split** (same op, many cores): a single moldable task with `p > 1`.
  *(v1.1 — deferred; the `ntt1024_dev_mc` data says this is only ~8%.)*
- **Spatial co-scheduling** (NTT ∥ modmul on different data): two independent DAG tasks
  whose time intervals overlap on **disjoint** core sets. The scheduler produces this
  automatically whenever the DAG has independent branches and cores are free.
  *(v1 — this is the bigger lever, and it works even at `p = 1`.)*

So we do not need two mechanisms — one moldable-task scheduler expresses both. **v1 fixes
`p = 1` and relies on co-scheduling alone; moldable `p > 1` is v1.1.**

> **✅ Verified on silicon (n150, 2026-07-17).** The whole premise depends on TT being able to
> run *different* kernels on *disjoint* core-ranges concurrently *within one dispatched
> program*, and on that beating serial dispatch. Both hold — `tests/common/coschedule_probe.cpp`:
> - **Heterogeneity:** `modmul` on core (0,0) and `modadd` on core (1,0) in **one** program —
>   both outputs bit-exact.
> - **Concurrency:** two equal-cost `modmul`s (256 tiles each) on two cores, one program vs two
>   serial programs → **2.00× speedup** (7.77 ms fused vs 15.53 ms serial). Adding the second
>   `modmul` cost *zero* extra wall-time — it fully overlapped the first.
>
> Mechanism: one `Program`, multiple `CreateKernel` on disjoint `CoreRange`s, one
> `EnqueueMeshWorkload`. This is the codegen shape the mapper must emit (see the pre-build
> notes); it works and delivers the predicted win. Corollary confirmed: fusing a *cheap* op
> onto an expensive one saves little (heterogeneous run was only 1.04×) — the gain comes from
> overlapping *comparable* work, exactly what the cost model must reason about.

### The v1 algorithm: HEFT (greedy, comm-aware list scheduling)

Do not invent a scheduler. **HEFT** (Heterogeneous Earliest-Finish-Time) is exactly the
greedy, locally-optimal approach: (1) rank every op by its critical-path length to the sink;
(2) walk ops in that order and place each on the core(-set) where it **finishes earliest**,
charging communication from its already-placed predecessors. The **DRAM / NoC / SRAM tiers
are HEFT's communication-cost term** — a same-core predecessor costs 0 (that is Fusion B), a
NoC hop costs the transfer, DRAM costs the round-trip. So placement + memory-tier + Fusion B
(pipeline stage 6) are one HEFT pass. Moldable `p` (v1.1) just adds "also try a few core
counts per op and keep the best finish time."

### The Physical IR (the output of mapping)

A **Physical IR** is the L2 (kernel) DAG plus a binding:

- a map `op → core-set`  (**placement**)
- a map `op → time-interval`  (**scheduling**)
- explicit **NoC transfer** ops inserted wherever a producer and consumer are placed
  differently (this is `all_to_all_tiles` made physical).

Its **verifier** checks scheduling invariants instead of shapes:
1. no core is double-booked across overlapping time intervals;
2. every consumer starts after its producer finishes + the transfer completes;
3. per-core SRAM residency fits — the core-local memory TT calls "L1", *not* the IR level
   (a 32×32×4B tile ≈ 4 KB; Wormhole core-local SRAM ≈ 1.5 MB/core).

The **functional golden is unchanged** — mapping is a pure refinement, so the existing numpy
interpreter still validates correctness. A **separate timing simulator** consumes the
Physical IR + cost model and predicts makespan. Correctness and performance are validated by
two different checkers over the same IR.

---

## 3. The linchpin: the cost model `T_op(p)` — and the warning our own data gives

Placement is guesswork without a cost model. And we already have the cautionary tale:
`docs/matmul/history/NTT_1024_MULTICORE.md` — splitting the int8 matmul across 4 cores yielded only
**~8%**, because (a) the matmul was *not* the bottleneck (recombine + twiddle were, and
stayed single-core), and (b) every core still paid the **fixed per-program dispatch cost**.

So the model must be Amdahl-shaped with an explicit dispatch term:

```
T_op(tiles, p) ≈ dispatch_fixed
               + serial_frac · W
               + (1 − serial_frac) · W / p
               + comm(tiles, placement)
```

**The design consequence:** the scheduling win comes mostly from
1. **filling idle cores with *independent* work** (spatial co-scheduling), and
2. **fusion to kill dispatch overhead**,

**not** from throwing more cores at a single op. The multi-core NTT result is the proof.

We can fit this model from data we already have: `docs/he/ct_mul_perf.md`,
`docs/he/kernels.md` (per-RISC breakdown), and the `NTT_1024_MULTICORE` table.

---

## 4. Suggested sequencing (to be broken into steps later)

1. **Build L1 (RNS-Poly IR)** for `ct_mul` and PIR as `Program`s emitted by the host driver;
   reuse the existing IR classes + numpy golden. Valuable on its own: it makes the DAG (and
   its parallelism) explicit.
2. **Extract a cost model** from the existing profiler CSVs — fit `dispatch_fixed`,
   `serial_frac`, `comm` per kernel. **Validate it by *predicting* the known 12 ms `ct_mul`
   and 7.1 ms PIR from a hand schedule.** If the model can't reproduce numbers we already
   measured, any scheduler built on it is fiction.
3. **Physical IR + verifier + timing simulator.** Start from a *fixed hand schedule*
   (today's all-serial-on-1-core), confirm the simulator reproduces measured latency, *then*
   introduce a scheduler.
4. **Scheduler, simplest first: HEFT at `p = 1`** (§2). Greedy comm-aware list scheduling —
   rank by critical path, place each op where it finishes earliest, DRAM/NoC/SRAM as the comm
   cost (so placement + Fusion B are one pass). Add Fusion A (stage 5, same-stream packing)
   before it. Defer moldable `p > 1` and cross-stream packing to v1.1; ILP only for small
   provably-optimal reference cases.
5. **Extend the viz.** It is already multi-level; add a placement view (grid heatmap + a
   Gantt over cores) so a schedule is inspectable the same way decompositions are.

---

## 5. Open decisions (need your call)

- **Mapping level — decided (§2.0).** Map at **L2 (kernel IR), one level, HE-unaware**; L3
  work-split is the moldable allotment `p`, not a separate pass. What is still open is the
  **clustering granularity** on top of L2: coarse (cluster by `prov` ≈ one L2 region per L1
  op — fast, coarse) vs fine (per-tile — slow, closer to optimal). Recommend starting coarse
  and refining only where the cost model says it pays.
- **Static vs dynamic.** TT dispatch is host-driven, so a fully **static** compile-time
  schedule fits naturally and is far simpler to reason about. Recommend starting static.
- **Build vs. align with tt-mlir.** Tenstorrent's MLIR stack does grid placement/sharding
  but has no HE ops (modmul, recombine, base-convert). Recommend keeping the custom IR (we
  need HE semantics + the bit-exact golden) but borrowing tt-mlir's grid/placement
  *vocabulary*. A short spike could confirm.
- **Routing fidelity.** Model the NoC as **uniform cost first** (volume-based); add
  distance/(x,y)-aware routing only if the uniform model proves inaccurate. The multi-core
  data suggests dispatch, not NoC distance, dominates at this scale.
- **Objective.** Latency (makespan) for a single `ct_mul`, or throughput for batched
  workloads? They lead to different schedules.

---

## 6. Risks / constraints to respect

- **The cost model is make-or-break.** Validate it against known measured numbers (§4.2)
  before trusting any schedule it produces.
- **Fused-FBC n≠4 bug.** `baseconv_fbc` is only correct when the modulus count divides the
  CB depth (n ∈ {2,4,8}); base conversion currently runs padded to width 4. The scheduler
  must treat this as a **placement/shape constraint** (base-convert tasks are width-4),
  not something to schedule around, until the real fix lands.
- **Single physical device, run serially.** One n150; device programs must not run
  concurrently at the *host-dispatch* level even though the *schedule* models on-chip
  spatial concurrency. The simulator models grid concurrency; the real device still executes
  one dispatched program's grid at a time.

---

## 7. Relationship to the existing `ntt_matmul` compiler

| `ntt_matmul` today | this proposal |
|---|---|
| IR for one op (NTT) | same IR classes, lifted to L1 (RNS-Poly) for whole workloads |
| Levels L1–L4 (all *vertical* lowering) | add the *horizontal* mapping axis on top of L1/L2 |
| `all_to_all_tiles` = abstract corner turn | becomes physical NoC routing under a placement |
| numpy interpreter = correctness golden | unchanged; add a separate timing simulator |
| `prov` attr links decomposition levels | same mechanism links L1→L2→L3 and op↔placement |
| multi-level web viewer | add placement (grid heatmap) + Gantt views |

The vertical machinery is done and verified. This proposal reuses it wholesale and adds the
horizontal (mapping) axis plus the new L1 (RNS-Poly) top level that exposes the parallelism to map.

---

## 8. Deferred to post-v1 (future exploration)

Explicitly out of v1 scope, kept here so nothing is lost. None of these is a correctness gap.

- **Way 2 — recombine-hoist (int8 lowering).** v1 uses **Way 1** (tile → decompose; one
  `recombine_modq` per K-tile). Way 2 accumulates int32 planes across all of K and recombines
  **once** — bit-exact equivalent, and what the **device kernels already do** (the IR just
  models Way 1). Add it as a **peephole on the decomposed graph** (per-K-tile recombine +
  `add_modq` → plane-accumulate + single recombine); no pipeline reorder needed. **Payoff is
  zero when K ≤ T** (the current 3-factor NTT / whole HE workload) and large only for
  **large-K** matmuls (2-factor big-matmul plan; future matmul-based base conversion).
  Precision guard required (Way 2 defers the mod-q reduction, so int32 planes grow with K).
  Full detail in §1, "Two consequences of tile-then-decompose ordering." Purely a lowering
  concern — does not touch p=1 or the scheduler.
- **Moldable `p > 1` (mapping).** v1 fixes p=1 (co-scheduling only). v1.1: let HEFT also try a
  few core-counts per op and keep the best finish time (§2). Low priority per the
  `ntt1024_dev_mc` ~8% result.
- **Cross-stream tile packing (Fusion A, part 2).** v1 packs only same-stream partial tiles.
  Packing partial tiles from *different* data streams needs co-location, so it becomes a
  placement-time (post-stage-6) opportunistic pass (§1.5).
- **One feedback iteration 5 → 6 → 5.** v1 runs the pipeline once; revisit packing after
  placement only if measurement shows it pays (§1.5).
- **Finer scheduling granularity.** v1 clusters coarse (one L2 region per L1 op, by `prov`);
  drop to per-tile only where the cost model says it helps (§5).
- **Throughput objective / batching.** v1 optimizes single-`ct_mul` latency (makespan);
  batched-throughput scheduling is a different objective and schedule (§5).
- **Distance-aware NoC routing.** v1 uses uniform (volume-based) comm cost; add (x,y)-aware
  routing only if the uniform model proves inaccurate (§5).
- **Non-square / non-32 tiles.** v1 assumes a square 32×32 tile (Wormhole matrix engine).
  Other tile shapes/sizes (other hardware, or a larger `TILE_NTT` over `TILE_MM`) are a
  stage-3 generalization (§1.5 "v1 parameter defaults").
- **Recursive NTT plans.** v1 takes a flat factor tuple `(a, b, c)`; nested/recursive
  factorization `(a, (b, c))` (a Cooley-Tukey recursion tree) is a future planner
  generalization (§1.5).
- **Multi-Q kernels (per-row / per-tile prime) — REQUIRED for limb-major Fusion A, not just an
  optimization.** IR uses per-value `q` (decided). Two uses: (a) *dispatch reduction* — pack tiles
  of different limbs into one SFPU dispatch via a per-tile prime table (à la `modmul_rns_sfpi`);
  (b) *the essential one* — **limb-major base conversion**, where one tile's rows are different
  primes and the reduction is mod a **different `p_j` per row**. Real HE (`N ≥ 1024`) has full
  coefficient tiles, so the only under-fill worth packing is the limb direction (§1.5 Fusion A) —
  and that packing is impossible without per-row multi-Q. **So multi-Q reduction is a prerequisite
  for Fusion A at production `N`, promoted from deferred.** `matmul_modq`'s int8 body stays
  prime-agnostic (products < q); the per-row prime lives in the recombine/reduction. (ops.md
  Conventions.)
- **Expanded-K int8 matmul for limb-major (few limbs).** `matmul_modq` has two int8 lowerings,
  chosen by layout: (a) **coeff-major / plane** — decompose A into 4 radix-128 limbs, 4 matmuls,
  accumulate planes, recombine (v1, the `fused_modmatmul` path); (b) **limb-major / expanded-K** —
  when the contraction axis is the RNS limb count `L` and `L ≤ 8`, stack the 4 radix-128 A-limbs
  *into the contraction rows* (`K → 4L ≤ 32`) so one wider int8 matmul does the radix-128 A-limb
  sum as its natural row-contraction (`128^p` baked into the folded constant `b_{p,r}`), leaving
  only the 4-way `128^r` recombine. Eliminates the 4 plane-tiles + the `add_modq`-over-`p`, and
  fills the rows that limb-major-with-few-limbs otherwise wastes. Bound: `C_r = Σ_{4L} 127² <
  2^19 < q`, exact. Payoff on the FPU base-conversion matmul; deferred until that path is built.
