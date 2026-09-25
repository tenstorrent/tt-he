# The HE compiler — CKKS front end, four mapping passes seeded on measured p150 data, one entry point

**Status: WORKING (small cases end-to-end on device).** `python3 compiler/he.py` compiles a CKKS
`ct_mul`, runs four optimisation passes, schedules it across 130 cores, emits a dispatch plan, builds
and dispatches a host on the p150, verifies **bit-exact** against its own golden, and serves a visual
tool showing the tile DAG and the scheduling trace. **Eight demo cases, 1.9 s to 21.7 s**, six of them
verified by both decrypt and a bit-exact device run. **[M-p150]**

`compiler/he.py` · `compiler/hw.py` · `compiler/passes.py` · `compiler/viz_app.py` ·
`reference/he/ckks_mul_ref.py` · baseline `compiler/README.md`

---

## 0. Headline

| | |
|---|--:|
| demo cases green (6 decrypt- and device-verified, 2 structural) | **8 / 8** |
| all 8 demo cases wall time | **1.9 s – 21.7 s** |
| pointwise ops executed on device, verified bit-exact | **518 / 518 (100%)** |
| **matmuls executed on the MATRIX ENGINE, verified bit-exact** | **up to 3408** (`n8k`, all primes) |
| cost-model retrodiction of the measured 4-rung hardware ladder | **0.2% worst rung** |
| rungs PREDICTED from independent benches (vs calibrated) | **2 of 3** |
| modelled end-to-end win from the four passes (best: `n1k`) | **2.24×** |
| the real hardware shape (L_Q=48, P=16, dnum=3) compiles in | **17.3 s** (70,448 tile ops) |

---

## 1. First: it was broken

The compiler did not run at all. The repository restructure moved the package from `reference/ir/` to
`compiler/` and the HE references to `reference/he/`, but **nine of the ten modules still
carried `sys.path.insert(0, .../"..", "he")`** — a directory that no longer exists. Only `run_he.py`
had been updated. Seven of the eight self-tests died on import; `run.sh` sourced `../../env.sh`, one
level too high.

| fix | |
|---|---|
| `compiler/_paths.py` | one resolver, imported for effect, replacing all nine stale inserts — one place to change when the layout moves again |
| `numpy` | not installed for the system interpreter and system `pip` is PEP 668-blocked. Installed with `--break-system-packages` rather than writing a pure-Python object-array shim: `reference.py` is the **bit-exact oracle** and is the last module worth reimplementing to save a dependency |
| `run.sh`, `README.md`, `USAGE.md` | paths corrected |

All nine self-tests green again, and that was the precondition for everything below.

## 2. A CKKS front end (the existing one was BFV)

`MulParams` has a plaintext modulus `t` and `Δ=⌊Q/t⌋` — the compiler modelled **BFV**, while the
hardware `ct_mul` is **CKKS**. The difference is only the order and the rescale:

```
BFV    tensor over Q ∪ S  →  rescale (t/Q, BEHZ two-base floor)  →  relin  →  add
CKKS   tensor over Q      →  relin  →  add  →  rescale (drop a prime)
```

so CKKS needs no aux base `S`, and its rescale is a modulus change plus a subtract plus a scalar
multiply per surviving limb instead of a two-base FBC over a wide accumulator. **That is why the
hardware rescale is three small kernels** (`bconv_one`, `modsub_reduce`, `modmul_scalar`) against the
base conversions' hundreds.

Everything else is shared *verbatim*: `_emit_relin_dnum` reads only `pp.{Q,P,QP,N,Pprod,Qprod}`, which
`CKKSParams` provides, so the hybrid key-switch is reused rather than forked.

**No canonical embedding.** Encode/decode sit entirely outside the op graph, and the RNS graph is
identical either way, so the reference validates against an **exact negacyclic integer product** —
`round(dec/scale) == m1*m2`, equal, not "close in 40 bits". That is a stronger gate than a slot
comparison and it costs nothing.

Two things fell out of building it:

* **`dnum=1` does not work in this noise budget, and that is not a bug.** At one digit the mod-up is
  `Q→P` over the *whole* 112-bit `Q`, so the approximate FBC's dropped `ε·Q_j` term is `ε·Q` and
  decrypt misses by tens. Shrinking that term is exactly what hybrid key-switching's `dnum>1` buys.
  The path now requires `dnum ≥ 2` **with the reason in the error message**; the hardware runs `dnum=3`.
* **Δ must equal what the rescale removes.** Multiply takes the scale to `Δ²` and each rescale divides
  by one prime, so `Δ = prod(Q[-ndrop:])` is what returns the scale to `Δ` and keeps the exact gate
  available at any `ndrop`. With `Δ` fixed at one prime, `ndrop=2` over-rescales to scale ≈ 1 and the
  plaintext is unrecoverable — correct behaviour, wrong parameters.

Gate (`compiler/ckks_gate.py`): decrypt **and** `run(L1)==run(L2)==run(L3)`, across N, `LQ`, `dnum`,
single-DFT and four-step NTT, `ndrop∈{1,2}`. All green.

## 3. A cost model that has to earn it — `hw.py`

`cost_model.py` was seeded on an **n150** against the *old* kernels (`modmul` at 30.34 µs/tile, most
entries `"guess"`/`"doc"`) and — the reason a new module exists — **had no notion of which engine an op
runs on.** Blackhole's two compute engines are 20× apart on the multiply, so a model that cannot tell
them apart cannot express the largest decision the mapper makes. `cost_model.py` is left untouched for
the modules that already import it.

Every constant in `hw.py` is measured on this card and carries its source:

| family | source | what it prices |
|---|---|---|
| arithmetic, both regimes | [`../vector/exp_roofline001.md`](../vector/exp_roofline001.md) | add / mul / montmul at R=1 **and** at the compute ceiling |
| the workload | [`../he/exp_tg018ntt.md`](../he/exp_tg018ntt.md) §4d/§4e | per-kernel census of the real ct_mul, both engine arms |
| movement | [`../he/dram_guide.md`](../he/dram_guide.md), [`../he/exp_dram001.md`](../he/exp_dram001.md) | 436 GB/s both-NoC, ~238 GB/s one-NoC wall, the 4.29× on-chip corner turn |

Derived, per-core, per 32×32 tile:

| op | streaming (R=1) | full reuse | reuse win |
|---|--:|--:|--:|
| plain add | 4.756 µs | 0.121 µs | **39.2×** |
| modular add | 4.775 | 0.251 | 19.0× |
| multiply | 4.825 | 0.838 | 5.8× |
| **Montgomery modmul** | **5.526** | **2.843** | **1.9×** |

**That last column is why fusion pays**, and the model knows it: an op whose operands are already in L1
is priced at the reuse end. Movement: `sram` 0, `noc` 0.285 µs/tile, `dram` 1.221 µs/tile. Dispatch tax
**0.840 µs**, backed out of the measured pair (23.991 ms device busy against a 22.917 ms critical path
over 1278 dispatches).

**The matrix engine's speedup is a model with three anchors, two of which fix it and one of which
checks it.** `exp_tg018r` measured the matrix-engine base conversion against the SFPU over the whole
`(Ls,Ld)` grid — 1.19× at 8→8, 10.3× at 32→32 — which fixes a power law
`1.19·(Ls·Ld/64)^0.778`. Evaluated at the real pipeline's `(16,24)` it gives **4.80×** against the
**4.99×** measured in §4e, from an entirely different bench. **−3.8%.**

### The ladder gate

The hardware work produced a measured four-rung ladder in which each rung is exactly one of the four
passes. A cost model that cannot retrodict it has no business predicting anything, so `hw.py` is
required to reproduce it:

| rung | from | model | measured | err | provenance |
|---|--:|--:|--:|--:|---|
| on-chip all-to-all | 50.863 | 43.877 | 43.932 | −0.1% | **PREDICTED** (corner-turn ratio) |
| fused NTT | 43.877 | 32.393 | 32.448 | −0.2% | CALIBRATED (fitted to this rung) |
| base conversion on the FPU | 32.393 | 23.951 | 23.991 | −0.2% | **PREDICTED** (`fpu_speedup(16,24)`) |

**Worst rung 0.2%; two of three predicted rather than fitted.** The fused-NTT rung is honestly
labelled calibrated: its constant (7.195 µs per eliminated dispatch, lumping dispatch tax + barrier +
CB refill + L1 reload) was fitted to that rung, because the mechanism is not separable from the one
measurement available. Saying so is the point — a model whose every constant is fitted to the thing it
predicts is a curve, not a model.

## 4. The four passes — `passes.py`

**Nothing rewrites the program.** Every pass writes into a `Mapping` (engine per op, tier per edge,
dispatch group per op, kernel variant per op), so the computed function is untouched *by construction*
and `run(prog)` stays the oracle. Same discipline as the lowering: a mapping decision that can change
the answer is a bug, not a trade-off. The self-test re-runs the program after all four passes and
diffs.

| pass | what it decides | hardware rung it corresponds to |
|---|---|---|
| **`place_comm`** | every producer→consumer edge into a memory tier; keeps the four-step NTT's corner turn **core-to-core** when the exchange block fits the grid | 50.863 → 43.932 (4.29×, measured on that stage) |
| **`fuse_dispatches`** | merges private chains into one dispatch **and marks their operands resident**, moving every op inside a chain from the streaming to the reuse end of the roofline | 43.932 → 32.448 (2884 → 1288 dispatches) |
| **`select_engines`** | per-op SFPU vs matrix engine, priced by `fpu_speedup`, over-wide destination axes **chunked** rather than refused | 32.448 → 23.991 (1.355×) |
| **`legalize`** | binds a kernel variant + params; enforces the 256-arg budget, `Ld ≤ 24`, and the L1 working set | no rung — it is what makes the other three legal |

The order breaks a genuine cycle (engines want to know about residency, fusion wants to know about
tiers) by running comm → fuse → engines and letting `legalize` repair whatever the first three produced
that the hardware cannot take.

### Two bugs the passes had, both of which hid a win

* **`legalize` demoted the ops it had just legalised.** It set `dst_chunk=24` on an over-wide
  contraction and then re-checked legality against the **raw** `Ld`, so every op it chunked was
  immediately pushed back to the SFPU. `tile32` reported *"256 chunked, 256 demoted"* and the large case
  kept **every** contraction on the SFPU — the engine pass looked like it had nothing to find. Legality
  is a property of the chunk, not of the original width. Fixed: 608 ops reach the matrix engine and
  `legalize` now costs 0.000 ms.
* **The baseline had nothing to improve on.** `place_comm` was measured against a default that already
  put the all-to-all on the NoC, so it reported a 0.000 ms delta against its own outcome. The
  unoptimised mapping now round-trips every all-to-all through DRAM — which is what a naive lowering
  actually does, and is rung 0 of the measured ladder, not a strawman.

### What the passes find, per case

| case | L2 ops | a2a on-chip | dispatches removed | ops → matrix engine | modelled |
|---|--:|--:|--:|--:|--:|
| `tiny` | 1163 | 176 | 294 | 0 | 1.500 → 1.243 ms (1.21×) |
| `small` | 1163 | 176 | 294 | 0 | 1.664 → 1.408 (1.18×) |
| `small-deep` | 2325 | 352 | 586 | 32 | 2.682 → 2.148 (1.25×) |
| `mid` | 1965 | 244 | 566 | 264 | 2.514 → 1.916 (1.31×) |
| `mid-alpha8` | 5683 | 704 | 1640 | 832 | 5.752 → 4.153 (1.39×) |
| **`tile32`** | 4211 | 352 | 1368 | **608** | **5.093 → 2.960 (1.72×)** |
| `big` | 9731 | 352 | 3576 | 1376 | **11.258 → 5.830 (1.93×)** |
| `hw-shape` | 70448 | 2592 | 25968 | **10272** | 76.621 → 40.787 (1.88×) |

`tiny` and `small` move **nothing** to the matrix engine, and that is the model being right rather than
the pass being broken: at N=16–32 every contraction is 8×4, *half* the 8→8 anchor, so the matrix engine
has nothing to chew on. The pass fires once contractions reach tile scale (`tile32`) or the key-switch
digit gets wide enough (`mid-alpha8`, α=8).

## 5. What actually runs on hardware

The existing device kernels (`kernels/ir/`) cover the pointwise ops and the spec format carries one
prime, so what can be dispatched **today** is the pointwise sub-DAG, per prime — driven with the **real
intermediate values** the ops see in place (`ir.run(all_values=True)`), not a synthetic workload that
merely resembles them.

```
mid, --all-primes:
  q=268432897   74 ops  7 waves   183 us   OK
  q=268428161   74 ops  7 waves   184 us   OK
  ... (8 primes)
  executed: 518 of 518 pointwise ops (100%) in 8 dispatch programs, 1227 us
  correctness: OK (bit-exact vs the compiler's golden, every run)
```

**100% of the compiled program's pointwise ops execute on the p150 and verify bit-exact.** Values
shorter than a tile are zero-padded to 1024 — the kernels process 1024 elements and `0*0 = 0+0 = 0`,
so padding cannot change the answer, and the golden is padded identically.

**What does NOT run:** the matmuls, the NTT, the base conversions, the structural relayouts. They are
emitted into `generated/dispatch_plan.json` with kernel, params, engine, core and dispatch group — the
complete mapping — but no kernel is generated for them, which was the agreed scope. The report says
`518 of 518 pointwise` rather than implying full coverage, because a coverage number that quietly means
something narrower than it sounds is the easiest way for a compiler to lie about itself.

## 5b. Matmuls on the matrix engine — the format was the blocker, not the kernel

The first version of this work reported that only the pointwise ops could run. That was true but the
reason was worth fixing: **`kernels/ir/compute/matmul_modq.cpp` had been in the tree the whole time**,
device-validated by `tests/ir/test_ir_matmul.cpp`. What could not express a matmul was the *spec*:
`codegen.py` encodes every op as `(kind ∈ {mul,add,sub}, in0, in1, out, core)` — no shape, no
contraction axis, no per-op constant — and `codegen_run.cpp` is a literal three-way switch. So a second
emitter and runtime lift the limit (`compiler/devgen.py`, `tests/ir/he_run.cpp`, spec v2); the original
pair is untouched because `run_he.py` depends on it.

**This is why the degree has to be ≥ 1024.** `matmul_modq` is a fused decompose + int8 matmul +
recombine over **one 32×32 tile**, so a dispatchable matmul is exactly `(32,32) const @ (32,32) value`.
In the four-step NTT that block only reaches 32×32 when the factors do — which happens at `N = 1024`
and not below. At `N = 256` with factors `(16,16)` the very same matmuls are 16×16 and this kernel
cannot take them. Larger N is not decoration here; it is the precondition.

Two properties made it drop into the existing wave scheduler rather than needing a new one:

* **Tilization.** The matmul kernel wants tilized data and the pointwise kernels do not care. Tilization
  is a fixed permutation and elementwise ops are permutation-invariant, so `devgen.py` tilizes
  everything in Python — inputs, golden, folded weights — and the two families share one layout with no
  tilizer in the runtime.
* **Montgomery passes straight through a plain-constant matmul.** The runtime keeps values Montgomery-
  encoded. A matmul whose weight is plain is *linear* in its data operand, so
  `A @ mont(B) = A @ (B·R) = (A@B)·R = mont(A@B)`. The domain survives with no conversion, so matmuls
  chain with the pointwise ops for free. Had this not held, every matmul would have needed a
  decode/encode pair around it.

**One bug worth recording.** `N ≥ 2048` first reported **zero** dispatchable matmuls. Past a factor of
32, `tile_matmul` slices the weight into 32×32 blocks, so the matmul's left operand is a `tile.slice`
*of* a constant, not a constant — and a direct `is it a const` test misses every one. Const-ness has to
be **transitive** through the structural ops. With that fixed, `n2k` went from 0 to 528 matmuls.

### Measured, single-prime (the default) — every row bit-exact

| case | N | L_Q | L2 ops | **wall** | pointwise | **matmul** | ladder |
|---|--:|--:|--:|--:|--:|--:|--:|
| **`n1k`** | 1024 | 4 | 4 705 | **3 s** | 74 | **34** | **2.24×** |
| `n2k` | 2048 | 4 | 11 217 | **7 s** | 108 | **102** | 1.95× |
| `n4k` | 4096 | 3 | 18 506 | **11 s** | 210 | **272** | 1.86× |
| `n8k` | 8192 | 3 | 41 456 | **26 s** | 210 | **816** | 1.91× |
| `n16k` | 16384 | 3 | 92 894 | 65 s | 74 | **2 176** | 1.94× |

With `--all-primes` (full coverage, one program per prime): `n1k` 370 pointwise + **176** matmul in
11 s; `n8k` 864 + **3 408** in 77 s. All `OK`.

**The mapping numbers the passes find at these sizes** — `place_comm` finally has something to do, and
`select_engines` fires on everything:

| case | a2a kept on-chip | dispatches removed | ops → matrix engine |
|---|--:|--:|--:|
| `n1k` | 176 | 1 724 | 176 |
| `n2k` | 176 | 4 364 | 528 |
| `n4k` | 142 | 7 022 | 1 136 |
| `n8k` | 142 | 15 918 | 3 408 |
| `n16k` | 142 | 35 982 | 9 088 |

**Honest caveat on the operands.** Past `N = 256`, obtaining the *real* intermediate values needs
`ir.run` over the whole L2 program, which is minutes of object-array interpretation. Above that
threshold the slice is driven with **deterministic pseudorandom leaves** instead, and the golden is
computed by interpreting the slice itself. The op structure, the folded weights and the primes are the
compiled program's own; only the leaf data is synthetic, and the output line says which was used
(`[real intermediate operands]` vs `[pseudorandom leaves]`).

## 6. The visual tool

`generated/compiler_viz.html`, self-contained, **no CDN** — a live demo must not need a network.
Three views:

* **pass ladder** — what each pass bought, with the **measured hardware ladder** underneath. The point
  of the tool: predicted rungs beside the numbers the hand-tuned pipeline actually hit.
* **tile DAG** — canvas (survives tens of thousands of nodes), every node coloured by the engine the
  mapper chose, drag to pan, wheel to zoom, hover for opcode / kernel / prime / dispatch group. A
  per-pass **diff** selector highlights what each pass changed; picking `place_comm` redraws edges in
  their *baseline* tier so the amber DRAM round-trips it removed are visible.
* **scheduling trace** — one lane per core, x = µs, bars coloured by engine, dispatch-group boundaries
  optional. Fusion is visible here as bars merging into runs.

Colour carries the engine — Tenstorrent purple for the SFPU, teal for the matrix
engine, amber for dataflow — and never alone: the legend is always present and every tooltip repeats it.

## 7. Demo cases

One script, config block at the top, `--case NAME` to override it:

```
python3 compiler/he.py --case mid          # 2.3 s, everything, then serves the viz
python3 compiler/he.py --list              # the case table
python3 compiler/he.py --case mid --all-primes   # 100% pointwise coverage on device
python3 compiler/he.py --case tile32 --no-device --no-serve
```

**Measured wall time, this box, full pipeline including the device run and the viz:**

| case | N | L_Q | L_P | dnum | NTT | **wall** | decrypt | device | modelled win | good for |
|---|--:|--:|--:|--:|---|--:|---|---|--:|---|
| `tiny` | 16 | 4 | 2 | 2 | (4,4) | **2.1 s** | OK | OK | 1.21× | fastest possible sanity check |
| `small` | 32 | 4 | 2 | 2 | (8,4) | **1.9 s** | OK | OK | 1.18× | |
| `small-deep` | 32 | 8 | 4 | 2 | (8,4) | **2.4 s** | OK | OK | 1.25× | more limbs, same degree |
| **`mid`** | 64 | 6 | 2 | 2 | (8,8) | **2.7 s** | OK | OK | 1.31× | **best all-round demo** — all four passes fire, decrypt- and device-verified, fast |
| `mid-alpha8` | 64 | 16 | 8 | 2 | (8,8) | **4.2 s** | OK | OK | 1.39× | wide key-switch digit (α=8) |
| **`tile32`** | 256 | 8 | 4 | 2 | (16,16) | **12.4 s** | OK | OK | **1.72×** | full 32×32 tiles, and the largest case that still decrypt- and device-verifies |
| **`n1k`** | 1024 | 4 | 2 | 2 | (32,32) | **3 s** | skipped | **OK + matmul** | **2.24×** | **best demo** — matmuls on the matrix engine, 3 s |
| `n2k` | 2048 | 4 | 2 | 2 | (32,64) | **7 s** | skipped | OK + matmul | 1.95× | |
| `n4k` | 4096 | 3 | 2 | 2 | (64,64) | **11 s** | skipped | OK + matmul | 1.86× | |
| `n8k` | 8192 | 3 | 2 | 2 | (64,128) | **26 s** | skipped | OK + matmul | 1.91× | largest under 30 s |
| `n16k` | 16384 | 3 | 2 | 2 | (128,128) | 65 s | skipped | OK + matmul | 1.94× | past the 30 s budget |
| `big` | 1024 | 8 | 4 | 2 | (32,32) | **3.2 s** | skipped | OK + matmul | 1.93× | more limbs at N=1024 |
| `hw-shape` | 1024 | **48** | **16** | **3** | (32,32) | **21.7 s** | skipped | skipped | 1.88× | the real hardware parameter shape, 70k tile ops |

**Ranges that keep a case under 30 s.** With `validate=True` (decrypt + device): `N ≤ 256`, `LQ` up to
~16 at `N ≤ 64` and ~8 at `N = 256`, `dnum ∈ {2,3}`, NTT factors multiplying to `N`. With
`validate=False` (structural): `N = 1024` and `LQ` to 48 both fit — `hw-shape` is 17.3 s.

**The bottleneck was not where it looked.** `big` first took **174 s**, and essentially all of it was
the correctness stage: `run(L1)` then `run(L2)` over ~10k ops of object arrays. But in structural mode
the inputs and keys are dummy **zeros**, so that check was verifying that zero equals zero — vacuous.
Skipping it in structural mode (and saying so in the output) took `big` from **174 s to 2.5 s**, which
is what made `hw-shape` viable at all. Mapping decisions depend on the graph, not the data.

**For a live demo:**

| what you want to show | run | takes |
|---|---|---|
| **matmuls on the matrix engine + the biggest ladder (2.24×)** | `--case n1k` | **3 s** |
| the whole pipeline decrypt-verified end to end | `--case mid` | **2.7 s** |
| 100% of the pointwise ops on device, verified | `--case mid --all-primes` | ~10 s |
| scale: 816 matmuls on hardware | `--case n8k` | 26 s |
| the real hardware parameter shape | `--case hw-shape` | 21.7 s |

## 8. Honest limits

1. **Pointwise ops and 32×32 const-LHS matmuls execute; the rest is planned, not run.** What is still
   only emitted: the sub-tile base-conversion matmuls (`(1,k)@(k,32)`), the structural relayouts, and
   the `int8` L3 form. So the *measured* device number covers a growing slice while the *modelled*
   makespan covers all of it — §5b gives the counts rather than a percentage, because a coverage
   percentage that quietly means something narrower than it sounds is how a compiler lies about itself.
2. **`place_comm` barely moves the small cases** (−0.010 ms) — its win is a large-`N` effect, since at
   `N ≤ 256` a limb is a single tile and the corner turn moves almost nothing. The `n*` cases (§5b) are
   where it has real traffic to place. The measured 18%-of-multiply corner turn is at `N = 2^16`.
3. **The modelled makespans are not the hardware's.** The demo cases are 2–3 orders of magnitude smaller
   than the measured `ct_mul`. What transfers is the *decisions* and the *ratios*, which is why the
   ladder gate (§3) exists at all.
4. **The fused-NTT rung is calibrated, not predicted** — stated in §3 and in `hw.py`.
5. **`dnum=1` is refused on the CKKS path**, for the reason in §2. BFV still accepts it.
6. **`Ld=32` is refused on the matrix engine**, inheriting `exp_tg018ntt` §4e's unexplained wrong answer
   at exactly the full-tile width; the cap is the verified 24.
7. **`hw-shape` is the hardware's PARAMETERS at N=1024, not the hardware's N.** The measured pipeline
   runs `N=2^16`; the op graph is `N`-independent (degree only sets tile counts) so the structure and
   every mapping decision match, but the modelled 40.787 ms is for `N=1024` tile counts and is **not**
   comparable to the measured 23.991 ms. Reaching `N=2^16` needs 256×256 four-step constants, which is
   minutes of pure-Python `pow` — a reference-interpreter artifact, not a pipeline limit.
8. **Structural mode skips both the decrypt check and the device run**, because dummy zero data makes
   the first vacuous and gives the second nothing to verify against. `big` and `hw-shape` are therefore
   *mapping* demonstrations; `tiny`..`tile32` are the end-to-end-verified ones.
9. **Not attempted:** kernel generation, L3 (int8) mapping through the passes, a Fusion-B-aware
   placement that clusters chains instead of spreading for parallelism.

## 9. Files

| file | role |
|---|---|
| `compiler/he.py` | **the entry point** — config block, all six stages, serves the viz, prints artifact paths |
| `compiler/hw.py` | p150 model: roofs, engines, tiers, dispatch, census, **the ladder gate** |
| `compiler/passes.py` | the four passes + the pass manager + the mapping-aware scheduler |
| `compiler/viz_app.py` | the three-view visual tool |
| `compiler/ckks_gate.py` | CKKS front-end gate (decrypt + meaning-preserving lowering) |
| `compiler/sweep_cases.py` | time candidate configs, see which passes fire |
| `compiler/_paths.py` | the one repo-layout resolver |
| `reference/he/ckks_mul_ref.py` | CKKS oracle, coefficient domain, exact |
| `compiler/generated/` | dispatch plan, device spec, viz page (all emitted) |

Self-tests, each runnable alone: `python3 <file>.py` for `reference · ir · cost_model · hw · builders ·
lower · schedule · passes · mlir · codegen_draft · ckks_gate`.
