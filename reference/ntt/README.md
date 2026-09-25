# Decomposed NTT via Basic Kernels + an IR — Design Plan

## 0. Goal

Build a **finite-field NTT** for FHE that is expressed entirely in terms of a few
**basic 2D kernels**, where the **decomposition strategy** is a first-class,
flexible object (an IR), separate from the kernels that execute it.

Two things must stay decoupled:

- **Decomposition strategy** — *how* we factor `N` into stages (the search space).
- **Kernels** — *what* actually runs (matmul / elementwise-mul / transpose).

A strategy compiles to an **IR** (a list of kernel ops); a **backend** runs the IR.
The numpy backend is the bit-exact golden oracle; a tt-metal backend can consume the
same IR later.

---

## 1. Scope and field

- **Sizes:** `N = 2^k` only (FHE/RLWE always uses power-of-two ring degree),
  `k ≤ 25` for the chosen prime. Typical `k = 10 … 16`.
- **Prime:** `q = 167772161 = 5·2^25 + 1` (NTT-friendly, 28-bit, `q−1 = 2^25·5`).
  - NOTE: the repo's `201326593` used by `vector_modmul` is **composite**
    (`13 × 15486661`). Fine for Montgomery modmul (any odd modulus), but it has no
    prime field / N-th root of unity, so it is **not** usable for the NTT.
- All arithmetic is in `GF(q)`. Every op outputs values already reduced mod `q`.

Because `N = 2^k`, the only decomposition family that matters is **Cooley–Tukey
mixed-radix**. Prime-factor (Good–Thomas), Rader, and Bluestein need coprime/prime
sizes and never apply. Split-radix is the one CT-adjacent variant we *deliberately
drop* (asymmetric, hardware-unfriendly).

---

## 2. The math (Cooley–Tukey four-step, matmul form)

NTT: `X[j] = Σ_n x[n]·ω^{nj} (mod q)`, `ω` a primitive `N`-th root of unity.

Factor `N = N₁·N₂`. With index split `n = n₁ + N₁·n₂`, `j = j₂ + N₂·j₁`, the
transform becomes six tensor ops (constants `W₁`, `W₂`, `T` baked at plan time):

```
1.  Xin = reshape(x, (N2, N1))        # view, free
2.  G   = matmul_modq(W2, Xin)        # (N2×N2)@(N2×N1)  — N2-pt NTTs on all columns
3.  G'  = mul_modq(G, T)              # (N2×N1) elementwise twiddle, T[k2,n1]=ω^{k2·n1}
4.  H   = matmul_modq(G', W1)         # (N2×N1)@(N1×N1)  — N1-pt NTTs on all rows
5.  O   = transpose(H)                # (N1×N2)
6.  out = reshape(O, (N,))            # view
```

**Key fact:** the "batch of small NTTs" is *not* a batched kernel — it is absorbed
into the **free dimension of a single 2D matmul**. A single four-step level is pure
2D, no batching.

Recursing on `W₁`/`W₂` (replacing a sub-NTT by its own decomposition) is what gives
flexibility — but see §5 for why recursion is replaced by a flat sweep under the
2D-only constraint.

---

## 3. Basic kernels (the only things a backend must implement)

All operate on **2D matrices** (no batched/3D tensors), all mod `q`:

| kernel          | signature                         | notes |
|-----------------|-----------------------------------|-------|
| `matmul_modq`   | `(m×k) @ (k×n) → (m×n)`           | optional `transpose_a`/`transpose_b` flags (see §6) |
| `mul_modq`      | `(m×n) ⊙ (m×n) → (m×n)`           | elementwise (Hadamard); broadcast-from-constant ok |
| `transpose`     | `(m×n) → (n×m)`                    | full 2D transpose only — **no per-block transpose** |
| `slice`         | `(…) → (…)` along one axis        | extract a contiguous sub-range (view) |
| `split`         | one tensor → N equal tiles        | tile the data's free dim (§13); one op, many outputs |
| `concat`        | join along one axis               | reassemble tiles (§13) |
| `all_to_all`    | N tiles → M tiles                  | inter-stage re-tile = block transpose (§13); fuses concat→reshape→split |

`reshape` is a free view (no data movement) and is not a real kernel.

---

## 4. Representation — nested n-ary groups

A decomposition strategy is a **nested group expression** over the prime factors of
`N` (here all `2`'s, but written as the chosen radices):

```
Plan := int                      # a leaf factor (a base transform size, dense matmul)
      | (Plan, Plan, ... )       # a group of ≥1 children
```

**Semantics (recursive):**

> A group's **children are the axes swept together** (transform each, CT twiddles
> between). **Everything outside the group is batch** (folded into the free dim).

Example `((a,b),(c,d,e),f)`, `N = a·b·c·d·e·f`:

```
top group [G1=(a,b), G2=(c,d,e), G3=f]  →  sweep super-axes of size ab, cde, f
  transform G1: recursively sweep (a,b),  batch = cde·f
  transform G2: recursively sweep (c,d,e), batch = ab·f
  transform G3: plain matmul,             batch = ab·cde
```

This single form **subsumes both earlier models**:

| form                         | meaning here |
|------------------------------|--------------|
| flat list `(2,4,8,16)`       | one top group, all axes swept, **no batch** |
| binary tree `((4,4),8)`      | every group has 2 children (every split has a batch) |
| hybrid `((a,b),(c,d,e),f)`   | mixed arity — some axes swept together, some batched |

**The grouping structure is exactly the "what is batched" decision** — which is
exactly the decision that requires batched kernels. That is why it is the right axis
to formalize.

Orthogonal attributes (not part of the tree):

```
Strategy = {
  plan:      Plan,                  # the nested-group decomposition
  direction: "DIT" | "DIF",         # twiddle before vs after the stage matmul
  fuse:      bool,                   # opportunistic, diagonal-twiddle only (§7)
}
```

---

## 5. Lowering — two backends, one IR

```
front-end:  ((a,b),(c,d,e),f)
                  │
   ┌──────────────┴───────────────┐
2D-only backend            batched-kernel backend (future)
flatten → [a,b,c,d,e,f]    execute groups as batched sub-sweeps
(grouping ignored)         (grouping = batch granularity + transpose locality)
```

### Why 2D-only forces flattening

Under 2D-only kernels there are **no batched ops**. The "batch" implied by a group is
exactly what would require a **batched / per-block transpose** (transpose each block
independently), which a full 2D `transpose` kernel **cannot** express:

```
full transpose of stacked blocks  ≠  transpose each block in place
```

So any grouped plan is **flattened to a single ordered factor list** and executed as a
**multidimensional sweep**: view `x` as a logical tensor `(f₁,…,f_m)`, transform one
axis per stage. Each stage:

```
reshape   → current factor on rows:   (f_i × N/f_i)     # free view
matmul    → W_{f_i} @ M                                 # batch folded into columns
mul       → ⊙ CT twiddle table                          # mandatory between stages
transpose → full corner-turn to expose the next factor  # the only real data movement
```

Example `[2,4,8,16]`, `N=1024` — stage matrices `(f_i, N/f_i)`:

```
f1=2  : (2, 512)   w_2  @ M   ⊙ twid   [transpose → expose f2]
f2=4  : (4, 256)   w_4  @ M   ⊙ twid   [transpose → expose f3]
f3=8  : (8, 128)   w_8  @ M   ⊙ twid   [transpose → expose f4]
f4=16 : (16, 64)   w_16 @ M            [final reorder for output]
```

**Caution:** the `→` between stages is **transpose + reshape**, not a bare reshape — a
naive contiguous reshape mixes factor bits (e.g. `(2,512)→(4,256)` makes the row
index `n₁·2 + ⌊n₂/2⌋`, not `n₂`). And the **CT twiddles between stages are required**;
without them you compute a multidimensional DFT, not the size-`N` NTT.

Under 2D-only, `((a,b),(c,d,e),f)` and flat `[a,b,c,d,e,f]` execute **identically** —
grouping only yields distinct schedules once batched kernels exist.

---

## 6. Transpose handling

Transposes are the only real data movement; minimize them.

**Free axis selection (no data move):** a 2D matrix exposes exactly 2 axes. Transform
- the **column** axis with `W @ M`,
- the **row** axis with `M @ W`.

Choosing the multiply side = a transpose with no data movement. Because the matrices
are **constants**, we can pre-store `Wᵀ` and carry a logical "transposed" flag.

**Transpose-elimination lowering pass:**

```
- track a logical "transposed" flag through the schedule
- realize axis selection by matmul SIDE, using pre-stored Wᵀ
- emit matmul_modq with transpose_a/transpose_b when the kernel supports it
- pre-permute constant twiddle tables at plan time to match the emitted layout
→ deletes transposes adjacent to a matmul-with-constant
```

**What it removes:** **2-factor → zero transposes.**

**What it cannot remove (the irreducible residual):** with **≥3 factors** a 2D matmul
only ever exposes 2 of the ≥3 logical axes; the third is *folded* and no multiply-side
choice or constant pre-transpose can reach it. Re-exposing it is **genuine data
movement** — a full corner-turn transpose. This residual:

- stays as an explicit `transpose` op on a **dense contiguous-only matmul**,
- is mostly **absorbed** if `matmul_modq` supports a `transpose_b` flag (read columns),
- can be driven to **zero** with strided/gather IO (Stockham-style addressing).

**Design lever:** assume a `transpose_a/transpose_b`-capable `matmul_modq`. That single
capability collapses the residual corner-turns.

**Two folding strategies (both implemented in ntt_passes.py):**

1. *Operand-side* (`eliminate_transposes`): fold a transpose that feeds **into** a
   matmul, via `transpose_a/b`, when separated only by identity reshapes. Removes the
   2-factor corner-turn; ≥3-factor residual stays (transpose feeds a non-identity
   reshape). This is the conservative §6 result.

2. *Result-side* (`fold_result_transposes`): fold a transpose **of** a matmul result
   using `transpose(A·B) = Bᵀ·Aᵀ` — reorder operands, pre-transpose the constant W and
   the twiddle T (free at plan time), and set `transpose_a` on the **data** operand:

   ```
   transpose(matmul(W,M) ⊙ T)  ->  matmul(M, Wᵀ, transpose_a) ⊙ Tᵀ
   ```

   This fires at the matmul that *produced* the data, so it sidesteps the
   non-identity-reshape blocker and removes the transpose at **every** stage (verified:
   2/3/4-factor all → 0 explicit transposes, output bit-identical). The cost is a
   `transpose_a` read on the data operand — exactly the §6 lever. On a kernel without
   that capability the `transpose_a` is itself a real transpose (cost relocated, not
   deleted), consistent with the "dense contiguous-only" caveat above.

---

## 7. Twiddle fusion (NOT general)

Fusion = bake twiddle constants into an adjacent matmul's constant matrix at plan time
(both are compile-time constants), removing a `mul_modq` pass. **Only valid when the
twiddle is diagonal w.r.t. the matmul's contraction axis and shared across the batch:**

- intra-butterfly twiddle: `[[1,W],[1,−W]] = [[1,1],[1,−1]]·diag(1,W)` ✅
- negacyclic ψ pre/post weights (RLWE) ✅
- inverse `1/N` scaling ✅

The **four-step inter-stage twiddle** `T[k₂,n₁]` couples a batch axis with the next
contraction axis → **not** foldable into one shared matrix (would need a distinct
weight per batch row). It stays a standalone `mul_modq`.

⇒ `mul_modq` is always a required kernel; fusion only removes *uses* of it, never the
kernel. Canonical IR is **unfused**; fusion is an optional plan-time rewrite, validated
against the golden reference.

---

## 8. IR

Python dataclass op-list, **MLIR-printable** (SSA values, ops with attributes). Not a
real MLIR dialect yet — but shaped so it can lower to **tt-mlir** later.

### 8.1 Class structure

Every instruction is one `Op`: an **opcode**, its **typed I/O** (operands + result,
each an SSA `Value` carrying a shape), and **optional attributes**. Shapes are tracked
on every value so the planner/passes can verify shape correctness without running.

```python
class OpCode(Enum):
    CONST       = "const"        # materialize a baked constant (twiddle / NTT matrix / ψ)
    RESHAPE     = "reshape"      # free view, no data movement
    TRANSPOSE   = "transpose"    # full 2D transpose
    MATMUL_MODQ = "matmul_modq"  # 2D modular matmul, optional operand-transpose flags
    MUL_MODQ    = "mul_modq"     # elementwise (Hadamard) mod q

@dataclass(frozen=True)
class Value:
    name:  str                   # SSA name, e.g. "%3", "%x"
    shape: tuple[int, ...]       # 1-D or 2-D; dtype is implicitly int in [0, q)

@dataclass
class Op:
    opcode:   OpCode
    operands: list[Value]        # input values (by reference)
    result:   Value              # output value (name + shape)
    attrs:    dict               # optional per-op attributes (see 8.3)

@dataclass
class Program:
    q:       int                 # field modulus, shared by all ops
    inputs:  list[Value]         # program inputs (e.g. the length-N vector)
    ops:     list[Op]            # SSA, topologically ordered
    output:  Value               # the value returned
    consts:  dict[str, "ndarray"]  # name -> baked constant data (CONST results)
```

Invariants the verifier checks: SSA (each `Value.name` defined once, used after def),
shape rules per opcode (8.2), and that every numeric op carries `q == Program.q`.

### 8.2 Per-op I/O + shape rules

| opcode        | operands (shape)              | result (shape)          | rule |
|---------------|-------------------------------|-------------------------|------|
| `CONST`       | —                             | `attrs.shape`           | data in `Program.consts[attrs.name]` |
| `RESHAPE`     | `x (S)`                       | `attrs.shape`           | `prod(S) == prod(attrs.shape)`; no compute |
| `TRANSPOSE`   | `x (m,n)`                     | `(n,m)`                 | full 2D transpose |
| `MATMUL_MODQ` | `a (m,k)`, `b (k,n)`          | `(m,n)`                 | `k` matches; flags swap operand dims first (8.3) |
| `MUL_MODQ`    | `a (m,n)`, `b (m,n)`          | `(m,n)`                 | shapes equal (constant `b` may be the same shape) |

### 8.3 Optional attributes

| opcode        | attribute      | type   | default | meaning |
|---------------|----------------|--------|---------|---------|
| `CONST`       | `name`         | str    | —       | key into `Program.consts` (e.g. `"W8"`, `"T0"`, `"PSI"`) |
| `CONST`       | `shape`        | tuple  | —       | declared shape of the constant |
| `RESHAPE`     | `shape`        | tuple  | —       | target view shape |
| `MATMUL_MODQ` | `transpose_a`  | bool   | `False` | read `a` transposed (operand stored `(k,m)`) |
| `MATMUL_MODQ` | `transpose_b`  | bool   | `False` | read `b` transposed (operand stored `(n,k)`) — the §6 lever |
| `*` (numeric) | `q`            | int    | —       | field modulus; must equal `Program.q` |

Attributes are where lowering passes record decisions: the transpose-elimination pass
(§6) sets `transpose_a/transpose_b` and pre-transposes the matching `CONST`; twiddle
fusion (§7) folds a `MUL_MODQ` constant into an adjacent `CONST`. The op set itself
never grows — passes only toggle attributes and rewire operands.

### 8.4 Textual (MLIR-printable) form

```
module {q = 167772161}
  %x   = input            : 128
  %w8  = const "W8"       : 8x8
  %t0  = const "T0"       : 8x16
  %0   = reshape %x       {shape=[8,16]}
  %1   = matmul_modq %w8, %0          {q=167772161}            : 8x16
  %2   = mul_modq %1, %t0             {q=167772161}            : 8x16
  %3   = matmul_modq %2, %w16 {transpose_b=true, q=167772161} : 8x16
  %4   = transpose %3                                          : 16x8
  %out = reshape %4       {shape=[128]}
  return %out
```

---

## 9. Backend now: numpy reference (golden oracle)

- Bit-exact mod-`q` interpreter for the IR.
- Modular matmul must avoid int64 overflow (`q<2^28` ⇒ product `<2^56`; reduce
  per-row, or use Python `object` dtype). Reuse the approach in `ntt_matrix.py`.
- **Verification:** every strategy/IR for size `N` must match:
  - `ntt_matrix.py` (schoolbook `O(N²)` reference), and
  - `ntt_butterfly.py` (radix-2 `O(N log N)`),
  for several plans (flat lists, trees, mixed radices) at `N = 1024, 32768`.
- Round-trip `intt(ntt(x)) == x`.

---

## 10. File layout

```
algorithm/
  ntt_matrix.py     # DONE — schoolbook matrix NTT (golden reference)
  ntt_butterfly.py  # DONE — radix-2 Cooley–Tukey (cross-check)
  ntt_ir.py         # DONE — IR op dataclasses + numpy interpreter + MLIR printer + verifier
  ntt_planner.py    # DONE — plan → IR (flatten lowering, flat-sweep emission, fwd/inv)
  ntt_passes.py     # DONE — transpose-elimination + twiddle-fusion rewrites
  test_ntt_ir.py    # DONE — vs ntt_matrix / ntt_butterfly, round-trip, grouping≡flatten, passes
  PLAN.md           # this document
```

Implementation status: Phases 1–4 complete and green; pass-level §6/§7 rewrites done
(Phase "Optimization passes"). Output convention: **natural input → bit-reversed
output** (digit-reversal perm → natural; fwd/inv pair cancels it). Verified bit-exact
for many plans at N=1024 (vs both references) and N=32768 (vs butterfly).
Remaining (Phase 5, deferred): negacyclic ψ NTT, DIT variant, tt-metal backend,
batched-kernel backend + cost-model search.

---

## 11. Implementation phases

1. **IR core** (`ntt_ir.py`): op dataclasses, numpy mod-`q` interpreter, MLIR printer.
   - Implement `matmul_modq` (with `transpose_a/b`), `mul_modq`, `transpose`, `reshape`,
     `const`.
2. **Planner** (`ntt_planner.py`): parse a nested-group `plan`; flatten to an ordered
   factor list for the 2D-only backend; emit the per-stage
   `reshape → matmul → twiddle → transpose` IR with correctly baked constants
   (`W_{f_i}` and the CT twiddle tables). Derive `ω`, twiddles from `q` and `N`.
   Support `direction` (DIT/DIF).
3. **Verify** (`test_ntt_ir.py`): cross-check several plans at `N=1024, 32768` against
   `ntt_matrix` / `ntt_butterfly`; round-trip; assert flat-list ≡ grouped (post-flatten).
4. **Optimization passes** (`ntt_passes.py`): transpose-elimination (side-switch +
   `transpose_b` + pre-transposed constants), twiddle-fusion (diagonal only). Each pass
   re-validated against the golden reference.
5. **Extensions (later):**
   - Negacyclic (ψ) NTT for `x^N+1` (RLWE) — pre/post diagonal weights, fusible.
   - Inverse NTT (`ω⁻¹` matrices + `1/N`).
   - tt-metal backend consuming the same IR.
   - Batched-kernel backend that executes groups as batched sub-sweeps (makes the
     grouping structure meaningful), plus a cost model to search the strategy space.

---

## 13. Tiling the matmul to a tile size (`tile_dataflow`, fold mode)

`tile=T` config + `tile_dataflow(prog, T)` pass. Runs on the **fold-mode (transpose-free)**
program, so there are no transposes to worry about — the only layout op between stages
is a `reshape`. For each stage `matmul(data, Wᵀ, transpose_a) [⊙ Twᵀ]`:

- **`split` the NTT-data operand on its free (non-reduction) dim** into equal tiles —
  one `split` op, many outputs. Chunk = **largest divisor of the dim ≤ T** (even tiles,
  no ragged remainder).
- **k (reduction) is never split**; the weight is shared across tiles.
- **one matmul per tile**, then the **twiddle `mul` runs directly on each tile** (the
  twiddle is pre-split into per-tile constants — preprocessed, no runtime op on it).
- **`concat` the tile results** before the stage-boundary `reshape`. No transpose.

So while the computation fits the tiling we do **no** layout transformation; the only
boundary work is `concat → reshape` (→ next stage's `split`). Pure partition → output
bit-identical. Verified across plans × T∈{8,16,32}, including sub-tile factors (4, 8).

Example `(4,4)`, `T=2`: `split %cur {parts=2}` → 2×(matmul→mul with `T0__r0/r1`) →
`all_to_all` → next stage. Tiling requires `transpose="fold"`.

**Inter-stage transition = `all_to_all` (block transpose).** Within a stage the tiles
are embarrassingly parallel (per-tile matmul + twiddle, no cross-tile traffic). The
re-tile between stages — `concat → reshape → split` — is a **block transpose**: each
destination tile gathers from (essentially) every source tile, i.e. **all-to-all**
communication (the FFT "corner-turn"). `fuse_all_to_all` collapses that trio into a
single `all_to_all` op (N tiles → M tiles) so the one expensive communication step is
explicit. Compute is local; all-to-all lives only at the `m−1` stage boundaries — so
fewer/larger factors = fewer all-to-all rounds (the radix trade-off).

Deferred: tiling k (reduction) would need cross-tile accumulation (`add_modq`) — out of
scope per design (k is never tiled).

## 12. Open knobs (deferred, not blocking)

- `transpose_b` capability of the device `matmul` (decides residual transpose count).
- Cost model for searching strategies (op count, transpose count, weight bandwidth).
- Whether to expose a real MLIR `ntt` dialect lowering to tt-mlir.
```

## The three `.mlir` are committed on purpose

`example.py` writes `example_ntt_keep.mlir` (all transposes), `example_ntt_fold.mlir`
(transpose-free) and `example_ntt_tiled.mlir` (fold + tiled), so they are generated output — but
`viz.py` loads them and committing them means reading the lowered IR needs no Python. Same rule as
the committed `.bin` fixtures (see `reference/pir/.gitignore`).

`example_ntt.json` is NOT committed, and the difference is the point: nothing read it except
`example.py`'s own round-trip self-check, and it was ~1 MB. Running `python3 example.py` regenerates
all four; they are deterministic, so a clean `git status` afterwards is the check.
