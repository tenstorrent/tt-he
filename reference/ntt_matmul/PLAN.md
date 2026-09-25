# Plan: lowering NTT `matmul_modq` to int8 HW — full M/N/K tiling + precision decomposition

**Status: original design plan — the work it describes is built.** Entry point is [README.md](README.md); this file is kept for the derivations, and `docs/ir/ops.md` §4 cites its §4.

## 0. Goal

Extend the NTT IR compiler (`reference/ntt/`) so a high-precision modular matmul
`R = (A @ B) mod q` (`q < 2^28`) is lowered all the way to the **int8 tile matmul** the
Tensix HW exposes (`i8 × i8 → i32`). Two orthogonal extensions:

1. **Tile M, N, *and* K to ≤ tile size `T`.** Today `tile_dataflow` (ntt_passes.py)
   only splits the free (M/N) dim; K (the reduction) is never split (PLAN.md §13, deferred).
   We add K-tiling, which needs cross-tile modular accumulation (`add_modq`).
2. **Decompose each ≤`T` modular matmul into int8 matmuls** (the verified algorithm in
   `reference/matmul/tt_int8_modular_matmul_32x32.py` and the on-device
   `tests/ntt/ntt1024_dev.cpp`). This needs the two new IR ops you identified:
   - **`decompose`** — bitcast one `i32` matrix → 4 `i8` limb matrices (radix-128).
   - **`recombine_modq`** — merge 4 `i32` matmul-result limbs → one `i32` matrix, with mod q.

The numpy interpreter stays the **bit-exact golden** (object/bignum), so every lowering step is
verified against `ntt_matrix.py` / `ntt_butterfly.py` exactly as today.

## 1. Where this fits

- Upstream: `ntt/PLAN.md` (the NTT → 2D-op IR), `ntt_planner.build_program`, the passes in
  `ntt_passes.py` (`eliminate_transposes`, `fold_result_transposes`, `fuse_diagonal_twiddles`,
  `tile_dataflow`, `fuse_all_to_all`).
- Algorithm reference (already device-verified): radix-128 limbs, plane formulation,
  recombine = modmul-by-`128^r` + modular sum — see `reference/matmul/` and `tests/ntt/ntt1024_dev.cpp`
  (`0/1024` vs golden on a real Wormhole; `docs/matmul/history/NTT_1024_ONDEVICE.md`).
- New code lives in `reference/ntt_matmul/` and reuses `ntt_ir.py` (add opcodes) and a new pass
  module; the existing NTT lowering is untouched up to the point these passes run.

## 2. Two-layer lowering picture

```
NTT plan ──ntt_planner──▶ i32 mod-q IR (matmul_modq / mul_modq / transpose / reshape)
        ──existing passes──▶ transpose-free, M/N-tiled
        ──NEW tile_k───────▶ every matmul_modq has M,N,K ≤ T   (adds add_modq across K)
        ──NEW lower_int8───▶ each ≤T matmul_modq replaced by:
                              decompose + fold(mul_modq) + i8 matmul_modq + add_modq + recombine_modq
        ──▶ tt-mlir / TT kernels (decompose_limbs, matmul_compute, modmul, modadd_reduce)
```

Layer A (mod-q) and layer B (int8) are cleanly separated: the int8 ops only appear *inside* the
expansion of a single ≤`T` `matmul_modq`; everything between matmuls stays mod-q.

## 3. Goal 1 — tile K to ≤ T (`tile_k` pass)

`A (M×K) @ B (K×N) mod q` with `K > T` is split into `⌈K/T⌉` K-tiles. Because `mod q` is a ring
homomorphism over addition, partial products may each be reduced and then summed mod q:

```
A @ B mod q  =  ( Σ_kt  (A[:, kt] @ B[kt, :]) mod q ) mod q
```

Lowering (mirrors `tile_dataflow`'s M/N split, but on the shared K dim and with an accumulate):
```
A_kt = slice(A, axis=1, kt)         # M × Kt   (Kt ≤ T)
B_kt = slice(B, axis=0, kt)         # Kt × N
P_kt = matmul_modq(A_kt, B_kt)      # M × N  (partial)
R    = add_modq(P_0, P_1, …)        # M × N  mod q
```
- Reuses `SLICE` (existing) to cut A on axis 1 and B on axis 0 (constants pre-sliced at plan time,
  exactly like the twiddle pre-split in `tile_dataflow`).
- **`add_modq`** (new op, §5.3) accumulates the K-partials mod q.
- Interacts with the M/N tiling: apply `tile_k` after the existing M/N `tile_dataflow` (or fold both
  into one pass) so the final matmuls satisfy `M,N,K ≤ T` simultaneously.
- Chunk size: `Kt = largest divisor of K that is ≤ T` (same `_chunk_size` policy), no ragged tail.

For the NTT the per-stage `K` equals the transformed factor `f ≤ T` already, so K-tiling mainly
matters for (a) factors `> T` and (b) making the int8 contraction fit (§4).

## 4. Goal 2 — int8 decomposition of one ≤T modular matmul (`lower_matmul_int8` pass)

For `R = (A @ B) mod q` (all dims ≤ `T`), one operand is the **parameter** (a CONST — the NTT
matrix `W` / a twiddle) and the other is **dynamic data**. Fixed radix `β = 128`, `nlimbs = 4`
(32-bit → 8-bit, `β^4 = 2^28 > q`). `T` is a user-chosen tile-size parameter (§3, §11).

**Constants are never computed on at runtime.** The parameter operand is precomputed into its int8
limb form *at plan time* and baked directly as CONST tiles — including the `β^p` fold. So there is
**no runtime `mul_modq` and no op that reads a constant**. The dynamic operand gets only a plain
shift+mask `decompose` at runtime.

Put the parameter on the **fold side** (second operand). Baked offline:
```
b_{p,r} = limb_r( (β^p · param) mod q )           # 16 i8 CONST tiles, p,r ∈ 0..3   (PLAN-TIME)
```
Runtime lowering of the matmul:
```
a_0..a_3 = decompose(data)                        # NEW op: 4 i8,  a_p = (data >> 7p) & 127
for r in 0..3:
    C_r  = add_modq( a_0@b_{0,r}, a_1@b_{1,r}, a_2@b_{2,r}, a_3@b_{3,r} )   # i8 matmul_modq + add_modq
R = recombine_modq(C_0, C_1, C_2, C_3)            # NEW op: Σ_r β^r · C_r  mod q
```
Runtime ops: **1 `decompose` + 16 i8 `matmul_modq` + 4 `add_modq` + 1 `recombine_modq`** — zero
`mul_modq`, nothing reading a constant.

**Orientation.** When the parameter is the LHS (`W @ data`), reorient to put it on the fold side via
the `transpose(A·B) = Bᵀ·Aᵀ` identity already used by `fold_result_transposes` (`W` is symmetric for
the NTT, so `Wᵀ = W`), keeping the parameter as the baked fold operand and the dynamic side a plain
decompose. The *only* case that would emit a runtime `mul_modq` is a matmul whose **dynamic** operand
must be the fold side and cannot be reoriented — then the fold runs on dynamic data (still never on a
constant). For the NTT, every matmul's parameter is `W`, so this never arises.

**Why fold into the constant / why 4 result limbs.** Folding `β^p` into the parameter absorbs the
A-side weights, so `recombine_modq` merges exactly **4** limbs (weights `β^0..β^3`) and the pre-mod
merge stays `< 2^42 < q·R` (what the on-device REDC recombine needs). Because the parameter is
constant, the fold is free at runtime. (Un-folded → 16-way merge up to ~`2^65`, infeasible on HW.)

**Why the i8 matmul reuses `matmul_modq`.** `C_r = Σ_p a_p@b_{p,r}` sums `4·K ≤ 4·T` products of two
7-bit numbers → `< 2^21 < q` (NTT primes `q ≈ 2^27`). So each product and each `C_r` is already
`< q`; the `%q` in `matmul_modq`/`add_modq` is an exact no-op. No "raw matmul" op needed. (Only if
`q ≤ 2^21` would a `reduce=false` attr be required — not for these primes.)

**K and the tile budget.** Each i8 `matmul_modq` contracts `K ≤ T` lanes (one limb per lane, planes
kept separate). So `K ≤ T` (Goal 1) is exactly the precondition that lets the int8 matmuls fit one
HW tile — the concrete link between the two goals.

## 5. New IR ops (slot into `ntt_ir.py`, mirroring existing conventions)

`Value` gains an optional `dtype: str = "i32"` (values `"i32"` = field/mod-q element, `"i8"` = raw
limb). Only used by `decompose`/`recombine_modq` semantics and by `print_mlir` (i8 vs i32 tensor
types); the numpy interpreter is dtype-agnostic (all Python ints). Default keeps every existing
Value at `"i32"`, so nothing upstream changes.

### 5.1 `DECOMPOSE` — bitcast i32 → `nlimbs` × i8
- **opcode:** `DECOMPOSE = "decompose"`
- **operands:** `[x]` — one Value `(M, N)`, dtype i32.
- **result:** `list[Value]` of length `nlimbs` (multi-output, like `SPLIT`), each `(M, N)`, dtype i8.
- **attrs:** `{radix: 128, nlimbs: 4}`.
- **verify:** result is a list of `nlimbs` Values, each shape == `x.shape`.
- **interpret:** `out_p = (x // radix**p) % radix` for `p in 0..nlimbs-1` (object arrays; for
  `radix=128` this is `(x >> 7p) & 127`).
- **print:** `a0, a1, a2, a3 = decompose x {radix=128, nlimbs=4} : MxN`

### 5.2 `RECOMBINE_MODQ` — merge `nlimbs` × i32 → i32 (mod q)
- **opcode:** `RECOMBINE_MODQ = "recombine_modq"`
- **operands:** `[C_0, …, C_{nlimbs-1}]` — `nlimbs` Values `(M, N)`.
- **result:** one Value `(M, N)`, dtype i32 (field).
- **attrs:** `{radix: 128, q}`.
- **verify:** all operands share shape; result == that shape; `q == program.q`.
- **interpret:** `val = Σ_r radix**r * C_r ; out = val % q`  (exact bignum).
- **print:** `R = recombine_modq C0, C1, C2, C3 {radix=128, q=…} : MxN`

### 5.3 `ADD_MODQ` — modular sum (K-tiling + per-limb accumulate)
- **opcode:** `ADD_MODQ = "add_modq"`
- **operands:** `[a, b, …]` — ≥2 Values, same shape.
- **result:** one Value, same shape.
- **attrs:** `{q}`.
- **verify:** operands share shape (broadcast allowed like `mul_modq`); `q == program.q`.
- **interpret:** `val = sum(env[o] for o in operands) % q`.
- **print:** `s = add_modq a, b, c {q=…} : MxN`

All three follow the exact verifier/interpreter/printer pattern documented for `MATMUL_MODQ` /
`MUL_MODQ` / `SPLIT` in `ntt_ir.py` (object dtype, `_as_modq`, multi-output handling for `SPLIT`).

## 6. Passes to add (`ntt_matmul/passes.py`)

- **`tile_k(prog, T)`** — split every `matmul_modq` whose K > T into K-tiles (SLICE both operands
  on the shared dim, pre-slice constant operands at plan time), emit per-tile `matmul_modq` +
  one `add_modq`. Compose with the existing M/N `tile_dataflow`.
- **`lower_matmul_int8(prog)`** — fixed `radix=128, nlimbs=4`. For each (≤T) `matmul_modq`: detect
  which operand is a CONST (parameter); **precompute its 16 folded limb tiles
  `b_{p,r}=limb_r(β^p·param mod q)` at plan time and bake them as new CONSTs** (drop the original
  parameter CONST). Emit runtime `decompose` only on the dynamic operand, then per-r `matmul_modq`(i8)
  + `add_modq`, then `recombine_modq`. No runtime `mul_modq`. Reorient (transpose identity) so the
  parameter is the fold side. Tag limb Values `dtype="i8"`.
- Both run **after** `eliminate_transposes` / `fold_result_transposes` / `fuse_diagonal_twiddles`
  (those operate at the mod-q layer and must see ordinary `matmul_modq`s).

## 7. Precision / overflow ledger (all verified by the device impl)

| quantity | bound | note |
|---|---|---|
| limb `a_p`, `b_{p,r}` | `[0,127]` | radix 128 → valid non-negative int8 |
| `Vp = β^p·B mod q` | `< q < 2^28` | exact 4-limb decomposition |
| one i8 product | `≤ 127·127` | int8×int8 |
| `C_r = Σ_p a_p@b_{p,r}` | `< 4·T·127² ≈ 2^21 < q` | fits i32; already reduced |
| merge `Σ_r β^r·C_r` (pre-mod) | `< 2^42 < q·R` | what the HW REDC recombine needs |

So `q` must be `< 2^28` (radix-128 / 4-limb), and `K ≤ T` keeps `C_r < q`.

## 8. Mapping to HW (the lowered ops are 1:1 with verified kernels)

| IR op | TT kernel (in `device/`) |
|---|---|
| `decompose` | `kernels/shared/compute/decompose_limbs.cpp` (shift+mask, `TensorAccessor` int8 I/O) |
| `mul_modq` (fold) | `kernels/shared/compute/modmul_compute.cpp` (Montgomery, ×`β^p`) |
| i8 `matmul_modq` | `kernels/shared/compute/matmul_compute.cpp` (int8 tile matmul) |
| `add_modq` | `kernels/shared/compute/modadd_reduce.cpp` |
| `recombine_modq` | `modmul`(×`β^r`) + `modadd_reduce` (= `tests/ntt/ntt1024_dev.cpp` stages A4a/A4b) |

The full NTT-1024 (N=32×32) already runs this exact op sequence on one Tensix core, so the IR
lowering is a faithful, pre-validated target.

## 9. Verification

1. **Golden equivalence:** run the int8-lowered IR through the numpy interpreter; assert
   bit-identical to (a) the un-lowered mod-q IR, (b) `ntt_matrix.ntt`, (c) `ntt_butterfly` — at
   `N = 1024, 32768`, `q = 167772161`, several plans (flat / tree / mixed radix).
2. **Per-op unit checks:** `decompose`∘`recombine_modq` round-trips a random i32<q matrix;
   `tile_k` output == un-tiled matmul; `lower_matmul_int8` output == the `matmul_modq` it replaced.
3. **HW cross-check:** dump a lowered op sequence for one 32×32 stage and diff against
   `tt_int8_modular_matmul_32x32.py` and the device `ntt1024_dev` intermediates.
4. **Round-trip:** `intt(ntt(x)) == x` with the lowered pipeline.

## 10. Implementation phases

1. **IR ops** — add `DECOMPOSE`, `RECOMBINE_MODQ`, `ADD_MODQ` to `ntt_ir.py` (enum + verify +
   interpret + print + `print_mlir` types) and the optional `Value.dtype`. Unit-test each.
2. **`lower_matmul_int8`** — the §4 expansion on a single `matmul_modq`: bake the parameter's 16
   folded limb CONSTs offline, emit runtime `decompose` on the dynamic operand only; verify vs the
   op it replaces and vs `tt_int8_modular_matmul_32x32.py`.
3. **`tile_k`** — K-tiling with `add_modq`; verify vs un-tiled; compose with M/N tiling so all
   dims ≤ T.
4. **End-to-end** — wire both passes into the planner's `_apply_passes`; full golden + round-trip
   at N=1024/32768; emit `print_mlir` for a tt-mlir backend.
5. **(later)** negacyclic ψ weights; cost model for radix/tile choices; tt-mlir lowering of the new
   ops to the kernels in §8.

## 11b. Decomposed matmul as an MLIR function (fold/expand)

Rather than inline the §4 expansion at every matmul site, emit it as a **reusable MLIR function**
and `func.call` it from `@ntt`. This dedups the body across all stages/tiles, keeps `@ntt` readable
(one call node per matmul), and gives a single natural unit to fold/expand in the viewer.

**IR representation.** `lower_matmul_int8` rewrites each `matmul_modq` into a *macro op* that carries
its lowered body as a nested region (a small sub-`Program`): operands = the dynamic input + the
baked param-limb CONSTs; body = `decompose → 16 i8 matmul_modq → 4 add_modq → recombine_modq`. The
macro keeps the original `matmul_modq` result shape, so upstream/downstream ops are unchanged.

**`print_mlir(..., inline: bool)`** then has two modes from the *same* IR:
- **folded** (`inline=False`): emit one shared `func.func @matmul_modq_i8(%data, %b0..%b15) -> ...`
  holding the body, and a `func.call @matmul_modq_i8(...)` at each site.
- **expanded** (`inline=True`): splice each body inline (today's flat single-block form).

```mlir
func.func @matmul_modq_i8(%data: tensor<32x32xi32>, %b: 16 x tensor<32x32xi8>) -> tensor<32x32xi32> {
  %a0..%a3 = ntt.decompose %data {radix=128, nlimbs=4}
  %c0 = ... ntt.matmul_modq + ntt.add_modq ...        // per r
  %r  = ntt.recombine_modq %c0,%c1,%c2,%c3 {radix=128, q=...}
  return %r
}
func.func @ntt(%x: tensor<1024xi32>) -> tensor<1024xi32> {
  ...
  %g = func.call @matmul_modq_i8(%xin, %Wb0, …, %Wb15) : ...
  ...
}
```
Baked param limbs are passed as call args (or module-level globals); one generic callee serves every
stage. The verifier/interpreter treat the macro by running its body (so the golden is unchanged).

## 11c. Expandable / foldable visualization

`viz.py` currently parses **flat single-block** MLIR only (it says so). Extend it so a matmul node can
be expanded into its int8 sub-graph and folded back, via an option button:

1. **Parse multiple functions + `func.call`.** Index each `func.func` body; a `func.call` becomes a
   node tagged with its callee.
2. **Fold = the call node; expand = the callee body** drawn as a sub-graph. Two UX levels:
   - **Global toggle** (simplest, robust): a `dcc.Checklist`/button "Expand int8 matmul" that swaps
     the whole graph between the folded view (call nodes) and the expanded view (inlined bodies) —
     backed by `print_mlir(inline=False)` vs `inline=True`, or two pre-parsed element sets.
   - **Per-node fold/expand** (nicer): draw the callee body inside a cytoscape **compound (parent)
     node** and use the `cytoscape-expand-collapse` extension so double-clicking a matmul node
     collapses/expands just that one. Color the new ops (`decompose`, `recombine_modq`, `add_modq`)
     by adding them to `COLORS`/`_category`.
3. Recommend shipping the **global toggle first** (small change: a button + switching element sets),
   then the per-node compound-node version.

This pairs 1:1 with §11b: folded graph ⇔ `func.call`, expanded graph ⇔ the function body.

## 11. Decisions & open questions

**Decided (per request):**
- **`T` is a free parameter** chosen by the user — `tile_k(prog, T)` / the tiling honor it; the int8
  path only requires `K ≤ T`. No hardcoded 32.
- **Always 4 limbs** — fixed `radix = 128`, `nlimbs = 4` (32-bit → 8-bit). No generalization.
- **Never compute on constants** — every parameter (NTT matrix / twiddle) is precomputed (folded +
  decomposed) at plan time and baked as CONST; runtime touches only dynamic data. The fold is offline.

**Open:**
1. **Reuse vs new "raw matmul".** Keep reusing `matmul_modq` for the i8 matmul (relies on `C_r < q`,
   true for `q ≈ 2^27`), or add an explicit `reduce=false` attr for robustness if a small `q` is ever
   used? (Recommend: reuse now, add the attr only if needed.)
2. **`add_modq` arity.** Variadic (one op sums all K-tiles / all 4 planes) vs binary — affects how a
   tt-mlir backend schedules the accumulation.
3. **Orientation fallback.** For a hypothetical matmul whose *dynamic* operand must be the fold side
   and can't be reoriented, accept a runtime `mul_modq` on data — does not occur for the NTT (param
   is always `W`), so deferred.
