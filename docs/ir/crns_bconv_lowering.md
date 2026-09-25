# CRNS base-conversion — L2/L3 IR lowering design

**Status:** L2 and L3 lowering **implemented + validated** (`run(L1)==run(L2)==run(L3)`, exact +
signed, multiple base sizes). L2 form is **explicit** (matmul + elementwise visible). See §9.
**Context:** part of getting the full `ct_mul` (tensor → rescale → relin → add) through the HE
compiler to L2/L3 and the scheduler. Base conversion is the default/general primitive under
rescale and relin, and its default realization is **CRNS** (Cox-Rower / Kawamura, m_sk-free).
**Related:** `docs/bconv/base_conversion_overview.md` §2.5,
`reference/he/bfv_bconv_crns.py`, `compiler/lower.py` (`_emit_base_convert_crns`),
`compiler/reference.py` (`base_convert_via_crns`), `tests/baseconv/baseconv_matmul_crns.cpp`,
`kernels/shared/compute/eps_correct_crns.cpp`.

---

## 1. Goal

Lower `rns.base_convert` (default = CRNS) to L2/L3 ops **built from the existing int8-matmul
machinery** — decompose + low-precision (int8) matmul + recombine — so that:

- there is **no new compute engine** (no separate "wide matmul", no separate SFPU accumulator pass);
- `matmul_wide` and `fixed_slice` are expressed *within* the radix-plane representation the int8
  matmul already produces;
- the CRNS correction (`k·g`, and the signed `−neg·M`) reuses `recombine_modq`;
- the structure is visible as **matmul + elementwise ops** at L2 (for the "MLIR of L2 and L3"
  artifact) and fuses at the plane level at L3.

---

## 2. Background — what CRNS base_convert computes

Exact RNS base change `x mod src → x mod dst`, without materializing the big integer, no sentinel
`m_sk`. From `reference.base_convert_via_crns`:

```
E[i][j] = I_i mod dst_j         I_i = ICRT factor = (M/s_i)^{-1}·(M/s_i) mod M,  M = ∏ src
f_i     = ⌈ I_i · 2^u / M ⌉     (fixed-point, integer)
g_j     = −M mod dst_j

s   = Σ_i r_i · f_i             # wide fixed-point accumulator ≈ 2^u·(k + x/M)
k   = s >> u                    # CRT overflow (integer part)
neg = (s >> (u-1)) & 1          # sign bit (signed/centered sources only, e.g. rescaled ẽ)
out_j = ( Σ_i r_i·E[i][j] + k·g_j − neg·(M mod dst_j) ) mod dst_j
```

`u = crns_safe_u(src)` guarantees exactness for all `x`; for BFV-bounded inputs
(`x < ∏src / max(src)`, a full spare limb) far smaller `u` is exact — see the u-sweep in
`bfv_bconv_crns.py`.

### `matmul_wide` and `fixed_slice` (the two CRNS-only ops today)

- **`matmul_wide`** (`reference.py:151`) — integer matmul with **no modular reduction**. The one use
  is `s = F @ RES` (F = the f-vector). "Wide" because `s` is a fixed-point value whose *high* bits
  (≥ u) are the overflow `k`; reducing mod anything would destroy exactly the bits you need.
- **`fixed_slice`** (`reference.py:161`) — arithmetic right-shift (+ optional low-bit mask) that reads
  bit-fields out of `s`: `k = s>>u`, `neg = (s>>u-1)&1`.

---

## 3. The three implementations diverge structurally

Same math, three different placements of the correction — important to keep straight.

| | matmul contains… | overflow (ε/k) computed on… | correction applied… |
|---|---|---|---|
| **JAX reference** (external: Google [jaxite](https://github.com/google/jaxite), `jaxite_ec/finite_field_context.py` `_jax_crns`) | `[E | f_T]` stacked → **one** matmul | same matmul (f_T column) | element-wise `x_n + v·g` |
| **IR lowering today** (`_emit_base_convert_crns`) | augmented `[E_.j, g_j, −M]` matmul_modq **+** separate `matmul_wide` for `s` | `matmul_wide` (matrix engine) | folded into the augmented matmul |
| **Device** (`baseconv_matmul_crns` + `eps_correct_crns`) | **E only** (base extension) | **SFPU** fixed-point MAC, kept `< 2^31` (no wide acc) | element-wise on SFPU (P3) |

Key facts:
- The **JAX ref** puts the `f_T` correction limb **inside** the matmul; only `x_n + v·g` is elementwise.
- The **device matmul does NOT include the correction limb.** P1 (`matmul_fbc_packed`) computes the
  E base-extension planes only; P3 (`eps_correct_crns.cpp`) computes ε on the SFPU as
  `ε = (Σ_i (y_i>>d)·gs_i + off) >> sh` with `gs_i = round(2^32/q_i)`, pre-shifting `y_i>>d` (d=16)
  so the accumulator stays `< 2^31` — **there is no `matmul_wide` on the device** (kernel header:
  "no wide accumulator"). ε is bounded small (`ε < L ≤ 8`, see `face_modreduce`).
- The **IR lowering as written matches the JAX/reference structure, not the deployed device kernel.**

This is the divergence the new design resolves.

---

## 4. Agreed design — CRNS as decompose + int8 matmul + recombine

The insight: `matmul_modq` at L3 already is **decompose (radix-128 → 4 int8 limbs) → int8 matmuls →
recombine_modq (Σ radix^s·plane_s, then mod q)**. CRNS reuses all of it with an **augmented weight**
`[E | f_T]`. The un-recombined **radix planes ARE the wide accumulator** — so `matmul_wide` is just
"recombine the f_T column by **place value instead of mod q**," and `fixed_slice` is a shift across
those chunks.

### Flow

```
R (src residues) ─decompose─┐
                            ├─ int8 matmul against  [ E | f_T ]  ─→  radix-plane matrix
[E | f_T] ─decompose────────┘
                                        │
                   ┌────────────────────┴─────────────────────┐
              E columns' planes                         f_T column's planes
                   │                              (the wide accumulator, as radix chunks — NO mod)
                   │                                          │
                   │                                    recombine-wide  →  s   (2× i32, ~58 bit)
                   │                                    fixed_slice(s, >>u [, >>u-1 & 1])
                   │                                          │
                   │                                    k  (now ONE i32)     [+ neg bit]
                   │                                          │
                   │                          k·g_j  ─decompose→ radix planes   ("special solve")
                   │                              (signed: also  neg·(M mod d_j) → planes)
                   │                                          │
                   └──────────── plane-wise add ─────────────┘
                                        │
                                 recombine_modq   ( Σ radix^s·plane_s ) mod d_j
                                        │
                                  x mod d_j   ✅ exact
```

### Why it is correct
`recombine_modq` is linear and reduces once, at the end, so adding `k·g`'s plane decomposition to the
E-extension planes and recombining once gives
`recombine(E_planes + kg_planes) = (acc_wide + k·g_j) mod d_j = x mod d_j`. The plane sums must stay
`< 2^31` before recombine (they do: base int8 planes are bounded, and `k·g`'s planes are small).

### Why it is good
- **Matches the device's kernel-reuse philosophy** (P1/P2 reuse `baseconv_matmul_packed`), but cleaner:
  ε becomes an extra matmul *column*, not a separate SFPU pass.
- **More exact than the current device kernel.** The device shaves precision (`y_i>>d` before the
  SFPU MAC) to avoid a wide accumulator; here `f_T` rides the **exact int8 matmul**, so `s` is exact
  and `>>u` is exact. The only approximation left is the `f_i = ⌈I_i·2^u/M⌉` rounding, already
  verified in `bfv_bconv_crns.py`.
- **Op-set growth is ~one op:** `recombine_wide` (place-value combine, no mod). Everything else
  (`decompose`, int8 `matmul_modq`, `add`, `recombine_modq`, `fixed_slice`) already exists.

---

## 5. Tiling belongs at L3 (decompose → tile → recombine), not L2

Both matmul-based L2 ops (`matmul_modq` and CRNS) must be **tiled at L3, on the int8 matmuls** — not
tiled at L2 as whole `matmul_modq` ops. This is a core reason L3 exists as a distinct level: **tiling
is a property of the int8 matmul, not of `matmul_modq`.**

**Wrong order — tile at L2 (what to avoid):**
```
matmul_modq  →  tile into T×T blocks  →  lower EACH block to L3
  each block: decompose(A_blk) + decompose(B_blk) + int8 matmuls + recombine → mod q
  then add the recombined partials mod q
```
Every output/K block forces a **recombine (mod-q reduce) at its boundary**, and decompose/recombine
are **replicated per block**. E.g. a matmul tiled 4×4 in the output → 16 `matmul_modq`, each carrying
its own decompose + recombine → large redundant overhead, and the mod-q reduction at every K boundary
breaks the native accumulation.

**Right order — decompose → tile → recombine (tile at L3):**
```
matmul_modq  →  decompose WHOLE A, WHOLE B once  →  int8 matrices
             →  tile the int8 matmuls (normal tiling; K accumulates in int32)
             →  recombine ONCE per output tile → mod q
```
- decompose runs **once per matrix** (not once per tile);
- the **K-dimension accumulates natively in the int32 accumulator** across tiles — no intermediate
  mod-q reduction ("int8→int32 matmul accumulates across calls", see the int8-matmul memory / TT
  datapath);
- **recombine / reduce only at the very end**, once per output tile.

**CRNS tiles the same way.** The augmented `[E | f_T]` int8 matmul: decompose `R` and `[E|f_T]` once,
tile + int32-accumulate the int8 matmuls, then recombine at the boundary only — **mod-q recombine**
for the E columns, **wide recombine** for the f_T column (→ `s` → `fixed_slice` → `k`). The `k·g`
correction (one `k` per coefficient broadcast against per-dest-modulus `g_j` — an outer product,
§8) folds into the E planes *before* that final recombine.
So both L2 matmul ops share one L3 story: **decompose once, tile + accumulate int8, recombine once.**

### Implication for the current pipeline (ordering must flip)
`lower_l1_to_l2(factors, T)` currently calls `tile_matmul(out, T)` — it **tiles `matmul_modq` at L2**
— and only then `lower_l2_to_l3` decomposes each resulting tile. That is the **tile-then-decompose**
(wrong) order. This needs to flip to **decompose-then-tile**: run L2→L3 first (one decompose per
matrix), and apply the tiling pass to the **int8 matmuls at L3**, with recombine at the boundary only.

---

## 6. Three things to lock down

1. ~~Pick `u ≤ 28` so `f_T` shares the 4-limb decomposition of `E`.~~ **DISPROVEN by the u-sweep.**
   `u = 28` mismatches ≈50% even for BFV-bounded inputs; exactness needs **`u ≥ 32`** (0 mismatch
   from u=32, `bfv_bconv_crns.py:u_sweep`). At radix-128 (7 bits/limb) `u=32` ⇒ `f_T` needs **5
   limbs**, so the "uniform 4-limb with E" idea does **not** hold. Resolution: the L3 lowering sizes
   **each operand's limb count from its value range** (`_nlimbs_for` in `lower.py`) rather than a
   global `nlimbs=4` — `f` (~2^u) gets `⌈(u+1)/7⌉` limbs, residues/`k` get theirs. (radix-256/byte
   limbs would fit `u=32` in 4, as the jaxite JAX reference does, but bytes are `uint8`, not the signed-`int8`
   non-negative radix-128 the TT matmul needs.)

2. **`recombine_wide` is the one new primitive:** place-value combine of the f_T planes → `s`
   (~58-bit, 2× i32) → `fixed_slice(>>u)` → `k` (1× i32). (Optimization: since `>>u` discards the low
   bits, `k` can often be recombined from the *high* planes only, avoiding materializing full `s`.)

3. **CRNS is a *dedicated* L2→L3 lowering, not generic L2 ops.** Folding `k·g` into `recombine_modq`
   requires injecting the correction at the **plane level**, which only exists after stage-4. So
   `rns.base_convert` keeps its own CRNS lowering pass (plane-aware), rather than expanding to generic
   `matmul_modq + mul + add` at L2 (each would hide its own recombine and couldn't be merged). At L2
   the op stays semantically `base_convert(src→dst, signed)`; the plane fusion happens in its L3
   expansion.

---

## 7. RESOLVED — L2 form is explicit

- **Explicit (preferred):** L2 shows the CRNS structure as matmul + elementwise
  (`matmul_modq(E-aug)` + `fixed_slice` + correction), and the plane-level fusion appears only at L3.
  Better for the "MLIR of L2 and L3" milestone — you see the structure at *both* levels.
- **Opaque:** a single `base_convert` op at L2 that only reveals decompose/matmul/recombine at L3.
  More compact, but hides the structure at L2.

**Chosen: explicit** — `_emit_base_convert_crns` keeps `matmul_wide` / `fixed_slice` / the augmented
`matmul_modq` visible at L2; stage-4 (`lower_l2_to_l3`) does the int8 realization. Inspecting both
levels is the point of the first milestone.

---

## 8. Bit-width notes (to verify during implementation)

**The `k·g` correction is an outer product (one `k` per coefficient).** `k` (the CRT overflow) is a
**single scalar per coefficient**, independent of the destination modulus `j`; `g_j = −M mod d_j`
varies per dest modulus. So the correction is `k ⊗ g` — a `(N,1)` column (one overflow per
coefficient) times a `(1,P)` row (`g` per dest limb) → `(N,P)`, matching the base-extension output
`x_n : (N,P)`. In the jaxite JAX reference (`_jax_crns`) this is `jnp.multiply(vector_v, crns_vector_g)`, an
**elementwise multiply with broadcasting** (`vector_v (N,1)`, `crns_vector_g (1,P)`), *not* an
einsum/contraction. `N`=degree, `Q`=source/before limbs, `P`=dest/after limbs.



- `f_i < 2^u`; with `u ≤ 28` → 4 radix-128 limbs, uniform with `E` (28-bit `E_ij`).
- `s = Σ_i r_i·f_i`: `r_i,f_i < 2^28`, 4 terms → `s < 2^58` → held as ~2× i32 wide value.
- `k = s>>u`: `k < nₛ·max(src)` (reference / unreduced form). **This scales with the source modulus
  count `nₛ`, which reaches ~60 for deep circuits** — so k is NOT a single i32 in general:
  measured max-k bits vs nₛ (28-bit primes): nₛ=4→29b, 8→30b, 16→31b, 32→32b, **60→33-34b**.
  So k crosses signed-i32 (2³¹) at nₛ≈16, unsigned-i32 (2³²) at nₛ≈32, and **needs 2 i32 chunks
  (>32 bits) by nₛ≈48-60**. The `fixed_slice s>>u` is therefore a multi-chunk (cross-limb) shift and
  k must be represented as a **2-chunk (≤~34-bit) value**, not one i32. In radix-128 form k is
  **≤ 5 limbs** across all nₛ≤60 (5·7=35 bits), so the int8 augmented-matmul path handles it via
  `nlimbs_b = _nlimbs_for(nₛ·max(src))` (=5 at nₛ=60) with no change; only native-uint32
  materializations (device `ε`/k registers) need the explicit 2 chunks. `k·g_j` is then ~34+28≈62
  bits before `recombine_modq` reduces it. (The reduced Cox-Rower form would cap `k < nₛ` → ~6 bits =
  1 limb even at nₛ=60, at the cost of a per-source `mod sᵢ`.)
- `k·g_j`: `< 2^31`-ish before reduction → a handful of radix-128 planes; reduced to `< d_j` by the
  final `recombine_modq`.
- Because only `k mod d_j` matters (`Σ r_i·I_i ≡ x + k·M (mod d_j)`), `k` may be reduced before or
  after the multiply — the final recombine reduces regardless.

---

## 9. Implementation status (what landed)

L2 and L3 CRNS `base_convert` lowering is implemented and validated in `compiler/`:

- **`reference.py`** — `recombine_wide(planes, radix, exps)`: place-value combine, **no** modular
  reduction (the wide f-accumulator's exact bignum combine). Added to `OP_TABLE`.
- **`ir.py`** — interpreter case `tile.recombine_wide` (dispatches to `ref.recombine_wide`).
- **`lower.py`**
  - `_nlimbs_for(maxv)` — smallest `nlimbs` with `radix^nlimbs > maxv` (§6 resolution).
  - `_emit_base_convert_crns` — **common CRNS form** (matmul does the base extension only; the
    overflow correction is elementwise). Per dst limb `j`:
    `xE_j = matmul_modq(E_.j, RES, d_j)`, then `out_j = (xE_j + k·g_j − neg·(M mod d_j)) mod d_j` via
    `mul_modq`/`add_modq`/`sub_modq`. `s = matmul_wide(F, RES)`; `k, neg = fixed_slice(s)`. Only the
    `matmul_wide` carries the wide f-vector (`nlimbs_a = ⌈(u+1)/7⌉`); the base-extension matmuls stay
    4-limb. **k never enters a matmul → never int8-decomposed**, so the wide/multi-chunk k
    (§8, up to ~34 bits at nₛ≈60) costs nothing structural.
  - `ir.py` verify — elementwise ops now allow a **raw (`q=None`) operand** (k is a wide value with no
    single modulus), implicitly reduced into the result's prime; all *primed* values must still agree.

  **Superseded (my earlier over-fusion):** an initial version folded `x@E + k·g + −neg·M` into ONE
  augmented `matmul_modq` (`[E|g|−M] @ [residues; k; neg]`), which forced k in as a matmul operand
  and thus int8-decomposed the wide k (5 limbs). That is *not* the common way — k·g is pointwise, not
  a contraction. Replaced by the elementwise form above. (The `−(M mod d_j)`-as-non-negative-residue
  fix from that version is moot now; the elementwise `neg·(M mod d_j)` uses `M mod d_j` directly.)
  - `lower_l2_to_l3` — expands **both** `tile.matmul_modq` and `tile.matmul_wide`; per-operand limb
    counts from attrs; the wide path emits int8 `matmul_wide` pieces + `recombine_wide` (place value,
    no reduce); the mod-q path is unchanged (group by `p+r`, `recombine_modq`).
- **`mlir.py`** — the MLIR emitter renders the whole CRNS structure at L2 (`matmul_wide`,
  `fixed_slice`×2, augmented `matmul_modq`) and at L3 (decompose → int8 `matmul_wide`/`matmul_modq`
  → `recombine_wide`/`recombine_modq`). Artifacts: `example_base_convert_crns_l{2,3}.mlir`.

**Validation:** `run(L1)==run(L2)==run(L3)==(x mod dst)` for exact and signed, over several base
sizes (BFV-bounded inputs). Tensor lowering + all prior self-tests (`reference.py`, `ir.py`,
`lower.py`) still pass.

**Note — `k·g` is elementwise, not folded into a matmul.** The base extension `Σ_i r_i·E_ij` is a
contraction (matmul); `k·g_j` is pointwise (one k per coefficient × scalar `g_j`, no contraction), so
it is done with `mul_modq`/`add_modq` after the matmul. This is what keeps the wide k off the matrix
engine. The only new L3 machinery is `matmul_wide`→`recombine_wide` (for the `s` accumulator).

**Deferred (unchanged):** tiling still happens at L2 (`tile_matmul`) — the decompose-then-tile
reorder (§5) is a later, performance-oriented change, not needed for the correctness milestone.
