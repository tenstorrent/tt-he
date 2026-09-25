# RNS Base Conversion on Tenstorrent — Overview

**Audience:** engineering management / reviewers with light project context.
**Scope:** RNS **fast base conversion** `x mod Q → x mod P` — convert the residues of a value
(a polynomial's coefficients) from one set of prime moduli `Q = {q_i}` to another `P = {p_j}`,
without reconstructing the big integer. The core primitive under BFV/CKKS **mod-up, mod-down,
key-switch, and relinearization**.
**Status:** implemented and validated bit-exact on real silicon (n150 / Wormhole B0), single
core; both SFPU and matrix-engine variants, each at 4→4 and 8→8 moduli; approximate and exact.
A **third exactness route (CRNS / fixed-point ε, matmul-native, m_sk-free)** is verified in the
Python reference (§2.5) — device kernel pending.
**Last updated:** 2026-07-17

---



## 1. What base conversion is, and why it is the hard FHE primitive

In RNS, a big integer `x < Q` is stored as its residues `x_i = x mod q_i`. Base conversion
re-expresses the *same integer* under a different prime set `P`. Conceptually it is
"**reconstruct** `x` **from its** `Q`**-residues (CRT), then reduce mod each** `p_j`" — but doing the
reconstruction explicitly would form a multi-hundred-bit integer, exactly what RNS exists to
avoid. The whole game is to get `x mod p_j` *without materializing* `x`.

This is **the** non-trivial RNS-FHE primitive: addition, multiplication and NTT are
element-wise per limb, but base conversion **couples all the limbs** (it is a CRT mix across
the modulus axis), and it is what mod-up / mod-down / key-switch are built from
(`docs/he/bfv_build_plan.md`).

---



## 2. The algorithm



### 2.1 The CRT identity and the ε overflow

With `q̂_i = Q/q_i` and `y_i = [x_i · q̂_i⁻¹]_{q_i} ∈ [0, q_i)`:

```
x  =  ( Σ_i y_i · q̂_i )  −  ε·Q ,      ε = ⌊ Σ_i y_i / q_i ⌋ ∈ {0, …, L−1}
```

so for each destination modulus `p_j`:

```
x mod p_j  =  ( Σ_i y_i·(q̂_i mod p_j)  −  ε·(Q mod p_j) )  mod p_j
```

Two pieces:

- **the linear combination** `Σ_i y_i·(q̂_i mod p_j) mod p_j` — pure modular multiply-add
(this is what the hardware does), and
- **the ε correction** — a *floor of a sum of fractions*, **not** a modular vector op. This is
the one part that falls outside "modular vector op + matmul".



### 2.2 Approximate vs. exact

- **Approximate FBC** drops ε → returns `(x + ε·Q) mod p_j`. The error is a bounded multiple
of `Q`; for **key-switch mod-up/down** it folds into the noise budget, so approximate is the
standard there. Pure modular MAC, no correction kernel.
- **Exact FBC** computes ε and subtracts `ε·(Q mod p_j)`. Needed where the result must be exact
(e.g. the BFV `t/Q` scaling). Two ways to get ε:
  - **host integer** (`ε = ⌊Σ_i y_i·q̂_i / Q⌋`, big-int) — exact, computed off-device;
  - **on-device Shenoy–Kumaresan (SK)** — carry a sentinel modulus `m_sk`, compute
  `A_sk = (x+εQ) mod m_sk`, then `ε = (A_sk − x_sk)·Q⁻¹ mod m_sk` — exact, **all integer
  modular ops on device** (no float). The fp32 (HPS) path is *not viable on Wormhole*: the
  SFPU's 24-bit mantissa is too narrow for a 28-bit-prime ratio, and Wormhole's float→int
  cast has no truncation mode (`RoundMode::Zero` is Blackhole-only).

A **third route** — CRNS / fixed-point ε — recovers `ε` with an integer shift instead of a sentinel
prime; it is matmul-native and needs no `m_sk` limb. See **§2.5**.



### 2.3 Canonical residues are required

The combine reinterprets `y_i`'s integer value under a *different* modulus `p_j`, so each `y_i`
must be the **canonical** representative in `[0, q_i)` — a lazy `[0, 2q_i)` value (`y_i` or
`y_i+q_i`) has a different residue mod `p_j` and gives the wrong answer. Hence every modmul in
the FBC uses the **finalizing** `modmul_sfpi_full` (canonical), not the lazy hybrid. (Mature
RNS libraries do the same — e.g. JAX `BConvBATLazy` forces a strict reduction right before the
combine.)

### 2.4 Montgomery-domain I/O is free (constant merge, no kernel change)

FHE pipelines keep polynomials in **Montgomery form** (`ṽ = v·R mod m`, `R = 2³²`) for the NTT
and pointwise multiplies. Base conversion can take Montgomery-domain RNS limbs in *and* return
Montgomery-domain limbs out at **zero cost** — by folding the `R` powers into the (offline)
constants. Nothing about the device kernels changes, and `y_i` stays **plain/canonical**
internally (still required by §2.3); only the baked constants carry the `R` bookkeeping.

The device modmul is `REDC(a·b) = a·b·R⁻¹ mod m`, so tracking `R`:

- **Input** `x̃_i = x_i·R`: stage 1 must yield **plain** `y_i`.
`modmul(x̃_i, C1) = x_i·R·C1·R⁻¹ = x_i·C1` ⇒ set `C1 = q̂_i⁻¹` (plain, one *fewer* `R` than
the plain-I/O version's `mont(q̂⁻¹)`). For the matmul version, stage 1 is host-side and
produces the same plain `y_i` regardless, so input-Montgomery is absorbed for free.
- **Output** `õut_j = out_j·R`: inject one `R` into the combine's constant.
  - *SFPU fused:* `C2 = c_ij·R²` (double-Montgomery) ⇒ each product `= y_i·c_ij·R`, sum `= õut_j`.
  - *matmul (FPU):* fold `R` into the CRT-fold matrix, `B_r = limb_r((128^p·c_ij·R) mod p_j)`;
  the recombine then reduces `Σ_r 128^r·plane_r = R·Σ y_i c_ij = õut_j` unchanged.


| version         | input `R`                                         | output `R`                         |
| --------------- | ------------------------------------------------- | ---------------------------------- |
| SFPU fused      | drop `R` from `C1` (→ plain `q̂⁻¹`)               | add `R` to `C2` (→ `c_ij·R²`)      |
| matmul / packed | absorbed by host stage-1 (`y_i` plain either way) | fold `R` into `B` (`128^p·c_ij·R`) |


Verified bit-exact against a Montgomery-encoded reference on all three versions, at **identical
latency** (same op count — only constant values differ). Takeaway: the FBC drops into a
Montgomery pipeline with **no convert-in / convert-out step**; the domain crossing lives entirely
in its offline constants (settles the domain-tracking note in `bfv_build_plan.md` §5).



### 2.5 CRNS: exact ε by fixed-point (a third route — matmul-native, no `m_sk`)

The SK sentinel (§2.2) is *one* way to make the FBC exact. A second, from Langowski–Devadas
(eprint 2025/1068, "Change RNS modulus" = **CRNS**), recovers the **same** overflow `ε` with an
**integer fixed-point floor** instead of a redundant modulus. It maps cleanly onto the matrix
engine and drops the `m_sk` channel entirely. **Verified bit-exact in the Python reference
(`reference/he/bfv_bconv_crns.py`); device kernel not yet built.**

**Built and measured on device (n150, 4→4, N=1024): 254 µs — 2.04× faster than the exact SK
kernel (519 µs) and the fastest *exact* FBC.** Reuses the packed combine (P1) and recombine (P2)
kernels unchanged, adds one SFPI program (P3, `eps_correct_crns`) for ε + the correction; exactness
costs only +51 µs (+25%) over the approximate packed matmul (203 µs). See §4/§5 and
`tests/baseconv/baseconv_matmul_crns.cpp`.

**The mechanism.** Fold the stage-1 inverse into a constant `I_i = [q̂_i⁻¹]_{q_i}·q̂_i (mod Q)`
so the raw residues `x_i` drive everything, and precompute three offline constants: a matrix
`E[i][j] = I_i mod p_j`, a fixed-point vector `f_i = ⌈I_i·2ᵘ / Q⌉`, and `g_j = −Q mod p_j`.
Then per coefficient:

```
s   = Σ_i x_i · f_i            (one wide fixed-point accumulator; s/2ᵘ ≈ Σ x_i I_i / Q = ε + V/Q)
k   = s >> u                   ε (the CRT overflow)  — integer part, no float
out_j = ( Σ_i x_i · E[i][j]  +  k · g_j )  mod p_j          combine (x·E) is an int8 matmul
```

`Σ_i x_i·E[i][j]` is exactly the §3.2 radix-128 int8 matmul combine; CRNS only *changes the
constants and adds the `k·g` correction*. The `ε` estimate is a dot-product plus a shift — pure
integer, so it **sidesteps the fp32 obstacle** that ruled out the HPS float route (§2.2) *and*
carries no sentinel prime.

**Exactness condition.** `k` is exact when the fixed-point precision `u` is large enough and the
value is bounded `x < Q` (both hold in the BFV rescale, where `ẽ < ∏S`). Measured on our params
(4 source limbs, `∏S ≈ 112 b`): exact for the BFV-bounded input at **u ≥ 32**, for the full
`[0,∏S)` range at **u ≥ 44**; the guaranteed worst-case bound is `u = ⌈log₂(L·max(q_i)·Q)⌉+1`.
The file's `max_modulus_bits + log₂L + 1` heuristic is a *lower* sanity bound, **not** the
exactness threshold. (Gotcha: compute `f_i` with **integer** ceil `((I_i<<u)+Q−1)//Q` — a float
`math.ceil(·/Q)` silently loses precision at large `u`.)

**Signed values need centering — and `m_sk`'s real rescale job was this, not float-avoidance.**
The BFV rescaled `ẽ = ⌊t·e/Q⌋ − ε` is **signed** (tensor coeffs go negative). Plain CRNS returns
the unsigned `[0,∏S)` representative `V`, giving `(V mod ∏S) mod p_j` — wrong for the ~half of
coefficients where `ẽ < 0` (they land near `∏S`). SK hides this because its `m_sk` also recovers
`⌊W/∏S⌋`; i.e. in the rescale the sentinel is doing **sign / representative recovery**, not just
ε. CRNS gets the sign bit for free from the **same accumulator**:

```
neg = (s >> (u−1)) & 1        set iff V ≥ ∏S/2  ⇔  ẽ < 0    (top fractional bit of s)
out_j = ( Σ_i x_i·E[i][j] + k·g_j  −  neg·(∏S mod p_j) )  mod p_j
```

No big-integer reconstruction — the sign is a second slice of the accumulator CRNS already
computes (`k = s>>u`, `neg = (s>>(u−1))&1`). It is robust because `|ẽ| ≪ ∏S/2`, so `V` sits near
`0` or near `∏S`, never near the `∏S/2` boundary. Verified `0/1024` vs the reconstruct-based
decision.

**What it verifies to.** In `reference/he/bfv_bconv_crns.py`: the standalone S→Q exact conversion
matches both ground truth and the SK path `0/200000`; the full `ct_mul` with the CRNS rescale is
**bit-identical to the `m_sk` rescale and decrypts to `m₁·m₂`, all seeds**.  On device
(`baseconv_matmul_crns`, 4→4): **exact `0/4096` across 5 seeds, 254 µs**.

**Device ε params (u=40).** gs_i=round(2^40/q_i)≈4096, d=16, sh=24, off=2^23 (α=½), giving ε via a
per-lane truncated MAC + shift, accumulator ≤ 2^28 (< 2^31, SFPU-safe). u=40 (not 32) is required:
at u=32 gs≈16, whose ½-LSB rounding is ~3% — larger than the α=½ resolution, so it over-estimates ε
by 1 on some coefficients; u=40 shrinks the rounding to ~0.01% and it is exact (verified 0/200000 in
Python and 0/4096 on device). The product (y_i>>d)·gs_i must use `mul_int_tile` (SFPU has no integer
`*` operator) and stays < 2^31 (low-32 exact). Harness note: the host reference needs Q=∏q_i to fit
u128, so this harness validates L≤4 (Q≤112 b) — the BFV pipeline's regime (Q=3, S=4); the kernel
itself is L-agnostic (params verified for L=8 in Python).

**Trade vs SK.** CRNS replaces (a) the sentinel prime carried through the whole tensor — `Q∪S∪{m_sk}`
drops from 8 limbs to `Q∪S` = 7 — and (b) the extra SK reduction phase (§5, the `+46%` sk_eps +
correction), with **one wide `u`-bit accumulator** (`u ≈ 44` here ⇒ `s` needs ~`u+28+⌈log₂L⌉` ≈
74-bit accumulation) sliced twice. The combine stays the §3.2 int8 matmul. So CRNS is the natural
way to make a **matmul-regime FBC both fully on-device *and* exact** — the open item flagged in §7 —
without the SK channel. Device cost (the wide accumulator vs the SK phase) is not yet measured.

---



## 3. Two hardware regimes for the combine

The linear combination `out_j = Σ_i y_i·C[i][j] mod p_j` (with `C[i][j] = q̂_i mod p_j`) can run
on either Tensix engine, and we built both:

### 3.1 SFPU regime (small L) — element-wise modular MAC

Each `out_j` is a sum of per-`(i,j)` scalar `modmul`s on the **32-lane vector engine**. One
32×32 tile = **1024 coefficients** processed at once; the `L=4` limbs are *separate tiles*.
Constants are pre-Montgomery'd so `modmul(y, mont(c)) = y·c` in the plain domain (base
conversion is **plain-domain**, never Montgomery — Montgomery's per-modulus `R` scaling would
corrupt the cross-modulus reinterpretation; the Montgomery REDC is just how the multiply is
implemented, with `R` folded into the constant).

### 3.2 Matrix-engine regime (large L) — the combine as an int8 matmul

The combine is a matrix–vector product over the limb axis: `out = Y · C`. With **radix-128**
decomposition, **8 source moduli × 4 limbs = 32 contraction lanes** fill the int8 matmul tile
*exactly*, so `out_j[c] = Σ_i y_i[c]·C[i][j]` becomes a 32×32 int8 modular matmul (coefficients
= the 32 tile rows). The **matrix engine** produces the 4 int32 radix planes; a **per-column
recombine** reduces each output column `j` mod `p_j`. This is the same "Basis-Aligned
Transformation" used by GPU RNS libraries (byte-decomposition matmul lowering), specialized to
TT's *signed* int8 → **radix 128** (7-bit limbs) instead of radix 256 (uint8), and Montgomery
recombine instead of Barrett.

#### Tiling: 1024 coefficients × 8 moduli × 4 limbs onto 32×32 tiles

The combine contracts the **source-modulus axis** `i`, so map it to `out = Y·C` with
`Y[c][i]` (dynamic) and `C[i][j]` (constant); the matmul engine contracts the **column** axis of
a 32×32 tile, so the contracted thing (source moduli × their limbs) goes in columns and
coefficients go in rows. Three axes, three placements:

**Axis 1 — the 1024 coefficients → tile ROWS.** A tile holds 32 coefficient-rows, so
N=1024 → **32 coefficient-tiles** (`NCT=32`); tile `t` holds coeffs `[32t, 32t+32)`.

**Axis 2 — source moduli × input radix limbs → 32 tile COLUMNS (the contraction).** Each
`y_i < 2²⁸` splits into 4 radix-128 limbs `p`; with 8 source moduli that is `8×4 = 32`, filling
the column axis exactly:

```
A_t  (one int8 tile per coeff-tile t):                 col = i*4 + p   (i=src 0..7, p=limb 0..3) → 32
  A_t[cp][i*4 + p] = limb_p( y[32t+cp][i] )            row = cp        (coeff in tile, 0..31)
```

**Axis 3 — dest moduli → output COLUMNS; output radix limbs → 4 separate PLANE tiles.** The
constant `C` is folded and decomposed into 4 output limbs `r`, giving 4 shared RHS tiles:

```
B_r  (r = 0..3, int8, constant, shared across all coeff-tiles):   col = j  (dst 0..7) → 8 used
  B_r[i*4 + p][j] = limb_r( (128^p · (q̂_i mod p_j)) mod p_j )      row = i*4+p (matches A's cols) → 32
```

Per coefficient-tile `t`, four int8 matmuls (one per output limb `r`):

```
plane_r[cp][j] = matmul_tiles(A_t, B_r) = Σ_{i,p} A_t[cp][i*4+p]·B_r[i*4+p][j]   ← contracts the 32 (i,p) lanes
```

→ 4 int32 plane tiles `[32 coeff × 8 dst]` per coeff-tile (128 planes total for N=1024). Then the
per-column recombine: `out[cp][j] = (Σ_r 128^r·plane_r[cp][j]) mod p_j` → 32 output tiles.

```
 1024 coeffs ─ 32 row-tiles ─► A_0..A_31      A_t · B_r (r=0..3) ─► 4 planes ─► recombine(Σ128^r) mod p_j ─► out_t
                cols = (i,p)=32 lanes          B_r cols = dst j=8                  per-column modulus
```

Two distinct radix axes (same double-decomposition as the modular matmul): `p` folds the
input into the contraction (A's columns); `r` keeps the output limbs as *separate plane
tiles* so the recombine sums them at the **same lane position** (no cross-lane reduction), with
the per-`p_j` reduction handled as a per-column modulus.

This is the **transpose** of the SFPU layout: there a tile's 1024 elements *are* the coefficients
and each modulus is a separate tile; here coefficients are rows (32/tile → 32 tiles) and the
`8×4` moduli-limbs are the 32-wide contraction. (The 32 int8 A-tiles span >12 DRAM banks, so the
reader uses `TensorAccessor`, not `InterleavedAddrGenFast` — see §5's implementation note.)

### 3.3 The bottleneck is the recombine — and the "block-packing" optimization

Profiling the matrix-engine version (§5) shows the **matmul combine is cheap (~~73 µs) and the
SFPU recombine dominates (~~640 µs, ~88%)**. And the recombine is badly *underutilized*: each
Montgomery reduction runs over a full 32×32 tile (all 1024 lanes), but the output is only the
`L` **destination moduli wide** — at L=8 that is 8 of 32 columns, so **~75% of every reduction
reduces padding**.

The fix (`baseconv_matmul_packed`): pack `NBL = 32/L` coefficient-blocks side-by-side into
one recombine tile so all 32 columns carry real data. The trick avoids any data reshuffle — it
**shifts each block's RHS to a disjoint column range and uses matmul accumulation**:

```
B[bl][r]  carries the CRT-fold limb only in columns [bl·L, bl·L + L),  zeros elsewhere.
per super-block sb, per output limb r:
    plane_r = Σ_{bl=0..NBL-1}  A_{sb·NBL+bl} @ B[bl][r]      (matmul dst-accumulate)
```

Because the `B[bl][r]` occupy disjoint columns, the accumulation lands the `NBL` blocks in
`NBL` separate 8-column (L-column) groups → one **fully-packed** plane. Recombine then runs on
`NSB = NCT / NBL = L` **super-blocks** (per-column modulus `p_{col mod L}`) instead of 32.

The striking consequence: **the recombine count equals** `L`**.**

```
NSB = NCT / NBL = 32 / (32/L) = L        →  L=4: 4 recombines,  L=8: 8 recombines
```

So *smaller* `L` packs more blocks per tile and needs *fewer* recombines — the matmul work is a
constant 128 int8 matmuls regardless of `L`, and only the (bottleneck) recombine count moves.
This is what makes the packed 4→4 the fastest configuration (§5).

---



## 4. The implemented versions

All single-core, N=1024 coefficients, `q,p` distinct ~28-bit primes, validated bit-exact
(`0` mismatches) vs a big-int reference across seeds.


| binary                   | combine                              | L                 | ε / exactness                | programs | new kernels                                            |
| ------------------------ | ------------------------------------ | ----------------- | ---------------------------- | -------- | ------------------------------------------------------ |
| `baseconv`               | SFPU                                 | 4→4               | exact, **host** ε            | 16       | — (modmul_full, modadd)                                |
| `baseconv_sk`            | SFPU                                 | 4→4               | exact, **on-device SK** ε    | 24       | modsub, reader_ab                                      |
| `baseconv_fused`         | SFPU                                 | **4 or 8** (argv) | approximate                  | **1**    | baseconv_fbc                                           |
| `baseconv_fused_exact`   | SFPU                                 | 4→4               | exact (host ε folded)        | **1**    | baseconv_fbc_exact                                     |
| `baseconv_matmul`        | **int8 matrix engine**               | **4 or 8** (argv) | approximate, fully on-device | 2        | matmul_fbc, recombine_rns, reader_fbc/recombine_rns    |
| `baseconv_matmul_packed` | **int8 matrix engine, block-packed** | **4 or 8** (argv) | approximate, fully on-device | 2        | matmul_fbc_packed, reader_fbc_packed (+ recombine_rns) |
| `baseconv_matmul_crns`   | **int8 matrix engine, block-packed** | 4→4 (harness)     | **EXACT, fully on-device (CRNS ε, no m_sk)** | 3 | eps_correct_crns, reader_eps_correct (+ matmul_fbc_packed, recombine_rns reused) |


"programs" = device program dispatches; the fused versions collapse the multi-launch pipeline
into one, keeping the `y_i` intermediates in L1 (no DRAM round-trips between stages).
`baseconv_fused`, `baseconv_matmul`, and `baseconv_matmul_packed` take `L ∈ {4,8}` on argv, so
every regime is measured at both moduli counts (§5). `baseconv_matmul_packed` is the §3.3
block-packing optimization (fewest recombines).

`baseconv_fused`, `baseconv_matmul`, and `baseconv_matmul_packed` use **Montgomery-domain I/O**
(§2.4) — Montgomery limbs in and out, via the constant merge, at no latency cost.

---



## 5. Performance

Measured on n150, single Tensix core, **all at N = 1024 coefficients**, 100-iteration
steady-state device latency (host setup/validation excluded). Both engine regimes tested at
**4→4 and 8→8** moduli. The two regimes pack data differently — the SFPU keeps 1024 coefficients
per tile (limbs are separate tiles), the matmul puts coefficients on the 32 tile rows so 1024
coeffs = 32 coefficient-tiles — but the **op is the same size**, so the µs are directly
comparable.


| version                                                    | 4→4 (N=1024) | 8→8 (N=1024) |
| ---------------------------------------------------------- | ------------ | ------------ |
| **SFPU fused** (`baseconv_fused`)                          | 356 µs       | 1277 µs      |
| **matrix engine** (`baseconv_matmul`)                      | 732 µs       | 733 µs       |
| **matrix engine, block-packed** (`baseconv_matmul_packed`) | **204 µs**   | **285 µs**   |


**Exact FBC, fully on-device — two routes (4→4, N=1024):**

| exact kernel | ε method | µs | vs SK |
| --- | --- | --- | --- |
| `baseconv_fused_sk` | Shenoy–Kumaresan sentinel `m_sk` (SFPU combine) | 519 | 1.0× |
| **`baseconv_matmul_crns`** | **CRNS fixed-point ε (FPU matmul combine + SFPI ε)** | **254** | **2.04× faster** |

CRNS-exact is the fastest exact FBC: it moves the combine onto the FPU matrix engine (packed) and gets
ε from a cheap SFPI truncated MAC + shift instead of the SK sentinel's modular-reduction phase.
Exactness costs only **+51 µs (+25 %)** over the approximate packed matmul (203 µs), vs the SK route's
+46 % over its SFPU baseline. Verified exact `0/4096`, 5 seeds.

(Exact variant — **fully on-device ε**, `baseconv_fused_sk` 4→4 = **520 µs**. This recovers the CRT
overflow ε on the device via the integer Shenoy–Kumaresan sentinel `m_sk` — a real extra SFPU phase
(A_sk = Σ yᵢ·(q̂ᵢ mod m_sk), then ε = (A_sk − x_sk)·Q⁻¹, then per-limb correction), so it costs
**+46%** over the 356 µs approximate, *not* the ~1% of the old host-ε shortcut. The earlier
`baseconv_fused_exact` folded ε on the host — removed, since a real pipeline never has the plaintext
to precompute ε from; on-device ε is the only correct exact FBC.)

### 5.1 Latency decomposition (per-invocation device zones, cycles ≈ ns at 0.985 GHz)

Where the time goes, read on the MATH core (TRISC_1):


| version (8→8)        | combine                      | reduce (recombine / phase 2)   | dominant          |
| -------------------- | ---------------------------- | ------------------------------ | ----------------- |
| SFPU fused           | phase 1: ~140 µs (8 modmuls) | phase 2: ~1128 µs (64 modmuls) | **reduction 89%** |
| matrix engine        | matmul: ~73 µs               | recombine: ~641 µs (32 tiles)  | **reduction 88%** |
| matrix engine packed | matmul: ~73 µs               | recombine: ~160 µs (8 tiles)   | reduction ~55%    |


**In every version the SFPU modular reduction is the bottleneck** — the same canonical Montgomery
mulmod as the vector-modmul primitive (`no native integer multiply → reduction dominates`). The
matmul versions win by moving the combine's `L²` multiplies onto the FPU (~73 µs); the *packed*
version additionally cuts the recombine from 32 tiles → `L` tiles (§3.3), which is the largest
single drop.

### 5.2 Reading the numbers

- **SFPU fused scales with** `L²`**.** Combine = `L + L²` modmuls: 4→4 = 20 (356 µs), 8→8 = 72
(1277 µs) — 3.6× for 2× the moduli, the `L²` term.
- **Unpacked matmul is** `L`**-flat** (732 ≈ 733): the 32×32 tile is fixed; L=4 just zero-pads it.
- **Packed matmul is the fastest, and *smaller* L is faster** (204 µs at 4→4 vs 285 at 8→8) —
because the recombine count is `NSB = L` (§3.3): 4 reductions at L=4, 8 at L=8. The combine is a
constant 128 int8 matmuls; only the bottleneck recombine count moves with L.
- **Best-to-date: packed 4→4 at 204 µs** — 1.75× over SFPU fused (356) and 3.6× over the unpacked
matmul (732); packed 8→8 at 285 µs is 4.5× over SFPU fused (1277) and 2.6× over unpacked (733).
- **On-device exact ε is not free** (356 → 520, **+46%**): the SK `m_sk` channel is a full extra
reduction phase (per-call device zones: phase1 ~69 µs, sk_eps ~87 µs, phase2+correction ~348 µs).
The old "+1%" figure was the host-ε shortcut (removed) — it only added one modular-add because the
host handed ε in as a constant, which a real pipeline cannot do.
- **Fusion eliminates dispatch/DRAM**: the multi-launch SFPU versions (`baseconv` 16 programs,
`baseconv_sk` 24) pay a dispatch + DRAM round-trip per stage; the fused/matmul versions above
dispatch once (fused) or twice (matmul: combine + recombine).

> **Implementation note** (a TT gotcha worth recording): the matmul versions' int8 LHS spans
>
> > 12 DRAM banks, so their readers must use `TensorAccessor`, not `InterleavedAddrGenFast` —
> > the latter mis-addresses sub-4096-byte (int8) tiles past bank 12 (manifested as exactly the
> > tiles ≥ 12 being wrong). The int32 buffers (4096-byte pages) are unaffected.

---



## 6. Crossover and scaling

Two things to read from §5:

**Unpacked: SFPU vs matrix engine crosses over between L=4 and L=8.**

- L=4: unpacked SFPU (356) beats unpacked matmul (732) — the matmul tile is half-empty.
- L=8: unpacked matmul (733) beats SFPU (1277) — full tile vs the SFPU's `L²` modmuls.

**Packed changes the picture: the block-packed matmul is fastest at both L**, and *smaller* L is
faster (204 µs at 4→4, 285 at 8→8). Once you pack the coefficient-blocks (§3.3), the matmul tile
is always full and the only L-dependent term is the recombine count `NSB = L`. So the packed
matrix engine dominates the SFPU everywhere here (1.75×–4.5×), and there is no crossover — it is
the right choice at all L we measured.

Underlying scaling:

- **SFPU fused** ≈ `L + L²` modmuls → quadratic in `L`.
- **packed matmul** = constant 128 int8 matmuls (combine) + `L` **recombine reductions** → the
bottleneck grows only *linearly* in `L`, on the FPU-offloaded combine plus a small SFPU term.

So for **real BFV (**`L ≈ 30–60`**)** the packed matrix-engine lowering is clearly the path: combine
stays on the FPU, and the recombine cost scales as `L` (with the CRT matrix tiled across multiple
32×32 int8 matmuls, the `matmul_compute_mtk` K-loop pattern, once `L > 8`). Same "matmul is cheap,
reduction dominates" story as the NTT / modular-matmul work — and the packing is what keeps the
reduction from dominating.

Remaining recombine headroom (the ~160 µs at packed 8→8): the per-tile reload of the 5 param
tiles is loop-invariant and could be hoisted; and the two device programs (combine, recombine)
could be fused to save a dispatch + the plane DRAM round-trip.

---



## 7. The "all-on-device" status


| stage                      | SFPU versions                                     | matmul version                           |
| -------------------------- | ------------------------------------------------- | ---------------------------------------- |
| stage 1 `y = x·q̂⁻¹ mod q` | on device (SFPU modmul)                           | host (input prep)                        |
| decompose to limbs         | n/a (SFPU) / on device                            | host (input prep)                        |
| combine                    | on device (SFPU)                                  | **on device (matrix engine)**            |
| ε                          | **device SK, fused** (`baseconv_fused_sk`)        | host                                     |
| recombine + `mod p_j`      | on device                                         | **on device (per-column RNS recombine)** |


`baseconv_fused_sk` is now the most self-contained: the whole exact conversion (stage-1, combine,
on-device ε via `m_sk`, and recombine) runs on device in **one dispatch**, taking the previous HE
kernel's per-limb residue tiles as input — the correct exact FBC for a pipeline (no host arithmetic,
no host-known plaintext). It is SFPU-only (520 µs, 4→4).

### I/O domain: Montgomery in / Montgomery out (concatenation-ready)

Both fused SFPU kernels are **Montgomery-in / Montgomery-out**, matching the BFV plan's end-to-end
convention ("all RNS limbs in Montgomery form"), so they drop into the pipeline **with no
`to_montgomery`/`from_montgomery` glue** at their boundaries — they concatenate directly with NTT/INTT,
pointwise modmul, and relin.

| kernel | input | output |
| --- | --- | --- |
| `baseconv_fbc` (approx) | `x̃_i = mont_{q_i}(x_i)` | `õut_j = mont_{p_j}(x mod p_j)` |
| `baseconv_fbc_sk` (exact) | all Montgomery: `x̃_i` **and** `x̃_sk = mont_{m_sk}(x mod m_sk)` | `õut_j = mont_{p_j}(x mod p_j)` |

Two caveats behind the "it's Montgomery" contract:

- **Internally the CRT combine is still plain-domain** (it must be — §2: per-modulus `R` scaling would
  corrupt the cross-modulus reinterpretation). `y_i` and `ε` stay plain/canonical inside the kernel;
  what makes the *external* contract Montgomery is that `R` is **folded into the baked constants**
  (`C1` plain, `C2`/`Ecorr`/`Dsk` = `R²`, `QinvM` plain). The constants absorb the domain difference
  for free, so the kernel is unchanged vs. a plain-I/O version — only the host-baked constants differ.
- **The exact (`_sk`) version needs the upstream kernel to also emit the sentinel residue** `x mod m_sk`
  (in Montgomery) — one extra limb the producer must carry. The approx version needs no `m_sk` channel.

`baseconv_matmul{,_packed}` move the combine onto the matrix engine (packed = 204 µs, the fastest
combine+recombine), but their stage-1/decompose are still host input-prep, and they are approximate
(no ε). Making a matmul version fully on-device **and** exact requires (a) an on-device stage-1
modmul + a 4→32 tile relayout (transpose + radix-decompose; the previous kernel emits 4 per-limb
tiles, the radix-matmul wants 32 coeff-major tiles with (modulus,radix) on the contraction axis) and
(b) an ε channel. **(a) is now done and the relayout turned out to be avoidable entirely — see the
box below and `docs/he/history/exp_tg018r.md`. (b) is still open**, so what exists on device today is the
approximate conversion on both routes. **The CRNS route (§2.5) is the natural fit for (b):** its ε is a fixed-point slice
of the matmul accumulator (no SK sentinel, no float), and its combine *is* the §3.2 int8 matmul with
different constants — so the matmul-regime exact FBC gets ε essentially for free once the combine
runs on device. Verified in the Python reference; device kernel pending.

> ### RESOLVED — pure-on-device FPU/matmul FBC now WINS.  `docs/he/history/exp_tg018r.md`
> **This box used to say "deferred, revisit at large N". Large N arrived and so did a third option
> the box did not consider, so the conclusion is reversed.**
>
> Measured at **N = 2^16 on 64 cores**, both routes fully on device, both verified against the same
> host golden, `(Ls, Ld)` swept over `{8,16,32}²` — **the matrix engine wins every cell**: 1.19× at
> `8→8`, **10.3× at `32→32`**, 1.98× at the `Ls=16, Ld=8` shape exp_Tg018 runs. The SFPU route costs
> `≈ Ls·(6.1 + 5.38·Ld)` µs; the matrix route is essentially **flat** at `≈ 265 + 8.5·Ls + 0.6·Ld`.
>
> **What the original argument missed.** It correctly showed that both ways of *reshaping the data*
> lose:
> - **Gather** (pull column `t` from 16 decomposed tiles per output tile): a strided element gather,
>   slow enough to erase the FPU win.
> - **Transpose** (`transpose_wh_tile` each limb tile): the combine then collapses to a
>   scalar-weighted sum of 16 tiles on the SFPU — not a matmul at all. Strictly worse.
>
> There is a third option: **do not reshape the data, transpose the OPERANDS.**
>
> ```
> reshape route:   plane [c][j] = Σ_lane A [c][lane] · B [lane][j]     ← needs A coeff-major
> operand route:   plane'[j][c] = Σ_lane Bt[j][lane] · At[lane][c]     ← needs At LIMB-major
> ```
>
> Both contract over the `(modulus, radix)` axis. The second wants its operand **limb-major**, which
> is what the HE kernels already emit — so the 1→32 reshape never happens. A tilized row is two
> contiguous 16-element runs, so building `Y'[i][c] = y_i[r·32 + c]` is **2 NoC copies of 64 bytes
> per (coefficient-group, limb)**, done by the matmul's own reader; the mirror image in its writer
> puts the result back limb-major. No transpose op, no extra pass, no scratch buffer, and the radix
> planes never leave L1. The whole thing costs ~10 µs of 277.
>
> The output modulus becomes a **row** index rather than a column index, so `recombine_rns` runs
> unchanged with its 5 parameter tiles built row-wise on the host.
>
> **The `Ld ≤ 8` cap was also not what it looked like** — it was `baseconv_fbc`'s stack arrays
> (`uint32_t C2[16][12]`), not the dispatch budget (`max_runtime_args_tensix = 4096`). Reading the
> args in place lifts it at no measurable cost.
>
> The n150 table in §5 below (`baseconv_matmul` 733 µs, packed 285 µs at 8→8) still stands for what
> it measured: n150, `L = 8`, one core, host-built input layout. Blackhole is ~3.3× faster across the
> board and exp_Tg018r supersedes it for any on-device comparison.

---



## 8. Pointers


| What                                          | Where                                                                                                                                                  |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| NumPy reference (approximate + exact ε)       | `reference/he/base_convert.py`                                                                                                                         |
| **CRNS exact FBC reference (§2.5) + verification** | `reference/he/bfv_bconv_crns.py` (run it: unit S→Q, u-sweep, full ct_mul cross-check vs `m_sk`)                                                    |
| CRNS source paper                             | Langowski–Devadas, [ePrint 2025/1068](https://eprint.iacr.org/2025/1068)                                              |
| BFV build plan (where FBC fits; ε discussion) | `docs/he/bfv_build_plan.md`                                                                                                                       |
| SFPU exact (host ε)                           | `tests/baseconv/baseconv.cpp`                                                                                                                         |
| SFPU exact (on-device SK ε)                   | `tests/baseconv/baseconv_sk.cpp`                                                                                                                      |
| SFPU fused approx (1 dispatch)                | `tests/baseconv/baseconv_fused.cpp`                                                                                                                   |
| SFPU fused **exact, on-device ε** (1 dispatch)| `tests/baseconv/baseconv_fused_sk.cpp`                                                                                                                |
| Matrix-engine (combine+recombine on device)   | `tests/baseconv/baseconv_matmul.cpp`                                                                                                                  |
| **Matrix-engine, block-packed (fastest approx)** | `tests/baseconv/baseconv_matmul_packed.cpp`                                                                                                        |
| **Matrix-engine EXACT via CRNS ε (fastest exact, 254 µs)** | `tests/baseconv/baseconv_matmul_crns.cpp` + `kernels/shared/compute/eps_correct_crns.cpp`, `kernels/shared/dataflow/reader_eps_correct.cpp`             |
| Kernels                                       | `kernels/shared/compute/{modmul_sfpi_full,modadd_reduce,modsub_reduce,baseconv_fbc,baseconv_fbc_sk,matmul_fbc,matmul_fbc_packed,recombine_rns}.cpp`    |
| Modular-multiply / matmul foundations         | `docs/vector/modular_multiplication.md`, `docs/matmul/modular_matmul_overview.md`                                                             |


