# Plan: 32×32 modular matmul (28-bit) on one Tensix core via the int8 matmul API

> **Status: Historical.** Superseded by [`../modular_matmul_overview.md`](../modular_matmul_overview.md).
> The original 32×32 modular matmul plan; its status section reads "DONE & VERIFIED ON DEVICE".
> Measured numbers, if any, are a snapshot from Wormhole n150.

## 0. Status — DONE & VERIFIED ON DEVICE

- **Algorithm + exact tile layout validated** in numpy:
  `reference/matmul/tt_int8_modular_matmul_32x32.py` — all 7 cases PASS.
- **Device kernel implemented and passing:** `scratch/modular_matmul_28b/`
  (host + reader/writer/compute kernels + CMake). Verified `mismatches=0/1024` across
  6 primes × 3 seeds, including the largest 28-bit primes (268435399, 167772161).
  Run: `./build-local/modular_matmul_28b [q] [seed]`.
- **M1 resolved:** `matmul_tiles(...,0,0,0)` **does** accumulate int32 across repeated calls
  within one `tile_regs_acquire()` for the Int8→Int32 path (the 4 `bk` matmuls sum correctly
  into one dst). The §7 fallback is therefore not needed.
- Milestones M1–M3 complete. Remaining: M4 (perf cleanup — cache L tiles), M5 (optional
  on-device recombine+mod).

## 1. Goal

Compute, on a **single Tensix core**, the modular matrix product

```
C = (A @ B) mod q        A, B ∈ Z_q^{32×32},   q prime, q < 2^28
```

using only the hardware **signed int8 tile matmul** (`matmul_tiles`), which is restricted to
`32×32` tiles of int8 operands accumulating into a `32×32` int32 output.

## 2. Data-type strategy: radix 128, not radix 256

TT-metal's tile matmul takes **signed int8** (Tensix encodes int8 as sign-magnitude). A naive
radix-256 byte decomposition yields limbs in `[0,255]`, which does **not** fit int8.

**Fix:** decompose in **radix 128 (7-bit limbs)**. Every limb is then in `[0,127]` — a valid
**non-negative** int8 in *any* encoding (sign-magnitude, two's complement, unsigned all agree
for non-negative values ≤ 127), so the signedness mismatch disappears completely. And
`128⁴ = 2²⁸`, so any value `< 2²⁸` splits into **exactly 4 int8 limbs**. Hence the constraint
`q < 2²⁸` — this is the "small modulus so it splits into 4 int8" requirement.

The math (p, r ∈ {0,1,2,3} are limb indices):
- Decompose dynamic A:    `A[i,k] = Σ_p a[i,k,p]·128^p`,  `a ∈ [0,128)`
- Precompute parameter B:  `Bp[k,p,j] = (128^p · B[k,j]) mod q`   (< q < 2²⁸)
- Byte-decompose Bp:       `Bp[k,p,j] = Σ_r bp[k,p,j,r]·128^r`,  `bp ∈ [0,128)`

```
C[i,j] = Σ_r 128^r · ( Σ_{k,p} a[i,k,p] · bp[k,p,j,r] )   mod q
                       └────── signed-int8 × int8 → int32 matmul ──────┘
```
The `128^p` weights are baked into `bp` by the offline mod-q precompute. Only the `128^r`
recombination (shifts of 7 bits: 0/7/14/21) and one final `mod q` remain.

## 3. Hardware constraints

| Constraint | Value | Consequence |
|---|---|---|
| Tile shape | 32×32 | all operands/outputs are 32×32 tiles |
| Input dtype | `DataFormat::Int8` | limbs must be non-negative ≤ 127 → **radix 128** (§2) |
| Output / accumulator | `DataFormat::Int32`, `fp32_dest_acc_en = true` | 32-bit integer accumulator |
| Contraction lanes / matmul | 32 | only 32 `(k,p)` MAC terms per call |
| Output columns / matmul | 32 | only 8 logical `j` fit (each needs 4 limb-columns) |
| SFPU integer math | weak (Wormhole: no native int mul) | **recombine + `mod q` on host** (v1) |

Reference for the working path: a local `matmul_int8` scratch project (Int8→Int32 matmul); the
validated equivalent now lives in `tests/matmul/` + `kernels/shared/compute/`.

## 4. Dimensional mapping (the core of this design)

Each logical element costs **4 lanes** (the `p` limbs) on the A side and **4 columns**
(the `r` limbs) on the B side. So one 32×32 matmul absorbs **8 contraction values `k`,
8 output values `j`, and all 32 rows `i`**. The logical problem is 32×32, so we tile only
**K** and **J** into 4 groups of 8 (the I axis fits whole):

```
k = bk·8 + κ      bk ∈ {0..3},  κ  ∈ {0..7}     (contraction)
j = bj·8 + jσ     bj ∈ {0..3},  jσ ∈ {0..7}     (output column)
lane   m   = κ·4 + p      ∈ [0,32)              (p = limb of A)
outcol     = jσ·4 + r     ∈ [0,32)              (r = limb of B)
```

**LHS limb-tile (A), one per `bk`:**
`L_bk[i, κ·4+p] = a[i, bk·8+κ, p]` — i.e. the int8-limb view of `A[0:32, bk·8:bk·8+8]`
(32×8 → 32×32 int8; only the column axis expands ×4, since A's 32 rows already fill the tile).

**RHS limb-tile (B), one per `(bk,bj)`:**
`R_{bk,bj}[κ·4+p, jσ·4+r] = bp[bk·8+κ, p, bj·8+jσ, r]` — from the offline mod-q precompute on
the 8×8 sub-block `B[bk·8:bk·8+8, bj·8:bj·8+8]` (8×8 → 32×32 int8; both axes expand ×4).

> This is the "decompose to 8×8 32-bit, bitcast to 32×32 int8" framing: exact for **B**
> (8×8 → 32×32). For **A** the natural unit is 32×8 → 32×32.

**Matmul with K-accumulation over `bk`** (per output column-group `bj`):
```
P_bj[i, jσ·4+r] = Σ_{bk=0..3} ( L_bk @ R_{bk,bj} )[i, jσ·4+r]
```
**16 matmuls** (4 bj × 4 bk) → **4 output tiles** `P_0..P_3`.

**Recombine (host):** `C[i, bj·8+jσ] = ( Σ_{r=0..3} 128^r · P_bj[i, jσ·4+r] ) mod q`.

## 5. Overflow analysis (verified in the oracle by assertion)

- A, B < q < 2²⁸ → limbs ∈ [0,127] (valid int8); top limb (p=3 / r=3) ∈ [0,8).
- `Bp = 128^p·B mod q < 2²⁸ = 128⁴` → exact 4-limb decomposition. ✓
- One `P_bj` entry = Σ over `(bk,κ,p)` = **128 terms**, each ≤ 127·127 = 16129.
  Max ≈ 128 · 16129 ≈ 2.06×10⁶ < **2²¹** ≪ 2³¹. **int32 accumulator safe.** ✓
- Host recombine: `Σ_r 128^r·P ≤ (1+128+128²+128³)·2²¹ ≈ 2⁴²` → fits **uint64**; then `mod q`. ✓

No intermediate `mod q` inside accumulation — the recombined integer is congruent to true `A@B`.

## 6. Host-side data preparation

- **Offline (B → 16 RHS tiles):** adapt `precompute_B` from the oracle: per 8×8 block compute
  `Bp[κ,p,jσ] = (128^p · B[κ,jσ]) mod q`, decompose along `r`, reshape `[8,4,8,4] → 32×32` int8.
- **Online (A → 4 LHS tiles):** `decompose_A`: `a[:, bk·8:bk·8+8, :].reshape(32,32)` int8.
- **Tilization:** convert each 32×32 row-major int8 array to TT tile (face) layout before
  `EnqueueWriteMeshBuffer`; untilize the int32 output tiles on readback. (Confirm the tilizer
  handles `sizeof(int8)` page sizes — mirror `scratch/matmul_int8`.)

## 7. Device kernel design (single core)

Mirror `scratch/matmul_int8`. Three kernels on core `{0,0}`.

**Circular buffers:** `c_0` Int8 (LHS), `c_1` Int8 (RHS), `c_16` Int32 (output), each 2-deep.

**Reader (`RISCV_1`):** stream the 16 `(bj,bk)` operand pairs, `bj`-major:
`for bj: for bk: push L_bk→c_0, push R_{bk,bj}→c_1` (L tiles re-fed per bj in v1; optimize in M4).

**Compute** (`ComputeConfig{ HiFi4, fp32_dest_acc_en=true, math_approx_mode=false }`):
```
mm_init(c_0, c_1, c_16);
for bj in 0..3:
    tile_regs_acquire();
    for bk in 0..3:
        cb_wait_front(c_0,1); cb_wait_front(c_1,1);
        matmul_tiles(c_0, c_1, 0, 0, 0);     // accumulate into dst tile 0
        cb_pop_front(c_0,1); cb_pop_front(c_1,1);
    tile_regs_commit(); tile_regs_wait();
    cb_reserve_back(c_16,1); pack_tile(0, c_16); cb_push_back(c_16,1);
    tile_regs_release();
```

**Writer (`RISCV_0`):** write the 4 `P_bj` int32 tiles to DRAM.

### ⚠ Verify first (milestone M1)
Does `matmul_tiles(...,0,0,0)` called repeatedly within one `tile_regs_acquire()`
**accumulate** into the dst for the Int8→Int32 path (as float K-loops do)?
- **Yes** → plan above stands.
- **No** → fallback: emit all 16 `P_{bk,bj}` separately and sum the 4 `bk` partials on the host
  during recombination (host already recombines+mods, so this is cheap; 4× output bandwidth).

## 8. Post-processing (host, v1)

After untilizing the 4 int32 tiles into `P[bj][32][32]`:
```
for i, bj, jσ:
    acc = Σ_{r=0..3} (uint64)P[bj][i][jσ*4+r] << (7*r)   // 128^r == 1<<(7r)
    C[i][bj*8+jσ] = acc % q
```
**On-device alternative (M5, optional):** SFPU `128^r` fold + Barrett/Montgomery `mod q`.
Deferred — Wormhole SFPU int-multiply weakness makes the 42→28-bit reduction fiddly.

## 9. Build & test

Layout mirrors `scratch/matmul_int8/`:
```
scratch/modular_matmul_28b/
  CMakeLists.txt              # find_package(TT-Metalium); OVERRIDE_KERNEL_PREFIX -> kernels/
  modular_matmul_28b.cpp      # host: prep, CBs, kernels, recombine+mod, verify
  kernels/shared/dataflow/{reader,writer}.cpp
  kernels/shared/compute/mm.cpp
```
Build: `cmake -B build-local -G Ninja && ninja -C build-local`, run on device.
**Oracle:** `reference/matmul/tt_int8_modular_matmul_32x32.py` (already passing) — dump A, B, q,
and device C; compare to its output and to the big-int `reference_modular_matmul`.

## 10. Milestones

1. **M1 — accumulation probe.** Confirm `matmul_tiles` accumulates int32 across repeated calls.
   Decides §7 path vs fallback.
2. **M2 — single column-group.** One `bj` end-to-end (4 bk matmuls → 1 tile → host recombine+mod
   for 8 columns); validate vs oracle on all 32 rows.
3. **M3 — full 32×32.** Loop all 4 `bj`, assemble C, validate vs oracle across all test primes.
4. **M4 — cleanup.** Cache the 4 L tiles (4-deep `c_0`) to avoid re-reads; tidy tilize/untilize.
5. **M5 (optional) — on-device recombine+mod** (SFPU + Barrett).

## 11. Resolved / open

- **Resolved:** radix 128 + signed int8 (was uint8/radix 256); `q` a prime `< 2²⁸`; recombine+mod
  on host for v1.
- **Open:**
  1. Which 28-bit prime to ship as the default `q`? (oracle tests 268435399 and 167772161.)
  2. A's range — strictly `[0,q)` (oracle assumes this) or any `< 2²⁸`? Either works; affects only
     whether A needs a pre-reduction.
  3. v1 scope: straight to full 32×32 (M3) or land M2 first as a correctness anchor?
