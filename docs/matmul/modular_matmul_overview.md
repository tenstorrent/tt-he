# Modular Matrix Multiplication on Tenstorrent — Project Overview

**Audience:** engineering management / reviewers with light project context.
**Scope:** the modular matrix product `C = (A · B) mod q`, lowered onto the Tensix **int8
matrix engine** for the multiply-accumulate and the **32-bit vector engine (SFPU)** for the
recombination and modular reduction.
**Status:** 32×32 single-core path implemented and validated bit-exact on real silicon
(n150 / Wormhole B0); recombine+reduce currently on host, on-device SFPU path scoped.
**Last updated:** 2026-06-26

---

## 1. What this is and why we need it

Where the element-wise modular multiply (`docs/vector/modular_multiplication.md`)
is the *twiddle* of an NTT, the **modular matrix multiply** `C = (A·B) mod q` is the other
half of the FHE workload: the **linear-algebra stages** — key-switching, the matrix form of
small NTTs, and any `Z_q`-linear map applied to encrypted data. These are dominated by
**integer matmul under a prime modulus**, exactly the operation here.

The challenge is that the fast matrix hardware on Tensix multiplies **8-bit** integers, while
our data is **28-bit values reduced modulo a 28-bit prime `q`**. This document explains how a
32-bit-wide modular matmul is built on top of an 8-bit matmul engine, and where the 32-bit
vector engine picks up the rest of the work.

Concrete parameters we target:

- **Modulus `q`** is a **28-bit prime** (e.g. `268435399 = 2²⁸ − 57`, or the NTT-friendly
  `167772161 = 5·2²⁵ + 1`). The 28-bit cap is what makes the int8 lowering fit exactly (§4).
- Operands `A, B ∈ Z_q^{32×32}`; the natural data unit is the hardware **32×32 tile**.

---

## 2. The decisive idea — split the op across two engines

A modular matmul has two distinct kinds of arithmetic, and Tensix has two distinct engines.
The whole design is the recognition that they line up:

| sub-problem | engine | why |
|---|---|---|
| `Σ_k A[i,k]·B[k,j]` — the **multiply-accumulate** | **int8 matrix engine** (FPU / `matmul_tiles`) | the only fast dense-MAC unit; 32×32 int8 → int32 per tile, hundreds of MACs/cycle |
| `(…) mod q` — the **recombine + reduction** | **32-bit vector engine** (SFPU) | element-wise, needs 32-bit integer add/shift and a Montgomery/Barrett reduction |

Neither engine can do the whole op: the matrix engine only multiplies 8-bit integers and has
no notion of "mod q"; the vector engine has the 32-bit width and the reduction logic but no
fast dense matmul. So we **lower the 28-bit MAC onto the int8 engine** by decomposing every
value into 7-bit limbs (§4), let the int8 engine accumulate exact partial products into a
32-bit integer, and then **hand the 32-bit partial products to the vector engine** to
recombine the limb weights and reduce mod `q`.

```
  A,B (28-bit, mod q)
        │  decompose to 7-bit limbs (radix 128)            ← host / vector engine
        ▼
   int8 32×32 tiles ──►  INT8 MATRIX ENGINE  ──►  int32 partial-product tiles
                         Σ over limbs & k          (exact, no mod yet)
        │                                                   matmul_tiles, K-accumulate
        ▼
   int32 tiles ──►  32-bit VECTOR ENGINE (SFPU)  ──►  C = (A·B) mod q
                    recombine 128^r  +  Montgomery/Barrett reduce
```

The rest of this doc is: how the lowering works (§3–§6), how it maps onto the kernels (§7),
and what each engine actually runs (§8).

---

## 3. The hardware reality that shapes the design

Two engines, two sets of constraints:

**The int8 matrix engine** (`matmul_tiles`):
- Operates on **32×32 tiles**, inputs **signed int8** (`DataFormat::Int8`), accumulating into
  a **32×32 int32** output (`fp32_dest_acc_en = true`).
- Tensix encodes int8 as **sign-magnitude**, but every limb we feed is **non-negative ≤ 127**,
  so sign-magnitude / two's-complement / unsigned all coincide — the signedness question
  disappears (this is *why* we use 7-bit, not 8-bit, limbs; §4).
- One matmul contracts **32 lanes** and produces **32 output columns** — but each logical
  value costs 4 lanes/columns (its limbs), so a tile only carries 8 logical `k` and 8
  logical `j` (§5).
- **Key property (verified):** `matmul_tiles(...,0,0,0)` called repeatedly inside one
  `tile_regs_acquire()` window **accumulates** int32 into the same dst tile for the Int8→Int32
  path — the standard float K-loop pattern works for ints. This lets the 4 contraction blocks
  sum in-register with no host adds.

**The 32-bit vector engine (SFPU)** — the same unit the vector modmul runs on:
- 32-lane SIMD over 32-bit integers; native add/shift/and/or but **no native integer
  multiply** (see the vector overview, §2). It is, however, exactly the right place for the
  `128^r` recombination (shifts + adds) and the `mod q` reduction (Montgomery).
- This is why the reduction is *not* on the matrix engine: "mod q" is element-wise 32-bit
  work, not dense MAC.

---

## 4. The algorithm — double radix-128 decomposition

To run a 28-bit MAC on an 8-bit multiplier we decompose **both** operands into 7-bit limbs
(radix `128`), exploiting that `128⁴ = 2²⁸`: any value `< 2²⁸` splits into **exactly 4 int8
limbs**, each in `[0,127]`.

**Decompose the dynamic operand `A`** into limbs (`p = 0..3`, online):
```
A[i,k] = Σ_p a[i,k,p] · 128^p ,        a[i,k,p] ∈ [0,128)
```

**Precompute the parameter `B`** offline, folding the `A`-side limb weight `128^p` *into* `B`
and reducing mod `q` so it stays < `2²⁸`, then decomposing *that* into limbs (`r = 0..3`):
```
Bp[k,p,j] = (128^p · B[k,j]) mod q             (< q < 2²⁸)
Bp[k,p,j] = Σ_r bp[k,p,j,r] · 128^r ,          bp[k,p,j,r] ∈ [0,128)
```

Then, working mod `q`:
```
C[i,j] = Σ_{k,p} a[i,k,p] · (128^p · B[k,j])
       = Σ_{k,p} a[i,k,p] · Bp[k,p,j]
       = Σ_r 128^r · ( Σ_{k,p} a[i,k,p] · bp[k,p,j,r] )      mod q
                      └──── signed int8 × int8 → int32 matmul ────┘
```

The crucial move: the `128^p` weights **never appear at runtime** — they are baked into the
parameter `bp` by the offline mod-`q` precompute. At runtime the matrix engine sees only
non-negative int8 limbs, and all that remains afterward is the `128^r` recombination (shifts
of 7 bits: 0/7/14/21) and **one** final `mod q`.

**Why radix 128 and not 256.** A byte (radix-256) decomposition produces limbs in `[0,255]`,
which overflows signed int8. Radix-128 keeps every limb non-negative ≤ 127 — valid int8 in
any encoding — and `128⁴ = 2²⁸` gives the clean 4-limb split. This is the same radix-128
trick the project uses elsewhere; it is *the* reason `q < 2²⁸`.

---

## 5. Dimensional mapping — how 28-bit values tile onto 32×32 int8

Each logical element costs **4 lanes** on the `A` side (its `p` limbs) and **4 columns** on
the `B` side (its `r` limbs). A 32×32 matmul therefore absorbs **8 contraction values `k`, 8
output values `j`, and all 32 rows `i`** at once. The logical problem is 32×32, so we tile
**K** and **J** into 4 groups of 8 (the `I` axis fits whole):

```
k = bk·8 + κ      bk ∈ {0..3},  κ  ∈ {0..7}     (contraction)
j = bj·8 + jσ     bj ∈ {0..3},  jσ ∈ {0..7}     (output column)
lane   m   = κ·4 + p      ∈ [0,32)              (p = limb of A)
outcol     = jσ·4 + r     ∈ [0,32)              (r = limb of B)
```

- **LHS limb-tile (A), one per `bk`:** `L_bk[i, κ·4+p] = a[i, bk·8+κ, p]` — the int8-limb view
  of `A[0:32, bk·8 : bk·8+8]` (32×8 logical → 32×32 int8; only the column axis expands ×4).
- **RHS limb-tile (B), one per `(bk,bj)`:** `R_{bk,bj}[κ·4+p, jσ·4+r] = bp[bk·8+κ, p, bj·8+jσ, r]`
  — from the offline precompute on the 8×8 block `B[bk·8:+8, bj·8:+8]` (8×8 → 32×32 int8; both
  axes expand ×4).

**Matmul with K-accumulation over `bk`** (per output column-group `bj`):
```
P_bj[i, jσ·4+r] = Σ_{bk=0..3} ( L_bk @ R_{bk,bj} )[i, jσ·4+r]
```
That is **16 int8 matmuls** (4 `bj` × 4 `bk`) accumulating into **4 int32 output tiles**
`P_0..P_3` — all on the matrix engine, the 4 `bk` summed in-register thanks to the §3
accumulation property.

---

## 6. Overflow analysis — why the int32 accumulator is safe

(Asserted in the numpy oracle and the host reference.)

- `A, B < q < 2²⁸` → all limbs ∈ `[0,127]` (valid int8). ✓
- `Bp = 128^p·B mod q < 2²⁸ = 128⁴` → an **exact** 4-limb decomposition (no truncation). ✓
- One `P_bj` entry sums over `(bk, κ, p)` = **128 terms**, each ≤ `127·127 = 16129`.
  Max ≈ `128 · 16129 ≈ 2.06×10⁶ < 2²¹` — far below the `2³¹` int32 limit. **Accumulator safe,
  with no intermediate `mod q`.** ✓
- Host/SFPU recombine: `Σ_r 128^r·P ≤ (1+128+128²+128³)·2²¹ ≈ 2⁴²` → fits **uint64** comfortably
  before the final `mod q`. ✓

Because there is **no `mod q` inside the accumulation**, the recombined integer is exactly
congruent to the true `A·B` — the modular reduction happens once, at the very end.

---

## 7. Implementation — how it maps onto tt-metal

Single Tensix core, three kernels on separate RISC-Vs (read / compute / write), mirroring the
standard matmul layout. Host: `tests/matmul/modular_matmul.cpp`; design rationale: `docs/matmul/history/PLAN.md`.

**Host data preparation** (`tests/matmul/modular_matmul.cpp`):
- *Offline (B → 16 RHS tiles):* per 8×8 block compute `Bp[κ,p,jσ] = (128^p·B[κ,jσ]) mod q`,
  decompose along `r`, reshape `[8,4,8,4] → 32×32` int8, **tilize**.
- *Online (A → 4 LHS tiles):* `a[:, bk·8:+8, :].reshape(32,32)` int8, tilize.
- Buffers: `a_dram` (4 int8 tiles), `b_dram` (16 int8 tiles), `out_dram` (4 int32 tiles);
  CBs `c_0`/`c_1` Int8 and `c_16` Int32, all 2-deep. Note the int8 tile is **1024 B** vs the
  int32 output tile **4096 B**.

**Compute kernel** (`kernels/shared/compute/matmul_compute.cpp`) — the matrix-engine core:
```cpp
mm_init(c_0, c_1, c_16);
for (bj = 0; bj < 4; ++bj) {
    tile_regs_acquire();
    for (bk = 0; bk < 4; ++bk) {
        cb_wait_front(c_0,1); cb_wait_front(c_1,1);
        matmul_tiles(c_0, c_1, 0,0,0);   // accumulate 4 bk into dst tile 0
        cb_pop_front(c_0,1); cb_pop_front(c_1,1);
    }
    tile_regs_commit(); tile_regs_wait();
    cb_reserve_back(c_16,1); pack_tile(0, c_16); cb_push_back(c_16,1);
    tile_regs_release();
}
```
This emits the 4 int32 product tiles `P_0..P_3`. **Config:** `MathFidelity::HiFi4`,
`fp32_dest_acc_en = true` (required for the 32-bit Int32 accumulator), `math_approx_mode = false`.

**Recombine + reduce** — the 32-bit vector part (§8). In v1 this runs on the **host** for
simplicity and exactness:
```cpp
for (i, bj, jσ):
    acc = Σ_{r=0..3} (uint64) P_bj[i][jσ*4+r] << (7*r)   // 128^r == 1<<(7r)
    C[i][bj*8+jσ] = acc % q
```

---

## 8. The 32-bit vector-engine half — recombine and reduce

The matrix engine stops at the int32 partial products `P_bj`. Turning those into `C mod q`
is pure **32-bit element-wise** work — exactly what the SFPU vector engine does:

1. **Recombine the `r` limbs:** `acc = Σ_r 128^r · P_bj[i, jσ·4+r]`. The weights `128^r` are
   shifts of 7 bits (`<<0`, `<<7`, `<<14`, `<<21`); `acc` is up to ~`2⁴²` (§6), so this needs
   a 64-bit (or two-limb 32-bit) intermediate.
2. **Reduce mod q:** `acc mod q`. With a 28-bit prime this is **the same Montgomery reduction
   the vector modmul already runs** (`docs/vector/modular_multiplication.md` §3) — the
   `≤ 2⁴²` value is reduced into `[0,q)` division-free, reusing the 16-bit-limb hand-built
   high-half assembly and the lazy-reduction trick.

**Why it is on the host in v1, and the on-device plan.** The recombination is trivial, but the
reduction leans on the SFPU integer-multiply path that the vector-modmul project had to build
from scratch (no native int multiply, §3). v1 keeps it on the host to land a correct,
verified matmul; the on-device version (PLAN milestone **M5**) drops in the existing
Montgomery `mod q` kernel after a `128^r` fold. Because that machinery is already bit-exact
and timed (see the vector overview), the on-device port is **integration, not new algorithm
work** — and it is what removes the host round-trip when this matmul is chained inside an NTT.

This is the cleanest illustration of the two-engine split: the **8-bit matrix engine** does
the dense MAC it is best at, and the **32-bit vector engine** does the reduction it is best at;
the int32 tile is the handoff between them.

---

## 9. Validation & status

- **Algorithm + exact tile/lane layout** validated in numpy:
  `reference/matmul/tt_int8_modular_matmul_32x32.py` (TT-faithful, signed int8 / radix 128) —
  all cases PASS, including the largest 28-bit primes (`268435399`, `167772161`).
  `reference/matmul/int8_modular_matmul.py` is the more general radix-256 reference.
- **On device:** `tests/matmul/modular_matmul.cpp` verifies the full 32×32 result element-by-element
  against an exact `__int128` reference, **mismatches = 0 / 1024** across multiple 28-bit primes
  and seeds.
- The int32 K-accumulation property of `matmul_tiles` for the Int8→Int32 path is confirmed on
  silicon (PLAN milestone M1), so the 4 `bk` partials sum in-register — no host-side `bk` adds.
- **Fully on-device, fused path** (`tests/matmul/fused_modmatmul.cpp`): decompose + matmul +
  recombine in **one program**, all reduction on the SFPU (M5 done), bit-exact `0/1024` across
  seeds for `q=167772161`. This is the version measured in §10.

---

## 10. Performance

Measured on n150, single Tensix core, `q = 167772161`, 100-iteration steady state. Two
optimizations were applied on top of the baseline (all-building-block, three separate
programs): porting the recombine's Montgomery reduction to **SFPI** register passes, and
**fusing** decompose+matmul+recombine into one program with the limb (int8) and plane (int32)
intermediates kept **L1-resident** (no DRAM round-trip, one dispatch instead of three).

| version | decompose | matmul | recombine | **total** | speedup |
|---|---:|---:|---:|---:|---:|
| building-block, 3 programs | 13.2 | 14.4 | 38.8 | **66.4 µs** | 1.0× |
| + SFPI recombine, 3 programs | 14.0 | 14.5 | 23.0 | **51.5 µs** | 1.29× |
| **+ fused, 1 program** | *(fused — see partition below)* | | | **29.8 µs** | **2.23×** |

**Throughput:** 32³ = 32,768 modular MACs in 29.8 µs ≈ **1.1 G modular-MAC/s** (the *full* op,
including the mod-q reduction; the int8 MAC alone runs ~3.8 G/s — the rest is decompose +
reduction).

### Measured latency decomposition (fused, 1 tile, 1 core)

Profiler-clean run (`TT_METAL_DEVICE_PROFILER=1`, device clock ≈0.985 GHz so cycles ≈ ns).
Read the **MATH core (TRISC_1)** — the bottleneck RISC; UNPACK/PACK overlap with it.

| phase (TRISC_1 / MATH) | cycles | share | what runs |
|---|---:|---:|---|
| `ph_decompose` | 1,932 | 7% | int32 → 4 int8 limbs (1 SFPI pass) |
| `ph_matmul` | 4,448 | 16% | 16 int8 matmuls → 4 int32 planes |
| **`ph_recombine`** | **20,784** | **76%** | 128^r fold + Montgomery `mod q` |
| `fused_total` | 27,315 | 100% | (~27.3 µs device; ~29.8 µs wall incl. dispatch) |

Cross-RISC note: the **matmul** phase is *operand-delivery bound*, not MAC bound — its UNPACK
cost (7,360 cyc) exceeds its MATH cost (4,448), because the radix-128 lowering turns one
logical matmul into 16 int8 tile-matmuls (operand feeding dominates the FPU work). The
**recombine** phase is the opposite — MATH-bound (20,784 cyc) on the Montgomery reduction.

**The headline finding:** the modular **reduction dominates the op (~76%)**, while the int8
**matmul is the cheapest phase (~16%)**. The 8-bit matrix engine does the multiply-accumulate
almost for free; the cost is the 32-bit vector-engine `mod q` — the system-level echo of the
vector-modmul result (no native integer multiply ⇒ the reduction is the bottleneck).

### Where the fusion win came from

- **SFPI recombine (38.8 → 23.0 µs):** the reduction collapsed from ~50 building-block tile-op
  passes (each a DST round-trip) to 6 register passes + 9 `mul_int_tile`. Compute-bound win, ~1.7×.
- **Fusion (51.5 → 29.8 µs):** eliminated 2 of 3 program dispatches and both intermediate DRAM
  round-trips (limbs 4 KB×2 + planes 16 KB×2). Notably this is what removed decompose's cost:
  decompose's compute is trivial, so collapsing its passes in isolation did nothing — its ~13 µs
  was *all* dispatch + DRAM overhead, which only fusion deletes.

### Caveats / next levers

- **Prime range:** the recombine's lazy reduction is tuned for a 28-bit prime near 2²⁸
  (`low < 1.61q`). 26/27-bit primes need the dual-Montgomery recombine (reduce the low half too:
  `low0 = mulmod(low, mont(1))`), ~+7 µs, making it prime-agnostic and constant-time.
- **Remaining matmul headroom:** the matmul phase is operand-feed bound (§7) — caching the LHS
  limb tiles / batching reads attacks the UNPACK stall.

---

## 11. Where this fits in the bigger project

This modular matmul is the **matrix-engine counterpart** to the vector modmul, sharing the
same arithmetic foundation (`q < 2²⁸`, radix-128 limbs, Montgomery reduction). Together they
back:

- the on-device **NTT** (1024 and 32768): butterfly/twiddle stages use the vector modmul,
  while the matrix-form sub-transforms and key-switch-like maps use this modular matmul;
- general `Z_q`-linear maps over encrypted data in the FHE pipeline.

So the choices here — int8 lowering via radix-128, int32 in-register accumulation, and
deferring the reduction to the 32-bit vector engine — are the matrix-side half of the same
foundation documented for the vector modmul.

---

## 12. Pointers

| What | Where |
|---|---|
| TT-faithful oracle (signed int8, radix 128) | `reference/matmul/tt_int8_modular_matmul_32x32.py` |
| General reference (radix 256) | `reference/matmul/int8_modular_matmul.py` |
| Host — matmul-only (host decompose/recombine) | `tests/matmul/modular_matmul.cpp` |
| **Host — fully fused (measured in §10)** | `tests/matmul/fused_modmatmul.cpp` |
| Fused compute kernel (decompose+matmul+recombine) | `kernels/shared/compute/fused_matmul_a.cpp` |
| Matmul-only compute kernel (int8, K-accumulate) | `kernels/shared/compute/matmul_compute.cpp` |
| SFPI recombine kernel | `kernels/shared/compute/recombine_compute.cpp` |
| Fused reader / int32 writer | `kernels/shared/dataflow/reader_fused.cpp`, `writer_i32.cpp` |
| Design + tile-mapping rationale | `docs/matmul/history/PLAN.md` |
| Vector modmul (shared reduction machinery) | `docs/vector/modular_multiplication.md` |
