# Plan: size-32768 NTT as (32, 32, 32), multi-core, on-device

> **Status: Historical.** Superseded by [`../ntt_32768.md`](../ntt_32768.md).
> The plan for the 32×32×32 factorization, written before it was built.
> Numbers are a snapshot, measured on Wormhole n150.

**Goal.** A fully on-device NTT of length `N = 32^3 = 32768`, decomposed three-factor as
`(32, 32, 32)`. Every 32×32 modular matmul and every 32×32 twiddle (scalarmul) is one tile on one
Tensix core. Stages are coupled only by **one tile-granular all-to-all** (the "stage-convert"
corner turn). Reuses the verified NTT-1024 (`ntt1024_dev.cpp`) and int8-modular building blocks
unchanged.

`q = 167772161` (28-bit prime), same radix-128 / plane / Montgomery machinery as the 1024 case.

---

## 1. Why the data is 32 tiles

`N = 32768` values = a `32×32×32` cube. A 32×32 matmul tile holds 1024 values, so the cube is
**32 tiles** of 32×32. We index a global element as

```
x[n],   n = n1 + 32·n2 + 1024·n3,   n1, n2, n3 ∈ [0,32)
```

and lay it out so **tile `n1`** holds the 32×32 slab `Xin_{n1}[n3][n2] = x[n1 + 32·n2 + 1024·n3]`
(row = n3, col = n2). The host writes the input in this layout once (a strided tilize — free, it is
the one-time input write), so there is **no all-to-all to load** the cube.

---

## 2. Algorithm (recursive four-step → three passes)

Split `N = 32 · 1024` and apply the proven two-factor four-step, with the inner 1024 itself split
`32 · 32`. The two-factor identity (derived & verified in the 1024 work) is: for `N=P·Q`,
`n=a+P·b`, transform over `b` (outer) → twiddle `ω_N^{a·u}` → transform over `a` (inner) →
output `X[u + Q·v]`. Applying it twice gives three 32-point DFT passes over the three cube axes,
with two twiddle (scalarmul) stages and one corner turn:

```
        per tile n1 (= the inner NTT-1024, entirely on-tile)
   ┌─────────────────────────────────────────────────────────┐
   │ Pass 1 (DFT over n3):   B  = W @ Xin_{n1}                 │   contract rows  (n3)
   │ Twiddle a (scalarmul):  B' = B  ⊙ Ta,  Ta[s][n2]=ω_1024^{n2·s}
   │ Pass 2 (DFT over n2):   C  = B' @ W                       │   contract cols  (n2), NO corner turn
   └─────────────────────────────────────────────────────────┘
   Twiddle b (scalarmul):    C' = C ⊙ Tb_{n1},  Tb_{n1}[s][t]=ω_32768^{ n1·(s+32t) }
   ─────────────  ALL-TO-ALL corner turn (n1 ↔ s)  ─────────────
   Pass 3 (DFT over n1):     E_s = W @ D_s                     │   contract n1 (was the tile axis)
   Output:                   X[s + 32·t + 1024·v] = E_s[v][t]
```

- **Pass 1** is a left-multiply `W @ tile` (W = 32-pt NTT matrix, `W[a][b]=ω_32^{ab}`), contracting
  the row axis n3. **Pass 2** is a right-multiply `tile @ W`, contracting the column axis n2 — so,
  exactly as in NTT-1024, **no transpose / no all-to-all is needed between Pass 1 and Pass 2.**
  Passes 1+2 + twiddle a are *literally the existing NTT-1024 run on tile `n1`*.
- **Twiddle b** couples the inner result with the outer factor: `ω_N^{n1·u}`, `u = s+32t`. 32
  distinct constant tiles `Tb_{n1}` (precomputed offline).
- **The only cross-tile communication** is the all-to-all before Pass 3, because Pass 3 contracts
  `n1`, which until now is the *tile index* (the batch axis), not a within-tile axis.

Correctness is established the same way as before: generate every constant offline, never compute on
a constant, and check bit-exact against the direct `O(N²)` golden for several seeds.

---

## 3. The all-to-all (the "stage-convert" corner turn)

After twiddle b the data is `C'[n1][s][t]` — 32 tiles indexed by `n1`, each a 32×32 `[s][t]`.
Pass 3 needs `n1` as a within-tile **row** axis, grouped by `s`. So move

```
element (n1, s, t):   tile n1, row s, col t   ──▶   tile s, row n1, col t      (col t unchanged)
```

i.e. **swap the tile-index with the row-index** (axes 0↔1 of the `[n1][s][t]` cube). Each source
tile `n1` scatters its 32 rows to 32 destination tiles; each destination tile `s` gathers row `n1`
from every source. This is a full **tile-granular all-to-all** — pure data movement, no compute —
and is exactly the `ALL_TO_ALL_TILES` opcode (permutation + per-tile row regroup) from
`reference/ntt_matmul`. Implement as one dataflow kernel: read source tiles from DRAM via
`TensorAccessor`, write to permuted destination tiles in DRAM (DRAM→DRAM reshuffle, NOC-driven).

This is the single point that couples the 32 otherwise-independent tile pipelines.

---

## 4. Core / tile schedule

n150 usable grid ≥ 32 cores, so **one tile per core, 32 cores**, all stages multi-core (one program
per stage, 32 cores each — same pattern as `ntt1024_dev_mc.cpp`). Barriers only between stages.

| stage | what runs per core (one tile) | kernels (reuse) | # tiles |
|---|---|---|---|
| **Block 1** | full NTT-1024 on tile `n1`: A1 decompose → A2 matmul(W@) → A3 recombine → B twiddle a → C1 decompose → C2 matmul(@W) → C3 recombine | `ntt1024_dev.cpp` 7-stage pipeline, **unchanged** | 32 |
| **Twiddle b** | scalarmul `C ⊙ Tb_{n1}` | `modmul_compute.cpp` | 32 |
| **All-to-all** | corner turn n1↔s | new `all_to_all_tiles.cpp` (dataflow only) | 32 |
| **Block 2 (Pass 3)** | modular matmul `W @ D_s`: decompose → matmul(folded W, LHS) → recombine | `decompose_limbs` + `reader_matmul_foldlhs` + `recombine_compute`, **unchanged** | 32 |
| output | reindex `m = s+32t+1024v` on the single host read | host | — |

Matmuls stay int8-plane modular: decompose i32→4×i8 limbs, 16 int8 tile-matmuls → 4 i32 planes,
recombine `Σ_r 128^r C_r mod q` (one Montgomery reduction). **W is folded into the constant offline**
in every matmul (Pass 1/2 already do this in NTT-1024; Pass 3 uses `reader_matmul_foldlhs`), so there
is **no runtime fold modmul** anywhere.

Each tile pipeline runs on a single core (no need for the 4-core matmul split: 32 tiles × 4 cores =
128 > 64, and the matmul was shown not to be the bottleneck — the Montgomery reductions are).

---

## 5. New pieces vs. existing

Reused unchanged: NTT-1024 pipeline, `decompose_limbs`, `matmul_compute`/`reader_matmul_foldlhs`,
`recombine_compute`, `modmul_compute`, tilize/untilize, Montgomery context.

New, small:
1. **Host constants** — 32 twiddle-b tiles `Tb_{n1}[s][t]=ω_N^{n1(s+32t)}` (Montgomery form);
   strided input tilize into 32 cube tiles; output reindex permutation.
2. **`kernels/shared/dataflow/all_to_all_tiles.cpp`** — DRAM→DRAM corner turn (n1↔s), tile-granular,
   compute-free.
3. **`ntt32768_dev.cpp`** — host orchestrator: 32 cube buffers, runs Block 1 (7 multi-core stages) →
   twiddle b (multi-core) → all-to-all → Block 2 (3 multi-core stages), one read at the end. No
   host↔device traffic between stages.

---

## 6. Performance projection

From measured single-tile, single-core numbers: decompose ≈14 µs, int8 matmul ≈14 µs, recombine
≈39 µs, twiddle ≈36 µs; chained NTT-1024 ≈147 µs; modular matmul (decompose+matmul+recombine)
≈67 µs. With 32 tiles spread one-per-core, each stage is a 32-core program whose wall-clock is the
single-tile time inflated by DRAM-bandwidth contention (assume ≈1.5× as a planning factor; **to be
measured**).

| block | basis | projected wall |
|---|---|--:|
| Block 1 (NTT-1024 ×32, parallel) | 147 µs × ~1.5 | ~220 µs |
| Twiddle b | 36 µs × ~1.5 | ~55 µs |
| All-to-all | 32 tiles × 4 KB DRAM↔DRAM | ~30 µs |
| Block 2 (Pass 3 ×32, parallel) | 67 µs × ~1.5 | ~100 µs |
| **Total (N=32768)** | | **~400 µs** |

So **~32× the data for ~2.5–3× the wall-clock** of one NTT-1024 (≈147 µs), thanks to 32-way tile
parallelism — ≈ 2.5k NTT-32768/s. As before, the 3 Montgomery-reduction families (recombines +
twiddles) dominate; the matmuls are cheap. Levers (later): fuse decompose+matmul+recombine per
stage to cut dispatch; parallelize the reduction tiles further if cores remain.

---

## 6b. Core-count scaling experiment (1, 2, 4, … cores)

Run the *same* transform with the 32 tile-pipelines mapped onto a varying core count `C` and compare
wall-clock — this isolates how much the parallelism actually buys, and where DRAM contention caps it.

**Mechanism.** Keep the algorithm identical; only change how the 32 tiles are distributed. With `C`
cores, each stage processes its 32 tiles in `ceil(32/C)` **waves** (a core takes tiles
`c, c+C, c+2C, …`). Concretely, make `C` a runtime parameter of `ntt32768_dev` and, per stage, either
(a) launch the multi-core program over a `C`-core range and loop it `ceil(32/C)` times with a tile
offset, or (b) set per-core runtime args so each core walks its wave list. Same kernels, same
buffers, same correctness check — only the core grid and wave count differ. The all-to-all is run
once regardless (it is a global reshuffle), but it too can be split across `C` cores.

**Sweep:** `C = 1, 2, 4, 8, 16, 32` (and 32-with-2-cores-per-tile if grid allows, to confirm the
matmul isn't the limit). For each `C`: verify `0/32768`, record total + per-block wall-clock, compute
speedup vs `C=1` and parallel efficiency `S(C)/C`.

Measured (`ntt32768_dev.cpp`, n150; full results + analysis in `NTT_32768_PERF.md`), after the
multi-core + row-chunk all-to-all (`all_to_all_corner_mc.cpp`):

| cores C | waves = ⌈32/C⌉ | total µs | speedup vs C=1 | efficiency |
|--:|--:|--:|--:|--:|
| 1 | 32 | 7154 | 1.00× | 100% |
| 2 | 16 | 3610 | 1.98× | 99% |
| 4 | 8 | 1850 | 3.87× | 97% |
| 8 | 4 | 988 | 7.24× | 90% |
| 16 | 2 | 573 | 12.48× | 78% |
| 32 | 1 | 407 | 17.59× | 55% |

Near-linear to 8 cores. (Before optimizing the corner turn it flattened at C=32 to 1285 µs / 18%
efficiency, bottlenecked by a single-core per-element all-to-all; see `NTT_32768_PERF.md`.)

**What to expect.** `C=1` is ≈ 32 × the single-NTT-1024 work (≈ 32 × 150 µs ≈ 4.8 ms for Block 1
alone) — a serial baseline. Doubling cores should roughly halve wall-clock **while compute-bound**;
the curve flattens once **DRAM bandwidth saturates** (the reductions/twiddles read+write whole
tiles), so efficiency drops at high `C`. The knee of that curve is the useful result: it tells us the
best cores-per-NTT operating point and whether 32-way is worth it over, say, 8- or 16-way. Plot
speedup vs `C`; the gap from the ideal line is the contention + fixed per-stage dispatch (Amdahl).
Report the per-block breakdown at each `C` so we can see *which* block (Block 1 reductions vs Pass 3
vs all-to-all) stops scaling first.

## 6c. Oracle & the IR-vs-plan all-to-all difference

`reference/ntt_matmul/oracle_32768.py` is the **device-faithful NumPy golden**: it computes the
exact stages of this plan (cube layout, `W@`/twiddle-a/`@W` per tile, twiddle-b `ω_N`, the `n1↔s`
corner turn, Pass 3, reindex) in exact mod-q integer arithmetic and returns every intermediate
(`Xin, B, Ba, C, Cb, D, E`) — so each device DRAM buffer can be diffed against the matching stage.
Verified `0/32768` vs the direct `O(N²)` DFT (`ntt_matrix.ntt`), seeds 0/1/7/42.

Note a deliberate structural difference from the IR (`main.py`, `PLAN=(32,32,32)`): the **IR emits
2 `all_to_all`** (a uniform 3-pass form, a corner turn between every pass: 96 `matmul_modq`,
64 `mul_modq`, 2 `all_to_all`), whereas this plan folds passes 1+2 into an **on-tile NTT-1024**
(Pass 2 is a right-multiply `@W`, needing no corner turn) and so uses **1 `all_to_all`**. Both are
bit-exact to the same golden; the device follows the 1-all-to-all design (fewer cross-core
shuffles, maximal reuse of the verified NTT-1024). `oracle_32768.py` — not `ir.run` — is therefore
the per-stage authority for the device, because its intermediates match the device stage layout
1:1. The IR remains the whole-transform cross-check.

## 7. Verification

1. **NumPy oracle** first: cube layout + 3-pass + twiddles + corner turn + reindex vs. direct
   `Σ x ω_N^{nk}`, bit-exact mod q (object dtype). Confirms the index map and both twiddle tables.
2. On device: `mismatches == 0/32768` vs. the direct golden, seeds 0/1/7/42, run **serially**
   (single n150 — never two device binaries at once) with a timeout.
3. Then the per-stage timing report, as for NTT-1024, **and the core-count sweep of §6b**
   (`C = 1, 2, 4, 8, 16, 32`) with the speedup/efficiency table filled in.

---

## 8. Risks / watch-list

- **DRAM bandwidth** with 32 cores hammering DRAM each stage — the projection's 1.5× is a guess;
  measure and, if needed, stagger or keep more of each tile pipeline L1-resident.
- **All-to-all** correctness (the n1↔s swap, col fixed) is the one genuinely new index move — pin it
  in the NumPy oracle before device.
- **Usable core count** — confirm ≥32 compute cores on the specific n150 grid (harvesting);
  if fewer, run tiles in 2 waves.
- **L1 capacity** per core (CBs for a 7-stage pipeline) × 32 cores — should fit (single-tile CBs),
  verify at build.

---

## 9. Relation to the `ntt_matmul` compiler

This is exactly what the IR pipeline should emit for `PLAN=(32,32,32)`, `T=32`: three matmul passes,
two element-wise twiddles done **per-tile** (no concat/slice in compute), and the cross-stage
corner turn as an explicit `ALL_TO_ALL_TILES` op. The hand-built `ntt32768_dev.cpp` doubles as the
golden reference for extending the compiler from two-factor to three-factor.
