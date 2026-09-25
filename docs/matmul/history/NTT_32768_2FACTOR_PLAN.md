# Plan: naive two-factor (128, 256) NTT-32768 — the "just throw big matmuls at TT" baseline

> **Status: Historical.** Superseded by [`../ntt_32768.md`](../ntt_32768.md).
> The plan for the deliberately-unoptimized 128×256 baseline, written before it was built.
> Numbers are a snapshot, measured on Wormhole n150.

**Goal.** A deliberately *un*-optimized NTT-32768 to compare against the hand-tuned 3-factor
(`ntt32768_dev.cpp`). Decompose `N = 32768 = 128 × 256` and run the textbook two-factor four-step —
**matmul → scalarmul → matmul** — handing TT two *large-shape* int8 modular matmuls and one
large i32 modular scalarmul. **No NTT-level fold/tiling/corner-turn.** The only tiling is the
unavoidable mechanical one a normal matmul does internally (M/N/K into 32×32 tiles). Run on up to
32 cores and measure the same `C = 1,2,4,8,16,32` sweep, so we get a like-for-like comparison.

The question this answers: *on TT, where modular reduction and data movement — not matmul FLOPs —
dominate, how much does the NTT-level decomposition actually buy?*

---

## 1. Algorithm (two-factor four-step)

Reshape `x` to a `128×256` matrix `A[n1][n2] = x[n1 + 128·n2]` (n1∈[0,128), n2∈[0,256)). Then:

```
M1  S    = A @ W256          (128×256)@(256×256), contract n2 (K=256)   -> S[n1][u]
TW  S'   = S ⊙ Tb            twiddle, Tb[n1][u] = ω_N^{n1·u}            (128×256 elementwise)
M2  R    = W128 @ S'         (128×128)@(128×256), contract n1 (K=128)   -> R[v][u]
out[u + 256·v] = R[v][u]     (host reindex of the single read)
```

`W256[a][b]=ω_256^{ab}`, `W128[a][b]=ω_128^{ab}`, `ω_256 = ω_N^{128}`, `ω_128 = ω_N^{256}`,
`ω_N = ω_32768`. Both DFT matrices are **constants folded offline** (no runtime fold), exactly like
the 3-factor's A2/C2: M1 folds into the RHS (`W256`), M2 folds into the LHS (`W128`). Exact index
map / twiddle pinned by the numpy oracle + a direct-DFT spot check (the proven methodology).

No corner-turn: M1 contracts the column axis (right-multiply), M2 the row axis (left-multiply).

---

## 2. The crux: large-K modular matmul (recombine overflow)

Each modular matmul is the int8 plane scheme: operands → 4 radix-128 limbs, 16 int8 limb-matmuls,
accumulate int32 planes `C_r`, recombine `Σ_r 128^r C_r mod q`. With K>1 tile the matmul also
accumulates over K-tiles, so a plane is

```
C_r = Σ_{limb pairs} Σ_{k<K} (7-bit)·(7-bit)   ≤  K · 127²
```

| K (contraction) | max plane C_r | vs recombine assumption (C_r < 2²¹) |
|---|--:|---|
| 32 (3-factor) | 32·127² ≈ 2¹⁹ | OK |
| 128 (M2) | ≈ 2²¹ | borderline |
| **256 (M1)** | ≈ 2²² | **exceeds** |

- **int32 accumulation is fine** even at K=256 (`4.1M < 2³¹`), so a single big matmul accumulating
  over all K-tiles is correct.
- **The optimized recombine breaks**: it assumes `low = C0 + (C1«7) < 1.61q`. At C_r≈2²², `low ≈ 2²⁹
  ≈ 3.8q`, so its two fixed conditional subtracts under-reduce.

**Fix (cheap):** a `recombine_bigK` variant — same one Montgomery reduction, but replace the fixed
2 conditional subtracts with a short subtract loop (handles `low < ~4q`, `R < ~5q`). Still **one
recombine per output tile** — we do NOT block K into many partial reductions (that would add
reductions and destroy the comparison). This is the key correctness detail and is also the
hardware-aware point: "no tiling at all" is impossible (K must be bounded for int32/recombine),
but *NTT-level* restructuring is genuinely avoided.

---

## 3. Op-count comparison (predicted)

| quantity | 2-factor (128,256) | 3-factor (32,32,32) | ratio |
|---|--:|--:|--:|
| MACs = N·Σfactors | 12.58 M | 3.14 M | **4.0× more** |
| int8 tile-matmuls | 2048 (M1) + 4096 (M2)... see note | 1536 | 4× more |
| recombines (reductions) | 64 | 96 | 0.67× |
| twiddles | 32 | 64 | 0.5× |
| corner-turns | **0** | 1 | — |
| pipeline stages | **7** | 12 | fewer |

Note tiles: M1 = 32 out-tiles × (4 planes·... ) — M1 K=8 → 32·(16·8)=4096; M2 K=4 → 32·(16·4)=2048;
total 6144 int8 tile-matmuls (4× the 3-factor's 1536). Fewer of everything else, no corner-turn.

## 4. Predicted performance (C=32, from measured per-stage anchors)

| stage | est. µs | note |
|---|--:|---|
| decompose ×2 | ~30 | 2 dynamic operands |
| M1 matmul (K=8) | ~120–180 | 4× tiles, K-looped |
| M2 matmul (K=4) | ~60–100 | |
| twiddle ×1 | ~43 | |
| recombine ×2 (bigK) | ~96 | one per out-tile |
| corner-turn | 0 | — |
| **total @ C=32** | **~360–450 µs** | vs 3-factor **407 µs** |

**Prediction: roughly comparable (±15%).** The 4× matmul penalty is largely hidden because int8
matmul is cheap on TT; the 2-factor's fewer reductions + no corner-turn offset it. Swing factor =
how well the K-looped matmul pipelines (amortized dispatch). The sweep will show whether 2-factor
scales as well (it has fewer, coarser stages → possibly better efficiency at high C).

---

## 5. Core mapping & sweep

Both matmuls output 32 tiles (M1: 4×8; M2: 4×8); twiddle/decompose are 32-tile stages. So the exact
same multi-core sweep applies: stage on `C` cores, core `c` handles output tiles `{c, c+C, …}`
(`32/C` each), `C ∈ {1,2,4,8,16,32}`. Reuse the `(start,stride)` pattern and `_g` reader/writers.

7 stages: `decompose(A) → M1 → recombine1 → twiddle → decompose(S') → M2 → recombine2`, then reindex
on the single read.

---

## 6. New vs reused

**Reused unchanged:** `decompose_limbs`, `modmul_compute` (twiddle), the strided `_g`
reader/writers, Montgomery context, tilize/untilize.

**New:**
1. **K-looped plane-matmul** — `matmul_compute_mtk.cpp` (outer over assigned out-tiles; per out-tile:
   4 planes × {4 limbs × K-tiles} accumulations) + `reader_matmul_mtk.cpp` (feeds the folded constant
   block + dynamic-limb K-column, both modes) + a writer (4 planes per out-tile). Generalizes the
   existing `*_mt` kernels with a K-tile loop and M/N/K tile addressing.
2. **`recombine_bigK.cpp`** — recombine with a subtract loop for `low < ~4q` (large-K planes).
3. **Host** `ntt32768_2factor_dev.cpp` — constants (W256/W128 folded, Tb 32 twiddle tiles, A layout,
   reindex), 7-stage sweep, golden + per-stage self-check.
4. **Oracle** `oracle_32768_2factor.py` — 2-factor four-step in exact mod-q ints; verify vs direct
   DFT (spot check) and return per-stage goldens (`A, S, S', R`).

---

## 7. Verification

1. NumPy oracle: 2-factor four-step vs direct DFT (spot-check indices, like the 3-factor oracle).
2. Device: `0/32768` vs the 3-pass... vs the 2-factor golden, seeds 0/1/7/42, run **serially** with
   a timeout. Per-stage self-check (`S, S', R`) for bring-up.
3. The sweep table (C=1…32) with mism=0 at every C.

## 8. Comparison deliverable (`NTT_32768_COMPARISON.md`)

Side-by-side at matched C: wall-clock, per-stage breakdown, op counts, efficiency. Headline question
answered: **does NTT-level decomposition help on TT, given matmul is cheap and reduction/movement
dominate?** Expected takeaway: for N=32768 the naive 2-factor is competitive (the decomposition's
MAC savings don't pay off until larger N), but it does 4× the matmul work and needs the `bigK`
recombine; the 3-factor wins clearly only as N grows (matmul work `O(N√N)` vs `O(N·N^{1/3})`).

## 9. Risks / watch-list
- **bigK recombine bound**: verify `low < 4q` actually holds at K=256 (`C_r ≤ 2²²` ⇒ `low ≤ 2²⁹ ≈
  3.8q` ✓) and the subtract loop count is enough; unit-test it like `test_recombine_one`.
- **K-looped matmul DRAM traffic**: M1 reads 8 K-tiles per out-tile → bandwidth may bind; measure.
- **Core grid** for C=32 (harvesting), as before.
- Keep runs serial on the single n150.
