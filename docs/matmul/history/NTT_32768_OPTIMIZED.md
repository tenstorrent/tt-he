# NTT-32768 — known-optimization pass and updated 2-factor vs 3-factor comparison

> **Status: Historical.** Consolidated into [`../ntt_32768.md`](../ntt_32768.md) (§1) and [`../ntt_32768_perf.md`](../ntt_32768_perf.md) (§2–8).
> The optimization pass that reversed the 2-factor vs 3-factor conclusion. Carried forward in full.
> Numbers are a snapshot, measured on Wormhole n150.

**Programs:** `tests/ntt/ntt32768_dev.cpp` (3-factor 32×32×32), `tests/ntt/ntt32768_2factor_dev.cpp`
(2-factor 128×256). Single Tensix core scaled over `C = 1…32` cores, Wormhole n150,
`q = 167772161`, host wall-clock (20 iters, one `Finish`). Both verified `0/32768` vs the direct
DFT at **every** C in the sweep.
**Updates** `NTT_32768_COMPARISON.md`, whose numbers predate the optimizations below.

## 1. How the 32768-point NTT is computed

The NTT is `X[k] = Σ_n x[n]·ω^{nk} mod q` (`ω` a primitive 32768-th root of unity mod `q`). Done
directly it is a 32768×32768 mod-`q` matmul — far too much work. Both variants instead use a
**multi-factor four-step** decomposition that turns it into a few **32×32-tile** modular matmuls
plus element-wise twiddles. They reduce to the *same two primitives* — the int8 radix-128 modular
matmul (`modular_matmul_overview.md`: decompose → 16 int8 limb-matmuls → recombine `mod q`) and the
Montgomery twiddle modmul (`modular_multiplication_overview.md`) — composed differently. Both also
**fold the DFT matrices into constants offline**, so there is no runtime fold multiply.

### 3-factor (32×32×32) — `ntt32768_dev.cpp`

Treat `x` as a **32×32×32 cube**, index `n = n1 + 32·n2 + 1024·n3`, laid out as **32 tiles of 32×32**
(tile `n1` holds `Xin[n1][n3][n2]`). Transform along the three axes, with twiddles between and one
data reshuffle:

```
Block 1 (per tile, independent over n1) = the verified on-tile NTT-1024 over (n2,n3):
   A: G  = W @ Xin      (decompose → int8 matmul(W) → recombine)
   B: G' = G ⊙ Ta       (twiddle a, ω_1024)
   C: H  = G' @ W       (decompose → matmul(W) → recombine)
TB:  Cb = H ⊙ Tb         (twiddle b, ω_N — couples the third axis)
A2A: D[s][n1][t] = Cb[n1][s][t]   (all-to-all "corner turn": bring the tile axis n1 in-tile)
Block 2 (Pass 3, contract n1):
   D: E = W @ D          (decompose → matmul(W) → recombine)
out[s + 32·t + 1024·v] = E[s][v][t]   (reindex on the single output read)
```

12 stages: 3 recombines, 2 twiddles, **1 corner turn**. The corner turn is needed because the `n1`
axis is the *tile index* and the matrix engine only contracts *within* a tile — so before Block 2
can contract `n1`, the all-to-all moves it to an in-tile axis (every tile feeds every tile).

### 2-factor (128×256) — `ntt32768_2factor_dev.cpp`

Treat `x` as a **128×256 matrix**, `A[n1][n2] = x[n1 + 128·n2]`. Textbook four-step:

```
M1:  S  = A @ W256       contract n2 (K=256)   — 256-point column DFTs
TW:  S' = S ⊙ Tb         twiddle, Tb[n1][u] = ω_N^{n1·u}
M2:  R  = W128 @ S'       contract n1 (K=128)   — 128-point row DFTs
out[u + 256·v] = R[v][u]  (reindex on the single read)
```

7 stages: 2 recombines, 1 twiddle, **0 corner turns** (M1 right-multiplies to contract columns, M2
left-multiplies to contract rows — neither axis needs to cross tiles). The price is **large-K**
matmuls (`K=256`/`128`), which the matmul handles by a K-tile accumulation loop and which push the
int32 planes large enough to need the `recombine_bigK` reduction (§2).

### Why two variants

The 3-factor is the classic FFT decomposition (fewer MACs, `O(N·N^{1/3})`) but pays an extra
reduction, an extra twiddle, and the corner turn. The 2-factor is the "just hand TT two big
matmuls" baseline (4× the MACs, `O(N·√N)`) but with no corner turn and fewer reductions. The rest
of this doc measures both after applying the known kernel optimizations, and compares them.

## 2. What was optimized

The same three kernel-level wins established on NTT-1024, applied here:

| optimization | what it does | applied to |
|---|---|---|
| **SFPI recombine** | Montgomery `mod q` reduction in fused register passes (no DST round-trips) instead of ~50 building-block tile-ops | 3-factor (`recombine_compute.cpp`) |
| **SFPI bigK recombine** | same, large-K variant (5 conditional subtracts for `low < ~4q`) | 2-factor (`recombine_bigK.cpp`) |
| **hybrid SFPI twiddle** | drop-in `modmul_compute.cpp → modmul_sfpi_hybrid.cpp` (~75 → ~16 passes) | both (twiddle stages) |

Not yet applied: **fusing** decompose+matmul+recombine into one program (done for NTT-1024; both
32768 variants still run those as separate chained stages), and multi-coring the 3-factor's
all-to-all input staging.

## 3. Before → after (C=32, µs / NTT-32768)

| variant | baseline | optimized | speedup | what moved |
|---|---:|---:|---:|---|
| 3-factor | 407 | **319.5** | 1.27× | recombine 143→96 (SFPI), twiddle 87→55 (hybrid) |
| 2-factor | 386 | **334.7** | 1.15× | recombine 98→65 (SFPI bigK), twiddle 43→26 (hybrid) |

The 3-factor gained more because it has **more reduction stages** (3 recombines + 2 twiddles vs the
2-factor's 2 + 1) — and reduction is exactly what these optimizations attack.

## 4. Updated head-to-head sweep

| cores C | 3-factor µs | 2-factor µs | winner |
|--:|--:|--:|---|
| 1  | 4574 | 5550 | 3-factor |
| 2  | 2318 | 2820 | 3-factor |
| 4  | 1201 | 1461 | 3-factor |
| 8  |  666 |  757 | 3-factor |
| 16 |  407 |  428 | 3-factor |
| 32 | **319.5** | 334.7 | 3-factor |

**Conclusion reversed.** Before optimization the naive 2-factor was marginally *ahead* (386 vs 407 at
C=32). After applying the known reductions to both, the **3-factor is now ahead at every C** — its
extra reduction passes, once cheap, no longer offset its matmul savings. The gap is modest at C=32
(~5%) but consistent.

## 5. Where the time goes now (C=32 per-stage, µs)

| cost class | 3-factor | 2-factor |
|---|--:|--:|
| matmul | 75.6 (A2 25.3 + C2 23.8 + D2 26.5) | **239** (M1 K=8: 158, M2 K=4: 81) |
| recombine | 95.9 (3×32) | 64.9 (2×32) |
| twiddle | 54.7 (B 29 + TB 25) | 25.6 (1×) |
| decompose | 50.0 (3×~16) | 32.4 (2×16) |
| **all-to-all (corner turn)** | **95.8** | **0** |
| **total (chained)** | **319.5** | **334.7** |

After optimization the **structural costs are exposed**, and they differ sharply:

- **2-factor is matmul-bound**: M1+M2 = 239 µs (71% of the op). This is the fundamental 4× MAC cost
  of the coarse factorization — the optimizations can't touch it (the matmul is already int8 + K-loop).
- **3-factor is movement/reduction-bound**: the single-core **all-to-all corner turn (95.8 µs)** is now
  its largest stage, tied with its 3 recombines (95.9 µs); its matmul is cheap (75.6 µs).

So the two designs still trade the same budget, but the optimizations sharpened the picture: the
2-factor's wall is matmul FLOPs; the 3-factor's wall is the corner turn + having one more reduction
and twiddle.

## 6. Remaining levers (ranked, per variant)

**3-factor** (319.5 µs):
1. **Multi-core the all-to-all input staging** — the corner turn is the top single cost (95.8 µs) and
   each core still reads all 32 input tiles; a permuted/strided read makes the read divide by C
   (`NTT_32768_PERF.md`). Biggest win.
2. **Fuse** A1–A3 and C1–C3 / D1–D3 like NTT-1024 — eliminates per-stage dispatch + DRAM round-trips
   for the on-tile NTT-1024 blocks.

**2-factor** (334.7 µs):
1. **Matmul operand delivery** — M1 (K=8) at 158 µs is the bottleneck and is operand-feed bound
   (output-stationary K-loop re-reads K-tiles per output tile). Cache/coalesce the K-column reads.
2. Little reduction headroom left — recombine + twiddle are already SFPI.

## 7. Scaling outlook (unchanged by these optimizations)

At N=32768 the two are close (319.5 vs 334.7). The crossover still favors the 3-factor as N grows:
its matmul scales `O(N·N^{1/3})` vs the 2-factor's `O(N·√N)`, so the 2-factor's matmul wall (already
71% of its op) blows up faster. The optimizations don't change that asymptote — they just made the
3-factor win at N=32768 too, by cutting the reduction overhead that had been hiding its matmul
advantage.

## 8. Validation

Both variants: `mismatches = 0/32768` vs the direct O(N²) DFT golden at **every** C ∈ {1,2,4,8,16,32}
(seed 0; methodology matches the prior docs). The SFPI recombine / bigK / hybrid-twiddle kernels are
bit-identical to the building-block versions they replace.

## 9. Pointers

| What | Where |
|---|---|
| 3-factor host | `tests/ntt/ntt32768_dev.cpp` |
| 2-factor host | `tests/ntt/ntt32768_2factor_dev.cpp` |
| SFPI recombine / bigK | `kernels/shared/compute/{recombine_compute,recombine_bigK}.cpp` |
| Hybrid twiddle modmul | `kernels/shared/compute/modmul_sfpi_hybrid.cpp` |
| K-looped matmul (2-factor) | `kernels/shared/compute/matmul_compute_mtk.cpp` |
| All-to-all corner turn | `kernels/shared/dataflow/all_to_all_corner_mc.cpp` |
| Prior (pre-optimization) comparison | `docs/matmul/history/NTT_32768_COMPARISON.md` |
| Algorithm plans | `docs/matmul/NTT_32768_{2FACTOR,3FACTOR}_PLAN.md` |
