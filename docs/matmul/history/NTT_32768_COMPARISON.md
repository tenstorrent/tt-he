> **Superseded (numbers pre-date the SFPI/hybrid optimization pass).** See

> **Status: Historical.** Superseded by [`../ntt_32768_perf.md`](../ntt_32768_perf.md).
> Pre-optimization head-to-head; its conclusion (2-factor marginally ahead) was REVERSED once the known reductions were applied to both. Self-declared superseded even before this consolidation.
> Numbers are a snapshot, measured on Wormhole n150.
> `NTT_32768_OPTIMIZED.md` for current measurements — after SFPI recombine, SFPI bigK recombine,
> and the hybrid twiddle, the comparison **reverses**: the 3-factor now wins at every C
> (319.5 vs 334.7 µs at C=32). The structural analysis below still holds.

# NTT-32768: naive 2-factor (128,256) vs hand-tuned 3-factor (32,32,32)

Both fully on-device, `q = 167772161`, int8 modular matmul, same `C = 1…32` core-count sweep, same
host-wall-clock methodology (20 iters, one `Finish`). Both verified `0/32768` vs the direct DFT,
seeds 0 & 7.

- **3-factor** (`ntt32768_dev.cpp`): NTT-level decomposition — 32×32 tiles, fold offline, on-tile
  NTT-1024, one tile-granular corner turn. 12 stages.
- **2-factor** (`ntt32768_2factor_dev.cpp`): the "just throw big matmuls at TT" baseline —
  `128×256` four-step `matmul → twiddle → matmul`, **no** NTT-level fold/tiling/corner-turn; the only
  tiling is the mechanical 32×32 a matmul does internally. 7 stages. (Needs `recombine_bigK`: K=256
  pushes radix-128 planes to ~2²².)

## Wall-clock (µs / NTT-32768)

| cores C | 3-factor | 2-factor | 2-factor speedup | 2-factor eff |
|--:|--:|--:|--:|--:|
| 1  | 7154 | 7222 | 1.00×  | 100% |
| 2  | 3610 | 3655 | 1.98×  | 99% |
| 4  | 1850 | 1881 | 3.85×  | 96% |
| 8  |  988 |  967 | 7.48×  | 93% |
| 16 |  573 |  534 | 13.6×  | 85% |
| 32 |  407 |  386 | 18.7×  | 59% |

**They are essentially tied — the naive 2-factor is even marginally faster at high C** (386 vs 407 µs
at C=32), and scales slightly better (fewer, coarser stages → less dispatch).

## Why — per-stage attribution at C=32

| cost class | 2-factor | 3-factor |
|---|--:|--:|
| matmul | **240 µs** (M1 K=8: 159, M2 K=4: 81) | 73 µs |
| recombine (modular reduction) | 98 (2×49) | 143 (3×48) |
| twiddle | 43 (1×) | 87 (2×) |
| decompose | 32 (2×) | 45 (3×) |
| corner-turn (all-to-all) | **0** | 95 |
| **total (chained)** | **386** | 407 |

The two designs trade the *same* budget between two pools:

- **2-factor**: 4× the matmul MACs (12.6M vs 3.1M) ⇒ matmul dominates at 240 µs — but it has **one
  fewer matmul pass, fewer reductions (2 vs 3 recombines, 1 vs 2 twiddles), and no corner turn**.
- **3-factor**: the NTT decomposition cuts matmul to 73 µs — but spends the savings on an extra
  reduction pass, an extra twiddle, and the corner-turn (324 µs of non-matmul work).

Net: a wash. **On TT, matmul is cheap and modular reduction + data movement are the real cost**, so
the classic FFT win (fewer MACs) doesn't translate into a wall-clock win at this size — the saved
MACs just move the bottleneck to reductions/movement, which the 2-factor has less of.

## When each wins

- **N = 32768**: 2-factor ties (slightly ahead). The decomposition's MAC saving (4×) is offset by
  its extra reduction/movement.
- **Larger N**: 2-factor matmul scales `O(N·√N)`, 3-factor `O(N·N^{1/3})`. At N=2²⁰ the matmul gap is
  ~16× (not 4×); matmul then dominates the 2-factor and the decomposition wins decisively. The
  crossover is roughly where `matmul_work × (gap−1)` exceeds the 3-factor's extra reduction+movement.
- **Simplicity / bring-up**: the 2-factor is far simpler (7 stages, no corner turn, reuses stock
  matmul) and competitive — a good default until N grows enough to need the decomposition.

## Costs the 2-factor pays that aren't wall-clock

- **Folded constant size**: `W256` folded is 16 limb-pairs × 8×8 tiles = **1024 int8 tiles (1 MB)**
  vs the 3-factor's 16 tiles. Bigger constant footprint / write-once cost.
- **`recombine_bigK`**: K=256 forces the wider-finalize recombine (planes ~2²²); "no tiling at all"
  is impossible — K must be bounded for the int32/recombine path (unit-tested in
  `test_recombine_bigk`, 0/1024).
- **Matmul DRAM traffic**: M1's K=8 reads 8 K-tiles per output tile; the 159 µs M1 stage is
  read-bound, not FPU-bound.

## Takeaway

For NTT-32768 on a single Wormhole, **the hardware-aware NTT decomposition does not beat naively
handing TT two big modular matmuls** — both land at ~390–410 µs on 32 cores. The decomposition's
value is asymptotic (larger N) and in constant footprint, not raw throughput at this size, because
the bottleneck on this hardware is modular reduction and data movement, not matmul FLOPs.

## Reproduce
```
cd device && ninja -C build ntt32768_dev ntt32768_2factor_dev
./build/ntt32768_dev        [q] [seed]   # 3-factor sweep
./build/ntt32768_2factor_dev [q] [seed]  # 2-factor sweep
```
