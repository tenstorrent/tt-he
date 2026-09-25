# Performance report: pure on-device NTT-1024

> **Status: Historical.** Superseded by [`../ntt_1024_perf.md`](../ntt_1024_perf.md).
> Per-stage numbers for the 147 µs version. Its "reduction dominates, not matmul" analysis is carried forward.
> Numbers are a snapshot of the code at the time and were measured on Wormhole n150.

**Program:** `tests/ntt/ntt1024_dev.cpp` (single Tensix core, Wormhole n150), `q = 167772161`.
**Pure on-device:** host writes input + baked constants once, issues 8 chained device programs that
pass intermediates through DRAM, and reads the result once. No host↔device traffic between stages,
no host arithmetic in the transform. Correct: `0/1024` vs the direct O(N²) golden, multiple seeds.

## Headline

```
147.0 us / NTT-1024   (single core, 7 chained device programs)   ≈ 6.8k NTT/s
```
Progression: 1167 µs (host-orchestrated baseline) → 465 µs (pure on-device) → 287 µs (single-kernel
recombine) → **147 µs (fold baked into the constant W — no runtime fold)**.

## Per-stage breakdown

Steady-state host wall-clock: warm-up enqueue+`Finish`, then 100 `EnqueueMeshWorkload` of that one
program bracketed by one `Finish` (includes dispatch + kernel; excludes host pre/post).

| stage | op | µs | note |
|---|---|--:|---|
| A1 | decompose `Xin` → limbs | 14.1 | mask & shift (was a 127 µs fold modmul — now baked into W) |
| A2 | int8 matmul (folded W) → `C_p` | 14.2 | 16 i8 tile-matmuls (the FPU; cheap) |
| A3 | recombine → `G` | 38.9 | 1 Montgomery reduction + shifts/adds |
| B  | twiddle `G ⊙ T` | 36.3 | 1 Montgomery modmul (1 tile) |
| C1 | decompose `G'` → limbs | 14.0 | mask & shift |
| C2 | int8 matmul (folded W) → `C_r` | 14.2 | |
| C3 | recombine → `H` | 38.7 | 1 Montgomery reduction |
| | **sum** | **~170** | (chained pipeline overlaps to 147) |

## Analysis

**The Montgomery modular reduction is the cost driver, not the matmul.** A single-tile Montgomery
reduction is ~37 µs (~50 int32 SFPU ops over a 1024-element tile); the int8 matmul — the O(n³)
"heavy" part — is only **~14 µs** (matrix engine), and decompose (mask&shift) is ~14 µs. After
baking the fold into `W`, exactly **3 reductions remain**: the two recombines (A3, C3) and the
twiddle (B), ~112 µs total ≈ 75% of the NTT. Everything else (2 matmuls + 2 decomposes) is ~57 µs.

The big win here was **the radix fold**: it used to run on the dynamic `Xin` as 4 modmuls (127 µs);
folding it into the constant `W` offline (as the IR always did, and stage C already did on device)
removed it entirely — `Xin` now only needs a mask&shift `decompose`.

## Optimization opportunities (ranked)

1. **Cut dispatch overhead by fusing programs.** 7 sequential `EnqueueMeshWorkload`s each pay launch
   latency; for single-tile stages that's a large fraction of the 147 µs (sum of kernel-only times
   ≈ 170, but the chained run is 147 — dispatch/overlap is significant). Fusing
   decompose+matmul+recombine into one program per stage would shrink it.
2. **Cheaper twiddle / recombine reduction.** The 3 Montgomery reductions are now the bulk. The
   recombine already uses the 1-reduction trick; the twiddle is an unavoidable modmul, but could be
   fused with the adjacent recombine.
3. **A real device profiler pass** (`TT_METAL_DEVICE_PROFILER=1`) for true per-RISC kernel cycles —
   the numbers here are host wall-clock and include dispatch.

## Multi-core matmul variant (`ntt1024_dev_mc.cpp`)

The plane-matmul produces 4 independent int32 result planes, so it was split across **4 Tensix
cores** (core `oi` computes plane `C_oi = Σ_k a_k @ b_k` and writes one tile; one core then gathers
+ recombines). New kernels: `compute/matmul_plane.cpp`, `dataflow/reader_matmul_mc.cpp`. Correct:
`0/1024`, seeds 0/1/7/42. See `NTT_1024_MULTICORE.md` for the design.

```
135 us / NTT-1024   (matmul on 4 cores)     vs   147 us (single core)   — ~8% faster
```

| stage | single-core µs | 4-core µs |
|---|--:|--:|
| A2 matmul | 14.2 | **12.6** |
| C2 matmul | 14.2 | **12.1** |
| (all other stages) | unchanged | unchanged |

**The matmul was never the bottleneck**, so the gain is small: the matmul stage dropped only
~14 → ~12.5 µs, not ~4×. At single-tile sizes a matmul stage is dominated by program-dispatch +
CB/NOC overhead, not the FPU, so splitting the (already ~14 µs) 16 tile-matmuls across 4 cores trims
only the small compute fraction — every core still pays the fixed dispatch + one DRAM tile write.
The 3 Montgomery reductions (≈ 83% of runtime) are still single-core; parallelizing *those*, or
fusing stages to cut dispatch, is where the real win is. The 4-core matmul layout is the right shape
for larger M/N/K, where the matmul becomes the dominant term.

## Caveats

- Host wall-clock, steady-state enqueue→`Finish`; includes dispatch overhead (significant for
  single-tile workloads), so it overstates pure kernel time.
- Single core, single NTT, batch = 1, N = 1024 (32×32, factor = T so no cross-tile all_to_all).
- Forward NTT only.

## Reproduce
```
cd device && cmake -B build -G Ninja && ninja -C build ntt1024_dev
./build/ntt1024_dev [q] [seed]    # prints total + per-stage timing
```
