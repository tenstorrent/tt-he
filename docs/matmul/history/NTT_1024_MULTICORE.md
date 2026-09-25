# Multi-core variant: int8 matmul split across 4 Tensix cores

> **Status: Historical.** Consolidated into [`../ntt_1024.md`](../ntt_1024.md) §5.1.
> Full design detail of the 4-core plane-matmul variant.
> Numbers are a snapshot of the code at the time and were measured on Wormhole n150.

**Program:** `tests/ntt/ntt1024_dev_mc.cpp` (Wormhole n150), `q = 167772161`. Same fully-on-device
pipeline as `ntt1024_dev.cpp`, but the int8 plane-matmul runs on **4 cores** instead of 1.

## Idea

The plane-matmul produces 4 int32 result planes `C_0..C_3` (one per radix limb of the output).
Each plane is an independent sum of `NB=4` int8 tile-matmuls. So we give **each plane its own core**:

```
core oi:  C_oi = sum_k a_k @ b_k      (NB int8 matmuls accumulated locally — the per-core "add")
          write C_oi -> DRAM tile oi
```

All 4 cores run one program (`CoreRange {0,0}..{3,0}`) with per-core runtime args; each reads its
operands from the shared DRAM constants and writes its single tile. A single core then **gathers the
4 planes and recombines** them (`recombine_compute.cpp`, unchanged). The matmul orientation differs
per stage and is selected by `mode` in `reader_matmul_mc.cpp`:

| stage | mode | LHS `a` | RHS `b` | plane formula |
|---|---|---|---|---|
| A (`G=W@Xin`) | 0 | folded W (16 tiles) | Xin limbs (4) | `C_p = Σ_r a[p·NB+r] @ b[r]` |
| C (`H=G'@W`)  | 1 | G' limbs (4) | folded W (16 tiles) | `C_r = Σ_p a[p] @ b[p·NB+r]` |

New kernels: `compute/matmul_plane.cpp` (one plane: NB matmuls → 1 tile),
`dataflow/reader_matmul_mc.cpp` (feeds plane `oi`'s operands).

## Result (verified `0/1024`, multiple seeds)

```
135 us / NTT-1024   (matmul on 4 cores)     vs   147 us (single core)
```

| stage | single-core µs | 4-core µs |
|---|--:|--:|
| A1 decompose(Xin) | 14.1 | 12.5 |
| A2 matmul | 14.2 | **12.6** |
| A3 recombine | 38.9 | 38.8 |
| B  twiddle | 36.3 | 36.3 |
| C1 decompose | 14.0 | 12.8 |
| C2 matmul | 14.2 | **12.1** |
| C3 recombine | 38.7 | 38.8 |

## Analysis

The matmul split works and is correct, but the **speedup is small (~8%)** — and the matmul stage
only dropped ~14 → ~12.5 µs, not ~4×. Why: the int8 matmul was never the bottleneck. A single-tile
matmul stage is dominated by **program-dispatch + circular-buffer/NOC overhead**, not by the FPU
matmul itself, so splitting the (already cheap) ~16 tile-matmuls across 4 cores trims only the small
compute fraction; every core still pays the same fixed dispatch + one DRAM tile write.

The real cost is the **3 Montgomery reductions** (2 recombines + 1 twiddle ≈ 112 µs ≈ 83% of the
135 µs), all still single-core. To actually move the needle, parallelize *those* — e.g. split each
1024-element recombine/twiddle tile across cores, or fuse decompose+matmul+recombine into one
program per stage to cut the 7-enqueue dispatch overhead. The matmul-on-4-cores is the right shape
for scaling to larger M/N/K (many planes/tiles), where the matmul does become the dominant term.

## Reproduce
```
cd device && cmake -B build -G Ninja && ninja -C build ntt1024_dev_mc
./build/ntt1024_dev_mc [q] [seed]
```
