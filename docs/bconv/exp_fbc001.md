# exp_fbc001 — FBC on the SFPU vs the matrix engine, both as ONE fused dispatch

**Status: DONE (measurement).** Both routes now run the whole approximate base conversion as a
**single dispatch with no DRAM access during the computation**, from and to the same limb-major
coefficient tiles. Swept over `(Ls, Ld) ∈ {8,16,32}²` at `N = 2^16` on 64 cores; **`0` mismatches in
all 18 cells** against the same host golden.

`host/experiments/exp_fbc001.cpp` · `kernels/experiments/exp_fbc001/` · data `data/fbc001_grid.csv` ·
predecessor [`../he/history/exp_tg018r.md`](../he/history/exp_tg018r.md)

---

## 1. What changed from exp_Tg018r

`exp_Tg018r` already had the SFPU route as one fused dispatch, but the matrix-engine route was **two**,
handing `y` between them through DRAM — `Ls` tiles out and `Ls` tiles back per core, 128 KB each way at
`Ls=32`. This experiment folds that scale stage in, so no `y` buffer exists.

**Why that was not trivial.** The limb→coefficient transpose is free *only* because it is fused into
the read (`reader_fbc_tr_fused.cpp` gathers it as 2 × 64 B NoC copies per `(group, limb)`), so anything
*before* the transpose would have to precede the reader — impossible inside one dispatch. The scale
therefore has to run **after** the transpose, where its constants vary along the **row** axis (row = source
limb) instead of being uniform per tile. That is exactly the shape phase 3's recombine already had
("PER-ROW modulus"), reading `q, qinv, qhigh, qlow` from parameter *tiles*. Phase 0 reuses that
sequence verbatim with a second set of 5 tiles for the source moduli, so `cb_par` grows 5 → 10.

Phase 0 and phase 1 share one `tile_regs_acquire`: the mulmod leaves `y` in DST 4 and `S_decompose`
reads DST 4, so **`y` never reaches even a CB**.

## 2. The result

Device µs for ONE conversion of all `N = 2^16` coefficients, 64 cores, 3 replays (min), from
tt-metal's own per-RISC `*-KERNEL` zones. **[M-p150]**

| Ls | Ld | SFPU | **FPU (fused)** | FPU/SFPU | FPU in exp_Tg018r (2 dispatches) |
|--:|--:|--:|--:|--:|--:|
| 8 | 8 | 397.5 | **440.8** | 1.11 | 333.0 |
| 8 | 16 | 748.3 | **441.5** | 0.59 | 333.1 |
| 8 | 32 | 1447.7 | **442.3** | 0.31 | 334.4 |
| 16 | 8 | 788.1 | **444.2** | 0.56 | 399.2 |
| 16 | 16 | 1476.5 | **444.1** | 0.30 | 399.3 |
| 16 | 32 | 2854.0 | **445.4** | 0.16 | 400.9 |
| 32 | 8 | 1563.7 | **457.0** | 0.29 | 535.7 |
| 32 | 16 | 2929.1 | **457.2** | 0.16 | 539.0 |
| 32 | 32 | 5659.4 | **457.3** | **0.08** | 550.4 |

Ratio grid (< 1.00 means the matrix engine wins):

| Ls\Ld | 8 | 16 | 32 |
|--:|--:|--:|--:|
| **8** | 1.11 | 0.59 | 0.31 |
| **16** | 0.56 | 0.30 | 0.16 |
| **32** | 0.29 | 0.16 | **0.08** |

**The SFPU column reproduces exp_Tg018r to 0.1%** (397.5 vs 397.48, 5659.4 vs 5659.95) — that route is
unchanged, so it is the control confirming the harness is the same measurement.

### The fused FPU route is essentially flat in BOTH parameters

| | fit, µs per `N=2^16` conversion, 64 cores |
|---|---|
| SFPU | `Ls · (6.1 + 5.38·Ld)` — pays for the shape |
| FPU, exp_Tg018r (2 dispatches) | `265 + 8.5·Ls + 0.6·Ld` |
| **FPU, fused (this)** | **`435 + 0.68·Ls + 0.06·Ld`** |

**Fusing removed the `Ls` term almost entirely — slope 8.5 → 0.68 µs per source limb, 12.5×** — because
that slope *was* the separate scale dispatch and its `y` round trip. Total spread across the whole grid
is now **16 µs on 441**, under 4%, against a 14× change in `Ls·Ld`.

## 3. But fusing is NOT uniformly better, and that is the interesting part

The constant rose **265 → 435 µs** while the `Ls` slope fell. So fusing is a **loss at small `Ls` and a
win at large**:

| Ls | fused | 2-dispatch | |
|--:|--:|--:|---|
| 8 | 440.8 | 333.0 | **108 µs worse** |
| 16 | 444.1 | 399.3 | 45 µs worse |
| 32 | 457.3 | 550.4 | **93 µs better** |

Break-even at **`Ls ≈ 24`** (`170 µs / 7.9 µs per limb`).

**The mechanism is the loop bound, not the DRAM saving.** The old scale ran once per *limb* tile — `Ls`
passes per our-tile. Phase 0 runs once per *coefficient group* — **32 passes, whatever `Ls` is**, because
after the transpose the source-limb axis is inside a tile and the group axis is what remains to iterate.
At `Ls=32` the counts match and the removed round trip is pure profit; at `Ls=8` it is 4× the scale work
for a round trip that was only 32 KB. Removing a DRAM round trip is not free if the fusion changes what
you iterate over.

Consequence for the SFPU/FPU crossover: it moves from `Ls·Ld ≳ 55` to **`Ls·Ld ≳ 72`** — visible in the
grid as `(8,8)` being the one cell the SFPU still wins.

## 4. Constant parameter size — the mirror image of the latency

| | formula | 8→8 | 32→32 |
|---|---|--:|--:|
| **SFPU** | `4·(Ls·Ld + 3·Ls + 2·Ld + 2)` bytes ≈ **4·Ls·Ld** | **424 B** | **4.6 KB** |
| **FPU** | **56 KB, flat** = 16 KB `Bt` + 40 KB `par` | 56 KB | 56 KB |

**The scaling is inverted relative to the timing.** The SFPU pays `Ls·Ld` in *size* for the same reason
it pays `Ls·Ld` in *time* — one constant and one modmul per (source, destination) pair. The FPU is flat
in both, for the same reason too: a tile is 32×32 whatever the shape.

**Root cause: the matrix engine has no scalar operand path.** `matmul_tiles` reads its operands through
the unpacker, so they must be tiles in L1. The SFPU's `mul_int_tile` works on DST and takes a scalar
broadcast in-register with `vUInt(x)` (`../he/exp_modmul001.md`). So the SFPU can consume a scalar *as*
a scalar and the matrix engine cannot — everything below follows from that.

Splitting storage from information content:

| | information | stored | ratio |
|---|--:|--:|--:|
| `Bt` at 32→32 | 3.5 KB | 16 KB | **4.6×** — *inherent* |
| `Bt` at 8→8 | 224 B | 16 KB | 73× — 4.6× inherent, the rest padding |
| `par` at 32→32 | 1.25 KB | 40 KB | **32×** — *overhead* |
| `par` at 8→8 | 320 B | 40 KB | 128× |

* **`Bt`'s 16 bytes per C2 value are not waste.** `Bt[rr*NP+p] = limb_rr((128^p · v_ij) mod p_j)`: the
  `p` axis exists because the dynamic operand is radix-128 split into `NP=4` limbs, each needing its own
  weight plane, and `rr` because the product must return as `NR=4` int8 planes to accumulate. That
  `NB² = 16` fold is the price of modular arithmetic on an int8 engine — the same fold as the NTT's
  `WfoldA`. At 32→32 there is no padding at all.
* **`par`'s 40 KB is genuine overhead.** Ten tiles carry only `5·Ls + 5·Ld` distinct values because each
  is **row-constant**: the SFPI reduce reads its modulus as `dst_reg[qIdx*NV+i]`, elementwise, so making
  `q` vary by row requires physically replicating it across all 32 columns. There is no row-broadcast
  primitive. Avoidable in principle with an index register and a select; not cheaply.

> **Half of `par` is this experiment's own cost.** The 5 source-side tiles added to fuse the scale took
> `par` from 5 tiles to 10, i.e. **+20 KB**. So the fusion cost 20 KB of resident constants on top of the
> +170 µs constant time in §3 — recorded because it is easy to present the fusion as free and it is not.

**In time it is ~2%, though.** 56 KB/core × 64 cores = 3.5 MB chip-wide ≈ **8.4 µs at 436 GB/s against a
~440 µs kernel**. Loaded once, amortised. The footprint only binds when L1 is contested — which is
exactly what happened in [`../matmul/exp_ntt001.md`](../matmul/exp_ntt001.md), where the 2-factor NTT's
196 KB of resident twiddles *was* the dominant term. Here 56 KB is 3.9% of L1 and does not bite.

**One hard ceiling the FPU does not have.** The SFPU's constants are runtime args, capped at 4096 per
kernel, so `Ls·Ld + 3Ls + 2Ld + 2 ≤ 4096` ⇒ **`Ls·Ld ≲ 3900`** (about `Ls=Ld=62`). Past that `C2` must
become a buffer and the size advantage goes with it. (`../he/history/exp_tg018r.md` §3 notes the earlier
256-arg limit was self-imposed; 4096 is the real one.)

## 5. Two bugs, both caught by the per-cell golden

1. **The host tilized only the first 5 parameter tiles** (`for t<5`), so the 5 new source-side tiles
   reached the device in row-major order and the kernel read garbage constants — `mism 524288`, every
   output wrong. A one-character fix, and completely invisible without the golden: the kernel ran, at
   plausible speed, and produced plausible-looking numbers.
2. A dropped `copy_tile_init(cb_par)` relative to phase 3's sequence. Fixed, though it turned out not
   to be the failure — worth recording so the next person does not re-suspect it.

## 6. Running it

```
. ./env.sh
TT_METAL_DEVICE_PROFILER=1 FBC_CSV=docs/bconv/data/fbc001_grid.csv ./build/exp_fbc001 [G=64] [ITERS=3]
```

| env | meaning |
|---|---|
| `FBC_G` | cores; must equal the our-tile count `N/1024` |
| `FBC_N` | ring degree, default 65536 |
| `FBC_LS`, `FBC_LD` | grid lists, default `8,16,32` each |
| `FBC_ROUTE` | `sfpu` \| `fpu` \| `both` |
| `FBC_CSV` | write the grid |

`fpu_scale_us` is `0` in the CSV **by construction** — there is no separate scale dispatch. Both
routes' kernels are in `kernels/experiments/exp_fbc001/`; the ones shared with `exp_tb001` are copies, so that
tree is untouched.
