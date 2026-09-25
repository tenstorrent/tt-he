# exp_Tg018r — SFPU vs matrix engine for the base conversion, on equal footing

**Status: Historical** — its conclusion was carried into the multiply and is now live in [`../exp_tg018ntt.md`](../exp_tg018ntt.md) §4e (`TG18NTT_FBC=fpu`, 32.498 → 23.991 ms). Kept for the controlled `(Ls, Ld)` sweep behind that decision.

**Status: DONE (measurement).** Both routes fully on device, both verified against the same host
golden at every point, swept over `(Ls, Ld) ∈ {8,16,32}²` at `N = 2^16` on 64 cores. **The matrix
engine wins every cell**, from 1.19× at 8→8 to **10.3× at 32→32**.

This also **corrects exp_Tg018k part C**, which concluded the opposite. See §5.

`host/experiments/exp_tg018r.cpp` · data `docs/he/data/tg018r_grid.csv` · method `docs/he/dram_guide.md`

---

## 1. The result

Device µs for ONE conversion of all `N = 2^16` coefficients, 64 cores, one 1024-coefficient tile per
core, 3 replays (min taken), from tt-metal's own per-RISC `*-KERNEL` zones with a verified 1-1
dispatch join (108 groups, 108 dispatches).

| Ls | Ld | SFPU | FPU scale | FPU main | **FPU total** | **FPU / SFPU** | mism S / F |
|---|---|---|---|---|---|---|---|
| 8 | 8 | 397.5 | 59.9 | 273.2 | **333.0** | **0.84** | 0 / 0 |
| 8 | 16 | 747.6 | 59.2 | 273.8 | **333.1** | **0.45** | 0 / 0 |
| 8 | 32 | 1447.0 | 59.9 | 274.4 | **334.4** | **0.23** | 0 / 0 |
| 16 | 8 | 788.7 | 122.2 | 277.0 | **399.2** | **0.51** | 0 / 0 |
| 16 | 16 | 1477.0 | 122.5 | 276.8 | **399.3** | **0.27** | 0 / 0 |
| 16 | 32 | 2853.5 | 122.5 | 278.4 | **400.9** | **0.14** | 0 / 0 |
| 32 | 8 | 1563.5 | 238.7 | 297.0 | **535.7** | **0.34** | 0 / 0 |
| 32 | 16 | 2928.3 | 239.0 | 300.0 | **539.0** | **0.18** | 0 / 0 |
| 32 | 32 | 5659.9 | 239.6 | 310.8 | **550.4** | **0.10** | 0 / 0 |

**Scope.** Both routes compute the **approximate** FBC — `out_j = Σ_i y_i·C2[i][j] mod p_j`, no ε
correction — which is what `baseconv_fbc` and exp_Tg018 run today. Exactness is a separate axis
(`docs/bconv/base_conversion_overview.md` §2.5: the CRNS ε is a fixed-point slice of the same matmul
accumulator, so it is cheaper to add on the matrix route than on the SFPU one, but it is not measured
here). Nothing about the comparison depends on it: both sides compute the same function and are
checked against the same golden.

`FPU total = scale + main`. **Nothing is excluded**: `main` is one dispatch containing the
limb→coefficient transpose, the radix-128 decomposition, 512 int8 matmuls, the modular recombine and
the coefficient→limb transpose. `scale` is stage 1 (`y_i = x_i·C1_i mod q_i`), which the SFPU number
also includes. The only host-side work on either route is building the constants.

### The shape of the two costs

| | fit (µs per `N=2^16` conversion, 64 cores) | error |
|---|---|---|
| SFPU | `Ls · (6.1 + 5.38·Ld)` | < 1% |
| FPU | `265 + 8.5·Ls + 0.6·Ld` | < 1% |

**The FPU route is essentially flat and the SFPU route is `Ls × Ld`.** `main` moves only 273 → 311 µs
across the entire grid, because every stage of it runs full 32-wide tiles whatever `Ls` and `Ld` are:
32 coefficient-groups, 4 radix planes, 4 lane-groups, one recombine. The SFPU pays one modular
multiply-accumulate per `(source, destination)` pair, so it pays for the shape.

Crossover from the fits: the matrix engine wins for `Ls > 6.7` at `Ld = 8`, `Ls > 14` at `Ld = 4`,
`Ls > 32` at `Ld = 2` — roughly **whenever `Ls · Ld ≳ 55`**. Every shape exp_Tg018 runs is past it.

---

## 2. What made this possible: transpose the operands, not the data

The matrix engine contracts only **within** a tile, and the axis being contracted is the source-limb
axis — which in exp_Tg018's limb-major storage is a **tile** axis. The standard reading of that (and
`docs/he/roadmap.md` item 4, and exp_Tg018k) is that the matrix route needs an on-device all-to-all
transpose pass first. **It does not.**

```
limb-major form:    plane [c][j] = Σ_lane A [c][lane] · B [lane][j]     ← needs A coefficient-major
exp_Tg018r:         plane'[j][c] = Σ_lane Bt[j][lane] · At[lane][c]     ← needs At LIMB-major
```

Both contract over the limb axis. The second wants `At` limb-major — and *that is what a strided read
of the tiles we already store produces.* A tile in exp_Tg018 is 1024 coefficients of one residue,
tilized as four 16×16 faces:

```
element (r, c)  ->  offset ((r/16)*2 + c/16)*256 + (r%16)*16 + (c%16)
```

so tile **row** `r` is exactly two contiguous runs of 16 elements, and row `i` of the operand
`Y'[i][c] = y_i[r·32 + c]` is likewise two contiguous runs. The whole transpose is therefore **2 NoC
copies of 64 bytes per (coefficient-group, source limb)** — `reader_fbc_tr.cpp` on the way in,
`writer_fbc_tr.cpp` on the way out, both inside the same dispatch as the matmul.

Consequences, all of which show up in the numbers:

* **no transpose pass, no scratch buffer, no extra dispatch** — the transpose costs about 10 µs of
  the 277, which is why `main` at `Ls=16, Ld=8` (277.0) lands so close to exp_Tg018k's transpose-free
  matmul+recombine floor (240.3);
* **the radix planes never leave L1** — the two-program split in exp_Tg018k pushed 1.15 MB per
  our-tile through DRAM and saturated the bus at 235 GB/s past 16 cores;
* the output modulus becomes a **row** index rather than a column index, so `recombine_rns` runs
  unchanged with its 5 parameter tiles built row-wise on the host.

Lane order is p-major (`lane = p·32 + i`), so `At[p]` is `limb_p` of `Y'` elementwise — no lane
permutation anywhere. It costs 4 A-tiles even when `Ls < 32` (rows ≥ `Ls` are killed by zeroed `Bt`
columns); at `Ls = 32` it is exactly optimal, which is visible as `main` rising only 273 → 297 µs
from `Ls=8` to `Ls=32`.

---

## 3. The other blocker: the `Ld` cap was never a dispatch limit

`baseconv_fbc.cpp` copies the constants into **stack arrays** — `uint32_t C2[16][12]` — and its own
header warns that overflowing them silently scribbles the compute RISC's stack. That is what capped
exp_Tg018 at `Ld = 8`, and it was written up (here and in exp_Tg018k) as a *runtime-arg budget*
limit. It is not:

```
tt_metal/impl/kernels/kernel.cpp:632   constexpr uint32_t max_runtime_args_tensix = 4096;
```

`Ls=32, Ld=32` needs 1186 args. `baseconv_fbc_wide.cpp` reads the same args **in place** with
`get_arg_val` at the point of use — no stack arrays, no cap — and costs nothing measurable: at
`Ls=16, Ld=8` it measures 788.7 µs against the original kernel's 776.98 µs at the same shape, 1.5%.

That change alone is what let the `Ls=32` row of the grid exist.

---

## 4. DRAM

Both routes get the same treatment, from `docs/he/dram_guide.md`:

* every buffer is laid out `page = our_tile · L + limb`, so the `L` pages one core touches are
  **consecutive** and therefore land on consecutive banks. The obvious layout (`limb · NOT + our_tile`)
  puts all of a core's pages on **one** bank whenever `NOT` is a multiple of 8;
* readers put the whole batch in flight before the barrier (`reader_seq_batch.cpp`), rather than
  paying a round trip per tile.

Neither route is DRAM-bound at this size: at `Ls=Ld=32` the FPU route moves 8 MB in 550 µs (15 GB/s)
and the SFPU 12 MB in 5660 µs (2 GB/s), both far under the ~250 GB/s ceiling. This is a compute
comparison, which is what makes it a fair one.

---

## 5. Correction to exp_Tg018k part C

**exp_Tg018k part C reported the matrix engine losing by 2.54×. That was an artifact and the sign is
wrong.**

`baseconv_fbc` converts **one** 1024-coefficient tile per dispatch whatever `T` is: its reader takes
`page = k·T` — a page *offset*, not a count (`reader_gather_batch_pg.cpp`, runtime arg 2). Parts B
and C divided that one-tile span by `T = 8`, so **every SFPU number in exp_Tg018k is 8× too small.**
The FPU side genuinely looped `nct = T·32` coefficient-tiles, so its numbers were right — and the
mismatch is the whole of part C's conclusion.

Measured control, same kernel, same shape (`Ls=16, Ld=8`, G=64):

| | span |
|---|---|
| original `baseconv_fbc`, T=1 | 776.98 µs |
| original `baseconv_fbc`, T=8 | 830.60 µs |

Eight times the work for 7% more time — because it is not eight times the work. Corrected, part C's
own `Ls=16, Ld=8` numbers read **FPU 240.3 µs vs SFPU ~756 µs at one core**, i.e. the matrix engine
was already 3.1× ahead there, before the fusion and the transposes measured here.

Two independent checks that this grid is the right one:

* `788.7 µs` at `Ls=16, Ld=8` matches exp_Tg018m's production FBC dispatch (`11.897 ms / 15 =
  793 µs`) to 0.5%;
* `main = 277 µs` decomposes as recombine ≈ 198 + matmul ≈ 42 + decompose ≈ 32 + transposes ≈ 10,
  using exp_Tg018k part C's independently measured recombine and matmul costs.

The derived claims in exp_Tg018k that rested on the bad column — the `Ld ≈ 22` crossover, the "199
GMAC/s still loses", the "do not port at `Ld = 8`" recommendation — are all withdrawn. §1 above
replaces them.

---

## 6. What to do with it

**At the shape exp_Tg018 runs today** (`Ls=16, Ld=8`), swapping `baseconv_fbc` for the fused matrix
route is **788.7 → 399.2 µs, 1.98×**. The FBC is 11.897 ms of the 64.1 ms multiply, so that is
11.897 → 6.02 ms, **9.2% off the whole multiply**, with no change to the data layout — the transpose
lives entirely inside the new reader and writer.

**With the `Ld` cap lifted too**, the conversion exp_Tg018 actually wants is `Ls=16 → Ldt=48`, run
today as six `Ld=8` chunks that each re-read the sources and redo stage 1:

| | per `N=2^16`, composed from measured cells |
|---|---|
| today: 6 × SFPU(`Ls=16, Ld=8`) | 6 × 788.7 = **4732 µs** |
| FPU, 2 × 32-wide column groups + one scale | 2 × 278 + 122 = **~678 µs** |

**~7×**, taking the FBC to roughly 1.7 ms. That second row composes measured cells rather than being
a measured cell itself — `Ld = 48` needs two passes of the 32-wide kernel — so it is the one number
here to re-measure before quoting it as a result.

Order of work:

1. **Port the fused matrix FBC into exp_Tg018m at `Ls=16, Ld=8`.** Measured 1.98×, no layout change,
   no cap to lift first. 9.2% of the multiply.
2. **Then lift the `Ld` cap** (`baseconv_fbc_wide`'s in-place arg reads, already written and
   measured) and run 32-wide column groups. Takes it to ~7× on the FBC.
3. **Then look at `scale`** — at `Ls=32` it is 239 µs of the FPU route's 550 (43%), and it is a plain
   SFPU modmul per source. Folding it into the fused kernel as a per-row multiply, the way the
   recombine already handles its per-row modulus, is the obvious next lever.

---

## 7. Running it

```
. ./env.sh
TT_METAL_DEVICE_PROFILER=1 TGR_CSV=docs/he/data/tg018r_grid.csv ./build/exp_tg018r [G=64] [ITERS=3]
```

| env | meaning |
|---|---|
| `TGR_G` | cores; must equal the our-tile count `N/1024` (both compute kernels convert one per dispatch) |
| `TGR_N` | ring degree, default 65536 |
| `TGR_LS`, `TGR_LD` | grid lists, default `8,16,32` each |
| `TGR_ROUTE` | `sfpu` \| `fpu` \| `both` |
| `TGR_CSV` | write the grid |

Kernels added: `compute/baseconv_fbc_wide.cpp`, `compute/fbc_scale_wide.cpp`,
`compute/fbc_fpu_tr.cpp`, `dataflow/reader_fbc_tr.cpp`, `dataflow/writer_fbc_tr.cpp`,
`dataflow/reader_seq_batch.cpp`, `dataflow/writer_seq_batch.cpp` — all in `kernels/experiments/exp_tb001/`.
