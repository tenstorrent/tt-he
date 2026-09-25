# exp_Tg018k — the two compute kernels that are left

**Status: Historical** — parts B and C are arithmetically wrong (they divide by `T`); see the caveat below. Kept because part A's kernel census is sound and the error itself is instructive.

> ## ⚠ SUPERSEDED IN PART — read `docs/he/history/exp_tg018r.md` first
>
> **Parts B and C divide by `T` when they must not.** `baseconv_fbc` converts ONE 1024-coefficient
> tile per dispatch whatever `T` is — its reader takes `page = k*T`, a page OFFSET, not a count
> (`reader_gather_batch_pg.cpp`, runtime arg 2). So **every SFPU µs/tile figure below is 8× too
> small.** The FPU side genuinely looped `nct = T*32`, so its numbers are right, and that mismatch
> is the whole of part C's conclusion.
>
> Measured control (same kernel, `Ls=16, Ld=8`, G=64): **776.98 µs at T=1, 830.60 µs at T=8** —
> eight times the work for 7% more time, because it is not eight times the work.
>
> **Withdrawn:** "the FPU route is 2.54× slower", the `Ld ≈ 22` crossover, "199 GMAC/s still loses",
> and recommendation 5 ("do not port the matrix-engine FBC at `Ld = 8`"). exp_Tg018r measures both
> routes on equal footing at `N = 2^16` over `(Ls, Ld) ∈ {8,16,32}²` with the transpose fused in, and
> the **matrix engine wins every cell** — 1.19× at 8→8, 10.3× at 32→32, 1.98× at the shape
> exp_Tg018 runs.
>
> **Part A has ALSO been superseded — `docs/he/exp_modmul001.md`.** Two defects: the hybrid kernel
> carries ~8 `DeviceZoneScopedN` markers inside its per-tile loop while `sfpi_full` carries one, so
> the timing compared instrumentation as well as arithmetic; and it ran at G=130 where DRAM
> contention compresses every ratio. Re-measured on one core with zero zones and per-tile cost
> FITTED rather than divided: `sfpi_full` 12.432, `sfpi_hybrid` **4.807**, `mul24` 5.376 µs/tile.
> **The ordering flips** — the hybrid is 12% faster than `mul24`, not slower. Everything below about
> the relative standing of those two is withdrawn.
>
> **Part A's T-division was at least sound.** The modmul kernels genuinely loop `T`
> tiles — `reader_ab_depth` takes `n` as a COUNT, not a page offset — verified by control:
> `sfpi_full` spans 18.04 µs at T=1 and 130.56 µs at T=8, a marginal tile of 16.07 µs against the
> 16.32 µs/tile reported. **Part A has since been superseded upward, not downward:**
> `docs/he/history/exp_tg018s.md` adds an `sfpmul24` variant that is **1.75× faster and bit-exact**.

**Status: DONE (measurement).** Part A kills a lead I had flagged twice, finds a free 2.7%, and turns
up one variant that is 1.70× faster and *not equivalent*. Part B establishes what the FBC's cost is
actually made of. Part C then **builds** the matrix-engine FBC at our shape and measures it against
the SFPU on the same card in the same run — and it is **2.54× slower at one core, 6.78× at 130**,
which reverses what Part B's model and the older n150 numbers both predicted. The crossover is at
`Ld ≈ 22`; we run at `Ld = 8`.

`host/experiments/exp_tg018k.cpp` · parent `docs/he/exp_tg018m.md` · method `docs/he/dram_guide.md` ·
data `docs/he/data/tg018k_fpu.csv`

## Why

`exp_Tg018m` took the multiply from 238.7 ms to 64.9 ms by fixing memory and occupancy. What survives
is arithmetic, measured with a **verified 1-1 dispatch map** (`dispatch_csv.py`, all 2879 rows agreeing
between host and trace):

| kernel | calls | total | share |
|---|---|---|---|
| `modmul_sfpi_full` | 1088 | 20.357 ms | **32.2%** |
| `baseconv_fbc` | 15 | 11.897 ms | **18.8%** |

51% of the multiply, and neither responds to scheduling any more.

## Part A — five implementations of the same Montgomery multiply

The tree contains **five** and uses one. They differ in how the 32×32-bit product is formed, which is
the expensive part. All take the same runtime args and CBs, so they are swappable in place; only
`_half` needs a different `ComputeConfig` (half-sync).

G=130, T=8 tiles/core, one dispatch each, all against the same host golden:

| variant | µs/tile | vs in-use | mismatches | |
|---|---|---|---|---|
| `sfpi_full` *(what exp_Tg018m runs)* | 16.321 | 1.00× | **0** | |
| `compute` — native `mul_int_tile` | 18.194 | **1.11× slower** | **0** | |
| `sfpi_hybrid` | 9.743 | 0.60× | 16,519 | **not equivalent** |
| `sfpi_half` — half-sync, 4 DST tiles | 16.008 | 0.98× | **0** | |
| **`mul24`** — Blackhole `sfpmul24` | **9.563** | **0.59×** | **0** | `exp_tg018s.md` |
| **`mul24` + half-sync** | **9.323** | **0.57× = 1.75×** | **0** | **take this one** |
| `mullo` | 6.522 | 0.40× | 1,064,960 | a different operation |

> **`mul24` is faster than `sfpi_hybrid` AND exact**, which closes the section below: there is no
> longer any reason to characterise the hybrid's 16,519 mismatches. `docs/he/history/exp_tg018s.md` has the
> measured `SFPMUL24` semantics and why `mul_int_tile` could not deliver this.

### The native-multiply lead is dead

I flagged this twice: `modmul_sfpi_full`'s header says *"Wormhole SFPI has no integer multiply"*, so
every product is a four-way fp32 byte-split — while `fused_matmul_a` happily uses `mul_int_tile` on
the same card. The obvious inference was that the emulation is legacy and the native path would win.

**It loses by 11%.** `modmul_compute` implements the same algorithm on `mul_int_tile` and measures
18.144 µs/tile against 16.410. The fp32 byte-split is not a Wormhole leftover being carried
needlessly — on Blackhole it is the faster of the two. Lead closed, and it cost one run rather than a
rewrite.

### `sfpi_half` is a free 2.7%

Same answer, bit-exact, 15.965 against 16.410. It is scheduled into 4 DST tiles so it can run
**half-sync** (`dst_full_sync_en = false`), letting math and pack overlap instead of alternating
across the full DST. Worth ~0.55 ms of the multiply for a one-line host change — the only catch is
that the host must set the flag per kernel, which `exp_Tg018m` currently sets globally.

### `sfpi_hybrid` is 1.70× faster and computes something else

9.743 µs/tile, but **16,519 of 1,064,960 elements differ (1.55%)**. Its header says outputs are left
in **lazy form** — `[0, 2q)` rather than canonical `[0, q)`, "so outputs chain back as lazy operands".

**Correction.** This section used to claim 1.55% was "far too low for a plain lazy/canonical mismatch,
so the contract is something narrower". That was wrong arithmetic on my part. Montgomery REDC returns
`< q + t/R`; with `q < 2^28` and `R = 2^32` the excess over `q` is at most `2^24 ~ q/16`, so **at most
~6%** of results can land in `[q, 2q)` — not the ~50% I was implicitly comparing against. 1.55% sits
inside that bound. The difference is exactly the documented skipped subtract and there is nothing
unidentified about it.

**Superseded as a speed lead** — `mul24` (`docs/he/history/exp_tg018s.md`) is faster *and* canonical, and is
now what exp_Tg018m runs. But the hybrid is still interesting for a different reason: its speed does
**not** come from its multiply, which is the slower `mul_int_tile` route. It comes from **4 face
passes instead of 7**, keeping intermediates in LREGs instead of round-tripping through DST. That
lever is independent of the multiply-instruction lever, so a hybrid-structured kernel using
`fractional_mul` should beat both 9.563 and 9.743. Not yet built or measured.

### `mullo` is not a candidate

Every element differs. The name is the tell — it is a low-half multiply, not a modular one. Included
so the shootout is complete rather than curated.

## Part B — what the FBC's 792 µs is made of

`baseconv_fbc` scales each of `Ls` source residues by a constant, then accumulates `Ls × Ld` modular
multiply-accumulates per element. Sweeping the shape at G=130, T=8:

| | Ld=2 | Ld=4 | Ld=8 |
|---|---|---|---|
| **Ls=4** | 12.543 | 18.331 | 32.515 |
| **Ls=8** | 24.864 | 35.570 | 57.345 |
| **Ls=16** | 49.961 | 71.081 | **113.372** |

*(µs per tile; all exact against the host golden)*

**Cost is linear in `Ls` with no constant term** — doubling `Ls` doubles the time at every `Ld`
(ratios 1.94–2.01). In `Ld` it is affine. Fitting:

$$\text{cost} \;\approx\; L_s \times \big(1.8 + 0.66\,L_d\big)\ \mu s\ \text{per tile}$$

At the production shape `Ls=16, Ld=8` that is `28.8 + 84.5 = 113.3` against 113.372 measured. So:

| term | µs/tile | share | what it is |
|---|---|---|---|
| `Ls × 1.8` | 28.8 | 26% | one modmul per source residue |
| **`Ls × Ld × 0.66`** | **84.5** | **74%** | **the CRT multiply-accumulate** |

### The matrix-engine FBC is not a lead — it is built, measured, and on the roadmap

**Correction to how this was first written up here.** `docs/bconv/base_conversion_overview.md` already
measured the matrix-engine combine end to end, and `docs/he/roadmap.md` carries it as item 4:

| version | 4→4 | 8→8 |
|---|---|---|
| SFPU fused (`baseconv_fused`) | 356 µs | 1277 µs |
| matrix engine (`baseconv_matmul`) | 732 µs | 733 µs |
| **matrix engine, block-packed** (`baseconv_matmul_packed`) | **204 µs** | **285 µs** |

**4.5× at 8→8**, and the SFPU version scales 3.6× from 4 to 8 moduli while the matrix engine scales
1.4× — we run at `Ls = 16`, further out that curve than anything in that table. An exact variant
exists too (`baseconv_matmul_crns`, 254 µs, 2.04× faster than the Shenoy–Kumaresan route, `0/4096`).

> **Those numbers are n150 and `Ls = Ld = 8`.** Re-measured on this Blackhole card they are 3.3×
> faster across the board (SFPU 8→8: 1277 µs there, 385 µs here) and the ratio drops to 2.23×. Part C
> below re-runs the comparison at **our** shape on **this** card, and the sign flips.

**Why exp_Tg018 still runs the SFPU one:** `kernels/experiments/exp_tb001/` contains `baseconv_fbc.cpp` and
nothing else FBC-related. When exp_Tb001 forked its own kernel tree — deliberately, so kernel work
could not regress the shared binaries — only the SFPU FBC came across. `matmul_fbc`,
`matmul_fbc_packed`, `recombine_rns` and `eps_correct_crns` stayed in `kernels/shared/compute/`, and every
experiment built on that tree inherited the gap. It was never a decision to prefer the SFPU version.

### What adopting it actually requires — the layout is the work

Reading the kernels rather than the summary changes the estimate. `matmul_fbc` is **coefficient-major**:

    A[c][i·NR + p] = limb_p( y[c][i] )      rows = coefficients, columns = (source limb, radix limb)

and in `tests/baseconv/baseconv_matmul.cpp` those A tiles are built **on the host**. exp_Tg018m stores
**limb-major** — one limb is 64 tiles, each tile 32×32 = 1024 coefficients — so the conversion is a
real data movement, on device:

| | |
|---|---|
| lanes needed | `Ls × NR` = 64 → **2 A-tiles per 32-coefficient group** (the kernel is written for 1) |
| B matrix | 64 rows × `Ld`=8 columns → uses **8 of 32** output columns (25%) |
| per one of our tiles | 32 coefficient-groups × 2 = **64 A-tiles, built from 16 source tiles** |

So three genuinely new pieces, not a port:

1. **An on-device limb-major → coefficient-major transform** that also does the `y` scaling and the
   radix-128 decomposition. This is precisely what roadmap item 4 already names — *"fed by an
   all-to-all transpose (limb-major ↔ coeff-major)"* — and it is structurally required, for the same
   reason the NTT needs its corner turn: **the matrix engine contracts only within a tile**, and the
   axis being contracted here is the limb axis, which is a tile axis.
2. **A 16-source fold** — two A-tiles accumulated per group, where the kernel does one.
3. **Column packing**, or the matmul wastes 75% of its output width. This is exactly what
   `matmul_fbc_packed` is for, and it is where 3.6× of the measured 4.5× comes from.

### Finishing the arithmetic at OUR shape — and it does not favour the matrix engine

Before building anything, the same cost model from Part B applied to the matrix path:

| | per one of our 1024-coefficient tiles |
|---|---|
| coefficient-groups | 32 |
| A-tiles per group | 2 (`Ls·NR` = 64 lanes / 32) |
| tile-matmuls | 32 × 2 × 4 = **256** |
| MACs performed | 8.39 M |
| MACs landing in a used output column | 2.10 M → **25%** (`Ld`=8 of 32) |

**`Ld = 8` fills 8 of the matmul's 32 output columns**, so 75% of every product is thrown away.
Costing that against the SFPU's measured 84.5 µs MAC term, at two plausible matmul rates:

| matmul rate | unpacked | packed 4× | break-even transpose budget |
|---|---|---|---|
| 34 GMAC/s (`fused_matmul_a` in situ) | 246.7 µs — **loses** | 61.7 µs | **8.9 µs/tile** |
| 159 GMAC/s (its matmul phase alone) | 52.8 µs | 13.2 µs | **57.4 µs/tile** |

That was the estimate. **Part C replaced it with a measurement, and the estimate was wrong in an
interesting way: the matmul is faster than either guess, and the estimate omitted the stage that
actually decides the answer.**

## Part C — the matrix-engine FBC, built and measured at our shape

`matmul_fbc` is written for `Ls = 8` (32 contraction lanes = one tile). At `Ls = 16` the contraction
is 64 lanes, so each output plane is the sum of **two** tile-matmuls accumulated in DST before the
pack — `kernels/experiments/exp_tb001/compute/matmul_fbc_ls16.cpp`, with `reader_fbc_ls16.cpp` holding the
`NR·NA = 8` int8 B-tiles resident for the whole dispatch. The 4 radix planes are then recombined
`(Σ_r 128^r · plane_r) mod p_j` by the existing `recombine_rns`, per-column modulus, on the SFPU.

Both stages run at G cores × T of our tiles, timed in separate `Finish`'d loops so the profiler sees
one dispatch group per replay, and **every point is verified against a host golden**: at G=130, T=8
that is `0 mismatches / 8,519,680 outputs`. The measurement is matmul + recombine; the limb-major →
coefficient-major transpose is built on the host and is **not** in these numbers.

**µs per one of our 1024-coefficient tiles, `Ls=16 → Ld=8`:**

| G | FPU matmul | SFPU recombine | **FPU route** | **SFPU `baseconv_fbc`** | FPU / SFPU |
|---|---|---|---|---|---|
| 1 | 42.07 | 198.28 | **240.34** | **94.55** | 2.54× |
| 4 | 42.45 | 198.29 | **240.74** | **94.93** | 2.54× |
| 16 | 92.62 | 198.61 | **291.23** | **97.03** | 3.00× |
| 64 | 217.18 | 212.56 | **429.74** | **103.88** | 4.14× |
| 130 | 406.52 | 362.73 | **769.25** | **113.47** | 6.78× |

**The matrix engine is not the problem — it is excellent.** 8.39 M int8 MACs in 42.07 µs is
**199 GMAC/s per core**, 3× the rate `fused_matmul_a` achieves in situ and 128× the SFPU's 1.55
GMAC/s MAC term. The optimistic 52.8 µs row above was nearly right.

**Two things kill it anyway.**

**1. The recombine, 198 µs — 4.7× the matmul it serves.** It runs once per *coefficient*-tile, so 32
times per one of our tiles, at 6.2 µs each. The SFPU FBC produces `Ld`=8 finished output tiles for
the same 1024 coefficients. So the matrix route pays 32 tile-reductions to the SFPU route's 8 output
tiles — and 24 of those 32 columns are the unused ones.

**2. Above ~16 cores, the planes saturate DRAM.** The split into two programs pushes 4 int32 planes
per coefficient-tile through DRAM: **1.15 MB per our-tile** against the SFPU path's 96 KB. At G=130
the recombine alone moves 681 MB per replay = **235 GB/s**, which is the measured DRAM ceiling
(`docs/he/dram_guide.md`). That is the whole G column: 2.54× at one core, 6.78× at 130.

### The crossover is in `Ld`, and it is at `Ld ≈ 22`

Both FPU stages run full 32-wide tiles whatever `Ld` is, so the FPU route's cost should not depend on
`Ld` at all. Measured at G=1 — identical to three decimals, each verified exact:

| `Ld` | 8 | 16 | 32 |
|---|---|---|---|
| FPU route, µs/our-tile | 240.354 | 240.354 | 240.348 |

The SFPU is linear in `Ld` (G=1, `Ls`=16; `Ld` ≤ 11 is where the kernel's runtime-arg budget stops):

| `Ld` | 2 | 4 | 6 | 8 | 10 | 11 |
|---|---|---|---|---|---|---|
| SFPU, µs/our-tile | 31.25 | 52.34 | 73.45 | 94.56 | 115.60 | 126.15 |

fit **10.16 + 10.545·`Ld`**, max residual < 0.1 µs. Setting that equal to the FPU's flat 240.35:

> **crossover at `Ld` ≈ 21.8.** Below it the SFPU wins; above it the matrix engine does. We run at
> `Ld = 8`, less than half way. At `Ld = 32` the FPU route would be **1.45× faster** (240 vs 348).

**So the matrix-engine FBC is not a lead at the shape we run, and the reason is the same `Ld = 8` cap
that costs the SFPU path 21%.** Lift the cap and the matrix engine becomes the right answer; leave it
and the SFPU is 2.5× ahead before the transpose is even counted.

Data: `docs/he/data/tg018k_fpu.csv`.

### The real problem is the `Ld = 8` chunk, and it hurts both paths

`Ld` is capped at 8 because the `Ls × Ld` CRT constant block travels in **runtime args**, and
`Ls=16, Ld=16` would need 338 of a 256 budget. That single cap does two things:

**It makes the SFPU repeat its `y` scaling.** The per-source term is paid once per chunk:

| Ld | chunks for Ldt=48 | µs/tile | runtime args |
|---|---|---|---|
| **8** *(today)* | 6 | **679.7** | 194 ✓ |
| 16 | 3 | 593.3 | 338 ✗ |
| 24 | 2 | 564.5 | 482 ✗ |
| 48 | 1 | **535.7** | 914 ✗ |

**21% of the FBC — ~2.5 ms of the multiply — is the y-scale being redone six times.**

**And it caps the matrix engine at 25% output utilisation**, which is exactly what the table above
shows killing it. At `Ld = 32` the matmul would be 100% efficient.

So the first move is neither porting `matmul_fbc` nor leaving things alone: **move the `Ls × Ld`
constant block out of runtime args and into a buffer.** That is a contained change to
`baseconv_fbc.cpp` — read the constants from a CB at kernel entry instead of `get_arg_val` — and it

* wins **~21% of the FBC on the SFPU path immediately**, and
* removes the cap that makes the matrix-engine path lose, so that decision can be re-taken at
  `Ld = 32` where its efficiency is 4× better.

### What is left on the matrix-engine path, if `Ld` is ever lifted

Two levers remain, and Part C bounds both:

**Fusing the two programs.** Keeping the 4 planes in CBs instead of DRAM — exactly what
`fused_matmul_a` does for the NTT matmul — removes 1.15 MB of DRAM traffic per our-tile and with it
the entire G column of the table above. It cannot go below the one-core floor Part C measured
(**240.3 µs**), because at G=1 there is no contention to remove. So fusion is worth up to
**3.2×** at G=130 and **nothing** at G=1.

**Column packing** (`matmul_fbc_packed`). Four `Ld`=8 groups packed into one 32-wide matmul is the
same thing as running at `Ld` = 32, which Part C measured directly: the FPU route does not get
cheaper, it produces 4× more output. That is the crossover result, arrived at from the other side.

Neither touches the recombine, which is 82% of the one-core floor.

### Why the matrix engine looked like the right target

The MAC term is 74% of the FBC, and the SFPU is a bad place to do it:

| | MACs per tile | time | rate per core |
|---|---|---|---|
| `baseconv_fbc` MAC term (SFPU) | 131,072 | 84.5 µs | **1.55 GMAC/s** |
| `fused_matmul_a` (int8 matrix engine) | 524,288 | ~8 µs | **65.5 GMAC/s** |
| **`matmul_fbc_ls16`, measured in Part C** | **8,388,608** | **42.07 µs** | **199 GMAC/s** |

That reasoning was sound and the rate is even better than it assumed — **128× apart**. What it missed
is that the matrix engine only does the *multiply*: the radix-128 emulation costs 16 MACs per real
one and the `Ld`=8 output width wastes a further 4×, so 199 GMAC/s of int8 delivers **1.56 GMAC/s of
useful modular MAC** — level with the SFPU — and then the recombine is charged on top. **Rate was the
wrong figure of merit; useful-MACs-per-tile-op is the right one.**

The earlier estimate of a **3.5× ceiling on the FBC** does not survive Part C. At `Ls=16, Ld=8` the
matrix route is 2.5× the wrong way.

## What this experiment says to do next, in order

1. **Lift the `Ld = 8` cap** — move the `Ls × Ld` CRT constant block out of runtime args and into a
   buffer. It is a contained change to `baseconv_fbc.cpp`, it wins **~21% of the FBC on the path we
   already run**, and it is the precondition for everything on the matrix-engine path. Now measured
   as the single decision that separates the two routes.
2. **Characterise `sfpi_hybrid`'s 1.55%.** Largest prize (1.70× on 32%), cheapest to investigate — it
   is one diff-dump away from either a free 8 ms or a closed door.
3. **Take `sfpi_half`.** 2.7% of modmul for a per-kernel `dst_full_sync_en`, with bit-exact output.
4. **Do not pursue `mul_int_tile` for modmul.** Measured 11% slower; the fp32 byte-split stays.
5. **Do not port the matrix-engine FBC at `Ld = 8`.** Measured 2.54× slower at one core, 6.78× at
   130, before the limb→coefficient transpose is counted at all. Re-take this decision at `Ld ≥ 22`,
   where Part C puts the crossover — and note that step 1 is what makes that shape reachable.

## Running it

```
. ./env.sh
TT_METAL_DEVICE_PROFILER=1 ./build/exp_tg018k [G=130] [T=8] [ITERS=3]
```

| env | meaning |
|---|---|
| `TGK_PART` | `a` (modmul variants) \| `b` (SFPU FBC shape) \| `c` (matrix-engine FBC) \| `both` = a+b \| `all` |
| `TGK_G`, `TGK_T` | cores, tiles per core |
| `TGK_LS`, `TGK_LD` | part B shape lists, e.g. `TGK_LS=4,8,16 TGK_LD=2,4,8` |
| `TGK_LDMAX` | part B destination-buffer pool (default 8); `Ls=16` admits `Ld ≤ 11` |
| `TGK_TC`, `TGK_LDC` | part C tiles/core and destination count (defaults `TGK_T`, 8) |

Part A reports mismatches as **data** — finding out which implementations agree is the point — so
only Parts B and C, which must be exact, can fail the run.

Part C's blobs are large: at G=130, T=8 it builds 68 MB of int8 A-tiles on the host and allocates
545 MB of planes on device. Reproducing the one-core floor (`TGK_G=1`) costs seconds.
