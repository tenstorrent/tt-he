# exp_modmul001 — three modmul kernels, one core, clean footing

**Status: CLOSED — the tile×tile modmul is optimised, ported, and at its floor.**

**`12.432 → 3.086 µs/tile, 4.03×`** on one core, bit-exact and canonical `[0, q)` throughout. Ported
into exp_Tg018m as `compute/modmul_fused3.cpp`: **`64.159 → 54.100 ms` end to end, 15.7% of the whole
CKKS multiply**, at `0/6029312` and 39.0 bits.

Seven kernels were built and measured on one core with zero profiler zones in any of them, per-tile
cost **fitted** from `span(T)` rather than divided. **This supersedes exp_Tg018k part A**, which was
not a fair comparison — see §4.

### The ladder, and what each rung cost

| kernel | µs/tile | what changed | needs Blackhole? |
|---|---|---|---|
| `sfpi_full` | 12.432 | the starting point: fp32 byte-split multiply, 14 SFPI passes, 4 tile-fills | — |
| `full_nofill` | 12.074 | four `fill_tile_int` → scalar args broadcast with `vUInt(x)` | **no** |
| `sfpi_full_bh` | 5.376 | `mul16` → two `sfpmul24` instructions. **Nothing else.** | yes |
| `hybrid_canon` | 4.930 | *(a different route: `mul_int_tile`, 4 passes — superseded)* | no |
| `fused_bh` | 4.660 | the four `a·b` products fused into one pass | yes |
| `fused2_bh` | 4.092 | all three product groups fused — 3 passes | yes |
| **`fused3_bh`** | **3.086** | one pass, no fills. **DST traffic 608 → 96 per tile.** | yes |

### Why it is closed

* **One SFPI pass, 2 loads + 1 store** — the floor for a two-operand elementwise op. You cannot read
  fewer than both inputs or write fewer than the output.
* **~2 cycles per operation** (130 cycles per face-row for ~62 ops), near the SFPU issue rate. The
  kernel is arithmetic-bound, not DST-bound.
* Every remaining lead was investigated and is small or refuted: Shoup is **10 products against
  Montgomery's 11**, not the 3× I claimed (§2b); lazy output is both blocked in ct_mul and now slower
  than `fused3_bh` (§3b); half-sync is ~2% and needs per-kernel config plumbing.

### Explicitly NOT done

* **`modmul_scalar`** (multiply by a runtime scalar, 2.3 ms in the census) and **`tensor2_fused`** are
  *different kernels*, still carry the fp32 byte-split, and have had none of this applied. The same
  technique should work; no number is quoted here because none was measured.
* Half-sync (`dst_full_sync_en = false`), ~2%, orthogonal to everything above.

### Three things worth carrying forward

1. **The biggest win came from a stale comment, not a new instruction.** *"One product per pass keeps
   SFPI register pressure low enough to compile"* was true of the fp32 code and false once the
   multiply changed. `sfpmul24`'s value was not its instruction count — it was dropping from 8
   registers per product to 2, which is what made the fusion **compile at all** (§2b control).
2. **An optimisation's value is set by the bottleneck, not by what it removes.** Deleting the four
   tile-fills removes 21% of DST accesses in both kernels; it is worth 2.9% on the arithmetic-bound
   fp32 kernel and materially more on the fused one (§2b).
3. **Measure against the best available option, not the one that is running.** `sfpmul24` looked like
   a 2.31× headline until a kernel already in the tree turned out to be faster — see the banner on
   `docs/he/history/exp_tg018s.md`.

`host/experiments/exp_modmul001.cpp` · `kernels/experiments/exp_modmul001/` · data `docs/he/data/modmul001.csv`

---

## 1. The result — three CANONICAL kernels, one core

One core (0,0), `q = 268435399` (28-bit), T swept 1…128, 5 replays each, min taken.
All three return a residue in `[0, q)` and score **0 canonical mismatches at every T**, so they are
interchangeable at any call site and no number here carries a contract caveat.

| kernel | **per-tile µs** | fixed µs | max resid | vs `full` | SFPI passes | DST/face-row | FILLs |
|---|---|---|---|---|---|---|---|
| `sfpi_full` | **12.432** | 1.12 | 0.03 | 1.00 | 14 | 53 | 4 |
| `full_nofill` *(no Blackhole feature)* | **12.074** | 1.17 | 0.03 | 1.03× | 14 | 53 | **0** |
| `sfpi_full_bh` | **5.376** | 1.03 | 0.08 | 2.31× | 14 | 53 | 4 |
| `hybrid_canon` | **4.930** | 1.08 | 0.08 | 2.52× | 4 + 9 tile ops | — | 4 |
| `fused_bh` | **4.660** | 1.02 | 0.07 | 2.67× | 9 | 33 | 4 |
| `fused2_bh` | **4.092** | 1.05 | 0.06 | 3.04× | 3 | 15 | 4 |
| **`fused3_bh`** | **3.087** | 1.02 | 0.06 | **4.03×** | **1** | **3** | **0** |

All canonical `[0, q)`, all `0` mismatches at every T.

Device spans (firmware `*-KERNEL` envelope), which is what the fit is over:

| T | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| `sfpi_full` | 13.59 | 25.89 | 50.84 | 100.57 | 200.00 | 398.90 | 796.63 | 1592.18 |
| `sfpi_full_bh` | 6.45 | 11.70 | 22.54 | 44.04 | 87.04 | 173.06 | 345.06 | 689.10 |
| `hybrid_canon` | 6.07 | 10.87 | 20.80 | 40.53 | 79.96 | 158.88 | 316.64 | 632.14 |
| `fused_bh` | 5.73 | 10.28 | 19.66 | 38.32 | 75.59 | 150.14 | 299.27 | 597.50 |
| `fused2_bh` | 5.15 | 9.18 | 17.43 | 33.82 | 66.54 | 132.01 | 262.95 | 524.83 |
| `fused3_bh` | 4.12 | 7.13 | 13.38 | 25.72 | 50.41 | 99.82 | 198.58 | 396.13 |

**The linear model is exact to 0.006%** — max residual 0.09 µs against a 1592 µs span at T=128 — and
all three carry the same ~1.1 µs fixed dispatch cost. So the per-tile figures are the arithmetic, not
the harness, and there is no divide-by-T assumption anywhere.

Two independent runs agree to better than 0.05% (12.431/5.376/4.932 then 12.430/5.375/4.930).

### What to take from it

* **`fused3_bh` is the fastest, at 3.087 µs/tile — 4.03× `sfpi_full`.**
* It is **1.74× ahead of `sfpi_full_bh`**, the kernel exp_Tg018m runs today, with the same canonical
  output and the same interface — a drop-in path swap.
* It is also faster than the LAZY kernel (4.805), so the lazy contract now buys nothing at all.
* It came from a stated hypothesis that was then tested — §2b.

The lazy kernel (`modmul_sfpi_hybrid`, output in `[0, 2q)`) is deliberately **not** in this table.
It is faster still — 4.805 µs/tile — but ct_mul cannot consume it (§3b), and mixing contracts in a
headline comparison is how the earlier write-up went wrong. Run it with `MM1_LAZY=1`.

---

## 2. What the three are

All three take the same runtime args (`n_tiles, q, q_inv_32, q_low16, q_high16`), the same CBs
(`c_0` = a, `c_1` = b, `c_16` = out), the same per-tile loop and the same `copy_tile`/`pack_tile`.
They are fed by one reader and drained by one writer that are byte-identical across the three.

| | how the multiply is done | non-multiply arithmetic | SFPI face passes | output |
|---|---|---|---|---|
| `sfpi_full` | **fp32 byte-split in SFPI** — split to 8-bit halves, 4 fp32 multiplies, 4 `int32_to_float` + 4 `float_to_uint16`, reassemble | SFPI | **7** | canonical |
| `sfpi_full_bh` | **`sfpi::fractional_mul`** — Blackhole's SFPMUL24, from SFPI | SFPI | 7 | canonical |
| `hybrid_canon` | **`mul_int_tile`** — the tt-metal tile-op API — 9 calls | SFPI, values kept in vector registers | **4** | canonical |
| `sfpi_hybrid` *(reference only)* | `mul_int_tile` | SFPI in registers | 4 | **lazy** |

`sfpi_full` and `sfpi_full_bh` differ **only** in the body of `mul16()`. Verified by diff — nothing
else in those two files differs. So that pair isolates the multiply primitive exactly.

### Where each kernel's cost actually is

Counted from the per-tile loop of each kernel (`S_*` = one SFPI face pass; `MUL` = one
`mul_int_tile`):

| | SFPI passes | `mul_int_tile` | FILL | `copy_tile` | **total DST ops** | µs/tile |
|---|---|---|---|---|---|---|
| `sfpi_full` | 14 | 0 | 4 | 2 | **20** | 12.430 |
| `sfpi_full_bh` | 14 | 0 | 4 | 2 | **20** | 5.375 |
| `hybrid_canon` | **4** | 9 | 4 | 2 | **19** | 4.930 |

**`sfpi_full` and `sfpi_full_bh` have identical operation counts and differ 2.31×.** That win is
entirely instructions *inside* the passes — `mul16` going from ~21 to 4 — and nothing structural.

**`hybrid_canon` has essentially the same operation count as `sfpi_full_bh` (19 vs 20) and is 8%
faster.** So its win is not "fewer operations" either: it is that a `mul_int_tile` op is cheaper than
a hand-written SFPI pass. The Blackhole LLK says why — `ckernel_sfpu_mul_int.h` uses `SFPLOADMACRO`
to fuse load+multiply+store into a hardware-sequenced macro at **1–3 cycles per input row**, where an
SFPI pass pays discrete load/compute/store per face-row plus `_llk_math_eltwise_sfpu_start_/done_`
bracketing.

**Correction.** An earlier version of this section called these "two independent levers" that should
compose, and gave the pass counts as 7 and 4. Both were wrong: the counts are 14 and 4, and the
levers are not orthogonal — using `fractional_mul` *means* doing the multiply in an SFPI pass, which
is precisely what `mul_int_tile` avoids. They are alternative implementations, not stackable ones.

### The experiment that IS implied: fuse products into passes

The four products `aL·bL, aH·bH, aL·bH, aH·bL` are four separate `mul_int_tile` ops — four DST round
trips — even though all four share two source operands that are already in registers. Doing them in
one SFPI pass was never possible before, and `modmul_sfpi_full`'s own comment says why:

> *"One product per pass keeps SFPI register pressure low enough to compile."*

The fp32 byte-split needs 4 floats plus 4 uint16 temporaries per product. `fractional_mul` needs
**two registers**. So four products in one pass should now fit — and **that**, not the instruction
count, is the reason `sfpmul24` matters for this kernel.

Sketch: fuse limbs + 4 products (1 pass), `S_lohi` (1), qinv-mul + split (1), the 4 `t·q` products
(1), reduce (1) → **~5 operations against 19**, with no `mul_int_tile` at all.

**Stated risk:** register pressure is the same thing that blocked this before. If four products do
not fit, the SFPI compiler fails or spills, and spilling to DST restores exactly the traffic being
removed. Hence an experiment, not a plan.

---

## 2b. The register-pressure hypothesis, tested

§2 predicted that `fractional_mul`'s much lower register cost should allow several partial products
to share one SFPI pass, collapsing passes whose only purpose was to shuttle intermediates through
DST. `modmul_sfpi_full`'s own comment states the old constraint:

> *"One product per pass keeps SFPI register pressure low enough to compile."*

The fp32 byte-split needs 4 `vFloat` plus 4 `vUInt` temporaries per product. `fractional_mul` needs
**two**. Two kernels test it, each with its failure mode stated in advance (a spill to DST would
restore the traffic being removed and make the kernel *slower*):

| | what was fused | SFPI passes | µs/tile | |
|---|---|---|---|---|
| `sfpi_full_bh` | — | 14 | 5.376 | baseline |
| `fused_bh` | `S_limbs` + 4× `S_mul` + `S_lohi` → **`S_ab`** | 9 | **4.660** | −13.3% |
| `fused2_bh` | also `S_mullo_a`+`S_mullo_b`+`S_split` → **`S_t`**, and 4× `S_mul`+`S_reduce` → **`S_final`** | 3 | **4.092** | −23.9% |
| `fused3_bh` | all three into **`S_all`**, and the four `fill_tile_int` replaced by scalar arguments broadcast with `vUInt(x)` | **1** | **3.087** | **−42.6%** |

**DST traffic per tile: 608 → 96 accesses.** `fused3_bh` reads `a` and `b` and writes `out`, and that
is the entire DST footprint of a Montgomery multiply — the floor for a two-operand elementwise op.

**The `fill_tile_int` calls turned out to be avoidable.** Each wrote a whole DST tile just to
broadcast a scalar every lane needs identically: 128 of the 608 accesses, 21% of all DST traffic. The
question was whether SFPI can materialise a *runtime* `uint32_t` into a vector register, since
`SFPLOADI` takes an assembly-time immediate — which is presumably why `fill_tile_int` exists at all.
It can: passing the constants through the variadic tail of
`_llk_math_eltwise_binary_sfpu_params_` and writing `vUInt(QINV & 0xFFFFu)` compiles and is exact.

**No spill at any step**, and every output is bit-exact against the golden — including `fused3_bh`,
whose peak liveness is about a dozen values through the `t*q` section. The whole Montgomery multiply
is now one call:

```
S_all(a, b, out, q_inv_32, q_high, q_low, q)
```

`a` and `b` are read once, every partial product and intermediate lives and dies in registers, and
`out` is written once.

### Where the floor is

At 3.087 µs/tile a tile is ~4167 cycles over 32 face-rows ≈ **130 cycles per face-row** for roughly
22 `sfpmul24` plus ~40 integer ops — about 2 cycles per operation, which is near the SFPU's issue
rate. The kernel is now arithmetic-bound rather than DST-bound, and no further fusion is available:
2 loads + 1 store is the minimum for a two-operand elementwise op. What remains outside the pass is
`copy_tile` ×2, `pack_tile`, and the `tile_regs` handshake.

**The algorithm is untouched** — same Montgomery reduce, same constants, same canonical `[0, q)`
output. The entire 3.04× over `sfpi_full` is instruction selection plus keeping intermediates off
DST. Nothing about the mathematics changed.

This is also what the `sfpmul24` finding was actually worth. On its own (`sfpi_full_bh`) it was 2.31×
but lost to a kernel already in the tree; its real value is that it made the fusion *possible*, and
the fused kernel beats everything including the lazy one.

### How much was reachable WITHOUT Blackhole?

Fair question, since one of the two changes — replacing the four `fill_tile_int` broadcasts with
scalar arguments broadcast via `vUInt(x)` — uses nothing Blackhole-specific. It is plain SFPI and was
available from the day `modmul_sfpi_full` was written, on Wormhole too. `modmul_full_nofill.cpp` is
exactly that and nothing else: same fp32 byte-split multiply, same 14 passes, four fewer tile-fills.

| kernel | µs/tile | needs Blackhole? |
|---|---|---|
| `sfpi_full` | 12.432 | — |
| **`full_nofill`** | **12.074 — 2.9%** | **no** |
| `fused3_bh` | 3.086 | yes |

**So the kernel was not fully optimised before — but only ~3% was on the table.**

The gap between "removes 21% of DST accesses" and "is worth 2.9%" is the point. With the fp32
byte-split each face-row runs about 170 instructions (eight `mul16` at ~21 each); against that,
deleting 128 of 608 DST accesses is noise. **The kernel was arithmetic-bound, so its memory traffic
did not matter.** The same change is worth much more inside `fused3_bh`, where the arithmetic is ~62
instructions per face-row and DST traffic is a large share of the total.

> **An optimisation's value is set by what the bottleneck is, not by what it removes.** The FILLs were
> 21% of DST traffic in both kernels; they were 2.9% in one and material in the other.

Of the 4.03× total, roughly **1.03× was reachable on Wormhole**. The rest required `sfpmul24` — not
for its instruction count, but because dropping from 8 registers per product to 2 is what made the
fusion compile at all (see the control below).

### The control: the fusion does NOT work without `sfpmul24`

The claim that `fractional_mul` is what *enabled* the fusion was, until now, an inference from
`modmul_sfpi_full`'s comment plus a register count. `modmul_fused3_fp32.cpp` tests it directly: the
identical full fusion — one pass, scalar constants, everything — with the fp32 byte-split `mul16` put
back. It **does not compile**:

```
sfpi_classes.h:147:12: internal compiler error: cannot store sfpu register (register spill)
```

Not slower — the SFPI compiler cannot even spill it. The fp32 byte-split needs 4 `vFloat` plus 4
`vUInt` temporaries per product, and the fused pass holds about a dozen values live through the `t*q`
section; `fractional_mul` needs two registers per product.

**So the two changes are not independent, and the order matters:**

| | | |
|---|---|---|
| the multiply swap alone | `sfpi_full` → `sfpi_full_bh` | **2.31×** |
| the fusion alone | `sfpi_full_bh` → `fused3_bh` | **1.74×** |
| the fusion without the multiply swap | — | **impossible — will not compile** |

The kernel is kept in the tree, unbuilt and out of the bench's variant list, as the evidence.

### Shoup / precomputed-quotient: analysed, and much weaker than claimed

I proposed replacing Montgomery with Shoup's constant-operand multiply for the NTT twiddle passes
(where the second operand is a fixed twiddle tile), claiming "3 products instead of 9". **That count
was wrong** — it counts 32×32 multiply *operations*, but this machine's multiplier is 23×23, so every
32×32 costs three or four of them. Counting 16×16 products, as the kernels actually issue them:

| | products |
|---|---|
| Montgomery | `a*b` full 64 = 4, `t = lo*q_inv` low 32 = 3, `m*q` high = 4 → **11** |
| Shoup | `mulhi(x, w')` = 4, `mullo(x, w)` = 3, `mullo(hi, q)` = 3 → **10** |

**~9% fewer products, not 3×** — and it costs a third operand read per tile (the `w'` companion
tile) and doubles the twiddle buffers. Not built; on this analysis it is not worth building ahead of
the items below.

---

## 3. The catch on the fastest one

`sfpi_hybrid` deliberately skips the final canonical subtract, so its output is in `[0, 2q)` — its
header says so: *"outputs chain back as lazy operands"*. The bench scores every kernel against its
**own** contract:

* canonical: `out == mont_mul(a,b)`
* lazy: `out == mont_mul(a,b)` or `out == mont_mul(a,b) + q`, and `out < 2q`

| | canonical mismatches | lazy mismatches |
|---|---|---|
| `sfpi_full` | 0 at every T | 0 |
| `sfpi_hybrid` | 19 / 40 / 69 / 131 / 258 / 520 (T = 1…32) | **0 at every T** |
| `sfpi_full_bh` | 0 at every T | 0 |

So the hybrid is **exactly correct under its stated contract** — 0 lazy mismatches across all 63
tiles measured. Its canonical mismatch rate is ~1.9%, which is inside the bound: REDC returns
`< q + t/R`, and with `q < 2^28`, `R = 2^32` the excess over `q` is at most `2^24 ≈ q/16`, so at most
~6% of results can exceed `q`.

**Whether the hybrid is usable is a caller question, not a kernel question.** A consumer that
immediately feeds the result into another modmul taking lazy operands can take the extra 12%; one
that needs a canonical residue cannot, and `sfpi_full_bh` is then the fastest correct option.

---

## 3b. Can ct_mul use the lazy kernel? No — and it does not need to

`hybrid_canon` is `sfpi_hybrid` with the canonical finalize restored — the ONLY difference is three
lines at the end of `face_reduce` (`b = hi + q - t_final; bq = b - q; out = (bq < 0) ? b : bq`).
Adding it separates the hybrid's two advantages, which were previously conflated:

| | µs/tile | difference |
|---|---|---|
| `sfpi_hybrid` (lazy, 4 passes) | 4.805 | |
| `hybrid_canon` (canonical, 4 passes) | 4.932 | **+0.127 = the cost of the finalize → laziness is worth 2.6%** |
| `sfpi_full_bh` (canonical, 7 passes) | 5.376 | **+0.444 = the 4-pass structure is worth 9.0%** |

**Three quarters of the hybrid's margin is structural, and needs no contract change.**

### And ct_mul cannot take lazy operands as it stands

Every modmul output in `ct_mul` lands in a consumer that requires a canonical residue:

| modmul call site (`host/experiments/exp_tg018m.cpp`) | what it is | consumer | lazy OK? |
|---|---|---|---|
| `pw` / `pw_jobs` | tensor-product pointwise multiply | `modadd_map`, `modsub_reduce` | **no** |
| the NTT twiddle pass (~line 1150) | `x * twiddle` | `fused_matmul_a` decompose | **no** |
| `twpass` (~line 1281) | `x * twiddle` | matmul / corner turn | **no** |

Two hard reasons, both in the consumers' own code:

* **`fused_matmul_a`'s radix-128 decompose covers exactly 28 bits** — `x & 127`, `(x>>7) & 127`,
  `(x>>14) & 127`, `(x>>21) & 127`. With `q = 268435399 ≈ 2^28`, a lazy value reaches `2^29` and bits
  28+ are silently dropped. Not an error, just a wrong answer.
* **`modadd_map`'s header states its contract**: *"Inputs a,b canonical `[0, q)`; a+b < 2q, so **one**
  conditional subtract gives canonical"*. Its body does exactly one (`add`, `sub q`, `min`). With
  lazy inputs `a+b < 4q` and one subtract is not enough.

There is also **no modmul → modmul chain anywhere in ct_mul** — the NTT alternates matmul → twiddle
modmul → matmul, and the tensor product is modmul → modadd. The lazy contract exists for a chaining
pattern this pipeline does not have.

Making lazy usable would mean a conditional subtract in the decompose and a second one in
`modadd`/`modsub` — i.e. moving the finalize to the consumer, for a 2.6% prize. **Not worth it.**
Take `hybrid_canon` instead: 9.0% over the kernel exp_Tg018m runs today, canonical output, no
consumer touched.

---

## 4. Why exp_Tg018k part A had to be redone

Two defects, both of which flattered or penalised specific kernels rather than affecting all equally:

1. **Unequal instrumentation.** `kernels/experiments/exp_tb001/compute/modmul_sfpi_hybrid.cpp` carries about
   **eight `DeviceZoneScopedN` markers inside its per-tile loop**
   (`modmul_tile_total`, `modmul_cb_wait`, `modmul_copy_in`, `modmul_math`, `modmul_limbs`,
   `modmul_ab_muls`, `modmul_lohi`, `modmul_qinv`, `modmul_tq_muls`, `modmul_reduce`,
   `modmul_pack_out`, `modmul_cb_push_pop`), while `modmul_sfpi_full.cpp` carries **one** at kernel
   entry. Those markers cost cycles. Any timing of those two files compared instrumentation as well
   as arithmetic — against the hybrid.
2. **Divide-by-T.** Part A reported `span / T`, which folds the ~1.1 µs fixed dispatch cost into the
   per-tile number. Small here, but it is the same assumption that made part B wrong by 8×
   (`history/exp_tg018r.md` §5), so it is not worth relying on. This experiment fits instead, and reports the
   residual so a kernel that does not scale with `T` would be visible.

It also ran at **G=130**, where the cores contend for DRAM, which compresses every ratio. The clean
one-core numbers are substantially larger:

| | exp_Tg018k part A (G=130, span/T, zones present) | **exp_modmul001 (1 core, fitted, no zones)** |
|---|---|---|
| `sfpi_full` | 16.321 | **12.432** |
| `sfpi_hybrid` | 9.743 → 1.68× | **4.807 → 2.59×** |
| `mul24` / `sfpi_full_bh` | 9.563 → 1.71× | **5.376 → 2.31×** |

**And the ordering flips.** Part A had `mul24` beating the hybrid (9.563 vs 9.743); on equal footing
the hybrid is 12% ahead (4.807 vs 5.376). The earlier claim that *"`mul24` is faster than
`sfpi_hybrid` and it is exact"* was an artifact of the hybrid carrying eight extra profiler zones per
tile. **Withdrawn.**

---

## 5. What this does and does not say

**Does say:** on one core, with everything else held equal, the arithmetic ranks
hybrid (4.807) < full_bh (5.376) < full (12.432) µs/tile, all three exact under their own contract.

**Does not say** what any of them is worth in the full pipeline. exp_Tg018m runs at G=130 where the
kernels are partly bound by DRAM and dispatch rather than arithmetic — which is why porting
`full_bh`'s multiply into it returned **11.9% end to end** (64.159 → 56.527 ms) rather than the 2.31×
the kernel shows here. Kernel ratios are an upper bound on pipeline effect, never a prediction.

Next, in order:

1. **DONE — `fused3_bh` is ported** as `kernels/experiments/exp_tb001/compute/modmul_fused3.cpp`. Measured end to
   end: **56.527 → 54.100 ms, 4.3%**, at `0/6029312` and 39.0 bits. As with the previous port the
   pipeline gain is well under the kernel ratio (1.74× of kernel → 4.3% of pipeline), because modmul
   dispatches are not 100% of the critical path.
2. **`hybrid_canon` and the lazy kernel are both superseded** by `fused2_bh` and need no further
   work. So does the lazy question — §3b was already "no", and lazy is now slower as well.
3. **Do not pursue lazy in ct_mul** — §3b. Worth 2.6%, blocked by two consumers whose range
   assumptions are documented in their own headers, and now slower than `fused3_bh` anyway.
4. **The tile×tile modmul is DONE.** One pass, 2 loads + 1 store — the floor for a two-operand
   elementwise op — and ~2 cycles per operation, near the SFPU issue rate. What is left is small and
   elsewhere: half-sync (~2%), and `modmul_scalar` / `tensor2_fused`, which are different kernels
   still carrying the fp32 byte-split and have had none of this applied.

---

## 6. Running it

```
. ./env.sh
TT_METAL_DEVICE_PROFILER=1 MM1_CSV=docs/he/data/modmul001.csv ./build/exp_modmul001 [ITERS=5]
```

| env | meaning |
|---|---|
| `MM1_T` | tile counts to sweep (default `1,2,4,8,16,32,64,128`) |
| `MM1_LAZY` | also run the lazy kernel, for reference — not in the default comparison |
| `MM1_CORE` | which core, as `x,y` (default `0,0`) |
| `MM1_CSV` | write the table |

A kernel only passes if it is exact under its own contract, so a variant that is fast because it
computes something else cannot be reported as a win.
