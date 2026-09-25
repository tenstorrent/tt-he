# exp_modmul002 — Barrett vs Montgomery, at equal effort

**Status: DONE (measurement).** Measured as **pure computation latency on the device trace**,
**Montgomery is 2.3% faster than Barrett** — 3.0866 vs 3.1578 µs/tile, resolved at a fit residual of
0.0065%. The two carry identical multiply counts; the gap is Barrett's extra shifts and its
correction, most of which the schedule hides.

`host/experiments/exp_modmul002.cpp` · `kernels/experiments/exp_modmul002/` · data `docs/he/data/modmul002.csv`

---

## 1. The result — pure computation latency

MATH (`TRISC_1`) `*-KERNEL` span per dispatch, one core, `q = 268435399`, T swept 1…128, per-tile
cost **fitted** from `span(T) = fixed + T·per_tile`. Both bit-exact at every T.

| T | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| `montgomery_1pass` | 3.73 | 6.78 | 12.96 | 25.32 | 50.01 | 99.38 | 198.16 | 395.71 |
| `barrett` | 3.83 | 6.99 | 13.29 | 25.90 | 51.18 | 101.71 | 202.75 | 404.86 |

| kernel | **per-tile µs** | fixed µs | max resid | vs Montgomery |
|---|---|---|---|---|
| `montgomery_1pass` | **3.0866** | 0.622 | 0.0256 | 1.0000 |
| `barrett` | **3.1578** | 0.659 | 0.0257 | **1.0231** |

**Montgomery is 2.3% faster, and the measurement resolves it.** Max residual is 0.026 µs on a 396 µs
span — **0.0065%** — while the difference is 0.071 µs, about 3× the residual.

Two independent confirmations:

* **`montgomery_1pass` measures 3.0866 here and 3.086 in `docs/he/exp_modmul001.md`** — different
  bench, different harness, same number.
* **It reconciles with the instruction counts.** 104/97 predicts 7.2%; measured 2.3%. The schedule
  hides roughly two-thirds of Barrett's extra work (§3), but not all of it.

### Why this measurement and not the earlier one

The `*-KERNEL` zone is emitted by firmware around `kernel_main()`, so it excludes the host enqueue,
the command queue, and `wait_for_go_message()` — which sits outside the zone. What remains is the
kernel's own execution:

| measured by | fixed cost per dispatch |
|---|---|
| host wall-clock | **~27 µs** |
| device `*-KERNEL` trace | **0.62 µs** |

Forty times smaller, and the fit removes even that. The host wall-clock protocol earlier in this
document could only bound the difference under ~4% of kernel cost; the trace resolves it at 2.3%.

**The route to this number was a sequence of measurement errors worth recording**, because each one
produced a confident wrong answer:

1. **Dividing `ms` by `T`** folds in a fixed cost that penalises the faster kernel — it reported
   Barrett 5% slow when the kernels differed by 2%.
2. **Unpaired runs** made the answer depend on a kernel's *position* in the dispatch sequence;
   removing an unrelated third variant flipped the sign, and this document briefly claimed Barrett
   was 2.6% *faster*.
3. **A single run** was published before the noise floor was known. Five runs then straddled 1.0.
4. **The paired protocol** (interleaved, 10 repetitions) was honest but underpowered: `+0.013 ±
   0.056 µs`, sign test 5 of 10 — correctly reported as unresolved.

Only the device trace has the precision. **A few-percent kernel comparison on this hardware needs the
`*-KERNEL` fit; host wall-clock cannot do it at any number of repetitions.**

### Instruction counts, from the compiled MATH TRISC

Full breakdown, per face-row, disassembled from the JIT-built kernels — see §3 for what each column
means. All three issue **exactly 22 `SFPMUL24`**; that is structural (§2), not a coincidence.

---

## 1b. Why Barrett needs only ONE subtract here

The textbook Barrett in `reference/vector/modmul_reference.py` needs one subtract, and
`barrett_1pass` used two only because HAC 14.42's `r < 3q` is a worst case over all parameters. Ours
is far from worst. Writing `s = 2^56 − μ·q`, the quotient deficit is

```
d = z1/q + z·s/(q·2^56)  <  (2^27 + s)/q
```

so `q3 ≥ ⌊z/q⌋ − 1`, hence `r < 2q`, whenever **`2^27 + s < q`**. For `q = 2^28 − 57` we get
`μ = 2^28 + 57` and `s = 3249`, giving `0.50001` — comfortably inside. The bench **asserts** that
precondition and prints it; a 400k-case sweep agrees (max `r/q` = 1.4993) but is not relied on,
because sampling can disprove a bound and never establish one.

**`barrett_1sub` therefore carries a precondition a caller must check.** The general Barrett still
needs two subtracts, and on a modulus where `s` is large it would.

---

## 2. Why the multiply counts are identical

This surprises people, so it is worth spelling out. Both algorithms perform **three** multiplies:

| step | Montgomery | bits | Barrett | bits |
|---|---|---|---|---|
| 1 | `a · b` | 28×28 → 56 | `a · b` | 28×28 → 56 |
| 2 | `lo · q_inv` *(low 32 only)* | **32×32** | `q1 · μ` *(high part)* | 29×29 → 58 |
| 3 | `t · q` *(high 32)* | **32×28 → 60** | `q3 · q` *(low 32 only)* | 29×28 → 57 |

**Montgomery's widest product is 60 bits; Barrett's is 58.** Barrett does *not* carry a
higher-precision multiply — if anything it is narrower.

And it does not matter, because `mul16` works on **16-bit limbs**, so cost is `limbs(x) × limbs(y)`.
28, 29 and 32 bits are all **2 limbs**, so every one of these is a 2×2 grid = **4 partial products**,
falling to **3** when only the low 32 bits are needed. Each algorithm has exactly one such step:

```
Montgomery:  4 (a·b full) + 3 (lo·q_inv, low) + 4 (t·q, high)  = 11 mul16 = 22 sfpmul24
Barrett:     4 (a·b full) + 4 (q1·μ, high)    + 3 (q3·q, low)  = 11 mul16 = 22 sfpmul24
```

Mirrored, and confirmed in the compiled MATH TRISC: **22 `SFPMUL24` in all three kernels** — the
subtract count changes nothing about the multiplies.

### Why this kernel differs from the Python reference

`reference/vector/modmul_reference.py` gives the **textbook** Barrett, which multiplies the
full 56-bit `z` by `m` and takes the high half — flagged there as *"WIDE product, needs high half"*.
That is the clearer statement of the algorithm but a bad fit for 16-bit limbs: 56b × 29b is **4 × 2 =
eight** partial products where 29b × 29b is **four**. Following it literally gives Barrett
`4 + 8 + 3 = 15 mul16` against Montgomery's 11 — it would lose before it started. (Truncating it by
dropping `z0·m0` and `z0·m1`, whose bits all fall below bit 56, still leaves six.)

So the kernel uses **HAC 14.42**, pre-shifting `z` to 29 bits first (`q1 = z >> 27`), which costs only
shifts. The price is exactly **one extra conditional subtract**, since that bounds `r < 3q` instead of
`2q`. About three instructions bought for about eight saved. Both files now cross-reference each
other: the reference is the specification, the kernel is the limb-efficient realisation.

---

## 3. The operation counts

Per face-row, from `objdump` on the compiled MATH TRISC of each kernel:

| instruction | `montgomery` | `barrett_2sub` | **`barrett`** | what it is |
|---|---|---|---|---|
| `sfpmul24` | **22** | **22** | **22** | the multiplies — identical by construction (§2) |
| `sfpshft` | 24 | **29** | **29** | Barrett's `>>27`, `<<5`, `>>29`, `<<3` realignments |
| `sfpiadd` | **25** | 24 | 23 | |
| `sfploadi` | 10 | 11 | 11 | constants materialised into LREGs |
| `sfpand` | 8 | 9 | 9 | limb masking |
| `sfpmov` | 5 | 5 | 4 | |
| `sfpor` | 1 | **4** | **4** | |
| `sfpencc` | 1 | 2 | 1 | condition-code push/pop per `v_if` |
| `sfpsetcc` | 1 | 2 | 1 | |
| **TOTAL** | **97** | **108** | **104** | |

Reading it:

* **The multiplies are a wash.** 22 `SFPMUL24` in every kernel. Neither algorithm has a
  higher-precision multiply and neither issues more of them — §2 explains why that is forced by the
  16-bit limb decomposition rather than being a property of the algorithms.
* **Barrett costs +7 instructions in shifts and masks** (`sfpshft` +5, `sfpand` +1, `sfploadi` +1)
  because it must realign `z` down to `q1` and the product back down to `q3`. Montgomery's `R = 2^32`
  makes its equivalent alignments free — they are word boundaries, so they are just which half of a
  register you read.
* **Each conditional subtract costs ~4 instructions** — visible as `sfpencc` and `sfpsetcc` going
  2→1 and `sfpiadd` 24→23 between the two Barrett variants, total 108→104. That is the one
  difference in this table that shows up cleanly in the timing: ~7%.
* **`sfpor` 1 → 4** is the `v_if` bodies merging their results back.

**97 vs 104 instructions is a 7% gap that the measurement does not resolve** (§1: the ratio straddles
1.0 with ~5% run-to-run spread). The schedule evidently absorbs most of Barrett's extra work — the
kernel sustains ~1.34 instructions per cycle, so 7 extra instructions in a ~130-cycle face-row is
within what the pipeline can hide.

---

## 4. What this does and does not say

**Does say:** at equal engineering effort, on Blackhole, with both written as one SFPI pass over
`sfpmul24`, **Montgomery is 2.3% faster in pure computation** — small, but resolved. Its advantage is
that `R = 2^32` makes its realignments free (word boundaries), where Barrett must shift `z` down to
`q1` and the product down to `q3`. Keep Montgomery. Note also that the correction count matters more
than the choice of reduction: one conditional subtract is ~7%, three times the gap between the
algorithms.

**Does not say** which is better for a pipeline. They compute different functions:

* Montgomery's domain is free **only if the whole pipeline stays in it**. exp_Tg018's does, so
  conversions never appear in the profile — that is why the 5% is the whole story here.
* Barrett needs no domain and can mix with plain-domain data at no cost. In a pipeline that must
  convert in and out, Barrett would win despite being slower per multiply.

That is a pipeline question, not a kernel question, and this bench deliberately does not answer it.

**Untried:** truncated Barrett — dropping the low-low partial `q1L·mL` in step 2, whose contribution
to `(q1·μ) >> 29` is at most one unit of quotient error, absorbed by another conditional subtract.
That trades **one `mul16` (two `sfpmul24`) for ~3 instructions**, which on these numbers could close
or invert the 5% gap. Not built, not measured — a hypothesis.

---

## 5. Running it

```
. ./env.sh
MM2_CSV=docs/he/data/modmul002.csv ./build/exp_modmul002 [ITERS=5]
```

| env | meaning |
|---|---|
| `MM2_T` | tile counts to sweep (default `1,2,4,8,16,32,64,128`) |
| `MM2_CORE` | which core, as `x,y` (default `0,0`) |
| `MM2_CSV` | write the table |

Each kernel is checked against **its own** golden — Montgomery against `a·b·R⁻¹ mod q`, Barrett
against `a·b mod q` — so a kernel that is fast because it computes something else cannot pass.

**Note on timing:** editing either kernel changes its JIT hash, and the next run absorbs compilation
into its first timed points, which wrecks the fit (a negative slope is the tell). Run twice after any
edit and use the second.
