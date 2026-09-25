# RNS-BFV on Tenstorrent — kernel hierarchy mapped to our primitives

Companion to `bfv_rns_kernel_hierarchy.md`. That note gives a clean, **correct** generic
RNS-BFV composition (L1 RNS polys → L2 evaluator → L3 base conversion → L4 modulus management
→ L5 key switch). This note answers the practical question: **given the kernels we have actually
built on TT, is that the right hierarchy, and what do we still need?**

Short answer: the L1–L5 *composition* is right and we should keep it. But for our stack it is
missing (a) the **hardware layer below** its "lowest kernels", (b) the **28-bit-prime constraints**
that dominate RNS-base design, and (c) a hardware-specific **matmul-lowering** of base conversion.
And three of its assumed base kernels don't exist yet.

---

## 1. The generic doc's "lowest kernels" are a MIDDLE layer for us

`bfv_rns_kernel_hierarchy.md` assumes these as primitive black boxes:
`vec_modadd, vec_modsub, vec_modmul, vec_mul_const, ntt, intt`.

On TT, **`ntt` and `vec_modmul` are themselves composed** from lower hardware kernels we built.
Our true bottom is:

```
L0  Hardware primitives (what TT actually runs)
    - int8 radix-128 modular matmul     (matmul_compute*, fused_matmul_*)   ← matrix engine
    - decompose_limbs                   (int32 → 4 int8 limbs)              ← SFPU
    - recombine (+ bigK)                (Σ128^r + Montgomery mod q)         ← SFPU
    - SFPU element-wise: modmul (Montgomery, lazy), modadd                  ← SFPU

L0.5  Transforms, built from L0
    - ntt / intt  = decompose → matmul(W, folded) → recombine, twiddle modmul, (corner turn)
                    [ntt1024_dev, ntt32768_dev / _2factor]

L1+  exactly the generic doc (RNS polys → BFV eval → base conv → mod mgmt → key switch)
```

So the corrected mental model has **two extra layers below** where the generic doc starts. The
generic doc's hierarchy sits unchanged on top of L0.5 — it's the right shape, just not the bottom.

---

## 2. Build status — what we have vs. what the doc assumes

| doc primitive | our kernel | status |
|---|---|---|
| `vec_modmul` | `modmul_sfpi_hybrid.cpp` (Montgomery, lazy [0,2q)); `modmul_rns_sfpi.cpp` (≤16 primes/launch) | ✅ built, optimized |
| `vec_modadd` | `modadd_reduce.cpp` (sums n≤4 int32 tiles mod q) | ✅ (2-input is n=2) |
| `vec_modsub` | — | ❌ **need** (trivial: `a + (q−b)`, or a dedicated SFPU kernel) |
| `vec_mul_const` | — (but = `modmul(x, fill(mont(c)))`, the recombine's K14 trick) | ❌ **need wrapper** (cheap) |
| `ntt` | `ntt1024_dev.cpp`, `ntt32768_dev.cpp` / `_2factor` | ✅ built, optimized, multi-core |
| `intt` | — | ❌ **need** (same machinery: inverse roots/twiddles + final `×N⁻¹`; reuses matmul+recombine) |
| (matmul) | `fused_matmul_*`, `matmul_compute*` | ✅ — *not in the generic doc, but central for us (NTT + base conv)* |

So to stand up the generic L1–L5 we still need: **`vec_modsub`, `vec_mul_const`, `intt`** (all small,
all reuse existing machinery), then the L3–L5 schedules are pure composition (Python-style, as the
doc shows).

---

## 3. The constraint the generic doc ignores: 28-bit NTT-friendly primes

This is the biggest "our case" delta. Our modulus is not free — every RNS limb `q_i` must satisfy
**all** of:

1. **`q_i < 2²⁸`** — radix-128, 4 int8 limbs for the matmul lowering.
2. **NTT-friendly** — `N | (q_i − 1)` (and the four-step factor structure), so the per-limb NTT works.
3. **Near 2²⁸** — the recombine's lazy reduction is tuned for `q ≈ 2²⁸`; below that use the
   `recombine_bigK` / dual-Montgomery variant (we have `bigK`).

Standard RNS-BFV uses **~50–60-bit** primes. We are capped at **~28-bit**, with direct consequences
the schedule layer must account for:

- **~2× more limbs** for the same total modulus `Q` (each limb carries ~28 bits, not ~55). Every
  per-limb cost — NTTs, pointwise muls, base-conversion rows — scales with the limb count, so our
  `L = num_limbs` is roughly double a textbook implementation's.
- **Restricted prime pool**: moduli must be NTT-friendly *and* near 2²⁸. We need to enumerate enough
  such primes for the main base `Q` **and** the auxiliary base `P` (key switch / mod-down). This is a
  parameter-selection task to do up front (e.g. primes of the form `k·2^m + 1` near 2²⁸ with the
  right 2-adic valuation for `N`).
- **More NTTs per op** ⇒ NTT throughput (which we've optimized) is even more dominant than in a
  64-bit implementation. Good news: it's exactly the kernel we made fast.

This constraint belongs at the **top** of any BFV plan for our hardware — it sizes `L`, the prime
table, and therefore the cost of every L1–L5 operation.

---

## 4. Hardware-specific win: base conversion is a (small) matmul

The generic doc implements `fast_base_convert` as nested `vec_mul_const` + `vec_modadd` over the
limb dimension. But look at its structure:

```
dst[j] = Σ_i  (x_i · q̂_i⁻¹ mod q_i) · (q̂_i mod p_j)  mod p_j
```

Over the limb axis this is a **matrix–vector product**: an `L_dst × L_src` constant CRT matrix times
the `L_src`-vector of `α_i = x_i · q̂_i⁻¹`, reduced per output modulus `p_j`, **batched across all N
coefficients**. We have an int8 modular matmul (and `modmul_rns_sfpi` for the per-`p_j` multi-modulus
reduction). So **mod-up / mod-down / base extension can be lowered onto the matmul engine** instead of
`L_src × L_dst` separate scalar passes — the same engine we already use for the NTT. This is a
reuse opportunity the generic doc can't see because it doesn't know we have a matmul.

(Caveat: each output limb `j` carries a different modulus `p_j`, so it's the RNS / per-modulus matmul,
not a single-`q` one — `modmul_rns_sfpi`'s multi-prime support is the relevant primitive.)

Net: for us, **base conversion and key-switch are matmul-shaped**, not just vector-shaped. That may
flip where the cost sits versus the generic doc's "everything reduces to vector kernels" mental model.

---

## 5. Montgomery / lazy-domain bookkeeping (a real correctness layer)

Our `vec_modmul` is **Montgomery-domain** with **lazy [0, 2q)** outputs; the generic doc assumes
plain integers in `[0, q)`. So the schedule layer must track a domain bit per `RNSPoly`, in addition
to `is_ntt`:

- values live in Montgomery form between multiplies; conversions (`a → aR`) happen once at boundaries;
- `vec_mul_const` needs the constant pre-converted (`mont(c)`), like the recombine's `K14 = mont(2¹⁴)`;
- base conversion mixes `mul_const` and `add` — the adds must be in a consistent domain/range, and
  lazy `[0, 2q)` operands must be canonicalized where the CRT sum's bound requires it;
- NTT input/output domain must be pinned (is the NTT output in Montgomery form? — fix one convention
  end-to-end and assert it, or pointwise-multiply will be off by a factor of `R`).

Recommend extending the doc's `RNSPoly` with `is_montgomery` and asserting domain at every kernel
boundary, exactly as it already asserts `is_ntt`.

---

## 6. Recommended build order for our stack

1. **Parameter selection (do first).** Enumerate NTT-friendly ~28-bit primes near 2²⁸; pick the `Q`
   base (`L` limbs) and auxiliary `P` base; precompute all CRT / Montgomery constants. Sizes everything.
2. **Fill the L0/L0.5 gaps:** `vec_modsub`, `vec_mul_const` (wrappers over existing modmul/add), and
   **`intt`** (inverse-root NTT + `×N⁻¹`). These unblock all of L1.
3. **L1 RNS poly kernels** (generic doc, verbatim composition) over our kernels — add the
   `is_montgomery` domain bit.
4. **L2 evaluator** (`ct_add`, `ct_plain_mul`, `ct_mul_raw`) — pure scheduling; cheap to validate.
5. **L3 base conversion**, implemented as the **RNS matmul** (§4), with a vector-pass fallback for
   bring-up. Validate against a CPU CRT reference.
6. **L4 mod-up/down + mod-switch**, then **L5 relinearization / key switch** — the heaviest op,
   NTT- and base-conversion-bound, i.e. exactly our optimized kernels.

---

## 7. Verdict

- **Keep** `bfv_rns_kernel_hierarchy.md` as the algorithm-level spec — its L1–L5 decomposition and
  schedules are correct and we should implement them as written.
- **Add, for our hardware:** the L0/L0.5 layer (matmul + decompose + recombine + SFPU ops, with NTT
  built on them); the 28-bit / NTT-friendly / near-2²⁸ prime constraints and their `L`-doubling
  consequences; the matmul-lowering of base conversion; and the Montgomery/lazy domain tracking.
- **Build next:** `vec_modsub`, `vec_mul_const`, `intt` — then the schedules drop straight on top of
  what we already have (modmul, modadd, NTT, matmul, decompose, recombine).

The generic hierarchy is the right skeleton; this note is the hardware-aware body around it.
