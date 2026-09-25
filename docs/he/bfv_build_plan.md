# Plan: building BFV evaluation on top of vector-mod + NTT (the hard parts)

**Assumption.** We have, over 28-bit NTT-friendly primes, the basic kernels:
`vec_modadd/sub/mul/mul_const`, `ntt/intt`, plus RNS-base bookkeeping. Everything in the *easy*
layer — `rns_add/sub`, `rns_poly_mul`, `ct_add`, `ct_plain_add/mul`, the **tensor** part of
`ct_mul_raw` — is pure composition over those kernels and is specified in
`bfv_rns_kernel_hierarchy.md`. **This plan covers only the operations that are NOT expressible as
plain vector-mod + NTT**, which is where BFV actually gets hard:

1. **Fast base conversion (FBC)** — the one primitive under everything below.
2. **BFV multiplication's `⌊(t/Q)·e⌉` scaling** — the heaviest, and *missing* from the generic doc.
3. **Key-switch / relinearization** and **mod-up / mod-down**.
4. **Modulus switching** (cross-modulus reduce).

The recurring theme: each is a CRT **linear combination (vector ops) + a rounding/overflow
correction (NOT a modular vector op)**. The whole plan is about implementing that correction.

---

## 0. Why these aren't "just vector ops + NTT"

Converting a value `x` from RNS base `Q = {q_0,…,q_{L-1}}` to another base is a CRT reconstruction.
The CRT gives, with `q̂_i = Q/q_i` and `y_i = [x_i · q̂_i⁻¹]_{q_i} ∈ [0, q_i)`:

```
x = ( Σ_i y_i · q̂_i )  −  ε·Q ,      ε = ⌊ Σ_i y_i / q_i ⌋ ,   ε ∈ {0,…,L−1}
```

So `x mod p_j = ( Σ_i y_i·(q̂_i mod p_j) − ε·(Q mod p_j) ) mod p_j`.

- The `Σ_i y_i·(q̂_i mod p_j) mod p_j` part is **`vec_mul_const` + `vec_modadd`** — easy.
- The `ε` term is **a floor of a sum of fractions** — *not* a modular vector op. Dropping it gives
  the **approximate FBC**, which returns `(x + ε·Q) mod p_j` (error up to `(L−1)·Q`).

Everything hard in BFV evaluation is "how do I handle `ε`": ignore it (when noise absorbs it),
compute it in float, or compute it exactly with an extra modulus. That choice is the plan.

---

## 1. Fast base conversion — the core kernel (build first)

### 1a. Approximate FBC (vector-only)

```
FBC_approx(x in Q → P):
    y_i      = vec_mul_const(x_i, q̂_i⁻¹ mod q_i, q_i)          # one per source limb
    dst_j    = Σ_i vec_mul_const(y_i, q̂_i mod p_j, p_j)  mod p_j   # accumulate per dst limb
    return (x + ε·Q) mod P      # ε silently included
```

Pure `vec_mul_const`/`vec_modadd`. Cost = `L_src × L_dst` constant-mults + adds per coefficient
(this is the **matmul-shaped** op — can be lowered onto our int8 RNS matmul; see §7). Precompute
`q̂_i⁻¹ mod q_i` and `q̂_i mod p_j` on the host.

**Where the `ε·Q` error is acceptable:** key-switch **mod-up** and **mod-down**. There the error is a
bounded additive term that folds into the key-switch noise budget — this is the standard "approximate
(hybrid) key switch". So **relin / key-switch / mod-up / mod-down can use FBC_approx and are then
fully vector-mod + NTT.** Just bound the noise (≤ `L·Q`-scaled, divided out by `P`).

### 1b. Exact FBC (needs the `ε` correction)

Required wherever the result must be exact — chiefly the BFV multiplication scaling (§2). Two ways:

- **Float path (HPS):** compute `ε = round(Σ_i y_i / q_i)` in fp, then subtract `ε·(Q mod p_j)`.
  - *TT caveat:* the SFPU is **fp32** (24-bit mantissa) but `q_i ≈ 2²⁸`, so `y_i/q_i` loses ~4 low
    bits; summed over `L` terms the floor can be off when the fractional sum is within `~L·2⁻²⁴` of
    an integer. Tolerable for noise-absorbed ops, **risky for exact scaling** → verify numerically or
    use the integer path.
- **Integer path (Shenoy–Kumaresan), recommended on TT:** carry one extra "sentinel" modulus `m_sk`
  (≈28-bit, coprime to `Q`/`P`, *not* NTT-friendly — never transformed). Over-extend to `P ∪ {m_sk}`,
  then recover `ε` **exactly** from the `m_sk` channel by integer arithmetic, and subtract `ε·Q`.
  All modular vector ops — **no floating point**, which avoids the fp32 precision problem entirely,
  at the cost of one extra modulus and a few more `vec_mul_const`/`add`.

**Recommendation:** build `FBC_approx` first (unblocks key-switch). Build `FBC_exact` via the
**integer SK path** for the multiplication scaling. (Exact formulas for the SK `ε`-recovery: follow
BEHZ; pin them against a CPU big-int reference.)

---

## 2. BFV multiplication — the `⌊(t/Q)·e⌉` scaling (hardest; missing from the generic doc)

The generic doc's `ct_mul` = tensor → relinearize. **That is incomplete** — BFV multiply *must*
divide the tensor product by `Q` and multiply by `t` with rounding, or the message comes out scaled
by an extra `Q/t`. Pipeline (BEHZ-style):

```
ct_mul(a, b):
  1. extend a, b to base  Q ∪ Bsk     (Bsk = auxiliary base B ∪ {m_sk})      ← FBC_approx ×(2·2 polys)
  2. tensor:  e = a ⊗ b   in Q∪Bsk    (e0,e1,e2)                              ← rns_poly_mul (NTT)
  3. SCALE:   ẽ = ⌊ (t/Q)·e ⌉  mod Bsk   — the "fast RNS floor"               ← FBC_exact + round
  4. convert ẽ from Bsk back to Q                                            ← FBC_exact
  5. relinearize ẽ2  (§3)
```

Step 3 is the crux and the genuinely-not-vector-op part: a **fast RNS floor/round**, built from
`FBC_exact` (using `m_sk` for the SK correction) plus the explicit `t`-multiply and `⌊·⌉`. The
auxiliary base `Bsk` gives headroom so the tensor products don't overflow before scaling, and the
`m_sk` sentinel makes the floor exact.

**Build strategy:** stand up a **CPU big-int reference** of the whole BFV mult first (encode →
tensor → scale → relin → decode) and validate noise growth; then port step-by-step, replacing each
stage with the RNS/TT kernel and checking `mismatches = 0` against the reference at each step. Do
**not** try to land step 3 directly on device — it has the most subtle constants.

---

## 3. Key-switch / relinearization (vector+NTT + approximate FBC)

The schedule is correct in the generic doc and is **fully vector-mod + NTT** once FBC exists:

```
key_switch(c2):
  for each RNS decomposition group g:        # group = limb slice — free, no compute
    c2_g          = extract_group(c2, g)
    c2_g_qp       = mod_up_group_to_QP(c2_g) = FBC_approx(group → QP)          ← §1a
    c2_g_qp_ntt   = rns_ntt(c2_g_qp)                                          ← NTT
    acc0 += c2_g_qp_ntt * evk0_g ;  acc1 += c2_g_qp_ntt * evk1_g              ← pointwise modmul + add
  acc0,acc1 = rns_intt(acc0,acc1)                                            ← INTT
  k0,k1     = mod_down_QP_to_Q(acc0,acc1)                                    ← §4
relinearize: (c0,c1,c2) → (c0+k0, c1+k1)
```

- **Decomposition = limb slicing** (the doc's key point — no integer digit decomposition, no compute).
- The only hard kernels are the two FBCs (mod-up, mod-down), both **approximate** here → error is
  key-switch noise. Keep `evk` in NTT form (precompute) to skip its transforms.
- This is the right target *after* `FBC_approx` + `intt` exist; it's the workhorse and exercises
  everything.

---

## 4. Mod-up / mod-down / mod-switch

| op | recipe | new kernel needed |
|---|---|---|
| **mod-up Q→QP** | `FBC_approx(Q→P)` then concat limbs | — (FBC) |
| **mod-down QP→Q** | `x_P→Q = FBC_approx(P→Q)`; `y_i = (x_i − x_{P→Q,i})·P⁻¹ mod q_i` | — (FBC + vec_modsub + vec_mul_const) |
| **mod-switch (drop last)** | `y_i = (x_i − [x_L]_{q_i})·q_L⁻¹ mod q_i` | **cross-modulus reduce** `[x_L]_{q_i}` |

**Cross-modulus reduce** (`reduce_small_int(x_L, q_i)`): `x_L` is a residue `< q_L < 2²⁸` that must be
re-reduced mod `q_i`. It's an exact single modular reduction of a known-small integer per element —
build it from the Barrett/Montgomery path we already have in the recombine. It's *not* `vec_modmul`,
but it's small and exact (no `ε`). This is the one extra kernel mod-switch needs.

---

## 5. The complete "new kernels beyond vector-mod + NTT" list

Everything BFV needs that the basic kernel set does **not** already give:

| new kernel | used by | nature |
|---|---|---|
| `FBC_approx` (CRT linear combo) | mod-up, mod-down, key-switch, mult base-extend | `vec_mul_const`+`add` (matmul-shaped, §7) |
| **`ε` correction** (SK integer, recommended; or fp32) | `FBC_exact` | extra modulus `m_sk` + integer ops (**or** SFPU fp32) |
| **RNS round** `⌊(t/Q)·e⌉` | BFV multiply scaling | `FBC_exact` + t-mul + floor |
| `reduce_small_int(x, q_i)` | mod-switch | single modular reduce (Barrett/Montgomery) |

So the answer to "is BFV doable with vector ops + NTT?": **the schedules are, and key-switch /
mod-up/down work with approximate FBC.** But you must add **one correction kernel** (integer-SK or
fp32) and the **RNS-round** for BFV multiply — those are the pieces that fall outside the
"modular vector op + NTT" set.

---

## 6. TT-specific notes

- **28-bit primes ⇒ ~2× the limbs** of a textbook (50–60-bit) build for the same `Q`. Every per-limb
  cost (NTTs, FBC rows, key-switch) scales with `L`, so `L` is large here — NTT throughput dominates
  (the kernel we've optimized). Budget for it.
- **SFPU is fp32-native**, so the float `ε` path is *available* — but 24-bit mantissa vs 28-bit
  primes makes exact rounding marginal. **Prefer the integer SK path** for the exact scaling; keep
  fp32 as a fast approximate option for noise-tolerant conversions if validation allows.
- **Montgomery / lazy domain:** our `vec_modmul` is Montgomery with lazy `[0,2q)`. FBC mixes
  `mul_const` and `add`; track an `is_montgomery` bit and canonicalize where the CRT-sum bound
  requires it. Pre-convert all CRT constants to Montgomery (`mont(q̂_i mod p_j)`), like the recombine's
  `K14`.
- **FBC as matmul (§7):** because it's matmul-shaped, the int8 RNS matmul (`modmul_rns_sfpi` /
  fused matmul) can do base conversion — reuse, not a new engine.

---

## 7. FBC = a small RNS matmul (reuse our matmul)

`dst_j = Σ_i y_i · C[i][j] mod p_j` with `C[i][j] = q̂_i mod p_j` is a `[L_dst × L_src]` constant
matrix times the `L_src` vector `y`, **per output modulus `p_j`**, batched over all `N` coefficients.
That is exactly an RNS modular matmul (each output row carries its own modulus `p_j`) — so mod-up,
mod-down, and the multiplication base-extensions can be **lowered onto the matmul engine** instead of
`L_src×L_dst` scalar passes. Relevant once `L` is large (which, per §6, it is for us).

---

## 8. Build order (each milestone validated vs a CPU big-int reference)

1. **Parameters:** enumerate NTT-friendly ~28-bit primes near 2²⁸ for `Q`, auxiliary `P`/`B`, plus a
   non-NTT sentinel `m_sk`. Precompute all CRT + Montgomery constants. (Sizes everything.)
2. **`intt`, `vec_modsub`, `vec_mul_const`** — small, reuse existing machinery; unblock L1.
3. **`FBC_approx`** + CPU CRT check. Then **mod-up / mod-down** on top.
4. **Key-switch / relinearization** (approximate FBC + NTT) — validate end-to-end vs reference;
   this is the workhorse and the first "real" BFV-hard op fully working.
5. **`reduce_small_int`** → **mod-switch / level drop**.
6. **`FBC_exact`** (integer SK with `m_sk`) + the **RNS round** → **full BFV multiply** with the
   `t/Q` scaling. Land last; validate noise growth and `0`-mismatch decryption over a multiply chain.
7. Optimize: FBC-as-matmul (§7), keep evk + hot polys in NTT form, lazy reduction.

---

## 9. References

- **BEHZ** (Bajard–Eynard–Hasan–Zucca) — exact RNS-BFV with the `m_sk` sentinel and fast RNS floor.
- **HPS** (Halevi–Polyakov–Shoup) — float-based RNS scaling variant.
- **Lattigo**, **Microsoft SEAL** — production RNS-BFV (`BConv`, `DivRoundByQ`, hybrid key-switch).
- Shenoy–Kumaresan — the exact base-extension correction via an extra modulus.

---

## 10. One-line summary

The BFV *schedules* (relin, key-switch, mod-up/down) are vector-mod + NTT **once you have an FBC**,
and the **approximate FBC** suffices for all the key-switch-family ops (error → noise). The parts
that are genuinely *not* vector-op + NTT are the **`ε` correction** and the **`⌊t·e/Q⌉` scaling** in
BFV multiply — both reduce to "compute the CRT overflow exactly," which we do with an integer
**Shenoy–Kumaresan extra modulus** (preferred on TT over the fp32 float path). Build FBC →
key-switch → mod-switch → exact-scaling-multiply, validating each against a CPU big-int reference.
