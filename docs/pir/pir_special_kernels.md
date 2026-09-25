# PIR-specific kernels: what OnionPIR v2 needs beyond our BFV HE scheme

Scope: the operations OnionPIR v2 (ePrint 2025/1142) requires that our existing **BFV
multiply/relin** path does **not** use. Everything else PIR touches — ciphertext add,
plaintext×ct multiply (the initial-dimension matmul), mod-switch, NTT/INTT, modmul, modadd —
we already built and verified on silicon (see `history/onionpir_build_plan.md` §0 and the `device/` tree).

The headline: PIR **skips** the hardest BFV op (ct×ct multiply / `⌊t/Q⌉` scaling / hybrid relin)
entirely, and replaces it with the **external product** (additive noise). The new work is one
core arithmetic primitive plus a small family of data-movement ops and compositions.

---

## The list (1–7)

### Genuinely new — no BFV analog at all

**1. External product** `RGSW(2ℓ×2) ⊡ BFV(2) → BFV(2)`
The defining PIR op. **Replaces** BFV ct×ct multiplication. BFV's multiply blows noise up
multiplicatively and needs the `⌊t/Q⌉` scale + relinearization; the external product's noise is
*additive* (in PIR the RGSW plaintext is a single bit, so noise ≈ `O(nB²·Err(C) + Err(d))`).
Structure: gadget-decompose the 2 BFV polys into `2ℓ` digits → NTT each → pointwise
mul-accumulate against the `2ℓ×2` RGSW matrix (kept in NTT) → 2 output polys.
Cost ≈ `2ℓ` digit-NTTs + `4ℓ` pointwise-mul-acc.

**2. RGSW ciphertext type**
A `2ℓ×2` matrix `C = Z + m·G`, where each row of `Z` is a BFV encryption of 0 and `G = I₂ ⊗ g`
is the gadget matrix (`g` = base-`B` gadget vector, length `ℓ`). Note `m` is **not** scaled by
`∆`. No such object exists in BFV. Needs its own data type + an encrypt reference.
**#2 is exactly the left-operand type consumed by #1** — the external product is the *only*
consumer of RGSW in the whole scheme. RGSW appears in two roles, both fed to external products:
  - `C_RGSW[i]` — the per-higher-dimension **selection-bit** RGSW (hypercube fold, Alg. 4).
  - `RGSW(s)` — an RGSW encryption of the client **secret key**, used in `QueryUnpack` (Alg. 3)
    to derive the bottom `ℓ` rows of each dimension's RGSW from the packed top `ℓ` rows.

**3. Galois automorphism `x → xʲ`** (the permute inside `Subs`)
Coefficient **index permutation** `i ↦ i·j mod 2n` with a negacyclic sign flip on wrap.
See the decomposition section below — the full `Subs(·,j)` is permute + negate + **key-switch**.

**4. Monomial multiply `x^{±k}`**
A negacyclic cyclic-shift of coefficients (with sign). Used in the `ExpandBFV` tree.
Pure data movement — see below.

### New, but a variant of a primitive we already have

**5. `gadget_decompose_signed(B, ℓ)`** — *the one core new arithmetic kernel*
We have `decompose_limbs` (fixed base-128, **unsigned**, exactly 4 digits, for the int8 matmul).
The gadget version needs **signed** digits in `[−B/2, B/2)` (smaller L∞ → lower external-product
noise), **arbitrary base `B`**, and **configurable `ℓ`**. This kernel — plus our existing
modmul+modadd for the mul-accumulate — is what *builds* the external product, BV key-switch,
and `Subs`. It is the main thing to write. Two parameter sets in practice: larger `ℓ` for query
unpacking, smaller `ℓ` for the remaining dimensions (Spiral's two-set trick).

**6. BV-style key-switch**
We built the **hybrid** (special-modulus, mod-up/down) key-switch. PIR uses **BV-style** instead:
gadget-decompose one component → multiply by the key-switch-key matrix → accumulate. No mod-up/down.
Simpler, and it reuses #5. Different kernel from the hybrid one we have.

**7. composite-NTT bridge**
RNS (28-bit limbs, great for the init-dim matmul) ↔ single-modulus view of the ~58-bit `q`
(so gadget decomposition avoids per-digit CRT). No BFV analog; the main TT-specific design point.

*(Also NTT at `n = 2048` — a new parameter set for machinery we have (NTT-1024 / NTT-32768),
not a new kernel.)*

### The shared shape

Items **1, 5, 6, and the key-switch inside `Subs`** all reduce to one shape:

```
gadget-decompose (signed, base-B, ℓ digits) → NTT each digit
    → pointwise mul-accumulate against key/RGSW material (in NTT) → sum
```

So the real new-kernel count is small: build **`gadget_decompose_signed`** once (the only new
*arithmetic*), add the **RGSW type**, the **Galois permute**, and the **monomial shift** (both
data-movement) — external product, BV key-switch, and `Subs` fall out as compositions over
kernels we already have (modmul, modadd, NTT).

---

## Discussion result: decomposing #3 and #4

#3 and #4 are **not** monolithic new ops. They break into: **shaping**, **negation**, and (for #3
only) **key-switch**.

|                              | shaping (permute)        | negation (mod q)      | key-switch |
|------------------------------|:------------------------:|:---------------------:|:----------:|
| **#4 monomial `x^{±k}`**     | ✅ cyclic coeff shift    | ✅ on wrapped coeffs  | ❌         |
| **#3 Galois `Subs(·,j)`**    | ✅ index permute `i↦i·j mod 2n` | ✅ on wrapped coeffs | ✅ |

- **#4 = shaping + negation.** Stays under the same key → **no** key-switch. Pure data movement,
  no arithmetic tax.
- **#3 = shaping + negation + key-switch.** The permute+negate *is* the automorphism; the
  key-switch is the unavoidable tax for the key getting dragged along by the automorphism.

### Why the negacyclic negation

In `Z[x]/(xⁿ+1)`, `xⁿ = −1`. Whenever a coefficient's exponent wraps past `n`, it picks up `−1`.
So both ops are **permutation + conditional negate-mod-q** (`v → q−v`) on the wrapped coefficients
— not literally "same values relocated."

### Why there is a key-switch in #3 (but not #4)

`Subs(·,j)` applies the ring map `σⱼ: x ↦ xʲ` to **every** polynomial. Apply it to the BFV
decryption identity `c₀ + c₁·s = ∆m + e (mod q)`:

```
σⱼ(c₀) + σⱼ(c₁)·σⱼ(s)  =  ∆·σⱼ(m) + σⱼ(e)   (mod q)
```

The transformed pair `(σⱼ(c₀), σⱼ(c₁))` **does** encrypt the message we want, `σⱼ(m) = m(xʲ)`
(the coefficient permutation) — **but the identity only closes with `σⱼ(s) = s(xʲ)` in the key
slot, not `s`.** So it is a valid ciphertext of the right message under the **wrong key** `s(xʲ)`.

Everything else in the protocol (and the client's decryption) is under `s(x)`; you cannot add,
external-product, or decrypt across mismatched keys. Converting a ciphertext from key `s'` back to
`s` without changing the message **is** key-switching. You can't apply `σⱼ` to only the message and
spare the key — `σⱼ` acts on polynomials and `s` sits inside the decryption inner product; the key
transforms whether you like it or not, and you can't hand the client a new key per automorphism.

The client instead provides, once at setup, a **Galois / key-switch key** = a gadget-form
encryption of `s(xʲ)` under `s(x)`. Key-switching then computes

```
(σⱼ(c₀), 0)  +  ⟨ G⁻¹(σⱼ(c₁)),  KSK ⟩
```

— gadget-decompose the offending `σⱼ(c₁)`, mul-accumulate against the KSK rows → a fresh `(d₀,d₁)`
with `d₀ + d₁·s ≈ ∆·σⱼ(m) + e'`: same message, now under `s(x)`. The gadget decomposition keeps
the added noise small.

Concretely in query expansion `j = n+1`, using `x^{i(n+1)} = (xⁿ)ⁱ·xⁱ = (−1)ⁱxⁱ`, this is what
lets `c ± Subs(c, n+1)` isolate even/odd coefficients — but only *after* the key-switch drops the
result back onto `s(x)` so `c` and `Subs(c,n+1)` share a key and can be added.

### Two refinements to the mental model

1. **Negation isn't really its own kernel** — it's a conditional negate-mod-q on the wrapped
   coefficients, and it fuses into the shaping (write the moved coefficient with a flipped sign).
   So #4 is *one* data-movement kernel carrying the sign; the automorphism part of #3 likewise.
2. **Key-switch isn't primitive** — it decomposes into #5 (signed gadget-decompose) + #6
   (mul-accumulate) against the Galois key. "Key-switch" on the #3 row is really "the gadget
   primitive again."

**Net:** the genuinely new *primitive* hiding in #3/#4 is just the **shape+sign data-movement
kernel** (permute with negacyclic sign). Everything else in #3 is the gadget-decompose /
mul-accumulate machinery already being built for the external product (#1).
