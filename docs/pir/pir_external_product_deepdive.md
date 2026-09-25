# External product & RNS bit-width analysis (OnionPIR v2 on our TT stack)

Companion to `pir_special_kernels.md` (which lists the 7 PIR-specific kernels and dissects the
#3/#4 permutations). This doc is the deep-dive on **kernel #1, the external product** — the
"special mul" that replaces BFV ct×ct multiply — worked out concretely, plus the **RNS /
composite-NTT data flow** and the **bit-width conclusion** for our hardware.

**Headline result (why we care):** the entire OnionPIR pipeline can run with **every modular
reduction ≤ 28-bit**. No wide-prime (56/58-bit) modular reduction is ever required. This matches
exactly the modmul / NTT / modular-matmul primitives we've already verified on silicon.

---

## 1. The external product, concretely (single modulus view, `n=1024`)

Drop RNS for a moment; picture one modulus `q`. Ring `R_q = Z_q[x]/(x^n+1)`, `n=1024`.

**A BFV ciphertext** `d` = two length-1024 coefficient vectors mod `q`:
```
d0 = [a0, ..., a1023],   d1 = [b0, ..., b1023]      each coeff ∈ [0, q)
```

**An RGSW ciphertext** `C` is **not** two vectors — it is a **grid of `2ℓ × 2` polynomials**
(every cell a length-1024 vector). With `ℓ=6` (higher-dim value) → `2ℓ=12` rows × 2 cols =
**24 polynomials**. Each *row* is itself a BFV ciphertext, structured by the gadget vector
`g = (g0,...,g5)` (fixed powers of base `B`, e.g. `gk = B^(5−k)`):
```
row  0..5   (top ℓ):     C[k] = ( −aₖ·s + eₖ + m·gₖ ,   aₖ )       ← message in column 0
row  6..11  (bottom ℓ):  C[k] = ( −aₖ·s + eₖ ,           aₖ + m·gₖ ) ← message in column 1
```
`m` = RGSW plaintext (in PIR, the selection **bit** 0/1). All 24 polys are fixed key material,
stored pre-NTT'd.

### The operation = gadget-decompose, then vector×matrix product

```
d ⊡ C = G⁻¹(d) · C
```

**Step 1 — gadget-decompose `d0,d1` into small digits.** Write each coefficient in signed base
`B` with `ℓ` digits (digits in `[−B/2, B/2)`). Toy example (`B=16`, digits in `[−8,8)`):
```
1000  →  (4, −1, −8)      since 4·256 − 1·16 − 8·1 = 1000
```
Doing this for all 1024 coefficients gives `ℓ` digit-polynomials per input:
```
d0 → u0,...,u5    d1 → w0,...,w5     G⁻¹(d) = [u0..u5, w0..w5]  (2ℓ small polys)
```
Reconstruction property: `Σₖ uk·gk = d0`, `Σₖ wk·gk = d1`.

**Step 2 — multiply the `(1×2ℓ)` digit vector by the `(2ℓ×2)` RGSW grid, accumulate down columns:**
```
out0 = u0⊛C[0][0] + ... + u5⊛C[5][0] + w0⊛C[6][0] + ... + w5⊛C[11][0]
out1 = u0⊛C[0][1] + ... + u5⊛C[5][1] + w0⊛C[6][1] + ... + w5⊛C[11][1]
```
`⊛` = negacyclic poly-mult mod `(x^1024+1)` (done via NTT); `+` = coeff-wise add mod `q`.
Both `⊛` and `+` are **length-1024 vector operations**. That's `4ℓ=24` negacyclic multiplies +
adds → output `(out0, out1)` = a BFV ciphertext.

### Why it works / why noise is additive
`C = Z + m·G`, so `G⁻¹(d)·C = G⁻¹(d)·Z + m·(G⁻¹(d)·G) = G⁻¹(d)·Z + m·d`.
- `m·d` → decrypts to `m·m_d` (the wanted product).
- `G⁻¹(d)·Z` → a combination of encryptions-of-0 weighted by **tiny digits** (`|·|<B/2`), so noise
  stays small and **adds** rather than multiplies. This is the entire reason OnionPIR uses the
  external product instead of BFV ct×ct multiply (which multiplies noise and needs `⌊t/Q⌉`+relin).

### In PIR: the 1-of-2 multiplexer (`b·(y−x)+x`)
Higher dimensions fold two halves `x,y` (BFV cts) with selection bit `b` as RGSW:
```
DBdiff = y − x
out    = ExternalProduct(C_RGSW, DBdiff) + x     # b=1→y, b=0→x
```
One external product per selection, `log₂(N/N₀)` times.

---

## 2. Coefficient sizes — don't confuse "small digit" with "small operand"

| thing | where | coefficient size |
|---|---|---|
| RGSW cell `C[i][j]` | the 24 key polys | **full width, `[0,q)` ≈ 56-58 bit** |
| BFV ct `d0,d1` | the input | full width, `[0,q)` |
| gadget digits `uk,wk` = `G⁻¹(d)` | after decompose | **small, `[−B/2,B/2)` ≈ ~10 bit** (ℓ=6) |

Only the **decomposed digits of the BFV input** are small. RGSW is full-size mod `q` (each row is a
real BFV encryption; `aₖ` is uniform in `[0,q)`). Storage ≈ `2ℓ·2·n·⌈log q⌉` — for `ℓ=6, n=2048,
q≈56b`: ~**393 KB** per RGSW (paper's seed trick halves it).

**Key subtlety:** the digit being ~10-bit helps only in the **coefficient domain** (it bounds the
noise term `Σ digitₖ·eₖ`). Once you `NTT(uk)`, the transform spreads it into 1024 **full-width**
residues, so the pointwise multiply in NTT domain is **full×full**, not 10×full. Decomposition buys
noise control, **not** a narrower multiplier.

---

## 3. Field vs ring

- All arithmetic is mod `q`, in the ring `R_q = Z_q[x]/(x^n+1)`.
- `q = q₁·q₂` is **composite → `Z_q` is a RING, not a field** (`q₁,q₂` are zero divisors). There is
  **no 56-bit prime**, by design.
- The NTT needs a `2n`-th root of unity → lives in the **prime fields** `GF(q₁)`, `GF(q₂)` (each
  `qᵢ` prime, `≡ 1 mod 2n`). So NTT + pointwise modmul run in the prime fields (RNS).
- Gadget-decompose needs the full composite value → works on the single `Z_q` (56-bit) integer.

---

## 4. Does the paper use RNS? — yes, selectively (composite NTT)

Paper §3.4: ciphertext modulus is composite `q=q₁q₂` (~58-bit, two ~29-bit primes) so multiplies
are fast `32×32→64` instead of `64×64→128`. But it refuses to run the *whole* algorithm in RNS,
because gadget decomposition would then need CRT/inverse-CRT around every `G⁻¹(·)`. The
**composite-NTT** technique (Faster-Spiral [21,23], our kernel #7) resolves this:
- **Initial dimension** (big dot-product/matmul) → **RNS/CRT**, per-channel 28-bit.
- **Higher dimensions** (gadget + external product) → treat `q` as a **single modulus** so gadget
  decompose avoids per-fold CRT churn.

Our stack: int8 matmul caps primes at 28-bit → pick two 28-bit primes `≡1 mod 2n`
(e.g. `167772161 = 5·2²⁵+1` + one more) → `q ≈ 56-bit` (vs paper's 58). Cost of 56 vs 58: ~2 bits
less noise headroom (`log q − log t`); retune `t`/`ℓ`/dims. Nothing forces a >28-bit reduction.

---

## 5. Full external-product pipeline, tracking representation (`n=1024, ℓ=6, q≈56b`)

The switch between "RNS 28-bit channels" and "single 56-bit value" happens **only at gadget
decompose**. An external product is a dance across that boundary:

| step | what | representation | arithmetic width |
|---|---|---|---|
| in | ct `d=(d0,d1)` arrives | RNS, NTT domain, per channel | — |
| 1. INTT | leave NTT domain (per channel) | RNS, coeff domain | 28-bit |
| 2. CRT-combine | `(x mod q₁, x mod q₂) → x` | **single 56-bit** coeffs | reduction is mod q₂ (28-bit) |
| 3. gadget-decompose | `X = Σ digitₖ·gₖ`, digits `[−B/2,B/2)` | 2ℓ **~10-bit** digit polys | 56-bit *integer* ops (shift/mask), **no mod-q** |
| 4. split to RNS | reduce digits mod q₁,q₂ (trivial, digit<28b) | RNS, coeff domain | — |
| 5. NTT digits | forward NTT per channel | RNS, NTT domain | 28-bit |
| 6. pointwise MAC vs RGSW | `Σ digitₖ ⊙ C[·]` down 2 cols | RNS, NTT domain | **28×28→56 mod 28-bit prime** |
| 7. INTT | back to coeffs, per channel | RNS, coeff domain | 28-bit |
| out | `d'=(out0,out1)` → next fold | RNS | — |

RGSW is **pre-stored** in step-6 form (per channel, NTT domain) and never moves.

**"How does it convert to digit decompose?"** → steps 1–3: INTT, CRT-combine the two 28-bit
residues into the single 56-bit coefficient, base-B signed-decompose *that*. You cannot decompose
`(x mod q₁, x mod q₂)` directly — the digit pattern depends on the magnitude of `x`.

**"How does it multiply the RGSW matrix?"** → steps 4–7: small digits go back into the two channels
(free — a 10-bit number is `< q₁,q₂`), NTT per channel, then the `2ℓ×2` product is just per-channel
pointwise modmul-accumulate against the pre-NTT'd RGSW rows, INTT back.

### Step 6 is nothing special
In NTT domain the convolution `⊛` has collapsed to elementwise `⊙`, so step 6 is exactly our
verified `vector_modmul` + modadd: per channel, over 1024 lanes, `28×28→56 modmul` + modadd. The
`2ℓ×2` "matmul" is tiny (`K=12, N=2`) → genuinely elementwise MAC, **no matmul engine** (that's the
*initial dimension*, a different part of the protocol). Accumulation can be **lazy** (defer
reduction, sum 12 terms, reduce once — the paper's delayed-reduction trick).

### What composite-NTT optimizes
Naive RNS does steps 1–2 + 5 (INTT+CRT-combine, then re-split+NTT) around **every** external
product → many whole-polynomial CRT round-trips over `log₂(N/N₀)` folds. Because both primes are
`≡1 mod 2n`, a `2n`-th root of unity also exists in `Z_q`, so the composite NTT runs *logically over
the single 56-bit `q`* (keeping values in the form gadget-decompose wants) while each 56-bit twiddle
modmul is still executed as two 28-bit channel mults + combine. Net: values stay in "mod-q 56-bit"
representation across the fold chain → **no explicit CRT round-trip per fold**; the CRT cost is paid
per-multiply (cheap, fused) instead of per-polynomial-per-fold.

---

## 6. Bit-width conclusion (the thing we care about for hardware)

**Every modular reduction in the entire pipeline is ≤ 28-bit** — against a 28-bit prime `q₁` or `q₂`:

| operation | reduction | width |
|---|---|---|
| initial-dimension matmul | mod qᵢ | 28-bit ✓ |
| NTT / INTT twiddles | mod qᵢ | 28-bit ✓ |
| external-product pointwise MAC (step 6) | mod qᵢ | 28-bit ✓ |
| ct add/sub (`y−x`), accumulation | mod qᵢ | 28-bit ✓ |

The only 56-bit arithmetic anywhere is **not** modular reduction:
1. **gadget-decompose** — 56-bit integer shifts/masks/subtracts (kernel #5), no mod-q.
2. **CRT-combine** — widening multiply-add producing a 56-bit int; its only reduction is mod q₂ (28-bit).
3. **final mod-switch** — runs **once** on one ciphertext at the end; output modulus `q′≈22-bit`.

`q` is composite, so there is **no 56-bit prime** to reduce against in the first place — this is
the deliberate point of the composite-modulus + composite-NTT design.

> **All modular reduction ≤ 28-bit is sufficient to run the entire OnionPIR pipeline** — matching
> the modmul / NTT / modular-matmul kernels already verified on our Wormhole silicon.
