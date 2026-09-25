# exp_Tb001b — a correct BFV ct_mul at the exp_Tb001 ring, verified by decryption

`host/experiments/exp_tb001b.cpp` · shares `kernels/experiments/exp_tb001/` · `./build/exp_tb001b [seed]`

## Why this exists

[`exp_tb001.md`](exp_tb001.md) measures a multiply that is a faithful **cost model** of CROSS Set-B:
the right ring, the right limb counts, the right kernel sequence, bit-exact against a CPU oracle. It
is not a semantically correct BFV multiply, and three things stop it being one:

| | exp_Tb001 | BFV requires |
|---|---|---|
| rescale | drop one prime (CKKS-shaped) | `⌊t·e/Q⌉` — divide by the **whole** modulus |
| evaluation key | random | generated from the secret key, encrypting `P·s²` |
| correctness test | device == CPU oracle | `decrypt(ct_mul) == m₁·m₂` in `R_t` |

The third row is the important one, and the distinction is easy to lose. "Device equals oracle" says
**the device computed what the host meant**. It says nothing about whether what the host meant is a
ciphertext of the product — both sides can agree perfectly on arithmetic that decrypts to noise. Only
an end-to-end decrypt test can tell you the scheme is right, and exp_Tb001 cannot run one, because it
has no secret key.

exp_Tb001b closes exactly that gap and nothing else. Same ring, same kernel tree, same measurement
stack; a real key, a real encryption, BFV's own rescale, and a test that decrypts.

## Parameters

| | | |
|---|---|---|
| `N` | 8192 | the CROSS Set-B ring, same as exp_Tb001 |
| `Q` | 8 × 28-bit = **224-bit** | ciphertext modulus |
| `P` | 4 × 28-bit | keyswitch base (`dnum=2`, as exp_Tb001) |
| `S` | 9 × 28-bit | **auxiliary base — exp_Tb001 has no analogue** |
| `t` | 12289 | plaintext modulus, as `lib/he` |

22 primes of the form `k·2N+1` below 2²⁸; 40 exist, so the base is not tight.

`S` is the parameter that marks this as BFV. The tensor product of two ciphertexts has coefficients up
to `N·Q²` ≈ 2⁴⁶¹ and **must not be reduced mod Q** before the rescale divides by Q — so it is computed
in the extended base `Q|S`, 17 limbs, ≈ 2⁴⁷⁶. A drop-prime rescale divides by one limb and needs no
such base at all, which is why exp_Tb001 never pays for it.

## The scheme

Following `lib/he/he_api.cpp`, which `examples/he_demo` validates at N=1024, LQ=3 by
`decrypt(6 ⊗ 7) == 42`.

```
encrypt(m)   c0 = Δ·m + e − a·s ,  c1 = a          Δ = ⌊Q/t⌋
decrypt(ct)  m = round(t/Q · [c0 + c1·s]_Q) mod t
multiply     tensor over ℤ in Q|S  →  ⌊t·e/Q⌉ back into Q  →  relinearize d₂ over Q|P  →  add
```

Two deliberate departures from the N=1024 reference, both forced by the scale:

**Full-polynomial messages.** `he_api` encrypts one integer into coefficient 0 and decrypts only that
coefficient. A constant plaintext is a weak test: it cannot detect a negacyclic wrap error, which is
precisely the bug class a ring implementation has. Here `m` is uniform in `R_t` and all 8192
coefficients are checked.

**Exact big integers.** Decryption CRT-reconstructs modulo `Q`; at LQ=8 that is a 224-bit value whose
per-term products exceed 128 bits. `he_api`'s `u128` arithmetic is correct only because LQ=3 puts `Q`
at ~84 bits. `u512` in this file is the minimum that makes the reconstruction exact rather than lucky,
and every base conversion here goes **through an exact integer** — no approximate RNS conversion, no
error terms. That is deliberate: this file is the *specification* the device path will be checked
against, so it must have no error of its own to explain away.

## Ciphertext I/O is COEFFICIENT form — the opposite of exp_Tb001, and measured

exp_Tb001's headline saving is that eval-form input removes **32 forward NTTs** from every multiply.
That works because CKKS's drop-prime rescale keeps the whole operation in the evaluation domain, so a
coefficient input would be converted once and never converted back.

**BFV cannot do that**, and it is worth being explicit because the reasoning does not carry over. Its
base extension `Q → Q|S` and its rescale `⌊t·e/Q⌉` are *coefficient-wise* — integer operations on
residues, not pointwise ring operations — so the multiply round-trips through coefficients whatever
domain it is handed:

| | eval I/O | coeff I/O |
|---|--:|--:|
| input → coefficients (for the lift) | 32 INTT | 0 |
| lift, then transform for the tensor | 36 NTT (only the 9 new `S` limbs — the `Q` eval forms are already in hand) | 68 NTT (all 17) |
| output back to eval | 16 NTT | 0 |
| **at the ends** | **84** | **68** |

So eval I/O costs **16 transforms more**, not 32 fewer — 171 against 155. Coefficient form is
therefore the contract: the fixture stores coefficient-domain ciphertexts and the device consumes and
produces them. `TB001B_EVAL_IN=1` runs the eval path instead, and both must land on the **same**
coefficient-domain ciphertext — that equality is the A/B, and it passes `0/131072`.

This is a deliberate divergence from exp_Tb001, which keeps eval-form I/O because for CKKS it genuinely
removes 32 transforms. Two schemes, two different right answers; the interfaces differ because the
arithmetic does.

## Base conversion: which method goes where

Every conversion reduces to one small integer per coefficient — **ε**, the number of times the CRT sum
`Σᵢ yᵢ·q̂ᵢ` wrapped past `Q`. Everything else is small-modulus arithmetic:

```
x mod p_j = ( Σᵢ yᵢ·(q̂ᵢ mod p_j)  −  ε·(Q mod p_j) )  mod p_j
```

Three ways to get ε, all present in this repo:

| | ε via | exact on unbounded input? | cost |
|---|---|---|---|
| **approximate FBC** | dropped entirely | no — off by `εQ` | nothing (`baseconv_fbc.cpp`) |
| **SK** | *measured* from a redundant residue mod `m_sk` | **yes, any range** | one extra prime (`baseconv_sk.cpp`) |
| CRNS | *estimated* from a fixed-point accumulator | only if `u` is large enough | one wide accumulator (`baseconv_matmul_crns.cpp`) |

**The precision question, measured.** CRNS's ε is `⌊Σ yᵢ/qᵢ⌋`, whose fractional part is exactly `x/Q`
— so the precision needed is set by how close `x` can get to `Q`, and the guaranteed bound is
`u = ⌈log₂(L·q_max·Q)⌉+1 ≈ 256` here. But the *practical* threshold is far lower. `TB001B_USWEEP=1`
measures it on real ciphertext coefficients with exact CRT as ground truth:

| `u` | 24 | 30 | 32 | 40 | **44** | 64 … 254 |
|---|--:|--:|--:|--:|--:|--:|
| wrong ε (of 32768) | 32768 | 165 | 165 | 7 | **0** | 0 |

The transition is sharp and lands near **44**, not 227 — matching this repo's own measured figure at a
different limb count. Note 32768 samples can only resolve rates to ~3×10⁻⁵, so `u=44` means "below
that", not "provably zero"; the guaranteed bound is still ~256.

**Where each ended up.** The deciding property turned out not to be exactness but **composition**:

* **input lift `Q → S` — SK (exact).** A fresh ciphertext's encryptor holds the integer, so it can
  supply the centered sentinel. Exactness is affordable here.
* **rescale, mod-down, relin mod-up — approximate FBC.** These are *interior* conversions on computed
  values, and their ε is absorbed (see the two sections below). Crucially they need no representative
  from anyone, so they compose.
* **CRNS — not used.** Once the interior conversions were approximate and the boundary one was SK,
  nothing was left that wanted a fixed-point ε. The sweep above is kept because it is the measurement
  that would justify switching, if a sentinel prime ever became the thing worth removing.

**The principle, learned the hard way** (see *The whole multiply runs on device* below): exactness is
affordable at the **boundary**, where a value enters the system with its provenance known, and must be
avoided in the **interior**, where every value is computed and nobody can supply a representative.

## The fixed reference case

`reference/he/tb001b_case.bin` (2.6 MB), regenerated with
`TB001B_GEN=<path> ./build/exp_tb001b <seed>`. It carries the eval-form inputs, the secret and
evaluation keys, the **expected eval-form output**, and the plaintexts.

Correctness is checked against that file, not against data generated in the same process. Two reasons
this matters once the device is in the loop:

* a device failure must be reproducible from the same bytes on any machine — a run-time RNG makes
  "it passed on my card" and "it failed on yours" impossible to compare;
* the expected **output** is stored, so the device is checked against a specific ciphertext rather
  than against whatever the host computes alongside it. A host-side bug can no longer cancel a
  device-side one.

The test adopts the file's keys, so it does not depend on this process's RNG at all — running
`./build/exp_tb001b 99` against a case generated at seed 0 still passes, which is the property that
makes the file a fixture rather than a coincidence.

## What it verifies

```
exp_Tb001b  CORRECT BFV, decrypt-verified   N=8192  LQ=8 (224-bit Q)  LP=4  LS=9  t=12289
[1]  decrypt(encrypt(m))             mismatches = 0/8192    [PASS]
[2]  decrypt(enc(m1) + enc(m2))      mismatches = 0/8192    [PASS]
[3a] host vs FIXED case (eval I/O)   mismatches = 0/131072  [PASS]
[3b] decrypt(case output) == m1*m2   mismatches = 0/8192    [PASS]
[3]  decrypt(enc(m1) (x) enc(m2))    mismatches = 0/8192    [PASS]
```

Seeds 0, 1, 7 and 42 all pass; the run takes ~3.7 s. Test [3] is the one that matters and it is a
**different kind of check** from anything in exp_Tb001: `want` comes from a schoolbook integer
convolution of the plaintexts, `got` from the RNS + big-integer HE pipeline. The two share no code, so
agreement on uniform inputs is not something a bug can fake — the sample line printed alongside makes
the compared values visible rather than asking you to trust a mismatch count of zero.

Test [1] is not redundant. It isolates keygen, encryption, decryption and the big-integer layer, so
when [3] fails the failure is in the multiply and not underneath it.

## Status: the whole multiply runs on device

**Verified end to end.** Parameters, keygen, encrypt, decrypt, HE add, the reference multiply, the
**coefficient-form I/O contract**, the **fixed reference case**, and **all five device stages plus the
chained multiply**:

```
[D1] device NTT (coeff->eval)  vs host                      0/131072  [PASS]   (TB001B_DEV=1)
[D2] device INTT(eval->coeff)  vs host                      0/131072  [PASS]
[D3] device SK lift Q->S (exact, centered) vs reference      0/73728  [PASS]   <- stage 2
[D4] device tensor in Q|S|m_sk vs reference                 0/442368 [PASS]   <- stage 3
[D5] device BEHZ rescale floor(t*e/Q)-eps vs reference       0/65536 [PASS]   <- stage 4
[D6] device relin+combine vs the FIXED case output         0/131072 [PASS]   <- stage 5
```

```
[D7] WHOLE MULTIPLY on device vs the FIXED case             0/131072  [PASS]   (TB001B_CHAIN=1)
[D8] decrypt(device output) == m1*m2                          0/8192  [PASS]
```

**Every stage is bit-exact, and so is the chained multiply.** D6 and D7 compare against the stored
fixture ciphertext rather than a freshly computed reference, so the device reproduces the exact bytes
the case file requires — and D8 confirms those bytes decrypt to `m1·m2`.

The two checks divide the labour, and both are needed: **D8 alone would have passed a broken chain.**
When the relin's mod-up was still the exact centered lift, decryption succeeded while every ciphertext
coefficient was wrong — a fault that would have compounded across chained multiplies. Bit-exactness is
what caught it; decryption is what proves the scheme.

`TB001B_STAGE3=1` adds the tensor, and with it paramgen for all 22 primes (transform constants are
needed for the whole tensor base, not just `Q`).

Both directions, 2 polynomials × 8 limbs × 8192 coefficients, bit-exact. The forward transform is the
one the coefficient-form contract actually uses (coeff in → eval for the tensor); the inverse is
checked too because the rescale comes back through it. That single check pins down more than it
looks: the `E`/`C''` tile layouts agree between the host reference and the shared device transform,
the root choices agree (`psi = g^((q-1)/2N)` from the same `prim_root`), and the shared kernels —
fused 8-partial contraction, SFPI twiddle, transpose — are correct at these primes. Everything
downstream rides on this transform, so it is verified alone first: if `eval -> coeff` were wrong,
every later stage would be wrong in a way that is far harder to localise.

The four device blocks are lifted **verbatim** from `host/experiments/exp_tb001.cpp` — they are the validated,
optimised versions and retyping them is how they drift. Only their environment differs: 22 primes
instead of 12, and the profiling/trace names stubbed out. Paramgen is `O(N2²)` per (prime, direction),
so stage 1 generates constants for the `Q` limbs only.

### Stage 2: the SK lift, and how centering resolves

Validated on the host against the exact reference — `TB001B_SKLIFT=1`, **0/147456** (2 polynomials ×
8192 coefficients × 9 destination limbs).

The open question was centering. BFV needs the representative in `[−Q/2, Q/2)`, SK naturally returns
the one in `[0,Q)`, and RNS has no magnitude comparison to tell them apart. The resolution needs no
extra machinery at all:

> **Carry the sentinel residue of the CENTERED value.** The `q`-residues are unchanged (`x` and `x_c`
> are the same class mod `Q`), but the sentinel reports `x_c = x − w·Q`. Then
> `A_sk − x_sk = (ε + w)·Q`, so the recovered correction is `ε + w`, and subtracting `(ε+w)·Q` from
> the combine yields `x − w·Q = x_c` directly.

The CRT overflow and the centering wrap are the same kind of quantity — a multiple of `Q` — so one
subtraction removes both. No offset trick, no sign bit, no second accumulator.

The consequence for the ciphertext format: it carries **9 residues**, the 8 `Q`-limbs plus
`x_c mod m_sk`. Encryption produces it free (the host has the integer); a multiply's rescale produces
it as one more destination limb.

**The algorithm, with the constants the kernel needs:**

```
  precompute   qhinv_i = (Q/q_i)^-1 mod q_i        qhat_sk_i = (Q/q_i) mod m_sk
               qhat_p[i][j] = (Q/q_i) mod p_j       Q_p[j] = Q mod p_j
               Qinv_sk = Q^-1 mod m_sk

  1.  y_i   = x_i · qhinv_i  mod q_i                          <- phase 1 of baseconv_fbc, unchanged
  2.  A_sk  = Σ_i y_i · qhat_sk_i  mod m_sk                   <- ONE more destination column
  3.  eps   = (A_sk − x_sk) · Qinv_sk  mod m_sk               <- new: 1 sub + 1 modmul
  4.  A_j   = Σ_i y_i · qhat_p[i][j]  mod p_j                 <- phase 2 of baseconv_fbc, unchanged
  5.  out_j = (A_j − eps · Q_p[j])  mod p_j                    <- new: 1 fused mul-sub per destination
```

Steps 1 and 4 are the existing kernel. The device delta is **one extra destination column plus a short
epilogue** — `x_sk` arrives as one more input tile, and `eps` is an elementwise tile like any other.

### Carry exp_Tb001's FBC optimisations into every conversion here

The base conversions in tb001b are **much wider** than tb001's, so the fixes from
[`exp_tb001.md`](exp_tb001.md) §2/§4 are not refinements here — one of them is a correctness
requirement and the rest decide whether this experiment is conversion-bound.

| conversion | `Ls × Ld` | pairs | vs tb001's measured 167 µs @ 28 pairs |
|---|---|--:|---|
| lift `Q → S\|m_sk` | 8 × 10 | 80 | ~480 µs × 4 polys = **1.9 ms** |
| rescale `Q\|S\|m_sk → Q\|m_sk` | 18 × 9 | 162 | ~970 µs × 3 = **2.9 ms** |

~4.8 ms of base conversion against tb001's total of 7.494 ms. In tb001 the FBC was 8.9% of the
critical path; here it is likely to be the largest single line item. So:

1. **Loop over the actual `Ls`/`Ld`, never a padded width (§4).** In tb001 this was a 2.3× saving. Here
   it is **mandatory**: the rescale has `Ls = 18`, which does not even fit the old `FBCW = 8` layout.
   Phase 1 is `O(Ls)` and phase 2 is `O(Ls·Ld)`, so a padded width costs quadratically.
2. **Size every CB to the actual width (§4's trap).** `c_3` is reserved once per destination inside the
   phase-2 loop, so a depth that is not a multiple of `Ls` leaves a reserve straddling the buffer end,
   and `pack_tile(.., cb, i)` offsets **without wrapping**. It surfaced as 4 wrong limbs out of 14, not
   a crash. At `Ls = 18` the CB is 72 KB of L1 — size it deliberately.
3. **Loop the new ε correction over `Ld` too.** `out_j = A_j − ε·(Q mod p_j)` is one fused multiply-sub
   per destination; padding it wastes the same way phase 2 does.
4. **`bconv_one` does NOT apply.** Its `Ls = 1` collapse is a property of CKKS's drop-prime rescale.
   BFV's conversions are `Ls = 8` and `Ls = 18`; there is no single-source case to specialise.
5. **The mod-down trick (§1) DOES apply.** BFV's keyswitch is the same hybrid, so "only the `P` limbs
   leave the eval domain" carries over unchanged — it is host-side scheduling, not scheme-specific.
6. **The Montgomery fold (§3) needs re-deriving.** tb001 folded the tensor's `R` pre-scale into the
   rescale's constant. BFV's rescale multiplies by nothing (it is a division), so there may be no
   constant to fold into — check before assuming.

**Stage 2 landed.** `compute/baseconv_sk.cpp` — phases 1 and 2 are `baseconv_fbc`'s unchanged
(including its width fix), the sentinel is one more destination column, and steps 3/5 are a short
epilogue. Bit-exact against the exact big-integer reference on the first device run, centering
included.

**Stage 3 landed.** The tensor needs no new kernel — it is `modmul_sfpi_full` and `modadd_map` over 18
limbs. Two details worth carrying: `modmul_sfpi_full` is a **Montgomery** multiply, so one side is
montified before upload (the NTT is `Z_q`-linear, so `mont()` rides through it unchanged); and the
**`m_sk` limb is verified too**, against `e` reconstructed exactly from the 17 `Q|S` limbs and reduced
— it is what stage 4's SK reads, so an unchecked fault there would surface in the rescale where it is
much harder to localise.

### Stage 4: the rescale is deliberately NOT exact, and the reference now says so

Reading the validated N=1024 path (`lib/he/he_api.cpp`) settles what the device must compute. Its
rescale folds `t` into phase 1 (`C1a[i] = mont(t·q̂ᵢ⁻¹ mod qᵢ)`) and derives the scaled value from an
**approximate** base conversion of `[t·e]_Q`, so what BEHZ produces is

```
r = ⌊t·e/Q⌋ − ε          ε = the CRT overflow of that conversion
```

Measured exactly on these parameters (`TB001B_BEHZ=1`), over 24576 coefficients:

| ε | 1 | 2 | 3 | 4 | 5 | 6 |
|---|--:|--:|--:|--:|--:|--:|
| share | 0.54% | 10.6% | 38.6% | 38.6% | 11.0% | 0.69% |

bounded by `L_Q = 8`. Against a noise budget of ~2²⁰⁹ that is irrelevant to correctness — `he_demo`
decrypts fine at N=1024 for exactly this reason.

**But it is not irrelevant to testing**, and that is the decision this forced. Had the reference kept
computing the exact value, the device could only ever have been checked by decryption — which says
"the scheme works" but not "stage 4 is right", and defers every fault to a decrypt failure three
stages downstream. So `rescale_behz()` now models the ε, and it is what `multiply()` calls:

* the **device is checkable bit-exactly** through every stage, so faults localise;
* the **decrypt test is unchanged** and remains the independent semantic check — it passes on seeds
  0/1/7 with the BEHZ rescale in place, confirming the ε really is absorbed;
* the exact form survives as `scale_round()`, as the mathematical statement of what is being
  approximated.

The fixture was regenerated, so `reference/he/tb001b_case.bin` now stores the BEHZ result — the
value the device is actually required to produce.

**Stage 4 landed, with no new kernel.** The rescale is two conversions the tree already has:

* **phase A** — `baseconv_fbc` from `Q` to `S|m_sk` with `t` folded into its phase-1 constant
  (`C1[i] = mont(t·q̂ᵢ⁻¹ mod qᵢ)`), then `r_j = (t·e_j − A_j)·Q⁻¹ mod p_j` in three pointwise ops;
* **phase B** — `baseconv_sk` unchanged, `S → Q`, **with phase A's `m_sk` channel as the sentinel**.

That is the whole reason `m_sk` is carried through the tensor: phase A computes it so phase B can be
exact. The builders are generalised over arbitrary source/destination prime lists, so the lift and the
rescale's phase B are the same code with different arguments.

**A trap worth recording.** The kernels declared their constant arrays at exp_Tb001's widths
(`C2[8][8]`). Phase A needs `Ld = 10` and phase B `Ls = 9`, so both overflowed and scribbled the
compute RISC's stack — presenting as **65536/65536 wrong**, with nothing to point at the cause. A
total failure is a useful signature: subtle arithmetic bugs give partial mismatches, so "everything
wrong" says structural, not numerical. Both kernels are now sized for the widest caller, and
exp_Tb001 re-verified (`0/114688`, `0/131072`) since `baseconv_fbc` is shared.

**Stage 5 landed**, again with no new kernel: lift `d₂` from `Q` to `P` (the same `baseconv_sk`,
sentinel = `d₂ mod m_sk` which the rescale already produced), transform to eval over `QP`, the evk
MAC, back to coefficients, mod-down by `P` (`baseconv_fbc` from `P` to `Q` plus a subtract and a
scale), then the combine.

The mod-down needed the same treatment as the rescale: it derives `k = ⌊v/P⌋` from an approximate
conversion, so BEHZ produces `⌊v/P⌋ − ε` with `ε` bounded by `L_P = 4`. `moddown_behz()` models it, and
the multiply still decrypts on seeds 0/1/7/42 with **both** approximations in place — which is the
statement that matters: the two ε terms together are absorbed by the noise budget.

### The whole multiply runs on device — and getting there found a real limit

```
[D7] WHOLE MULTIPLY on device vs the FIXED case         0/131072  [PASS]
[D8] decrypt(device output) == m1*m2                      0/8192  [PASS]
```

`TB001B_CHAIN=1` runs lift → tensor → rescale → relinearise → combine on device from its own
intermediates, reproducing the fixture's ciphertext **bit for bit** and decrypting to `m1·m2`.

It did not work first time, and the failure was worth more than the fix. The chain initially returned
`131072/131072` wrong on D7 while D8 passed — every coefficient different, plaintext exactly right.
That pairing is a diagnosis, not a contradiction: the device had computed a *different but equally
valid* ciphertext.

**Why.** The relinearisation lifts `d₂` from `Q` to `P`, and SK needs `d₂`'s sentinel — the residue of
the representative being lifted. The chain feeds it `r mod m_sk`, where `r` is the scaled value the
rescale produced. But `r` is **not** `d₂`'s centered representative: `|r| ≲ t·N·Q/4 ≈ 2²⁵·Q`, so
`d₂ = r mod Q` is a genuine reduction and `centred(d₂) = r − kQ` for some `k ≠ 0`.

Lifting `r` instead of `centred(d₂)` keyswitches a value that is congruent mod `Q` but larger. The
output ciphertext therefore differs — yet its **phase is unchanged mod `Q`**, because the extra term is
`k·Q·s²  ≡ 0 (mod Q)`. Decryption reads the phase mod `Q`, so it is unaffected. Hence D7 fails and D8
passes, and the only real cost is that the keyswitch noise is inflated by roughly `|r|/Q ≈ 2²⁵` —
invisible against a 2²⁰⁹ budget, but not what an implementation should do.

**The architectural point, which is the actual finding:** the centered-sentinel trick (§stage 2) is
exact and free for a *fresh* ciphertext, because the encryptor holds the integer and can emit
`x_c mod m_sk` directly. **It does not compose.** A computed ciphertext cannot produce its own centered
sentinel, because doing so needs `centred(d mod Q)` — a magnitude comparison, which is precisely what
RNS cannot do and what the sentinel was introduced to avoid.

This is why BEHZ does not lift centered representatives at all: it uses an approximate lift and strips
the resulting `q`-overflow *after* the tensor with a small Montgomery reduction. That path composes,
because nothing downstream needs a representative anyone has to choose.

**The fix, and why it is the right one.** The relinearisation's mod-up does not need to be exact —
standard hybrid keyswitching uses the **approximate** conversion there, and so does exp_Tb001. Its
`ε·Q` error contributes `ε·Q·s²` to the output phase, which is `0 mod Q` and therefore invisible to
decryption. More importantly it **composes**: a pure matmul on residues needs no representative from
anyone, so nothing downstream has to produce one.

Both sides now use it — `build_fbc` on device, and the reference models the same conversion instead of
the exact centered lift. The exact SK lift remains where it belongs, at the **input** lift (stage 2),
where the encryptor holds the integer and can supply the sentinel.

That is the general shape worth remembering: **exactness is affordable at the boundary, where a value
enters the system with its provenance known, and must be avoided in the interior, where every value is
computed.** The centered-sentinel trick is exact and free in the first position and impossible in the
second.

**What is left:** measurement. Every stage and the chain are verified; there is no end-to-end latency
number yet. Each stage is verified against reference inputs, which is what makes a
fault localise; the device does not yet run the whole multiply from its own intermediates. That is
plumbing rather than new algorithm — every piece is proven — but it is not done, and until it is there
is no end-to-end device latency number to quote. Each is checkable on its
own against the reference, because the fixture stores the expected output. The reference and its
fixture exist first on purpose — dispatching kernels against a specification that has not been shown to
decrypt is how you get a fast wrong answer, and without a stored expected output there is nothing for
the device to be wrong against.

It is worth being honest in advance about what the port implies, because it is not a re-parameterisation
of exp_Tb001. Counting transforms per multiply:

| stage | transforms |
|---|--:|
| input INTT, eval → coefficients (the lift is coefficient-wise) | 32 |
| NTT the 9 new `S` limbs for the tensor | 36 |
| tensor in `Q\|S` | pointwise, 0 |
| INTT the three degree-2 terms out of `Q\|S` | 51 |
| relin: NTT `d₂` over `Q\|P` | 12 |
| mod-down: INTT the `P` limbs only (exp_Tb001 §1's trick applies unchanged) | 8 |
| mod-down: NTT the corrections back to eval | 16 |
| output `d₀`, `d₁` → eval | 16 |
| **total** | **171** |

**That table predates the port and the coefficient-I/O decision — the built chain is what counts.**
Counting what `TB001B_CHAIN=1` actually dispatches:

| | transforms | conversion pairs (`Ls×Ld`) |
|---|--:|--:|
| NTT the 18-limb tensor base, 4 polynomials | 72 | — |
| INTT the three degree-2 terms | 54 | — |
| relin: NTT `d₂` + evk over `QP`, INTT back | 60 | — |
| lift `Q → S\|m_sk`, ×4 | — | 4 × 80 |
| rescale: FBC `Q → S\|m_sk` + SK `S → Q`, ×3 | — | 3 × 161 |
| relin mod-up `Q → P`, mod-down `P → Q` ×2 | — | 32 + 64 |
| **total** | **186** | **~900** |

24 of those transforms are the evk, which a production implementation would keep in evaluation form
rather than re-transforming per multiply — so **162** is the steady-state figure.

At exp_Tb001's measured ~62.9 µs per transform and ~6 µs per conversion pair, that projects to
**~16 ms of critical path busy, roughly 2× exp_Tb001's 7.494 ms**.

**This is a projection from operation counts, not a measurement**, and it is the one number in this
document that has not been measured. It is also the honest framing of the whole experiment: the
CROSS-aligned cost model is cheaper than the scheme it models, and this is what turns that from an
assumption into a number — once the trace stack is wired.

The optimisations already landed in exp_Tb001 apply directly — the contraction fusion, the FBC width
fix and the SFPI modmul are all in `kernels/experiments/exp_tb001/`, which this experiment **shares**. That
sharing is the point: a kernel change is measured against both experiments at once, and the two cannot
drift apart.

## Running it

```sh
./build/exp_tb001b                     # host reference only: no device, no env.sh needed
./build/exp_tb001b 42                  # any seed; 0/1/7/42 are known good

TB001B_DEV=1   NTT_CORES=8 ./build/exp_tb001b       # + device stages 1-2 (D1-D3)
TB001B_STAGE3=1 ...                                 # + tensor, rescale, relin (D4-D6); all 22 primes
TB001B_CHAIN=1  ...                                 # + the whole multiply on device (D7-D8)
```

Each flag is additive, and each stage checks against the reference independently — that is what makes a
fault localise to one stage rather than surfacing as a decrypt failure several stages later.

| env | what it does |
|---|---|
| `TB001B_GEN=<path>` | regenerate the fixed reference case |
| `TB001B_EVAL_IN=1` | run the eval-form I/O path; must agree with coefficient form bit-for-bit |
| `TB001B_SKLIFT=1` | host-only: the SK lift vs the exact reference (settled the centering question) |
| `TB001B_BEHZ=1` | host-only: the distribution of the rescale's ε |
| `TB001B_USWEEP=1` | host-only: CRNS ε error rate vs `u` |

The host-only modes need no card and run in seconds; they are where each algorithm was settled before
any kernel was written, which is why every device stage passed on its first or second run.

## Relationship to the other programs

| | ring | correctness claim |
|---|---|---|
| `he_demo` / `lib/he` | N=1024, LQ=3 | decrypt-verified BFV, 1 core |
| `exp_Tb001` | N=8192, LQ=8 | bit-exact vs a CPU oracle of the same arithmetic |
| **`exp_Tb001b`** | **N=8192, LQ=8** | **decrypt-verified BFV** |

exp_Tb001b is `he_demo`'s correctness standard at exp_Tb001's scale. Neither of the others is
redundant: `lib/he` is the validated small reference this follows, and exp_Tb001 remains the
performance vehicle, where the kernel work happens.
