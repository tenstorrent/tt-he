# Modular Multiplication on Tenstorrent

**Status:** Current — design **and** measurements for the vector modmul.
*Consolidated 2026-08. Unlike the NTT docs this subsystem keeps one file: its performance section is
the evidence for its bottleneck argument, and splitting them would separate a finding from its proof.
The original plan, the full results report and the roofline analysis are in [`history/`](history/).*

**Audience:** engineering management / reviewers with light project context.
**Scope:** the element-wise modular multiply primitive `C[i] = (A[i] · B[i]) mod q`.
**Status:** implemented, validated bit-exact on real silicon (n150 / Wormhole B0), multi-core, timed.
**Last updated:** 2026-06-26

---

## 1. What this is and why we need it

Modular multiplication — "multiply two integers, then take the remainder modulo a fixed
prime `q`" — is the single most-executed arithmetic operation in **homomorphic encryption
(FHE)**. Every encrypted-data computation reduces to large polynomial multiplications, and
the fast way to multiply polynomials is the **Number-Theoretic Transform (NTT)**, whose
inner loop is *millions* of modular multiplies. If modular multiply is slow, FHE is slow.

Our goal is to make this operation fast on Tenstorrent hardware. This document covers the
**element-wise vector** version (`C[i] = A[i]·B[i] mod q`), which is both a standalone
building block (the NTT "twiddle" stage is exactly this) and the proving ground for the
arithmetic tricks reused everywhere else in the project.

Concrete parameters we target:

- **Modulus `q`** is a **28-bit prime** (e.g. `167772161 = 5·2²⁵ + 1`). 28 bits is not
  arbitrary — it is what makes the hardware tricks below fit exactly (§4).
- Values are 32-bit unsigned integers; the natural data unit is a **32×32 tile = 1024
  elements**.

---

## 2. The hardware reality that shapes the whole design

The relevant engine is the Tensix **SFPU** — a 32-lane SIMD vector unit, the right place
for element-wise work (the matrix engine only does dot-products). The decisive finding,
established by reading the toolchain on this machine:

> **The Wormhole SFPU has no native integer multiply.** The only real multiplier is
> **fp32** (24-bit mantissa). There is no `vInt * vInt` instruction.

Integer multiply *is* available, but it is **emulated**: the library routine
(`mul_int_tile`) splits each operand into small limbs (7–11 bits), casts the limbs to
fp32, multiplies and accumulates them with fused multiply-add, and reassembles the integer
result. Crucially, this emulated multiply returns **only the low 32 bits** of the product.

This one fact drives every algorithmic choice:

1. A modular reduction needs the **high** bits of intermediate products, so we cannot just
   call one integer-multiply and be done — we must build the high half ourselves out of
   smaller multiplies whose results we *can* see in full.
2. Because each "multiply" is really ~28 emulated fp32 operations, the cost of the whole
   primitive is **dominated by how many multiplies we issue**, not by data movement. This
   makes the op **compute-bound**, and tells us exactly what to optimize.

---

## 3. The algorithm — why Montgomery, and how it compares to Barrett

To compute `(a · b) mod q` you need two things: the product `a·b`, and a way to reduce a
~56-bit product back down to `[0, q)` **without a hardware integer division** (division is
even more expensive than multiply, and not available on the SFPU). The two standard
division-free reduction methods are **Barrett** and **Montgomery**.

Both of the annotated listings below are runnable:
`reference/vector/modmul_reference.py` checks them against Python's `%` across all five
28-bit moduli the device tests sweep, and asserts the range claim the annotations rest on — that one
conditional subtract suffices for each. It gives them as full modular **multiplies**, with `a·b` as
the first line and the three multiplies marked, because the equal multiply count is the result
[`../he/exp_modmul002.md`](../he/exp_modmul002.md) §2 turns on; the listings below are the reduction
halves alone.

### The shared skeleton: `z = k·q + r`

Both methods start from the division identity. For any integer `z ≥ 0` and modulus `q`
there is a **unique** pair `(k, r)`:

```
z = k·q + r ,   with   k = ⌊z/q⌋   and   0 ≤ r < q
```

so the remainder is `r = z − ⌊z/q⌋·q`. The *only* hard part is `⌊z/q⌋`, because the SFPU
has no integer division. **Barrett and Montgomery are both tricks to isolate "the multiple
of `q` to remove" without dividing** — they differ in *which end of `z` they attack*:

- **Barrett** removes a multiple of `q` from the **top**: it estimates `k = ⌊z/q⌋` directly
  (by multiplying by a precomputed reciprocal of `q`), then `r = z − k·q` is the answer in
  `[0, q)`. This is the skeleton above, taken literally.
- **Montgomery** removes a multiple of `q` from the **bottom**: it picks the multiple
  `k = z·(−q⁻¹) mod R` so that `z + k·q` becomes divisible by `R = 2³²` (its low 32 bits
  vanish), then *shifts* down by 32 — a free word-select instead of a division. The result
  lands scaled by `R⁻¹`, i.e. `z·R⁻¹ mod q`, which is why Montgomery needs a domain (the
  `a → a·R` conversion). Same skeleton — isolate a multiple of `q` and remove it — opposite
  end: Barrett subtracts to leave the true remainder, Montgomery adds-then-shifts to leave a
  shiftable number.

Bit-widths below use the targeted parameters: `q < 2²⁸`, `R = 2³²`, inputs `a, b < q`, so
the product `z = a·b < 2⁵⁶`.

### Barrett reduction
Estimates the quotient `⌊z/q⌋` by multiplying by a precomputed reciprocal of `q`
(`m ≈ ⌊2^k / q⌋`), then subtracts `q·⌊z/q⌋`. It works directly on ordinary integers — no
domain change — so it is convenient when you do an occasional reduction.

- **Pros:** operands stay in the normal integer domain; nothing to convert in/out.
- **Cons:** the quotient estimate needs the **high bits** of a wide product (`z · m`),
  i.e. it leans on exactly the high-half multiply the SFPU does *not* give us cheaply, and
  it needs a correction step (the estimate can be off by one). More awkward on this
  hardware.

One operation per line, with I/O bit-widths marked:

```python
# precompute (host):
#   L = 56                          # >= bitlength(z)
#   m = floor(2**L / q)             # reciprocal of q                 out: ~29b

def barrett(z):                     # in:  z = a*b,  0 <= z < 2**56      (56b)
    t  = z * m                      # in: 56b * 29b   out: 85b   <-- WIDE product, needs high half
    k  = t >> L                     # in: 85b >> 56   out: ~29b  (k ≈ floor(z/q))
    kq = k * q                      # in: 29b * 28b   out: 57b
    r  = z - kq                     # in: 56b - 57b   out: ~30b  (r in [0, 2q))
    if r >= q: r = r - q            # in: 30b         out: 28b   (correction: 1–2 subs, r in [0,q))
    return r                        # out: 28b
```

The pain on this hardware is `t = z * m`: an 85-bit product whose **high 56 bits** (`>> 56`)
are exactly what we keep — it leans entirely on the high-half multiply the SFPU does *not*
give us, *plus* the `if r >= q` correction. Hence the "less well" verdict in the table below.

### Montgomery reduction (what we use)
Works in a transformed "Montgomery domain" where numbers are scaled by `R = 2³²`.
Reduction becomes `REDC(z) = z·R⁻¹ mod q`. Input is a product already in Montgomery form,
`z = ã·b̃ < q² < 2⁵⁶`. One operation per line, I/O bits marked:

```python
# precompute (host):
#   R    = 2**32
#   qinv = pow(q, -1, R)            # q^{-1} mod R                    out: 32b

def redc(z):                        # in:  z = ã*b̃,  0 <= z < 2**56     (56b)
    z_low  = z & (R - 1)            # in: 56b mask    out: 32b   (= z mod R, free low word)
    z_high = z >> 32                # in: 56b >>32    out: 24b   (= floor(z/R), free high word)
    t  = z_low * qinv               # in: 32b * 32b   out: 64b   <-- keep LOW 32 only => SFPU-native, FREE
    t  = t & (R - 1)                # in: 64b mask    out: 32b   (= t mod R)
    tq = t * q                      # in: 32b * 28b   out: 60b
    tf = tq >> 32                   # in: 60b >>32    out: 28b   <-- needs HIGH half of t*q
    r  = z_high + q - tf            # in: 24b + 28b   out: ~29b  (r in [0, 2q))
    if r >= q: r = r - q            # in: 29b         out: 28b   (optional; lazy reduction skips this)
    return r                        # out: 28b/29b
```

The key advantage on this hardware: **`R = 2³²` means "divide by R" and "mod R" are just
taking the high or low 32-bit word — free** (`z_low`, `z_high`, `tf` are all word selects).
No general division, no reciprocal estimate, no off-by-one correction. And `z_low * qinv`
only needs its **low 32 bits** — *exactly* what the emulated `mul_int_tile` returns
natively, so that multiply costs no high-half assembly. Only `t * q` (`>> 32`) needs a
hand-built high half. Barrett's `z * m` needed a high half *and* a correction; Montgomery
needs one high half and (with lazy reduction) **no** correction branch.

| | **Montgomery (chosen)** | **Barrett** |
|---|---|---|
| Division replaced by | shifts by 32 bits (pick a word) | multiply by reciprocal + shift |
| Correction step | none (one conditional subtract) | yes (quotient may be off by 1) |
| Domain | requires in/out conversion (`a → a·R`) | none — works on plain integers |
| Fits "no high-half multiply" HW | **well** — `mod R`/`÷R` are word selects | less well — still needs high-half product |
| Best when | the **same** values are multiplied many times (NTT!) | occasional one-off reductions |

**Why Montgomery wins here:** in an NTT the values are converted to Montgomery form
**once** at the start and back **once** at the end; in between there are millions of
multiplies, and the per-multiply cost is what matters. Montgomery makes that inner cost a
sequence of word-selects and small multiplies with **no correction branch** — a perfect
match for the SFPU's strengths and weaknesses. The conversion overhead is amortized to
nothing.

### Two engineering refinements we use
- **Lazy reduction.** We keep operands in the redundant range `[0, 2q)` instead of forcing
  them into `[0, q)` after every step. This drops the final conditional subtraction from
  the hot path; results chain directly back as inputs. (A single branchless `min(b, b−q)`
  canonicalizes at the very end when needed.)
- **16-bit limb decomposition.** Since the emulated multiply only returns low-32 bits, we
  hand-build the 64-bit product (and the high half of `t·q`) from **16×16→32** sub-products
  that *do* fit in 32 bits. The `q < 2²⁸` cap guarantees none of the partial-sum carries
  overflow 32 bits — this is proven in the code's bounds comments and verified by test.

The CPU reference for all of this is `host/tests/common/montgomery.hpp` (`mulmod = reduce(mul_hi_lo(a,b))`),
which is the bit-exact golden the device output is checked against.

---

## 4. Why the modulus must be 28-bit

`q < 2²⁸` is the load-bearing constraint:

- It keeps every 16×16 limb product and every partial-sum carry **inside 32 bits**, so the
  hand-built high-half assembly is exact with no overflow.
- `2²⁸ = 128⁴`, so any value `< q` splits into **exactly four 7-bit limbs** — the basis for
  the int8-matmul lowering used by the modular *matrix* multiply elsewhere in the project.

So "28-bit prime" is a hardware-driven choice, not a cryptographic limitation we picked
arbitrarily.

---

## 5. Implementation — how it maps onto the tt-metal API

The structure mirrors a standard tt-metal element-wise kernel: data lives in DRAM, is
streamed tile-by-tile into on-core SRAM (L1) through **circular buffers (CBs)**, processed,
and streamed back. Three kernels run concurrently on each Tensix core, on separate RISC-V
processors so read / compute / write overlap:

```
 DRAM (A,B)                          per Tensix core                          DRAM (C)
 ──────────►  reader kernel  ──►  CB ──►  COMPUTE kernel  ──►  CB  ──►  writer kernel ──►
  (RISCV_1)   noc_async_read     (L1)    Montgomery modmul   (L1)   noc_async_write   (RISCV_0)
                                          on the SFPU
```

**Host program** (`tests/vector/vector_modmul.cpp`): creates the device mesh, allocates three
interleaved DRAM buffers (A, B, C) at 4096 B/tile, creates the CBs (`UInt32`,
double-buffered), splits the tiles across cores with `split_work_to_cores`, sets per-core
runtime args, enqueues the write→compute→read, and checks every element against the CPU
golden. It is parameterized by env vars (`MODMUL_PRIME`, `MODMUL_NTILES`, `MODMUL_KERNEL`,
`MODMUL_MODE`) so we can sweep primes / sizes / kernels without rebuilding.

**The compute kernel — three generations, same bit-exact result:**

1. **Building-block kernel** (`modmul_compute.cpp`) — composes the modmul out of already-
   validated library tile-ops: `mul_int_tile` (emulated 32×32→low-32 multiply),
   `add/sub_int_tile`, logical shifts, `bitwise_or`, `min_uint32`, `fill_tile_int`. Easy to
   trust (every op is individually tested) but pays a full emulated-multiply + a DST
   round-trip *per operation* — measured at **~75 SFPU tile-op passes per tile**.

2. **Hybrid SFPI kernel** (`modmul_sfpi_hybrid.cpp`, the current best) — the insight: SFPI
   (the low-level vector language) *does* have native integer add/sub/and/or/shift/select —
   it only lacks multiply. So we collapse **everything except the multiplies** into four
   custom hand-written SFPI register passes (`face_limbs`, `face_lohi`, `face_split`,
   `face_reduce`) that keep intermediates in vector registers with **no DST round-trips**,
   and keep the **9 multiplies** on the proven emulated `mul_int_tile`. This leaves the one
   error-prone primitive on a trusted path while eliminating the bulk of the overhead
   (~75 passes → ~16). The compile-time constants (`q`, `q⁻¹ mod 2³²`, `q_low`, `q_high`)
   are baked in. **Result: ~1.7× faster, bit-exact on the first run.**

3. **RNS variant** (`modmul_rns_sfpi.cpp`) — handles up to 16 independent vectors each with
   its **own** prime in one launch (Residue Number System), selecting each tile's modulus by
   `tile_id / tiles_per_vector`. This is how FHE actually uses many small primes at once.

**The actual per-element math (what every core computes, all three kernels identical).**
Below is the device computation in Python form — *only* the arithmetic, dropping the
tile/face loop bookkeeping. Every variable is a **32-bit** value (the SFPU never forms a
64-bit number); the reduction algorithm is §3's Montgomery, and the 32×32→64 product is
hand-built from 16×16→32 limbs. The **9** lines marked `# MUL` are the emulated
`mul_int_tile` calls (the bottleneck of §7); everything else is native SFPI add/shift/mask/or.
This is a line-for-line transcription of `host/tests/common/montgomery.hpp` and `modmul_sfpi_hybrid.cpp`.

```python
# baked-in constants (host-precomputed at kernel init):
#   q        : 28b   prime modulus
#   q_inv_32 : 32b   q^-1 mod 2^32
#   q_low    : 16b   q & 0xFFFF
#   q_high   : 12b   q >> 16          (q < 2^28  =>  high limb < 2^12)
MASK16 = 0xFFFF

def modmul(a, b):                  # in: a, b lazy in [0, 2q) < 2^29   (29b each)

    # 1. split each operand into 16-bit limbs            (SFPI: face_limbs)
    aL, aH = a & MASK16, a >> 16   # 16b , 13b
    bL, bH = b & MASK16, b >> 16   # 16b , 13b

    # 2. four limb-products of a*b                        (4 emulated multiplies)
    ll = aL * bL                   # 16b*16b -> 32b   # MUL 1
    hh = aH * bH                   # 13b*13b -> 26b   # MUL 2
    c1 = aL * bH                   # 16b*13b -> 29b   # MUL 3
    c2 = aH * bL                   # 13b*16b -> 29b   # MUL 4

    # 3. assemble  a*b = hi*2^32 + lo   (= z_high, z_low) (SFPI: face_lohi)
    cross = c1 + c2                # 29b + 29b -> 30b      (q<2^28 keeps carries <32b)
    mid   = (ll >> 16) + (cross & MASK16)        # 16b + 16b -> 17b
    lo    = (ll & MASK16) | (mid << 16)          # 32b   low word of a*b  (= z_low)
    hi    = hh + (cross >> 16) + (mid >> 16)     # 28b   high word of a*b (= z_high < q)

    # 4. Montgomery factor  t = (z mod R) * q^-1  mod R   (mul + SFPI: face_split)
    t  = (lo * q_inv_32) & 0xFFFFFFFF   # 32b*32b, keep LOW 32b only   # MUL 5  (SFPU-native)
    tL, tH = t & MASK16, t >> 16        # 16b , 16b

    # 5. four limb-products of t*q                        (4 emulated multiplies)
    pHH = tH * q_high              # 16b*12b -> 28b   # MUL 6
    pMH = tH * q_low               # 16b*16b -> 32b   # MUL 7
    pML = tL * q_high              # 16b*12b -> 28b   # MUL 8
    pLL = tL * q_low               # 16b*16b -> 32b   # MUL 9

    # 6. t_final = floor(t*q / 2^32), then lazy result    (SFPI: face_reduce)
    mid_low  = (pMH & MASK16) + (pML & MASK16) + (pLL >> 16)   # -> 18b
    mid_high = (pMH >> 16)    + (pML >> 16)    + (mid_low >> 16)  # -> 18b
    t_final  = pHH + mid_high      # 28b   = floor(t*q / 2^32)  (high half of t*q)
    r = hi + q - t_final           # in [0, 2q) -> 29b   lazy: NO conditional subtract
    return r                       # (a*b) * R^-1 mod q, lazy in [0, 2q)

def finalize(r):                   # canonical [0, q) only at end of pipeline
    return r - q if r >= q else r  # on device: branchless min(r, r - q)
```

Note the **only multiply that needs >32 output bits handled specially is `t*q`** (step 5–6,
recovered as `t_final` via the `>>16` assembly); `lo * q_inv_32` keeps just its low 32 bits,
which is exactly what `mul_int_tile` returns natively — the §3 "free low-word" advantage.

**One mandatory config flag:** `fp32_dest_acc_en = true`. Int32 results require the 32-bit
DST accumulator mode; the float-fidelity knobs (`HiFi/LoFi`, `math_approx_mode`) are
irrelevant because we are on the **integer** SFPU path, which is exact.

**Validation:** bit-exact against `host/tests/common/montgomery.hpp` for every element, swept across
multiple 28-bit primes (`167772161`, `201326593`, `251658241`, `268435399`, `268435455`),
including boundary operands (`0`, `q−1`, `2q−1`, and the lazy-reduction edge).

### SFPI vs. building-block tile-ops — a worked example

The whole ~1.7× win comes from *how* the non-multiply glue is expressed. Take one step both
kernels perform identically: assembling the 64-bit product `a·b = hi·2³² + lo` from the four
limb sub-products (`ll, hh, cross = c1+c2`). Same math, two ways to run it on the SFPU.

**Without SFPI** — the building-block kernel (`modmul_compute.cpp`) composes it from
library tile-ops. The `ADD`/`DUP`/`SHR`/… names are thin wrappers: each one re-inits its
SFPU mode and then issues **one library op that sweeps the whole 1024-element tile**, reading
operands from and writing the result back to the DST register file. That is what "one pass"
means below:

```cpp
constexpr auto FMT = DataFormat::UInt32;
// each wrapper = (mode init) + (one full-tile SFPU op writing DST)
inline void DUP(uint32_t in, uint32_t out)            { copy_dest_values_init();  copy_dest_values<FMT>(in, out); }      // out = in (copy)
inline void ADD(uint32_t a, uint32_t b, uint32_t o)   { add_int_tile_init();      add_int_tile<FMT>(a, b, o); }          // o   = a + b
inline void OR (uint32_t a, uint32_t b, uint32_t o)   { binary_bitwise_tile_init(); bitwise_or_binary_tile<FMT>(a,b,o); }// o   = a | b
inline void SHR(uint32_t in, uint32_t sh, uint32_t o) { binary_shift_tile_init(); binary_logical_right_shift_tile<FMT>(in, sh, o); }
inline void SHL(uint32_t in, uint32_t sh, uint32_t o) { binary_shift_tile_init(); binary_left_shift_tile<FMT>(in, sh, o); }
inline void LOW16(uint32_t d) { SHL(d, 7, d); SHR(d, 7, d); }   // x & 0xFFFF == (x<<16)>>16  -> itself TWO passes
```

`DUP` is an explicit copy needed because each op overwrites its destination; `LOW16` is two
passes. Assembling `lo`/`hi` then reads as a long sequence of these one-pass calls:

```cpp
// mid = (ll>>16) + (cross & 0xFFFF)
DUP(4,0); SHR(0,SH,0);   // d0 = ll>>16           (2 passes)
DUP(6,1); LOW16(1);      // d1 = cross & 0xFFFF    (3 passes: DUP + SHL + SHR)
ADD(0,1,0);              // d0 = mid               (1)
// lo = (ll & 0xFFFF) | (mid<<16)
DUP(4,1); LOW16(1);      // d1 = ll & 0xFFFF       (3)
DUP(0,2); SHL(2,SH,2);   // d2 = mid<<16           (2)
OR(1,2,1);               // d1 = lo                (1)
// hi = hh + (cross>>16) + (mid>>16)
DUP(6,2); SHR(2,SH,2);   // d2 = cross>>16         (2)
ADD(5,2,5);              // d5 = hh + cross>>16    (1)
DUP(0,2); SHR(2,SH,2);   // d2 = mid>>16           (2)
ADD(5,2,5);              // d5 = hi                (1)
// → ~18 SFPU tile-op passes, ~14 DST round-trips, for this one step
```

**With SFPI** — the hybrid kernel (`modmul_sfpi_hybrid.cpp`) writes the same expression as
one custom register pass (`face_lohi`). `cross`, `mid` and the shifted temporaries live in
SFPU **vector registers (LREGs)**; only the two real outputs (`lo`, `hi`) are written to DST.
A 32×32 tile is four 16×16 **faces**, and each face is 8 vectors of 32 lanes — so the function
body is a `for i in 0..8` loop over the face's vectors, and the `..._params_` wrapper invokes
it **once per face** (4 faces), covering all 1024 elements:

```cpp
constexpr uint32_t NV = 32;  // dst_reg slots per tile; dst_reg[idx*NV + i] = vector i of tile idx

// in: ll, hh, c1=aL*bH, c2=aH*bL  -> out: lo, hi.  ONE SFPU pass for the whole face.
inline void face_lohi(uint32_t ll, uint32_t hh, uint32_t c1, uint32_t c2,
                      uint32_t lo, uint32_t hi) {
    for (uint32_t i = 0; i < 8; i++) {                 // the 8 vectors of one face
        sfpi::vUInt vll = sfpi::dst_reg[ll*NV + i];
        sfpi::vUInt vhh = sfpi::dst_reg[hh*NV + i];
        sfpi::vUInt cross = sfpi::vUInt(sfpi::dst_reg[c1*NV+i])
                          + sfpi::vUInt(sfpi::dst_reg[c2*NV+i]);   // stays in an LREG
        sfpi::vUInt mid   = (vll >> 16u) + (cross & 0xFFFFu);      // stays in an LREG
        sfpi::dst_reg[lo*NV + i] = (vll & 0xFFFFu) | (mid << 16u); // only outputs touch DST
        sfpi::dst_reg[hi*NV + i] = vhh + (cross >> 16u) + (mid >> 16u);
    }
}

// host/compute side: run face_lohi over all 4 faces of the tile, one SFPU pass total.
inline void S_lohi(uint32_t ll, uint32_t hh, uint32_t c1, uint32_t c2, uint32_t lo, uint32_t hi) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_lohi, ll, hh, c1,
                                               ckernel::VectorMode::RC, c2, lo, hi));
}
```

`cross` and `mid` never leave the register file — contrast the building-block version, which
materializes each of them in DST via a `DUP`+shift. The whole face computes in one pass.

**Why SFPI is faster** — four compounding reasons, all visible above:

1. **No DST round-trips.** Library ops shuttle every intermediate (`ll>>16`, `cross&m`,
   `mid`, …) out to the DST register file and back; the write→read latency sits on the
   critical path between *every* op. SFPI keeps the whole dependency chain in LREGs and
   writes DST only for the genuine results — here, 2 writes instead of ~14.
2. **One pass, not ~18.** Each tile-op has fixed per-pass overhead (mode setup, face/vector
   loop issue over 4×8 vectors). SFPI fuses the entire expression into a single pass, paying
   that overhead once. The kernel-wide effect is the doc's **~75 passes → ~16**.
3. **No `DUP`/copy ops.** Tile-ops overwrite their destination, so values that are still
   needed must be copied first (`DUP`). In registers an SSA value is simply referenced again
   — those copies vanish entirely.
4. **No per-op re-init.** Every library wrapper reprograms its SFPU mode (`binary_shift_tile_init()`
   before each shift, etc.). The native SFPI operators (`>>`, `&`, `|`, `+`) are plain vector
   instructions with no mode reprogramming between them.

The one thing SFPI **cannot** do is the multiply — there is no integer-multiply vector
instruction (§2) — so the 9 multiplies stay on `mul_int_tile`. SFPI eliminates only the glue
*around* them; that is exactly why §7's remaining bottleneck is the multiplies themselves.

---

## 6. Performance

Measured on n150, 50-iteration steady state, full modmul. Throughput is "G modmul/s"
(billions of modular multiplies per second).

| N (elements) | tiles | cores used | building-block | **hybrid (current)** | speedup |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 4 | 4 | 37.9 µs · 0.11 G/s | **21.8 µs · 0.19 G/s** | 1.74× |
| 65,536 | 64 | 64 | 46.9 µs · 1.40 G/s | **28.1 µs · 2.33 G/s** | 1.67× |
| 1,048,576 | 1024 | 64 | 511 µs · 2.05 G/s | **297 µs · 3.53 G/s** | 1.72× |

Reading the table:

- **Small N (4k) is latency-bound, not throughput.** Only 4 tiles → only 4 of 64 cores can
  be used, and host-launch (dispatch) overhead dominates. This is the correctness/latency
  point, not a throughput claim.
- **Throughput scales with cores until the 64-core grid saturates,** then plateaus at the
  per-core compute rate. The op is embarrassingly parallel (one tile = one independent unit
  of work), so the ceiling is set by **SFPU instructions per element**, exactly as predicted.
- The hybrid kernel saturates around **~3.5 G modmul/s** on the full grid.

---

## 7. The bottleneck — and where the headroom is

Detailed profiling at the grid-saturating size (2¹⁶) localizes the cost precisely
(`history/report_modmul_hybrid_real_vs_theory.md`):

- The op is **SFPU-compute-bound, not memory-bound.** DRAM read effective bandwidth is
  ~115 GB/s and the read/write stages overlap with compute; data movement is *not* the limit.
- Inside the compute kernel, the **9 emulated `mul_int32` multiplies are the bottleneck.**
  Each is ~28 dependent fp32 operations (load, shift, mask, cast, multiply-add, mantissa-
  extract, store). They make up the large majority of the ~9,900-instruction theoretical
  math body, and their **dependency chains** prevent issuing at the ideal 1-instr/cycle.
- The kernel currently reaches **~59 % of the optimistic SFPU roof** (16,851 measured vs
  9,888 theoretical cycles for the math body). The optimistic full-grid ceiling is
  **~6.6 G modmul/s**; we are at ~3.5.

**In one sentence:** the limit is the *count* and *dependency depth* of the emulated integer
multiplies, because the hardware has no native integer multiplier.

### Measured per-tile latency partition (1 tile, 1 core, n150)

A profiler-clean single-tile run (`TT_METAL_DEVICE_PROFILER=1 MODMUL_ITERS=1 MODMUL_NTILES=1`,
hybrid kernel; device clock 0.985 GHz so cycles ≈ ns) isolates where the math core spends its
time. Read the **MATH core (TRISC_1)** — it is the bottleneck RISC; reader/writer/unpack/pack
finish far sooner and overlap. (Wall-clock for a single tile is ~87 µs, but that is almost all
**host dispatch latency** — the latency-bound small-N regime of §6, not compute.)

| zone (TRISC_1) | cycles | share of `modmul_math` | what it is |
|---|---:|---:|---|
| `modmul_tile_total` | 18,454 | — | whole per-tile kernel body |
| ` ├ modmul_copy_in` | 1,309 | — | copy A,B from CB → DST (outside math) |
| ` └ modmul_math` | **16,851** | 100% | the §5 arithmetic |
| `    ├ modmul_tq_muls` | 5,938 | 35.2% | **4 muls** (`t·q` limbs) + 3 const fills |
| `    ├ modmul_ab_muls` | 5,547 | 32.9% | **4 muls** (`a·b` limbs) |
| `    ├ modmul_qinv` | 1,829 | 10.9% | **1 mul** (`t = lo·q⁻¹`) + fill + face_split |
| `    ├ modmul_reduce` | 1,397 | 8.3% | SFPI `face_reduce` (t_final, lazy result) |
| `    ├ modmul_lohi` | 1,290 | 7.7% | SFPI `face_lohi` (assemble hi/lo) |
| `    └ modmul_limbs` | 569 | 3.4% | SFPI `face_limbs` (16-bit split) |

This confirms the diagnosis quantitatively: `ab_muls + tq_muls` alone is **11,485 cyc = 68 %**
of the math body for 8 of the 9 multiplies; with the multiply inside `qinv` it is **~70 %+**.
The four SFPI assembly passes (`limbs`+`lohi`+`split`+`reduce`) together are only ~3,250 cyc
(~19 %). The `modmul_math = 16,851` here reproduces the 2¹⁶ report figure bit-for-bit, so the
per-tile MATH cost is stable down to a single isolated tile.

### Next levers (ranked)
1. **Constant-fold the `×q` multiplies into SFPI** (Task 2). Four of the nine multiplies are
   `(variable limb) × (compile-time constant limb)`. Done as direct fp32 multiply-adds with
   the constant limbs pre-cast at init, they avoid the full emulated-multiply machinery and
   its DST round-trips — this is the path past the ~3.5 G/s `mul_int_tile`-bound plateau.
2. **Fewer multiplies via algebra.** Any reduction in the 9-multiply count translates almost
   linearly into throughput, since multiplies dominate.
3. **Keep operands L1-resident when chained** (sharding). For the standalone op the DRAM
   round-trip is noise, but when this modmul is one stage of an NTT, staying in L1 across
   stages avoids repeated DRAM traffic. Relevant downstream, not now.

---

## 8. Where this fits in the bigger project

This vector modmul is the **foundational arithmetic layer**. The same Montgomery + limb
machinery, with `q < 2²⁸` and the radix-128 decomposition, is reused to build:

- the **modular matrix multiply** (28-bit `A@B mod q` lowered onto the hardware int8 matmul), and
- the on-device **NTT** (sizes 1024 and 32768), where the per-tile twiddle stage *is* exactly
  this element-wise modmul.

So the algorithm choices documented here (Montgomery over Barrett, lazy reduction, 16-bit
limbs, hybrid SFPI) are not local to one kernel — they are the arithmetic foundation the FHE
pipeline is built on.

---

## 9. Pointers

| What | Where |
|---|---|
| CPU golden reference | `host/tests/common/montgomery.hpp`, `host/tests/common/test_montgomery.cpp` |
| Host program | `tests/vector/vector_modmul.cpp`, `tests/vector/vector_modmul_rns.cpp` |
| Current best kernel | `kernels/shared/compute/modmul_sfpi_hybrid.cpp` |
| Building-block kernel | `kernels/shared/compute/modmul_compute.cpp` |
| Design + algorithm rationale | `docs/vector/history/plan_vector_modmul.md` |
| Full results report | `docs/vector/history/report_vector_modmul.md` (awaiting consolidation) |
| Real-vs-theory bottleneck analysis | `docs/vector/history/report_modmul_hybrid_real_vs_theory.md` |
| Build & run knobs | `device/README.md` |

## 10. Running it

```bash
. env.sh && cmake -B build -G Ninja
cmake --build build --target vector_modmul
./build/vector_modmul                 # default: full modmul, prime 251658241, 4 tiles (N=4096)
```

## Env knobs (validate stages / scale without rebuilding)
| var | default | meaning |
|---|---|---|
| `MODMUL_KERNEL` | `compute/modmul_compute.cpp` | compute kernel rel-path under `kernels/shared/` |
| `MODMUL_MODE` | `modmul` | golden to check against: `passthrough\|mullo\|mulhi\|modmul` |
| `MODMUL_PRIME` | `251658241` | 28-bit modulus |
| `MODMUL_NTILES` | `4` | number of 1024-elem tiles (N = 1024·tiles) |

Examples:
```bash
MODMUL_KERNEL=compute/modmul_mullo.cpp MODMUL_MODE=mullo ./build/vector_modmul   # stage-1 multiply check
MODMUL_NTILES=1024 ./build/vector_modmul                                          # throughput (~2 G modmul/s, 64 cores)
MODMUL_PRIME=201326593 ./build/vector_modmul
```

## Device profiling traces
```bash
TT_METAL_DEVICE_PROFILER=1 MODMUL_ITERS=1 MODMUL_NTILES=64 \
  MODMUL_KERNEL=compute/modmul_sfpi_hybrid.cpp ./build/vector_modmul
# per-core/per-RISC trace -> $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
```
`MODMUL_ITERS` (default 50) controls timed enqueues; use 1 for a clean trace. Archived
traces for 2^12..2^18 are in `generated/profiler/traces/`; see report §11 for the
per-RISC cycle breakdown (MATH/SFPU is the bottleneck).

## RNS / multi-modulus version (up to 16 vectors, each its own prime)
```bash
./build/vector_modmul_rns                                    # 16 vectors x 2^16 elems, 16 primes
MODMUL_NVEC=8 MODMUL_VEC_TILES=64 ./build/vector_modmul_rns   # 8 vectors x 2^16
MODMUL_NVEC=16 MODMUL_VEC_TILES=1 ./build/vector_modmul_rns   # stress: prime switches every tile
```
Env: `MODMUL_NVEC` (1..16), `MODMUL_VEC_TILES` (tiles/vector, 64 → 2^16 elems),
`MODMUL_KERNEL` (default `compute/modmul_rns_sfpi.cpp`). Data is vector-major;
the kernel selects each tile's prime by `tile_id / tiles_per_vector`.

## 11. Deeper analyses (historical)

- [`history/report_modmul_hybrid_real_vs_theory.md`](history/report_modmul_hybrid_real_vs_theory.md) —
  roofline comparison at N=2¹⁶ on the full 64-core grid: the measured math body reaches
  **9,888 theory cycles / 16,851 measured = 59% of the optimistic SFPU roof**. The clearest statement
  of how much headroom the hybrid kernel still has.
- [`history/report_vector_modmul.md`](history/report_vector_modmul.md) — the full results report.
- [`history/plan_vector_modmul.md`](history/plan_vector_modmul.md) — the original plan.
