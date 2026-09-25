# exp_Tg018s — Blackhole has an integer multiply, and `modmul` was not using it

**Status: Historical** — and read the caveat below before the numbers: the headline was measured against the kernel that was running, not the best one. The settled modmul result is [`../exp_modmul001.md`](../exp_modmul001.md).

> ## ⚠ THE HEADLINE CLAIM IS OVERSTATED — read `docs/he/exp_modmul001.md` first
>
> **This experiment measured against the kernel that was RUNNING, not against the best kernel
> AVAILABLE.** On a clean footing (one core, zero profiler zones, per-tile fitted):
>
> | | µs/tile | already in the tree before this experiment? |
> |---|---|---|
> | `sfpi_full` — what exp_Tg018m ran | 12.430 | — |
> | `sfpi_full_bh` — what this experiment built and ported | 5.375 | no |
> | **`hybrid_canon`** — existing `sfpi_hybrid` + a 3-line canonical finalize | **4.930** | **yes** |
> | `sfpi_hybrid` — existing, untouched | 4.805 | **yes** |
>
> **A faster modmul was already in the repo.** `sfpi_hybrid` went unused because its output is lazy
> `[0, 2q)` and ct_mul cannot consume that — but restoring the canonical subtract is three lines and
> yields 4.930, which beats this experiment's `sfpmul24` kernel by 9% **without using `sfpmul24` at
> all**. So against the right baseline, the `sfpmul24` multiply is worth nothing; against what was
> actually running, 2.31×.
>
> The root cause was exp_Tg018k part A, which ranked `mul24` above `sfpi_hybrid` (9.563 vs 9.743)
> because the hybrid kernel carries ~8 profiler zones inside its per-tile loop. Reading that
> backwards is what led to porting the wrong kernel.
>
> **What still stands:** the end-to-end port is real and measured (64.159 → 56.527 ms, 11.9%,
> `0/6029312`); the SFPMUL24 semantics in §2; and §4 on `mul_int_tile` never issuing `MOD1_UPPER`.
> Most of the available prize was banked — 12.430 → 5.375 of a possible 12.430 → 4.930 — leaving
> ~8% of modmul, roughly 1% end to end.

**Status: DONE (measurement + fix), headline superseded.** `modmul_sfpi_full` emulates every 16×16
product with four fp32 multiplies and twelve int↔float conversions, because it was written for
Wormhole, "which has no integer multiply (and no sfpmul24)". Blackhole has `sfpmul24`. Replacing
**one function** with two instructions takes that kernel from 12.430 → 5.375 µs/tile, bit-exact —
but see the banner: it is not the fastest kernel available.

`modmul` is 20.357 ms of exp_Tg018m's 64.159 ms multiply. **Ported and measured end to end: 64.159 →
56.527 ms, 11.9%**, at `0/6029312` and 39.0 bits.

`host/experiments/exp_tg018s.cpp` (semantics probe) · `host/experiments/exp_tg018k.cpp` part A (the shootout) ·
kernels `compute/modmul_mul24.cpp`, `compute/modmul_mul24_half.cpp`

---

## 1. The result

G=130, T=8 tiles/core, one dispatch each, all against the same host golden:

| variant | µs/tile | vs in-use | mismatches |
|---|---|---|---|
| `sfpi_full` *(what exp_Tg018m runs)* | 16.321 | 1.00× | **0** |
| `compute` — via `mul_int_tile` | 18.194 | 1.11× slower | **0** |
| `sfpi_hybrid` | 9.743 | 0.60× | 16,519 — **not equivalent** |
| `sfpi_half` — half-sync only | 16.008 | 0.98× | **0** |
| **`mul24`** — sfpmul24 only | **9.563** | **0.59×** | **0** |
| **`mul24 + half-sync`** | **9.323** | **0.57× — 1.75×** | **0** |
| `mullo` | 6.522 | 0.40× | 1,064,960 — a different operation |

> **WITHDRAWN: "`mul24` is faster than `sfpi_hybrid`".** That was an artifact. The hybrid kernel
> measured here carries ~8 profiler zones inside its per-tile loop; `sfpi_full` carries one. Re-run
> on one core with zero zones in any kernel and per-tile cost fitted rather than divided
> (`docs/he/exp_modmul001.md`): `sfpi_full` 12.432, **`sfpi_hybrid` 4.807**, `mul24` 5.376 µs/tile.
> **The hybrid is 12% FASTER than `mul24`**, and is exact under its own lazy contract (0 mismatches
> against `gold` or `gold + q` at every T). What remains true is that `mul24` is the fastest
> **canonical** kernel, which is why it is the one that could be dropped into exp_Tg018m unchanged.

The two wins are independent and compose: `sfpi_half`'s half-sync (math/pack overlap) is worth 2%
on its own and 2.5% on top of `mul24`.

### The per-tile figures here are sound

Part B of exp_Tg018k divided a one-tile conversion by `T` (see `exp_tg018r.md` §5), so the same check
was run here. The modmul kernels genuinely loop `T` tiles — `reader_ab_depth` takes `n` as a **count**
(runtime arg 2), not a page offset:

| | span at T=1 | span at T=8 | marginal tile |
|---|---|---|---|
| `sfpi_full` | 18.04 µs | 130.56 µs | (130.56 − 18.04)/7 = **16.07 µs** ✓ |
| `mul24` | 11.00 µs | 76.50 µs | **9.36 µs** ✓ |

and the ratio holds at both sizes (1.64× at T=1, 1.71× at T=8).

---

## 2. What SFPMUL24 actually computes

The instruction is exposed by `sfpi_lib.h` behind `#if __riscv_xtttensixbh` as

```cpp
vMag fractional_mul(a, b, FractionalHalf::Low | FractionalHalf::High)
```

Its semantics had to be **measured**, not inferred: the name says *fractional*, the return type is a
sign-magnitude `vMag`, and the one in-tree wide multiply built on it
(`ckernel_sfpu_div_int32_floor.h`) reassembles with a shift of **23**, not 24. `host/experiments/exp_tg018s.cpp`
emits the raw halves for 532 structured operand pairs and fits candidate models on the host:

| model | lo ok | hi ok | both |
|---|---|---|---|
| low/high 23, full operands | 532 | 348 | 348 |
| low/high 24, full operands | 346 | 132 | 130 |
| **low/high 23, operands & (2²³−1)** | **532** | **532** | **532** |
| low/high 24, operands & (2²⁴−1) | 346 | 148 | 146 |

One model fits **every** probe class — powers of two, `2^k − 1`, and random operands at 11, 16, 22,
23, 24, 28 and 32 bits:

> **`SFPMUL24(a, b)` masks both operands to 23 bits and returns an exact 23×23 → 46-bit product.**
> `Low` is bits [22:0], `High` is bits [45:23]. **Bits ≥ 23 of either operand are silently dropped** —
> `a = 2²³, b = 1` returns `LO = 0, HI = 0`.

It is **not** 24 bits despite the name, and the silent truncation is the trap: an implementation that
feeds it 32-bit operands gets a wrong answer with no indication.

---

## 3. The fix

`modmul_sfpi_full` funnels every product through one function. Its operands are always < 2¹⁶, hence
< 2²³, so nothing is truncated and the two halves reassemble the exact 32-bit product:

```cpp
// before — 4 fp32 multiplies, 4 int→float, 4 float→uint16, plus reassembly
vUInt x0 = x & 0xFFu, x1 = x >> 8u, y0 = y & 0xFFu, y1 = y >> 8u;
vFloat fx0 = int32_to_float(...); /* ...four of these... */
vUInt p00 = float_to_uint16(fx0 * fy0, ...); /* ...four of these... */
return p00 + ((p01 + p10) << 8u) + (p11 << 16u);

// after — two instructions
vUInt lo = vUInt(fractional_mul(x, y, FractionalHalf::Low));    // bits [22:0]
vUInt hi = vUInt(fractional_mul(x, y, FractionalHalf::High));   // bits [45:23]
return lo + (hi << 23u);
```

Nothing else changes — same algorithm, same 8-tile DST layout, same runtime args — so the measurement
isolates the multiply primitive.

---

## 4. Why `mul_int_tile` did not already give this

`modmul_compute` routes its products through `mul_int_tile`, the "native" integer multiply, and
measures **11% slower** than the fp32 emulation. That looked like a hardware fact. It is an API fact:

`tt_llk_blackhole/.../ckernel_sfpu_mul_int.h` issues exactly one `SFPMUL24` per element, always with
`SFPMUL24_MOD1_LOWER`, on operands loaded with `InstrModLoadStore::LO16`. **`MOD1_UPPER` is never
used, on any branch.** The upper half is the whole reason a wide multiply is cheap on this
instruction, and it is unreachable through `mul_int_tile` — so a caller that needs more than the low
part has to compose it out of extra `mul_int_tile` calls, which is what `modmul_compute` does and why
it loses.

**The lesson generalises: the tile-op API is not a complete view of the SFPU.** Going one level down
to SFPI, where `fractional_mul` exposes both halves, is what turned an 18-instruction emulation into
two instructions.

---

## 5. What to do with it

1. **`mul24` is taken — exp_Tg018m runs it.** Measured **64.159 → 56.527 ms end to end, 11.9%**,
   bit-identical. Still available: `mul24 + half-sync` is another 2.5% at kernel level (9.563 → 9.323
   µs/tile) but needs `dst_full_sync_en = false` threaded per kernel, which exp_Tg018m sets globally.
2. **Apply the same swap to the other three kernels carrying the byte-split** — `modmul_scalar`
   (2.339 ms in the profile), `modmul_sfpi_half`, `tensor2_fused`. Same one-line change; not yet
   measured in those, so measure rather than assume.
3. **`sfpi_hybrid` is NOT retired** — see the withdrawal above. On equal footing it is the fastest of
   the three (4.807 vs 5.376 µs/tile) and exact under its lazy `[0, 2q)` contract. It is unusable
   only where a caller needs a canonical residue. Its real lesson is structural: its speed comes from
   4 SFPI passes instead of 7, which is a lever independent of the multiply instruction, and no
   kernel yet has both.
4. **Do not pursue `mul_int_tile`.** Measured 11% slower, and §4 explains why it cannot win as it
   stands.

The matrix-engine FBC (`docs/he/history/exp_tg018r.md`, 1.98× on 11.897 ms) is still unported; on the same
reasoning it is worth roughly 5 ms more, which would take the multiply to the low 50s. Measure it
after porting rather than trusting that arithmetic — this change came in at 11.9% against a predicted
13.6%.

---

## 6. Running it

```
. ./env.sh
./build/exp_tg018s                                              # the semantics probe
TT_METAL_DEVICE_PROFILER=1 TGK_PART=a ./build/exp_tg018k 130 8 3   # the shootout
```

The probe is one core, one tile, no profiler needed; it prints the model fit, a per-probe-class
breakdown, and raw `(a, b, LO, HI, a*b)` samples so the semantics can be re-derived by eye.
