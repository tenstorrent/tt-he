# BFV on Tenstorrent — build plan & record

> **Status: Historical.** Superseded by [`../README.md`](../README.md) and [`../blackhole_port.md`](../blackhole_port.md).
> The build record: what was constructed, in order, with the decisions behind it. Kept for the reasoning (Q=3 vs Q=4, the FBC width-4 pad, Montgomery-transparent NTT).
> Measured on Wormhole n150.

The record of what was built, in order, with the key decisions and the final status. Read
`README.md` first for the overview and demo; this doc is the "how we got here" and "what's left."

**End state:** a correct BFV evaluator (HE add + HE multiply) with **all arithmetic on device**,
bit-exact to an exact big-int golden, exposed through a clean `HEContext` API and a demo
(`./build/he_demo a b`).

---

## 1. Goal & scope

Build a single-core BFV evaluator — **HE addition** and **HE multiplication with relinearization** —
entirely from TT's modular-arithmetic primitives, and prove it decrypts correctly.

Locked parameters: `N = 1024` (`x¹⁰²⁴+1`); ciphertext modulus `Q` = 3 RNS limbs (~84-bit); relin
special base `P` = 3 limbs; rescale aux base `S` = 4 limbs + sentinel `m_sk`; plaintext modulus
`t = 12289`; single Tensix core. All limbs NTT-friendly (`k·2N+1 < 2²⁸`) and in Montgomery form.

Out of scope (host, by BFV convention): encryption, decryption, key generation — they supply/read
test data; the *evaluator* is what runs on device.

---

## 2. Build order & final status

Bottom-up; every step validated bit-exact vs the CPU big-int golden (`reference/he/bfv_mul_ref.py`,
`bfv_relin_ref.py`).

| # | milestone | status | artifact |
|--:|---|:--:|---|
| 1 | 28-bit modular 32×32 matmul via radix-128 int8 | ✅ | `matmul/`, `docs/matmul/history/PLAN.md` |
| 2 | modmul / modadd / modsub (Montgomery) | ✅ | `kernels/shared/compute/*` |
| 3 | NTT-1024 forward + inverse, negacyclic (four-step) | ✅ | `ntt/nntt1024_dev.cpp`, `nintt1024_dev.cpp` |
| 4 | base conversion FBC (approx + exact `m_sk`) | ✅ | `kernels/shared/compute/baseconv_fbc*.cpp` |
| 5 | **HE add** | ✅ 0/8192 | `he/bfv_add.cpp` |
| 6 | **tensor product** (NTT-domain pointwise) | ✅ 0/12288 | `he/bfv_mul_raw.cpp` |
| 7 | **relinearization** at Q=4 | ✅ 0/8192 | `he/bfv_relin.cpp` |
| 8 | **rescale** `⌊t·e/Q⌉` on device, Q=3 | ✅ 0/3072 | `he/bfv_rescale.cpp` |
| 9 | relinearization at **Q=3** (+ FBC pad fix, §3) | ✅ 0/6144 | `he/bfv_relin.cpp` (`CFG_LQ/LP`) |
| 10 | device-chained full `ct_mul` (correctness mode) | ✅ | `he/bfv_ct_mul.cpp` |
| 11 | **tensor on device** (8-limb, Montgomery I/O) → all arithmetic on device | ✅ 0/24576 | `he/bfv_ct_mul.cpp` |
| 12 | on-device `reindex` (tile transpose) + buffer-resident tensor & relin | ✅ | `kernels/shared/compute/transpose_tile.cpp` |
| 13 | **demo**: `HEContext` API (encrypt/add/multiply/decrypt), hides TT-Metalium | ✅ | `he/he_api.{hpp,cpp}`, `he_demo.cpp` |

Correctness end-to-end: `decrypt(ct_a ⊞ ct_b)=a+b` and `decrypt(ct_a ⊠ ct_b)=a·b (mod t)`, all seeds.
Device-active `ct_mul` ≈ 12.8 ms (501 dispatches). Details in `ct_mul_perf.md`.

---

## 3. Key decisions & findings

1. **`Q` = 3 limbs, not 4.** The rescale's `ẽ = ⌊t·e/Q⌉` is ~one limb larger than `Q`, so the aux
   base needs `|Q|+1` limbs. At `Q=3` every base conversion is ≤ 4 limbs → fits the existing 4-slot
   FBC kernels, avoiding an `L_src→L_dst` FBC generalization. 84-bit `Q` has ample margin for one
   multiply (verified in the golden).

2. **Fused-FBC `n≠4` bug + pad workaround.** `baseconv_fbc.cpp` is only correct when the modulus
   count divides the CB depth (8): n∈{2,4,8}. At n=3/5/6 the first `(n−2)` destination limbs read
   stale CB data (batched-reserve vs per-tile-drain; never hit before because every earlier caller was
   n=4). **Fix:** always run the FBC at **width 4 with an inert padding limb** (pad source `C1=0`;
   pad dest dummy prime, output discarded). Verified via `bfv_moddown_q{2,3,5,6}`. This unblocked
   relin at Q=3.

3. **Unified Montgomery data format, no conversions between kernels.** NTT/INTT are `Z_q`-linear, so
   feeding Montgomery in gives Montgomery out with **no kernel change**; the pointwise multiply is
   natively `mont⊙mont→mont`. The rescale/FBC — which reconstruct the integer via CRT and can't be
   Montgomery-transparent — absorb the `R⁻¹` decode / `R` encode into their **existing edge constants**
   (raw vs double-Montgomery), so a uniform Montgomery format costs **zero extra device ops**.

4. **`reindex` is a tile transpose → done on device.** The NTT four-step's output permutation is
   exactly a 32×32 transpose (`transpose_wh_tile`). Moving it on-device (`transpose_tile.cpp`) lets
   `transform` leave its result in DRAM, enabling buffer-resident chaining (tensor readbacks 96→24/run,
   relin 5→2/limb). For a pure `ct_mul` the order is irrelevant (all ops are pointwise / per-limb), so
   the transpose can ultimately be dropped by keeping a consistent permuted order.

5. **The `s²` keygen bug (found in the demo).** A one-char transcription error in the demo's keygen
   — the negacyclic wrap term was `s2[k] -= …` instead of `s2[k-N] -= …` — dropped `s²`'s wrap-around,
   giving a wrong relin key and a key-dependent constant offset in the multiply (add was unaffected).
   Isolated by proving each layer (pipeline with golden keys, `to8`, encrypt/decrypt via add), then
   running the C++ keys through the Python golden to pin it to keygen, then to `s²`.

---

## 4. What remains (no correctness/arithmetic gaps)

Everything correctness-critical is done. Remaining items are performance / zero-host purity:

- **Dispatch fusion.** `ct_mul` is 501 mostly single-tile dispatches → dispatch-bound. Fusing (esp.
  the NTT twiddle into the adjacent matmul, and the FBC recombine) is the main speed lever
  (`kernels.md` §3).
- **Zero-host data movement.** The remaining host touches are stage *handoffs* (rescale/FBC/moddown
  still take host arrays) — no arithmetic. Buffer-input paths on those + a build-once steady-state
  loop close it.
- **Depth ≥ 2 / larger `N`.** Q=3 is a one-multiply budget; deeper circuits need a re-check of the
  noise margin or more limbs. PIR (`reference/pir/`) needs only the easy ops already built.

---

## 5. Superseded planning docs

This consolidates four earlier docs (now historical; git-preserved): the planning docs
`bfv_prototype_plan.md` (original add/mul/relin scope) and `BFV_new_plan.md` (the rescale bring-up
plan, executed autonomously), plus the perf/status docs `BFV_MULTIPLY_PERF.md` (status + per-step task
reports) and `BFV_KERNELS_PERF.md` (base-kernel device profile) — the latter two merged into
`kernels.md`, the full-multiply latency into `ct_mul_perf.md`.
