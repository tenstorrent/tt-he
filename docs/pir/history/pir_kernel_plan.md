# Kernel plan: OnionPIR v2 end-to-end on-device

> **Status: Historical.** Superseded by [`../pir_demo.md`](../pir_demo.md) and [`../pir_perf.md`](../pir_perf.md).
> The PIR kernel plan, written before the pipeline existed.
> Measured numbers, if any, are a snapshot from Wormhole n150.

**Goal.** Enumerate every kernel needed to run OnionPIR v2 (ePrint 2025/1142) end-to-end on a TT
device, and sort each against what we already built in `device/`. Three buckets:

- **A. Have** — reuse as-is (maybe new runtime args / more tiles).
- **B. Have, needs small modification** — a variant of an existing kernel (new params, sign, base,
  or dropping a step).
- **C. Don't have** — genuinely new kernel or host schedule.

Companion design docs (the "what/why"): `onionpir_build_plan.md`,
`../pir_special_kernels.md`, `../pir_external_product_deepdive.md`.
This doc is the "which code to write."

Key result carried over from the deep-dive: **every modular reduction in the pipeline is ≤ 28-bit**
(composite modulus `q = q₁·q₂`, two 28-bit NTT-friendly primes) — so all our verified 28-bit
modmul / NTT / matmul primitives apply directly. The only 56-bit work is *non-modular* (gadget
decompose = shifts/masks; CRT-combine = a widening MAC whose only reduction is mod q₂).

---

## 1. The pipeline → kernels it touches

```
client → 1 packed BFV query ct + key material (Galois keys, RGSW(s))
server:
  (a) QueryUnpack : expand 1 packed ct → N₀ BFV cts + per-dim RGSW cts
                    [ Subs = Galois-permute + BV key-switch ; monomial mult ; ct add ]
  (b) Initial dim : response = Σ_k DB_ntt[k] ⊙ query_ntt[k]     [ int8 modular matmul / pointwise MAC ]
  (c) Higher dims : fold hypercube, one external product per 1-of-2 (b·(y−x)+x)
                    [ gadget_decompose_signed → NTT → pointwise MAC vs RGSW → INTT ; modsub ; modadd ]
  (d) Mod-switch response → send back                          [ rescale/moddown family ]
```

The bulk of compute is **(b)** (the throughput number). **(a)** and **(c)** carry the noise and
use the one genuinely new arithmetic primitive (signed gadget decompose).

---

## 2. Bucket summary

| # | Kernel / building block | Bucket | Existing basis in `device/` |
|---|---|:---:|---|
| 1 | pointwise modmul, RNS (28-bit) | **A** | `vector/vector_modmul_rns.cpp`, `kernels/shared/compute/modmul_rns_sfpi.cpp`, `modmul_sfpi_hybrid.cpp` |
| 2 | modular add | **A** | `kernels/shared/compute/modadd_reduce.cpp` |
| 3 | modular sub (`y−x`, negation) | **A** | `kernels/shared/compute/modsub_reduce.cpp` |
| 4 | int8 modular matmul (28-bit) | **A** | `matmul/modular_matmul.cpp`, `matmul/fused_modmatmul.cpp`, `kernels/shared/compute/matmul_fbc*.cpp` |
| 5 | negacyclic NTT / INTT (n=1024) | **A** | `ntt/nntt1024_dev.cpp`, `ntt/nintt1024_dev.cpp` |
| 6 | Montgomery per-prime recombine | **A** | `kernels/shared/compute/recombine_rns.cpp`, `recombine_compute.cpp` |
| 7 | base conversion / FBC | **A** | `baseconv/*`, `kernels/shared/compute/baseconv_fbc*.cpp` |
| 8 | NTT/INTT at PIR ring size (n=2048/4096) | **B** | reparametrize four-step NTT (1024=32×32, 32768=32×32×32) → 2048=32×64 or 4096=64×64 |
| 9 | `gadget_decompose_signed(B, ℓ)` | **B** | extend `kernels/shared/compute/decompose_limbs.cpp` (radix-128, unsigned, 4 fixed limbs) |
| 10 | BV-style key-switch | **B** | strip `he/bfv_relin.cpp` (hybrid) down to gadget-decompose + MAC vs KSK |
| 11 | final modulus-switch | **B** | `he/bfv_moddown.cpp` / `he/bfv_rescale.cpp` with new small target q′ |
| 12 | initial-dimension wiring (query×DB batched) | **B** | reuse matmul kernel #4; host batches N₀ + composite-NTT layout |
| 13 | CRT-combine 2×28-bit → single 56-bit (+ split back) | **C** | new; pairs with #9 — the composite-NTT bridge glue |
| 14 | Galois automorphism `x→xʲ` | **C** | new data-movement; borrow `kernels/shared/dataflow/all_to_all_corner*.cpp` reindex |
| 15 | monomial multiply `x^{±k}` | **C** | new data-movement (negacyclic shift + sign) |
| 16 | external product (RGSW ⊡ BFV) | **C** | new host composition over #9,#13,#5,#1,#2 + RGSW type |
| 17 | RGSW type + encrypt reference (offline/host) | **C** | new host code, no device kernel |
| 18 | QueryUnpack / ExpandBFV schedule | **C** | new host orchestration (Alg. 1–3) |
| 19 | hypercube-fold schedule | **C** | new host orchestration (Alg. 4, `b·(y−x)+x`) |
| 20 | composite-NTT (single-mod view across folds) | **C** (opt) | optimization layer over #8,#13 |

---

## 3. Bucket A — have, reuse as-is

These are on silicon and directly used by PIR. No kernel change; only new runtime args, tile counts,
or call sites.

- **pointwise modmul (RNS)** — external-product step 6 (the `2ℓ×2` MAC collapses to per-lane
  modmul in NTT domain), NTT twiddles, and the initial-dimension elementwise multiply. `_rns`
  variant already selects a per-tile prime → matches the two-channel `q₁,q₂` layout.
- **modadd / modsub** — ct add (dimension accumulation, expansion-tree adds), `y−x` in the 1-of-2
  mux, and `+x` after the external product. Negation `q−v` = modsub with `a=0` (or a 1-line SFPI
  variant if we want it fused).
- **int8 modular matmul** — the initial-dimension dot product *is* this kernel; it's our strongest.
  (Wiring for PIR's batch shape is bucket B #12, but the kernel itself is unchanged.)
- **negacyclic NTT/INTT (n=1024)** — the `nNTT/niNTT` used by relin are exactly the transforms the
  external product needs; only the ring size changes (#8).
- **recombine / baseconv (FBC)** — Montgomery per-prime recombine backs matmul & NTT; FBC backs
  mod-down/mod-switch. Reused unchanged.

**Watch-outs (from memory / prior bugs):**
- Single n150 → **run device binaries serially**; concurrent runs wedge the device.
- `baseconv_fbc` is only correct when the modulus count divides CB depth (n∈{2,4,8}) — **pad to 4**
  where it appears in mod-switch.
- int8 tile writes past 12 DRAM banks need `TensorAccessor`, not `InterleavedAddrGenFast`.

---

## 4. Bucket B — have, small modification

### #8 NTT/INTT at PIR ring size (n = 2048 or 4096)
Same four-step machinery as `ntt1024_dev` (32×32) and `ntt32768_dev` (32×32×32). Need a new
factorization + baked twiddle constants + host reindex:
- 2048 = 32×64 (two-factor) or 4096 = 64×64.
- The per-stage `decompose → matmul(folded W) → recombine` structure is unchanged; only W/T tables
  and the reindex permutation are new.
- **Effort:** medium — new params + new golden, no new kernel logic. Pick n once params are fixed
  (step 1 of build order).

### #9 `gadget_decompose_signed(B, ℓ)` — the one core new arithmetic kernel
`decompose_limbs.cpp` today: radix-128, **unsigned**, **exactly 4** limbs `(v>>7p)&127`, one SFPI
pass. PIR needs:
- **signed** digits in `[−B/2, B/2)` (carry-propagate: if a digit ≥ B/2, subtract B and carry +1),
- **arbitrary base B** (shift/mask by `log₂B`, or general base via mul-hi if B not a power of two —
  prefer power-of-two B so it stays shifts/masks),
- **configurable ℓ** (loop bound / output CB depth),
- input is the **single 56-bit** coefficient (two 28-bit limbs after CRT-combine, #13), so the pass
  operates on a 56-bit value split across two vUInt registers.
- **Effort:** medium-high — this is *the* new arithmetic kernel, but it's a variant of an SFPI
  kernel we've written before (same `face_*` + `_llk_math_eltwise_binary_sfpu_params_` pattern).
- Two parameter sets in practice: larger ℓ for query unpack, smaller ℓ for higher dims (Spiral trick).

### #10 BV-style key-switch  — DONE (end-to-end on device)
`(c0,c1)ₛ' → (d0,d1)ₛ = (c0,0) + Σ_k G⁻¹(c1)_k ⊛ KSK[k]`. Exactly the external-product machinery
restricted to ONE decomposed component (c1) plus adding c0. Validated bit-exact vs
`reference/pir/gen_ks_case.py` (onionpir_ref.keyswitch): `tests/pir/test_bv_keyswitch.cpp`, 0/4096 across
seeds. Chain: CRT-combine(c1) → gadget → digit_to_rns → nNTT → MAC vs KSK (ℓ×2) → niNTT → +c0.
No new kernels. Unblocks `Subs` (#3 = automorph #14 ✅ + this) → query expansion (#18).

### #11 final modulus-switch  — DONE (device CRT + once-per-response finalize)
`c'_i = round(q'·c_i/q) mod q'`, q'≈22-bit. Runs once on the final response. Device CRT-combine
(validated) reconstructs the 56-bit C per coefficient; the scale-round is a once-per-response host
finalize. Validated vs `reference/pir/gen_modswitch_case.py` (onionpir_ref.mod_switch):
`tests/pir/test_modswitch.cpp`, 0/2048. (`mod_switch`/`decrypt_modswitched` added to onionpir_ref,
self-test passes.) A device scale-round kernel is possible later but unnecessary for a once-per-response op.

### #12 initial-dimension wiring (query × precomputed DB)  — DONE (operation validated on device)
`response[r][comp] = Σ_{k<N0} DB[k*rest+r] ⊛ query[k][comp]` (plaintext×ct) as an NTT-domain MAC:
DB pre-nNTT'd + Montgomery on device, then `Σ_k query_ntt ⊙ DB_ntt` (modmul+modadd) → niNTT.
Validated bit-exact vs `reference/pir/gen_initdim_case.py`: `tests/pir/test_initdim.cpp`, 0/8192 across
seeds. Correctness path uses the pointwise MAC; the **throughput** version swaps the per-slot MAC for
the int8 matmul engine over the (N0×rest) DB — a perf optimization, same operation.

---

## 5. Bucket C — don't have (new kernels + schedules)

### #13 CRT-combine (2×28-bit → single 56-bit) + split back — composite-NTT bridge glue
The boundary between "RNS 28-bit channels" and "single 56-bit value" that gadget-decompose needs.
- **combine:** `x = CRT(x mod q₁, x mod q₂)` → one 56-bit int per coefficient (widening MAC; its
  only reduction is mod q₂, 28-bit). We have Montgomery recombine (`recombine_rns`) but that reduces
  RNS→*one prime*; this reconstructs the *full composite* value into a 2-register 56-bit int.
- **split back:** reduce each small digit mod q₁,q₂ — trivial, digit < 28-bit, so it's a copy into
  both channels.
- **Effort:** medium — new SFPI kernel, same register-pass style as `recombine_rns`. Pairs 1:1 with
  #9; consider fusing combine→decompose and split→NTT to avoid DST round-trips.

### #14 Galois automorphism `x → xʲ`  — DONE
Coefficient index permutation `i ↦ i·j mod 2n` with a negacyclic sign flip on wrap. Realized as
**host reindex over the tile layout** (`lib/pir/pir_permute.hpp::automorph`), applied during the
existing inter-stage transfers — like the NTT output reindex, no SFPU kernel. Validated bit-exact vs
`reference/pir/gen_permute_case.py` (from onionpir_ref) for all expansion `j = n/2^d+1`, n∈{1024,2048}:
`tests/pir/test_permute.cpp`. Note `j=n+1` degenerates to "negate odd coefficients" (no permute).
Followed by BV key-switch (#10) to land back on `s(x)`.

### #15 monomial multiply `x^{±k}`  — DONE
Negacyclic cyclic shift with sign (`pir_permute.hpp::monomial`), same host-reindex path; validated
(incl. negative k) in `test_permute.cpp`. No key-switch (stays under the same key).

### #16 external product `RGSW(2ℓ×2) ⊡ BFV(2) → BFV(2)`  — DONE (end-to-end on device)
**Full operation validated on silicon:** `tests/pir/test_extprod.cpp` runs the complete external
product `d=(d0,d1) ⊡ C → (out0,out1)` across both RNS channels and matches the golden
(`reference/pir/gen_extprod_case.py`, from onionpir_ref) — front-half digits exact + 0/4096 output, across
seeds. Chain (no new kernels):
`CRT-combine + gadget (front-half) → digit_to_rns → nNTT(digit)/channel → MAC (Σ_r digit_r⊙C[r][col],
modmul+modadd) → niNTT/channel`. RGSW rows are nNTT'd on-device (feed mont(C) → device nNTT →
mont(nNTT(C))) so there's no Python/device NTT-convention matching. Core primitive proven separately in
`tests/ntt/test_negconv.cpp` (`nNTT⊙niNTT = negacyclic conv`).
Bug found in assembly: `crt_mul_split` emits X0,X1 as consecutive tiles in one buffer → feed the gadget
via `reader_i32` (2 tiles), not `reader_gadget` (which expects two separate buffers).

Not a monolithic kernel — a **host-orchestrated composition**:
`INTT → CRT-combine(#13) → gadget_decompose_signed(#9) → split → NTT(#8) → pointwise MAC vs RGSW(#1,#2) down 2 cols → INTT(#8)`.
Accumulation over the `2ℓ` rows can be **lazy** (sum then reduce once). This is the op that replaces
BFV ct×ct multiply.
- **Effort:** medium — orchestration + validation vs a CPU RGSW⊡RLWE reference. **Core milestone.**

### #17 RGSW type + encrypt reference
A `2ℓ×2` matrix `C = Z + m·G`; each row a BFV encryption, stored **pre-NTT'd** in step-6 form. Host
data type + an encrypt routine + a CPU reference. No device kernel.
- **Effort:** low-medium (host/offline).

### #3 Subs  — DONE (wired on device)
`Subs(ct,j)` = automorph (#14 ✅) + BV key-switch (#10 ✅). Validated via `reference/pir/gen_subs_case.py`
(emits a key-switch case with pre-automorphed inputs) run through `device/pir/test_bv_keyswitch`,
j=513, 0/4096. Vs `onionpir_ref.subs`.

### #18 QueryUnpack / ExpandBFV schedule (Alg. 1–3)
Host tree: `Subs` (#3 ✅) + monomial mult (#15 ✅) + ct add (#2). Pure composition; loops the wired
`Subs`. *(Schedule remaining; CPU golden validates it end-to-end.)*

### #19 hypercube-fold schedule (Alg. 4)  — atom DONE on device
Host loop over higher dims: `out = mux_1of2(C, x, y) = ExternalProduct(C, y−x) + x`, `log₂(N/N₀)`
times. The **mux atom is validated on device** (`tests/pir/test_mux.cpp` + `gen_mux_case.py`, both
bits 0/4096, private selection by encrypted bit). Full fold = loop it; CPU golden validates the loop.
- **Effort:** low-medium.

### #20 composite-NTT (optimization)
Keep values in the single-mod (56-bit) view across the fold chain so we don't do a whole-polynomial
CRT round-trip per fold; the CRT cost is paid per-multiply instead. Layered on #8 + #13. Do **after**
correctness (build-order step 6).
- **Effort:** medium-high — perf optimization, not needed for first end-to-end.

---

## 6. Build order (each validated vs a CPU reference)

**Status:** ALL BLOCKS READY (validated on silicon). CPU golden `reference/pir/onionpir_ref.py` (all
stages + end-to-end retrieve PASS). Device blocks, each bit-exact vs golden:
NTT/INTT-2048 (`nntt2048_dev`,`nintt2048_dev`) · gadget (`test_gadget_decompose`) · CRT-combine
(`test_crt_combine`) · digit→RNS (`test_digit_to_rns`) · negacyclic-conv (`test_negconv`) ·
**external product** (`test_extprod`) · **BV key-switch** (`test_bv_keyswitch`) · Galois+monomial
(`test_permute`) · **initial dimension** (`test_initdim`) · **mod-switch** (`test_modswitch`).
**END-TO-END DEMO DONE:** `examples/pir_demo.cpp` — private retrieval of `DB[idx]` from an
`N₀×2` hypercube with the server (initial dim → fold → mod-switch) on device; recovers DB[i] for all
idx, 0 mismatches. See `docs/pir/pir_demo.md`. Remaining is optimization only (query expansion #18 to
compress the request; int8-matmul initial dim for throughput; composite-NTT #20; scale-up).

**Confirmed params from the paper's own repo** (`chenyue42/OnionPIRv2`,
`src/includes/database_constants.h`, config `N2048_K1_COMP` — the composite variant that matches
our 28-bit RNS): `n=2048`; composite `q` via `FirstDimRNSMods={29,29}` (we use two **28-bit**
primes → `q≈56-bit`) with single-modulus view `RnsMods={58}` for the gadget dims; `L_EP=5–6`
(small ℓ, higher dims), `L_KEY=8–10` / `L_KS=8` (large ℓ, expansion / key-switch);
`TREE_HEIGHT=10`; `PlainMod=13–14 bit`; final mod-switch `SmallQWidth=22 bit`. Gadget base
`B ≈ 2^(⌈log₂q⌉/ℓ)` (power-of-two → #9 stays shifts/masks); note balanced digits need
`B^ℓ ≥ 2q`.

1. **Params + NTT (#8).** Lock the params above; build/validate NTT/INTT at `n=2048` (32×64 four-step).
   - **Oracle DONE:** `reference/pir/ntt2048_ref.py` — negacyclic NTT-2048 as four-step (N1=32, N2=64),
     validated bit-exact vs schoolbook for all three primes. Pins the constants the device bakes:
     stage-1 (64-DFT) `WA[k2][n2]=ω₆₄^{k2·n2}·ψ^{32 n2}` (ψ folded), twiddle
     `Tw[n1][k2]=ψ^{n1}·ω^{n1 k2}`, stage-2 (32-DFT) `WC[k1][n1]=ω₃₂^{k1·n1}`; reshape
     `n=n1+32 n2`, reindex `k=k2+64 k1`. (`reference/ntt_matmul` PLAN=(32,64) validates the
     cyclic skeleton.)
   - **Device NTT-2048 DONE:** `tests/ntt/nntt2048_dev.cpp`, validated 0/2048 at both PIR primes
     (268369921, 268361729) + the tool prime, multiple seeds. **No new kernels** — reuses
     `fused_matmul_a/_c`, `modadd_reduce`, `modmul_compute`. 10-program chain: the 64-DFT is a 2×2
     hardware-tiled matmul (4 `fused_a` + 2 `modadd` K-accumulate), twiddle (2 `modmul`), 32-DFT
     (2 `fused_c`). Bug found: fused-matmul compute kernels need their `{q,QINV,QLO,QHI,K14}`
     runtime args set (recombine constants) — easy to miss vs the dataflow-only kernels.
   - **Device INTT-2048 DONE:** `tests/ntt/nintt2048_dev.cpp`, validated 0/2048 at all three primes.
     Mirror chain (ω→ω⁻¹, N⁻¹ folded into stage 1, ψ⁻ⁿ post-scale folded); DFT sizes swap stages
     (stage 1 = 32-DFT / 2 `fused_a`, stage 2 = 64-DFT / 4 `fused_c` + 2 `modadd`). No new kernels.
   - **#8 COMPLETE:** forward + inverse NTT at the PIR ring size, on silicon.
2. **`gadget_decompose_signed` (#9) + CRT-combine (#13) → external product (#16).** Validate
   ext-product correctness + noise vs CPU RGSW⊡RLWE. *Core milestone — unblocks (a) and (c).*
   - **#9 DONE (single-pass, ℓ≤6):** `kernels/shared/compute/gadget_decompose_signed.cpp` +
     `tests/common/test_gadget_decompose.cpp`, validated bit-exact vs `reference/pir/gen_gadget_case.py`
     (`gadget_ep.bin`, B=1024/ℓ=6/n=2048): 0/49152 mismatches incl. edge cases. One SFPI pass, the
     running value in local registers (no DST slots), balanced carry across the two 28-bit limbs.
     *TODO:* chunked variant for ℓ=8–10 (expansion/key-switch) — DST holds only ℓ+2 slots at once.
   - **#13 DONE:** CRT-combine (2×28→56), 3-stage pipeline `garner_diff → modmul_compute (reused) →
     crt_mul_split`, validated bit-exact vs `gadget_case.bin`: 0/16384. New kernels
     `compute/garner_diff.cpp` (SFPI), `compute/crt_mul_split.cpp` (SFPI + `mul_int_tile`),
     `dataflow/reader_crt.cpp`; test `tests/test_crt_combine.cpp`. Reuses the proven Montgomery
     modmul for the one mod-q₂ multiply. *TODO (optimize):* fuse the 3 stages + fuse #13→#9
     (composite-NTT, #20) to drop DRAM round-trips.
   - **#9+#13 together** now cover: RNS residues → single 56-bit → balanced base-B digits.
   - **digit→RNS split DONE:** `compute/digit_to_rns.cpp` (+ `tests/test_digit_to_rns.cpp`),
     signed digit → (d mod q₁, d mod q₂), validated vs golden digits: 0/49152. Pure SFPI.
   - **MAC = no new kernel:** the 2ℓ×2 accumulation is `vector_modmul` (products) + `modadd_reduce`
     (sum) — both already on silicon (deep-dive: "step 6 is nothing special").
   - **External-product kernel inventory COMPLETE.** All pieces exist & are validated:
     INTT (nintt1024) · CRT-combine (#13) · gadget (#9) · digit→RNS · NTT (nntt1024) · MAC
     (modmul+modadd). Remaining for #16 end-to-end = **orchestration** + the **n=2048 NTT (#8)**
     to run at the PIR ring size (nntt/nintt are currently n=1024).
3. **Galois (#14) + monomial (#15) + BV key-switch (#10) → `Subs` → QueryUnpack (#18).** Validate
   the expansion produces the right per-bit ciphertexts.
4. **Initial dim (#12) over precomputed DB NTT; higher dims (#19) via external products.** Assemble
   full query→response.
5. **Final mod-switch (#11); end-to-end correctness** (retrieve right entry) + throughput number.
6. **Optimize:** composite-NTT (#20), keep all key material (Galois keys, RGSW, DB) in NTT form,
   lazy reduction.

---

## 7. Effort rollup

| bucket | items | net new device-kernel work |
|---|---|---|
| **A. have** | #1–7 | none (reuse) |
| **B. modify** | #8–12 | 1 medium NTT reparam, 1 medium-high decompose variant, 3 low-med re-wires |
| **C. new** | #13–20 | 1 medium bridge kernel, 2 data-movement kernels, 1 host op-composition, host types + 3 schedules, 1 opt |

**Bottom line:** the only genuinely new *arithmetic* device kernels are **`gadget_decompose_signed`
(#9)** and **CRT-combine (#13)**; the only new *data-movement* kernels are **Galois automorphism
(#14)** and **monomial multiply (#15)**. Everything else is either already on silicon (bucket A),
a parametric/structural variant of an existing kernel (bucket B), or host-side composition
(external product, RGSW type, and the two schedules). PIR skips BFV ct×ct multiply / `⌊t/Q⌉`
scaling / hybrid relin entirely.

---

## 8. Open design points

- **Ring size:** 2048 (32×64) vs 4096 (64×64) — noise headroom vs NTT cost. Decide at step 1.
- **Base B / ℓ:** power-of-two B keeps #9 as shifts/masks; pick the two ℓ sets (expand vs fold).
- **Fuse #13↔#9 and split↔NTT** to avoid DST round-trips (mirrors how `decompose_limbs` does all
  4 limbs in one pass).
- **Composite-NTT (#20) scope:** how much of the fold chain stays single-mod before re-splitting.
- **DB storage budget:** NTT+RNS roughly doubles raw DB size; confirm it fits DRAM for the target N.
