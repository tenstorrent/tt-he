# Matching OnionPIRv2's default cell on Tenstorrent — design plan (N=2048, 128 MB, ≤32 cores)

> **Status: Historical.** Superseded by [`../onion_match_perf.md`](../onion_match_perf.md).
> Plan for matching OnionPIR; the work it describes was built and measured.
> Measured numbers, if any, are a snapshot from Wormhole n150.

**Goal.** Build a **device-resident (no CPU compute), multi-core (≤32 Tensix)** OnionPIR *server* that runs
the OnionPIRv2 library's **default** workload — `CONFIG_N2048_K1`: `N=2048`, ~60-bit ciphertext modulus,
`t≈2¹⁴`, a **128 MB database**, full query **expand → unpack → initial-dim scan → hypercube fold →
mod-switch** — and measure its latency and throughput.

This is the scale-up of the working 4–8-entry demo (`pir_demo.md`) to the library's real cell. Everything
correctness-critical (expand, unpack, all HE ops) already runs on device at n=1024; this plan is about
**parameters, a multi-core datapath, one new kernel, and the benchmark.**

> **Two deliberate, HW-driven deviations from the library default (agreed):**
> 1. **Ciphertext modulus:** the library's single **60-bit** `q` becomes **`q = q₁·q₂·q₃`, three 28-bit RNS
>    limbs (~84-bit)** — the int8-matmul caps each limb at 28 bits, and **3×28 gives noise headroom above
>    the library's 60** (removes the risk that motivated §7). Cost: generalize the 2-channel machinery to
>    **K=3** (`crt_and_gadget`, `digit_to_rns`, and the gadget's input limbs 2→3) and pay **~1.5× the
>    transforms**. HE properties identical. *Fairness caveat:* 84-bit is ~24 bits fatter than the library's
>    60, so our per-op cost / response size are larger than a true Q-match — stated in the perf writeup.
>    (Exact-Q alternative at the same K=3 cost: **3×20-bit = 60-bit**; chosen only if a Q-matched comparison
>    is wanted over noise-safety.)
> 2. **Gadget length ℓ=8:** the library's expansion key-switch uses `L_KEY=L_KS=8`, above our single-pass
>    `gadget_decompose_signed` limit (ℓ≤6, DST register file) — and with K=3 the input adds a 3rd limb, so the
>    DST need is `ℓ+3 = 11` slots. We **build the chunked (multi-pass) gadget kernel** to hit ℓ=8. One new kernel.
>
> Everything else is matched: `N=2048`, `t≈2¹⁴`, `L_EP=5` (folds), `L_KEY=L_KS=8` (expansion), expansion
> tree height 10, mod-switch to 22-bit, 128 MB DB. The datapath is **fully on device — no CPU arithmetic,
> no per-op host round-trips.**

---

> **Update 2026-07-31 (Blackhole p150):** milestone #5 is largely done —
> [latency_optimization.md](../latency_optimization.md) records the answer pipeline going **3.45 s → 164 ms
> per query (21×)**, bit-exact, via workload caching + a fixed-address buffer pool + buffer residency.
> Note this plan's prescription of **kernel fusion** for #5 was the wrong lever: the cost was fresh-program
> serialization, and caching/trace addresses it without touching kernels. Remaining to a 128 MB end-to-end
> number: trace replay, the C++ decrypt-check, `scan_mm`+`recombine` as the real initial dimension, and
> chaining expansion at full width.

## Current status (2026-07-24) — snapshot

**Functional goal met + throughput advantage demonstrated.** The full OnionPIR query→answer runs
device-resident on the n150 at the library ring (N=2048) and 84-bit RNS, bit-exact vs the golden.

| # | milestone | status |
|---|---|---|
| 1 | Golden noise gate (K=3, N=2048) | ✅ passes with margin |
| 2 | K=3 (~84-bit) device arithmetic — gadget(3-limb), digit→RNS(3-out), CRT-combine(Garner) | ✅ 0 mismatch, K=2 regressions clean |
| 3 | Chunked ℓ=8 gadget | ✅ 0/131072 |
| 4 | Full op set at n=2048, K=2 **and** K=3 — NTT, external product, key-switch, Subs, expand, unpack | ✅ all bit-exact (see §perf table) |
| 5 | Shared engine `pir2048::Engine` (all ops, tests are thin consumers) | ✅ done · device-residency = **kernel fusion** (diagnosed below), not yet built |
| 6 | 32-core int8 scan throughput | ✅ ~28 GB/s @ 32 cores, validated + compared |
| 7 | Answer pipeline (init-dim scan + hypercube folds) → recover DB[idx] | ✅ 0 mismatch, single- & multi-fold; full chain proven transitively |
| 8 | `onion_match_perf.md` — correctness + scan throughput + latency diagnosis | 🟡 done except end-to-end latency (gated on #5 fusion) |

**Key result (§ perf).** Scan (dominant, DB-linear op) at ~28 GB/s / 891 GMAC/s on 32 cores. All crypto correctness locked in behind
`pir2048::Engine`.

**Sole remaining gap — end-to-end latency.** The host-orchestrated answer is dispatch-bound (~1 ms ×
thousands of tiny programs; a 32-cell answer ≈ 4.3 s wall-clock, **not** device compute). Empirically
diagnosed: **program dispatch**, not readbacks — a buffer-resident MAC that cut readbacks made it
*slower* by adding programs. The fix is **kernel fusion** (fuse the four-step NTT's ~10 stages; batch
NTTs/MAC into single multi-tile programs). The 28 GB/s scan is itself a fused kernel — direct evidence
fusion recovers device speed. This is the one open task; it changes no correctness and no throughput
headline. Scoped in `onion_match_perf.md §2` + the `pir_onion_match_plan` memory.

**Artifacts.** Kernels: `gadget_decompose_signed` (KLIMBS), `digit_to_rns` (DR_K), `crt_mul_split` +
`crt_limb3` + `reader_crt_stride`/`reader_pq` (Garner), `scan_mm` + `reader_scan_mm` (scan),
`reader_i32_stride`. Engine: `lib/pir/pir_engine2048.hpp`. Tests: `test_{extprod,bv_keyswitch,
query_expand,pir_answer}2048`, `test_scan_mc`. Goldens: `reference/pir/gen_*_case.py` (all K-parameterized,
`--K 2|3`). Docs: this file + `onion_match_perf.md`; debug lessons in `docs/debugging.md §8–9`.

---

## 1. Target parameters

| parameter | OnionPIRv2 `CONFIG_N2048_K1` | our TT build | note |
|---|---|---|---|
| ring `N` | 2048 | 2048 | `nntt2048`/`nintt2048` validated |
| ct modulus `q` | single 60-bit | `q₁·q₂·q₃`, 3×28-bit (~84-bit) | HW-forced RNS; K=3, noise-safe (>60) |
| plaintext `t` | `2¹⁴` (`PlainMod=14`, 13 usable bits/coeff) | same | full packing |
| fold gadget `L_EP` | 5 | 5 | single-pass gadget OK |
| expand gadget `L_KEY=L_KS` | 8 | 8 | **chunked gadget kernel (new)** |
| expansion tree height | 10 (→ 2¹⁰ cts) | 10 | Subs tree |
| mod-switch `q'` | 22-bit | 22-bit | response shrink |
| database | 128 MB | 128 MB | fully-packed random payloads |

---

## 2. Database shape (128 MB → hypercube)

Following `utils::calculate_db_shape` (OnionPIRv2 `src/utils.cpp`, line 368 at commit `1676dd7`):

```
usable bits/coeff = PlainMod − 1        = 13
plaintext size    = 13·2048/8           = 3328 B
DB cells (num_pt) = 128 MiB / 3328 B    ≈ 40,330
expansion capacity= 2^TREE_HEIGHT       = 2^10 = 1024 BFV cts
```
With `FST_DIM_POW2=true`, `reserved = (num_dims−1)·L_EP`, `slack = capacity − reserved`,
`fst_dim_sz = largest pow2 ≤ slack`, capacity `= fst_dim_sz · 2^(num_dims−1)`:

```
num_dims = 8  →  reserved = 7·5 = 35,  slack = 989,  fst_dim_sz = 512,  capacity = 512·2^7 = 65,536 ≥ 40,330 ✓
num_dims = 7  →  capacity = 512·2^6 = 32,768 < 40,330 ✗
```
**Matched shape: `fst_dim_sz = 512`, `num_dims = 8` (one initial dimension of 512 + seven binary folds).**
~40k cells packed into a 512×2⁷ capacity (rest zero-padded).

**Query (per retrieval):**
- initial dimension: a **512-slot one-hot** → expanded to 512 BFV selectors (`m_exp = 9` Subs levels).
- each of the **7 folds**: an `L_EP=5`-row gadget ladder → expanded, then **unpacked to an RGSW** via `RGSW(s)`.
- expansion key-switch runs at **ℓ=8**; folds at **ℓ=5**. Total expanded slots `512 + 7·5 = 547 ≤ 1024`.
- one-time key material (Galois keys for 10 levels, `RGSW(s)`) uploaded once, amortized.

**DRAM budget (resident):** DB NTT'd in 3×28-bit RNS as uint32 = `N·K·4 = 2048·3·4 = 24 KB/cell`;
`40,330 · 24 KB ≈ 967 MB`. Plus keys/scratch. Comfortable in n150's 12 GB (Blackhole p150: 32 GB).

---

## 3. What runs where (all on device)

**Offline precompute (once, amortized over all queries):**
- DB → forward NTT (Montgomery), resident in DRAM (`645 MB`). *No transforms on the DB per query.*
- Galois keys (10 levels) and `RGSW(s)` → NTT-Montgomery, resident.

**Online (per query), buffer-resident — data stays in DRAM across stages, no host round-trip:**
```
expand (Subs tree, ℓ=8 KS)  ->  unpack (RLWE->RGSW via RGSW(s), ℓ=8)  ->  scan (initial dim, int8 matmul)
   ->  fold (7 external products, ℓ=5)  ->  mod-switch  ->  one response ct
```
The `reindex` transpose stays on-device; **automorph / monomial-shift / ct-add / mod-switch-finalize become
device kernels** (they were host in the demo — see §4/§6). No `EnqueueReadMeshBuffer` between ops.

---

## 4. Kernel inventory

**Reuse as-is (silicon-validated):** `nntt2048`/`nintt2048`, `modmul_compute`, `modadd_reduce`, int8
`fused_matmul_{a,c}`, `crt_mul_split`, `garner_diff`, `digit_to_rns`, `gadget_decompose_signed` (ℓ≤6, for
the ℓ=5 folds), the external-product and BV-key-switch compositions.

**Modify (parameter / dataflow, no new algorithm):**
- **n=1024 → 2048** across the PIR host harnesses & CB sizing (2 tiles/poly). `test_extprod`/`test_bv_keyswitch`
  hardcode `N=1024` + wire the n=1024 four-step NTT → swap to `nntt2048`/`nintt2048`. (`PIR_DEMO`/gadget are
  already n-agnostic.)
- **K=2 → K=3 RNS.** Generalize `crt_and_gadget` (reconstruct 84-bit from 3 residues → 3 radix-2²⁸ limbs),
  `digit_to_rns` (1 digit → 3 residues), and the extprod/keyswitch per-channel loops from 2 to 3 channels.
  The HE stack already runs 3-limb RNS (`ct_mul` Q=3, CRNS bconv) — reuse those patterns; the PIR-specific
  `crt_mul_split`/`garner_diff` need the 3-prime CRT combine.
- **On-device automorph / monomial-shift / ct-add** — currently host reindex+add in the demo. For device
  residency they become dataflow/compute kernels: a signed index-permute (`i→i·j mod 2N`, negate-on-wrap)
  and elementwise mod-add. Cheap; needed to kill the host round-trips.
- **On-device mod-switch finalize** — the `round(q'·c/q)` scale-round, currently host, becomes a small SFPU kernel.

**New (the one genuinely-new kernel):**
- **`gadget_decompose_signed_chunked` (ℓ=8, K=3 input).** DST need is `ℓ + K = 8 + 3 = 11` int32 slots vs the
  8-slot fp32 DST → must chunk. **Chunk the digit output over ≥2 passes**: pass 1 emits digits 0..k while
  carrying the running value (now across **3** radix-2²⁸ limbs) in local vUInt registers; later pass(es) emit the
  rest. Balanced-carry state stays in registers across the digit loop (as today); only the *output* tiles chunk.
  Validate bit-exact vs `onionpir_ref.gadget_decompose_signed` at ℓ=8 (extend `gen_gadget_case.py --ell 8`, 3 limbs).

---

## 5. Parallel strategy (≤32 cores)

Map each stage to the grid by its parallel structure (from `PIR_PRIMER §7` and the expand analysis):

| stage | parallel structure | mapping across ≤32 cores |
|---|---|---|
| **scan (initial dim)** — dominant, **linear in DB** | `acc[r] = Σ_{k<512} sel[k]·DB[k·128+r]`, over 128 fold-slots × 2 comps × 2 ch — a **512-wide batched pointwise MAC** = an int8 matmul | **shard the 512-wide contraction across cores** (16 rows/core → partial accumulators → tree-reduce with ct-adds), *and* batch the 128×4 outputs. Embarrassingly parallel; this is the throughput kernel. Run it on the **int8 matmul engine** (lever #4), not pointwise. |
| **expand (Subs tree)** — log-depth, wide levels | level `a` has `2^a` independent nodes; each node = automorph + ℓ=8 key-switch | **fan the `2^a` nodes across cores, level-by-level (barrier per level).** Deep levels (`a≥5` → ≥32 nodes) saturate the grid; thin early levels (1,2,4,8,16 nodes) underutilize — acceptable, they're a small fraction. |
| **unpack (RLWE→RGSW)** | 7 folds × ℓ=5 conversions, all independent | distribute the `7·5 = 35` external products across cores (each is itself NTT-parallel). |
| **fold** — log-depth, 7 external products on accumulators | the 7 folds are **sequential** (data-dependent chain), but each external product parallelizes over its `2ℓ` rows / per-limb NTTs | parallelize *within* each external product across cores; the 7-deep chain is the latency floor. |
| **mod-switch** | one ct | single core. |

**Core-budget sketch (per stage, illustrative):** scan → all 32 (sharded); expand → up to 32 at the wide
levels; unpack → 32 (35 independent jobs); fold → 32 within each of the 7 steps. The scan dominates wall-clock
at 128 MB, so **prioritize its int8-matmul multi-core form** — that is the headline `bytes/s` number.

Concurrency uses one program with disjoint core-ranges (the co-scheduling pattern already shown to give ~2×,
`he_compiler_coschedule_verified`). NTTs use the existing multi-core variant (`ntt1024_dev_mc` → 2048 analogue).

---

## 6. Device-resident dataflow (no CPU)

The demo reads every intermediate back to host. For a fair latency number and "no CPU":
- **Buffer-in/buffer-out signatures** on every stage: residues stay in DRAM across expand→unpack→scan→fold→
  mod-switch. Chain like the standalone NTT already does.
- **Build-once, single steady-state program** per query (or a small fixed set), so JIT is amortized, not per-op.
- **All arithmetic + all reindex/add/scale on device** (§4). The only host touch is enqueueing the query buffer
  and reading the final response ct — I/O, not compute.

---

## 7. Noise budget — verify FIRST, in the golden (gate)

Before any device work, extend `onionpir_ref.py --pipeline` to the matched cell and confirm decryption margin:
```
N=2048, q=q₁·q₂ (~56-bit), t=2¹⁴, L_EP=5, L_KEY=L_KS=8, expansion m_exp=9 + 7 folds, mod-switch 22-bit
```
at **3×28-bit (~84-bit)**. Open questions this answers:
- **84-bit vs the library's 60** — we have *more* headroom than a config the library already ships, so decrypt
  should be safe; this run **measures the actual margin** (and confirms 84-bit isn't so large it needs a bigger
  `t`). Our earlier ell=6 result was at t=257/56-bit and a shallow hypercube — **not** transferable; must re-run.
- Confirms the **ℓ split** (8 expand / 5 fold) is sufficient at the full height-9 expansion + 7-deep fold.
- If margin is *huge*, it may be worth revisiting **3×20-bit (exact 60)** for a Q-matched comparison; if it's
  comfortable, keep 3×28 for simplicity.

numpy-accelerated `negacyclic_mul` makes an n=2048 full-pipeline run ~seconds/index, so this sweep is cheap.

---

## 8. Measurement methodology

**The library's default 128 MB DB and retrieval.** Report:
- **Latency:** end-to-end server (device-active *and* wall-clock, now that wall-clock is meaningful) +
  per-stage split (expand / unpack / scan / fold / mod-switch), profiler zones as in `pir_perf.md`.
- **Throughput:** `MB/s` scanned and `queries/sec/card`, over the 128 MB DB.
- **Caveats stated inline:** 3×28-bit RNS (~84-bit) vs the default 60-bit — *fatter*, so we carry more bits
  per op; the 2-packed-ct-vs-1 query encoding residual (`PIR_DEMO`).

Deliverable: a `docs/pir/onion_match_perf.md` in the shape of `pir_perf.md`.

---

## 9. Build order (milestones) — with status

Status legend: ✅ done+validated on device · 🟡 partial · ⏳ not started.

1. ✅ **Golden gate (§7)** — `onionpir_ref.py` at N=2048/K=3(~84-bit)/t≈2¹⁴ decrypts with margin
   (ell=8 side final ~41.5b, ell=5 side ~28b; both OK). 3×28-bit confirmed safe.
2. ✅ **K=3 RNS generalization** — golden done (`Params(K=3)`, K-prime `negacyclic_mul`, bit-exact) **and
   all device arithmetic validated on n150:** gadget 3-limb (`test_gadget_decompose_k3` **0/131072**),
   `digit_to_rns` 3-out (`test_digit_to_rns_k3` **0/131072**), CRT-combine Garner-3prime
   (`test_crt_combine_k3` **0/16384**). K=2 regressions all clean. New code: `reader_crt_stride`,
   `reader_pq`, `crt_limb3` (rest reuses validated K=2 kernels). See [[pir_onion_match_plan]] memory.
3. ✅ **Chunked gadget kernel (ℓ=8)** on device — `gadget_decompose_signed.cpp` multi-pass; ℓ=8 n=2048
   **0/131072**, ℓ=6 regression **0/98304** (target `test_gadget_decompose_l8`). *(3-limb input = part of #2.)*
4. ✅ **n=2048 parameterization (K=2)** — full OnionPIR op set on device at the library ring: `fwd2048`/`inv2048`
   (0/2048), external product (`test_extprod2048` 0/8192), BV key-switch + Subs (`test_bv_keyswitch2048` 0/8192),
   **query expansion + unpack** (`test_query_expand2048` selectors 0/32768 + unpack 0/49152).
   **K=3 (~84-bit) op set COMPLETE + validated** (unified K-general tests, K read from V2 case): external product
   `test_extprod2048` 0/12288, BV key-switch + Subs `test_bv_keyswitch2048` 0/12288, **query-expand + unpack
   `test_query_expand2048` selectors 0/49152 + unpack 0/98304**; all K=2 regressions clean. Every device-resident
   crypto op OnionPIR needs now runs bit-exact at n=2048/84-bit. ✅ **Milestones 1-4 done at K=3.**
5. 🟡 **Shared engine done** — `lib/pir/pir_engine2048.hpp` (`pir2048::Engine`) unifies the K-general
   device ops; the 3 op tests are now thin consumers, all re-validated identical (K=2 + K=3). Remaining #5:
   true device-residency — buffer-chain stages + on-device automorph/shift/add + mod-switch finalize
   (removes per-op host round-trips; now a single-location change in the engine).
6. ✅ **32-core int8-matmul scan** throughput kernel (`test_scan_mc` + `compute/scan_mm.cpp`): int8
   matmul-accumulate validated bit-exact; sharded 1..32 cores. Headline: **27.85 GB/s DB-scan @ 32 cores**
   (15× scaling, DRAM-bound). (Recombine+mod-q
   = validated SFPU tail, not timed; full 128MB end-to-end = #7.)
7. 🟡 **Answer pipeline done** — `test_pir_answer2048` (init-dim scan + chained folds via `pir2048::Engine`)
   recovers DB[idx] bit-exact vs `onionpir_ref.pir_answer`: K=2 [4,2] 0/8192; K=3 [4,2] 0/12288; K=3 [8,2,2]
   (2 folds) 0/12288. Full query→answer proven transitively (expand+unpack #4 + answer each == golden).
   Remaining: scale the scan to the 128 MB shape + one chained expand→unpack→answer test.
8. 🟡 **`onion_match_perf.md` started** — op-set correctness table + scan throughput + comparison
   methodology recorded. Remaining: end-to-end latency + QPS/card after #5/#7.

Each step is independently validatable; 1–4 de-risk before the datapath rewrite (5–7).

### Progress record (session log)
- **Foundations validated on n150:** chunked ℓ=8 gadget (0/131072), n=2048 NTT fwd `nntt2048_dev`
  (0/2048) + inv `nintt2048_dev` (0/2048), and **`fwd2048`∘`inv2048` round-trip** (`test_ntt2048_rt`, 0/2048).
- **Golden (Python):** K=3/84-bit end-to-end noise gate passes; `rlwe_to_rgsw` + chained expand→unpack→answer
  written & bit-exact (`onionpir_ref.py`); numpy-accelerated any-K `negacyclic_mul`.
- **Debugging:** the ℓ=8 "hang" was a **test-harness bug** (kernel JIT `.defines` not forwarded → ran ℓ=6);
  found via Watcher (`CWFW` on writer BRISC) + DPRINT. Full playbook + worked example in `docs/debugging.md`.
- **Immediate next:** `test_extprod2048.cpp` — port the extprod (`crt→gadget→digit_rns→fwd2048→MAC→inv2048`)
  to 2 tiles/poly on the validated NTT layer; validate `gen_extprod_case --n 2048 --ell 5` (K=2), then K=3.

---

## 10b. n=2048 external-product port — implementation recipe (milestone #4)

Foundation **validated on device**: `nntt2048_dev` / `nintt2048_dev` both 0/2048. Gadget chunked ℓ=8
done (§9 #3). The port replaces the n=1024 four-step in `test_extprod`/`test_bv_keyswitch` with the
32×64 machinery lifted from `ntt/nntt2048_dev.cpp` (no new kernels — reuses `fused_matmul_a/c`,
`modadd_reduce`, `modmul_compute`).

**`fwd2048(chan, x[2048]) → X[2048]`** (per RNS channel), ~10 device programs:
1. reshape `x` → 2 input tiles `XinT[bn2]` (bn2=0,1), `tilize32`.
2. **stage-A (64-DFT), K-accumulate over bn2:** for `bk2∈{0,1}`: `A[bk2] = modadd( fusedA(XinT[0],
   WAfold[bk2][0]), fusedA(XinT[1], WAfold[bk2][1]) )` → 2 tiles (`b_A` depth 2).
3. **twiddle:** `B[bk2] = modmul(A[bk2], TwT[bk2])` (Montgomery), 2 tiles.
4. **stage-C (32-DFT):** `Xout[bk2] = fusedC(B[bk2], WCtfold)`, 2 tiles.
5. **reindex (host):** `X[k2 + 64·k1] = Xout[bk2][k2loc][k1]`, `k2 = bk2·32 + k2loc`.
**`inv2048`** is the *transpose* (K-accumulation moves to stage-2), from `nintt2048_dev`:
1. reshape freq `X`: `Xtile[bk2][k1][k2loc] = X[(bk2·32+k2loc) + 64·k1]`.
2. **stage-1 (32-DFT):** `P[bk2] = fusedA(Xtile[bk2], Winv32fold)`; twiddle `Q[bk2] = modmul(P[bk2],
   TwInvT[bk2])` (inverse twiddles `ω⁻ⁿ¹ᵏ²·ψ⁻ⁿ¹`). [2 fusedA + 2 modmul]
3. **stage-2 (64-DFT), K-accumulate over bk2:** `R[bn2] = modadd( fusedC(Q[0], Winv64fold[bn2][0]),
   fusedC(Q[1], Winv64fold[bn2][1]) )`. [4 fusedC + 2 modadd]  `N⁻¹` folded into `Winv32`.
4. **reindex (host):** `x[n1 + 32·n2] = R[bn2][n1][n2loc]`, `n2 = bn2·32 + n2loc`.
Per-channel constants baked once (like `make_chan`): fwd `WAfold[4]/TwT[2]/WCtfold`, inv
`Winv32fold/TwInvT[2]/Winv64fold[4]`.

**Tile-streaming note:** the CRT/gadget/digit helpers take a **tile-count runtime arg** (test_extprod
passes `1u`) — the kernels loop it (gadget confirmed at n_tiles=2). For n=2048 pass `2u` and size the
scratch buffers/CBs for 2 tiles (`mkbuf(2,…)`, `bX`=`mkbuf(4,…)`). This is the main mechanical care in
the port: get every 1→2 tile sizing right, or a CB depth mismatch hangs (→ Watcher).

**Resolved details for the port (from studying the code):**
- `pir::tilize_poly(v, ntiles)` / `untilize_poly` already handle multi-tile → `til2(v)=tilize_poly(v,2)`,
  `unt2(v)=untilize_poly(v,2)` for 2048-vecs. Clean.
- **2-tile ordering gotcha:** `crt_mul_split` writes `bX` in *block* order `[X0_t0,X0_t1,X1_t0,X1_t1]`,
  but the gadget kernel wants *per-coeff-tile interleave* `[X0_t0,X1_t0,X0_t1,X1_t1]`. **Fix:** feed the
  gadget via the validated **`reader_gadget` two-buffer path** (X0 buffer + X1 buffer, as
  `test_gadget_decompose` does at n=2048) rather than the single `bX` buffer + `reader_i32`. Reuses proven
  code and avoids the interleave mismatch.
- `fwd2048`/`inv2048` are ready as lambdas in `test_ntt2048_rt.cpp` (round-trip 0/2048) — lift into
  `test_extprod2048.cpp` verbatim. `pmul`/`padd`/`digit_rns` become `n_tiles=2` variants of the same.
- Validate the front half first (device digits vs golden `digitsG`, as test_extprod does), then the full
  MAC vs golden `outG` — isolates a CRT/gadget bug from a MAC/NTT bug.

**Extprod/keyswitch changes:** swap `fwd/inv → fwd2048/inv2048`; everything else
(`crt_and_gadget`, `digit_to_rns`, `pmul`, `padd`) is already tile-streaming — run with `n_tiles=2`
(2 tiles/poly). MAC loops over `2ℓ` rows × 2 tiles. Validate: `gen_extprod_case --n 2048 --ell 5`
(K=2 first to isolate the NTT change), then K=3.

**K=3 (84-bit) additions (milestone #2 device part):** `crt_mul_split` + `garner_diff` → 3-residue
CRT combine; `digit_to_rns` → 3 outputs; gadget input 2→3 radix-2²⁸ limbs (running value `lo,mid,hi`).
Gen: extend `gen_*_case.py` to emit 3 residues / 3 limbs (golden already K=3-capable).

**Debug:** any hang → Watcher (`device/generated/watcher/watcher.log`) + DPRINT per `docs/debugging.md`.
Reminder (§8 lesson): forward all kernel params as JIT `.defines`.

## 10. Risks / open items

- **Chunked gadget** correctness + cost (extra passes add dispatches; keep it off the fold's hot path — folds
  stay ℓ=5 single-pass).
- **K=3 generalization** (`crt_and_gadget`, `digit_to_rns`, gadget input 2→3 limbs, 3-channel extprod/keyswitch
  loops, ~1.5× transforms) — the main added cost of 3×28-bit. Noise itself is safe (84 > 60); step 1 measures margin.
- **Multi-core scheduling / CB pressure** at 32 cores, 2 tiles/poly, ℓ=8 — watch L1/CB budgets.
- **Query encoding**: we keep the 2-packed-ct split (Δ-scaled init-dim + scaleless fold ladders); collapsing
  to the library's single ct is the unified-encoding trick, cosmetic for upload — out of scope here.
- **Dispatch count**: 547-slot expansion + 7 folds at ℓ=8 is many single-tile dispatches; fusion (twiddle→
  matmul) and batching are the same levers as BFV `ct_mul` — track but don't block first numbers.
```
