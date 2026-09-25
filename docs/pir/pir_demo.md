# OnionPIR on Tenstorrent — end-to-end private retrieval

A working **single-server PIR** demo built on the BFV/HE stack (`docs/he`) on one Tenstorrent Tensix
core (Wormhole n150). The client asks for one database entry; the **server touches the whole database
but never learns which entry** and returns an encrypted answer only the client can open. All the
server's homomorphic work — the initial-dimension dot product, the hypercube fold, and the final
mod-switch — **runs on the device**.

```
$ ./build/pir_demo 3             # retrieve index 3   (usage: pir_demo [index] [db_entries])

===== OnionPIR — private information retrieval on Tenstorrent =====

scheme:   OnionPIR, N=1024, composite q = q1*q2 (28-bit RNS), plaintext modulus t=257
database: 4 entries (hypercube 2 x 2), each slot holds one integer:

            DB[0] = 174
            DB[1] = 20
            DB[2] = 168
            DB[3] = 52   <- requested

client wants index 3  (kept secret from the server)

[1] client -> server:  ONE packed query ct + one-time keys (Galois keys, RGSW(s))
[2] SERVER on device:  expand query (Subs tree) -> unpack fold bit (RLWE->RGSW)
                       ->  initial-dimension dot product  ->  hypercube fold (external product)
                       ->  mod-switch the response
[3] server -> client:  one encrypted response ciphertext
[4] client decrypts.

  retrieved DB[3] = 52    (expected 52)   OK
  the server touched all 4 entries but never learned index 3.

Private retrieval succeeded.
```

Each DB slot holds one integer (a constant-poly plaintext, fixed across queries); the index is a CLI
argument and the retrieved value is printed. Correct for every index and DB size (e.g.
`./build/pir_demo 6 8` → an 8-entry DB). The encrypted query selects `DB[i]` without the server
learning `i`. All TT-Metalium device control is hidden behind the `pir::*` API
(`lib/pir/pir_demo_api.hpp` / `pir_demo_api.cpp`), so the demo reads like the protocol — the same shape as
`docs/he`'s `he_demo`. `PIR_VERBOSE=1` keeps the device logs.

Scheme: **OnionPIR v2** (ePrint 2025/1142, Chen–Ren). Design + kernel plan: `pir_kernel_plan.md`.
Device-active latency breakdown: `pir_perf.md` (7.1 ms end-to-end; NTT-bound, ~79% transforms).
Overview of *why* PIR is the right TT-accelerated HE showcase, plus the new-op deep dives:
`docs/pir/` (`history/onionpir_build_plan.md`, `pir_special_kernels.md`,
`pir_external_product_deepdive.md`). The paper is OnionPIR v2, Chen–Ren,
[ePrint 2025/1142](https://eprint.iacr.org/2025/1142).

---

## 1. What runs where

| stage | who | where | op |
|---|---|---|---|
| keygen, encrypt query, RGSW(bit) | client | host (CPU ref) | BFV encrypt / RGSW encrypt |
| **(b) initial dimension** `Σ_k DB[k]·query[k]` | server | **device** | plaintext×ct MAC (#12) |
| **(c) hypercube fold** `mux(C,x,y)=C⊡(y−x)+x` | server | **device** | external product (#16) + ct add/sub |
| **(d) mod-switch** `round(q'·c/q)` | server | **device** (CRT) + host finalize | #11 |
| decrypt → `DB[i]` | client | host (CPU ref) | BFV decrypt |

The server's homomorphic evaluation is the BFV/HE ops of `docs/he` — **ct add**, **plaintext×ct
multiply**, and the **external product** (which *replaces* BFV ct×ct multiply for additive noise;
OnionPIR never uses the hard `⌊t/Q⌉`+relin path). Every device block is validated bit-exact against
the CPU golden `onionpir_ref.py` (see the status table below).

## 2. Parameters

- Ring `n = 1024` (`x¹⁰²⁴+1`); the PIR ring-size machinery is validated at `n = 2048` too
  (`nntt2048_dev`/`nintt2048_dev`).
- Composite ciphertext modulus `q = q₁·q₂`, two **28-bit** NTT-friendly primes (RNS) — matches the
  paper's composite-modulus design and our int8-matmul 28-bit cap.
- Gadget base `B = 2²⁰`, `ℓ = 3` (external product); plaintext `t`; response mod-switch to
  `q' ≈ 22-bit`.
- Hypercube `N = N₀ × 2` (demo: `2 × 2 = 4` entries; one initial dimension + one fold).

## 3. Quick start

```bash
cd device && cmake --build build --target pir_demo
./build/pir_demo 3          # retrieve index 3 of a 4-entry DB (builds the query internally)
./build/pir_demo 6 8        # index 6 of an 8-entry DB
```
The demo builds the client's encrypted query for the requested index via the CPU reference
(`reference/pir/gen_pir_demo_case.py`, needs `python3`); override its path with `PIR_GEN=`. Run from
`device/`. The individual block tests still take an explicit case (`gen_*.py` → `.bin`).

## 4. The block library (all validated on silicon)

Every PIR building block runs on device (measured on Wormhole n150; see the support matrix in the root README), bit-exact vs `onionpir_ref.py`:

| block | program (device) | golden gen | result |
|---|---|---|---|
| NTT / INTT at n=2048 (32×64) | `nntt2048_dev`, `nintt2048_dev` | `ntt2048_ref.py` | ✅ 0/2048 |
| signed gadget decompose (#9) | `test_gadget_decompose` | `gen_gadget_case.py` | ✅ 0/49152 |
| CRT-combine 2×28→56 (#13) | `test_crt_combine` | `gen_gadget_case.py` | ✅ 0/16384 |
| digit → RNS | `test_digit_to_rns` | `gen_gadget_case.py` | ✅ 0/49152 |
| negacyclic conv via NTT (#16 core) | `test_negconv` | — | ✅ 0/1024 |
| **external product** (#16) | `test_extprod` | `gen_extprod_case.py` | ✅ 0/4096 |
| **BV key-switch** (#10) | `test_bv_keyswitch` | `gen_ks_case.py` | ✅ 0/4096 |
| **Subs** (#3 = automorph + key-switch) | `test_bv_keyswitch` | `gen_subs_case.py` | ✅ 0/4096 |
| Galois automorphism + monomial (#14/#15) | `test_permute` | `gen_permute_case.py` | ✅ 0/2048 |
| **1-of-2 mux** (#19 atom) | `test_mux` | `gen_mux_case.py` | ✅ 0/4096 (both bits) |
| **initial dimension** (#12) | `test_initdim` | `gen_initdim_case.py` | ✅ 0/8192 |
| **mod-switch** (#11) | `test_modswitch` | `gen_modswitch_case.py` | ✅ 0/2048 |
| **end-to-end retrieval** | `pir_demo` | `gen_pir_demo_case.py` | ✅ recover DB[i], idx 0..3 |

The genuinely new *arithmetic* kernels are `gadget_decompose_signed` (#9) and `crt_mul_split`/
`garner_diff` (CRT-combine, #13); `digit_to_rns` is a trivial split; Galois/monomial are host
reindex. Everything else reuses kernels already on silicon (modmul, modadd, int8 matmul,
recombine, the fused NTT stages). See `pir_kernel_plan.md` for the full have/modify/new breakdown.

## 5. Files

```
docs/pir/                  reading docs (this folder is .md only)
  pir_demo.md              this file — the end-to-end demo
  pir_kernel_plan.md       have / modify / new kernel breakdown + build order + status
reference/pir/                   CPU reference + golden generators (Python)
  onionpir_ref.py          CPU big-int golden (BFV + RGSW + external product + Subs + fold + mod-switch)
  ntt2048_ref.py           negacyclic NTT-2048 (32×64) oracle
  gen_*.py                 golden case generators (one per block + the demo)
  (.bin cases are generated by gen_*.py and git-ignored)
docs/pir/                  design docs: history/onionpir_build_plan.md, pir_special_kernels.md,
                           pir_external_product_deepdive.md  (paper: ePrint 2025/1142)
device/
  ntt/    nntt2048_dev.cpp, nintt2048_dev.cpp, test_negconv.cpp
  host/tests/  test_gadget_decompose.cpp, test_crt_combine.cpp, test_digit_to_rns.cpp
  pir/    pir_demo_api.hpp, pir_demo_api.cpp   (server device pipeline + client decrypt; all TT control here)
          pir_demo.cpp               (clean narrative demo, like docs/he/he_demo.cpp)
          pir_permute.hpp, test_permute.cpp, test_extprod.cpp, test_bv_keyswitch.cpp,
          test_initdim.cpp, test_modswitch.cpp, test_mux.cpp
  kernels/shared/compute/  gadget_decompose_signed.cpp, crt_mul_split.cpp, garner_diff.cpp, digit_to_rns.cpp
  kernels/shared/dataflow/ reader_gadget.cpp, reader_crt.cpp
```

## 6. Status & next

**Done:** every OnionPIR building block + the end-to-end retrieval run on device, validated bit-exact
against the CPU reference. This is the proof that TT's modular-arithmetic primitives compose into a
real, private, single-server PIR.

**Query expansion + unpack now run ON DEVICE** (#18): the client sends ONE packed query ct + one-time
keys (Galois keys, `RGSW(s)`); the server **expands** it into the N₀ selectors (a `Subs` tree —
host automorph/shift/add + device BV key-switch) and **unpacks** the fold bit into an RGSW
(RLWE→RGSW via `RGSW(s)`, i.e. `extprod` against the message rows), then scans/folds/mod-switches.
Params use `ℓ=6, B=2¹⁰` for expand/unpack (the fold's `ℓ=3, B=2²⁰` fails the extra noise; verified via
`onionpir_ref.py --pipeline`). Standalone check: `./build/test_query_expand` (0/16384 selectors +
0/24576 unpack rows). Nothing is pre-expanded.

**Next (optimization, not correctness):**
- **Throughput**: run the initial dimension on the **int8 matmul engine** (vs the pointwise MAC used
  here) — same operation, the headline `bytes/s` number.
- **Composite-NTT** (#20) and fusion to cut the CRT/NTT round-trips per fold.
- Scale up the hypercube (more dimensions / larger N₀) and the ring to n=2048.
