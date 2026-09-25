# BFV homomorphic encryption on Tenstorrent

A working **BFV homomorphic-encryption evaluator** built from scratch on a single Tenstorrent
Tensix core (Wormhole n150). You can encrypt two integers, **add and multiply them homomorphically
— with all arithmetic running on the device — and decrypt the correct result**. This is the proof
that TT's modular-arithmetic primitives (int8 matmul → modmul → NTT → base conversion) compose into
real homomorphic evaluation.

```
$ ./build/he_demo 6 7
  decrypt(ct_a (+) ct_b) = 13     (expected 13)   OK
  decrypt(ct_a (x) ct_b) = 42     (expected 42)   OK
```

- **Parameters:** ring degree `N = 1024` (`x¹⁰²⁴+1`), ciphertext modulus `Q` = 3 RNS limbs (~84-bit),
  relin special base `P` = 3 limbs, rescale aux base `S` = 4 limbs + a sentinel `m_sk`, plaintext
  modulus `t = 12289`. Single Tensix core.
- **Status:** HE **add** and HE **multiply** (tensor → rescale → relinearize) both run with **all
  arithmetic on device**, bit-exact to an exact big-int golden that decrypts to `m₁·m₂`.
- **Device-active time:** ~**12 ms** per full `ct_mul` (add ~24 µs). See §5.

---

## 1. Quick start — the demo

The demo exposes a tiny API and hides all TT-Metalium control:

```cpp
#include "he_api.hpp"
he::HEContext he;                       // params + keygen + open device
auto ca   = he.encrypt(6);              // host
auto cb   = he.encrypt(7);
auto csum = he.add(ca, cb);             // HE add  — ON DEVICE
auto cpro = he.multiply(ca, cb);        // HE mul  — ON DEVICE (tensor→rescale→relin)
int  s    = he.decrypt(csum);           // 13
int  p    = he.decrypt(cpro);           // 42
```

Build & run:

```bash
cd device && cmake --build build --target he_demo
./build/he_demo 6 7           # clean output (TT-Metalium device logs are suppressed)
./build/he_demo 20 25
HE_VERBOSE=1 ./build/he_demo 6 7   # show the full device driver logs
```

Takes any two integers `a b` on the command line; keep `a·b < t = 12289` (integers are encoded as
constant polynomials, so the result lives in coefficient 0). Files:

| file | role |
|---|---|
| `examples/he_demo.cpp` | the `main` — encrypt → add & multiply → decrypt → print. No device code. |
| `lib/he/he_api.hpp`  | public API: `HEContext` = `encrypt` / `add` / `multiply` / `decrypt`. |
| `lib/he/he_api.cpp`  | **all** TT-Metalium control + the device pipeline + host crypto, hidden here. |

Encrypt / decrypt / keygen are host-side (they only supply/read the test data — the *evaluator* is
what runs on device, per BFV convention).

---

## 2. The kernel stack (bottom-up)

Everything is built from one hardware primitive — the **signed int8 32×32 tile matmul** — up to
homomorphic multiply. Each layer is validated bit-exact against a CPU big-int reference.

| layer | what it is | device kernels |
|---|---|---|
| **1. Modular matmul** | `C = (A·B) mod q`, `q < 2²⁸`, via **radix-128 int8** decomposition (each 28-bit value = 4 int8 limbs in `[0,127]`, so the int8 matmul stays exact and non-negative). | `matmul/`, `kernels/shared/compute/fused_matmul_{a,c}.cpp` |
| **2. Modular vector ops** | `modmul` / `modadd` / `modsub`, **Montgomery form** (`R = 2³²`). SFPU. | `kernels/shared/compute/modmul_sfpi_full.cpp`, `modadd_reduce.cpp`, `modsub_reduce.cpp` |
| **3. NTT-1024** | negacyclic forward/inverse NTT as a **32×32 four-step** (matmul → twiddle → matmul); the four-step's output permutation (`reindex`) is a **32×32 tile transpose** done on the matrix engine. | `ntt/nntt1024_dev.cpp`, `nintt1024_dev.cpp`, `kernels/shared/compute/transpose_tile.cpp` |
| **4. Base conversion (FBC)** | RNS base change `x mod A → x mod B` (approximate + exact via `m_sk`), fused into one dispatch; **padded to width 4** (see §4 gotcha). | `kernels/shared/compute/baseconv_fbc.cpp`, `baseconv_fbc_sk.cpp` |
| **5. Rescale** | `⌊t·e/Q⌉` (BEHZ fast-RNS-floor) = two FBCs + per-limb modmul/modsub. | `he/bfv_rescale.cpp` |
| **6. HE ops** | **add** = per-limb modadd; **tensor** = NTT-domain pointwise; **relinearize** = mod-up → NTT → ×evk → INTT → mod-down. | `he/bfv_add.cpp`, `bfv_mul_raw.cpp`, `bfv_relin.cpp` |

The **integer multiply** (Montgomery reduction) is the compute bottleneck throughout — data movement
is cheap. See `kernels.md` for the per-RISC / per-phase profile.

---

## 3. How the HE operations are implemented

A BFV ciphertext is a pair of polynomials `(c0, c1)` in RNS (base-`Q` residues). `c0 + c1·s ≈ Δ·m`
where `Δ = ⌊Q/t⌋` and `s` is the secret key.

**HE add** — `(c0,c1) + (d0,d1) = (c0+d0, c1+d1)`: per-limb `modadd`. Decrypts to `m₁+m₂`.

**HE multiply** — `ct_mul = tensor → rescale → relinearize`, all on device:

```
1. tensor      (c0,c1)×(d0,d1) → (e0,e1,e2)        degree-2, over base Q∪S∪{m_sk} (8 limbs)
                 per limb: NTT the 4 inputs → pointwise (e0=c0·d0, e1=c0·d1+c1·d0, e2=c1·d1) → INTT
2. rescale     ⌊t·e/Q⌉  → (ẽ0,ẽ1,ẽ2)  base Q      restores scale Δ (divides out the extra Q/t)
3. relinearize (ẽ2 under s²) → (k0,k1) base Q       key-switch: mod-up Q→QP, NTT, ×evk, INTT, mod-down P→Q
4. out = (ẽ0+k0, ẽ1+k1)                             degree-1 again; decrypts to m₁·m₂
```

Key implementation choices:

- **`Q` = 3 limbs (84-bit), not 4.** The rescale's `ẽ` is ~one limb larger than `Q`, so the aux base
  needs `|Q|+1` limbs. At `Q=3` every base conversion is ≤ 4 limbs, fitting the existing 4-slot FBC
  kernels — no `L_src→L_dst` FBC generalization needed.
- **Unified Montgomery data format.** NTT/INTT and the pointwise multiply all carry values in
  Montgomery form; the transforms are `Z_q`-linear so Montgomery rides through with no kernel change.
  The rescale/FBC (which reconstruct the integer via CRT) fold the Montgomery decode/encode into their
  existing edge constants — zero extra device ops.
- **Buffer-resident chaining.** Within the tensor and relin, data stays in device DRAM across
  NTT → pointwise → INTT (the `reindex` transpose runs on-device), so only stage handoffs read back.
- **Two-mode harness.** A correctness path threads an exact big-int golden per stage; a performance
  path chains device stages. The demo is the assembled, self-contained version.

The exact big-int ground truth is `reference/he/bfv_mul_ref.py` (`decrypt(ct1·ct2)==m1·m2`, verified
all seeds). The device output is bit-exact to it.

---

## 4. Gotcha worth knowing

The fused FBC kernel (`baseconv_fbc.cpp`) is only correct when the modulus count `n` **divides the CB
depth (8)** — n∈{2,4,8}. At n=3 the first `(n−2)` destination limbs read stale circular-buffer data
(a batched-reserve vs per-tile-drain mismatch; never triggered before because every earlier caller ran
at n=4). **Workaround (in use):** always run the FBC at **width 4, padding with an inert limb** (pad
source `C1=0` ⇒ contributes nothing; pad dest gets a dummy prime, output discarded). This keeps
everything on the validated n=4 path. See `PLAN.md` §3 and `kernels.md` §1.

---

## 5. Results

- **Correctness:** HE add & multiply decrypt to `m₁+m₂` / `m₁·m₂ (mod t)`, bit-exact to the golden,
  all seeds. The device tensor matches the golden `0/24576`; rescale, relin, final all `0`.
- **Device-active time:** **12.04 ms / `ct_mul`** (profiler; 12.81 ms steady-state incl. dispatch
  overhead), 501 dispatches. NTT-bound (55%) and dispatch-bound; kernel fusion is the main lever.
  HE add ~24 µs. The cold-run wall-clock (~6.6 s) is JIT compilation + per-stage host readbacks, not
  device work. Stage split: tensor 47% · rescale 28% · relin 24% · add 0.1%.

Full latency distribution (per-kernel + per-stage + methodology): **`ct_mul_perf.md`**. Base-kernel
device profile: `kernels.md`.

---

## 6. Document index

**Three experiments build on this evaluator at the CROSS Set-B ring (N=8192, L=8), sharing one kernel
tree (`kernels/experiments/exp_tb001/`):** `exp_tb001` is the performance vehicle — a faithful *cost model*,
bit-exact against a CPU oracle, 7.494 ms critical path busy at `G=8`; `exp_tb001b` is the **BFV**
correctness vehicle — a real key, BFV's own `⌊t·e/Q⌉` rescale, and `decrypt(enc(a) ⊗ enc(b)) == a·b`
end to end on device; `exp_tg000` is the **CKKS** correctness vehicle — canonical embedding, pointwise
tensor, drop-prime rescale, and `decode(decrypt(enc(z₁) ⊗ enc(z₂))) ≈ z₁ ⊙ z₂` on device, built with
no kernel of its own; 7.600 ms critical path busy at `G=8` and ~45 bits of slot precision. None replaces the others, and a kernel change is measured against all three at
once. See `exp_tb001.md`, `exp_tb001b.md` and `exp_tg000.md`.

| doc | contents |
|---|---|
| **`README.md`** (this) | overview, kernel stack, implementation, demo usage. |
| **`PLAN.md`** | the record of what was built, in order — milestones, decisions, final status. |
| **`roadmap.md`** | roadmap from here — performance, zero-host/capability, applications (prioritized). |
| **`kernels.md`** | per-kernel **correctness** (bit-exact vs golden) + standalone **device profile** (per-RISC, per-phase relin) + optimization. |
| **`ct_mul_perf.md`** | **latency distribution of the full `ct_mul`** — per-kernel (calls/total/per-call) + per-stage decomposition + methodology. |
| **`exp_tg018.md`** | CKKS `ct_mul` at **L_Q=48, dnum=3, N=65536** on a 3-level (32,32,64) NTT — the scaled-up parameter set, plan + running record + the per-kernel latency breakdown. |
| **`dram_guide.md`** | **the measured DRAM/NoC reference for kernel optimisation** — the six-level bandwidth ladder (core port, row, channel, chip), the `M + T·C` latency model, the recipe that reaches 433 GB/s of 436 achievable, the null results not worth retrying, and the code to copy. Start here before optimising any memory-bound kernel. |
| **`exp_tg018m.md`** | **exp_Tg018 optimised end to end — 4.69× on the whole CKKS multiply** (238.663 → 50.920 ms) at bit-identical output. Starts with one allocation change (every tile had been on DRAM bank 0, 1.79× alone), then occupancy, batching, and finally the `sfpmul24` modmul. Also records the one guide rule that inverted. |
| **`exp_modmul002.md`** | **Barrett vs Montgomery at equal effort** — both written as one SFPI pass over `sfpmul24`, both bit-exact, per-tile cost fitted. **they tie** — 22 `sfpmul24` each and a ratio that straddles 1.0 over five runs. What does cost is a *second* conditional subtract (~7%), which Barrett does not need for our modulus (`2^27 + s < q`, asserted). Explains why the counts match (16-bit limbs make 28/29/32-bit operands all cost the same) and why the kernel deviates from the Python reference's textbook form. |
| **`exp_modmul001.md`** | **CLOSED — the modmul, optimised to its floor.** — one core, a fresh self-contained kernel tree with **zero profiler zones**, per-tile cost **fitted** from `span(T)` rather than divided. One core, zero profiler zones, per-tile cost fitted rather than divided: `sfpi_full` 12.432 → **`fused3_bh` 3.086 µs/tile, 4.03×**, canonical and bit-exact, by collapsing 14 SFPI passes to 1 and deleting the four `fill_tile_int` broadcasts (608 → 96 DST accesses per tile). Ported: **64.159 → 54.100 ms end to end.** Records why the fusion needs `sfpmul24` (it will not compile without it), how much was reachable on Wormhole (~3%), and why lazy output is dead once `fractional_mul`'s low register cost let several partial products share a pass. **Supersedes exp_Tg018k part A**, whose hybrid carried ~8 zones per tile and whose ordering was therefore wrong. |
| **`history/exp_tg018s.md`** | Blackhole has an integer multiply and `modmul` was not using it — **but the headline is overstated, see the banner and `exp_modmul001.md`**: it measured against the kernel that was running, while a faster one (`sfpi_hybrid` + a 3-line canonical finalize, 4.930 µs/tile) was already in the tree. `modmul_sfpi_full` emulates every 16×16 product in fp32 because it was written for Wormhole; `sfpmul24` does it in two instructions. **1.75×, bit-exact, one function changed — 13.6% off the whole multiply.** Includes the measured `SFPMUL24` semantics (an exact 23×23→46 multiply that silently truncates past bit 23) and why `mul_int_tile` cannot reach it. |
| **`history/exp_tg018r.md`** | **the base conversion on the SFPU vs the int8 MATRIX ENGINE, on equal footing** — N=2^16, 64 cores, `(Ls,Ld)` swept over {8,16,32}², both routes fully on device, both exact. The matrix engine wins **every** cell (1.19× at 8→8, **10.3× at 32→32**) once its limb↔coefficient transpose is fused into its own reader and writer instead of being a pass. Corrects exp_Tg018k parts B and C. |
| **`history/exp_tg018k.md`** | the two surviving compute kernels: a five-way modmul shootout (part A, stands) and the FBC's cost model (parts B/C, **superseded** — see the banner and exp_Tg018r). |
| **`history/exp_tg018dram.md`** | the **DRAM sweep** for exp_Tg018's two dominant kernels — tiles-per-core × cores, one dispatch per point. Locates the ~220 GB/s ceiling, shows which kernel is compute- vs memory-bound, and finds the 1.80× `w_share` win. Narrows the depth conclusion in `exp_tg018.md`. |
| **`blackhole_port.md`** | porting to Blackhole **p150**: install/env flow (`env.sh`, `tools/install_ttmetalium.sh`), the **mid-kernel re-init deadlock** in `fused_matmul_a/c` and its fix, p150 results + how to compare them against the n150 numbers. |

**Reusing the optimisations:** `kernels/OPTIMISED.md` is the index of the optimised kernel set —
which kernel to use instead of which, and the three SFPU techniques plus the DRAM rules they are
built from, so they can be applied to kernels not yet converted. `examples/ctmul_fast_kernels.cpp`
runs the optimised modular multiply against the original on device, verified, in ~20 seconds.

Related primitive docs live in `docs/matmul/` (NTT, modular matmul), `docs/bconv/` (base conversion),
and `docs/vector/` (modular multiplication). The exact big-int references are in `reference/he/`.
