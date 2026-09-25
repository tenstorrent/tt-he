and # IR op reference — L1 / L2 / L3

Companion to `formalization.md`. One entry per op: **name**, **I/O signature** (MLIR-style),
**attributes** (with the meaning of each), and **what the op does**. The op sets are grounded
in the two applications we actually have — `he_demo` (`lib/he/he_api.cpp`) and `pir_demo`
(`lib/pir/pir_demo_api.cpp`) — and §4 proves both are buildable from these ops.

Dialects (proposed names): **`rns`** = L1 (RNS-poly), **`tile`** = L2/L3 (tile / kernel).
L2 and L3 opcodes already exist in `reference/ntt_matmul/ir.py` (currently un-namespaced);
this doc gives them stable names and signatures.

### The three levels at a glance

| level | dialect | value | what it expresses | key ops |
|---|---|---|---|---|
| **L1** | `rns` | `!rns.poly<N×q, coeff\|eval>` — one RNS limb | the HE **algebra** (per-limb poly math) | `ntt` `intt` `mul` `add` `sub` `base_convert` `crt_combine` `gadget_decompose` `const` |
| **L2** | `tile` | `tensor<M×N × i32>`, tiles ≤ T=32 | **kernel IR** — NTT as tile matmuls + corner-turns | `matmul_modq` `mul_modq` `add_modq` `all_to_all(_tiles)` `reshape` `transpose` `const` |
| **L3** | `tile` | `tensor<M×N × i8>` — radix-128 limbs | **int8 hardware** decomposition of the matmul | `decompose` `recombine_modq` (+ reuses `matmul_modq`/`add_modq` on i8) |

Lowering: **L1 → L2** expands each `rns.ntt`/`rns.base_convert` into tile matmuls, twiddles,
and `all_to_all` corner-turns (multi-factor decomposition, ≤T tiles). **L2 → L3** decomposes
every `tile.matmul_modq` into `1 decompose + 16 i8 matmul + 4 add + 1 recombine` (radix-128).
Only `matmul_modq` reaches L3; pointwise ops (`mul`/`add`) stay at L2 (SFPU).

### Conventions

- **L1 value:** `!rns.poly<N × q, dom>` — a single RNS limb: `N` coefficients, residues mod
  one prime `q`, in domain `dom ∈ {coeff, eval}` (`eval` = post-NTT / pointwise domain).
  Representation is **Montgomery throughout** (the NTT is `Z_q`-linear, so Montgomery rides
  through unchanged; see `docs/he/README.md` §3). An RNS ciphertext polynomial with `L` limbs
  is a *bundle* of `L` such values — so limb-parallelism is explicit (an NTT over an
  `L`-limb poly is `L` independent `rns.ntt` ops). A **ciphertext** is 2 polynomials
  (`c0`,`c1`) → `2L` values.
- **Multi-modulus — DECIDED: per-value `q`.** The prime lives on the **`Value`** (its type
  `!rns.poly<N × q>`), there is **no global `program.q`**, and `verify()` checks operand primes
  are consistent (and `base_convert`/`crt_combine` src/dst primes). This is forced: `base_convert`
  and `crt_combine` span multiple primes in one op, so a single-`q` program cannot even represent
  rescale / mod-switch. (Only relaxes the old `ir.py` `op.q == program.q` check.)
  - **Codegen contract:** kernels **never hardcode `q`**. `q` + its derived Montgomery scalars
    (`q_inv`, `q_low`, `q_high`) are **runtime args**; per-prime data (twiddles, folded matrix,
    evk) are **selectable input buffers**, chosen by the HE context. One prime-agnostic binary per
    op. This is already how `modmul` (and `modmul_rns_sfpi`) work.
  - **v1: one prime per dispatch** (scalar `q` arg). Mixed-prime packing (per-tile prime table,
    à la `modmul_rns_sfpi`) is a later optimization — see `formalization.md` §8.
- **L2/L3 value:** `tensor<M×N × i32>` (field element in `[0,q)`) or `tensor<M×N × i8>`
  (radix-128 limb). Tiles are ≤ `T` (v1: `T = 32`, square).
- **Data layout (physical; on the `Value`).** A tile has one of two layouts, transparent to
  the golden (which computes logical values) but load-bearing for lowering / codegen:
  - **`coeff`** (vector-slice, default) — the 1024 elements are coeffs of **one** limb. SFPU
    pointwise ops + NTT consume this; **one tile = one prime** (per-value `q` is a scalar).
  - **`limb`** (limb-major) — **rows are limbs**, cols are coeffs; **one tile spans a base**
    (`q` is a tuple of primes). Used by the **FPU/matmul `base_convert`** (CRT recombine as a
    matmul over the limb axis).
  `coeff ↔ limb` is **not free** — an explicit **relayout** op (all-to-all transpose,
  `roadmap.md` Track A.4) that the compiler inserts when a consumer's layout differs from the
  producer's, and the scheduler must see. v1 SFPU/NTT/matmul all use `coeff`; `limb` arrives
  with the FPU base-conversion path.
- **Domain tags** on L1 polys (`coeff`/`eval`) are compile-time; the interpreter is
  domain-agnostic (all Python ints), exactly like `ir.py`'s dtype.

---

## 1. L1 — RNS-poly ops (`rns`)

### 1a. Primitives (each maps to one device kernel, or is irreducible)

These are the L1 primitive vocabulary (ten ops). §4 shows the two demos need only **nine** of
them — all but `automorph`/`monomial`, which are future query-expansion / rotations.

---

**`rns.const`** — materialize a baked constant poly.
```
%w = rns.const { name, N, q, dom } : !rns.poly<N × q, dom>
```
- `name` — key into the constant table (twiddles, ψ-powers, evk, folded DB, gadget rows).
- `N`, `q`, `dom` — shape / prime / domain of the baked value.
- **Does:** emits a compile-time constant (folded + Montgomery-encoded at plan time; never
  computed on at runtime). Mirrors L2 `tile.const`.

---

**`rns.ntt`** — forward negacyclic NTT of one limb.
```
%X = rns.ntt %x { N, q } : !rns.poly<N × q, coeff> -> !rns.poly<N × q, eval>
```
- `N` — ring degree (`x^N + 1`). `q` — the limb prime (NTT-friendly, `k·2N+1`).
- **Does:** negacyclic number-theoretic transform (coeff → eval), so that a coefficient-wise
  `rns.mul` in the eval domain equals negacyclic polynomial multiplication. Lowers to the
  four-step (matmul → twiddle → matmul + reindex) at L2.

**`rns.intt`** — inverse NTT.
```
%x = rns.intt %X { N, q } : !rns.poly<N × q, eval> -> !rns.poly<N × q, coeff>
```
- Inverse of `rns.ntt` (eval → coeff), including the `N⁻¹` scale.

---

**`rns.mul`** — pointwise (Hadamard) modular multiply.
```
%c = rns.mul %a, %b { q } : (!rns.poly<N × q, D>, !rns.poly<N × q, D>) -> !rns.poly<N × q, D>
```
- `q` — modulus. Both operands (and result) share domain `D`.
- **Does:** `c[i] = a[i]·b[i] mod q`, coefficient-wise. The workhorse: eval-domain ct×ct,
  plaintext×ct, twiddle, ×evk, ×constant (one operand may be an `rns.const`). Device kernel:
  `modmul_sfpi_full.cpp` / `modmul_compute.cpp` (SFPU, Montgomery). Lowers to L2 `tile.mul_modq`.

**`rns.add`** — modular add (variadic, ≥2).
```
%s = rns.add %a, %b, ... { q } : (!rns.poly<N × q, D>, ...) -> !rns.poly<N × q, D>
```
- `q` — modulus. **Does:** `s[i] = (Σ operands[i]) mod q`. Used by HE add, `e1 = c0·d1 +
  c1·d0`, MAC accumulation, mux `+ acc0`. Kernel: `modadd_reduce.cpp`. → L2 `tile.add_modq`.

**`rns.sub`** — modular subtract.
```
%d = rns.sub %a, %b { q } : (!rns.poly<N × q, D>, !rns.poly<N × q, D>) -> !rns.poly<N × q, D>
```
- `q` — modulus. **Does:** `d[i] = (a[i] − b[i]) mod q`. Used by the rescale floor
  correction, mod-down, and the mux difference `acc1 − acc0`. Kernel: `modsub_reduce.cpp`.

---

**`rns.base_convert`** — fast RNS base conversion (FBC).
```
%y:Ld = rns.base_convert %x:Ls
        { src_base=[q…], dst_base=[p…], exact } : (Ls × !rns.poly<N,coeff>) -> (Ld × !rns.poly<N,coeff>)
```
- `src_base` — the `Ls` source primes; `dst_base` — the `Ld` destination primes.
- `exact` (bool) — `false` = approximate (fast, small error); `true` = exact via the `m_sk`
  sentinel correction.
- **Does:** given `x`'s residues in base A, produce its residues in base B — the CRT
  reconstruction `x = Σ_i (x_i·(Q/q_i)⁻¹ mod q_i)·(Q/q_i)`, remapped to base B. Multi-in /
  multi-out. Kernel: `baseconv_fbc.cpp` (+ `_sk`). **Constraint:** correct only at width
  `Ls,Ld ∈ {2,4,8}` (the n≠4 bug) — v1 pads to 4 (`tt_fused_fbc_n4_bug`). Also expressible as
  per-dst-limb `rns.mul`-by-constant + `rns.add` (how `he_api.cpp` builds rescale today).

---

**`rns.crt_combine`** — CRT/Garner reconstruction to a wider residue.
```
%w = rns.crt_combine %x0, %x1 { src_base=[q1,q2], out_q } : (2 × !rns.poly<N,coeff>) -> !rns.poly<N × out_q, coeff>
```
- `src_base` — the source primes (PIR: two 28-bit primes). `out_q` — the wider output modulus
  (or a bigint-limb pair) the combined value lands in.
- **Does:** reconstruct the integer `x` from its RNS residues via Garner (`crt_mul_split` +
  `garner_diff`), i.e. `2×28 → 56-bit` in PIR (#13). Used by mod-switch. Kernel:
  `crt_mul_split.cpp`.

---

**`rns.gadget_decompose`** — signed gadget (digit) decomposition.
```
%d:ell = rns.gadget_decompose %x { base_B, ell, signed=true, q } : !rns.poly<N × q> -> (ell × !rns.poly<N × q>)
```
- `base_B` — the gadget base (PIR: `2²⁰`). `ell` (ℓ) — number of digits. `signed` — digits in
  `[−B/2, B/2)` (keeps the external-product noise low).
- **Does:** write each coefficient as `x = Σ_{j<ℓ} d_j·B^j` with signed digits (#9). Multi-out.
  Kernel: `gadget_decompose_signed.cpp`. (Trivial split into RNS = `digit_to_rns`.)

---

**`rns.automorph`** — Galois automorphism (ring substitution).  *(block-library; not used by
the current demos — needed for query expansion #18)*
```
%y = rns.automorph %x { k, N } : !rns.poly<N × q, coeff> -> !rns.poly<N × q, coeff>
```
- `k` — the Galois element: `σ_k : a(X) ↦ a(X^k)` with the negacyclic sign fold.
- **Does:** coefficient permutation + sign (#14). A companion `rns.monomial %x {j}` multiplies
  by `X^j` (#15). Implemented as a host/reindex permutation today. Listed for completeness;
  §4 shows the two demos do **not** require it.

### 1b. Composites (macros — lower to §1a primitives; also serve as `prov` cluster labels)

Each is a named subgraph — **no new arithmetic of its own**; every composite expands down to
the §1a primitives (some *via* other composites: `relin` uses `mod_up`/`mod_down`, `mux` uses
`extern_product` — the nesting always terminates at 1a). The host emits these when building
the L1 program, and each name becomes the provenance group for the viewer / clustering. **Sole
exception:** `rns.mod_switch`'s device part is `crt_combine` (1a), but its final `round(q'·c/q)`
runs on the **host** (true integer division, not modular) and is deliberately not an IR op.

| Op | Expands to | Meaning / where |
|---|---|---|
| `rns.ct_mul_raw` | per-limb `ntt`×4 → `mul`/`add` (e0,e1,e2) → `intt` | raw ct×ct → degree-2 product, pre-rescale/relin (BFV multiply step 1; device `bfv_mul_raw.cpp`) |
| `rns.rescale` | `base_convert`×2 + `mul` + `sub` | `⌊t·e/Q⌉` BEHZ fast-floor (multiply step 2) |
| `rns.mod_up` / `rns.mod_down` | `base_convert` (+ `sub` for down) | extend Q→QP / reduce P→Q (relin edges) |
| `rns.relin` (= key-switch) | `mod_up` → `ntt` → `mul`(×evk) → `intt` → `mod_down` | relinearize e2 under s² (multiply step 3) |
| `rns.initial_dim` | `mul` + `add` (MAC) over k | `Σ_k DB[k]·query[k]` (PIR #12) |
| `rns.extern_product` | `gadget_decompose` → `ntt` → `mul`(×RGSW) → `add` → `intt` | `C ⊡ d` additive-noise product (PIR #16) |
| `rns.mux` | `sub` (acc1−acc0) → `extern_product` → `add` (+acc0) | hypercube fold `mux(C,x,y)=C⊡(y−x)+x` (PIR #19) |
| `rns.mod_switch` | `crt_combine` (device) + scale-round (**host**) | `round(q'·c/q)` response shrink (PIR #11) |

The scale-round in `mod_switch` and BFV encrypt/decrypt/keygen are **host** (see
`formalization.md` scope decision) — they are not L1 ops.

---

## 2. L2 — tile / kernel ops (`tile`)

Verbatim from `reference/ntt_matmul/ir.py` (opcodes, attrs, shape rules). These are the
**kernel IR** — the single mapping level (`formalization.md` §2.0). Values are
`tensor<M×N × i32>` mod the program `q`, tiles ≤ `T`.

**Compute**

- **`tile.matmul_modq`** — `%r = tile.matmul_modq %a, %b { q, transpose_a?, transpose_b? } : … -> tensor<M×N × i32>`
  · `q` modulus (must match program q) · `transpose_a/b` (bool) transpose that operand first ·
  **does:** `R = (Aᵀ?·Bᵀ?) mod q`. Runs on the FPU/matrix unit → decomposes to L3 (§3).
- **`tile.mul_modq`** — `%c = tile.mul_modq %a, %b { q } : … ` (numpy broadcast) ·
  **does:** `C = (A⊙B) mod q`, Hadamard. SFPU. (L1 `rns.mul` lowers here.)
- **`tile.add_modq`** — `%s = tile.add_modq %a, %b, … { q }` (≥2 operands, broadcast) ·
  **does:** `S = (Σ operands) mod q`. Also the K-tile / per-plane accumulate.

**Layout / structural** (no arithmetic; mostly free views)

- **`tile.const`** — `%c = tile.const { name, shape }` — materialize a baked constant tile;
  `name` indexes `program.consts`.
- **`tile.reshape`** — `%y = tile.reshape %x { shape }` — free view; element count must match.
- **`tile.transpose`** — `%y = tile.transpose %x` — full 2D transpose `(m,n)→(n,m)`.
- **`tile.slice`** — `%y = tile.slice %x { axis, start, size }` — contiguous sub-range (view).
- **`tile.split`** — `%a,%b,… = tile.split %x { axis, parts }` — one tensor → `parts` equal
  tiles along `axis` (multi-out); `dim % parts == 0`.
- **`tile.concat`** — `%y = tile.concat %a,%b,… { axis }` — join along `axis` (off-axis dims
  must match).
- **`tile.all_to_all`** — `%o… = tile.all_to_all %i… { concat_axis, reshape_to, split_axis,
  parts }` — reshape-based cross-stage retile (`concat → reshape → split`); appears when tiles
  aren't square `T×T`.
- **`tile.all_to_all_tiles`** — `%o… = tile.all_to_all_tiles %i… { perm, transpose? }` —
  pure tile permutation/gather: `outputs[k] = inputs[perm[k]]`, optionally per-tile
  transposed. **This is the NoC corner-turn** — under a placement it becomes physical routing
  (`formalization.md` §0). `perm` length == #results; `transpose` is a per-result bool list.

Only `tile.slice`/`tile.concat` at the very begin/end cut/feed real data; no `slice` ever cuts
a compute result and no `concat` ever feeds a compute op (an `ntt_matmul` invariant).

---

## 3. L3 — int8 decomposition (`tile`, matmul-only)

Only `tile.matmul_modq` lowers here (`formalization.md` §1). Values gain `i8` limbs.

- **`tile.decompose`** — `%a0,%a1,%a2,%a3 = tile.decompose %x { radix=128, nlimbs=4 } :
  tensor<M×N × i32> -> 4 × tensor<M×N × i8>` · **does:** radix-128 limb split
  `a_p = (x >> 7p) & 127`. Multi-out; each result shares `x`'s shape.
- **`tile.recombine_modq`** — `%r = tile.recombine_modq %C0,%C1,%C2,%C3 { radix=128, q } :
  4 × tensor<M×N × i32> -> tensor<M×N × i32>` · **does:** `R = (Σ_r radix^r · C_r) mod q`.
  All operands share shape; `q` = program q.
- The int8 matmul itself **reuses `tile.matmul_modq`** on `i8` operands (products `< q`, so the
  `mod q` is an exact no-op), and the per-plane K-accumulate **reuses `tile.add_modq`** — no
  new opcodes. (v1 uses Way 1; Way 2 recombine-hoist is deferred, `formalization.md` §8.)

Expansion of one `tile.matmul_modq` (parameter pre-folded to 16 baked limb consts at plan
time): `1 decompose + 16 i8 matmul_modq + 4 add_modq + 1 recombine_modq` — zero runtime
`mul_modq`, nothing reads a constant (`ntt_matmul/PLAN.md` §4).

---

## 4. Coverage — the two applications, built from L1 ops

Proof that the op set is sufficient. `[i]` ranges over RNS limbs; primitives only.

### `he.add(a, b)` — `lib/he/he_api.cpp::add`
```
out.c0[i] = rns.add a.c0[i], b.c0[i]   { q=Q[i] }      for i in 0..LQ
out.c1[i] = rns.add a.c1[i], b.c1[i]   { q=Q[i] }
```
Ops used: **`rns.add`**.

### `he.multiply(A, B)` — `he_api.cpp::multiply` (ct_mul_raw → rescale → relin → add)
```
# ct_mul_raw (per limb q over the 8-limb base Q∪S∪{m_sk}):    rns.ct_mul_raw
  C0 = rns.ntt c0; C1 = rns.ntt c1; D0 = rns.ntt d0; D1 = rns.ntt d1
  E0 = rns.mul C0,D0 ; E2 = rns.mul C1,D1
  E1 = rns.add (rns.mul C0,D1), (rns.mul C1,D0)
  e0 = rns.intt E0 ; e1 = rns.intt E1 ; e2 = rns.intt E2
# rescale each e_k  ⌊t·e/Q⌉ :                                  rns.rescale
  = rns.base_convert (×2, exact via m_sk) + rns.mul + rns.sub
# relinearize e2 :                                             rns.relin
  mod_up   = rns.base_convert (Q→QP)
  ×evk     = rns.mul (rns.ntt …, evk) ; then rns.intt
  mod_down = rns.base_convert (P→Q) + rns.sub
# combine:
  out.c0[i] = rns.add ẽ0[i], k0[i] ; out.c1[i] = rns.add ẽ1[i], k1[i]
```
Ops used: **`rns.ntt`, `rns.intt`, `rns.mul`, `rns.add`, `rns.sub`, `rns.base_convert`,
`rns.const`** (evk / twiddles).

### `pir.server_answer(case)` — `lib/pir/pir_demo_api.cpp::server_answer`
```
# pre-transform DB / selectors / RGSW:                         rns.ntt, rns.const
# (b) initial dimension  acc = Σ_k sel[k] ⊙ DB[k] :            rns.initial_dim
  acc = rns.add ( rns.mul sel[k], DB[k] for k )                 (MAC)
# (c) hypercube fold  ans = mux(C1, acc0, acc1) :              rns.mux
  d   = rns.sub acc1, acc0
  ep  = rns.extern_product C1, d
        = rns.gadget_decompose d -> rns.ntt -> rns.mul(×RGSW) -> rns.add -> rns.intt
  ans = rns.add ep, acc0
# (d) mod-switch each component  round(q'·c/q) :               rns.mod_switch
  w = rns.crt_combine ans[·q1], ans[·q2]   ; then host scale-round
```
Ops used: **`rns.ntt`, `rns.intt`, `rns.mul`, `rns.add`, `rns.sub`, `rns.gadget_decompose`,
`rns.crt_combine`, `rns.const`**.

### Result

The union over both demos is exactly **9 primitive L1 ops** —
`const, ntt, intt, mul, add, sub, base_convert, crt_combine, gadget_decompose` — plus the
composites that name their subgraphs. **`rns.automorph`/`rns.monomial` are *not* needed by the
current demos** (they belong to query expansion / rotations, future work). Everything lowers
through the L2 `tile` ops (with `tile.matmul_modq` → L3), all of which already exist and are
verified in `ntt_matmul`.

---

## 5. Op → device kernel map (verified against `kernels/`)

Every op below was checked against the actual kernel tree. **Result: every compute op has an
existing, silicon-validated kernel** (sometimes a different name, or built from a few kernels).
Two expected exceptions: *view* ops need no compute kernel, and `automorph` is host-only.

| Op | Existing kernel(s) | Status |
|---|---|---|
| `rns.mul` / `tile.mul_modq` | `modmul_sfpi_full`, `modmul_compute`, `modmul_sfpi_hybrid`, `modmul_rns_sfpi` (multi-modulus) | ✅ direct |
| `rns.add` / `tile.add_modq` | `modadd_reduce` | ✅ direct |
| `rns.sub` | `modsub_reduce` | ✅ direct |
| `rns.ntt` / `rns.intt` | `fused_matmul_a` + `fused_matmul_c` + twiddle `modmul_compute` + `transpose_tile` | ✅ composite (not one kernel) |
| `rns.base_convert` | `baseconv_fbc` (+ `_sk`); FPU path `matmul_fbc`(`_packed`) | ✅ direct (two impls) |
| `rns.crt_combine` | `crt_mul_split`, `garner_diff` | ✅ direct |
| `rns.gadget_decompose` | `gadget_decompose_signed` (+ `digit_to_rns`) | ✅ direct |
| `rns.const` / `tile.const` | — baked data (host DRAM write + reader) | ✅ no compute kernel needed |
| `tile.matmul_modq` (i8) | `matmul_compute`(`_mt`/`_mtk`), `matmul_plane` | ✅ direct (several tilings) |
| `tile.decompose` | `decompose_limbs` | ✅ direct |
| `tile.recombine_modq` | `recombine_compute`, `recombine_bigK`, `recombine_rns` | ✅ direct |
| `tile.transpose` | `transpose_tile` | ✅ direct |
| `tile.all_to_all` / `all_to_all_tiles` | `all_to_all_corner`(`_mc`) | ✅ direct |
| `tile.reshape` / `slice` / `split` / `concat` | — views, realized in reader/writer addressing | ✅ no compute kernel by design |
| `rns.automorph` / `rns.monomial` | — host reindex (`pir_permute.hpp`) | ⚠️ host-only, no device kernel (not needed by current demos) |

So the **9 primitives the demos need**, and all L2/L3 ops, map to kernels that already run
bit-exact on the n150 (`docs/he/kernels.md`, `docs/pir/pir_demo.md` §4). The IR lowers to a
pre-validated target — the only op lacking a device kernel (`automorph`) is unused by
`he_demo`/`pir_demo`. Bonus: `modmul_rns_sfpi` already handles per-tile multi-modulus, which
supports the multi-prime L1 → L2 lowering (see Conventions).
