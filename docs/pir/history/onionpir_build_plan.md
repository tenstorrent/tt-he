# Plan: building OnionPIR v2 (single-server PIR) on our BFV/TT stack

Based on **OnionPIR v2** (ePrint 2025/1142, Chen–Ren). Single-server PIR that composes **BFV
(RLWE)** with **RGSW** via the **external product** for noise control, over a database laid out as a
multi-dimensional hypercube. This note answers: *given our BFV ops, what else must we build?*

---

## What is PIR? (in plain terms)

**Private Information Retrieval** lets you look up one entry in a server's database **without the
server learning which entry you asked for**.

The everyday picture: imagine a **library** where you want one specific book, but you don't want the
librarian to know which book interests you (say it reveals your health, politics, or finances). The
*trivial* private solution is to **borrow the entire library**, read your book at home, and return
everything — the librarian learns nothing about your interest, but you've hauled the whole library
back and forth. That's perfect privacy at absurd cost. **PIR is the cryptographic way to get just
your one book while the librarian still learns nothing — and you only download about one book's
worth of data, not the whole library.**

Why is that even hard? To keep your choice hidden, the server must **touch every entry** (if it
skipped some, it would learn your answer isn't among them) — yet it must return **only your entry**,
in a form it **cannot read itself**. Homomorphic encryption is what makes that possible: you send an
**encrypted** query, the server **computes on the ciphertext** (it never decrypts), and it returns an
encrypted answer only you can open.

Real, deployed-or-proposed examples people can relate to:
- **"Was my password leaked?"** — check your password against a breach database without revealing it.
- **Private Safe Browsing** — ask "is this website malicious?" without telling the provider which
  sites you visit.
- **Private contact discovery** — "is my friend on this app?" without uploading your whole address book.
- **Private lookups** in medical, legal, patent, or stock databases, where *the query itself* is sensitive.

OnionPIR is a **single-server** PIR scheme (one server does all the work, using HE) — which is what
makes it a clean, self-contained target for a single accelerator.

## Why PIR is a good demo application for TT-accelerated HE

PIR is arguably the **best near-term showcase** for Tenstorrent HE support, for several reasons:

1. **Its bottleneck is exactly TT's strength.** The dominant cost (the initial-dimension dot product,
   §4) is a huge **modular multiply-accumulate over the whole database** — i.e. **NTT + modular
   matmul**, the precise kernels we've built and optimized. The headline PIR metric, **throughput =
   database-bytes ÷ server-time**, is *directly* improved by faster modular arithmetic on TT. A
   speedup on our modmul/matmul/NTT translates 1:1 into that throughput.
2. **Compute-bound and embarrassingly parallel.** The server must scan the entire database, and that
   scan is a massively parallel dot product — a perfect fit for TT's many-core throughput and exactly
   the kind of `C = 1…32`-core scaling story we already measured for the NTT.
3. **Achievable with what we have — it skips the hardest HE op.** OnionPIR replaces BFV ct×ct
   multiplication (the `t/Q`-scaling + relinearization monster) with the additive-noise **external
   product**. So a working PIR demo needs the *easy* BFV ops plus one new primitive (gadget
   decomposition, §2) — a realistic milestone, not a full-FHE moonshot.
4. **A clean, tight benchmark loop.** The plaintext database is public, so its NTT is **precomputed
   offline**; the online server work reduces to a batched modular multiply-accumulate against those
   precomputed tables — a well-defined, measurable kernel that's easy to profile and compare against
   CPU/GPU baselines.
5. **Single server, single accelerator, end-to-end story.** No multi-party coordination; one device
   takes an encrypted query and returns an encrypted answer. The demo is concrete and legible:
   *retrieve the correct entry privately, at X MB/s* — a number a non-expert immediately understands.
6. **Real-world relevance.** PIR is a deployed, commercially meaningful primitive (password checkup,
   safe browsing, private discovery). Demonstrating it makes the case that TT can accelerate *useful*
   privacy technology, not just an HE microbenchmark.

In short: PIR turns "TT does modular matmul/NTT fast" into "TT serves private queries over a real
database fast" — the same kernels, wrapped in an application whose headline metric is throughput,
which is precisely what the hardware delivers.

## 0. The good news: PIR needs the EASY BFV ops, not the hard one

OnionPIR **never uses BFV ciphertext×ciphertext multiplication** — it replaces it with the external
product (additive noise instead of multiplicative). So **the hardest BFV op we planned** (the
`⌊t/Q⌉` scaling + hybrid key-switch relin, see `docs/he/bfv_build_plan.md`) **is not on the PIR
path at all.** The BFV ops PIR actually uses:

| BFV op | role in PIR | status in our plan |
|---|---|---|
| ciphertext **add** | dimension accumulation, expansion tree | easy (have) |
| **plaintext×ciphertext** multiply | the **initial-dimension dot product** (the throughput bottleneck) | easy (have) |
| **key-switch** (BV-style, gadget) | Galois `Subs` in query expansion | needs the gadget primitive (§2) |
| **modulus switching** | shrink the final response | planned (mod-switch) |
| NTT / INTT, mul_const | everywhere | have / planned |

So PIR is *easier* on the BFV side than general BFV. The new work is the **RGSW + external product +
Galois** family below.

---

## 1. The server pipeline (what we must run)

```
client → 1 packed BFV query ciphertext + key materials (Galois keys, RGSW(s))
server:
  (a) QueryUnpack:  expand the 1 packed ct → N0 BFV cts (initial dim)
                                            + RGSW cts (one per higher dim)   [§3]
  (b) Initial dimension:  response = Σ  DB_plaintext · query_BFV   (plaintext×ct dot product) [§4]
  (c) Higher dimensions:  fold the hypercube with EXTERNAL PRODUCTS,
                          one per 1-of-2 selection via  b·(y−x)+x            [§2,§4]
  (d) Modulus-switch the response → send back
```

The bulk of compute is **(b)**, the initial dimension (`~2N` poly mults).
**(a)** and **(c)** are logarithmic in `N` but carry the noise.

---

## 2. The core new primitive: gadget-decompose + NTT multiply-accumulate

**One primitive gives three of the new operations.** The external product, BV key-switch, and the
Galois `Subs` all have the identical shape:

```
G⁻¹(·): signed gadget-decompose a poly into ℓ base-B digits (each digit a small poly)
        → NTT each digit → pointwise-multiply-accumulate against precomputed key/RGSW NTTs → sum
```

- **External product** `RGSW(2ℓ×2) ⊡ BFV(2)`: decompose the 2 BFV polys into `2ℓ` digits, multiply by
  the `2ℓ×2` RGSW matrix (rows = BFV encryptions, kept in NTT), accumulate → 2 output polys.
  Cost ≈ `2ℓ` digit-NTTs + `4ℓ` pointwise-mul-acc. **Additive** noise (the reason OnionPIR is fast).
- **BV key-switch** (used for `Subs`): decompose one component into `ℓ` digits, multiply by the
  key-switch key matrix, accumulate. Same structure.

So **build once**: `gadget_decompose` (signed, base-`B`, `ℓ` digits) + the existing NTT pointwise
mul-accumulate against key material in NTT form. That single kernel underlies external product,
key-switch, and `Subs`.

**New kernels here:**
- `gadget_decompose_signed(poly, B, ℓ)` → `ℓ` digit-polys in `[−B/2, B/2)`. Generalizes our
  radix-128 `decompose_limbs` (different base, signed, `ℓ` configurable). **This is the main new
  arithmetic kernel.**
- RGSW / key-material data type, stored in NTT form (offline).
- The mul-accumulate is `rns_mul_pointwise` + `rns_add` — already in the BFV layer.

---

## 3. Query expansion (the other new family)

Expands one packed BFV ciphertext into the `N0` initial-dim cts and the higher-dim RGSW cts
(SealPIR-style, Algorithms 1–3). It is a **tree of three small ops**:

| op | what it is | new kernel? |
|---|---|---|
| `Subs(c, j)` (automorphism) | apply `x → xʲ` (coefficient permutation, negacyclic) **then BV key-switch** back to `s(x)` | **Galois permute** (data movement) + key-switch (§2) |
| monomial mult `· x^{±k}` | negacyclic coefficient shift (+ sign) | tiny **permutation kernel** |
| add | ciphertext add | have |
| RLWE→RGSW | `ExternalProduct(RGSW(s), c)` to derive the bottom `ℓ` RGSW rows | reuses §2 |

**New kernels here:**
- **Galois automorphism** `x→xʲ`: a coefficient **index permutation** `i ↦ i·j mod 2n` with a sign
  flip on wrap. On TT this is a gather/reindex — reuse the corner-turn / reindex machinery.
- **Monomial multiply** `x^{±k}`: a cyclic shift of coefficients with negacyclic sign — cheap permute.
- The **Galois keys** (one key-switch key per automorphism `j` used in the tree) are client-provided
  key material, stored in NTT form.

Everything else in expansion is composition (the tree schedule in Algorithms 2–3).

---

## 4. Database layout & dimension folding

- **Hypercube**: `N` entries as `N0 × 2 × 2 × …` (initial dim ~ a few hundred, rest = 2). Offline.
- **Initial dimension = a big modular matmul** (see §5): the response is `Σ_k DB[k]·query[k]`, i.e. the
  `N0`-long query-ciphertext vector times the plaintext-database matrix, in NTT domain. The plaintext
  DB is **public → precompute its NTT once**. Hot loop = batched **pointwise-mul-accumulate** of query
  NTTs against precomputed DB NTTs — the throughput-critical kernel.
- **Higher dimensions**: one **external product** per 1-of-2 selection using the `b·(y−x)+x` trick
  (one ext-product instead of two). Folds the hypercube down to the answer.

---

## 5. TT-specific mapping (and the one real challenge)

**Wins — reuse what we built:**
- **Initial dimension ⇒ our modular matmul / NTT.** The dot product is exactly an NTT-domain
  matmul (query vector × precomputed plaintext-DB), the same shape we optimized for the NTT. This is
  the part that sets the throughput number — and it's our strongest kernel.
- **Galois permute / monomial shift ⇒ our reindex / corner-turn data-movement** machinery.
- **Modulus switching, NTT, pointwise modmul, add ⇒ already planned/built.**

**The real challenge — gadget decomposition vs. our 28-bit RNS:**
OnionPIR uses a **~60-bit composite modulus `q = q1·q2`** (CRT) precisely so the bulk multiplies are
`32×32→64`. **Our 28-bit RNS *is* that composite representation** — natural fit for the initial
dimension. But **gadget decomposition wants the *full* `q` value**, which in RNS is split across
limbs → it would need repeated CRT/inverse-CRT around every `G⁻¹(·)`. OnionPIR's answer is the
**composite-NTT** technique (Spiral/Faster-Spiral [21,23]): CRT/RNS in the initial dimension, but
treat `q` as a *single* modulus in the higher dimensions so gadget decomposition avoids per-digit CRT.
**Decision for us:** adopt composite-NTT — RNS-matmul the initial dimension, and run the external-
product dimensions in a single-modulus view (our 28-bit limbs reconstructed to the ~56-bit `q` only
where the gadget decompose needs it). This is the main design point to nail.

**Ring degree:** PIR uses `n ≥ 2048` (typically 2048/4096); we have NTT-1024 and NTT-32768. **Need an
NTT at the PIR ring size** (e.g. 2048 = 32×64 four-step, or 4096) — same machinery, new parameters.

**Other standard tricks to carry over (all align with what we do):** signed gadget decomposition
(`[−B/2,B/2)`, lower noise), **BV-style key-switch** (no hybrid mod-up/down — simpler, gadget-based),
delayed/lazy modular reduction (we already do lazy `[0,2q)`), pseudorandom-seed for the random ct
component (client-side request-size trick).

---

## 6. The full "new kernels beyond BFV" list

| new building block | enables | nature |
|---|---|---|
| **`gadget_decompose_signed(B, ℓ)`** | external product, BV key-switch, Subs | base-B signed digit extraction (extends `decompose_limbs`) |
| **gadget mul-accumulate** (NTT) | same | `rns_mul_pointwise` + `rns_add` vs. key NTTs (have) |
| **RGSW type + external product** | higher dimensions, RLWE→RGSW | composition over the two above |
| **Galois automorphism** `x→xʲ` | `Subs` / query expansion | coefficient permutation (reindex/gather) |
| **BV key-switch** | `Subs`, RLWE→RGSW | gadget-decompose + key-mul-acc |
| **monomial mult** `x^{±k}` | expansion tree | negacyclic coefficient shift |
| **NTT at PIR ring size** (2048/4096) | everything | four-step NTT, new params |
| **composite-NTT bridge** | RNS init dim ↔ single-mod gadget dims | RNS↔single-modulus reconciliation (§5) |

Plus the schedules (QueryUnpack tree, hypercube fold) which are pure composition.

---

## 7. Build order (validate each vs. a CPU reference)

1. **Params:** ring `n` (2048/4096), composite `q = q1·q2` from two ~28-bit NTT-friendly primes, plaintext
   `t`, RGSW base `B` + `ℓ` (two sets: larger `ℓ` for expansion, smaller for higher dims). Build the
   **NTT at `n`**.
2. **`gadget_decompose_signed` + gadget mul-accumulate** → **external product**. Validate ext-product
   noise + correctness vs. a CPU RGSW⊡RLWE reference. *(Core milestone — unblocks everything.)*
3. **Galois automorphism + BV key-switch** → **`Subs`** → **ExpandBFV / QueryUnpack**. Validate the
   expansion produces the right per-bit ciphertexts.
4. **Initial dimension** as the RNS modular matmul over the precomputed plaintext-DB NTT; **higher
   dimensions** via external products + `b·(y−x)+x`. Assemble the **full PIR query→response**.
5. **Modulus-switch** the response; end-to-end correctness (retrieve the right entry) + throughput.
6. Optimize: composite-NTT to keep the init dim in RNS-matmul while gadget dims run single-modulus;
   keep all key material (Galois keys, RGSW, DB) in NTT form; lazy reduction.

---

## 8. One-line summary

Given BFV (add, plaintext×ct, key-switch, mod-switch, NTT), building OnionPIR v2 needs **one new
arithmetic primitive — signed gadget decomposition + NTT multiply-accumulate against key material —
which yields the external product, BV key-switch, and Galois `Subs`**; plus the **Galois automorphism
/ monomial permutations** and the **query-expansion + hypercube-fold schedules**. The throughput-
critical **initial dimension is a modular matmul** (our strongest kernel), and the main TT-specific
design point is the **composite-NTT bridge** between our 28-bit RNS (great for the matmul) and the
single-modulus view the gadget decomposition wants. Notably, PIR **skips** the hardest BFV op
(ct×ct multiply / `t/Q` scaling / relin) entirely.
