# PIR on Tenstorrent — primer, storage model, and per-card sizing

**Purpose.** A self-contained write-up of what PIR is, why OnionPIR is the right scheme for a
Tenstorrent accelerator, how data is actually stored and processed, and **how big a workload one card
can realistically serve, and why capacity scales linearly with cards.** Written to be read start-to-finish by someone new to the project.

Companion docs (the "how it's built" detail): `history/onionpir_build_plan.md`,
`pir_external_product_deepdive.md`, `pir_special_kernels.md`, and the measured device numbers in
`../../docs/pir/pir_perf.md` / `pir_demo.md`. Scheme: OnionPIR v2, ePrint 2025/1142 (Chen–Ren).

> **Status honesty up front.** Everything correctness-critical runs on device today (the `pir_demo`
> retrieves the right entry, bit-exact to the CPU golden). The throughput/sizing figures in §7 are a
> **projection** built on one *measured* multi-core rate (~3.5 G modmul/s) plus a back-of-envelope —
> the real per-card `bytes/s` benchmark over a large database is the recommended next build step (§8).
> Numbers are from a single **n150 (Wormhole)** running **unoptimized proof-of-concept** code.

---

## 1. What PIR is (the idea)

**Private Information Retrieval** = look up one record in a server's database *without the server
learning which record you asked for*.

The tension is that those two goals seem contradictory: normally, to answer "give me record #5," the
server must know you wanted #5. PIR breaks that link — it returns record #5, but genuinely cannot
tell whether you asked for #5, #500, or #5000.

The mental model: you want one book from a library, but which book you pick is private. The
dumb-but-private solution is to check out the *entire* library, read your book at home, and return
everything — the librarian learns nothing, but you hauled the whole library home. PIR is the
cryptographic version: you get the effect of "touched everything" so nothing leaks, but you only
download one book's worth of data.

**Why it is inherently hard / expensive.** For your choice to stay hidden, the server *must touch
every entry*. If it ever skipped some records, the skip itself would leak information. So there is no
shortcut: the server does work proportional to the whole database on every query. Making that
per-entry work fast is the entire game — and that work is exactly what TT accelerates.

---

## 2. PIR landscape — why FHE, why OnionPIR

### Two branches (the split is about trust, not just speed)

| | Multi-server (information-theoretic) | Single-server (HE-based) |
|---|---|---|
| Servers | ≥2, must **not collude** | 1 |
| Privacy from | secret-sharing (unconditional) | cryptographic hardness (lattices / LWE) |
| Server compute | cheap (XOR / shares) | expensive (homomorphic) |
| Deployment friction | finding independent non-colluding servers | making the crypto fast |

**The benefit of using homomorphic encryption** is that it removes the non-collusion assumption: one
server, standard cryptographic privacy, no trusted second party. That is what makes single-server PIR
deployable (it is what password-checkup and safe-browsing ship). The cost you accept is computation —
homomorphic arithmetic over the whole database — which is precisely the bet here: buy the strong
trust model, pay it back in silicon.

Note you do **not** need *fully* homomorphic encryption (arbitrary unbounded computation). You need
*just enough*: additions plus a controlled amount of multiplication. Less homomorphic machinery →
smaller parameters → less data moved and less compute. A good PIR scheme does the job with the least
machinery possible.

### The central problem inside HE-based PIR

Every HE-PIR scheme fights the same two-front battle:

- **Upload (query size).** Naively one ciphertext per database slot → query as big as the DB. Fixed
  by *query expansion* (SealPIR): client sends one small packed ciphertext, server expands it. (§3.3)
- **Download (answer size / "rate").** To scale, the DB is a multi-dimensional hypercube folded
  dimension by dimension — but folding requires *multiplying* ciphertexts, and **BFV ct×ct multiply
  multiplies noise**. Deep folds force a big modulus to leave room for noise → fat ciphertexts → poor
  download rate. This is the killer.

### OnionPIR's advantage

OnionPIR folds the hypercube with the **RGSW external product** instead of a BFV ct×ct multiply. The
external product decomposes one operand into *tiny* gadget digits before multiplying, so its noise
growth is **additive, not multiplicative**. That one property cascades:

- **Deep folds stay cheap** — noise grows linearly in depth ("onion" = thin layers peeled one at a
  time).
- **Small modulus → excellent download rate** — the answer ciphertext stays small.
- **Skips the hardest FHE op** — no `⌊t/Q⌉` rescale + relinearization. This is why the project could
  build working PIR without a full-FHE stack: it needs the *easy* ops (add, plaintext×ct, key-switch)
  plus one new primitive (signed gadget decomposition).
- **Small upload too**, via inherited SealPIR-style query expansion. Wins on both fronts at once.

The structure: the **initial (large) dimension** is a plaintext×ciphertext dot product (cheap noise,
no ct×ct, the throughput bottleneck); only the **higher dimensions** use external products, and those
are logarithmically few. Bulk of the work is the cheapest operation.

**v2 (the version followed here)** adds a *composite modulus* `q = q₁·q₂` (~56–58 bit from two
~28–29-bit primes) + the *composite-NTT* technique, so the bulk multiplies run as fast `32×32→64`
ops. This lands naturally on TT: the composite design keeps **every modular reduction ≤ 28-bit** —
exactly the width the int8-matmul-based modmul supports.

---

## 3. How data is stored

### 3.1 The database — plaintext, enlarged ~7×

**Key point: the database is NOT encrypted.** In PIR the records are public to the server; it is the
*query* that is encrypted. Each record is stored as a **plaintext polynomial** in
`Z_t[x]/(x^N+1)` — up to `N` coefficients, each a value mod the plaintext modulus `t`. One polynomial
packs up to `N` values (a whole record payload), not one number. *(The demo uses a constant
polynomial — value in coeff 0, rest zero — for legibility; a real DB packs all `N` coefficients.)*

The storage overhead is **not** ciphertext expansion. It comes from representing each plaintext
coefficient in the form the homomorphic arithmetic needs: lifted into the ciphertext modulus `q`,
split into RNS residues (per prime channel), NTT'd offline.

```
storage expansion = log₂q / log₂t ≈ 56 / 8 ≈ 7×      (with q ≈ 56-bit, t = 257)
```

It is the **ratio of bit-widths, not q/t.** `q/t ≈ 2⁴⁸` is a different quantity — the BFV scaling
factor `Δ = ⌊q/t⌋`, i.e. how much the plaintext is lifted in *magnitude*.

| quantity | formula | value | meaning |
|---|---|---|---|
| storage expansion | `log₂q / log₂t` | ~7× | bytes stored ÷ bytes of payload |
| noise budget (bits) | `log₂q − log₂t` | ~48 bits | headroom before decryption fails |
| scaling factor `Δ` | `q / t` | ~2⁴⁸ | magnitude the plaintext is lifted by |

**Elegant consequence:** the expansion factor *is* the noise budget. Push `t` up toward `q` → less
storage overhead but less computation headroom; push `t` down → more headroom, fatter DB. Choosing
`t` relative to `q` **is** choosing that trade. OnionPIR's additive noise lets you run a larger `t` /
smaller `q`, which directly shrinks the footprint.

### 3.2 The query — ciphertext, huge per bit, but tiny information

The query encodes the wanted **index**, *positionally*, not as a number:

- **Initial dimension → one-hot selector.** Coordinate `c0` along the big axis becomes a one-hot
  vector (a `1` at position `c0`, `0` elsewhere); each entry is separately encrypted as a BFV
  ciphertext (`= (c0, c1)`, two polynomials).
- **Each fold dimension → an RGSW-encrypted bit.** Coordinate `c1` (a bit) is encrypted as one RGSW
  ciphertext (a `2ℓ × 2` grid of polynomials).

Each selector encrypts **one bit** but is stored as `2 × N × log₂q ≈ 14 KB` — ~100,000× per useful
bit. That is inherent to lattice encryption (the bit hides in a full random-looking polynomial pair).
The *uncompressed* demo query is ~128 KB to retrieve from a 4-record DB — bigger than the DB. That
motivates query expansion.

### 3.3 Query expansion — packed upload, and who pays for it

Real OnionPIR does not send the expanded selectors. The client sends **one packed BFV ciphertext**
(~2 polys, ~14 KB; the seed trick for the uniform `c1` halves it to ~8 KB) whose coefficients encode
the index. The **server expands** it into the selectors + RGSW bits via a `Subs`-automorphism tree.
So per-query upload is **one ciphertext, independent of database size**. One-time key material
(Galois keys, RGSW(s)) is sent once and amortized.

**Can the client do the expansion itself instead?** Yes — and it costs *less* client CPU, not more.
The server's expansion is expensive only because it lacks the secret key and must reconstruct the
selectors homomorphically. The client *has* the key, so it simply **encrypts the selectors directly**
(a handful of fresh encryptions). The cost it pays is **upload bandwidth**, not CPU.

| | Server expands (standard) | Client sends expanded (this demo) |
|---|---|---|
| Upload | 1 packed ct (~8 KB), size-independent of DB | all selectors (~128 KB here), grows with dims |
| Server work | pays the expansion stage (~50% of server time, see §5) | none — jumps to the scan |
| One-time keys | must upload Galois keys + RGSW(s) | **not needed** |
| Noise | expansion adds noise | fresh selectors are noise-clean (better) |
| Privacy | identical | identical |

**For a TT accelerator demo, client-side expansion is often preferable**: it lets the device spend all
its time on the scan-and-fold that showcases the hardware. That is exactly why the current demo sends
the query pre-expanded.

---

## 4. The server pipeline

```
[0] expand query      (once per query, DB-independent)   ← skipped in the current demo (client pre-expands)
[1] initial dimension: response = Σ_k DB[k]·query[k]      ← TOUCHES EVERY RECORD; the throughput bottleneck
[2] higher dimensions: fold hypercube, 1 external product per size-2 axis  (log-depth)
[3] mod-switch the answer → return one ciphertext
```

**Yes, it operates on all records — it must.** Step [1] reads every DB entry exactly once and
multiplies it by its encrypted selector (0 or 1), then sums. The encrypted 0s zero out every record
but the wanted one; the encrypted 1 keeps it — and because the selectors are ciphertexts, the server
cannot tell which was the 1. If it touched only the wanted record, the access pattern would leak the
index. This scan is the throughput metric (database-bytes ÷ time) and is a big batched
plaintext×ciphertext MAC in the NTT domain — TT's strongest kernel (the int8 modular matmul).

Steps [2]–[3] operate on the *accumulators*, not the raw DB, so they do not re-scan.

---

## 5. What's built on TT (and what query expansion still needs)

Everything for the demo pipeline (scan → fold → mod-switch) is on silicon and bit-exact. For the
**server-side query expansion** (step [0]), **no fundamentally new kernel is missing** — the
primitives are all built and individually validated:

| piece expansion needs | status | device test |
|---|---|---|
| Galois automorphism `x→xʲ` | done | `test_permute` |
| monomial multiply `x^{±k}` | done | `test_permute` |
| BV key-switch | done | `test_bv_keyswitch` |
| signed gadget decompose (ℓ≤6) | done | `test_gadget_decompose` |
| NTT/INTT at PIR ring `n=2048` | done | `nntt2048_dev` / `nintt2048_dev` |
| external product (RLWE→RGSW convert) | done | `test_extprod` |
| ciphertext add / MAC | done | reused |

What remains is **not** a new primitive:
1. **The ExpandBFV schedule (#18)** — host orchestration that chains the above into the `Subs` tree
   (paper Alg. 1–3). Classified as "optimization / not-yet-assembled," same category as the demo's
   server pipeline.
2. **A chunked `gadget_decompose_signed` variant for large ℓ (8–10)** — the one real kernel-level
   caveat. The current single-pass kernel is validated for ℓ≤6 (fine for the folds); the key-switch
   in expansion typically uses larger ℓ, which overflows the DST register file and needs a chunked
   pass. (Or run key-switch at ℓ≤6 with a coarser base and eat the extra noise — no new kernel.)

Reference cost breakdown (CPU OnionPIRv2, why expansion is the heavy stage): Expand **51%**, initial
dimension 33%, fold 9%, convert 6%, mod-switch ~0%.

---

## 6. Example applications

PIR fits lookups where the query itself is sensitive and the database is large: password-breach
checking, private record lookups (for example market data, drug interactions or genomic markers), and
private retrieval for AI. In each, the hot loop is the initial-dimension scan.

Each becomes an **index search** through a *public, client-side* key→index map (the map leaks nothing —
PIR hides *which* index, not the map): a breach check uses a **public hash → bucket**; a record lookup
uses a **public directory → row number**. A breach list is public and identity-free (it records *which*
password strings are compromised, not *whose*), so a bucket of neighboring hashes reveals no secret —
public breach-check services already work this way, and PIR additionally hides the query. For
sensitive, identity-linked databases use size-1 buckets (cuckoo hashing) or per-record encryption so
only your own item is recoverable.

For retrieval-augmented generation, PIR covers the *retrieval* layer only: the result stays private
only if the model runs on the client, on-prem, or under confidential inference.

---

## 7. Per-card sizing and scaling

**State per-card capacity as database *bytes* per card** — the scan cost is linear in DB size. Two
independent limits agree on the same number, which is what makes it a clean claim:

**Bound 1 — DRAM capacity.** Keep the DB resident (streaming from host each query kills throughput).
n150 has 12 GB; at ~7× storage expansion, resident payload ≈ 12 / 7 ≈ **~1.5 GB of real data**.

**Bound 2 — latency × throughput.** The scan touches the whole DB once per query. Optimized target
throughput ≈ **~1 GB/s** (below); at ~1 s latency that is **~1 GB scanned per query**.

→ **Headline: ~1 GB of payload per n150, at roughly sub-second query latency.** The DB fits in DRAM
with headroom *and* is scannable in about a second — both limits land on the same figure.

### Where ~1 GB/s comes from (projection, honestly labeled)

Anchored on the one measured multi-core figure, ~3.5 G modmul/s across the card:
- per DB element (`n=2048`, 2 RNS channels, 2 ct components): `2048 × 2 × 2 ≈ 8192` modmuls
- `3.5e9 / 8192 ≈ 430K` elements/s; packed element ≈ `2048 × 13/8 ≈ 3.3 KB`
- → `430K × 3.3 KB ≈ **1.4 GB/s**`.

Caveats: the *current* PoC is single-core and unoptimized (`2.3 ms` per external product at
n=1024/ℓ=3). The levers are known and not fundamental — multi-core fan-out, twiddle-modmul fusion, device-resident pipeline (no host round-trips),
composite-NTT. The scan is comfortably **compute-bound, not DRAM-bound** (~10 GB stored read in
~35 ms at 288 GB/s ≪ 1 s), which is why adding compute helps.

### Why "100× data → 100 cards, ~same latency" is genuinely true

The initial-dimension response `Σ_k DB[k]·query[k]` is a **linear sum over database entries**, so it
shards cleanly: split the DB rows across cards, each card scans its shard into a partial ciphertext
accumulator, then **sum the partial accumulators** (cheap ct-adds) and fold once. Near-perfect linear
scaling — the only fixed (non-scaling) work is query expansion (once, or offloaded to the client) and
the log-depth fold + combine. **Capacity scales linearly with cards, at constant latency** — "add
cards = add database." This is a cleaner scaling story than most HE workloads.

### ~1 GB/card mapped to the applications

| application | record | records / card (~1 GB) | real-world anchor |
|---|---|---|---|
| Breach / password check | ~20 B hash | **~50 M** | A ~1 B-entry breached-password list (~20 GB) → **~20 cards** |
| Private RAG / AI retrieval | ~2 KB chunk | **~500 K** | One enterprise's private KB per card; 100 cards → ~50 M chunks |
| Finance / healthcare lookup | ~5 KB record | **~200 K** | A hospital patient index / patent sub-corpus per card |

### n150 (Wormhole) today → p150 (Blackhole) upside

All numbers above are a single **n150** running unoptimized code. **Blackhole p150 improves both
bounds that set per-card size:**

- **Memory (the direct win):** ~32 GB GDDR6 vs 12 GB on n150 → ~2.7× → resident payload rises from
  ~1.5 GB to **~4–4.5 GB per card**. Bigger DB stays resident, fewer cards for the same corpus.
- **Compute:** more Tensix cores + higher clocks → materially higher scan throughput (the modmul/NTT
  rate that sets §7's ~1 GB/s). Directional, not yet measured — but it moves the latency×throughput
  bound up in step with the memory bound.

*(Exact p150 throughput is a projection pending the device benchmark in §8; the memory figure is the
concrete, load-bearing improvement.)* Net: the per-card database size and QPS both scale up on
Blackhole, so the "one card solves X, N cards solve N·X" line only gets stronger — a larger X per
card and the same linear scaling.

### Server cost model — how work scales with database size (10M → 50M)

Going from a 10M to a 50M database changes **nothing structurally** — the same kernels run in the
same order. What changes is the *count* (~linear in database size) and the fold depth (+2). The
server work splits into two parts, both scaling with the number of PIR slots `S` (∝ database size):

- **Initial-dimension scan** — touches every slot → cost **linear in `S`**. Pure pointwise
  multiply-accumulate; the database is pre-NTT'd offline, so **no transforms run on the database
  here**. This is the dominant term at real sizes and is TT's throughput kernel (int8 matmul).
- **Folds** — total external products `≈ rest − 1 = S/N₀ − 1`, so also **linear in `S`** for fixed
  `N₀`. But the fold *depth* (sequential critical path) is only `d = log₂(S/N₀)` — logarithmic.

> **Law: total work ∝ database size; fold depth ∝ log(database size).**

Two configurations (`N₀ = 256`, ~200 hashes/bucket):

| | 10M | 50M |
|---|--:|--:|
| slots `S` | ~65,536 (2¹⁶) | ~262,144 (2¹⁸) |
| fold dims `d` | 8 | 10 |
| accumulators after scan `rest = 2^d` | 256 | 1,024 |
| total external products (`rest−1`) | ~255 | ~1,023 |

Approximate **arithmetic-operation** counts (one op = kernel on one full polynomial-channel;
per-external-product bundle at `n=2048, ℓ=3`: ~16 transforms + 2 gadget-decompose + 6 digit→RNS +
~4 CRT + a `2ℓ×2` MAC — anchored on the demo's measured per-ext-product counts):

| kernel / operation | scales with | 10M | 50M | ratio |
|---|---|--:|--:|--:|
| **initial-dim pointwise modmul (scan)** | `4·S` | ~262,000 | ~1,050,000 | ~4× |
| initial-dim modadd (accumulate) | `~4·S` | ~262,000 | ~1,050,000 | ~4× |
| selector NTTs (query, once) | `4·N₀` — *fixed* | ~1,024 | ~1,024 | 1× |
| fold transforms (NTT+INTT) | `16·(S/N₀)` | ~4,100 | ~16,400 | ~4× |
| gadget-decompose | `2·(S/N₀)` | ~510 | ~2,050 | ~4× |
| digit→RNS | `6·(S/N₀)` | ~1,530 | ~6,140 | ~4× |
| CRT-combine (garner + split) | `~4·(S/N₀)` | ~1,020 | ~4,100 | ~4× |
| fold pointwise MAC | `48·(S/N₀)` | ~12,200 | ~49,100 | ~4× |

**Reading of the table:**
1. **Everything is ~4× (≈5× for a true 5× data jump)** — same kernels, more calls, no new *type* of work.
2. **The scan dominates at real sizes.** The initial-dim modmul (~262K → ~1M) is 50–100× everything
   else combined. The tiny 4-entry demo looked *transform-bound* (~79% NTT) only because its scan was
   2 slots; a real DB is dominated by the **pointwise-MAC scan = TT's int8-matmul kernel**. Bigger DB
   → profile shifts *toward* the kernel TT is best at.
3. **The "new" PIR arithmetic stays ~1%** (gadget/CRT/digit ∝ `S/N₀`, a few thousand vs a million
   scan ops) — consistent with the measured device profile; scaling the DB doesn't change the ratio.
4. **Query-selector transforms don't grow** with the database (∝ `N₀`), because the query encodes the
   *index*, not the data.
5. **Latency vs work:** one card takes ~4× longer for 50M than 10M; but fold depth only went 8→10 and
   the scan is embarrassingly parallel, so ~4× more cores/cards serves 50M at the *same* latency —
   the linear-scaling story, now visible at the kernel level.

**Caveat — arithmetic ops vs dispatches.** The counts above are arithmetic operations. Actual kernel
*dispatches* depend on batching: the current single-tile PoC has dispatches ≈ ops (dispatch-bound),
but the **matmul formulation of the initial dimension** (lever #4) batches a `32×32` block per
dispatch — same arithmetic, ~1000× fewer launches. That is exactly why the dominant, linearly-growing
scan should run as the int8 matmul at scale.

---

## 8. Next build step

Replace the 4-entry toy DB with a real ~1 GB database benchmark —
1. run the initial dimension through the **int8 modular matmul at n=2048** (the transforms already
   exist on device),
2. use **fully-packed** plaintext polynomials with real payloads (not constant polys),
3. report **MB/s and queries/sec/card**,
4. demonstrate the **multi-card linear scaling** (shard → partial-accumulate → combine).

That measured per-card throughput replaces the §7 projection with a measurement.

---

## Appendix — key numbers referenced

| item | value | source |
|---|---|---|
| demo `pir_demo` device-active | 7.14 ms, 264 dispatches (4-entry DB, n=1024) | `pir_perf.md` |
| PIR is NTT-bound | ~79% transforms / ~19% MAC / ~1% new arithmetic (fused) | `pir_perf.md` |
| external product, n=1024/ℓ=3, 1 core | TT 2.3 ms (unoptimized PoC) | `pir_perf.md` |
| measured multi-core modmul rate | ~3.5 G modmul/s | `../vector/modular_multiplication.md` |
| storage expansion | ~7× (`log q / log t`) | this doc §3.1 |
| projected per-card DB (n150) | ~1 GB payload, ~1 s/query | this doc §7 (projection) |
| projected per-card DB (p150 Blackhole) | ~4–4.5 GB payload (memory-bound) | this doc §7 (projection) |
| params (real PIR) | n=2048, q=q₁·q₂ two 28-bit primes (~56-bit), t≈13-bit, B=2²⁰ | `pir_kernel_plan.md` |
