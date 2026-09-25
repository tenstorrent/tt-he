# Real 65 536-record PIR demo — plan and progress

**Goal.** A demo that privately retrieves one row out of a **real 65 536-row text database** on the
n150: the server never learns which row. Four deliverables, from the request:

1. a generated text database — one row per line, ASCII, `<row index> <famous company> <sentence>`;
2. a **high-performance parallel CPU encoder** that converts the text into the exact on-device PIR
   format and writes it to a file, so a demo run loads it directly with little or no preprocessing;
3. the demo itself, on the PIR pipeline already built (`docs/pir/ONION_MATCH_PERF.md`);
4. decode the retrieved plaintext back to a string and print it.

Status legend: ✅ done · 🟡 in progress · ⏳ not started · ⚠️ blocked//needs a decision.

> **DONE — the demo runs end to end.** `./build/pir_demo65536` privately retrieves a row from the real
> 65 536-row database and prints it back as text:
>
> ```
>   index> 31415
>   client generated the query in 305 ms (c0=245, fold bits=0110111)
>   [0] fold RGSWs -> device 55 ms | [1] expand 288 ms | [2] convert 52 ms
>   [3] scan 407 ms (+14 ms DB) | [4] folds 78 ms | [5] decrypt 24 ms
>   row 31415:
>     "31415 Mastercard - Mastercard operates supply chains that span several continents ..."
>   leading index matches request: YES   full row matches db65536.txt: YES
> ```
>
> **It is interactive**: `./pir_demo65536 -i` prompts for a row index and answers it. That needs a
> client fast enough to be typed at, which the Python reference (73 s/query) is not — see §10.
>
> This is the first time anything in this project has gone **plaintext → plaintext at n=2048**: every
> earlier test checked one stage against a golden. See §9 for what it cost and what it proved.

---

## 0. Why this shape works — the grounding facts

Checked in the reference before planning, because the whole design hangs on them:

- A DB cell **is a BFV plaintext polynomial**: `pir_answer` calls `pt_ct_mul(pp, db[cell], sel_ct)`,
  so `db[cell]` is a list of `n` coefficients mod `t`.
- `decrypt` returns `round(v/Δ) mod t` per coefficient — i.e. **the plaintext comes back as
  coefficients mod t**, nothing else to undo.
- Default `t = 257` (`Params.__init__`). ASCII is 0–127, so **one coefficient holds exactly one
  character** with room to spare — which is precisely the encoding requested.
- `n = 2048` ⇒ **2048 characters per row**, far more than a company name and a sentence need.

So the encode is *ASCII byte → coefficient*, and the decode is *coefficient → ASCII byte*. No packing
tricks, no bit-slicing. (The library packs 13 bits/coeff for density; we deliberately do not, because
the demo's point is legibility, and 2048 chars/row is already generous.)

**Hypercube.** `N0 = 512` first dimension × `rest = 128` remaining cells = **65 536 rows exactly** —
the same shape as OnionPIRv2's `CONFIG_N2048_K1`, and the number the request asks for. 7 binary folds
(128 → 1).

---

## 1. Text database generator ✅

`cpu/pir/gen_db_text.py` → `db65536.txt`, 65 536 lines.

Row format: `<5-digit zero-padded index> <Company> — <sentence about that company>`, ASCII only,
≤ 2048 characters. The index leads the row so a retrieved row is **self-identifying**: the demo prints
what it got and you can see the index matches what was asked for, without trusting the plumbing.

Companies cycle through a list of well-known names with varied sentence templates so rows differ in
content and length.

## 2. Parallel CPU encoder ✅ (both outputs verified against the device)

`cpu/pir/encode_db.cpp` → `db65536.pir`. Reads the text, emits the **device-ready** database.

What "device-ready" means here — the scan consumes the DB already in NTT domain, Montgomery form, per
RNS channel, tilized (`bDB[ch]`, cell `c` at tiles `2c`; see `Engine::scan_levelB`). So the encoder does
per row: ASCII → 2048 coefficients mod t → forward NTT per channel → Montgomery → 32×32 tilize → write.

- **Parallelism**: rows are independent ⇒ trivially parallel over rows (`std::thread` /
  OpenMP-style chunking), which is the whole reason this is a separate offline program rather than
  Python. Target: full 65 536 rows in well under a minute on this VM.
- **Size**: `K=3 × 65 536 cells × 2048 coeffs × 4 B` = **1.61 GB**. Acceptable on disk; the host has
  31 GB and the n150 12 GB.
- **Format**: a small header (magic, n, K, t, primes, n_rows, layout id) then the tile data, so the
  demo can `mmap`/stream it straight to device buffers.

**Resolved (direction set by the user): the encoder must emit the layout the EXISTING scan wants.**
The OnionPIRv2-matched scan is the **int8 matmul path** (`matmul_modq_mbatch` +
reshuffle — `test_scan_ntt2048` / `test_scan_kt`, the ~28 GB/s and ~30 ms figures). That path consumes
the DB **int8-folded, slot-major**, not the elementwise-MAC layout emitted so far.

The right place to absorb that is here, in the offline encoder — which is exactly its job ("do the
expensive prep once so a demo run loads and goes"). It is a change to an offline program, **not** a
change to the device configuration or to the benchmarked pipeline.

*Course correction recorded:* an earlier revision of this plan added a new `scan_chunked` engine method
(elementwise MAC, chunked to bound memory) so the already-written encoder format could be reused. That
would have demoed a **different implementation from the one we benchmarked**, which is the wrong
trade — removed. The config is unchanged throughout: n=2048, K=3 with primes
{268369921, 268361729, 268271617} (= `onionpir_ref.ntt_primes(3,2048)`), ell=8, B=2^11, N0=512,
7 folds, t=257.

## 3. Demo ✅ (`examples/pir_demo65536.cpp`, passing — see §9)

`examples/pir_demo65536.cpp`. **Must use the optimized kernels — the OnionPIRv2-matched
build** — not any simpler substitute:

| stage | implementation to use | why this one |
|---|---|---|
| expand | level-batched Subs tree + **MeshTrace replay** | 285 ms at m_exp=9; the traced path is the measured one |
| scan | int8 **matmul** (`matmul_modq_mbatch` + reshuffle), ×4 m-tiles | the ~28 GB/s path behind the comparison; verified on this exact file |
| folds | `fold_levelB` (batched, plane-domain reduce) | 77 ms for 7 folds, flat in cell count |
| decrypt/decode | host | client side by definition |

Shape per run: query → **512 selectors** (expand) → **4 × 32 = 128 cells** (scan) → **1 ciphertext**
(7 folds) → 2048 coefficients → ASCII.

### Scan wiring lifted into the Engine ✅

Done. `Engine::scan_mbatch_pass` / `scan_add_pass` / `scan_gather` / `scan_scatter` / `scan_mtile` are
the ~80 lines of inline lambdas from `test_scan_ntt2048` moved verbatim — the only edits were
`mkcb`→`mkcb_cr` and routing through `run(p)` so they join the workload cache and can be traced.

**Verified by re-pointing the proof at the port:** `test_fold_file_check`'s lambdas now delegate to the
Engine methods, so the test that already proved the inline wiring correct on this database now proves
the lifted wiring correct — **0/393 216 on m-tiles 0 and 2**. All three stages the demo needs
(`subs_levelB`, `scan_mtile`, `fold_levelB`) are now Engine calls.

Three layout conversions the demo must get right (each is the silent-failure kind this task has hit
three times already, so each deserves a check before the full run):

1. **selectors → scan input.** Expand emits a ct-batch `[comp][ch][half][node]` in the *coefficient*
   domain. The scan wants `Q_ntt[ch][k*2+c]` — so: `fwd2048_batchM` per channel (M=1024, 12 programs),
   then the per-k-tile `til2` + `gather` the wiring already does.
2. **scan output → fold input.** The scan produces `A_ntt` (eval domain, `rest*2` polys × 2 tiles).
   The folds want coefficient-domain cts in ct-batch layout with B=128 — so `inv2048_batchM` then a
   strided copy (`copy_polys`, already added) into the `[comp][ch][half][node]` layout.
3. **four m-tiles → one 128-cell batch.** m-tile `mt` yields rows `mt*32 .. mt*32+31`; they must land
   at node offset `mt*32` of the B=128 ct-batch.

### Decrypt (client, host)

`q = q0·q1·q2` is ~84-bit, so the phase computation needs `__int128` (or per-prime + Garner). Per
coefficient: `round(v/Δ) mod t` with `Δ = q/t`. Then coefficient → ASCII byte.

## 4. Decode and print ✅

Coefficients mod t → bytes → trim padding → print. Also assert the retrieved row's leading index equals
the requested one, which is the end-to-end correctness check the pipeline has never had (see below).

---

## 5. Prerequisites in the existing code — what is genuinely missing

This is the honest part. The pipeline is fast and each stage is validated, but a *chained, real-shape*
run has never been done, and three specific things stand between here and the demo:

**(a) Scan at `rest = 128` — largely dissolved.** Grouping the fold by m-tile means the demo runs the
already-validated `rest=32` scan four times, once per m-tile: a host-side loop, no kernel change.
Verified on m-tiles 0 and 3 (below). Original framing kept for context:
 - **DB layout** — the matmul scan wants the DB int8-folded and slot-major. Produced offline by the
   encoder (§2), so the demo still just loads.
 - **m-tiling** — `test_scan_ntt2048` hard-caps `rest ≤ 32` because it computes a single output m-tile
   (32 rows); `rest = 128` needs four. The kernels already do m-tiling (`test_scan_kt`); only the
   wiring is missing. *(Not a memory or time limit — see `ONION_MATCH_PERF.md` §1g.)*

`Engine::scan_levelB` is **not** the path to use: besides being a different implementation from the
benchmarked one, at this size its product buffer is `512·2·128·2` tiles ≈ 1 GB.

**(b) No chained n=2048 path.** Expand, unpack, scan and folds are each validated against goldens
*separately*, each fed from a golden rather than from the previous stage. Nothing at n=2048 goes
plaintext → plaintext. This demo is exactly that missing test, which is a good reason to build it.

**(c) Client side at n=2048/K=3.** `gen_pir_demo_case.py` produces the packed query + Galois keys +
`RGSW(s)` only at n=1024. Needs the K-general n=2048 version.

**Deliberately skipped: mod-switch.** It only shrinks the response ciphertext; the client can decrypt
the full-modulus ct directly (`onionpir_ref.decrypt`). Dropping it removes the one真 gap in §1a of the
perf doc and costs the demo nothing.

**Correctness at depth is unverified.** Expand is validated to m_exp=5 (N0=32), folds to 16 cells, scan
to 16 384 cells. The demo runs m_exp=9 / 65 536 cells. If a row comes back with the right index and
readable text, that is *itself* strong evidence — a wrong index or garbled text would be immediately
visible, which is a nice property of an ASCII payload.

---

## 6. Build order

1. §1 text generator — self-contained, no device.
2. §2 encoder — self-contained; verify by round-tripping encode→decode on host without any device.
3. §5(c) client query at n=2048/K=3.  ✅
4. §2b encoder emits the int8-folded slot-major DB the benchmarked scan consumes.
5. §5(a) wire the existing matmul scan's m-tile loop to `rest=128`.
6. §3+§4 chain it and print.

Steps 1–2 are independent of every device gap and are worth doing first.

---

## 7. Progress log

*(newest last; each entry records what was done, what was measured, and anything that turned out
differently than planned)*

- **Plan written.** Grounded the encoding in `onionpir_ref` (`t=257`, one ASCII char per coefficient,
  2048 chars/row) and confirmed the 512×128 = 65 536 hypercube matches both the request and the
  library config. Recorded the three real prerequisites (§5) rather than assuming the pipeline is
  demo-ready.

- **§1 done.** `cpu/pir/gen_db_text.py` → `db65536.txt`: 65 536 rows, ASCII-only (checked), longest row
  129 chars against the 2048 limit, 0.17 s. Rows are self-identifying, e.g.
  `00000 Tenstorrent - Tenstorrent is frequently named when people discuss the future of its industry.`

- **§2 done.** `cpu/pir/encode_db.cpp` → `db65536.pir`: **65 536 rows in 6.5 s on 8 threads
  (10 078 rows/s)**, 1.54 GB, layout `[channel][row][2 tiles]` behind an 8-word header. A demo run now
  does zero crypto preprocessing — it seeks and uploads.

- **A convention bug the self-check could not have caught.** `encode_db --verify` (fast NTT vs its own
  slow O(n²) NTT) passed while the file was **100% wrong** against the device: 36 864/36 864 mismatches.
  Cause: the Montgomery lift. `montgomery.hpp` defines `to_montgomery(a) = (a << 32) % q`, i.e. `a·R`;
  the encoder multiplied by `R² mod q`. Both of the encoder's own paths shared the wrong constant, so
  they agreed with each other perfectly. Fixed; now **0/36 864 against the device**, covering row 0,
  row 65 535 and random rows across all 3 channels.
  *Lesson worth keeping: a self-consistency check validates an implementation against itself, not
  against the convention it has to interoperate with. `tests/pir/test_db_encode_check.cpp` exists
  precisely to compare the two independent implementations, and it earned its keep immediately.*

- **§5(c) done.** `cpu/pir/gen_query65536.py` → `query65536.bin`: **8.7 MB in 73 s** (keygen+query 0.1 s,
  Galois keys 2.8 s, 7 fold RGSW 70 s). For `idx=40000` it reports `c0=312 (of 512), fold bits
  [1,0,0,0,0,0,0]` via the reference's own `_coords`, so the hypercube decomposition is not
  re-derived by hand. The database is deliberately *not* in this file: 1.5 GB of DB loads once, and any
  index then costs a ~9 MB query.
  Two things skipped on purpose and recorded: the Python reference answer (it would run a 512-term scan
  over 65 536 cells in bigint arithmetic — hours; the demo verifies against the known text row instead),
  and ladder expansion for the fold bits (`expand_ladder=False`) which keeps generation at ~1 min
  without weakening privacy — the server still only ever sees ciphertexts.
  *A layout trap avoided by reading the validated writer first:* `galois_keys_for_expand` returns a
  **dict keyed by `j = (n>>a)+1`**, not a level-indexed list, and each KSK is `ksk[k][col]`. Copying
  `gen_expand_case.py`'s serialisation exactly is what kept the device side unchanged.

- **Course correction (user).** Use the OnionPIRv2-matched TT implementation; do not
  drift the config. I had begun adding a new `scan_chunked` (elementwise MAC) so the encoder's existing
  output format could be reused — that would have demoed a different scan from the one we measured.
  Removed. Instead the **encoder** grows a second output: the int8-folded slot-major DB the benchmarked
  matmul scan consumes. Offline change, no config change. Config confirmed unchanged: n=2048, K=3
  {268369921, 268361729, 268271617} (= `onionpir_ref.ntt_primes(3,2048)`), ell=8, B=2^11, N0=512,
7 folds, t=257.

- **Both encoder/generator bugs now VERIFIED fixed, not just fixed.** Asked whether the bugs were
  actually fixed, the honest answer at that moment was "one of them":

  | bug | fix | verification |
  |---|---|---|
  | encoder Montgomery lift used `R²` instead of `R` | patched | **0/36 864** vs the device (`test_db_encode_check`) |
  | Galois keys serialized as a level-indexed list, but `galois_keys_for_expand` returns a **dict keyed by `j=(n>>a)+1`** with `ksk[k][col]` | patched by copying `gen_expand_case.py`'s writer | **was untested** — nothing consumed `query65536.bin` |

  The second was the same class of defect as the first (a convention mismatch between two
  implementations) and would have surfaced only as garbage at the end of the demo, after expand → scan
  → folds. So it got the same treatment: `gen_query65536.py --golden` now also emits the reference
  selectors (tractable at small N0), and `tests/pir/test_query_file_check.cpp` reads the query file,
  runs the **real level-batched Subs tree** with the client's keys, and compares.
  Result at `--N0 32 --nfold 2 --idx 100`: **0/393 216 over 32 selectors** — the config is otherwise
  identical to the 65 536 case (n=2048, K=3, ell=8, B=2^11, t=257), so this validates the *layout*;
  only N0 differs. It also incidentally proves the client-key → device-expand path works end to end,
  which nothing had shown before.

- **§2b done (built, not yet verified).** `encode_db --fold db65536.fold --rest 128` emits the
  int8-folded **slot-major** database the benchmarked matmul scan reads: **6.00 GB in 24.6 s**,
  laid out `[m-tile][channel][k-tile][slot][16 int8 tiles]`.

  Grouping by m-tile is the key choice. The scan produces 32 output rows at a time — which is exactly
  why `test_scan_ntt2048` caps `rest ≤ 32` — so with the file grouped this way the demo runs the
  **already-validated rest=32 scan four times**, once per m-tile, and needs **no kernel change at
  all**. It also means the device can either hold all 6.0 GB resident (of 12 GB) or stream 1.5 GB per
  m-tile.

  **A convention trap caught by reading the consumer first:** the matmul scan uses **plain NTT values,
  NOT Montgomery** — `DBn[ch][i] = eng.fwd2048(...)` with no `to_mont`, because `matmul_modq` is a
  plain modular matmul whereas the `pmul` MAC path needs one operand in Montgomery form. The existing
  `db65536.pir` is Montgomery-lifted and would have been silently wrong here. The encoder now keeps the
  plain NTT alongside and lifts only for the `.pir` output.

  **Not yet verified.** Nothing has consumed `db65536.fold`. This is precisely the state the query file
  was in before `test_query_file_check`, and the same reasoning applies: a layout error is silent. The
  verification to run is to feed the file's fold into the validated scan and confirm it still gives
  0 mismatches, rather than trusting that the formula was transcribed correctly.

- **§2b VERIFIED.** `tests/pir/test_fold_file_check.cpp` is `test_scan_ntt2048` with exactly two
  substitutions — DB_ntt read from `db65536.pir` (un-Montgomery'd) instead of random, and the int8
  fold **read from `db65536.fold`** instead of recomputed inline. Everything else is the validated
  path: gather → K-tiled mbatch matmul → scatter, against the eval-domain golden.

  **m-tile 0: 0/393 216. m-tile 3: 0/393 216.** So the encoder's fold formula *and* its byte layout
  are what the benchmarked scan expects — not merely self-consistent.

  A side effect worth recording: **`scan-wall` fell from ~32 s to 1.7 s**. In `test_scan_ntt2048` that
  figure is dominated by computing the DB NTT on device; here the NTT arrives precomputed in the file.
  That is the encoder's entire purpose — "load and go" — showing up as a 19× drop in setup.

  The m-tile grouping also validates the §5(a) plan: the scan ran unmodified at `rest=32` and the four
  m-tiles tile the `rest=128` database, so no m-tile loop needs to be added to the kernels.

- **Doc updated; demo scoped precisely.** With §2b verified, both blockers are gone and the demo is
  assembly of verified parts. Scoped the remaining work honestly rather than starting it half-way: the
  scan must be *lifted* from `test_fold_file_check`'s inline lambdas into an Engine method verbatim
  (that exact wiring is now proven against this database), and three layout conversions sit between
  the stages — selectors→scan (coeff→NTT), scan→folds (NTT→coeff + ct-batch layout), and m-tile→node
  offset. Each is the same silent-failure class that has already cost three bugs here (R vs R²,
  dict vs list, Montgomery vs plain), so each gets a check before the full run rather than after.

- **Scan lifted and re-verified.** The benchmarked scan wiring is now `Engine::scan_mtile` (+ the four
  passes it composes), moved verbatim rather than rewritten. Rather than trust the port, the existing
  proof was re-pointed at it: `test_fold_file_check` delegates to the Engine methods and still returns
  **0/393 216** (m-tiles 0 and 2). The demo can now call all three stages —
  `subs_levelB` → `scan_mtile` ×4 → `fold_levelB` — with no inline device wiring of its own.

---

## 8. HANDOFF — everything needed to finish, without re-deriving anything

Written to be self-sufficient: a reader who has only this file should be able to finish the demo.

### 8.1 Config (unchanged from the OnionPIRv2-matched build — do not drift it)

```
n = 2048           ring degree = characters per row
K = 3              RNS channels, primes {268369921, 268361729, 268271617}   (all < 2^28, 4096 | q-1)
                   == onionpir_ref.ntt_primes(3, 2048).  The CLIENT fixes these; see the trap in 8.5.
ell = 8            gadget length          B = 2^11 = 2048     gadget base
N0 = 512           first hypercube dim    rest = 128          remaining cells  -> 512*128 = 65536 rows
nfold = 7          binary folds (128 -> 1)
t = 257            plaintext modulus: ONE ASCII CHARACTER PER COEFFICIENT
R = 2^32           Montgomery radix (cpu/montgomery.hpp)
m_exp = 9          expansion levels = log2(N0)
KB = 16            k-tiles (N0/32)        MB = 4            m-tiles (rest/32)
```

### 8.2 Artifacts — all verified, with regeneration commands

| file | size | what | verified by |
|---|--:|---|---|
| `cpu/pir/db65536.txt` | 9 MB | 65 536 ASCII rows, `<5-digit idx> <Company> - <sentence>` | ASCII+length asserted in generator |
| `cpu/pir/db65536.pir` | 1.5 GB | NTT + **Montgomery**, `[ch][row][2 tiles]`, hdr `PDB1` 8×u32 | `test_db_encode_check` 0/36 864 |
| `cpu/pir/db65536.fold` | 6.0 GB | int8 fold, **PLAIN NTT**, `[m-tile][ch][k-tile][slot][16 tiles]`, hdr `PFLD` 8×u32 | `test_fold_file_check` 0/393 216 (mt 0,2,3) |
| `cpu/pir/query_last.bin` | 8.7 MB | second demo query, `idx=65535` (far corner of the hypercube) | demo end-to-end |
| `cpu/pir/query65536.bin` | 8.7 MB | query + Galois keys + fold RGSWs + secret, hdr `Q65S` 10×u32 + K primes | `test_query_file_check` 0/393 216 (small case) |

```bash
cd cpu/pir
python3 gen_db_text.py                                            # 0.2 s
g++ -O3 -march=native -pthread -o encode_db encode_db.cpp
./encode_db db65536.txt db65536.pir --fold db65536.fold --rest 128   # 6.5 s + 24.6 s
python3 gen_query65536.py --idx 40000                             # 73 s
python3 gen_query65536.py --idx 65535 --out query_last.bin         # 73 s, the second index
python3 gen_query65536.py --N0 32 --nfold 2 --idx 100 --golden --out query_small.bin   # for the check
cd ../../device/build
./test_db_encode_check ; ./test_fold_file_check 0 ; ./test_query_file_check ../../cpu/pir/query_small.bin
./pir_demo65536                                                   # the demo itself
```

### 8.3 The three Engine stages the demo calls (all verified, no inline wiring needed)

```cpp
// EXPAND — level-batched Subs tree; see test_expand_bench / test_query_expand2048 for the loop shape.
//   per level a: build comb = [cts | monomul(cts)] as a 2B batch, then ONE subs, then ctadd
void subs_levelB(MeshBuf bIn, uint32_t B, int64_t j,
                 const std::vector<std::vector<MeshBuf>>& KnB, MeshBuf out, MeshBuf scratch);
void permute_ctb_into(MeshBuf in, uint32_t Bin, MeshBuf out, uint32_t Bout, uint32_t node0,
                      uint32_t A, uint32_t Bc);            // A=1,Bc=0 is an identity copy
void monomul_ctb_into(MeshBuf in, uint32_t Bin, MeshBuf out, uint32_t Bout, uint32_t node0, int64_t k);
void ctadd_ctb(MeshBuf a, MeshBuf b, MeshBuf o, uint32_t B);

// SCAN — the benchmarked int8 matmul path, one m-tile (32 output rows) per call
void scan_gather(MeshBuf in, MeshBuf out, uint32_t ROWS, uint32_t COLS, uint32_t SR, uint32_t SC,
                 uint32_t out_base, uint32_t h);           // query poly->slot, per k-tile, per half
void scan_mtile(Chan& c, MeshBuf sX, MeshBuf sW, MeshBuf sPar, MeshBuf sAcc0, MeshBuf sAcc1,
                MeshBuf out, uint32_t KB, uint32_t sOT, uint32_t rest);

// FOLDS — batched, plane-domain reduce
std::vector<std::vector<MeshBuf>> rgsw_operand(const std::vector<std::vector<std::vector<uint32_t>>>& Cn);
MeshBuf fold_levelB(MeshBuf acc, uint32_t Bold, const std::vector<std::vector<MeshBuf>>& Cn);

// helpers
MeshBuf mkctb(uint32_t B); MeshBuf upload_ctb(...); std::vector<...> download_ctb(MeshBuf, uint32_t B);
void fwd2048_batchM(Chan&, MeshBuf in, uint32_t in_base, uint32_t in_pstride, uint32_t in_tstride,
                    uint32_t M, MeshBuf out, uint32_t out_base, uint32_t out_pstride, uint32_t out_tstride=1);
void inv2048_batchM(...same signature...);
void copy_polys(MeshBuf in, MeshBuf out, uint32_t npoly, uint32_t q,
                uint32_t ib,uint32_t ips,uint32_t its, uint32_t ob,uint32_t ops,uint32_t ots);
void prewarm_batch(uint32_t maxB);   // MUST call before a traced/pooled run
```

**Ct-batch (CTB) layout**, used by expand and folds:
`tile of poly (comp, ch, node), half h  =  ((comp*K + ch)*2 + h)*B + node`. At B=1 this *is* the
per-ciphertext layout, which makes `B=1` a free equivalence check for anything new.

### 8.4 Demo steps, with the three layout conversions spelled out

1. Read `query65536.bin`; upload Galois keys as **NTT + Montgomery** per (level, ch, col) into a
   `2*ell`-tile buffer with key `k` at tiles `2k, 2k+1` (that ordering is what the batched MAC
   broadcasts against). Same for the fold RGSWs, but via `rgsw_operand()` which packs them
   `(k, comp, half)` = tile `4k + 2*comp + t`.
2. **Expand**: `upload_ctb({query})` → 9 levels → CTB with **B=512** selectors, coefficient domain.
3. **Conversion 1 — selectors → scan input.** Scan wants `Q_ntt[ch][k*2+c]`, so per channel
   `fwd2048_batchM` (M=1024) then, per k-tile and per half, `til2` + `scan_gather(..., ROWS=min(32,N0-kt*32),
   COLS=2, SR=2, SC=1, out_base=kt*sOT, h)`. `sOT = N = 2048`.
4. **Scan**, per m-tile `mt` in 0..3: upload that m-tile's fold slice from `db65536.fold`
   (offset `sizeof(hdr) + mt*per_mt + ch*KB*N*16*1024`, `per_mt = K*KB*N*16*1024`), then
   `scan_mtile(...)` → `A_ntt`, `rest*2 = 64` polys × 2 tiles, eval domain.
5. **Conversion 2 — scan output → fold input.** `inv2048_batchM` back to coefficient domain, then
   `copy_polys` into the CTB layout of a **B=128** batch.
6. **Conversion 3 — m-tile → node offset.** m-tile `mt` supplies rows `mt*32 .. mt*32+31`, so it lands
   at node offset `mt*32` of the B=128 batch.
7. **Folds**: `fold_levelB` × 7, B: 128→64→…→1.
8. **Decrypt (host)**: `q = q0*q1*q2` (~84 bit) so use `__int128` or per-prime + Garner.
   `phase = c0 + c1*s mod q` (negacyclic), then per coefficient `round(center(v)/Δ) mod t`, `Δ = q/t`.
9. **Decode**: coefficient → ASCII byte, stop at the first 0, print. **Assert the leading 5 digits
   equal the requested index** — rows are self-identifying precisely so this check exists.

### 8.5 Traps — every one of these has already cost real time here

*Silent-failure class (wrong answer, no error) — check each in isolation before a full run:*
- **The RNS primes are fixed BY THE CLIENT, not by the device.** `onionpir_ref.ntt_primes(3,2048)` =
  {268369921, 268361729, 268271617}; `encode_db.cpp` originally hardcoded a different, equally valid
  triple {167772161, 104857601, 268271617}. Both sides passed every self-check, the demo produced
  fluent garbage, and only the one prime the two sets share (268271617) decrypted — which is exactly
  what localized it: **channel 2 correct, channels 0 and 1 wrong, and channel 1's device output was
  numerically larger than channel 1's own modulus.** Anything that names a prime must derive it from
  the artifact it has to agree with, never from a local constant.
- **Montgomery vs plain.** The `pmul` MAC path needs one operand Montgomery-lifted; the **matmul scan
  wants PLAIN NTT**. `db65536.pir` is Montgomery, `db65536.fold` is plain. Mixing them up is silent.
- **`to_montgomery(a) = (a<<32) % q` is `a·R`, not `a·R²`.** Cost one full wrong 1.5 GB file.
- **`galois_keys_for_expand` returns a DICT keyed by `j=(n>>a)+1`**, not a level-indexed list, and each
  KSK is `ksk[k][col]`. Copy `gen_expand_case.py`'s writer rather than guessing.
- **A self-consistency check proves nothing about a convention.** `encode_db --verify` passed on a file
  that was 100 % wrong. Always cross-check against the *other* implementation.

*Infrastructure class (hang or crash):*
- **`DEFAULT_TRACE_REGION_SIZE` is 0.** `begin_mesh_trace` without a region **hangs the device**
  (needs `tt-smi -r`). Always `create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 64u<<20)` or larger.
- **A program cannot be BUILT inside a trace capture** ("Writes are not supported during trace
  capture"). Build+enqueue once outside, then re-enqueue the same workload inside. Program reuse is a
  prerequisite for trace, not an alternative.
- **Trace bakes buffer ADDRESSES** → buffers must be pooled (`pool_enable`) and the query slot
  allocated *before* `pool_enable`, or `pool_reset()` hands it back to the tree as scratch.
- **The workload cache is index-replayed and cannot self-validate.** If a shape changes between passes
  call `caches_clear()`, or stale programs run against new buffers (this produced a fake 3.8 M NTT/s).
- **Adding a runtime arg to a shared dataflow kernel is not a compile error at the call sites.** Grep
  every `SetRuntimeArgs` for that kernel (cost: an uninitialized `b_cycle` that still passed tests).
- **`split()` caps tiles/core but not core count**; use `split_stream()` for streaming ops or you ask
  for 256 cores on a 64-core grid ("Illegal kernel placement").

### 8.6 Depth coverage — what the demo closed

- Expand was validated to **m_exp=5 (N0=32)** against a Python golden; the demo runs **m_exp=9** and its
  selectors decrypt one-hot at c0 — so the library-depth tree is now correct, not just fast.
- Folds were validated to 16 cells; the demo runs **128**. Scan was validated to 16 384 cells; the demo
  runs **65 536**.
- **Plaintext → plaintext at n=2048 now exists** (§9). It is the check every earlier test lacked.
- Device DRAM: the fold is 6.0 GB of 12 GB. The demo **streams 1.5 GB per m-tile** (1.6-1.9 s per query
  from page cache) rather than holding it resident; making it resident is a pure win left on the table.

---

## 9. Result — the demo, what it proved, and what it cost

`examples/pir_demo65536.cpp` (+ `add_executable(pir_demo65536 …)`). Run from `device/build`:

```bash
./pir_demo65536                 # with the isolation checks (adds ~20 s of host golden work)
./pir_demo65536 --no-checks     # the demo as demoed
```

### 9.1 Measured, one n150 (per query, after §11's host-side work)

| stage | where | cold (1st query) | warm |
|---|---|--:|--:|
| client generates the query (host, 7 threads) | host | 293 | 293 |
| fold RGSWs -> NTT + Montgomery -> device | host | 55 | 55 |
| expand — 1 ct -> 512 selectors (m_exp=9), 1131 programs | device | 1047 | **288** |
| convert — selectors -> scan operand, 144 programs | device | 114 | **52** |
| scan — 65 536 rows x 512 terms, 600 programs | device | 678 | **407** |
| folds — 128 -> 1 (7 folds), 1092 programs | device | 1034 | **78** |
| **server on-device total** (2 967 programs) | device | **2928** | **881** |
| decrypt + ASCII decode | host (client) | 24 | 24 |
| **wall clock, index typed to row printed** | | ~3.3 s | **~1.2 s** |

One-time at startup: host client key setup 81 ms, Galois keys to device 174 ms, and the 6.0 GB folded
database made resident 1.5-1.7 s.

### 9.1b A second, independent index

One passing index could be luck, so a second query was generated against the far corner of the
hypercube — `idx=65535` (c0=511, **all seven fold bits set**, m-tile 3, i.e. the last row):

```
  retrieved row 65535:
    "65535 Adobe - Adobe is often cited in business schools as a case study in scaling an organisation."
  leading index matches request : YES    full row matches db65536.txt : YES
```

Different first-dimension cell, different m-tile, and the opposite fold path — 2609 ms, the same as
idx=40000 to within noise, as it should be (PIR touches every row regardless of which one is wanted).

### 9.2 What it proves that nothing before it did

- **Plaintext → plaintext at n=2048.** Every earlier test compared one stage against a golden. This
  runs the actual claim: a client encrypts an index, the server touches all 65 536 rows without
  learning which one, and the client gets back the row's text.
- **Expand at library depth is CORRECT, not just fast.** It was validated to m_exp=5; the 512
  selectors here decrypt one-hot at c0=312.
- **The noise budget survives the full pipeline** — 9 expansion levels, a 512-term scan against
  plaintexts up to 128, and 7 external-product folds, at K=3 (~84-bit q) with t=257. This was an open
  question; nobody had run it.
- **The offline artifacts are load-and-go.** The 6.0 GB folded database goes to the device with no
  preprocessing; the encoder's whole purpose, visible as the 1.9 s streaming line.

### 9.3 The bug that mattered, and how it was caught

The first full run produced fluent garbage. Three staged checks (each comparing against the *other*
implementation, never against itself) localized it in one run:

| check | result |
|---|---|
| selectors decrypt one-hot at c0 | ok → expand fine |
| `bQ` = batched selector NTT in scan layout | ok → conversion 1 fine |
| `sX` = slot gather with the new per-k-tile `in_base` | ok → conversion 1 fine |
| scan output vs host golden (from `bQ` + `db65536.pir`) | **ch0 FAIL, ch1 FAIL, ch2 ok** |

Two channels wrong and one right is not an arithmetic bug, and channel 1's device output was *larger
than channel 1's modulus*. The encoder had hardcoded `{167772161, 104857601, 268271617}` while the
client's keys were generated on `onionpir_ref.ntt_primes(3,2048)` = `{268369921, 268361729, 268271617}`
— the two sets share exactly one prime, and that is the channel that worked. Fixed at the source
(`encode_db.cpp` now carries the reference's primes, with the reasoning in a comment), re-encoded
(38 s), and `test_fold_file_check` now reads the primes from `db65536.pir`'s header instead of a local
constant, so the class of bug cannot recur silently there.

That is the fourth convention mismatch in this task (R vs R², dict vs list, Montgomery vs plain, and
now the prime set). All four were invisible to self-consistency checks and all four were caught by
comparing against the other implementation. The staged checks are kept in the demo (default on,
`--no-checks` to skip) for exactly that reason.

### 9.4 One Engine change was needed

`scan_gather` gained a defaulted `in_base` argument. The gather kernel's `in_off` is a pure additive
term on the input tile index, so passing the k-tile's block base there lets one `4*N0`-tile buffer of
expanded selectors feed every k-tile — no per-k-tile staging buffer and no host round-trip. With
`in_base=0` it is byte-identical to before, which `test_fold_file_check` confirms (0/393 216, m-tile 1
and m-tile 2 re-run after the change).

### 9.5 Left on the table (in the order worth doing)

1. **Trace the demo.** Biggest single win: expand 1027 → ~285 ms, folds 892 → ~77 ms measured.
2. **Keep the fold resident** (6.0 GB of 12 GB) — removes the 1.9 s streaming from every query.
3. **Fix the scan's column underfill** (2 of 32 output columns used at B=1) — the known ~2× on the
   scan, documented in `ONION_MATCH_PERF.md` §3.
4. Batch several queries: everything except the folds is already batch-shaped.

---

## 10. Interactive mode — and the host client that makes it possible

```bash
./pir_demo65536 -i            # prompt for a row index, answer it, repeat
./pir_demo65536 --idx 12345   # one index, non-interactive
./pir_demo65536               # the reference query file (unchanged path, still verified)
./pir_demo65536 --checks      # add the per-stage isolation checks
```

```
index> 0
  client generated the query in 296 ms (c0=0, fold bits=0000000)
  row 0: "00000 Tenstorrent - Tenstorrent is frequently named when people discuss the future of its industry."
index> 65535
  client generated the query in 284 ms (c0=511, fold bits=1111111)
  row 65535: "65535 Adobe - Adobe is often cited in business schools as a case study in scaling an organisation."
```

### 10.1 Why a new client was needed

Only two things depend on the requested index: the packed query ciphertext and the 7 fold RGSWs. The
secret key and the Galois keys do not — so they are loaded once from the existing key bundle and the
device's uploaded Galois keys stay valid for every query in the session.

`gen_query65536.py` takes **73 s** per index, ~70 s of it in the fold RGSWs, because the reference does
negacyclic multiplication over an 84-bit composite modulus in Python bigints. That is a property of the
reference, not of the client's real work. `cpu/pir/pir_client.hpp` does the same arithmetic in RNS with
the host NTT: **306 ms**, and 7 threads (the folds are independent) take it there from 1.57 s.
**240× faster than the reference, and it is the honest client cost.**

The same host NTT also replaced the device round-trip that was prepping key material one polynomial at
a time: **key prep 4.6 s → 174 ms**.

### 10.2 Cost per interactive query (measured)

| | ms |
|---|--:|
| client generates the query (host, 7 threads) | ~300 |
| fold RGSWs → NTT + Montgomery → device | ~255 |
| expand / convert / scan / folds (device) | ~2650 |
| database streaming (6.0 GB, page cache) | ~1800–2300 |
| decrypt + decode (host) | 24 |
| **wall clock, index typed to row printed** | **~5 s** |

One-time at startup: host client key setup 81 ms, Galois keys to device 174 ms.

### 10.3 How the client is verified

`cpu/pir/test_client.cpp` (`g++ -O2 -pthread -o test_client test_client.cpp && ./test_client`) — every
check compares against something independent of the thing under test, which is the discipline this
project arrived at the hard way:

| check | compared against |
|---|---|
| `intt(ntt(x)) == x` on all three channels | the forward transform, itself device-verified |
| RNS-NTT multiply | O(n²) schoolbook negacyclic convolution mod q |
| `dec(enc(m)) == m` | the plaintext |
| balanced gadget digits | the recomposition identity Σ dₖBᵏ ≡ a, and the range bound |
| `extprod(RGSW(s), RLWE(μ))` | s·μ computed directly |
| all 7 fold RGSWs in a mux | which of two plaintexts a mux is supposed to select |
| the packed query | one-hot at c0, scaled by (2^m)⁻¹ mod t |

**All pass.** On the device side, the host NTT is re-checked against `eng.fwd2048` on every startup
(refusing to run if they disagree), and a client-generated query was put through the full per-stage
isolation checks — selectors one-hot, both halves of conversion 1, all 12 scan (m-tile × channel)
comparisons against a host golden, and the post-scan accumulator — **all ok**, then the row itself.

### 10.4 What the interactive mode does not change

The server path is byte-for-byte the OnionPIRv2-matched one: the same `subs_levelB`,
`scan_mtile` and `fold_levelB` calls on the same buffers. The client is host-side by definition and
touches nothing the server measurement covers.

---

## 11. Host-side latency work on the demo

The first working demo had only three of the four host-side steps. Step 4 — program reuse — was
deliberately deferred while correctness was in doubt (a fast wrong answer is the worst outcome), then
applied and measured here. **Warm query: 2928 ms → 881 ms on device, ~3.3 s → ~1.2 s wall clock.**

| step | what it means here | state |
|---|---|---|
| 1. paramgen hoisted | offline artifacts; key prep out of the query; host NTT instead of a device round-trip per polynomial (**4.6 s → 174 ms**) | ✅ |
| 2. constant residency | twiddles, gadget and CRT constants baked in the Engine ctor | ✅ |
| 3. buffer residency | `persist` + the scratch pool; nothing reallocated per query | ✅ |
| 4. **program reuse** | `wl_enable` + one `wl_reset()` per query | ✅ **(this section)** |
| 4b. MeshTrace | not needed: reuse already reaches the traced figures (see below) | ⏭ |

### 11.1 What each change gave

**Program reuse (3× on device).** 2 967 programs run per query and the sequence is identical every
time, so the index-replayed workload cache is sufficient: query 1 builds and records, later queries
re-enqueue. Cold vs warm per program: expand 0.89 → 0.25 ms, folds 1.00 → 0.07 ms. The build, not the
dispatch, was the cost.

**Prerequisite that makes reuse safe.** Program reuse bakes buffer ADDRESSES into the recorded runtime
args, so the two per-query *inputs* — the packed query and the 7 fold RGSW operands — can no longer be
reallocated each query. They are now fixed buffers re-filled in place (`Engine::upload_ctb_into`,
`Engine::rgsw_operand_into`). Reallocating them would have left the cached programs reading the
previous query's memory: a wrong answer with no error.

**Resident database (removed 1.7 s/query).** The folded DB is 6.0 GB and the n150 has 12 GB, so it now
loads once (1.5 s) instead of streaming 6.0 GB per query: **1690 ms → 14 ms**. It genuinely fits
alongside the expand working set; `--stream` forces the old behaviour, and an allocation failure falls
back to streaming automatically rather than dying.

**Threaded fold-RGSW prep (253 → 55 ms).** 672 host NTTs + Montgomery lifts, 7 independent folds;
transform in parallel, keep only the enqueues serial (one command queue).

### 11.2 Why MeshTrace was not needed after all

Warm expand is **288 ms against the traced benchmark's 285 ms**, and warm folds are **78 ms against 77
ms**. Trace removes host dispatch overhead; reuse had already removed the host *build* cost, which was
the whole gap. Trace would still help if dispatch dominated — it does not here — and it carries real
hazards (baked addresses, no program build inside a capture, device hangs on a missing trace region).

### 11.3 Verification of the optimized path

Program reuse is exactly the change that can silently replay a stale program, so it was checked the
same way as everything else: `--checks` run on a **warm** query passes every per-stage isolation check
(selectors one-hot, both halves of conversion 1, all 12 scan m-tile×channel comparisons against a host
golden, post-scan accumulator), and three different indices answered in one warm session all return
their own row. Different indices returning different correct rows from cached programs is the evidence
that matters.

### 11.4 A latent hazard the batch work exposed: pool arenas

The three sections of a query — expand, scan, folds — each allocate a *different* sequence of scratch
buffers, but all three called `pool_reset()`. The pool matches by size in call order, so at every
section boundary it saw a mismatch, dropped its tail and reallocated the lot — on every query and every
m-tile. Each section now starts at its own mark (`arena_scan`, `arena_fold`), so each replays its own
sequence exactly. Beyond the wasted churn this was a correctness hazard: reused programs bake buffer
addresses, and free/realloc only returned the same addresses because the allocator happened to be
deterministic. Measured effect at batch 4 was small (581 → 574 ms/query); the reason to do it is that
it removes the hazard, and the churn compounds with batch size.

### 11.5 What is left

1. **The scan, 407 ms** — now the largest device term. The known ~2× is the column underfill
   (2 of 32 output columns used at B=1), documented in `ONION_MATCH_PERF.md` §3.
2. **The client, 293 ms** — now comparable to the whole server compute. Barrett/Shoup reduction in the
   host NTT would take it under 100 ms.
3. Batch several queries: everything except the folds is already batch-shaped.

---

## 12. Batching — yes, and it is the only way to fill the scan

```bash
./pir_demo65536 --batch 4              # 4 spread-out indices, one shared scan
./pir_demo65536 --batch 3,777,40000    # ...or an explicit list
```

### 12.1 Why batching is the only lever on the scan

Per slot the scan computes `A[rest x N0] x B[N0 x 2]`: **the DB matrix A is fully used**, but B — the
query — occupies 2 of the matmul's 32 columns. The waste is in B's padded columns, so the fix has to
add *columns*, and the only source of columns is more queries: 16 x 2 components = 32, exactly full.

**This corrects §4 of `ONION_MATCH_PERF.md`**, which suggested a single query could fill the width by
letting "the `rest` axis fill the output width instead of the query axis". `rest` is already the *row*
axis; moving it to the columns just moves the padding. There is no single-query restructuring that
fills those columns — batching is the mechanism, which is exactly why the 30 ms figure was always
quoted per query *in a batch of 16*.

### 12.2 What batching does and does not buy

Expand and folds are inherently per query — each query has its own Subs tree and its own fold RGSWs —
so only the scan is shared. Measured (batch 4, steady state):

| | Q=1 | Q=4 |
|---|--:|--:|
| expand + convert + folds | 418 ms | ~406 ms/query |
| scan | 407 ms | **109 ms/query** (436 ms for all four) |
| **device per query** | **881 ms** | **574 ms** |
| latency for *your* query | 881 ms | 2.4 s (you wait for the batch) |

So it is a **throughput** mode. The interactive path stays Q=1 on purpose.

### 12.3 The cold pass is O(Q)

Program count scales with the batch: 2 967 at Q=1, 10 356 at Q=4. Every one is built on the first
batch, so pass 1 costs **15.4 s at Q=4** against 2.4 s in steady state. `--batch` therefore runs the
batch twice and reports both, and cross-checks that the two passes return identical rows — which is
also the check that reused programs are not stale.

### 12.4 Verified — and the limit found

All 4 rows correct, cold and warm, at batch 4 — four different indices spread across the database, each
returning its own row from a single shared scan.

**Q=16 does not currently work.** Twice it failed to finish its cold (program-building) pass inside 12
minutes, with the process pinned at ~110 % CPU and ~2 GB RSS, before printing anything. The first
attempt was blamed on pool churn (§11.4); fixing that changed nothing, so the cause is still unknown —
the per-query program count is only *linear* in Q (2 967 → 10 356 from Q=1 to Q=4), so something else
is superlinear. `--batch` therefore warns above 4. The full-32-column figure (~25 ms/query of scan)
remains **projected, not measured**; what is measured is Q=4: 109 ms/query.

## 10. Measured on Blackhole p150b (2026-08-11) — supersedes the n150 figures above

All three demo modes pass on p150b. The timings earlier in this document were taken on Wormhole
**n150**; they are a different machine *and*, for the batch rows, a different measurement. Do not
compare them directly.

| run | result |
|---|---|
| `--checks` | every per-stage isolation check ok; row 40000 matches `db65536.txt` |
| `--idx 65535` | correct row; **4299 ms** on-device |
| `--batch 3,777,40000` | **3/3 correct** |

Steady state, batch of 3 (`--batch 3,777,40000`):

    [0] fold RGSWs 224 ms | [1] expand 280 ms | [2] convert 79 ms
    [3] scan 190 ms (+8 ms DB) | [4] folds 93 ms | [5] decrypt 72 ms
    server on-device compute: 867 ms for 3 queries = 289 ms/query (scan 63.3 ms/query)
    3/3 correct | 949 ms wall for the batch = 316 ms/query

The cold pass is **21869 ms**, so record/replay buys ~25x. `programs: ... total 0` on the query path
confirms nothing is rebuilt per query: compile once per DB *structure*, replay per query.

`--batch` widths 1, 2, 3 and **5** all return correct rows (Q=5: `3,777,40000,12345,65535` -> 5/5).
The constraint is cold-pass cost, not correctness:

| Q | cold pass | steady state | per query | scan/query |
|---|---|---|---|---|
| 3 | 21.9 s | 867 ms | 289 ms | 63.3 ms |
| 5 | **571 s** | 1317 ms | 263 ms | **39.8 ms** |

Steady state gets *better* with width (the shared scan amortises), but the cold pass grew ~32x for two
extra queries — which is why Q=16 never finished its. **Budget for it on every process start**: a second
Q=5 run with the on-disk kernel cache warm cost 579 s again, so it is not first-ever JIT compilation.
Program count is linear in Q (7 821 at Q=3, 12 699 at Q=5); the per-program cost is what grows, and at
Q=3 the host split is 99.9 % workload ENQUEUE at 2.004 ms/call against 0.019 ms/call on replay. See
`docs/pir/pir.md` for what has been ruled out. Above Q=5 is unprofiled.

See `docs/pir/pir.md` §7.3 for the four bugs fixed to get here, one of which (`bZeroN` drawing from
the scratch pool) was found only by running `--batch`, because it needs a cold *and* a warm pass.

> **STAGE ORDER NOTE (2026-08-17).** Older snippets in this file print stages as
> `[0] fold RGSWs | [1] expand | [2] convert | [3] scan | [4] folds | [5] decrypt`. That was the
> print order, **not** the execution order. Measured order is
> `expand -> convert -> scan -> fold RGSWs -> folds -> decrypt`: "fold RGSWs" prepares this
> query's fold operands (672 host NTTs + Montgomery) and runs immediately before the folds that
> consume it. The numbers in those snippets are correct; only their order was misleading. The CLI
> and the UI now both print in execution order.
