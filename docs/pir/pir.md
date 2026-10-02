# PIR — root document and merge manifest

> ## STATUS: MERGE COMPLETE AND VERIFIED on Blackhole p150b (2026-08-11)
>
> This document was written on the **n150** as a merge instruction. The merge has been executed onto the
> restructured tree, all four bugs it exposed are fixed, and the result is verified end to end.
> **Read this box before trusting any path, status flag or timing further down.**
>
> ### Verification summary
> | check | result |
> |---|---|
> | `ctest -j1` (whole repo, 102 targets rebuilt first) | **100 / 100 passed**, 0 failed (1082 s) |
> | Batched key-switch at B=1 vs single-ct (§7.3) | **0 / 12288**; every chain intermediate 0 |
> | `crtK_b` vs HOST big-integer Garner | **0 / 6144** (was 3787 / 6144) |
> | `crt_and_gadget_b` vs `crt_and_gadget` | **0 / 16384** (was 8785 / 16384) |
> | Query file -> device expansion, 32 selectors | **0 / 393216** (was 393216 / 393216) |
> | Demo `--checks` | all per-stage isolation checks ok; row matches `db65536.txt` |
> | Demo `--idx 65535` | correct row, 4299 ms on-device |
> | Demo `--batch 3,777,40000` | **3 / 3 correct** |
>
> **65 536-row PIR on p150b: 289 ms/query on-device** (867 ms for a batch of 3; scan 63.3 ms/query;
> 316 ms/query wall). Cold pass 21869 ms -> record/replay buys ~25x, with `programs: total 0` on the
> query path.
>
> ### Four bugs, all one family: the engine was ported, the kernels it depends on were not
> TT-Metal **silently ignores surplus runtime args**, so a kernel/host mismatch yields plausible-but-wrong
> values or a hang, never an error. Full detail in §7.3 below.
> 1. `reader_crt_stride.cpp` -- 6 args in-tree vs the branch's 7 (`r1_off`). **Ported.**
> 2. `reader_i32_groups.cpp` -- a **filename collision**: two different kernels, same name; `reduce_groups`
>    got the wrong one. Landed the branch's as `reader_i32_groups_2l.cpp`. **Both are required.**
> 3. A `//` comment appended to a line holding three `SetRuntimeArgs` calls swallowed two of them.
>    **Self-inflicted during the merge.**
> 4. `bZeroN` drew from the scratch pool while memoizing -- a slot consumed on the cold pass and not on
>    the warm one. **Only `--batch` could expose it** (it needs a cold *and* a warm pass).
>
> ### New / changed artifacts
> * `kernels/shared/dataflow/reader_i32_groups_2l.cpp` -- new, keep alongside the same-named original.
> * `tests/pir/test_ks_batched_b1.cpp` (+ `he_device_target`) -- the regression test the four batched
>   helpers never had; uses host Garner as ground truth so it assumes no device path is correct.
> * `.gitignore` -- 7.6 GB of generated demo artifacts (`*.pir`, `*.fold`, `db65536.txt`, `query*.bin`)
>   were NOT ignored and would have been swept into a commit.
>
> ### Batch width: correct at Q=5, but the COLD pass scales badly
> Q = 1, 2, 3 and 5 all return correct rows (`--batch 3,777,40000,12345,65535` -> **5/5 correct**).
> The limit is not correctness, it is compile/record time:
>
> | Q | cold pass | steady state | per query | scan/query |
> |---|---|---|---|---|
> | 3 | 21.9 s | 867 ms | 289 ms | 63.3 ms |
> | 5 | **571 s** | 1317 ms | 263 ms | **39.8 ms** |
>
> Steady state *improves* with width -- the shared scan amortises, so scan/query drops 63.3 -> 39.8 ms --
> but the cold pass grows ~32x for two extra queries. That, not a functional limit, is why Q=16 "did not
> complete its cold pass".
>
> **The cold pass is paid on every process start, not once.** A second Q=5 run with the on-disk kernel
> cache fully warm cost 579 s again, so this is not first-ever JIT compilation (a hypothesis the re-run
> killed). `WC` lives in memory; nothing survives the process.
>
> What actually scales -- measured with `prof::report`, after fixing the counters below:
> | Q | programs built | cold pass | ms/program | cold ENQ/call | warm ENQ/call |
> |---|---|---|---|---|---|
> | 3 | 7 821 | 18.0 s | 2.3 ms | **2.004 ms** | **0.019 ms** |
> | 5 | 12 699 (1.6x) | 579 s (32x) | 45.6 ms (20x) | (see below) | |
>
> Program *count* is linear in Q (expand and folds are per-query loops); the per-program cost is what
> explodes. The split is **100% workload ENQUEUE** at both widths -- 2.004 ms/call (Q=3) and 45.699
> ms/call (Q=5) cold, against 0.019 / 0.021 ms/call on replay. Replay is flat in Q; only first enqueue
> hurts. So the cost is first-enqueue device setup, not program building.
>
> **It is positional, not global.** Cold ms/program by stage, in execution order:
> | stage | Q=3 | Q=5 | ratio |
> |---|---|---|---|
> | expand | 1.26 | 1.82 | 1.4x |
> | convert | 1.71 | 3.69 | 2.2x |
> | scan | 2.06 | 12.18 | 5.9x |
> | **folds** (last) | **3.52** | **101.98** | **29x** |
>
> The later a program is enqueued, the more it costs, and **folds alone is 556.8 s of the 580.3 s cold
> pass (96 %)** while being only 43 % of the programs. That is the signature of a per-enqueue cost that
> grows with the number of workloads already recorded -- O(N^2) over the pass -- not of anything
> specific to folding.
>
> Ruled out by experiment, so nobody repeats them: on-disk JIT compilation (a second Q=5 run with the
> cache warm cost the same 579 s); and `WC` vector growth copying its elements (`MeshWorkload`'s move
> ctor *is* `noexcept`, checked with a `static_assert`, so growth moves).
>
> **Where to attack it**, in order of expected payoff:
> 1. **Cut program count.** 12 699 distinct workloads for one batch is the root input to an O(N^2)
>    term. folds is 5 460 of them (1 092 per query) and dominates the cost.
> 2. **Use the trace path.** The engine already has `TR_PHASE` 1/2 with `begin_mesh_trace` /
>    `replay_mesh_trace`; the demo drives only `wl_enable` (the workload cache). A trace records one
>    command per segment instead of one workload per dispatch.
> 3. Only then look inside `EnqueueMeshWorkload` for the O(N) term.
>
> Not attempted here -- this is a characterisation, not a fix.
>
> ### Still open
> * Cold-pass scaling above Q=5 is unprofiled (see the table above). Q=16 untested.
> * Nothing is committed; the work sits on `merge-wh-pir`.
>
> ### Path mapping applied
> | this document says | the repo now uses |
> |---|---|
> | `device/pir/{pir_engine2048,pir_permute,pir_api}.*` | `lib/pir/` |
> | `device/pir/pir_demo65536.cpp` | `examples/` |
> | `device/pir/test_*.cpp` | `tests/pir/` |
> | `device/kernels/` · `device/kernels_ir/` | `kernels/` · `kernels/ir/` |
> | `cpu/pir/`, `cpu/montgomery.hpp` | `reference/pir/`, `tests/common/montgomery.hpp` |
> | `device/CMakeLists.txt` target blocks | root `CMakeLists.txt` via `he_device_target(...)` |
>
> ### Corrections to this document
> 1. **The `U`/`T` status column is STALE.** Every file it marks `U` ("untracked — will not come across in
>    a git-based merge") *is* tracked on the branch, including all 22 kernels, `PIR65536_DEMO.md`,
>    `pir_client.hpp`, `encode_db.cpp` and `pir_demo65536.cpp`. The merge was ordinary git content; §2's
>    headline warning no longer applies. **Ignore the status column.**
> 2. **§5.1 undercounts: it lists 5 cross-tree includes; there are 23.** The other 18 are
>    `#include "../../cpu/montgomery.hpp"` in tests (12 in `device/pir/`, 7 in `device/kernels_ir/`) plus
>    `pir_api.cpp`. Its *recommended* fix (a repo-root include dir) covers all 23 — only the inventory was
>    short. The destination tree already used that convention (`HE_INCLUDE_DIRS`), so the files were
>    normalized to `#include "common/montgomery.hpp"` and repo-root-relative data paths.
> 3. **§5.3 also lists only the `../../cpu/...` form.** `test_permute2048` used a single-`../` path
>    (`"../cpu/pir/permute_case.bin"`) and was missed by a pass written from the table. Scan for both.
> 4. **§5.2's "on every PIR target" is over-broad.** 23 of 39 targets legitimately lack
>    `OVERRIDE_KERNEL_PREFIX_IR`: the `kernels_ir` tests point their *primary* prefix there, and older tests
>    never reach IR kernels. The precise rule is **every target that uses the engine's dual-root paths**.
>    The hazard is real and now guarded: `he_device_target` gained a `KERNELS_IR` option with an
>    `IS_DIRECTORY` check, because a wrong/absent prefix is otherwise a runtime-only failure.
> 5. **Timings in §1 and §7 are n150 figures** (881 ms device, ~1.2 s wall) -- historical, and NOT
>    comparable to the p150b numbers in the box above: different machine *and* different measurement.
>    Blackhole has now been measured (see the verification summary and `pir65536_demo.md` §10).
>
> ### §5.7 — the coupling this document could not know about: the Blackhole re-init deadlock
> `unary_op_init_common` and `mm_init` both re-run the DEST-sync handshake; re-running it mid-kernel
> deadlocks MATH against PACK on Blackhole. It **survives on Wormhole**, where this branch was developed,
> so nine kernels here carried it. Six were unique to this branch and have been fixed in the merge —
> `kernels/shared/compute/fused_matmul_{a2,c2}.cpp` and `kernels/ir/compute/matmul_modq_mbatch{,_c,_k2,_c_k2}.cpp`
> — by doing one full init at entry then `matmul_init` + `reconfig_data_format*` per phase, mirroring
> `mm_init`'s reversed `(in1, in0)` order. **Diagnose with Watcher waypoints, never DPRINT** — DPRINT
> serialises the TRISCs and hides it. Reference: `docs/he/blackhole_port.md` §2.
>
> ### Merge verification so far
> | step | result |
> |---|---|
> | §7.1 `test_client` | ✅ |
> | §7.2 `test_db_encode_check` | ✅ **0/36864** |
> | §7.3 `test_query_file_check` | ❌ **393216/393216 — OPEN, see below** |
> | §7.4 `test_fold_file_check` 0 & 3 | ✅ **0/393216** each |
> | §7.5–7 `pir_demo65536` | pending `query65536.bin` |
> | 13 ported device tests | ✅ |
> | pre-existing PIR gates | ✅ `0/12288` ×3, decrypt `0/2048` |
> | full build / host suite | ✅ 0 errors / 2-2 |
>
> ### OPEN: §7.3 fails — isolated to the CTB path
> Every value mismatches; device output shows non-zero at even coefficient indices and zero at odd.
> **Ruled out by test:** the tree's own expand path (`test_query_expand2048` = 0/32768 + 0/49152 on the
> merged engine); `onionpir_ref.py` divergence (differs only in a doc-path comment); `pir_permute.hpp`
> (differs only in comments); signature drift on the 115 name-colliding methods (only `madd2` differs, and
> the tree's is a superset with `os = 0`); the RNS primes (§5.6 verification passes — ref, `encode_db.cpp`
> and the file header all agree). Pooling-vs-workload-cache conflation was found and fixed along the way
> (`mkbuf` pooled whenever `CACHE_ON` was set; now `pooling` is independent, as on wh) but did not change
> the result.
> **Note `test_expand_bench` is NOT evidence here** — it drives the identical CTB sequence but compares
> against no golden, so it only proves the path runs.
> **Eliminated by test (do not re-check):**
> | hypothesis | how ruled out |
> |---|---|
> | CTB layout / `til2`/`unt2` / `wrb`/`rdb` | `upload_ctb`→`download_ctb` identity probe = **0/12288** |
> | batch automorph / monomial | `test_permute2048` = 0/24576 (K=3 batch automorph) |
> | this tree's expand path | `test_query_expand2048` = 0/32768 + 0/49152 on the merged engine |
> | `onionpir_ref.py` divergence | differs only in a doc-path comment |
> | `pir_permute.hpp` (held file) | differs only in comments |
> | signature drift on name-colliding methods | only `madd2`; the tree's is a superset (`os = 0`) |
> | RNS primes (§5.6) | ref / `encode_db.cpp` / file header all `{268369921, 268361729, 268271617}` |
> | Galois key order (§5.5) | generator writes `lvl→ch→k→col`, test reads `a→ch→k→col`, both `j=(n>>a)+1` |
> | `fwd2048` divergence | different implementation (`fusedA2` fused vs `fusedA`+`madd2`), same output reindex |
>
> ## §7.3 ROOT CAUSE FOUND AND FIXED: two kernels the merge left behind
> The engine was ported from `wh`; two of the *kernels* it depends on were not. Both failures are
> **silent**: surplus runtime args are ignored by the kernel, so a stale kernel reads the wrong tiles
> and produces plausible-but-wrong values instead of an error.
>
> | # | kernel | what went wrong | fix |
> |---|---|---|---|
> | 1 | `dataflow/reader_crt_stride.cpp` | tree's copy takes **6** args (`r1[i] = tile i`); `wh`'s takes **7**, adding `r1_off` so `r1` can be one poly inside a bigger buffer. The ported engine passes `r1_off+o`, which the old kernel dropped — so `stage_crt` read residue 0 from tile 0 instead of tile 6, and under the 2-core split core 1 read r1 tile 0 instead of tile 1 (only core 0 was ever right). | ported `wh`'s 7-arg kernel; added the explicit `0u` r1_off at the two 6-arg call sites (`stage_crt` convenience overload, `tests/common/test_crt_combine_k3.cpp`) |
> | 2 | `dataflow/reader_i32_groups.cpp` | **filename collision, not version skew.** This tree's kernel reads groups of CONSECUTIVE tiles (`{addr,n_groups,gsize,gstride,base}`) for the 7-plane mod-q recombine. `wh`'s kernel of the same name is two-level strided (`{addr,G,n_in,gdiv,gstride,istride,g_start}`) for the level-wide MAC. `reduce_groups` was silently getting this tree's kernel: `gdiv` read as `gsize`, `istride`/`g_start` dropped. | landed `wh`'s as **`reader_i32_groups_2l.cpp`** and repointed only `reduce_groups`. **Both kernels are required — do not merge them.** |
>
> ### How it was found (bisect, each step now a permanent probe in `tests/pir/test_query_file_check.cpp`)
> | probe | result |
> |---|---|
> | `upload_ctb`→`download_ctb` identity | 0/12288 — layout helpers correct |
> | `keyswitch_levelB(B=1)` vs single-ct key-switch | 12288/12288 — minimal repro |
> | `crt_and_gadget_b` vs `crt_and_gadget` | 8785/16384 |
> | `crtK_b` vs `crtK` | 3787/6144 — narrowed to the CRT |
> | `crtK_b` with residues repacked to `src0=0` | 2417/6144 — *not* offset handling; deterministic, so not a race |
> | **`crtK_buf` vs HOST Garner (independent ground truth)** | **0/6144 — the single-ct path is correct** |
> | **`crtK_b` vs HOST Garner** | **3787/6144 — `crtK_b` is genuinely broken** |
> | chain stage-by-stage vs host: `D01,v1,A,Bt,CC,v2` | 0/2048 each — Garner arithmetic clean, so the bug is in the `stage_crt` tail |
>
> Two things that cost time and are worth remembering:
> * **`crtK_b` and all three `stage_*` helpers are byte-identical to `wh`'s.** Diffing the ported C++ can
>   never find this class of bug — the divergence was in a kernel, one directory away.
> * `garner_diff(a,b)` computes **`(b-a) mod q`** (Garner needs `r1-r0`). An early probe "failed"
>   2048/2048 purely because the host formula was backwards; the device was right.
>
> ### Third bug in the same class -- and this one was self-inflicted
> While porting fix 1 I appended `// arg6=r1_off` to a line that held **three** statements:
> `SetRuntimeArgs(p,rd,...); SetRuntimeArgs(p,wr,...); SetRuntimeArgs(p,cp,...)`. The `//` swallowed the
> `wr` and `cp` calls, so `writer_i32` and `crt_mul_split` ran with **zero** runtime args.
>
> Symptoms, and why they misled: `crtK_buf` (this tree's own, previously-correct path) began to hang,
> and every subsequent run hung too because a timeout leaves the Blackhole wedged. I wrongly recorded
> this as a "pre-existing, benign OOB" -- it was neither pre-existing nor benign. What actually found it:
> **Watcher**, which reported `BRISC accessed unique runtime arg index out of bounds. Current kernel:
> writer_i32.cpp`, plus per-stage `Finish` + flush prints to attribute the trip to `crtK_buf`.
>
> Two lessons worth more than the fix:
> * This file packs several statements per line. **Never append a `//` comment to such a line** -- put it
>   at the end, or split the line first. A scan for `//[^"]*SetRuntimeArgs` finds no remaining instance.
> * A "kernel needs N args" bug and a "host passes N args" bug look identical from the outside. Watcher
>   distinguishes them immediately and DPRINT cannot (it serializes the TRISCs). Reach for Watcher first.
>
> ### Verified after all three fixes (`tests/pir/test_ks_batched_b1.cpp`)
> | probe | before | after |
> |---|---|---|
> | `ctb upload->download` identity | 0/12288 | 0/12288 |
> | `crtK_buf` vs HOST Garner | 0/6144 | 0/6144 |
> | `crtK_b` vs HOST Garner | 3787/6144 | **0/6144** |
> | `crt_and_gadget_b` vs `crt_and_gadget` | 8785/16384 | **0/16384** |
> | `keyswitch_levelB(B=1)` vs single-ct key-switch | 12288/12288 | **0/12288** |
>
> The last row exercises `digit_rns_all_b`, `fwd2048_batchM`, `mmul_bcast` and `reduce_groups`, so fix 2
> (`reader_i32_groups_2l`) is covered as well. `test_crt_combine_k3`: 0/16384, passes under Watcher.
>
> ### Suite verification after all three fixes
> `ctest -j1`: **100% passed, 0 failed out of 100** (1074 s serial, Blackhole p150b).
> Includes the two new/changed tests and every pre-existing gate.
>
> **Rebuild ALL targets before running ctest after touching `pir_engine2048.hpp`.** A partial rebuild
> (two targets) left 97 stale binaries carrying the bug-3 line, which produced a 47-failure ctest run
> that looked exactly like a broad regression, plus two wrong diagnoses chased from it. Every test
> statically links the header, so a header edit invalidates all of them.
>
> ### Fourth bug, found by running the 65536 demo: `bZeroN` drew from the scratch pool
> `--batch` failed with `pool order diverged at slot 13: recorded (nt=4,pg=4096) but asked for (nt=32...)`,
> at the same slot for **every** batch width including Q=1, while `--idx` was fine.
>
> Cause: `--batch` runs `answer()` twice (cold, then steady state); `--idx` runs it once. `bZeroN(nt)`
> memoizes into `zeroCache` and so calls `mkbuf` **only on a cache miss** -- it consumed a pool slot on
> the cold pass and none on the warm one, shifting every later slot. Slot 13 held the cold pass's nt=4
> zero buffer; the warm pass put the next real (nt=32) buffer there.
>
> Fix: allocate the zero operand with `pooling` temporarily off. It is a read-only constant and should
> live for the Engine's lifetime, never in the recycled arena.
>
> **General rule for anything the pool feeds: allocation must be UNCONDITIONAL.** Any memoized or
> lazily-created buffer that draws from the pool breaks the invariant the moment its cache hits. The
> pool's order check is what makes this loud instead of silent aliasing -- keep it.
>
> ### 65 536-row demo on Blackhole p150b (all three §7 steps green)
> | run | result |
> |---|---|
> | `--checks` | all per-stage isolation checks ok; row 40000 matches `db65536.txt`; **Demo Passed** |
> | `--idx 65535` | correct row; 4299 ms on-device; **Demo Passed** |
> | `--batch 3,777,40000` | **3/3 correct**; **Demo Passed** |
>
> Steady-state batch of 3: `fold RGSWs 224 | expand 280 | convert 79 | scan 190 | folds 93 | decrypt 72` ms
> -> **867 ms on-device for 3 queries = 289 ms/query** (scan 63.3 ms/query), 949 ms wall = 316 ms/query.
> Cold pass 21869 ms, i.e. the record/replay JIT buys ~25x. `programs: ... total 0` on the query path
> confirms nothing is rebuilt per query.
>
> These supersede the n150 figures quoted from the `wh` branch (881 ms / 1.2 s), which were a different
> machine AND a different measurement -- do not compare them directly.
>
> **Operational note:** every test timeout leaves the Blackhole wedged, and the *next* run then hangs
> with no output — which looks exactly like a new hang in whatever you just changed. `tt-smi -r` plus a
> `test_ntt2048_buf` sanity run between attempts; three "hangs" here were this, not the code.
>
 `subs_levelB` = `automorph_ctb` + `keyswitch_levelB`; the
> automorph half is validated (`test_permute2048`, 0/24576). Inside `keyswitch_levelB`:
> | helper | covered by a passing test? |
> |---|---|
> | `fwd2048_batchM` / `inv2048_batchM` | ✅ `test_ntt_batch_fwd` validates both vs a per-poly device golden |
> | grid geometry (wh ran 8x8, this p150 is 13x10) | ✅ ruled out — forcing GX=GY=8 gives the IDENTICAL failure; `shard_at`/`shard_set` are self-consistent for any GX |
> | `crt_and_gadget_b` | ❌ **none** |
> | `digit_rns_all_b` | ❌ **none** |
> | `mmul_bcast` | ❌ **none** |
> | `reduce_groups` | ❌ **none** |
>
> `test_extprod_fast2048` exercises only the SINGLE-ct forms (`crt_and_gadget`, `digit_rns`, `fwd2048`),
> and `test_expand_bench` drives the batch path but compares against no golden. **The bug therefore sits in
> the repo's one coverage hole**, which is also why every other check came up green. Each of the four has a
> validated single-ct counterpart in this tree, so at B=1 (where CTB layout is defined to equal the
> per-ciphertext layout) each can be diffed against its counterpart in turn — that names the culprit and,
> whatever the outcome, the probes are worth keeping as the regression tests that were missing.
>
> (superseded) Narrowed to: `subs_levelB` (the batched key-switch) and what it consumes — i.e. the ported
> application methods' interaction with this tree's *shared core*, not the layout, keys or primes.
> **Next probe:** bisect the Subs tree by level — compare after level 0 (`Bl=1`, where CTB layout is
> *defined* to equal the per-ciphertext layout, per §3.4, making it a free equivalence check) before
> trusting deeper levels; then check `upload_ctb`→`download_ctb` as an identity round-trip to separate the
> layout helpers from the compute.


Entry point for the private-information-retrieval application in this repo: what it is, every file it
consists of, how those files depend on each other, and **what has to be re-pointed when the repo is
restructured**.

Written to be usable as a merge instruction into a differently-organised tree. The paths below are
this VM's; the couplings in §5 are the things that must survive the move, and §7 is the acceptance
test that proves they did.

---

## 1. What the application is

A working OnionPIR server on one Tenstorrent n150. A client encrypts a row index; the server touches
all 65 536 rows of a real ASCII database without learning which one is wanted; the client decrypts the
row's text. **~881 ms of device compute per warm query, ~1.2 s wall clock, index typed to row printed.**

```
client (offline)  gen_query65536.py ──► query65536.bin   secret + Galois keys (index-INDEPENDENT)
client (online)   pir_client.hpp    ──► query ct + 7 fold RGSWs        306 ms per index
server (offline)  encode_db.cpp     ──► db65536.pir (1.5 GB) + db65536.fold (6.0 GB)
server (online)   pir_engine2048.hpp   expand 1 ct ─► 512 selectors ─► scan 65 536 rows ─► 7 folds
client (online)   pir_demo65536.cpp    Garner-lift, decrypt, ASCII-decode, print
```

### The invariant config — do not drift it

```
n     = 2048      ring degree = characters per row
K     = 3         RNS channels, primes {268369921, 268361729, 268271617}
                  == onionpir_ref.ntt_primes(3, 2048).  THE CLIENT FIXES THESE.  See §5.6.
ell   = 8         gadget length         B = 2^11 = 2048   gadget base
N0    = 512       first hypercube dim   rest = 128        512 * 128 = 65 536 rows
nfold = 7         binary folds (128 -> 1)
t     = 257       plaintext modulus: ONE ASCII CHARACTER PER COEFFICIENT
R     = 2^32      Montgomery radix (cpu/montgomery.hpp)
m_exp = 9         expansion levels      KB = 16 k-tiles   MB = 4 m-tiles
```

This is the same shape as OnionPIRv2's `CONFIG_N2048_K1`, and the config the measurements in
`docs/pir/ONION_MATCH_PERF.md` were taken at. Changing any of it invalidates both the artifacts and
those measurements.

---

## 2. Merge quick-start

Five groups, in dependency order. Everything in §3 is source; nothing generated needs to move (§8).

| # | group | files | notes |
|---|---|--:|---|
| 1 | device kernels | 45 | **22 untracked** — the most likely thing to be lost in a merge |
| 2 | shared headers | 3 | `cpu/montgomery.hpp`, `device/pir/pir_permute.hpp`, `device/pir/pir_engine2048.hpp` |
| 3 | client + offline | 5 | `pir_client.hpp`, `encode_db.cpp`, `onionpir_ref.py`, `gen_db_text.py`, `gen_query65536.py` |
| 4 | demo + tests | 5 | `pir_demo65536.cpp` + 3 file-check tests + `test_client.cpp` |
| 5 | build + docs | 2 + 8 | `device/CMakeLists.txt` target blocks, `docs/pir/*.md` |

Then fix the six couplings in §5, build (§6), and run the ladder in §7.

---

## 3. File manifest

`T` = tracked in git, `T*` = tracked and locally modified, `U` = **untracked — will not come across in a
git-based merge**.

### 3.1 Documentation — `docs/pir/`

| st | file | lines | what it is |
|---|---|--:|---|
| U | `PIR65536_DEMO.md` | 756 | **The primary doc.** Plan, progress log, §8 self-sufficient handoff, §8.5 traps, §9 results, §10 interactive mode, §11 latency ladder, §12 batching |
| T* | `ONION_MATCH_PERF.md` | 769 | Correctness + performance record for the op set |
| T* | `ONION_MATCH_PLAN.md` | 349 | Plan behind that pipeline |
| T* | `PIR_DEMO.md` | 155 | The earlier, smaller n=1024 demo |
| T | `PIR_PERF.md` | 177 | Scan performance history |
| T | `SCAN_ALGORITHM.md` | 185 | First-dimension scan algorithm |
| T | `SCAN_STATUS.md` | 102 | Scan status record |
| T* | `pir_kernel_plan.md` | 310 | Kernel-level plan |

Read order for someone new: this file → `PIR65536_DEMO.md` §0, §8, §9 → `ONION_MATCH_PERF.md` §1a, §2.

### 3.2 Client and shared host code — `cpu/`

| st | file | what it is |
|---|---|---|
| T | `cpu/montgomery.hpp` | `MontgomeryContext`, `to_montgomery(a) = (a<<32) % q`, `modular_inverse`. **Used by both the device engine and the client** — the one file both sides share |
| U | `cpu/pir/pir_client.hpp` | The whole host client: per-prime negacyclic NTT, BFV enc/dec, balanced gadget, external product, `RGSW(s)`, `make_query(idx)`. 306 ms/query vs the reference's 73 s |
| U | `cpu/pir/test_client.cpp` | Client self-test. Every check compares against something independent of the thing under test (§7) |
| T | `cpu/pir/onionpir_ref.py` | The Python reference. **Authority for the prime set, the gadget convention and the hypercube decomposition** |

### 3.3 Offline preparation — `cpu/pir/`

| st | file | produces | cost |
|---|---|---|--:|
| U | `gen_db_text.py` | `db65536.txt` — 65 536 self-identifying ASCII rows | 0.2 s |
| U | `encode_db.cpp` | `db65536.pir` (NTT + **Montgomery**, `[ch][row][2 tiles]`) and, with `--fold`, `db65536.fold` (int8 fold, **PLAIN NTT**, `[m-tile][ch][k-tile][slot][16 tiles]`) | 6.5 s + 24.6 s |
| U | `gen_query65536.py` | `query65536.bin` — query ct + Galois keys + fold RGSWs + secret | 73 s |
| T | `gen_expand_case.py` | (not run by the demo) — but `gen_query65536.py`'s Galois-key serialisation is copied from it verbatim. **Keep them together**; see §5.5 |

### 3.4 Device engine and demo — `device/pir/`

| st | file | lines | what it is |
|---|---|--:|---|
| T* | `pir_engine2048.hpp` | 1207 | **The op library.** Per-channel NTT/CRT/gadget constants, batched NTT, Garner CRT, gadget, key-switch, `subs_levelB`, `scan_mtile`, `fold_levelB`, the CTB batch layout, the scratch pool and workload cache |
| T* | `pir_permute.hpp` | 84 | `tilize_poly` / `untilize_poly` (32×32 face layout) + host automorphism/monomial reference |
| U | `pir_demo65536.cpp` | 596 | **The demo.** Assembly + the three layout conversions + host decrypt/decode. `-i`, `--idx`, `--batch`, `--checks`, `--stream` |

The engine's three stage entry points, all proven by the tests in §3.5:

```cpp
void subs_levelB(MeshBuf bIn, uint32_t B, int64_t j,
                 const std::vector<std::vector<MeshBuf>>& KnB, MeshBuf out, MeshBuf scratch);
void scan_mtile(Chan& c, MeshBuf sX, MeshBuf sW, MeshBuf sPar, MeshBuf sAcc0, MeshBuf sAcc1,
                MeshBuf out, uint32_t KB, uint32_t sOT, uint32_t rest, uint32_t cols = 2);
MeshBuf fold_levelB(MeshBuf acc, uint32_t Bold, const std::vector<std::vector<MeshBuf>>& Cn);
```

**CTB (ct-batch) layout**, used by expand and folds — memorise this, every conversion is stated in it:
`tile of poly (comp, ch, node), half h = ((comp*K + ch)*2 + h)*B + node`. At `B=1` this *is* the
per-ciphertext layout, which makes `B=1` a free equivalence check for anything new.

### 3.5 Verification tests — `device/pir/`

These are the merge acceptance test. Each compares against the *other* implementation, never itself.

| st | file | proves | expected |
|---|---|---|---|
| U | `test_db_encode_check.cpp` | encoder's NTT+Montgomery == device `to_mont(fwd2048(...))` | 0 / 36 864 |
| U | `test_fold_file_check.cpp` | `db65536.fold`'s fold formula and byte layout == what the benchmarked scan reads | 0 / 393 216 per m-tile |
| U | `test_query_file_check.cpp` | `query65536.bin`'s Galois-key serialisation == what the device expand consumes | 0 / 393 216 |
| U | `test_scan_ntt2048.cpp` | the scan itself, against an eval-domain golden | — |
| U | `test_expand_bench.cpp`, `test_fold_bench.cpp` | stage timings at library depth | — |

Also present and PIR-related but not on the 65 536 path: `pir_demo_api.{hpp,cpp}`, `pir_demo.cpp` (n=1024),
`test_extprod2048`, `test_bv_keyswitch2048`, `test_query_expand2048`, `test_permute2048`,
`test_ntt2048_buf`, `test_mux`, `test_modswitch`, `test_initdim`, `test_trace_probe`.

### 3.6 Device kernels — **the highest-risk group**

The engine names kernels as **runtime path strings** resolved at `CreateKernel`. A missing kernel is
not a compile error — it is a runtime failure, or worse a silently different program. 45 distinct
kernels are required; **22 are untracked.**

`device/kernels/` (32 required, 12 untracked) — reached via `OVERRIDE_KERNEL_PREFIX`:

| | compute/ | dataflow/ |
|---|---|---|
| **T** | `crt_limb3` `crt_mul_split` `digit_to_rns` `fused_matmul_a` `fused_matmul_c` `gadget_decompose_signed` `garner_diff` `modadd_map` `modadd_reduce` `modmul_compute` | `reader_ab` `reader_crt_stride`(*modified*) `reader_fused` `reader_gadget` `reader_i32` `reader_i32_stride` `reader_modmul` `reader_pq` `writer_i32` `writer_modmul` |
| **U** | `fused_matmul_a2` `fused_matmul_c2` `modadd_reduce_groups` `modsub_map` | `gather_table` `permute_poly` `reader_fused2` `reader_fused_c2` `reader_fused_off` `reader_gadget_il` `reader_i32_groups` `reader_modmul_bcast` |

`device/kernels_ir/` (13 required, 10 untracked) — reached via `OVERRIDE_KERNEL_PREFIX_IR`:

| | compute/ | dataflow/ |
|---|---|---|
| **T** | `add_modq` | `reader_binop` `writer_i32` |
| **U** | `matmul_modq_mbatch` `matmul_modq_mbatch_c` `matmul_modq_mbatch_k2` `matmul_modq_mbatch_c_k2` | `reader_mbatch` `reader_mbatch_c` `reader_mbatch_k2` `reader_mbatch_c_k2` `reshuffle_gather_mc` `reshuffle_scatter_mc` |

`kernels_ir/dataflow/writer_i32.cpp` is **byte-identical** to `kernels/shared/dataflow/writer_i32.cpp` but
both must exist — the two prefixes are independent roots and each is resolved separately.

To regenerate this list on any tree:

```bash
grep -o 'OVERRIDE_KERNEL_PREFIX "[a-z0-9_/]*\.cpp"'    device/pir/pir_engine2048.hpp | sed 's/.*"\(.*\)"/\1/' | sort -u
grep -o 'OVERRIDE_KERNEL_PREFIX_IR "[a-z0-9_/]*\.cpp"' device/pir/pir_engine2048.hpp | sed 's/.*"\(.*\)"/\1/' | sort -u
grep -o 'stage_ab("[a-z0-9_/]*\.cpp"'                  device/pir/pir_engine2048.hpp | sort -u   # garner_diff, named dynamically
```

The last line matters: `garner_diff.cpp` is passed as a *parameter* to `stage_ab`, so a grep for the
macro alone misses it.

### 3.7 Build — `device/CMakeLists.txt`

One flat standalone tt-metalium project, 90 `add_executable` targets, of which the PIR ones are the
concern. The demo target:

```cmake
add_executable(pir_demo65536 pir/pir_demo65536.cpp)
target_link_libraries(pir_demo65536 PRIVATE TT::Metalium Threads::Threads)
target_compile_definitions(pir_demo65536
    PRIVATE OVERRIDE_KERNEL_PREFIX="${CMAKE_CURRENT_SOURCE_DIR}/kernels/"
            OVERRIDE_KERNEL_PREFIX_IR="${CMAKE_CURRENT_SOURCE_DIR}/kernels_ir/")
```

`test_fold_file_check`, `test_query_file_check`, `test_db_encode_check`, `test_expand_bench` and
`test_fold_bench` use the identical pair of definitions but do **not** link `Threads::Threads` —
only the demo does (the client threads the fold-RGSW prep).

---

## 4. Dependency graph

```
                    cpu/montgomery.hpp
                     │                │
     ┌───────────────┘                └──────────────┐
     ▼                                               ▼
cpu/pir/pir_client.hpp                    device/pir/pir_engine2048.hpp
     │        ▲                                │        ▲
     │        │                                │        │  pir_permute.hpp
     │   test_client.cpp                       │        │  (tilize/untilize)
     │                                         │
     └────────────────┬────────────────────────┘
                      ▼
        device/pir/pir_demo65536.cpp ──runtime──► 45 kernels under
                                                  kernels/ and kernels_ir/
        cpu/pir/encode_db.cpp        (standalone; shares NO header — the NTT is
                                      duplicated into pir_client.hpp on purpose)
        cpu/pir/gen_query65536.py ──import──► cpu/pir/onionpir_ref.py
```

Two duplications are deliberate and must not be "fixed" during a merge:

- **`encode_db.cpp`'s NTT is copied into `pir_client.hpp`.** Both are independently checked against the
  device (`test_db_encode_check`, and the demo's startup `host NTT == device fwd2048` assertion). The
  duplication is what makes the cross-check meaningful.
- **`gen_query65536.py`'s Galois-key writer is copied from `gen_expand_case.py`.** Copying rather than
  re-deriving is what avoided the dict-vs-list bug (§5.5).

---

## 5. Couplings that break on restructure — fix these six

Every one is a *silent* failure or a runtime-only failure. None is caught by the compiler.

### 5.1 Relative `#include` paths crossing the cpu/device boundary

| file | line | include |
|---|--:|---|
| `device/pir/pir_engine2048.hpp` | 26 | `#include "../../cpu/montgomery.hpp"` |
| `device/pir/pir_engine2048.hpp` | 27 | `#include "pir_permute.hpp"` |
| `device/pir/pir_demo65536.cpp` | 39 | `#include "../../cpu/pir/pir_client.hpp"` |
| `device/pir/test_fold_file_check.cpp` | 25 | `#include "../../cpu/montgomery.hpp"` |
| `cpu/pir/pir_client.hpp` | 32 | `#include "../montgomery.hpp"` |

The device tree reaching two levels up into the host tree is the structural assumption most likely to
break. If `cpu/` and `device/` stop being siblings, prefer adding
`target_include_directories(... PRIVATE ${REPO_ROOT})` and rewriting these five to root-relative
includes, rather than patching the `../..` counts.

### 5.2 `OVERRIDE_KERNEL_PREFIX` and `OVERRIDE_KERNEL_PREFIX_IR` — both, on every PIR target

The engine has fallbacks:

```cpp
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
#ifndef OVERRIDE_KERNEL_PREFIX_IR
#define OVERRIDE_KERNEL_PREFIX_IR ""
#endif
```

so **forgetting `_IR` compiles cleanly** and then fails at the first `scan_mtile` — after expand has
already run. Any new PIR target needs both definitions. If the two kernel directories are merged into
one in the new layout, point both macros at it and keep the two `writer_i32.cpp` copies.

### 5.3 Runtime working directory and default data paths

All defaults assume the process runs from `device/build/`:

| file | line | default |
|---|--:|---|
| `pir_demo65536.cpp` | 101–104 | `../../cpu/pir/{query65536.bin, db65536.fold, db65536.txt, db65536.pir}` |
| `test_db_encode_check.cpp` | 28–29 | `../../cpu/pir/db65536.{txt,pir}` |
| `test_fold_file_check.cpp` | 56–57 | `../../cpu/pir/db65536.{pir,fold}` |
| `test_query_file_check.cpp` | 29 | `../../cpu/pir/query_small.bin` |

All are overridable on the command line (`--query`, `--fold`, `--txt`, `--pir`, or positional for the
tests), so the merge can either preserve the layout or always pass explicit paths. Preferring explicit
paths in any new scripts is the lower-risk choice.

### 5.4 `gen_query65536.py` imports `onionpir_ref` as a sibling module

`import onionpir_ref as ref` — resolved by CWD. The two files must stay in the same directory, or
`PYTHONPATH` must include `onionpir_ref.py`'s directory. `cpu/pir/__pycache__/` is stale build output;
do not merge it.

### 5.5 Conventions that are copied, not shared

- `galois_keys_for_expand` returns a **dict keyed by `j = (n>>a)+1`**, not a level-indexed list, and each
  KSK is `ksk[k][col]`. `gen_query65536.py` copies `gen_expand_case.py`'s writer for exactly this reason.
- The demo reads the key bundle back in that same order (`pir_demo65536.cpp:140–143`).

If either file moves without the other, this is a silent garbage-at-the-end failure.

### 5.6 The RNS primes are fixed by the CLIENT

`encode_db.cpp:64` hardcodes `{268369921, 268361729, 268271617}` with the reasoning in a comment. These
must equal `onionpir_ref.ntt_primes(3, 2048)`. An earlier revision used a different, equally valid
triple; both sides passed every self-check, the demo produced fluent garbage, and only the single
shared prime's channel decrypted. `test_fold_file_check` now reads the primes from `db65536.pir`'s
header rather than a local constant, so the class of bug cannot recur silently there — preserve that.

**Verification after a merge:** `python3 -c "import onionpir_ref as r; print(r.ntt_primes(3,2048))"`
must match `encode_db.cpp`'s `PRIMES[3]` and the `K` primes in `query65536.bin`'s header.

---

## 6. Build and run

```bash
# --- offline artifacts (host only, no device needed) ---
cd cpu/pir
python3 gen_db_text.py                                                  # 0.2 s  -> db65536.txt
g++ -O3 -march=native -pthread -o encode_db encode_db.cpp
./encode_db db65536.txt db65536.pir --fold db65536.fold --rest 128      # 6.5 s + 24.6 s
python3 gen_query65536.py --idx 40000                                   # 73 s   -> query65536.bin
python3 gen_query65536.py --N0 32 --nfold 2 --idx 100 --golden --out query_small.bin
g++ -O2 -pthread -o test_client test_client.cpp && ./test_client        # client self-test

# --- device build ---
cd ../../device
cmake -B build -G Ninja && cmake --build build
#   requires tt-metalium installed and TT_METAL_HOME, TT_METAL_RUNTIME_ROOT,
#   CMAKE_PREFIX_PATH, LD_LIBRARY_PATH exported

# --- run, from device/build ---
cd build
./pir_demo65536 -i                    # INTERACTIVE: type a row index, get the row
./pir_demo65536 --idx 12345           # one index, generated by the C++ client
./pir_demo65536                       # the reference query file
./pir_demo65536 --checks              # + per-stage isolation checks (needs db65536.pir)
./pir_demo65536 --batch 3,777,40000   # throughput: one shared scan (verified to 4; see below)
./pir_demo65536 --stream              # do not hold the 6 GB database resident
```

Disk needed for the artifacts: **~8.1 GB / 7.6 GiB** (fold + pir + txt + three query files).
Device DRAM: the fold is 6.0 GB of the n150's 12 GB and is held resident by default, falling back to
streaming automatically if the allocation fails.

---

## 7. Merge acceptance test — run in this order

Each step compares against an implementation independent of the thing under test. Stop at the first
failure; the order localises it.

| # | command | expected | if it fails |
|---|---|---|---|
| 1 | `./test_client` (in `cpu/pir`) | all checks pass | client-only; device untouched |
| 2 | `./test_db_encode_check` | `0 / 36 864` | encoder NTT or Montgomery convention (§5.6, §5.1) |
| 3 | `./test_query_file_check ../../cpu/pir/query_small.bin` | `0 / 393 216` | Galois-key serialisation (§5.5) |
| 4 | `./test_fold_file_check 0` and `./test_fold_file_check 3` | `0 / 393 216` each | fold layout, or a missing `kernels_ir` kernel (§5.2) |
| 5 | `./pir_demo65536 --checks --idx 40000` | selectors one-hot at c0=312; conversion 1 both halves ok; 12 scan m-tile×channel comparisons ok; post-scan acc ok | see the per-stage line that failed |
| 6 | `./pir_demo65536 --idx 65535` | row 65535 returned, `leading index matches request: YES`, `full row matches db65536.txt: YES` | — |
| 7 | `./pir_demo65536 --batch 3,777,40000` | 3/3 correct, cold and warm agree | program reuse / pool arenas |

Step 6's index is deliberately the far corner of the hypercube: `c0=511`, all seven fold bits set,
m-tile 3. Step 5 and step 6 together cover a different first-dimension cell, a different m-tile and the
opposite fold path.

Expected warm timings once it passes (one n150, per query):

```
[0] fold RGSWs 55 ms | [1] expand 288 ms | [2] convert 52 ms
[3] scan 407 ms (+14 ms DB) | [4] folds 78 ms | [5] decrypt 24 ms
server on-device total 881 ms (2 967 programs)   |   wall clock ~1.2 s
```

A first query is cold (~2 928 ms device) because the 2 967 programs are built then; later queries
re-enqueue. One-time at startup: client key setup 81 ms, Galois keys to device 174 ms, 6.0 GB database
resident 1.5–1.7 s.

---

## 8. Do NOT merge — regenerate instead

`cpu/pir/.gitignore` already excludes `*.bin` and `__pycache__/`, but the large artifacts are not
covered and should be excluded explicitly in the new tree:

| path | size | regenerate with |
|---|--:|---|
| `cpu/pir/db65536.fold` | 6.0 GB | `./encode_db ... --fold db65536.fold --rest 128` |
| `cpu/pir/db65536.pir` | 1.5 GB | `./encode_db db65536.txt db65536.pir` |
| `cpu/pir/db65536.txt` | 7.3 MB | `python3 gen_db_text.py` |
| `cpu/pir/query*.bin` | 8.7 MB each | `python3 gen_query65536.py --idx N` |
| `cpu/pir/*.bin` (goldens) | ~35 MB | the matching `gen_*_case.py` |
| `cpu/pir/encode_db`, `cpu/pir/test_client` | binaries | rebuild |
| `cpu/pir/__pycache__/`, `device/build/` | — | rebuild |

`db65536.txt` **is** deterministic given `gen_db_text.py`, so regenerating it does not invalidate a
`db65536.pir` produced from an earlier run — but the safe order is always: regenerate the text, then
re-encode, then regenerate queries.

---

## 9. Traps carried over

The full list with its history is `docs/pir/PIR65536_DEMO.md` §8.5. The ones a restructure can
reintroduce:

**Silent (wrong answer, no error):**
- The RNS primes come from the artifact you must agree with, never a local constant (§5.6).
- **Montgomery vs plain.** `db65536.pir` is Montgomery-lifted (the `pmul` MAC path needs it);
  `db65536.fold` is **plain NTT** (`matmul_modq` is a plain modular matmul). Mixing them is silent.
- `to_montgomery(a) = (a<<32) % q` is `a·R`, **not** `a·R²`. Cost one wrong 1.5 GB file.
- A self-consistency check proves nothing about a convention. `encode_db --verify` passed on a file
  that was 100 % wrong against the device. Always cross-check against the *other* implementation —
  which is what every test in §7 does, and why they are worth carrying across.

**Infrastructure (hang or crash):**
- `DEFAULT_TRACE_REGION_SIZE` is 0; `begin_mesh_trace` without a region **hangs the device** (needs
  `tt-smi -r`). Always `create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 64u << 20)` or larger — the demo does.
- A program cannot be built inside a trace capture; program reuse is a prerequisite for trace, not an
  alternative.
- Program reuse and trace both bake buffer **addresses**: the per-query inputs must be fixed buffers
  refilled in place (`upload_ctb_into`, `rgsw_operand_into`), and each section needs its own pool arena
  (`arena_scan`, `arena_fold`) or the pool drops its tail and reallocates every query.
- The workload cache is index-replayed and cannot self-validate; changing a shape between passes
  requires `caches_clear()`.
- `split()` caps tiles/core but not core count — use `split_stream()` for streaming ops.

---

## 10. Known limits

- **Batching above Q=4 does not work.** Q=16 (the full 32 matmul columns) twice failed to finish its
  cold program-building pass in 12 minutes, at ~110 % CPU and ~2 GB RSS. Program count is only *linear*
  in Q (2 967 → 10 356 from Q=1 to Q=4), so something else is superlinear; cause still unknown. The demo
  warns above 4. Measured: Q=4 gives 109 ms/query of scan against 407 at Q=1. The ~25 ms/query
  full-width figure is **projected, not measured**.
- **The scan (407 ms) is the largest device term**, and batching is the only lever: the query occupies
  2 of the matmul's 32 output columns, and `rest` is already the row axis, so no single-query
  restructuring fills them. (`PIR65536_DEMO.md` §12.1 records this as a correction to
  `ONION_MATCH_PERF.md` §4.)
- **Mod-switch is deliberately skipped** — it only shrinks the response ciphertext, and the client
  decrypts the full-modulus ct directly.
- The client's 293 ms is now comparable to the whole server compute; Barrett/Shoup reduction in the host
  NTT would take it under 100 ms.
- Dead code, harmless: the `if (Q > 4)` warning at `pir_demo65536.cpp:194` is unreachable (`Q` is 0
  there, set later in `resize_for`). The live check is the identical one at `:514`.

> **STAGE ORDER NOTE (2026-08-17).** Older snippets in this file print stages as
> `[0] fold RGSWs | [1] expand | [2] convert | [3] scan | [4] folds | [5] decrypt`. That was the
> print order, **not** the execution order. Measured order is
> `expand -> convert -> scan -> fold RGSWs -> folds -> decrypt`: "fold RGSWs" prepares this
> query's fold operands (672 host NTTs + Montgomery) and runs immediately before the folds that
> consume it. The numbers in those snippets are correct; only their order was misleading. The CLI
> and the UI now both print in execution order.
