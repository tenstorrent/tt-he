# Porting to Blackhole (p150) — build flow + the mid-kernel re-init deadlock

Record of moving this repo from its original Wormhole **n150** VM to a Blackhole **p150** VM
(2026-07-28). Two independent problems had to be fixed: the **build** (tt-metalium was never
installed on the new VM) and a **latent TRISC deadlock** in the fused matmul kernels that hung every
NTT-based binary. Both are fixed; `tools/bench/hemul_cross.sh` runs bit-exact end to end.

The deadlock is **not Blackhole-specific** — it was an unsupported use of the LLK init API that
happened to survive on Wormhole. The fix belongs on both architectures.

---

## 1. Build: tt-metalium must be *installed*, not just built

A tt-metal source build is not consumable by this repo. `find_package(TT-Metalium)` needs
`Metalium.cmake` and `tt-logger-targets.cmake`, which **only `cmake --install` generates** — pointing
`CMAKE_PREFIX_PATH` at `tt-metal/build_Release` fails.

`cmake --install` on that build tree also does not work out of the box, for two non-obvious reasons:

1. Several CPM deps and tools are `EXCLUDE_FROM_ALL` yet still carry `install()` rules, so a normal
   `ninja` never builds them and the install dies partway with
   `file INSTALL cannot find .../libfmt.so.11.1.4`. The set that must be built explicitly:
   **`fmt`, `generate_rank_bindings`, `precompile_fw`, `precompile-fw`** (the last is the custom
   command that generates `tt_metal/pre-compiled/`).
2. A full-tree install additionally pulls unbuilt test/benchmark deps (gtest, benchmark). Installing
   only the **metalium components** avoids them.

> **Trap:** `ninja -n` reports only the CMake-regen step here, so the build looks complete when it is
> not. Do not use it to conclude the tree is fully built.

`tools/install_ttmetalium.sh` encodes all of the above:

```bash
sh tools/install_ttmetalium.sh -s /path/to/tt-metal                      # -> $HOME/work/ttm-install
sh tools/install_ttmetalium.sh -s /path/to/tt-metal -p /usr/local -S     # system-wide (sudo)
```

### Unified env flow

Superseded on 2026-07-31 by a config-file layer (restructure plan §3.1): dependency locations live in
`config.local.env` (per-machine, gitignored) over `config.defaults.env` (committed), and **both `env.sh`
and CMake read them** — so `cmake` works whether or not the env was sourced. `env.sh` still owns
`TT_METAL_HOME`, `TT_METAL_RUNTIME_ROOT`, `CMAKE_PREFIX_PATH`, `LD_LIBRARY_PATH`; nothing else sets them.

```bash
sh tools/setup.sh      # detect the prefix, write config.local.env, verify everything
sh tools/doctor.sh     # re-check any time; every failure prints the command that fixes it
. env.sh && cd device && cmake -B build -G Ninja && cmake --build build
```

Resolution order: environment → `config.local.env` → `config.defaults.env` → autodetect
(`/usr/local`, `$HOME/work/ttm-install`, `<repo>/../ttm-install`). A candidate is valid only with both
`lib/cmake/tt-metalium/tt-metalium-config.cmake` and `libexec/tt-metalium`; an *explicitly configured*
prefix that fails that test is a hard error rather than a silent fallback to some other SDK.

Both entry-point scripts source it (`tools/bench/hemul_cross.sh`, `compiler/run.sh`); neither
hardcodes `/usr/local` any more. `run.sh` previously used hard `export`, which **silently overrode**
a correct environment — that was its own portability bug.

---

## 2. The deadlock: full re-init mid-kernel hangs MATH against PACK

### Symptom

Every binary that goes through `kernels/shared/compute/fused_matmul_a.cpp`, `fused_matmul_c.cpp` or
`kernels_ir/compute/matmul_modq.cpp` hung
forever (host spin-polling a completion that never arrives). Everything else ran correctly.

| works | hangs |
|---|---|
| `vector_modmul`, `test_decompose`, `test_recombine` (SFPU only) | `fused_modmatmul` |
| `modular_matmul` (int8 matrix engine, **separate programs** per phase) | `ntt1024_dev`, `nntt{1024,2048,8192}_dev` |
| `baseconv_fused` (fused, but SFPU only) | `he_mul_cross`, and the whole NTT chain |

The split is exact: those three are the **only** kernels in the repo
that mix `unary_op_init_common` (SFPU) with `mm_init` (matrix engine) in one kernel. The int8 matrix
engine itself is fine — `modular_matmul` passes `mismatches=0/1024` — as is multi-phase fusion per se
(`baseconv_fused` passes).

### Root cause

`unary_op_init_common` and `mm_init` both re-run the **DEST-sync handshake**:

```
MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
PACK((llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>()));
```

Re-running that *mid-kernel*, once per phase, deadlocks MATH against PACK. Watcher waypoints planted
at the phase boundaries show it directly on the one active core:

```
CWFW, CRBW, A2MQ, A2MP, A2MP
 │     │     │     │     └─ PACK   (TRISC2): stuck INSIDE mm_init
 │     │     │     └─────── MATH   (TRISC1): stuck INSIDE mm_init
 │     │     └───────────── UNPACK (TRISC0): passed mm_init  (A2MQ)
 │     └─────────────────── reader (NCRISC): cb_reserve_back, input CB full
 └───────────────────────── writer (BRISC) : cb_wait_front, c_16 never arrives
```

Two of three TRISCs never return from `mm_init`; the waypoint *inside* the matmul loop (`A2MM`) is
never reached. No shipped tt-metal kernel mixes those two inits — the pattern is outside the
supported path.

### Fix

One full init at kernel entry, then only **light-weight** per-phase switches. `matmul_init`
(previously `mm_init_short`) configures math+unpack for matmul and does **not** touch dest-sync;
formats are switched with `reconfig_data_format*` / `pack_reconfig_data_format`.

```cpp
// phase 2 — switch to the matrix engine (fused_matmul_a: in0=cb_w, in1=cb_limbs)
reconfig_data_format(cb_limbs, cb_w);      // srcA<-cb_limbs, srcB<-cb_w (mm_init's reversed order)
pack_reconfig_data_format(cb_planes);      // pack int8 (c_2) -> int32 (c_3)
matmul_init(cb_w, cb_limbs, 0 /*transpose*/);

// phase 3 — switch back to the SFPU
reconfig_data_format_srca(cb_planes);
pack_reconfig_data_format(cb_out);
```

Fusion is preserved — still one program dispatch per stage, no perf given up. The fix also *removes*
work (a full `hw_configure` per phase becomes a light reconfig).

**Three files carry this pattern, not two.** `kernels_ir/compute/matmul_modq.cpp` has the identical
sequence and was fixed later, on 2026-07-31, when a full 76-binary sweep found `test_ir_matmul` as the
only remaining hang (exit 124). The original investigation missed it because the pattern grep covered
`kernels/shared/compute/` only — `kernels_ir/` is a second, parallel kernel tree. **Lesson: grep the
whole repo for `mm_init` + `*_init_common` co-occurrence, not one kernel directory.**

### Debugging notes (for the next time this shape of bug appears)

- **DPRINT hides it.** The same binary passes with `TT_METAL_DPRINT_CORES` set and hangs without it —
  DPRINT serializes the TRISCs. Anything timing-sensitive must be diagnosed with **watcher waypoints**
  (`api/debug/waypoint.h`, `WAYPOINT("ABCD")`, 4 chars): a single L1 mailbox write, no synchronization.
  Run with `TT_METAL_WATCHER=<seconds>` and read `generated/watcher/watcher.log`; the 5 waypoint fields
  per core are `BRISC, NCRISC, TRISC0-unpack, TRISC1-math, TRISC2-pack`.
- Waypoints ending in `D`/`Q`-style names are often *completion* markers (`UPAD` = unpack **done**,
  `MWDD` = math dest-wait **done**), not blocks — check the LLK header before concluding a thread is
  stuck there.
- **A hung run wedges the device.** The *next*, innocent program will also hang until `tt-smi -r`.
  Reset between runs while bisecting, or you will chase phantoms.
- **`tt-smi -r` returns before the card is ready.** It prints "Re-initializing boards after reset..."
  and exits while that is still happening, so a program started immediately after can fail spuriously —
  observed as `exit=2` from one binary and `SIGABRT` from the next, both passing on retry. Anything that
  resets automatically must pause afterwards (`tools/doctor.sh --fix` and `tools/run_all.sh` sleep 3),
  otherwise the reset manufactures exactly the false failures it exists to prevent.
- API drift on current tt-metal (`v0.76.0-dev20260728`): DPRINT is printf-style now
  (`DPRINT("fmt", ...)`; old `DPRINT << x` is a hard `static_assert`), and debug headers moved under
  `api/` (`api/debug/dprint.h`, `api/debug/waypoint.h`).

---

## 3. Results on p150 (Blackhole)

All previously hanging, all now bit-exact:

| target | result |
|---|---|
| `fused_modmatmul` | `mismatches=0/1024` |
| `ntt1024_dev` / `nntt1024_dev` | `mismatches=0/1024` |
| `nntt2048_dev` | `mismatches=0/2048` |
| `nntt8192_dev` | `mismatches=0/8192` |
| `he_mul_cross` | `device-vs-oracle = 0/114688` **PASS** |

`tools/bench/hemul_cross.sh` (traced resident, profiler on, 20 iters):

| metric | 16 cores, as ported | **64 cores, optimized (current)** |
|---|--:|--:|
| host wall | 25.521 | **24.429** |
| DEVICE-BUSY span | 20.640 | **14.092** |

The optimized column is 2026-08-03; see §5 for what changed and why. Both are bit-exact
`device-vs-oracle = 0/114688`, 46 segment replays.

### Comparison against the n150 numbers in `OPTIMIZATION_REPORT.md` — read with care

The report's traced-resident figures (host wall 83.8, device-busy 38.0) were measured at
**102 trace segments / 2065 cached workloads**. The current code produces **46 / 1219** — roughly half
the dispatches — so the code changed after that report was written. Since the report attributes much of
its 38 ms to *"102 per-segment replay/sync overheads"*, that structural change would also speed up the
n150. **The 3.3× / 1.8× apparent gain therefore conflates hardware with code changes and should be
treated as an upper bound on the hardware gain.**

A better-isolated hardware comparison is the standalone NTT (one kernel, no host orchestration), both
bit-exact, same core counts, vs `cross_comparison.md` §7.5:

| `nntt8192_mc` | n150 (docs) | p150 | ratio |
|---|--:|--:|--:|
| 32 cores | 162 µs | **80.8 µs** | 2.00× |
| 64 cores | 137 µs | **68.9 µs** | 1.99× |

~2× consistently. Caveat: this path also goes through `fused_matmul_a/c`, and the fix above removes a
per-phase `hw_configure`, so some of the 2× may be the fix rather than the silicon. Splitting the two
requires running the **current** code on an n150.

Note `he_mul_cross` defaults to the full grid when `NTT_CORES` is unset — on p150 that is 13×10 = 130
cores (n150: 8×8), which reports 380.6 ms/mul at `HE_ITERS=2`. The bench script pins `NTT_CORES=16`.

---

## 4. Open items

- ~~`tools/bench/hemul_cross.sh` step [3/3] skips its trace breakdown~~ — **fixed 2026-08-03**: `trace_hemul.py`
  did not exist anywhere in the repo because it lived in a gitignored `generated/` path on the old VM.
  The analyzers now live in [`tools/traces/`](../../tools/traces/README.md), which is tracked, and step
  [3/3] runs. Both headline numbers come from step [2/3] and were never affected.
  Note that `HE_PROF2`'s capture is *incomplete* — the device profiler buffer overflows partway through a
  ct_mul and drops markers. For a full trace use `HE_PROF_DRAIN=1`; see
  [`tools/traces/README.md`](../../tools/traces/README.md).
- ~~`device/he/generated/*` tracked and rewritten by every run~~ — **fixed 2026-07-31**: the ignore rule
  covered `device/generated/` only, while tt-metalium writes to `<cwd>/generated`, so artifacts landed in
  three different places. Rule is now `generated/` (any depth) and the 7 tracked files are untracked.
- Driver logs `Failed to set initial power state: -22` / `Setting power state failed` on this VM. Benign
  so far — every benchmark passes bit-exact.


## 5. Optimization pass (2026-08-03) — 20.85 → 14.09 ms/mul

> **Timebase.** The per-kernel µs figures in this section come from the profiler and were computed
> with the old `1 cycle = 1 ns` assumption. The CSV is in **device cycles at AICLK = 1350 MHz**, so
> every absolute µs here is **1.35× too high** — divide by 1.35 for real time. Every *ratio* and
> percentage is unaffected (the factor cancels), so the conclusions and the ranking of optimizations
> stand, as do the device-busy latencies, which are host-measured. See
> [`exp_tb001.md`](exp_tb001.md#device-timebase--every-trace-derived-number-was-135-too-high).

**The workload is dispatch-bound, not kernel-bound.** Measured: 1219 program dispatches per ct_mul at
~16 µs of wall each ≈ 19.5 ms, which is the whole device-busy span. Per dispatch only ~11.5 µs was
compute — the other ~4.5 µs is fixed launch/teardown. So the lever is *fewer, fuller programs*, and
making a kernel faster without removing a dispatch buys almost nothing.

| change | dispatches | device-busy | |
|---|--:|--:|---|
| baseline (16 cores) | 1219 | 20.85 ms | — |
| `reader_fused_batch.cpp` — one barrier instead of 17 | 1219 | 17.95 ms | 1.16× |
| `modadd_twiddle.cpp` — fused reduce+twiddle (16 cores) | 1130 | 17.68 ms | 1.18× |
| **`NTT_CORES=64`** — wider matmul waves | 764 | **14.09 ms** | **1.48×** |

Cores-per-mul sweep on the final build (device-busy / dispatches):

| cores | 4 | 8 | 16 | 32 | **64** |
|---|--:|--:|--:|--:|--:|
| device-busy (ms) | 41.62 | 22.90 | 17.68 | 15.16 | **14.09** |
| dispatches | 3236 | 1618 | 1130 | 886 | **764** |
| µs per dispatch | 12.86 | 14.15 | 15.64 | 17.11 | 18.46 |

The last row is the load-bearing one: **per-dispatch cost grows only ~1.44× for 16× the cores.** That
sub-linearity is what makes wider programs win, and it is also the basis for batching independent muls
(§7.4 of `cross_comparison.md`). Note the optimum moved: before the reduce+twiddle fusion, 64 cores
measured 18.06 ms and 32 was best — the fusion removed a dispatch whose cost scaled badly with core
count. Re-run the sweep after any change to the dispatch structure.

1. **Reader barrier-per-tile** (`reader_fused.cpp` did `read; barrier; push` for each of 17 tiles, so
   every read paid full DRAM latency in series — 8.77 µs to deliver 17 KB, ~2 GB/s). The reads are
   independent; issuing them all behind one barrier took `ph_matmul` from **6.82 → 1.25 µs** (−82%)
   and the whole `fused_matmul_a` from **16.44 → 11.45 µs**. The matmul phase had been tracking the
   reader (0.5 µs/tile delivered vs 0.43 µs/tile consumed), not the matrix engine.
2. **Core count.** Stage-1 is 64 tile-jobs chunked by `NTT_CORES`, so more cores = fewer waves.
   64 is the optimum on the final build; see the sweep above.
3. **Fused reduce+twiddle.** `modadd_reduce` → `modmul_compute` were consecutive elementwise passes
   with a full 8-tile DRAM round trip between them purely to hand one tile to the next program.
   Merged, the accumulator stays in DEST. Forward NTT only — the inverse chain's twiddle has a
   different producer.

Left on the table, in rough value order: the 8-way stage-1 contraction is still materialized to DRAM
(64 tiles out, 64 back in) and could accumulate int32 planes in DEST instead — that would delete
`modadd` entirely and cut 64 recombines to 8-16, but the 8-way plane sum reaches ~2.11e9 against an
int32 ceiling of 2.147e9, so it needs a 4-way split. `ph_recombine` is now the single fattest phase
(5.82 of 11.45 µs). FBC is 12% of device compute at 511 µs/call. And occupancy is still low: 32 of
130 cores, ~8.6 busy on average within them.
