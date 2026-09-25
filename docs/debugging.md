# Debugging Tenstorrent kernels — hangs, deadlocks, and wrong output

Practical reference for debugging our device kernels (BFV, NTT, PIR). Written against Wormhole n150; the techniques apply unchanged on Blackhole p150 (see [`he/blackhole_port.md`](he/blackhole_port.md) §2 for a worked example on p150). Distilled from the
tt-metal repo (`tech_reports/Debugging/Kernel_Debugging_Tips.md`, `tt_metal/tt-llk/docs/`) and the
[Watcher](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/watcher.html) /
[DPRINT](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_print.html) docs,
adapted to our **standalone C++ binaries** (not pytest).

> **First reflex on any hang:** the device is probably wedged. After a `TENSIX TIMED OUT` / kill, run
> **`tt-smi -r`** before the next attempt (`tt_single_device_serial` memory). Reset only for runtime
> hangs, never to "fix" a compile error.

---

## 0. Why compute kernels hang (the mental model)

A Tensix core is **5 RISC-V cores** driving a 3-thread coprocessor:
- **BRISC / NCRISC** — the two **dataflow** RISCs (our `reader_*` / `writer_*` kernels): NOC + circular buffers.
- **TRISC0 / TRISC1 / TRISC2** — **compute** (unpack / **math** / pack): our `compute/*.cpp`.

Almost every *hang* (as opposed to wrong-answer) is a **producer/consumer stall** on a circular buffer
or the tile-register handshake:
- `cb_wait_front(cb, n)` blocks forever if the producer never `cb_push_back`es `n` tiles.
- `cb_reserve_back(cb, n)` blocks forever if the consumer never `cb_pop_front`s enough to free `n`.
- **counts must match end-to-end**: reader pushes = compute pops, compute pushes = writer pops. A
  per-tile mismatch (e.g. multi-pass compute pushing a different total than the writer expects) deadlocks.
- `tile_regs_acquire/commit/wait/release` must be balanced; packing outside a `wait`/`release` window,
  or holding a `cb_reserve_back` across multiple `tile_regs` cycles, can stall the pack thread.

> This is exactly the class of bug in the chunked-gadget `ℓ=8` hang (`ONION_MATCH_PLAN.md` §4): the
> multi-pass path changed the CB push pattern; suspect a reader/compute/writer **count mismatch** or a
> pack/`tile_regs` sync across passes.

---

## 1. Watcher — the primary hang tool

Instruments firmware + kernels and polls device state on a host thread. On a hang it tells you **which
core, which RISC, and the last "waypoint"** each RISC reached.

```bash
# prefix our binary (no pytest):
TT_METAL_WATCHER=1 ./build/test_query_expand reference/pir/expand_case.bin
#   TT_METAL_WATCHER=<seconds>  poll interval (1 = aggressive; larger to perturb timing less)
#   TT_METAL_WATCHER_DUMP_ALL=1 dump full state each poll (heavy; use a large interval)
#   TT_METAL_WATCHER_APPEND=1   append instead of truncating the log
```

When it hangs, **kill it and read the log** (startup prints the exact path):
```
$TT_METAL_HOME/generated/watcher/watcher.log
```
What to look for per core line:
- **Waypoint** (≤4 chars) — the last code block each RISC executed. A RISC parked on a CB-wait waypoint
  (e.g. `CWFW` = cb_wait_front wait, `CRBW` = cb_reserve_back wait) with its partner *not* on the matching
  op = your producer/consumer mismatch. The **stuck RISC + waypoint localizes the hang to reader vs
  compute vs writer**.
- **NOC sanitization** — Watcher stops a RISC that issues a transaction with a bad coord/addr (soft hang);
  the log names the offending core/addr. Catches out-of-bounds DRAM/L1 access.
- **Ring buffer** (`WATCHER_RING_BUFFER_PUSH()`) — last 31 values pushed per core.
- **Stack usage** — per-RISC high-water mark after each kernel (catches stack overflow).

Add your own waypoints to bracket suspect regions in a kernel:
```cpp
#include "debug/waypoint.h"
WAYPOINT("PS0");   // before pass 0
WAYPOINT("PS1");   // before pass 1
```

> **Develop with `TT_METAL_WATCHER=1` on.** Caveat: if a hang *disappears* under Watcher, it's timing-
> sensitive — reduce Watcher features rather than chase it blind.

---

## 2. tt-triage — stack traces on every RISC while hung

When the process is **still hung** (don't kill it), from another shell:
```bash
cd $TT_METAL_HOME && ./tools/tt-triage.py --verbosity=4 --dev=0
```
Dumps stack traces on all RISCs of all cores — pinpoints the exact instruction each core is spinning on.
(Still maturing; may not always attach.)

---

## 3. DPRINT — deterministic hang + wrong-output debug

Print from inside a kernel (compile-time gated; near-zero cost when off):
```cpp
#include "debug/dprint.h"
DPRINT("pass={} k_lo={}\n", p, k_lo);          // scalar trace
```
Enable per core when launching:
```bash
TT_METAL_DPRINT_CORES="(0,0)" ./build/test_query_expand ...      # our demo uses core {0,0}
#   TT_METAL_DPRINT_CORES=all           every core
#   TT_METAL_DPRINT_RISCVS=TR1          restrict to a RISC (BR/NC/TR0/TR1/TR2)
```
Prints appear on stdout **interleaved with the run**. Trace *how far* the loop got before the stall —
the last DPRINT before silence brackets the hang deterministically.

**Print CB tile contents** (wrong-answer / PCC debug) — from
`tt_metal/hw/inc/debug/dprint_pages.h`:
```cpp
// compute kernel: dump a tile from a CB
print_full_tile(cb_id, tile_id, /*untilize=*/false);
// dataflow kernel: print_bf16_pages / print_f32_pages
```

---

## 4. The stdout-buffering gotcha (bit us already)

Our binaries `fmt::print` to **stdout**, which is **fully buffered when not a TTY** (i.e. behind a pipe
or `>`). On a `timeout`-kill the buffer is **lost** — you see nothing and can't tell where it stalled.

- **Don't** diagnose a hang through `| grep …` — the buffer never flushes.
- The **tt-metal logger writes to stderr** and is far less buffered — its lines (JIT build, device open,
  `EnqueueMeshWorkload`) *do* stream, so redirect and watch **stderr**:
  ```bash
  TT_METAL_WATCHER=1 timeout 120 ./build/test_query_expand case.bin > run.log 2>&1 &
  tail -f run.log     # watch device-log progress; last line ≈ where it stalled
  ```
- If it stalls **before** any device log → JIT/compile issue. **After** `EnqueueMeshWorkload` with no
  read completion → device-side CB/compute deadlock → go to Watcher (§1).

---

## 5. General reduction tactics (fastest path to a fix)

From the tt-metal debugging checklist:
1. **Minimize to a unit test**: one op, smallest shape. E.g. reproduce the gadget hang at `n=1024,
   nvec=1` (one tile) before the full n=2048/128 MB case — a 1-tile hang is far easier to reason about.
2. **Single core, single device** — we already are.
3. **Program cache off** if used.
4. **Fixed inputs** (all 0s / 1s) to separate control-flow bugs from data bugs.
5. **`tt-smi -r` between runs** so state doesn't leak (a wedged device makes the *next* run look broken).

---

## 6. Wrong-answer (not hang) checklist

- Recompute/JIT: confirm the binary's compile-time defines (`GADGET_ELL`, `GADGET_B`, `LOG2B`) match the
  case header — our tests assert this, but a stale build can mismatch.
- Tilize/untilize order (`pir::tilize_poly`/`untilize_poly`) and the four-step `reindex` — a permutation
  bug shows as a *shuffled* result, not random.
- Montgomery domain: mixing plain vs Montgomery inputs to `modmul` gives a consistent scale-factor-off
  result. Cross-check one tile against the Python golden with `print_full_tile`.
- Always validate against the CPU golden (`reference/pir/onionpir_ref.py`, `reference/he/*_ref.py`) tile-exact.

---

## 7. Quick recipe for the current gadget-`ℓ=8` hang

```bash
tt-smi -r
# 1-tile reproducer, watcher on, output to a file (not a pipe):
python3 reference/pir/gen_gadget_case.py --n 1024 --ell 8 --B 256 --nvec 1 --out gadget_l8_tiny.bin
TT_METAL_WATCHER=1 timeout 120 ./build/test_gadget_decompose_l8 reference/pir/gadget_l8_tiny.bin \
    > /tmp/l8.log 2>&1 &
tail -f /tmp/l8.log                                   # JIT vs device stall?
# on hang: kill, then
cat $TT_METAL_HOME/generated/watcher/watcher.log      # which RISC + waypoint is stuck?
```
Expected read of the result: if TRISC2 (pack) or the writer's BRISC is parked on a CB-wait while the
compute finished fewer pushes than `G·ELL`, it's the **multi-pass push-count** bug; if a compute RISC
(TRISC1) is stuck mid-kernel, it's the SFPU/`tile_regs` re-entry across passes.

---

## 8. Worked example — the chunked-gadget "hang" (resolved)

Symptom: `test_gadget_decompose_l8` (ℓ=8) ran forever; ℓ=6 was fine. Debug path that worked:
1. **Buffering** hid the output → redirected to a file, watched **stderr** (§4). Saw it stall *after*
   device dispatch, not in JIT → device-side.
2. **Watcher** (`TT_METAL_WATCHER=1`, log at `device/generated/watcher/watcher.log`): worker core
   waypoints `CWFW, W, W, W, W`, k_ids `8|7|9|9|9` → **BRISC (`writer_i32`) stuck on `cb_wait_front`**
   for `cb_out`; reader + all 3 compute TRISCs idle. ⇒ compute **under-produced** the tiles the writer
   awaited (writer wanted `G·ℓ=8`, got 6).
3. **DPRINT** (`DPRINT_MATH("… {}\n", …)`, `TT_METAL_DPRINT_CORES="(0,0)"`) printed
   `ELL=6 CHUNK=6 NPASS=1` — the kernel was running **ℓ=6**, not 8.
4. **Root cause (not the kernel):** `test_gadget_decompose.cpp` never passed `GADGET_ELL/B/LOG2B` as
   compute-kernel JIT **`.defines`** — the kernel used its `#ifndef` default `ELL=6`. The host's
   `-DGADGET_ELL=8` only affected host code. Fix: forward the macros via `ComputeConfig{.defines=…}`.
   (`test_extprod`/`test_bv_keyswitch` already did this; the gadget unit-test did not.)

Lesson: a compute kernel's behavior comes from its **JIT `.defines`**, not the host binary's macros —
always forward parameters both places. `DPRINT`-ing the compile-time constants is the fastest way to
catch a silent default.

## 9. Worked example — CircularBuffer depth must be a multiple of the group size (bit 3×)

Symptom while bringing up K=3 (3-limb / 3-output) kernels: the **low-order** output tile is correct
everywhere, the **high-order** tiles are wrong, and *only on alternating (odd) coefficient-tiles*.
Root cause: a CB whose DEPTH is not an integral multiple of the per-group push/pop count. When the
producer runs ahead of the consumer, one group's tiles straddle the CB wrap boundary; `get_write_ptr`
+ manual `ts` offsets (readers) or the reserve/pack window (compute) then land a tile in the wrong
L1 slot. Because the corruption only hits groups that happen to wrap, it looks tile-parity-dependent.

Seen twice in one session:
- `gadget_decompose_signed` input `c_0` depth **5** with a group of **3** limbs (K=3) → hi limb (bits
  56-83, i.e. digits 5-7) wrong on odd tiles; digits 0-4 use only lo/mid so looked fine.
- `crt_limb3` output `c_16` depth **4** with a push-group of **3** → X1/X2 wrong on odd tiles, X0 fine.

Fix: size every CB as `depth = 2 * group` (clean double-buffer) — never an awkward `group+slack`.
Diagnosis path: readback the *intermediate* DRAM buffers and confirm they are correct (isolating the
fault to the consuming kernel, not its inputs). Note `EnqueueReadMeshBuffer` **requires
`blocking=true`** — a non-blocking read throws `TT_FATAL ... Non-Blocking reads are not supported`.

## Sources
- `tt-metal/tech_reports/Debugging/Kernel_Debugging_Tips.md`
- [Watcher — TT-Metalium docs](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/watcher.html)
- [Device Print (DPRINT) — TT-Metalium docs](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_print.html)
- `tt-metal/tt_metal/tt-llk/docs/tests/debugging_guide.md`, `tt_metal/tt-llk/.claude/CLAUDE.md` (Tensix model)
