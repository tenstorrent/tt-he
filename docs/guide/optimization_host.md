# Optimization — host side

**Status:** Current · One of three, split by where the time goes. Start at
[`optimization.md`](optimization.md) for the order of attack; measure with
[`profiling.md`](profiling.md) before and after.

| | |
|---|---|
| [`optimization_host.md`](optimization_host.md) | dispatch count, occupancy, batching, trace |
| [`optimization_dram.md`](optimization_dram.md) | read depth, bank spreading, NoC, the ceilings |
| [`optimization_kernel.md`](optimization_kernel.md) | SFPI passes, `sfpmul24`, DST traffic, registers |

---

**Diagnosis:** ENQ or BUILD dominates the split. ENQ cost *is* fresh-program serialization — binaries,
CB configs and runtime args re-sent per dispatch — so it and BUILD are one problem, not two.

### The ladder, in the order the data demanded it

PIR answer pipeline, `test_pir_answer2048`, K=3, dims=[8,2,2], bit-exact `0/12288` at every rung
([`../pir/latency_optimization.md`](../pir/latency_optimization.md) §1):

| step | ms/query | vs baseline |
|---|--:|--:|
| baseline (host-orchestrated, warm JIT) | 3450 | 1.0× |
| **1 — workload cache + fixed-address buffer pool** | **241** | **14.3×** |
| 2 — packed transform readbacks | 221 | 15.6× |
| 3 — resident MAC in the external product | 183 | 18.9× |
| 4 — resident digit/RGSW transforms (packed, offset reader) | 164.5 | 21.0× |
| **5 — Metal Trace capture/replay** | **112.5** | **30.7×** |
| **7 — co-scheduled MAC (`mmul_batch`)** | **103.8** | **33.2×** |

The mechanisms, cheapest-first, with the isolated per-dispatch cost from `test_dispatch_micro`
(single-tile modmul, 32 cores, all bit-exact):

| regime | ms/dispatch | vs fresh |
|---|--:|--:|
| build fresh + enqueue | 0.663 | 1× |
| persistent re-enqueue (**workload cache**) | 0.0318 | 21× |
| **traced** replay | 0.0313 | 21× |
| **co-scheduled** — one program, 32 cores | **0.0025** | **260×** |

### The four levers

1. **Workload cache + fixed-address buffer pool** — record programs on pass 1, replay them after. This
   is a **compile-once (JIT) model**, and it is *proven* query-independent: recording on one query and
   replaying a different one rebuilt **0 workloads** and was bit-exact against its own golden. Cached
   workloads bake in **structure** only; data arrives as untraced buffer writes. The buffer pool must
   come with it — trace bakes in addresses, so every buffer inside a trace must be allocated once.
2. **Buffer residency** — keep intermediates on device instead of round-tripping through the host.
   Justified by the READ term, *and* a prerequisite for tracing: **trace segments break at every host
   boundary**, so tracing before residency fragments into ~1,668 segments and gains nothing.
3. **Metal Trace capture/replay** — O(1) in dispatch count, so it only separates from the plain cache
   at large dispatch counts (~183 ms persistent vs a few ms traced at PIR's ~5,898 dispatches).
4. **Co-scheduling** — the largest lever, and the one most often missed. **Separate programs serialize
   on the single command queue regardless of which cores they target.** Only *one program with
   disjoint core ranges* runs them concurrently. Trace removes per-dispatch *host* cost but still
   issues N sequential dispatches; co-scheduling removes the N.

### Fewer, fuller programs

Per-dispatch cost grows **only ~1.44× for 16× the cores**, which is why wider programs win
([`../he/blackhole_port.md`](../he/blackhole_port.md) §5):

| cores/mul | 4 | 8 | 16 | 32 | **64** |
|---|--:|--:|--:|--:|--:|
| device-busy (ms) | 41.62 | 22.90 | 17.68 | 15.16 | **14.09** |
| dispatches | 3236 | 1618 | 1130 | 886 | **764** |
| µs per dispatch | 12.86 | 14.15 | 15.64 | 17.11 | 18.46 |

**Re-run this sweep after any change to dispatch structure.** The optimum was 32 cores until a kernel
fusion removed a dispatch that scaled badly with core count, which moved it to 64.

### Null and negative results — do not re-litigate

| idea | result |
|---|---|
| **Residency first, before the workload cache** (HE Step-3) | **2.8× SLOWER** — 727 → 2047 ms. The on-device transpose added ~720 dispatches, and at 0.57 ms/dispatch that swamped the readback saved. Residency is safe only *after* caching makes a dispatch cost 0.020 ms |
| **Co-scheduling `digit_rns`** in the answer pipeline | **2–3% slower.** Only 48 of 5,619 dispatches were `digit_rns` — consolidating them changes little, while the single large readback lands on the critical path. **Reverted.** Co-scheduling pays in proportion to how many independent ops of that shape exist; re-measure per call site, never assume it transfers |
| **Packed readbacks** (−31% readbacks) as a *speedup* | Total was flat (239.6 → 241.4 ms) — the 25 ms saved was inside the ~12% enqueue noise. Its real value is structural: 522 fewer host boundaries = fewer trace segments |

### Open

**The cold/record pass is O(N²) in dispatch count and is paid on every process start** — 18.0 s at
Q=3 vs 579 s at Q=5 for only 1.6× the programs; 100% first-enqueue, positional (the last stage is 96%
of the cost). Ruled out: on-disk JIT, `MeshWorkload` copy-on-growth. Attack order: cut program count →
drive the existing trace path → only then look inside `EnqueueMeshWorkload`.
([`../pir/pir.md`](../pir/pir.md))

---
