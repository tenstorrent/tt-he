# Optimization — where to start

**Status:** Current · Normative. Measure with [`profiling.md`](profiling.md) before and after; a
number produced outside those rules is not quotable.

Split by where the time goes, because the three have almost nothing in common:

| | covers | typical lever |
|---|---|---|
| [`optimization_host.md`](optimization_host.md) | dispatch count, occupancy, batching, trace | fewer, fuller programs |
| [`optimization_dram.md`](optimization_dram.md) | read depth, bank spreading, both NoCs, the ceilings | reads in flight before the barrier |
| [`optimization_kernel.md`](optimization_kernel.md) | SFPI passes, `sfpmul24`, DST traffic, registers | collapse passes, keep intermediates in registers |

Ready-to-use kernels carrying these: [`../../kernels/OPTIMISED.md`](../../kernels/OPTIMISED.md).

---

Fix the term that dominates, then **re-measure** — the profile inverts as you go, and the previous
winner stops being the lever.

| the split says | bottleneck | section |
|---|---|---|
| ENQ / BUILD dominate | **host** — dispatch-bound | [§1](#1-host-side) |
| READ (blocking) dominates | **host** — round-trip-bound | [§1](#1-host-side), residency |
| device inside kernels, achieved GB/s ≪ ceiling | **DRAM / NoC** | [§2](#2-dram-and-noc) |
| device inside kernels, GB/s near ceiling | **kernel** | [§3](#3-kernel-side) |

Two facts that set the priorities for the whole repo:

* **The HE multiply is dispatch-bound, not kernel-bound.** 1219 dispatches × ~16 µs ≈ 19.5 ms = the
  entire device-busy span, of which only ~11.5 µs per dispatch was compute. **Making a kernel faster
  without removing a dispatch buys almost nothing.**
  ([`../he/blackhole_port.md`](../he/blackhole_port.md) §5)
* **A kernel costs ~3.3× more inside the pipeline than alone.** `modmul_fused3`: 3.086 µs/tile on one
  core, 10.3 µs at G=130. That inflation, not the arithmetic, is most of what a dispatch costs.
  ([`../he/exp_tg018m.md`](../he/exp_tg018m.md))

---

---

* **Re-measure after every rung.** The dominant term moves — the PIR profile went READ 5% → 43% in
  four steps, and the HE core-count optimum moved 32 → 64 after an unrelated fusion.
* **A new kernel file, not an edit,** when a shared kernel has many call sites. `reader_fused.cpp` had
  25 others and runtime args are positional, so `reader_fused_batch.cpp` / `reader_fused_off.cpp`
  landed as siblings. TT-Metal **silently ignores surplus runtime args** — a host/kernel mismatch
  yields plausible-but-wrong values or a hang, never an error.
* **Record negative results.** Every null above cost a device run; they are the cheapest thing in this
  repo to re-derive by accident.
* **Regression-gate every step on bit-exactness**, not on the headline. Each rung of both ladders
  above carries its mismatch count.
