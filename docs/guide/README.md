# Guide — read these first

Normative guides for working in this repo. A number produced outside `profiling.md` is not quotable.

| | |
|---|---|
| [`profiling.md`](profiling.md) | **How to measure.** The four pipeline latency metrics, the host-phase split, and — §4b — the one metric and four rules for comparing two kernels. Includes the fit code and the cross-checks. |
| [`optimization.md`](optimization.md) | **What to do** once a term dominates — the order of attack, then one doc per domain. |
| ├ [`optimization_host.md`](optimization_host.md) | dispatch count, occupancy, batching, trace |
| ├ [`optimization_dram.md`](optimization_dram.md) | read depth, bank spreading, both NoCs, the ceilings |
| └ [`optimization_kernel.md`](optimization_kernel.md) | SFPI passes, `sfpmul24`, DST traffic, registers |

Every lever carries its measured effect and a reference. Null results are included so they are not
re-litigated.

Companion: [`../../kernels/OPTIMISED.md`](../../kernels/OPTIMISED.md) — the optimised kernel set,
which kernel to use instead of which, and the three SFPU techniques written up for reuse.

**The two rules that cost the most to learn:**

1. **Fit, never divide.** `time/T` folds a fixed per-dispatch cost into every tile and penalises the
   faster kernel. Four separate wrong answers in this repo trace to it.
2. **Kernel ratios bound pipeline effect; they never predict it.** A 2.31× kernel returned 11.9% end
   to end; a 1.74× returned 4.3%. Always re-measure after porting.
