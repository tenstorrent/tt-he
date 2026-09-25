# matmul/history — superseded records

Kept, not deleted: each described the code accurately when written, and the reasoning behind a
superseded approach is often the most useful part of the record. Every file carries a
`Status: Historical` header naming what replaced it.

**Do not cite numbers from here as current.** They are snapshots, all measured on Wormhole n150.
Current: [`../ntt_1024.md`](../ntt_1024.md) / [`../ntt_1024_perf.md`](../ntt_1024_perf.md) and
[`../ntt_32768.md`](../ntt_32768.md) / [`../ntt_32768_perf.md`](../ntt_32768_perf.md).

| file | what it recorded |
|---|---|
| `NTT_1024_PERF.md` | the original host-orchestrated version (1167 µs) and its host-cost attribution |
| `NTT_1024_ONDEVICE.md` | design of the 7-program on-device version, pre-fusion |
| `NTT_1024_ONDEVICE_PERF.md` | per-stage numbers at 147 µs; source of the "reduction dominates" finding |
| `NTT_1024_MULTICORE.md` | full design of the 4-core plane-matmul variant (summarized in `../ntt_1024.md` §5.1) |
| `NTT_32768_OPTIMIZED.md` | the optimization pass that reversed the 2-factor vs 3-factor conclusion |
| `NTT_32768_PERF.md` | core-count sweep + the all-to-all corner-turn fix (972 → 94.8 µs) |
| `NTT_32768_3FACTOR_PLAN.md` | plan for the 32×32×32 factorization |
| `NTT_32768_2FACTOR_PLAN.md` | plan for the 128×256 baseline |
| `NTT_32768_COMPARISON.md` | pre-optimization head-to-head; its conclusion was later reversed |
