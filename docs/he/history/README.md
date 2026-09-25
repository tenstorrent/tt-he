# he/history — superseded records

Kept, not deleted: each was accurate when written. Every file carries a `Status: Historical` header
naming its replacement.

**Do not cite these numbers as current.** They are snapshots. The first two were measured on
Wormhole **n150** — and `OPTIMIZATION_REPORT.md` at a code state producing 102 trace segments where
today's produces 46. The four `exp_Tg018*` records below were measured on Blackhole **p150**: they
are here because the lineage moved on, not because the hardware changed.
Current: [`../README.md`](../README.md), [`../ct_mul_perf.md`](../ct_mul_perf.md),
[`../blackhole_port.md`](../blackhole_port.md), and for that lineage
[`../exp_tg018m.md`](../exp_tg018m.md) → [`../exp_tg018ntt.md`](../exp_tg018ntt.md).

| file | hw | what it recorded |
|---|---|---|
| `PLAN.md` | n150 | the build record, in order, with the design decisions behind it |
| `OPTIMIZATION_REPORT.md` | n150 | the Set-B optimization ladder (5044 → 727 → 38 ms) and its per-phase attribution |
| `exp_tg018dram.md` | p150 | the DRAM sweep. **Negative results, which do not expire** — five of six levers did nothing or hurt |
| `exp_tg018r.md` | p150 | SFPU vs matrix engine for base conversion, swept over `(Ls, Ld)`. Its winner is live in `../exp_tg018ntt.md` §4e |
| `exp_tg018s.md` | p150 | `sfpmul24` vs the fp32 byte-split. **Headline overstated** — measured against the running kernel, not the best one |
| `exp_tg018k.md` | p150 | the surviving compute kernels. Part A's census is sound; **parts B and C divide by `T` and are wrong** |
