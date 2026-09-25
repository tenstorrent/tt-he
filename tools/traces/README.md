# Device-trace analyzers

Post-processors for tt-metalium's device profiler CSV. They read
`$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv` — written when a program runs under
`TT_METAL_DEVICE_PROFILER=1` — and turn per-RISC `DeviceZoneScopedN` zones into a latency breakdown
or a timeline. Pure standard-library Python 3 (`trace_app.py` is the one exception, see below);
nothing here touches the device.

**These live in `tools/` on purpose.** They used to sit under `generated/profiler/traces/he/`, which
is gitignored — so they were never committed, and `bench/hemul_cross.sh` step [3/3] silently skipped
its breakdown on every machine but the one they were written on.

| script | output | use it for |
|---|---|---|
| `trace_hemul.py <csv> [out_prefix]` | stdout table + `.json` (Perfetto) + `.html` | **the CROSS Set-B HE-mul breakdown** — per-op-type device time, NTT / pointwise / FBC buckets. Called by `tools/bench/hemul_cross.sh`. |
| `busy_metrics.py <csv> --expect N [--wall-ms X] [--host-busy-ms X] [--clock-mhz F]` | stdout | **the four latency metrics** defined in [`docs/he/exp_tb001.md`](../../docs/he/exp_tb001.md#latency-terminology) — wall time and host device busy (host-measured), busy span and critical path busy (trace-derived) — and nothing else. Verifies a 1:1 map against the host's own dispatch count and refuses to report on a capture with dropped markers. The CSV is in device **cycles**, so `--clock-mhz` matters (default 1350). |
| `analyze_ctmul.py <csv>` | stdout | per-kernel latency of a full BFV `ct_mul` (calls / total / per-call) |
| `analyze2.py <csv>...` | stdout | per-kernel device-active time, one line per kernel |
| `zone_phases.py <csv> [--clock-mhz F]` | stdout | **inner PHASE zones**, summed as max-over-cores per occurrence — the only way to attribute decompose / matmul / recombine once they are FUSED into one kernel (build with `-DNTT_PHASE_ZONES` / `-DTB001_PHASE_ZONES`). Phase rows need not sum to the kernel total and include CB waits — read its docstring first |
| `analyze.py <csv>` | stdout | per-phase breakdown within one kernel |
| `trace_viz.py <csv> <out.html>` | self-contained `.html` | interactive canvas Gantt, no server and no libraries — just open the file |
| `trace_riscs.py <csv> <out.html>` | self-contained `.html` | per-RISC lanes colored by role (reader / unpack / math / pack / writer) |
| `trace_app2.py <csv> [PORT] [lo_ms hi_ms] [--expect N]` | live server on `0.0.0.0:<port>` | the richest view: searchable canvas Gantt, hover for a RISC's role, click a bar for op detail. Idle between dispatches is compressed (gaps >30 µs → an 8 µs stub) and the header reports how much was removed. **`--expect N` is required for a `HE_PROF_DRAIN` capture** — the default `pick_window()` splits on >5 ms gaps, and a drained capture stalls ~23 ms every `HE_DRAIN_K` dispatches, so it would shatter one multiply into ~40 clusters and show a fortieth of it. |
| `trace_app.py <csv> [port]` | live server | older Dash/Plotly viewer — **needs `pip install dash plotly`**, unlike everything else here |

## Typical flow

```bash
# 1. run something with the profiler on
TT_METAL_DEVICE_PROFILER=1 HE_TRACE2=1 HE_PROF2=1 NTT_CORES=16 ./build/he_mul_cross 0

# 2. break it down
CSV="$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
python3 tools/traces/trace_hemul.py "$CSV" generated/traces/hemul
```

`sh tools/bench/hemul_cross.sh` does both and prints the summary.

**For a *complete* trace, use `HE_PROF_DRAIN=1` instead of `HE_PROF2=1`.** The device-side profiler
buffer overflows partway through a full ct_mul and silently drops markers; `HE_PROF_DRAIN` drains it
every `HE_DRAIN_K` dispatches (default 100) so the capture keeps every zone:

```bash
TT_METAL_DEVICE_PROFILER=1 HE_PROF_DRAIN=1 NTT_CORES=16 ./build/he_mul_cross 0
python3 tools/traces/trace_app2.py 8747     # then forward the port and open it
```
