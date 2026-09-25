# OnionPIR — device-active latency (device-profiler measurement)

Where the time goes in the PIR server pipeline, measured on device (single Tensix, Wormhole n150,
n=1024, composite `q = q₁·q₂` two 28-bit primes, gadget `B=2²⁰`/`ℓ=3`). Same methodology as
`docs/he/ct_mul_perf.md`. See `pir_demo.md` for the pipeline and `pir_kernel_plan.md` for the build.

> **Hardware.** Numbers here were measured on **Wormhole n150**. Current hardware is Blackhole p150 — see the support matrix in the [root README](../../README.md) and [`../he/blackhole_port.md`](../he/blackhole_port.md) §3 for why they are not directly comparable.


## Headline

```
END-TO-END pir_demo (retrieve 1 of 4 entries):  7.14 ms device-active   (264 dispatches)

  by functional bucket:                     refined (twiddle-modmul folded into transforms):
    NTT transforms   3.61 ms  50.6%           NTT transforms      5.61 ms  78.7%
    modmul           3.38 ms  47.4%           pointwise MAC       1.38 ms  19.3%
    CRT / gadget     0.09 ms   1.2%           CRT+gadget (new)    0.09 ms   1.2%
    mod-add          0.06 ms   0.8%           mod-add             0.06 ms   0.8%
```

PIR is **NTT-bound** — like BFV `ct_mul`. Every server op (initial-dimension MAC, external product,
mod-switch prep) surrounds a handful of pointwise ops with negacyclic NTT/INTT, and the four-step
transforms dominate. The **genuinely new PIR arithmetic — signed gadget decompose (#9) and
CRT-combine (#13) — is ~1.2%** of device time; the cost is all in the transforms and the pointwise
modmul, primitives already on silicon. `modmul` is 47% as its own bucket, but ~⅔ of those calls are
NTT *twiddle* multiplies — folding them into the transforms gives the "refined" column: **~79%
transforms, ~19% pointwise MAC, ~1% new arithmetic.**

---

## 1. Methodology

**Metric: device-active latency.** Each program dispatch is bracketed on device by a `-KERNEL`
profiler zone per RISC; RISCs run concurrently, so a dispatch's device latency = **max over RISCs of
its `-KERNEL` duration**, and the end-to-end figure is the sum over dispatches (dispatches run
serially on one command queue). Profiler clock 1 GHz → **1 cycle = 1 ns**. Excludes host
dispatch/launch and the correctness harness's host round-trips (see caveats).

**Zones (the key step).** Every compute kernel carries a distinct top-level `DeviceZoneScopedN`. The
reused BFV/NTT kernels already had them (`fused_total`, `fusedC_total`, `modmul_compute`,
`modadd_reduce`); this work added zones to the four new PIR kernels — `gadget_decompose`,
`crt_mul_split`, `garner_diff`, `digit_to_rns` — so each dispatch attributes to the named zone nested
in TRISC_1's i-th `-KERNEL` window.

**Reproduce.**
```bash
cd device
CSV="$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
(cd ../cpu/pir && python3 gen_pir_demo_case.py --idx 3)
rm -f "$CSV"; TT_METAL_DEVICE_PROFILER=1 ./build/pir_demo          # single n150 → run serially
python3 ../cpu/pir/analyze_pir.py "$CSV"
```

---

## 2. Per-kernel latency — end-to-end `pir_demo`

| kernel | calls | total (µs) | µs/call | % of device |
|---|--:|--:|--:|--:|
| modmul (NTT twiddle + pointwise MAC) `modmul_compute` | 108 | 3378.9 | 31.29 | 47.4 % |
| NTT stage-C matmul `fused_matmul_c` | 64 | 1833.8 | 28.65 | 25.7 % |
| NTT stage-A matmul `fused_matmul_a` | 64 | 1776.4 | 27.76 | 24.9 % |
| modular add `modadd_reduce` | 12 | 57.8 | 4.81 | 0.8 % |
| CRT combine+split (#13) `crt_mul_split` | 4 | 47.3 | 11.82 | 0.7 % |
| digit→RNS split `digit_to_rns` | 6 | 16.1 | 2.68 | 0.2 % |
| gadget decompose (#9) `gadget_decompose_signed` | 2 | 13.5 | 6.73 | 0.2 % |
| CRT garner-diff (#13) `garner_diff` | 4 | 11.5 | 2.88 | 0.2 % |
| **TOTAL device-active** | **264** | **7135.3** | | **100 %** |

- **A negacyclic transform = 3 dispatches** (`fused_matmul_a` → twiddle `modmul_compute` →
  `fused_matmul_c`). There are 64 transforms here; each `modmul_compute` twiddle (~31 µs) is counted
  in the modmul row, so the true transform cost is `fused_a + fused_c + 64 twiddles ≈ 5.61 ms (79%)`.
- **The four new PIR kernels together = 88.4 µs (1.2%)** over 16 calls — the signed gadget decompose
  and the CRT-combine bridge are cheap; they just feed the transforms.
- Same shape as BFV `ct_mul` (`docs/he/ct_mul_perf.md`): NTT-bound, dispatch-bound, single-tile.

## 3. Per-operation latency (measured standalone)

Each server op profiled on its own (includes the one-time nNTT of its key material — see note):

| operation | program | dispatches | device (µs) | NTT % | modmul % | new-arith % |
|---|---|--:|--:|--:|--:|--:|
| external product (#16) | `test_extprod` | 162 | 4415.6 | 51.1 | 46.8 | 1.3 |
| initial dimension (#12) | `test_initdim` | 160 | 4560.4 | 49.5 | 49.4 | — |
| mod-switch (#11) | `test_modswitch` | 6 | 92.1 | — | 67.8 | 32.2 |

- **Mod-switch is cheap (92 µs, 6 dispatches):** it runs once on the response — a CRT-combine
  (`garner_diff` + `modmul` + `crt_mul_split`) per component; the scale-round finalize is host.
- **External product & initial dimension each include one-time key-material nNTT.** For the external
  product, NTT'ing the 24 RGSW rows is **precompute (offline in real PIR)** ≈ 72 dispatches / 2.1 ms;
  the **online** external product (keys pre-NTT'd) is ≈ 90 dispatches / **2.3 ms** — 16 digit
  transforms + the 2ℓ×2 pointwise MAC + the gadget/CRT front-half (~59 µs). Likewise the initial
  dimension's DB nNTT is offline precompute.

## 4. Where the time goes → optimization levers (ranked)

1. **NTT transforms (~79%, the lever).** Fuse the **twiddle modmul** (the largest single kernel
   bucket) into the adjacent matmul stage — the NTT report already flags it as fusible; cheaper
   matmul recombine is the second sub-lever. This is the same #1 lever as BFV `ct_mul`.
2. **Keep all key material NTT'd offline.** DB and RGSW/Galois keys are static → their nNTT is
   one-time precompute, not per-query (≈ half of the standalone external-product / initial-dimension
   time). The demo re-NTTs them each run for self-containment; a server caches them.
3. **Composite-NTT (#20).** Keep values in the single-mod view across the fold chain to avoid an
   INTT+NTT round-trip per external product — cuts transforms on the fold path directly.
4. **int8-matmul initial dimension.** The correctness path uses the pointwise MAC; the throughput
   version runs the initial dimension on the int8 matmul engine over the whole DB (the headline
   `bytes/s` number). Same operation, far fewer dispatches for a large N₀.
5. **Dispatch count / host round-trips.** The demo reads every intermediate back to host (the
   correctness harness) → high wall-clock. A resident device pipeline (chain through DRAM, as the
   standalone NTT does) removes that; device-active is already the number above.

## 5. Caveats

Single core, n=1024, seed-fixed, a 4-entry demo DB (`N₀=2 × 2`). Device-active excludes host
dispatch overhead and the correctness harness's per-op host readbacks/reshapes (so wall-clock ≫
device-active). Per-call figures are averages (notably `modmul_compute` spans NTT-twiddle and
pointwise-MAC calls — same kernel, so they share a zone). The standalone per-op numbers each include
that op's key-material nNTT, which is offline precompute in a real deployment (§3 note). This is a
"where does the time go" report, not a throughput benchmark — the headline PIR `bytes/s` number comes
from the int8-matmul initial dimension over a real database (lever #4), future work.
