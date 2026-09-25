# BFV `ct_mul` — latency distribution (device-profiler measurement)

Per-kernel and per-stage latency of the full homomorphic multiply `ct_mul = tensor → rescale →
relinearize → add`, measured on device (single Tensix, Wormhole n150, N=1024, Q=3/P=3/S=4, t=12289).
This is the "where does the time go" report; see `README.md` for the pipeline and `PLAN.md` for the
build.

> **Hardware.** Numbers here were measured on **Wormhole n150**. Current hardware is Blackhole p150 — see the support matrix in the [root README](../../README.md) and [`blackhole_port.md`](blackhole_port.md) §3 for why they are not directly comparable.


## Headline

```
END-TO-END ct_mul:  12.04 ms device-active   (501 program dispatches)

  by stage:                          by functional bucket:
    tensor        5.71 ms  47.4%       NTT transforms   6.62 ms  55.0%
    rescale       3.40 ms  28.2%       modmul           4.13 ms  34.3%
    relinearize   2.91 ms  24.2%       FBC              1.05 ms   8.7%
    add           0.02 ms   0.1%       mod-add/sub      0.24 ms   2.0%
```

`ct_mul` is **NTT-bound** (55% in the four-step transforms) and **dispatch-bound** (501 mostly
single-tile programs). HE **add** alone is ~24 µs device (`bfv_add`).

---

## 1. Methodology — how this is measured

**Metric: device-active latency.** Each program dispatch is bracketed on device by a `-KERNEL`
profiler zone per RISC (BRISC / NCRISC / TRISC_0/1/2). The RISCs run concurrently within a dispatch,
so a dispatch's device latency = **max over RISCs of its `-KERNEL` duration**; the end-to-end figure
is the sum over dispatches. Profiler clock is 1 GHz, so **1 cycle = 1 ns**. This excludes host
dispatch/launch overhead (measured separately) — it is the pure on-device compute latency.

**Marking the zones (the key step).** To attribute each dispatch to *which* kernel, every compute
kernel used by `ct_mul` carries a distinct top-level `DeviceZoneScopedN(...)`:

| kernel | zone | kernel | zone |
|---|---|---|---|
| `fused_matmul_a` | `fused_total` | `modmul_sfpi_full` | `modmul_sfpi_full` |
| `fused_matmul_c` | `fusedC_total` | `baseconv_fbc` | `fbc_total` |
| `modmul_compute` | `modmul_compute` | `modadd_reduce` | `modadd_reduce` |
| `transpose_tile` | `transpose_tile` | `modsub_reduce` | `modsub_reduce` |

Since dispatches run serially on one command queue, the i-th `-KERNEL` pair on each RISC is dispatch
`i`; the kernel of dispatch `i` is the named zone nested inside TRISC_1's i-th `-KERNEL` window
(same RISC → tight nesting, no cross-RISC skew). Stages run sequentially (tensor → rescale → relin →
add) with distinct kernel signatures — rescale has no transforms, relin begins at the first FBC
(mod-up), add is the trailing `modadd` run — so dispatches bucket into stages by index range.

**Reproduce.**
```bash
rm -f "$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
TT_METAL_DEVICE_PROFILER=1 ./build/bfv_ct_mul reference/he/ctmul_case.bin   # one device → run serially
mkdir -p generated/traces
cp "$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv" generated/traces/bfv_ct_mul.csv
python3 tools/traces/analyze_ctmul.py generated/traces/bfv_ct_mul.csv
```

---

## 2. Per-kernel latency (calls · total · per-call)

| kernel | calls | total (µs) | µs/call | % of device |
|---|--:|--:|--:|--:|
| pointwise / rescale modmul (`modmul_sfpi_full`) | 134 | 4126.1 | 30.79 | 34.3 % |
| NTT twiddle modmul (`modmul_compute`) | 74 | 2316.3 | 31.30 | 19.2 % |
| NTT stage-C matmul (`fused_matmul_c`) | 74 | 2120.6 | 28.66 | 17.6 % |
| NTT stage-A matmul (`fused_matmul_a`) | 74 | 2052.1 | 27.73 | 17.0 % |
| base conversion (`baseconv_fbc`) | 3 | 1048.9 | 349.64 | 8.7 % |
| modular add (`modadd_reduce`) | 41 | 176.9 | 4.32 | 1.5 % |
| NTT reindex transpose (`transpose_tile`) | 74 | 127.6 | 1.73 | 1.1 % |
| modular sub (`modsub_reduce`) | 27 | 67.3 | 2.49 | 0.6 % |
| **TOTAL device-active** | **501** | **12035.8** | | **100 %** |

Notes:
- **A negacyclic transform = 4 dispatches** (`fused_matmul_a` → `modmul_compute` twiddle →
  `fused_matmul_c` → `transpose_tile` reindex). It runs **74×** = 56 in the tensor (8 limbs ×
  (4 fwd + 3 inv)) + 18 in relin (6 QP limbs × (1 fwd + 2 inv)). The three ~28–31 µs stages dominate;
  the on-device reindex transpose is nearly free (1.7 µs).
- **`modmul_sfpi_full` per-call (30.8 µs) is an average** over single-tile ops (tensor pointwise,
  ~21 µs) and multi-tile ops (rescale processes `L`=3–4 tiles/call, ~3–4× longer) — that mix is why
  it is the single largest kernel bucket.
- **FBC is expensive per call (350 µs)** but only 3 calls (relin mod-up + 2× mod-down); its cost is
  the Montgomery recombine (see `kernels.md`).

---

## 3. Stage decomposition (end-to-end)

| stage | dispatches | device (µs) | % | composition |
|---|--:|--:|--:|---|
| **tensor**      | 264 | 5707.6 | 47.4 % | 56 transforms (224 disp.) + 32 modmul + 8 modadd, over 8 limbs `Q∪S∪m_sk` |
| **rescale**     | 144 | 3398.7 | 28.2 % | ×3 polys: 90 modmul + 27 modadd + 27 modsub (2 FBCs built from primitives) |
| **relinearize** |  87 | 2911.6 | 24.2 % | 18 transforms (72 disp.) + 12 ×evk modmul + 3 FBC (mod-up/down) |
| **add**         |   6 |   18.0 |  0.1 % | 6 `modadd` (`(ẽ0+k0, ẽ1+k1)` over 3 limbs each) |
| **TOTAL**       | 501 | 12035.8 | 100 % | |

The **tensor is the largest stage (47%)** — it does the most NTTs (8 limbs, 4 in + 3 out per limb =
56 transforms, vs relin's 18). Rescale (28%) is all element-wise modmul/add/sub. Relin (24%) is
transforms + FBC. Add is negligible.

---

## 4. End-to-end: device vs wall-clock

| measure | value | what it is |
|---|--:|---|
| device-active (profiler) | **12.04 ms** | pure on-device kernel compute (this report) |
| device steady-state (`TIME=1`) | 12.81 ms | build-once, re-enqueue loop — adds ~0.8 ms dispatch overhead (~1.6 µs × 501) |
| cold wall-clock | ~6.6 s | one correctness run — dominated by JIT kernel compile + per-stage blocking readbacks, **not device work** |
| HE add (`bfv_add`) | ~24 µs | for comparison — HE addition is ~500× cheaper than multiply |

---

## 5. Where the time goes → optimization levers (ranked)

1. **NTT transforms (55%, biggest lever).** 296 transform dispatches. Fuse the **twiddle modmul**
   (19.2%, the largest single kernel bucket) into the adjacent matmul stage — it is a standalone
   single-tile Montgomery modmul the NTT report already flags as fusible. Cheaper matmul recombine is
   the second sub-lever.
2. **Dispatch count (501, mostly single-tile).** Device-active is ~12 ms but the pipeline pays
   per-dispatch launch overhead 501×; batching limbs into wider tiles / fusing per-limb ops cuts the
   dispatch tax and is the path to a lower wall-clock.
3. **`modmul` (34%).** Largely inherent (Montgomery reduction is the compute bottleneck everywhere),
   but the rescale's multi-tile modmuls could share setup.
4. **FBC (8.7%).** 3 calls, recombine-bound — the `kernels.md` target (cheaper recombine).

Caveats: single core, N=1024, seed 0, one `ct_mul`. Per-call figures are averages over a kernel's
calls (notably `modmul_sfpi_full` spans 1- and multi-tile ops). Device-active excludes host dispatch
overhead and the host readbacks/permutations of the current correctness harness.
