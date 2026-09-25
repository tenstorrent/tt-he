# BFV kernels — correctness & device performance

Correctness (bit-exact vs the exact big-int golden) and standalone device profile of the building-block
kernels. For the composed full-multiply latency distribution see `ct_mul_perf.md`; for the build history
and design decisions (Q=3, the FBC pad, Montgomery-transparent NTT, the on-device reindex) see `PLAN.md`.

> **Hardware.** Numbers here were measured on **Wormhole n150**. Current hardware is Blackhole p150 — see the support matrix in the [root README](../../README.md) and [`blackhole_port.md`](blackhole_port.md) §3 for why they are not directly comparable.


Single Tensix core, Wormhole n150, N=1024, t=12289. Profiler clock 1 GHz → 1 cycle = 1 ns.

---

## 1. Correctness — bit-exact vs the golden

Ground truth: `reference/he/bfv_mul_ref.py` (exact big-int: encrypt → tensor → rescale → relin →
decrypt). `decrypt(ct1·ct2) == m1·m2 mod t`, 0/1024, all seeds, ~76-bit noise margin below Q.

| kernel / stage | program | params | result |
|---|---|---|---|
| HE add | `bfv_add` | Q=4 | ✅ 0/8192 |
| tensor product (NTT-domain) | `bfv_mul_raw` | Q=4 | ✅ 0/12288 |
| relinearization | `bfv_relin` | Q=4/P=4 | ✅ 0/8192 |
| relinearization | `bfv_relin_q3` | Q=3/P=3 | ✅ 0/6144 |
| rescale `⌊t·e/Q⌉` | `bfv_rescale` | Q=3/S=4/m_sk | ✅ 0/3072 |
| NTT reindex = tile transpose | `transpose_probe` | — | ✅ 0/1024 (device == host) |
| **full `ct_mul` on device** | `bfv_ct_mul` | Q=3 | ✅ tensor 0/24576, rescale/relin/final 0 |
| **end-to-end** (encrypt → add & mul → decrypt) | golden + demo | Q=3 | ✅ `a+b`, `a·b` all seeds |

All arithmetic runs on device (add, tensor, rescale, relinearize). Empirically pinned in the golden:
**floor suffices** (round vs floor ≤1, noise-absorbed); the rescale's `Q→S` FBC may be **approximate**
(drops ε<3), the `S→Q` FBC must be **exact** (`m_sk`) with `ẽ` **centered** before the mod-Q reduce.
The final parameters use **Q=3** (84-bit) so every base conversion is ≤4 limbs (see `PLAN.md` §3).

---

## 2. Base-kernel device performance (standalone)

Measured at Q=4/P=4 (the base kernels' native config; the composed `ct_mul` runs at Q=3 — see
`ct_mul_perf.md`). All bit-exact (seed 0).

```
bfv_add       24.0 us device / 105.7 us host-wall   (8 dispatches)   — HE addition
bfv_mul_raw  348.6 us device / 376.1 us host-wall   (20 dispatches)  — tensor product (NTT-domain)
bfv_relin   3490.5 us device                        (91 dispatches)  — key-switch / relinearize
```

`device` = device-active (Σ per-dispatch max-RISC `-KERNEL` cycles; RISCs concurrent, host gaps
excluded). `host-wall` = steady-state host wall-clock (warm-up + 100× enqueue, one `Finish`; includes
dispatch overhead). `bfv_relin` has no host-wall figure — it does a blocking read + NTT `reindex()`
between stages, so the device-only loop does not apply.

### Per-RISC breakdown (Σ over all dispatches, µs)

| kernel | BRISC (write) | NCRISC (read) | TRISC_0 (unpack) | TRISC_1 (math) | TRISC_2 (pack) |
|---|--:|--:|--:|--:|--:|
| `bfv_add`     |   24.0 |   9.4 |   13.4 |   20.3 |   2.4 |
| `bfv_mul_raw` |  348.6 |  18.4 |   29.2 |  341.3 |   6.6 |
| `bfv_relin`   | 3490.5 | 422.4 | 2411.1 | 3453.5 | 1434.3 |

In every kernel **TRISC_1 (SFPU math) tracks the BRISC envelope** — the integer-multiply / Montgomery
reduction is the bottleneck, not data movement. Same conclusion as the modmul and NTT-1024 reports.

### `bfv_relin` per-phase breakdown (Q=4, 91 dispatches)

| phase | kernel | dispatches | device-active | share |
|---|---|--:|--:|--:|
| NTT stage-A matmul | `fused_matmul_a` | 24 |  666.6 µs | 19.1 % |
| NTT twiddle modmul | `modmul_compute`  | 24 |  751.3 µs | 21.5 % |
| NTT stage-C matmul | `fused_matmul_c` | 24 |  687.7 µs | 19.7 % |
| ×evk element-wise modmul | `modmul_sfpi_full` | 16 |  338.8 µs |  9.7 % |
| FBC base conversion (1 mod-up + 2 mod-down) | `baseconv_fbc` | 3 | 1048.7 µs | 30.0 % |
| | | **91** | **3493.2 µs** | |

**Rolled up:** NTT/INTT **60.3 %** (fusedA + twiddle + fusedC) · FBC **30.0 %** · ×evk **9.7 %**.

A negacyclic transform is **3–4 dispatches** — `fused_matmul_a` (int8 four-step matmul) → twiddle
(`modmul_compute`, single-tile Montgomery modmul) → `fused_matmul_c` (int8 matmul) [→ `transpose_tile`
reindex, on device]. The single biggest phase is the **twiddle modmul (21.5%)**, edging out both
matmuls. For FBC the Montgomery **recombine** dominates each call (`fbc_phase2` ~288 µs of ~346 µs).

---

## 3. Optimization opportunities (ranked)

1. **NTT/INTT transforms (60% of relin — biggest lever).** (a) **Fuse the twiddle modmul into the
   adjacent matmul** (`fused_matmul_a/c`) — it is a standalone single-tile Montgomery modmul and the
   largest single phase; the NTT-1024 report already flags it as fusible. (b) Cheaper matmul-stage
   recombine.
2. **Cheaper FBC recombine.** The 3 base conversions cost ~1.05 ms (30%); `fbc_phase2` (a Montgomery
   reduction per destination modulus) is ~85% of each. Same target the bconv doc flags.
3. **Dispatch count.** Both `bfv_add` (78% launch latency across 8 single-tile programs) and the full
   `ct_mul` (501 dispatches) are dispatch-bound — batching limbs into wider tiles / fusing per-limb
   ops cuts the launch tax. `bfv_mul_raw` is already at its floor (int-multiply-bound, ~7% dispatch).
4. **×evk — leave it.** ~10% and compute-bound at the tile level (fusing its dispatches saves < 1 µs each).

---

## 4. Full `ct_mul` composed latency

The end-to-end multiply (Q=3) is **12.04 ms device-active, 501 dispatches** — 55% NTT transforms,
34% modmul, 8.7% FBC; by stage: tensor 47% · rescale 28% · relin 24% · add 0.1%. Full per-kernel and
per-stage breakdown + methodology: **`ct_mul_perf.md`**.

---

## 5. Reproduce

```bash
./build/bfv_add ; ./build/bfv_mul_raw            # print "TIMING:" host-wall lines
mkdir -p generated/traces
for p in bfv_add bfv_mul_raw bfv_relin; do       # device profiler — run SERIALLY (one device)
  TT_METAL_DEVICE_PROFILER=1 ./build/$p
  cp "$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv" generated/traces/$p.csv
done
python3 tools/traces/analyze2.py generated/traces/*.csv                     # per-kernel device-active
python3 tools/traces/analyze.py  generated/traces/bfv_relin_zoned.csv       # per-phase
(cd reference/he && python3 bfv_mul_ref.py)      # golden + end-to-end HE verify
```

**Caveats.** Single core, N=1024, batch=1, seed 0. Device-active excludes host gaps; host-wall includes
dispatch. `bfv_relin` device-active excludes the host `reindex()`/tilize between stages (now moved
on-device in `bfv_ct_mul` via `transpose_tile`). Sub-op device time is context-independent (FBC and the
fused-matmul stage measure identically standalone and inside relin, < 0.1% drift). Method matches the
repo's prior work (device profiler + 1 GHz→ns; host wall-clock steady-state loop).
