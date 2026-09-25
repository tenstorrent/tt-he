# OnionPIR on Tenstorrent n150 — correctness + performance record

> ⚠️ **STALE (n150) + one correction. See [latency_optimization.md](latency_optimization.md) first.**
> Every number below was measured on a Wormhole **n150**; the repo now runs on a Blackhole **p150**, where
> the scan is 48.61 GB/s @32 cores (not 27.85) and the answer pipeline is **164 ms/query** (not 4.3 s)
> after the device-residency work. Also: §2's "Current end-to-end latency" prescribes *kernel fusion* —
> that ordering was wrong; workload caching + trace replay is the lever (14.3× measured).

> **Hardware.** Numbers here were measured on **Wormhole n150**. Current hardware is Blackhole p150 — see the support matrix in the [root README](../../README.md) and [`../he/blackhole_port.md`](../he/blackhole_port.md) §3 for why they are not directly comparable.


**Scope.** Running OnionPIRv2's default workload (N=2048, ~60-bit-class modulus, t≈2¹⁴, 128 MB DB →
512×2⁷ hypercube) **device-resident** on a single Tenstorrent Wormhole n150. Plan:
`docs/pir/history/ONION_MATCH_PLAN.md`.

**Status (2026-07-24).** All cryptographic operations — through the full query→answer, recovering
DB[idx] — validated bit-exact on device at the library ring size (n=2048) and at a **conservative
84-bit RNS modulus** (3×28-bit, chosen over the library's ~60-bit for noise margin). The dominant
DB-scan is benchmarked across cores (~28 GB/s @ 32 cores). The **one remaining gap is end-to-end
latency**, which is program-dispatch-bound today and requires a kernel-fusion pass (§2, "Current
end-to-end latency") — it changes no correctness result and no throughput headline.

---

## 1. Correctness — full OnionPIR op set on device (n=2048)

Every operation OnionPIR needs, validated bit-exact vs the `onionpir_ref.py` golden, on the n150.
Tests are K-general (`K=2` ~56-bit and `K=3` ~84-bit share one code path; K read from the case header).

| Operation | test | K=2 (~56b) | K=3 (~84b) |
|---|---|---|---|
| signed gadget decompose (chunked ℓ=8) | `test_gadget_decompose*` | 0/131072 | 0/131072 |
| digit → RNS | `test_digit_to_rns*` | 0/131072 | 0/131072 |
| CRT-combine (Garner) | `test_crt_combine*` | 0/16384 | 0/16384 |
| forward/inverse NTT (four-step 32×64) | `test_ntt2048_rt` | 0/2048 | — |
| external product (folds + unpack core) | `test_extprod2048` | 0/8192 | 0/12288 |
| BV key-switch | `test_bv_keyswitch2048` | 0/8192 | 0/12288 |
| Subs (automorph + key-switch) | `test_bv_keyswitch2048` | 0/8192 | 0/12288 |
| query expansion (Subs tree) | `test_query_expand2048` | 0/32768 | 0/49152 |
| RLWE→RGSW unpack | `test_query_expand2048` | 0/49152 | 0/98304 |
| **answer** (init-dim scan + hypercube folds) → recover DB[idx] | `test_pir_answer2048` | 0/8192 | 0/12288 |

The answer pipeline recovers DB[idx] bit-exact vs `onionpir_ref.pir_answer` for dims=[4,2] and the
2-fold dims=[8,2,2] (32-cell hypercube). Since expand+unpack and the answer each match the golden
bit-exact, the full query→answer chain is proven transitively. All device ops are unified behind
`pir2048::Engine` (`lib/pir/pir_engine2048.hpp`); the op tests are thin consumers.

Everything runs on device (the host only orchestrates enqueues + the automorph/monomul/ct-add reindex,
which move on-device in milestone #5). The 84-bit path uses a 3-limb gadget, a 3-output digit split,
and a Garner CRT chain built by reusing the validated K=2 kernels (`crt_mul_split` ×3 + `crt_limb3`).

## 2. Performance — the first-dimension DB scan (the dominant, DB-linear op)

The scan `response[r] = Σ_{k<512} query[k] ⊛ DB[k][r]` is the throughput-critical PIR op: linear in DB
size, and (in the NTT/eval domain, per coefficient slot) a **modular matmul contracting the 512-wide
first dimension** — exactly the Tensix int8 matrix engine's strength (radix-128 int8 limbs →
int32 accumulate → mod-q recombine on the SFPU).

`host/experiments/test_scan_mc.cpp` benchmarks the matmul core of the scan (`compute/scan_mm.cpp`), sharded
across cores, streaming the DB from DRAM (unique tiles per output) against the shared query. The
recombine + mod-q reduction is the separately-validated SFPU tail (`matmul_fbc`/`recombine_rns`) and is
not timed — the scan is matmul/DRAM-bandwidth bound, which is what this measures. The int8
matmul-accumulate is validated bit-exact vs a CPU golden.

Workload OT=8192 output plane-tiles, KB=16 (512-wide contraction); n150 compute grid 8×8:

| cores | time (ms) | DB GB/s | int8 GMAC/s | speedup |
|------:|----------:|--------:|------------:|--------:|
| 1  | 72.44 | 1.85 | 59.3 | 1.00× |
| 2  | 36.69 | 3.66 | 117.1 | 1.97× |
| 4  | 19.19 | 6.99 | 223.8 | 3.78× |
| 8  | 10.34 | 12.98 | 415.3 | 7.00× |
| 16 | 5.94 | 22.58 | 722.7 | 12.19× |
| 32 | 4.82 | **27.85** | **891.1** | 15.03× |

Near-linear scaling to 8 cores, then DRAM-bandwidth bound (~28 GB/s ceiling) — the correct regime for
a database stream. Using ≤32 cores as specified; the grid has 64. The ceiling is **stable across scale**:
re-run at 256 MiB (OT=16384/KB=16 and OT=8192/KB=32) gives 29.5 and 26.9 GB/s at 32 cores (peak
**945 GMAC/s**), int8 matmul validated at every point. OT=8192·KB=16 = exactly 128 MiB of int8 DB tiles.

### Current end-to-end latency — dispatch-bound (motivates #5)

Measured answer-pipeline latency (`test_pir_answer2048`, dims=[8,2,2] = 32-cell hypercube, K=3):
**scan 1.5 s, folds 2.8 s, total 4.3 s**. This is *not* device compute — it is host-orchestration
overhead. **The bottleneck was diagnosed empirically to be PROGRAM DISPATCH, not readbacks:** a
buffer-resident MAC that removed the per-product blocking readbacks made the scan *slower*
(1.45→1.96 s) because it added programs. The [8,2,2] scan issues ~1440 tiny programs (144 forward
NTTs × ~10 stage-programs each) at ~1 ms dispatch each ≈ the measured 1.5 s. So the real
device-residency win is **kernel fusion — fewer, bigger programs** (fuse the 10-stage four-step NTT;
batch the many NTTs/MAC into one multi-tile program), not buffer-chaining. That is the sole remaining
gap to a meaningful end-to-end latency. The scan's own *device*
throughput is measured directly (§2, 28 GB/s — a single fused kernel over many tiles) and is unaffected
by this per-op dispatch overhead, which is exactly the evidence that fusion recovers the device speed.

### Scan throughput — raw and plaintext-normalized

**(a) Raw scan-engine throughput** (int8 DB-limb bytes streamed + contracted / s): **~27.85 GB/s**
(peak 29.5) on the n150 at 32 cores.

**(b) Plaintext-normalized.** On TT the scan streams the DB in eval-domain RNS radix-128 int8 limbs:
each plaintext coefficient (~13 usable bits ≈ 1.6 B) becomes K·4 int8 bytes = 8 B (K=2) … 12 B (K=3) —
a **~5–7.4× representation blow-up**. Dividing the raw rate by that blow-up gives a plaintext-scan rate
of **~3.8–5.4 GB/s**. (The blow-up is intrinsic to lowering a 28-bit modular MAC onto an 8-bit engine.)

**Caveats.** (1) The figure times the matmul/bandwidth-bound scan kernel; the mod-q recombine is the
separately-validated SFPU tail (not timed, but small — one recombine per output tile). (2) The 84-bit
modulus is reflected in the K=3 end of the blow-up range. (3) This is the *scan* only; the full 128 MB
end-to-end pipeline (#7) — expand + scan + folds chained device-resident — will give the true query
latency + QPS/card headline.

## 3. Remaining work

The cryptographic pipeline is complete and validated (§1) and the scan headline is measured (§2). What
remains is the integration for a single end-to-end *latency* number:

- **#5 device-residency** — the shared engine (`pir2048::Engine`) is done; the remaining piece is
  buffer-chaining the stages + moving automorph/monomul/ct-add/mod-switch on-device to remove the
  per-op host round-trips (today's per-op wall-clock is dispatch-bound, not device-compute-bound).
- **#7 scale-up** — run the scan at the 128 MB hypercube shape (already measured at scale in §2) and a
  single chained expand→unpack→answer test (functionally proven transitively today).
- **#8 end-to-end perf** — per-stage latency + QPS/card, once #5 removes the round-trips.

## 4. Reproduce

```
# correctness (K=3 shown; drop _k3 / use K=2 goldens for ~56-bit)
cd reference/pir && python3 gen_extprod_case.py --K 3 --n 2048 --ell 8 --B 2048 --out extprod_k3_n2048.bin
cd device/build && ./test_extprod2048 ../../cpu/pir/extprod_k3_n2048.bin
./test_query_expand2048 ../../cpu/pir/expand_k3_n2048.bin
# scan throughput + core scaling
./test_scan_mc 8192 16
```
