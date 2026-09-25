# PIR scan (first dimension) — status record

**As of 2026-08-04.** Companion to `SCAN_ALGORITHM.md` (algorithm + hardware mapping) and
`ONION_MATCH_PERF.md` (the wider perf record). Records the **live, validated** scan path and what
remains. (Superseded experiments — the slot-loop kernel, single-kernel K-accumulation, 1-tile-per-core
sharding — have been removed from the tree; this file describes only what is kept.)

## TL;DR

The PIR first-dimension scan is a **batched modular matmul** (contract the N0 first-dim axis), run
**dispatch-free** on the n150: phase-major M-batch, K-tiled, multi-core, with the coeff↔eval NTTs and
the poly↔slot reshuffle also on device. The full **coeff-in / coeff-out** scan is validated end-to-end
(NTT → device reshuffle → matmul → device reshuffle → NTT), host-free.

## The operation

Per RNS channel, per NTT slot `j`:
```
A[r,c,j] = Σ_k DB_ntt[k,r,j] · Q_ntt[k,c,j]   mod q     "krj,kcj->rcj"
```
= per slot a modular matmul `DB_jᵀ · Q_j = [rest,N0]·[N0,c] → [rest,c]`, contract `k=N0`.
DB is pre-known → int8-folded **offline**; Q is decomposed **on-device**; recombine mod q. The DB NTT
is offline (one-time); the online scan is O(DB). Full derivation + real dims (`N0=512, rest=128, c=2,
N=2048, K=3`) in `SCAN_ALGORITHM.md`.

## Validated on device (n150)

| what | test | result |
|---|---|---|
| algorithm: scan == iNTT(batched matmul) == int8 lowering | `reference/ir/validate_scan.py` | bit-exact |
| end-to-end: NTT+matmul+iNTT decrypts to DB[c0·rest+r] | `reference/ir/validate_scan_full.py` | ALL PASS |
| query-batch (B queries in tile columns), K-tiled | `kernels_ir/test_scan_qbatch` | B=16 0/2048; B=64 0/8192 |
| **M-batch phase-major** (dispatch-free matmul tile) | `kernels_ir/test_scan_mbatch` | 0 bad; **1.115 M tiles/s @ 64c** |
| **full first-dim K-tiled scan** (contract N0=512) | `kernels_ir/test_scan_kt` (mbatch + add_modq) | 0 bad; **60 502 out-tiles/s @ 64c** |
| **batched forward NTT** (online NTT, phase-major) | `pir/test_ntt_batch_fwd` | 0 bad; **470 009 NTTs/s @ 64c** |
| **batched inverse NTT** (round-trip) | `pir/test_ntt_batch_fwd` | 0 bad — `inv(fwd(x))==x` |
| **reshuffle poly→slot** (query/DB in, all-to-all gather) | `kernels_ir/test_reshuffle` | 0 bad; scales past L1 (P=1024 @ 64c) |
| **reshuffle slot→poly** (output, all-to-all scatter) | `kernels_ir/test_reshuffle_rev` | 0 bad, multi-core |
| **full on-device scan, host-free reshuffle** (n=1024) | `pir/test_scan_ntt` (fwd → gather → mbatch → scatter → inv) | 0 bad (KB=1 0/32768, KB=2 0/65536) |
| **REAL CONFIG scan** (n=2048, K=3 RNS, N0=512) | `pir/test_scan_ntt2048` (real `fwd2048` → n=2048 gather → mbatch → scatter, eval golden) | **0/24576, Test Passed** |
| **REAL CONFIG external product** (fast path) | `pir/test_extprod_fast2048` (gadget → digNtt → **mbatch MAC contract 2ℓ, RGSW folded** → reshuffle) | **0/24576** — the atomic op of unpack (stage 1) + folds (stage 3) |

### M-batch phase-major throughput (`test_scan_mbatch`, OT=4096)
```
   M    cores   tiles/s
   16    64      713 909
   32    64    1 115 442   ← sweet spot (M=48/64 exceed 1.5 MB L1)
   32     1       35 443
```
`matmul_modq_mbatch` runs M independent output tiles in ONE program **phase-major** — all M decomposes
(1 unary init), then all M matmuls (1 mm init), then all M recombines (1 unary init): **3 engine-mode
switches total, not 3·M**. Each tile is an independent contract-32 modular matmul (own 4 planes,
recombined separately) → no K-accumulation, no plane overflow. M-intermediates live in L1 (planes cap
M at 32). This removed the per-tile dispatch that capped the earlier 1-tile-per-core path (20× gain).

## The building blocks (kept, reusable for the real end-to-end)

| kernel (`device/kernels_ir/`) | role |
|---|---|
| `compute/matmul_modq.cpp` | reference single-tile modular matmul (radix-128 int8 lowering) |
| `compute/matmul_modq_mbatch.cpp` + `dataflow/reader_mbatch.cpp` | **scan matmul** (phase-major) / **NTT stage A** |
| `compute/matmul_modq_mbatch_c.cpp` + `dataflow/reader_mbatch_c.cpp` | **NTT stage C** (shared/resident twiddle) |
| `compute/add_modq.cpp` + `dataflow/reader_binop.cpp` | K-tiling running accumulate `acc=(acc+partial) mod q` |
| `dataflow/reshuffle_gather_mc.cpp` | **poly→slot** reshuffle (all-to-all), `poly = row·SR + col·SC` covers DB & query |
| `dataflow/reshuffle_scatter_mc.cpp` | **slot→poly** reshuffle (output side) |
| `compute/transpose.cpp` | 32×32 tile transpose (NTT reindex) |

Throughput levers, all data-level on these kernels: **query-batch** (B queries → 2B tile columns),
**K-tiling** (contract N0=512 = 16 tiles via mbatch passes + `add_modq`), **multi-core** (shard output
tiles / polys across the grid).

## Remaining work

1. **Scale the fast path to n=2048 / K=3 — DONE, validated.** `pir/test_scan_ntt2048`: the real-config
   scan (real ring n=2048, 3×28-bit RNS K=3, first-dim **N0=512**) runs on device — real `fwd2048`
   (engine) → **n=2048 poly→slot gather** → K-tiled **mbatch matmul** → **slot→poly scatter** — and
   validates **0/24576** vs a host eval-domain golden. (`rest=2` here — the free/m-tile axis; rest=128
   is the same op ×4 m-tiles, validated in `test_scan_kt`. Compared in the eval domain so the golden is
   cheap; the coeff `inv2048` tail is validated by round-trip.) Supporting measurements:
   - **Matmul** at real-config scale — `test_scan_kt` OT=8192 (2.1 GB DB, KB=16) = **52 601 out-tiles/s
     @ 64c, 0 bad**; real OT=24 576 → **~30 ms/query** (measured-rate; direct OT=24 576 OOMs *host*
     data-gen, not device).
   - **Reshuffle** at n=2048 (2 tiles/poly, `in_stride`/`in_off`, one pass per half): `test_reshuffle
     32 32 2` **0 bad @ S=2048**.
   - Two wiring bugs found+fixed here: gather conflated DRAM-tile-index with slot-index (split
     `slot_start`/`out_tile_base`); output buffer sized `rest*2` not `rest*2*HALVES` (2 tiles/poly).
2. **Folds + expansion onto the same fast path** — a fold is gadget → NTTs → MAC(contract 2ℓ) → NTTs;
   the MAC is an mbatch matmul, the NTTs batch on the batched-NTT kernels, the reshuffle is the same —
   so it is *composition of the kept kernels*, **no new kernels**. Chain expand→scan→folds device-resident.
3. **Perf refinements** — replace the reshuffle's C·P DRAM reads with the NoC-distributed a2a
   (`bench_noc_a2a`, 3.8× @ G=64, [[tt_noc_vs_dram_reshuffle]]); wire the batched NTT into `test_scan_ntt`
   (it still round-trips fwd/inv to host there).
4. **End-to-end headline** — one wall-clock run at n=2048/K=3, then GB/s + QPS in
   `ONION_MATCH_PERF.md`.

## Files

- Kernels + tests: see the building-blocks table above; tests `test_scan_mbatch`, `test_scan_kt`,
  `test_scan_qbatch`, `test_reshuffle`, `test_reshuffle_rev` (`device/kernels_ir/`), `test_scan_ntt`,
  `test_ntt_batch_fwd` (`device/pir/`, reuse the `test_initdim` n=1024 NTT).
- Goldens/algorithm (`reference/ir/`): `validate_scan.py`, `validate_scan_full.py`, `reference.py`.
- Docs: `SCAN_ALGORITHM.md` (algorithm + hw mapping), this file, `ONION_MATCH_PERF.md`.
