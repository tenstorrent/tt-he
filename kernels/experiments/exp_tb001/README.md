# `kernels/experiments/exp_tb001/` — private kernel tree for exp_Tb001

The 29 Tensix kernels `host/experiments/exp_tb001.cpp`, `host/experiments/exp_tb001b.cpp` and `host/experiments/exp_tg000.cpp` dispatch,
as its **own** copies. Nothing else in the repo compiles or JITs from this directory, and these
experiments JIT from nowhere else.

```
he_device_target(exp_tb001 SOURCES host/experiments/exp_tb001.cpp KERNELS kernels/experiments/exp_tb001 TEST hw)
```

`KERNELS` sets `OVERRIDE_KERNEL_PREFIX` to this directory, so the host's
`OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a.cpp"` resolves here and not in `kernels/shared/`. The
mechanism and the trailing-slash requirement are documented in `cmake/HeDeviceTarget.cmake`;
`kernels/ir/` is the other tree that uses it.

## Why a private copy

So kernel-level experiments for exp_Tb001 cannot regress the validated tree. `kernels/shared/` is shared by
**76** device binaries, including `he_mul_cross`, the whole NTT suite, `lib/he` and `lib/pir` — a
change there to try something for this experiment is a change to all of them. The separation is
applied up front here, rather than after a break.

It also removes a real trap: two of these kernels, `fused_matmul_a.cpp` and `fused_matmul_c.cpp`,
carry the mid-kernel re-init deadlock fix (`docs/he/blackhole_port.md` §2). Editing the shared copies
while chasing a performance idea is how that fix gets lost for every NTT binary at once.

## Invariant: these start byte-identical to `kernels/shared/`

They were copied verbatim, so exp_Tb001's first run reproduces `he_mul_cross`'s arithmetic exactly.
Check what has diverged since at any time:

```sh
for f in $(cd kernels/experiments/exp_tb001 && find . -name '*.cpp'); do
    cmp -s "kernels/experiments/exp_tb001/$f" "kernels/shared/$f" || echo "diverged: $f"
done
```

When you *do* diverge one, say so at the top of that file and in `docs/he/exp_tb001.md` — otherwise
the next person cannot tell an intentional experiment from a stale copy.

## Divergences from `kernels/shared/` — the record

| file | why |
|---|---|
| `compute/bconv_one.cpp` | **new here, no shared counterpart.** Single-source base conversion (`out_j = x mod q_j`) for the rescale's Ls=1 case, which the general FBC was computing through the full CRNS machinery at 386 µs instead of 7.8 µs. See `docs/he/exp_tb001.md` "Optimizations landed at this workload". |
| `compute/fused_matmul_a.cpp` | inner `ph_decompose` / `ph_matmul` / `ph_recombine` zones compiled out behind `TB001_PHASE_ZONES` — their marker volume overflows the profiler's per-RISC DRAM buffer at `G=8`, which breaks the 1-1 dispatch map the metrics depend on. |
| `compute/modadd_twiddle.cpp` | same, for `ph_reduce` / `ph_twiddle`. |
| `compute/baseconv_fbc.cpp` | same, for `fbc_phase1` / `fbc_phase2`. **Also diverged arithmetically:** takes `Ls` and `Ld` as separate args and loops each over its actual bound, instead of one padded `n = FBCW = 8`. 386 → 167 µs/dispatch. The shared copy still pads. |
| `compute/fused_matmul_a_acc.cpp` | **new here, no shared counterpart.** The 8-block NTT contraction with the int32 planes accumulated across all partials and the recombine run ONCE (forward stage 1). 8 dispatches per output block → 1. |
| `compute/fused_matmul_c_acc.cpp` | **new here, no shared counterpart.** Same, for the inverse chain's stage 2, where the limbs are the matmul's LHS. |
| `dataflow/reader_fused_acc.cpp` | **new here, no shared counterpart.** Feeds the two above: `NACC` input tiles + `NACC×16` folded-W tiles, all reads in flight, one barrier. |
| `compute/baseconv_sk.cpp` | **new here, no shared counterpart.** EXACT base conversion via Shenoy–Kumaresan: ε is *measured* from a redundant residue mod `m_sk` rather than dropped, and the centered representative falls out of the same subtraction. Used by `exp_tb001b`'s input lift, where an approximate conversion would corrupt the plaintext. Phases 1–2 are `baseconv_fbc`'s verbatim. |
| `dataflow/a2a_corner_bufs.cpp` | **new here, no shared counterpart.** All-to-all corner turn over PER-TILE buffers, for exp_Tg018's 3-level NTT-65536. `kernels/shared/dataflow/all_to_all_corner_mc.cpp` does the same arithmetic but addresses the NT input tiles as pages of ONE buffer; every other stage in this tree takes each tile as its own single-tile buffer, and a TensorAccessor cannot be pointed at "page t of a larger buffer" by address arithmetic (DRAM interleaving puts page i on bank i mod nbanks). Teaching the corner turn per-tile addresses is a smaller change than teaching every matmul reader a page offset. The row-chunk copy is verbatim from the shared version. |
| `compute/baseconv_fbc.cpp` | (second divergence) constant arrays sized for the **widest** caller, not exp_Tb001's: `exp_tb001b` converts `Q → S\|m_sk` (`Ld=10`). Overflowing them silently corrupts the compute RISC's stack. |

Everything else is still byte-identical to `kernels/shared/`; the `cmp` loop above is what tells you.

`exp_tb001` (the CROSS-aligned cost model), `exp_tb001b` (the decrypt-verified BFV) and `exp_tg000`
(the decrypt+decode-verified CKKS) all build from this tree, deliberately: a kernel change is then
measured against all three at once and they cannot drift.

`exp_tg000` added **no kernel and no divergence** — CKKS's tensor is pointwise mod Q and its rescale is
the single-source `bconv_one`, both of which were already here.

`bconv_one.cpp` returns the remainder in `[0, q_L)`, and that non-centred remainder is the dominant
CKKS precision term (`docs/he/exp_tg000.md`). It does **not** need a centred variant: adding `qHalf`
to the value before a plain `[0, q_L)` rescale is identically the centred rescale, because
`[c + qHalf]_{q_L} = [c]_{q_L} + qHalf − q_L·[[c]_{q_L} > qHalf]`. exp_Tg000 recovers 3.7 bits that
way, with one `modadd_map` per limb and this kernel untouched.

## Contents

| | |
| `compute/recombine_rns.cpp` | the two `DeviceZoneScopedN` markers (`recrns_total`, `recrns_wait_in`) are **deleted outright**, not guarded. Same reason as the three above -- marker volume overflows the profiler's per-RISC DRAM buffer at `G=8` and breaks the 1-1 dispatch map the metrics depend on -- but note the difference: the others sit behind `#ifdef TB001_PHASE_ZONES` and come back with a build flag, so this one cannot. No arithmetic change. |
|---|---|
| `compute/` (15) | `bconv_one` (single-source base conversion, this tree only) · `fused_matmul_a` `fused_matmul_c` (int8 NTT stages, one partial per dispatch) · `fused_matmul_a_acc` `fused_matmul_c_acc` (the same stages with all 8 partials fused, this tree only — the default) · `modadd_twiddle` (fused reduce+twiddle) · `modmul_compute` `modmul_sfpi_full` (Montgomery) · `baseconv_fbc` (approximate FBC) · `baseconv_sk` (exact, sentinel-corrected) · `modadd_map` `modadd_reduce` `modsub_reduce` · `tensor2_fused` · `transpose_tile` |
| `dataflow/` (14) | `reader_fused_batch` (batched W/input reads) · `reader_fused_acc` (the fused contraction's feed, this tree only) · `a2a_corner_bufs` (corner turn over per-tile buffers, this tree only) · `reader_reduce_tw` (fused reduce+twiddle feed) · `reader_ab` `reader_cd_rc` `reader_gather` `reader_i32` `reader_modmul` · `writer_i32` `writer_modmul` `writer_scatter` · `bench_dram_rt` `bench_noc_a2a` (only reached via `NOC_BENCH`) |
