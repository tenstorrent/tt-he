# Fully on-device size-1024 NTT (single Tensix core)

> **Status: Historical.** Superseded by [`../ntt_1024.md`](../ntt_1024.md).
> Design of the earlier 7-program on-device version, before the stages were fused.
> Numbers are a snapshot of the code at the time and were measured on Wormhole n150.

**Program:** `tests/ntt/ntt1024_dev.cpp`  ·  **Status:** correct (`0/1024` vs direct-NTT golden, seeds 0,1,2,7), `q = 167772161`.

The entire NTT runs on the device. The host writes the input + baked constants **once**, issues
10 chained device programs that pass intermediates through DRAM, and reads the result **once** —
there is no host↔device data movement between stages.

## Pipeline (plane formulation)

`C = A@B mod q` is done with **plain** radix-128 limb tiles (no interleaving): decompose A into
`a_p = limb_p(A)` and B into `b_{p,r} = limb_r(128^p·B mod q)`, run 16 int8 tile-matmuls
`C_r = Σ_p a_p@b_{p,r}` (4 int32 planes), then recombine `Σ_r 128^r·C_r mod q` in **one kernel**
(`recombine_compute.cpp`) with a single Montgomery reduction:
`low=C0+(C1<<7); hi=C2+(C3<<7); R=(low + (hi·2^14 mod q)) mod q` — only the `hi·2^14` needs a
reduction; the shifts/adds and the final two conditional subtracts are cheap. (Earlier this was
4 `modmul` ×128^r + `modadd`; the optimization cuts it to 1 reduction.)

The radix fold is **baked into the constant W offline** for both stages (never computed on the
dynamic data), so neither stage has a runtime fold modmul — `Xin`/`G'` only get a mask&shift
`decompose`:

```
A: G  = W @ Xin     decompose(Xin) → matmul(folded W) → recombine
B: G' = G ⊙ T       twiddle modmul (T pre-converted to Montgomery)
C: H  = G' @ W      decompose(G') → matmul(folded W) → recombine
out[j2+32·j1] = H[j2,j1]   (host reindex of the single output read)
```

7 device programs, chained through DRAM buffers (`Xin→xlimbs→CrA→G→Gp→aC→CrC→H`).

## Kernels

| kernel | role | status |
|---|---|---|
| `compute/decompose_limbs.cpp` | int32 → 4 int8 radix-128 limb tiles | new (unit-tested) |
| `compute/matmul_compute.cpp` | int8 plane matmul `C_r=Σ_p a_p@b_{p,r}` | reused |
| `compute/modmul_compute.cpp` | Montgomery modmul (the twiddle) | reused |
| `dataflow/reader_matmul_foldlhs.cpp` | stage-A matmul reader: 16-tile folded-W LHS, 4-tile Xin RHS | new |
| `compute/modadd_reduce.cpp` | sum n tiles mod q | new |
| `compute/recombine_compute.cpp` | one-kernel recombine `Σ 128^r·C_r mod q`, 1 reduction | new |
| `dataflow/reader_matmul_planes.cpp` | feeds `a[p]`, `b[p·NB+r]` | new |
| `dataflow/{reader_i32,writer_i8,writer_i32}.cpp` | int32/int8 tile I/O (TensorAccessor) | new |

## Performance

```
147 us / NTT-1024   (device, 7 chained stages, single core)
```
vs the host-orchestrated `ntt1024` at **~1167 µs** end-to-end → **~8× faster**. Progression:
465 µs (pure on-device) → 287 µs (single-kernel recombine) → **147 µs** (radix fold baked into the
constant `W`, removing the 4-modmul runtime fold on `Xin`). See `NTT_1024_ONDEVICE_PERF.md` for the
per-stage breakdown; the 3 Montgomery reductions (2 recombines + 1 twiddle) are now the bulk, and
the int8 matmul is only ~14 µs. Remaining win: fuse stages to cut the 7-enqueue dispatch overhead.

## Notes / caveats

- Single core, batch = 1. No multi-core scaling.
- Forward NTT only (no `intt` / round-trip yet).
- Wormhole-specific gotcha hit & fixed: device int8 tile **writes** past `NUM_DRAM_BANKS` (12)
  corrupt with `InterleavedAddrGenFast`; use `TensorAccessor` (see `writer_i8.cpp`).

## Reproduce

```
cd device && cmake -B build -G Ninja && ninja -C build ntt1024_dev
./build/ntt1024_dev [q] [seed]      # default 167772161 0
```
Unit tests: `test_decompose`, `test_recombine`.
