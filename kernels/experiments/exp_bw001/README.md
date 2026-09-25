# `kernels/experiments/exp_bw001/` — private kernel tree for exp_Bw001

The four Tensix kernels `host/experiments/exp_bw001.cpp` dispatches. Nothing else in the repo compiles or JITs
from this directory, and the experiment JITs from nowhere else.

```
he_device_target(exp_bw001 SOURCES host/experiments/exp_bw001.cpp KERNELS kernels/experiments/exp_bw001 TEST hw)
```

Same discipline as `kernels/experiments/exp_tb001/`: `kernels/shared/` is shared by 76 device binaries, so a kernel
change made to chase a bandwidth number there is a change to `he_mul_cross`, the whole NTT suite,
`lib/he` and `lib/pir` at once. `KERNELS` sets `OVERRIDE_KERNEL_PREFIX` to this directory — see
`cmake/HeDeviceTarget.cmake` for the mechanism and its trailing-slash requirement.

Unlike `exp_tb001/`, **nothing here is a copy**. All four files are new, so there is no
"diverged from the shared tree" record to keep.

## Contents

| file | what it does |
|---|---|
| `compute/mul_i32_stream.cpp` | `C = A * B` on int32 tiles, `BATCH` tiles per DST acquire. Bare SFPU `mul_int32` — no modular reduction, unlike every other multiply kernel in this repo. |
| `compute/mul_i32_l1.cpp` | The same multiply with all memory traffic removed: one tile pair into DST once, then 2^n multiplies. Measures the SFPU ceiling alone. |
| `dataflow/reader_mul_stream.cpp` | Strided/wrapped feed of one core's tile column, `BATCH` iterations of reads in flight per barrier. |
| `dataflow/writer_mul_stream.cpp` | The matching drain, same addressing, same batching. |

## Two invariants worth not breaking

**BATCH ≤ 4.** DST holds 8 int32 tiles under `fp32_dest_acc_en + dst_full_sync_en`, and each tile in
flight needs an `(a, b)` pair resident, so 4 pairs is the ceiling. `DST_PAIRS` in the host program is
the same constant; they must move together.

**CB depth must be a multiple of BATCH.** The reader and writer address a batch as `base + j*ts`,
which is only contiguous while the reserved run does not straddle the circular buffer's wrap point.
The host guarantees it by sizing every CB to `4*BATCH` slots and only ever reserving/pushing exactly
`BATCH`. `n_iter` is a power of two and `BATCH` divides it, so no batch is ever partial. Break either
half of that and the kernel does not fail loudly — it silently reads or writes the wrong L1 offsets.

Measurements and what they mean: `docs/vector/exp_bw001.md`.
