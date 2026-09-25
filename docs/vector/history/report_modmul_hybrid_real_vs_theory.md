# Report: Hybrid Modmul Real vs Theory at 2^16

> **Status: Historical.** Referenced from [`../modular_multiplication.md`](../modular_multiplication.md) §11.
> Roofline analysis at N=2^16: the math body reaches 59% of the optimistic SFPU roof. Kept in full — the current doc cites the result but not the derivation.
> Numbers are a snapshot, measured on Wormhole n150.

**Kernel:** `kernels/shared/compute/modmul_sfpi_hybrid.cpp`  
**Workload:** `N = 2^16 = 65,536` elements = 64 tiles  
**Core use:** 64 tiles on 64 cores, one tile/core, so the compute grid is fully occupied  
**Trace:** `device/generated/profiler/traces/trace_modmul_components_2p16.csv`

This note compares measured device latency against simple theory limits for the hybrid
modular multiplication kernel. The comparison is only for the 2^16 case, because that is
the smallest size that fully uses the 64-core grid.

## Measurement Setup

The profile was captured with additional `DeviceZoneScopedN` regions in
`modmul_sfpi_hybrid.cpp`:

```bash
TT_METAL_DEVICE_PROFILER=1 MODMUL_ITERS=1 MODMUL_NTILES=64 \
  MODMUL_KERNEL=compute/modmul_sfpi_hybrid.cpp ./build/vector_modmul
```

The real column below uses the median cycle count across active cores from the trace.
The profiler clock is 1 GHz, so cycles and ns are numerically the same.

## Real vs Theory

| Component | Real measured latency | Theory lower bound |
|---|---:|---:|
| DRAM read A+B into input CBs (`NCRISC-KERNEL`) | 4,544 cyc = 4.54 us | measured-implied: 524,288 B / 115 B/cyc = 4,544 cyc |
| Whole compute tile path (`modmul_tile_total`) | 21,976 cyc = 21.98 us | 10,080 cyc = 10.08 us |
| CB to DST copy for A+B (`modmul_copy_in`) | 4,834 cyc = 4.83 us | 128 cyc = 0.13 us |
| Math body total (`modmul_math`) | 16,851 cyc = 16.85 us | 9,888 cyc = 9.89 us |
| `face_limbs` | 569 cyc = 0.57 us | 320 cyc = 0.32 us |
| 4 x `a,b` limb multiplies | 5,547 cyc = 5.55 us | 3,584 cyc = 3.58 us |
| `face_lohi` | 1,290 cyc = 1.29 us | 544 cyc = 0.54 us |
| `lo*q_inv` + split | 1,829 cyc = 1.83 us | 1,096 cyc = 1.10 us |
| 4 x `t*q` limb multiplies + constants | 5,938 cyc = 5.94 us | 3,704 cyc = 3.70 us |
| `face_reduce` lazy output | 1,397 cyc = 1.40 us | 640 cyc = 0.64 us |
| DST to output CB pack (`modmul_pack_out`) | 64 cyc = 0.06 us | 64 cyc = 0.06 us |
| Output CB push + input CB pop (`modmul_cb_push_pop`) | 21 cyc = 0.02 us | not modeled |
| DRAM write C from output CB (`BRISC-KERNEL`) | 22,768 cyc = 22.77 us | 8,738 cyc = 8.74 us |

**Caption, real column.** Real values come from
`device/generated/profiler/traces/trace_modmul_components_2p16.csv`. For `modmul_*`
rows, the named zones are inside `modmul_sfpi_hybrid.cpp`. The DRAM read row is the
reader dataflow kernel, `kernels/shared/dataflow/reader_modmul.cpp`, and the DRAM write
row is the writer dataflow kernel, `kernels/shared/dataflow/writer_modmul.cpp`. The
writer time includes waiting for compute output, so it is an end-to-end pipeline proxy,
not pure useful DRAM-write time.

**Caption, theory column.** Compute theory assumes an optimistic 1 SFPU instruction/cycle.
`mul_int_tile<UInt32>` dispatches to `mul_int32`, which has 28 visible `TT_*` / `TTI_*`
SFPU instruction calls per row in
`tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h`.
One tile has 32 rows. Custom SFPI pass counts come from the current
`modmul_sfpi_hybrid.cpp`, counting each `dst_reg` load/store and vector ALU operation as
one SFPU instruction. `fill_tile_int<UInt32>` is counted from
`tt-metal/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h`.
CB to DST and DST to CB theory uses 64 B/cycle per active core. For DRAM read, the
requested 30 GB/s assumption is slower than the measured reader, so it is not used as a
roof; the read row instead reports the measured-implied effective bandwidth. DRAM write
keeps the conservative 30 GB/s full-vector estimate, but the measured writer row includes
waiting for compute output.

## How The Theory Numbers Were Counted

Math-body instruction count:

| Math component | Instruction count |
|---|---:|
| `face_limbs` | 10 inst/row x 32 rows = 320 |
| 4 x `a,b` limb multiplies | 4 x 28 inst/row x 32 rows = 3,584 |
| `face_lohi` | 17 inst/row x 32 rows = 544 |
| `lo*q_inv` + split | 896 + 40 + 160 = 1,096 |
| 4 x `t*q` limb multiplies + constants | 4 x 896 + 3 x 40 = 3,704 |
| `face_reduce` lazy output | 20 inst/row x 32 rows = 640 |
| **Math total** | **9,888** |

Memory/copy lower bounds:

| Movement | Bytes | Assumed bandwidth | Theory |
|---|---:|---:|---:|
| DRAM read A+B, full vector | 64 tiles x 2 x 4096 B = 524,288 B | real measured: ~115 B/cycle; 30 B/cycle is too low | 4,544 cycles measured |
| CB to DST copy A+B, per core | 2 x 4096 B = 8192 B | local 64 B/cycle | 128 cycles |
| DST to output CB pack, per core | 4096 B | local 64 B/cycle | 64 cycles |
| DRAM write C, full vector | 64 tiles x 4096 B = 262,144 B | shared 30 B/cycle | 8,738 cycles |

The whole compute tile-path theory is:

```text
CB->DST copy + math body + DST->CB pack
= 128 + 9,888 + 64
= 10,080 cycles/tile
```

The full pipeline lower bound is not the sum of reader, compute, and writer, because those
stages overlap. With shared-device DRAM bandwidth and per-core compute/copy bandwidth, the
pipeline lower bound is approximately the maximum stage:

```text
max(DRAM read 4,544, compute tile path 10,080, DRAM write 8,738)
= 10,080 cycles
```

## Deep Dive: `modmul_copy_in`

`modmul_copy_in` covers this code in `modmul_sfpi_hybrid.cpp`:

```cpp
copy_tile_init(cb_in0); copy_tile(cb_in0, 0, 0);  // d0 = a
copy_tile_init(cb_in1); copy_tile(cb_in1, 0, 1);  // d1 = b
```

It is not a raw SRAM byte copy. `copy_tile()` is the compute API's CB-to-DST path:

- `copy_tile_init()` calls `copy_tile_to_dst_init_short()`.
- `copy_tile_to_dst_init_short()` configures both sides:
  - UNPACK side: `llk_unpack_A_init(...)`
  - MATH side: `llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, ...>(...)`
- `copy_tile()` then does both:
  - UNPACK side: `llk_unpack_A(...)`, which unpacks the tile from the CB into source registers.
  - MATH side: `llk_math_eltwise_unary_datacopy<DataCopyType::A2D, ...>(...)`, which moves the unpacked source data into the selected DST tile.

The same source-level `DeviceZoneScopedN("modmul_copy_in")` appears on multiple TRISCs,
but they do different parts of the operation. From `trace_modmul_components_2p16.csv`:

| RISC | Median cycles in `modmul_copy_in` | Interpretation |
|---|---:|---|
| TRISC_0 | 385 | UNPACK-side work is small: unpack setup + `llk_unpack_A` for two input tiles |
| TRISC_1 | 4,834 | MATH-side path gates the operation: datacopy init/run plus waiting/synchronization |
| TRISC_2 | 18 | PACK side is effectively idle for this region |

The table above uses the TRISC_1 number because it is the latency seen by the MATH path
before `modmul_math` can start.

The `64 B/cycle` theory is only the ideal local-movement bandwidth:

```text
2 input tiles * 4096 B/tile / 64 B/cycle = 128 cycles
```

The real path is slower because `copy_tile` does more than move bytes:

- **LLK init is inside the measured region.** The kernel calls `copy_tile_init(cb_in0)` and
  `copy_tile_init(cb_in1)` every tile. Those calls configure unpack/datacopy state and
  program/reset MOP/counters.
- **UInt32 + fp32 DST mode does not use the simplest MOVA2D path.** In
  `llk_math_eltwise_unary_datacopy.h`, `DataCopyType::A2D` uses an `ELWADD` MOP when
  `fp32_dest_acc_en` is enabled and the destination format is not UInt16. This kernel
  requires `fp32_dest_acc_en=true` for int32 correctness, so the operation is not just a
  64 B/cycle MOVA2D copy.
- **UNPACK and MATH synchronize.** The unpacker has to produce source-register data before
  the math-side datacopy can write DST. The measured TRISC_1 zone includes this handoff.
- **DST ownership and addressing add overhead.** The copy happens after
  `tile_regs_acquire()` and before later SFPU math uses DST. The LLK sets DST write
  address state, clears it, and updates counters/address modifiers.

So `128 cycles` is a physical lower bound for local data movement, not a realistic
cycle-accurate estimate for this `copy_tile_init + copy_tile` implementation path.

## Summary

The measured math body reaches:

```text
9,888 theory cycles / 16,851 measured cycles = 59% of the optimistic SFPU roof
```

The measured whole compute tile path reaches:

```text
10,080 theory cycles / 21,976 measured cycles = 46% of the optimistic compute-tile roof
```

At full grid, the math-body theory throughput is:

```text
64 cores * 1024 elements/tile * 1 GHz / 9,888 cycles/tile
  = 6.63 G modmul/s
```

The current gap is mainly inside the compute kernel, especially the 9 emulated `mul_int32`
calls and their wrapper/scheduling overhead. The observed BRISC writer duration is close
to the compute tile time because the writer waits for compute output; it should not be
read as pure DRAM-write cost.

## Why Real Differs From Theory

The theory column is an optimistic lower bound, not a cycle-accurate model. It assumes
perfect 1 SFPU instruction/cycle issue for the source-level instruction count, plus ideal
bandwidth for local CB/DST movement and shared DRAM movement. Real hardware execution has
several extra costs:

- **`mul_int32` dependency chain:** the emulated integer multiply is a sequence of loads,
  shifts, masks, casts, MADs, mantissa extracts, shifts, adds, and a store. Many of these
  operations depend on the previous result, so they cannot all issue at the ideal rate.
- **LLK wrapper overhead:** each `mul_int_tile` and custom SFPI pass goes through LLK
  start/done code, vector-mode face iteration, DST address setup/increment, and stall
  waits. The theory count mostly covers only the arithmetic body.
- **Address and DST access overhead:** expressions like `dst_reg[idx * NV + i]` are
  counted as one load or store in the theory, but real code also pays address setup and
  possible DST access stalls.
- **Copy and synchronization overhead:** the whole compute tile path includes CB waits,
  `tile_regs_acquire`, `copy_tile`, `tile_regs_commit`, `tile_regs_wait`, pack, and CB
  push/pop. These are outside the pure math-body instruction count.
- **Profiler perturbation:** this trace is compiled with `PROFILE_KERNEL=1` and includes
  `DeviceZoneScopedN` markers. The markers are useful for breakdowns, but they inflate
  absolute cycle counts compared with a clean run.

For DRAM, the read row is faster than the conservative 30 GB/s assumption:

```text
524,288 B / 4.544 us = ~115 GB/s effective read bandwidth
524,288 B / 30 GB/s = 17.48 us, which is slower than real and therefore not a valid roof
```

That means 30 GB/s is not a true hardware peak for this path; it is only a conservative
assumption. The write row cannot be used the same way because `BRISC-KERNEL` includes
waiting for the compute output.
