# vector_modmul — element-wise modular multiplication on Wormhole B0

> **Status: Historical.** Consolidated into [`../modular_multiplication.md`](../modular_multiplication.md) §10.
> The old device/README: env knobs, profiling commands and the RNS variant. Its build commands predate the root build dir.
> Numbers are a snapshot, measured on Wormhole n150.

Computes `C[i] = (A[i] * B[i]) mod q` for vectors of uint32, where `A`/`B` are
Montgomery-domain residues (lazy `[0, 2q)`) and `q` is a 28-bit prime. Same
Montgomery algorithm as `../cpu/montgomery.hpp`, running on the Tensix SFPU.

## Build & run
```bash
cd tt-he
. env.sh                              # resolves the tt-metalium install, exports all 4 env vars
cd device
cmake -B build -G Ninja && cmake --build build
./build/vector_modmul                 # default: full modmul, prime 251658241, 4 tiles (N=4096)
```
`env.sh` is the single source of truth for `TT_METAL_HOME`, `TT_METAL_RUNTIME_ROOT`,
`CMAKE_PREFIX_PATH` and `LD_LIBRARY_PATH` — never set them by hand. It auto-detects the install
prefix (`$TT_METALIUM_PREFIX`, then `/usr/local`, then `$HOME/work/ttm-install`) and fails with
instructions if none is usable.

No install yet on this machine? Create one from a tt-metal source build:
```bash
sh tools/install_ttmetalium.sh -s /path/to/tt-metal          # -> $HOME/work/ttm-install
sh tools/install_ttmetalium.sh -s /path/to/tt-metal -p /usr/local -S   # or system-wide (sudo)
```
A tt-metal **build tree** (`build_Release`) is not usable directly — `find_package(TT-Metalium)`
needs `Metalium.cmake`, which only `cmake --install` generates.

## Env knobs (validate stages / scale without rebuilding)
| var | default | meaning |
|---|---|---|
| `MODMUL_KERNEL` | `compute/modmul_compute.cpp` | compute kernel rel-path under `kernels/` |
| `MODMUL_MODE` | `modmul` | golden to check against: `passthrough\|mullo\|mulhi\|modmul` |
| `MODMUL_PRIME` | `251658241` | 28-bit modulus |
| `MODMUL_NTILES` | `4` | number of 1024-elem tiles (N = 1024·tiles) |

Examples:
```bash
MODMUL_KERNEL=compute/modmul_mullo.cpp MODMUL_MODE=mullo ./build/vector_modmul   # stage-1 multiply check
MODMUL_NTILES=1024 ./build/vector_modmul                                          # throughput (~2 G modmul/s, 64 cores)
MODMUL_PRIME=201326593 ./build/vector_modmul
```

## Device profiling traces
```bash
TT_METAL_DEVICE_PROFILER=1 MODMUL_ITERS=1 MODMUL_NTILES=64 \
  MODMUL_KERNEL=compute/modmul_sfpi_hybrid.cpp ./build/vector_modmul
# per-core/per-RISC trace -> $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
```
`MODMUL_ITERS` (default 50) controls timed enqueues; use 1 for a clean trace. Archived
traces for 2^12..2^18 are in `generated/profiler/traces/`; see report §11 for the
per-RISC cycle breakdown (MATH/SFPU is the bottleneck).

## RNS / multi-modulus version (up to 16 vectors, each its own prime)
```bash
./build/vector_modmul_rns                                    # 16 vectors x 2^16 elems, 16 primes
MODMUL_NVEC=8 MODMUL_VEC_TILES=64 ./build/vector_modmul_rns   # 8 vectors x 2^16
MODMUL_NVEC=16 MODMUL_VEC_TILES=1 ./build/vector_modmul_rns   # stress: prime switches every tile
```
Env: `MODMUL_NVEC` (1..16), `MODMUL_VEC_TILES` (tiles/vector, 64 → 2^16 elems),
`MODMUL_KERNEL` (default `compute/modmul_rns_sfpi.cpp`). Data is vector-major;
the kernel selects each tile's prime by `tile_id / tiles_per_vector`.

## Layout

Host programs are grouped by operation; the `kernels/` tree is **shared** across all of them
(`OVERRIDE_KERNEL_PREFIX` points every target at `kernels/`):

```
device/
  vector/      vector_modmul.cpp, vector_modmul_rns.cpp
  matmul/      modular_matmul.cpp, fused_modmatmul.cpp
  ntt/         ntt1024{,_dev,_dev_mc}.cpp, ntt32768_dev.cpp, ntt32768_2factor_dev.cpp
  host/tests/       test_decompose.cpp, test_recombine{,_bigk,_one}.cpp
  kernels/     compute/ + dataflow/  (shared by all)
```
All executables still land in `build/` (e.g. `./build/vector_modmul`).

## Files
- `vector/vector_modmul.cpp` — single-prime host: buffers, multi-core work split, runtime args, timing, golden check.
- `vector/vector_modmul_rns.cpp` — RNS host: V vectors each with its own prime, per-vector validation.
- `kernels/shared/compute/modmul_rns_sfpi.cpp` — multi-modulus hybrid SFPI kernel (per-tile prime lookup).
- `kernels/shared/dataflow/{reader,writer}_modmul.cpp` — DRAM↔CB tile movement (per-core tile range).
- `kernels/shared/compute/modmul_compute.cpp` — full Montgomery modmul, building-block tile-ops.
- `kernels/shared/compute/modmul_sfpi_hybrid.cpp` — **hybrid**: non-multiply ops fused into
  custom SFPI register passes, multiply still via `mul_int_tile`. ~1.7× faster, bit-exact.
- `kernels/shared/compute/modmul_mullo.cpp` — stage-1 bring-up kernel (raw `mul_int_tile`).

```bash
MODMUL_KERNEL=compute/modmul_sfpi_hybrid.cpp MODMUL_NTILES=1024 ./build/vector_modmul   # ~3.5 G modmul/s
```

See `../docs/vector/plan_vector_modmul.md` for design, hardware notes, and measured results.
