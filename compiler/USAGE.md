# Running the HE compiler driver (`run_he.py`)

One command takes your **workload** parameters, defaults the **hardware**, and runs the whole
pipeline — compile → schedule → generate code → build device runtime → dispatch on the n150 →
**print a final performance number** (correctness, predicted vs measured latency, throughput).

---

## TL;DR — copy-paste

```bash
# (fresh shell only) resolve the tt-metalium env — single source of truth for all 4 vars
. "$REPO/env.sh"          # $REPO = this repository's root

# run it (works from any directory)
python3 "$REPO/compiler/run_he.py" --groups 64 2>/dev/null
```
`sh run.sh` sources `env.sh` itself, so it needs no exported vars at all.

`2>/dev/null` hides the device driver log lines; drop it if you want to see them.

---

## Prerequisites

- The tt-metalium env must be resolved via `. <repo>/env.sh` (see above). If it reports no usable
  install, create one with `tools/install_ttmetalium.sh` — a tt-metal source build alone is not
  enough, it must be *installed*.
- The **n150 must be free** — only one TT device program at a time (a concurrent job wedges the
  device and needs `tt-smi -r`). Run one `run_he.py` at a time.
- First run may take a few extra seconds if a kernel needs JIT compiling; it is cached after.

---

## The command

```bash
python3 "$REPO/compiler/run_he.py" [options]
```

| option | default | meaning |
|---|---|---|
| `--groups K` | 64 | **workload**: K independent pointwise groups (`s = w*x + y*z`, 2 muls + 1 add each) |
| `--cores P` | 64 | **hardware**: Tensix cores to use (64 = full grid) — defaulted, override if you want |
| `--prime Q` | 251658241 | 28-bit modulus |
| `--N n` | 1024 | elements per tile (one 32×32 tile; leave at 1024) |
| `--help` | — | list all options |

### Copy-paste examples

```bash
# default workload, default hardware
python3 "$REPO/compiler/run_he.py" --groups 64 2>/dev/null

# bigger workload
python3 "$REPO/compiler/run_he.py" --groups 128 2>/dev/null

# same workload, fewer cores (see it scale)
python3 "$REPO/compiler/run_he.py" --groups 64 --cores 16 2>/dev/null

# a different prime
python3 "$REPO/compiler/run_he.py" --groups 256 --prime 201326593 2>/dev/null

# all options
python3 "$REPO/compiler/run_he.py" --help
```

---

## Reading the output

```
== workload ==  64 groups (s = w*x + y*z), N=1024, prime=251658241
== hardware ==  64 cores (default)

[compile]  192 tile-ops, 448 buffers
[schedule] 64 cores used, predicted makespan 66 us (61.9x vs serial)
[build]    cmake --build (codegen_run) ... ok
[run]      dispatching on device ...

========================================================
FINAL PERFORMANCE
========================================================
  correctness       : OK  (bit-exact vs golden)     <- result matches the exact CPU golden
  dispatch waves    : 4                              <- device programs (dependency waves)
  predicted makespan:       66 us   (cost model)     <- what the scheduler predicts
  measured device   :      696 us/run  (steady-state, 64 cores)
  throughput        :    188.3 M modmul/s
========================================================
```

- **correctness** — the device result is compared bit-exact against the compiler's golden
  (`ir.run`). `OK` = the generated code is provably correct.
- **predicted vs measured** — the cost-model estimate vs the real device time. The gap is per-op
  dispatch overhead (this codegen emits one program per op); the batched path is faster.
- **throughput** — modular multiplies per second for this workload.

---

## What it runs under the hood

`run_he.py` chains the compiler stages you can also run individually:

1. `codegen.build_pointwise_dag` → the L1/L2 program (compile)
2. `ir.run` → the exact golden
3. `schedule.schedule` → place ops across cores (stage 6)
4. `codegen.emit` → `codegen_spec.bin` (stage 7 codegen)
5. `cmake --build … codegen_run` → the device runtime
6. `build/codegen_run` → dispatch on the n150, decode, verify

---

## Troubleshooting

| symptom | fix |
|---|---|
| `cannot open … codegen_spec.bin` | run from the driver (it writes the spec); don't run `codegen_run` alone without a spec |
| device hang / `tt-smi` needed | another TT job was running concurrently → `tt-smi -r`, then retry one at a time |
| `find_package(TT-Metalium)` / build error | the tt-metalium env vars aren't exported (see TL;DR) |
| noisy UMD/topology log lines | append `2>/dev/null` |

---

## Scope (so the numbers aren't misread)

The workload is the **parallel pointwise core** of an HE stage (independent modmul/modadd groups),
and the codegen emits **one program per op**, so the reported throughput is **dispatch-bound**.
The batched harness `kernels/ir/stage7_run.cpp` shows the same work reaches **near-linear
57.9× at 64 cores** when fused into wide dispatches — that fusion is the optimization that closes
the predicted-vs-measured gap. Matmul / NTT / full-`ct_mul` codegen is the pipeline's remaining
generalization (see `README.md`).
