#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""run_he.py — one-shot HE compiler driver: workload params in, hardware performance out.

You pick the WORKLOAD; hardware params default (P = full Tensix grid). It runs the whole pipeline:
  build L1/L2 program -> schedule (place across cores) -> emit spec (codegen) -> compile the
  device runtime -> dispatch on the n150 -> report predicted vs measured performance + correctness.

  python3 run_he.py --groups 64                 # 64 parallel (w*x + y*z) groups, default 64 cores
  python3 run_he.py --groups 128 --cores 32     # override the hardware core count
  python3 run_he.py --groups 256 --prime 201326593

Workload = K independent pointwise groups (`s = w*x + y*z`, 2 muls + 1 add each), the parallel
core of an HE stage. (The codegen runtime currently dispatches SFPU tile-ops; matmul/structural
codegen is the pipeline's remaining generalization.)
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))     # <repo>/compiler
ROOT = os.path.abspath(os.path.join(HERE, ".."))      # <repo>
BUILD = os.path.join(ROOT, "build")                  # single build tree at the repo root
BIN = os.path.join(BUILD, "codegen_run")
SPEC = os.path.join(HERE, "artifacts", "codegen_spec.bin")

import _paths                                     # noqa: F401  (repo layout; see _paths.py)
import ir                                # noqa: E402
import schedule as sched                 # noqa: E402
import codegen as cg                     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="HE compiler driver (workload -> hardware performance)")
    ap.add_argument("--groups", type=int, default=64, help="K independent pointwise groups (the workload)")
    ap.add_argument("--cores", type=int, default=64, help="Tensix cores to use (HW default 64 = full grid)")
    ap.add_argument("--prime", type=int, default=251658241, help="28-bit modulus")
    ap.add_argument("--N", type=int, default=1024, help="elements per tile (fixed 1024 = one 32x32 tile)")
    a = ap.parse_args()

    print(f"== workload ==  {a.groups} groups (s = w*x + y*z), N={a.N}, prime={a.prime}")
    print(f"== hardware ==  {a.cores} cores (default)\n")

    # 1. build + golden + schedule + emit spec  (the compiler)
    prog, data = cg.build_pointwise_dag(a.groups, a.N, a.prime)
    golden = ir.run(prog, data)
    sc = sched.schedule(prog, a.cores)
    nbuf, nop = cg.emit(prog, data, golden, sc, SPEC, a.prime, a.N)
    print(f"[compile]  {nop} tile-ops, {nbuf} buffers")
    print(f"[schedule] {sc['cores_used']} cores used, predicted makespan {sc['makespan']:.0f} us "
          f"({sc['serial']/sc['makespan']:.1f}x vs serial)")

    # 2. compile the device runtime
    print("[build]    cmake --build (codegen_run) ...", end=" ", flush=True)
    r = subprocess.run(["cmake", "--build", "build", "--target", "codegen_run"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED\n" + r.stderr[-2000:]); sys.exit(1)
    print("ok")

    # 3. run on hardware
    print("[run]      dispatching on device ...", flush=True)
    r = subprocess.run([BIN, SPEC], cwd=ROOT, capture_output=True, text=True)
    out = r.stdout
    waves = re.search(r"in (\d+) waves", out)
    dev = re.search(r"steady-state device (\d+)", out)
    ok = re.search(r"bit-exact vs golden: (\w+)", out)
    if not (waves and dev and ok):
        print("could not parse device output:\n" + out + r.stderr[-1500:]); sys.exit(1)

    # 4. final performance report
    dev_us = float(dev.group(1))
    total_mul = a.groups * 2 * a.N            # modular multiplies in the workload
    print("\n" + "=" * 56)
    print("FINAL PERFORMANCE")
    print("=" * 56)
    print(f"  correctness       : {ok.group(1)}  (bit-exact vs golden)")
    print(f"  dispatch waves    : {waves.group(1)}")
    print(f"  predicted makespan: {sc['makespan']:8.0f} us   (cost model)")
    print(f"  measured device   : {dev_us:8.0f} us/run  (steady-state, {a.cores} cores)")
    print(f"  throughput        : {total_mul / dev_us:8.1f} M modmul/s")
    print("=" * 56)


if __name__ == "__main__":
    main()
