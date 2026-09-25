#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""zone_phases.py — sum inner PHASE zones the way critical-path busy is defined.

  Usage: zone_phases.py <profile_log_device.csv> [--clock-mhz F]

For each zone occurrence it takes the MAX OVER CORES, then sums the occurrences -- the same rule
busy_metrics.py applies to whole dispatches, so a phase total is comparable with a dispatch total.

WHAT IT IS FOR.  Once decompose/matmul/recombine are fused into one kernel they stop being separate
dispatches, so dispatch_csv.py can no longer attribute them. Building the fused kernels with
-DNTT_PHASE_ZONES (compute/ntt32768_3f_fused.cpp, compute/fused_matmul_k_1pass.cpp) or
-DTB001_PHASE_ZONES (compute/fused_matmul_a_1pass.cpp) emits inner zones this script can total.

TWO LIMITS, BOTH LOAD-BEARING -- read before quoting a number:

  1. THE PHASE ROWS NEED NOT SUM TO THE KERNEL TOTAL. Zones on different TRISCs (unpack / math / pack)
     overlap by design, so their sum can exceed the enclosing zone. Measured on the single-dispatch
     NTT-32768: phases 43.6 us against a 33.9 us kernel. Read a phase as "which phase held the zone",
     not as a slice of an additive budget.
  2. ZONES INCLUDE CB WAITS. A phase that begins with cb_wait_front charges the wait to itself. In the
     same capture ph_decompose reads 22.71 us mostly because it waits on the reader and on the 32-core
     exchange -- it is not 22.71 us of decomposing.

Marker volume roughly triples with phase zones on, which can overflow the profiler's per-RISC DRAM
buffer and break the 1-1 dispatch map (kernels/experiments/exp_tb001/README.md). Keep them opt-in, and re-check
the dispatch count with busy_metrics.py when they are enabled.
"""
import csv, sys, collections

def main():
    args = [a for a in sys.argv[1:]]
    mhz = 1350.0
    if "--clock-mhz" in args:
        i = args.index("--clock-mhz"); mhz = float(args[i + 1]); del args[i:i + 2]
    if not args:
        print(__doc__); return 2
    path = args[0]

    ev = collections.defaultdict(list)
    with open(path) as f:
        f.readline(); r = csv.reader(f); next(r)
        for x in r:
            if len(x) < 12: continue
            zone, kind = x[10].strip(), x[11].strip()
            if not zone: continue
            ev[(zone, (x[1].strip(), x[2].strip()), x[3].strip())].append((int(x[5]), kind))

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for (zone, core, _risc), lst in ev.items():
        lst.sort(); stack, durs = [], []
        for ts, kind in lst:
            if kind == "ZONE_START": stack.append(ts)
            elif kind == "ZONE_END" and stack: durs.append(ts - stack.pop())
        if durs: per[zone][core].append(durs)

    print(f"  device clock {mhz:.0f} MHz; per occurrence: max over cores, then summed\n")
    print(f"  {'zone':24s} {'occurrences':>11s} {'us':>10s}")
    for zone, cores in sorted(per.items()):
        flat = collections.defaultdict(list)
        for _core, runs in cores.items():
            for durs in runs:
                for i, d in enumerate(durs): flat[i].append(d)
        print(f"  {zone:24s} {len(flat):11d} {sum(max(v) for v in flat.values()) / mhz:10.2f}")
    print("\n  NB: phase rows need not sum to the enclosing kernel zone, and they include CB waits."
          "\n      See this file's docstring before quoting them.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
