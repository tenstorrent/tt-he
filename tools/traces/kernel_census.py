#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""kernel_census.py — device busy by CENSUS, for workloads too large to capture 1-1.

  Usage: kernel_census.py <csv> --labels FILE [--clock-mhz F] [--out FILE]

WHY THIS EXISTS

  busy_metrics.py derives `critical path busy` as  sum over dispatches of ( max over its cores ),
  and to do that it needs a verified 1-1 map: every dispatch the host issued present in the trace with
  all 5 RISCs. Past a few thousand dispatches that is not obtainable — the profiler's per-RISC DRAM
  buffer overflows and tt-metal drops markers. exp_Tg018 at N=65536 issues 5412 and the drained
  capture plateaus at ~5554 of 6020, identically at every drain interval.

  But the sum has structure. A workload of this shape is the SAME FEW KERNELS repeated thousands of
  times at a fixed core count and a fixed tile shape, so

      critical path busy  ~=  sum over kernel types k of  ( N_k  x  mean_busy_k )

  where N_k comes from the HOST (the label file, which is authoritative and complete) and mean_busy_k
  comes from whatever sample of dispatches the trace did capture. Dropped markers stop being fatal:
  they cost sample size, not correctness.

WHAT IT ASSUMES, AND HOW IT CHECKS

  The estimate is only as good as "every dispatch of kernel k costs about the same". That is an
  assumption about the workload, not a fact, so this tool does not take it on trust — it prints the
  per-kernel spread (min / mean / max and the coefficient of variation) and flags any kernel whose
  CV exceeds 15%. A tight distribution earns the extrapolation; a wide one means that kernel is being
  dispatched at more than one shape and should be split before the number is believed.

WHAT IT IS AND IS NOT

  IS      an estimate of `critical path busy` — device time with the inter-dispatch idle removed.
  IS NOT  `busy span` (a window, so it still contains the idle) and not `wall time`.

  So the ordering to expect against the host-measured figure is

      host device busy  >=  critical path busy (this estimate)

  because the host stopwatch additionally contains intra-segment device idle and the per-call blocking
  overhead. If this estimate comes out ABOVE host device busy, something is wrong — either the sample
  is unrepresentative or the census does not match the run that was captured.

  The parsing rule (which zones count, how dispatch envelopes are grouped) is byte-for-byte
  dispatch_csv.py's, so the two tools cannot disagree about what a dispatch is.

WHY N x mean IS THE RIGHT COMPOSITION (both halves verified on exp_Tg018's capture)

  N counts PROGRAM DISPATCHES, not kernel instances, and mean is already the MAX over the dispatch's
  cores -- never their sum. So the two kinds of parallelism are handled separately:

  * WITHIN a dispatch the cores run concurrently, and `max` collapses that. Measured: sum-over-cores
    divided by max-over-cores is 50.6 (median 53.4) against 64 cores, so they are ~80% overlapped.
    Had this been summed instead of maxed the answer would be ~50x too large.
  * BETWEEN dispatches there is no overlap -- they are serialized on the command queue. Measured: 0
    overlapping adjacent pairs out of 6412. That is what makes the composition a sum.

  A corollary worth knowing: do NOT just add up the trace's group latencies. On exp_Tg018 the capture
  holds 6412 groups against 5412 dispatches the host actually issued (drain-boundary fragmentation
  plus residue), and summing them gives 289.2 ms against a host device busy of 238.4 ms -- 21% high.
  Taking N from the host and only the MEAN from the trace gives 236.6 ms, within 0.8%. The census is
  robust to spurious markers as well as to dropped ones, which is the whole reason it exists.
"""
import bisect
import collections
import csv
import math
import sys

SKIP = {"BRISC-FW", "NCRISC-FW", "TRISC-FW"}


def load(path):
    """Identical to dispatch_csv.py's loader: -KERNEL zones are dispatch envelopes, the rest named."""
    st, env, named = collections.defaultdict(list), [], []
    with open(path) as f:
        f.readline()
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        ix = {n: hdr.index(n) for n in
              ("core_x", "core_y", "RISC processor type", "time[cycles since reset]", "zone name", "type")}
        for x in r:
            if len(x) < len(hdr):
                continue
            z = x[ix["zone name"]].strip()
            ty = x[ix["type"]].strip()
            core = (x[ix["core_x"]].strip(), x[ix["core_y"]].strip())
            risc = x[ix["RISC processor type"]].strip()
            t = int(x[ix["time[cycles since reset]"]])
            k = (core, risc, z)
            if ty == "ZONE_START":
                st[k].append(t)
            elif ty == "ZONE_END" and st[k]:
                s0 = st[k].pop()
                if z.endswith("-KERNEL"):
                    env.append((s0, t, core, risc))
                elif not z.endswith("-FW") and z not in SKIP:
                    named.append((s0, t, core, z))
    env.sort()
    named.sort()
    return env, named


ZONE2HOST = {"fused_total": "fused_matmul_a", "fusedC_total": "fused_matmul_c",
             "fusedAcc_total": "fused_matmul_a_acc", "fusedCacc_total": "fused_matmul_c_acc",
             "fbc_total": "baseconv_fbc"}


def main():
    args, pos, lblf, clk, out, dfo_name = sys.argv[1:], [], None, 1350.0, None, None
    while args:
        a = args.pop(0)
        if a == "--labels":
            lblf = args.pop(0)
        elif a == "--dataflow-only":
            dfo_name = args.pop(0)
        elif a == "--clock-mhz":
            clk = float(args.pop(0))
        elif a == "--out":
            out = args.pop(0)
        else:
            pos.append(a)
    if not pos or lblf is None:
        print(__doc__)
        return 2

    # ---- host census: authoritative counts, straight from the dispatch labels ----
    host = collections.Counter()
    hop = collections.defaultdict(collections.Counter)      # kernel -> he_op -> count
    with open(lblf) as f:
        for d in csv.DictReader(f):
            k = d["host_kernel"].strip()
            host[k] += 1
            hop[k][d["he_op"].strip()] += 1
    n_host = sum(host.values())

    # ---- trace sample: group envelopes into dispatches exactly as dispatch_csv.py does ----
    env, named = load(pos[0])
    if not env:
        print("no kernel envelopes in", pos[0])
        return 1
    groups, cur, end = [], [env[0]], env[0][1]
    for e in env[1:]:
        if e[0] <= end:
            cur.append(e); end = max(end, e[1])
        else:
            groups.append(cur); cur = [e]; end = e[1]
    groups.append(cur)
    # A dispatch with no TRISC envelope ran NO compute kernel -- it is dataflow-only. Those carry no
    # named zone either, so they cannot be identified by name; --dataflow-only attributes them to a
    # host kernel, and the count check below is what makes that inference falsifiable.
    kern = [g for g in groups if any(e[3].startswith("TRISC") for e in g)]
    dfo  = [g for g in groups if not any(e[3].startswith("TRISC") for e in g)]
    ws = [min(e[0] for e in g) for g in kern]
    we = [max(e[1] for e in g) for g in kern]

    kof = {}
    for s0, _e0, _c, z in named:
        i = bisect.bisect_right(ws, s0) - 1
        if 0 <= i < len(kern) and s0 <= we[i]:
            kof.setdefault(i, set()).add(z)

    # per-dispatch busy = max over participating cores, as in dispatch_csv.py
    samp = collections.defaultdict(list)
    for i, g in enumerate(kern):
        per = collections.defaultdict(lambda: [1 << 62, 0])
        for s0, e0, c, _r in g:
            per[c][0] = min(per[c][0], s0)
            per[c][1] = max(per[c][1], e0)
        lat_us = max(hi - lo for lo, hi in per.values()) / clk
        zs = sorted(kof.get(i, ()))
        if len(zs) != 1:
            continue                                        # ambiguous envelope, not a clean sample
        samp[ZONE2HOST.get(zs[0], zs[0])].append(lat_us)

    dfo_note = ""
    if dfo_name and dfo:
        for g in dfo:
            per = collections.defaultdict(lambda: [1 << 62, 0])
            for s0, e0, c, _r in g:
                per[c][0] = min(per[c][0], s0)
                per[c][1] = max(per[c][1], e0)
            samp[dfo_name].append(max(hi - lo for lo, hi in per.values()) / clk)
        ok = "matches" if abs(len(dfo) - host.get(dfo_name, 0)) <= 0.15 * max(1, host.get(dfo_name, 0)) else "DOES NOT MATCH"
        dfo_note = (f"  dataflow-only dispatches: {len(dfo)} found, attributed to '{dfo_name}'"
                    f" (host census {host.get(dfo_name, 0)}) -- {ok}\n")

    print(f"\n  capture: {len(kern)} dispatch envelopes sampled;  host census: {n_host} dispatches")
    print(f"  device clock {clk:.0f} MHz")
    if dfo_note: print(dfo_note, end="")
    print()
    print(f"  {'kernel':<22}{'N (host)':>9}{'sampled':>9}{'mean us':>10}{'min':>9}{'max':>9}{'CV':>7}"
          f"{'N x mean':>12}{'%':>7}")
    print("  " + "-" * 96)

    rows, total, wide = [], 0.0, []
    for k in sorted(host, key=lambda k: -host[k] * (sum(samp[k]) / len(samp[k]) if samp.get(k) else 0)):
        v = samp.get(k)
        if not v:
            rows.append((k, host[k], 0, None, None, None, None, None))
            continue
        mean = sum(v) / len(v)
        cv = (math.sqrt(sum((x - mean) ** 2 for x in v) / len(v)) / mean) if mean else 0.0
        tot = host[k] * mean / 1000.0
        total += tot
        if cv > 0.15:
            wide.append((k, cv))
        rows.append((k, host[k], len(v), mean, min(v), max(v), cv, tot))

    for k, n, ns, mean, lo, hi, cv, tot in rows:
        if mean is None:
            print(f"  {k:<22}{n:>9}{0:>9}{'— not sampled —':>45}")
        else:
            print(f"  {k:<22}{n:>9}{ns:>9}{mean:>10.1f}{lo:>9.1f}{hi:>9.1f}{cv*100:>6.1f}%"
                  f"{tot:>11.2f}m{100*tot/total:>6.1f}%")
    print("  " + "-" * 96)
    print(f"  {'ESTIMATED critical path busy':<22}{'':>9}{'':>9}{'':>10}{'':>9}{'':>9}{'':>7}{total:>11.2f}m")

    miss = [k for k, n, ns, m, *_ in rows if m is None]
    if miss:
        print(f"\n  WARNING: no trace sample for {miss} — their contribution is MISSING from the total.")
    if wide:
        print(f"\n  WARNING: wide spread (CV > 15%), so the census assumption is weak for: "
              + ", ".join(f"{k} ({cv*100:.0f}%)" for k, cv in wide))
        print("           that kernel is probably dispatched at more than one shape; split it before"
              " trusting its row.")
    if not miss and not wide:
        print("\n  every kernel sampled, every CV <= 15% — the census extrapolation is well conditioned.")

    print("\n  NOTE: this estimates CRITICAL PATH BUSY (inter-dispatch idle removed). It is not busy"
          "\n        span and not wall time. Expect  host device busy >= this estimate.\n")

    if out:
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel", "n_host", "n_sampled", "mean_us", "min_us", "max_us", "cv",
                        "total_ms", "pct", "he_ops"])
            for k, n, ns, mean, lo, hi, cv, tot in rows:
                w.writerow([k, n, ns,
                            f"{mean:.3f}" if mean is not None else "",
                            f"{lo:.3f}" if lo is not None else "",
                            f"{hi:.3f}" if hi is not None else "",
                            f"{cv:.4f}" if cv is not None else "",
                            f"{tot:.4f}" if tot is not None else "",
                            f"{100*tot/total:.2f}" if tot is not None else "",
                            ";".join(f"{o}:{c}" for o, c in hop[k].most_common())])
        print(f"  wrote {out}")
    return 0


sys.exit(main())
