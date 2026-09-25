#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""busy_metrics.py — the four latency metrics. Nothing else.

  Usage: busy_metrics.py <csv> --expect N [--wall-ms X] [--host-busy-ms X] [--clock-mhz F]
                         [--dataflow-ok]   accept dispatches that run no compute kernel
                         [--drain]         a drained capture: critical path busy only, no busy span

  `--expect N` is the host's own dispatch count for the captured multiply. exp_tb001 prints the
  exact command to run, with both values filled in:

      HE_PROF2: 1-1 check -> tools/traces/busy_metrics.py <csv> --expect 636 --wall-ms 24.736

THE DEVICE TIMEBASE — READ THIS BEFORE TRUSTING ANY DEVICE-SIDE NUMBER

  The CSV column is "time[cycles since reset]": device CYCLES, not nanoseconds. Cycles tick at AICLK,
  which on this p150 is 800 MHz idle and boosts to a sustained 1350 MHz under load (tt-smi; the cap
  AICLK_LIMIT_MAX = 0x546 = 1350). Every tool in this directory used to divide by 1000 and call the
  result microseconds, i.e. assume 1 GHz, which overstates every device-side duration by 1.35x.

  That assumption is not merely imprecise, it is provably wrong: at 1 GHz the host's own stopwatch
  around a blocking segment replay comes out 128 us/call SHORTER than the device work it brackets,
  which cannot happen. At 1350 MHz the same comparison gives +104 us/call, i.e. ordinary blocking-call
  overhead. Default here is therefore --clock-mhz 1350; override it if the card is clocked otherwise.

  Note this does NOT affect the host-measured numbers (wall time, and the programs' "DEVICE-BUSY
  span", which is a host stopwatch). It affects only what is derived from the trace.

THE FOUR METRICS

  Two are measured on the host and two are derived from the trace. Having both sides is the point:
  turning the profiler on perturbs wall time and busy span, but barely touches host device busy
  (measured: 11.18-11.23 ms un-profiled vs 11.27-11.49 ms profiled, ~0.6%), so the host pair anchors
  the trace pair.

  HOST-MEASURED (std::chrono; immune to the device-timebase issue below)

  wall time            host end-to-end. Pass it with --wall-ms.
  host device busy     sum of the host's stopwatch around each BLOCKING segment replay. Excludes host
                       stalls BETWEEN segments; includes device idle inside a segment plus ~52 us/call
                       of blocking-call overhead. Pass it with --host-busy-ms.

  TRACE-DERIVED (device cycles; needs --clock-mhz)

  busy span            first cycle the device starts running -> last cycle it stops. A window, so the
                       idle between dispatches is still inside it.
  critical path busy   busy span minus the idle:  sum over dispatches of ( max over its cores ). The
                       host dispatches sequentially, dispatches do not overlap, and within a dispatch
                       each participating core runs one kernel — so this is the critical path.

  GUARANTEED ORDERINGS (the tool checks these)

      wall time  >=  busy span
      wall time  >=  host device busy  >=  critical path busy

  host device busy vs busy span is NOT ordered in general: the host figure drops the inter-segment
  stalls but adds its own per-call overhead. Which is larger tells you which dominates.

  All four must come from ONE execution or the orderings can appear violated — profiling slows the
  device down. exp_tb001's HE_PROF2 mode measures its own profiled replay for exactly this reason.

THE 1-1 REQUIREMENT, AND WHY WINDOWING IS NEEDED

  Every host dispatch must appear in the trace exactly once, with all of its RISCs. Two things get
  in the way, and this tool handles both:

  1. CONTAMINATION. ReadMeshDeviceProfilerResults rewrites the CSV from the host's full accumulated
     marker history on every call, so deleting the file beforehand achieves nothing — a capture holds
     setup and warm-up dispatches as well as the one multiply (measured: 1576 dispatches present for
     a 636-dispatch multiply). The profiled multiply runs LAST, so this tool windows to the final N
     kernel-bearing dispatches.
  2. DROPPED MARKERS. Each RISC's markers go to a DRAM buffer that can overflow, at which point
     tt-metal warns "Profiler DRAM buffers were full, markers were dropped! ... Please either
     decrease the number of ops being profiled or run read device profiler more often". A dispatch
     that lost its TRISC markers still shows BRISC/NCRISC, so it masquerades as a dispatch with no
     compute. This tool refuses to report if any windowed dispatch is missing RISCs.

     Those no-TRISC groups ARE dropped markers, not buffer transfers: host<->device copies are done by
     the dispatch infrastructure and emit no worker-core kernel envelope at all, so they never appear
     as a group. Evidence: a raw drain capture (which logs drop warnings) has 258 of them, while a
     clean 636-dispatch window has ZERO -- yet both do exactly the same 32 uploads and 14 readbacks.

     Measured remedies: draining more often (HE_DRAIN_K=20) did NOT clear the warnings, and
     TT_METAL_PROFILER_ACCUMULATE=1 made them worse. What does hold is that the drops land in the
     setup/warm-up phase, so the final N dispatches come out complete — verified 636/636 with all
     5 RISCs. Hence: window first, verify, and only then report.
"""
import csv, sys, collections


def load(path):
    """-> sorted [(start_cyc, end_cyc, core, risc)] for kernel-execution envelopes. CYCLES, not ns."""
    st, out = collections.defaultdict(list), []
    with open(path) as f:
        f.readline()
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        ix = {n: hdr.index(n) for n in
              ("core_x", "core_y", "RISC processor type", "time[cycles since reset]", "zone name", "type")}
        for x in r:
            if len(x) < len(hdr):
                continue
            if not x[ix["zone name"]].strip().endswith("-KERNEL"):
                continue
            core = (x[ix["core_x"]].strip(), x[ix["core_y"]].strip())
            risc = x[ix["RISC processor type"]].strip()
            t = int(x[ix["time[cycles since reset]"]])
            k = (core, risc)
            if x[ix["type"]].strip() == "ZONE_START":
                st[k].append(t)
            elif st[k]:
                out.append((st[k].pop(), t, core, risc))
    out.sort()
    return out


def dispatch_groups(ev):
    """Overlap-connected groups. Dispatches never overlap, so one group == one dispatch."""
    out, cur, end = [], [ev[0]], ev[0][1]
    for e in ev[1:]:
        if e[0] <= end:
            cur.append(e); end = max(end, e[1])
        else:
            out.append(cur); cur = [e]; end = e[1]
    out.append(cur)
    return out


def main():
    args, pos, expect, wall, hostbusy, clk = sys.argv[1:], [], None, None, None, 1350.0
    dataflow_ok = False
    drain = False
    while args:
        a = args.pop(0)
        if a == "--expect":
            expect = int(args.pop(0))
        elif a == "--dataflow-ok":
            dataflow_ok = True
        elif a == "--drain":
            drain = True
        elif a == "--wall-ms":
            wall = float(args.pop(0))
        elif a == "--host-busy-ms":
            hostbusy = float(args.pop(0))
        elif a == "--clock-mhz":
            clk = float(args.pop(0))
        else:
            pos.append(a)
    if not pos or expect is None:
        print(__doc__)
        return 2

    ev = load(pos[0])
    if not ev:
        print("no kernel envelopes in", pos[0])
        return 1
    groups = dispatch_groups(ev)
    # A dispatch that runs NO compute kernel emits no TRISC envelope, so the default filter drops it
    # STRUCTURALLY -- not because markers were lost. exp_Tg018's corner turn is dataflow-only, and its
    # 608 dispatches are counted in --expect, so the 1-1 map could never close without this.
    # Opt-in, so every existing caller keeps the stricter rule.
    if dataflow_ok:
        kern = [g for g in groups if any(e[3].startswith("TRISC") for e in g)
                or all(not e[3].startswith("TRISC") for e in g)]
    else:
        kern = [g for g in groups if any(e[3].startswith("TRISC") for e in g)]

    if len(kern) < expect:
        print(f"\n  1-1 map : FAIL — capture has {len(kern)} kernel dispatches, fewer than the {expect} "
              f"the host issued.\n            Markers were dropped; see this file's header.\n")
        return 1
    win = kern[-expect:]                                  # the profiled multiply runs last
    # 5 RISCs is the completeness rule for a reader+compute+writer dispatch. A dataflow-only dispatch
    # legitimately has fewer, so it is checked for >=1 instead of ==5.
    if dataflow_ok:
        incomplete = [g for g in win
                      if (len({e[3] for e in g}) != 5 and any(e[3].startswith("TRISC") for e in g))]
    else:
        incomplete = [g for g in win if len({e[3] for e in g}) != 5]

    print()
    print(f"  1-1 map : host dispatches {expect}   trace dispatches {len(win)}"
          f"   (windowed from {len(kern)} in the capture)")
    if incomplete:
        print(f"  1-1 map : FAIL — {len(incomplete)} of {expect} dispatches are missing RISC markers.")
        print( "            The capture is incomplete, so the metrics would understate. Profile fewer")
        print( "            markers or drain more often, then re-capture.\n")
        return 1

    # CONTIGUITY. Having N dispatches each with 5 RISCs is NOT enough: if the profiled run itself lost
    # markers, the "last N" reaches back into an earlier phase and the window straddles a boundary.
    # Measured case: at G=8 only 750 of 1266 dispatches survived, so the window pulled in 516 from a
    # REC pass 77 SECONDS earlier -- every per-dispatch check passed and the busy span came out
    # 77,979 ms. Detect it by scale, not by ratio: legitimate inter-segment stalls are single-digit ms
    # (already ~4,500x the ~0.6 us median gap), whereas a phase boundary is seconds. The host wall is
    # the natural yardstick -- the device cannot be busy for materially longer than the host waited.
    ws = [min(e[0] for e in gp) for gp in win]
    we = [max(e[1] for e in gp) for gp in win]
    win_span_ms = (we[-1] - ws[0]) / (clk * 1e3)
    gaps = sorted(ws[i + 1] - we[i] for i in range(len(win) - 1))
    biggest_ms = gaps[-1] / (clk * 1e3) if gaps else 0.0
    limit_ms = wall * 1.5 if wall is not None else 1000.0     # 1 s fallback when no wall is given
    # --drain: a DRAINED capture interleaves Finish + profiler reads with the dispatches, so the window
    # is deliberately non-contiguous and its span is meaningless. `critical path busy` is a SUM of
    # per-dispatch maxes and so is unaffected by what happens between dispatches -- it stays valid, and
    # is the only reason to run a drained capture. `busy span` does NOT survive and is suppressed.
    if drain:
        print(f"  1-1 map : OK — all {expect} dispatches present, RISCs complete")
        print(f"  1-1 map : DRAINED capture — window spans {win_span_ms:,.1f} ms with drain pauses"
              f" (largest {biggest_ms:,.1f} ms).\n            busy span is therefore NOT reported;"
              f" critical path busy is a sum and is unaffected.\n")
    elif win_span_ms > limit_ms:
        print(f"  1-1 map : FAIL — the window is not contiguous. It spans {win_span_ms:,.1f} ms against a")
        print(f"            {'wall of %.3f ms' % wall if wall is not None else ' 1 s sanity limit'}, "
              f"with a single internal gap of {biggest_ms:,.1f} ms.")
        print( "            The profiled run lost markers, so the window reached back into an earlier")
        print( "            pass and these numbers would mix two executions.")
        print( "            Use HE_PROF_DRAIN (keeps every marker) for critical path busy, which is a sum")
        print( "            and so idle-insensitive; a busy span needs an undropped single pass.\n")
        return 1
    else:
        print(f"  1-1 map : OK — all {expect} dispatches present, RISCs complete, contiguous\n")

    span = max(e[1] for g in win for e in g) - min(e[0] for g in win for e in g)
    crit = 0
    for g in win:
        per = collections.defaultdict(lambda: [1 << 62, 0])
        for s_, e_, c, _r in g:
            per[c][0] = min(per[c][0], s_)
            per[c][1] = max(per[c][1], e_)
        crit += max(hi - lo for lo, hi in per.values())

    cyc2ms = 1.0 / (clk * 1e3)          # cycles -> ms at the real device clock
    # Printed largest-to-smallest, which is also the guaranteed ordering. The source column matters:
    # the two host figures are immune to the device-timebase issue, the two trace figures are not.
    if wall is not None:
        print(f"  wall time             {wall:10.3f} ms   host")
    else:
        print( "  wall time                    n/a       host  — pass --wall-ms")
    if drain:
        print( "  busy span                    n/a       trace — invalid for a drained capture")
    else:
        print(f"  busy span             {span*cyc2ms:10.3f} ms   trace")
    if hostbusy is not None:
        print(f"  host device busy      {hostbusy:10.3f} ms   host")
    else:
        print( "  host device busy             n/a       host  — pass --host-busy-ms")
    print(f"  critical path busy    {crit*cyc2ms:10.3f} ms   trace")
    print(f"  (device clock {clk:.0f} MHz; the CSV is in cycles — see this file's header)")
    print()
    # Device timestamps are converted at 1 cycle = 1 ns. That is an approximation of the actual
    # device clock, so busy span can land a fraction of a percent above a correctly-measured wall.
    # Only flag a real ordering violation, not timebase rounding.
    TOL = 0.03
    bad = []
    if not drain and wall is not None and span * cyc2ms > wall * (1 + TOL):
        bad.append("wall >= busy span")
    if wall is not None and hostbusy is not None and hostbusy > wall * (1 + TOL):
        bad.append("wall >= host device busy")
    if hostbusy is not None and crit * cyc2ms > hostbusy * (1 + TOL):
        bad.append("host device busy >= critical path busy")
    if bad:
        print("  WARNING: violated beyond the " + f"{100*TOL:.0f}% timebase tolerance: " + "; ".join(bad))
        print("           The values are probably not all from one execution, or --clock-mhz is wrong.\n")
    elif wall is not None and span * cyc2ms > wall:
        print(f"  note: busy span exceeds wall by {100*(span*cyc2ms/wall-1):.1f}% — within the "
              f"{100*TOL:.0f}% tolerance.\n")
    if hostbusy is not None:
        d = hostbusy - crit * cyc2ms
        print(f"  host device busy - critical path busy = {d:+.3f} ms  "
              f"(intra-segment device idle + ~52 us/call blocking overhead)")
        if span * cyc2ms > hostbusy:
            print(f"  busy span - host device busy         = {span*cyc2ms-hostbusy:+.3f} ms  "
                  f"(host stalls BETWEEN segments — the orchestration target)\n")
    return 0


sys.exit(main())
