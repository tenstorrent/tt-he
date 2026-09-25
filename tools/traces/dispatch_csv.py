#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""dispatch_csv.py — one row per program dispatch, for optimisation deep-dives.

  Usage: dispatch_csv.py <csv> --expect N [--labels FILE] [--clock-mhz F] [--out FILE] [--drain]

Emits, in dispatch execution order:

  seq                        0-based program-dispatch index within the multiply
  kernel                     the compute kernel that dispatch ran
  trace_kernel_latency_us    that dispatch's max-over-cores, in microseconds

With `--labels FILE` (the file exp_tb001 writes under TB001_LABELS), three more, joined on seq:

  he_op                      rescale | tensor | keyswitch | combine
  he_phase                   e.g. rescale.ntt, ks.modup_fbc, ks.moddown_tail, tensor.pointwise
  operand                    which instance: c0/c1/d0/d1, digit group j0/j1, acc0/acc1, limb

The join is verified: the host records the kernel it dispatched, and every row's host kernel must
agree with the kernel the trace observed. A mismatch means the two are not the same execution.

`--expect N` is the host's own dispatch count (exp_tb001 prints it). The same guards as
busy_metrics.py apply, because a CSV built from a bad window is worse than no CSV:

  * window to the LAST N kernel-bearing dispatches — a capture also holds setup and warm-up passes,
    since ReadMeshDeviceProfilerResults rewrites the file from the full accumulated marker history;
  * every windowed dispatch must carry all 5 RISCs, else markers were dropped;
  * the window must be temporally contiguous, else it straddles a phase boundary.

`--drain` for a HE_PROF_DRAIN capture: skips the contiguity check, because that mode deliberately
stalls to drain the profiler and those stalls look exactly like a phase boundary. It is safe for THIS
tool -- every value here is a per-dispatch DURATION (max-over-cores), and a duration is immune to idle
inserted around it. The window is instead validated by the per-kernel histogram, which is printed: if
the window had straddled, those counts would not match the host's own (exp_tb001's TB001_KLAT prints
them). Do NOT use a drained capture for a busy span, which is idle-sensitive.

Timebase: the CSV column is "time[cycles since reset]" -- device CYCLES at AICLK, 1350 MHz on this
p150 under load. Do not assume 1 GHz. See docs/he/exp_tb001.md "Device timebase".
"""
import csv, sys, collections, bisect

SKIP = {"ph_recombine", "ph_matmul", "ph_decompose", "ph_reduce", "ph_twiddle",
        "fbc_phase1", "fbc_phase2"}          # inner sub-zones, not a dispatch's identity


def load(path):
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


def main():
    args, pos, expect, clk, out, drain, lblf = sys.argv[1:], [], None, 1350.0, None, False, None
    dataflow_ok = False
    while args:
        a = args.pop(0)
        if a == "--expect":
            expect = int(args.pop(0))
        elif a == "--clock-mhz":
            clk = float(args.pop(0))
        elif a == "--out":
            out = args.pop(0)
        elif a == "--drain":
            drain = True
        elif a == "--dataflow-ok":
            dataflow_ok = True
        elif a == "--labels":
            lblf = args.pop(0)
        else:
            pos.append(a)
    if not pos or expect is None:
        print(__doc__)
        return 2

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
    # --dataflow-ok: keep dispatches that run no compute kernel (no TRISC envelope). They carry no
    # named zone either, so the host label is the only name available -- accepted rather than inferred,
    # since there is nothing in the trace for it to contradict. Opt-in; default rule unchanged.
    if dataflow_ok:
        kern = [g for g in groups if any(e[3].startswith("TRISC") for e in g)
                or all(not e[3].startswith("TRISC") for e in g)]
    else:
        kern = [g for g in groups if any(e[3].startswith("TRISC") for e in g)]

    if len(kern) < expect:
        print(f"FAIL: capture has {len(kern)} kernel dispatches, fewer than the {expect} the host issued "
              f"— markers were dropped.")
        return 1
    win = kern[-expect:]
    # 5 RISCs is the completeness rule for a reader+compute+writer dispatch; a dataflow-only one
    # legitimately has fewer, so under --dataflow-ok it is exempted rather than failed.
    if dataflow_ok:
        bad = [g for g in win
               if len({e[3] for e in g}) != 5 and any(e[3].startswith("TRISC") for e in g)]
    else:
        bad = [g for g in win if len({e[3] for e in g}) != 5]
    if bad:
        print(f"FAIL: {len(bad)} of {expect} windowed dispatches are missing RISC markers.")
        return 1
    ws = [min(e[0] for e in g) for g in win]
    we = [max(e[1] for e in g) for g in win]
    gaps = sorted(ws[i + 1] - we[i] for i in range(len(win) - 1))
    span_ms = (we[-1] - ws[0]) / (clk * 1e3)
    if not drain and gaps and gaps[-1] / (clk * 1e3) > 1000.0:   # a >1 s internal gap is a boundary
        print(f"FAIL: window not contiguous — largest internal gap {gaps[-1]/(clk*1e3):,.1f} ms, "
              f"span {span_ms:,.1f} ms.  (--drain if this is a HE_PROF_DRAIN capture)")
        return 1

    bounds = list(zip(ws, we))
    kof = {}
    for s0, _e0, _c, z in named:
        i = bisect.bisect_right(ws, s0) - 1
        if 0 <= i < len(win) and s0 <= bounds[i][1]:
            kof.setdefault(i, set()).add(z)

    rows = []
    for i, g in enumerate(win):
        per = collections.defaultdict(lambda: [1 << 62, 0])
        for s0, e0, c, _r in g:
            per[c][0] = min(per[c][0], s0)
            per[c][1] = max(per[c][1], e0)
        lat_us = max(hi - lo for lo, hi in per.values()) / clk
        ks = sorted(kof.get(i, ()))
        rows.append((i, "+".join(ks) if ks else "unlabelled", lat_us))

    # ---- optional HE-poly-op labels, joined on seq ----
    ZONE2HOST = {"fused_total": "fused_matmul_a", "fusedC_total": "fused_matmul_c",
                 "fusedAcc_total": "fused_matmul_a_acc", "fusedCacc_total": "fused_matmul_c_acc",
                 "fbc_total": "baseconv_fbc"}
    lbl = {}
    if lblf:
        with open(lblf) as lf:
            for d in csv.DictReader(lf):
                lbl[int(d["seq"].strip())] = {k.strip(): v.strip() for k, v in d.items()}
        if len(lbl) != len(rows):
            print(f"FAIL: labels file has {len(lbl)} rows, trace window has {len(rows)}.")
            return 1
        bad = 0
        for i, k, _lat in rows:
            hk = lbl[i]["host_kernel"]
            if dataflow_ok and k == "unlabelled":
                continue                      # dataflow-only: no trace zone to agree or disagree with
            if ZONE2HOST.get(k, k) != hk:
                bad += 1
        if bad:
            print(f"FAIL: {bad} of {len(rows)} rows disagree on the kernel between host and trace — "
                  f"not the same execution.")
            return 1
        print(f"  join verified: all {len(rows)} rows agree on the kernel between host and trace")
        # A dataflow-only dispatch has no trace zone, so the host label is the only name it can carry.
        if dataflow_ok:
            rows = [(i, lbl[i]["host_kernel"] if k == "unlabelled" else k, lat) for i, k, lat in rows]

    dest = open(out, "w", newline="") if out else sys.stdout
    w = csv.writer(dest)
    head = ["seq", "kernel", "trace_kernel_latency_us"]
    if lblf:
        head += ["he_op", "he_phase", "operand"]
    w.writerow(head)
    for i, k, lat in rows:
        r = [i, k, f"{lat:.3f}"]
        if lblf:
            r += [lbl[i]["he_op"], lbl[i]["he_phase"], lbl[i]["operand"]]
        w.writerow(r)
    if out:
        dest.close()
        tot = sum(r[2] for r in rows)
        print(f"wrote {out}: {len(rows)} dispatches, "
              f"critical path busy = {tot/1000:.3f} ms @ {clk:.0f} MHz"
              + ("   [--drain: contiguity check skipped]" if drain else ""))
        hist = collections.Counter(r[1] for r in rows)
        cost = collections.Counter()
        for _i, k, l in rows:
            cost[k] += l
        print("  per-kernel histogram — cross-check against the host's own dispatch counts:")
        for k, n in hist.most_common():
            print(f"    {n:>5}  {k:<20} {cost[k]/1000:>8.3f} ms  {100*cost[k]/tot:>5.1f}%")
    return 0


sys.exit(main())
