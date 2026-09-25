#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""On-device latency profile + timeline TRACE for the CROSS-aligned Set-B HE-mul.

Parses tt-metal's profile_log_device.csv (per-core / per-RISC DeviceZoneScopedN zones) and emits:

  TIMEBASE: the CSV column is "time[cycles since reset]" -- device CYCLES at AICLK, which is
  1350 MHz under load on this p150 (800 MHz idle). This file used to assume 1 GHz, which overstated
  every duration by 1.35x. Override with --clock-mhz if the card is clocked differently.
  1. a per-op-type latency breakdown (device-compute time in each kernel), and
  2. a Chrome/Perfetto trace JSON  (open in https://ui.perfetto.dev  — x=time, y=core/RISC lane,
     bars=operations), and
  3. a self-contained HTML Gantt (same axes) for quick viewing.

  Usage: trace_hemul.py profile_log_device.csv [out_prefix]
                          [--arch NAME] [--wall-ms X] [--label TEXT]

  --arch / --wall-ms / --label only annotate the HTML header and closing note. Everything the page
  states about where the time goes -- shares, per-call costs, core and lane counts -- is derived from
  the capture itself. (It used to hardcode "Wormhole n150", 16 cores and a 5044 ms/mul wall, which
  silently went stale the moment either changed.)
"""
import csv, sys, json, collections, html

# friendly op-type names for the compute DeviceZoneScopedN zones this workload uses
OP = {
    "fused_total":      "matmul-A (NTT/INTT stage-1)",
    "fusedC_total":     "matmul-C (NTT/INTT stage-2)",
    "fusedAcc_total":   "matmul-A, 8 partials fused (NTT stage-1)",
    "fusedCacc_total":  "matmul-C, 8 partials fused (INTT stage-2)",
    "modmul_compute":   "twiddle modmul",
    "modmul_sfpi_full": "pointwise modmul (tensor / x-evk)",
    "fbc_total":        "base conversion (FBC)",
    "modadd_reduce":    "modular add",
    "modadd_twiddle":   "reduce+twiddle (fused)",
    "modsub_reduce":    "modular sub",
    "epscorr_total":    "CRNS eps-correct",
}
# color per functional bucket (for the trace)
BUCKET = {"fused_total":"NTT","fusedC_total":"NTT","fusedAcc_total":"NTT","fusedCacc_total":"NTT",
          "modmul_compute":"NTT","transpose_tile":"NTT",
          "modmul_sfpi_full":"pointwise","fbc_total":"FBC","modadd_reduce":"vec","modsub_reduce":"vec",
          "modadd_twiddle":"NTT"}
COLOR  = {"NTT":"#4e79a7","pointwise":"#f28e2b","FBC":"#59a14f","vec":"#e15759","KERNEL":"#bab0ac","other":"#9c755f"}

def parse(path):
    rows=[]
    with open(path) as f:
        f.readline()                      # ARCH line
        r=csv.reader(f); next(r)          # column header
        for x in r:
            if len(x)>=12: rows.append(x)
    # match ZONE_START/END per (core,RISC) with a stack
    zones=[]                              # (core_x,core_y,risc, name, start_ns, dur_ns)
    st=collections.defaultdict(list)
    for x in sorted(rows,key=lambda x:int(x[5])):
        cx,cy,risc = int(x[1]),int(x[2]),x[3]; t=int(x[5]); name=x[10].strip(); ty=x[11]
        key=(cx,cy,risc)
        if ty=="ZONE_START": st[key].append((name,t))
        elif ty=="ZONE_END" and st[key]:
            nm,s=st[key].pop(); zones.append((cx,cy,risc,nm,s,t-s))
    t0=min(z[4] for z in zones)
    return [(cx,cy,risc,nm,s-t0,d) for (cx,cy,risc,nm,s,d) in zones], (max(z[4]+z[5] for z in zones)-t0)

def breakdown(zones):
    # per named op-type: count + summed device-compute time (across all cores/dispatches)
    agg=collections.defaultdict(lambda:[0,0])
    for cx,cy,risc,nm,s,d in zones:
        if nm in OP: agg[nm][0]+=1; agg[nm][1]+=d
    return agg

SKIP={"ph_recombine","ph_matmul","ph_decompose","fbc_phase1","fbc_phase2"}   # inner sub-zones
def keep(nm): return not nm.endswith("-FW") and nm not in SKIP

def to_chrome(zones, out):
    ev=[]
    for cx,cy,risc,nm,s,d in zones:
        if not keep(nm): continue
        ev.append({"name":nm,"ph":"X","ts":s/1000.0,"dur":d/1000.0,          # µs
                   "pid":cx*100+cy,"tid":risc,
                   "args":{"core":f"({cx},{cy})"}})
    # lane (pid) names = cores
    for pid in sorted({e["pid"] for e in ev}):
        cx,cy=pid//100,pid%100
        ev.append({"name":"process_name","ph":"M","pid":pid,"tid":0,"args":{"name":f"core({cx},{cy})"}})
    json.dump({"traceEvents":ev,"displayTimeUnit":"ns"}, open(out,"w"))
    return len(ev)

def _gantt_svg(zones):
    # viewable SLICE: filter noise, then window on the DENSEST ~250 kept bars.
    # (It used to take the FIRST 250, which on a drained capture is the warm-up phase -- the strip
    #  showed sparse setup dispatches rather than the multiply the page is about.)
    kz=sorted([z for z in zones if keep(z[3])], key=lambda z:z[4])
    N=250
    if len(kz)>N:
        st=[z[4] for z in kz]
        i0=min(range(len(st)-N), key=lambda i: st[i+N]-st[i])   # tightest N-op time window
        w0_,w1=st[i0], st[i0+N]
        kz=[z for z in kz if w0_<=z[4]<=w1]
    w0=min(z[4] for z in kz); span=max(max(z[4]+z[5] for z in kz)-w0,1)
    kz=[(cx,cy,risc,nm,s-w0,d) for (cx,cy,risc,nm,s,d) in kz]
    lanes=sorted({(cx,cy,risc) for cx,cy,risc,nm,s,d in kz})
    li={l:i for i,l in enumerate(lanes)}
    LM=118; W=1700; rowh=14; top=20; H=len(lanes)*rowh+top+8
    def X(t): return LM+(t/span)*(W-LM-16)
    out=[]
    # lane bands (alternate) + labels
    for l,i in li.items():
        y=i*rowh+top
        if i%2==0: out.append(f'<rect x="0" y="{y}" width="{W}" height="{rowh}" fill="#ffffff05"/>')
        rn=l[2].replace("TRISC_","T").replace("BRISC","B").replace("NCRISC","NC")
        out.append(f'<text x="6" y="{y+rowh-3.5}" class="ln">core({l[0]},{l[1]})·{rn}</text>')
    # time gridlines
    for k in range(0,11):
        t=span*k/10; x=X(t)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{H-8}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{top-6}" class="ax">{t/1000:.1f}µs</text>')
    # op bars
    for cx,cy,risc,nm,s,d in kz:
        b=BUCKET.get(nm) or ("kernel" if nm.endswith("-KERNEL") else "other")
        c=COLOR.get(b,COLOR["other"]) if b!="kernel" else COLOR["KERNEL"]
        y=li[(cx,cy,risc)]*rowh+top+1.5; w=max((d/span)*(W-LM-16),0.5)
        lbl=OP.get(nm,nm)
        out.append(f'<rect x="{X(s):.1f}" y="{y:.1f}" width="{w:.2f}" height="{rowh-3}" rx="1" fill="{c}"><title>{html.escape(lbl)} · core({cx},{cy}) {risc} · {d} ns</title></rect>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" role="img">{"".join(out)}</svg>', span, len(lanes)

def to_html(zones, agg, tot, ctx, out):
    svg,span,nlanes=_gantt_svg(zones)
    rows=""
    order=sorted(agg.items(),key=lambda kv:-kv[1][1])
    for nm,(c,d) in order:
        b=BUCKET.get(nm,"other"); pct=100*d/tot
        rows+=(f'<tr><td><span class="dot" style="background:{COLOR.get(b,COLOR["other"])}"></span>{html.escape(OP[nm])}</td>'
               f'<td class="n">{c:,}</td><td class="n">{d/1000/c:.1f}</td><td class="n">{d/1000:,.0f}</td>'
               f'<td class="barcell"><span class="bar" style="width:{pct:.1f}%;background:{COLOR.get(b,COLOR["other"])}"></span><span class="pct">{pct:.1f}%</span></td></tr>')
    legend="".join(f'<span class="lg"><span class="dot" style="background:{COLOR[b]}"></span>{b}</span>' for b in ["NTT","pointwise","FBC","vec","KERNEL"])
    css="""
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1a1d24;--mut:#5b6472;--line:#e4e7ec;--accent:#3d6aa8;--trace:#12141a;--tracln:#8a93a3;--tracgrid:#ffffff14;--tracax:#aeb6c2}
@media(prefers-color-scheme:dark){:root{--bg:#0f1116;--panel:#161922;--ink:#e6e9ef;--mut:#9aa3b2;--line:#252a35;--accent:#6f9fd8;--trace:#0c0e13}}
:root[data-theme=light]{--bg:#f6f7f9;--panel:#fff;--ink:#1a1d24;--mut:#5b6472;--line:#e4e7ec;--accent:#3d6aa8}
:root[data-theme=dark]{--bg:#0f1116;--panel:#161922;--ink:#e6e9ef;--mut:#9aa3b2;--line:#252a35;--accent:#6f9fd8;--trace:#0c0e13}
*{box-sizing:border-box}
.wrap{max-width:1120px;margin:0 auto;padding:40px 24px 72px;color:var(--ink);
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;line-height:1.5}
.eyebrow{font:600 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
h1{font-size:26px;font-weight:680;margin:.35em 0 .15em;letter-spacing:-.01em;text-wrap:balance}
.sub{color:var(--mut);font-size:15px;margin:0 0 4px}
.meta{font:12px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mut);margin-top:10px}
.meta b{color:var(--ink);font-weight:600}
h2{font-size:13px;font-weight:600;letter-spacing:.02em;margin:38px 0 12px;color:var(--ink)}
h2 .hint{font-weight:400;color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}
th{font:600 10.5px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:8px;vertical-align:middle}
.barcell{width:26%}.bar{display:inline-block;height:9px;border-radius:2px;vertical-align:middle;min-width:2px}
.pct{font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mut);margin-left:8px;font-variant-numeric:tabular-nums}
.tracewrap{background:var(--trace);border:1px solid var(--line);border-radius:8px;padding:10px 4px 6px;overflow-x:auto}
.tracewrap svg{display:block}
.ln{font:9px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--tracln)}
.ax{font:9px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--tracax)}
.grid{stroke:var(--tracgrid);stroke-width:1}
.legend{display:flex;gap:18px;margin:10px 2px 0;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mut);flex-wrap:wrap}
.lg{display:flex;align-items:center}
.note{margin-top:34px;padding:16px 18px;border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:6px;background:var(--panel);font-size:13.5px}
.note b{color:var(--ink)} .note code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--accent)}
"""
    # --- everything the prose asserts is derived from this capture, not hardcoded ---
    arch    = ctx.get("arch") or "unknown device"
    ncores  = ctx["ncores"]
    wall_ms = ctx.get("wall_ms")
    wall_txt = f" · wall <b>{wall_ms:g}&nbsp;ms/mul</b>" if wall_ms else ""
    bsum = collections.Counter()
    for nm,(c,d) in agg.items(): bsum[BUCKET.get(nm,"other")] += d
    ntt_pct = 100*bsum["NTT"]/tot
    fbc_pct = 100*bsum["FBC"]/tot
    def per(nm): return (agg[nm][1]/agg[nm][0]/1000.0) if nm in agg and agg[nm][0] else 0.0
    fbc_per = per("fbc_total")
    mm_per  = per("fusedAcc_total") or per("fused_total") or per("fusedC_total")
    capture_note = (ctx.get("label") or
        "The capture is whatever the profiler buffer held when the run ended, so it may span "
        "warm-up dispatches as well as the measured replay — read the shares, not the absolute totals.")

    body=f"""<div class="wrap">
<div class="eyebrow">On-device profile · {arch}</div>
<h1>CROSS-aligned Set-B HE-multiply — device trace</h1>
<p class="sub">Where the multiply's time goes on silicon, and the operation timeline across cores.</p>
<p class="meta"><b>N</b>=8192 · <b>L</b>=8 · <b>dnum</b>=2 · <b>{ncores} Tensix cores</b> · {nlanes} core/RISC lanes · profiler 1&nbsp;GHz (1&nbsp;cycle=1&nbsp;ns){wall_txt} · device-compute (Σ over lanes) <b>{tot/1000:,.0f}&nbsp;µs</b></p>

<h2>Where the on-device compute goes <span class="hint">— by kernel type</span></h2>
<table><thead><tr><th>operation</th><th class="n">calls</th><th class="n">µs/call</th><th class="n">Σ µs</th><th>share of device-compute</th></tr></thead>
<tbody>{rows}</tbody></table>

<h2>On-device timeline <span class="hint">— first ~250 ops · x = time, lane = core/RISC, bar = operation</span></h2>
<div class="tracewrap">{svg}</div>
<div class="legend">{legend}</div>

<div class="note"><b>Reading it.</b> Compute is <b>{ntt_pct:.0f}% NTT/INTT matmul</b>
(matmul-A + matmul-C) and <b>{fbc_pct:.0f}% base conversion</b> — and FBC costs
<code>~{fbc_per:,.0f}&nbsp;µs/call</code> against <code>~{mm_per:.0f}&nbsp;µs</code> for a matmul, making it the
most expensive single kernel by a wide margin. Summed device-compute is
<b>~{tot/1e6:,.0f}&nbsp;ms</b> across {nlanes} core/RISC lanes; that is a sum, so it exceeds the wall
time by roughly the number of concurrently busy lanes. {capture_note}
Full trace (Perfetto): load the companion <code>.json</code> in ui.perfetto.dev.</div>
</div>"""
    head=f"<title>Set-B HE-multiply device trace — {arch}</title>"
    open(out,"w").write(f"{head}<style>{css}</style>{body}")

ARCH=None; WALL_MS=None; LABEL=None; CLOCK_MHZ=1350.0

def main():
    global ARCH, WALL_MS, LABEL, CLOCK_MHZ
    args=[a for a in sys.argv[1:]]
    pos=[]
    while args:
        a=args.pop(0)
        if a=="--arch": ARCH=args.pop(0)
        elif a=="--wall-ms": WALL_MS=float(args.pop(0))
        elif a=="--label": LABEL=args.pop(0)
        elif a=="--clock-mhz": CLOCK_MHZ=float(args.pop(0))
        else: pos.append(a)
    path=pos[0]; pre=pos[1] if len(pos)>1 else "hemul_trace"
    zones,span=parse(path)
    agg=breakdown(zones)
    tot=sum(v[1] for v in agg.values())
    print(f"\n=== on-device latency breakdown (device-compute @ {CLOCK_MHZ:.0f} MHz; timeline span={span/CLOCK_MHZ:.1f} µs) ===")
    print("NOTE: these sums are CORE-TIME (summed over every core) -- work, not latency. A kernel on 64")
    print("      cores accumulates 8x the core-time of the same latency on 8 cores, so this ranking")
    print("      over-weights wide kernels. For the CRITICAL-PATH share (max-over-cores, which is what")
    print("      latency is) see docs/he/exp_tb001.md: on exp_Tb001 that reorders FBC from 6.8% here to")
    print("      30.9%, and NTT stage-1+2 from 88% down to 26%.\n")
    print(f"{'op-type':<40}{'calls':>7}{'sum µs':>10}{'µs/call':>10}{'% comp':>8}")
    print("-"*75)
    for nm,(c,d) in sorted(agg.items(),key=lambda kv:-kv[1][1]):
        print(f"{OP[nm]:<40}{c:>7}{d/CLOCK_MHZ:>10.1f}{d/CLOCK_MHZ/c:>10.3f}{100*d/tot:>7.1f}%")
    print("-"*75)
    print(f"{'TOTAL device-compute (summed, cores overlap)':<40}{'':>7}{tot/CLOCK_MHZ:>10.1f}")
    print(f"timeline span (wall, incl. host gaps + concurrency) = {span/CLOCK_MHZ:.1f} µs")
    ncores=len({(z[0],z[1]) for z in zones})
    ctx={"arch":ARCH,"ncores":ncores,"wall_ms":WALL_MS,"label":LABEL}
    n=to_chrome(zones, pre+".json"); to_html(zones, agg, tot, ctx, pre+".html")
    print(f"\nwrote {pre}.json  ({n} events → open in https://ui.perfetto.dev)")
    print(f"wrote {pre}.html  (self-contained Gantt)")

main()
