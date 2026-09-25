#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""trace_app2.py — interactive device-trace app, in the mould of compiler/viz.py.

Parses the device profiler CSV into a data model and SERVES an interactive canvas Gantt:
  * y = one lane per (core, RISC),  x = time (us),  bars = each RISC's activity
  * bars colored by OP; lane labels colored by RISC ROLE (rd/T0/T1/T2/wr)
  * hover a UNIT NAME (left gutter) -> what that RISC does;  click a bar -> op detail
  * search / legend / pan / zoom
Run:  python3 trace_app2.py <profile_log_device.csv> [PORT] [lo_ms hi_ms] [--expect N]
then forward PORT and open http://localhost:PORT/

  --expect N  window to the last N program dispatches (N = the host's own dispatch count, which
              exp_tb001 prints). REQUIRED for a HE_PROF_DRAIN capture: the default pick_window()
              splits on >5 ms gaps, and a drained capture stalls ~23 ms every HE_DRAIN_K dispatches,
              so it would shatter one multiply into ~40 clusters and show you a fortieth of it.

Idle between dispatches is already compressed (THRESH/STUB below) -- gaps over 30 us collapse to an
8 us stub, and the header reports how much was removed. Without that a G=8 drained capture is 97%
dead space.
"""
import sys, os, csv, json, collections, http.server, socketserver, functools

PORT_DEFAULT = 8747

# op zone-name -> (category, color)   [TRISC named compute zones]
NAMED = {
    "fused_total": ("matmul", "#f0883e"), "fusedC_total": ("matmul", "#f0883e"),
    # the FUSED block contraction (compute/fused_matmul_{a,c}_acc.cpp) -- its own colour, because it is
    # 1 dispatch where the plain kernels above are 8, and the Gantt should make that obvious
    "fusedAcc_total": ("matmul (8-fused)", "#e3b341"), "fusedCacc_total": ("matmul (8-fused)", "#e3b341"),
    "modmul_compute": ("twiddle", "#a371f7"),
    "modmul_sfpi_full": ("pointwise", "#3fb950"), "tensor2_fused": ("pointwise", "#3fb950"),
    "baseconv_fbc": ("FBC", "#db61a2"), "fbc_total": ("FBC", "#db61a2"),
    "bconv_one": ("FBC (Ls=1)", "#f778ba"),
    "transpose_tile": ("transpose", "#39c5cf"),
    "modadd_reduce": ("vec", "#59a14f"), "modadd_map": ("vec", "#59a14f"), "modsub_reduce": ("vec", "#59a14f"),
    # fused reduce+twiddle (compute/modadd_twiddle.cpp) -- its own colour so the merge is visible in
    # the timeline: where the forward NTT used to show a vec bar followed by a twiddle bar with a DRAM
    # round trip between them, there is now one bar.
    "modadd_twiddle": ("reduce+tw (fused)", "#d2a8ff"),
}
KWRAP = {"BRISC": ("reader", "#58a6ff"), "NCRISC": ("writer", "#e3b341")}
# RISC -> (short lane tag, bright label color, what-it-does explanation)
RISC_INFO = {
    "BRISC":   ("rd",      "#58a6ff", "reader — reads tiles from DRAM into on-core L1 (over the NoC)"),
    "TRISC_0": ("T0·unpk", "#7ee787", "unpack — moves a tile from L1 into the math engine's input registers"),
    "TRISC_1": ("T1·math", "#3fb950", "math — the actual compute: matmul, modular multiply/add"),
    "TRISC_2": ("T2·pack", "#56d364", "pack — writes the result from math registers back to L1"),
    "NCRISC":  ("wr",      "#f0b429", "writer — writes result tiles from L1 back out to DRAM"),
}
SKIP = {"ph_recombine", "ph_matmul", "ph_decompose", "fbc_phase1", "fbc_phase2"}
LANE_ORDER = {"BRISC": 0, "TRISC_0": 1, "TRISC_1": 2, "TRISC_2": 3, "NCRISC": 4}


def parse(path):
    st = collections.defaultdict(list); ev = []
    with open(path) as f:
        f.readline(); r = csv.reader(f); next(r)
        for x in r:
            if len(x) < 12: continue
            cx, cy, ri = int(x[1]), int(x[2]), x[3].strip()
            t = int(x[5]); nm = x[10].strip(); ty = x[11].strip(); k = (cx, cy, ri)
            if ty == "ZONE_START":
                st[k].append((nm, t))
            elif ty == "ZONE_END" and st[k]:
                n, s = st[k].pop()
                if n in SKIP or n.endswith("-FW"): continue
                if n.endswith("-KERNEL"):
                    if ri in KWRAP: ev.append((cx, cy, ri, s, t, KWRAP[ri][0], KWRAP[ri][1], ri.split('_')[0]))
                elif n in NAMED:
                    cat, col = NAMED[n]; ev.append((cx, cy, ri, s, t, n, col, cat))
    return ev


def pick_window(ev, lo_ms, hi_ms):
    starts = sorted(e[3] for e in ev); t0 = starts[0]
    if lo_ms is not None: return t0 + int(lo_ms * 1e6), t0 + int(hi_ms * 1e6)
    # pick the region with the most COMPUTE (named TRISC ops), not the longest idle cluster —
    # split on >5ms gaps, then choose the cluster whose compute-op count is highest.
    comp_ts = sorted(e[3] for e in ev if e[7] not in ("reader", "writer"))   # named compute-op starts
    if not comp_ts: comp_ts = starts
    clusters = []; cur = [comp_ts[0]]
    for a, b in zip(comp_ts, comp_ts[1:]):
        if b - a > 5_000_000: clusters.append(cur); cur = [b]
        else: cur.append(b)
    clusters.append(cur)
    best = max(clusters, key=len)   # most compute ops
    return best[0], best[-1] + 2_000_000


def window_last_dispatches(path, expect):
    """Bounds of the last `expect` program dispatches, grouped on the -KERNEL ENVELOPES read straight
    from the raw file — byte-for-byte the rule dispatch_csv.py / busy_metrics.py apply.

    It re-reads the file rather than reusing parse()'s events ON PURPOSE. parse() keeps the -KERNEL
    envelope only for BRISC/NCRISC (they become the reader/writer bars) and drops the TRISC ones,
    keeping named compute zones instead. Grouping THAT by overlap groups different objects: dataflow
    bars from upload-only programs merge with compute zones, and any dispatch whose kernel is absent
    from NAMED vanishes entirely. The counts then drift from the host's, so "the last N groups" is not
    "the last N dispatches" -- it reached ~76 s back into setup and showed a few stray fwd8192 warm-up
    dispatches (modmul_compute, fused_total) followed by a long rd/wr desert before the real pass.
    Windowing on the envelopes makes the viewer and the CSV describe the same execution by construction.

    pick_window() is the alternative and is wrong here: it splits on >5 ms gaps and takes the densest
    cluster, but a HE_PROF_DRAIN capture stalls ~23 ms every HE_DRAIN_K dispatches, so it shatters one
    multiply into ~40 clusters. The idle compression below flattens those stalls instead."""
    st, env = collections.defaultdict(list), []
    with open(path) as f:
        f.readline(); r = csv.reader(f); next(r)
        for x in r:
            if len(x) < 12: continue
            k = (int(x[1]), int(x[2]), x[3].strip())
            t = int(x[5]); nm = x[10].strip(); ty = x[11].strip()
            if ty == "ZONE_START":
                st[k].append((nm, t))
            elif ty == "ZONE_END" and st[k]:
                n, s0 = st[k].pop()
                if n.endswith("-KERNEL"): env.append((s0, t, k[2]))
    env.sort()
    groups, cur, end = [], [env[0]], env[0][1]
    for e in env[1:]:
        if e[0] <= end: cur.append(e); end = max(end, e[1])
        else: groups.append(cur); cur = [e]; end = e[1]
    groups.append(cur)
    kern = [g for g in groups if any(e[2].startswith("TRISC") for e in g)]
    if len(kern) < expect:
        print(f"  WARNING: capture holds {len(kern)} dispatches, fewer than the {expect} expected "
              f"— markers were dropped; showing what is there.")
        expect = len(kern)
    win = kern[-expect:]
    return min(e[0] for e in win[0]), max(e[1] for e in win[-1]), expect


def build(path, lo_ms=None, hi_ms=None, expect=None):
    ev = parse(path)
    if expect:
        lo, hi, n = window_last_dispatches(path, expect)
        print(f"  windowed to the last {n} program dispatches "
              f"({(hi-lo)/1.35e6:.1f} ms span @ 1350 MHz)")
    else:
        lo, hi = pick_window(ev, lo_ms, hi_ms)
    ev = [e for e in ev if e[3] >= lo and e[3] <= hi]
    lanes = sorted({(e[0], e[1], e[2]) for e in ev}, key=lambda l: (l[0], l[1], LANE_ORDER.get(l[2], 9)))
    li = {l: i for i, l in enumerate(lanes)}
    lane_lbl, lane_col, lane_exp = [], [], []
    for cx, cy, ri in lanes:
        tag, col, exp = RISC_INFO.get(ri, (ri, "#c9d1d9", ri))
        lane_lbl.append(f"c({cx},{cy}) {tag}"); lane_col.append(col)
        lane_exp.append(f"({cx},{cy}) {ri} — {exp}")
    cats = {}
    for e in ev: cats.setdefault(e[7], e[6])
    ev.sort(key=lambda e: e[3])
    busy = collections.defaultdict(float); NODES = []
    for cx, cy, ri, s, t, lab, col, cat in ev:
        busy[cat] += (t - s) / 1e6
        NODES.append({"i": len(NODES), "lane": li[(cx, cy, ri)], "s": round((s-lo)/1000.0, 3),
                      "d": round(max((t-s)/1000.0, 0.05), 3), "op": lab, "col": col, "cat": cat,
                      "core": f"({cx},{cy})", "risc": ri})
    span = round((hi - lo) / 1000.0, 3)
    # --- compress idle gaps (drain stalls / dead time) so ops render back-to-back ---
    THRESH, STUB = 30.0, 8.0   # µs: gaps longer than THRESH collapse to STUB
    removed = 0.0
    if NODES:
        iv = sorted((n["s"], n["s"]+n["d"]) for n in NODES)
        merged = [list(iv[0])]
        for a, b in iv[1:]:
            if a <= merged[-1][1] + 0.001: merged[-1][1] = max(merged[-1][1], b)
            else: merged.append([a, b])
        off = merged[0][0]; segs = []   # remove leading dead time too
        for i, (a, b) in enumerate(merged):
            if i > 0:
                gap = a - merged[i-1][1]
                if gap > THRESH: off += (gap - STUB)
            segs.append((a, off))
        starts = [s for s, _ in segs]
        import bisect
        for n in NODES:
            j = bisect.bisect_right(starts, n["s"]) - 1
            n["s"] = round(n["s"] - segs[max(0, j)][1], 3)
        removed = round(off, 1)
        span = round(max(n["s"]+n["d"] for n in NODES), 3)
    meta = {"span": span, "lanes": len(lanes), "ops": len(NODES), "removed_us": removed,
            "busy": {k: round(v, 1) for k, v in sorted(busy.items(), key=lambda kv: -kv[1])}}
    return {"nodes": NODES, "lanes": lane_lbl, "laneCol": lane_col, "laneExp": lane_exp, "cats": cats, "meta": meta}


_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>ct_mul device trace</title><style>
 *{box-sizing:border-box} html,body{margin:0;height:100%;font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;background:#0e1116;color:#d7dde5}
 #top{position:fixed;top:0;left:0;right:0;height:46px;display:flex;align-items:center;gap:8px;padding:0 12px;background:#161b22;border-bottom:1px solid #30363d;z-index:10;flex-wrap:wrap}
 #top b{color:#eaf0f7;font-size:14px;margin-right:6px}
 button.act{padding:4px 9px;border:1px solid #30363d;border-radius:6px;background:#0e1116;color:#9aa7b4;cursor:pointer;font-size:11px}
 #search{padding:4px 8px;background:#0e1116;border:1px solid #30363d;border-radius:6px;color:#d7dde5;width:150px;font-size:11px}
 #meta{margin-left:auto;color:#7d8998;font-size:11px}
 #wrap{position:absolute;top:46px;left:0;right:322px;bottom:0} canvas{display:block;width:100%;height:100%;cursor:grab} canvas.drag{cursor:grabbing}
 #side{position:fixed;top:46px;right:0;bottom:0;width:322px;background:#161b22;border-left:1px solid #30363d;overflow:auto;padding:12px}
 #side .op{font-size:15px;font-weight:700}.kv{margin:8px 0;padding:8px;background:#0e1116;border:1px solid #30363d;border-radius:6px;word-break:break-all}
 .kv .k{color:#7d8998;font-size:11px;text-transform:uppercase} .muted{color:#7d8998}
 #legend{position:fixed;left:10px;bottom:8px;background:#161b22cc;border:1px solid #30363d;border-radius:8px;padding:5px 8px;font-size:10px;z-index:9}
 #legend span.sw{width:9px;height:9px;border-radius:2px;display:inline-block;margin-right:3px}
 #note{position:fixed;left:10px;bottom:44px;color:#c9a35b;font-size:10px;z-index:9;max-width:55%}
 #tip{position:fixed;display:none;max-width:320px;background:#1b2430;border:1px solid #3d5573;border-radius:6px;padding:7px 10px;font-size:12px;color:#e6edf3;z-index:20;pointer-events:none;box-shadow:0 4px 14px #0008}
 #tip b{color:#fff}
</style></head><body>
<div id="top"><b>ct_mul device trace</b>
 <button class="act" onclick="fit()">fit</button>
 <input id="search" placeholder="search op (matmul/fbc/reader…)"><span id="meta"></span></div>
<div id="wrap"><canvas id="cv"></canvas></div>
<div id="note"></div><div id="legend"></div><div id="side"></div><div id="tip"></div>
<script>
const D=__DATA__, N=D.nodes;
const LM=170;                         // wider left gutter for readable unit labels
let scaleX=1,panX=LM,panY=26,drag=null,sel=-1,hl=new Set();
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
const TOP=26,ROWMIN=10;
function nlanes(){return D.lanes.length;}
function rowPix(){return Math.max(ROWMIN,(cv.height-TOP-6)/nlanes());}
function Xt(t){return t*scaleX+panX;} function Yl(l){return l*rowPix()+panY;}
function fit(){scaleX=(cv.width-LM-8)/Math.max(D.meta.span,1)*0.99;panX=LM;panY=TOP;draw();}
function nstep(x){const p=Math.pow(10,Math.floor(Math.log10(x||1)));const m=x/p;return(m<1.5?1:m<3?2:m<7?5:10)*p;}
function laneAt(sy){const l=Math.floor((sy-panY)/rowPix());return (l>=0&&l<nlanes())?l:-1;}
function draw(){const w=cv.width,h=cv.height,rp=rowPix(),nl=nlanes();ctx.clearRect(0,0,w,h);
  // row backgrounds
  for(let l=0;l<nl;l++){const y=Yl(l);ctx.fillStyle=l%2?'#141a22':'#0e1116';ctx.fillRect(0,y,w,rp);}
  // time grid + axis
  ctx.strokeStyle='#20262e';ctx.fillStyle='#7d8998';ctx.font='10px ui-monospace';ctx.textAlign='center';
  const mk=D.meta.span,step=nstep(mk/(w/110));for(let t=0;t<=mk;t+=step){const x=Xt(t);if(x<LM||x>w)continue;ctx.beginPath();ctx.moveTo(x,TOP);ctx.lineTo(x,h);ctx.stroke();ctx.fillText(t.toFixed(mk<40?1:0)+'µs',x,13);}
  // bars
  ctx.textAlign='left';const dim=sel>=0||hl.size>0;
  for(const n of N){const x0=Math.max(LM,Xt(n.s)),x1=Math.max(x0+0.6,Xt(n.s+n.d)),y=Yl(n.lane);
    if(x1<LM||x0>w||y>h||y+rp<TOP)continue;
    ctx.globalAlpha=dim&&!(hl.has(n.i)||n.i===sel)?.18:1;ctx.fillStyle=n.col;ctx.fillRect(x0,y+1,x1-x0,rp-2);
    if(n.i===sel){ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.strokeRect(x0,y+1,x1-x0,rp-2);}}
  ctx.globalAlpha=1;
  // LEFT GUTTER: solid bg + large, bright, role-colored unit labels (drawn on top of bars)
  ctx.fillStyle='#0c0f14';ctx.fillRect(0,TOP,LM,h);
  const fs=Math.min(15,Math.max(11,rp-3));ctx.font='bold '+fs+'px ui-monospace';ctx.textBaseline='middle';
  for(let l=0;l<nl;l++){const y=Yl(l);if(y+rp<TOP||y>h)continue;ctx.fillStyle=D.laneCol[l];ctx.fillText(D.lanes[l],8,y+rp/2);}
  ctx.strokeStyle='#3d5573';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(LM,TOP);ctx.lineTo(LM,h);ctx.stroke();}
function hitBar(sx,sy){const rp=rowPix(),l=laneAt(sy);if(l<0||sx<LM)return -1;
  for(let i=N.length-1;i>=0;i--){const n=N[i];if(n.lane!==l)continue;const x0=Math.max(LM,Xt(n.s)),x1=Math.max(x0+2,Xt(n.s+n.d));if(sx>=x0-1&&sx<=x1)return i;}return -1;}
function side(i){const el=document.getElementById('side');
  if(i<0){el.innerHTML='<span class="muted">click a bar to inspect · hover a unit name (left) for what it does</span>';return;}
  const n=N[i];el.innerHTML=`<div class="op" style="color:${n.col}">${n.op}</div><div class="muted">#${i} · ${n.cat}</div>`+
    `<div class="kv"><div class="k">placement</div>core ${n.core} · ${n.risc}</div>`+
    `<div class="kv"><div class="k">timing</div>${n.s.toFixed(2)} → ${(n.s+n.d).toFixed(2)} µs · dur <b>${n.d.toFixed(2)} µs</b></div>`;}
function showTip(txt,x,y){tip.innerHTML=txt;tip.style.display='block';tip.style.left=Math.min(x+14,innerWidth-330)+'px';tip.style.top=(y+14)+'px';}
cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,px:panX,py:panY,mv:false};cv.classList.add('drag');});
window.addEventListener('mousemove',e=>{if(drag){panX=Math.max(LM,drag.px+(e.clientX-drag.x));panY=drag.py+(e.clientY-drag.y);if(Math.abs(e.clientX-drag.x)+Math.abs(e.clientY-drag.y)>3)drag.mv=true;draw();return;}
  const r=cv.getBoundingClientRect();const mx=e.clientX-r.left,my=e.clientY-r.top;
  if(mx>=0&&mx<LM){const l=laneAt(my);if(l>=0){showTip('<b>'+D.lanes[l]+'</b><br>'+D.laneExp[l],e.clientX,e.clientY);cv.title='';return;}}
  tip.style.display='none';
  const id=hitBar(mx,my);cv.title=id>=0?(N[id].op+' · '+N[id].d.toFixed(2)+'µs'):'';});
window.addEventListener('mouseup',e=>{if(drag&&!drag.mv){const r=cv.getBoundingClientRect();const mx=e.clientX-r.left;if(mx>=LM)select(hitBar(mx,e.clientY-r.top));}drag=null;cv.classList.remove('drag');});
cv.addEventListener('mouseleave',()=>{tip.style.display='none';});
function select(i){sel=i;hl.clear();side(i);draw();}
cv.addEventListener('wheel',e=>{e.preventDefault();const r=cv.getBoundingClientRect(),mx=Math.max(LM,e.clientX-r.left);const t=(mx-panX)/scaleX;scaleX*=Math.exp(-e.deltaY*0.0015);scaleX=Math.max(1e-4,scaleX);panX=mx-t*scaleX;draw();},{passive:false});
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();hl.clear();sel=-1;
  if(q){for(const n of N)if((n.op+' '+n.cat).toLowerCase().includes(q))hl.add(n.i);}draw();});
function resize(){const r=cv.parentElement.getBoundingClientRect();cv.width=r.width;cv.height=r.height;draw();}
window.addEventListener('resize',resize);
document.getElementById('meta').textContent=`${D.meta.ops.toLocaleString()} ops · ${D.meta.lanes} lanes · span ${D.meta.span.toFixed(0)}µs · hover a unit name (left) for what it does`;
document.getElementById('note').textContent="rd→T0→T1→T2→wr = DRAM→unpack→math→pack→DRAM · idle gaps compressed ("+(D.meta.removed_us/1000).toFixed(1)+"ms of drain-stall removed) · busy(ms): "+Object.entries(D.meta.busy).map(([k,v])=>k+" "+v).join(" · ");
const lg=document.getElementById('legend');for(const k in D.cats)lg.innerHTML+=`<span><span class="sw" style="background:${D.cats[k]}"></span>${k}</span>&nbsp; `;
side(-1);resize();fit();
</script></body></html>"""


def main():
    args, pos, expect = sys.argv[1:], [], None
    while args:
        a = args.pop(0)
        if a == "--expect": expect = int(args.pop(0))
        else: pos.append(a)
    path = pos[0]
    port = int(pos[1]) if len(pos) > 1 else PORT_DEFAULT
    lo = float(pos[2]) if len(pos) > 3 else None
    hi = float(pos[3]) if len(pos) > 3 else None
    data = build(path, lo, hi, expect)
    print(f"built trace: {data['meta']['ops']} ops · {data['meta']['lanes']} lanes · span {data['meta']['span']:.0f} us")
    out = os.path.join(os.path.dirname(os.path.abspath(path)), "ctmul_trace_app.html")
    open(out, "w").write(_HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    d = os.path.dirname(out) or "."
    class H(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", ""): self.path = "/ctmul_trace_app.html"
            return super().do_GET()
        def log_message(self, *a): pass
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), functools.partial(H, directory=d)) as httpd:
        print(f"\n  serving at http://0.0.0.0:{port}/   (forward this port; Ctrl-C to stop)")
        httpd.serve_forever()


main()
