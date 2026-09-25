# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""viz_app.py — the compiler's visual tool: pass ladder + tile DAG + scheduling trace, one page.

Three views, tabs to switch, everything inline (no CDN -- a live demo must not need a network):

  LADDER   what each pass bought, against the MEASURED hardware ladder. The point of the whole tool:
           the compiler's predicted rungs next to the numbers the hand-tuned pipeline actually hit.
  TILE DAG the lowered tile graph, every node coloured by the ENGINE the mapper chose, with a
           per-pass diff -- select a pass and the nodes it changed light up. Canvas, so it survives
           tens of thousands of nodes; drag to pan, wheel to zoom, hover for the op.
  TRACE    the schedule: one lane per core, x = microseconds, bars coloured by engine, dispatch-group
           boundaries marked. This is where fusion is visible as bars merging into runs.

Colour carries the ENGINE, which is the one thing this tool exists to show, and it is never the only
channel: every node has its opcode on hover, the legend is always present, and the trace repeats the
engine in its tooltip. Palette is Tenstorrent purple for the SFPU, teal for the matrix engine,
amber for dataflow -- three hues that stay separable
under deuteranopia and in greyscale by lightness.
"""
from __future__ import annotations
import json
import os

import _paths                                     # noqa: F401
import hw
import passes as P
from ir import outputs
from mlir_viz import program_graph

ENG_COLOR = {
    "sfpu": "#7C68FA",      # Tenstorrent purple -- the vector engine, where ct_mul mostly lives
    "fpu":  "#00C2A8",      # teal -- the matrix engine
    "dma":  "#F2A93B",      # amber -- movement
    "free": "#AEB6C2",      # structural: no kernel at all
}
CLS_LABEL = {"sfpu": "SFPU (vector)", "fpu": "matrix engine", "dma": "dataflow / a2a",
             "free": "structural (no kernel)"}


def collect(prog, mp, est, rungs, meta, max_nodes=20000):
    """Everything the page needs, as plain JSON-able dicts."""
    nodes = program_graph(prog)
    # Layers are recomputed here rather than taken from mlir_viz.layout, so the page does not depend on
    # that helper's exact return shape -- longest-path layer is all the DAG view needs.
    npre = len(prog.inputs)
    layers = [0] * len(nodes)
    for nd in nodes:
        if nd["n"]:
            layers[nd["i"]] = max(layers[p] for p in nd["n"]) + 1
    rows = {}
    for nd in nodes:
        rows.setdefault(layers[nd["i"]], []).append(nd["i"])
    # WRAP each layer into a compact block instead of one long row. ct_mul's widest layer is ~750 ops
    # (every limb of every digit, independent), which as a single row is a 19,000 px canvas -- fit-to-
    # window then makes every node sub-pixel and the view useless. Wrapped, a layer reads as a BAND
    # whose colour mix shows which engines that depth uses, and the whole DAG fits a screen.
    WRAP = 224          # chosen so the wrapped DAG lands near the canvas aspect (~2:1)
    pos, y = {}, 0
    band = []
    for l in sorted(rows):
        ids = rows[l]
        # order within a layer by (class, engine) so the bands separate visually rather than dithering
        ids.sort(key=lambda i: (nodes[i]["o"], i))
        h = (len(ids) + WRAP - 1) // WRAP
        for k, nid in enumerate(ids):
            pos[nid] = (k % WRAP, y + k // WRAP)
        band.append(dict(layer=l, y0=y, h=h, n=len(ids)))
        # No blank row: with 77 depth levels the separators would dominate the height. The alternating
        # band backgrounds drawn behind the nodes do the separating instead, for free.
        y += h

    out_nodes = []
    for nd in nodes:
        nid = nd["i"]
        oi = nid - npre                       # index into prog.ops (inputs come first)
        eng = "free"
        grp = -1
        cls = "input"
        var = ""
        if 0 <= oi < len(prog.ops):
            eng = mp.engine[oi]
            grp = mp.group[oi]
            cls = P.op_class(prog.ops[oi])
            var = (mp.variant[oi][0] or "")
        col, row = pos[nid]
        out_nodes.append(dict(i=nid, oi=oi, o=nd["o"], c=cls, e=eng, g=grp, x=col, y=row,
                              L=layers[nid], q=nd["q"], v=var, p=nd["n"]))
        if len(out_nodes) >= max_nodes:
            break

    keep = {n["i"] for n in out_nodes}
    edges = []
    for n in out_nodes:
        for pid in n["p"]:
            if pid not in keep:
                continue
            a, b = pid - npre, n["oi"]
            t = mp.tier.get((a, b), "noc") if (a >= 0 and b >= 0) else "noc"
            edges.append([pid, n["i"], t])

    trace = []
    for i in range(len(prog.ops)):
        d = est["dur"][i]
        if d <= 0:
            continue
        trace.append(dict(i=i, c=est["core"][i], s=round(est["start"][i], 4),
                          d=round(d, 4), e=mp.engine[i], g=mp.group[i],
                          o=prog.ops[i].opcode))

    lad = [dict(name=r.get("pass_name", "?"), ms=round(r.get("makespan", 0) / 1e3, 4),
                groups=r.get("groups", 0), note=r.get("note", ""),
                skipped=bool(r.get("skipped")),
                delta=round(r.get("delta", 0) / 1e3, 4)) for r in rungs]
    measured = [dict(name=n, ms=m, pas=p) for n, m, p in hw.LADDER]

    return dict(nodes=out_nodes, edges=edges, trace=trace, ladder=lad, measured=measured,
                bands=band, wrap=WRAP,
                meta=meta, truncated=len(out_nodes) < len(nodes),
                n_total=len(nodes), engcol=ENG_COLOR, englab=CLS_LABEL,
                P=est["P"], cores_used=est["cores_used"],
                makespan=round(est["makespan"] / 1e3, 4))


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>HE compiler — %(title)s</title><style>
:root{--bg:#0f1115;--pan:#171a21;--ink:#e6e9ef;--dim:#9aa4b2;--line:#262b35;--acc:#7C68FA}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:18px;flex-wrap:wrap}
h1{margin:0;font-size:16px;font-weight:650;letter-spacing:.2px}
.sub{color:var(--dim);font-size:12px}
.tabs{display:flex;gap:2px;padding:0 20px;border-bottom:1px solid var(--line);background:var(--pan)}
.tab{padding:9px 16px;cursor:pointer;color:var(--dim);border-bottom:2px solid transparent;font-weight:550}
.tab.on{color:var(--ink);border-bottom-color:var(--acc)}
.view{display:none;padding:18px 20px}.view.on{display:block}
canvas{border:1px solid var(--line);border-radius:6px;background:#12141a;display:block;cursor:grab}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:10px 0 14px}
.lg{display:flex;align-items:center;gap:7px;color:var(--dim)}
.sw{width:12px;height:12px;border-radius:3px}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:6px 12px;text-align:right;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:550;text-align:right}
td.l,th.l{text-align:left}
.note{color:var(--dim);font-size:12px;max-width:62ch}
.bar{height:14px;border-radius:3px;background:var(--acc);display:inline-block;vertical-align:middle}
.bar2{height:14px;border-radius:3px;background:#3d4450;display:inline-block;vertical-align:middle}
#tip{position:fixed;pointer-events:none;background:#0b0d11;border:1px solid var(--line);
  border-radius:5px;padding:6px 9px;font-size:12px;display:none;z-index:9;max-width:44ch;box-shadow:0 4px 14px #0008}
.ctl{display:flex;gap:14px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
select,button{background:var(--pan);color:var(--ink);border:1px solid var(--line);border-radius:5px;padding:5px 9px;font:inherit}
button{cursor:pointer}
.kv{color:var(--dim)}.kv b{color:var(--ink);font-weight:600}
.warn{color:#F2A93B}
#prog{margin:0;padding:10px 20px;background:#12141a;border-bottom:1px solid var(--line);
  font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;color:#b9c2d0;white-space:pre-wrap;
  max-height:130px;overflow:auto}
#prog .k{color:var(--acc);font-weight:600}
#prog .c{color:#5d6672}
</style></head><body>
<header><h1>HE compiler</h1><div class="sub" id="hdr"></div></header>
<pre id="prog"></pre>
<div class="tabs">
  <div class="tab on" data-v="lad">pass ladder</div>
  <div class="tab" data-v="dag">tile DAG</div>
  <div class="tab" data-v="tr">scheduling trace</div>
</div>
<div class="view on" id="v-lad"></div>
<div class="view" id="v-dag">
  <div class="ctl">
    <label class="kv">highlight what changed in: <select id="diff"></select></label>
    <button id="fit">fit</button><span class="kv" id="daginfo"></span>
  </div>
  <div class="legend" id="leg1"></div>
  <canvas id="cdag" width="1600" height="720"></canvas>
</div>
<div class="view" id="v-tr">
  <div class="ctl"><span class="kv" id="trinfo"></span>
    <label class="kv"><input type="checkbox" id="grp"> mark dispatch groups</label></div>
  <div class="legend" id="leg2"></div>
  <canvas id="ctr" width="1600" height="720"></canvas>
</div>
<div id="tip"></div>
<script>
const D = %(data)s;
const EC = D.engcol, EL = D.englab;
const $ = s => document.querySelector(s);
const tip = $("#tip");
function showTip(e, html){ tip.style.display="block"; tip.innerHTML=html;
  tip.style.left=Math.min(innerWidth-330, e.clientX+12)+"px"; tip.style.top=(e.clientY+12)+"px"; }
function hideTip(){ tip.style.display="none"; }

$("#hdr").textContent = D.meta.line;
if(D.meta.program){
  $("#prog").innerHTML = D.meta.program.split("\n").map(l =>
    l.trim().startsWith("#") ? `<span class="c">${l}</span>`
      : l.replace(/\b(mul|add)\b/g, '<span class="k">$1</span>')).join("\n");
} else { $("#prog").style.display = "none"; }

document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.remove("on"));
  document.querySelectorAll(".view").forEach(x=>x.classList.remove("on"));
  t.classList.add("on"); $("#v-"+t.dataset.v).classList.add("on");
  if(t.dataset.v==="dag") drawDag(); if(t.dataset.v==="tr") drawTr();
});

/* ---------------- legends ---------------- */
function legend(el){
  el.innerHTML = Object.keys(EC).map(k=>
    `<div class="lg"><div class="sw" style="background:${EC[k]}"></div>${EL[k]}</div>`).join("")
    + `<div class="lg"><div class="sw" style="background:#8a5a12"></div>DRAM round-trip (pick place_comm to see what it removed)</div>`
    + `<div class="lg"><div class="sw" style="background:#2f3646"></div>NoC hop</div>`
    + `<div class="lg"><div class="sw" style="background:#1e2430;border:1px solid #333"></div>stayed in L1</div>`;
}
legend($("#leg1")); legend($("#leg2"));

/* ---------------- LADDER ---------------- */
(function(){
  const L = D.ladder, M = D.measured;
  const mx = Math.max(...L.map(r=>r.ms));
  let h = `<table><thead><tr><th class="l">pass</th><th>predicted</th><th>delta</th>
    <th>dispatches</th><th class="l" style="width:340px">&nbsp;</th><th class="l">what it did</th></tr></thead><tbody>`;
  L.forEach((r,k)=>{
    const w = Math.max(2, 320*r.ms/mx);
    h += `<tr><td class="l">${r.skipped?'<span class="warn">'+r.name+' (skipped)</span>':r.name}</td>
      <td>${r.ms.toFixed(3)} ms</td>
      <td>${k===0?'':(r.delta>0?'+':'')+r.delta.toFixed(3)}</td>
      <td>${r.groups||''}</td>
      <td class="l"><span class="bar" style="width:${w}px"></span></td>
      <td class="l note">${r.note||''}</td></tr>`;
  });
  h += `</tbody></table>`;
  const first=L[0].ms, last=L[L.length-1].ms;
  h += `<p class="kv" style="margin-top:14px">end to end <b>${first.toFixed(3)} &rarr;
    ${last.toFixed(3)} ms = ${(first/last).toFixed(2)}&times;</b> &nbsp;·&nbsp;
    ${D.cores_used}/${D.P} cores used</p>`;

  const mmx = Math.max(...M.map(r=>r.ms));
  h += `<h3 style="margin:26px 0 8px;font-size:14px">the MEASURED hardware ladder &mdash; what the
    hand-tuned pipeline actually hit</h3>
    <p class="note" style="margin:0 0 10px">Each rung is one of the passes above, measured on this
    p150 at N=2<sup>16</sup>, L<sub>Q</sub>=48, P=16, dnum=3, every arm verified
    <code>0/6029312</code>. The compiler's job is to find these decisions, not to reproduce these
    absolute times &mdash; the demo workload is far smaller.</p>
    <table><thead><tr><th class="l">rung</th><th>measured</th><th class="l">pass that does it</th>
    <th class="l" style="width:300px">&nbsp;</th></tr></thead><tbody>`;
  M.forEach(r=>{
    h += `<tr><td class="l">${r.name}</td><td>${r.ms.toFixed(3)} ms</td>
      <td class="l">${r.pas?'<code>'+r.pas+'</code>':'&mdash;'}</td>
      <td class="l"><span class="bar2" style="width:${Math.max(2,300*r.ms/mmx)}px"></span></td></tr>`;
  });
  h += `</tbody></table>`;
  $("#v-lad").innerHTML = h;
})();

/* ---------------- DAG ---------------- */
const cd = $("#cdag"), gd = cd.getContext("2d");
let vx=0, vy=0, vs=1, dragging=false, lx=0, ly=0;
const NX = 13, NY = 15;                       // grid pitch (wrapped blocks, see collect())
const diffSel = $("#diff");
D.ladder.forEach((r,k)=>{ if(k>0){ const o=document.createElement("option");
  o.value=r.name; o.textContent=r.name; diffSel.appendChild(o);} });
const o0=document.createElement("option"); o0.value=""; o0.textContent="(nothing)";
diffSel.insertBefore(o0, diffSel.firstChild); diffSel.value="";
diffSel.onchange = drawDag;

function nodeXY(n){ return [n.x*NX, n.y*NY]; }
function hilite(n){
  const w = diffSel.value;
  if(!w) return false;
  if(w==="select_engines") return n.e==="fpu";
  if(w==="fuse_dispatches") return n.g>=0 && D.grpsize && D.grpsize[n.g]>1;
  if(w==="place_comm") return n.c==="a2a";
  if(w==="legalize") return !!(n.v && n.v.indexOf("chunk")>=0) || n.c==="contract";
  return false;
}
function drawDag(){
  gd.setTransform(1,0,0,1,0,0); gd.clearRect(0,0,cd.width,cd.height);
  gd.setTransform(vs,0,0,vs,vx,vy);
  gd.lineWidth = 1/vs;
  // Edge colour carries the memory TIER: amber = a DRAM round-trip (what place_comm removes),
  // mid-grey = a NoC hop, near-invisible = stayed in L1. With place_comm selected the edges are drawn
  // in their BASELINE tier instead of the final one -- every inbound edge of an all-to-all started as
  // a DRAM round-trip -- because otherwise the pass that removed them would show nothing.
  const TC = {dram:"#8a5a12", noc:"#2f3646", sram:"#1e2430"};
  const showBefore = diffSel.value === "place_comm";
  for(const [a,b,t] of D.edges){
    const na=D.nodes[a], nb=D.nodes[b]; if(!na||!nb) continue;
    const [x1,y1]=nodeXY(na), [x2,y2]=nodeXY(nb);
    const tt = (showBefore && nb.c === "a2a") ? "dram" : t;
    gd.strokeStyle = TC[tt] || "#232a36";
    gd.lineWidth = (tt === "dram" ? 2 : 1)/vs;
    gd.beginPath(); gd.moveTo(x1+5,y1+9); gd.lineTo(x2+5,y2); gd.stroke();
  }
  gd.lineWidth = 1/vs;
  for(const n of D.nodes){
    const [x,y]=nodeXY(n);
    const hl = hilite(n);
    gd.fillStyle = EC[n.e] || "#556";
    if(hl){ gd.fillStyle = "#fff"; }
    gd.fillRect(x, y, 10, 10);
    if(hl){ gd.strokeStyle=EC[n.e]||"#556"; gd.lineWidth=2/vs;
            gd.strokeRect(x-1.5,y-1.5,13,13); gd.lineWidth=1/vs; }
  }
  $("#daginfo").innerHTML = `<b>${D.nodes.length}</b> nodes` +
    (D.truncated?` <span class="warn">(truncated from ${D.n_total} for rendering)</span>`:"") +
    ` · <b>${D.edges.length}</b> edges · <b>${D.bands.length}</b> depth levels · drag to pan, wheel to zoom`;
}
function fit(){
  let mxx=1,mxy=1;
  for(const n of D.nodes){ const [x,y]=nodeXY(n); mxx=Math.max(mxx,x); mxy=Math.max(mxy,y); }
  vs = Math.min((cd.width-40)/(mxx+20), (cd.height-40)/(mxy+20));
  vs = Math.max(0.02, Math.min(vs, 2)); vx=20; vy=20; drawDag();
}
$("#fit").onclick = fit;
cd.onmousedown = e => { dragging=true; lx=e.clientX; ly=e.clientY; cd.style.cursor="grabbing"; };
addEventListener("mouseup", ()=>{ dragging=false; cd.style.cursor="grab"; });
cd.onmousemove = e => {
  if(dragging){ vx += e.clientX-lx; vy += e.clientY-ly; lx=e.clientX; ly=e.clientY; drawDag(); return; }
  const r = cd.getBoundingClientRect();
  const mx=(e.clientX-r.left-vx)/vs, my=(e.clientY-r.top-vy)/vs;
  let hit=null;
  for(const n of D.nodes){ const [x,y]=nodeXY(n);
    if(mx>=x-2&&mx<=x+12&&my>=y-2&&my<=y+12){ hit=n; break; } }
  if(hit) showTip(e, `<b>${hit.o}</b><br>depth ${hit.L} · class <b>${hit.c}</b> · engine <b>${hit.e}</b>` +
    (hit.v?`<br>kernel <b>${hit.v}</b>`:"") + (hit.q?`<br>q=${hit.q}`:"") +
    `<br>dispatch group ${hit.g}`);
  else hideTip();
};
cd.onmouseleave = hideTip;
cd.onwheel = e => { e.preventDefault();
  const r=cd.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const f = e.deltaY<0 ? 1.15 : 1/1.15;
  vx = mx-(mx-vx)*f; vy = my-(my-vy)*f; vs*=f; drawDag(); };

/* ---------------- TRACE ---------------- */
const ct = $("#ctr"), gt = ct.getContext("2d");
$("#grp").onchange = drawTr;
function drawTr(){
  gt.setTransform(1,0,0,1,0,0); gt.clearRect(0,0,ct.width,ct.height);
  const T = D.trace; if(!T.length){ return; }
  const tmax = Math.max(...T.map(b=>b.s+b.d));
  const cores = Math.max(...T.map(b=>b.c))+1;
  const padL=44, padT=18, padB=26;
  const W = ct.width-padL-14, H = ct.height-padT-padB;
  const rh = Math.max(1.2, Math.min(16, H/cores));
  const sx = W/tmax;
  gt.strokeStyle="#262b35"; gt.lineWidth=1;
  gt.beginPath(); gt.moveTo(padL,padT+cores*rh+2); gt.lineTo(padL+W,padT+cores*rh+2); gt.stroke();
  gt.fillStyle="#9aa4b2"; gt.font="11px ui-sans-serif";
  for(let k=0;k<=5;k++){ const t=tmax*k/5, x=padL+t*sx;
    gt.fillText(t.toFixed(0)+"us", x-8, padT+cores*rh+18);
    gt.strokeStyle="#1c212b"; gt.beginPath(); gt.moveTo(x,padT); gt.lineTo(x,padT+cores*rh); gt.stroke(); }
  if(rh>=7) for(let c=0;c<cores;c+=Math.ceil(cores/24)) gt.fillText("c"+c, 6, padT+c*rh+rh-1);
  const mark = $("#grp").checked;
  for(const b of T){
    gt.fillStyle = EC[b.e]||"#556";
    const x=padL+b.s*sx, w=Math.max(0.7, b.d*sx), y=padT+b.c*rh;
    gt.fillRect(x, y, w, Math.max(1, rh-1));
    if(mark && w>2){ gt.strokeStyle="#0f1115"; gt.lineWidth=0.6;
      gt.beginPath(); gt.moveTo(x,y); gt.lineTo(x,y+rh-1); gt.stroke(); }
  }
  $("#trinfo").innerHTML = `<b>${T.length}</b> scheduled ops · <b>${cores}</b> cores ·
    makespan <b>${D.makespan.toFixed(3)} ms</b> (compute + dispatch tax)`;
  ct._geo = {padL,padT,rh,sx,cores};
}
ct.onmousemove = e => {
  const g = ct._geo; if(!g) return;
  const r = ct.getBoundingClientRect();
  const t=(e.clientX-r.left-g.padL)/g.sx, c=Math.floor((e.clientY-r.top-g.padT)/g.rh);
  let hit=null;
  for(const b of D.trace) if(b.c===c && t>=b.s && t<=b.s+b.d){ hit=b; break; }
  if(hit) showTip(e, `<b>${hit.o}</b><br>core ${hit.c} · engine <b>${hit.e}</b><br>` +
    `start ${hit.s.toFixed(2)}us · ${hit.d.toFixed(3)}us<br>dispatch group ${hit.g}`);
  else hideTip();
};
ct.onmouseleave = hideTip;

/* group sizes for the fusion highlight */
D.grpsize = {}; for(const n of D.nodes) if(n.g>=0) D.grpsize[n.g]=(D.grpsize[n.g]||0)+1;
fit();
</script></body></html>
"""


def write(path, prog, mp, est, rungs, meta, max_nodes=20000):
    data = collect(prog, mp, est, rungs, meta, max_nodes)
    html = HTML % dict(title=meta.get("title", "ct_mul"), data=json.dumps(data, separators=(",", ":")))
    with open(path, "w") as f:
        f.write(html)
    return os.path.getsize(path)
