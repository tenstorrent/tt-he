# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""sched_viz.py — Gantt viewer for a scheduled ct_mul (the post-scheduling view).

The DAG viewer (mlir_viz.py) shows structure; once scheduled, each op has a CORE and a START/END, so
the right view is a Gantt: cores on Y, time on X, each op a bar. Builds the full ct_mul at L3, applies
Fusion A, schedules it (reuse-aware) for several core counts, and writes a SELF-CONTAINED sched_gantt
.html: bars colored by op category, click a bar to inspect it + see its data handoffs drawn as arrows
(SRAM same-core vs NoC cross-core), hover for a tooltip, time zoom/pan, and a P (core-count) switch.

Run:  python3 sched_viz.py            # writes sched_gantt.html
      python3 sched_viz.py --serve    # + serves it on :8000 (path / -> the gantt)
"""
from __future__ import annotations
import os, sys, json
from collections import defaultdict

import _paths                                     # noqa: F401  (repo layout; see _paths.py)
from ir import outputs                                       # noqa: E402

CAT = {  # opcode -> (category, color)
    "tile.matmul_modq": ("matmul", "#f0883e"), "tile.matmul_wide": ("matmul_wide", "#e3b341"),
    "tile.mul_modq": ("elem", "#3fb950"), "tile.add_modq": ("elem", "#3fb950"), "tile.sub_modq": ("elem", "#3fb950"),
    "tile.decompose": ("decompose", "#a371f7"), "tile.recombine_modq": ("recombine", "#db61a2"),
    "tile.recombine_wide": ("recombine", "#db61a2"), "tile.fixed_slice": ("slice", "#39c5cf"),
    "tile.transpose": ("struct", "#6e7681"), "tile.reshape": ("struct", "#586069"),
    "tile.concat": ("struct", "#586069"), "tile.split": ("struct", "#586069"),
    "tile.const": ("const", "#8b949e"), "rns.const": ("const", "#8b949e"),
}
def catcol(op): return CAT.get(op, ("other", "#768390"))


def build():
    import random
    from bfv_mul_ref import MulParams, encrypt
    from bfv_relin_ref import small_poly, keygen_relin
    from builders import build_ct_mul_rns
    from lower import lower_l1_to_l2, lower_l2_to_l3, fuse_a
    import schedule as sched

    pp = MulParams(N=16, LQ=3, LP=3, LB=4)
    rng = random.Random(4); s = small_poly(pp.N, rng); e0, e1 = keygen_relin(pp, s, rng)
    ctA = encrypt(pp, [rng.randrange(pp.t) for _ in range(pp.N)], s, rng)
    ctB = encrypt(pp, [rng.randrange(pp.t) for _ in range(pp.N)], s, rng)
    prog, _, _, _ = build_ct_mul_rns(pp, ctA, ctB, e0, e1)
    l3 = lower_l2_to_l3(fuse_a(lower_l1_to_l2(prog)))

    prod_of = {}
    for i, op in enumerate(l3.ops):
        for v in outputs(op):
            prod_of[v.name] = i
    nodes = []
    for i, op in enumerate(l3.ops):
        outs = outputs(op)
        ins = [prod_of[o.name] for o in op.operands if o.name in prod_of]
        cat, col = catcol(op.opcode)
        nodes.append({"i": i, "o": op.opcode, "c": cat, "col": col,
                      "r": (outs[0].name if outs else ""), "in": ins})
    scheds = {}
    for P in (8, 16, 32, 64):
        sc = sched.schedule(l3, P, reuse=True)
        tier = {(d, j): t for (d, j, t) in sc["edges"]}
        it = [[("noc" if tier.get((d, i)) == "noc" else "sram") for d in nodes[i]["in"]] for i in range(len(nodes))]
        scheds[str(P)] = {
            "core": sc["core"], "start": [round(x, 2) for x in sc["start"]],
            "end": [round(x, 2) for x in sc["end"]], "intier": it,
            "makespan": round(sc["makespan"], 1), "used": sc["cores_used"],
            "reuse": 100 * sc["reuse_hits"] // max(1, sc["accesses"]),
            "fusedB": 100 * sc["fused"] // max(1, sc["n_edges"]),
        }
    return nodes, scheds


_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>ct_mul schedule — Gantt</title><style>
 *{box-sizing:border-box} html,body{margin:0;height:100%;font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;background:#0e1116;color:#d7dde5}
 #top{position:fixed;top:0;left:0;right:0;height:44px;display:flex;align-items:center;gap:8px;padding:0 12px;background:#161b22;border-bottom:1px solid #30363d;z-index:10}
 #top b{color:#eaf0f7;font-size:14px;margin-right:6px}
 .tab{padding:5px 10px;border:1px solid #30363d;border-radius:6px;background:#0e1116;color:#9aa7b4;cursor:pointer}
 .tab.on{background:#1f6feb;border-color:#1f6feb;color:#fff}
 #stat{margin-left:auto;color:#9aa7b4}
 button.act{padding:5px 10px;border:1px solid #30363d;border-radius:6px;background:#0e1116;color:#9aa7b4;cursor:pointer}
 #wrap{position:absolute;top:44px;left:0;right:320px;bottom:0}
 canvas{display:block;width:100%;height:100%;cursor:grab} canvas.drag{cursor:grabbing}
 #side{position:fixed;top:44px;right:0;bottom:0;width:320px;background:#161b22;border-left:1px solid #30363d;overflow:auto;padding:12px}
 #side .op{font-size:15px;font-weight:700}.kv{margin:8px 0;padding:8px;background:#0e1116;border:1px solid #30363d;border-radius:6px;word-break:break-all}
 .kv .k{color:#7d8998;font-size:11px;text-transform:uppercase}.lnk{color:#58a6ff;cursor:pointer}.lnk:hover{text-decoration:underline}
 #legend{position:fixed;left:12px;bottom:10px;background:#161b22cc;border:1px solid #30363d;border-radius:8px;padding:6px 9px;font-size:11px;z-index:9}
 #legend span.sw{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:4px}
 .muted{color:#7d8998}
</style></head><body>
<div id="top"><b>ct_mul schedule · Gantt</b><span id="tabs"></span>
 <button class="act" onclick="fit()">fit</button>
 <label style="color:#9aa7b4"><input type="checkbox" id="arrows" checked> handoffs (selected)</label>
 <span id="stat"></span></div>
<div id="wrap"><canvas id="cv"></canvas></div>
<div id="legend"></div>
<div id="side"><span class="muted">click a bar to inspect the op + its data handoffs</span></div>
<script>
const NODES=__NODES__, SCH=__SCH__;
const CATS={matmul:'#f0883e',matmul_wide:'#e3b341',elem:'#3fb950',decompose:'#a371f7',recombine:'#db61a2',slice:'#39c5cf',struct:'#6e7681',const:'#8b949e',other:'#768390'};
let P=Object.keys(SCH)[0], s=null, succ=[];
let scaleX=1, panX=0, panY=0, drag=null, sel=-1, hover=-1;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const TOP=26, ROWMIN=7;

function setP(p){ P=p; s=SCH[p]; sel=-1;
  succ=NODES.map(()=>[]); for(const n of NODES) for(const d of n.in) succ[d].push(n.i);
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.dataset.p===p));
  document.getElementById('stat').textContent=`P=${p} · makespan ${s.makespan}us · ${s.used} cores · reuse ${s.reuse}% · FusionB ${s.fusedB}%`;
  fit(); }
function ncores(){ return Math.max(...s.core)+1; }
function rowPix(){ return Math.max(ROWMIN,(cv.height-TOP-6)/ncores()); }
function fit(){ const mk=s.makespan||1; scaleX=(cv.width-70)/mk; panX=60; panY=TOP; draw(); }
function X(t){ return t*scaleX+panX; } function Y(c){ return c*rowPix()+panY; }
function invT(px){ return (px-panX)/scaleX; } function invC(py){ return Math.floor((py-panY)/rowPix()); }

function draw(){ const w=cv.width,h=cv.height,rp=rowPix(),nc=ncores(); ctx.clearRect(0,0,w,h);
  // core rows + labels
  ctx.font='10px ui-monospace'; ctx.textBaseline='middle';
  for(let c=0;c<nc;c++){ const y=Y(c); ctx.fillStyle=c%2?'#12161c':'#0e1116'; ctx.fillRect(0,y,w,rp);
    if(rp>=9){ ctx.fillStyle='#4d5562'; ctx.fillText('c'+c, 4, y+rp/2); } }
  // time axis ticks
  ctx.strokeStyle='#20262e'; ctx.fillStyle='#5b6470'; ctx.textAlign='center';
  const mk=s.makespan, step=niceStep(mk/ (w/110));
  for(let t=0;t<=mk;t+=step){ const x=X(t); if(x<50||x>w)continue; ctx.beginPath();ctx.moveTo(x,TOP);ctx.lineTo(x,h);ctx.stroke();
    ctx.fillText(t.toFixed(0)+'us', x, 13); }
  ctx.textAlign='left';
  // bars (viewport-culled)
  const dim = sel>=0;
  for(const n of NODES){ const i=n.i, c=s.core[i]; if(c<0)continue;
    const x0=X(s.start[i]), x1=Math.max(x0+1,X(s.end[i])), y=Y(c);
    if(x1<50||x0>w||y>h||y+rp<TOP)continue;
    ctx.globalAlpha = dim && i!==sel && !isNbr(i) ? 0.28 : 1;
    ctx.fillStyle=n.col; ctx.fillRect(x0,y+1,x1-x0,rp-2);
    if(i===sel){ ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.strokeRect(x0,y+1,x1-x0,rp-2); }
    if(rp>=12 && x1-x0>34){ ctx.fillStyle='#0b0e12'; ctx.font='9px ui-monospace';
      ctx.fillText(n.o.replace(/^(tile|rns)\./,'').slice(0,Math.floor((x1-x0)/6)), x0+3, y+rp/2); }
  }
  ctx.globalAlpha=1;
  // handoff arrows for selected op
  if(sel>=0 && document.getElementById('arrows').checked){
    const rp2=rp;
    const arrow=(a,b,t)=>{ ctx.strokeStyle=t==='noc'?'#f0883e':'#3fb950'; ctx.lineWidth=t==='noc'?1.6:1.2;
      ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
      const ang=Math.atan2(b[1]-a[1],b[0]-a[0]); ctx.beginPath();
      ctx.moveTo(b[0],b[1]); ctx.lineTo(b[0]-7*Math.cos(ang-.4),b[1]-7*Math.sin(ang-.4));
      ctx.lineTo(b[0]-7*Math.cos(ang+.4),b[1]-7*Math.sin(ang+.4)); ctx.closePath(); ctx.fillStyle=ctx.strokeStyle; ctx.fill(); };
    NODES[sel].in.forEach((d,k)=>{ const t=s.intier[sel][k];
      arrow([X(s.end[d]),Y(s.core[d])+rp2/2],[X(s.start[sel]),Y(s.core[sel])+rp2/2],t); });
    for(const u of succ[sel]){ const k=NODES[u].in.indexOf(sel); const t=s.intier[u][k]||'sram';
      arrow([X(s.end[sel]),Y(s.core[sel])+rp2/2],[X(s.start[u]),Y(s.core[u])+rp2/2],t); }
  }
}
function niceStep(x){ const p=Math.pow(10,Math.floor(Math.log10(x||1))); const m=x/p; return (m<1.5?1:m<3?2:m<7?5:10)*p; }
function isNbr(i){ if(sel<0)return false; return NODES[sel].in.includes(i)||succ[sel].includes(i); }

function hit(sx,sy){ const rp=rowPix(); const c=invC(sy), t=invT(sx); if(c<0)return -1;
  for(const n of NODES){ if(s.core[n.i]!==c)continue; if(t>=s.start[n.i]&&t<=Math.max(s.end[n.i],s.start[n.i]+0.5/scaleX*scaleX))
    if(sx>=X(s.start[n.i])-1 && sx<=Math.max(X(s.start[n.i])+2,X(s.end[n.i]))) return n.i; }
  return -1; }
function select(i){ sel=i; side(i); draw(); }
function lnk(i){ const n=NODES[i]; return `<span class="lnk" onclick="select(${i})">${n.o.replace(/^(tile|rns)\./,'')}·c${s.core[i]}</span>`; }
function side(i){ const el=document.getElementById('side');
  if(i<0){ el.innerHTML='<span class="muted">click a bar to inspect the op + its data handoffs</span>'; return; }
  const n=NODES[i]; let h=`<div class="op" style="color:${n.col}">${n.o}</div><div class="muted">#${i} · ${n.c}</div>`;
  h+=`<div class="kv"><div class="k">placement</div>core ${s.core[i]} · start ${s.start[i]}us · end ${s.end[i]}us · dur ${(s.end[i]-s.start[i]).toFixed(2)}us</div>`;
  h+=`<div class="kv"><div class="k">result</div>${n.r}</div>`;
  const inh=n.in.map((d,k)=>`${lnk(d)} <span style="color:${s.intier[i][k]==='noc'?'#f0883e':'#3fb950'}">[${s.intier[i][k]}]</span>`).join('<br>');
  h+=`<div class="kv"><div class="k">inputs (${n.in.length})  · green=SRAM(on-core)  orange=NoC</div>${inh||'<span class=muted>none</span>'}</div>`;
  const cs=succ[i]; h+=`<div class="kv"><div class="k">used by (${cs.length})</div>${cs.slice(0,30).map(lnk).join('<br>')||'<span class=muted>none</span>'}</div>`;
  el.innerHTML=h; }

cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,px:panX,py:panY,mv:false};cv.classList.add('drag');});
window.addEventListener('mousemove',e=>{ if(drag){panX=drag.px+(e.clientX-drag.x);panY=drag.py+(e.clientY-drag.y);if(Math.abs(e.clientX-drag.x)+Math.abs(e.clientY-drag.y)>3)drag.mv=true;draw();return;}
  const r=cv.getBoundingClientRect(); const id=hit(e.clientX-r.left,e.clientY-r.top);
  if(id!==hover){hover=id; cv.title=id>=0?NODES[id].o+' → '+NODES[id].r:''; }});
window.addEventListener('mouseup',e=>{ if(drag&&!drag.mv){const r=cv.getBoundingClientRect();select(hit(e.clientX-r.left,e.clientY-r.top));} drag=null;cv.classList.remove('drag');});
cv.addEventListener('wheel',e=>{e.preventDefault(); const r=cv.getBoundingClientRect(),mx=e.clientX-r.left; const t=invT(mx);
  scaleX*=Math.exp(-e.deltaY*0.0015); scaleX=Math.max(1e-4,scaleX); panX=mx-t*scaleX; draw(); },{passive:false});
document.getElementById('arrows').addEventListener('change',draw);
function resize(){const r=cv.parentElement.getBoundingClientRect();cv.width=r.width;cv.height=r.height;draw();}
window.addEventListener('resize',resize);
const tabs=document.getElementById('tabs'); for(const p of Object.keys(SCH)){const b=document.createElement('span');b.className='tab';b.dataset.p=p;b.textContent=p+' cores';b.onclick=()=>setP(p);tabs.appendChild(b);}
const lg=document.getElementById('legend'); for(const k in CATS) lg.innerHTML+=`<span><span class="sw" style="background:${CATS[k]}"></span>${k}</span>&nbsp; `;
resize(); setP(P);
</script></body></html>"""


if __name__ == "__main__":
    nodes, scheds = build()
    html = (_HTML.replace("__NODES__", json.dumps(nodes, separators=(",", ":")))
                 .replace("__SCH__", json.dumps(scheds, separators=(",", ":"))))
    out = os.path.join(os.path.dirname(__file__), "sched_gantt.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({os.path.getsize(out)//1024} KB) — {len(nodes)} ops, P in {list(scheds)}")
    for p, sc in scheds.items():
        print(f"  P={p}: makespan {sc['makespan']}us, {sc['used']} cores, reuse {sc['reuse']}%, FusionB {sc['fusedB']}%")

    if "--serve" in sys.argv:
        import http.server, socketserver, functools
        port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
        d = os.path.dirname(out) or "."
        class H(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", ""):
                    self.path = "/sched_gantt.html"
                return super().do_GET()
            def log_message(self, *a): pass
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("0.0.0.0", port), functools.partial(H, directory=d)) as httpd:
            print(f"serving Gantt at http://0.0.0.0:{port}/  — Ctrl-C to stop")
            httpd.serve_forever()
