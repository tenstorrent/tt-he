# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""mlir_viz.py — interactive DAG viewer for the L1/L2/L3 programs (the "MLIR").

Builds the full ct_mul at all three levels, extracts each as a dependency DAG with a precomputed
layered layout, and writes a SELF-CONTAINED `mlir_dag.html` (no external libraries) that renders the
DAG on a <canvas> with zoom/pan, op-type coloring, search, dependency highlighting, and a side panel
of per-op info. Built for the large-graph case: L3 is ~3k nodes, so nodes are drawn on canvas with
viewport culling and the layout is precomputed here (not in the browser).

Run:  python3 mlir_viz.py            # writes mlir_dag.html — open it in a browser
      python3 mlir_viz.py --tensor   # visualize the tensor instead of the full ct_mul
"""
from __future__ import annotations
import os, sys, json, random
from collections import defaultdict

import _paths                                     # noqa: F401  (repo layout; see _paths.py)
from ir import outputs                                        # noqa: E402


# --------------------------------------------------------------------------- #
# Program -> graph (nodes = ops, edges = value dependencies)
# --------------------------------------------------------------------------- #
def _ty(v):
    return ("x".join(str(d) for d in v.shape) + "x" + v.dtype) if v.shape else ("1x" + v.dtype)


def _attrs(op):
    parts = []
    for k, val in op.attrs.items():
        if k in ("psi", "fn"):
            continue
        if k == "name":
            parts.append(f'name="{val}"')
        elif isinstance(val, (list, tuple)):
            parts.append(f"{k}={list(val)}")
        else:
            parts.append(f"{k}={val}")
    return ", ".join(parts)


def program_graph(prog):
    """-> list of node dicts (short keys): i id, k kind, o opcode, r result names, t type,
       q prime, a attrs, n predecessor ids."""
    producer, nodes = {}, []

    def add(kind, op, res, ty, q, attrs, ins):
        nid = len(nodes)
        nodes.append({"i": nid, "k": kind, "o": op, "r": res, "t": ty,
                      "q": (str(q) if q is not None else None), "a": attrs, "n": ins})
        return nid

    for v in prog.inputs:
        producer[v.name] = add("input", "input", [v.name], _ty(v), v.q, "", [])
    for op in prog.ops:
        outs = outputs(op)
        ins = [producer[o.name] for o in op.operands if o.name in producer]
        kind = "const" if op.opcode.endswith(".const") else "op"
        nid = add(kind, op.opcode, [v.name for v in outs], _ty(outs[0]), outs[0].q, _attrs(op), ins)
        for v in outs:
            producer[v.name] = nid
    return nodes


# --------------------------------------------------------------------------- #
# layered layout — longest-path layers, barycenter ordering, source pull-down
# --------------------------------------------------------------------------- #
def layout(nodes, dx=190, dy=70):
    n = len(nodes)
    succ = defaultdict(list)
    for nd in nodes:
        for p in nd["n"]:
            succ[p].append(nd["i"])
    layer = [0] * n
    for nd in nodes:                                          # nodes are SSA/topological
        if nd["n"]:
            layer[nd["i"]] = max(layer[p] for p in nd["n"]) + 1
    # pull sources (const/input, no preds) down to just above their earliest consumer
    for nd in nodes:
        if not nd["n"] and succ[nd["i"]]:
            layer[nd["i"]] = max(0, min(layer[c] for c in succ[nd["i"]]) - 1)

    rows = defaultdict(list)
    for nd in nodes:
        rows[layer[nd["i"]]].append(nd["i"])
    order = {}
    for lay in rows:
        for idx, nid in enumerate(rows[lay]):
            order[nid] = idx
    for _ in range(6):                                        # barycenter sweeps to cut crossings
        for lay in sorted(rows):
            rows[lay].sort(key=lambda nid: (sum(order[p] for p in nodes[nid]["n"]) / len(nodes[nid]["n"]))
                           if nodes[nid]["n"] else order[nid])
            for idx, nid in enumerate(rows[lay]):
                order[nid] = idx
    width = max((len(r) for r in rows.values()), default=1)
    for lay, row in rows.items():
        off = (width - len(row)) / 2.0                        # center each layer
        for idx, nid in enumerate(row):
            nodes[nid]["x"] = round((off + idx) * dx)
            nodes[nid]["y"] = lay * dy
            nodes[nid]["l"] = lay
    return nodes


# --------------------------------------------------------------------------- #
# HTML (self-contained: canvas renderer + side panel + search)
# --------------------------------------------------------------------------- #
_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 *{box-sizing:border-box} html,body{margin:0;height:100%;font:13px/1.4 ui-monospace,Menlo,Consolas,monospace;background:#0e1116;color:#d7dde5}
 #top{position:fixed;top:0;left:0;right:0;height:44px;display:flex;align-items:center;gap:8px;padding:0 12px;background:#161b22;border-bottom:1px solid #30363d;z-index:10}
 #top b{color:#eaf0f7;font-size:14px;margin-right:6px}
 .tab{padding:5px 12px;border:1px solid #30363d;border-radius:6px;background:#0e1116;color:#9aa7b4;cursor:pointer}
 .tab.on{background:#1f6feb;border-color:#1f6feb;color:#fff}
 #search{margin-left:auto;padding:5px 9px;background:#0e1116;border:1px solid #30363d;border-radius:6px;color:#d7dde5;width:220px}
 button.act{padding:5px 10px;border:1px solid #30363d;border-radius:6px;background:#0e1116;color:#9aa7b4;cursor:pointer}
 #wrap{position:absolute;top:44px;left:0;right:320px;bottom:0}
 canvas{display:block;width:100%;height:100%;cursor:grab}
 canvas.drag{cursor:grabbing}
 #side{position:fixed;top:44px;right:0;bottom:0;width:320px;background:#161b22;border-left:1px solid #30363d;overflow:auto;padding:12px}
 #side h3{margin:0 0 8px;font-size:13px;color:#eaf0f7}
 #side .op{font-size:15px;font-weight:700}
 .kv{margin:8px 0;padding:8px;background:#0e1116;border:1px solid #30363d;border-radius:6px;word-break:break-all}
 .kv .k{color:#7d8998;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
 .lnk{color:#58a6ff;cursor:pointer}.lnk:hover{text-decoration:underline}
 #legend{position:fixed;left:12px;bottom:12px;background:#161b22cc;border:1px solid #30363d;border-radius:8px;padding:8px 10px;font-size:11px;z-index:9}
 #legend div{display:flex;align-items:center;gap:6px;margin:2px 0}.sw{width:11px;height:11px;border-radius:2px;display:inline-block}
 #hud{position:fixed;left:12px;top:52px;color:#7d8998;font-size:11px;z-index:9}
 .muted{color:#7d8998}
</style></head><body>
<div id="top"><b>MLIR DAG</b>
 <span id="tabs"></span>
 <button class="act" onclick="fit()">fit</button>
 <input id="search" placeholder="search op / value name…">
</div>
<div id="wrap"><canvas id="cv"></canvas></div>
<div id="hud"></div>
<div id="legend"></div>
<div id="side"><span class="muted">click a node to inspect its op</span></div>
<script>
const DATA = __DATA__;
const CAT = {
  matmul:{c:'#f0883e',t:['tile.matmul_modq','tile.matmul_wide']},
  ntt:{c:'#a371f7',t:['rns.ntt','rns.intt']},
  elem:{c:'#3fb950',t:['rns.mul','rns.add','rns.sub','tile.mul_modq','tile.add_modq','tile.sub_modq']},
  bconv:{c:'#db61a2',t:['rns.base_convert']},
  int8:{c:'#e3b341',t:['tile.decompose','tile.recombine_modq','tile.recombine_wide']},
  slice:{c:'#39c5cf',t:['tile.fixed_slice']},
  struct:{c:'#6e7681',t:['tile.reshape','tile.transpose','tile.concat','tile.slice','tile.split','tile.all_to_all_tiles']},
  const:{c:'#8b949e',t:['tile.const','rns.const']},
  input:{c:'#1f6feb',t:['input']},
};
function catOf(op){for(const k in CAT)if(CAT[k].t.includes(op))return k;return 'other';}
function colOf(op){const k=catOf(op);return k==='other'?'#8b949e':CAT[k].c;}

let level=Object.keys(DATA)[0], nodes=[], succ=[], sel=-1, hl=new Set();
let scale=1, px=0, py=0, drag=null, hoverId=-1;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const NW=150, NH=34;

function setLevel(name){
  level=name; nodes=DATA[name].nodes; sel=-1; hl.clear();
  succ=nodes.map(()=>[]); for(const n of nodes) for(const p of n.n) succ[p].push(n.i);
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.dataset.l===name));
  document.getElementById('hud').textContent=`${name}: ${nodes.length} nodes`;
  fit();
}
function fit(){
  if(!nodes.length)return;
  let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
  for(const n of nodes){x0=Math.min(x0,n.x);y0=Math.min(y0,n.y);x1=Math.max(x1,n.x+NW);y1=Math.max(y1,n.y+NH);}
  const w=cv.width, h=cv.height;
  scale=Math.min(w/(x1-x0+80), h/(y1-y0+80), 1.4);
  px=w/2-(x0+x1)/2*scale; py=40-y0*scale; draw();
}
function W2S(x,y){return [x*scale+px, y*scale+py];}
function S2W(x,y){return [(x-px)/scale,(y-py)/scale];}

function draw(){
  const w=cv.width,h=cv.height; ctx.clearRect(0,0,w,h);
  const dim = sel>=0 || hoverId>=0;
  // edges
  ctx.lineWidth=Math.max(.6,scale*.8);
  for(const n of nodes){
    const [ax,ay]=W2S(n.x+NW/2,n.y);
    for(const p of n.n){
      const s=nodes[p]; const [bx,by]=W2S(s.x+NW/2,s.y+NH);
      if((bx<-50&&ax<-50)||(bx>w+50&&ax>w+50)||(by<-50&&ay<-50)||(by>h+50&&ay>h+50))continue;
      const on = dim && (hl.has(n.i)&&hl.has(p));
      ctx.strokeStyle = on?'#58a6ff': (dim?'#22262d':'#343b45');
      ctx.beginPath();ctx.moveTo(bx,by);
      ctx.bezierCurveTo(bx,by+(ay-by)*.5, ax,ay-(ay-by)*.5, ax,ay);ctx.stroke();
    }
  }
  // nodes (viewport-culled)
  const showText=scale>.45;
  ctx.font='11px ui-monospace,Menlo,monospace'; ctx.textBaseline='middle';
  for(const n of nodes){
    const [sx,sy]=W2S(n.x,n.y); const sw=NW*scale, sh=NH*scale;
    if(sx>w||sy>h||sx+sw<0||sy+sh<0)continue;
    const faded = dim && !hl.has(n.i);
    ctx.globalAlpha = faded?0.22:1;
    ctx.fillStyle = colOf(n.o); ctx.strokeStyle = n.i===sel?'#fff':'#0e1116'; ctx.lineWidth=n.i===sel?2:1;
    roundRect(sx,sy,sw,sh,5*scale); ctx.fill(); if(n.i===sel||n.i===hoverId)ctx.stroke();
    if(showText){
      ctx.fillStyle='#0b0e12'; const lbl=n.o.replace(/^(tile|rns)\./,'');
      ctx.fillText(lbl.length>18?lbl.slice(0,17)+'…':lbl, sx+7, sy+sh*0.36);
      ctx.fillStyle='#0b0e12dd'; ctx.font='9px ui-monospace';
      const rn=n.r[0]||''; ctx.fillText(rn.length>20?rn.slice(0,19)+'…':rn, sx+7, sy+sh*0.72);
      ctx.font='11px ui-monospace,Menlo,monospace';
    }
  }
  ctx.globalAlpha=1;
}
function roundRect(x,y,w,h,r){r=Math.max(0,Math.min(r,h/2));ctx.beginPath();
  ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();}

function hit(sx,sy){
  const [wx,wy]=S2W(sx,sy);
  for(let i=nodes.length-1;i>=0;i--){const n=nodes[i];
    if(wx>=n.x&&wx<=n.x+NW&&wy>=n.y&&wy<=n.y+NH)return i;}
  return -1;
}
function select(id){
  sel=id; hl.clear();
  if(id>=0){hl.add(id); for(const p of nodes[id].n)hl.add(p); for(const c of succ[id])hl.add(c);}
  sidePanel(id); draw();
}
function nodeLink(id){const n=nodes[id];return `<span class="lnk" onclick="select(${id})">${n.o.replace(/^(tile|rns)\./,'')} · ${(n.r[0]||'').slice(0,22)}</span>`;}
function sidePanel(id){
  const s=document.getElementById('side');
  if(id<0){s.innerHTML='<span class="muted">click a node to inspect its op</span>';return;}
  const n=nodes[id];
  let h=`<div class="op" style="color:${colOf(n.o)}">${n.o}</div><div class="muted">node #${n.i} · layer ${n.l} · ${catOf(n.o)}</div>`;
  h+=`<div class="kv"><div class="k">result${n.r.length>1?'s':''}</div>${n.r.join('<br>')}</div>`;
  h+=`<div class="kv"><div class="k">type</div>tensor&lt;${n.t}&gt;</div>`;
  if(n.q)h+=`<div class="kv"><div class="k">prime q</div>${n.q}</div>`;
  if(n.a)h+=`<div class="kv"><div class="k">attrs</div>${n.a.replace(/</g,'&lt;')}</div>`;
  h+=`<div class="kv"><div class="k">operands (${n.n.length})</div>${n.n.map(nodeLink).join('<br>')||'<span class=muted>none</span>'}</div>`;
  const cs=succ[n.i];
  h+=`<div class="kv"><div class="k">used by (${cs.length})</div>${cs.map(nodeLink).join('<br>')||'<span class=muted>none</span>'}</div>`;
  s.innerHTML=h;
}

// interaction
cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,px,py,moved:false};cv.classList.add('drag');});
window.addEventListener('mousemove',e=>{
  if(drag){px=drag.px+(e.clientX-drag.x);py=drag.py+(e.clientY-drag.y);if(Math.abs(e.clientX-drag.x)+Math.abs(e.clientY-drag.y)>3)drag.moved=true;draw();return;}
  const r=cv.getBoundingClientRect(); const id=hit(e.clientX-r.left,e.clientY-r.top);
  if(id!==hoverId){hoverId=id;cv.title=id>=0?nodes[id].r[0]||nodes[id].o:'';draw();}
});
window.addEventListener('mouseup',e=>{
  if(drag&&!drag.moved){const r=cv.getBoundingClientRect();select(hit(e.clientX-r.left,e.clientY-r.top));}
  drag=null;cv.classList.remove('drag');
});
cv.addEventListener('wheel',e=>{e.preventDefault();
  const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  const [wx,wy]=S2W(mx,my); scale*=Math.exp(-e.deltaY*0.0012); scale=Math.max(.05,Math.min(3,scale));
  px=mx-wx*scale; py=my-wy*scale; draw();
},{passive:false});
document.getElementById('search').addEventListener('input',e=>{
  const q=e.target.value.toLowerCase(); hl.clear();
  if(q){let first=-1;for(const n of nodes){if((n.o+' '+n.r.join(' ')).toLowerCase().includes(q)){hl.add(n.i);if(first<0)first=n.i;}}
    if(first>=0){const n=nodes[first];const r=cv.getBoundingClientRect();px=r.width/2-(n.x+NW/2)*scale;py=r.height/2-(n.y+NH/2)*scale;}}
  sel=-1; draw();
});
function resize(){const r=cv.parentElement.getBoundingClientRect();cv.width=r.width;cv.height=r.height;draw();}
window.addEventListener('resize',resize);

// build tabs + legend
const tabs=document.getElementById('tabs');
for(const k of Object.keys(DATA)){const b=document.createElement('span');b.className='tab';b.dataset.l=k;
  b.textContent=`${k} (${DATA[k].nodes.length})`;b.onclick=()=>setLevel(k);tabs.appendChild(b);}
const lg=document.getElementById('legend');
for(const k in CAT){lg.innerHTML+=`<div><span class="sw" style="background:${CAT[k].c}"></span>${k}</div>`;}
resize(); setLevel(level);
</script></body></html>"""


def build_levels(which):
    import reference as ref                                   # noqa: E402
    from lower import lower_l1_to_l2, lower_l2_to_l3, fuse_a   # noqa: E402
    from bfv_mul_ref import MulParams, encrypt                # noqa: E402
    from bfv_relin_ref import small_poly, keygen_relin        # noqa: E402
    pp = MulParams(N=16, LQ=3, LP=3, LB=4)
    rng = random.Random(4); s = small_poly(pp.N, rng)
    if which == "tensor":
        from builders import build_tensor
        l1, _, _ = build_tensor(pp, encrypt(pp, [rng.randrange(pp.t) for _ in range(pp.N)], s, rng),
                                encrypt(pp, [rng.randrange(pp.t) for _ in range(pp.N)], s, rng))
        title = "tensor"
    else:
        from builders import build_ct_mul_rns
        evk0, evk1 = keygen_relin(pp, s, rng)
        ctA = encrypt(pp, [rng.randrange(pp.t) for _ in range(pp.N)], s, rng)
        ctB = encrypt(pp, [rng.randrange(pp.t) for _ in range(pp.N)], s, rng)
        l1, _, _, _ = build_ct_mul_rns(pp, ctA, ctB, evk0, evk1)
        title = "ct_mul"
    l2 = lower_l1_to_l2(l1); l3 = lower_l2_to_l3(l2)
    l2f = fuse_a(l2); l3f = lower_l2_to_l3(l2f)               # stage 5 — Fusion A, and its int8 form
    levels = {}
    for tag, prog in (("L1", l1), ("L2", l2), ("L3", l3), ("L2+FusionA", l2f), ("L3+FusionA", l3f)):
        levels[tag] = {"nodes": layout(program_graph(prog))}
    return title, levels


if __name__ == "__main__":
    which = "tensor" if "--tensor" in sys.argv else "ct_mul"
    title, levels = build_levels(which)
    html = _HTML.replace("__DATA__", json.dumps(levels, separators=(",", ":"))).replace("__TITLE__", f"MLIR DAG — {title}")
    out = os.path.join(os.path.dirname(__file__), "mlir_dag.html")
    with open(out, "w") as f:
        f.write(html)
    sizes = " / ".join(f"{t}:{len(levels[t]['nodes'])}" for t in levels)
    print(f"wrote {out}  ({os.path.getsize(out) // 1024} KB)   {title} nodes {sizes}")

    if "--serve" in sys.argv:
        import http.server, socketserver, functools
        port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
        d = os.path.dirname(out) or "."
        class H(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):                                 # "/" -> the DAG page
                if self.path in ("/", ""):
                    self.path = "/mlir_dag.html"
                return super().do_GET()
            def log_message(self, *a):                        # quiet
                pass
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("0.0.0.0", port), functools.partial(H, directory=d)) as httpd:
            print(f"serving MLIR DAG at  http://0.0.0.0:{port}/   (also /mlir_dag.html) — Ctrl-C to stop")
            httpd.serve_forever()
    else:
        print("open mlir_dag.html in a browser  ·  or:  python3 mlir_viz.py --serve --port 8000")
