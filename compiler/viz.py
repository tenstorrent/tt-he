# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""viz.py — ONE app for the HE compiler: DAG (L1/L2/L3/+FusionA) + schedule Gantt, tabs to switch.

Edit the PARAMETERS block below, then just run `python3 viz.py`: it builds the ct_mul with those
params, lowers it (L1→L2→L3, optional four-step NTT, Fusion A), schedules it across the core counts
you list (reuse-aware), writes one self-contained page, and serves it — open the printed URL and flip
between views with the top tabs. No flags needed.
"""
from __future__ import annotations
import os, sys, json, random, time, http.server, socketserver, functools

import _paths                                     # noqa: F401  (repo layout; see _paths.py)

# ============================== PARAMETERS (edit me) ============================== #
N        = 2**8                       # ring degree (power of two)
LQ       = 2                        # ciphertext-modulus limbs (Q)
LP       = 2                        # relin special-base limbs (P)
LB       = 3                        # rescale aux-base limbs (S);  >= LQ+1
T_PT     = 12289                    # plaintext modulus t
DNUM     = 1                        # hybrid key-switch digits (1 .. LQ)
NTT      = "auto"                  # four-step NTT factorization (the 'decompose' param):
                                    #   None       -> single DFT matmul (no four-step)
                                    #   "auto"     -> split N as (n1,n2) automatically
                                    #   (n1, n2)   -> explicit factors, n1*n2 == N
TILE     = 32                       # matrix-engine tile size: matmuls tiled to TILE x TILE (32 = HW tile)
FUSION_A = ("ntt", "baseconv")      # enabled Fusion-A strategies (() to disable)
CORES    = [8, 16, 32, 64]          # core counts to schedule (Gantt sub-tabs)
REUSE    = True                     # residency / reuse-aware scheduling
MAX_NODES = None                   # skip laying out / embedding any view above this op count; None = no limit
VALIDATE = None                     # decrypt-check the graph (needs real keygen/encrypt — O(N^2)).
                                    #   None -> auto (only when N<=256); True/False to force.
                                    #   When off, keys/data are DUMMY zeros: the op-graph STRUCTURE is
                                    #   identical (all the scheduler/DAG/Gantt need) and the expensive
                                    #   crypto is skipped — the big win at large N.
PORT     = 8000                     # serve here
# ================================================================================= #

from ir import outputs                                        # noqa: E402
from mlir_viz import program_graph, layout                    # noqa: E402  (DAG extraction + layered layout)
from sched_viz import catcol                                  # noqa: E402  (op -> category,color)
from lower import lower_l1_to_l2, lower_l2_to_l3, fuse_a       # noqa: E402
from builders import build_ct_mul_rns, recon_poly             # noqa: E402
from bfv_mul_ref import MulParams, encrypt, decrypt           # noqa: E402
from bfv_relin_ref import small_poly, keygen_relin, negacyclic_mul  # noqa: E402
from bfv_relin_dnum import keygen_relin_dnum                  # noqa: E402
import schedule as sched                                      # noqa: E402


def _factors(n, tile=None):
    """auto multi-factor split of n=2^k into factors of size TILE (=32) so matmuls stay ≤ tile:
    e.g. 2^15 -> (32,32,32), 2^13 -> (32,32,8), 2^10 -> (32,32), 2^4 -> (16,)."""
    tile = tile or TILE
    k = n.bit_length() - 1; e = tile.bit_length() - 1        # n=2^k, tile=2^e
    qd, r = divmod(k, e)
    fs = [tile] * qd + ([1 << r] if r else [])
    return tuple(fs) if fs else (n,)


def build():
    START = time.time()
    def log(msg):
        print(f"  [{time.time()-START:6.1f}s] {msg}", flush=True)

    factors = None if NTT is None else (_factors(N) if NTT == "auto" else tuple(NTT))
    if factors:
        p = 1
        for f in factors:
            p *= f
        if p != N:
            raise ValueError(f"NTT factors {factors} must multiply to N={N}")
    cap = float("inf") if MAX_NODES is None else MAX_NODES
    validate = VALIDATE if VALIDATE is not None else (N <= 256)
    log(f"params: N={N} L=({LQ},{LP},{LB}) t={T_PT} dnum={DNUM} "
        f"NTT={'four-step '+str(factors) if factors else 'single-DFT'} tile={TILE} fuseA={list(FUSION_A)} "
        f"validate={validate} max_nodes={MAX_NODES}")

    pp = MulParams(N=N, LQ=LQ, LP=LP, LB=LB, t=T_PT)
    rng = random.Random(4); s = None
    if validate:                                             # real keys+data (needed only for decrypt-check)
        log("keygen + encrypt (real — for decrypt check) ...")
        s = small_poly(N, rng)
        if DNUM > 1:
            evks, groups = keygen_relin_dnum(pp, s, rng, DNUM); ks = dict(ks=(evks, groups))
        else:
            e0, e1 = keygen_relin(pp, s, rng); ks = dict(evk0=e0, evk1=e1)
        m1 = [rng.randrange(pp.t) for _ in range(N)]; m2 = [rng.randrange(pp.t) for _ in range(N)]
        ct1, ct2 = encrypt(pp, m1, s, rng), encrypt(pp, m2, s, rng)
    else:                                                    # DUMMY zeros — same graph STRUCTURE, skips O(N^2) crypto
        log("structural mode: dummy zero keys/data (no keygen/encrypt) ...")
        z = lambda: [0] * N
        ct1 = ct2 = (z(), z())
        if DNUM > 1:
            from bfv_relin_dnum import q_groups
            groups = q_groups(LQ, DNUM)
            evks = [([z() for _ in pp.QP], [z() for _ in pp.QP]) for _ in groups]
            ks = dict(ks=(evks, groups))
        else:
            ks = dict(evk0=[z() for _ in pp.QP], evk1=[z() for _ in pp.QP])

    log("building L1 (tensor→rescale→relin→add) ...")
    l1, inputs, o0n, o1n = build_ct_mul_rns(pp, ct1, ct2, **ks)
    log(f"L1: {len(l1.ops)} ops")
    log("lowering L1→L2 (four-step NTT + tiling) ...")
    l2 = lower_l1_to_l2(l1, factors=factors, T=TILE)
    log(f"L2: {len(l2.ops)} ops")
    levels = [("L1", l1), ("L2", l2)]
    l3 = l3a = None
    if len(l2.ops) > cap:                                    # L3 is ~5× L2 — skip the expensive int8 lowering
        log(f"L2 ({len(l2.ops)}) > MAX_NODES={MAX_NODES}: SKIPPING L3 int8 lowering + Fusion A + Gantt "
            f"(int8 would be ~5× bigger, unrenderable) — lower N/LQ or raise/None MAX_NODES")
    else:
        log("lowering L2→L3 (int8) ...")
        l3 = lower_l2_to_l3(l2); log(f"L3: {len(l3.ops)} ops   ·   Fusion A ...")
        l2a = fuse_a(l2, FUSION_A) if FUSION_A else l2
        l3a = lower_l2_to_l3(l2a); log(f"L3+FusionA: {len(l3a.ops)} ops")
        levels += [("L3", l3), ("L2+FusionA", l2a), ("L3+FusionA", l3a)]

    correct = "not checked"
    if validate and l3a is not None and N <= 256:
        log("decrypt check (small N) ...")
        import ir as _ir
        env = _ir.run(l3a, inputs)
        o0 = recon_poly([env[n] for n in o0n], pp.Q); o1 = recon_poly([env[n] for n in o1n], pp.Q)
        correct = "OK" if decrypt(pp, [o0, o1], s) == negacyclic_mul(m1, m2, pp.t) else "WRONG"
        log(f"decrypt {correct}")

    dag, skipped = {}, []
    for name, prog in levels:
        if len(prog.ops) > cap:
            skipped.append(f"{name}({len(prog.ops)})"); continue        # too big to lay out / embed
        t = time.time(); dag[name] = {"nodes": layout(program_graph(prog))}
        log(f"DAG {name}: laid out {len(prog.ops)} nodes ({time.time()-t:.1f}s)")
    if skipped:
        log(f"SKIPPED DAG (> MAX_NODES={MAX_NODES}): {', '.join(skipped)}  — raise MAX_NODES or lower N/LQ to view")

    gantt = {"nodes": [], "sched": {}}
    if l3a is not None and len(l3a.ops) <= cap:
        prod_of = {}
        for i, op in enumerate(l3a.ops):
            for v in outputs(op):
                prod_of[v.name] = i
        gnodes = []
        for i, op in enumerate(l3a.ops):
            outs = outputs(op); cat, col = catcol(op.opcode)
            gnodes.append({"i": i, "o": op.opcode, "c": cat, "col": col,
                           "r": (outs[0].name if outs else ""),
                           "in": [prod_of[o.name] for o in op.operands if o.name in prod_of]})
        scheds = {}
        for P in CORES:
            t = time.time(); sc = sched.schedule(l3a, P, reuse=REUSE)
            tier = {(d, j): t2 for (d, j, t2) in sc["edges"]}
            it = [[("noc" if tier.get((d, i)) == "noc" else "sram") for d in gnodes[i]["in"]] for i in range(len(gnodes))]
            scheds[str(P)] = {"core": sc["core"], "start": [round(x, 2) for x in sc["start"]],
                              "end": [round(x, 2) for x in sc["end"]], "intier": it,
                              "makespan": round(sc["makespan"], 1), "used": sc["cores_used"],
                              "reuse": 100 * sc["reuse_hits"] // max(1, sc["accesses"]),
                              "fusedB": 100 * sc["fused"] // max(1, sc["n_edges"])}
            log(f"scheduled P={P}: makespan {sc['makespan']:.0f}us, {sc['cores_used']} cores, "
                f"reuse {scheds[str(P)]['reuse']}%, FusionB {scheds[str(P)]['fusedB']}% ({time.time()-t:.1f}s)")
        gantt = {"nodes": gnodes, "sched": scheds}
    elif l3a is not None:
        log(f"SKIPPED Gantt ({len(l3a.ops)} ops > MAX_NODES={MAX_NODES})")

    meta = dict(N=N, LQ=LQ, LP=LP, LB=LB, t=T_PT, dnum=DNUM,
                ntt=(f"four-step {factors}" if factors else "single-DFT"), tile=TILE,
                fusionA=list(FUSION_A), reuse=REUSE, correct=correct,
                l1=len(l1.ops), l2=len(l2.ops),
                l3=(len(l3.ops) if l3 is not None else None),
                l3a=(len(l3a.ops) if l3a is not None else None))
    return {"dag": dag, "gantt": gantt, "meta": meta}


# --------------------------------------------------------------------------- #
_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>ct_mul — DAG + schedule</title><style>
 *{box-sizing:border-box} html,body{margin:0;height:100%;font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;background:#0e1116;color:#d7dde5}
 #top{position:fixed;top:0;left:0;right:0;height:46px;display:flex;align-items:center;gap:6px;padding:0 12px;background:#161b22;border-bottom:1px solid #30363d;z-index:10;flex-wrap:wrap}
 #top b{color:#eaf0f7;font-size:14px;margin-right:6px}
 .tab{padding:4px 9px;border:1px solid #30363d;border-radius:6px;background:#0e1116;color:#9aa7b4;cursor:pointer;font-size:11px}
 .tab.on{background:#1f6feb;border-color:#1f6feb;color:#fff}
 .pt{padding:3px 7px;border:1px solid #30363d;border-radius:5px;background:#0e1116;color:#9aa7b4;cursor:pointer;font-size:10px}
 .pt.on{background:#238636;border-color:#238636;color:#fff}
 #meta{margin-left:auto;color:#7d8998;font-size:11px}
 button.act{padding:4px 9px;border:1px solid #30363d;border-radius:6px;background:#0e1116;color:#9aa7b4;cursor:pointer;font-size:11px}
 #wrap{position:absolute;top:46px;left:0;right:322px;bottom:0}
 canvas{display:block;width:100%;height:100%;cursor:grab} canvas.drag{cursor:grabbing}
 #side{position:fixed;top:46px;right:0;bottom:0;width:322px;background:#161b22;border-left:1px solid #30363d;overflow:auto;padding:12px}
 #side .op{font-size:15px;font-weight:700}.kv{margin:8px 0;padding:8px;background:#0e1116;border:1px solid #30363d;border-radius:6px;word-break:break-all}
 .kv .k{color:#7d8998;font-size:11px;text-transform:uppercase}.lnk{color:#58a6ff;cursor:pointer}.lnk:hover{text-decoration:underline}
 #search{padding:4px 8px;background:#0e1116;border:1px solid #30363d;border-radius:6px;color:#d7dde5;width:150px;font-size:11px}
 #legend{position:fixed;left:10px;bottom:8px;background:#161b22cc;border:1px solid #30363d;border-radius:8px;padding:5px 8px;font-size:10px;z-index:9}
 #legend span.sw{width:9px;height:9px;border-radius:2px;display:inline-block;margin-right:3px} .muted{color:#7d8998}
 #ptabs{display:none;gap:4px} #ptabs.show{display:inline-flex}
</style></head><body>
<div id="top"><b>ct_mul</b><span id="tabs"></span><span id="ptabs"></span>
 <button class="act" onclick="fit()">fit</button>
 <input id="search" placeholder="search op/value"><span id="meta"></span></div>
<div id="wrap"><canvas id="cv"></canvas></div>
<div id="legend"></div>
<div id="side"></div>
<script>
const DATA=__DATA__;
const DCAT={matmul:'#f0883e',matmul_wide:'#e3b341',elem:'#3fb950',ntt:'#a371f7',bconv:'#db61a2',int8:'#e3b341',slice:'#39c5cf',struct:'#6e7681',const:'#8b949e',input:'#1f6feb',other:'#768390'};
const GCAT={matmul:'#f0883e',matmul_wide:'#e3b341',elem:'#3fb950',decompose:'#a371f7',recombine:'#db61a2',slice:'#39c5cf',struct:'#6e7681',const:'#8b949e',other:'#768390'};
const NTTC=['rns.ntt','rns.intt'],MMC=['tile.matmul_modq','tile.matmul_wide'],ELC=['rns.mul','rns.add','rns.sub','tile.mul_modq','tile.add_modq','tile.sub_modq'];
function dcat(o){if(o==='input')return'input';if(MMC.includes(o))return'matmul';if(NTTC.includes(o))return'ntt';if(ELC.includes(o))return'elem';if(o==='rns.base_convert')return'bconv';if(['tile.decompose','tile.recombine_modq','tile.recombine_wide'].includes(o))return'int8';if(o==='tile.fixed_slice')return'slice';if(o.endsWith('.const'))return'const';return'struct';}
function dcol(o){return DCAT[dcat(o)]||'#768390';}

const HASG=(DATA.gantt.nodes&&DATA.gantt.nodes.length>0);
let MODE=Object.keys(DATA.dag)[0]||(HASG?'Gantt':null), P=Object.keys(DATA.gantt.sched)[0]||null;
let scale=1,scaleX=1,panX=0,panY=0,drag=null,sel=-1,hover=-1,hl=new Set();
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const NW=150,NH=34,TOP=26,ROWMIN=7;
const isG=()=>MODE==='Gantt';
function nodes(){ return isG()?DATA.gantt.nodes:DATA.dag[MODE].nodes; }
function sc(){ return DATA.gantt.sched[P]; }
let succ=[];
function rebuildSucc(){ const nn=nodes(); succ=nn.map(()=>[]); for(const n of nn){const ins=isG()?n.in:n.n; for(const d of ins) if(succ[d])succ[d].push(n.i);} }

function setMode(m){ MODE=m; sel=-1; hl.clear();
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.dataset.m===m));
  document.getElementById('ptabs').classList.toggle('show',isG());
  rebuildSucc(); side(-1); fit(); }
function setP(p){ P=p; sel=-1; document.querySelectorAll('.pt').forEach(t=>t.classList.toggle('on',t.dataset.p===p));
  rebuildSucc(); hdr(); fit(); }
function hdr(){ const m=DATA.meta; let g=isG()?` · P=${P} makespan ${sc().makespan}us · ${sc().used}c · reuse ${sc().reuse}% · FusionB ${sc().fusedB}%`:'';
  document.getElementById('meta').textContent=`N=${m.N} L=(${m.LQ},${m.LP},${m.LB}) t=${m.t} dnum=${m.dnum} · NTT ${m.ntt} · tile ${m.tile}×${m.tile} · fuseA[${m.fusionA}] · decrypt ${m.correct}`+g; }

// ---------- transforms ----------
function ncores(){ return Math.max(...sc().core)+1; }
function rowPix(){ return Math.max(ROWMIN,(cv.height-TOP-6)/ncores()); }
function fit(){ const nn=nodes(),w=cv.width,h=cv.height;
  if(isG()){ scaleX=(w-70)/(sc().makespan||1); panX=60; panY=TOP; }
  else{ let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9; for(const n of nn){x0=Math.min(x0,n.x);y0=Math.min(y0,n.y);x1=Math.max(x1,n.x+NW);y1=Math.max(y1,n.y+NH);}
    scale=Math.min(w/(x1-x0+80),h/(y1-y0+80),1.4); panX=w/2-(x0+x1)/2*scale; panY=40-y0*scale; }
  hdr(); draw(); }

function draw(){ isG()?drawG():drawD(); }
// ---------- DAG ----------
function W2S(x,y){return[x*scale+panX,y*scale+panY];}
function S2W(x,y){return[(x-panX)/scale,(y-panY)/scale];}
function drawD(){ const w=cv.width,h=cv.height,nn=nodes(); ctx.clearRect(0,0,w,h);
  const dim=sel>=0||hl.size>0;
  ctx.lineWidth=Math.max(.6,scale*.8);
  for(const n of nn){ const [ax,ay]=W2S(n.x+NW/2,n.y);
    for(const p of n.n){ const s2=nn[p],[bx,by]=W2S(s2.x+NW/2,s2.y+NH);
      if((bx<-50&&ax<-50)||(bx>w+50&&ax>w+50))continue;
      const on=dim&&hl.has(n.i)&&hl.has(p); ctx.strokeStyle=on?'#58a6ff':(dim?'#20242b':'#343b45');
      ctx.beginPath();ctx.moveTo(bx,by);ctx.bezierCurveTo(bx,by+(ay-by)*.5,ax,ay-(ay-by)*.5,ax,ay);ctx.stroke(); } }
  const showT=scale>.45; ctx.font='11px ui-monospace';ctx.textBaseline='middle';
  for(const n of nn){ const [sx,sy]=W2S(n.x,n.y),sw=NW*scale,sh=NH*scale; if(sx>w||sy>h||sx+sw<0||sy+sh<0)continue;
    ctx.globalAlpha=dim&&!hl.has(n.i)?.22:1; ctx.fillStyle=dcol(n.o); ctx.strokeStyle=n.i===sel?'#fff':'#0e1116'; ctx.lineWidth=n.i===sel?2:1;
    rr(sx,sy,sw,sh,5*scale);ctx.fill(); if(n.i===sel||n.i===hover)ctx.stroke();
    if(showT){ctx.fillStyle='#0b0e12';const l=n.o.replace(/^(tile|rns)\./,'');ctx.fillText(l.length>18?l.slice(0,17)+'…':l,sx+7,sy+sh*.4);} }
  ctx.globalAlpha=1; }
function rr(x,y,w,h,r){r=Math.max(0,Math.min(r,h/2));ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();}
// ---------- Gantt ----------
function Xt(t){return t*scaleX+panX;} function Yc(c){return c*rowPix()+panY;}
function drawG(){ const w=cv.width,h=cv.height,s=sc(),nn=nodes(),rp=rowPix(),nc=ncores(); ctx.clearRect(0,0,w,h);
  ctx.font='10px ui-monospace';ctx.textBaseline='middle';
  for(let c=0;c<nc;c++){const y=Yc(c);ctx.fillStyle=c%2?'#12161c':'#0e1116';ctx.fillRect(0,y,w,rp); if(rp>=9){ctx.fillStyle='#4d5562';ctx.fillText('c'+c,4,y+rp/2);}}
  ctx.strokeStyle='#20262e';ctx.fillStyle='#5b6470';ctx.textAlign='center';
  const mk=s.makespan,step=nstep(mk/(w/110)); for(let t=0;t<=mk;t+=step){const x=Xt(t);if(x<50||x>w)continue;ctx.beginPath();ctx.moveTo(x,TOP);ctx.lineTo(x,h);ctx.stroke();ctx.fillText(t.toFixed(0)+'us',x,13);}
  ctx.textAlign='left'; const dim=sel>=0;
  for(const n of nn){const i=n.i,c=s.core[i];if(c<0)continue;const x0=Xt(s.start[i]),x1=Math.max(x0+1,Xt(s.end[i])),y=Yc(c);
    if(x1<50||x0>w||y>h||y+rp<TOP)continue; ctx.globalAlpha=dim&&i!==sel&&!nbr(i)?.28:1; ctx.fillStyle=n.col;ctx.fillRect(x0,y+1,x1-x0,rp-2);
    if(i===sel){ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.strokeRect(x0,y+1,x1-x0,rp-2);}
    if(rp>=12&&x1-x0>34){ctx.fillStyle='#0b0e12';ctx.font='9px ui-monospace';ctx.fillText(n.o.replace(/^(tile|rns)\./,'').slice(0,Math.floor((x1-x0)/6)),x0+3,y+rp/2);}}
  ctx.globalAlpha=1;
  if(sel>=0){const arr=(a,b,t)=>{ctx.strokeStyle=t==='noc'?'#f0883e':'#3fb950';ctx.lineWidth=t==='noc'?1.6:1.1;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
      const an=Math.atan2(b[1]-a[1],b[0]-a[0]);ctx.beginPath();ctx.moveTo(b[0],b[1]);ctx.lineTo(b[0]-7*Math.cos(an-.4),b[1]-7*Math.sin(an-.4));ctx.lineTo(b[0]-7*Math.cos(an+.4),b[1]-7*Math.sin(an+.4));ctx.closePath();ctx.fillStyle=ctx.strokeStyle;ctx.fill();};
    nodes()[sel].in.forEach((d,k)=>arr([Xt(s.end[d]),Yc(s.core[d])+rp/2],[Xt(s.start[sel]),Yc(s.core[sel])+rp/2],s.intier[sel][k]));
    for(const u of succ[sel]){const k=nodes()[u].in.indexOf(sel);arr([Xt(s.end[sel]),Yc(s.core[sel])+rp/2],[Xt(s.start[u]),Yc(s.core[u])+rp/2],(s.intier[u]||[])[k]||'sram');}}
}
function nstep(x){const p=Math.pow(10,Math.floor(Math.log10(x||1)));const m=x/p;return(m<1.5?1:m<3?2:m<7?5:10)*p;}
function nbr(i){return sel>=0&&(nodes()[sel].in.includes(i)||succ[sel].includes(i));}

// ---------- interaction ----------
function hit(sx,sy){ const nn=nodes();
  if(isG()){const s=sc(),rp=rowPix(),c=Math.floor((sy-panY)/rp),t=(sx-panX)/scaleX; if(c<0)return -1;
    for(const n of nn){if(s.core[n.i]!==c)continue; if(sx>=Xt(s.start[n.i])-1&&sx<=Math.max(Xt(s.start[n.i])+2,Xt(s.end[n.i])))return n.i;} return -1;}
  const [wx,wy]=S2W(sx,sy); for(let i=nn.length-1;i>=0;i--){const n=nn[i];if(wx>=n.x&&wx<=n.x+NW&&wy>=n.y&&wy<=n.y+NH)return i;} return -1; }
function select(i){ sel=i; hl.clear(); if(i>=0&&!isG()){hl.add(i);for(const p of nodes()[i].n)hl.add(p);for(const c of succ[i])hl.add(c);} side(i); draw(); }
function lnk(i){const n=nodes()[i];const c=isG()?(' · c'+sc().core[i]):'';return `<span class="lnk" onclick="select(${i})">${n.o.replace(/^(tile|rns)\./,'')}${c}</span>`;}
function side(i){ const el=document.getElementById('side');
  if(i<0){el.innerHTML='<span class="muted">click a '+(isG()?'bar':'node')+' to inspect</span>';return;}
  const n=nodes()[i]; let h=`<div class="op" style="color:${isG()?n.col:dcol(n.o)}">${n.o}</div>`;
  if(isG()){const s=sc();h+=`<div class="muted">#${i} · ${n.c}</div><div class="kv"><div class="k">placement</div>core ${s.core[i]} · ${s.start[i]}→${s.end[i]}us · dur ${(s.end[i]-s.start[i]).toFixed(2)}us</div>`;
    h+=`<div class="kv"><div class="k">result</div>${n.r}</div>`;
    h+=`<div class="kv"><div class="k">inputs — green SRAM / orange NoC</div>${n.in.map((d,k)=>lnk(d)+` <span style="color:${s.intier[i][k]==='noc'?'#f0883e':'#3fb950'}">[${s.intier[i][k]}]</span>`).join('<br>')||'<span class=muted>none</span>'}</div>`;
    h+=`<div class="kv"><div class="k">used by (${succ[i].length})</div>${succ[i].slice(0,30).map(lnk).join('<br>')||'<span class=muted>none</span>'}</div>`;
  }else{ h+=`<div class="muted">#${i} · layer ${n.l} · ${dcat(n.o)}</div><div class="kv"><div class="k">result</div>${(n.r||[]).join('<br>')}</div>`;
    h+=`<div class="kv"><div class="k">type</div>tensor&lt;${n.t}&gt;${n.q?' @'+n.q:''}</div>`; if(n.a)h+=`<div class="kv"><div class="k">attrs</div>${n.a.replace(/</g,'&lt;')}</div>`;
    h+=`<div class="kv"><div class="k">operands (${n.n.length})</div>${n.n.map(lnk).join('<br>')||'<span class=muted>none</span>'}</div>`;
    h+=`<div class="kv"><div class="k">used by (${succ[i].length})</div>${succ[i].slice(0,30).map(lnk).join('<br>')||'<span class=muted>none</span>'}</div>`; }
  el.innerHTML=h; }

cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,px:panX,py:panY,mv:false};cv.classList.add('drag');});
window.addEventListener('mousemove',e=>{if(drag){panX=drag.px+(e.clientX-drag.x);panY=drag.py+(e.clientY-drag.y);if(Math.abs(e.clientX-drag.x)+Math.abs(e.clientY-drag.y)>3)drag.mv=true;draw();return;}
  const r=cv.getBoundingClientRect();const id=hit(e.clientX-r.left,e.clientY-r.top);if(id!==hover){hover=id;cv.title=id>=0?(nodes()[id].r||nodes()[id].o):'';if(!isG())draw();}});
window.addEventListener('mouseup',e=>{if(drag&&!drag.mv){const r=cv.getBoundingClientRect();select(hit(e.clientX-r.left,e.clientY-r.top));}drag=null;cv.classList.remove('drag');});
cv.addEventListener('wheel',e=>{e.preventDefault();const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  if(isG()){const t=(mx-panX)/scaleX;scaleX*=Math.exp(-e.deltaY*0.0015);scaleX=Math.max(1e-4,scaleX);panX=mx-t*scaleX;}
  else{const [wx,wy]=S2W(mx,my);scale*=Math.exp(-e.deltaY*0.0012);scale=Math.max(.05,Math.min(3,scale));panX=mx-wx*scale;panY=my-wy*scale;} draw();},{passive:false});
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();hl.clear();
  if(q&&!isG()){let f=-1;for(const n of nodes()){if((n.o+' '+(n.r||[]).join(' ')).toLowerCase().includes(q)){hl.add(n.i);if(f<0)f=n.i;}}if(f>=0){const n=nodes()[f];const r=cv.getBoundingClientRect();panX=r.width/2-(n.x+NW/2)*scale;panY=r.height/2-(n.y+NH/2)*scale;}}
  sel=-1;draw();});
function resize(){const r=cv.parentElement.getBoundingClientRect();cv.width=r.width;cv.height=r.height;draw();}
window.addEventListener('resize',resize);
const tabs=document.getElementById('tabs'); for(const m of Object.keys(DATA.dag)){const b=document.createElement('span');b.className='tab';b.dataset.m=m;b.textContent=m;b.onclick=()=>setMode(m);tabs.appendChild(b);}
if(HASG){const b=document.createElement('span');b.className='tab';b.dataset.m='Gantt';b.textContent='Gantt';b.onclick=()=>setMode('Gantt');tabs.appendChild(b);}
const pt=document.getElementById('ptabs'); for(const p of Object.keys(DATA.gantt.sched)){const b=document.createElement('span');b.className='pt';b.dataset.p=p;b.textContent=p+'c';b.onclick=()=>setP(p);pt.appendChild(b);}
if(!MODE){document.getElementById('side').innerHTML='<span class="muted">all views exceeded MAX_NODES — see the terminal for op counts + schedule stats; lower N/LQ or raise MAX_NODES to view.</span>';document.getElementById('meta').textContent='(nothing to render)';}
function updLegend(){const lg=document.getElementById('legend');lg.innerHTML='';const C=isG()?GCAT:DCAT;for(const k in C)lg.innerHTML+=`<span><span class="sw" style="background:${C[k]}"></span>${k}</span>&nbsp; `;}
const _sm=setMode; setMode=function(m){_sm(m);document.querySelectorAll('.pt').forEach(t=>t.classList.toggle('on',t.dataset.p===P));updLegend();};
resize(); if(MODE) setMode(MODE);
</script></body></html>"""


if __name__ == "__main__":
    data = build()
    m = data["meta"]
    print(f"built ct_mul: N={m['N']} L=({m['LQ']},{m['LP']},{m['LB']}) t={m['t']} dnum={m['dnum']} "
          f"· {m['ntt']} NTT · fuseA={m['fusionA']} · decrypt {m['correct']}")
    print(f"  L1 {m['l1']} · L2 {m['l2']} · L3 {m['l3']} · L3+FusionA {m['l3a']} ops  (None = skipped, > MAX_NODES)")
    html = _HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = os.path.join(os.path.dirname(__file__), "ctmul_viz.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"  wrote {out} ({os.path.getsize(out)//1024} KB)")
    d = os.path.dirname(out) or "."
    class H(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", ""):
                self.path = "/ctmul_viz.html"
            return super().do_GET()
        def log_message(self, *a): pass
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), functools.partial(H, directory=d)) as httpd:
        print(f"\n  serving at  http://0.0.0.0:{PORT}/   — tabs: DAG L1/L2/L3/+FusionA · Gantt (Ctrl-C to stop)")
        httpd.serve_forever()
