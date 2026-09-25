#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""trace_riscs.py — SELF-CONTAINED per-RISC timeline for one ct_mul (MLIR-viz style).

Renders the device profiler KERNEL-wrapper zones as a Gantt chart: x = time (µs),
y = one lane per (core, RISC), bars = that RISC's activity, colored by ROLE so the
data-movement-vs-compute story is visible at a glance:
    BRISC   = reader  (DRAM -> L1)      blue
    NCRISC  = writer  (L1 -> DRAM)      orange
    TRISC_1 = math                      green
    TRISC_0 = unpack / TRISC_2 = pack   grey-green
Reader (blue) bars being longer/denser than math (green) = DRAM-read-bound.

Auto-crops to ONE mul (the last dense cluster, dropping stale/contaminated entries),
or pass a window:  python3 trace_riscs.py profile_log_device.csv [out.html] [lo_ms hi_ms]
No libraries, no server — open the .html in a browser (pan=drag, zoom=wheel, hover=details).
"""
import csv, sys, json, collections

ROLE = {"BRISC":("reader (DRAM->L1)","#4e79a7"), "NCRISC":("writer (L1->DRAM)","#f28e2b"),
        "TRISC_0":("unpack","#8cae7a"), "TRISC_1":("math","#59a14f"), "TRISC_2":("pack","#b0c4a0")}
ORDER = {"BRISC":0,"TRISC_0":1,"TRISC_1":2,"TRISC_2":3,"NCRISC":4}

def parse(path):
    st=collections.defaultdict(list); z=[]
    with open(path) as f:
        f.readline(); r=csv.reader(f); next(r)
        for x in r:
            if len(x)<12: continue
            cx,cy,ri=int(x[1]),int(x[2]),x[3].strip(); t=int(x[5]); nm=x[10].strip(); ty=x[11].strip()
            if not nm.endswith("-KERNEL"): continue
            k=(cx,cy,ri)
            if ty=="ZONE_START": st[k].append(t)
            elif ty=="ZONE_END" and st[k]:
                s=st[k].pop(); z.append((cx,cy,ri,s,t))
    return z

def pick_window(z, lo_ms, hi_ms):
    starts=sorted(e[3] for e in z); t0=starts[0]
    if lo_ms is not None:
        return t0+int(lo_ms*1e6), t0+int(hi_ms*1e6)
    # auto: last dense cluster (>5ms gaps delimit)
    clusters=[]; cur=[starts[0]]
    for a,b in zip(starts,starts[1:]):
        if b-a>5_000_000: clusters.append((cur[0],cur[-1])); cur=[b]
        else: cur.append(b)
    clusters.append((cur[0],cur[-1]))
    s,e=max(clusters,key=lambda c:c[1]-c[0])   # widest cluster = one full mul
    return s, e+2_000_000

def build(path, lo_ms=None, hi_ms=None):
    z=parse(path); lo,hi=pick_window(z,lo_ms,hi_ms)
    z=[e for e in z if e[3]>=lo and e[3]<=hi]
    lanes=sorted({(cx,cy,ri) for cx,cy,ri,s,t in z}, key=lambda l:(l[0],l[1],ORDER.get(l[2],9)))
    li={l:i for i,l in enumerate(lanes)}
    lbl=[f"({l[0]},{l[1]}) {l[2].replace('TRISC_','T').replace('BRISC','B-rd').replace('NCRISC','NC-wr')}" for l in lanes]
    roles=sorted(ROLE.items(), key=lambda kv:ORDER[kv[0]])
    rnames=[v[0] for k,v in roles]; rcol=[v[1] for k,v in roles]; ridx={k:i for i,(k,v) in enumerate(roles)}
    z.sort(key=lambda e:e[3])
    EV=[[li[(cx,cy,ri)], round((s-lo)/1000.0,3), round(max((t-s)/1000.0,0.05),3), ridx[ri]] for cx,cy,ri,s,t in z]
    span=round((hi-lo)/1000.0,3)
    # per-role busy (summed) for the header
    busy=collections.defaultdict(float)
    for cx,cy,ri,s,t in z: busy[ri]+=(t-s)/1e6
    return {"lanes":lbl,"roles":rnames,"rcol":rcol,"ev":EV,"span":span,
            "busy":{ROLE[k][0]:round(busy[k],1) for k in ROLE if k in busy}}

HTML=r"""<!doctype html><html><head><meta charset="utf-8"><title>ct_mul device trace (per-RISC)</title><style>
*{box-sizing:border-box} html,body{margin:0;height:100%;font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;background:#0e1116;color:#d7dde5;overflow:hidden}
#top{display:flex;align-items:center;gap:10px;padding:8px 12px;border-bottom:1px solid #21262d;flex-wrap:wrap}
#top b{color:#eaf0f7;font-size:13px} #top .meta{color:#7d8998}
.lg{display:inline-flex;align-items:center;gap:5px;padding:3px 8px;border:1px solid #30363d;border-radius:6px;cursor:pointer;user-select:none}
.lg.off{opacity:.3} .sw{width:10px;height:10px;border-radius:2px;display:inline-block}
button.act{margin-left:auto;padding:4px 10px;border:1px solid #30363d;border-radius:6px;background:#0e1116;color:#9aa7b4;cursor:pointer}
#wrap{position:absolute;top:64px;left:0;right:0;bottom:0} canvas{display:block;cursor:grab} canvas.drag{cursor:grabbing}
#hud{position:fixed;right:12px;top:74px;background:#161b22ee;border:1px solid #30363d;border-radius:6px;padding:8px 10px;max-width:320px;pointer-events:none;display:none}
#hud .o{font-size:13px;color:#eaf0f7} #hud .k{color:#7d8998} #note{padding:2px 12px;color:#c9a35b;font-size:11px}
</style></head><body>
<div id="top"><b>ct_mul device trace — one multiply, per-RISC activity</b><span class="meta" id="meta"></span><span id="leg"></span>
<button class="act" onclick="fit()">reset view</button></div>
<div id="note"></div>
<div id="wrap"><canvas id="cv"></canvas></div><div id="hud"></div>
<script>
const D=__DATA__;
const cv=document.getElementById('cv'),cx=cv.getContext('2d');
const LM=150, TOP=24, RH=12;
let scale=1, px=LM, py=TOP;
const on=D.roles.map(_=>true);
document.getElementById('meta').textContent=`${D.ev.length.toLocaleString()} kernel-runs · ${D.lanes.length} lanes · span ${D.span.toFixed(0)}µs`;
document.getElementById('note').textContent="reader (blue) busy ≫ math (green) ⇒ math cores stall waiting on DRAM reads (DRAM-read-bound).  busy(ms): "+Object.entries(D.busy).map(([k,v])=>k+" "+v).join(" · ");
const leg=document.getElementById('leg');
D.roles.forEach((b,i)=>{const s=document.createElement('span');s.className='lg';s.innerHTML=`<span class="sw" style="background:${D.rcol[i]}"></span>${b}`;s.onclick=()=>{on[i]=!on[i];s.classList.toggle('off',!on[i]);draw();};leg.appendChild(s);});
function lo(t){let a=0,b=D.ev.length;while(a<b){const m=(a+b)>>1;if(D.ev[m][1]+D.ev[m][2]<t)a=m+1;else b=m;}return a;}
function fit(){scale=(cv.width-LM-16)/Math.max(D.span,1);px=LM;py=TOP;draw();}
function draw(){
  cx.clearRect(0,0,cv.width,cv.height);
  const t0=(LM-px)/scale, t1=(cv.width-px)/scale;
  cx.fillStyle='#0e1116';cx.fillRect(0,0,cv.width,TOP);
  cx.fillStyle='#7d8998';cx.font='10px ui-monospace';cx.textAlign='center';
  const step=niceStep((t1-t0)/8);
  for(let t=Math.ceil(t0/step)*step;t<t1;t+=step){const X=px+t*scale;cx.strokeStyle='#171c23';cx.beginPath();cx.moveTo(X,TOP);cx.lineTo(X,cv.height);cx.stroke();cx.fillText(t.toFixed(t1-t0<40?1:0)+'µs',X,15);}
  const i0=lo(t0);
  for(let i=i0;i<D.ev.length;i++){const e=D.ev[i];if(e[1]>t1)break;if(!on[e[3]])continue;
    const y=TOP+e[0]*RH+py-TOP;if(y+RH<TOP||y>cv.height)continue;
    const X=px+e[1]*scale,W=Math.max(e[2]*scale,0.6);
    cx.fillStyle=D.rcol[e[3]];cx.fillRect(X,y+1,W,RH-2);}
  cx.fillStyle='#0e1116';cx.fillRect(0,TOP,LM,cv.height);
  cx.textAlign='left';cx.font='9px ui-monospace';
  for(let l=0;l<D.lanes.length;l++){const y=TOP+l*RH+py-TOP;if(y+RH<TOP||y>cv.height)continue;
    cx.fillStyle=(l&1)?'#7d8998':'#9aa7b4';cx.fillText(D.lanes[l],4,y+RH-3);}
  cx.strokeStyle='#21262d';cx.beginPath();cx.moveTo(LM,0);cx.lineTo(LM,cv.height);cx.stroke();
}
function niceStep(x){const p=Math.pow(10,Math.floor(Math.log10(x)));const f=x/p;return (f<1.5?1:f<3.5?2:f<7.5?5:10)*p;}
function hit(mx,my){if(mx<LM)return -1;const t=(mx-px)/scale;const lane=Math.floor((my-TOP-(py-TOP))/RH);
  const i0=lo(t-0.001);for(let i=i0;i<D.ev.length;i++){const e=D.ev[i];if(e[1]>t)break;if(e[0]!==lane||!on[e[3]])continue;if(t>=e[1]&&t<=e[1]+e[2])return i;}return -1;}
const hud=document.getElementById('hud');let drag=null;
cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,px,py};cv.classList.add('drag');});
addEventListener('mouseup',()=>{drag=null;cv.classList.remove('drag');});
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect();const mx=e.clientX-r.left,my=e.clientY-r.top;
  if(drag){px=drag.px+(e.clientX-drag.x);py=drag.py+(e.clientY-drag.y);draw();return;}
  const i=hit(mx,my);if(i<0){hud.style.display='none';return;}
  const e2=D.ev[i];hud.style.display='block';hud.style.left=(e.clientX+14)+'px';hud.style.top=(e.clientY+14)+'px';
  hud.innerHTML=`<div class="o" style="color:${D.rcol[e2[3]]}">${D.roles[e2[3]]}</div><div class="k">${D.lanes[e2[0]]}</div><div>start <b>${e2[1].toFixed(2)}</b> µs · dur <b>${e2[2].toFixed(2)}</b> µs</div>`;});
cv.addEventListener('mouseleave',()=>{hud.style.display='none';});
cv.addEventListener('wheel',e=>{e.preventDefault();const r=cv.getBoundingClientRect();const mx=e.clientX-r.left;const t=(mx-px)/scale;const f=e.deltaY<0?1.2:1/1.2;scale*=f;px=mx-t*scale;draw();},{passive:false});
function resize(){cv.width=cv.parentElement.clientWidth;cv.height=cv.parentElement.clientHeight;draw();}
addEventListener('resize',resize);resize();fit();
</script></body></html>"""

def main():
    path=sys.argv[1]; out=sys.argv[2] if len(sys.argv)>2 else "trace_riscs.html"
    lo_ms=float(sys.argv[3]) if len(sys.argv)>4 else None
    hi_ms=float(sys.argv[4]) if len(sys.argv)>4 else None
    data=build(path, lo_ms, hi_ms)
    open(out,"w").write(HTML.replace("__DATA__", json.dumps(data,separators=(",",":"))))
    print(f"wrote {out}  ({len(data['ev'])} kernel-runs, {len(data['lanes'])} lanes, span {data['span']:.0f} µs)")
    print("per-role busy (ms, summed over cores):", data["busy"])

main()
