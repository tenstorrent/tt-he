#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""trace_viz.py — SELF-CONTAINED interactive timeline for the HE-mul device profiler.

Same style as compiler/mlir_viz.py: parses profile_log_device.csv and writes ONE
self-contained HTML (no libraries, no server) that renders the trace on a <canvas> with
pan/zoom/hover + viewport culling (handles 100k+ ops). Just open the .html in a browser —
no port-forward needed.  x = time (µs), y = core/RISC lane, bar = one on-device operation.

  python3 trace_viz.py profile_log_device.csv [out.html]
"""
import csv, sys, json, collections

OP = {"fused_total":"matmul-A (NTT stage-1)","fusedC_total":"matmul-C (NTT stage-2)",
      "modmul_compute":"twiddle modmul","modmul_sfpi_full":"pointwise modmul",
      "fbc_total":"base conversion (FBC)","modadd_reduce":"modular add","modsub_reduce":"modular sub"}
BUCKET={"fused_total":0,"fusedC_total":0,"modmul_compute":0,"transpose_tile":0,
        "modmul_sfpi_full":1,"fbc_total":2,"modadd_reduce":3,"modsub_reduce":3}   # 0 NTT,1 pw,2 FBC,3 vec,4 kernel
BNAME=["NTT","pointwise","FBC","vec","kernel"]
BCOL =["#4e79a7","#f28e2b","#59a14f","#e15759","#8a8f98"]
SKIP={"ph_recombine","ph_matmul","ph_decompose","fbc_phase1","fbc_phase2"}

def parse(path):
    rows=[]
    with open(path) as f:
        f.readline(); r=csv.reader(f); next(r)
        for x in r:
            if len(x)>=12: rows.append(x)
    st=collections.defaultdict(list); z=[]
    for x in sorted(rows,key=lambda x:int(x[5])):
        cx,cy,risc=int(x[1]),int(x[2]),x[3]; t=int(x[5]); nm=x[10].strip(); ty=x[11]
        k=(cx,cy,risc)
        if ty=="ZONE_START": st[k].append((nm,t))
        elif ty=="ZONE_END" and st[k]:
            n,s=st[k].pop()
            if n.endswith("-FW") or n in SKIP: continue
            z.append((cx,cy,risc,n,s,t-s))
    t0=min(e[4] for e in z)
    return [(cx,cy,risc,nm,s-t0,d) for cx,cy,risc,nm,s,d in z]

def build(path):
    z=parse(path)
    rn=lambda r:r.replace('TRISC_','T').replace('BRISC','B').replace('NCRISC','NC')
    lanes=sorted({(cx,cy,risc) for cx,cy,risc,nm,s,d in z}, key=lambda l:(l[0],l[1],l[2]))
    lane_i={l:i for i,l in enumerate(lanes)}
    lane_lbl=[f"({l[0]},{l[1]}) {rn(l[2])}" for l in lanes]
    opnames=[]; opidx={}
    def opi(nm):
        fn=OP.get(nm,"kernel" if nm.endswith("-KERNEL") else nm)
        if fn not in opidx: opidx[fn]=len(opnames); opnames.append(fn)
        return opidx[fn]
    z.sort(key=lambda e:e[4])                                  # by start (for x-culling binary search)
    EV=[[lane_i[(cx,cy,risc)], round(s/1000.0,3), round(max(d/1000.0,0.001),3),
         BUCKET.get(nm,4), opi(nm)] for cx,cy,risc,nm,s,d in z]
    span=max((s+d)/1000.0 for cx,cy,risc,nm,s,d in z)
    return {"lanes":lane_lbl,"ops":opnames,"bnames":BNAME,"bcol":BCOL,"ev":EV,"span":round(span,3)}

HTML=r"""<!doctype html><html><head><meta charset="utf-8"><title>HE-mul device trace</title><style>
*{box-sizing:border-box} html,body{margin:0;height:100%;font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;background:#0e1116;color:#d7dde5;overflow:hidden}
#top{display:flex;align-items:center;gap:10px;padding:8px 12px;border-bottom:1px solid #21262d;flex-wrap:wrap}
#top b{color:#eaf0f7;font-size:13px} #top .meta{color:#7d8998}
.lg{display:inline-flex;align-items:center;gap:5px;padding:3px 8px;border:1px solid #30363d;border-radius:6px;cursor:pointer;user-select:none}
.lg.off{opacity:.35} .sw{width:10px;height:10px;border-radius:2px;display:inline-block}
button.act{margin-left:auto;padding:4px 10px;border:1px solid #30363d;border-radius:6px;background:#0e1116;color:#9aa7b4;cursor:pointer}
#wrap{position:absolute;top:41px;left:0;right:0;bottom:0} canvas{display:block;cursor:grab} canvas.drag{cursor:grabbing}
#hud{position:fixed;right:12px;top:52px;background:#161b22ee;border:1px solid #30363d;border-radius:6px;padding:8px 10px;max-width:320px;pointer-events:none;display:none}
#hud .o{font-size:13px;color:#eaf0f7} #hud .k{color:#7d8998}
</style></head><body>
<div id="top"><b>HE-mul device trace</b><span class="meta" id="meta"></span><span id="leg"></span>
<button class="act" onclick="fit()">reset view</button></div>
<div id="wrap"><canvas id="cv"></canvas></div><div id="hud"></div>
<script>
const D=__DATA__;
const cv=document.getElementById('cv'),cx=cv.getContext('2d');
const LM=132, TOP=26, RH=13;               // left margin, top axis, row height
let scale=1, px=LM, py=TOP;                 // scale = px per µs; px,py = pan offsets
const on=D.bnames.map(_=>true);             // bucket visibility
document.getElementById('meta').textContent=`N=8192·L=8·dnum=2·16 cores · ${D.ev.length.toLocaleString()} ops · ${D.lanes.length} lanes · span ${D.span.toFixed(0)}µs`;
const leg=document.getElementById('leg');
D.bnames.forEach((b,i)=>{const s=document.createElement('span');s.className='lg';s.innerHTML=`<span class="sw" style="background:${D.bcol[i]}"></span>${b}`;s.onclick=()=>{on[i]=!on[i];s.classList.toggle('off',!on[i]);draw();};leg.appendChild(s);});
// first-visible-event index by start time (binary search; D.ev sorted by start)
function lo(t){let a=0,b=D.ev.length;while(a<b){const m=(a+b)>>1;if(D.ev[m][1]+D.ev[m][2]<t)a=m+1;else b=m;}return a;}
function fit(){scale=(cv.width-LM-16)/Math.max(D.span,1);px=LM;py=TOP;draw();}
function draw(){
  cx.clearRect(0,0,cv.width,cv.height);
  const t0=(LM-px)/scale, t1=(cv.width-px)/scale;               // visible time window (µs)
  // time axis
  cx.fillStyle='#0e1116';cx.fillRect(0,0,cv.width,TOP);
  cx.strokeStyle='#21262d';cx.fillStyle='#7d8998';cx.font='10px ui-monospace';cx.textAlign='center';
  const step=niceStep((t1-t0)/8);
  for(let t=Math.ceil(t0/step)*step;t<t1;t+=step){const X=px+t*scale;cx.beginPath();cx.moveTo(X,TOP);cx.lineTo(X,cv.height);cx.strokeStyle='#171c23';cx.stroke();cx.fillText(t.toFixed(t1-t0<20?2:0)+'µs',X,16);}
  // bars (cull by x via binary search on start, and by lane y-range)
  const i0=lo(t0);
  for(let i=i0;i<D.ev.length;i++){const e=D.ev[i];if(e[1]>t1)break;if(!on[e[3]])continue;
    const y=TOP+e[0]*RH+py-TOP;if(y+RH<TOP||y>cv.height)continue;
    const X=px+e[1]*scale,W=Math.max(e[2]*scale,0.6);
    cx.fillStyle=D.bcol[e[3]];cx.fillRect(X,y+1,W,RH-2);}
  // lane labels (left gutter)
  cx.fillStyle='#0e1116';cx.fillRect(0,TOP,LM,cv.height);
  cx.textAlign='left';cx.font='9px ui-monospace';
  for(let l=0;l<D.lanes.length;l++){const y=TOP+l*RH+py-TOP;if(y+RH<TOP||y>cv.height)continue;
    cx.fillStyle=(l&1)?'#7d8998':'#9aa7b4';cx.fillText(D.lanes[l],4,y+RH-3);}
  cx.strokeStyle='#21262d';cx.beginPath();cx.moveTo(LM,0);cx.lineTo(LM,cv.height);cx.stroke();
}
function niceStep(x){const p=Math.pow(10,Math.floor(Math.log10(x)));const f=x/p;return (f<1.5?1:f<3.5?2:f<7.5?5:10)*p;}
// pick event under cursor
function hit(mx,my){if(mx<LM)return -1;const t=(mx-px)/scale,l=Math.round((my-TOP-py+TOP)/RH-0);
  const lane=Math.floor((my-TOP-(py-TOP))/RH);
  const i0=lo(t-0.001);
  for(let i=i0;i<D.ev.length;i++){const e=D.ev[i];if(e[1]>t)break;if(e[0]!==lane||!on[e[3]])continue;
    if(t>=e[1]&&t<=e[1]+e[2])return i;}return -1;}
const hud=document.getElementById('hud');
let drag=null;
cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,px,py};cv.classList.add('drag');});
addEventListener('mouseup',()=>{drag=null;cv.classList.remove('drag');});
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect();const mx=e.clientX-r.left,my=e.clientY-r.top;
  if(drag){px=drag.px+(e.clientX-drag.x);py=drag.py+(e.clientY-drag.y);draw();return;}
  const i=hit(mx,my);
  if(i<0){hud.style.display='none';return;}
  const e2=D.ev[i];hud.style.display='block';hud.style.left=(e.clientX+14)+'px';hud.style.top=(e.clientY+14)+'px';
  hud.innerHTML=`<div class="o" style="color:${D.bcol[e2[3]]}">${D.ops[e2[4]]}</div><div class="k">${D.lanes[e2[0]]} · ${D.bnames[e2[3]]}</div><div>start <b>${e2[1].toFixed(3)}</b> µs · dur <b>${e2[2].toFixed(3)}</b> µs</div>`;});
cv.addEventListener('mouseleave',()=>{hud.style.display='none';});
cv.addEventListener('wheel',e=>{e.preventDefault();const r=cv.getBoundingClientRect();const mx=e.clientX-r.left;
  const t=(mx-px)/scale;const f=e.deltaY<0?1.2:1/1.2;scale*=f;px=mx-t*scale;draw();},{passive:false});
function resize(){cv.width=cv.parentElement.clientWidth;cv.height=cv.parentElement.clientHeight;draw();}
addEventListener('resize',resize);resize();fit();
</script></body></html>"""

def main():
    path=sys.argv[1]; out=sys.argv[2] if len(sys.argv)>2 else "trace_hemul.html"
    data=build(path)
    open(out,"w").write(HTML.replace("__DATA__", json.dumps(data,separators=(",",":"))))
    print(f"wrote {out}  ({len(data['ev']):,} ops, {len(data['lanes'])} lanes, span {data['span']:.0f} µs) — open in a browser")

main()
