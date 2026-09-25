#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Interactive on-device trace viewer (Dash + Plotly) for the HE-mul profiler CSV.

Live Python server → forward the port → explore in your browser: zoom (box-select / scroll),
pan, hover for op name + duration, toggle op-types in the legend, filter by RISC.
x = time (µs), y = core/RISC lane, bar = one on-device operation.

  python3 trace_app.py profile_log_device.csv [port]      # default port 8050
  ssh -L 8050:localhost:8050 <vm>   then open  http://localhost:8050
"""
import csv, sys, collections
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

OP = {"fused_total":"matmul-A (NTT stage-1)","fusedC_total":"matmul-C (NTT stage-2)",
      "modmul_compute":"twiddle modmul","modmul_sfpi_full":"pointwise modmul",
      "fbc_total":"base conversion (FBC)","modadd_reduce":"modular add","modsub_reduce":"modular sub"}
BUCKET={"fused_total":"NTT","fusedC_total":"NTT","modmul_compute":"NTT",
        "modmul_sfpi_full":"pointwise","fbc_total":"FBC","modadd_reduce":"vec","modsub_reduce":"vec"}
COLOR={"NTT":"#4e79a7","pointwise":"#f28e2b","FBC":"#59a14f","vec":"#e15759","kernel":"#8a8f98"}
SKIP={"ph_recombine","ph_matmul","ph_decompose","fbc_phase1","fbc_phase2"}
MAXBARS=8000   # cap for smooth interaction (earliest window); full trace lives in the Perfetto .json

def load(path):
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
    rn=lambda r:r.replace('TRISC_','T').replace('BRISC','B').replace('NCRISC','NC')
    recs=[]                                                                # dict per op
    for cx,cy,risc,nm,s,d in z:
        recs.append({"risc":risc,"lane":f"({cx},{cy}) {rn(risc)}","start":(s-t0)/1000.0,"dur":d/1000.0,
                     "op":OP.get(nm,"kernel" if nm.endswith("-KERNEL") else nm),"bucket":BUCKET.get(nm,"kernel")})
    recs.sort(key=lambda e:e["start"])
    if len(recs)>MAXBARS: recs=recs[:MAXBARS]
    return recs

df=load(sys.argv[1]); PORT=int(sys.argv[2]) if len(sys.argv)>2 else 8050
lanes=sorted({r["lane"] for r in df}, key=lambda s:(s.split()[0], s.split()[1]))

def figure(sel_riscs):
    d=[r for r in df if (not sel_riscs or r["risc"] in sel_riscs)]
    fig=go.Figure()
    for bkt,color in COLOR.items():
        g=[r for r in d if r["bucket"]==bkt]
        if not g: continue
        fig.add_bar(y=[r["lane"] for r in g], x=[r["dur"] for r in g], base=[r["start"] for r in g],
                    orientation="h", name=bkt, marker_color=color, marker_line_width=0, width=0.8,
                    customdata=[[r["op"],r["dur"],r["start"]] for r in g],
                    hovertemplate="<b>%{customdata[0]}</b><br>%{y}<br>start %{customdata[2]:.2f} µs · dur %{customdata[1]:.3f} µs<extra></extra>")
    fig.update_layout(barmode="overlay", template="plotly_dark", height=max(420,18*len(lanes)+120),
                      margin=dict(l=90,r=20,t=30,b=40), bargap=0.15, legend_title="op-type",
                      paper_bgcolor="#12141a", plot_bgcolor="#12141a", font=dict(family="ui-monospace,monospace",size=11))
    fig.update_xaxes(title="time (µs)", gridcolor="#2a2e38", zeroline=False)
    fig.update_yaxes(categoryorder="array", categoryarray=lanes[::-1], gridcolor="#20242c", automargin=True)
    return fig

app=Dash(__name__)
app.layout=html.Div(style={"background":"#0f1116","color":"#e6e9ef","fontFamily":"ui-monospace,monospace","padding":"14px 20px","minHeight":"100vh"},children=[
    html.H3("CROSS-aligned Set-B HE-mul — on-device trace (interactive)",style={"margin":"0 0 2px"}),
    html.Div(f"N=8192 · L=8 · dnum=2 · 16 cores · {len(df):,} ops shown (earliest window) · zoom=box/scroll, pan=drag, hover=details",
             style={"color":"#9aa3b2","fontSize":"12px","marginBottom":"10px"}),
    dcc.Checklist(id="risc", options=[{"label":f" {r}","value":r} for r in ["BRISC","NCRISC","TRISC_0","TRISC_1","TRISC_2"]],
                  value=["TRISC_1","TRISC_2","BRISC","NCRISC","TRISC_0"], inline=True,
                  style={"marginBottom":"8px","fontSize":"12px"}, inputStyle={"marginRight":"3px","marginLeft":"12px"}),
    dcc.Graph(id="g", config={"scrollZoom":True}),
])
@app.callback(Output("g","figure"), Input("risc","value"))
def _(sel): return figure(sel)

if __name__=="__main__":
    print(f"serving interactive trace on http://127.0.0.1:{PORT}  ({len(df)} bars, {len(lanes)} lanes)")
    app.run(host="127.0.0.1", port=PORT, debug=False)
