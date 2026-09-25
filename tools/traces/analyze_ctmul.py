#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Per-kernel latency breakdown of the full BFV ct_mul from a device-profiler trace.

Dispatches run serially on one command queue, so the i-th `-KERNEL` start/end pair on each RISC
is dispatch i (same method as analyze2.py). Per-dispatch device-active latency = max over RISCs of
that dispatch's `-KERNEL` envelope (RISCs run concurrently; 1 GHz → 1 cycle = 1 ns). Each dispatch
runs one compute kernel, identified by the named DeviceZoneScopedN zone whose start-time falls in
the dispatch's BRISC window. Group by kernel → calls / total / per-call.
  Usage: analyze_ctmul.py bfv_ct_mul.csv
"""
import csv, sys, collections, bisect

KID = {   # top-level zone -> (friendly kernel name, functional bucket)
    "fused_total":     ("NTT stage-A matmul (fused_matmul_a)",       "NTT"),
    "modmul_compute":  ("NTT twiddle modmul (modmul_compute)",       "NTT"),
    "fusedC_total":    ("NTT stage-C matmul (fused_matmul_c)",       "NTT"),
    "transpose_tile":  ("NTT reindex transpose (transpose_tile)",    "NTT"),
    "modmul_sfpi_full":("pointwise/rescale modmul (modmul_sfpi_full)","modmul"),
    "fbc_total":       ("base conversion (baseconv_fbc)",            "FBC"),
    "modadd_reduce":   ("modular add (modadd_reduce)",               "add/sub"),
    "modsub_reduce":   ("modular sub (modsub_reduce)",               "add/sub"),
}

def analyze(path):
    rows=[]
    with open(path) as f:
        f.readline(); r=csv.reader(f); next(r)
        for x in r:
            if len(x)>=12: rows.append(x)
    rows.sort(key=lambda x:int(x[5]))
    # per-RISC ordered list of (start,dur) for -KERNEL zones; KID named-zone starts on TRISC_1
    kern=collections.defaultdict(list); open_k=collections.defaultdict(list)
    t1named=[]                                  # (start, zone) KID zones on TRISC_1
    for x in rows:
        risc=x[3]; t=int(x[5]); z=x[10].strip(); ty=x[11]
        if z.endswith("-KERNEL"):
            if ty=="ZONE_START": open_k[risc].append(t)
            elif ty=="ZONE_END" and open_k[risc]:
                s=open_k[risc].pop(); kern[risc].append((s,t-s))
        elif z in KID and ty=="ZONE_START" and risc=="TRISC_1":
            t1named.append((t,z))
    for risc in kern: kern[risc].sort()
    t1named.sort()
    riscs=list(kern); ndisp=max(len(v) for v in kern.values())
    # per-dispatch device-active = max over riscs of the i-th -KERNEL dur (serial → index-aligned)
    dev=[max(kern[r][i][1] for r in riscs if i<len(kern[r])) for i in range(ndisp)]
    # kernel of dispatch i = the KID named zone nested inside TRISC_1's i-th -KERNEL window
    t1k=kern["TRISC_1"]; nstarts=[s for s,_ in t1named]
    kof=[None]*ndisp
    for i,(s,d) in enumerate(t1k):
        j=bisect.bisect_left(nstarts,s)
        if j<len(t1named) and s<=nstarts[j]<=s+d: kof[i]=KID[t1named[j][1]][0]
    per=collections.defaultdict(lambda:[0,0]); bkt=collections.defaultdict(lambda:[0,0]); tot=sum(dev)
    fb={v[0]:v[1] for v in KID.values()}
    for i in range(ndisp):
        k=kof[i] or "(unattributed)"; per[k][0]+=1; per[k][1]+=dev[i]
        b=fb.get(k,"other"); bkt[b][0]+=1; bkt[b][1]+=dev[i]
    # ---- stage decomposition (pipeline runs sequentially: tensor → rescale → relin → add) ----
    TRANSF={KID[z][0] for z in ("fused_total","modmul_compute","fusedC_total","transpose_tile")}
    FBCk=KID["fbc_total"][0]; ADDk=KID["modadd_reduce"][0]
    R=next((i for i in range(ndisp) if kof[i]==FBCk), ndisp)          # relin = first FBC (mod-up)
    A=ndisp                                                            # add = trailing modadd run
    while A>0 and kof[A-1]==ADDk: A-=1
    T=1+max((i for i in range(R) if kof[i] in TRANSF), default=-1)     # tensor = up to last transform < R
    stg=[("tensor",0,T),("rescale",T,R),("relinearize",R,A),("add",A,ndisp)]
    stage={n:[e-s, sum(dev[s:e])] for n,s,e in stg}
    return per,bkt,tot,ndisp,stage

def us(c): return c/1000.0
per,bkt,tot,ndisp,stage=analyze(sys.argv[1])
print(f"\n=== BFV ct_mul — per-kernel device-active latency ({ndisp} dispatches, 1 GHz) ===\n")
print(f"{'kernel':<44}{'calls':>7}{'total µs':>11}{'µs/call':>10}{'%':>7}")
print("-"*79)
for k,(c,cyc) in sorted(per.items(), key=lambda kv:-kv[1][1]):
    print(f"{k:<44}{c:>7}{us(cyc):>11.1f}{us(cyc)/c:>10.3f}{100*cyc/tot:>6.1f}%")
print("-"*79)
print(f"{'TOTAL device-active':<44}{ndisp:>7}{us(tot):>11.1f}")
print(f"\n=== functional buckets ===")
for b,(c,cyc) in sorted(bkt.items(), key=lambda kv:-kv[1][1]):
    print(f"  {b:<12}{c:>5} calls{us(cyc):>10.1f} µs{100*cyc/tot:>7.1f}%")
print(f"\n=== stage decomposition (end-to-end ct_mul) ===")
for n in ("tensor","rescale","relinearize","add"):
    c,cyc=stage[n]; print(f"  {n:<14}{c:>5} dispatches{us(cyc):>10.1f} µs{100*cyc/tot:>7.1f}%")
