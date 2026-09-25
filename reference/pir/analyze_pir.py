#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Per-kernel device-active latency from a device-profiler trace (PIR programs).

Same method as docs/he/analyze_ctmul.py: dispatches run serially on one command queue, so the i-th
`-KERNEL` start/end pair on each RISC is dispatch i. Per-dispatch device-active latency = max over
RISCs of that dispatch's `-KERNEL` envelope (RISCs run concurrently; 1 GHz -> 1 cycle = 1 ns). Each
dispatch's kernel is the named DeviceZoneScopedN zone nested in TRISC_1's i-th `-KERNEL` window.
  Usage: analyze_pir.py profile_log_device.csv
"""
import csv, sys, collections, bisect

KID = {   # zone -> (friendly name, functional bucket)
    "fused_total":      ("NTT stage-A matmul (fused_matmul_a)", "NTT"),
    "fusedC_total":     ("NTT stage-C matmul (fused_matmul_c)", "NTT"),
    "modmul_compute":   ("modmul (NTT twiddle + pointwise MAC)", "modmul"),
    "modadd_reduce":    ("modular add (accumulate)",            "add/sub"),
    "modsub_reduce":    ("modular sub",                          "add/sub"),
    "garner_diff":      ("CRT garner-diff (#13)",               "CRT/gadget"),
    "crt_mul_split":    ("CRT combine+split (#13)",             "CRT/gadget"),
    "gadget_decompose": ("gadget decompose (#9)",               "CRT/gadget"),
    "digit_to_rns":     ("digit->RNS split",                    "CRT/gadget"),
}

def analyze(path):
    rows=[]
    with open(path) as f:
        f.readline(); r=csv.reader(f); next(r)
        for x in r:
            if len(x)>=12: rows.append(x)
    rows.sort(key=lambda x:int(x[5]))
    kern=collections.defaultdict(list); open_k=collections.defaultdict(list); t1named=[]
    for x in rows:
        risc=x[3]; t=int(x[5]); z=x[10].strip(); ty=x[11]
        if z.endswith("-KERNEL"):
            if ty=="ZONE_START": open_k[risc].append(t)
            elif ty=="ZONE_END" and open_k[risc]: s=open_k[risc].pop(); kern[risc].append((s,t-s))
        elif z in KID and ty=="ZONE_START" and risc=="TRISC_1": t1named.append((t,z))
    for risc in kern: kern[risc].sort()
    t1named.sort()
    riscs=list(kern); ndisp=max(len(v) for v in kern.values())
    dev=[max(kern[r][i][1] for r in riscs if i<len(kern[r])) for i in range(ndisp)]
    t1k=kern["TRISC_1"]; nstarts=[s for s,_ in t1named]; kof=[None]*ndisp
    for i,(s,d) in enumerate(t1k):
        j=bisect.bisect_left(nstarts,s)
        if j<len(t1named) and s<=nstarts[j]<=s+d: kof[i]=t1named[j][1]
    per=collections.defaultdict(lambda:[0,0]); bkt=collections.defaultdict(lambda:[0,0]); tot=sum(dev)
    for i in range(ndisp):
        z=kof[i]; name=KID.get(z,(z or "(unattributed)","other"))[0]; b=KID.get(z,("","other"))[1]
        per[name][0]+=1; per[name][1]+=dev[i]; bkt[b][0]+=1; bkt[b][1]+=dev[i]
    return per,bkt,tot,ndisp

def us(c): return c/1000.0
per,bkt,tot,ndisp=analyze(sys.argv[1])
print(f"\n=== PIR — per-kernel device-active latency ({ndisp} dispatches, 1 GHz) ===\n")
print(f"{'kernel':<40}{'calls':>7}{'total us':>11}{'us/call':>10}{'%':>7}")
print("-"*75)
for k,(c,cyc) in sorted(per.items(), key=lambda kv:-kv[1][1]):
    print(f"{k:<40}{c:>7}{us(cyc):>11.1f}{us(cyc)/c:>10.3f}{100*cyc/tot:>6.1f}%")
print("-"*75)
print(f"{'TOTAL device-active':<40}{ndisp:>7}{us(tot):>11.1f}")
print(f"\n=== functional buckets ===")
for b,(c,cyc) in sorted(bkt.items(), key=lambda kv:-kv[1][1]):
    print(f"  {b:<12}{c:>5} calls{us(cyc):>10.1f} us{100*cyc/tot:>7.1f}%")
