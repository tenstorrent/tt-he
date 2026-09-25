# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
import csv, sys, collections

def analyze(path):
    rows=[]
    with open(path) as f:
        f.readline(); r=csv.reader(f); next(r)
        for x in r:
            if len(x)>=12: rows.append(x)
    rows.sort(key=lambda x:int(x[5]))
    # per (core,risc) ordered list of KERNEL and FW durations via stack
    kern=collections.defaultdict(list)   # (core,risc)->[durations] KERNEL
    stack=collections.defaultdict(list)
    for x in rows:
        core=(x[1],x[2]); risc=x[3]; t=int(x[5]); z=x[10]; ty=x[11]
        key=(core,risc,z)
        if ty=="ZONE_START": stack[key].append(t)
        elif ty=="ZONE_END" and stack[key]:
            d=t-stack[key].pop()
            if z.endswith("-KERNEL"): kern[(core,risc)].append(d)
    # single core assumed (these are 1-core programs)
    riscs=sorted(set(r for (c,r) in kern))
    # number of dispatches = max kernel-count across riscs
    ndisp=max(len(kern[(c,r)]) for (c,r) in kern)
    core=sorted(set(c for (c,r) in kern))[0]
    # per-dispatch device-active = max over riscs of that dispatch's kernel dur
    perrisc={r:kern[(core,r)] for r in riscs}
    dev_active=0
    for i in range(ndisp):
        m=max((perrisc[r][i] for r in riscs if i<len(perrisc[r])), default=0)
        dev_active+=m
    busy={r:sum(perrisc[r]) for r in riscs}
    return dict(ndisp=ndisp, dev_active=dev_active, busy=busy, ncore=len(set(c for (c,r) in kern)))

for p in sys.argv[1:]:
    a=analyze(p); name=p.split("/")[-1].replace(".csv","")
    print(f"\n######## {name} ########")
    print(f"  cores used            : {a['ncore']}")
    print(f"  program dispatches    : {a['ndisp']}")
    print(f"  DEVICE-ACTIVE time     : {a['dev_active']:>10,} cyc = {a['dev_active']/1000:8.2f} us   (sum of per-dispatch max-RISC; excludes host gaps)")
    print(f"  per-RISC total busy (sum over all dispatches):")
    for r in sorted(a['busy']):
        print(f"      {r:<8}: {a['busy'][r]:>9,} cyc = {a['busy'][r]/1000:7.2f} us")
