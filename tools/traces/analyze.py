# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
import csv, sys, collections

def analyze(path):
    rows = []
    with open(path) as f:
        f.readline()  # ARCH line
        r = csv.reader(f)
        header = next(r)
        for x in r:
            if len(x) < 12: continue
            rows.append(x)
    # columns
    def C(x): return (x[1],x[2],x[3])  # core_x, core_y, risc
    times = [int(x[5]) for x in rows]
    span = max(times) - min(times)
    dispatches = sorted(set(int(x[7]) for x in rows))
    cores = sorted(set((x[1],x[2]) for x in rows), key=lambda t:(int(t[0]),int(t[1])))
    riscs = sorted(set(x[3] for x in rows))

    # pair KERNEL zones per (core,risc,run_id) -> sum busy cycles
    # match ZONE_START/END by zone name stack per (core,risc,runid)
    busy = collections.defaultdict(int)          # risc -> total kernel cycles
    perdisp = collections.defaultdict(lambda: collections.defaultdict(int))  # runid -> risc -> cyc
    stacks = collections.defaultdict(list)
    for x in sorted(rows, key=lambda x:int(x[5])):
        core_x,core_y,risc,tid,t,data,runid = x[1],x[2],x[3],x[4],int(x[5]),x[6],int(x[7])
        zname,ztype = x[10],x[11]
        key=(core_x,core_y,risc,runid,zname)
        if ztype=="ZONE_START":
            stacks[key].append(t)
        elif ztype=="ZONE_END" and stacks[key]:
            dur = t - stacks[key].pop()
            if zname.endswith("-KERNEL"):
                busy[risc]+=dur
                perdisp[runid][risc]+=dur
    return dict(span=span, ndisp=len(dispatches), ncores=len(cores),
                riscs=riscs, busy=busy, perdisp=perdisp, cores=cores)

for p in sys.argv[1:]:
    a=analyze(p)
    name=p.split("/")[-1].replace(".csv","")
    print(f"\n######## {name} ########")
    print(f"device wall span      : {a['span']:>12,} cyc = {a['span']/1000:.2f} us")
    print(f"program dispatches    : {a['ndisp']}")
    print(f"distinct cores touched: {a['ncores']}  {[f'{x},{y}' for x,y in a['cores']][:8]}{'...' if a['ncores']>8 else ''}")
    tot=sum(a['busy'].values())
    print(f"total KERNEL busy (sum over RISCs, all dispatches): {tot:,} cyc = {tot/1000:.2f} us")
    for risc in sorted(a['busy']):
        print(f"    {risc:<8}: {a['busy'][risc]:>10,} cyc = {a['busy'][risc]/1000:8.2f} us")
    # compute risc = TRISC0/1/2; show top dispatches by TRISC time
    comp = lambda d: sum(v for r,v in a['perdisp'][d].items() if r.startswith('TRISC'))
    top = sorted(a['perdisp'], key=comp, reverse=True)[:5]
    print(f"    heaviest dispatches by compute (TRISC) cyc:")
    for d in top:
        c=comp(d); tr=a['perdisp'][d]
        print(f"       runid {d:>3}: TRISC={c:>8,}  BRISC={tr.get('BRISC',0):>7,} NCRISC={tr.get('NCRISC',0):>7,}")
