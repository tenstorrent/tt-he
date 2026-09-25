# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""schedule.py — stage 6: placement + scheduling (greedy HEFT-lite, p=1).

Takes an L2 (kernel-IR) Program and P cores, assigns each op a core + a time so that independent
ops run concurrently on different cores. HE-unaware (works on tile ops, not HE semantics). This is
a *pure annotation* — it does not change what is computed (mapping is meaning-preserving), only
where/when. Durations + the DRAM cross-core handoff come from cost_model.py.

Algorithm (HEFT-lite, docs/ir/FORMALIZATION §2/§6):
  1. rank each op by critical-path length to the sink (upward rank).
  2. walk ready ops in rank order; place each on the core that lets it FINISH earliest.
     - Fusion B: a same-core producer→consumer stays in SRAM (free); cross-core pays a NoC hop.
     - reuse=True: a per-core LRU working set models residency — a constant (NTT weight, twiddle,
       evk, CRNS tables) or hot intermediate already resident on the chosen core is reused for free
       instead of re-loaded, and placement is biased to co-locate consumers of a shared operand
       (operand affinity). Reports the reuse rate + load traffic saved.
Not claimed optimal — a reasonable greedy split across cores.
"""
from __future__ import annotations
import heapq
from math import prod
from collections import OrderedDict

import cost_model as cm
from ir import outputs


# op -> duration (us), from cost_model; structural ops are ~free
def op_dur(op) -> float:
    r = outputs(op)[0]
    tiles = max(1, int(prod(r.shape)) // 1024) if getattr(r, "shape", None) else 1
    oc = op.opcode
    if oc in ("tile.matmul_modq", "tile.matmul_wide"):       return cm.OPS["i8_matmul"].setup   # one tile matmul
    if oc == "tile.mul_modq":         return cm.op_cost("modmul", tiles)
    if oc in ("tile.add_modq", "tile.sub_modq"): return cm.op_cost("modadd", tiles)
    if oc == "tile.decompose":        return cm.OPS["decompose"].setup
    if oc in ("tile.recombine_modq", "tile.recombine_wide"): return cm.OPS["recombine"].setup
    return 0.5                                                                  # reshape/slice/concat/const/transpose/fixed_slice

# Memory tiers (docs/ir/FORMALIZATION §6). A producer->consumer edge costs:
#   same core   -> SRAM  (stays in L1 — this IS Fusion B — free)
#   diff core   -> NoC   (a tile hop)
#   (spill      -> DRAM  — not modeled separately in v1)
SRAM, NOC = cm.K["sram"].value, cm.K["noc"].value + cm.DISPATCH.value
COMM = NOC                                            # cross-core handoff (kept for back-compat)
CONST_LD = cm.K["dram"].value + cm.DISPATCH.value     # load a baked constant DRAM->L1 (once per core)
RELOAD = cm.K["dram"].value + cm.DISPATCH.value       # re-fetch a spilled/evicted intermediate


def schedule(prog, P: int, reuse: bool = False, cap: int = 64) -> dict:
    """Greedy HEFT-lite placement (p=1). Meaning-preserving (annotation only).

    reuse=False (baseline): a producer->consumer edge is free if same core (SRAM/Fusion B), else NoC.

    reuse=True (residency model): each core keeps an LRU **working set** of `cap` resident tiles.
    An operand already resident on the chosen core — a reused constant (NTT weight, twiddle, evk,
    CRNS tables) or a hot intermediate — costs 0; otherwise it's loaded (a const from DRAM, an
    intermediate over NoC, or a re-fetch if it was evicted). Because placement picks the earliest
    finish and tie-breaks on load cost, consumers of a shared operand are pulled onto the core where
    it is already resident (operand affinity) — so a weight is loaded once and reused, not re-streamed
    to every core. Reports the reuse rate + load traffic saved."""
    ops = prog.ops
    n = len(ops)
    prod_of, is_const = {}, set()
    for i, op in enumerate(ops):
        for v in outputs(op):
            prod_of[v.name] = i
        if op.opcode in ("tile.const", "rns.const"):
            for v in outputs(op):
                is_const.add(v.name)
    opnd = [[o.name for o in op.operands] for op in ops]
    deps = [ {prod_of[nm] for nm in opnd[i] if nm in prod_of} for i in range(n) ]
    succ = [[] for _ in range(n)]
    for i in range(n):
        for d in deps[i]:
            succ[d].append(i)
    dur = [op_dur(op) for op in ops]

    # upward rank (critical path to sink)
    rank = [0.0] * n
    for i in reversed(range(n)):
        rank[i] = dur[i] + max((rank[s] for s in succ[i]), default=0.0)

    core = [-1] * n; start = [0.0] * n; end = [0.0] * n
    core_free = [0.0] * P
    ws = [OrderedDict() for _ in range(P)]              # per-core LRU working set (value name resident)
    reuse_hits = reload_ct = const_ld = load_us = 0.0
    accesses = 0
    indeg = [len(deps[i]) for i in range(n)]
    ready = [(-rank[i], i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(ready)
    while ready:
        _, i = heapq.heappop(ready)
        best = None
        for c in range(P):
            rt = ld = 0.0
            if reuse:
                for nm in opnd[i]:
                    d = prod_of.get(nm)
                    base = end[d] if d is not None else 0.0        # never start before the producer finishes
                    if nm in ws[c]:
                        xfer = 0.0                                 # resident -> reuse, free
                    elif nm in is_const:
                        xfer = CONST_LD                            # load baked const DRAM->L1
                    elif d is not None:
                        xfer = NOC if core[d] != c else RELOAD     # from another core, or evicted
                    else:
                        xfer = 0.0
                    rt = max(rt, base + xfer); ld += xfer
            else:
                for d in deps[i]:
                    rt = max(rt, end[d] + (COMM if core[d] != c else 0.0))
            st = max(core_free[c], rt)
            fin = st + dur[i]
            key = (fin, ld)                                        # tie-break: prefer fewer loads (affinity)
            if best is None or key < best[0]:
                best = (key, c, st)
        (fin_ld, c, st) = best
        core[i], start[i], end[i] = c, st, fin_ld[0]
        core_free[c] = fin_ld[0]
        if reuse:
            load_us += fin_ld[1]
            for nm in opnd[i]:                                     # update working set + reuse stats
                accesses += 1
                if nm in ws[c]:
                    reuse_hits += 1; ws[c].move_to_end(nm)
                else:
                    if nm in is_const: const_ld += 1
                    elif nm in prod_of: reload_ct += 1
                    ws[c][nm] = True
            for v in outputs(ops[i]):
                ws[c][v.name] = True; ws[c].move_to_end(v.name)
            while len(ws[c]) > cap:
                ws[c].popitem(last=False)                          # evict LRU
        for s in succ[i]:
            indeg[s] -= 1
            if indeg[s] == 0:
                heapq.heappush(ready, (-rank[s], s))

    makespan = max(end) if end else 0.0
    serial = sum(dur)
    used = len({c for c in core if c >= 0})
    edges = []
    for i in range(n):
        for d in deps[i]:
            edges.append((d, i, "sram" if core[d] == core[i] else "noc"))
    fused = sum(1 for _, _, t in edges if t == "sram")
    comm_us = sum((SRAM if t == "sram" else NOC) for _, _, t in edges)
    return dict(core=core, start=start, end=end, dur=dur, makespan=makespan,
               serial=serial, cores_used=used, P=P, deps=deps,
               edges=edges, n_edges=len(edges), fused=fused, comm_us=comm_us,
               reuse=reuse, cap=cap, accesses=int(accesses), reuse_hits=int(reuse_hits),
               const_loads=int(const_ld), reloads=int(reload_ct), load_us=load_us)


def fusion_b_report(prog, sch) -> str:
    """Fusion B = producer→consumer edges kept on the SAME core (stay in SRAM, no NoC hop). Reports
    how many of the DAG's edges were fused vs moved, and the communication cost that Fusion B saved."""
    e = sch["edges"]; n = len(e); fused = sch["fused"]; moved = n - fused
    saved = moved and (moved * (NOC - SRAM))
    return (f"Fusion B: {fused}/{n} edges same-core (SRAM, fused), {moved} cross-core (NoC)  "
            f"[{100*fused//max(1,n)}% fused]\n"
            f"  comm cost {sch['comm_us']:.1f} us   (Fusion B saved ~{moved*(NOC-SRAM):.1f} us vs "
            f"all-NoC on the moved edges)")


def reuse_report(prog, sch) -> str:
    """Operand-reuse under the residency model: how many operand fetches hit a value already resident
    on the placed core (a reused constant / hot intermediate) vs had to be loaded. Compares the load
    traffic against a naive 'load every operand every time' baseline (no residency)."""
    if not sch.get("reuse"):
        return "reuse model OFF (schedule(..., reuse=True) to enable)"
    acc, hit = sch["accesses"], sch["reuse_hits"]
    naive = sch["const_loads"] + sch["reloads"] + hit                 # every fetch would be a load
    return (f"Operand reuse (cap={sch['cap']} tiles/core): {hit}/{acc} operand fetches reused from L1 "
            f"[{100*hit//max(1,acc)}%]  ·  {sch['const_loads']} const loads + {sch['reloads']} reloads\n"
            f"  load traffic {sch['load_us']:.1f} us  vs ~{(naive)*CONST_LD:.1f} us if every fetch "
            f"re-loaded (residency avoided {hit} re-loads)")


def check_valid(sch) -> bool:
    """Every op starts no earlier than its producers finish (schedule respects the DAG)."""
    for i, ds in enumerate(sch["deps"]):
        for d in ds:
            if sch["start"][i] + 1e-9 < sch["end"][d]:
                return False
    return True


def report(prog, sch) -> str:
    L = [f"schedule: {len(prog.ops)} ops on {sch['P']} cores  ({sch['cores_used']} used)",
         f"  makespan {sch['makespan']:.1f} us   vs serial {sch['serial']:.1f} us   "
         f"=> {sch['serial']/sch['makespan']:.2f}x   valid={check_valid(sch)}"]
    # tiny per-core occupancy
    busy = [0.0] * sch["P"]
    for i, c in enumerate(sch["core"]):
        if c >= 0: busy[c] += sch["dur"][i]
    L.append("  per-core busy us: " + " ".join(f"{b:.0f}" for b in busy if b > 0))
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# self-test: schedule the lowered tensor across cores; check valid + parallel
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os, sys, random
    import _paths                                     # noqa: F401  (repo layout; see _paths.py)
    from bfv_mul_ref import MulParams, encrypt
    from bfv_relin_ref import small_poly
    import ir, reference as ref
    from builders import build_tensor
    from lower import lower_l1_to_l2, fuse_a

    pp = MulParams(N=64)
    rng = random.Random(0); s = small_poly(pp.N, rng)
    ct = [encrypt(pp, [rng.randrange(pp.t) for _ in range(pp.N)], s, rng) for _ in range(2)]
    l1, inp, _ = build_tensor(pp, *ct)
    l2 = fuse_a(lower_l1_to_l2(l1, factors=(8, 8), T=32))     # a realistic L2 tensor
    r_ref = ir.run(l1, inp)

    print(f"workload: tensor N={pp.N}, {len(l2.ops)} L2 ops\n")
    for P in (1, 4, 8, 16):
        sch = schedule(l2, P)
        print(report(l2, sch))
    # meaning-preserving: scheduling is annotation only — results unchanged
    same = all(ref._eq(r_ref[k], ir.run(l2, inp)[k]) for k in r_ref)
    print(f"\n  meaning-preserving (results unchanged by scheduling): {'OK' if same else 'FAIL'}")
    print("\nSCHEDULE OK" if same and check_valid(schedule(l2, 8)) else "\nFAIL")
