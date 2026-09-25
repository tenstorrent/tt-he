# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""passes.py — the four mapping optimisations, as an ordered pass manager over an L2/L3 Program.

Each pass is a rung of the MEASURED hardware ladder in docs/he/exp_tg018ntt.md, so each one can be
checked against a number rather than argued about:

    pass              hardware rung                              measured
    ---------------------------------------------------------------------------
    place_comm        corner turn DRAM -> core-to-core           50.863 -> 43.932
    fuse_dispatches   six NTT passes -> ONE dispatch             43.932 -> 32.448
    select_engines    base conversion SFPU -> matrix engine      32.448 -> 23.991
    legalize          (no rung: it is what makes the rest legal)

NOTHING HERE REWRITES THE PROGRAM. Every pass writes into a `Mapping` -- engine per op, tier per edge,
dispatch group per op, kernel variant per op -- so the computed function is untouched by construction
and `run(prog)` stays the oracle. That is the same discipline the lowering uses (`run(L1)==run(L2)==
run(L3)`): a mapping decision that can change the answer is a bug, not a trade-off.

WHY THIS ORDER. `select_engines` needs to know whether operands will be resident (fusion) to price the
reuse regime, and `fuse_dispatches` needs to know which edges are on-chip (comm) to know what may fuse.
The cycle is broken by running comm -> fuse -> engines and letting `legalize` reject anything the first
three produced that the hardware cannot actually take. Each pass reports what it changed, and the
manager keeps the running predicted makespan so the ladder is a data structure the viz can render.

Run:  python3 passes.py
"""
from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from math import prod

import _paths                                     # noqa: F401
import hw
from ir import outputs


# --------------------------------------------------------------------------- #
# op classification
# --------------------------------------------------------------------------- #
STRUCTURAL = ("tile.reshape", "tile.slice", "tile.concat", "tile.const", "rns.const",
              "tile.fixed_slice")
ALLTOALL = ("tile.transpose", "tile.all_to_all")
CONTRACT = ("tile.matmul_modq", "tile.matmul_wide", "tile.base_extend")


def _tiles(v) -> int:
    if not getattr(v, "shape", None):
        return 1
    n = int(prod(v.shape))
    return max(1, (n + hw.TILE_ELEMS - 1) // hw.TILE_ELEMS)


def op_shape(op):
    """(contracted extent ls, destination extent ld, output tiles) for a contraction; else (1,1,tiles)."""
    r = outputs(op)[0]
    ot = _tiles(r)
    if op.opcode in CONTRACT and len(op.operands) >= 2:
        a, b = op.operands[0], op.operands[1]
        sa = getattr(a, "shape", None) or (1,)
        sb = getattr(b, "shape", None) or (1,)
        ls = sa[-1] if len(sa) > 1 else 1                 # the contracted axis
        ld = sb[-1] if len(sb) > 1 else 1                 # the destination axis
        return int(ls), int(ld), ot
    return 1, 1, ot


def op_class(op) -> str:
    oc = op.opcode
    if oc in STRUCTURAL:
        return "structural"
    if oc in ALLTOALL:
        return "a2a"
    if oc in CONTRACT:
        return "contract"
    if oc in ("tile.mul_modq", "rns.mul"):
        return "modmul"
    if oc in ("tile.add_modq", "tile.sub_modq", "rns.add", "rns.sub"):
        return "addm"
    if oc == "tile.decompose":
        return "decompose"
    if oc in ("tile.recombine_modq", "tile.recombine_wide"):
        return "recombine"
    return "other"


# --------------------------------------------------------------------------- #
# the mapping
# --------------------------------------------------------------------------- #
@dataclass
class Mapping:
    """Everything the four passes decide. Pure annotation over a fixed Program."""
    n: int
    engine: list = field(default_factory=list)        # op -> "sfpu" | "fpu" | "dma" | "free"
    tier: dict = field(default_factory=dict)          # (producer, consumer) -> "sram"|"noc"|"dram"
    group: list = field(default_factory=list)         # op -> dispatch group id
    variant: list = field(default_factory=list)       # op -> (kernel name, params dict)
    resident: list = field(default_factory=list)      # op -> operands expected already in L1
    chain: list = field(default_factory=list)         # op -> length of the fused chain it sits in
    reports: list = field(default_factory=list)       # per-pass report dicts

    @classmethod
    def initial(cls, prog):
        n = len(prog.ops)
        m = cls(n=n)
        m.engine = [("free" if op_class(o) == "structural" else
                     hw.DMA if op_class(o) == "a2a" else hw.SFPU) for o in prog.ops]
        m.group = list(range(n))                      # worst case: one dispatch per op
        m.variant = [(None, {}) for _ in range(n)]
        m.resident = [False] * n
        m.chain = [1] * n
        # The naive starting point: every all-to-all spills to DRAM (write the whole set out, read it
        # back) and everything else takes a NoC hop. This is rung 0 of the ladder, not a strawman --
        # it is what the pipeline actually did before the corner turn was made core-to-core.
        deps, _, _ = dag(prog)
        for i, op in enumerate(prog.ops):
            t = "dram" if op_class(op) == "a2a" else "noc"
            for d in deps[i]:
                m.tier[(d, i)] = t
        return m


def dag(prog):
    """(deps, succ, producer-of-value) for a Program."""
    ops = prog.ops
    n = len(ops)
    prod_of = {}
    for i, op in enumerate(ops):
        for v in outputs(op):
            prod_of[v.name] = i
    deps = [sorted({prod_of[o.name] for o in ops[i].operands if o.name in prod_of}) for i in range(n)]
    succ = [[] for _ in range(n)]
    for i in range(n):
        for d in deps[i]:
            succ[d].append(i)
    return deps, succ, prod_of


# --------------------------------------------------------------------------- #
# duration under a mapping
# --------------------------------------------------------------------------- #
def op_us(op, mp: Mapping, i: int) -> float:
    """Per-core microseconds for op `i` under the current mapping.

    The REUSE REGIME is the whole reason fusion pays: exp_roofline001 measures every op twice, once
    streaming from DRAM (R=1) and once with operands resident, and the gap is 1.9x on montmul rising to
    39x on a plain add. An op whose operands are already in L1 (because the producer was fused into the
    same dispatch) is priced at the reuse end; everything else streams."""
    cls = op_class(op)
    reuse = mp.resident[i]
    if cls == "structural":
        return 0.0
    if cls == "a2a":
        # movement, not arithmetic -- priced by the tier place_comm picked for its inbound edge
        return 0.0
    ls, ld, ot = op_shape(op)
    if cls == "contract":
        per = hw.us_per_tile("modmul", reuse) + hw.us_per_tile("addm", reuse)
        base = ls * per * ot
        if mp.engine[i] == hw.FPU:
            return base / hw.fpu_speedup(ls, min(ld, int(float(hw.FPU_LD_CAP))))
        return base
    if cls == "modmul":
        return hw.us_per_tile("modmul", reuse) * ot
    if cls == "addm":
        return hw.us_per_tile("addm", reuse) * ot
    if cls == "decompose":
        return hw.us_per_tile("addp", reuse) * ot * 4          # 4 radix-128 planes, shift+mask each
    if cls == "recombine":
        return hw.us_per_tile("modmul", reuse) * ot * 2        # place-value combine + one reduction
    return hw.us_per_tile("addm", reuse) * ot


_DAG_CACHE = {}


def dag_cached(prog):
    k = id(prog)
    if k not in _DAG_CACHE:
        _DAG_CACHE[k] = dag(prog)
    return _DAG_CACHE[k]


def estimate(prog, mp: Mapping, P: int) -> dict:
    """Greedy list-schedule under the mapping -> predicted makespan. HEFT-lite, engine/tier aware."""
    ops = prog.ops
    n = len(ops)
    deps, succ, _ = dag_cached(prog)
    dur = [op_us(ops[i], mp, i) for i in range(n)]

    dtiles = [_tiles(outputs(ops[i])[0]) for i in range(n)]
    rank = [0.0] * n
    for i in reversed(range(n)):
        rank[i] = dur[i] + max((rank[s] for s in succ[i]), default=0.0)

    core = [-1] * n
    start = [0.0] * n
    end = [0.0] * n
    free = [0.0] * P
    indeg = [len(deps[i]) for i in range(n)]
    ready = [(-rank[i], i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(ready)
    while ready:
        _, i = heapq.heappop(ready)
        # candidate cores: those already holding a producer (transfer-free) plus the earliest-free one
        # (wait-free). Any other core is dominated by one of these two, so scanning all P is waste.
        cand = {core[d] for d in deps[i] if core[d] >= 0}
        cand.add(min(range(P), key=lambda c: free[c]))
        best = None
        for c in cand:
            rt = 0.0
            for d in deps[i]:
                t = mp.tier.get((d, i))
                if t == "dram":
                    k = float(hw.K_DRAM)              # spilled / all-to-all too big for the grid
                elif core[d] == c:
                    k = 0.0                           # same core: it is already in L1
                else:
                    k = float(hw.K_NOC)
                rt = max(rt, end[d] + k * dtiles[d])
            st = max(free[c], rt)
            if best is None or st + dur[i] < best[0]:
                best = (st + dur[i], c, st)
        fin, c, st = best
        core[i], start[i], end[i] = c, st, fin
        free[c] = fin
        for s in succ[i]:
            indeg[s] -= 1
            if indeg[s] == 0:
                heapq.heappush(ready, (-rank[s], s))

    ngroups = len({g for g in mp.group})
    compute = max(end) if end else 0.0
    dispatch = ngroups * float(hw.DISPATCH_US)
    return dict(makespan=compute + dispatch, compute=compute, dispatch_us=dispatch,
                groups=ngroups, core=core, start=start, end=end, dur=dur, deps=deps,
                serial=sum(dur), cores_used=len({c for c in core if c >= 0}), P=P)


# --------------------------------------------------------------------------- #
# PASS 1 — place_comm
# --------------------------------------------------------------------------- #
def place_comm(prog, mp: Mapping, P: int) -> dict:
    """Classify every producer->consumer edge into a memory tier, and keep the all-to-all ON CHIP.

    The all-to-all (the four-step NTT's corner turn, `tile.transpose`) is the one movement pattern in
    ct_mul that looks like it needs DRAM: every output tile draws from every input tile, so a naive
    lowering writes the whole polynomial set out and reads it back. It does not need DRAM -- the tiles
    are already on the chip, and a core-to-core exchange with a semaphore barrier moves them directly.
    Measured 9.108 -> 2.122 ms, 4.29x, on exactly this stage.

    The condition is capacity: the exchanging block must fit the grid at once, because every core has
    to hold its slice of the exchange while it waits. Blocks that do not fit fall back to DRAM."""
    deps, succ, _ = dag_cached(prog)
    a2a_on, a2a_off, noc, sram = 0, 0, 0, 0
    for i, op in enumerate(prog.ops):
        is_a2a = op_class(op) == "a2a"
        for d in deps[i]:
            ot = _tiles(outputs(prog.ops[d])[0])
            if is_a2a:
                # does the whole exchange block fit on the grid at once?
                if ot <= P:
                    mp.tier[(d, i)] = "noc"
                    a2a_on += 1
                else:
                    mp.tier[(d, i)] = "dram"
                    a2a_off += 1
            elif len(succ[d]) == 1:
                mp.tier[(d, i)] = "sram"          # a private chain can stay on-core
                sram += 1
            else:
                mp.tier[(d, i)] = "noc"
                noc += 1
    return dict(pass_name="place_comm", a2a_onchip=a2a_on, a2a_dram=a2a_off,
                sram=sram, noc=noc, edges=a2a_on + a2a_off + noc + sram,
                note=f"{a2a_on} all-to-all edges kept core-to-core "
                     f"({float(hw.A2A_RATIO):.2f}x cheaper than DRAM, measured)")


# --------------------------------------------------------------------------- #
# PASS 2 — fuse_dispatches
# --------------------------------------------------------------------------- #
def fuse_dispatches(prog, mp: Mapping, P: int, max_chain: int = 64) -> dict:
    """Merge producer->consumer CHAINS into one dispatch, and mark their internal operands resident.

    This is the pass that reproduced the largest rung: the six-pass NTT went from 2884 dispatches to
    1288 and the multiply from 43.932 to 32.448 ms. It pays twice, and the second is the bigger half:

      1. one launch instead of many (0.84 us each, measured);
      2. the chain's intermediates never leave L1, so every op inside the chain moves from the
         STREAMING end of the roofline to the REUSE end -- 1.9x on montmul, up to 39x on a plain add.

    A chain is a maximal run of ops where each has exactly one consumer, that consumer has exactly one
    producer worth fusing, and the edge between them was placed in SRAM by place_comm. Ops with several
    consumers end a chain: their result has to be visible to cores that are not in this dispatch."""
    deps, succ, _ = dag_cached(prog)
    n = len(prog.ops)
    gid = [-1] * n
    g = 0
    for i in range(n):
        if gid[i] != -1:
            continue
        gid[i] = g
        cur, ln = i, 1
        while ln < max_chain:
            if len(succ[cur]) != 1:
                break
            nxt = succ[cur][0]
            if gid[nxt] != -1 or len(deps[nxt]) != 1:
                break
            if mp.tier.get((cur, nxt)) != "sram":
                break
            gid[nxt] = g
            mp.resident[nxt] = True               # its operand is the previous op's result, still in L1
            cur, ln = nxt, ln + 1
        g += 1
    before = len({x for x in mp.group})
    mp.group = gid
    cl = {}
    for g in gid:
        cl[g] = cl.get(g, 0) + 1
    mp.chain = [cl[g] for g in gid]                    # per-CORE working set, not the batched width
    nres = sum(1 for r in mp.resident if r)
    return dict(pass_name="fuse_dispatches", groups_before=before, groups_after=g,
                removed=before - g, resident_ops=nres,
                note=f"{before - g} dispatches removed; {nres} ops now read operands from L1 "
                     f"(priced at the reuse end of the roofline)")


# --------------------------------------------------------------------------- #
# PASS 2b — batch_dispatches   (the DATA-PARALLEL axis)
# --------------------------------------------------------------------------- #
def batch_dispatches(prog, mp: Mapping, P: int) -> dict:
    """Put INDEPENDENT instances of the same kernel into ONE dispatch across the grid.

    `fuse_dispatches` merges along the SEQUENTIAL axis -- a producer chain collapsing into one kernel.
    That is the wrong axis for the launch tax, and leaving it at that is what made the compiler
    dispatch-bound: at the hardware parameter shape it emitted 44480 groups against the hand-tuned
    pipeline's 1278, so 91.6% of the modelled makespan was launch overhead against the real pipeline's
    4.5%. The compiler was paying 0.84 us to start a program that did ONE TILE on ONE CORE.

    The missing axis is DATA PARALLELISM: ct_mul applies the same op to 64 independent limbs, and every
    limb is many tiles. The hand-written pipeline dispatches that as one program covering the grid --
    literally `for base in 0..NTILE step NC` -- and this pass recovers the same shape from the graph.

    Two ops may share a dispatch when they are (a) mutually independent and (b) the same kernel at the
    same shape. Independence is guaranteed structurally rather than tested pairwise: ops are bucketed by
    their longest-path LEVEL, and two ops at the same level cannot have a path between them. Each bucket
    is then cut into chunks of at most P, one per core -- exactly the grid-filling loop.

    A fused CHAIN is kept intact and batched as a unit: its level is its head's, so chains of the same
    shape ride together and the sequential fusion is not undone."""
    deps, succ, _ = dag_cached(prog)
    n = len(prog.ops)
    # level of each op (longest path from a source), then lift to the fused GROUP
    lvl = [0] * n
    for i in range(n):
        if deps[i]:
            lvl[i] = max(lvl[d] for d in deps[i]) + 1
    glvl, gops = {}, {}
    for i in range(n):
        g = mp.group[i]
        glvl[g] = max(glvl.get(g, 0), lvl[i])
        gops.setdefault(g, []).append(i)

    def sig(g):
        """The kernel signature: same class, engine and tile count => the same program can run it."""
        i = gops[g][0]
        ls, ld, ot = op_shape(prog.ops[i])
        return (op_class(prog.ops[i]), mp.engine[i], ot, len(gops[g]),
                ls if op_class(prog.ops[i]) == "contract" else 0)

    buckets = {}
    for g in gops:
        buckets.setdefault((glvl[g], sig(g)), []).append(g)

    before = len(gops)
    newg, nid = {}, 0
    for key in sorted(buckets, key=lambda k: k[0]):
        gs = buckets[key]
        for base in range(0, len(gs), P):                    # cut to the grid, one op per core
            for g in gs[base:base + P]:
                newg[g] = nid
            nid += 1
    for i in range(n):
        mp.group[i] = newg[mp.group[i]]
    widest = max((len(v) for v in buckets.values()), default=0)
    return dict(pass_name="batch_dispatches", groups_before=before, groups_after=nid,
                removed=before - nid, buckets=len(buckets), widest_bucket=widest,
                note=f"{before} -> {nid} dispatches ({before/max(1,nid):.0f}x fewer); "
                     f"{len(buckets)} (level, kernel) buckets, widest {widest} ops")


# --------------------------------------------------------------------------- #
# PASS 3 — select_engines
# --------------------------------------------------------------------------- #
def select_engines(prog, mp: Mapping, P: int) -> dict:
    """Per-op choice of the SFPU or the matrix engine, priced, with legality checked.

    Blackhole has two compute engines and the multiply roofs are 20x apart, so this is the largest
    single decision in the mapping -- and the one the hand-tuned pipeline got wrong for months: the
    matrix-engine base conversion existed, was measured at 1.19x-10.3x, and was simply never wired in
    (exp_tg018ntt 4d wrote it up as future work while the kernel sat in the tree). A pass cannot forget.

    Only contractions are eligible, and only within a tile: the matrix engine contracts along the
    source-limb axis, so one tile holds at most Ld destinations. Ld=32 is the algorithmic limit but
    Ld=24 is the widest VERIFIED one, so that is the cap enforced here."""
    moved, kept, ineligible = 0, 0, 0
    gain = 0.0
    for i, op in enumerate(prog.ops):
        if op_class(op) != "contract":
            ineligible += 1
            continue
        ls, ld, ot = op_shape(op)
        # A destination axis wider than the tile is CHUNKED, not refused: the matrix engine takes
        # FPU_LD_CAP destinations at a time and the op runs ceil(ld/cap) times. Refusing instead of
        # chunking is precisely how a pass would miss the real pipeline's FBC, whose Ld is 48.
        eff_ld = min(ld, int(float(hw.FPU_LD_CAP)))
        if ls > float(hw.FPU_LS_CAP) or op.opcode not in CONTRACT:
            kept += 1
            continue
        sp = hw.fpu_speedup(ls, eff_ld)
        if sp <= 1.0:
            kept += 1                             # too small a contraction to beat the SFPU
            continue
        sf = op_us(op, mp, i)
        mp.engine[i] = hw.FPU
        mp.variant[i] = (mp.variant[i][0], dict(mp.variant[i][1], dst_chunk=eff_ld, fpu_speedup=round(sp, 3)))
        fp = op_us(op, mp, i)
        moved += 1
        gain += sf - fp
    return dict(pass_name="select_engines", moved_to_fpu=moved, kept_on_sfpu=kept,
                not_a_contraction=ineligible, modelled_gain_us=gain,
                note=f"{moved} contractions moved to the matrix engine, "
                     f"modelled saving {gain:.1f} us")


# --------------------------------------------------------------------------- #
# PASS 4 — legalize
# --------------------------------------------------------------------------- #
# Hardware limits that are REAL and were each hit during the hand-tuning.
ARG_BUDGET = 256            # runtime args per kernel; LS=16,LD=12 is already 266 -> chunking
L1_TILES = 64               # working-set tiles per core before spilling
CB_MAX = 32                 # circular-buffer depth

VARIANTS = {
    # op class      -> (kernel, engine, notes)
    ("modmul", hw.SFPU):   ("modmul_sfpi_1pass", "one-pass sfpmul24 montmul"),
    ("addm", hw.SFPU):     ("modadd_map", "elementwise add + conditional subtract"),
    ("contract", hw.SFPU): ("fbc_combine", "SFPU contraction, Ls*Ld modmuls"),
    ("contract", hw.FPU):  ("fbc_fpu_tr", "matrix engine, transpose fused into the I/O"),
    ("decompose", hw.SFPU): ("decompose_i8", "radix-128 planes"),
    ("recombine", hw.SFPU): ("recombine_modq", "place-value combine + reduce"),
    ("a2a", "free"):        ("a2a_corner_x", "core-to-core exchange, semaphore barrier"),
}


def legalize(prog, mp: Mapping, P: int) -> dict:
    """Bind each op to a concrete kernel variant + parameters, and enforce the hardware's real limits.

    Three limits, each of which actually bit during the hand-tuning:
      * runtime-arg budget 256 -- LS=16 with LD=12 is already 266 args, which is why the SFPU base
        conversion chunks its destination list at all;
      * Ld <= 24 on the matrix engine (the verified cap);
      * L1 working set -- a fused chain that needs more than L1_TILES resident spills, which undoes
        the reuse pricing fusion was granted, so the chain is split instead.
    Anything the earlier passes produced that violates these is repaired here, not left to fail on
    device. Reports every repair, because a silent legalisation is how a mapper lies about its own win."""
    chunked, demoted, split, bound = 0, 0, 0, 0
    for i, op in enumerate(prog.ops):
        cls = op_class(op)
        ls, ld, ot = op_shape(op)
        # arg budget: a contraction carrying ls*ld constants as runtime args must chunk its dest list
        params = {}
        if cls == "contract":
            need = ls * ld + 2 * ld + 2
            if need > ARG_BUDGET and mp.engine[i] != hw.FPU:
                ch = max(1, (ARG_BUDGET - 2) // max(1, ls + 2))
                params["dst_chunk"] = ch
                chunked += 1
            # CHUNK FIRST, THEN CHECK. An earlier version re-checked legality against the RAW ld
            # after setting dst_chunk, so every op it had just made legal was immediately demoted --
            # which silently kept the whole four-step NTT on the SFPU at real tile sizes and made the
            # engine pass look like it had nothing to find. Legality is a property of the CHUNK.
            eff_ld = ld
            if mp.engine[i] == hw.FPU and ld > float(hw.FPU_LD_CAP):
                eff_ld = int(float(hw.FPU_LD_CAP))
                params["dst_chunk"] = eff_ld
                params["dst_chunks"] = -(-ld // eff_ld)
                chunked += 1
            if mp.engine[i] == hw.FPU and not hw.fpu_legal(op.opcode, ls, eff_ld):
                mp.engine[i] = hw.SFPU
                demoted += 1
        # L1 capacity: an over-long fused CHAIN would spill, cancelling the reuse it was priced at.
        # This must be the chain length, NOT the dispatch group: after batch_dispatches a group spans
        # up to P cores, and sizing L1 against it over-counts by the batch width.
        if mp.resident[i] and mp.chain[i] * ot > L1_TILES:
            mp.resident[i] = False
            split += 1
        key = (cls, mp.engine[i])
        kern = VARIANTS.get(key)
        if kern is None:
            kern = ("<structural>", "no kernel: relayout / constant")
        mp.variant[i] = (kern[0], params)
        bound += 1
    return dict(pass_name="legalize", bound=bound, chunked=chunked, demoted_from_fpu=demoted,
                chains_split_for_l1=split,
                note=f"{bound} ops bound to kernel variants; {chunked} chunked for the "
                     f"{ARG_BUDGET}-arg budget, {demoted} demoted off the FPU, {split} chains split for L1")


# --------------------------------------------------------------------------- #
# the pass manager
# --------------------------------------------------------------------------- #
PIPELINE = [place_comm, fuse_dispatches, batch_dispatches, select_engines, legalize]


def run_passes(prog, P: int, enable=None, verbose: bool = True) -> tuple:
    """Run the pipeline, recording the predicted makespan after each pass -> the compiler's own ladder.

    `enable` = set of pass names to run (None = all). Disabled passes are still reported, as skipped,
    so the ladder has the same shape whatever is turned on."""
    mp = Mapping.initial(prog)
    est0 = estimate(prog, mp, P)
    rungs = [dict(pass_name="(unoptimised)", makespan=est0["makespan"], groups=est0["groups"],
                  compute=est0["compute"], note="one dispatch per op, every operand streamed")]
    if verbose:
        print(f"  {'pass':<18} {'makespan':>11} {'groups':>8} {'delta':>9}   note")
        print(f"  {'-'*18} {'-'*11} {'-'*8} {'-'*9}   {'-'*52}")
        print(f"  {'(unoptimised)':<18} {est0['makespan']/1e3:>8.3f} ms {est0['groups']:>8} "
              f"{'':>9}   one dispatch per op, all operands streamed")
    prev = est0["makespan"]
    for fn in PIPELINE:
        nm = fn.__name__
        if enable is not None and nm not in enable:
            rungs.append(dict(pass_name=nm, skipped=True, makespan=prev))
            if verbose:
                print(f"  {nm:<18} {'-- skipped --':>11}")
            continue
        rep = fn(prog, mp, P)
        est = estimate(prog, mp, P)
        rep["makespan"] = est["makespan"]
        rep["groups"] = est["groups"]
        rep["compute"] = est["compute"]
        rep["delta"] = est["makespan"] - prev
        rungs.append(rep)
        if verbose:
            print(f"  {nm:<18} {est['makespan']/1e3:>8.3f} ms {est['groups']:>8} "
                  f"{rep['delta']/1e3:>+8.3f}ms   {rep['note']}")
        prev = est["makespan"]
    mp.reports = rungs
    final = estimate(prog, mp, P)
    if verbose:
        tot = rungs[0]["makespan"] / max(1e-9, final["makespan"])
        print(f"\n  end to end: {rungs[0]['makespan']/1e3:.3f} -> {final['makespan']/1e3:.3f} ms "
              f"= {tot:.2f}x   ({final['cores_used']}/{P} cores used)")
    return mp, final, rungs


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import random
    from ckks_mul_ref import CKKSParams, encrypt
    from bfv_relin_dnum import keygen_relin_dnum
    from bfv_relin_ref import small_poly
    from builders import build_ct_mul_ckks
    from lower import lower_l1_to_l2
    import ir
    import reference as ref

    pp = CKKSParams(N=32, LQ=4, LP=2)
    rng = random.Random(0)
    s = small_poly(32, rng)
    ks = keygen_relin_dnum(pp, s, rng, 2)
    m1 = [rng.randrange(-3, 4) for _ in range(32)]
    m2 = [rng.randrange(-3, 4) for _ in range(32)]
    l1, inp, o0, o1, base = build_ct_mul_ckks(pp, encrypt(pp, m1, s, rng), encrypt(pp, m2, s, rng), ks)
    l2 = lower_l1_to_l2(l1, factors=(8, 4), T=32)
    gold = ir.run(l2, inp)

    print(f"=== CKKS ct_mul N=32 LQ=4 dnum=2 -> {len(l2.ops)} L2 ops, 130 cores ===\n")
    mp, final, rungs = run_passes(l2, hw.CORES)

    print("\n=== the mapping is ANNOTATION: the program still computes the same function ===")
    same = all(ref._eq(gold[k], ir.run(l2, inp)[k]) for k in gold)
    print(f"  run(L2) unchanged after all four passes: {'OK' if same else 'FAIL'}")

    print("\n=== engine histogram ===")
    hist = {}
    for i, e in enumerate(mp.engine):
        k = (op_class(l2.ops[i]), e)
        hist[k] = hist.get(k, 0) + 1
    for (cls, eng), c in sorted(hist.items(), key=lambda x: -x[1]):
        print(f"  {cls:<12} {eng:<6} {c:>5}")

    print("\n=== tier histogram ===")
    th = {}
    for t in mp.tier.values():
        th[t] = th.get(t, 0) + 1
    for t, c in sorted(th.items(), key=lambda x: -x[1]):
        print(f"  {t:<6} {c:>5} edges")

    ok = same and final["makespan"] > 0
    print("\nPASSES OK" if ok else "\nPASSES FAIL")
