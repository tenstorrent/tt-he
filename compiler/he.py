#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""he.py — THE entry point. One script: compile -> optimise -> schedule -> codegen -> run -> visualise.

    python3 he.py                 # everything, using the CONFIG block below
    python3 he.py --case tile32    # a named demo case instead (see CASES)
    python3 he.py --list           # what the demo cases are and how long each takes

Edit CONFIG at the top and run it. It compiles the workload, runs the four optimisation passes,
schedules across the grid, emits a device spec, builds and dispatches the host on the p150, verifies
bit-exact against the compiler's own golden, writes the visual tool and serves it, then prints where
the generated host and binary are.

ONE DEVICE JOB AT A TIME. A second concurrent tt-metal program wedges the card and needs `tt-smi -r`.
"""
from __future__ import annotations

# ============================== CONFIG (edit me) ============================== #
HARDWARE = dict(
    cores      = 130,          # Tensix cores to schedule across (this p150 reports 130 usable)
    # everything else about the card -- roofs, bandwidths, dispatch tax, engine speedups -- is
    # MEASURED and lives in hw.py with its provenance. Do not guess it here.
)

WORKLOAD = dict(
    scheme     = "ckks",       # "ckks" (validated against ckks_mul_ref) or "bfv"
    N          = 256,          # ring degree
    LQ         = 8,            # ciphertext modulus limbs
    LP         = 4,            # special-base limbs (key-switch)
    dnum       = 2,            # hybrid key-switch digits (CKKS path needs >= 2, see ckks_mul_ref)
    ndrop      = 1,            # primes the rescale drops
    ntt        = (16, 16),     # four-step NTT factors; None = single DFT matmul
    tile       = 32,           # matrix-engine tile edge
    validate   = True,         # real keygen+encrypt and a decrypt check (costs O(N^2); off = same graph)
)

PASSES = ("place_comm", "fuse_dispatches", "batch_dispatches",              # subset to run, in order
           "select_engines", "legalize")

STAGES = dict(
    l3         = False,        # also lower to int8 (L3) -- ~5x more ops, slower, not needed for mapping
    device     = True,         # emit a spec, build the host, dispatch on the p150, verify bit-exact
    viz        = True,         # write the visual tool
    serve      = True,         # ... and serve it
    port       = 8017,
)
# ============================================================================== #

import argparse
import functools
import http.server
import json
import os
import random
import re
import shutil
import socketserver
import struct
import subprocess
import sys
import time

import _paths                                     # noqa: F401  (repo layout; see _paths.py)
import hw
import ir
import passes as P
import reference as ref
import devgen
import viz_app
from ir import Op, Program, Value, outputs

ROOT = _paths.ROOT
GEN = _paths.GEN
SPEC = os.path.join(GEN, "codegen_spec.bin")
DEV_BIN = os.path.join(ROOT, "build", "he_run")
DEV_SPEC = os.path.join(GEN, "he_spec.bin")


# --------------------------------------------------------------------------- #
# demo cases -- timings measured on this box, see docs/compiler/he_compiler.md
# --------------------------------------------------------------------------- #
CASES = {
    #  name             N    LQ  LP dnum  ntt        validate  note
    "tiny":       dict(N=16,  LQ=4,  LP=2, dnum=2, ntt=(4, 4),   validate=True),
    "small":      dict(N=32,  LQ=4,  LP=2, dnum=2, ntt=(8, 4),   validate=True),
    "small-deep": dict(N=32,  LQ=8,  LP=4, dnum=2, ntt=(8, 4),   validate=True),
    "mid":        dict(N=64,  LQ=6,  LP=2, dnum=2, ntt=(8, 8),   validate=True),
    "mid-alpha8": dict(N=64,  LQ=16, LP=8, dnum=2, ntt=(8, 8),   validate=True),
    "tile32":     dict(N=256, LQ=8,  LP=4, dnum=2, ntt=(16, 16), validate=True),
    # N >= 1024 with a SMALL limb count. The degree is what matters for the device path: matmul_modq
    # takes a 32x32 tile, so the four-step factors have to reach (32,32) for a DFT block to fill one --
    # which happens at N=1024 and not below. Small L keeps the op count (and the wall time) down.
    "n1k":        dict(N=1024, LQ=4,  LP=2,  dnum=2, ntt=(32, 32), validate=False),
    "n2k":        dict(N=2048, LQ=4,  LP=2,  dnum=2, ntt=(32, 64), validate=False),
    "n4k":        dict(N=4096, LQ=3,  LP=2,  dnum=2, ntt=(64, 64), validate=False),
    "n8k":        dict(N=8192, LQ=3,  LP=2,  dnum=2, ntt=(64, 128), validate=False),
    "n16k":       dict(N=16384, LQ=3, LP=2,  dnum=2, ntt=(128, 128), validate=False),
    "big":        dict(N=1024, LQ=8, LP=4, dnum=2, ntt=(32, 32), validate=False),
    "hw-shape":   dict(N=1024, LQ=48, LP=16, dnum=3, ntt=(32, 32), validate=False),
}


def log(msg, t0=[None]):
    if t0[0] is None:
        t0[0] = time.time()
    print(f"  [{time.time()-t0[0]:6.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 1. build the workload
# --------------------------------------------------------------------------- #
def build(w):
    """-> dict with the L1/L2(/L3) programs, inputs, and enough key material to decrypt-check."""
    from builders import build_ct_mul_ckks, build_ct_mul_rns
    from lower import lower_l1_to_l2, lower_l2_to_l3
    from bfv_relin_dnum import keygen_relin_dnum, q_groups
    from bfv_relin_ref import small_poly

    N, LQ, LP, dnum = w["N"], w["LQ"], w["LP"], w["dnum"]
    rng = random.Random(0)
    out = {}

    if w["scheme"] == "ckks":
        from ckks_mul_ref import CKKSParams, encrypt
        pp = CKKSParams(N=N, LQ=LQ, LP=LP, ndrop=w["ndrop"])
        if w["validate"]:
            log("keygen + encrypt (real, for the decrypt check) ...")
            s = small_poly(N, rng)
            ks = keygen_relin_dnum(pp, s, rng, dnum)
            m1 = [rng.randrange(-3, 4) for _ in range(N)]
            m2 = [rng.randrange(-3, 4) for _ in range(N)]
            ct1, ct2 = encrypt(pp, m1, s, rng), encrypt(pp, m2, s, rng)
        else:
            # STRUCTURAL mode: dummy zero keys/data. The op GRAPH is bit-identical -- N only sets tile
            # counts -- so every mapping decision is the same; only the decrypt check is unavailable.
            log("structural mode: dummy keys/data (identical graph, no O(N^2) crypto) ...")
            s, m1, m2 = None, [0] * N, [0] * N
            z = [0] * N
            groups = q_groups(LQ, dnum)
            ks = ([([list(z) for _ in pp.QP], [list(z) for _ in pp.QP]) for _ in groups], groups)
            ct1 = ct2 = (list(z), list(z))
        log(f"building L1 (tensor -> relin -> add -> rescale), CKKS ...")
        l1, inp, o0, o1, base = build_ct_mul_ckks(pp, ct1, ct2, ks, ndrop=w["ndrop"])
        out.update(base=base, o0=o0, o1=o1)
    else:
        from bfv_mul_ref import MulParams, encrypt
        pp = MulParams(N=N, LQ=LQ, LP=LP, LB=LQ + 1)
        log("keygen + encrypt (BFV) ...")
        s = small_poly(N, rng)
        ks = keygen_relin_dnum(pp, s, rng, dnum) if dnum > 1 else None
        m1 = [rng.randrange(pp.t) for _ in range(N)]
        m2 = [rng.randrange(pp.t) for _ in range(N)]
        ct1, ct2 = encrypt(pp, m1, s, rng), encrypt(pp, m2, s, rng)
        log("building L1 (tensor -> rescale -> relin -> add), BFV ...")
        if ks:
            l1, inp, o0, o1 = build_ct_mul_rns(pp, ct1, ct2, ks=ks)
        else:
            from bfv_relin_ref import keygen_relin
            e0, e1 = keygen_relin(pp, s, rng)
            l1, inp, o0, o1 = build_ct_mul_rns(pp, ct1, ct2, evk0=e0, evk1=e1)
        out.update(base=list(pp.Q), o0=o0, o1=o1)

    log(f"L1: {len(l1.ops)} ops")
    log(f"lowering L1 -> L2 (tile ops, four-step NTT {w['ntt']}, tile {w['tile']}) ...")
    l2 = lower_l1_to_l2(l1, factors=w["ntt"], T=w["tile"])
    log(f"L2: {len(l2.ops)} ops")
    l3 = None
    if STAGES["l3"]:
        log("lowering L2 -> L3 (int8) ...")
        l3 = lower_l2_to_l3(l2)
        log(f"L3: {len(l3.ops)} ops")
    out.update(pp=pp, l1=l1, l2=l2, l3=l3, inputs=inp, s=s, m1=m1, m2=m2, ks=ks)
    return out


# --------------------------------------------------------------------------- #
# 2. correctness
# --------------------------------------------------------------------------- #
def check_correct(w, b):
    """Decrypt the compiled program's own output, and confirm the tile lowering is meaning-preserving."""
    res = {}
    if not w["validate"]:
        # STRUCTURAL MODE: inputs and keys are dummy zeros, so run(L1)==run(L2) would be checking that
        # zero equals zero -- vacuous, and at N=1024 it is also the single slowest thing in the whole
        # pipeline (~170 s of object-array interpretation over ~10k ops). Skipped, and SAID to be
        # skipped. Mapping decisions depend on the graph, not on the data.
        log("structural mode: skipping the interpreter (dummy data -- the check would be vacuous)")
        res["skipped"] = True
        return res
    log("running the reference interpreter (the golden) ...")
    g1 = ir.run(b["l1"], b["inputs"])
    log("run(L1) == run(L2) ? (the lowering must be a refinement, not a hope) ...")
    g2 = ir.run(b["l2"], b["inputs"])
    res["lowering"] = all(ref._eq(g1[k], g2[k]) for k in g1)
    if b["l3"] is not None:
        g3 = ir.run(b["l3"], b["inputs"])
        res["lowering_l3"] = all(ref._eq(g1[k], g3[k]) for k in g1)
    if w["validate"] and w["scheme"] == "ckks":
        from ckks_mul_ref import decrypt, crt_poly, _exact_product
        from math import prod
        pp, N = b["pp"], w["N"]
        r0 = [[int(v) for v in g1[n]] for n in b["o0"]]
        r1 = [[int(v) for v in g1[n]] for n in b["o1"]]
        ct = (crt_poly(r0, b["base"], N), crt_poly(r1, b["base"], N))
        got = decrypt(pp, ct, b["s"], Qprod=prod(b["base"]),
                      scale=(pp.Delta ** 2, prod(pp.Q[len(b["base"]):])))
        want = _exact_product(b["m1"], b["m2"], N)
        res["decrypt"] = got == want
        if not res["decrypt"]:
            res["decrypt_err"] = max(abs(x - y) for x, y in zip(got, want))
    res["golden"] = g1
    return res


# --------------------------------------------------------------------------- #
# 3. device: slice the pointwise sub-DAG out of the REAL compiled program and run it
# --------------------------------------------------------------------------- #
OPMAP = {"rns.mul": "tile.mul_modq", "rns.add": "tile.add_modq", "rns.sub": "tile.sub_modq",
          "tile.mul_modq": "tile.mul_modq", "tile.add_modq": "tile.add_modq",
          "tile.sub_modq": "tile.sub_modq"}
DEV_N = 1024                                    # the device kernels work on one 32x32 tile


def device_primes(prog, inputs):
    """Which primes have a dispatchable pointwise sub-DAG, largest first."""
    from collections import Counter
    qc = Counter(outputs(op)[0].q for op in prog.ops if op.opcode in OPMAP)
    return [q for q, _ in qc.most_common()], sum(qc.values())


def device_slice(prog, inputs, q=None, want_n=DEV_N, max_ops=4096, env=None):
    """The largest single-prime pointwise sub-DAG of the compiled program, as a runnable Program.

    WHY A SLICE. The existing device kernels (`kernels/ir/`) cover the pointwise ops -- modmul, modadd,
    modsub -- and the spec format carries ONE prime, so what can actually be dispatched today is the
    biggest single-prime pointwise sub-DAG. That is a real part of the compiled ct_mul, driven with the
    REAL intermediate values it sees in place (ir.run(all_values=True)), not a synthetic workload that
    merely resembles it. Everything else in the multiply is emitted into the dispatch plan but not run;
    the report says so rather than implying full coverage.

    Values shorter than a tile are zero-padded to `want_n`: the kernels process 1024 elements and
    0*0 = 0+0 = 0, so padding cannot change the answer, and the golden is padded identically."""
    if env is None:
        env = ir.run(prog, inputs, all_values=True)
    cand = [(i, op) for i, op in enumerate(prog.ops) if op.opcode in OPMAP]
    if not cand:
        return None
    if q is None:
        from collections import Counter
        qc = Counter(outputs(op)[0].q for _, op in cand)
        q = qc.most_common(1)[0][0]
    sel = [(i, op) for i, op in cand if outputs(op)[0].q == q][:max_ops]
    if not sel:
        return None
    made = {outputs(op)[0].name for _, op in sel}

    def pad(name):
        v = env.get(name)
        if v is None:
            return None
        flat = v.reshape(-1).tolist() if hasattr(v, "reshape") else list(v)
        lst = [int(x) % q for x in flat]
        if len(lst) == 1:                       # a scalar constant: broadcast it across the tile
            lst = lst * want_n
        return (lst + [0] * want_n)[:want_n]

    nv, data, ops, in_vals = {}, {}, [], []

    def get(o):
        if o.name in nv:
            return nv[o.name]
        val = Value(o.name, (want_n,), q=q)
        nv[o.name] = val
        if o.name not in made:                  # a leaf of the slice -> an input carrying real data
            d = pad(o.name)
            if d is None:
                return None
            data[o.name] = d
            in_vals.append(val)
        return val

    ok = []
    for i, op in sel:
        a = get(op.operands[0]) if len(op.operands) > 0 else None
        bb = get(op.operands[1]) if len(op.operands) > 1 else None
        if a is None or bb is None:
            continue
        r = outputs(op)[0]
        rv = Value(r.name, (want_n,), q=q)
        nv[r.name] = rv
        ops.append(Op(OPMAP[op.opcode], [a, bb], rv, {"q": q}))
        ok.append(r.name)
    if not ops:
        return None
    consumed = {o.name for op in ops for o in op.operands}
    outs = [nv[n] for n in ok if n not in consumed]
    sub = Program(inputs=in_vals, ops=ops, outputs=outs)
    return dict(prog=sub, data=data, q=q, n=want_n, n_src_ops=len(sel))


_BUILT = [False]


def run_device_v2(rep, quiet=False):
    """Build he_run if needed, dispatch the v2 spec, parse the verdict."""
    out_rep = dict(rep)
    if not _BUILT[0]:
        log("building the device host (cmake --target he_run) ...")
        r = subprocess.run(["cmake", "--build", "build", "--target", "he_run"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            out_rep["error"] = "build failed:\n" + r.stderr[-1500:]
            return out_rep
        _BUILT[0] = True
    out_rep["binary"] = DEV_BIN
    if not quiet:
        log(f"spec: {rep['ops']} ops, {rep['buffers']} buffers, {rep['weights']} folded weights, "
            f"{rep['bytes']} bytes -> {DEV_SPEC}")
        log("dispatching on the p150 ...")
    r = subprocess.run([DEV_BIN, DEV_SPEC], cwd=ROOT, capture_output=True, text=True)
    o = r.stdout + r.stderr
    m_ok = re.search(r"bit-exact vs golden:\s*(\w+)", o)
    m_w = re.search(r"in (\d+) waves", o)
    m_d = re.search(r"steady-state device (\d+)", o)
    out_rep["correct"] = m_ok.group(1) if m_ok else "UNKNOWN"
    out_rep["waves"] = int(m_w.group(1)) if m_w else None
    out_rep["device_us"] = float(m_d.group(1)) if m_d else None
    if not m_ok:
        out_rep["raw"] = o[-1200:]
    return out_rep


def run_device(sl, cores, quiet=False):
    """Emit the spec, build the host, dispatch, verify. -> report dict."""
    import codegen as cg
    import schedule as sched
    os.makedirs(GEN, exist_ok=True)
    golden = ir.run(sl["prog"], sl["data"])
    sc = sched.schedule(sl["prog"], cores)
    nbuf, nop = cg.emit(sl["prog"], sl["data"], golden, sc, SPEC, sl["q"], sl["n"])
    rep = dict(spec=SPEC, spec_bytes=os.path.getsize(SPEC), buffers=nbuf, ops=nop,
               cores=sc["cores_used"], predicted_us=sc["makespan"])
    if not quiet:
        log(f"spec: {nop} ops, {nbuf} buffers, {rep['spec_bytes']} bytes -> {SPEC}")
    if not _BUILT[0]:
        log("building the device host (cmake --target codegen_run) ...")
        r = subprocess.run(["cmake", "--build", "build", "--target", "codegen_run"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            rep["error"] = "build failed:\n" + r.stderr[-1500:]
            return rep
        _BUILT[0] = True
    rep["binary"] = DEV_BIN
    if not quiet:
        log("dispatching on the p150 ...")
    r = subprocess.run([DEV_BIN, SPEC], cwd=ROOT, capture_output=True, text=True)
    out = r.stdout + r.stderr
    m_ok = re.search(r"bit-exact vs golden:\s*(\w+)", out)
    m_w = re.search(r"in (\d+) waves", out)
    m_d = re.search(r"steady-state device (\d+)", out)
    rep["correct"] = (m_ok.group(1) if m_ok else "UNKNOWN")
    rep["waves"] = int(m_w.group(1)) if m_w else None
    rep["device_us"] = float(m_d.group(1)) if m_d else None
    if not m_ok:
        rep["raw"] = out[-1200:]
    return rep


# --------------------------------------------------------------------------- #
# 4. the dispatch plan for the WHOLE program (emitted, not executed)
# --------------------------------------------------------------------------- #
def write_plan(prog, mp, est, path):
    """Per op: kernel variant, params, engine, core, dispatch group. The full mapping, as JSON."""
    plan = []
    for i, op in enumerate(prog.ops):
        kern, params = mp.variant[i]
        ls, ld, ot = P.op_shape(op)
        plan.append(dict(i=i, opcode=op.opcode, cls=P.op_class(op), engine=mp.engine[i],
                         kernel=kern, params=params, core=est["core"][i],
                         group=mp.group[i], tiles=ot, ls=ls, ld=ld,
                         start_us=round(est["start"][i], 4), dur_us=round(est["dur"][i], 5)))
    doc = dict(ops=len(plan), groups=est["groups"], cores=est["cores_used"],
               makespan_us=round(est["makespan"], 3), plan=plan)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return doc


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def serve(directory, port):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"\n  serving  http://localhost:{port}/compiler_viz.html    (ctrl-C to stop)\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  stopped.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", help="a named demo case (overrides the WORKLOAD block)")
    ap.add_argument("--list", action="store_true", help="list the demo cases")
    ap.add_argument("--cores", type=int, help="override HARDWARE['cores']")
    ap.add_argument("--no-device", action="store_true")
    ap.add_argument("--all-primes", action="store_true",
                    help="dispatch the pointwise slice of EVERY prime (full coverage, one program each)")
    ap.add_argument("--no-serve", action="store_true")
    ap.add_argument("--port", type=int)
    a = ap.parse_args()

    if a.list:
        print("demo cases (python3 he.py --case NAME):\n")
        print(f"  {'name':<12} {'N':>6} {'LQ':>4} {'LP':>4} {'dnum':>5} {'ntt':>10} {'decrypt':>8}")
        for nm, c in CASES.items():
            print(f"  {nm:<12} {c['N']:>6} {c['LQ']:>4} {c['LP']:>4} {c['dnum']:>5} "
                  f"{str(c['ntt']):>10} {'yes' if c['validate'] else 'no':>8}")
        print("\nsee docs/compiler/he_compiler.md for measured compile times.")
        return 0

    w = dict(WORKLOAD)
    if a.case:
        if a.case not in CASES:
            print(f"unknown case {a.case!r}; try --list"); return 2
        w.update(CASES[a.case])
    cores = a.cores or HARDWARE["cores"]
    if a.no_device:
        STAGES["device"] = False
    if a.no_serve:
        STAGES["serve"] = False
    port = a.port or STAGES["port"]
    os.makedirs(GEN, exist_ok=True)

    t_all = time.time()
    print("=" * 78)
    print(f"HE COMPILER   {w['scheme'].upper()} ct_mul   N={w['N']} L_Q={w['LQ']} P={w['LP']} "
          f"dnum={w['dnum']} ndrop={w['ndrop']}")
    print(f"              NTT={w['ntt']} tile={w['tile']}   ->  {cores} cores (Blackhole p150)")
    print("=" * 78)

    # ---- compile ----
    print("\n[1/6] compile")
    b = build(w)

    # ---- correctness ----
    print("\n[2/6] correctness")
    cc = check_correct(w, b)
    if cc.get("skipped"):
        print("      skipped -- structural mode (dummy data; the graph is what matters here)")
    else:
        print(f"      lowering  run(L1)==run(L2) : {'OK' if cc['lowering'] else 'FAIL'}")
        if "lowering_l3" in cc:
            print(f"      lowering  ==run(L3)        : {'OK' if cc['lowering_l3'] else 'FAIL'}")
        if "decrypt" in cc:
            print(f"      decrypt   == exact m1*m2   : "
                  f"{'OK' if cc['decrypt'] else 'FAIL (err ' + str(cc.get('decrypt_err')) + ')'}")

    # ---- optimise ----
    print("\n[3/6] optimisation passes")
    mp, est, rungs = P.run_passes(b["l2"], cores, enable=set(PASSES))

    # ---- plan ----
    print("\n[4/6] dispatch plan")
    plan_path = os.path.join(GEN, "dispatch_plan.json")
    doc = write_plan(b["l2"], mp, est, plan_path)
    kh = {}
    for p in doc["plan"]:
        kh[(p["kernel"], p["engine"])] = kh.get((p["kernel"], p["engine"]), 0) + 1
    for (k, e), c in sorted(kh.items(), key=lambda x: -x[1]):
        print(f"      {c:>6}  {k:<22} {e}")
    log(f"wrote {plan_path}")

    # ---- device ----
    dev = None
    if STAGES["device"]:
        print("\n[5/6] device")
        src = b["l2"]
        from collections import Counter
        qc = Counter(outputs(op)[0].q for op in src.ops
                     if op.opcode in devgen.OPK or op.opcode in devgen.MATMUL)
        primes = [q for q, _ in qc.most_common()]
        # Real intermediate values need ir.run over the whole L2 program, which costs minutes past
        # N=256. Past that the slice is driven with deterministic pseudorandom leaves instead -- the op
        # structure, weights and primes are still the compiled program's, and the report says which.
        env = None
        if w["validate"] and len(src.ops) <= 6000:
            log("evaluating the program for real intermediate operands ...")
            env = ir.run(src, b["inputs"], all_values=True)
        else:
            log("large program: slice driven with pseudorandom leaves (structure/weights are the "
                "compiled program's)")
        todo = primes if a.all_primes else primes[:1]
        runs, tot_pw, tot_mm, tot_us, allok = [], 0, 0, 0.0, True
        for qi in todo:
            sl = devgen.build_slice(src, qi, env=env)
            if sl is None:
                continue
            import schedule as sched
            ncore = min(cores, 64)
            corelist = [i % ncore for i in range(len(sl["ops"]))]
            rep = devgen.emit(sl, corelist, DEV_SPEC)
            d = run_device_v2(rep, quiet=len(todo) > 1)
            if "error" in d:
                print("      " + d["error"].splitlines()[0]); allok = False; break
            runs.append(d); tot_pw += sl["n_pw"]; tot_mm += sl["n_mm"]
            tot_us += (d["device_us"] or 0.0)
            allok &= (d["correct"] == "OK")
            print(f"      q={qi:<11} {sl['n_pw']:>4} pointwise + {sl['n_mm']:>4} MATMUL  "
                  f"{d['waves']} waves  {d['device_us'] or 0:>7.0f} us   {d['correct']}")
        if runs:
            dev = dict(runs[-1])
            dev.update(n_runs=len(runs), pw=tot_pw, mm=tot_mm, total_us=tot_us,
                       correct="OK" if allok else "FAIL", synthetic=(env is None))
            print("      ---")
            print(f"      correctness   : {dev['correct']}  (bit-exact vs the compiler's golden, "
                  f"every run)")
            print(f"      executed      : {tot_pw} pointwise + {tot_mm} matmul ops on the matrix "
                  f"engine, {len(runs)} program(s)")
            print(f"      device        : {tot_us:.0f} us total"
                  + ("   [pseudorandom leaves]" if env is None else "   [real intermediate operands]"))
    else:
        print("\n[5/6] device  -- skipped")

    # ---- visualise ----
    print("\n[6/6] visual tool")
    viz_path = os.path.join(GEN, "compiler_viz.html")
    meta = dict(title=f"{w['scheme']} ct_mul N={w['N']} LQ={w['LQ']}",
                line=(f"{w['scheme'].upper()} ct_mul · N={w['N']} L_Q={w['LQ']} P={w['LP']} "
                      f"dnum={w['dnum']} · NTT={w['ntt']} · {len(b['l2'].ops)} tile ops · "
                      f"{cores} cores · Blackhole p150"))
    if STAGES["viz"]:
        n = viz_app.write(viz_path, b["l2"], mp, est, rungs, meta)
        log(f"wrote {viz_path} ({n/1e6:.2f} MB, self-contained)")

    # ---- where everything is ----
    print("\n" + "=" * 78)
    print("GENERATED ARTIFACTS")
    print("=" * 78)
    print(f"  directory       : {GEN}")
    print(f"  dispatch plan   : {plan_path}")
    if STAGES["viz"]:
        print(f"  visual tool     : {viz_path}")
    if dev and "binary" in dev:
        print(f"  device spec     : {DEV_SPEC}  ({dev.get('bytes', 0)} bytes)")
        print(f"  host source     : {os.path.join(ROOT, 'tests', 'ir', 'he_run.cpp')}")
        print(f"  host binary     : {dev['binary']}")
        print(f"  re-run directly : {dev['binary']} {DEV_SPEC}")
    print(f"\n  total {time.time()-t_all:.1f}s")
    print("=" * 78)

    if STAGES["viz"] and STAGES["serve"]:
        serve(GEN, port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
