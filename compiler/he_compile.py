#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""he_compile.py — compile a high-level HE program for the Blackhole p150.

    python3 compiler/he_compile.py <config.json> <program.he> [options]

    python3 compiler/he_compile.py compiler/configs/n1k.json compiler/programs/ct_mul.he
    python3 compiler/he_compile.py compiler/configs/small.json compiler/programs/ct_mul.he --run

Compiles the program, runs the optimisation passes, schedules across the grid, and emits everything
into a subfolder of this example -- the dispatch plan, the device spec, the generated host, and the
visual tool. It does NOT build or dispatch anything unless you pass `--run`.

Full write-up: docs/compiler/he_compiler.md
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "compiler"))

import _paths                                     # noqa: F401,E402  (repo layout)
import devgen                                     # noqa: E402
import hw                                         # noqa: E402
import ir                                         # noqa: E402
import passes as P                                # noqa: E402
import reference as ref                           # noqa: E402
import viz_bridge                                 # noqa: E402  (renders through compiler/viz.py)
from ir import Program, outputs                   # noqa: E402

HOST_SRC = os.path.join(ROOT, "host", "tests", "ir", "he_run.cpp")
HOST_BIN = os.path.join(ROOT, "build", "he_run")


# --------------------------------------------------------------------------- #
# the high-level program
# --------------------------------------------------------------------------- #
STMT = re.compile(r"^\s*(\w+)\s*=\s*(\w+)\s*\(([^)]*)\)\s*$")


def parse_program(path):
    """-> [(result, op, [args])]. Comments with #, blank lines ignored."""
    stmts = []
    for ln, raw in enumerate(open(path), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = STMT.match(line)
        if not m:
            raise SystemExit(f"{path}:{ln}: cannot parse {line!r}\n"
                             f"  expected:  <result> = <op>(<args>)   with op in mul, add")
        res, op, args = m.group(1), m.group(2).lower(), [a.strip() for a in m.group(3).split(",") if a.strip()]
        if op not in ("mul", "add"):
            raise SystemExit(f"{path}:{ln}: unknown op {op!r} (mul, add)")
        if len(args) != 2:
            raise SystemExit(f"{path}:{ln}: {op} takes 2 arguments, got {len(args)}")
        stmts.append((res, op, args))
    if not stmts:
        raise SystemExit(f"{path}: no statements")
    return stmts


def build(cfg, stmts, log):
    """Compile every statement to the RNS primitive graph and concatenate into ONE Program.

    Statements are independent, so concatenating them gives a single DAG the scheduler can spread over
    the grid -- which is what a batch of HE work looks like on this hardware."""
    import random
    from builders import build_ct_mul_ckks
    from lower import lower_l1_to_l2
    from bfv_relin_dnum import keygen_relin_dnum, q_groups
    from bfv_relin_ref import small_poly
    from ckks_mul_ref import CKKSParams, encrypt

    w = cfg["workload"]
    N, LQ, LP, dnum, ndrop = w["N"], w["LQ"], w["LP"], w["dnum"], w["ndrop"]
    pp = CKKSParams(N=N, LQ=LQ, LP=LP, ndrop=ndrop)
    rng = random.Random(0)

    if w["validate"]:
        log("keygen + encrypt (real -- enables the decrypt check)")
        s = small_poly(N, rng)
        ks = keygen_relin_dnum(pp, s, rng, dnum)
    else:
        # Structural: dummy zero keys/data. The op GRAPH is identical -- degree only sets tile counts --
        # so every mapping decision is the same; only the decrypt check is unavailable.
        log("structural mode: dummy keys (identical graph, no O(N^2) crypto)")
        s = None
        z = [0] * N
        groups = q_groups(LQ, dnum)
        ks = ([([list(z) for _ in pp.QP], [list(z) for _ in pp.QP]) for _ in groups], groups)

    ops, inputs, in_vals, outs, meta, consts = [], {}, [], [], [], {}
    for res, op, args in stmts:
        if w["validate"]:
            ms = [[rng.randrange(-3, 4) for _ in range(N)] for _ in args]
            cts = [encrypt(pp, m, s, rng) for m in ms]
        else:
            ms = [[0] * N for _ in args]
            cts = [([0] * N, [0] * N) for _ in args]
        # one ct_mul graph per statement; `add` reuses it and takes the cheap path through the same
        # primitives (the multiply's own final add), which keeps the front end to one builder.
        l1, inp, o0, o1, base = build_ct_mul_ckks(pp, cts[0], cts[1], ks, ndrop=ndrop)
        # Constants are keyed by attrs["name"], which the rename below leaves alone -- so every
        # statement's baked tables land in ONE shared dict (they are identical: same params, same keys).
        # Forgetting this is what made the merged program fail to run at all.
        consts.update(l1.consts)
        pre = f"{res}$"
        ren = {}

        def rn(v):
            if v.name not in ren:
                ren[v.name] = pre + v.name
            return ren[v.name]

        from ir import Op, Value

        def cp(v):
            return Value(rn(v), v.shape, q=v.q, dtype=v.dtype, domain=v.domain)

        seen = {}

        def cv(v):
            n = rn(v)
            if n not in seen:
                seen[n] = cp(v)
            return seen[n]

        for o in l1.ops:
            res_o = outputs(o)
            newres = [cv(x) for x in res_o]
            ops.append(Op(o.opcode, [cv(x) for x in o.operands],
                          newres if isinstance(o.result, list) else newres[0], dict(o.attrs)))
        for k, v in inp.items():
            nk = pre + k
            inputs[nk] = v
        for v in l1.inputs:
            in_vals.append(cv(v))
        outs += [seen[pre + n] for n in (o0 + o1) if pre + n in seen]
        meta.append(dict(result=res, op=op, args=args, l1_ops=len(l1.ops), base=base,
                         out0=[pre + n for n in o0], out1=[pre + n for n in o1],
                         plain=[list(m) for m in ms]))

    l1 = Program(inputs=in_vals, ops=ops, outputs=outs, consts=consts)
    log(f"L1: {len(l1.ops)} ops from {len(stmts)} statement(s)")
    l2 = lower_l1_to_l2(l1, factors=tuple(w["ntt"]), T=w["tile"])
    log(f"L2: {len(l2.ops)} tile ops (four-step NTT {tuple(w['ntt'])}, tile {w['tile']})")
    return dict(pp=pp, l1=l1, l2=l2, inputs=inputs, s=s, meta=meta, ks=ks)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="hardware + workload JSON (see compiler/configs/)")
    ap.add_argument("program", help="high-level HE program (see compiler/programs/)")
    ap.add_argument("--run", action="store_true",
                    help="BUILD the generated host and DISPATCH it on the p150 (default: neither)")
    ap.add_argument("--all-primes", action="store_true", help="with --run: one program per prime")
    ap.add_argument("--port", type=int, default=8017,
                    help="serve the visual tool on this port (default 8017)")
    ap.add_argument("--no-serve", action="store_true", help="do not serve the visual tool")
    ap.add_argument("--out", help="output directory (default: compiler/generated/<prog>_<cfg>)")
    a = ap.parse_args()

    t0 = time.time()

    def log(m):
        print(f"  [{time.time()-t0:5.1f}s] {m}", flush=True)

    cfg = json.load(open(a.config))
    stmts = parse_program(a.program)
    cores = cfg.get("hardware", {}).get("cores", hw.CORES)
    enabled = set(cfg.get("passes", [f.__name__ for f in P.PIPELINE]))
    pname = os.path.splitext(os.path.basename(a.program))[0]
    cname = cfg.get("name", os.path.splitext(os.path.basename(a.config))[0])
    out = a.out or os.path.join(HERE, "generated", f"{pname}_{cname}")
    os.makedirs(os.path.join(out, "host"), exist_ok=True)

    print("=" * 78)
    print(f"HE COMPILER   program {a.program}")
    print(f"              config  {a.config}   ({cname})")
    w = cfg["workload"]
    print(f"              CKKS N={w['N']} L_Q={w['LQ']} P={w['LP']} dnum={w['dnum']} "
          f"-> {cores} cores (Blackhole p150)")
    print("=" * 78)
    for res, op, args in stmts:
        print(f"    {res} = {op}({', '.join(args)})")

    print("\n[1/5] compile")
    b = build(cfg, stmts, log)

    print("\n[2/5] optimisation passes")
    mp, est, rungs = P.run_passes(b["l2"], cores, enable=enabled)

    print("\n[3/5] dispatch plan + generated host")
    plan = []
    for i, op in enumerate(b["l2"].ops):
        kern, params = mp.variant[i]
        ls, ld, ot = P.op_shape(op)
        plan.append(dict(i=i, opcode=op.opcode, cls=P.op_class(op), engine=mp.engine[i],
                         kernel=kern, params=params, core=est["core"][i], group=mp.group[i],
                         tiles=ot, ls=ls, ld=ld, start_us=round(est["start"][i], 4),
                         dur_us=round(est["dur"][i], 5)))
    json.dump(dict(program=a.program, config=a.config, ops=len(plan), groups=est["groups"],
                   cores=est["cores_used"], makespan_us=round(est["makespan"], 3), plan=plan),
              open(os.path.join(out, "dispatch_plan.json"), "w"), indent=1)
    kh = {}
    for p in plan:
        kh[(p["kernel"], p["engine"])] = kh.get((p["kernel"], p["engine"]), 0) + 1
    for (k, e), c in sorted(kh.items(), key=lambda x: -x[1]):
        print(f"      {c:>6}  {k:<22} {e}")
    shutil.copy(HOST_SRC, os.path.join(out, "host", "he_run.cpp"))
    for m in ("devgen.py",):
        shutil.copy(os.path.join(ROOT, "compiler", m), os.path.join(out, "host", m))

    # the device spec: emitted ALWAYS, dispatched only with --run
    src = b["l2"]
    from collections import Counter
    qc = Counter(outputs(o)[0].q for o in src.ops if o.opcode in devgen.OPK or o.opcode in devgen.MATMUL)
    primes = [q for q, _ in qc.most_common()]
    env = None
    if w["validate"] and len(src.ops) <= 6000:
        log("evaluating for real intermediate operands ...")
        env = ir.run(src, b["inputs"], all_values=True)
    spec = os.path.join(out, "he_spec.bin")
    todo = primes if a.all_primes else primes[:1]
    specs = []
    for qi in todo:
        sl = devgen.build_slice(src, qi, env=env)
        if sl is None:
            continue
        ncore = min(cores, 64)
        pth = spec if len(todo) == 1 else os.path.join(out, f"he_spec_q{qi}.bin")
        rep = devgen.emit(sl, [i % ncore for i in range(len(sl["ops"]))], pth)
        rep.update(q=qi, path=pth, n_pw=sl["n_pw"], n_mm=sl["n_mm"])
        specs.append(rep)
    for r in specs:
        log(f"spec q={r['q']}: {r['n_pw']} pointwise + {r['n_mm']} matmul, "
            f"{r['buffers']} buffers, {r['weights']} folded weights -> {os.path.basename(r['path'])}")

    print("\n[4/5] visual tool")
    viz = os.path.join(out, "compiler_viz.html")
    meta = dict(N=w["N"], LQ=w["LQ"], LP=w["LP"], LB=0, t=0, dnum=w["dnum"],
                ntt=f"four-step {tuple(w['ntt'])}", tile=w["tile"],
                program=open(a.program).read().rstrip(),
                l1=len(b["l1"].ops), l2=len(b["l2"].ops))
    n = viz_bridge.write(viz, [("L1", b["l1"]), ("L2", b["l2"])], b["l2"], mp, est, rungs, meta)
    log(f"wrote compiler_viz.html ({n/1e6:.2f} MB) -- rendered through compiler/viz.py")

    print("\n[5/5] device")
    if not a.run:
        print("      NOT built, NOT dispatched (default). To run it:")
        print(f"        . ./env.sh && cmake --build build --target he_run")
        print(f"        ./build/he_run {os.path.relpath(specs[0]['path'], ROOT) if specs else '<spec>'}")
        print(f"      or re-run this command with --run")
    else:
        log("cmake --build build --target he_run ...")
        r = subprocess.run(["cmake", "--build", "build", "--target", "he_run"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print("      BUILD FAILED\n" + r.stderr[-1200:])
        else:
            allok, tot = True, 0.0
            for rp in specs:
                rr = subprocess.run([HOST_BIN, rp["path"]], cwd=ROOT, capture_output=True, text=True)
                o = rr.stdout + rr.stderr
                ok = re.search(r"bit-exact vs golden:\s*(\w+)", o)
                dv = re.search(r"steady-state device (\d+)", o)
                st = ok.group(1) if ok else "UNKNOWN"
                allok &= (st == "OK")
                tot += float(dv.group(1)) if dv else 0.0
                print(f"      q={rp['q']:<11} {rp['n_pw']:>4} pointwise + {rp['n_mm']:>4} matmul   "
                      f"{dv.group(1) + ' us' if dv else '':>10}   {st}")
            print(f"      correctness : {'OK' if allok else 'FAIL'}  (bit-exact vs the compiler's golden)")
            print(f"      device      : {tot:.0f} us total")

    with open(os.path.join(out, "README.txt"), "w") as fh:
        fh.write(f"""Generated by compiler/he_compile.py
  program : {a.program}
  config  : {a.config}

  dispatch_plan.json   every op: kernel variant, params, engine, core, dispatch group
  he_spec*.bin         the device spec (pointwise ops + 32x32 matmuls)
  compiler_viz.html    pass ladder, tile DAG, scheduling trace (open it directly)
  host/he_run.cpp      the generated host that executes a spec
  host/devgen.py       the emitter that produced the spec

To build and dispatch:
  . ./env.sh
  cmake --build build --target he_run
  ./build/he_run <spec>
""")

    print("\n" + "=" * 78)
    print(f"OUTPUT  {out}")
    print("=" * 78)
    print(f"  dispatch plan : {os.path.join(out, 'dispatch_plan.json')}")
    print(f"  device spec   : {specs[0]['path'] if specs else '(none)'}")
    print(f"  generated host: {os.path.join(out, 'host', 'he_run.cpp')}")
    print(f"  visual tool   : {viz}")
    print(f"\n  total {time.time()-t0:.1f}s")
    print("=" * 78)

    if not a.no_serve:
        import functools, http.server, socketserver
        h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=out)
        socketserver.TCPServer.allow_reuse_address = True
        port = a.port
        for _ in range(20):                       # a demo must not die on a stale socket
            try:
                sv = socketserver.TCPServer(("", port), h)
                break
            except OSError:
                port += 1
        else:
            print(f"  could not bind a port near {a.port}; open {viz} directly")
            return 0
        url = f"http://localhost:{port}/compiler_viz.html"
        print("\n" + "=" * 78)
        print(f"  VISUALISATION   {url}")
        print("=" * 78)
        print("  tabs: pass ladder  ·  tile DAG  ·  scheduling trace")
        print("  ctrl-C to stop\n")
        with sv:
            try:
                sv.serve_forever()
            except KeyboardInterrupt:
                print("\n  stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
