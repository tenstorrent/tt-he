# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""codegen.py — stage 7 (final): emit a placed L2 program as a runnable spec for the device.

Walks an L2 Program (SFPU tile ops) + a schedule and writes a compact binary spec that the C++
runtime (kernels/ir/codegen_run.cpp) executes: a DRAM buffer per Value, each op dispatched
as reader_binop -> mul/add/sub -> writer_out, independent ops placed on different cores. The
runtime validates its output bit-exact against the golden emitted here (ir.run). This closes the
loop: Python compiler -> spec -> device dispatch -> golden-checked result.

Data is UNIFIED MONTGOMERY on device (mul is mont⊙mont→mont, add/sub linear), so the runtime
Montgomery-encodes inputs and decodes the output; the golden here is the logical (natural) result.

Spec (little-endian u32): magic,N,q,nbuf,nop | kind[nbuf] (0=inter,1=in,2=out) |
  nop×(opkind{0mul,1add,2sub}, in0, in1, out, core) | in-buffer data (logical) | out golden (logical).
"""
from __future__ import annotations
import os, struct, sys, random

import _paths                                     # noqa: F401  (repo layout; see _paths.py)
import ir                                    # noqa: E402
from ir import Value, Op, Program            # noqa: E402
import schedule as sched                     # noqa: E402

OPK = {"tile.mul_modq": 0, "tile.add_modq": 1, "tile.sub_modq": 2}


def build_pointwise_dag(K: int, N: int, q: int):
    """K independent groups: s = w*x + y*z  (2 muls in parallel, then an add). Real deps + parallelism."""
    rng = random.Random(1)
    inputs, ops, outs, data = [], [], [], {}
    for k in range(K):
        v = {}
        for nm in ("w", "x", "y", "z"):
            val = Value(f"{nm}{k}", (N,), q=q); inputs.append(val); v[nm] = val
            data[val.name] = [rng.randrange(q) for _ in range(N)]
        m0 = Value(f"m0_{k}", (N,), q=q); m1 = Value(f"m1_{k}", (N,), q=q); s = Value(f"s{k}", (N,), q=q)
        ops += [Op("tile.mul_modq", [v["w"], v["x"]], m0, {"q": q}),
                Op("tile.mul_modq", [v["y"], v["z"]], m1, {"q": q}),
                Op("tile.add_modq", [m0, m1], s, {"q": q})]
        outs.append(s)
    return Program(inputs, ops, outs), data


def emit(prog: Program, data: dict, golden: dict, sch: dict, path: str, q: int, N: int):
    order, idx = [], {}
    def bid(v):
        if v.name not in idx:
            idx[v.name] = len(order); order.append(v)
        return idx[v.name]
    for v in prog.inputs: bid(v)
    for op in prog.ops:
        for o in op.operands: bid(o)
        bid(op.result)
    for v in prog.outputs: bid(v)
    inset = {v.name for v in prog.inputs}; outset = {v.name for v in prog.outputs}
    kind = [1 if order[i].name in inset else 2 if order[i].name in outset else 0 for i in range(len(order))]

    with open(path, "wb") as f:
        f.write(struct.pack("<5I", 0xC0DE6E00, N, q, len(order), len(prog.ops)))
        f.write(struct.pack(f"<{len(kind)}I", *kind))
        for i, op in enumerate(prog.ops):
            f.write(struct.pack("<5I", OPK[op.opcode], bid(op.operands[0]), bid(op.operands[1]),
                                bid(op.result), sch["core"][i]))
        for i, v in enumerate(order):
            if kind[i] == 1:
                f.write(struct.pack(f"<{N}I", *[int(x) % q for x in data[v.name]]))
        for i, v in enumerate(order):
            if kind[i] == 2:
                f.write(struct.pack(f"<{N}I", *[int(x) % q for x in golden[v.name]]))
    return len(order), len(prog.ops)


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    P = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    N, q = 1024, 251658241
    prog, data = build_pointwise_dag(K, N, q)
    golden = ir.run(prog, data)                       # logical golden (the compiler's oracle)
    sc = sched.schedule(prog, P)
    path = os.path.join(os.path.dirname(__file__), "artifacts", "codegen_spec.bin")
    nbuf, nop = emit(prog, data, golden, sc, path, q, N)
    print(f"codegen: K={K} groups, {nop} ops, {nbuf} buffers, P={P} cores")
    print(f"  schedule: makespan {sc['makespan']:.0f} us predicted, {sc['cores_used']} cores")
    print(f"  wrote {path}  ({os.path.getsize(path)} bytes)")
    print(sched.report(prog, sc))
