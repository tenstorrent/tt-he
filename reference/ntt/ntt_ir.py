# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""NTT IR: opcodes, typed I/O with shapes, optional attributes (PLAN.md §8).

A decomposition strategy compiles to a `Program` (an SSA list of `Op`s). The numpy
interpreter here is the bit-exact golden backend; the same IR can later be lowered to
tt-metal kernels. Every numeric op is mod `q` and outputs reduced values.

The only real kernels a backend must implement are MATMUL_MODQ, MUL_MODQ, TRANSPOSE.
RESHAPE is a free view; CONST materializes a baked constant.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class OpCode(Enum):
    CONST       = "const"        # materialize a baked constant (twiddle / NTT matrix)
    RESHAPE     = "reshape"      # free view, no data movement
    TRANSPOSE   = "transpose"    # full 2D transpose
    SLICE       = "slice"        # extract a contiguous sub-range along one axis (view)
    SPLIT       = "split"        # one tensor -> N equal tiles along one axis (multi-output)
    CONCAT      = "concat"       # join values along one axis (reassemble tiles)
    ALL_TO_ALL  = "all_to_all"   # inter-stage re-tile: concat -> reshape -> split, one op
    MATMUL_MODQ = "matmul_modq"  # 2D modular matmul, optional transpose_a/transpose_b
    MUL_MODQ    = "mul_modq"     # elementwise (Hadamard) mod q, numpy broadcasting ok


@dataclass(frozen=True)
class Value:
    name: str
    shape: tuple                 # 1-D or 2-D; dtype implicitly int in [0, q)

    def __repr__(self):
        return f"{self.name}:{'x'.join(map(str, self.shape))}"


@dataclass
class Op:
    opcode:   OpCode
    operands: list                       # list[Value]
    result:   object                     # Value, or list[Value] for multi-output (SPLIT)
    attrs:    dict = field(default_factory=dict)


def outputs(op: "Op") -> list:
    """The result Values an op defines (a list even for single-output ops)."""
    return op.result if isinstance(op.result, list) else [op.result]


@dataclass
class Program:
    q:       int
    inputs:  list                        # list[Value]
    ops:     list                        # list[Op], SSA / topologically ordered
    output:  Value
    consts:  dict = field(default_factory=dict)   # name -> np.ndarray (object, mod q)
    meta:    dict = field(default_factory=dict)   # e.g. {"factors":..., "layout":"bitrev"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_modq(a, q):
    """Coerce to a Python-int (object) ndarray reduced mod q — exact, no overflow."""
    arr = np.asarray(a, dtype=object)
    return arr % q


def _matmul_shape(a, b, transpose_a, transpose_b):
    a = (a[1], a[0]) if transpose_a else a
    b = (b[1], b[0]) if transpose_b else b
    if a[1] != b[0]:
        raise ValueError(f"matmul inner dims disagree: {a} @ {b}")
    return (a[0], b[1])


# --------------------------------------------------------------------------- #
# shape verifier  (static; runs without data)
# --------------------------------------------------------------------------- #
def verify(program: Program) -> None:
    """Check SSA + per-op shape rules (PLAN §8.2). Raises on the first problem."""
    shapes = {v.name: tuple(v.shape) for v in program.inputs}

    def shape_of(v: Value):
        if v.name not in shapes:
            raise ValueError(f"use of undefined value {v.name}")
        return shapes[v.name]

    for op in program.ops:
        oc = op.opcode
        for v in outputs(op):
            if v.name in shapes:
                raise ValueError(f"SSA violation: {v.name} defined twice")

        if oc is OpCode.SPLIT:
            src = list(shape_of(op.operands[0]))
            ax, parts = op.attrs["axis"], op.attrs["parts"]
            if src[ax] % parts != 0:
                raise ValueError(f"split axis {ax} dim {src[ax]} not divisible by parts {parts}")
            src[ax] //= parts
            each = tuple(src)
            outs = outputs(op)
            if len(outs) != parts:
                raise ValueError(f"split declares {parts} parts but has {len(outs)} results")
            for v in outs:
                if tuple(v.shape) != each:
                    raise ValueError(f"split result {v.name}: declared {v.shape}, expected {each}")
                shapes[v.name] = tuple(v.shape)
            continue

        if oc is OpCode.ALL_TO_ALL:
            ins = [list(shape_of(o)) for o in op.operands]
            ca, rt = op.attrs["concat_axis"], tuple(op.attrs["reshape_to"])
            sa, parts = op.attrs["split_axis"], op.attrs["parts"]
            cat = list(ins[0]); cat[ca] = sum(s[ca] for s in ins)
            if int(np.prod(cat)) != int(np.prod(rt)):
                raise ValueError(f"all_to_all: concat {tuple(cat)} vs reshape_to {rt} element mismatch")
            if rt[sa] % parts != 0:
                raise ValueError(f"all_to_all: reshape_to[{sa}]={rt[sa]} not divisible by parts {parts}")
            each = list(rt); each[sa] //= parts; each = tuple(each)
            outs = outputs(op)
            if len(outs) != parts:
                raise ValueError(f"all_to_all declares {parts} parts but has {len(outs)} results")
            for v in outs:
                if tuple(v.shape) != each:
                    raise ValueError(f"all_to_all result {v.name}: declared {v.shape}, expected {each}")
                shapes[v.name] = tuple(v.shape)
            continue

        res = op.result
        if oc is OpCode.CONST:
            name = op.attrs["name"]
            if name not in program.consts:
                raise ValueError(f"const {name} not in program.consts")
            exp = tuple(op.attrs["shape"])

        elif oc is OpCode.RESHAPE:
            src = shape_of(op.operands[0])
            exp = tuple(op.attrs["shape"])
            if int(np.prod(src)) != int(np.prod(exp)):
                raise ValueError(f"reshape changes element count: {src} -> {exp}")

        elif oc is OpCode.TRANSPOSE:
            (m, n) = shape_of(op.operands[0])
            exp = (n, m)

        elif oc is OpCode.SLICE:
            src = list(shape_of(op.operands[0]))
            ax, st, sz = op.attrs["axis"], op.attrs["start"], op.attrs["size"]
            if not (0 <= st and st + sz <= src[ax]):
                raise ValueError(f"slice out of range on axis {ax}: {st}+{sz} > {src[ax]}")
            src[ax] = sz
            exp = tuple(src)

        elif oc is OpCode.CONCAT:
            shps = [list(shape_of(o)) for o in op.operands]
            ax = op.attrs["axis"]
            base = list(shps[0])
            base[ax] = sum(s[ax] for s in shps)
            for s in shps:
                if [d for i, d in enumerate(s) if i != ax] != [d for i, d in enumerate(shps[0]) if i != ax]:
                    raise ValueError(f"concat shape mismatch off axis {ax}: {shps}")
            exp = tuple(base)

        elif oc is OpCode.MATMUL_MODQ:
            a, b = shape_of(op.operands[0]), shape_of(op.operands[1])
            exp = _matmul_shape(a, b, op.attrs.get("transpose_a", False),
                                       op.attrs.get("transpose_b", False))
            if op.attrs.get("q") != program.q:
                raise ValueError(f"{oc} must carry q={program.q}")

        elif oc is OpCode.MUL_MODQ:
            a, b = shape_of(op.operands[0]), shape_of(op.operands[1])
            exp = tuple(np.broadcast_shapes(a, b))
            if op.attrs.get("q") != program.q:
                raise ValueError(f"{oc} must carry q={program.q}")
        else:
            raise ValueError(f"unknown opcode {oc}")

        if tuple(res.shape) != tuple(exp):
            raise ValueError(f"{oc.value} {res.name}: declared {res.shape}, expected {exp}")
        shapes[res.name] = tuple(res.shape)

    shape_of(program.output)


# --------------------------------------------------------------------------- #
# numpy interpreter  (bit-exact golden backend)
# --------------------------------------------------------------------------- #
def run(program: Program, x) -> np.ndarray:
    """Execute the IR. `x` is the single program input (array-like)."""
    verify(program)
    q = program.q
    if len(program.inputs) != 1:
        raise ValueError("run() supports a single input")
    env = {program.inputs[0].name: _as_modq(x, q)}

    for op in program.ops:
        oc = op.opcode
        if oc is OpCode.CONST:
            val = _as_modq(program.consts[op.attrs["name"]], q)
        elif oc is OpCode.RESHAPE:
            val = env[op.operands[0].name].reshape(tuple(op.attrs["shape"]))
        elif oc is OpCode.TRANSPOSE:
            val = env[op.operands[0].name].T
        elif oc is OpCode.SLICE:
            ax, st, sz = op.attrs["axis"], op.attrs["start"], op.attrs["size"]
            idx = [slice(None)] * env[op.operands[0].name].ndim
            idx[ax] = slice(st, st + sz)
            val = env[op.operands[0].name][tuple(idx)]
        elif oc is OpCode.SPLIT:
            chunks = np.split(env[op.operands[0].name], op.attrs["parts"], axis=op.attrs["axis"])
            for v, c in zip(outputs(op), chunks):
                env[v.name] = c
            continue
        elif oc is OpCode.ALL_TO_ALL:
            full = np.concatenate([env[o.name] for o in op.operands], axis=op.attrs["concat_axis"])
            r = full.reshape(tuple(op.attrs["reshape_to"]))
            for v, c in zip(outputs(op), np.split(r, op.attrs["parts"], axis=op.attrs["split_axis"])):
                env[v.name] = c
            continue
        elif oc is OpCode.CONCAT:
            val = np.concatenate([env[o.name] for o in op.operands], axis=op.attrs["axis"])
        elif oc is OpCode.MATMUL_MODQ:
            a = env[op.operands[0].name]
            b = env[op.operands[1].name]
            if op.attrs.get("transpose_a", False):
                a = a.T
            if op.attrs.get("transpose_b", False):
                b = b.T
            val = (a @ b) % q
        elif oc is OpCode.MUL_MODQ:
            a = env[op.operands[0].name]
            b = env[op.operands[1].name]
            val = (a * b) % q
        else:
            raise ValueError(f"cannot execute {oc}")
        env[op.result.name] = val

    return env[program.output.name]


# --------------------------------------------------------------------------- #
# MLIR-printable text form
# --------------------------------------------------------------------------- #
def print_program(program: Program) -> str:
    def ty(v):
        return "x".join(map(str, v.shape))

    lines = [f"module {{q = {program.q}}}"]
    for v in program.inputs:
        lines.append(f"  {v.name} = input            : {ty(v)}")

    for op in program.ops:
        oc = op.opcode
        attrs = {k: v for k, v in op.attrs.items() if k not in ("shape", "name")}
        # render bool flags as MLIR-ish, drop defaults
        attrs = {k: v for k, v in attrs.items() if not (isinstance(v, bool) and v is False)}
        astr = ""
        if oc is OpCode.MATMUL_MODQ or oc is OpCode.MUL_MODQ:
            parts = []
            for k in ("transpose_a", "transpose_b"):
                if op.attrs.get(k):
                    parts.append(f"{k}=true")
            parts.append(f"q={program.q}")
            astr = " {" + ", ".join(parts) + "}"

        if oc is OpCode.CONST:
            rhs = f'const "{op.attrs["name"]}"'
        elif oc is OpCode.RESHAPE:
            rhs = f'reshape {op.operands[0].name} {{shape={list(op.attrs["shape"])}}}'
        elif oc is OpCode.TRANSPOSE:
            rhs = f"transpose {op.operands[0].name}"
        elif oc is OpCode.SLICE:
            rhs = (f"slice {op.operands[0].name} "
                   f"{{axis={op.attrs['axis']}, start={op.attrs['start']}, size={op.attrs['size']}}}")
        elif oc is OpCode.SPLIT:
            lhs = ", ".join(v.name for v in outputs(op))
            lines.append(f"  {lhs} = split {op.operands[0].name} "
                         f"{{axis={op.attrs['axis']}, parts={op.attrs['parts']}}} : {ty(outputs(op)[0])}")
            continue
        elif oc is OpCode.ALL_TO_ALL:
            lhs = ", ".join(v.name for v in outputs(op))
            lines.append(f"  {lhs} = all_to_all {', '.join(o.name for o in op.operands)} "
                         f"{{reshape_to={list(op.attrs['reshape_to'])}, parts={op.attrs['parts']}}} "
                         f": {ty(outputs(op)[0])}")
            continue
        elif oc is OpCode.CONCAT:
            rhs = f"concat {', '.join(o.name for o in op.operands)} {{axis={op.attrs['axis']}}}"
        else:
            rhs = f"{oc.value} {', '.join(o.name for o in op.operands)}{astr}"
        lines.append(f"  {op.result.name} = {rhs:<46} : {ty(op.result)}")

    lines.append(f"  return {program.output.name}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# full MLIR-style form (typed func signature + per-op operand/result types)
# --------------------------------------------------------------------------- #
def print_mlir(program: Program, func_name: str = "ntt", elem: str = "i32") -> str:
    """Render a complete MLIR-style module: a `func.func` with typed inputs and
    output, and every op carrying full operand and result tensor types.

    Uses an `ntt` dialect spelling (ntt.matmul_modq / mul_modq / transpose / reshape
    / const). `elem` is the element type (q < 2^28 fits i32, matching the u32 datapath).
    """
    def ty(shape):
        return f"tensor<{'x'.join(map(str, shape))}x{elem}>"

    shapes = {v.name: tuple(v.shape) for v in program.inputs}

    def sh(name):
        return shapes[name]

    args = ", ".join(f"{v.name}: {ty(v.shape)}" for v in program.inputs)
    out = program.output
    L = [
        "module {",
        f"  func.func @{func_name}({args}) -> {ty(sh(out.name) if out.name in shapes else out.shape)} {{",
    ]

    for op in program.ops:
        oc = op.opcode
        if oc is OpCode.SPLIT:
            outs = outputs(op)
            s = op.operands[0]
            lhs = ", ".join(v.name for v in outs)
            rty = ty(outs[0].shape)
            L.append(f"    {lhs} = ntt.split {s.name} "
                     f"{{axis = {op.attrs['axis']} : index, parts = {op.attrs['parts']} : index}} : "
                     f"{ty(sh(s.name))} -> ({', '.join(rty for _ in outs)})")
            for v in outs:
                shapes[v.name] = tuple(v.shape)
            continue
        if oc is OpCode.ALL_TO_ALL:
            outs = outputs(op)
            lhs = ", ".join(v.name for v in outs)
            ins = ", ".join(o.name for o in op.operands)
            inty = ", ".join(ty(sh(o.name)) for o in op.operands)
            outty = ", ".join(ty(v.shape) for v in outs)
            L.append(f"    {lhs} = ntt.all_to_all {ins} "
                     f"{{reshape_to = {list(op.attrs['reshape_to'])}, parts = {op.attrs['parts']} : index}} : "
                     f"({inty}) -> ({outty})")
            for v in outs:
                shapes[v.name] = tuple(v.shape)
            continue
        r = op.result
        rt = ty(r.shape)
        q = program.q
        if oc is OpCode.CONST:
            line = f'{r.name} = ntt.const {{name = "{op.attrs["name"]}"}} : {rt}'
        elif oc is OpCode.RESHAPE:
            s = op.operands[0]
            line = f"{r.name} = ntt.reshape {s.name} : {ty(sh(s.name))} to {rt}"
        elif oc is OpCode.TRANSPOSE:
            s = op.operands[0]
            line = f"{r.name} = ntt.transpose {s.name} : {ty(sh(s.name))} to {rt}"
        elif oc is OpCode.SLICE:
            s = op.operands[0]
            line = (f"{r.name} = ntt.slice {s.name} "
                    f"{{axis = {op.attrs['axis']} : index, start = {op.attrs['start']} : index, "
                    f"size = {op.attrs['size']} : index}} : {ty(sh(s.name))} to {rt}")
        elif oc is OpCode.CONCAT:
            tys = ", ".join(ty(sh(o.name)) for o in op.operands)
            line = (f"{r.name} = ntt.concat {', '.join(o.name for o in op.operands)} "
                    f"{{axis = {op.attrs['axis']} : index}} : ({tys}) -> {rt}")
        elif oc in (OpCode.MATMUL_MODQ, OpCode.MUL_MODQ):
            a, b = op.operands
            flags = [f"q = {q} : {elem}"]
            for k in ("transpose_a", "transpose_b"):
                if op.attrs.get(k):
                    flags.append(f"{k}")
            opn = "matmul_modq" if oc is OpCode.MATMUL_MODQ else "mul_modq"
            line = (f"{r.name} = ntt.{opn} {a.name}, {b.name} "
                    f"{{{', '.join(flags)}}} : "
                    f"({ty(sh(a.name))}, {ty(sh(b.name))}) -> {rt}")
        else:
            raise ValueError(oc)
        L.append("    " + line)
        shapes[r.name] = tuple(r.shape)

    L.append(f"    return {out.name} : {ty(sh(out.name))}")
    L.append("  }")
    L.append("}")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# structured export (JSON)
# --------------------------------------------------------------------------- #
def export_dict(program: Program, include_const_data: bool = True) -> dict:
    """Serializable dict of the full instruction list (opcodes, typed I/O, attrs).

    With include_const_data, the baked constant matrices are emitted as int lists so
    the export is self-contained and re-executable by any backend.
    """
    def jattrs(d):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}

    consts = {}
    for name, arr in program.consts.items():
        entry = {"shape": list(arr.shape)}
        if include_const_data:
            entry["data"] = np.asarray(arr, dtype=object).astype(int).tolist()
        consts[name] = entry

    return {
        "q": program.q,
        "meta": program.meta,
        "inputs": [{"name": v.name, "shape": list(v.shape)} for v in program.inputs],
        "ops": [
            {
                "op": o.opcode.value,
                "operands": [v.name for v in o.operands],
                "results": [{"name": v.name, "shape": list(v.shape)} for v in outputs(o)],
                "attrs": jattrs(o.attrs),
            }
            for o in program.ops
        ],
        "output": program.output.name,
        "consts": consts,
    }


def export_json(program: Program, path: str, include_const_data: bool = True) -> None:
    with open(path, "w") as f:
        json.dump(export_dict(program, include_const_data), f, indent=2)


def from_dict(d: dict) -> Program:
    """Reconstruct a Program from export_dict output (needs include_const_data=True
    to be re-executable)."""
    shapes = {v["name"]: tuple(v["shape"]) for v in d["inputs"]}
    inputs = [Value(v["name"], tuple(v["shape"])) for v in d["inputs"]]
    consts = {
        name: (np.array(e["data"], dtype=object) if "data" in e else None)
        for name, e in d["consts"].items()
    }
    ops = []
    for o in d["ops"]:
        operands = [Value(n, shapes[n]) for n in o["operands"]]
        outs = [Value(r["name"], tuple(r["shape"])) for r in o["results"]]
        for v in outs:
            shapes[v.name] = tuple(v.shape)
        res = outs if len(outs) > 1 else outs[0]
        attrs = dict(o["attrs"])
        if "shape" in attrs:
            attrs["shape"] = tuple(attrs["shape"])
        ops.append(Op(OpCode(o["op"]), operands, res, attrs))
    output = Value(d["output"], shapes[d["output"]])
    return Program(d["q"], inputs, ops, output, consts, d.get("meta", {}))


def load_json(path: str) -> Program:
    """Load and reconstruct a Program from a JSON export; ready to run()."""
    with open(path) as f:
        return from_dict(json.load(f))
