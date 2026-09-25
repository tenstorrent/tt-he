# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ir.py — IR data structures for the HE compiler: Value / Op / Program + verify + run.

The in-memory representation of a *program* (a DAG of ops). `reference.py` gives the math of
each op; this gives the container that strings them into a graph you can build, verify, run,
lower, and schedule. Adapted from reference/ntt_matmul/ir.py with the locked decisions:

  * PER-VALUE PRIME. `q` lives on the Value (its type), there is NO global program.q. Ops read
    their modulus from operands/result; base_convert/crt_combine span primes (src = operand
    primes, dst = result primes). (docs/ir/ops.md Conventions, FORMALIZATION §5.)
  * ONE op set across L1/L2/L3, names == ops.md (rns.* / tile.*).
  * run() dispatches each op through reference.py — so the interpreter is the bit-exact golden.

Everything is exact (object/bignum) via reference.py. Structural only; passes/codegen build on it.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import reference as ref


# --------------------------------------------------------------------------- #
# core types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Value:
    name: str
    shape: tuple                     # e.g. (1024,) poly, (32, 32) tile
    dtype: str = "i32"               # "i32" field element | "i8" radix-128 limb
    q: object = None                 # the prime(s) this value is under:
                                     #   layout="coeff": one int (a tile = one limb)
                                     #   layout="limb" : a tuple base (a tile's rows = these primes)
    domain: str | None = None        # "coeff" | "eval" for polys (NTT domain tag)
    layout: str = "coeff"            # PHYSICAL tile layout (transparent to the golden):
                                     #   "coeff" = vector-slice, 1024 coeffs of one limb (SFPU/NTT)
                                     #   "limb"  = limb-major, rows = limbs of a base (FPU base_convert)
                                     # coeff<->limb needs an explicit relayout (all-to-all transpose).

    def __repr__(self):
        t = "x".join(map(str, self.shape))
        lay = "#limb" if self.layout == "limb" else ""
        extra = "".join(x for x in [f"@{self.q}" if self.q else "",
                                    f":{self.domain}" if self.domain else "",
                                    f":{self.dtype}" if self.dtype != "i32" else ""])
        return f"{self.name}:{t}{extra}{lay}"


@dataclass(slots=True)
class Op:
    opcode: str                      # ops.md name, e.g. "rns.ntt", "tile.matmul_modq"
    operands: list                   # list[Value]
    result: object                   # Value, or list[Value] for multi-output ops
    attrs: dict = field(default_factory=dict)


@dataclass
class Program:
    inputs: list                     # list[Value]
    ops: list                        # list[Op] (SSA / topologically ordered)
    outputs: list                    # list[Value]
    consts: dict = field(default_factory=dict)   # name -> ndarray (baked constants)
    meta: dict = field(default_factory=dict)


def outputs(op: Op) -> list:
    return op.result if isinstance(op.result, list) else [op.result]


# --------------------------------------------------------------------------- #
# verifier (static: SSA + prime consistency)
# --------------------------------------------------------------------------- #
_ELEMWISE = {"rns.mul", "rns.add", "rns.sub", "tile.mul_modq", "tile.add_modq"}

def verify(program: Program) -> None:
    defined = {v.name for v in program.inputs}
    for op in program.ops:
        for o in op.operands:
            if o.name not in defined:
                raise ValueError(f"{op.opcode}: operand {o.name!r} used before defined")
        for v in outputs(op):
            if v.name in defined:
                raise ValueError(f"SSA violation: {v.name!r} defined twice")
            defined.add(v.name)
        # per-value-q consistency: elementwise ops reduce into one shared prime. A raw operand
        # (q=None, e.g. the CRNS overflow k — a wide value with no single modulus) is allowed and is
        # implicitly reduced into the result's prime; every *primed* value must agree, and the result
        # must be primed.
        if op.opcode in _ELEMWISE:
            primes = {o.q for o in op.operands if o.q is not None} | {v.q for v in outputs(op) if v.q is not None}
            if len(primes) != 1 or any(v.q is None for v in outputs(op)):
                raise ValueError(f"{op.opcode}: operand/result primes disagree or result unprimed: "
                                 f"{[o.q for o in op.operands]} -> {[v.q for v in outputs(op)]}")
    for v in program.outputs:
        if v.name not in defined:
            raise ValueError(f"output {v.name!r} never defined")


# --------------------------------------------------------------------------- #
# interpreter (the bit-exact golden — dispatches through reference.py)
# --------------------------------------------------------------------------- #
def run(program: Program, inputs: dict, all_values: bool = False) -> dict:
    """inputs: {value_name: array-like}. Returns {output_name: exact ndarray}.

    all_values=True returns EVERY value in the program, not just the outputs. The device
    code-generator uses it to drive a sub-DAG with the real intermediate operands it would see in
    place, rather than with fresh random data that merely has the right shape."""
    verify(program)
    env: dict = {}
    for v in program.inputs:
        env[v.name] = ref.modq(inputs[v.name], v.q) if v.q else ref.A(inputs[v.name])

    for op in program.ops:
        oc, a = op.opcode, op.attrs
        ins = [env[o.name] for o in op.operands]
        outs = outputs(op)
        rq = outs[0].q

        # ---- macro / composite op: semantics supplied as attrs["fn"] (a subgraph stand-in) ----
        if oc == "macro":
            res = a["fn"](*ins)
            if isinstance(op.result, list):
                for v, r in zip(outs, res):
                    env[v.name] = r
            else:
                env[outs[0].name] = res
            continue

        # ---- multi-output ops ----
        if oc == "tile.split":
            res = ref.split(ins[0], a["axis"], a["parts"])
        elif oc == "tile.decompose":
            res = ref.decompose(ins[0], a.get("radix", 128), a.get("nlimbs", 4))
        elif oc == "rns.gadget_decompose":
            res = ref.gadget_decompose(ins[0], a["base_B"], a["ell"], a.get("signed", True))
        elif oc == "rns.base_convert":
            src_b, dst_b = [o.q for o in op.operands], [v.q for v in outs]
            if a.get("exact", True):
                res = ref.base_convert(ins, src_b, dst_b, a.get("signed", False))
            else:
                res = ref.fast_base_convert(ins, src_b, dst_b)   # approximate FBC (drops ε overflow)
        elif oc == "tile.all_to_all_tiles":
            res = ref.all_to_all_tiles(ins, a["perm"], a.get("transpose"))
        else:
            # ---- single-output ops ----
            if oc in ("rns.const", "tile.const"):
                val = ref.const(program.consts[a["name"]], rq)
            elif oc == "rns.ntt":
                val = ref.ntt(ins[0], rq, a["psi"])
            elif oc == "rns.intt":
                val = ref.intt(ins[0], rq, a["psi"])
            elif oc in ("rns.mul", "tile.mul_modq"):
                val = ref.mul(ins[0], ins[1], rq)
            elif oc in ("rns.add", "tile.add_modq"):
                val = ref.add(*ins, q=rq)
            elif oc in ("rns.sub", "tile.sub_modq"):
                val = ref.sub(ins[0], ins[1], rq)
            elif oc == "rns.automorph":
                val = ref.automorph(ins[0], a["k"], rq, a["N"])
            elif oc == "rns.crt_combine":
                val = ref.crt_combine(ins, [o.q for o in op.operands])
            elif oc == "tile.matmul_modq":
                val = ref.matmul_modq(ins[0], ins[1], rq, a.get("transpose_a", False), a.get("transpose_b", False))
            elif oc == "tile.matmul_wide":
                val = ref.matmul_wide(ins[0], ins[1], a.get("transpose_a", False), a.get("transpose_b", False))
            elif oc == "tile.fixed_slice":
                val = ref.fixed_slice(ins[0], a["shift"], a.get("mask"))
            elif oc == "tile.transpose":
                val = ref.transpose(ins[0])
            elif oc == "tile.reshape":
                val = ref.reshape(ins[0], a["shape"])
            elif oc == "tile.slice":
                val = ref.slice_(ins[0], a["axis"], a["start"], a["size"])
            elif oc == "tile.concat":
                val = ref.concat(ins, a["axis"])
            elif oc == "tile.recombine_modq":
                val = ref.recombine_modq(ins, rq, a.get("radix", 128))
            elif oc == "tile.recombine_wide":
                val = ref.recombine_wide(ins, a.get("radix", 128), a.get("exps"))
            else:
                raise ValueError(f"interpreter: unknown opcode {oc!r}")
            env[outs[0].name] = val
            continue

        if len(res) != len(outs):
            raise ValueError(f"{oc}: produced {len(res)} results, op declares {len(outs)}")
        for v, r in zip(outs, res):
            env[v.name] = r

    if all_values:
        return dict(env)
    return {v.name: env[v.name] for v in program.outputs}


# --------------------------------------------------------------------------- #
# pretty printer
# --------------------------------------------------------------------------- #
def print_program(program: Program) -> str:
    lines = ["program {"]
    lines += [f"  in  {v!r}" for v in program.inputs]
    for op in program.ops:
        res = ", ".join(repr(v) for v in outputs(op))
        ops_ = ", ".join(o.name for o in op.operands)
        at = "".join(f" {{{k}={v}}}" for k, v in op.attrs.items() if k != "psi")
        lines.append(f"  {res} = {op.opcode} {ops_}{at}")
    lines += [f"  out {v.name}" for v in program.outputs]
    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# self-test: build small Programs, run them, check vs reference directly
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import random
    rng = random.Random(3)
    ok = 0
    def check(name, cond):
        global ok
        assert cond, f"FAIL: {name}"
        print(f"  ok  {name}"); ok += 1

    q = 17

    # ---- he.add: two ciphertexts (c0,c1)+(d0,d1) per limb ----
    N = 8
    ins = {n: [rng.randrange(q) for _ in range(N)] for n in ("a0", "a1", "b0", "b1")}
    def V(n): return Value(n, (N,), q=q, domain="coeff")
    o0 = Value("o0", (N,), q=q, domain="coeff")
    o1 = Value("o1", (N,), q=q, domain="coeff")
    padd = Program(
        inputs=[V("a0"), V("a1"), V("b0"), V("b1")],
        ops=[Op("rns.add", [V("a0"), V("b0")], o0),
             Op("rns.add", [V("a1"), V("b1")], o1)],
        outputs=[o0, o1])
    r = run(padd, ins)
    check("he.add program",
          ref._eq(r["o0"], ref.add(ins["a0"], ins["b0"], q=q)) and
          ref._eq(r["o1"], ref.add(ins["a1"], ins["b1"], q=q)))

    # ---- NTT round-trip as a program: intt(ntt(x)) == x ----
    psi = ref.find_2nth_root(q, N)
    x = [rng.randrange(q) for _ in range(N)]
    xv = Value("x", (N,), q=q, domain="coeff")
    Xv = Value("X", (N,), q=q, domain="eval")
    yv = Value("y", (N,), q=q, domain="coeff")
    pntt = Program(
        inputs=[xv],
        ops=[Op("rns.ntt", [xv], Xv, {"psi": psi, "N": N}),
             Op("rns.intt", [Xv], yv, {"psi": psi, "N": N})],
        outputs=[yv])
    check("ntt round-trip program", ref._eq(run(pntt, {"x": x})["y"], ref.modq(x, q)))

    # ---- tile matmul_modq program ----
    Q = 251658241
    A_ = [[rng.randrange(Q) for _ in range(4)] for _ in range(4)]
    B_ = [[rng.randrange(Q) for _ in range(4)] for _ in range(4)]
    av = Value("A", (4, 4), q=Q); bv = Value("B", (4, 4), q=Q); cv = Value("C", (4, 4), q=Q)
    pmm = Program(inputs=[av, bv], ops=[Op("tile.matmul_modq", [av, bv], cv, {"q": Q})], outputs=[cv])
    check("matmul_modq program", ref._eq(run(pmm, {"A": A_, "B": B_})["C"], ref.matmul_modq(A_, B_, Q)))

    # ---- multi-output: decompose then recombine (L3 round-trip) as a program ----
    xs = [[rng.randrange(Q) for _ in range(4)] for _ in range(4)]
    xv2 = Value("x2", (4, 4), q=Q)
    limbs = [Value(f"l{p}", (4, 4), dtype="i8") for p in range(4)]
    rv = Value("r", (4, 4), q=Q)
    pdr = Program(inputs=[xv2],
                  ops=[Op("tile.decompose", [xv2], limbs, {"radix": 128, "nlimbs": 4}),
                       Op("tile.recombine_modq", limbs, rv, {"radix": 128, "q": Q})],
                  outputs=[rv])
    check("decompose+recombine program", ref._eq(run(pdr, {"x2": xs})["r"], ref.modq(xs, Q)))

    print("\n--- example program (he.add) ---")
    print(print_program(padd))
    print(f"\nALL {ok} IR CHECKS PASSED — Program build / verify / run over reference.py works.")
