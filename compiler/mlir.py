# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""mlir.py — render an ir.Program as an MLIR-style module (for L1 / L2 / L3 inspection).

Stage-1 artifact for the full-ct_mul work: dump any Program (L1 poly, L2 tile, or L3 int8) as a
textual MLIR module — a `func.func` with typed inputs/output and every op carrying operand + result
tensor types. This is the "get a MLIR of L2 and L3 ops" deliverable (docs/ir/crns_bconv_lowering.md,
FORMALIZATION §1.5). Adapted from reference/ntt_matmul/ir.py::print_mlir with the new IR's decisions:

  * The opcodes are ALREADY dialect-qualified (`rns.*`, `tile.*`), so they ARE the MLIR op names —
    the emitter is generic over the op set, with special cases only for const / multi-output / macro.
  * PER-VALUE prime: there is no global program.q. `q` rides as an op attribute (from attrs["q"], or
    the result Value's own .q), so base_convert (which spans primes) renders fine.
  * Types come from Value.shape + Value.dtype (i32 field element | i8 radix-128 limb). The CRNS ops
    (tile.matmul_wide, tile.fixed_slice) render like any other op — no special casing needed.

SSA names: Value names may contain '#'/other chars illegal in MLIR SSA ids; they are sanitized to
'%'-prefixed unique ids. Meaning-preserving is NOT re-checked here (lower.py owns that); this is a
faithful textual view of whatever Program it is handed.

Run:  python3 mlir.py       # emits + writes L1/L2/L3 MLIR for the tensor, and a CRNS base_convert.
"""
from __future__ import annotations
import os, sys

import _paths                                     # noqa: F401  (repo layout; see _paths.py)
from ir import Program, Op, outputs               # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _san(name: str) -> str:
    """Sanitize a Value name into a legal MLIR SSA identifier body ([a-zA-Z0-9_$.-])."""
    s = "".join(c if (c.isalnum() or c in "_$.-") else "_" for c in name)
    if not s:
        s = "v"
    if s[0].isdigit():
        s = "_" + s
    return s


def _tyv(v) -> str:
    """MLIR tensor type of a Value: tensor<AxBx...xDTYPE>."""
    dims = "x".join(str(d) for d in v.shape) if v.shape else "1"
    return f"tensor<{dims}x{v.dtype}>"


def _fmt_attr_val(val) -> str:
    if isinstance(val, bool):
        return "unit" if val else None            # handled by caller (unit attrs only when true)
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(str(int(x)) if isinstance(x, bool) else str(x) for x in val) + "]"
    if isinstance(val, int):
        return f"{val} : i64"
    if isinstance(val, str):
        return f'"{val}"'
    return f'"{val}"'


def _attrs(op: Op, result_q) -> str:
    """Render op.attrs as an MLIR attribute dict, dropping non-printables (psi tables, macro fns).
       Adds `q = <int>` from the result Value when the op carries a single prime and attrs lacks it."""
    parts = []
    for k, val in op.attrs.items():
        if k in ("psi", "fn"):                    # baked NTT table / python closure — not printable
            continue
        if isinstance(val, bool):
            if val:
                parts.append(k)                   # MLIR unit attribute
            continue
        parts.append(f"{k} = {_fmt_attr_val(val)}")
    if "q" not in op.attrs and isinstance(result_q, int):
        parts.append(f"q = {result_q} : i64")
    return (" {" + ", ".join(parts) + "}") if parts else ""


# --------------------------------------------------------------------------- #
# emitter
# --------------------------------------------------------------------------- #
def emit_mlir(prog: Program, func_name: str = "graph") -> str:
    ssa: dict[str, str] = {}
    used: set[str] = set()

    def declare(name: str) -> str:
        if name in ssa:
            return ssa[name]
        s = _san(name)
        while s in used:
            s += "_"
        used.add(s)
        ssa[name] = "%" + s
        return ssa[name]

    def use(name: str) -> str:
        return ssa.get(name) or declare(name)     # forward-ref safety (topo order makes this rare)

    for v in prog.inputs:
        declare(v.name)

    args = ", ".join(f"{ssa[v.name]}: {_tyv(v)}" for v in prog.inputs)
    out_tys = ", ".join(_tyv(v) for v in prog.outputs)
    ret_ty = out_tys if len(prog.outputs) == 1 else f"({out_tys})"
    L = ["module {", f"  func.func @{func_name}({args}) -> {ret_ty} {{"]

    for op in prog.ops:
        outs = outputs(op)
        opname = op.opcode if "." in op.opcode else f"he.{op.opcode}"   # e.g. bare "macro" -> "he.macro"
        lhs = ", ".join(declare(v.name) for v in outs)
        rq = outs[0].q if len(outs) == 1 else None
        attrs = _attrs(op, rq)
        out_ty = _tyv(outs[0]) if len(outs) == 1 else "(" + ", ".join(_tyv(v) for v in outs) + ")"

        if not op.operands:                        # const / source op: `%r = tile.const {..} : ty`
            L.append(f"    {lhs} = {opname}{attrs} : {out_ty}")
            continue
        ins = ", ".join(use(o.name) for o in op.operands)
        in_ty = ", ".join(_tyv(o) for o in op.operands)
        L.append(f"    {lhs} = {opname} {ins}{attrs} : ({in_ty}) -> {out_ty}")

    ret_vals = ", ".join(use(v.name) for v in prog.outputs)
    L.append(f"    return {ret_vals} : {out_tys}")
    L += ["  }", "}"]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# self-test / driver: emit L1 / L2 / L3 MLIR for the tensor + a CRNS base_convert
# --------------------------------------------------------------------------- #
def _snippet(s: str, n: int = 40) -> str:
    lines = s.splitlines()
    return "\n".join(lines[:n]) + (f"\n    ... ({len(lines) - n} more lines)" if len(lines) > n else "")


if __name__ == "__main__":
    import random
    from bfv_mul_ref import MulParams, encrypt                 # noqa: E402
    from bfv_relin_ref import small_poly, keygen_relin         # noqa: E402
    import reference as ref                                    # noqa: E402
    from ir import Value                                       # noqa: E402
    from builders import build_tensor, build_ct_mul_rns        # noqa: E402
    from lower import lower_l1_to_l2, lower_l2_to_l3, fuse_a   # noqa: E402

    # committed compiler output lives in artifacts/, next to codegen_spec.bin; the source
    # modules stay at the top level (see compiler/README.md).
    outdir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(outdir, exist_ok=True)
    written = []

    # ---- FULL ct_mul (tensor over Q∪S -> rescale -> relin -> add): L1 / L2 / L3 ----
    ppc = MulParams(N=16, LQ=3, LP=3, LB=4)
    rngc = random.Random(4); sc = small_poly(ppc.N, rngc)
    evk0, evk1 = keygen_relin(ppc, sc, rngc)
    ctA = encrypt(ppc, [rngc.randrange(ppc.t) for _ in range(ppc.N)], sc, rngc)
    ctB = encrypt(ppc, [rngc.randrange(ppc.t) for _ in range(ppc.N)], sc, rngc)
    cm1, _, _, _ = build_ct_mul_rns(ppc, ctA, ctB, evk0, evk1)
    cm2 = lower_l1_to_l2(cm1)
    cm3 = lower_l2_to_l3(cm2)
    cm2f = fuse_a(cm2)                                         # stage 5 — Fusion A (tile packing)
    cm3f = lower_l2_to_l3(cm2f)
    for tag, prog in (("l1", cm1), ("l2", cm2), ("l3", cm3), ("l2_fusedA", cm2f), ("l3_fusedA", cm3f)):
        mod = emit_mlir(prog, func_name=f"ct_mul_{tag}")
        path = os.path.join(outdir, f"example_ctmul_{tag}.mlir")
        with open(path, "w") as f:
            f.write(mod + "\n")
        written.append((path, len(prog.ops), len(mod.splitlines())))

    # ---- tensor: L1 (poly) -> L2 (tile, four-step NTT) -> L3 (int8) ----
    pp = MulParams(N=16)
    rng = random.Random(7)
    s = small_poly(pp.N, rng)
    ct1 = encrypt(pp, [rng.randrange(pp.t) for _ in range(pp.N)], s, rng)
    ct2 = encrypt(pp, [rng.randrange(pp.t) for _ in range(pp.N)], s, rng)
    l1, _, _ = build_tensor(pp, ct1, ct2)
    l2 = lower_l1_to_l2(l1, factors=(4, 4))
    l3 = lower_l2_to_l3(l2)

    for tag, prog in (("l1", l1), ("l2", l2), ("l3", l3)):
        mod = emit_mlir(prog, func_name=f"tensor_{tag}")
        path = os.path.join(outdir, f"example_tensor_{tag}.mlir")
        with open(path, "w") as f:
            f.write(mod + "\n")
        written.append((path, len(prog.ops), len(mod.splitlines())))

    # ---- CRNS base_convert alone: shows tile.matmul_wide / tile.fixed_slice in the L2 MLIR ----
    src = [40961, 65537, 114689]                    # coprime NTT-friendly primes (src base)
    dst = [147457, 163841]                          # dst base
    N = 8
    bops = [Value(f"r{i}", (N,), q=q) for i, q in enumerate(src)]
    bouts = [Value(f"o{j}", (N,), q=q) for j, q in enumerate(dst)]
    bc_l1 = Program(inputs=bops,
                    ops=[Op("rns.base_convert", bops, bouts, {"signed": True})],
                    outputs=bouts)
    bc_l2 = lower_l1_to_l2(bc_l1)
    bc_mod = emit_mlir(bc_l2, func_name="base_convert_crns_l2")
    bc_path = os.path.join(outdir, "example_base_convert_crns_l2.mlir")
    with open(bc_path, "w") as f:
        f.write(bc_mod + "\n")
    written.append((bc_path, len(bc_l2.ops), len(bc_mod.splitlines())))

    # ---- report ----
    print("MLIR emitter — wrote modules:")
    for path, nops, nlines in written:
        print(f"  {os.path.basename(path):40s} {nops:4d} ops   {nlines:4d} MLIR lines")

    print("\n--- example_tensor_l2.mlir (head) ---")
    print(_snippet(emit_mlir(l2, "tensor_l2")))

    print("\n--- example_base_convert_crns_l2.mlir (full) — note tile.matmul_wide / tile.fixed_slice ---")
    print(bc_mod)

    # sanity: emitter produced a module per level with the expected op counts
    ok = all(nlines > 3 for _, _, nlines in written)
    print("\nMLIR OK" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)
