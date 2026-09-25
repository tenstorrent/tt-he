# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""devgen.py — device code-gen v2: pointwise ops AND matmuls, from the real compiled program.

WHY A SECOND EMITTER. `codegen.py`'s spec encodes each op as (kind in {mul,add,sub}, in0, in1, out,
core): no shape, no contraction axis, no per-op constant, one global prime. A matmul is not expressible
in it, and that -- not a missing kernel -- is the only reason the matmuls never reached the device.
`kernels/ir/compute/matmul_modq.cpp` has been there all along, device-validated by
`tests/ir/test_ir_matmul.cpp`. This emitter lifts the format limit; `codegen.py` is left alone because
`run_he.py` still depends on it.

WHAT IS NOW DISPATCHABLE. `matmul_modq` is a fused decompose + int8 matmul + recombine over ONE 32x32
tile, with the weight matrix pre-folded on the host into 16 int8 tiles. So the ops it can take are
exactly `(32,32) const @ (32,32) value -> (32,32)`. In the four-step NTT that is the `W_n1 @ Y` half --
which only EXISTS at N >= 1024, because the factors have to reach (32,32) for a DFT block to fill a
tile. At N=256 with factors (16,16) the same matmuls are 16x16 and cannot use this kernel at all.

TWO THINGS THAT MAKE IT COMPOSE

  * TILIZATION. The matmul kernel wants tilized data; the pointwise kernels do not care. Tilization is
    a fixed permutation and elementwise ops are permutation-invariant, so EVERYTHING is tilized here
    (inputs, golden, folded weights) and the two op families interoperate. Done in Python so the
    runtime needs no tilizer.
  * MONTGOMERY PASSES THROUGH A PLAIN-CONSTANT MATMUL. The runtime keeps values in the Montgomery
    domain (mul is mont(x)*mont(y)->mont, add/sub linear). A matmul with a PLAIN folded weight is
    linear in its data operand, so  A @ mont(B) = A @ (B*R) = (A@B)*R = mont(A@B)  -- the domain is
    preserved with no conversion, and the matmul chains with the pointwise ops for free.

SPEC v2 (little-endian u32 unless stated):
    magic=0xC0DE6E02, N, q, nbuf, nop, nweight
    kind[nbuf]                      0=intermediate 1=input 2=output
    nop x (opkind, in0, in1, out, core)     opkind 0=mul 1=add 2=sub 3=matmul
                                            for matmul: in0 = data buffer, in1 = WEIGHT index
    input data   (tilized, logical) for every kind==1 buffer, N u32 each
    golden       (tilized, logical) for every kind==2 buffer, N u32 each
    weights      nweight x (16 * 1024) int8, already folded and tilized
"""
from __future__ import annotations
import os
import struct

import _paths                                     # noqa: F401
import ir
from ir import Op, Program, Value, outputs

T = 32
ELEMS = T * T
NB = 4                                            # radix-128 planes
POW = (1, 128, 16384, 2097152)
MAGIC = 0xC0DE6E02

OPK = {"tile.mul_modq": 0, "rns.mul": 0,
       "tile.add_modq": 1, "rns.add": 1,
       "tile.sub_modq": 2, "rns.sub": 2}
MATMUL = ("tile.matmul_modq",)


def tilize(flat):
    """Row-major 32x32 -> tt tile order (four 16x16 faces). The permutation the kernels expect."""
    o = [0] * ELEMS
    for r in range(T):
        for c in range(T):
            o[((r // 16) * 2 + c // 16) * 256 + (r % 16) * 16 + (c % 16)] = flat[r * T + c]
    return o


def untilize(t):
    o = [0] * ELEMS
    for r in range(T):
        for c in range(T):
            o[r * T + c] = t[((r // 16) * 2 + c // 16) * 256 + (r % 16) * 16 + (c % 16)]
    return o


def fold_weight(a_flat, q):
    """A (32x32, row-major, residues mod q) -> 16 int8 tiles, (p,r) order, tilized.

    Afold[p*NB+r][i] = limb_p( (128^r * A[i]) mod q ), exactly `tests/ir/test_ir_matmul.cpp`'s fold.
    The kernel then computes sum_r 128^r * (Afold[.,r] @ limb_r(X)) and reduces mod q."""
    out = bytearray(NB * NB * ELEMS)
    for p in range(NB):
        for r in range(NB):
            lim = [((POW[r] * int(a_flat[i])) % q >> (7 * p)) & 127 for i in range(ELEMS)]
            til = tilize(lim)
            base = (p * NB + r) * ELEMS
            for i, v in enumerate(til):
                out[base + i] = v
    return bytes(out)


def _flat(v, q, n=ELEMS):
    """A value as exactly `n` logical residues, zero-padded. Padding is safe: the kernels process a
    whole tile and 0*0 = 0+0 = 0, and the golden is padded identically."""
    if v is None:
        return None
    lst = v.reshape(-1).tolist() if hasattr(v, "reshape") else list(v)
    lst = [int(x) % q for x in lst]
    if len(lst) == 1:
        lst = lst * n                             # a scalar constant broadcasts across the tile
    return (lst + [0] * n)[:n]


def build_slice(prog, q, env=None, seed=0, max_ops=4096):
    """The dispatchable sub-DAG of `prog` at prime `q`: pointwise ops + 32x32 const-LHS matmuls.

    SELF-CONTAINED BY DESIGN. Leaf operands are filled either from `env` (the real intermediate values
    the ops see in place, when the caller already has them) or with deterministic pseudorandom residues.
    The golden is then computed by interpreting THE SLICE, not the whole program -- which is what makes
    this affordable at N=1024, where `ir.run` over ~10k L2 ops of object arrays costs minutes. The op
    STRUCTURE, the weights and the primes are the compiled program's own either way; only the leaf data
    may be synthetic, and `emit` records which.

    `matmul_modq` takes ONE 32x32 tile with the weight folded on the host, so a dispatchable matmul is
    exactly (32,32) const @ (32,32) value. The base-conversion extension matmuls are (1,k)@(k,32) --
    sub-tile -- and this kernel cannot take them."""
    import random
    rng = random.Random(seed)
    # "Constant" must be TRANSITIVE. Once a four-step factor exceeds the tile (N >= 2048), tile_matmul
    # slices the weight matrix into 32x32 blocks, so the matmul's left operand is a tile.slice OF a
    # constant rather than a constant -- and a direct check misses every one of them, which is why the
    # N=2048 and N=4096 cases first reported zero dispatchable matmuls. A value is const-derived if it
    # is a constant, or is produced by a STRUCTURAL op all of whose operands are const-derived.
    STRUCT = ("tile.slice", "tile.reshape", "tile.transpose", "tile.concat", "tile.fixed_slice")
    const_of = set()
    for op in prog.ops:
        if op.opcode.endswith(".const"):
            for v in outputs(op):
                const_of.add(v.name)
        elif op.opcode in STRUCT and op.operands and all(o.name in const_of for o in op.operands):
            for v in outputs(op):
                const_of.add(v.name)

    sel = []
    for op in prog.ops:
        r = outputs(op)[0]
        if r.q != q:
            continue
        if op.opcode in OPK and len(op.operands) == 2:
            sel.append(("pw", op))
        elif op.opcode in MATMUL and len(op.operands) == 2:
            a, b = op.operands
            if (getattr(a, "shape", None) == (T, T) and getattr(b, "shape", None) == (T, T)
                    and a.name in const_of):
                sel.append(("mm", op))
        if len(sel) >= max_ops:
            break
    if not sel:
        return None

    made = {outputs(op)[0].name for _, op in sel}
    val, data, ops, in_names, weights, widx, synth = {}, {}, [], [], [], {}, 0

    def leaf(name):
        """-> the 1024 logical residues for a leaf of the slice."""
        nonlocal synth
        if name in val:
            return val[name]
        d = _flat(env.get(name), q) if env is not None else None
        if d is None:
            d = [rng.randrange(q) for _ in range(ELEMS)]
            synth += 1
        val[name] = d
        data[name] = d
        in_names.append(name)
        return d

    produced = []
    for tag, op in sel:
        rn = outputs(op)[0].name
        if tag == "pw":
            an, bn = op.operands[0].name, op.operands[1].name
            a = val[an] if an in made and an in val else leaf(an)
            b = val[bn] if bn in made and bn in val else leaf(bn)
            k = OPK[op.opcode]
            if k == 0:
                val[rn] = [(x * y) % q for x, y in zip(a, b)]
            elif k == 1:
                val[rn] = [(x + y) % q for x, y in zip(a, b)]
            else:
                val[rn] = [(x - y) % q for x, y in zip(a, b)]
            ops.append(("pw", k, an, bn, rn))
        else:
            wn, xn = op.operands[0].name, op.operands[1].name
            x = val[xn] if xn in made and xn in val else leaf(xn)
            if wn not in widx:
                wf = _flat(env.get(wn), q) if env is not None else None
                if wf is None:
                    wf = [rng.randrange(q) for _ in range(ELEMS)]
                    synth += 1
                widx[wn] = len(weights)
                weights.append((fold_weight(wf, q), wf))
            A = weights[widx[wn]][1]
            out = [0] * ELEMS
            for i in range(T):                     # C = A @ X mod q, row-major 32x32
                Ai = A[i * T:(i + 1) * T]
                for j in range(T):
                    acc = 0
                    for kk in range(T):
                        acc += Ai[kk] * x[kk * T + j]
                    out[i * T + j] = acc % q
            val[rn] = out
            ops.append(("mm", 3, xn, widx[wn], rn))
        produced.append(rn)

    if not ops:
        return None
    consumed = set()
    for tag, k, a, b, r in ops:
        consumed.add(a)
        if tag == "pw":
            consumed.add(b)
    out_names = [n for n in produced if n not in consumed]
    return dict(ops=ops, in_names=in_names, out_names=out_names, data=data, val=val,
                weights=[w for w, _ in weights], q=q, synthetic_leaves=synth,
                n_pw=sum(1 for t, *_ in ops if t == "pw"),
                n_mm=sum(1 for t, *_ in ops if t == "mm"))


def emit(sl, cores, path):
    """Write spec v2. Golden = the slice's own evaluation, tilized."""
    order, idx = [], {}

    def bid(nm):
        if nm not in idx:
            idx[nm] = len(order)
            order.append(nm)
        return idx[nm]

    for nm in sl["in_names"]:
        bid(nm)
    for tag, k, a, b, r in sl["ops"]:
        bid(a)
        if tag == "pw":
            bid(b)
        bid(r)
    inset, outset = set(sl["in_names"]), set(sl["out_names"])
    kind = [1 if nm in inset else 2 if nm in outset else 0 for nm in order]
    q, n = sl["q"], ELEMS

    with open(path, "wb") as f:
        f.write(struct.pack("<6I", MAGIC, n, q, len(order), len(sl["ops"]), len(sl["weights"])))
        f.write(struct.pack(f"<{len(kind)}I", *kind))
        for i, (tag, k, a, b, r) in enumerate(sl["ops"]):
            in1 = b if tag == "mm" else bid(b)
            f.write(struct.pack("<5I", k, bid(a), in1, bid(r), cores[i]))
        for nm, kd in zip(order, kind):
            if kd == 1:
                f.write(struct.pack(f"<{n}I", *tilize(sl["data"][nm])))
        for nm, kd in zip(order, kind):
            if kd == 2:
                f.write(struct.pack(f"<{n}I", *tilize(sl["val"][nm])))
        for w in sl["weights"]:
            f.write(w)
    return dict(buffers=len(order), ops=len(sl["ops"]), weights=len(sl["weights"]),
                bytes=os.path.getsize(path), outputs=len(outset))
