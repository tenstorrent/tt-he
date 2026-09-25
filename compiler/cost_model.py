# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""cost_model.py — v1 cost model for HE mapping / scheduling.

See docs/ir/formalization.md §3. Times in MICROSECONDS (us); a "tile" is 32x32.

Formula (v1, p = 1 — one core per op):
    op compute:   compute(op, m) = setup[op] + per_tile[op] * m        (m = # tiles)
    edge comm:    comm(tier, m)  = k[tier] * m       tier: sram(=0) | noc | dram
    makespan:     T = sum_programs( d + max_core( sum compute ) ) + sum_edges comm
                  (disjoint cores overlap -> max; same core serializes -> sum;
                   programs run back-to-back -> sum, each paying one dispatch d)

v1.1 (moldable, p cores on one op) — not used yet:
    compute(op, m, p) = setup + f*(per_tile*m) + (1-f)*(per_tile*m)/p    (f = serial frac)

IMPORTANT: every constant below is a TEMPORARY seed from a prior experiment or a doc
(tagged .source / .confidence). Measure accurately later — see MEASURE_TODO at the bottom.
Confidence: "measured" (on n150) > "doc" (from a report) > "guess" (placeholder).
"""
from __future__ import annotations
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
@dataclass
class Const:
    value: float
    source: str
    confidence: str        # "measured" | "doc" | "guess"


# per-program dispatch cost
DISPATCH = Const(1.6, "docs/he/ct_mul_perf.md: 501 dispatches ~= 0.8 ms launch tax", "doc")

# comm tiers, us/tile
K = {
    "sram": Const(0.0,  "same-core producer->consumer stays in L1 (Fusion B)", "measured"),
    "noc":  Const(0.05, "PLACEHOLDER — micro-bench one 32x32 NoC hop",         "guess"),
    "dram": Const(0.60, "PLACEHOLDER — infer from a DRAM round-trip micro-bench", "guess"),
}


# --------------------------------------------------------------------------- #
# per-op compute cost
# --------------------------------------------------------------------------- #
@dataclass
class OpCost:
    setup: float           # us, fixed
    per_tile: float        # us/tile (0 for a 'fixed' region cost)
    kind: str              # "per_tile" | "fixed"
    source: str
    confidence: str
    serial_frac: float = 1.0   # f, only used by the v1.1 moldable path

    def cost(self, tiles: int = 1) -> float:
        return self.setup + (self.per_tile * tiles if self.kind == "per_tile" else 0.0)


# op name -> cost. Primitives are per-tile; NTT-family are fixed per 1024-poly limb
# (self-contained: the 147 us already includes the internal matmul/twiddle/recombine).
OPS: dict[str, OpCost] = {
    # --- primitives, per-tile ---
    "modmul":        OpCost(0.0, 30.34, "per_tile", "coschedule_probe: 7769us/256t, modmul_compute.cpp", "measured"),
    "modmul_hybrid": OpCost(0.0, 17.8,  "per_tile", "README: modmul_sfpi_hybrid ~1.7x faster than modmul_compute", "doc"),
    "modadd":        OpCost(0.0, 3.0,   "per_tile", "PLACEHOLDER — modadd << modmul (probe hetero = 1.04x)", "guess"),
    "modsub":        OpCost(0.0, 3.0,   "per_tile", "PLACEHOLDER ~ modadd", "guess"),
    # NTT sub-stages: doc numbers are per-STAGE over a whole 1024 limb (32 tiles), so 'fixed'.
    "decompose":     OpCost(14.1, 0.0, "fixed", "NTT_1024_MULTICORE: A1 decompose 14.1us / 1024 limb", "doc"),
    "recombine":     OpCost(38.9, 0.0, "fixed", "NTT_1024_MULTICORE: A3 recombine 38.9us / 1024 limb", "doc"),
    "i8_matmul":     OpCost(14.2, 0.0, "fixed", "NTT_1024_MULTICORE: A2 matmul 14.2us / 1024 limb single-core", "doc"),
    "twiddle":       OpCost(36.3, 0.0, "fixed", "NTT_1024_MULTICORE: B twiddle 36.3us / 1024 limb", "doc"),
    # --- region-level (fixed) costs for one 1024 limb ---
    "ntt_1024":      OpCost(147.0, 0.0, "fixed", "NTT_1024_MULTICORE: 147us single-core", "doc"),
    "intt_1024":     OpCost(147.0, 0.0, "fixed", "PLACEHOLDER ~ ntt_1024", "guess"),
    "base_convert":  OpCost(350.0, 0.0, "fixed", "PLACEHOLDER — 3 FBC ~1.05ms/ct_mul, ~85% recombine", "guess"),
    "crt_combine":   OpCost(50.0,  0.0, "fixed", "PLACEHOLDER — crt_mul_split, to measure", "guess"),
    "gadget_decompose": OpCost(50.0, 0.0, "fixed", "PLACEHOLDER — gadget_decompose_signed, to measure", "guess"),
}


# --------------------------------------------------------------------------- #
# cost functions
# --------------------------------------------------------------------------- #
def op_cost(name: str, tiles: int = 1, cores: int = 1) -> float:
    """Compute time of one op. cores>1 uses the v1.1 moldable model (else p=1)."""
    o = OPS[name]
    base = o.cost(tiles)
    if cores <= 1:
        return base
    par = base * (1.0 - o.serial_frac)      # only the per-tile part is treated as parallel here
    return base * o.serial_frac + par / cores


def comm_cost(tier: str, tiles: int) -> float:
    return K[tier].value * tiles


# A schedule:
#   programs: list of programs; each program is {core_id: [(op_name, tiles), ...]}
#   edges:    list of (tier, tiles) for data movement between ops
def makespan(programs: list[dict], edges: list[tuple] | None = None) -> float:
    total = 0.0
    for prog in programs:
        core_times = [sum(op_cost(n, m) for n, m in ops) for ops in prog.values()]
        total += DISPATCH.value + (max(core_times) if core_times else 0.0)
    for tier, m in (edges or []):
        total += comm_cost(tier, m)
    return total


# --------------------------------------------------------------------------- #
# introspection
# --------------------------------------------------------------------------- #
def dump_params() -> None:
    print(f"{'DISPATCH':<16} {DISPATCH.value:>8.2f} us/prog   [{DISPATCH.confidence}] {DISPATCH.source}")
    for t, c in K.items():
        print(f"k_{t:<14} {c.value:>8.3f} us/tile  [{c.confidence}] {c.source}")
    print()
    for name, o in OPS.items():
        rate = f"{o.per_tile:.2f}/tile" if o.kind == "per_tile" else f"{o.setup:.0f} fixed"
        print(f"{name:<18} {rate:>12}   [{o.confidence}] {o.source}")


# --------------------------------------------------------------------------- #
# self-check: reproduce the coschedule_probe (device/sched) result
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== parameters (TEMP seeds) ===")
    dump_params()

    print("\n=== self-check vs coschedule_probe (n150, W=256 modmul) ===")
    W = 256
    serial = makespan([{0: [("modmul", W)]}, {1: [("modmul", W)]}])   # 2 programs
    fused  = makespan([{0: [("modmul", W)], 1: [("modmul", W)]}])      # 1 program, 2 cores
    print(f"  serial predicted {serial:8.1f} us   (measured 15532.7)")
    print(f"  fused  predicted {fused:8.1f} us   (measured  7769.4)")
    print(f"  speedup predicted {serial/fused:.2f}x   (measured 2.00x)")

    # NOTE: modmul's per_tile was calibrated FROM this probe, so it reproduces by design.
    # The REAL validation is predicting the known 12 ms ct_mul / 7.1 ms PIR (not yet calibrated).


# --------------------------------------------------------------------------- #
# MEASURE_TODO — replace guesses with n150 measurements (docs/ir/formalization.md §3/§4.2)
# --------------------------------------------------------------------------- #
# 1. DISPATCH:      micro-bench at W=1 (probe hides it under compute at W=256).
# 2. k_noc, k_dram: micro-bench one 32x32 tile transfer over NoC and via DRAM round-trip.
# 3. per-op costs:  extract modadd/modsub, base_convert, crt_combine, gadget_decompose,
#                   intt from ct_mul_perf.md / kernels.md or dedicated micro-benches.
# 4. serial_frac:   fit per kernel from ntt1024_dev_mc (matmul p=4 -> 8% => high f).
# 5. VALIDATION GATE: hand-schedule ct_mul (501 dispatches) and PIR; makespan() must
#    reproduce ~12 ms and ~7.1 ms before the scheduler is trusted.
