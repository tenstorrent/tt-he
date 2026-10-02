# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""hw.py — Blackhole p150 hardware model with an ENGINE dimension. Supersedes cost_model.py's seeds.

`cost_model.py` was seeded on an **n150** against the *old* kernels: `modmul` at 30.34 us/tile, most
entries tagged "guess"/"doc", and -- the reason a new module was needed -- **no notion of which engine
an op runs on**. Blackhole has two compute engines with rooflines 20x apart on the multiply
(matrix-engine matmul vs SFPU montmul at 46.8 Gop/s), so a model that cannot tell them apart cannot
express the single largest decision the mapper makes. `cost_model.py` is left untouched for the modules
that already import it.

EVERY CONSTANT HERE IS MEASURED ON THIS CARD, and carries its source. Three families:

  * `docs/vector/exp_roofline001.md` -- per-element throughput for add / mul / montmul, in BOTH regimes
    (streaming R=1 and full-reuse arithmetic bound). This is what prices arithmetic.
  * `docs/he/exp_tg018ntt.md` §4d/§4e -- the per-kernel census of the real CKKS ct_mul at N=2^16,
    L_QP=64, dnum=3, 130 cores, with the SFPU/matrix split. This is what prices the workload and what
    the LADDER GATE below checks against.
  * `docs/he/dram_guide.md`, `docs/he/exp_dram001.md` -- 436 GB/s chip DRAM (both NoCs), a ~238 GB/s
    one-NoC wall, and the 4.29x on-chip-vs-DRAM corner-turn ratio. This is what prices movement.

THE LADDER GATE (`validate_ladder`) is the point of the file. The hardware work produced a measured
four-rung ladder -- 50.863 -> 43.932 -> 32.448 -> 23.991 ms -- where each rung is exactly one of the
four optimisation passes. A cost model that cannot retrodict that ladder has no business predicting
anything, so the model is required to reproduce it, and the gate reports which rungs are PREDICTED from
independent measurements and which are CALIBRATED (fitted to that rung). Two of four are predicted.

Run:  python3 hw.py
"""
from __future__ import annotations
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# the card
# --------------------------------------------------------------------------- #
CORES = 130                 # usable Tensix on the p150 these measurements ran on
CLOCK_MHZ = 1350            # sustained AICLK under load (tt-smi)
TILE = 32                   # matrix-engine tile edge
TILE_ELEMS = TILE * TILE    # 1024
TILE_BYTES = TILE_ELEMS * 4 # int32 tile

SFPU, FPU, DMA = "sfpu", "fpu", "dma"      # engine tags: vector, matrix, dataflow


@dataclass
class C:
    """A constant with provenance. `conf`: measured > doc > calibrated > guess."""
    value: float
    source: str
    conf: str = "measured"

    def __float__(self):
        return float(self.value)


# --------------------------------------------------------------------------- #
# 1. ARITHMETIC -- from exp_roofline001, the only per-element measurement we have
# --------------------------------------------------------------------------- #
# Chip-wide Gop/s at both ends of the roofline. `stream` = R=1, every operand from DRAM (where real
# kernels sit); `reuse` = the compute ceiling with operands resident (what fusion buys you).
ROOF = {                                     # op -> (stream Gop/s, arithmetic-bound Gop/s)
    "addp":   (C(27.99, "exp_roofline001 addp R=1"),   C(1097.57, "exp_roofline001 addp ceiling")),
    "addm":   (C(27.88, "exp_roofline001 add R=1"),    C(530.04,  "exp_roofline001 add ceiling")),
    "mul":    (C(27.59, "exp_roofline001 mul R=1"),    C(158.85,  "exp_roofline001 mul ceiling")),
    "modmul": (C(24.09, "exp_roofline001 modmul R=1"), C(46.83,   "exp_roofline001 modmul ceiling")),
}
MEM_ROOF_GBPS = C(335.9, "exp_roofline001: 2 reads + 1 write, reads on one NoC")
DRAM_GBPS = C(436.0, "dram_guide 3.1 / exp_dram001: both NoCs reading")
NOC1_GBPS = C(238.0, "exp_dram001 2: one-NoC wall, not tunable (3 kernels agree)")


def us_per_tile(op: str, reuse: bool = False) -> float:
    """Per-CORE microseconds for one 32x32 tile of `op`. The roofline figures are chip-wide, so a
    single core gets 1/CORES of them -- which is the number a per-core scheduler needs."""
    gops = float(ROOF[op][1 if reuse else 0]) / CORES
    return TILE_ELEMS / (gops * 1e3)


# --------------------------------------------------------------------------- #
# 2. THE MATRIX ENGINE -- when it is legal, and how much it wins
# --------------------------------------------------------------------------- #
# exp_tg018r measured the matrix-engine base conversion against the SFPU one over the whole
# (Ls, Ld) grid in {8,16,32}^2 and it won in every cell: 1.19x at 8->8, 10.3x at 32->32. Those two
# anchors fix a power law, which is then CHECKED against a third, independent point: the real
# pipeline's (Ls=16, Ld=24) FBC, measured at 4.99x in exp_tg018ntt §4e.
FPU_ANCHOR_LO = C(1.19, "exp_tg018r grid: Ls=8 -> Ld=8")
FPU_ANCHOR_HI = C(10.3, "exp_tg018r grid: Ls=32 -> Ld=32")
_K = 0.778                  # log(10.3/1.19) / log(1024/64)


def fpu_speedup(ls: int, ld: int) -> float:
    """Matrix-engine speedup over the SFPU for a contraction of `ls` sources into `ld` destinations.

    Power law through the two exp_tg018r anchors. It is only meaningful for a contraction: the win
    comes from the matrix engine doing ls*ld multiply-accumulates in the time the SFPU does a few, so
    it scales with the PRODUCT, not with either axis."""
    work = max(1, ls * ld) / 64.0
    return max(1.0, float(FPU_ANCHOR_LO) * work ** _K)


# The matrix engine contracts WITHIN a tile, and the contracted axis is the source limb -- so one tile
# holds at most TILE destinations. exp_tg018ntt §4e found Ld=32 returns a wrong answer through that
# port's readers while Ld=24 is bit-exact, so the legal cap is the verified one, not the algorithmic one.
FPU_LD_CAP = C(24, "exp_tg018ntt 4e: Ld=32 wrong through reader_fbc_tr_pg; 24 verified", "measured")
FPU_LS_CAP = C(32, "matrix engine contracts within a tile", "doc")


def fpu_legal(opcode: str, ls: int = 1, ld: int = 1) -> bool:
    """Can this op run on the matrix engine at all? Only contractions, and only inside the tile."""
    if opcode not in ("tile.matmul_modq", "tile.matmul_wide", "tile.base_extend"):
        return False
    return ls <= float(FPU_LS_CAP) and ld <= float(FPU_LD_CAP)


# --------------------------------------------------------------------------- #
# 3. MOVEMENT -- per-core us per tile
# --------------------------------------------------------------------------- #
# One core's share of chip DRAM bandwidth.
K_DRAM = C(TILE_BYTES / ((float(DRAM_GBPS) / CORES) * 1e3), "436 GB/s / 130 cores, 4 KB tile")
# Core-to-core. Priced from the ONE controlled A/B we have for exactly this choice: exp_tg018ntt's
# corner turn is the same all-to-all done via DRAM (9.108 ms) and core-to-core (2.122 ms).
A2A_RATIO = C(9.108 / 2.122, "exp_tg018ntt 2: corner turn DRAM 9.108 ms vs on-chip 2.122 ms")
K_NOC = C(float(K_DRAM) / float(A2A_RATIO), "K_DRAM / the measured on-chip:DRAM ratio")
K_SRAM = C(0.0, "same-core producer->consumer never leaves L1")

TIERS = {"sram": K_SRAM, "noc": K_NOC, "dram": K_DRAM}

# Device-side cost of one more dispatch, backed out of the measured pair: host device busy 23.991 ms
# against a 22.917 ms critical path over 1278 dispatches.
DISPATCH_US = C((23.991 - 22.917) * 1e3 / 1278, "exp_tg018ntt 4e: (23.991 - 22.917) ms / 1278")
# What one ELIMINATED dispatch is actually worth when a chain is fused -- dispatch tax plus the barrier,
# the CB refill and the L1 reload that go with it. Lumped, because that is how it was measured:
# the fused-NTT rung removed 1596 dispatches for 11.484 ms. CALIBRATED, not predicted.
FUSE_SAVING_US = C((43.932 - 32.448) * 1e3 / (2884 - 1288),
                   "exp_tg018ntt 4c: 11.484 ms / 1596 dispatches removed", "calibrated")
HOST_DISPATCH_US = C((42.032 - 23.991) * 1e3 / 1278,
                     "exp_tg018ntt 4e: (host wall - device busy) / dispatches")


# --------------------------------------------------------------------------- #
# 4. THE WORKLOAD -- the measured census of the real ct_mul, both engine arms
# --------------------------------------------------------------------------- #
@dataclass
class Kern:
    disp: int                # dispatches in one ct_mul
    ms: float                # measured critical-path contribution
    engine: str
    note: str = ""


# exp_tg018ntt 4d, the SFPU arm: 1288 dispatches, 31.443 ms critical path.
CENSUS_SFPU = {
    "ntt64_fused":   Kern(76,  11.065, "mixed", "62.7% SFPU / 5.9% matmul / 31.4% a2a+sync"),
    "fbc_combine":   Kern(15,  10.666, SFPU,    "the base conversion, pure SFPU"),
    "modmul_fused3": Kern(480, 3.422,  SFPU),
    "modmul_scalar": Kern(142, 2.334,  SFPU),
    "modadd_map":    Kern(328, 1.610,  SFPU),
    "modsub_reduce": Kern(142, 0.926,  SFPU),
    "fbc_scale":     Kern(5,   0.571,  SFPU),
    "modadd_scalar": Kern(96,  0.452,  SFPU),
    "bconv_one":     Kern(4,   0.397,  SFPU),
}
# exp_tg018ntt 4e, the matrix-engine arm: 1278 dispatches, 22.917 ms. Only the FBC row changes;
# the other eight reproduce to +-0.9%, which is what makes it a controlled A/B.
CENSUS_FPU = dict(CENSUS_SFPU)
CENSUS_FPU.pop("fbc_combine")
CENSUS_FPU["fbc_fpu_tr"] = Kern(5, 2.14, "mixed", "residual vs the 22.917 ms critical path, not a sample")

# The measured ladder. Each rung is exactly one pass.
LADDER = [
    ("baseline (exp_tg018m)",       50.863, None),
    ("on-chip all-to-all",          43.932, "place_comm"),
    ("fused NTT (one dispatch)",    32.448, "fuse_dispatches"),
    ("base conversion on the FPU",  23.991, "select_engines"),
]
CT_MUL_SHAPE = dict(N=2 ** 16, LQ=48, LP=16, dnum=3, alpha=16, cores=130, bits=28)


def census_ms(arm: str = "fpu") -> float:
    c = CENSUS_FPU if arm == "fpu" else CENSUS_SFPU
    return sum(k.ms for k in c.values())


def census_dispatches(arm: str = "fpu") -> int:
    c = CENSUS_FPU if arm == "fpu" else CENSUS_SFPU
    return sum(k.disp for k in c.values())


def sfpu_share(arm: str = "fpu") -> tuple:
    """(low, high) bound on the SFPU's share. `ntt64_fused` splits 62.7/5.9/31.4 by phase zones;
    `fbc_fpu_tr` was never zone-split, so on the FPU arm the share is a RANGE, not a number."""
    c = CENSUS_FPU if arm == "fpu" else CENSUS_SFPU
    tot = sum(k.ms for k in c.values())
    lo = hi = 0.0
    for name, k in c.items():
        if k.engine == SFPU:
            lo += k.ms; hi += k.ms
        elif name == "ntt64_fused":
            lo += 0.627 * k.ms; hi += 0.627 * k.ms
        else:                                   # fbc_fpu_tr: unattributed
            hi += k.ms
    return lo / tot, hi / tot


# --------------------------------------------------------------------------- #
# 5. THE LADDER GATE
# --------------------------------------------------------------------------- #
def validate_ladder(verbose: bool = True) -> dict:
    """Retrodict the measured ladder from the constants above. Reports per-rung error and says which
    rungs are PREDICTED from independent measurements vs CALIBRATED against that same rung."""
    rows = []
    cur = LADDER[0][1]

    # rung 1 -> 2: the corner turn moves from DRAM to core-to-core. PREDICTED: the delta is the
    # measured stage cost times (1 - 1/ratio), and the ratio came from the stage, not from the ladder.
    d = 9.108 * (1.0 - 1.0 / float(A2A_RATIO))
    rows.append(("on-chip all-to-all", cur, cur - d, LADDER[1][1], "PREDICTED (corner-turn ratio)"))
    cur = cur - d

    # rung 2 -> 3: fusion removes 1596 dispatches. CALIBRATED -- FUSE_SAVING_US was fitted here.
    d = (2884 - 1288) * float(FUSE_SAVING_US) / 1e3
    rows.append(("fused NTT", cur, cur - d, LADDER[2][1], "CALIBRATED (fitted to this rung)"))
    cur = cur - d

    # rung 3 -> 4: the FBC moves to the matrix engine. PREDICTED: fpu_speedup at the real (Ls, Ld)
    # comes from exp_tg018r's grid, an entirely different bench.
    sp = fpu_speedup(16, 24)
    fbc = CENSUS_SFPU["fbc_combine"].ms
    d = fbc - fbc / sp
    rows.append(("FBC on the FPU", cur, cur - d, LADDER[3][1],
                 f"PREDICTED (fpu_speedup(16,24)={sp:.2f}x vs 4.99x measured)"))
    cur = cur - d

    if verbose:
        print(f"{'rung':<24} {'from':>8} {'model':>9} {'measured':>9} {'err':>7}   provenance")
        print("-" * 104)
        for name, f, mdl, meas, prov in rows:
            err = 100 * (mdl - meas) / meas
            print(f"{name:<24} {f:>8.3f} {mdl:>9.3f} {meas:>9.3f} {err:>6.1f}%   {prov}")
        print("-" * 104)
        worst = max(abs(100 * (m - x) / x) for _, _, m, x, _ in rows)
        print(f"worst rung error {worst:.1f}%   ·   end of ladder: model {cur:.3f} ms "
              f"vs measured {LADDER[-1][1]:.3f} ms  ({100*(cur-LADDER[-1][1])/LADDER[-1][1]:+.1f}%)")
    return dict(rows=rows, final=cur, measured=LADDER[-1][1])


def dump() -> None:
    print(f"=== Blackhole p150 ===  {CORES} cores @ {CLOCK_MHZ} MHz, {TILE}x{TILE} tiles\n")
    print("ARITHMETIC (per-core us per 1024-element tile, from exp_roofline001)")
    print(f"  {'op':<10} {'stream (R=1)':>14} {'full reuse':>12} {'reuse win':>11}")
    for op in ROOF:
        a, b = us_per_tile(op), us_per_tile(op, reuse=True)
        print(f"  {op:<10} {a:>14.3f} {b:>12.3f} {a/b:>10.1f}x")
    print(f"\nMATRIX ENGINE  legal for contractions, Ls<={float(FPU_LS_CAP):.0f} Ld<={float(FPU_LD_CAP):.0f}")
    for ls, ld in ((8, 8), (16, 8), (16, 24), (32, 24)):
        print(f"  fpu_speedup({ls:>2}, {ld:>2}) = {fpu_speedup(ls, ld):>5.2f}x")
    print(f"\nMOVEMENT (per-core us per tile)")
    for t, k in TIERS.items():
        print(f"  {t:<6} {float(k):>8.3f}   [{k.conf}] {k.source}")
    print(f"\nDISPATCH")
    for nm, k in (("device", DISPATCH_US), ("host gap", HOST_DISPATCH_US), ("fusion saving", FUSE_SAVING_US)):
        print(f"  {nm:<14} {float(k):>8.3f} us   [{k.conf}] {k.source}")
    lo, hi = sfpu_share("fpu")
    lo0, hi0 = sfpu_share("sfpu")
    print(f"\nWORKLOAD  ct_mul N=2^16 L_Q=48 P=16 dnum=3")
    print(f"  SFPU arm  {census_ms('sfpu'):.3f} ms, {census_dispatches('sfpu')} dispatches, "
          f"SFPU share {100*lo0:.1f}%")
    print(f"  FPU  arm  {census_ms('fpu'):.3f} ms, {census_dispatches('fpu')} dispatches, "
          f"SFPU share {100*lo:.1f}-{100*hi:.1f}%")


if __name__ == "__main__":
    dump()
    print("\n=== LADDER GATE: retrodict the measured hardware ladder ===")
    r = validate_ladder()
    worst = max(abs(100 * (m - x) / x) for _, _, m, x, _ in r["rows"])
    print("\nHW MODEL OK" if worst < 10 else "\nHW MODEL FAIL (a rung is off by >10%)")
