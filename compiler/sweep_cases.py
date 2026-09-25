# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""sweep_cases.py — time candidate demo configs and see which passes fire. Writes cases.json."""
from __future__ import annotations
import json, random, sys, time
import _paths                                     # noqa: F401
import hw, passes, ir
from ckks_mul_ref import CKKSParams, encrypt
from bfv_relin_dnum import keygen_relin_dnum, q_groups
from bfv_relin_ref import small_poly
from builders import build_ct_mul_ckks
from lower import lower_l1_to_l2, lower_l2_to_l3


def build_case(N, LQ, LP, dnum, factors, T=32, validate=True, do_l3=False, ndrop=1, seed=0):
    t0 = time.time()
    pp = CKKSParams(N=N, LQ=LQ, LP=LP, ndrop=ndrop)
    rng = random.Random(seed)
    if validate:
        s = small_poly(N, rng)
        ks = keygen_relin_dnum(pp, s, rng, dnum)
        m1 = [rng.randrange(-3, 4) for _ in range(N)]
        m2 = [rng.randrange(-3, 4) for _ in range(N)]
        ct1, ct2 = encrypt(pp, m1, s, rng), encrypt(pp, m2, s, rng)
    else:                                          # structural: identical GRAPH, no O(N^2) crypto
        s = None
        z = [0] * N
        groups = q_groups(LQ, dnum)
        evks = [([list(z) for _ in pp.QP], [list(z) for _ in pp.QP]) for _ in groups]
        ks = (evks, groups)
        ct1 = ct2 = (list(z), list(z))
        m1 = m2 = z
    t_key = time.time() - t0
    t0 = time.time()
    l1, inp, o0, o1, base = build_ct_mul_ckks(pp, ct1, ct2, ks, ndrop=ndrop)
    t_l1 = time.time() - t0
    t0 = time.time()
    l2 = lower_l1_to_l2(l1, factors=factors, T=T)
    t_l2 = time.time() - t0
    l3 = None
    t_l3 = 0.0
    if do_l3:
        t0 = time.time(); l3 = lower_l2_to_l3(l2); t_l3 = time.time() - t0
    return dict(pp=pp, l1=l1, l2=l2, l3=l3, inp=inp, o0=o0, o1=o1, base=base, s=s,
                m1=m1, m2=m2, t_key=t_key, t_l1=t_l1, t_l2=t_l2, t_l3=t_l3)


CANDIDATES = [
    # name,                     N,    LQ, LP, dnum, factors,   validate
    ("tiny",                    16,   4,  2,  2,   (4, 4),     True),
    ("small",                   32,   4,  2,  2,   (8, 4),     True),
    ("small-deep",              32,   8,  4,  2,   (8, 4),     True),
    ("mid",                     64,   6,  2,  2,   (8, 8),     True),
    ("mid-alpha8",              64,  16,  8,  2,   (8, 8),     True),
    ("tile32",                 256,   8,  4,  2,   (16, 16),   True),
    ("tile32-full",           1024,   8,  4,  2,   (32, 32),   False),
    ("tile32-alpha8",         1024,  16,  8,  2,   (32, 32),   False),
    ("hw-shape-alpha16",      1024,  48, 16,  3,   (32, 32),   False),
]

if __name__ == "__main__":
    only = sys.argv[1:] or None
    out = []
    for name, N, LQ, LP, dnum, fac, val in CANDIDATES:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            c = build_case(N, LQ, LP, dnum, fac, validate=val)
        except Exception as ex:
            print(f"{name:<20} FAILED: {type(ex).__name__}: {ex}")
            continue
        tb = time.time() - t0
        t0 = time.time()
        mp, final, rungs = passes.run_passes(c["l2"], hw.CORES, verbose=False)
        tp = time.time() - t0
        fired = {r["pass_name"]: r for r in rungs if "delta" in r}
        moved = fired.get("select_engines", {}).get("moved_to_fpu", 0)
        rmvd = fired.get("fuse_dispatches", {}).get("removed", 0)
        a2a = fired.get("place_comm", {}).get("a2a_onchip", 0)
        chunk = fired.get("legalize", {}).get("chunked", 0)
        base_ms = rungs[0]["makespan"] / 1e3
        print(f"{name:<20} N={N:<5} LQ={LQ:<3} dnum={dnum}  L1={len(c['l1'].ops):<5} "
              f"L2={len(c['l2'].ops):<6} build={tb:>6.1f}s pass={tp:>5.1f}s  "
              f"a2a={a2a:<5} fuse={rmvd:<5} fpu={moved:<5} chunk={chunk:<4} "
              f"{base_ms:>8.2f}->{final['makespan']/1e3:>7.2f}ms {base_ms/(final['makespan']/1e3):>5.2f}x")
        out.append(dict(name=name, N=N, LQ=LQ, LP=LP, dnum=dnum, factors=list(fac), validate=val,
                        l1=len(c["l1"].ops), l2=len(c["l2"].ops), build_s=round(tb, 2),
                        pass_s=round(tp, 2), a2a=a2a, fuse=rmvd, fpu=moved, chunk=chunk,
                        base_ms=round(base_ms, 3), final_ms=round(final["makespan"] / 1e3, 3)))
    json.dump(out, open("cases.json", "w"), indent=1)
    print(f"\nwrote cases.json ({len(out)} configs)")
