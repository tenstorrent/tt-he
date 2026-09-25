#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""PIR answer pipeline (#19) test case: initial-dimension scan + binary folds over a hypercube DB,
recovering DB[idx].  Isolates the ANSWER (the genuinely-new device integration for milestone #7) —
expand/unpack are validated separately (test_query_expand2048).  Selectors are provided directly:
sel0 = the N0 initial-dim one-hot BFV cts; sel_bits = the per-fold RGSW selection ciphertexts.

K-parameterized (K=2 ~56-bit / K=3 ~84-bit).  Bit-exact vs onionpir_ref.pir_answer; also carries the
plaintext DB[idx] so the device test can confirm the recovered answer decrypts to it.

Binary (VERSION 1, little-endian):
    u32 MAGIC=0x50414E53('PANS'), VERSION=1, n, ell, B, K, t, ndims, N0, nfold(=ndims-1), idx
    ndims * u32 dims
    K * u32 primes
    s        : n i32                                   (secret, for the decrypt check)
    db       : per cell(Ncells): per ch(K): n u32      (plaintext residues; values < t)
    sel0     : per k(N0): per comp(2): per ch(K): n u32
    sel_bits : per f(nfold): per row(2ℓ): per col(2): per ch(K): n u32   (RGSW, coeff domain)
    ans      : per comp(2): per ch(K): n u32           (golden pir_answer output)
    dbidx    : n i32                                   (plaintext DB[idx], for decrypt sanity)

    python3 gen_pir_answer_case.py --n 2048 --ell 6 --B 1024 --dims 4,2          # K=2 default
    python3 gen_pir_answer_case.py --K 3 --n 2048 --ell 8 --B 2048 --dims 4,2 --idx 5 --out ans_k3.bin
"""
import argparse, random, struct
import onionpir_ref as ref

MAGIC = 0x50414E53
VERSION = 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--ell", type=int, default=6)
    ap.add_argument("--B", type=int, default=1 << 10)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--dims", type=str, default="4,2", help="hypercube dims, e.g. 4,2 or 8,2,2")
    ap.add_argument("--idx", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pir_answer_case.bin")
    ap.add_argument("--no-golden", action="store_true",
                    help="skip the O(cells) Python pir_answer golden (ans=0); device decrypt-checks vs dbidx. "
                         "Required for large hypercubes -- the golden is what costs days at 128 MB scale.")
    a = ap.parse_args()
    dims = [int(x) for x in a.dims.split(",")]
    N0 = dims[0]
    Ncells = 1
    for d in dims:
        Ncells *= d
    assert 0 <= a.idx < Ncells
    rng = random.Random(a.seed)
    pp = ref.Params(n=a.n, B=a.B, ell=a.ell, K=a.K)
    q, primes, t = pp.q, pp.primes, pp.t

    s = ref.keygen(pp, rng)
    db = ref.make_database(pp, dims, rng)
    c0, fold_bits = ref._coords(dims, a.idx)

    sel0 = ref.onehot_cts(pp, s, N0, c0, rng)                       # N0 BFV selectors
    gsw_key = ref.gen_gsw_key(pp, s, rng)
    # per-fold RGSW selection cts (expand_ladder=False: inject msg rows directly)
    sel_bits = []
    for b in fold_bits:
        msg_rows = ref.gsw_msg_rows(pp, s, b, None, rng, expand_ladder=False)
        sel_bits.append(ref.rlwe_to_rgsw(pp, msg_rows, gsw_key))    # 2ℓ×2 RGSW, coeff domain
    if a.no_golden:
        # SCALE-UP MODE. pir_answer() redoes the whole server computation in Python -- rest*N0 pt_ct_mul
        # plus the fold external products -- which is what makes a big hypercube take days (11 min for
        # just 32 cells at K=3). The device test does not need it: it can decrypt its own response with
        # the secret `s` below and compare against `dbidx`, which is exactly how the OnionPIRv2 `pir`
        # test validates itself. Everything still emitted here (keys, DB, selectors, s, dbidx) is
        # O(N0 + ell), NOT O(cells), so real 128 MB-class cases become generatable.
        ans = [[0] * a.n, [0] * a.n]
        print(f"  [--no-golden] skipped Python pir_answer; device must decrypt-check vs dbidx")
    else:
        ans = ref.pir_answer(pp, db, dims, sel0, sel_bits)           # golden response ct
        # sanity: recovered answer decrypts to DB[idx]
        dec = ref.decrypt(pp, s, ans)
        assert dec == db[a.idx], f"golden answer != DB[idx]"

    def wp(f, poly, qi): f.write(struct.pack(f"<{a.n}I", *[x % qi for x in poly]))

    with open(a.out, "wb") as f:
        f.write(struct.pack("<11I", MAGIC, VERSION, a.n, a.ell, a.B, a.K, t, len(dims), N0, len(dims) - 1, a.idx))
        f.write(struct.pack(f"<{len(dims)}I", *dims))
        f.write(struct.pack(f"<{a.K}I", *primes))
        f.write(struct.pack(f"<{a.n}i", *s))
        for cell in range(Ncells):
            for qi in primes: wp(f, db[cell], qi)
        for k in range(N0):
            for comp in range(2):
                for qi in primes: wp(f, sel0[k][comp], qi)
        for fb in sel_bits:
            for row in range(2 * a.ell):
                for col in range(2):
                    for qi in primes: wp(f, fb[row][col], qi)
        for comp in range(2):
            for qi in primes: wp(f, ans[comp], qi)
        f.write(struct.pack(f"<{a.n}i", *db[a.idx]))
    print(f"wrote {a.out}: n={a.n} ell={a.ell} B={a.B} K={a.K} dims={dims} Ncells={Ncells} idx={a.idx} "
          f"c0={c0} bits={fold_bits} q=~{q.bit_length()}b  "
          + ("ans=ZERO (no golden) -- DEVICE MUST DECRYPT-CHECK vs dbidx" if a.no_golden
             else "recovered==DB[idx] OK"))

if __name__ == "__main__":
    main()
