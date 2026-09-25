#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Client-side query material for the 65 536-row demo, at n=2048 / K=3.

Emits everything the SERVER needs and nothing it should not have:
  * one packed initial-dimension query ciphertext (the device expands it into 512 selectors)
  * Galois keys for the expansion tree
  * the per-fold RGSW selection ciphertexts
plus, for the CLIENT side of the demo, the secret key so it can decrypt the response.

The database is NOT in this file — it lives in db65536.pir (see encode_db.cpp), so the demo loads a
1.5 GB database once and can then answer any index with a ~9 MB query file.

Deliberately NOT done here: the reference answer. `pir_query_and_answer` would have to run the whole
pipeline in Python over 65 536 cells (a 512-term scan per cell, in bigint arithmetic) — hours. The
demo verifies differently and better: rows are self-identifying, so the decoded string must equal the
known line of db65536.txt. See docs/pir/PIR65536_DEMO.md §5.

Binary (little-endian):
    u32 MAGIC=0x51363553('Q65S'), n, K, ell, B, N0, m_exp, nfold, idx, t
    K u32 primes
    s        : n i32                                       (client secret)
    query    : per comp(2): per ch(K): n u32               (packed one-hot, Delta-scaled)
    gkeys    : per level(m_exp): per ch(K): per k(ell): per col(2): n u32
    has_gold : u32   (1 if the reference selectors follow, for the serialization check)
    selectors: per k(N0): per comp(2): per ch(K): n u32                    (only if has_gold)
    sel_bits : per f(nfold): per row(2ell): per col(2): per ch(K): n u32   (RGSW, coeff domain)

    python3 gen_query65536.py --idx 40000
"""
import argparse, random, struct, time
import onionpir_ref as ref

MAGIC = 0x51363553

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--ell", type=int, default=8)
    ap.add_argument("--B", type=int, default=1 << 11)
    ap.add_argument("--N0", type=int, default=512, help="first hypercube dimension")
    ap.add_argument("--nfold", type=int, default=7, help="binary fold dimensions (N0*2^nfold rows)")
    ap.add_argument("--idx", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="query65536.bin")
    ap.add_argument("--golden", action="store_true",
                    help="also emit reference selectors (small N0 only) so a device test can verify "
                         "this file's SERIALIZATION, not just its arithmetic")
    a = ap.parse_args()

    rows = a.N0 * (1 << a.nfold)
    assert 0 <= a.idx < rows, f"idx must be < {rows}"
    dims = [a.N0] + [2] * a.nfold
    m_exp = a.N0.bit_length() - 1
    assert (1 << m_exp) == a.N0, "N0 must be a power of two"

    rng = random.Random(a.seed)
    pp = ref.Params(n=a.n, t=257, B=a.B, ell=a.ell, K=a.K)
    t0 = time.time()
    s = ref.keygen(pp, rng)

    c0, fold_bits = ref._coords(dims, a.idx)
    print(f"idx={a.idx} of {rows}  ->  c0={c0} (of {a.N0}), fold bits={fold_bits}")

    # packed initial-dimension query: one-hot at c0, pre-scaled by (2^m_exp)^-1 mod t
    inv = ref.modinv(1 << m_exp, pp.t)
    packed = [0] * a.n
    packed[c0] = inv
    ctq = ref.encrypt(pp, s, packed, rng)
    t_q = time.time()

    gkeys = ref.galois_keys_for_expand(pp, s, m_exp, rng)
    t_g = time.time()

    # per-fold RGSW selection bits.  expand_ladder=False: the client sends the ell message rows
    # directly rather than a packed ladder the server must expand — the server still never sees the
    # bit, and it keeps generation minutes rather than hours.
    gsw_key = ref.gen_gsw_key(pp, s, rng)
    sel_bits = []
    for b in fold_bits:
        rowsm = ref.gsw_msg_rows(pp, s, b, gkeys, rng, False)
        sel_bits.append(ref.rlwe_to_rgsw(pp, rowsm, gsw_key))
    t_b = time.time()

    sel_golden = None
    if a.golden:
        # only tractable for small N0; this is a serialization check, not a scale test
        sel_golden = ref.expand_query(pp, ctq, m_exp, gkeys)[:a.N0]
        dec = [ref.decrypt(pp, s, c)[0] for c in sel_golden]
        assert dec == [1 if k == c0 else 0 for k in range(a.N0)], f"reference selectors not one-hot: {dec}"
        print(f"  golden: {a.N0} selectors verified one-hot at c0={c0}")
    t_sel = time.time()

    twoL = 2 * a.ell
    def wpoly(v):
        f.write(struct.pack(f"<{a.n}I", *[x % (1 << 32) for x in v]))
    with open(a.out, "wb") as f:
        f.write(struct.pack("<10I", MAGIC, a.n, a.K, a.ell, a.B, a.N0, m_exp, a.nfold, a.idx, pp.t))
        f.write(struct.pack(f"<{a.K}I", *pp.primes))
        f.write(struct.pack(f"<{a.n}i", *s))
        for comp in range(2):
            for ch in range(a.K):
                wpoly([x % pp.primes[ch] for x in ctq[comp]])
        for lvl in range(m_exp):                       # gkeys is a DICT keyed by j=(n>>a)+1
            ksk = gkeys[(a.n >> lvl) + 1]              # and each KSK is ksk[k][col]
            for ch in range(a.K):
                for k in range(a.ell):
                    for col in range(2):
                        wpoly([x % pp.primes[ch] for x in ksk[k][col]])
        f.write(struct.pack("<I", 1 if sel_golden else 0))
        if sel_golden:
            for k in range(a.N0):
                for comp in range(2):
                    for ch in range(a.K):
                        wpoly([x % pp.primes[ch] for x in sel_golden[k][comp]])
        for C in sel_bits:
            for r in range(twoL):
                for col in range(2):
                    for ch in range(a.K):
                        wpoly([x % pp.primes[ch] for x in C[r][col]])

    import os
    print(f"wrote {a.out}: {os.path.getsize(a.out)/2**20:.1f} MB")
    print(f"  keygen+query {t_q-t0:5.1f}s | galois keys {t_g-t_q:5.1f}s | "
          f"{a.nfold} fold RGSW {t_b-t_g:5.1f}s | total {time.time()-t0:5.1f}s")

if __name__ == "__main__":
    main()
