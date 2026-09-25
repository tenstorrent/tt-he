#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Query-expansion (#18) test case: ONE packed BFV query -> N0 one-hot selector cts via the Subs
tree, plus the RLWE->RGSW unpack (#18b).  The device test (test_query_expand2048) runs the tree on
device — host automorph/shift/add + the validated device BV key-switch — and checks the N0 selectors
+ unpack bottom rows bit-exact vs onionpir_ref.

K-parameterized (K = RNS limb count): K=2 (~56-bit q), K=3 (~84-bit q).

Binary (VERSION 2, little-endian):
    u32 MAGIC=0x45585044('EXPD'), VERSION=2, n, ell, B, N0, m_exp, K, t
    K * u32 primes
    s        : n i32                                    (client secret, ternary — for decrypt check)
    query    : per comp(2): per ch(K): n u32            (packed one-hot query)
    gkeys    : per level a(m_exp): per ch(K): per k(ell): per col(2): n u32   (Galois KSK_a)
    golden   : per k(N0): per comp(2): per ch(K): n u32                       (expand_query outputs)
    u32 bit
    RGSW(s)  : per ch(K): per row(2ℓ): per col(2): n u32
    msg      : per k(ell): per comp(2): per ch(K): n u32
    bottom   : per k(ell): per col(2): per ch(K): n u32

    python3 gen_expand_case.py --n 2048 --ell 6 --B 1024 --N0 4                # K=2 default
    python3 gen_expand_case.py --K 3 --n 2048 --ell 8 --B 2048 --N0 4 --out expand_k3.bin
"""
import argparse, random, struct
import onionpir_ref as ref

MAGIC = 0x45585044
VERSION = 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--ell", type=int, default=6)
    ap.add_argument("--B", type=int, default=1 << 10)
    ap.add_argument("--N0", type=int, default=4)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--target", type=int, default=2)
    ap.add_argument("--bit", type=int, default=1, help="fold bit for the unpack (RLWE->RGSW) section")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="expand_case.bin")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pp = ref.Params(n=a.n, B=a.B, ell=a.ell, K=a.K)
    q, primes, t = pp.q, pp.primes, pp.t
    m_exp = 0
    while (1 << m_exp) < a.N0:
        m_exp += 1
    assert (1 << m_exp) == a.N0, "N0 must be a power of two"

    s = ref.keygen(pp, rng)
    inv = ref.modinv(1 << m_exp, t)                      # client pre-scale by (2^m)^{-1} mod t
    packed = [0] * a.n
    packed[a.target] = inv
    ctq = ref.encrypt(pp, s, packed, rng)
    gkeys = ref.galois_keys_for_expand(pp, s, m_exp, rng)      # dict j -> KSK (ell rows [col0,col1])
    sel = ref.expand_query(pp, ctq, m_exp, gkeys)[:a.N0]       # golden N0 selector cts
    dec = [ref.decrypt(pp, s, c)[0] for c in sel]
    assert dec == [1 if i == a.target else 0 for i in range(a.N0)], f"golden not one-hot: {dec}"

    # ---- unpack (#18b): RGSW(s) + ℓ message rows (phase bit·B^k) -> RGSW(bit); golden bottom rows ----
    gsw_key = ref.gen_gsw_key(pp, s, rng)                            # RGSW(s), 2ℓ×2
    msg_rows = ref.gsw_msg_rows(pp, s, a.bit, gkeys, rng, expand_ladder=False)   # ℓ message rows [c0,c1]
    rgsw_bit = ref.rlwe_to_rgsw(pp, msg_rows, gsw_key)               # 2ℓ×2 ; top=msg_rows, bottom=extprod
    bottom = rgsw_bit[a.ell:]                                        # ℓ golden bottom rows [col0,col1]

    def wp(f, poly, qi):
        f.write(struct.pack(f"<{a.n}I", *[x % qi for x in poly]))

    with open(a.out, "wb") as f:
        f.write(struct.pack("<9I", MAGIC, VERSION, a.n, a.ell, a.B, a.N0, m_exp, a.K, t))
        f.write(struct.pack(f"<{a.K}I", *primes))
        f.write(struct.pack(f"<{a.n}i", *s))
        for comp in (ctq[0], ctq[1]):                          # packed query c0, c1
            for qi in primes: wp(f, comp, qi)
        for aa in range(m_exp):                                # Galois key per level a
            ksk = gkeys[(a.n >> aa) + 1]
            for qi in primes:
                for k in range(a.ell):
                    for col in range(2): wp(f, ksk[k][col], qi)
        for k in range(a.N0):                                  # golden selectors
            for comp in range(2):
                for qi in primes: wp(f, sel[k][comp], qi)
        f.write(struct.pack("<I", a.bit))
        for qi in primes:                                      # RGSW(s): per ch: per row(2ℓ): per col(2)
            for row in range(2 * a.ell):
                for col in range(2): wp(f, gsw_key[row][col], qi)
        for k in range(a.ell):                                 # message rows: per k: per comp(2): per ch
            for comp in range(2):
                for qi in primes: wp(f, msg_rows[k][comp], qi)
        for k in range(a.ell):                                 # golden bottom rows: per k: per col(2): per ch
            for col in range(2):
                for qi in primes: wp(f, bottom[k][col], qi)
    print(f"wrote {a.out}: n={a.n} ell={a.ell} B={a.B} N0={a.N0} m_exp={m_exp} K={a.K} target={a.target} "
          f"bit={a.bit} q=~{q.bit_length()}b  golden one-hot={dec}  OK")

if __name__ == "__main__":
    main()
