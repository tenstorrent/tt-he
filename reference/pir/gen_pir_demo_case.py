#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""End-to-end PIR demo case (client crypto = CPU reference; server runs on device).

Hypercube DB of N = N0 × 2 entries (one initial dimension + one fold). Client encrypts a query for
a chosen index i; the device server does initial-dimension dot product + 1-of-2 fold + mod-switch;
the client decrypts to recover DB[i].  This dumps everything the device demo needs + the golden.

Binary (LE): u32 MAGIC=0x50495244('PIRD'), n, ell, B, N0, q1, q2, qp, idx
    DB   : per ch(q1,q2): per e(N0*2): n u32          (plaintext DB entries mod q_ch)
    sel0 : per ch: per k(N0): per comp(2): n u32      (initial-dim one-hot selector cts)
    C1   : per ch: per r(2ℓ): per col(2): n u32       (RGSW of the fold bit)
    s    : n i32                                       (secret key, ternary — client)
    dbi  : n i32                                       (expected DB[idx], for the check)

    python3 gen_pir_demo_case.py [--idx 3]
"""
import argparse, random, struct
import onionpir_ref as ref
MAGIC = 0x50495245   # 'PIRE' — full-OnionPIR case (packed query + keys; server expands+unpacks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024); ap.add_argument("--ell", type=int, default=6)
    ap.add_argument("--B", type=int, default=1 << 10); ap.add_argument("--N0", type=int, default=2)
    ap.add_argument("--idx", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pir_demo_case.bin"); a = ap.parse_args()
    rng = random.Random(a.seed); pp = ref.Params(n=a.n, B=a.B, ell=a.ell)
    q, q1, q2, t = pp.q, pp.q1, pp.q2, pp.t
    dims = [a.N0, 2]; N = a.N0 * 2
    m_exp = 0
    while (1 << m_exp) < a.N0:
        m_exp += 1
    assert (1 << m_exp) == a.N0, "N0 must be a power of two"
    s = ref.keygen(pp, rng)
    # Database: each entry holds one integer value (constant polynomial), so it prints cleanly.
    dbrng = random.Random(0xDB)
    dbvals = [dbrng.randrange(t) for _ in range(N)]
    DB = [[dbvals[e]] + [0] * (a.n - 1) for e in range(N)]
    i = a.idx; c0 = i // 2; c1 = i % 2

    # ---- client's compact query: TWO packed cts + one-time key material (server expands both) ----
    m_lad = 0
    while (1 << m_lad) < a.ell:
        m_lad += 1
    mk = max(m_exp, m_lad)
    gkeys = ref.galois_keys_for_expand(pp, s, mk, rng)           # Galois KSK, max(m_exp,m_lad) levels
    gsw_key = ref.gen_gsw_key(pp, s, rng)                        # RGSW(s), the RLWE->RGSW unpack key
    # init-dim query: packed one-hot at c0, pre-scaled by (2^m_exp)^-1 mod t (Δ-scaled BFV)
    inv = ref.modinv(1 << m_exp, t)
    packed = [0] * a.n; packed[c0] = inv
    ctq = ref.encrypt(pp, s, packed, rng)
    # fold-ladder query: packed gadget ladder c1·B^k, pre-scaled by (2^m_lad)^-1 mod q (scaleless)
    g = ref.gadget_vector(a.B, a.ell, q)
    invf = ref.modinv(1 << m_lad, q)
    packed_f = [0] * a.n
    for k in range(a.ell): packed_f[k] = (invf * c1 * g[k]) % q
    qfold = ref.encrypt_scaleless(pp, s, packed_f, rng)

    # ---- golden self-check: expand BOTH queries + unpack + answer decrypts to DB[i] ----
    sel0 = ref.expand_query(pp, ctq, m_exp, gkeys)[:a.N0]
    msg = ref.expand_query(pp, qfold, m_lad, gkeys)[:a.ell]      # ℓ fold message rows (expanded)
    C1 = ref.rlwe_to_rgsw(pp, msg, gsw_key)
    ans = ref.pir_answer(pp, DB, dims, sel0, [C1])
    qp = ref.ntt_primes(1, a.n, bits=22)[0]
    rec = ref.decrypt_modswitched(pp, s, ref.mod_switch(pp, ans, qp), qp)
    assert rec == DB[i], f"reference retrieval failed: {rec[0]} != {DB[i][0]}"
    twoL = 2 * a.ell
    with open(a.out, "wb") as f:
        f.write(struct.pack("<11I", MAGIC, a.n, a.ell, a.B, a.N0, m_exp, q1, q2, qp, i, t))
        for qi in (q1, q2):                                      # DB
            for e in range(N): f.write(struct.pack(f"<{a.n}I", *[x % qi for x in DB[e]]))
        for comp in (ctq[0], ctq[1]):                            # init-dim query: per comp: per ch
            for qi in (q1, q2): f.write(struct.pack(f"<{a.n}I", *[x % qi for x in comp]))
        for comp in (qfold[0], qfold[1]):                        # fold-ladder query: per comp: per ch
            for qi in (q1, q2): f.write(struct.pack(f"<{a.n}I", *[x % qi for x in comp]))
        for aa in range(mk):                                     # Galois keys: per level: per ch: per k: per col
            ksk = gkeys[(a.n >> aa) + 1]
            for qi in (q1, q2):
                for k in range(a.ell):
                    for col in range(2): f.write(struct.pack(f"<{a.n}I", *[x % qi for x in ksk[k][col]]))
        for qi in (q1, q2):                                      # RGSW(s): per ch: per row: per col
            for r in range(twoL):
                for col in range(2): f.write(struct.pack(f"<{a.n}I", *[x % qi for x in gsw_key[r][col]]))
        f.write(struct.pack(f"<{a.n}i", *s))
        f.write(struct.pack(f"<{a.n}i", *DB[i]))
    print(f"wrote {a.out}: N={N} (N0={a.N0}x2) idx={i} (coords c0={c0},c1={c1}) m_exp={m_exp} m_lad={m_lad} "
          f"ell={a.ell} B={a.B} qp={qp}  ref retrieval OK (server expands BOTH queries + unpacks)")

if __name__ == "__main__":
    main()
