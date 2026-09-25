#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Initial-dimension dot product (#12) test case — the throughput-critical PIR op, n=1024.

response[r] = Σ_{k<N0} DB[k*rest+r] · query_ct[k]   (plaintext × ciphertext, accumulated).
Per output r and ct component (c0,c1):  acc = Σ_k DB[k*rest+r] ⊛ query[k][comp]  (negacyclic, mod q).
On device this is an NTT-domain multiply-accumulate: DB (public plaintext) is pre-NTT'd offline;
the hot loop is Σ_k query_ntt ⊙ DB_ntt. Validated bit-exact vs this reference.

Binary (little-endian):
    u32 MAGIC=0x494e4954('INIT'), u32 n, u32 N0, u32 rest, u32 q1, u32 q2
    query : per ch(q1,q2): per k(N0): per comp(2): n u32   (query[k][comp] mod q_ch)
    DB    : per ch: per idx(N0*rest): n u32                (DB[idx] mod q_ch; plaintext coeffs < t)
    out   : per ch: per r(rest): per comp(2): n u32        (acc[r][comp] mod q_ch)

    python3 gen_initdim_case.py            # n=1024, N0=4, rest=2 -> initdim_case.bin
"""
import argparse, random, struct
import onionpir_ref as ref

MAGIC = 0x494E4954

def gen(n, N0, rest, seed):
    rng = random.Random(seed)
    pp = ref.Params(n=n)
    q, q1, q2, t = pp.q, pp.q1, pp.q2, pp.t
    # query: N0 ciphertexts (c0,c1) — random mod q (operation test; BFV semantics covered by onionpir_ref)
    query = [[[rng.randrange(q) for _ in range(n)] for _ in range(2)] for _ in range(N0)]  # [k][comp]
    DB = [[rng.randrange(t) for _ in range(n)] for _ in range(N0 * rest)]                  # plaintext (< t)
    # golden: acc[r][comp] = Σ_k DB[k*rest+r] ⊛ query[k][comp] mod q
    out = [[[0] * n for _ in range(2)] for _ in range(rest)]
    for r in range(rest):
        for comp in range(2):
            acc = [0] * n
            for k in range(N0):
                acc = ref.poly_add(acc, ref.negacyclic_mul(DB[k * rest + r], query[k][comp], q), q)
            out[r][comp] = acc
    return {"n": n, "N0": N0, "rest": rest, "q1": q1, "q2": q2, "query": query, "DB": DB, "out": out}

def write(path, c):
    n, N0, rest = c["n"], c["N0"], c["rest"]
    with open(path, "wb") as f:
        f.write(struct.pack("<6I", MAGIC, n, N0, rest, c["q1"], c["q2"]))
        for qi in (c["q1"], c["q2"]):
            for k in range(N0):
                for comp in range(2):
                    f.write(struct.pack(f"<{n}I", *[x % qi for x in c["query"][k][comp]]))
        for qi in (c["q1"], c["q2"]):
            for idx in range(N0 * rest):
                f.write(struct.pack(f"<{n}I", *[x % qi for x in c["DB"][idx]]))
        for qi in (c["q1"], c["q2"]):
            for r in range(rest):
                for comp in range(2):
                    f.write(struct.pack(f"<{n}I", *[x % qi for x in c["out"][r][comp]]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--N0", type=int, default=4)
    ap.add_argument("--rest", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="initdim_case.bin")
    a = ap.parse_args()
    c = gen(a.n, a.N0, a.rest, a.seed)
    write(a.out, c)
    print(f"wrote {a.out}: n={c['n']} N0={c['N0']} rest={c['rest']} q1={c['q1']} q2={c['q2']}")

if __name__ == "__main__":
    main()
