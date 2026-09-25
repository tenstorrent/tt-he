#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Validate the PIR scan (initial dimension) as a BATCHED modular matmul — the Part-2 mapping in
docs/pir/SCAN_ALGORITHM.md. Proves, bit-exact in Python, the whole chain:

    coeff-domain scan  ==  iNTT( Σ_k NTT(DB) ⊙ NTT(Q) )              (NTT convolution theorem)
                       ==  iNTT( per-slot matmul_modq(DB_ntt, Q_ntt) )   (Step 2 = batched matmul)
                       ==  iNTT( per-slot matmul_modq_via_int8(...) )     (int8 lowering, exact)

Scan (coeff domain):  response[r][c] = Σ_{k<N0} DB[k·rest+r] ⊛ query[k][c]   (mod q),  ⊛ negacyclic.
Einsum (eval domain): A[r,c,j] = Σ_k DB_ntt[k,r,j]·Q_ntt[k,c,j]   "krj,kcj->rcj".

    python3 validate_scan.py
"""
import numpy as np
import reference as ref


def negmul(a, b, q):
    """a ⊛ b in Z_q[x]/(x^N+1) (negacyclic), coefficient domain."""
    N = len(a); c = [0] * N
    for i in range(N):
        for j in range(N):
            k = i + j
            if k < N: c[k] = (c[k] + a[i] * b[j]) % q
            else:     c[k - N] = (c[k - N] - a[i] * b[j]) % q
    return [x % q for x in c]


def scan_coeff(DB, Q, q, N0, rest):
    """Golden: response[r][c] = Σ_k DB[k·rest+r] ⊛ Q[k][c] mod q."""
    resp = [[[0] * len(DB[0]) for _ in range(2)] for _ in range(rest)]
    for r in range(rest):
        for c in range(2):
            acc = [0] * len(DB[0])
            for k in range(N0):
                prod = negmul(DB[k * rest + r], Q[k][c], q)
                acc = [(acc[i] + prod[i]) % q for i in range(len(acc))]
            resp[r][c] = acc
    return resp


def main():
    rng = np.random.default_rng(0)
    N, N0, rest, q = 8, 4, 2, 17          # small: q ≡ 1 mod 2N (17 mod 16 = 1) so a 2N-th root exists
    psi = ref.find_2nth_root(q, N)
    Ncells = N0 * rest

    DB = [[int(rng.integers(q)) for _ in range(N)] for _ in range(Ncells)]      # plaintext polys
    Q  = [[[int(rng.integers(q)) for _ in range(N)] for _ in range(2)] for _ in range(N0)]  # query cts

    # ---- golden (coeff domain) ----
    gold = scan_coeff(DB, Q, q, N0, rest)

    # ---- NTT everything (Step 0/1) ----
    DB_ntt = [ref.ntt(DB[c], q, psi) for c in range(Ncells)]                    # [Ncells][N]
    Q_ntt  = [[ref.ntt(Q[k][c], q, psi) for c in range(2)] for k in range(N0)]  # [N0][2][N]

    # ---- Step 2 three ways, then iNTT (Step 3) ----
    def intt_out(A_ntt):   # A_ntt: [rest][2][N] -> coeff [rest][2][N]
        return [[list(ref.intt(A_ntt[r][c], q, psi)) for c in range(2)] for r in range(rest)]

    # (a) einsum / pointwise MAC
    A_pw = [[[ (sum(DB_ntt[k * rest + r][j] * Q_ntt[k][c][j] for k in range(N0))) % q
               for j in range(N)] for c in range(2)] for r in range(rest)]
    # (b) per-slot matmul_modq :  A[:,:,j] = DB_j^T · Q_j , DB_j=[N0,rest], Q_j=[N0,2]
    A_mm = [[[0] * N for _ in range(2)] for _ in range(rest)]
    for j in range(N):
        DB_j = np.array([[DB_ntt[k * rest + r][j] for r in range(rest)] for k in range(N0)])  # [N0,rest]
        Q_j  = np.array([[Q_ntt[k][c][j] for c in range(2)] for k in range(N0)])              # [N0,2]
        Aj = ref.matmul_modq(DB_j, Q_j, q, transpose_a=True)                                  # [rest,2]
        for r in range(rest):
            for c in range(2): A_mm[r][c][j] = int(Aj[r][c])
    # (c) per-slot matmul via int8 (decompose both -> int8 planes -> recombine)
    A_i8 = [[[0] * N for _ in range(2)] for _ in range(rest)]
    for j in range(N):
        DB_j = np.array([[DB_ntt[k * rest + r][j] for r in range(rest)] for k in range(N0)])
        Q_j  = np.array([[Q_ntt[k][c][j] for c in range(2)] for k in range(N0)])
        Aj = ref.matmul_modq_via_int8(DB_j.T, Q_j, q)                                          # [rest,2]
        for r in range(rest):
            for c in range(2): A_i8[r][c][j] = int(Aj[r][c])

    got_pw, got_mm, got_i8 = intt_out(A_pw), intt_out(A_mm), intt_out(A_i8)

    def eq(x, y): return x == y
    ok = True
    for name, got in [("pointwise-MAC", got_pw), ("matmul_modq", got_mm), ("matmul_via_int8", got_i8)]:
        good = eq(got, gold)
        ok &= good
        print(f"  scan via {name:16s} == coeff-domain golden : {'OK' if good else 'MISMATCH'}")
    print(f"params: N={N} N0={N0} rest={rest} q={q} psi={psi}  ->  {'ALL PASS' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
