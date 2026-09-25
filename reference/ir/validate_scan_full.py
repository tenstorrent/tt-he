#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Full PIR first-dimension (scan) end-to-end, proving the MATMUL scan does real BFV selection.

Minimal single-prime BFV (one RNS channel = what the device matmul does per channel):
  keygen s;  enc(m) = (Δ·m − a·s + e, a);  dec(c0,c1) = round((c0 + c1·s)/Δ) mod t.
One-hot query:  sel[k] = Enc(1) if k==c0 else Enc(0).
Scan (golden, coeff):  acc[r][comp] = Σ_k DB[k·rest+r] ⊛ sel[k][comp]  (mod q)  → decrypts to DB[c0·rest+r].

Reproduce Step 2 as a BATCHED MATMUL through the NTT (docs/pir/SCAN_ALGORITHM.md):
  DB_ntt = NTT(DB), sel_ntt = NTT(sel);  A[r,c,j] = Σ_k DB_ntt[k,r,j]·sel_ntt[k,c,j]  (per-slot matmul_modq);
  acc2 = iNTT(A).  Assert acc2 == coeff-golden AND decrypts to DB[c0·rest+r].

    python3 validate_scan_full.py
"""
import numpy as np
import reference as ref


def negmul(a, b, q):
    N = len(a); c = [0] * N
    for i in range(N):
        for j in range(N):
            k = i + j
            if k < N: c[k] = (c[k] + a[i] * b[j]) % q
            else:     c[k - N] = (c[k - N] - a[i] * b[j]) % q
    return [x % q for x in c]


def main():
    rng = np.random.default_rng(0)
    N, N0, rest, q, t = 8, 4, 2, 40961, 3       # q ≡ 1 mod 2N; single NTT prime; big enough Δ for BFV noise
    psi = ref.find_2nth_root(q, N)
    Delta = q // t
    Ncells = N0 * rest
    c0_target = 2                                # the selected first-dim coordinate

    s = [int(rng.integers(-1, 2)) % q for _ in range(N)]                          # ternary secret
    def enc(m):
        a = [int(rng.integers(q)) for _ in range(N)]
        e = [int(rng.integers(-1, 2)) for _ in range(N)]
        as_ = negmul(a, s, q)
        c0 = [(Delta * m[i] - as_[i] + e[i]) % q for i in range(N)]
        return [c0, a]
    def dec(ct):
        phase = [(ct[0][i] + negmul(ct[1], s, q)[i]) % q for i in range(N)]
        return [round((p if p <= q // 2 else p - q) / Delta) % t for p in phase]

    DB = [[int(rng.integers(t)) for _ in range(N)] for _ in range(Ncells)]        # plaintext DB (mod t)
    sel = [enc([1 if k == c0_target else 0] + [0] * (N - 1)) for k in range(N0)]   # one-hot query cts

    # ---- golden scan (coeff): acc[r][comp] = Σ_k DB[k·rest+r] ⊛ sel[k][comp] ----
    def scan_coeff():
        out = [[[0] * N for _ in range(2)] for _ in range(rest)]
        for r in range(rest):
            for c in range(2):
                acc = [0] * N
                for k in range(N0):
                    acc = [(acc[i] + negmul(DB[k * rest + r], sel[k][c], q)[i]) % q for i in range(N)]
                out[r][c] = acc
        return out
    gold = scan_coeff()

    # ---- matmul-scan through the NTT ----
    DB_ntt = [ref.ntt(DB[c], q, psi) for c in range(Ncells)]
    sel_ntt = [[ref.ntt(sel[k][c], q, psi) for c in range(2)] for k in range(N0)]
    A = [[[0] * N for _ in range(2)] for _ in range(rest)]
    for j in range(N):                                                            # batch over slots
        DBj = np.array([[DB_ntt[k * rest + r][j] for r in range(rest)] for k in range(N0)])  # [N0,rest]
        Qj  = np.array([[sel_ntt[k][c][j] for c in range(2)] for k in range(N0)])            # [N0,2]
        Aj = ref.matmul_modq(DBj.T, Qj, q)                                        # [rest,2] contract k
        for r in range(rest):
            for c in range(2): A[r][c][j] = int(Aj[r][c])
    acc2 = [[list(ref.intt(A[r][c], q, psi)) for c in range(2)] for r in range(rest)]

    ok = (acc2 == gold)
    print(f"  matmul-scan (NTT+matmul_modq+iNTT) == coeff golden : {'OK' if ok else 'MISMATCH'}")

    # ---- decrypt each output row: must equal DB[c0·rest + r] ----
    sel_ok = True
    for r in range(rest):
        d = dec([acc2[r][0], acc2[r][1]])
        want = DB[c0_target * rest + r]
        good = (d == want)
        sel_ok &= good
        print(f"  row r={r}: decrypt(scan) = {d}  ==  DB[c0·rest+r] = {want} : {'OK' if good else 'MISMATCH'}")
    allok = ok and sel_ok
    print(f"params: N={N} N0={N0} rest={rest} q={q} t={t} c0={c0_target}  ->  {'ALL PASS' if allok else 'FAILED'}")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
