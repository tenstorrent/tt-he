#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Negacyclic NTT-2048 as a four-step (32, 64) — CPU golden for the device kernel (#8).

PIR needs the NTT at the ring size n=2048 (plan §8). We reuse the four-step / int8-matmul
machinery already on silicon (nntt1024_dev = 32×32); the only change is one factor is 64, so that
stage is a 64×64 modular matmul = 2×2 hardware tiles with K-accumulation. Insight/structure comes
from algorithm/ntt_matmul (PLAN=(32,64) validates the *cyclic* transform bit-exact); this file adds
the NEGACYCLIC folding (ψ = 4096-th root into the stage-1 W columns and the twiddle) that BFV/PIR
need for the ring x^2048 + 1, and validates the whole thing vs schoolbook.

Layout (matches nntt1024_dev's n = n1 + N1·n2 convention, generalized to N1=32, N2=64):
    n = n1 + 32·n2,   n1∈[0,32),  n2∈[0,64)        input  index
    k = k2 + 64·k1,   k1∈[0,32),  k2∈[0,64)        output index

Four-step (derivation in comments): stage 1 is the 64-DFT (ψ^{32 n2} folded into its columns),
then a twiddle carrying ψ^{n1}·ω^{n1 k2}, then stage 2 the 32-DFT.

    python3 ntt2048_ref.py          # validate vs schoolbook for the PIR primes
"""

import sys

def modpow(b, e, m):
    r = 1; b %= m
    while e:
        if e & 1: r = r * b % m
        b = b * b % m; e >>= 1
    return r

def is_prime(x):
    if x < 2: return False
    d = 2
    while d * d <= x:
        if x % d == 0: return False
        d += 1
    return True

def prim_root(q):
    phi = q - 1; n = phi; f = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            f.append(d)
            while n % d == 0: n //= d
        d += 1
    if n > 1: f.append(n)
    for g in range(2, q):
        if all(modpow(g, phi // p, q) != 1 for p in f): return g
    raise RuntimeError("no primitive root")


class NTT2048:
    """Negacyclic NTT of length N=2048 over Z_q, four-step (N1=32, N2=64)."""
    N1, N2, N = 32, 64, 2048

    def __init__(self, q):
        assert (q - 1) % (2 * self.N) == 0, "need q ≡ 1 (mod 4096) for a 4096-th root"
        self.q = q
        g = prim_root(q)
        self.w   = modpow(g, (q - 1) // self.N, q)         # ω_2048
        self.psi = modpow(g, (q - 1) // (2 * self.N), q)   # ψ (4096-th root, ψ²=ω)
        self.w64 = modpow(self.w, self.N1, q)              # ω_64 = ω^{32}
        self.w32 = modpow(self.w, self.N2, q)              # ω_32 = ω^{64}
        q_ = q
        # baked matrices (the device pre-computes these offline)
        # stage 1 (64-DFT over n2):  WA[k2][n2] = ω_64^{k2 n2} · ψ^{32 n2}
        self.WA = [[(modpow(self.w64, k2 * n2, q_) * modpow(self.psi, self.N1 * n2, q_)) % q_
                    for n2 in range(self.N2)] for k2 in range(self.N2)]
        # twiddle:  Tw[n1][k2] = ψ^{n1} · ω^{n1 k2}
        self.Tw = [[(modpow(self.psi, n1, q_) * modpow(self.w, n1 * k2, q_)) % q_
                    for k2 in range(self.N2)] for n1 in range(self.N1)]
        # stage 2 (32-DFT over n1):  WC[k1][n1] = ω_32^{k1 n1}
        self.WC = [[modpow(self.w32, k1 * n1, q_) for n1 in range(self.N1)] for k1 in range(self.N1)]

    def forward(self, x):
        q, N1, N2 = self.q, self.N1, self.N2
        # reshape: Xin2[n1][n2] = x[n1 + 32 n2]
        Xin = [[x[n1 + N1 * n2] for n2 in range(N2)] for n1 in range(N1)]
        # stage 1: A[n1][k2] = Σ_n2 Xin[n1][n2] · WA[k2][n2]      (64-DFT, ψ folded)
        A = [[sum(Xin[n1][n2] * self.WA[k2][n2] for n2 in range(N2)) % q for k2 in range(N2)]
             for n1 in range(N1)]
        # twiddle: B = A ⊙ Tw
        B = [[A[n1][k2] * self.Tw[n1][k2] % q for k2 in range(N2)] for n1 in range(N1)]
        # stage 2: Xout[k1][k2] = Σ_n1 WC[k1][n1] · B[n1][k2]     (32-DFT)
        Xout = [[sum(self.WC[k1][n1] * B[n1][k2] for n1 in range(N1)) % q for k2 in range(N2)]
                for k1 in range(N1)]
        # reindex: X[k2 + 64 k1] = Xout[k1][k2]
        X = [0] * self.N
        for k1 in range(N1):
            for k2 in range(N2):
                X[k2 + N2 * k1] = Xout[k1][k2]
        return X


def schoolbook_negacyclic_ntt(x, q):
    """X[k] = Σ_n x[n]·ψ^n·ω^{nk}  (the reference: pointwise-mul here = negacyclic conv)."""
    N = len(x)
    g = prim_root(q)
    w = modpow(g, (q - 1) // N, q)
    psi = modpow(g, (q - 1) // (2 * N), q)
    out = []
    for k in range(N):
        acc = 0
        for n in range(N):
            acc += x[n] * modpow(psi, n, q) % q * modpow(w, n * k, q) % q
        out.append(acc % q)
    return out


def run_tests():
    import random
    rng = random.Random(7)
    # the tool's prime + the two PIR primes (all ≡ 1 mod 4096)
    primes = [167772161, 268369921, 268361729]
    ok = True
    for q in primes:
        assert is_prime(q) and (q - 1) % 4096 == 0
        ntt = NTT2048(q)
        # same primitive-root convention as schoolbook (both call prim_root(q))
        x = [rng.randrange(q) for _ in range(2048)]
        got = ntt.forward(x)
        exp = schoolbook_negacyclic_ntt(x, q)
        match = got == exp
        ok &= match
        print(f"  [{'PASS' if match else 'FAIL'}] negacyclic NTT-2048 (32,64) vs schoolbook, q={q}")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
