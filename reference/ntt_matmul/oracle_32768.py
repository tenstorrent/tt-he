# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""NumPy oracle for the (32,32,32) on-device NTT-32768.

This mirrors the HAND-OPTIMIZED device design in docs/matmul/history/NTT_32768_3FACTOR_PLAN.md
(one all-to-all: passes 1+2 folded into an on-tile NTT-1024), NOT the IR's uniform
3-pass form (which uses two all-to-alls). Because it follows the device stages exactly,
every intermediate it returns is a per-stage golden you can diff against a device buffer.

Layout: N=32768 values as 32 tiles of 32x32. Global index n = n1 + 32*n2 + 1024*n3.
  tile n1 holds  Xin[n1][n3][n2] = x[n1 + 32*n2 + 1024*n3]   (row=n3, col=n2)

Stages (all exact, mod q):
  B   = W @ Xin                 per tile, contract n3 (rows)         -> B[n1][s][n2]
  Ba  = B  * Ta                 twiddle a, Ta[s][n2]=w1024^{n2*s}    -> Ba[n1][s][n2]
  C   = Ba @ W                  per tile, contract n2 (cols)         -> C[n1][s][t]
  Cb  = C  * Tb                 twiddle b, Tb[n1][s][t]=wN^{n1*(s+32t)}
  D   = all_to_all(Cb)          corner turn n1<->s                   -> D[s][n1][t]
  E   = W @ D                   per tile, contract n1 (rows)         -> E[s][v][t]
  out[s + 32*t + 1024*v] = E[s][v][t]

B,Ba,C together == the existing NTT-1024 run on each tile (W@ , twiddle w1024 , @W).
W is the 32-pt NTT matrix W[a][b]=w32^{a*b}; w32=wN^{1024}, w1024=wN^{32}, wN=ω_32768.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ntt"))
from ntt_matrix import nth_root_of_unity, ntt as golden_ntt  # noqa: E402

T = 32
N = 32768
DEFAULT_Q = 167772161


def _matmul_mod(a, b, q):
    """Exact 32x32 @ 32x32 mod q via python-int object arrays."""
    return (np.asarray(a, dtype=object) @ np.asarray(b, dtype=object)) % q


def build_constants(q):
    w = nth_root_of_unity(N, q)            # ω_N
    w32 = pow(w, N // T, q)                # ω_32
    w1024 = pow(w, N // 1024, q)           # ω_1024
    W = np.array([[pow(w32, (a * b) % T, q) for b in range(T)] for a in range(T)], dtype=object)
    Ta = np.array([[pow(w1024, (n2 * s) % 1024, q) for n2 in range(T)] for s in range(T)], dtype=object)
    # Tb depends on the tile index n1:  Tb[n1][s][t] = wN^{ n1*(s+32t) mod N }
    Tb = np.array([[[pow(w, (n1 * (s + T * t)) % N, q) for t in range(T)]
                    for s in range(T)] for n1 in range(T)], dtype=object)
    return dict(w=w, w32=w32, w1024=w1024, W=W, Ta=Ta, Tb=Tb)


def cube_from_x(x):
    """x[n] -> Xin[n1][n3][n2] = x[n1 + 32*n2 + 1024*n3]  (row=n3, col=n2)."""
    cube = np.empty((T, T, T), dtype=object)
    for n1 in range(T):
        for n3 in range(T):
            for n2 in range(T):
                cube[n1][n3][n2] = int(x[n1 + T * n2 + 1024 * n3])
    return cube


def run(x, q=DEFAULT_Q, return_stages=False):
    """Compute the NTT via the device's stages. Returns the length-N result (and stages)."""
    c = build_constants(q)
    W, Ta, Tb = c["W"], c["Ta"], c["Tb"]
    Xin = cube_from_x(x)

    B = np.empty((T, T, T), dtype=object)     # B[n1][s][n2]
    Ba = np.empty((T, T, T), dtype=object)
    C = np.empty((T, T, T), dtype=object)     # C[n1][s][t]  (== NTT-1024 output H per tile)
    Cb = np.empty((T, T, T), dtype=object)
    for n1 in range(T):
        B[n1] = _matmul_mod(W, Xin[n1], q)          # contract n3 (rows)
        Ba[n1] = (B[n1] * Ta) % q                   # twiddle a (w1024)
        C[n1] = _matmul_mod(Ba[n1], W, q)           # contract n2 (cols)
        Cb[n1] = (C[n1] * Tb[n1]) % q               # twiddle b (wN)

    # all-to-all corner turn: D[s][n1][t] = Cb[n1][s][t]
    D = np.empty((T, T, T), dtype=object)           # D[s][n1][t]
    for s in range(T):
        for n1 in range(T):
            for t in range(T):
                D[s][n1][t] = Cb[n1][s][t]

    E = np.empty((T, T, T), dtype=object)           # E[s][v][t]
    for s in range(T):
        E[s] = _matmul_mod(W, D[s], q)              # contract n1 (rows)

    out = np.empty(N, dtype=object)
    for s in range(T):
        for v in range(T):
            for t in range(T):
                out[s + T * t + 1024 * v] = E[s][v][t]

    if return_stages:
        return out, dict(Xin=Xin, B=B, Ba=Ba, C=C, Cb=Cb, D=D, E=E, **c)
    return out


def verify(q=DEFAULT_Q, seeds=(0, 1, 7, 42)):
    ok = True
    for seed in seeds:
        x = np.random.default_rng(seed).integers(0, q, size=N, dtype=np.int64)
        got = run(x, q)
        gold = golden_ntt(x, q, nth_root_of_unity(N, q)) % q
        mism = int(np.sum(got.astype(object) % q != gold.astype(object)))
        print(f"  seed={seed:3d}  mismatches={mism}/{N}  {'OK' if mism == 0 else 'FAIL'}")
        ok &= (mism == 0)
    return ok


if __name__ == "__main__":
    print(f"oracle (32,32,32) NTT-{N}, q={DEFAULT_Q}: device-faithful 3-pass, 1 all-to-all")
    print("vs direct O(N^2) golden (ntt_matrix.ntt):")
    assert verify(), "oracle does NOT match the direct DFT"
    print("ORACLE MATCHES THE DIRECT DFT.")
