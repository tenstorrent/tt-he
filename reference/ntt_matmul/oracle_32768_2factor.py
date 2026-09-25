# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""NumPy oracle for the naive two-factor (128,256) NTT-32768.

Textbook four-step, NO NTT-level tiling: matmul -> twiddle -> matmul. Mirrors the
device baseline (ntt32768_2factor_dev.cpp) so each intermediate is a per-stage golden.

  A[n1][n2] = x[n1 + 128*n2]            (128 x 256)
  S  = A @ W256                         contract n2 (K=256)   -> S[n1][u]
  S' = S * Tb,  Tb[n1][u]=wN^{n1*u}     twiddle
  R  = W128 @ S'                        contract n1 (K=128)   -> R[v][u]
  out[u + 256*v] = R[v][u]

W256[a][b]=w256^{ab}, W128[a][b]=w128^{ab}; w256=wN^128, w128=wN^256, wN=ω_32768.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ntt"))
from ntt_matrix import nth_root_of_unity  # noqa: E402

N = 32768
N1 = 128   # rows / second contraction
N2 = 256   # cols / first contraction
DEFAULT_Q = 167772161


def _mm(a, b, q):
    return (np.asarray(a, dtype=object) @ np.asarray(b, dtype=object)) % q


def build_constants(q):
    w = nth_root_of_unity(N, q)
    w256 = pow(w, N // N2, q)
    w128 = pow(w, N // N1, q)
    W256 = np.array([[pow(w256, (a * b) % N2, q) for b in range(N2)] for a in range(N2)], dtype=object)
    W128 = np.array([[pow(w128, (a * b) % N1, q) for b in range(N1)] for a in range(N1)], dtype=object)
    Tb = np.array([[pow(w, (n1 * u) % N, q) for u in range(N2)] for n1 in range(N1)], dtype=object)
    return dict(w=w, W256=W256, W128=W128, Tb=Tb)


def run(x, q=DEFAULT_Q, return_stages=False):
    c = build_constants(q)
    A = np.empty((N1, N2), dtype=object)
    for n1 in range(N1):
        for n2 in range(N2):
            A[n1][n2] = int(x[n1 + N1 * n2])
    S = _mm(A, c["W256"], q)               # contract n2
    Sp = (S * c["Tb"]) % q                  # twiddle
    R = _mm(c["W128"], Sp, q)               # contract n1
    out = np.empty(N, dtype=object)
    for v in range(N1):
        for u in range(N2):
            out[u + N2 * v] = R[v][u]
    if return_stages:
        return out, dict(A=A, S=S, Sp=Sp, R=R, **c)
    return out


def verify(q=DEFAULT_Q, seeds=(0, 1, 7), n_spot=32):
    ok = True
    for seed in seeds:
        rng = np.random.default_rng(seed)
        x = rng.integers(0, q, size=N, dtype=np.int64)
        out = run(x, q)
        w = nth_root_of_unity(N, q)
        r2 = np.random.default_rng(1000 + seed)
        bad = 0
        for _ in range(n_spot):
            k = int(r2.integers(0, N))
            acc = 0
            for n in range(N):
                acc += int(x[n]) * pow(w, (n * k) % N, q)
            if acc % q != int(out[k]):
                bad += 1
        print(f"  seed={seed:3d}  spot-checked {n_spot} indices vs direct DFT: {'OK' if bad == 0 else f'{bad} BAD'}")
        ok &= (bad == 0)
    return ok


if __name__ == "__main__":
    print(f"oracle two-factor (128,256) NTT-{N}, q={DEFAULT_Q}")
    assert verify(), "oracle does NOT match the direct DFT"
    print("ORACLE MATCHES THE DIRECT DFT.")
