# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Radix-2 butterfly Number Theoretic Transform (Cooley-Tukey).

This is the fast O(N log N) NTT, the finite-field counterpart of the radix-2 FFT.
It computes the same transform as the schoolbook matrix version in ntt_matrix.py

    X[k] = sum_{n=0}^{N-1} x[n] * w^{n*k}   (mod q)

but in log2(N) butterfly stages instead of one N x N matrix multiply. N must be a
power of two (and must divide q-1 so the N-th root of unity exists).

Algorithm: iterative decimation-in-time (DIT). We first permute the input into
bit-reversed order, then run log2(N) stages. Stage s (length m = 2, 4, ..., N)
combines pairs separated by m/2 using a "butterfly":

    u = a[j]
    t = w_m^j * a[j + m/2]          (twiddle multiply, mod q)
    a[j]       = u + t   (mod q)
    a[j + m/2] = u - t   (mod q)

where w_m is the primitive m-th root of unity. All arithmetic is in GF(q); a u32
product M*v < 2^56 fits in Python ints, so no overflow concerns here (numpy is
used only for the array container, not for batched multiply).

Reuses the prime, primitive-root, and root-of-unity helpers from ntt_matrix.
"""

import numpy as np

from ntt_matrix import DEFAULT_PRIME, _mod_pow, nth_root_of_unity


def _check_pow2(n):
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"butterfly NTT requires N to be a power of two, got N={n}")


def _bit_reverse(a):
    """Permute array a in place into bit-reversed index order.

    For N = 2^L, index i is sent to the integer formed by reversing its L bits.
    Done with the standard incremental reversed-counter trick, O(N).
    """
    n = a.shape[0]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]


def _transform(a, q, root):
    """In-place iterative radix-2 DIT butterfly using the given primitive root.

    `root` must be a primitive N-th root of unity (forward: w, inverse: w^{-1}).
    Does NOT apply the 1/N scaling -- the caller handles that for the inverse.
    """
    n = a.shape[0]
    _bit_reverse(a)

    m = 2
    while m <= n:
        # w_m = root^(N/m) is the primitive m-th root of unity for this stage.
        w_m = _mod_pow(int(root), n // m, q)
        half = m // 2
        for start in range(0, n, m):
            w = 1
            for j in range(half):
                u = int(a[start + j])
                t = (w * int(a[start + j + half])) % q
                a[start + j] = (u + t) % q
                a[start + j + half] = (u - t) % q       # Python % keeps it non-negative
                w = (w * w_m) % q
        m <<= 1
    return a


def ntt(x, q=DEFAULT_PRIME, w=None):
    """Forward butterfly NTT of vector x over GF(q). N must be a power of two.

    w : optional primitive N-th root of unity; derived from q if omitted.
    Returns a numpy int64 array of length N.
    """
    a = np.asarray(x, dtype=np.int64).copy() % q
    n = a.shape[0]
    _check_pow2(n)
    if w is None:
        w = nth_root_of_unity(n, q)
    return _transform(a, q, w)


def intt(X, q=DEFAULT_PRIME, w=None):
    """Inverse butterfly NTT:  x = N^{-1} * butterfly(X, w^{-1}) (mod q)."""
    a = np.asarray(X, dtype=np.int64).copy() % q
    n = a.shape[0]
    _check_pow2(n)
    if w is None:
        w = nth_root_of_unity(n, q)
    w_inv = _mod_pow(int(w), q - 2, q)        # w^{-1} via Fermat (q prime)
    _transform(a, q, w_inv)
    n_inv = _mod_pow(n, q - 2, q)             # N^{-1} mod q
    return (a * n_inv) % q


if __name__ == "__main__":
    import ntt_matrix

    q = DEFAULT_PRIME

    for N in (2, 4, 8, 16, 1024):
        rng = np.random.default_rng(N)
        x = rng.integers(0, q, size=N, dtype=np.int64)

        X = ntt(x, q)

        # Round-trip
        x_rt = intt(X, q)
        assert np.array_equal(x_rt, x % q), f"round-trip mismatch at N={N}"

        # Agreement with the O(N^2) matrix reference (shared root of unity)
        w = nth_root_of_unity(N, q)
        X_mat = ntt_matrix.ntt(x, q, w)
        assert np.array_equal(X, X_mat), f"butterfly disagrees with matrix NTT at N={N}"

        print(f"N={N:5d}: round-trip OK, matches matrix NTT")

    print("OK: all butterfly checks pass")
