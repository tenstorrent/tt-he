# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Matrix-based Number Theoretic Transform (NTT).

The NTT is the finite-field analogue of the DFT. Over the prime field GF(q) it
replaces the complex root of unity e^{-2*pi*i/N} with a primitive N-th root of
unity w in GF(q). The transform of a length-N vector x is

    X[k] = sum_{n=0}^{N-1} x[n] * w^{n*k}   (mod q),    k = 0 .. N-1

which is exactly a matrix-vector product  X = W @ x (mod q)  with the twiddle
matrix  W[k, n] = w^{(n*k) mod N} (mod q).

This module builds that twiddle matrix explicitly and applies it with numpy. It
is the O(N^2) "schoolbook" reference — slow next to a radix-2 butterfly NTT, but
trivially correct and handy for validating the device kernels in this repo.

Default prime is q = 167772161 = 5 * 2^25 + 1, an NTT-friendly 28-bit prime
(q - 1 = 2^25 * 5, so it supports power-of-two NTT sizes up to 2^25). Note: the
201326593 value used by the vector_modmul kernels is composite (13 * 15486661) --
fine for Montgomery modmul, which works for any odd modulus, but not a valid field
for the NTT, which needs a prime.
"""

import numpy as np

# 28-bit NTT-friendly prime: q - 1 = 2^25 * 5, so it supports NTT sizes up to 2^25.
DEFAULT_PRIME = 167772161


def _mod_pow(base, exp, mod):
    """Fast modular exponentiation: base**exp % mod (pure-Python big ints)."""
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result


def _factorize(n):
    """Return the set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def find_primitive_root(q):
    """Find a primitive root (generator) of the multiplicative group of GF(q)."""
    phi = q - 1
    factors = _factorize(phi)
    for g in range(2, q):
        if all(_mod_pow(g, phi // p, q) != 1 for p in factors):
            return g
    raise ValueError(f"no primitive root found for q={q}")


def nth_root_of_unity(n, q=DEFAULT_PRIME):
    """Return a primitive N-th root of unity w in GF(q).

    Requires N | (q - 1). w satisfies w^N == 1 and w^k != 1 for 0 < k < N.
    """
    if (q - 1) % n != 0:
        raise ValueError(f"N={n} must divide q-1={q - 1} for an N-th root to exist")
    g = find_primitive_root(q)
    return _mod_pow(g, (q - 1) // n, q)


def twiddle_matrix(n, w, q):
    """Build the N x N forward twiddle matrix  W[k, n] = w^{(k*n) mod N} mod q.

    Powers of w are precomputed once (w^0 .. w^{N-1}) and indexed by the outer
    product (k*n) mod N, so the whole matrix costs O(N^2) lookups, no pow calls.
    """
    powers = np.empty(n, dtype=np.int64)
    acc = 1
    for i in range(n):
        powers[i] = acc
        acc = (acc * w) % q
    idx = np.arange(n, dtype=np.int64)
    exponents = np.outer(idx, idx) % n          # (k*n) mod N
    return powers[exponents]                     # W[k, n] = w^{(k*n) mod N}


def ntt(x, q=DEFAULT_PRIME, w=None):
    """Forward NTT of vector x over GF(q):  X = W @ x (mod q).

    x : 1-D sequence of length N (N must divide q-1).
    w : optional primitive N-th root of unity; derived from q if omitted.
    Returns a numpy int64 array of length N.
    """
    x = np.asarray(x, dtype=np.int64) % q
    n = x.shape[0]
    if w is None:
        w = nth_root_of_unity(n, q)
    W = twiddle_matrix(n, w, q)
    # Reduce per term before summing to keep the int64 accumulator from overflowing
    # (q < 2^28, so each product < 2^56; we sum N of them -> stay modular).
    return _matvec_mod(W, x, q)


def intt(X, q=DEFAULT_PRIME, w=None):
    """Inverse NTT:  x = N^{-1} * (W^{-1} @ X) (mod q).

    The inverse transform uses the matrix of inverse twiddles W^{-1}[k,n] =
    w^{-(k*n) mod N} and a final scaling by the modular inverse of N.
    """
    X = np.asarray(X, dtype=np.int64) % q
    n = X.shape[0]
    if w is None:
        w = nth_root_of_unity(n, q)
    w_inv = _mod_pow(int(w), q - 2, q)           # w^{-1} via Fermat (q prime)
    Winv = twiddle_matrix(n, w_inv, q)
    n_inv = _mod_pow(n, q - 2, q)                 # N^{-1} mod q
    y = _matvec_mod(Winv, X, q)
    return (y * n_inv) % q


def _matvec_mod(M, v, q):
    """Modular matrix-vector product (M @ v) mod q without int64 overflow.

    q < 2^28 so each product M[k,n]*v[n] < 2^56; accumulating N of them would
    overflow int64 for large N, so we reduce the running sum row-by-row.
    """
    n = v.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for k in range(n):
        # Per-row dot product, reduced mod q to stay well inside int64.
        out[k] = int(np.sum((M[k] * v) % q) % q)
    return out


if __name__ == "__main__":
    q = DEFAULT_PRIME
    N = 16

    rng = np.random.default_rng(0)
    x = rng.integers(0, q, size=N, dtype=np.int64)

    w = nth_root_of_unity(N, q)
    print(f"q = {q}, N = {N}")
    print(f"primitive N-th root of unity w = {w}  (w^N mod q = {_mod_pow(int(w), N, q)})")
    print(f"input  x = {x.tolist()}")

    X = ntt(x, q)
    print(f"NTT(x)   = {X.tolist()}")

    # Round-trip: INTT(NTT(x)) == x
    x_rt = intt(X, q)
    print(f"INTT(NTT(x)) = {x_rt.tolist()}")
    assert np.array_equal(x_rt, x % q), "round-trip mismatch"

    # Cross-check forward NTT against the direct definition sum x[n] w^{nk}.
    X_ref = np.array(
        [sum(int(x[nn]) * _mod_pow(int(w), nn * k, q) for nn in range(N)) % q
         for k in range(N)],
        dtype=np.int64,
    )
    assert np.array_equal(X, X_ref), "forward NTT disagrees with definition"

    print("OK: round-trip and direct-definition checks pass")
