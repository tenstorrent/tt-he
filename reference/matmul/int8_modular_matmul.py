# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Modular matmul  C = (A @ B) mod q  emulated with an int8 x int8 -> uint32 kernel.

Setting
-------
We want C[i, j] = (sum_k A[i, k] * B[k, j]) mod q  with  q <= 2**31.
The only hardware primitive we assume is an int8 x int8 matmul that accumulates
into uint32 (8-bit inputs, 32-bit output).  One operand (B) is a known parameter
and may be preprocessed offline.

Idea (double byte-decomposition)
--------------------------------
Decompose the *dynamic* operand A into 4 little-endian bytes:
    A[i, k] = sum_p a[i, k, p] * 256**p           (p = 0..3, a in [0, 256))

Precompute the *parameter* B, folding byte-weight 256**p into it and reducing:
    Bp[k, p, j] = (256**p * B[k, j]) mod q        (still < 2**31)
and byte-decompose that as well:
    Bp[k, p, j] = sum_r bp[k, p, j, r] * 256**r   (r = 0..3, bp in [0, 256))

Then, working mod q,
    C[i, j] = sum_{k,p} a[i,k,p] * (256**p * B[k,j])
            = sum_{k,p} a[i,k,p] * Bp[k,p,j]
            = sum_r 256**r * ( sum_{k,p} a[i,k,p] * bp[k,p,j,r] )
                             \________ int8 x int8 -> uint32 matmul ________/

The 256**p weights never appear at runtime; they are baked into the parameter
through the mod-q precompute.  Only the 256**r recombination (shifts 0,8,16,24)
and a final mod q remain.
"""

import numpy as np


# ----------------------------------------------------------------------------
# Offline: convert the known parameter matrix (uint32, values in [0, q)) to the
# packed uint8 form  bp[k, p, j, r]  laid out as a 2-D (K*4, J*4) matrix.
#
# Independent numpy implementation of the byte-packing identity below.
# Output[k*4 + p, j*4 + r] = byte_r( (256**p * B[k, j]) mod q )
# ----------------------------------------------------------------------------
def modular_matrix_u32_to_u8(matrix: np.ndarray, modulus: int) -> np.ndarray:
    rows, cols = matrix.shape
    assert modulus <= 2**31
    matrix_u64 = matrix.astype(np.uint64)
    # 4 byte-shifted copies: B << (8*p) for p = 0..3, shape (4, rows, cols)
    shifted = np.array([matrix_u64 << np.uint64(8 * p) for p in range(4)], dtype=np.uint64)
    # interleave so the shift index p sits right after k -> rows become (k, p)
    shifted = shifted.transpose(1, 0, 2).reshape(rows * 4, cols)
    # reduce mod q (now < 2**31, fits in uint32)
    reduced = (shifted % modulus).astype(np.uint32)
    # split each uint32 into its 4 little-endian bytes along the column axis
    packed_u8 = reduced.view(np.uint8)  # shape (rows*4, cols*4)
    return packed_u8


# ----------------------------------------------------------------------------
# Online: decompose the dynamic operand A into bytes a[i, k, p].
# ----------------------------------------------------------------------------
def decompose_operand_u32_to_u8(matrix: np.ndarray) -> np.ndarray:
    rows, cols = matrix.shape
    bytes_u8 = matrix.astype(np.uint32).view(np.uint8)  # (rows, cols*4)
    return bytes_u8.reshape(rows, cols, 4)              # a[i, k, p]


# ----------------------------------------------------------------------------
# The int8 x int8 -> uint32 kernel + recombination, mirroring matmul_einsum_kernel.
#   lhs = a[i, k, p]        (I, K, 4)   uint8
#   rhs = bp[k, p, j, r]    (K, 4, J, 4) uint8
# Contract k, sum p, keep i, j, and the shift index r; then recombine over r.
# ----------------------------------------------------------------------------
def matmul_int8_kernel(a: np.ndarray, bp: np.ndarray, modulus: int) -> np.ndarray:
    # int8 x int8 matmul accumulating into uint32 (this is the "hardware" op)
    products = np.einsum("ikp,kpjr->ijr", a.astype(np.uint32), bp.astype(np.uint32))
    assert products.max() < 2**32, "uint32 accumulator overflow (reduce K or magnitudes)"

    # recombine the byte axis r with weights 256**r, then reduce mod q
    shift = np.array([0, 8, 16, 24], dtype=np.uint64)
    recombined = np.sum(products.astype(np.uint64) << shift, axis=-1)  # (I, J)
    return (recombined % modulus).astype(np.uint32)


def modular_matmul_via_int8(A: np.ndarray, B: np.ndarray, modulus: int) -> np.ndarray:
    """C = (A @ B) mod q, computed only through the int8 kernel above.

    A: (I, K) uint32, any value < 2**32.   B: (K, J) uint32, values in [0, q).
    """
    K, J = B.shape
    a = decompose_operand_u32_to_u8(A)                      # (I, K, 4)
    bp = modular_matrix_u32_to_u8(B, modulus)              # (K*4, J*4)
    bp = bp.reshape(K, 4, J, 4)                            # bp[k, p, j, r]
    return matmul_int8_kernel(a, bp, modulus)


def reference_modular_matmul(A: np.ndarray, B: np.ndarray, modulus: int) -> np.ndarray:
    """Ground truth using exact Python big-int arithmetic (object dtype)."""
    C = A.astype(object) @ B.astype(object)
    return (C % modulus).astype(np.uint64)


# ----------------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------------
def _test(I, K, J, q, seed):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, q, size=(I, K), dtype=np.uint64).astype(np.uint32)
    B = rng.integers(0, q, size=(K, J), dtype=np.uint64).astype(np.uint32)

    got = modular_matmul_via_int8(A, B, q).astype(np.uint64)
    expected = reference_modular_matmul(A, B, q)

    ok = np.array_equal(got, expected)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] I={I:4d} K={K:5d} J={J:4d} q={q:>11d}")
    if not ok:
        bad = np.argwhere(got != expected)[:5]
        for i, j in bad:
            print(f"   ({i},{j}) got={got[i, j]} expected={expected[i, j]}")
    return ok


if __name__ == "__main__":
    # K is bounded so the uint32 accumulator (<= 4*K*255*255) does not overflow:
    #   4 * K * 65025 < 2**32  =>  K < ~16511
    cases = [
        dict(I=8,   K=64,    J=8,   q=12289,        seed=0),   # NTT-friendly prime
        dict(I=16,  K=256,   J=16,  q=40961,        seed=1),
        dict(I=4,   K=1024,  J=4,   q=786433,       seed=2),
        dict(I=32,  K=512,   J=32,  q=2013265921,   seed=3),   # ~2**31, a real HE prime
        dict(I=2,   K=8192,  J=2,   q=2013265921,   seed=4),
        dict(I=1,   K=1,     J=1,   q=2**31,        seed=5),   # edge: max modulus
    ]
    all_ok = all(_test(**c) for c in cases)
    print("\nALL PASS" if all_ok else "\nSOME FAILED")
