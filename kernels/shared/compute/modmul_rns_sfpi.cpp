// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// RNS (multi-modulus) Montgomery modmul, hybrid SFPI kernel.
//   out[i] = (a[i] * b[i]) mod q_v,  where v = which vector tile i belongs to.
//
// Same hybrid structure as modmul_sfpi_hybrid.cpp (non-multiply ops in custom
// SFPI register passes, multiplies via mul_int_tile) — but the modulus constants
// are PER-VECTOR: data is laid out vector-major, tiles_per_vector tiles per
// vector, so global_tile_id / tiles_per_vector selects the prime. The kernel
// re-FILLs q/q_inv/q_low/q_high from per-vector arrays each tile.
//
// Runtime args:
//   0 = n_tiles (this core)        1 = start_tile (global)
//   2 = tiles_per_vector           3 = num_vectors V
//   4        .. 4+V-1   : q[v]
//   4+V      .. 4+2V-1  : q_inv_32[v]
//   4+2V     .. 4+3V-1  : q_low16[v]
//   4+3V     .. 4+4V-1  : q_high16[v]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/mul_int_sfpu.h"

namespace {
constexpr auto FMT = DataFormat::UInt32;
inline void MUL(uint32_t a, uint32_t b, uint32_t o) { mul_int_tile_init<FMT>(); mul_int_tile<FMT>(a, b, o); }
inline void FILL(uint32_t d, uint32_t v) { fill_tile_init(); fill_tile_int<FMT>(d, v); }
}  // namespace

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;  // dst_reg slots per tile

inline void face_limbs(uint32_t a, uint32_t b, uint32_t aL, uint32_t aH, uint32_t bL, uint32_t bH) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt va = sfpi::dst_reg[a * NV + i];
        sfpi::vUInt vb = sfpi::dst_reg[b * NV + i];
        sfpi::dst_reg[aL * NV + i] = va & 0xFFFFu;
        sfpi::dst_reg[aH * NV + i] = va >> 16u;
        sfpi::dst_reg[bL * NV + i] = vb & 0xFFFFu;
        sfpi::dst_reg[bH * NV + i] = vb >> 16u;
    }
}
inline void face_lohi(uint32_t ll, uint32_t hh, uint32_t c1, uint32_t c2, uint32_t lo, uint32_t hi) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vll = sfpi::dst_reg[ll * NV + i];
        sfpi::vUInt vhh = sfpi::dst_reg[hh * NV + i];
        sfpi::vUInt cross = sfpi::vUInt(sfpi::dst_reg[c1 * NV + i]) + sfpi::vUInt(sfpi::dst_reg[c2 * NV + i]);
        sfpi::vUInt mid = (vll >> 16u) + (cross & 0xFFFFu);
        sfpi::dst_reg[lo * NV + i] = (vll & 0xFFFFu) | (mid << 16u);
        sfpi::dst_reg[hi * NV + i] = vhh + (cross >> 16u) + (mid >> 16u);
    }
}
inline void face_split(uint32_t t, uint32_t tL, uint32_t tH) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vt = sfpi::dst_reg[t * NV + i];
        sfpi::dst_reg[tL * NV + i] = vt & 0xFFFFu;
        sfpi::dst_reg[tH * NV + i] = vt >> 16u;
    }
}
inline void face_reduce(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL,
                        uint32_t hi, uint32_t qIdx, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vpMH = sfpi::dst_reg[pMH * NV + i];
        sfpi::vUInt vpML = sfpi::dst_reg[pML * NV + i];
        sfpi::vUInt vpLL = sfpi::dst_reg[pLL * NV + i];
        sfpi::vUInt mid_low = (vpMH & 0xFFFFu) + (vpML & 0xFFFFu) + (vpLL >> 16u);
        sfpi::vUInt mid_high = (vpMH >> 16u) + (vpML >> 16u) + (mid_low >> 16u);
        sfpi::vUInt t_final = sfpi::vUInt(sfpi::dst_reg[pHH * NV + i]) + mid_high;
        sfpi::vUInt q = sfpi::dst_reg[qIdx * NV + i];
        sfpi::vUInt b = sfpi::vUInt(sfpi::dst_reg[hi * NV + i]) + q - t_final;
        sfpi::vUInt bq = b - q;
        sfpi::vUInt res = bq;
        v_if(sfpi::reinterpret<sfpi::vInt>(bq) < 0) { res = b; }
        v_endif;
        sfpi::dst_reg[out * NV + i] = res;
    }
}
}  // namespace
#endif

inline void S_limbs(uint32_t a, uint32_t b, uint32_t aL, uint32_t aH, uint32_t bL, uint32_t bH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_limbs, a, b, aL, ckernel::VectorMode::RC, aH, bL, bH));
}
inline void S_lohi(uint32_t ll, uint32_t hh, uint32_t c1, uint32_t c2, uint32_t lo, uint32_t hi) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_lohi, ll, hh, c1, ckernel::VectorMode::RC, c2, lo, hi));
}
inline void S_split(uint32_t t, uint32_t tL, uint32_t tH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_split, t, tL, tH, ckernel::VectorMode::RC));
}
inline void S_reduce(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL,
                     uint32_t hi, uint32_t qIdx, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_reduce, pHH, pMH, pML, ckernel::VectorMode::RC, pLL, hi, qIdx, out));
}

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t tiles_per_vector = get_arg_val<uint32_t>(2);
    const uint32_t V = get_arg_val<uint32_t>(3);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in0, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        // Select this tile's prime: vector index = global_tile_id / tiles_per_vector.
        const uint32_t vec = (start_tile + t) / tiles_per_vector;
        const uint32_t Q = get_arg_val<uint32_t>(4 + vec);
        const uint32_t QINV = get_arg_val<uint32_t>(4 + V + vec);
        const uint32_t QLOW = get_arg_val<uint32_t>(4 + 2 * V + vec);
        const uint32_t QHIGH = get_arg_val<uint32_t>(4 + 3 * V + vec);

        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        copy_tile_init(cb_in0); copy_tile(cb_in0, 0, 0);  // d0 = a
        copy_tile_init(cb_in1); copy_tile(cb_in1, 0, 1);  // d1 = b

        S_limbs(0, 1, 2, 3, 4, 5);   // aL=d2, aH=d3, bL=d4, bH=d5
        MUL(2, 4, 0);                // ll  = aL*bL
        MUL(3, 5, 1);                // hh  = aH*bH
        MUL(2, 5, 6);                // c1  = aL*bH
        MUL(3, 4, 7);                // c2  = aH*bL
        S_lohi(0, 1, 6, 7, 2, 3);    // lo=d2, hi=d3

        FILL(0, QINV);
        MUL(2, 0, 2);                // t = lo*q_inv_32
        S_split(2, 4, 5);            // tL=d4, tH=d5

        FILL(0, QHIGH);
        FILL(1, QLOW);
        MUL(5, 0, 6);                // pHH = tH*q_high
        MUL(5, 1, 5);                // pMH = tH*q_low
        MUL(4, 0, 0);                // pML = tL*q_high
        MUL(4, 1, 1);                // pLL = tL*q_low
        FILL(2, Q);
        S_reduce(6, 5, 0, 1, 3, 2, 4);  // out = (a*b) mod q_v -> d4

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(4, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
