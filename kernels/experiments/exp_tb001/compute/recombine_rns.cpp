// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Per-column-modulus (RNS) recombine: like recombine_compute.cpp, but each tile
// COLUMN carries its own destination modulus p_j (for base conversion, column j =
// dest modulus). The modulus-dependent constants (q, q^{-1} mod 2^32, q_low,
// q_high, K14 = mont(2^14)) are supplied as 5 per-column TILES on c_1 instead of
// scalars — value at column c uses p_c's parameter — so the same Montgomery
// recombine runs with a different modulus per lane. Makes baseconv_matmul fully
// on-device (matrix engine does the combine, this does the per-p_j reduction).
//
//   out[c][j] = (C0 + C1<<7 + C2<<14 + C3<<21) mod p_j
// via one Montgomery reduction (hi14 = (C2+C3<<7)·2^14 mod p_j).
//
// c_0: 4 int32 plane tiles C0..C3.   c_1: 5 param tiles [K14,QINV,QHIGH,QLOW,Q].
// c_16: 1 output tile.   Runtime arg: 0 = n_out (groups of 4 planes).
// Plane bound assumption: low = C0+(C1<<7) < 1.61·p_j (holds for matmul planes < ~2^20).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/mul_int_sfpu.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr auto FMT = DataFormat::UInt32;
inline void MUL_INIT() { mul_int_tile_init<FMT>(); }
inline void MUL(uint32_t a, uint32_t b, uint32_t o) { mul_int_tile<FMT>(a, b, o); }
}  // namespace

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;
inline void face_build_hi(uint32_t C2, uint32_t C3, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++)
        sfpi::dst_reg[out * NV + i] = sfpi::vUInt(sfpi::dst_reg[C2 * NV + i]) + (sfpi::vUInt(sfpi::dst_reg[C3 * NV + i]) << 7u);
}
inline void face_limbs(uint32_t a, uint32_t b, uint32_t aL, uint32_t aH, uint32_t bL, uint32_t bH) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt va = sfpi::dst_reg[a * NV + i], vb = sfpi::dst_reg[b * NV + i];
        sfpi::dst_reg[aL * NV + i] = va & 0xFFFFu; sfpi::dst_reg[aH * NV + i] = va >> 16u;
        sfpi::dst_reg[bL * NV + i] = vb & 0xFFFFu; sfpi::dst_reg[bH * NV + i] = vb >> 16u;
    }
}
inline void face_lohi(uint32_t ll, uint32_t hh, uint32_t c1, uint32_t c2, uint32_t lo, uint32_t hi) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vll = sfpi::dst_reg[ll * NV + i], vhh = sfpi::dst_reg[hh * NV + i];
        sfpi::vUInt cross = sfpi::vUInt(sfpi::dst_reg[c1 * NV + i]) + sfpi::vUInt(sfpi::dst_reg[c2 * NV + i]);
        sfpi::vUInt mid = (vll >> 16u) + (cross & 0xFFFFu);
        sfpi::dst_reg[lo * NV + i] = (vll & 0xFFFFu) | (mid << 16u);
        sfpi::dst_reg[hi * NV + i] = vhh + (cross >> 16u) + (mid >> 16u);
    }
}
inline void face_split(uint32_t t, uint32_t tL, uint32_t tH) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vt = sfpi::dst_reg[t * NV + i];
        sfpi::dst_reg[tL * NV + i] = vt & 0xFFFFu; sfpi::dst_reg[tH * NV + i] = vt >> 16u;
    }
}
inline void face_reduce_fin(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL, uint32_t hi, uint32_t qIdx, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vpMH = sfpi::dst_reg[pMH * NV + i], vpML = sfpi::dst_reg[pML * NV + i], vpLL = sfpi::dst_reg[pLL * NV + i];
        sfpi::vUInt mid_low = (vpMH & 0xFFFFu) + (vpML & 0xFFFFu) + (vpLL >> 16u);
        sfpi::vUInt mid_high = (vpMH >> 16u) + (vpML >> 16u) + (mid_low >> 16u);
        sfpi::vUInt t_final = sfpi::vUInt(sfpi::dst_reg[pHH * NV + i]) + mid_high;
        sfpi::vUInt q = sfpi::dst_reg[qIdx * NV + i];
        sfpi::vUInt b = sfpi::vUInt(sfpi::dst_reg[hi * NV + i]) + q - t_final;
        sfpi::vUInt bq = b - q, res = bq;
        v_if(sfpi::reinterpret<sfpi::vInt>(bq) < 0) { res = b; } v_endif;
        sfpi::dst_reg[out * NV + i] = res;
    }
}
inline void face_final(uint32_t C0, uint32_t C1, uint32_t hi14, uint32_t qIdx, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt q = sfpi::dst_reg[qIdx * NV + i];
        sfpi::vUInt low = sfpi::vUInt(sfpi::dst_reg[C0 * NV + i]) + (sfpi::vUInt(sfpi::dst_reg[C1 * NV + i]) << 7u);
        sfpi::vUInt r = low + sfpi::vUInt(sfpi::dst_reg[hi14 * NV + i]);
        sfpi::vUInt rq = r - q, res = rq;
        v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; } v_endif;
        r = res; rq = r - q; res = rq;
        v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; } v_endif;
        sfpi::dst_reg[out * NV + i] = res;
    }
}
}  // namespace
#endif

inline void S_build_hi(uint32_t C2, uint32_t C3, uint32_t o) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_build_hi, C2, C3, o, ckernel::VectorMode::RC)); }
inline void S_limbs(uint32_t a, uint32_t b, uint32_t aL, uint32_t aH, uint32_t bL, uint32_t bH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_limbs, a, b, aL, ckernel::VectorMode::RC, aH, bL, bH)); }
inline void S_lohi(uint32_t ll, uint32_t hh, uint32_t c1, uint32_t c2, uint32_t lo, uint32_t hi) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_lohi, ll, hh, c1, ckernel::VectorMode::RC, c2, lo, hi)); }
inline void S_split(uint32_t t, uint32_t tL, uint32_t tH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_split, t, tL, tH, ckernel::VectorMode::RC)); }
inline void S_reduce_fin(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL, uint32_t hi, uint32_t qIdx, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_reduce_fin, pHH, pMH, pML, ckernel::VectorMode::RC, pLL, hi, qIdx, out)); }
inline void S_final(uint32_t C0, uint32_t C1, uint32_t hi14, uint32_t qIdx, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_final, C0, C1, hi14, ckernel::VectorMode::RC, qIdx, out)); }

void kernel_main() {
    const uint32_t n_out = get_arg_val<uint32_t>(0);

    constexpr auto cb_in = tt::CBIndex::c_0;    // 4 plane tiles C0..C3
    constexpr auto cb_par = tt::CBIndex::c_1;   // 5 per-column param tiles: K14,QINV,QHIGH,QLOW,Q
    constexpr auto cb_out = tt::CBIndex::c_16;
    // param tile indices in cb_par
    constexpr uint32_t PK14 = 0, PQINV = 1, PQHIGH = 2, PQLOW = 3, PQ = 4;

    unary_op_init_common(cb_in, cb_out);
    MUL_INIT();

    cb_wait_front(cb_par, 5);
    for (uint32_t t = 0; t < n_out; ++t) {
        cb_wait_front(cb_in, 4);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        // hi = C2 + (C3<<7) -> d0 ;  K14 (per-col) -> d1
        copy_tile_init(cb_in); copy_tile(cb_in, 2, 2); copy_tile(cb_in, 3, 3);
        S_build_hi(2, 3, 0);
        copy_tile_init(cb_par); copy_tile(cb_par, PK14, 1);

        // hi14 = (hi · K14) mod p_j   (per-column Montgomery)  -> d4
        S_limbs(0, 1, 2, 3, 4, 5);
        MUL(2, 4, 0); MUL(3, 5, 1); MUL(2, 5, 6); MUL(3, 4, 7);
        S_lohi(0, 1, 6, 7, 2, 3);
        copy_tile_init(cb_par); copy_tile(cb_par, PQINV, 0); MUL(2, 0, 2);
        S_split(2, 4, 5);
        copy_tile(cb_par, PQHIGH, 0); copy_tile(cb_par, PQLOW, 1);
        MUL(5, 0, 6); MUL(5, 1, 5); MUL(4, 0, 0); MUL(4, 1, 1);
        copy_tile(cb_par, PQ, 2);                       // d2 = q (per-col), persists for S_final
        S_reduce_fin(6, 5, 0, 1, 3, 2, 4);              // hi14 -> d4

        // R = (C0 + C1<<7 + hi14) mod p_j  -> d5
        copy_tile_init(cb_in); copy_tile(cb_in, 0, 0); copy_tile(cb_in, 1, 1);
        S_final(0, 1, 4, 2, 5);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(5, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_in, 4);
    }
    cb_pop_front(cb_par, 5);
}
