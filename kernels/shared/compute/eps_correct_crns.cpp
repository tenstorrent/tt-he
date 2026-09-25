// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// CRNS exact-FBC finish stage (P3): compute the CRT overflow ε by a FIXED-POINT
// truncated MAC on the SFPU (no sentinel modulus, no Montgomery, no wide accumulator),
// then apply the exact correction to the packed matmul-combine output.
//
//   ε[cp][col] = ( Σ_i (y_i >> d)·gs_i  +  off ) >> sh        (small MAC, all < 2^31)
//   out[cp][col] = ( combine[cp][col] − (ε·(Q mod p_j) mod p_j) ) mod p_j     (plain domain)
//
// y_i are host-supplied per-source-modulus tiles in the packed layout (value replicated
// across the L dest columns of each coeff-block), so ε lands in the SAME [row cp, col]
// layout as the combine tile → the correction is fully element-wise, no relayout.
// gs_i = round(2^32 / q_i) (u=32), d/sh/off baked from the exactness analysis; valid when
// the input is bounded a full limb below Q (the BFV rescale case).
//
// c_0: L y_i tiles (per super-block).   c_1: 2 per-column param tiles [QMODP, P].
// c_2: 1 combine tile (plain (x+εQ) mod p_j).   c_16: 1 exact output tile.
// Runtime args: 0=nsb, 1=L, 2=d, 3=sh, 4=off, 5..5+L-1 = gs_i.

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
inline void MUL(uint32_t a, uint32_t b, uint32_t o) { mul_int_tile<FMT>(a, b, o); }
inline void FILL(uint32_t d, uint32_t v) { fill_tile_int<FMT>(d, v); }
}  // namespace

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;
// dst[out] = (dst[a] >> sh)                (element-wise right shift)
inline void face_shr(uint32_t a, uint32_t out, uint32_t sh) {
    for (uint32_t i = 0; i < 8; i++)
        sfpi::dst_reg[out * NV + i] = sfpi::vUInt(sfpi::dst_reg[a * NV + i]) >> sh;
}
// dst[acc] = dst[acc] + dst[add]           (plain add, no reduction; stays < 2^31)
inline void face_add(uint32_t acc, uint32_t add, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++)
        sfpi::dst_reg[out * NV + i] = sfpi::vUInt(sfpi::dst_reg[acc * NV + i]) + sfpi::vUInt(sfpi::dst_reg[add * NV + i]);
}
// dst[out] = (dst[a] + off) >> sh
inline void face_addshr(uint32_t a, uint32_t out, uint32_t off, uint32_t sh) {
    for (uint32_t i = 0; i < 8; i++)
        sfpi::dst_reg[out * NV + i] = (sfpi::vUInt(sfpi::dst_reg[a * NV + i]) + off) >> sh;
}
// t mod p via up-to-8 conditional subtracts (t = ε·(Q mod p) < 8·p, ε<L≤8).  t in [a], p in [pp].
inline void face_modreduce(uint32_t a, uint32_t pp, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt t = sfpi::dst_reg[a * NV + i];
        sfpi::vUInt p = sfpi::dst_reg[pp * NV + i];
        for (uint32_t k = 0; k < 8; ++k) {
            sfpi::vUInt tp = t - p;
            v_if(sfpi::reinterpret<sfpi::vInt>(tp) >= 0) { t = tp; } v_endif;
        }
        sfpi::dst_reg[out * NV + i] = t;
    }
}
// out = (a − b) mod p   (a,b ∈ [0,p) ⇒ one conditional add)
inline void face_modsub(uint32_t a, uint32_t b, uint32_t pp, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt r = sfpi::vUInt(sfpi::dst_reg[a * NV + i]) - sfpi::vUInt(sfpi::dst_reg[b * NV + i]);
        sfpi::vUInt res = r;
        v_if(sfpi::reinterpret<sfpi::vInt>(r) < 0) { res = r + sfpi::vUInt(sfpi::dst_reg[pp * NV + i]); } v_endif;
        sfpi::dst_reg[out * NV + i] = res;
    }
}
}  // namespace
#endif

inline void S_shr(uint32_t a, uint32_t out, uint32_t sh) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_shr, a, out, sh, ckernel::VectorMode::RC)); }
inline void S_add(uint32_t acc, uint32_t add, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_add, acc, add, out, ckernel::VectorMode::RC)); }
inline void S_addshr(uint32_t a, uint32_t out, uint32_t off, uint32_t sh) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_addshr, a, out, off, ckernel::VectorMode::RC, sh)); }
inline void S_modreduce(uint32_t a, uint32_t pp, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_modreduce, a, pp, out, ckernel::VectorMode::RC)); }
inline void S_modsub(uint32_t a, uint32_t b, uint32_t pp, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_modsub, a, b, pp, ckernel::VectorMode::RC, out)); }

void kernel_main() {
    const uint32_t nsb = get_arg_val<uint32_t>(0);
    const uint32_t L   = get_arg_val<uint32_t>(1);
    const uint32_t d   = get_arg_val<uint32_t>(2);
    const uint32_t sh  = get_arg_val<uint32_t>(3);
    const uint32_t off = get_arg_val<uint32_t>(4);
    uint32_t gs[8];
    for (uint32_t i = 0; i < L; ++i) gs[i] = get_arg_val<uint32_t>(5 + i);

    constexpr auto cb_y   = tt::CBIndex::c_0;   // L y_i tiles per super-block
    constexpr auto cb_par = tt::CBIndex::c_1;   // [QMODP, P] per-column param tiles
    constexpr auto cb_cmb = tt::CBIndex::c_2;   // combine (x+εQ) mod p_j
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t PQMODP = 0, PP = 1;

    unary_op_init_common(cb_y, cb_out);
    mul_int_tile_init<FMT>();
    fill_tile_init();

    cb_wait_front(cb_par, 2);
    DeviceZoneScopedN("epscorr_total");
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        cb_wait_front(cb_y, L);
        cb_wait_front(cb_cmb, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        // ---- ε = ( Σ_i (y_i>>d)·gs_i + off ) >> sh ----   acc in d0
        // first term
        copy_tile_init(cb_y); copy_tile(cb_y, 0, 1);        // y_0 -> d1
        S_shr(1, 1, d);                                     // y_0>>d
        FILL(2, gs[0]); MUL(1, 2, 0);                       // d0 = (y_0>>d)*gs_0
        for (uint32_t i = 1; i < L; ++i) {
            copy_tile(cb_y, i, 1); S_shr(1, 1, d);
            FILL(2, gs[i]); MUL(1, 2, 3);                   // d3 = (y_i>>d)*gs_i
            S_add(0, 3, 0);                                 // acc += d3
        }
        S_addshr(0, 0, off, sh);                            // ε -> d0
        // ---- corr = (ε·(Q mod p_j)) mod p_j ----   d4
        copy_tile_init(cb_par); copy_tile(cb_par, PQMODP, 1); copy_tile(cb_par, PP, 2);
        MUL(0, 1, 4);                                        // corr_raw = ε·(Q mod p_j)  (<2^31, exact)
        S_modreduce(4, 2, 4);                               // corr = corr_raw mod p_j -> d4
        // ---- out = (combine − corr) mod p_j ----   d5
        copy_tile_init(cb_cmb); copy_tile(cb_cmb, 0, 5);    // combine -> d5
        S_modsub(5, 4, 2, 5);                               // (combine-corr) mod p -> d5

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(5, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_y, L);
        cb_pop_front(cb_cmb, 1);
    }
    cb_pop_front(cb_par, 2);
}
