// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// baseconv_fbc_wide — SFPU base conversion with NO Ld cap: constants read from the runtime-arg region, not stack arrays.
//
// MEASURED: lifts Ld from 8; Ls=32,Ld=32 needs 1186 args of the 4096 allowed.  docs/he/exp_tg018r.md
//
// This is an OPTIMISED variant promoted into the shared kernel tree. The kernel it derives
// from (kernels/experiments/exp_tb001/compute/baseconv_fbc_wide.cpp) is RETAINED and unchanged, so nothing that
// depends on the old behaviour moves. See kernels/OPTIMISED.md for the full set and the
// three techniques they share.
//
// Original header follows.
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: Apache-2.0
//
// WIDE baseconv_fbc for exp_Tg018r.  Identical arithmetic to baseconv_fbc.cpp; the ONLY change is
// that the Ls x Ld constant block is no longer copied into stack arrays but read from the runtime-arg
// region at the point of use.
//
// WHY. baseconv_fbc declares `uint32_t C2[16][12]` and friends on the compute RISC's stack, and its
// header explains that overflowing them silently scribbles that stack.  That -- not the dispatch
// budget -- is what capped exp_Tg018 at Ld = 8: tt-metal allows 4096 runtime args per Tensix kernel
// (max_runtime_args_tensix), and Ls=32, Ld=32 needs 1186.  Reading through get_arg_val costs nothing
// (the args live in L1 either way) and uses zero stack, so the whole (Ls, Ld) grid becomes reachable.
//
//   y_i   = (x_i . get_arg_val<uint32_t>(oC1 + i)) mod q_i
//   out_j = SUM_i (y_i . C2[i][j]) mod p_j              [approximate FBC, no eps correction]
//
// Runtime args, in order: LS, LD, C1[LS], C2[LS*LD], Q[LS], QINV[LS], P[LD], PINV[LD].

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
constexpr uint32_t NB = 4;
inline void MUL(uint32_t a, uint32_t b, uint32_t o) { mul_int_tile<FMT>(a, b, o); }
inline void FILL(uint32_t d, uint32_t v) { fill_tile_int<FMT>(d, v); }
}  // namespace

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;
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
// canonical [0,q): one conditional subtract on the lazy [0,2q) value
inline void face_reduce_fin(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL,
                            uint32_t hi, uint32_t qIdx, uint32_t out) {
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
// acc = (acc + add) mod q   (acc,add ∈ [0,q) ⇒ acc+add < 2q ⇒ one conditional subtract)
inline void face_modadd(uint32_t accS, uint32_t addS, uint32_t out, uint32_t Q) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt r = sfpi::vUInt(sfpi::dst_reg[accS * NV + i]) + sfpi::vUInt(sfpi::dst_reg[addS * NV + i]);
        sfpi::vUInt t = r - Q, res = t;
        v_if(sfpi::reinterpret<sfpi::vInt>(t) < 0) { res = r; } v_endif;
        sfpi::dst_reg[out * NV + i] = res;
    }
}
}  // namespace
#endif

inline void S_limbs(uint32_t a, uint32_t b, uint32_t aL, uint32_t aH, uint32_t bL, uint32_t bH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_limbs, a, b, aL, ckernel::VectorMode::RC, aH, bL, bH)); }
inline void S_lohi(uint32_t ll, uint32_t hh, uint32_t c1, uint32_t c2, uint32_t lo, uint32_t hi) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_lohi, ll, hh, c1, ckernel::VectorMode::RC, c2, lo, hi)); }
inline void S_split(uint32_t t, uint32_t tL, uint32_t tH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_split, t, tL, tH, ckernel::VectorMode::RC)); }
inline void S_reduce_fin(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL, uint32_t hi, uint32_t qIdx, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_reduce_fin, pHH, pMH, pML, ckernel::VectorMode::RC, pLL, hi, qIdx, out)); }
inline void S_modadd(uint32_t accS, uint32_t addS, uint32_t out, uint32_t Q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_modadd, accS, addS, out, ckernel::VectorMode::RC, Q)); }

// canonical Montgomery mulmod: a in d0, b(=mont const) in d1 -> canonical result in d4. Scratch d0..d7.
inline void MULMOD(uint32_t Q, uint32_t QINV) {
    S_limbs(0, 1, 2, 3, 4, 5);
    MUL(2, 4, 0); MUL(3, 5, 1); MUL(2, 5, 6); MUL(3, 4, 7);
    S_lohi(0, 1, 6, 7, 2, 3);
    FILL(0, QINV); MUL(2, 0, 2);
    S_split(2, 4, 5);
    FILL(0, Q >> 16); FILL(1, Q & 0xFFFFu);
    MUL(5, 0, 6); MUL(5, 1, 5); MUL(4, 0, 0); MUL(4, 1, 1);
    FILL(2, Q);
    S_reduce_fin(6, 5, 0, 1, 3, 2, 4);
}

void kernel_main() {
    const uint32_t LS = get_arg_val<uint32_t>(0);
    const uint32_t LD = get_arg_val<uint32_t>(1);
    // Offsets into the runtime-arg region. No stack arrays -- that is the whole point of this variant.
    const uint32_t oC1 = 2, oC2 = oC1 + LS, oQ = oC2 + LS * LD, oQI = oQ + LS,
                   oP = oQI + LS, oPI = oP + LD;
    constexpr auto cb_x = tt::CBIndex::c_0;     // input residues x_i (LS tiles)
    constexpr auto cb_y = tt::CBIndex::c_2;     // intermediate y_i (LS, L1-resident)
    constexpr auto cb_p = tt::CBIndex::c_3;     // intermediate products per j (LS, L1)
    constexpr auto cb_out = tt::CBIndex::c_16;  // output residues (LD)

    unary_op_init_common(cb_x, cb_out);
    mul_int_tile_init<FMT>();
    fill_tile_init();

    

    // ===== phase 1: y_i = (x_i · C1[i]) mod q_i =====
    { 
#ifdef TB001_PHASE_ZONES
    DeviceZoneScopedN("fbc_phase1");
#endif
    cb_wait_front(cb_x, LS);
    cb_reserve_back(cb_y, LS);
    for (uint32_t i = 0; i < LS; ++i) {
        tile_regs_acquire();
        copy_tile_init(cb_x); copy_tile(cb_x, i, 0);   // x_i -> d0
        FILL(1, get_arg_val<uint32_t>(oC1 + i));                                 // b = mont(qhat_inv_i) -> d1
        MULMOD(get_arg_val<uint32_t>(oQ + i), get_arg_val<uint32_t>(oQI + i));                          // canonical y_i -> d4
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(4, cb_y, i);
        tile_regs_release();
    }
    cb_push_back(cb_y, LS);
    cb_pop_front(cb_x, LS);
    }

    // ===== phase 2: out_j = Σ_i (y_i · C2[i][j]) mod p_j =====
    // (accumulator can't survive MULMOD in DST — only 8 int32 dst tiles — so the 4
    //  products go to an L1 CB and a separate modadd window sums them.)
    { 
#ifdef TB001_PHASE_ZONES
    DeviceZoneScopedN("fbc_phase2");
#endif
    cb_wait_front(cb_y, LS);
    cb_reserve_back(cb_out, LD);
    for (uint32_t j = 0; j < LD; ++j) {
        // LS modmuls -> products in c_p
        cb_reserve_back(cb_p, LS);
        for (uint32_t i = 0; i < LS; ++i) {
            tile_regs_acquire();
            copy_tile_init(cb_y); copy_tile(cb_y, i, 0);  // y_i -> d0
            FILL(1, get_arg_val<uint32_t>(oC2 + i * LD + j));                             // b = mont(qhat_i mod p_j) -> d1
            MULMOD(get_arg_val<uint32_t>(oP + j), get_arg_val<uint32_t>(oPI + j));                          // canonical prod -> d4
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(4, cb_p, i);
            tile_regs_release();
        }
        cb_push_back(cb_p, LS);

        // sum the LS products mod p_j
        cb_wait_front(cb_p, LS);
        tile_regs_acquire();
        copy_tile_init(cb_p); copy_tile(cb_p, 0, 0);       // acc = prod_0 -> d0
        for (uint32_t i = 1; i < LS; ++i) {
            copy_tile(cb_p, i, 1);                          // prod_i -> d1
            S_modadd(0, 1, 0, get_arg_val<uint32_t>(oP + j));                        // acc = (acc + prod_i) mod p_j
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, j);
        tile_regs_release();
        cb_pop_front(cb_p, LS);
    }
    cb_push_back(cb_out, LD);
    cb_pop_front(cb_y, LS);
    }
}
