// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// EXACT base conversion x mod Q -> x mod p_j via Shenoy-Kumaresan, on the CENTERED representative.
//
// The fast base conversion (baseconv_fbc.cpp) computes (x + eps*Q) mod p_j and drops eps -- correct
// for CKKS, where the error is absorbed into the noise, and wrong for BFV, where an eps*Q error in a
// lift becomes an eps*t error in the plaintext. This kernel recovers eps EXACTLY, using a redundant
// residue mod a sentinel prime m_sk carried alongside the input:
//
//   A_sk = (Sum_i y_i * (qhat_i mod m_sk)) mod m_sk  =  (x + eps*Q) mod m_sk    <- carries the overflow
//   x_sk =                                                     x_c mod m_sk    <- carried with the ct
//   eps  = (A_sk - x_sk) * Q^-1  mod m_sk
//
// eps is MEASURED, not estimated, so it is exact for any input range -- which the lift needs, since a
// ciphertext coefficient is uniform on all of [0,Q). The fixed-point (CRNS) alternative infers eps
// from the magnitude of the sum and so needs precision ~1/Q; see docs/he/exp_tb001b.md.
//
// CENTERING FALLS OUT FREE. BFV needs the representative in [-Q/2, Q/2), SK naturally returns the one
// in [0,Q), and RNS has no magnitude comparison to tell them apart. But if the sentinel reports the
// CENTERED value x_c = x - w*Q, then A_sk - x_sk = (eps + w)*Q, so the recovered correction is eps + w
// and one subtraction removes the CRT overflow and the centering wrap together -- they are the same
// kind of quantity. No offset trick, no sign bit, no second accumulator. Verified 0/147456 against an
// exact big-integer reference (TB001B_SKLIFT=1).
//
// Phases 1 and 2 are baseconv_fbc's, unchanged, including its width fix: LS and LD are the ACTUAL
// widths and each loop runs to its own bound. The sentinel is simply one more destination column, so
// phase 2 runs over LD+1. Steps 3 and 5 are the new epilogue.
//
//   c_0   LS source tiles, then x_sk at index LS      c_2   y_i          c_3   per-destination products
//   c_4   A_j for LD+1 destinations                   c_5   eps          c_16  LD outputs
//   Runtime args: LS, LD, C1[LS], C2[LS*(LD+1)], Q[LS], QINV[LS], P[LD+1], PINV[LD+1], G[LD], QINVSK
//     where G[j] = mont(Q mod p_j) and QINVSK = mont(Q^-1 mod m_sk); index LD of P/PINV/C2 is m_sk.
//
// Fed by dataflow/reader_gather.cpp (n = LS+1) and drained by dataflow/writer_scatter.cpp (n_out = LD).
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
inline void face_modsub(uint32_t aS, uint32_t bS, uint32_t out, uint32_t Q) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vInt va = sfpi::reinterpret<sfpi::vInt>(sfpi::vUInt(sfpi::dst_reg[aS * NV + i]));
        sfpi::vInt vb = sfpi::reinterpret<sfpi::vInt>(sfpi::vUInt(sfpi::dst_reg[bS * NV + i]));
        sfpi::vInt d = va - vb;                 // both < 2^28, so this cannot overflow int32
        sfpi::vInt r = d;
        v_if(d < 0) { r = d + (int)Q; }
        v_endif;
        sfpi::dst_reg[out * NV + i] = sfpi::reinterpret<sfpi::vUInt>(r);
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
inline void S_modsub(uint32_t aS, uint32_t bS, uint32_t out, uint32_t Q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_modsub, aS, bS, out, ckernel::VectorMode::RC, Q));
}
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
    const uint32_t LD = get_arg_val<uint32_t>(1);      // REAL destinations; m_sk is one more
    const uint32_t LDT = LD + 1;                        // total destination columns

    // Sized for the widest caller: the rescale's second phase is S -> Q, so Ls = 9 (see the note in
    // baseconv_fbc.cpp -- these bounds are load-bearing, not cosmetic).
    uint32_t C1[10], C2[10][12], Q[10], QINV[10], P[12], PINV[12], G[12], QINVSK;
    uint32_t a = 2;
    for (uint32_t i = 0; i < LS; ++i) C1[i] = get_arg_val<uint32_t>(a++);
    for (uint32_t i = 0; i < LS; ++i) for (uint32_t j = 0; j < LDT; ++j) C2[i][j] = get_arg_val<uint32_t>(a++);
    for (uint32_t i = 0; i < LS; ++i) Q[i]    = get_arg_val<uint32_t>(a++);
    for (uint32_t i = 0; i < LS; ++i) QINV[i] = get_arg_val<uint32_t>(a++);
    for (uint32_t j = 0; j < LDT; ++j) P[j]    = get_arg_val<uint32_t>(a++);
    for (uint32_t j = 0; j < LDT; ++j) PINV[j] = get_arg_val<uint32_t>(a++);
    for (uint32_t j = 0; j < LD;  ++j) G[j]    = get_arg_val<uint32_t>(a++);
    QINVSK = get_arg_val<uint32_t>(a++);

    constexpr auto cb_x = tt::CBIndex::c_0;     // LS sources, then x_sk at index LS
    constexpr auto cb_y = tt::CBIndex::c_2;
    constexpr auto cb_p = tt::CBIndex::c_3;
    constexpr auto cb_a = tt::CBIndex::c_4;     // A_j, LD+1 of them
    constexpr auto cb_e = tt::CBIndex::c_5;     // eps
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_x, cb_out);
    mul_int_tile_init<FMT>();
    fill_tile_init();

    DeviceZoneScopedN("bconv_sk");

    // ===== phase 1: y_i = (x_i * C1[i]) mod q_i  =====
    cb_wait_front(cb_x, LS + 1);
    cb_reserve_back(cb_y, LS);
    for (uint32_t i = 0; i < LS; ++i) {
        tile_regs_acquire();
        copy_tile_init(cb_x); copy_tile(cb_x, i, 0);
        FILL(1, C1[i]);
        MULMOD(Q[i], QINV[i]);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(4, cb_y, i);
        tile_regs_release();
    }
    cb_push_back(cb_y, LS);

    // ===== phase 2: A_j = sum_i (y_i * C2[i][j]) mod p_j, over LD+1 destinations =====
    cb_wait_front(cb_y, LS);
    cb_reserve_back(cb_a, LDT);
    for (uint32_t j = 0; j < LDT; ++j) {
        cb_reserve_back(cb_p, LS);
        for (uint32_t i = 0; i < LS; ++i) {
            tile_regs_acquire();
            copy_tile_init(cb_y); copy_tile(cb_y, i, 0);
            FILL(1, C2[i][j]);
            MULMOD(P[j], PINV[j]);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(4, cb_p, i);
            tile_regs_release();
        }
        cb_push_back(cb_p, LS);

        cb_wait_front(cb_p, LS);
        tile_regs_acquire();
        copy_tile_init(cb_p); copy_tile(cb_p, 0, 0);
        for (uint32_t i = 1; i < LS; ++i) {
            copy_tile(cb_p, i, 1);
            S_modadd(0, 1, 0, P[j]);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_a, j);
        tile_regs_release();
        cb_pop_front(cb_p, LS);
    }
    cb_push_back(cb_a, LDT);
    cb_pop_front(cb_y, LS);

    // ===== step 3: eps = (A_sk - x_sk) * Q^-1 mod m_sk =====
    cb_wait_front(cb_a, LDT);
    cb_reserve_back(cb_e, 1);
    tile_regs_acquire();
    copy_tile_init(cb_a); copy_tile(cb_a, LD, 0);      // A_sk
    copy_tile_init(cb_x); copy_tile(cb_x, LS, 1);      // x_sk (the CENTERED value's residue)
    S_modsub(0, 1, 0, P[LD]);
    FILL(1, QINVSK);
    MULMOD(P[LD], PINV[LD]);                            // d4 = eps  (exact, in [0, LS+1])
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(4, cb_e, 0);
    tile_regs_release();
    cb_push_back(cb_e, 1);
    cb_pop_front(cb_x, LS + 1);

    // ===== step 5: out_j = (A_j - eps * (Q mod p_j)) mod p_j =====
    cb_wait_front(cb_e, 1);
    cb_reserve_back(cb_out, LD);
    for (uint32_t j = 0; j < LD; ++j) {
        tile_regs_acquire();
        copy_tile_init(cb_e); copy_tile(cb_e, 0, 0);
        FILL(1, G[j]);
        MULMOD(P[j], PINV[j]);                          // d4 = eps*Q mod p_j
        copy_tile_init(cb_a); copy_tile(cb_a, j, 5);    // A_j -- AFTER MULMOD, which clobbers d5
        S_modsub(5, 4, 0, P[j]);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, j);
        tile_regs_release();
    }
    cb_push_back(cb_out, LD);
    cb_pop_front(cb_a, LDT);
    cb_pop_front(cb_e, 1);
}
