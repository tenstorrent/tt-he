// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FUSED *exact* RNS base conversion  x mod Q -> x mod P  (4->4 moduli), ONE compute
// kernel / ONE dispatch, with the CRT overflow ε computed FULLY ON DEVICE via the
// integer Shenoy-Kumaresan method (a sentinel modulus m_sk; no floating point).
// Fused counterpart of the multi-dispatch baseconv_sk.cpp; same Montgomery I/O and
// baked constants as the approximate baseconv_fbc.cpp, plus the m_sk channel + ε.
//
//   phase 1 :  y_i  = (x̃_i · C1[i]) mod q_i                         [n modmul]   x̃(c_0)->y(c_2)
//   SK ε    :  A_sk = Σ_i y_i·Dsk[i] mod m_sk  = (x+εQ) mod m_sk
//              ε    = ((A_sk − x_sk)·Q^{-1}) mod m_sk               [exact integer ε]  ->eps(c_4)
//   phase 2 :  A_j  = Σ_i y_i·C2[i][j] mod p_j = (x+εQ)·R mod p_j   [approx, mont]
//              out_j= (A_j − ε·Ecorr[j]) mod p_j = x·R mod p_j      [exact correction] ->out(c_16)
//
// FULL Montgomery I/O (matches baseconv_fbc.cpp): ALL inputs are Montgomery —
// x̃_i = x_i·R mod q_i AND x̃_sk = x·R mod m_sk. y_i / ε stay plain/canonical internally.
// C1 = q̂_i⁻¹ (plain); C2, Ecorr, Dsk are DOUBLE-mont (·R²); QinvM is plain. Output õut_j = x·R mod p_j.
// (Dsk = R² makes A_sk Montgomery to match x̃_sk; QinvM plain makes ε plain for the correction.)
//
// Runtime args: 0=n; then C1[n], C2[n*n], Q[n], QINV[n], P[n], PINV[n], Dsk[n],
//   Ecorr[n], MSK, MSKINV, QINVM.  (x̃_i are cb_x tiles 0..n-1; x_sk is cb_x tile n.)

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
// acc = (acc + add) mod q   (acc,add ∈ [0,q) ⇒ one conditional subtract)
inline void face_modadd(uint32_t accS, uint32_t addS, uint32_t out, uint32_t Q) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt r = sfpi::vUInt(sfpi::dst_reg[accS * NV + i]) + sfpi::vUInt(sfpi::dst_reg[addS * NV + i]);
        sfpi::vUInt t = r - Q, res = t;
        v_if(sfpi::reinterpret<sfpi::vInt>(t) < 0) { res = r; } v_endif;
        sfpi::dst_reg[out * NV + i] = res;
    }
}
// out = (a − b) mod q   (a,b ∈ [0,q) ⇒ a−b ∈ (−q,q) ⇒ one conditional add)
inline void face_modsub(uint32_t aS, uint32_t bS, uint32_t out, uint32_t Q) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt r = sfpi::vUInt(sfpi::dst_reg[aS * NV + i]) - sfpi::vUInt(sfpi::dst_reg[bS * NV + i]);
        sfpi::vUInt res = r;
        v_if(sfpi::reinterpret<sfpi::vInt>(r) < 0) { res = r + Q; } v_endif;
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
inline void S_modsub(uint32_t aS, uint32_t bS, uint32_t out, uint32_t Q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_modsub, aS, bS, out, ckernel::VectorMode::RC, Q)); }

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
    const uint32_t n = get_arg_val<uint32_t>(0);
    uint32_t C1[8], C2[8][8], Q[8], QINV[8], P[8], PINV[8], DSK[8], ECORR[8], MSK, MSKINV, QINVM;
    uint32_t a = 1;
    for (uint32_t i = 0; i < n; ++i) C1[i] = get_arg_val<uint32_t>(a++);
    for (uint32_t i = 0; i < n; ++i) for (uint32_t j = 0; j < n; ++j) C2[i][j] = get_arg_val<uint32_t>(a++);
    for (uint32_t i = 0; i < n; ++i) Q[i]    = get_arg_val<uint32_t>(a++);
    for (uint32_t i = 0; i < n; ++i) QINV[i] = get_arg_val<uint32_t>(a++);
    for (uint32_t j = 0; j < n; ++j) P[j]    = get_arg_val<uint32_t>(a++);
    for (uint32_t j = 0; j < n; ++j) PINV[j] = get_arg_val<uint32_t>(a++);
    for (uint32_t i = 0; i < n; ++i) DSK[i]  = get_arg_val<uint32_t>(a++);
    for (uint32_t j = 0; j < n; ++j) ECORR[j]= get_arg_val<uint32_t>(a++);
    MSK    = get_arg_val<uint32_t>(a++);
    MSKINV = get_arg_val<uint32_t>(a++);
    QINVM  = get_arg_val<uint32_t>(a++);

    constexpr auto cb_x = tt::CBIndex::c_0;     // x̃_i (n tiles) + x_sk (tile n)
    constexpr auto cb_y = tt::CBIndex::c_2;     // y_i plain (n, L1)
    constexpr auto cb_p = tt::CBIndex::c_3;     // scratch products (n, L1)
    constexpr auto cb_eps = tt::CBIndex::c_4;   // ε plain (1, L1)
    constexpr auto cb_A = tt::CBIndex::c_5;     // A_j scratch (1, L1)
    constexpr auto cb_out = tt::CBIndex::c_16;  // exact output (n)

    unary_op_init_common(cb_x, cb_out);
    mul_int_tile_init<FMT>();
    fill_tile_init();

    DeviceZoneScopedN("fbc_total");

    cb_wait_front(cb_x, n + 1);   // n residues + x_sk

    // ===== phase 1: y_i = (x̃_i · C1[i]) mod q_i =====
    { DeviceZoneScopedN("fbc_phase1");
    cb_reserve_back(cb_y, n);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        copy_tile_init(cb_x); copy_tile(cb_x, i, 0);
        FILL(1, C1[i]);
        MULMOD(Q[i], QINV[i]);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(4, cb_y, i);
        tile_regs_release();
    }
    cb_push_back(cb_y, n);
    }

    // ===== SK ε: A_sk = Σ_i y_i·Dsk[i] mod m_sk ; ε = ((A_sk − x_sk)·Q^{-1}) mod m_sk =====
    { DeviceZoneScopedN("fbc_sk_eps");
    cb_wait_front(cb_y, n);
    cb_reserve_back(cb_p, n);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        copy_tile_init(cb_y); copy_tile(cb_y, i, 0);
        FILL(1, DSK[i]);
        MULMOD(MSK, MSKINV);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(4, cb_p, i);
        tile_regs_release();
    }
    cb_push_back(cb_p, n);
    cb_wait_front(cb_p, n);
    cb_reserve_back(cb_eps, 1);
    tile_regs_acquire();
    copy_tile_init(cb_p); copy_tile(cb_p, 0, 0);              // acc = prod_0
    for (uint32_t i = 1; i < n; ++i) { copy_tile(cb_p, i, 1); S_modadd(0, 1, 0, MSK); }  // A_sk -> d0
    copy_tile_init(cb_x); copy_tile(cb_x, n, 1);              // x_sk -> d1
    S_modsub(0, 1, 0, MSK);                                   // diff = A_sk − x_sk -> d0
    FILL(1, QINVM); MULMOD(MSK, MSKINV);                      // ε -> d4
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(4, cb_eps, 0);
    tile_regs_release();
    cb_push_back(cb_eps, 1);
    cb_pop_front(cb_p, n);
    cb_wait_front(cb_eps, 1);
    }

    cb_pop_front(cb_x, n + 1);

    // ===== phase 2: out_j = (Σ_i y_i·C2[i][j] − ε·Ecorr[j]) mod p_j =====
    { DeviceZoneScopedN("fbc_phase2");
    cb_wait_front(cb_y, n);
    cb_reserve_back(cb_out, n);
    for (uint32_t j = 0; j < n; ++j) {
        // 4 modmuls -> products in cb_p
        cb_reserve_back(cb_p, n);
        for (uint32_t i = 0; i < n; ++i) {
            tile_regs_acquire();
            copy_tile_init(cb_y); copy_tile(cb_y, i, 0);
            FILL(1, C2[i][j]);
            MULMOD(P[j], PINV[j]);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(4, cb_p, i);
            tile_regs_release();
        }
        cb_push_back(cb_p, n);
        cb_wait_front(cb_p, n);
        // A_j = Σ products mod p_j -> cb_A
        cb_reserve_back(cb_A, 1);
        tile_regs_acquire();
        copy_tile_init(cb_p); copy_tile(cb_p, 0, 0);
        for (uint32_t i = 1; i < n; ++i) { copy_tile(cb_p, i, 1); S_modadd(0, 1, 0, P[j]); }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_A, 0);
        tile_regs_release();
        cb_push_back(cb_A, 1);
        cb_wait_front(cb_A, 1);
        cb_pop_front(cb_p, n);
        // corr = ε·Ecorr[j] mod p_j ; out_j = (A_j − corr) mod p_j
        tile_regs_acquire();
        copy_tile_init(cb_eps); copy_tile(cb_eps, 0, 0);
        FILL(1, ECORR[j]); MULMOD(P[j], PINV[j]);            // corr -> d4
        copy_tile_init(cb_A); copy_tile(cb_A, 0, 0);         // A_j -> d0
        S_modsub(0, 4, 0, P[j]);                             // (A_j − corr) mod p_j -> d0
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, j);
        tile_regs_release();
        cb_pop_front(cb_A, 1);
    }
    cb_push_back(cb_out, n);
    cb_pop_front(cb_y, n);
    }
}
