// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// TB001_PHASE_ZONES: the inner per-phase zones below are OPTIONAL profiler markers, and they
// triple-to-quadruple the marker volume of this kernel. At small G every dispatch lands on the
// same few cores, so the per-RISC DRAM marker buffer overflows and tt-metal drops markers --
// which breaks the 1-1 dispatch map the metrics rely on. They are therefore compiled out by
// default here; build with -DTB001_PHASE_ZONES to get the phase breakdown back.
// This is a divergence from kernels/ -- see kernels/experiments/exp_tb001/README.md.
//
// FUSED approximate RNS base conversion  x mod Q -> (x+εQ) mod P  (4->4 moduli),
// one compute kernel, intermediates (the y_i limbs) kept in an on-core L1 CB —
// no DRAM round-trips, ONE program dispatch (vs ~12 chained launches).
//
//   phase 1:  y_i   = (x_i · C1[i]) mod q_i                    [4 modmul, mod q_i]   x(c_0)->y(c_2)
//   phase 2:  out_j = Σ_i (y_i · C2[i][j]) mod p_j             [16 modmul + 4 modadd] y(c_2)->out(c_16)
//
// Each modmul is the canonical Montgomery mulmod (a·mont(c) mod m); constants are
// pre-Montgomery'd on host. C1,C2 and the per-modulus (m, m^{-1} mod 2^32) are
// runtime args; m_low/m_high are derived on device. Result is the APPROXIMATE FBC
// (the ε·Q overflow is not corrected here — add the ε correction for exact;
// see baseconv.cpp / baseconv_sk.cpp and docs/he/bfv_build_plan.md §1).
//
// SPLIT: PHASE 2 ONLY -- out_j = sum_i (y_i * C2[i][j]) mod p_j, for exp_Tg018m. Takes the y_i that
// compute/fbc_scale.cpp produced once, so this is the only part that repeats per LD chunk. See
// fbc_scale.cpp for why the split exists.
//
// Runtime args: 0 = LS, 1 = LD, then C2[LS*LD], P[LD], PINV[LD]  (146 at LS=16, LD=8).

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
    // SOURCE and DESTINATION counts are separate and are the ACTUAL widths, not a padded one.
    // This kernel used to take a single n = FBCW = 8 and loop 8x8 whatever the real conversion was.
    // Phase 1 is O(LS) and phase 2 is O(LS*LD), so at the widths actually used (LS=4,LD=7 and
    // LS=3,LD=8) that computed 72 modmuls where 32 were needed -- 56% of the work was zero padding.
    // Layout: LS, LD, C1[LS], C2[LS*LD], Q[LS], QINV[LS], P[LD], PINV[LD].
    const uint32_t LS = get_arg_val<uint32_t>(0);
    const uint32_t LD = get_arg_val<uint32_t>(1);
    // Sized for the WIDEST conversion any caller performs, not for exp_Tb001's alone: exp_Tb001b's
    // BFV rescale converts Q -> S|m_sk, which is Ld = 10. Overflowing these silently scribbles the
    // compute RISC's stack and every output comes back wrong, with nothing to point at the cause.
    // Sized for the WIDEST caller, not the current one. exp_Tg018 converts a dnum=3 digit
    // (alpha = 16 source primes) so LS reaches 16; LD stays <= 12 because the runtime-arg budget,
    // not the stack, is the binding limit -- LS=16, LD=12 is already 266 args. Callers past that
    // chunk their destination list (exp_Tg018 uses LD=8). Overflowing these silently corrupts the
    // compute RISC's stack, which is why they are sized here rather than at the caller's width.
    uint32_t C2[16][12], P[12], PINV[12];
    uint32_t a = 2;
    for (uint32_t i = 0; i < LS; ++i) for (uint32_t j = 0; j < LD; ++j) C2[i][j] = get_arg_val<uint32_t>(a++);
    for (uint32_t j = 0; j < LD; ++j) P[j]    = get_arg_val<uint32_t>(a++);
    for (uint32_t j = 0; j < LD; ++j) PINV[j] = get_arg_val<uint32_t>(a++);

    constexpr auto cb_x = tt::CBIndex::c_0;     // input residues x_i (LS tiles)
    constexpr auto cb_y = tt::CBIndex::c_2;     // intermediate y_i (LS, L1-resident)
    constexpr auto cb_p = tt::CBIndex::c_3;     // intermediate products per j (LS, L1)
    constexpr auto cb_out = tt::CBIndex::c_16;  // output residues (LD)

    unary_op_init_common(cb_x, cb_out);
    mul_int_tile_init<FMT>();
    fill_tile_init();

    DeviceZoneScopedN("fbc_combine");

    // ===== phase 2: out_j = Σ_i (y_i · C2[i][j]) mod p_j =====
    // (accumulator can't survive MULMOD in DST — only 8 int32 dst tiles — so the 4
    //  products go to an L1 CB and a separate modadd window sums them.)
    { 
#ifdef TB001_PHASE_ZONES
    DeviceZoneScopedN("fbc_phase2");
#endif
    cb_wait_front(cb_x, LS);
    cb_reserve_back(cb_out, LD);
    for (uint32_t j = 0; j < LD; ++j) {
        // LS modmuls -> products in c_p
        cb_reserve_back(cb_p, LS);
        for (uint32_t i = 0; i < LS; ++i) {
            tile_regs_acquire();
            copy_tile_init(cb_x); copy_tile(cb_x, i, 0);  // y_i -> d0
            FILL(1, C2[i][j]);                             // b = mont(qhat_i mod p_j) -> d1
            MULMOD(P[j], PINV[j]);                          // canonical prod -> d4
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
            S_modadd(0, 1, 0, P[j]);                        // acc = (acc + prod_i) mod p_j
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, j);
        tile_regs_release();
        cb_pop_front(cb_p, LS);
    }
    cb_push_back(cb_out, LD);
    cb_pop_front(cb_x, LS);
    }
}
