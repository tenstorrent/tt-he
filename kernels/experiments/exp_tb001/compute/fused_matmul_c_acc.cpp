// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ACCUMULATING stage-C modular matmul: decompose + matmul + ONE recombine over NACC partials (limbs are the LHS here).
//
// WHY. The 8-block contraction of the four-step NTT ran as 8 independent dispatches per output
// block: each decomposed its input tile, did 16 int8 tile-matmuls, and then ran the FULL recombine
// (~30 SFPU ops: 64-bit build, Montgomery reduce, final reduction) to land a partial residue mod q.
// A separate modadd tree then summed the 8 partials mod q.
//
// But the planes are just integers, and the recombine is Z-linear in them:
//     sum_p 128^p * plane_p  =  sum_a sum_n W_{a,n} * X_{a,n}      (exactly, before any mod)
// so accumulating the int32 planes ACROSS partials and recombining ONCE gives the same residue.
// The matrix engine already accumulates into DEST across matmul_tiles calls, so the accumulation
// itself costs nothing: NACC partials become one dispatch with NACC*16 matmuls, ONE recombine
// (down from NACC) and a 8/NACC-wide modadd (down from 8).
//
// The one thing that does not ride along is the final reduction's fixed 2-subtract chain -- see
// face_final_n. Bound: low = plane0 + 128*plane1 <= NACC * 129 * (4*32*127*127) = NACC * 2.66e8, so
// NACC = 8 reaches ~2.40e9, which is past int32 -- so the chain does the compare in a way that is
// valid over the whole unsigned range (see face_final_n). 8 is the ceiling: it is the block count,
// and it keeps every value inside 32 bits (hi = plane2 + 128*plane3 peaks at ~2.13e9).
//
//   c_0   NACC int32 input tiles        c_1  NACC*16 int8 folded-W tiles (a-major, then (p,r))
//   c_2   NACC*4 int8 limb tiles        c_3  4 int32 planes        c_16  1 int32 result
//   Runtime args: 0=q, 1=q_inv_32, 2=q_low16, 3=q_high16, 4=mont(2^14), 5=NACC, 6=nsub
//
// Fed by dataflow/reader_fused_acc.cpp. This tree only -- there is no shared counterpart.
// See docs/he/exp_tb001.md "Optimizations landed at this workload".
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
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
inline void FILL_INIT() { fill_tile_init(); }
inline void FILL(uint32_t d, uint32_t v) { fill_tile_int<FMT>(d, v); }
}  // namespace

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;

inline void face_decompose(uint32_t v, uint32_t l0, uint32_t l1, uint32_t l2, uint32_t l3) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt x = sfpi::dst_reg[v * NV + i];
        sfpi::dst_reg[l0 * NV + i] = x & 127u;
        sfpi::dst_reg[l1 * NV + i] = (x >> 7u) & 127u;
        sfpi::dst_reg[l2 * NV + i] = (x >> 14u) & 127u;
        sfpi::dst_reg[l3 * NV + i] = (x >> 21u) & 127u;
    }
}
inline void face_build_hi(uint32_t C2, uint32_t C3, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt c2 = sfpi::dst_reg[C2 * NV + i];
        sfpi::vUInt c3 = sfpi::dst_reg[C3 * NV + i];
        sfpi::dst_reg[out * NV + i] = c2 + (c3 << 7u);
    }
}
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
inline void face_reduce_fin(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL,
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
inline void face_final(uint32_t C0, uint32_t C1, uint32_t hi14, uint32_t qIdx, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt c0 = sfpi::dst_reg[C0 * NV + i];
        sfpi::vUInt c1 = sfpi::dst_reg[C1 * NV + i];
        sfpi::vUInt q = sfpi::dst_reg[qIdx * NV + i];
        sfpi::vUInt low = c0 + (c1 << 7u);
        sfpi::vUInt r = low + sfpi::vUInt(sfpi::dst_reg[hi14 * NV + i]);
        sfpi::vUInt rq = r - q;
        sfpi::vUInt res = rq;
        v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; }
        v_endif;
        r = res;
        rq = r - q;
        res = rq;
        v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; }
        v_endif;
        sfpi::dst_reg[out * NV + i] = res;
    }
}

// ACCUMULATED final reduction. The single-partial kernel needs exactly TWO conditional subtracts,
// because low = plane0 + 128*plane1 is then < 2q. Accumulating NACC partials into the planes scales
// low by NACC, so the chain length has to scale with it -- the host passes nsub = floor(r_max/q)+1
// where r_max = NACC * 129 * (NB*32*127*127) + q. Everything else in the recombine already tolerates
// a 32-bit `hi`, which is why the accumulation is otherwise free.
inline void face_final_n(uint32_t C0, uint32_t C1, uint32_t hi14, uint32_t qIdx, uint32_t out,
                         uint32_t nsub) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt q = sfpi::dst_reg[qIdx * NV + i];
        sfpi::vUInt r = sfpi::vUInt(sfpi::dst_reg[C0 * NV + i])
                      + (sfpi::vUInt(sfpi::dst_reg[C1 * NV + i]) << 7u)
                      + sfpi::vUInt(sfpi::dst_reg[hi14 * NV + i]);
        // Conditional subtract that is correct over the FULL unsigned 32-bit range, which the
        // single-partial kernel's version is not: it tests sign(r - q), so it silently stops working
        // once r >= 2^31. Two levels instead: if r has its top bit set then r > q unconditionally
        // (q is 28-bit), so subtract; only below 2^31 does the borrow test decide. That is what lets
        // NACC reach 8, where r_max is ~2.40e9.
        for (uint32_t k = 0; k < nsub; ++k) {
            sfpi::vUInt rq = r - q;
            sfpi::vUInt res = rq;
            v_if(sfpi::reinterpret<sfpi::vInt>(r) >= 0) {
                v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; }
                v_endif;
            }
            v_endif;
            r = res;
        }
        sfpi::dst_reg[out * NV + i] = r;
    }
}
}  // namespace
#endif

inline void S_decompose(uint32_t v, uint32_t l0, uint32_t l1, uint32_t l2, uint32_t l3) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_decompose, v, l0, l1, ckernel::VectorMode::RC, l2, l3));
}
inline void S_build_hi(uint32_t C2, uint32_t C3, uint32_t o) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_build_hi, C2, C3, o, ckernel::VectorMode::RC));
}
inline void S_limbs(uint32_t a, uint32_t b, uint32_t aL, uint32_t aH, uint32_t bL, uint32_t bH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_limbs, a, b, aL, ckernel::VectorMode::RC, aH, bL, bH));
}
inline void S_lohi(uint32_t ll, uint32_t hh, uint32_t c1, uint32_t c2, uint32_t lo, uint32_t hi) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_lohi, ll, hh, c1, ckernel::VectorMode::RC, c2, lo, hi));
}
inline void S_split(uint32_t t, uint32_t tL, uint32_t tH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_split, t, tL, tH, ckernel::VectorMode::RC));
}
inline void S_reduce_fin(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL,
                         uint32_t hi, uint32_t qIdx, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_reduce_fin, pHH, pMH, pML, ckernel::VectorMode::RC, pLL, hi, qIdx, out));
}
inline void S_final_n(uint32_t C0, uint32_t C1, uint32_t hi14, uint32_t qIdx, uint32_t out,
                      uint32_t nsub) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_final_n, C0, C1, hi14, ckernel::VectorMode::RC, qIdx, out, nsub));
}

inline void S_final(uint32_t C0, uint32_t C1, uint32_t hi14, uint32_t qIdx, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_final, C0, C1, hi14, ckernel::VectorMode::RC, qIdx, out));
}

void kernel_main() {
    const uint32_t Q = get_arg_val<uint32_t>(0);
    const uint32_t QINV = get_arg_val<uint32_t>(1);
    const uint32_t QLOW = get_arg_val<uint32_t>(2);
    const uint32_t QHIGH = get_arg_val<uint32_t>(3);
    const uint32_t K14 = get_arg_val<uint32_t>(4);
    const uint32_t NACC = get_arg_val<uint32_t>(5);
    const uint32_t NSUB = get_arg_val<uint32_t>(6);

    constexpr auto cb_in = tt::CBIndex::c_0;       // NACC int32 inputs
    constexpr auto cb_w = tt::CBIndex::c_1;        // NACC*16 folded W int8
    constexpr auto cb_limbs = tt::CBIndex::c_2;    // NACC*4 int8 limbs
    constexpr auto cb_planes = tt::CBIndex::c_3;   // 4 int32 planes (accumulated over NACC)
    constexpr auto cb_out = tt::CBIndex::c_16;     // 1 int32 result
    constexpr uint32_t NB = 4;

    DeviceZoneScopedN("fusedCacc_total");

    // ===== phase 1: decompose each of the NACC inputs -> 4 int8 limbs =====
    unary_op_init_common(cb_in, cb_limbs);
    cb_wait_front(cb_in, NACC);
    cb_reserve_back(cb_limbs, NACC * NB);
    for (uint32_t a = 0; a < NACC; ++a) {
        tile_regs_acquire();
        copy_tile_init(cb_in); copy_tile(cb_in, a, 4);
        S_decompose(4, 0, 1, 2, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_limbs, a * NB + 0);
        pack_tile(1, cb_limbs, a * NB + 1);
        pack_tile(2, cb_limbs, a * NB + 2);
        pack_tile(3, cb_limbs, a * NB + 3);
        tile_regs_release();
    }
    cb_push_back(cb_limbs, NACC * NB);
    cb_pop_front(cb_in, NACC);

    // ===== phase 2: C_r = sum_a sum_p limb_p(X_a) @ Wrhs_a[p,r], accumulating over BOTH the limb index and the partial index =====
    // Light-weight engine switch only -- a full re-init here deadlocks MATH against PACK (see the
    // single-partial kernel's header for the diagnosis).
    reconfig_data_format(cb_w, cb_limbs);
    pack_reconfig_data_format(cb_planes);
    matmul_init(cb_limbs, cb_w, 0 /*transpose*/);
    cb_wait_front(cb_limbs, NACC * NB);
    cb_wait_front(cb_w, NACC * NB * NB);
    cb_reserve_back(cb_planes, NB);
    for (uint32_t r = 0; r < NB; ++r) {
        tile_regs_acquire();
        for (uint32_t a = 0; a < NACC; ++a)
            for (uint32_t p = 0; p < NB; ++p)
                matmul_tiles(cb_limbs, cb_w, a * NB + p, a * NB * NB + p * NB + r, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_planes, r);
        tile_regs_release();
    }
    cb_push_back(cb_planes, NB);
    cb_pop_front(cb_limbs, NACC * NB);
    cb_pop_front(cb_w, NACC * NB * NB);

    // ===== phase 3: recombine the accumulated planes -> R mod q (ONCE, not per partial) =====
    reconfig_data_format_srca(cb_planes);
    pack_reconfig_data_format(cb_out);
    MUL_INIT();
    FILL_INIT();
    cb_wait_front(cb_planes, NB);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    copy_tile_init(cb_planes);
    copy_tile(cb_planes, 2, 2);
    copy_tile(cb_planes, 3, 3);
    S_build_hi(2, 3, 0);                 // d0 = hi = plane2 + plane3<<7  (32-bit, NACC-scaled)
    FILL(1, K14);
    S_limbs(0, 1, 2, 3, 4, 5);
    MUL(2, 4, 0); MUL(3, 5, 1); MUL(2, 5, 6); MUL(3, 4, 7);
    S_lohi(0, 1, 6, 7, 2, 3);
    FILL(0, QINV); MUL(2, 0, 2);
    S_split(2, 4, 5);
    FILL(0, QHIGH); FILL(1, QLOW);
    MUL(5, 0, 6); MUL(5, 1, 5); MUL(4, 0, 0); MUL(4, 1, 1);
    FILL(2, Q);
    S_reduce_fin(6, 5, 0, 1, 3, 2, 4);   // d4 = hi14 = hi*2^14 mod q
    copy_tile(cb_planes, 0, 0);
    copy_tile(cb_planes, 1, 1);
    S_final_n(0, 1, 4, 2, 5, NSUB);      // d5 = R, with the NACC-scaled subtract chain
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(5, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
    cb_pop_front(cb_planes, NB);
}
