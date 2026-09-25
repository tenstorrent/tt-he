// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// fused_matmul_c_1pass — Stage-C modular matmul (operands mirrored vs stage A) with the same one-pass recombine.
//
// MEASURED: the three NTT matmuls 17.501 -> 12.943 ms in ct_mul.  docs/he/exp_tg018m.md
//
// This is an OPTIMISED variant promoted into the shared kernel tree. The kernel it derives
// from (kernels/experiments/exp_tb001/compute/fused_matmul_c.cpp) is RETAINED and unchanged, so nothing that
// depends on the old behaviour moves. See kernels/OPTIMISED.md for the full set and the
// three techniques they share.
//
// Original header follows.
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: Apache-2.0
//
// FUSED stage-C modular matmul: decompose + matmul + recombine in ONE compute
// kernel. The limb tiles (int8) and plane tiles (int32) stay in on-core CBs --
// no DRAM round-trip, one program dispatch instead of three.
//
//   phase 1  decompose: G'(int32, c_0) -> 4 int8 limbs limb_p (c_2)    [SFPU]
//   phase 2  matmul:     C_r = sum_p limb_p(c_2) @ Wrhs[p,r](c_1)       [matrix engine]
//                        -> 4 int32 planes (c_3)   (limbs are the LHS here; Wrhs resident)
//   phase 3  recombine:  R = (sum_k 128^k * plane_k) mod q (c_3 -> c_16) [SFPU]
//
// Engine switch between phases must NOT be a full re-init. unary_op_init_common and mm_init both
// re-run the DEST-sync handshake -- MATH(llk_math_pack_sync_init) + PACK(llk_pack_dest_init) --
// and re-running that mid-kernel deadlocks MATH against PACK on Blackhole. See fused_matmul_a.cpp
// for the full diagnosis. ONE full init at kernel entry, then light-weight switches per phase.
//
// Runtime args: 0=q, 1=q_inv_32, 2=q_low16, 3=q_high16, 4=K14 (= mont(2^14)).

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

// ---- FUSED RECOMBINE: the whole of phase 3 in ONE SFPI pass ----
// The original ran 6 SFPI passes, 9 mul_int_tile and 5 fill_tile_int to compute
//     R = (C0 + C1<<7 + C2<<14 + C3<<21) mod q
// via one Montgomery reduction (hi14 = (C2 + C3<<7) * 2^14 mod q). That is structurally the
// modmul, and docs/he/exp_modmul001.md took exactly this shape from 4.930 to 3.086 us/tile with
// three moves, all applied here:
//   * mul_int_tile -> sfpi::fractional_mul  (2 registers per product instead of 8)
//   * the six passes collapsed into one     (partial products never reach DST)
//   * all five fill_tile_int deleted        (each wrote a WHOLE DST tile to broadcast a scalar)
// The arithmetic is unchanged -- same Montgomery reduce, same constants, same canonical output.
inline sfpi::vUInt mul16(sfpi::vUInt x, sfpi::vUInt y) {
    using namespace sfpi;
    vUInt lo = vUInt(fractional_mul(x, y, FractionalHalf::Low));    // product bits [22:0]
    vUInt hi = vUInt(fractional_mul(x, y, FractionalHalf::High));   // product bits [45:23]
    return lo + (hi << 23u);
}
inline void face_recomb(uint32_t C0, uint32_t C1, uint32_t C2, uint32_t C3, uint32_t out,
                        uint32_t K14, uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt vhi = vUInt(dst_reg[C2 * NV + i]) + (vUInt(dst_reg[C3 * NV + i]) << 7u);
        // full 32x32 product vhi * K14 -> lo, hiP
        vUInt aL = vhi & 0xFFFFu, aH = vhi >> 16u;
        vUInt bL = vUInt(K14 & 0xFFFFu), bH = vUInt(K14 >> 16u);
        vUInt ll = mul16(aL, bL);
        vUInt cross = mul16(aL, bH) + mul16(aH, bL);
        vUInt hh = mul16(aH, bH);
        vUInt mid = (ll >> 16u) + (cross & 0xFFFFu);
        vUInt lo = (ll & 0xFFFFu) | (mid << 16u);
        vUInt hiP = hh + (cross >> 16u) + (mid >> 16u);
        // t = (lo * q_inv_32) mod 2^32
        vUInt qiL = vUInt(QINV & 0xFFFFu), qiH = vUInt(QINV >> 16u);
        vUInt lL = lo & 0xFFFFu, lH = lo >> 16u;
        vUInt t = mul16(lL, qiL) + ((mul16(lL, qiH) + mul16(lH, qiL)) << 16u);
        vUInt tL = t & 0xFFFFu, tH = t >> 16u;
        // the four t*q products and the reduce -> hi14
        vUInt vqh = vUInt(QHIGH), vql = vUInt(QLOW), vq = vUInt(Q);
        vUInt pHH = mul16(tH, vqh), pMH = mul16(tH, vql);
        vUInt pML = mul16(tL, vqh), pLL = mul16(tL, vql);
        vUInt mid_low  = (pMH & 0xFFFFu) + (pML & 0xFFFFu) + (pLL >> 16u);
        vUInt mid_high = (pMH >> 16u) + (pML >> 16u) + (mid_low >> 16u);
        vUInt t_final = pHH + mid_high;
        vUInt b1 = hiP + vq - t_final;
        vUInt bq1 = b1 - vq;
        vUInt hi14 = bq1;
        v_if(sfpi::reinterpret<sfpi::vInt>(bq1) < 0) { hi14 = b1; }
        v_endif;
        // R = (C0 + C1<<7 + hi14) mod q -- two conditional subtracts, as face_final did
        vUInt low = vUInt(dst_reg[C0 * NV + i]) + (vUInt(dst_reg[C1 * NV + i]) << 7u);
        vUInt r = low + hi14;
        vUInt rq = r - vq;
        vUInt res = rq;
        v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; }
        v_endif;
        r = res; rq = r - vq; res = rq;
        v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; }
        v_endif;
        dst_reg[out * NV + i] = res;
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
}  // namespace
#endif

inline void S_recomb(uint32_t C0, uint32_t C1, uint32_t C2, uint32_t C3, uint32_t out,
                     uint32_t K14, uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_recomb, C0, C1, C2, ckernel::VectorMode::RC,
                                               C3, out, K14, QINV, QHIGH, QLOW, Q));
}
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
inline void S_final(uint32_t C0, uint32_t C1, uint32_t hi14, uint32_t qIdx, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_final, C0, C1, hi14, ckernel::VectorMode::RC, qIdx, out));
}

void kernel_main() {
    DeviceZoneScopedN("fusedC_total");
    const uint32_t Q = get_arg_val<uint32_t>(0);
    const uint32_t QINV = get_arg_val<uint32_t>(1);
    const uint32_t QLOW = get_arg_val<uint32_t>(2);
    const uint32_t QHIGH = get_arg_val<uint32_t>(3);
    const uint32_t K14 = get_arg_val<uint32_t>(4);

    constexpr auto cb_in = tt::CBIndex::c_0;       // Xin int32
    constexpr auto cb_w = tt::CBIndex::c_1;        // folded W int8
    constexpr auto cb_limbs = tt::CBIndex::c_2;    // int8 limbs (intermediate, NB resident)
    constexpr auto cb_planes = tt::CBIndex::c_3;   // int32 planes (intermediate, NB)
    constexpr auto cb_out = tt::CBIndex::c_16;     // int32 result
    constexpr uint32_t NB = 4;

    // ===== phase 1: decompose Xin -> 4 int8 limbs =====
    {
    #ifdef TB001_PHASE_ZONES
    DeviceZoneScopedN("ph_decompose");
    #endif
    unary_op_init_common(cb_in, cb_limbs);
    cb_wait_front(cb_in, 1);
    tile_regs_acquire();
    copy_tile_init(cb_in); copy_tile(cb_in, 0, 4);   // d4 = Xin
    S_decompose(4, 0, 1, 2, 3);                       // d0..d3 = limb 0..3
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_limbs, NB);
    pack_tile(0, cb_limbs, 0);
    pack_tile(1, cb_limbs, 1);
    pack_tile(2, cb_limbs, 2);
    pack_tile(3, cb_limbs, 3);
    cb_push_back(cb_limbs, NB);
    tile_regs_release();
    cb_pop_front(cb_in, 1);
    }

    // ===== phase 2: matmul  C_r = sum_p limb_p @ Wrhs[p,r]  -> 4 int32 planes =====
    // limbs (LHS) and all 16 Wrhs tiles (RHS) are resident and indexed.
    // Light-weight switch to the matrix engine (see header note). in0=cb_limbs, in1=cb_w here,
    // so srcA<-cb_w / srcB<-cb_limbs; both int8. Pack -> int32 planes.
    reconfig_data_format(cb_w, cb_limbs);
    matmul_init(cb_limbs, cb_w, 0 /*transpose*/);
    cb_wait_front(cb_limbs, NB);                      // 4 limbs resident
    cb_wait_front(cb_w, NB * NB);                     // 16 Wrhs tiles resident
    // phases 2+3 FUSED: plane r accumulates into DST tile r inside ONE acquire, so the
    // recombine reads all four in place. Removes 4 pack_tile + 4 copy_tile and 3 acquire cycles.
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    {
#ifdef TB001_PHASE_ZONES
    DeviceZoneScopedN("ph_matmul");
#endif
    for (uint32_t r = 0; r < NB; ++r) {
            for (uint32_t p = 0; p < NB; ++p)
                matmul_tiles(cb_limbs, cb_w, p, p * NB + r, r);
    }
    }
    {
#ifdef TB001_PHASE_ZONES
    DeviceZoneScopedN("ph_recombine");
#endif
    S_recomb(0, 1, 2, 3, 4, K14, QINV, QHIGH, QLOW, Q);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(4, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
    cb_pop_front(cb_w, NB * NB);
    cb_pop_front(cb_limbs, NB);
}
