// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// fused_matmul_k_1pass — the 2-factor NTT's K-LOOPED matmul with the bigK recombine fused in, planes
// kept in DST.  The large-shape counterpart of fused_matmul_{a,c}_1pass.cpp.
//
//   phase 1  matmul:    plane p accumulates K_acc = NB*Kt int8 tile-matmuls into DST tile p
//   phase 2  recombine: R = (sum_k 128^k * plane_k) mod q, ONE SFPI pass, planes read in place
//
// WHY A SEPARATE KERNEL.  fused_matmul_a/c_1pass are K=1 (16 tile-matmuls per output tile); the
// 2-factor contracts K=256 and K=128, i.e. K_acc = 32 and 16 pairs per plane.  And the recombine must
// be the bigK variant: at K=256 the plane sums reach ~2^22, so the final reduction needs FIVE
// conditional subtracts instead of two (recombine_bigK.cpp).
//
// WHAT IT REMOVES: the four int32 planes (16 KB/core) that matmul_compute_mtk wrote to DRAM for
// recombine_bigK to read back, plus 3 of the 4 acquire/pack cycles.  What it does NOT remove is the
// decompose, which stays a separate dispatch here -- see docs/matmul/exp_ntt001.md: the 2-factor's
// matmuls are not tile-local (an output tile needs 8 input tiles for M1, 4 for M2), so folding the
// decompose in would make 8 cores each decompose the same 8 tiles.
//
// Pairs are consumed in the reader's (plane, inner, kt) order -- identical to matmul_compute_mtk --
// so reader_matmul_mtk{,_deep}.cpp work unchanged.
//
// Engine switch: ONE full init at entry, then matmul_init / reconfig only. Re-running a full init
// mid-kernel deadlocks MATH against PACK on Blackhole (docs/he/blackhole_port.md 2).
//
// Runtime args: 0=K_acc (= NB*Kt), 1=q, 2=q_inv_32, 3=q_low16, 4=q_high16, 5=K14.

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

#ifdef NTT_PHASE_ZONES
#define PHASE_ZONE(name) DeviceZoneScopedN(name)
#else
#define PHASE_ZONE(name)
#endif

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
        // bigK: K=256 pushes C_r to ~2^22, so low = C0+(C1<<7) reaches ~3.2q and R ~4.2q --
        // FIVE conditional subtracts, exactly as recombine_bigK.cpp documents. This is the only
        // line of difference from fused_matmul_a_1pass's recombine.
        vUInt low = vUInt(dst_reg[C0 * NV + i]) + (vUInt(dst_reg[C1 * NV + i]) << 7u);
        vUInt r = low + hi14;
        vUInt res = r;
        for (uint32_t k = 0; k < 5; ++k) {
            vUInt rq = r - vq;
            res = rq;
            v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; }
            v_endif;
            r = res;
        }
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

void kernel_main() {
    DeviceZoneScopedN("fusedK_total");
    const uint32_t K_acc = get_arg_val<uint32_t>(0);
    const uint32_t Q     = get_arg_val<uint32_t>(1);
    const uint32_t QINV  = get_arg_val<uint32_t>(2);
    const uint32_t QLOW  = get_arg_val<uint32_t>(3);
    const uint32_t QHIGH = get_arg_val<uint32_t>(4);
    const uint32_t K14   = get_arg_val<uint32_t>(5);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t NB = 4;

    mm_init(cb_in0, cb_in1, cb_out);            // the ONE full init

    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    { PHASE_ZONE("ph_matmul");
    for (uint32_t p = 0; p < NB; ++p) {         // plane p accumulates into DST tile p
        for (uint32_t a = 0; a < K_acc; ++a) {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);
            matmul_tiles(cb_in0, cb_in1, 0, 0, p);
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }
    } }
    { PHASE_ZONE("ph_recombine");
      S_recomb(0, 1, 2, 3, 4, K14, QINV, QHIGH, QLOW, Q); }   // planes read in place from d0..d3
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(4, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
}
