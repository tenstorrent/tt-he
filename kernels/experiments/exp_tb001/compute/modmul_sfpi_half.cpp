// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Montgomery modmul fully in SFPI, scheduled to fit in 4 DST tiles so the host
// can run it with dst_full_sync_en = false (half-sync: Dest split in half,
// MATH/PACK double-buffered).
//
// The multiply is the fp32 byte-split trick (Wormhole SFPI has no int multiply).
// Every product/limb lives in LREGs, never DST. Each pass does at most a few
// mul16 to stay within the SFPI register file. Tile schedule (only d0..d3):
//   T1: d2 = ll = aL*bL                 live {a,b,ll}
//   T2: d3 = cross = aL*bH + aH*bL      live {a,b,ll,cross}
//   T3: d0 = Tlo, d1 = Thi (uses hh)    live {a,b,ll,cross} -> {Tlo,Thi}
//   fill d2 = q_inv_32
//   M1: d3 = cross = TloL*qinvH + TloH*qinvL
//   M2: d0 = m = TloL*qinvL + cross<<16 live {m,Thi}
//   fill d2 = q
//   RA: d3 = mid_high(pLL,pML,pMH)      live {m,Thi,q,partial}
//   RB: d0 = (a*b) mod q (uses pHH,Thi) live {m,Thi,q,partial}
// Runtime args: 0=n_tiles, 1=q, 2=q_inv_32, 3=q_low16(unused), 4=q_high16(unused).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"

namespace {
constexpr auto FMT = DataFormat::UInt32;
inline void FILL(uint32_t d, uint32_t v) { fill_tile_int<FMT>(d, v); }
}  // namespace

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;

inline sfpi::vUInt mul16(sfpi::vUInt x, sfpi::vUInt y) {
    using namespace sfpi;
    vUInt x0 = x & 0xFFu, x1 = x >> 8u;
    vUInt y0 = y & 0xFFu, y1 = y >> 8u;
    vUInt p00 = float_to_uint16(int32_to_float(reinterpret<vInt>(x0), RoundMode::NearestEven) *
                                int32_to_float(reinterpret<vInt>(y0), RoundMode::NearestEven), RoundMode::NearestEven);
    vUInt p01 = float_to_uint16(int32_to_float(reinterpret<vInt>(x0), RoundMode::NearestEven) *
                                int32_to_float(reinterpret<vInt>(y1), RoundMode::NearestEven), RoundMode::NearestEven);
    vUInt p10 = float_to_uint16(int32_to_float(reinterpret<vInt>(x1), RoundMode::NearestEven) *
                                int32_to_float(reinterpret<vInt>(y0), RoundMode::NearestEven), RoundMode::NearestEven);
    vUInt p11 = float_to_uint16(int32_to_float(reinterpret<vInt>(x1), RoundMode::NearestEven) *
                                int32_to_float(reinterpret<vInt>(y1), RoundMode::NearestEven), RoundMode::NearestEven);
    vUInt mid = p01 + p10;
    return p00 + (mid << 8u) + (p11 << 16u);
}

inline void face_T1(uint32_t a, uint32_t b, uint32_t ll) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++)
        dst_reg[ll * NV + i] = mul16(vUInt(dst_reg[a * NV + i]) & 0xFFFFu, vUInt(dst_reg[b * NV + i]) & 0xFFFFu);
}
inline void face_T2(uint32_t a, uint32_t b, uint32_t cross) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt va = dst_reg[a * NV + i], vb = dst_reg[b * NV + i];
        dst_reg[cross * NV + i] = mul16(va & 0xFFFFu, vb >> 16u) + mul16(va >> 16u, vb & 0xFFFFu);
    }
}
inline void face_T3(uint32_t a, uint32_t b, uint32_t ll, uint32_t cross) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt hh = mul16(vUInt(dst_reg[a * NV + i]) >> 16u, vUInt(dst_reg[b * NV + i]) >> 16u);
        vUInt vll = dst_reg[ll * NV + i];
        vUInt vcross = dst_reg[cross * NV + i];
        vUInt mid = (vll >> 16u) + (vcross & 0xFFFFu);
        dst_reg[a * NV + i] = (vll & 0xFFFFu) | (mid << 16u);   // Tlo -> d(a)
        dst_reg[b * NV + i] = hh + (vcross >> 16u) + (mid >> 16u);  // Thi -> d(b)
    }
}

inline void face_M1(uint32_t lo, uint32_t qinv, uint32_t cross) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt vlo = dst_reg[lo * NV + i], vqi = dst_reg[qinv * NV + i];
        dst_reg[cross * NV + i] = mul16(vlo & 0xFFFFu, vqi >> 16u) + mul16(vlo >> 16u, vqi & 0xFFFFu);
    }
}
inline void face_M2(uint32_t lo, uint32_t qinv, uint32_t cross, uint32_t m) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt lolo = mul16(vUInt(dst_reg[lo * NV + i]) & 0xFFFFu, vUInt(dst_reg[qinv * NV + i]) & 0xFFFFu);
        dst_reg[m * NV + i] = lolo + (vUInt(dst_reg[cross * NV + i]) << 16u);
    }
}

// Reduce split across two passes to stay at <=2 mul16/pass with one scratch
// tile. RA1: partial = pLL>>16. RA2: partial = mid_high(pLL,pML,pMH).
inline void face_RA1(uint32_t m, uint32_t q, uint32_t partial) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt pLL = mul16(vUInt(dst_reg[m * NV + i]) & 0xFFFFu, vUInt(dst_reg[q * NV + i]) & 0xFFFFu);
        dst_reg[partial * NV + i] = pLL >> 16u;
    }
}
inline void face_RA2(uint32_t m, uint32_t q, uint32_t partial) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt vm = dst_reg[m * NV + i], vq = dst_reg[q * NV + i];
        vUInt mL = vm & 0xFFFFu, mH = vm >> 16u, qL = vq & 0xFFFFu, qH = vq >> 16u;
        vUInt pML = mul16(mL, qH);
        vUInt pMH = mul16(mH, qL);
        vUInt mid_low = (pMH & 0xFFFFu) + (pML & 0xFFFFu) + vUInt(dst_reg[partial * NV + i]);
        dst_reg[partial * NV + i] = (pMH >> 16u) + (pML >> 16u) + (mid_low >> 16u);
    }
}
inline void face_RB(uint32_t m, uint32_t q, uint32_t hi, uint32_t partial, uint32_t out) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt vm = dst_reg[m * NV + i], vq = dst_reg[q * NV + i];
        vUInt pHH = mul16(vm >> 16u, vq >> 16u);
        vUInt t_final = pHH + vUInt(dst_reg[partial * NV + i]);
        vUInt b = vUInt(dst_reg[hi * NV + i]) + vq - t_final;
        vUInt bq = b - vq;
        vUInt res = bq;
        v_if(reinterpret<vInt>(bq) < 0) { res = b; }
        v_endif;
        dst_reg[out * NV + i] = res;
    }
}
}  // namespace
#endif

inline void S_T1(uint32_t a, uint32_t b, uint32_t ll) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_T1, a, b, ll, ckernel::VectorMode::RC));
}
inline void S_T2(uint32_t a, uint32_t b, uint32_t cross) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_T2, a, b, cross, ckernel::VectorMode::RC));
}
inline void S_T3(uint32_t a, uint32_t b, uint32_t ll, uint32_t cross) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_T3, a, b, ll, ckernel::VectorMode::RC, cross));
}
inline void S_M1(uint32_t lo, uint32_t qinv, uint32_t cross) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_M1, lo, qinv, cross, ckernel::VectorMode::RC));
}
inline void S_M2(uint32_t lo, uint32_t qinv, uint32_t cross, uint32_t m) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_M2, lo, qinv, cross, ckernel::VectorMode::RC, m));
}
inline void S_RA1(uint32_t m, uint32_t q, uint32_t partial) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_RA1, m, q, partial, ckernel::VectorMode::RC));
}
inline void S_RA2(uint32_t m, uint32_t q, uint32_t partial) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_RA2, m, q, partial, ckernel::VectorMode::RC));
}
inline void S_RB(uint32_t m, uint32_t q, uint32_t hi, uint32_t partial, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_RB, m, q, hi, ckernel::VectorMode::RC, partial, out));
}

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t Q = get_arg_val<uint32_t>(1);
    const uint32_t QINV = get_arg_val<uint32_t>(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in0, cb_out);
    fill_tile_init();

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        copy_tile_init(cb_in0); copy_tile(cb_in0, 0, 0);  // d0 = a
        copy_tile_init(cb_in1); copy_tile(cb_in1, 0, 1);  // d1 = b

        S_T1(0, 1, 2);          // d2 = ll
        S_T2(0, 1, 3);          // d3 = cross
        S_T3(0, 1, 2, 3);       // d0 = Tlo, d1 = Thi

        FILL(2, QINV);          // d2 = q_inv_32
        S_M1(0, 2, 3);          // d3 = cross
        S_M2(0, 2, 3, 0);       // d0 = m

        FILL(2, Q);             // d2 = q
        S_RA1(0, 2, 3);         // d3 = pLL>>16
        S_RA2(0, 2, 3);         // d3 = mid_high partial
        S_RB(0, 2, 1, 3, 0);    // d0 = (a*b) mod q

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
