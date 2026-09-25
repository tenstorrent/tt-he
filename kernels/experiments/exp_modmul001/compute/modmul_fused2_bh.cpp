// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_modmul001 / modmul_fused2_bh — the register-pressure hypothesis, taken all the way.
//
// modmul_fused_bh proved the premise: with sfpi::fractional_mul costing two registers instead of the
// fp32 byte-split's eight, four 16x16 products fit in registers at once, so the six passes that
// existed only to shuttle them through DST collapse to one. Measured 5.376 -> 4.660 us/tile.
//
// This applies the same collapse to the other two product groups of the Montgomery reduce:
//
//   S_limbs + 4x S_mul + S_lohi        (6 passes)  ->  S_ab      lo, hi of a*b
//   S_mullo_a + S_mullo_b + S_split    (3 passes)  ->  S_t       t = lo*q_inv_32, split
//   4x S_mul + S_reduce                (5 passes)  ->  S_final   the four t*q products AND the reduce
//
// 14 SFPI passes -> 3. Every partial product is consumed in the register that produced it.
// The algorithm is untouched -- same Montgomery reduce, same constants, same canonical output.
//
// FAILURE MODE, stated in advance: S_final holds four products plus hi and q live simultaneously.
// If that spills, this is slower than modmul_fused_bh, and that is the result.
//
// OUTPUT: canonical, in [0, q).  Runtime args: 0=n_tiles, 1=q, 2=q_inv_32, 3=q_low16, 4=q_high16.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"

namespace {
constexpr auto FMT = DataFormat::UInt32;
inline void FILL(uint32_t d, uint32_t v) { fill_tile_init(); fill_tile_int<FMT>(d, v); }
}  // namespace

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;

// Exact 16x16 -> 32-bit product (x,y < 2^16). 8-bit split keeps each fp32
// product < 2^16, exactly representable, recovered with float_to_uint16.
inline sfpi::vUInt mul16(sfpi::vUInt x, sfpi::vUInt y) {
    using namespace sfpi;
    vUInt lo = vUInt(fractional_mul(x, y, FractionalHalf::Low));    // product bits [22:0]
    vUInt hi = vUInt(fractional_mul(x, y, FractionalHalf::High));   // product bits [45:23]
    return lo + (hi << 23u);
}

// FUSED: a,b -> lo,hi of the full 32x32 product, in ONE pass.
// Replaces six passes (S_limbs, four S_mul, S_lohi) whose only purpose was to shuttle the four
// 16x16 partial products through DST because the fp32 byte-split needed 4 floats + 4 uint16
// temporaries per product -- "One product per pass keeps SFPI register pressure low enough to
// compile", per modmul_sfpi_full.cpp. mul16 now costs TWO registers, so all four products can live
// in registers at once and the intermediates never reach DST.
// Peak live values are ordered to keep pressure down: the limbs die as each product is formed.
inline void face_ab(uint32_t a, uint32_t b, uint32_t lo, uint32_t hi) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt va = dst_reg[a * NV + i], vb = dst_reg[b * NV + i];
        vUInt aL = va & 0xFFFFu, aH = va >> 16u;
        vUInt bL = vb & 0xFFFFu, bH = vb >> 16u;
        vUInt ll = mul16(aL, bL);
        vUInt cross = mul16(aL, bH) + mul16(aH, bL);
        vUInt hh = mul16(aH, bH);
        vUInt mid = (ll >> 16u) + (cross & 0xFFFFu);
        dst_reg[lo * NV + i] = (ll & 0xFFFFu) | (mid << 16u);
        dst_reg[hi * NV + i] = hh + (cross >> 16u) + (mid >> 16u);
    }
}

// FUSED: t = (lo * q_inv_32) mod 2^32, split into 16-bit halves. Replaces S_mullo_a + S_mullo_b +
// S_split (three passes) -- those existed only because the fp32 mul16 could not hold three products
// live at once.
inline void face_t(uint32_t lo, uint32_t qinv, uint32_t tL, uint32_t tH) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt vlo = dst_reg[lo * NV + i], vqi = dst_reg[qinv * NV + i];
        vUInt lL = vlo & 0xFFFFu, lH = vlo >> 16u;
        vUInt qL = vqi & 0xFFFFu, qH = vqi >> 16u;
        vUInt t = mul16(lL, qL) + ((mul16(lL, qH) + mul16(lH, qL)) << 16u);
        dst_reg[tL * NV + i] = t & 0xFFFFu;
        dst_reg[tH * NV + i] = t >> 16u;
    }
}

// FUSED: the four t*q products AND the Montgomery reduce, in one pass. Replaces four S_mul plus
// S_reduce (five passes). The four products are consumed immediately by the reduce, so they never
// need to reach DST.
inline void face_final(uint32_t tL, uint32_t tH, uint32_t qhigh, uint32_t qlow,
                       uint32_t hi, uint32_t qIdx, uint32_t out) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt vtL = dst_reg[tL * NV + i], vtH = dst_reg[tH * NV + i];
        vUInt vqh = dst_reg[qhigh * NV + i], vql = dst_reg[qlow * NV + i];
        vUInt pHH = mul16(vtH, vqh);
        vUInt pMH = mul16(vtH, vql);
        vUInt pML = mul16(vtL, vqh);
        vUInt pLL = mul16(vtL, vql);
        vUInt mid_low  = (pMH & 0xFFFFu) + (pML & 0xFFFFu) + (pLL >> 16u);
        vUInt mid_high = (pMH >> 16u) + (pML >> 16u) + (mid_low >> 16u);
        vUInt t_final = pHH + mid_high;
        vUInt q = dst_reg[qIdx * NV + i];
        vUInt b = vUInt(dst_reg[hi * NV + i]) + q - t_final;
        vUInt bq = b - q;
        vUInt res = bq;
        v_if(sfpi::reinterpret<sfpi::vInt>(bq) < 0) { res = b; }
        v_endif;
        dst_reg[out * NV + i] = res;
    }
}

// One product per pass keeps SFPI register pressure low enough to compile.
inline void face_mul(uint32_t a, uint32_t b, uint32_t o) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        dst_reg[o * NV + i] = mul16(vUInt(dst_reg[a * NV + i]), vUInt(dst_reg[b * NV + i]));
    }
}

// Low 32 bits of a full 32x32 product, split across two passes (xL*yH + xH*yL
// in the first, xL*yL + cross<<16 in the second) to stay within registers.
// Uses dst[scratch] to carry the cross term.
inline void face_mullo_a(uint32_t x, uint32_t y, uint32_t scratch) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt vx = dst_reg[x * NV + i], vy = dst_reg[y * NV + i];
        dst_reg[scratch * NV + i] = mul16(vx & 0xFFFFu, vy >> 16u) + mul16(vx >> 16u, vy & 0xFFFFu);
    }
}
inline void face_mullo_b(uint32_t x, uint32_t y, uint32_t scratch, uint32_t o) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt lolo = mul16(vUInt(dst_reg[x * NV + i]) & 0xFFFFu, vUInt(dst_reg[y * NV + i]) & 0xFFFFu);
        dst_reg[o * NV + i] = lolo + (vUInt(dst_reg[scratch * NV + i]) << 16u);
    }
}

// ---- proven assembly passes, copied verbatim from modmul_sfpi_hybrid.cpp ----
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
inline void face_reduce(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL,
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
}  // namespace
#endif

inline void S_mul(uint32_t a, uint32_t b, uint32_t o) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_mul, a, b, o, ckernel::VectorMode::RC));
}
inline void S_mullo_a(uint32_t x, uint32_t y, uint32_t s) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_mullo_a, x, y, s, ckernel::VectorMode::RC));
}
inline void S_mullo_b(uint32_t x, uint32_t y, uint32_t s, uint32_t o) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_mullo_b, x, y, s, ckernel::VectorMode::RC, o));
}

inline void S_ab(uint32_t a, uint32_t b, uint32_t lo, uint32_t hi) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_ab, a, b, lo, ckernel::VectorMode::RC, hi));
}
inline void S_t(uint32_t lo, uint32_t qinv, uint32_t tL, uint32_t tH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_t, lo, qinv, tL, ckernel::VectorMode::RC, tH));
}
inline void S_final(uint32_t tL, uint32_t tH, uint32_t qhigh, uint32_t qlow,
                    uint32_t hi, uint32_t qIdx, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_final, tL, tH, qhigh, ckernel::VectorMode::RC,
                                               qlow, hi, qIdx, out));
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
inline void S_reduce(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL,
                     uint32_t hi, uint32_t qIdx, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_reduce, pHH, pMH, pML, ckernel::VectorMode::RC, pLL, hi, qIdx, out));
}

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t Q = get_arg_val<uint32_t>(1);
    const uint32_t QINV = get_arg_val<uint32_t>(2);
    const uint32_t QLOW = get_arg_val<uint32_t>(3);
    const uint32_t QHIGH = get_arg_val<uint32_t>(4);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in0, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        copy_tile_init(cb_in0); copy_tile(cb_in0, 0, 0);  // d0 = a
        copy_tile_init(cb_in1); copy_tile(cb_in1, 0, 1);  // d1 = b

        S_ab(0, 1, 2, 3);            // ONE pass: lo=d2, hi=d3          (was six)
        FILL(0, QINV);
        S_t(2, 0, 4, 5);             // ONE pass: tL=d4, tH=d5          (was three)
        FILL(0, QHIGH); FILL(1, QLOW); FILL(2, Q);
        S_final(4, 5, 0, 1, 3, 2, 6);// ONE pass: out=d6                 (was five)

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(6, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
