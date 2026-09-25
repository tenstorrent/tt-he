// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// modmul_sfpi_full with ONE line changed: the 16x16 product uses Blackhole's SFPMUL24.
//
// modmul_sfpi_full's premise is stated in its own header -- "Wormhole SFPI has no integer multiply
// (and no sfpmul24), so each product is the fp32 byte-split trick". Blackhole HAS sfpmul24, exposed
// as sfpi::fractional_mul(a, b, Low|High) behind `#if __riscv_xtttensixbh`. exp_Tg018s probed what
// it computes over 532 structured operand pairs and got one model on every one of them:
//
//     operands are masked to (a & (2^23-1)) and (b & (2^23-1))   -- bits >= 23 are DROPPED
//     Low  = bits [22:0]  of that 23x23 product
//     High = bits [45:23] of that 23x23 product
//
// i.e. an exact 23x23 -> 46-bit multiply in two instructions. It is NOT 24 bits despite the name,
// and `mul_int_tile` does not expose it: the Blackhole LLK loads LO16, takes MOD1_LOWER and stores
// LO16, so mul_int_tile is a 16x16 -> low-16 multiply. That is why modmul_compute, which routes its
// products through mul_int_tile, measured 11% SLOWER than the byte-split in exp_Tg018k part A.
//
// Every operand into mul16 here is < 2^16 < 2^23, so nothing is truncated and the two halves
// reassemble the exact 32-bit product. That replaces 4 int->float converts, 4 fp32 multiplies and 4
// float->uint16 converts with two multiplies. NOTHING ELSE in the kernel changes -- same algorithm,
// same 8-tile DST layout, same full-sync requirement, same runtime args -- so the measurement
// isolates the multiply primitive.
//
// Runtime args: 0=n_tiles, 1=q, 2=q_inv_32, 3=q_low16, 4=q_high16.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr auto FMT = DataFormat::UInt32;
inline void FILL(uint32_t d, uint32_t v) { fill_tile_init(); fill_tile_int<FMT>(d, v); }
}  // namespace

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;

// Exact 16x16 -> 32-bit product (x,y < 2^16), as TWO sfpmul24 instructions.
// x,y < 2^16 < 2^23 so neither operand is truncated; the product is < 2^32 so the high half is
// < 2^9 and the two pieces do not overlap.  Measured semantics: exp_Tg018s / docs/he/exp_tg018s.md.
inline sfpi::vUInt mul16(sfpi::vUInt x, sfpi::vUInt y) {
    using namespace sfpi;
    vUInt lo = vUInt(fractional_mul(x, y, FractionalHalf::Low));    // bits [22:0]
    vUInt hi = vUInt(fractional_mul(x, y, FractionalHalf::High));   // bits [45:23]
    return lo + (hi << 23u);
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
    DeviceZoneScopedN("modmul_mul24");
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

        S_limbs(0, 1, 2, 3, 4, 5);   // aL=d2, aH=d3, bL=d4, bH=d5
        S_mul(2, 4, 0);              // ll  = aL*bL -> d0
        S_mul(3, 5, 1);              // hh  = aH*bH -> d1
        S_mul(2, 5, 6);              // c1  = aL*bH -> d6
        S_mul(3, 4, 7);              // c2  = aH*bL -> d7
        S_lohi(0, 1, 6, 7, 2, 3);    // lo=d2, hi=d3

        FILL(0, QINV);               // d0 = q_inv_32
        S_mullo_a(2, 0, 4);          // cross -> d4
        S_mullo_b(2, 0, 4, 2);       // t = lo*q_inv_32 -> d2
        S_split(2, 4, 5);            // tL=d4, tH=d5

        FILL(0, QHIGH);              // d0 = q_high
        FILL(1, QLOW);               // d1 = q_low
        S_mul(5, 0, 6);              // pHH = tH*q_high -> d6
        S_mul(5, 1, 5);              // pMH = tH*q_low  -> d5
        S_mul(4, 0, 0);              // pML = tL*q_high -> d0
        S_mul(4, 1, 1);              // pLL = tL*q_low  -> d1
        FILL(2, Q);                  // d2 = q
        S_reduce(6, 5, 0, 1, 3, 2, 4);  // out = (a*b) mod q -> d4

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(4, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
