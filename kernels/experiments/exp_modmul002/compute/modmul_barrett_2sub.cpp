// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_modmul002 / modmul_barrett_2sub — GENERAL Barrett, two conditional subtracts.  FALLBACK ONLY.
// Montgomery kernel it is compared against, so the comparison is of the ALGORITHM and not of how
// hard someone optimised each one.
//
// PREFER modmul_barrett.cpp.  This variant applies HAC 14.42's worst-case bound r < 3q and so
// corrects twice.  For any modulus where 2^27 + s < q (s = 2^56 - mu*q) the bound is actually r < 2q
// and one correction is exact -- see modmul_barrett.cpp, which is ~7% faster and wins in every
// measured run.  This file exists for moduli where that precondition does NOT hold, and it is what
// established the ~7% cost of a conditional subtract in docs/he/exp_modmul002.md.
//
// Same three techniques as kernels/OPTIMISED.md:
//   * every product is sfpi::fractional_mul (SFPMUL24), never the fp32 byte-split
//   * the WHOLE reduction is ONE SFPI pass -- no intermediate touches DST
//   * q and mu arrive as SCALAR arguments broadcast with vUInt(), never fill_tile_int
//
// ALGORITHM (Handbook of Applied Cryptography 14.42), n = 28 so q < 2^28, a,b < q:
//
//   mu = floor(2^56 / q)                       precomputed on the host, < 2^29
//   z  = a*b                                   < 2^56, held as (zhi, zlo)
//   q1 = z >> 27                               < 2^29
//   q3 = (q1 * mu) >> 29                       ~ z/q, never over by more than 2
//   r  = zlo - (q3*q mod 2^32)                 in [0, 3q)
//   two conditional subtracts                  -> [0, q)
//
// PRODUCT COUNT: 4 (z) + 4 (q1*mu) + 3 (q3*q, low 32 only) = 11 mul16 -- exactly the same as the
// Montgomery kernel (4 + 3 + 4). The algorithms differ in their shifts and in needing TWO final
// conditional subtracts rather than one, not in multiply count.
//
// WHY THIS DIFFERS FROM reference/vector/modmul_reference.py.  The reference implements
// the TEXTBOOK Barrett, which multiplies the full 56-bit z by m directly:
//
//     t = z * m        # 56b * 29b -> 85b   <-- WIDE product, needs high half
//     k = t >> 56 ;  r = z - k*q            # r in [0, 2q) -> ONE subtract
//
// That is correct and is the clearer statement of the algorithm, but on 16-bit limbs it is much more
// expensive: cost is limbs(x) * limbs(y), so 56b x 29b is 4 x 2 = EIGHT partial products against the
// FOUR that 29b x 29b needs.  Following it literally would make Barrett 4 + 8 + 3 = 15 mul16 against
// Montgomery's 11 -- it would lose before it started.  (Truncating the reference form by dropping
// z0*m0 and z0*m1, whose bits all fall below bit 56, still leaves six.)
//
// So this kernel pre-shifts z down to 29 bits first, which costs only shifts.  The price is exactly
// one extra conditional subtract, because HAC 14.42 bounds r by 3q rather than 2q -- about three
// instructions bought for about eight saved.  Same algorithm, a different point on the same trade.
// The two implementations agree on every input; the reference is the specification, this is the
// limb-efficient realisation of it.
//
// WHAT IT COMPUTES: a*b mod q on PLAIN residues. The Montgomery kernel computes a*b*R^-1 mod q on
// Montgomery-domain residues. Different functions, each checked against its own golden -- see
// host/experiments/exp_modmul002.cpp for why that is still a fair latency comparison and where it is not.
//
// Runtime args: 0 = n_tiles, 1 = q, 2 = mu.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;

// Exact 16x16 -> 32-bit product as TWO sfpmul24. x,y < 2^16 < 2^23 so neither operand is truncated
// by the instruction's 23-bit input mask; the product is < 2^32 so the halves do not overlap.
inline sfpi::vUInt mul16(sfpi::vUInt x, sfpi::vUInt y) {
    using namespace sfpi;
    vUInt lo = vUInt(fractional_mul(x, y, FractionalHalf::Low));    // product bits [22:0]
    vUInt hi = vUInt(fractional_mul(x, y, FractionalHalf::High));   // product bits [45:23]
    return lo + (hi << 23u);
}

inline void face_barrett(uint32_t a, uint32_t b, uint32_t out, uint32_t Q, uint32_t MU) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt va = dst_reg[a * NV + i], vb = dst_reg[b * NV + i];

        // z = a*b, as (zhi, zlo)
        vUInt aL = va & 0xFFFFu, aH = va >> 16u;
        vUInt bL = vb & 0xFFFFu, bH = vb >> 16u;
        vUInt p00 = mul16(aL, bL);
        vUInt cr  = mul16(aL, bH) + mul16(aH, bL);
        vUInt p11 = mul16(aH, bH);
        vUInt md  = (p00 >> 16u) + (cr & 0xFFFFu);
        vUInt zlo = (p00 & 0xFFFFu) | (md << 16u);
        vUInt zhi = p11 + (cr >> 16u) + (md >> 16u);

        // q1 = z >> 27 ; q3 = (q1 * mu) >> 29     (zhi dies here)
        vUInt q1 = (zhi << 5u) | (zlo >> 27u);
        vUInt mL = vUInt(MU & 0xFFFFu), mH = vUInt(MU >> 16u);
        vUInt q1L = q1 & 0xFFFFu, q1H = q1 >> 16u;
        vUInt s00 = mul16(q1L, mL);
        vUInt scr = mul16(q1L, mH) + mul16(q1H, mL);
        vUInt s11 = mul16(q1H, mH);
        vUInt md2 = (s00 >> 16u) + (scr & 0xFFFFu);
        vUInt q2lo = (s00 & 0xFFFFu) | (md2 << 16u);
        vUInt q2hi = s11 + (scr >> 16u) + (md2 >> 16u);
        vUInt q3 = (q2hi << 3u) | (q2lo >> 29u);

        // r = zlo - (q3*q mod 2^32); the true value is < 3q < 2^30 so the low word is exact
        vUInt vq = vUInt(Q);
        vUInt qL = vq & 0xFFFFu, qH = vq >> 16u;
        vUInt q3L = q3 & 0xFFFFu, q3H = q3 >> 16u;
        vUInt sub = mul16(q3L, qL) + ((mul16(q3L, qH) + mul16(q3H, qL)) << 16u);
        vUInt r = zlo - sub;

        // two conditional subtracts: r in [0, 3q) -> [0, q)
        vUInt t1 = r - vq; vUInt s1 = r;
        v_if(sfpi::reinterpret<sfpi::vInt>(t1) >= 0) { s1 = t1; } v_endif;
        vUInt t2 = s1 - vq; vUInt s2 = s1;
        v_if(sfpi::reinterpret<sfpi::vInt>(t2) >= 0) { s2 = t2; } v_endif;
        dst_reg[out * NV + i] = s2;
    }
}
}  // namespace
#endif

inline void S_barrett(uint32_t a, uint32_t b, uint32_t out, uint32_t Q, uint32_t MU) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_barrett, a, b, out, ckernel::VectorMode::RC, Q, MU));
}

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t Q = get_arg_val<uint32_t>(1);
    const uint32_t MU = get_arg_val<uint32_t>(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0, cb_in1 = tt::CBIndex::c_1, cb_out = tt::CBIndex::c_16;
    unary_op_init_common(cb_in0, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in0, 1); cb_wait_front(cb_in1, 1); cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        copy_tile_init(cb_in0); copy_tile(cb_in0, 0, 0);
        copy_tile_init(cb_in1); copy_tile(cb_in1, 0, 1);
        S_barrett(0, 1, 2, Q, MU);
        tile_regs_commit(); tile_regs_wait();
        pack_tile(2, cb_out);
        cb_push_back(cb_out, 1); tile_regs_release();
        cb_pop_front(cb_in0, 1); cb_pop_front(cb_in1, 1);
    }
}
