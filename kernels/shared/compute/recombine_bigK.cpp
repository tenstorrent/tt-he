// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Large-K SFPI recombine + reduce, one int32 output tile per 4 limb planes.
// Same as recombine_compute.cpp (SFPI), but the int8 plane sums C_r are larger
// because the matmul contracts more K (e.g. K=256 -> C_r <= ~2^22 vs 2^19 at
// K=32). Then low = C0+(C1<<7) <= ~3.2q and R = low+hi14 < ~4.2q, so the final
// reduction uses 5 conditional subtracts instead of 2.
//
//   R = (C0 + C1<<7 + C2<<14 + C3<<21) mod q
//
// One Montgomery reduction (mulmod by K14 = mont(2^14)):
//   hi   = C2 + (C3 << 7)            // < 2^29
//   hi14 = (hi * K14) mod q          // = hi * 2^14 mod q, in [0, q)
//   low  = C0 + (C1 << 7)            // up to ~3.2q (large K)
//   R    = (low + hi14) mod q        // < ~4.2q -> FIVE conditional subtracts
//
// SFPI version: the non-multiply glue (limb build, lo/hi assembly, reduce,
// finalize) is done in custom register passes (no DST round-trips); only the 9
// mulmod sub-products use mul_int_tile. Same algorithm + result as the
// building-block version, far fewer SFPU passes (the Montgomery path mirrors
// modmul_sfpi_hybrid.cpp / modmul_sfpi_full.cpp).
//
// Inputs (CB c_0): groups of 4 int32 limb tiles C0,C1,C2,C3 (each a matmul plane, < 2^21).
// Output (CB c_16): one int32 tile per group, canonical in [0, q).
// Runtime args: 0=n_out, 1=q, 2=q_inv_32, 3=q_low16, 4=q_high16, 5=K14 (= mont(2^14)).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/mul_int_sfpu.h"

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

// hi = C2 + (C3 << 7)
inline void face_build_hi(uint32_t C2, uint32_t C3, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt c2 = sfpi::dst_reg[C2 * NV + i];
        sfpi::vUInt c3 = sfpi::dst_reg[C3 * NV + i];
        sfpi::dst_reg[out * NV + i] = c2 + (c3 << 7u);
    }
}
// 16-bit limbs of a, b  (mul_hi_lo step 1)
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
// assemble a*b = hi*2^32 + lo
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
// Montgomery reduce, finalized to [0, q).
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
        sfpi::vUInt b = sfpi::vUInt(sfpi::dst_reg[hi * NV + i]) + q - t_final;  // [0,2q)
        sfpi::vUInt bq = b - q;
        sfpi::vUInt res = bq;
        v_if(sfpi::reinterpret<sfpi::vInt>(bq) < 0) { res = b; }
        v_endif;
        sfpi::dst_reg[out * NV + i] = res;  // [0,q)
    }
}
// low = C0 + (C1<<7); R = (low + hi14) mod q  (two conditional subtracts).
inline void face_final(uint32_t C0, uint32_t C1, uint32_t hi14, uint32_t qIdx, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt c0 = sfpi::dst_reg[C0 * NV + i];
        sfpi::vUInt c1 = sfpi::dst_reg[C1 * NV + i];
        sfpi::vUInt q = sfpi::dst_reg[qIdx * NV + i];
        sfpi::vUInt low = c0 + (c1 << 7u);                                    // up to ~3.2q
        sfpi::vUInt r = low + sfpi::vUInt(sfpi::dst_reg[hi14 * NV + i]);       // < ~4.2q
        // 5 branchless conditional subtracts: r >= q ? r-q : r
        for (uint32_t s = 0; s < 5; s++) {
            sfpi::vUInt rq = r - q;
            sfpi::vUInt res = rq;
            v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; }
            v_endif;
            r = res;
        }
        sfpi::dst_reg[out * NV + i] = r;  // [0,q)
    }
}
}  // namespace
#endif

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
    const uint32_t n_out = get_arg_val<uint32_t>(0);
    const uint32_t Q = get_arg_val<uint32_t>(1);
    const uint32_t QINV = get_arg_val<uint32_t>(2);
    const uint32_t QLOW = get_arg_val<uint32_t>(3);
    const uint32_t QHIGH = get_arg_val<uint32_t>(4);
    const uint32_t K14 = get_arg_val<uint32_t>(5);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in, cb_out);
    MUL_INIT();
    FILL_INIT();

    for (uint32_t t = 0; t < n_out; ++t) {
        cb_wait_front(cb_in, 4);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        // hi = C2 + (C3 << 7)  -> d0 ;  b = K14 -> d1
        copy_tile_init(cb_in);
        copy_tile(cb_in, 2, 2);              // d2 = C2
        copy_tile(cb_in, 3, 3);              // d3 = C3
        S_build_hi(2, 3, 0);                 // d0 = hi
        FILL(1, K14);                        // d1 = K14

        // hi14 = (hi * K14) mod q  -> d4   (Montgomery, 9 muls, finalized [0,q))
        S_limbs(0, 1, 2, 3, 4, 5);           // aL=d2,aH=d3,bL=d4,bH=d5
        MUL(2, 4, 0);                        // ll  = aL*bL -> d0
        MUL(3, 5, 1);                        // hh  = aH*bH -> d1
        MUL(2, 5, 6);                        // c1  = aL*bH -> d6
        MUL(3, 4, 7);                        // c2  = aH*bL -> d7
        S_lohi(0, 1, 6, 7, 2, 3);            // lo=d2, hi=d3
        FILL(0, QINV); MUL(2, 0, 2);         // t = lo*q_inv_32 -> d2
        S_split(2, 4, 5);                    // tL=d4, tH=d5
        FILL(0, QHIGH); FILL(1, QLOW);
        MUL(5, 0, 6);                        // pHH = tH*q_high -> d6
        MUL(5, 1, 5);                        // pMH = tH*q_low  -> d5
        MUL(4, 0, 0);                        // pML = tL*q_high -> d0
        MUL(4, 1, 1);                        // pLL = tL*q_low  -> d1
        FILL(2, Q);                          // d2 = q
        S_reduce_fin(6, 5, 0, 1, 3, 2, 4);   // hi14 -> d4 (uses q=d2)

        // R = (C0 + C1<<7 + hi14) mod q  -> d5
        copy_tile(cb_in, 0, 0);              // d0 = C0
        copy_tile(cb_in, 1, 1);              // d1 = C1
        S_final(0, 1, 4, 2, 5);              // d5 = R (q still in d2)

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(5, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        cb_pop_front(cb_in, 4);
    }
}
