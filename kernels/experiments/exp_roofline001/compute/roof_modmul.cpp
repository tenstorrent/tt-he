// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// roof_modmul — the Montgomery modular multiply, R times per tile, for the exp_roofline001 sweep.
//
// This is `compute/modmul_sfpi_1pass.cpp`'s arithmetic verbatim (the repo's best modmul: whole
// Montgomery reduce in ONE SFPI pass over sfpmul24, 4.03x the fp32 byte-split -- exp_modmul001).
// The ONLY thing added is the R loop, which is what moves the operating point along the roofline:
//
//   arithmetic intensity = R * 1024 ops / (3 * 4096 bytes) = R/12 ops/byte
//
// R=1 is the streaming point (read A, read B, write C for one op -- about the worst intensity an op
// can have); large R is the compute-bound point, where the memory traffic is amortised to nothing and
// what is left is the SFPU's throughput on this op.
//
// WHY THE R LOOP IS NOT DATA-DEPENDENT.  Each iteration recomputes d2 = modmul(d0, d1) from the same
// operands, exactly as `exp_bw001/compute/mul_i32_l1.cpp` does. Two consequences, both wanted:
//   * it measures THROUGHPUT, not the SFPU's latency through a dependent chain;
//   * the result is independent of R, so the host golden is the same at every point of the sweep and
//     correctness is gated at all of them.
// The risk is the compiler collapsing R identical passes into one. It does not -- the pass is an
// _llk_math_eltwise_binary_sfpu_params_ call with side effects on dst_reg -- and the bench does not
// take that on trust: it FITS time(R) = fixed + R*slope and reports the residual, so a collapsed loop
// would show up as a broken fit rather than as a spuriously high ceiling.
//
// Runtime args: 0 = n_tiles, 1 = R, 2 = Q, 3 = QINV, 4 = QLOW, 5 = QHIGH.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;
inline sfpi::vUInt mul16(sfpi::vUInt x, sfpi::vUInt y) {
    using namespace sfpi;
    vUInt lo = vUInt(fractional_mul(x, y, FractionalHalf::Low));
    vUInt hi = vUInt(fractional_mul(x, y, FractionalHalf::High));
    return lo + (hi << 23u);
}
inline void face_modmul(uint32_t A, uint32_t B_, uint32_t out,
                        uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt a = dst_reg[A * NV + i], b = dst_reg[B_ * NV + i];
        vUInt aL = a & 0xFFFFu, aH = a >> 16u, bL = b & 0xFFFFu, bH = b >> 16u;
        vUInt ll = mul16(aL, bL);
        vUInt cross = mul16(aL, bH) + mul16(aH, bL);
        vUInt hh = mul16(aH, bH);
        vUInt mid = (ll >> 16u) + (cross & 0xFFFFu);
        vUInt lo = (ll & 0xFFFFu) | (mid << 16u);
        vUInt hi = hh + (cross >> 16u) + (mid >> 16u);
        vUInt qiL = vUInt(QINV & 0xFFFFu), qiH = vUInt(QINV >> 16u);
        vUInt lL = lo & 0xFFFFu, lH = lo >> 16u;
        vUInt t = mul16(lL, qiL) + ((mul16(lL, qiH) + mul16(lH, qiL)) << 16u);
        vUInt tL = t & 0xFFFFu, tH = t >> 16u;
        vUInt vqh = vUInt(QHIGH), vql = vUInt(QLOW), vq = vUInt(Q);
        vUInt pHH = mul16(tH, vqh), pMH = mul16(tH, vql);
        vUInt pML = mul16(tL, vqh), pLL = mul16(tL, vql);
        vUInt mid_low  = (pMH & 0xFFFFu) + (pML & 0xFFFFu) + (pLL >> 16u);
        vUInt mid_high = (pMH >> 16u) + (pML >> 16u) + (mid_low >> 16u);
        vUInt t_final = pHH + mid_high;
        vUInt bo = hi + vq - t_final;
        vUInt bq = bo - vq;
        vUInt res = bq;
        v_if(sfpi::reinterpret<sfpi::vInt>(bq) < 0) { res = bo; }
        v_endif;
        dst_reg[out * NV + i] = res;
    }
}
}  // namespace
#endif

inline void S_op(uint32_t A, uint32_t B_, uint32_t out,
                 uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_modmul, A, B_, out, ckernel::VectorMode::RC,
                                               QINV, QHIGH, QLOW, Q));
}

void kernel_main() {
    DeviceZoneScopedN("roof_modmul");
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t R       = get_arg_val<uint32_t>(1);
    const uint32_t Q       = get_arg_val<uint32_t>(2);
    const uint32_t QINV    = get_arg_val<uint32_t>(3);
    const uint32_t QLOW    = get_arg_val<uint32_t>(4);
    const uint32_t QHIGH   = get_arg_val<uint32_t>(5);

    constexpr auto cb0 = tt::CBIndex::c_0, cb1 = tt::CBIndex::c_1, cb_out = tt::CBIndex::c_16;
    unary_op_init_common(cb0, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb0, 1);
        cb_wait_front(cb1, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        copy_tile_init(cb0); copy_tile(cb0, 0, 0);
        copy_tile_init(cb1); copy_tile(cb1, 0, 1);
        for (uint32_t i = 0; i < R; ++i) S_op(0, 1, 2, QINV, QHIGH, QLOW, Q);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(2, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb0, 1);
        cb_pop_front(cb1, 1);
    }
}
