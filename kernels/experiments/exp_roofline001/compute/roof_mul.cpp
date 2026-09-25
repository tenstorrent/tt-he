// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// roof_mul — ONE 32x32 -> low-32 integer multiply, R times per tile: the MULTIPLY arm of
// exp_roofline001.
//
//   out = (a * b) mod 2^32      no modular reduction at all
//
// Built from the same sfpmul24 primitive as roof_modmul.cpp so the two are directly comparable:
// the low 32 bits of a 32x32 product need THREE 16x16 partials (aL*bL, aL*bH, aH*bL) = 6 sfpmul24,
// against the full Montgomery multiply's eleven = 22. Same shape otherwise -- one SFPI pass, R
// repeats, same CBs, same feed.
//
// This is the roofline's "what does a multiply cost" reference point. exp_bw001 measured the same
// question through the tile-op API (mul_int_tile, 517.5 Gmul/s L1-resident); this measures it through
// SFPI/sfpmul24, which is what every optimised kernel in this repo actually issues.
//
// Runtime args: 0 = n_tiles, 1 = R, 2 = Q (unused), 3..5 unused.

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
inline void face_mul(uint32_t A, uint32_t B_, uint32_t out,
                     uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt a = dst_reg[A * NV + i], b = dst_reg[B_ * NV + i];
        vUInt aL = a & 0xFFFFu, aH = a >> 16u, bL = b & 0xFFFFu, bH = b >> 16u;
        // low 32 bits only: the aH*bH partial contributes nothing below bit 32
        vUInt lo = mul16(aL, bL) + ((mul16(aL, bH) + mul16(aH, bL)) << 16u);
        dst_reg[out * NV + i] = lo;
    }
}
}  // namespace
#endif

inline void S_op(uint32_t A, uint32_t B_, uint32_t out,
                 uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_mul, A, B_, out, ckernel::VectorMode::RC,
                                               QINV, QHIGH, QLOW, Q));
}

void kernel_main() {
    DeviceZoneScopedN("roof_mul");
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
