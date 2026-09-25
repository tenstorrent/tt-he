// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// probe_mulchain — cycles per mul16, measured standalone.
//
// modmul_sfpi_1pass cannot be instrumented in place: adding a single probe variable to it makes the
// SFPI compiler spill (the kernel already sits at the 8-LREG limit, which is what technique 2 in
// kernels/OPTIMISED.md trades on). So this measures the multiply on its own: N chained mul16 in a
// dependency chain, so nothing is hoisted or dropped, and the per-tile cost is fitted over N.
//
// Runtime args: 0 = n_tiles, 1 = N (chained mul16 per element).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;
inline sfpi::vUInt mul16(sfpi::vUInt x, sfpi::vUInt y) {
    using namespace sfpi;
    vUInt lo = vUInt(fractional_mul(x, y, FractionalHalf::Low));
    vUInt hi = vUInt(fractional_mul(x, y, FractionalHalf::High));
    return lo + (hi << 23u);
}
inline void face_chain(uint32_t a, uint32_t b, uint32_t out, uint32_t N) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt x = dst_reg[a * NV + i], y = dst_reg[b * NV + i];
        for (uint32_t k = 0; k < N; ++k) x = mul16(x & 0xFFFFu, y & 0xFFFFu);
        dst_reg[out * NV + i] = x;
    }
}
}  // namespace
#endif
inline void S_chain(uint32_t a, uint32_t b, uint32_t out, uint32_t N) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_chain, a, b, out, ckernel::VectorMode::RC, N));
}

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t N = get_arg_val<uint32_t>(1);
    constexpr auto cb0 = tt::CBIndex::c_0, cb1 = tt::CBIndex::c_1, cbo = tt::CBIndex::c_16;
    unary_op_init_common(cb0, cbo);
    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb0, 1); cb_wait_front(cb1, 1); cb_reserve_back(cbo, 1);
        tile_regs_acquire();
        copy_tile_init(cb0); copy_tile(cb0, 0, 0);
        copy_tile_init(cb1); copy_tile(cb1, 0, 1);
        S_chain(0, 1, 2, N);
        tile_regs_commit(); tile_regs_wait();
        pack_tile(2, cbo); cb_push_back(cbo, 1); tile_regs_release();
        cb_pop_front(cb0, 1); cb_pop_front(cb1, 1);
    }
}
