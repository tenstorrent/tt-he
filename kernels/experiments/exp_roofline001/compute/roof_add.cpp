// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// roof_add — a modular ADD, R times per tile: the NON-MULTIPLY arm of exp_roofline001.
//
//   out = (a + b) mod q      one add, one conditional subtract, ZERO multiplies
//
// It is the same shape as roof_modmul.cpp in every other respect -- one SFPI pass, R repeats, same
// CBs, same feed -- so the pair isolates exactly what the multiply costs. Modular add rather than a
// bare add because that is the operation the HE pipeline actually issues (compute/add_modq.cpp), and
// because a bare add would be so cheap that the pass overhead, not the arithmetic, would set its
// ceiling.
//
// a, b < q < 2^28 so the sum is under 2^29 and one conditional subtract suffices; the sign test is
// the idiom the verified kernels already use.
//
// Runtime args: 0 = n_tiles, 1 = R, 2 = Q, 3 = QINV (unused), 4 = QLOW (unused), 5 = QHIGH (unused).

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
inline void face_add(uint32_t A, uint32_t B_, uint32_t out,
                     uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt s = vUInt(dst_reg[A * NV + i]) + vUInt(dst_reg[B_ * NV + i]);
        vUInt vq = vUInt(Q);
        vUInt sq = s - vq;
        vUInt res = sq;
        v_if(sfpi::reinterpret<sfpi::vInt>(sq) < 0) { res = s; }
        v_endif;
        dst_reg[out * NV + i] = res;
    }
}
}  // namespace
#endif

inline void S_op(uint32_t A, uint32_t B_, uint32_t out,
                 uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_add, A, B_, out, ckernel::VectorMode::RC,
                                               QINV, QHIGH, QLOW, Q));
}

void kernel_main() {
    DeviceZoneScopedN("roof_add");
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
