// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Modular subtraction:  out = (a − b) mod q,  a,b canonical in [0,q), q < 2^28.
//   r = a + q − b  ∈ (0, 2q)  → one conditional subtract → [0,q).
// Reads a on c_0, b on c_1; writes out on c_16.  Runtime arg: 0 = q.
// (The vec_modsub primitive from docs/he/bfv_build_plan.md.)

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace { constexpr auto FMT = DataFormat::UInt32; }

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;
// a in slot `as`, b in slot `bs` -> out in slot `out`;  out = (a + q − b) mod q.
inline void face_modsub(uint32_t as, uint32_t bs, uint32_t out, uint32_t Q) {
    for (uint32_t v = 0; v < 8; v++) {
        sfpi::vUInt a = sfpi::dst_reg[as * NV + v];
        sfpi::vUInt b = sfpi::dst_reg[bs * NV + v];
        sfpi::vUInt r = a + Q - b;               // (0, 2q)
        sfpi::vUInt t = r - Q;
        sfpi::vUInt res = t;
        v_if(sfpi::reinterpret<sfpi::vInt>(t) < 0) { res = r; } v_endif;
        sfpi::dst_reg[out * NV + v] = res;
    }
}
}  // namespace
#endif

inline void S_modsub(uint32_t out, uint32_t Q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_modsub, 0, 1, out, ckernel::VectorMode::RC, Q));
}

void kernel_main() {
    const uint32_t Q = get_arg_val<uint32_t>(0);
    constexpr auto cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1, cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_a, cb_out);
    DeviceZoneScopedN("modsub_reduce");

    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    copy_tile_init(cb_a); copy_tile(cb_a, 0, 0);   // a -> slot 0
    copy_tile_init(cb_b); copy_tile(cb_b, 0, 1);   // b -> slot 1
    S_modsub(2, Q);                                 // out -> slot 2
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(2, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, 1);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}
