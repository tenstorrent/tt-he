// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// CRT-combine (#13) stage A — Garner difference.
//   diff = (r2 - (r1 mod q2)) mod q2,   r1 = x mod q1 (< q1), r2 = x mod q2 (< q2).
// Because q1 > q2 (our two 28-bit primes), r1 can be >= q2, so it is reduced mod q2 first
// (single conditional subtract, since r1 < q1 < 2·q2). Pure SFPI (no multiply).
//
// c_0 = r1, c_1 = r2, out c_16 = diff.  Runtime arg: 0 = q2.

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
inline void face_garner(uint32_t r1s, uint32_t r2s, uint32_t out, uint32_t Q2) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt u1 = sfpi::dst_reg[r1s * NV + i];
        sfpi::vUInt u2 = sfpi::dst_reg[r2s * NV + i];
        sfpi::vInt r1 = sfpi::reinterpret<sfpi::vInt>(u1);
        sfpi::vInt r2 = sfpi::reinterpret<sfpi::vInt>(u2);
        sfpi::vInt r1r = r1 - (int)Q2;                 // r1 mod q2 (r1 < 2·q2)
        v_if(r1r < 0) { r1r = r1; } v_endif;
        sfpi::vInt d = r2 - r1r;                        // in (-q2, q2)
        v_if(d < 0) { d = d + (int)Q2; } v_endif;       // -> [0, q2)
        sfpi::dst_reg[out * NV + i] = sfpi::reinterpret<sfpi::vUInt>(d);
    }
}
}  // namespace
#endif

inline void S_garner(uint32_t r1s, uint32_t r2s, uint32_t out, uint32_t Q2) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_garner, r1s, r2s, out, ckernel::VectorMode::RC, Q2));
}

void kernel_main() {
    DeviceZoneScopedN("garner_diff");
    const uint32_t Q2 = get_arg_val<uint32_t>(0);
    constexpr auto cb_r1 = tt::CBIndex::c_0, cb_r2 = tt::CBIndex::c_1, cb_out = tt::CBIndex::c_16;
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);

    unary_op_init_common(cb_r1, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_r1, 1);
        cb_wait_front(cb_r2, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_tile_init(cb_r1); copy_tile(cb_r1, 0, 0);
        copy_tile_init(cb_r2); copy_tile(cb_r2, 0, 1);
        S_garner(0, 1, 2, Q2);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(2, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_r1, 1);
        cb_pop_front(cb_r2, 1);
    }
}
