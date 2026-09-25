// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Modular ADD of two separate DRAM buffers: out = (a + b) mod q, elementwise.
//
// modadd_reduce sums n tiles from ONE buffer, so it cannot add two independent buffers. This is the piece
// the resident Subs tree needs: ct_add and c0+mac both take operands that now live in different buffers.
// Structurally identical to garner_diff (reader_ab feeds c_0/c_1; pure SFPI, no multiply), so it carries the
// same low risk -- no matmul/SFPU engine switch.
//
// c_0 = a, c_1 = b, out c_16 = (a+b) mod q.  Runtime args: 0 = q, 1 = n_tiles.

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
inline void face_modadd(uint32_t as, uint32_t bs, uint32_t out, uint32_t Q) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vInt a = sfpi::reinterpret<sfpi::vInt>(sfpi::vUInt(sfpi::dst_reg[as * NV + i]));
        sfpi::vInt b = sfpi::reinterpret<sfpi::vInt>(sfpi::vUInt(sfpi::dst_reg[bs * NV + i]));
        sfpi::vInt s = a + b;                           // both < q < 2^28, so a+b < 2^29: no overflow
        v_if(s >= (int)Q) { s = s - (int)Q; } v_endif;   // single conditional subtract
        sfpi::dst_reg[out * NV + i] = sfpi::reinterpret<sfpi::vUInt>(s);
    }
}
}  // namespace
#endif

inline void S_modadd(uint32_t as, uint32_t bs, uint32_t out, uint32_t Q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_modadd, as, bs, out, ckernel::VectorMode::RC, Q));
}

void kernel_main() {
    DeviceZoneScopedN("modadd_ab");
    const uint32_t Q = get_arg_val<uint32_t>(0);
    constexpr auto cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1, cb_out = tt::CBIndex::c_16;
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);

    unary_op_init_common(cb_a, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_tile_init(cb_a); copy_tile(cb_a, 0, 0);
        copy_tile_init(cb_b); copy_tile(cb_b, 0, 1);
        S_modadd(0, 1, 2, Q);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(2, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
}
