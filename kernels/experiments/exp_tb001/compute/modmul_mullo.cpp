// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage-1 bring-up kernel: out = low 32 bits of (a * b).
// Validates the uint32 data path + the mul_int_tile primitive end-to-end.
#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/mul_int_sfpu.h"

void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in0, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_tile_init(cb_in0);
        copy_tile(cb_in0, 0, 0);
        copy_tile_init(cb_in1);
        copy_tile(cb_in1, 0, 1);

        mul_int_tile_init<DataFormat::UInt32>();
        mul_int_tile<DataFormat::UInt32>(0, 1, 0);  // dst0 = lo32(a*b)
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
