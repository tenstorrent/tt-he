// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for the 32x32 modular matmul (see docs/matmul/history/PLAN.md).
//
// For each output column-group bj (NB of them) we accumulate NB int8 tile
// matmuls (one per contraction block bk) into a single int32 dst tile, then
// pack it out. matmul_tiles() accumulates into the dst across repeated calls
// within one tile_regs_acquire()/commit() window -- the standard K-loop pattern.
//
// The int32 product tile P_bj[i, jsig*4+r] is then recombined with weights
// 128^r and reduced mod q on the host (the SFPU int path is too weak for a
// clean on-device reduction in v1).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;    // LHS Int8
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;    // RHS Int8
    constexpr uint32_t cb_out = tt::CBIndex::c_16;   // out Int32
    constexpr uint32_t NB = 4;                       // 32 / 8

    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t bj = 0; bj < NB; ++bj) {
        tile_regs_acquire();
        for (uint32_t bk = 0; bk < NB; ++bk) {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);
            matmul_tiles(cb_in0, cb_in1, 0, 0, 0);   // accumulate into dst tile 0
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}
