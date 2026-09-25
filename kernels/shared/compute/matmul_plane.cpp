// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One output plane of the int8 plane-matmul, for the multi-core NTT: each core
// computes a single result plane  C = sum_k a_k @ b_k  (NB int8 matmuls accumulated
// into one int32 dst tile), then packs one tile. The reader (reader_matmul_mc.cpp)
// feeds this core only the NB (a_k, b_k) pairs for its plane.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t NB = 4;

    mm_init(cb_in0, cb_in1, cb_out);

    tile_regs_acquire();
    for (uint32_t k = 0; k < NB; ++k) {
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
