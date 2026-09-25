// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Multi-tile plane-matmul compute. Same per-tile work as matmul_compute.cpp
// (NB output planes, each NB int8 matmuls accumulated), wrapped in an outer loop
// over n_tiles cube tiles. The reader (reader_matmul_mt.cpp) streams the 16
// (LHS,RHS) pairs per tile in (bj,bk) order; this kernel packs 4 plane tiles per
// cube tile (so the writer emits planes [ti*4 .. ti*4+3]).
//
// Runtime args: 0 = n_tiles.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t NB = 4;

    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        for (uint32_t bj = 0; bj < NB; ++bj) {
            tile_regs_acquire();
            for (uint32_t bk = 0; bk < NB; ++bk) {
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);
                matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
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
}
