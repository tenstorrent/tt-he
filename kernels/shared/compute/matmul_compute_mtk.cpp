// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// K-looped multi-tile plane-matmul for the large-shape modular matmul (2-factor NTT).
// For each assigned output tile, NB output planes; each plane accumulates K_acc int8
// tile-matmuls (= NB limbs x Kt contraction tiles) into one int32 dst tile. The reader
// (reader_matmul_mtk.cpp) streams the K_acc (LHS,RHS) pairs per plane in order.
//
// Runtime args: 0 = n_out (output tiles for this core), 1 = K_acc (= NB * Kt).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    const uint32_t n_out = get_arg_val<uint32_t>(0);
    const uint32_t K_acc = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t NB = 4;

    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t i = 0; i < n_out; ++i) {
        for (uint32_t plane = 0; plane < NB; ++plane) {
            tile_regs_acquire();
            for (uint32_t a = 0; a < K_acc; ++a) {
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
