// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// IR kernel: per-tile modular subtract  out[t] = (a[t] - b[t]) mod q, over n_tiles.
// a,b canonical [0,q), q < 2^28.  r = a + q - b in (0, 2q) -> min(r, r-q) -> [0,q).
// Inputs CB c_0 (a), c_1 (b); output CB c_16.  Args: 0 = n_tiles, 1 = q.
// Tile-batched map (kernels_ir contract); same int-SFPU building blocks as add_modq.

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace { constexpr auto FMT = DataFormat::UInt32; }

void kernel_main() {
    DeviceZoneScopedN("sub_modq");
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t Q = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in0, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        fill_tile_init(); fill_tile_int<FMT>(1, Q);          // d1 = q
        copy_tile_init(cb_in0); copy_tile(cb_in0, 0, 0);     // d0 = a
        copy_tile_init(cb_in1); copy_tile(cb_in1, 0, 2);     // d2 = b
        add_int_tile_init(); add_int_tile<FMT>(0, 1, 0);     // d0 = a + q
        sub_int_tile_init(); sub_int_tile<FMT>(0, 2, 0);     // d0 = a + q - b   in (0, 2q)
        sub_int_tile<FMT>(0, 1, 2);                          // d2 = (a+q-b) - q  (= a-b, wraps if <q)
        binary_min_uint32_tile_init(); binary_min_uint32_tile(0, 2, 0);  // d0 = min -> [0, q)
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
