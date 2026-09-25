// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Sum n_in int32 tiles elementwise, reduced mod q.  out = (Σ_i in_i) mod q.
// Each input tile is a canonical residue in [0, q); n_in*q must fit in 32 bits
// (n_in <= 4, q < 2^28  ->  sum < 2^30, fine). Output canonical [0, q).
//
// Inputs on CB c_0 (n_in tiles). Output on CB c_16 (1 tile).
// Runtime args: 0 = n_in, 1 = q.

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
    const uint32_t n_in = get_arg_val<uint32_t>(0);
    const uint32_t Q = get_arg_val<uint32_t>(1);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in, cb_out);
    DeviceZoneScopedN("modadd_reduce");

    cb_wait_front(cb_in, n_in);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    fill_tile_init(); fill_tile_int<FMT>(1, Q);          // d1 = q
    copy_tile_init(cb_in); copy_tile(cb_in, 0, 0);        // d0 = acc = in_0
    for (uint32_t i = 1; i < n_in; ++i) {
        copy_tile(cb_in, i, 2);                           // d2 = in_i
        add_int_tile_init(); add_int_tile<FMT>(0, 2, 0);  // d0 = acc + in_i  (< 2q each step)
        sub_int_tile_init(); sub_int_tile<FMT>(0, 1, 2);  // d2 = acc - q (wraps if acc<q)
        binary_min_uint32_tile_init(); binary_min_uint32_tile(0, 2, 0);  // d0 = min -> [0,q)
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, 1);
    cb_pop_front(cb_in, n_in);
}
