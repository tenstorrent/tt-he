// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Plain int32 elementwise multiply: out[t] = a[t] * b[t] (low 32 bits), over n_tiles.
//
// No modulus, no Montgomery — the bare operation, so a benchmark built on it measures the SFPU's
// int32 multiply rate and the surrounding dataflow, nothing else.  Note what that rate actually is:
// Wormhole has NO native integer multiplier, so `mul_int_tile` emulates it with fp32 limb products
// and keeps the low 32 bits. This kernel therefore measures the *emulated* int32 multiply.
//
// CBs: c_0 = a, c_1 = b, c_16 = out.  Runtime arg 0 = n_tiles.
// One multiply per tile, each tile distinct: no data is reused, so this is pure streaming.

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/mul_int_sfpu.h"

namespace { constexpr auto FMT = DataFormat::UInt32; }

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in0, cb_out);
    mul_int_tile_init<FMT>();

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_tile_init(cb_in0); copy_tile(cb_in0, 0, 0);     // d0 = a
        copy_tile_init(cb_in1); copy_tile(cb_in1, 0, 1);     // d1 = b
        mul_int_tile<FMT>(0, 1, 0);                          // d0 = a * b   (low 32)
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
