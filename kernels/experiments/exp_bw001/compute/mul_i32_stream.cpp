// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_Bw001 streaming compute — plain int32 elementwise multiply, C = A * B (low 32 bits, wrapping).
// No modular reduction: this is the raw SFPU `mul_int32` and nothing else, so the measured rate is
// the machine's elementwise-multiply rate rather than a Montgomery kernel's.
//
// One 32x32 tile per sequential iteration; n_iter = 2^n iterations per core.  The entire 2^n loop
// lives INSIDE the kernel, which is what lets the whole (2^n, C, 32, 32) tensor be one dispatch.
//
// BATCH tiles are processed per DST acquire.  DST holds 8 tiles at 32 bits (fp32_dest_acc_en +
// dst_full_sync_en), and each tile needs an (a, b) pair resident, so BATCH <= 4 is a hard ceiling:
//   d[2j] = a_j, d[2j+1] = b_j, then d[2j] = a_j * b_j in place.
// Batching amortizes the acquire/commit/wait/release handshake, which is per-acquire overhead the
// single-tile form pays 2^n times.
//
//   Runtime args: 0=n_iter, 1=BATCH

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/mul_int_sfpu.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr auto FMT = DataFormat::Int32;
}

void kernel_main() {
    DeviceZoneScopedN("mul_i32_stream");
    const uint32_t n_iter = get_arg_val<uint32_t>(0);
    const uint32_t BATCH  = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in0, cb_out);

    for (uint32_t i = 0; i < n_iter; i += BATCH) {
        cb_wait_front(cb_in0, BATCH);
        cb_wait_front(cb_in1, BATCH);
        cb_reserve_back(cb_out, BATCH);

        tile_regs_acquire();

        copy_tile_init(cb_in0);
        for (uint32_t j = 0; j < BATCH; ++j) copy_tile(cb_in0, j, 2 * j);
        copy_tile_init(cb_in1);
        for (uint32_t j = 0; j < BATCH; ++j) copy_tile(cb_in1, j, 2 * j + 1);

        mul_int_tile_init<FMT>();
        for (uint32_t j = 0; j < BATCH; ++j) mul_int_tile<FMT>(2 * j, 2 * j + 1, 2 * j);

        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < BATCH; ++j) pack_tile(2 * j, cb_out, j);
        tile_regs_release();

        cb_push_back(cb_out, BATCH);
        cb_pop_front(cb_in0, BATCH);
        cb_pop_front(cb_in1, BATCH);
    }
}
