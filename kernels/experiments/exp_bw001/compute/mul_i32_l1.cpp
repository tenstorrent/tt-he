// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_Bw001 L1-resident compute — the SAME int32 elementwise multiply as mul_i32_stream.cpp, with
// all memory traffic removed from the 2^n loop.
//
// One (a, b) tile pair is unpacked into DST once; then `mul_int_tile` runs 2^n times over it and one
// result tile is packed out at the end.  No DRAM read, no DRAM write, no CB handshake and no DST
// acquire inside the loop — nothing but the SFPU issuing multiplies.
//
// WHY BOTH KERNELS EXIST.  The streaming kernel's time is min(SFPU rate, DRAM bandwidth) and cannot
// tell you which of the two it hit.  This one measures the SFPU ceiling alone, so the pair gives a
// roofline: streaming/L1 is directly how much of the multiply rate the memory system is costing.
//
// IS THE LOOP REALLY RUNNING?  `mul_int_tile` expands to TTI_* hardware instruction macros through
// MATH(), not to anything the compiler can value-number away, so 2^n identical multiplies are 2^n
// real multiplies.  The empirical proof is in the measurement rather than the argument: the reported
// time is linear in 2^n over 12 doublings, which a hoisted loop could not be.
//
//   Runtime args: 0=n_iter (2^n)

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/mul_int_sfpu.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr auto FMT = DataFormat::Int32;
}

void kernel_main() {
    DeviceZoneScopedN("mul_i32_l1");
    const uint32_t n_iter = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in0, cb_out);

    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();

    copy_tile_init(cb_in0);
    copy_tile(cb_in0, 0, 0);   // d0 = a
    copy_tile_init(cb_in1);
    copy_tile(cb_in1, 0, 1);   // d1 = b

    mul_int_tile_init<FMT>();
    for (uint32_t i = 0; i < n_iter; ++i) {
        mul_int_tile<FMT>(0, 1, 2);   // d2 = a * b
    }

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(2, cb_out, 0);
    tile_regs_release();

    cb_push_back(cb_out, 1);
    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
}
