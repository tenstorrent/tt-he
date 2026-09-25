// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SINGLE-SOURCE base conversion:  out_j = x mod q_j,  for j = 0 .. LD-1, from ONE source limb.
//
// WHY THIS EXISTS. The general FBC (baseconv_fbc.cpp) is a full CRNS / Cox-Rower base conversion:
// per-source pre-scale, base-extension accumulation, a fixed-point overflow term and its k·g
// correction. Measured, it costs ~386 us per dispatch and that cost is FLAT IN THE SOURCE COUNT --
// Ls = 1, 3 and 4 all measure 386.1 / 386.3 / 386.1 us, so the machinery runs whether or not there is
// anything to accumulate.
//
// The rescale's conversion has Ls = 1, and there its constants collapse: with a single-prime source
// base S = q_L we get S/q_0 = 1, hence C1[0] = mont(1) and C2[0][j] = mont(1), and the whole thing
// reduces to  out_j = x mod q_j  -- an integer reduction into each destination modulus, no summation
// and no overflow term. That is a couple of conditional subtracts, not 386 us.
//
// x is a canonical residue in [0, q_L). Values are natural (non-Montgomery) integers here: the
// transforms are Z_q-linear so Montgomery would ride through, but this path only ever sees natural
// residues (up_natE uploads with mont=false, and the scale constants are the Montgomery-encoded side
// of the modmul). A plain conditional subtract is therefore representation-correct, which a Montgomery
// multiply by mont(1) would also have been -- just far more expensively.
//
// TWO conditional subtracts, not one: one covers q_L < 2·q_j, two cover q_L < 3·q_j. The primes are
// all 28-bit and within a small factor, so one would do; the second costs ~nothing and removes the
// dependence on their ordering.
//
//   c_0   1 source tile (x)          c_16   LD output tiles, pushed in destination order
//   Runtime args: 0 = LD, 1..LD = the destination moduli q_j
//
// Fed by dataflow/reader_gather.cpp (n=1) and drained by dataflow/writer_scatter.cpp (n_out=LD),
// the same pair the general FBC uses.

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace { constexpr auto FMT = DataFormat::UInt32; }

void kernel_main() {
    const uint32_t LD = get_arg_val<uint32_t>(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in, cb_out);
    DeviceZoneScopedN("bconv_one");

    cb_wait_front(cb_in, 1);

    for (uint32_t j = 0; j < LD; ++j) {
        const uint32_t Q = get_arg_val<uint32_t>(1 + j);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_tile_init(cb_in); copy_tile(cb_in, 0, 0);                    // d0 = x
        fill_tile_init(); fill_tile_int<FMT>(1, Q);                       // d1 = q_j
        // out = min(x, x-q) twice. The subtract wraps when x < q, so min picks the un-subtracted
        // value -- the standard branchless conditional subtract used by modadd_reduce.
        sub_int_tile_init(); sub_int_tile<FMT>(0, 1, 2);
        binary_min_uint32_tile_init(); binary_min_uint32_tile(0, 2, 0);
        sub_int_tile_init(); sub_int_tile<FMT>(0, 1, 2);
        binary_min_uint32_tile_init(); binary_min_uint32_tile(0, 2, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
    }

    cb_pop_front(cb_in, 1);
}
