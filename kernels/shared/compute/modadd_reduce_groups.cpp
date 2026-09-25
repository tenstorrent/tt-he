// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GROUPED modadd_reduce: G independent reductions in ONE program.  out[g] = (Σ_i in[g][i]) mod q,
// for g = 0..G-1, each group being n_in tiles.  (modadd_reduce does exactly one group.)
//
// Why: a level-wide key-switch MACs B ciphertexts at once, so the digit axis must be reduced B*2
// times (once per node per tile-half).  Doing that as B*2 separate `modadd_reduce` programs would
// make the MAC O(B) in dispatches and defeat the batching; folding the groups into one program keeps
// it O(1).  It also removes the reason the forward NTTs had to run as `ell` separate batches: with a
// grouped reduce the natural single batch (dN in [node][digit] order) works, and the reduction just
// strides over the digit axis.  See ONION_MATCH_PERF §1c.
//
// The group loop is safe batching: ONE engine mode for the whole program (`unary_op_init_common`
// hoisted out), tiles looped inside — the same shape as `modmul_compute`, not the mode-switch-per-item
// anti-pattern that broke earlier batched kernels.
//
// Inputs on CB c_0 (G*n_in tiles, group-major — the reader decides which tiles those are).
// Output on CB c_16 (G tiles).  Runtime args: 0 = G, 1 = n_in, 2 = q.

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
    const uint32_t G = get_arg_val<uint32_t>(0);
    const uint32_t n_in = get_arg_val<uint32_t>(1);
    const uint32_t Q = get_arg_val<uint32_t>(2);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in, cb_out);              // once, outside the group loop
    DeviceZoneScopedN("modadd_reduce_groups");

    for (uint32_t g = 0; g < G; ++g) {
        cb_wait_front(cb_in, n_in);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        fill_tile_init(); fill_tile_int<FMT>(1, Q);           // d1 = q
        copy_tile_init(cb_in); copy_tile(cb_in, 0, 0);         // d0 = acc = in_0
        for (uint32_t i = 1; i < n_in; ++i) {
            copy_tile(cb_in, i, 2);                            // d2 = in_i
            add_int_tile_init(); add_int_tile<FMT>(0, 2, 0);   // d0 = acc + in_i  (< 2q each step)
            sub_int_tile_init(); sub_int_tile<FMT>(0, 1, 2);   // d2 = acc - q (wraps if acc<q)
            binary_min_uint32_tile_init(); binary_min_uint32_tile(0, 2, 0);  // d0 = min -> [0,q)
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, n_in);
    }
}
