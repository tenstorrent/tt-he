// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// On-device 32x32 tile transpose (W<->H). This IS the NTT four-step `reindex`
// (out[c*32+r] = in[r*32+c] — swap the two base-32 index digits), done on the matrix
// engine instead of on host. One tile in (c_0) -> transposed tile out (c_16).
// Enables buffer-resident chaining: an NTT can leave its reindexed result in DRAM
// with no host round-trip.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

// IR kernel: per-tile 32x32 transpose over n_tiles (the NTT reindex / corner-turn).
// In c_0 -> transposed out c_16. Arg 0 = n_tiles.
void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    transpose_wh_init(cb_in, cb_out);
    DeviceZoneScopedN("transpose");

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        transpose_wh_tile(cb_in, 0, 0);      // d0 = transpose(in tile)
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out, 0);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
    }
}
