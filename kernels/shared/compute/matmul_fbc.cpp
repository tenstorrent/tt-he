// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Base-conversion CRT combine on the INT8 MATRIX ENGINE.
//   plane_r[c][j] = Σ_{(i,p)} yl[c][i,p] · Bl_r[(i,p)][j]      (one int8 matmul per r)
// where yl = radix-128 limbs of y (LHS, 8 src × 4 limbs = 32 contraction lanes) and
// Bl_r = limb r of the folded CRT matrix (RHS). The 4 r-planes are recombined
// (Σ_r 128^r · plane_r) mod p_j on the host (per-destination modulus).
//
// Reads A (1 int8 tile) on c_0 [reused across r] and B_r (4 int8 tiles) on c_1;
// writes 4 int32 plane tiles on c_16.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "hostdevcommon/kernel_structs.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t nct = get_arg_val<uint32_t>(0);  // coefficient-tiles (N/32)

    constexpr uint32_t cb_a = tt::CBIndex::c_0;    // LHS yl (Int8), one A-tile per coeff-tile
    constexpr uint32_t cb_b = tt::CBIndex::c_1;    // RHS Bl_0..3 (Int8), resident constants
    constexpr uint32_t cb_out = tt::CBIndex::c_16; // Int32 planes (4 per coeff-tile)
    constexpr uint32_t NR = 4;                     // radix-128 output limbs

    mm_init(cb_a, cb_b, cb_out);

    DeviceZoneScopedN("mmfbc_total");
    for (uint32_t ct = 0; ct < nct; ++ct) {
        { DeviceZoneScopedN("mmfbc_wait_a"); cb_wait_front(cb_a, 1); }
        for (uint32_t r = 0; r < NR; ++r) {
            { DeviceZoneScopedN("mmfbc_wait_b"); cb_wait_front(cb_b, 1); }
            tile_regs_acquire();
            { DeviceZoneScopedN("mmfbc_mm"); matmul_tiles(cb_a, cb_b, 0, 0, 0); }
            tile_regs_commit();
            tile_regs_wait();
            { DeviceZoneScopedN("mmfbc_pack"); cb_reserve_back(cb_out, 1); pack_tile(0, cb_out); cb_push_back(cb_out, 1); }
            tile_regs_release();
            cb_pop_front(cb_b, 1);
        }
        cb_pop_front(cb_a, 1);
    }
}
