// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Base-conversion CRT combine on the INT8 MATRIX ENGINE, at exp_Tg018's shape.
//   plane_r[c][j] = Σ_{(i,p)} yl[c][i,p] · Bl_r[(i,p)][j]      (NA int8 matmuls per r)
// where yl = radix-128 limbs of y (LHS, Ls src × 4 limbs = Ls*4 contraction lanes) and Bl_r = limb r
// of the folded CRT matrix (RHS). recombine_rns then does (Σ_r 128^r · plane_r) mod p_j.
//
// This is matmul_fbc.cpp generalised past ONE tile of contraction. matmul_fbc assumes Ls=8, so the
// 32 lanes fit a single 32x32 tile and each plane is one matmul. exp_Tg018 runs Ls=16 -> 64 lanes,
// so each plane is the sum of NA = Ls*4/32 = 2 tile-matmuls. They accumulate in DST before the one
// pack, which is what matmul_tiles does within a single tile_regs_acquire -- the same structure
// fused_matmul_a uses for its 4-limb contraction.
//
// The B limbs are constants: the reader loads all NR*NA of them once and never pops, so B costs one
// read for the whole dispatch instead of one per coefficient-tile.
//
// c_0: NA int8 A-tiles per coefficient-tile.   c_1: NR*NA int8 B-tiles, resident.
// c_16: NR int32 plane tiles per coefficient-tile.
// Runtime args: 0 = nct (coefficient-tiles), 1 = NA (lane-groups).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    const uint32_t nct = get_arg_val<uint32_t>(0);
    const uint32_t NA = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t NR = 4;

    mm_init(cb_a, cb_b, cb_out);

    cb_wait_front(cb_b, NR * NA);          // resident for the whole dispatch; never popped
    for (uint32_t ct = 0; ct < nct; ++ct) {
        cb_wait_front(cb_a, NA);
        for (uint32_t r = 0; r < NR; ++r) {
            tile_regs_acquire();
            for (uint32_t s = 0; s < NA; ++s) matmul_tiles(cb_a, cb_b, s, r * NA + s, 0);  // accumulate in DST
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_a, NA);
    }
}
