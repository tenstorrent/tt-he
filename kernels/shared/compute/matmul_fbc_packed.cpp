// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PACKED base-conversion combine on the int8 matrix engine: 4 coefficient-blocks
// packed into one 32-wide output tile so the downstream recombine runs on N/128
// super-blocks (4x fewer SFPU reductions — the measured bottleneck).
//
// Per super-block sb (= 4 coeff-blocks) and output limb r:
//   dst = Σ_{bl=0..3}  A_{sb*4+bl} @ B_{bl,r}
// where B_{bl,r} carries the CRT-fold limb only in columns [bl*8, bl*8+8), zeros
// elsewhere. The 4 blocks land in disjoint column ranges → one packed plane_r
// (32 coeff × 4 blocks·8 dst = 32 cols). Accumulation is the standard int32
// matmul_tiles dst-accumulate.  A is re-streamed per r (DRAM is not the limit).
//
// c_0 = A tiles (int8), c_1 = B_{bl,r} tiles (int8), c_16 = int32 packed planes.
// Runtime arg: 0 = nsb (super-blocks = NCT/4).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "hostdevcommon/kernel_structs.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t nsb = get_arg_val<uint32_t>(0);
    const uint32_t nbl = get_arg_val<uint32_t>(1);   // blocks packed per tile = 32/L

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t NR = 4;

    mm_init(cb_a, cb_b, cb_out);

    DeviceZoneScopedN("mmpack_total");
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        for (uint32_t r = 0; r < NR; ++r) {
            tile_regs_acquire();
            for (uint32_t bl = 0; bl < nbl; ++bl) {
                cb_wait_front(cb_a, 1);
                cb_wait_front(cb_b, 1);
                matmul_tiles(cb_a, cb_b, 0, 0, 0);   // accumulate block bl into dst 0
                cb_pop_front(cb_a, 1);
                cb_pop_front(cb_b, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
    }
}
