// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PIR first-dimension SCAN as int8 matmul (throughput kernel, #12/#6).  Each output tile is a
// K-accumulated int8 matmul:  C_o = Σ_{k<KB} A[o,k] @ B[k]  (int8 32×32 -> int32, accumulated in
// dst).  A = DB radix-128 limb tiles (streamed, unique per output tile — the bandwidth-dominant
// operand); B = query limb tiles (shared).  One int32 plane tile per output; the radix-128 recombine
// + mod-q reduction is the (separately validated) SFPU tail (matmul_fbc / recombine_rns), not timed
// here — the scan is matmul-bandwidth-bound, which is what this measures.
//
//   c_0 = A (int8), c_1 = B (int8), out c_16 = int32 plane.
//   Runtime args: 0 = n_out (output tiles for THIS core), 1 = KB (contraction tiles).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "hostdevcommon/kernel_structs.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    DeviceZoneScopedN("scan_mm");
    const uint32_t n_out = get_arg_val<uint32_t>(0);
    const uint32_t KB    = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;   // A (DB limbs), int8
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;   // B (query limbs), int8
    constexpr uint32_t cb_out = tt::CBIndex::c_16;  // int32 plane

    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t o = 0; o < n_out; ++o) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < KB; ++k) {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);
            matmul_tiles(cb_in0, cb_in1, 0, 0, 0);   // accumulate into dst tile 0
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}
