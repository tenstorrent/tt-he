// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Radix-128 limb decomposition, on-device (SFPI).
// Input : n_tiles int32 tiles (values in [0, 2^28)) on CB c_0.
// Output: for each input tile, 4 int8 limb tiles on CB c_16 (pushed in order
//         p = 0,1,2,3):   limb_p = (v >> 7p) & 127     (each in [0,127]).
//
// All four limbs are produced in ONE custom SFPI register pass (shifts+masks are
// native vUInt ops, no DST round-trips, no fills) instead of ~20 tile-op passes
// + 4 acquire/pack cycles. Bit-identical to the building-block version.
//
// Runtime args: 0 = n_tiles.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

namespace {
constexpr auto FMT = DataFormat::UInt32;
}

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;  // dst_reg slots per tile

// in: v -> out: l0,l1,l2,l3 where l_p = (v >> 7p) & 127
inline void face_decompose(uint32_t v, uint32_t l0, uint32_t l1, uint32_t l2, uint32_t l3) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt x = sfpi::dst_reg[v * NV + i];
        sfpi::dst_reg[l0 * NV + i] = x & 127u;
        sfpi::dst_reg[l1 * NV + i] = (x >> 7u) & 127u;
        sfpi::dst_reg[l2 * NV + i] = (x >> 14u) & 127u;
        sfpi::dst_reg[l3 * NV + i] = (x >> 21u) & 127u;
    }
}
}  // namespace
#endif

inline void S_decompose(uint32_t v, uint32_t l0, uint32_t l1, uint32_t l2, uint32_t l3) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_decompose, v, l0, l1, ckernel::VectorMode::RC, l2, l3));
}

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in, 1);

        tile_regs_acquire();
        copy_tile_init(cb_in); copy_tile(cb_in, 0, 4);   // d4 = v
        S_decompose(4, 0, 1, 2, 3);                       // d0..d3 = limbs 0..3
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out, 4);
        pack_tile(0, cb_out, 0);
        pack_tile(1, cb_out, 1);
        pack_tile(2, cb_out, 2);
        pack_tile(3, cb_out, 3);
        cb_push_back(cb_out, 4);
        tile_regs_release();

        cb_pop_front(cb_in, 1);
    }
}
