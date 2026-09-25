// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// CRT-combine (#13) K=3 tail — assemble the 84-bit value X from two 56-bit partials and split
// into three radix-2^28 limbs.  Given the Garner nested form  X = v0 + q0·(v1 + q1·v2), computed
// as three reuses of crt_mul_split:
//     w = q1·v2 + v1                       (56-bit, limbs w0,w1)
//     P = q0·w0 + v0                       (56-bit, limbs P0,P1)      <- carries v0
//     Q = q0·w1                            (56-bit, limbs Q0,Q1)
// then  X = P + Q·2^28  =  P0 + (P1+Q0)·2^28 + Q1·2^56, so the three output limbs are:
//     X0 = P0
//     X1 = (P1 + Q0) & (2^28-1)
//     X2 = Q1 + ((P1 + Q0) >> 28)
// (P1,Q0 < 2^28 so P1+Q0 < 2^29 -> carry is 0/1; X < q < 2^84 so X2 < 2^28.)  Pure SFPI, no mul.
//
//   c_0 = (P0, P1) per tile,  c_1 = (Q0, Q1) per tile.   out c_16 = (X0, X1, X2) per tile.
// Runtime arg: 0 = n_tiles.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace { constexpr auto FMT = DataFormat::UInt32; }

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;
constexpr uint32_t M28 = (1u << 28) - 1u;
// (P0,P1,Q0,Q1) at slots p0,p1,q0,q1 -> (X0,X1,X2) at slots x0,x1,x2.
inline void face_limb3(uint32_t p0s, uint32_t p1s, uint32_t q0s, uint32_t q1s,
                       uint32_t x0s, uint32_t x1s, uint32_t x2s) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt P0 = sfpi::dst_reg[p0s * NV + i];
        sfpi::vUInt P1 = sfpi::dst_reg[p1s * NV + i];
        sfpi::vUInt Q0 = sfpi::dst_reg[q0s * NV + i];
        sfpi::vUInt Q1 = sfpi::dst_reg[q1s * NV + i];
        sfpi::vUInt s1 = P1 + Q0;                        // < 2^29
        sfpi::dst_reg[x0s * NV + i] = P0 & M28;
        sfpi::dst_reg[x1s * NV + i] = s1 & M28;
        sfpi::dst_reg[x2s * NV + i] = Q1 + (s1 >> 28u);
    }
}
}  // namespace
#endif

inline void S_limb3(uint32_t p0s, uint32_t p1s, uint32_t q0s, uint32_t q1s,
                    uint32_t x0s, uint32_t x1s, uint32_t x2s) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_limb3, p0s, p1s, q0s, ckernel::VectorMode::RC,
                                               q1s, x0s, x1s, x2s));
}

void kernel_main() {
    DeviceZoneScopedN("crt_limb3");
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_p = tt::CBIndex::c_0, cb_q = tt::CBIndex::c_1, cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_p, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_p, 2);
        cb_wait_front(cb_q, 2);
        cb_reserve_back(cb_out, 3);

        tile_regs_acquire();
        copy_tile_init(cb_p); copy_tile(cb_p, 0, 0); copy_tile(cb_p, 1, 1);   // P0->0, P1->1
        copy_tile_init(cb_q); copy_tile(cb_q, 0, 2); copy_tile(cb_q, 1, 3);   // Q0->2, Q1->3
        S_limb3(0, 1, 2, 3, 4, 5, 6);                                          // X0,X1,X2 -> 4,5,6
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(4, cb_out, 0);
        pack_tile(5, cb_out, 1);
        pack_tile(6, cb_out, 2);
        cb_push_back(cb_out, 3);
        tile_regs_release();

        cb_pop_front(cb_p, 2);
        cb_pop_front(cb_q, 2);
    }
}
