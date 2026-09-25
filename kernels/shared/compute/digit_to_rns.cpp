// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// External-product (#16) step 4 — split a signed gadget digit into RNS channels so it can be
// NTT'd per channel.  Digit d in [-B/2, B/2), |d| < every q_j, so  d mod q_j = d (d>=0) or d+q_j.
//   in c_0  = signed digit tile (int32)
//   out c_16 = DR_K tiles: (d mod q1, d mod q2, [d mod q3])
// Pure SFPI (conditional add). Runtime args: 0 = n_tiles, 1 = q1, 2 = q2, [3 = q3].
// DR_K = RNS limb count K (2 -> ~56-bit q, 3 -> ~84-bit q).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifndef DR_K
#define DR_K 2
#endif

namespace { constexpr auto FMT = DataFormat::UInt32; }

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;
// dst0=digit slot, dst1/dst2 = residue slots 1/2; trailing args carry the primes (+ slot3/Q3 for K=3).
inline void face_to_rns(uint32_t ds, uint32_t r1s, uint32_t r2s, uint32_t Q1, uint32_t Q2,
                        uint32_t Q3, uint32_t r3s) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt du = sfpi::dst_reg[ds * NV + i];
        sfpi::vInt d = sfpi::reinterpret<sfpi::vInt>(du);
        sfpi::vInt a = d, b = d;
#if DR_K >= 3
        sfpi::vInt c = d;
        v_if(d < 0) { a = d + (int)Q1; b = d + (int)Q2; c = d + (int)Q3; } v_endif;
        sfpi::dst_reg[r3s * NV + i] = sfpi::reinterpret<sfpi::vUInt>(c);
#else
        v_if(d < 0) { a = d + (int)Q1; b = d + (int)Q2; } v_endif;
        (void)Q3; (void)r3s;
#endif
        sfpi::dst_reg[r1s * NV + i] = sfpi::reinterpret<sfpi::vUInt>(a);
        sfpi::dst_reg[r2s * NV + i] = sfpi::reinterpret<sfpi::vUInt>(b);
    }
}
}  // namespace
#endif

inline void S_to_rns(uint32_t ds, uint32_t r1s, uint32_t r2s, uint32_t Q1, uint32_t Q2,
                     uint32_t Q3, uint32_t r3s) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_to_rns, ds, r1s, r2s, ckernel::VectorMode::RC,
                                               Q1, Q2, Q3, r3s));
}

void kernel_main() {
    DeviceZoneScopedN("digit_to_rns");
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t Q1 = get_arg_val<uint32_t>(1);
    const uint32_t Q2 = get_arg_val<uint32_t>(2);
#if DR_K >= 3
    const uint32_t Q3 = get_arg_val<uint32_t>(3);
#else
    const uint32_t Q3 = 0;
#endif
    constexpr auto cb_in = tt::CBIndex::c_0, cb_out = tt::CBIndex::c_16;
    constexpr uint32_t K = DR_K;

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, K);

        tile_regs_acquire();
        copy_tile_init(cb_in); copy_tile(cb_in, 0, 0);
        S_to_rns(0, 1, 2, Q1, Q2, Q3, 3);       // digit slot 0 -> residue slots 1,2,[3]
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t j = 0; j < K; ++j) pack_tile(1 + j, cb_out, j);
        cb_push_back(cb_out, K);
        tile_regs_release();

        cb_pop_front(cb_in, 1);
    }
}
