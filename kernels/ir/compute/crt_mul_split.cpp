// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// CRT-combine (#13) stage C — reconstruct the 56-bit value and split into radix-2^28 limbs.
//   X = q1·t + r1,   t = diff·q1^{-1} mod q2 (< q2),  r1 = x mod q1 (< q1),  X in [0, q).
//   out: X0 = X & (2^28-1),  X1 = X >> 28   (the two limbs gadget_decompose_signed consumes).
//
// q1·t is a 28×28->56 widening multiply (no reduction): split t into 16-bit halves, multiply by
// the constant q1's halves (mul_int_tile), combine to (lo32, hi), add r1 via 16-bit limbs
// (carry-safe), then extract the two 28-bit limbs. SFPI glue + mul_int_tile, like recombine_rns.
//
// c_0 = t, c_1 = r1, c_2 = params [q1_low16 tile, q1_high16 tile].  out c_16 = (X0, X1) per tile.
// Runtime arg: 0 = n_tiles.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/compute/mul_int_sfpu.h"

namespace {
constexpr auto FMT = DataFormat::UInt32;
inline void MUL(uint32_t a, uint32_t b, uint32_t o) { mul_int_tile_init<FMT>(); mul_int_tile<FMT>(a, b, o); }
}  // namespace

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;
constexpr uint32_t M16 = 0xFFFFu;
constexpr uint32_t M28 = (1u << 28) - 1u;

// split t into 16-bit halves
inline void face_split16(uint32_t ts, uint32_t tLs, uint32_t tHs) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vt = sfpi::dst_reg[ts * NV + i];
        sfpi::dst_reg[tLs * NV + i] = vt & M16;
        sfpi::dst_reg[tHs * NV + i] = vt >> 16u;
    }
}

// (ll, hh, p1, p2, r1) -> X0, X1 where prod = ll + (p1+p2)<<16 + hh<<32 = q1·t, X = prod + r1.
inline void face_combine_split(uint32_t lls, uint32_t hhs, uint32_t p1s,
                               uint32_t p2s, uint32_t r1s, uint32_t X0s, uint32_t X1s) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt ll = sfpi::dst_reg[lls * NV + i];
        sfpi::vUInt hh = sfpi::dst_reg[hhs * NV + i];
        sfpi::vUInt cross = sfpi::vUInt(sfpi::dst_reg[p1s * NV + i]) + sfpi::vUInt(sfpi::dst_reg[p2s * NV + i]);
        sfpi::vUInt r1 = sfpi::dst_reg[r1s * NV + i];
        // 32x32->64 combine (montgomery.hpp mul_hi_lo): mid<<16 truncates, mid>>16 carries to hi
        sfpi::vUInt mid  = (ll >> 16u) + (cross & M16);
        sfpi::vUInt lo32 = (ll & M16) | (mid << 16u);
        sfpi::vUInt hi   = hh + (cross >> 16u) + (mid >> 16u);
        // X = prod + r1, adding r1 via 16-bit limbs so the carry into hi is explicit
        sfpi::vUInt sL = (lo32 & M16) + (r1 & M16);
        sfpi::vUInt sH = (lo32 >> 16u) + (r1 >> 16u) + (sL >> 16u);
        sfpi::vUInt newlo = (sL & M16) | ((sH & M16) << 16u);
        hi = hi + (sH >> 16u);
        // limbs: X0 = X & (2^28-1); X1 = X >> 28 = (newlo>>28) | (hi<<4)
        sfpi::dst_reg[X0s * NV + i] = newlo & M28;
        sfpi::dst_reg[X1s * NV + i] = (newlo >> 28u) | (hi << 4u);
    }
}
}  // namespace
#endif

inline void S_split16(uint32_t ts, uint32_t tLs, uint32_t tHs) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_split16, ts, tLs, tHs, ckernel::VectorMode::RC));
}
inline void S_combine_split(uint32_t lls, uint32_t hhs, uint32_t p1s, uint32_t p2s,
                            uint32_t r1s, uint32_t X0s, uint32_t X1s) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_combine_split, lls, hhs, p1s,
                                               ckernel::VectorMode::RC, p2s, r1s, X0s, X1s));
}

void kernel_main() {
    DeviceZoneScopedN("crt_mul_split");
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_t = tt::CBIndex::c_0, cb_r1 = tt::CBIndex::c_1, cb_par = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t PQL = 0, PQH = 1;   // param tiles: q1_low16, q1_high16

    unary_op_init_common(cb_t, cb_out);

    cb_wait_front(cb_par, 2);
    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_t, 1);
        cb_wait_front(cb_r1, 1);
        cb_reserve_back(cb_out, 2);

        tile_regs_acquire();
        copy_tile_init(cb_t);  copy_tile(cb_t, 0, 0);        // d0 = t
        copy_tile_init(cb_r1); copy_tile(cb_r1, 0, 1);       // d1 = r1
        S_split16(0, 2, 3);                                   // d2 = tL, d3 = tH
        copy_tile_init(cb_par); copy_tile(cb_par, PQL, 4); copy_tile(cb_par, PQH, 5);  // d4=q1L, d5=q1H
        MUL(2, 4, 6);   // ll = tL·q1L -> d6
        MUL(3, 5, 7);   // hh = tH·q1H -> d7
        MUL(2, 5, 2);   // p1 = tL·q1H -> d2
        MUL(3, 4, 3);   // p2 = tH·q1L -> d3
        S_combine_split(6, 7, 2, 3, 1, 0, 4);                 // -> X0=d0, X1=d4
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out, 0);
        pack_tile(4, cb_out, 1);
        cb_push_back(cb_out, 2);
        tile_regs_release();

        cb_pop_front(cb_t, 1);
        cb_pop_front(cb_r1, 1);
    }
    cb_pop_front(cb_par, 2);
}
