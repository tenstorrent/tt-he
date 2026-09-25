// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Vector Montgomery modular multiplication, full kernel.
// Computes out = (a * b) mod q  (canonical [0, q)), where a, b are
// Montgomery-domain residues in lazy form [0, 2q), q a 28-bit prime.
//
// Same algorithm as cpu/montgomery.hpp: mul_hi_lo (16-bit limbs) then
// reduce_optimized, then a branchless finalize. Built entirely from proven
// Wormhole int SFPU tile-ops. All right shifts are LOGICAL (the unary
// right_shift_tile is arithmetic, so we use binary_logical_right_shift_tile
// with a shift-amount tile = 16). DST budget: 8 int32 tiles (d0..d6 working,
// d7 = constant 16), enabled by fp32_dest_acc_en + dst_full_sync_en on host.
//
// Runtime args: 0=n_tiles, 1=q, 2=q_inv_32, 3=q_low16, 4=q_high16.

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/binary_max_min.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr auto FMT = DataFormat::UInt32;

// init+op wrappers (each SFPU op re-initializes its mode/constants).
inline void DUP(uint32_t in, uint32_t out) { copy_dest_values_init(); copy_dest_values<FMT>(in, out); }
inline void SHR(uint32_t in, uint32_t sh, uint32_t out) { binary_shift_tile_init(); binary_logical_right_shift_tile<FMT>(in, sh, out); }
inline void SHL(uint32_t in, uint32_t sh, uint32_t out) { binary_shift_tile_init(); binary_left_shift_tile<FMT>(in, sh, out); }
// x & 0xFFFF  ==  (x << 16) >> 16  (logical), using the shift-amount tile at index 7.
inline void LOW16(uint32_t d) { SHL(d, 7, d); SHR(d, 7, d); }
inline void MUL(uint32_t a, uint32_t b, uint32_t o) { mul_int_tile_init<FMT>(); mul_int_tile<FMT>(a, b, o); }
inline void ADD(uint32_t a, uint32_t b, uint32_t o) { add_int_tile_init(); add_int_tile<FMT>(a, b, o); }
inline void SUB(uint32_t a, uint32_t b, uint32_t o) { sub_int_tile_init(); sub_int_tile<FMT>(a, b, o); }  // o = a - b
inline void OR(uint32_t a, uint32_t b, uint32_t o) { binary_bitwise_tile_init(); bitwise_or_binary_tile<FMT>(a, b, o); }
inline void MIN(uint32_t a, uint32_t b, uint32_t o) { binary_min_uint32_tile_init(); binary_min_uint32_tile(a, b, o); }
inline void FILL(uint32_t d, uint32_t val) { fill_tile_init(); fill_tile_int<FMT>(d, val); }
}  // namespace

void kernel_main() {
    DeviceZoneScopedN("mul_modq");
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t Q = get_arg_val<uint32_t>(1);
    const uint32_t QINV = get_arg_val<uint32_t>(2);
    const uint32_t QLOW = get_arg_val<uint32_t>(3);
    const uint32_t QHIGH = get_arg_val<uint32_t>(4);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t SH = 7;  // DST tile holding the shift amount (16)

    unary_op_init_common(cb_in0, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        FILL(SH, 16);  // d7 = 16 (logical-shift amount)

        // ---- extract 16-bit limbs of a and b ----
        copy_tile_init(cb_in0); copy_tile(cb_in0, 0, 0);  // d0 = a
        DUP(0, 1);                                        // d1 = a
        LOW16(0);                  // d0 = aL
        SHR(1, SH, 1);                                    // d1 = aH
        copy_tile_init(cb_in1); copy_tile(cb_in1, 0, 2);  // d2 = b
        DUP(2, 3);                                        // d3 = b
        LOW16(2);                  // d2 = bL
        SHR(3, SH, 3);                                    // d3 = bH

        // ---- base products ----
        MUL(0, 2, 4);   // d4 = ll = aL*bL
        MUL(1, 3, 5);   // d5 = hh = aH*bH
        MUL(0, 3, 6);   // d6 = aL*bH
        MUL(1, 2, 0);   // d0 = aH*bL
        ADD(6, 0, 6);   // d6 = cross = aL*bH + aH*bL

        // ---- mid = (ll>>16) + (cross & 0xFFFF) ----
        DUP(4, 0); SHR(0, SH, 0);    // d0 = ll>>16
        DUP(6, 1); LOW16(1);                  // d1 = cross & 0xFFFF
        ADD(0, 1, 0);                // d0 = mid

        // ---- lo = (ll & 0xFFFF) | (mid<<16) ----
        DUP(4, 1); LOW16(1);                  // d1 = ll & 0xFFFF
        DUP(0, 2); SHL(2, SH, 2);    // d2 = mid<<16  (low 16 of mid, <<16)
        OR(1, 2, 1);                 // d1 = lo   (kept)

        // ---- hi = hh + (cross>>16) + (mid>>16) ----
        DUP(6, 2); SHR(2, SH, 2);    // d2 = cross>>16
        ADD(5, 2, 5);                // d5 = hh + cross>>16
        DUP(0, 2); SHR(2, SH, 2);    // d2 = mid>>16
        ADD(5, 2, 5);                // d5 = hi   (kept)
        // live: lo=d1, hi=d5, sh=d7

        // ---- t = lo * q_inv_32 (low 32) ----
        FILL(0, QINV);   // d0 = q_inv_32
        MUL(1, 0, 1);    // d1 = t
        DUP(1, 2); LOW16(2);                  // d2 = tL
        DUP(1, 3); SHR(3, SH, 3);    // d3 = tH

        // ---- the four t*q limb products ----
        FILL(0, QHIGH);  // d0 = q_high
        FILL(1, QLOW);   // d1 = q_low  (t no longer needed)
        MUL(3, 0, 4);    // d4 = pHH = tH*q_high
        MUL(3, 1, 6);    // d6 = pMH = tH*q_low
        MUL(2, 0, 0);    // d0 = pML = tL*q_high
        MUL(2, 1, 1);    // d1 = pLL = tL*q_low
        // live: pHH=d4, pMH=d6, pML=d0, pLL=d1, hi=d5, sh=d7

        // ---- mid_low = (pMH&m) + (pML&m) + (pLL>>16) ----
        DUP(6, 2); LOW16(2);                  // d2 = pMH & 0xFFFF
        DUP(0, 3); LOW16(3);                  // d3 = pML & 0xFFFF
        ADD(2, 3, 2);                // d2 = sum
        DUP(1, 3); SHR(3, SH, 3);    // d3 = pLL>>16
        ADD(2, 3, 2);                // d2 = mid_low

        // ---- mid_high = (pMH>>16) + (pML>>16) + (mid_low>>16) ----
        DUP(6, 3); SHR(3, SH, 3);    // d3 = pMH>>16
        DUP(0, 6); SHR(6, SH, 6);    // d6 = pML>>16
        ADD(3, 6, 3);                // d3 = sum
        DUP(2, 6); SHR(6, SH, 6);    // d6 = mid_low>>16
        ADD(3, 6, 3);                // d3 = mid_high

        // ---- t_final = pHH + mid_high ----
        ADD(4, 3, 4);                // d4 = t_final
        // live: t_final=d4, hi=d5, sh=d7

        // ---- b_out = hi + q - t_final  (in [0, 2q)) ----
        FILL(0, Q);      // d0 = q
        ADD(5, 0, 5);    // d5 = hi + q
        SUB(5, 4, 5);    // d5 = b_out

        // ---- finalize: min(b_out, b_out - q) -> [0, q) ----
        DUP(5, 1); SUB(1, 0, 1);     // d1 = b_out - q (wraps if b_out<q)
        MIN(5, 1, 5);                // d5 = result

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(5, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
