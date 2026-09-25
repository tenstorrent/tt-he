// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// TB001_PHASE_ZONES: the inner per-phase zones below are OPTIONAL profiler markers, and they
// triple-to-quadruple the marker volume of this kernel. At small G every dispatch lands on the
// same few cores, so the per-RISC DRAM marker buffer overflows and tt-metal drops markers --
// which breaks the 1-1 dispatch map the metrics rely on. They are therefore compiled out by
// default here; build with -DTB001_PHASE_ZONES to get the phase breakdown back.
// This is a divergence from kernels/ -- see kernels/experiments/exp_tb001/README.md.
//
// FUSED reduce + twiddle: out = mont_mul( (sum_i in_i) mod q , tw ) mod q.
//
// This is modadd_reduce.cpp followed by modmul_compute.cpp, merged into one kernel. In the forward
// NTT those two ran as separate programs, so the 8:1 reduction result made a full DRAM round trip
// (8 tiles written by the reducer, 8+1 read back by the twiddle) purely to hand one tile to the next
// program. Merged, the accumulator never leaves DEST: the reduce leaves it in d0 and the Montgomery
// multiply picks it up there instead of re-reading it through c_0.
//
// Saves per forward NTT: one program dispatch (~16 us of wall, of which only ~7 was compute) and one
// 8-tile DRAM write + 8-tile read.
//
//   c_0  n_in int32 partial tiles      c_1  1 int32 twiddle tile      c_16  1 int32 result
//
// Montgomery arithmetic is character-for-character the modmul_compute.cpp sequence (16-bit limb
// mul_hi_lo + reduce + branchless finalize); only the source of operand `a` differs. Inputs are
// Montgomery-domain residues, output is canonical [0, q).
//
// DST budget: 8 int32 tiles (fp32_dest_acc_en + dst_full_sync_en on the host side).
//   reduce phase uses d0 (acc), d1 (q), d2 (tmp); the multiply then uses d0..d7 with d7 = 16.
//
// Runtime args: 0=n_in, 1=q, 2=q_inv_32, 3=q_low16, 4=q_high16.

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

inline void DUP(uint32_t in, uint32_t out) { copy_dest_values_init(); copy_dest_values<FMT>(in, out); }
inline void SHR(uint32_t in, uint32_t sh, uint32_t out) { binary_shift_tile_init(); binary_logical_right_shift_tile<FMT>(in, sh, out); }
inline void SHL(uint32_t in, uint32_t sh, uint32_t out) { binary_shift_tile_init(); binary_left_shift_tile<FMT>(in, sh, out); }
inline void LOW16(uint32_t d) { SHL(d, 7, d); SHR(d, 7, d); }
inline void MUL(uint32_t a, uint32_t b, uint32_t o) { mul_int_tile_init<FMT>(); mul_int_tile<FMT>(a, b, o); }
inline void ADD(uint32_t a, uint32_t b, uint32_t o) { add_int_tile_init(); add_int_tile<FMT>(a, b, o); }
inline void SUB(uint32_t a, uint32_t b, uint32_t o) { sub_int_tile_init(); sub_int_tile<FMT>(a, b, o); }
inline void OR(uint32_t a, uint32_t b, uint32_t o) { binary_bitwise_tile_init(); bitwise_or_binary_tile<FMT>(a, b, o); }
inline void MIN(uint32_t a, uint32_t b, uint32_t o) { binary_min_uint32_tile_init(); binary_min_uint32_tile(a, b, o); }
inline void FILL(uint32_t d, uint32_t val) { fill_tile_init(); fill_tile_int<FMT>(d, val); }
}  // namespace

void kernel_main() {
    DeviceZoneScopedN("modadd_twiddle");
    const uint32_t n_in = get_arg_val<uint32_t>(0);
    const uint32_t Q = get_arg_val<uint32_t>(1);
    const uint32_t QINV = get_arg_val<uint32_t>(2);
    const uint32_t QLOW = get_arg_val<uint32_t>(3);
    const uint32_t QHIGH = get_arg_val<uint32_t>(4);

    constexpr auto cb_in = tt::CBIndex::c_0;    // n_in partials
    constexpr auto cb_tw = tt::CBIndex::c_1;    // twiddle
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t SH = 7;                  // DST tile holding the shift amount (16)

    unary_op_init_common(cb_in, cb_out);

    cb_wait_front(cb_in, n_in);
    cb_wait_front(cb_tw, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();

    // ===== phase 1: acc = (sum_i in_i) mod q, kept in d0 =====
    // Identical to modadd_reduce.cpp: each step adds (< 2q) then folds back via min(acc, acc-q).
    {
    
#ifdef TB001_PHASE_ZONES
    DeviceZoneScopedN("ph_reduce");
#endif
    FILL(1, Q);                                        // d1 = q
    copy_tile_init(cb_in); copy_tile(cb_in, 0, 0);     // d0 = acc = in_0
    for (uint32_t i = 1; i < n_in; ++i) {
        copy_tile(cb_in, i, 2);                        // d2 = in_i
        ADD(0, 2, 0);                                  // d0 = acc + in_i   (< 2q)
        SUB(0, 1, 2);                                  // d2 = acc - q      (wraps if acc < q)
        MIN(0, 2, 0);                                  // d0 = min -> [0, q)
    }
    }

    // ===== phase 2: out = mont_mul(acc, tw) — acc is ALREADY in d0, so no re-read =====
    {
    
#ifdef TB001_PHASE_ZONES
    DeviceZoneScopedN("ph_twiddle");
#endif
    FILL(SH, 16);                                      // d7 = 16 (logical-shift amount)

    // limbs of a (= acc, in d0) and b (= tw)
    DUP(0, 1);                                         // d1 = a
    LOW16(0);                                          // d0 = aL
    SHR(1, SH, 1);                                     // d1 = aH
    copy_tile_init(cb_tw); copy_tile(cb_tw, 0, 2);     // d2 = b
    DUP(2, 3);                                         // d3 = b
    LOW16(2);                                          // d2 = bL
    SHR(3, SH, 3);                                     // d3 = bH

    MUL(0, 2, 4);   // d4 = ll = aL*bL
    MUL(1, 3, 5);   // d5 = hh = aH*bH
    MUL(0, 3, 6);   // d6 = aL*bH
    MUL(1, 2, 0);   // d0 = aH*bL
    ADD(6, 0, 6);   // d6 = cross

    DUP(4, 0); SHR(0, SH, 0);        // d0 = ll>>16
    DUP(6, 1); LOW16(1);             // d1 = cross & 0xFFFF
    ADD(0, 1, 0);                    // d0 = mid

    DUP(4, 1); LOW16(1);             // d1 = ll & 0xFFFF
    DUP(0, 2); SHL(2, SH, 2);        // d2 = mid<<16
    OR(1, 2, 1);                     // d1 = lo

    DUP(6, 2); SHR(2, SH, 2);        // d2 = cross>>16
    ADD(5, 2, 5);
    DUP(0, 2); SHR(2, SH, 2);        // d2 = mid>>16
    ADD(5, 2, 5);                    // d5 = hi

    FILL(0, QINV);   // d0 = q_inv_32
    MUL(1, 0, 1);    // d1 = t
    DUP(1, 2); LOW16(2);             // d2 = tL
    DUP(1, 3); SHR(3, SH, 3);        // d3 = tH

    FILL(0, QHIGH);
    FILL(1, QLOW);
    MUL(3, 0, 4);    // d4 = pHH
    MUL(3, 1, 6);    // d6 = pMH
    MUL(2, 0, 0);    // d0 = pML
    MUL(2, 1, 1);    // d1 = pLL

    DUP(6, 2); LOW16(2);
    DUP(0, 3); LOW16(3);
    ADD(2, 3, 2);
    DUP(1, 3); SHR(3, SH, 3);
    ADD(2, 3, 2);                    // d2 = mid_low

    DUP(6, 3); SHR(3, SH, 3);
    DUP(0, 6); SHR(6, SH, 6);
    ADD(3, 6, 3);
    DUP(2, 6); SHR(6, SH, 6);
    ADD(3, 6, 3);                    // d3 = mid_high

    ADD(4, 3, 4);                    // d4 = t_final

    FILL(0, Q);
    ADD(5, 0, 5);                    // d5 = hi + q
    SUB(5, 4, 5);                    // d5 = b_out in [0, 2q)
    DUP(5, 1); SUB(1, 0, 1);         // d1 = b_out - q
    MIN(5, 1, 5);                    // d5 = result in [0, q)
    }

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(5, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, 1);
    cb_pop_front(cb_in, n_in);
    cb_pop_front(cb_tw, 1);
}
