// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Multi-Q Montgomery modmul: out[i] = (a[i]*b[i]) mod q[i], where the modulus is PER ELEMENT.
// Same algorithm as modmul_compute, but q / q_inv_32 / q_low16 / q_high16 are supplied as TILES
// (loaded once into CBs) instead of scalar runtime args — so each element reduces mod its own
// prime. The prime tile shares the data tile's layout, so this handles per-row (limb-major),
// per-tile, and uniform primes with no row/lane bookkeeping. Enables limb-major Fusion A.
//
// CBs: c_0=a, c_1=b (per tile) ; c_2=q, c_3=q_inv, c_4=q_low16, c_5=q_high16 (params, once) ; c_16=out.
// Runtime arg: 0 = n_tiles.

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
inline void FILL(uint32_t d, uint32_t v) { fill_tile_init(); fill_tile_int<FMT>(d, v); }
// load a param tile into DST slot `d` (per-element modulus constants)
inline void LOADP(uint32_t cb, uint32_t d) { copy_tile_init(cb); copy_tile(cb, 0, d); }
}  // namespace

void kernel_main() {
    DeviceZoneScopedN("mul_modq_multiq");
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1;
    constexpr auto cb_q = tt::CBIndex::c_2, cb_qi = tt::CBIndex::c_3, cb_ql = tt::CBIndex::c_4, cb_qh = tt::CBIndex::c_5;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t SH = 7;

    unary_op_init_common(cb_a, cb_out);
    cb_wait_front(cb_q, 1); cb_wait_front(cb_qi, 1); cb_wait_front(cb_ql, 1); cb_wait_front(cb_qh, 1);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_a, 1); cb_wait_front(cb_b, 1); cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        FILL(SH, 16);
        copy_tile_init(cb_a); copy_tile(cb_a, 0, 0); DUP(0, 1); LOW16(0); SHR(1, SH, 1);
        copy_tile_init(cb_b); copy_tile(cb_b, 0, 2); DUP(2, 3); LOW16(2); SHR(3, SH, 3);
        MUL(0, 2, 4); MUL(1, 3, 5); MUL(0, 3, 6); MUL(1, 2, 0); ADD(6, 0, 6);
        DUP(4, 0); SHR(0, SH, 0); DUP(6, 1); LOW16(1); ADD(0, 1, 0);
        DUP(4, 1); LOW16(1); DUP(0, 2); SHL(2, SH, 2); OR(1, 2, 1);
        DUP(6, 2); SHR(2, SH, 2); ADD(5, 2, 5); DUP(0, 2); SHR(2, SH, 2); ADD(5, 2, 5);
        LOADP(cb_qi, 0); MUL(1, 0, 1); DUP(1, 2); LOW16(2); DUP(1, 3); SHR(3, SH, 3);   // t = lo·q_inv
        LOADP(cb_qh, 0); LOADP(cb_ql, 1); MUL(3, 0, 4); MUL(3, 1, 6); MUL(2, 0, 0); MUL(2, 1, 1);
        DUP(6, 2); LOW16(2); DUP(0, 3); LOW16(3); ADD(2, 3, 2); DUP(1, 3); SHR(3, SH, 3); ADD(2, 3, 2);
        DUP(6, 3); SHR(3, SH, 3); DUP(0, 6); SHR(6, SH, 6); ADD(3, 6, 3); DUP(2, 6); SHR(6, SH, 6); ADD(3, 6, 3);
        ADD(4, 3, 4);
        LOADP(cb_q, 0); ADD(5, 0, 5); SUB(5, 4, 5);            // b_out = hi + q - t_final   in [0,2q)
        DUP(5, 1); SUB(1, 0, 1); MIN(5, 1, 5);                 // finalize -> [0,q)  (q still in slot 0)
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(5, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1); cb_pop_front(cb_a, 1); cb_pop_front(cb_b, 1);
    }
    cb_pop_front(cb_q, 1); cb_pop_front(cb_qi, 1); cb_pop_front(cb_ql, 1); cb_pop_front(cb_qh, 1);
}
