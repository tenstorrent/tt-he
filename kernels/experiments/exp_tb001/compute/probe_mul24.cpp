// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SEMANTICS PROBE for SFPMUL24 (`sfpi::fractional_mul`), the Blackhole-only integer multiply.
//
// WHY. modmul_sfpi_full's header says "Wormhole SFPI has no integer multiply (and no sfpmul24), so
// each product is the fp32 byte-split trick" -- four fp32 multiplies plus twelve conversions per
// 16x16 product. Blackhole HAS the instruction: sfpi_lib.h exposes it as fractional_mul(a, b, Low |
// High) behind `#if __riscv_xtttensixbh`. tt-metal's own mul_int_tile does NOT use its width -- the
// Blackhole LLK loads LO16, takes MOD1_LOWER and stores LO16, so mul_int_tile is a 16x16 -> low-16
// multiply and nothing more.
//
// What the instruction actually computes has to be measured, not inferred: the name is
// `fractional_mul`, the return type is a sign-MAGNITUDE vMag, and the one in-tree wide-multiply
// (ckernel_sfpu_div_int32_floor.h) reassembles with a shift of 23, not 24. This kernel emits the raw
// Low and High halves for caller-supplied operands so the host can fit the model.
//
// c_0 = a, c_1 = b (uint32 tiles).  c_16 = 2 tiles: [0] = Low, [1] = High.
// Runtime arg 0 = n tile-pairs.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;
inline void face_mul24(uint32_t a, uint32_t b, uint32_t lo, uint32_t hi) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt va = sfpi::dst_reg[a * NV + i];
        sfpi::vUInt vb = sfpi::dst_reg[b * NV + i];
        sfpi::vInt l = sfpi::fractional_mul(va, vb, sfpi::FractionalHalf::Low);
        sfpi::vInt h = sfpi::fractional_mul(va, vb, sfpi::FractionalHalf::High);
        sfpi::dst_reg[lo * NV + i] = l;
        sfpi::dst_reg[hi * NV + i] = h;
    }
}
}  // namespace
#endif

inline void S_mul24(uint32_t a, uint32_t b, uint32_t lo, uint32_t hi) {
    // signature is (func, in0, in1, out, vector_mode, args...) -> func(in0, in1, out, args...)
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_mul24, a, b, lo, ckernel::VectorMode::RC, hi));
}

void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0);
    constexpr auto cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1, cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_a, cb_out);
    for (uint32_t t = 0; t < n; ++t) {
        cb_wait_front(cb_a, 1); cb_wait_front(cb_b, 1);
        cb_reserve_back(cb_out, 2);
        tile_regs_acquire();
        copy_tile_init(cb_a); copy_tile(cb_a, 0, 0);
        copy_tile_init(cb_b); copy_tile(cb_b, 0, 1);
        S_mul24(0, 1, 2, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(2, cb_out, 0);
        pack_tile(3, cb_out, 1);
        cb_push_back(cb_out, 2);
        tile_regs_release();
        cb_pop_front(cb_a, 1); cb_pop_front(cb_b, 1);
    }
}
