// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Multi-core matmul reader: this core computes ONE output plane `oi` of the int8
// plane-matmul (paired with compute/matmul_plane.cpp). It feeds the NB (a_k, b_k)
// pairs whose products accumulate into plane `oi`.
//
// Two orientations select which tile of the 16-tile (NB*NB) operand belongs to
// plane oi at contraction step k:
//
//   mode 0  (stage A, fold into the LHS): a = folded W (NB*NB), b = Xin limbs (NB)
//           C_p = sum_r a[p*NB+r] @ b[r]        -> a_idx = oi*NB+k,  b_idx = k
//   mode 1  (stage C, fold into the RHS): a = G' limbs (NB), b = folded W (NB*NB)
//           C_r = sum_p a[p] @ b[p*NB+r]        -> a_idx = k,        b_idx = k*NB+oi
//
// One operand is the 16-tile folded constant, the other is the 4-tile dynamic limbs.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t oi     = get_arg_val<uint32_t>(2);   // this core's output plane
    const uint32_t mode   = get_arg_val<uint32_t>(3);   // 0 = fold-LHS (A), 1 = planes (C)

    constexpr uint32_t cb_in0 = 0;   // LHS, Int8
    constexpr uint32_t cb_in1 = 1;   // RHS, Int8
    constexpr uint32_t NB = 4;

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    for (uint32_t k = 0; k < NB; ++k) {
        const uint32_t a_idx = (mode == 0) ? (oi * NB + k) : k;
        const uint32_t b_idx = (mode == 0) ? k : (k * NB + oi);
        cb_reserve_back(cb_in0, 1);
        noc_async_read_tile(a_idx, a, get_write_ptr(cb_in0));
        cb_reserve_back(cb_in1, 1);
        noc_async_read_tile(b_idx, b, get_write_ptr(cb_in1));
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
        cb_push_back(cb_in1, 1);
    }
}
