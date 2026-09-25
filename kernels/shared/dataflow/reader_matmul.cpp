// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the 32x32 modular matmul (see docs/matmul/history/PLAN.md).
//
// The 32x32 logical problem is tiled into NB x NB = 4 x 4 blocks of 8.
//   - A is decomposed into NB LHS limb-tiles L_bk          (A DRAM buffer, NB tiles)
//   - B is precomputed into NB*NB RHS limb-tiles R_{bk,bj} (B DRAM buffer, NB*NB tiles)
//
// For each output column-group bj we accumulate over the NB contraction blocks bk,
// so we stream the (bj, bk) operand pairs in bj-major / bk-minor order. The compute
// kernel consumes them in the same order.
//
//   L tile index in A buffer  = bk
//   R tile index in B buffer  = bj * NB + bk

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in0 = 0;   // LHS (A limbs), Int8
    constexpr uint32_t cb_in1 = 1;   // RHS (B limbs), Int8
    constexpr uint32_t NB = 4;       // 32 / 8

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    for (uint32_t bj = 0; bj < NB; ++bj) {
        for (uint32_t bk = 0; bk < NB; ++bk) {
            cb_reserve_back(cb_in0, 1);
            noc_async_read_tile(bk, a, get_write_ptr(cb_in0));
            cb_reserve_back(cb_in1, 1);
            noc_async_read_tile(bj * NB + bk, b, get_write_ptr(cb_in1));
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
            cb_push_back(cb_in1, 1);
        }
    }
}
