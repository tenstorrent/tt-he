// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for fused_matmul_a2 (2-block K-accumulated stage-A/C).  Feeds:
//   Xin[0], Xin[1]  (2 int32 tiles)                          -> c_0
//   folded W in [p][b][r] order (32 int8 tiles) from TWO W buffers (block 0, block 1) -> c_1
// The matmul phase consumes W as foldW[p, b*NB+r] = W_b[p*NB+r], so per output plane p it reads
// W_0[p,0..3] then W_1[p,0..3].  Limb/plane intermediates never touch DRAM.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t xin_addr = get_arg_val<uint32_t>(0);   // Xin (2 int32 tiles, contiguous)
    const uint32_t w0_addr  = get_arg_val<uint32_t>(1);   // folded W block 0 (NB*NB int8)
    const uint32_t w1_addr  = get_arg_val<uint32_t>(2);   // folded W block 1 (NB*NB int8)

    constexpr uint32_t cb_in = 0, cb_w = 1, NB = 4, BLK = 2;

    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, xin_addr);
    constexpr auto w0_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto w0 = TensorAccessor(w0_args, w0_addr);
    constexpr auto w1_args = TensorAccessorArgs<w0_args.next_compile_time_args_offset()>();
    const auto w1 = TensorAccessor(w1_args, w1_addr);

    for (uint32_t b = 0; b < BLK; ++b) {
        cb_reserve_back(cb_in, 1);
        noc_async_read_tile(b, x, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }

    for (uint32_t p = 0; p < NB; ++p) {
        for (uint32_t b = 0; b < BLK; ++b) {
            for (uint32_t r = 0; r < NB; ++r) {
                cb_reserve_back(cb_w, 1);
                const uint32_t idx = p * NB + r;
                if (b == 0) noc_async_read_tile(idx, w0, get_write_ptr(cb_w));
                else        noc_async_read_tile(idx, w1, get_write_ptr(cb_w));
                noc_async_read_barrier();
                cb_push_back(cb_w, 1);
            }
        }
    }
}
