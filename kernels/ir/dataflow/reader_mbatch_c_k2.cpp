// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for matmul_modq_mbatch_c_k2 (stage C, 2-block K-accumulation).
//   c_0 <- 2*M Xin tiles: block 0 from x_base0 then block 1 from x_base1 (compute finds block blk's
//          limbs at (blk*M + m))
//   c_1 <- 2*16 shared Wrhs tiles, block 0's set then block 1's, resident for the whole phase
//   Runtime args: 0=xin, 1=w0, 2=w1, 3=x_base0, 4=x_base1, 5=M.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t xin_addr = get_arg_val<uint32_t>(0);
    const uint32_t w0_addr  = get_arg_val<uint32_t>(1);
    const uint32_t w1_addr  = get_arg_val<uint32_t>(2);
    const uint32_t x_base0  = get_arg_val<uint32_t>(3);
    const uint32_t x_base1  = get_arg_val<uint32_t>(4);
    const uint32_t M        = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_in = 0, cb_w = 1, NB = 4;
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, xin_addr);
    constexpr auto w0_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto w0 = TensorAccessor(w0_args, w0_addr);
    constexpr auto w1_args = TensorAccessorArgs<w0_args.next_compile_time_args_offset()>();
    const auto w1 = TensorAccessor(w1_args, w1_addr);

    for (uint32_t blk = 0; blk < 2; ++blk) {
        const uint32_t base = blk ? x_base1 : x_base0;
        for (uint32_t m = 0; m < M; ++m) {
            cb_reserve_back(cb_in, 1);
            noc_async_read_tile(base + m, x, get_write_ptr(cb_in));
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
    for (uint32_t blk = 0; blk < 2; ++blk)
        for (uint32_t i = 0; i < NB * NB; ++i) {
            cb_reserve_back(cb_w, 1);
            if (blk == 0) noc_async_read_tile(i, w0, get_write_ptr(cb_w));
            else          noc_async_read_tile(i, w1, get_write_ptr(cb_w));
            noc_async_read_barrier();
            cb_push_back(cb_w, 1);
        }
}
