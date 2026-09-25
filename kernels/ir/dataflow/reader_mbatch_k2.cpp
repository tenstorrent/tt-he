// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for matmul_modq_mbatch_k2 (2-block K-accumulation): feeds BOTH contraction blocks.
//   c_0 <- 2*M Xin tiles: block 0 from x_base0, block 1 from x_base1 (M each, in that order, so the
//         compute kernel finds block blk's limbs at (blk*M + m))
//   c_1 <- the folded W for block 0 then block 1, M*16 tiles each, streamed in phase-2 consumption
//         order [m][p][blk][r] — the compute kernel's block loop is INSIDE its p loop, so the two
//         blocks' W tiles must interleave per (m,p), not follow one another per block.
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
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t p = 0; p < NB; ++p)
            for (uint32_t blk = 0; blk < 2; ++blk)
                for (uint32_t r = 0; r < NB; ++r) {
                    cb_reserve_back(cb_w, 1);
                    const uint32_t idx = (m * NB + p) * NB + r;
                    if (blk == 0) noc_async_read_tile(idx, w0, get_write_ptr(cb_w));
                    else          noc_async_read_tile(idx, w1, get_write_ptr(cb_w));
                    noc_async_read_barrier();
                    cb_push_back(cb_w, 1);
                }
}
