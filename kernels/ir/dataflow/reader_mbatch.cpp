// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for matmul_modq_mbatch (M-batched, phase-major): feeds a core's M output tiles.
// All M Xin tiles -> c_0 first (phase 1 waits for all M), then all M·16 foldW tiles -> c_1
// (streamed; phase 2 consumes in [m][p][r] order = foldW buffer order).
//   Runtime args: 0 = xin_addr, 1 = w_addr, 2 = x_base (tile), 3 = w_base (tile), 4 = M.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t xin_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr   = get_arg_val<uint32_t>(1);
    const uint32_t x_base   = get_arg_val<uint32_t>(2);
    const uint32_t w_base   = get_arg_val<uint32_t>(3);
    const uint32_t M        = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in = 0, cb_w = 1, NB = 4;
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, xin_addr);
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto w = TensorAccessor(w_args, w_addr);

    for (uint32_t m = 0; m < M; ++m) {
        cb_reserve_back(cb_in, 1);
        noc_async_read_tile(x_base + m, x, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
    const uint32_t WT = M * NB * NB;
    for (uint32_t i = 0; i < WT; ++i) {
        cb_reserve_back(cb_w, 1);
        noc_async_read_tile(w_base + i, w, get_write_ptr(cb_w));
        noc_async_read_barrier();
        cb_push_back(cb_w, 1);
    }
}
