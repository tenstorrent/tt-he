// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused-stage-A reader: feeds the single fused decompose+matmul+recombine kernel.
//   Xin      (1 int32 tile)   -> c_0
//   folded W (NB*NB int8 tiles, (p,r) order) -> c_1
// The limb and plane intermediates never touch DRAM (they live in on-core CBs),
// so this reader only delivers the two true inputs.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t xin_addr = get_arg_val<uint32_t>(0);   // Xin (1 int32 tile)
    const uint32_t w_addr = get_arg_val<uint32_t>(1);     // folded W (NB*NB int8 tiles)

    constexpr uint32_t cb_in = 0;   // Xin, Int32
    constexpr uint32_t cb_w = 1;    // folded W, Int8
    constexpr uint32_t NB = 4;

    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, xin_addr);
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto w = TensorAccessor(w_args, w_addr);

    cb_reserve_back(cb_in, 1);
    noc_async_read_tile(0, x, get_write_ptr(cb_in));
    noc_async_read_barrier();
    cb_push_back(cb_in, 1);

    for (uint32_t i = 0; i < NB * NB; ++i) {
        cb_reserve_back(cb_w, 1);
        noc_async_read_tile(i, w, get_write_ptr(cb_w));
        noc_async_read_barrier();
        cb_push_back(cb_w, 1);
    }
}
