// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for fused_matmul_c2 (2-block K-accumulated stage-C, RHS fold).  Feeds:
//   Xin[0], Xin[1]  (2 int32 tiles)                     -> c_0
//   folded RHS W: block0's 16 tiles then block1's 16 (each (p,r) order) from 2 buffers -> c_1
// The matmul keeps all 32 W tiles resident and indexes them as b*16 + p*NB + r.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t xin_addr = get_arg_val<uint32_t>(0);   // Xin (2 int32 tiles)
    const uint32_t w0_addr  = get_arg_val<uint32_t>(1);   // RHS W block 0 (NB*NB int8)
    const uint32_t w1_addr  = get_arg_val<uint32_t>(2);   // RHS W block 1 (NB*NB int8)

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

    for (uint32_t b = 0; b < BLK; ++b) {
        for (uint32_t i = 0; i < NB * NB; ++i) {
            cb_reserve_back(cb_w, 1);
            if (b == 0) noc_async_read_tile(i, w0, get_write_ptr(cb_w));
            else        noc_async_read_tile(i, w1, get_write_ptr(cb_w));
            noc_async_read_barrier();
            cb_push_back(cb_w, 1);
        }
    }
}
