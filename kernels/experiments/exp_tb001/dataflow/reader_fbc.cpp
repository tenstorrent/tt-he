// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for matmul_fbc (N coefficient-tiles). Per coeff-tile: A_ct -> c_0, then
// B_0..3 (folded CRT constants, re-streamed) -> c_1. Uses TensorAccessor because
// the int8 A buffer spans > 12 DRAM banks (InterleavedAddrGenFast mis-addresses
// sub-4096-byte int8 tiles past bank 12).
// Compile args: TensorAccessorArgs(A) then TensorAccessorArgs(B).
// Runtime args: 0 = a_addr, 1 = b_addr, 2 = nct.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t nct = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1;
    constexpr uint32_t NR = 4;

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr, get_tile_size(cb_a));
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr, get_tile_size(cb_b));

    for (uint32_t ct = 0; ct < nct; ++ct) {
        cb_reserve_back(cb_a, 1);
        noc_async_read_tile(ct, a, get_write_ptr(cb_a));
        noc_async_read_barrier();
        cb_push_back(cb_a, 1);
        for (uint32_t r = 0; r < NR; ++r) {
            cb_reserve_back(cb_b, 1);
            noc_async_read_tile(r, b, get_write_ptr(cb_b));
            noc_async_read_barrier();
            cb_push_back(cb_b, 1);
        }
    }
}
