// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for matmul_fbc_ls16. The NR*NA int8 B-tiles (folded CRT constants) are pushed ONCE and
// stay resident; then NA A-tiles per coefficient-tile are streamed, starting at this core's slice.
// Uses TensorAccessor because the int8 buffers span > 12 DRAM banks (InterleavedAddrGenFast
// mis-addresses sub-4096-byte int8 tiles past bank 12).
//
// Compile args: TensorAccessorArgs(A) then TensorAccessorArgs(B).
// Runtime args: 0 = a_addr, 1 = b_addr, 2 = nct, 3 = NA, 4 = ct_start (coefficient-tiles).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t nct = get_arg_val<uint32_t>(2);
    const uint32_t NA = get_arg_val<uint32_t>(3);
    const uint32_t ct_start = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1;
    constexpr uint32_t NR = 4;

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr, get_tile_size(cb_a));
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr, get_tile_size(cb_b));
    const uint32_t ats = get_tile_size(cb_a), bts = get_tile_size(cb_b);

    cb_reserve_back(cb_b, NR * NA);
    {
        const uint32_t bp = get_write_ptr(cb_b);
        for (uint32_t i = 0; i < NR * NA; ++i) noc_async_read_tile(i, b, bp + i * bts);
        noc_async_read_barrier();
    }
    cb_push_back(cb_b, NR * NA);

    for (uint32_t ct = 0; ct < nct; ++ct) {
        cb_reserve_back(cb_a, NA);
        const uint32_t ap = get_write_ptr(cb_a);
        for (uint32_t s = 0; s < NA; ++s) noc_async_read_tile((ct_start + ct) * NA + s, a, ap + s * ats);
        noc_async_read_barrier();
        cb_push_back(cb_a, NA);
    }
}
