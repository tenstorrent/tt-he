// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Two-operand streaming reader with ONE barrier per tile pair instead of two.
//
// `reader_ab` barriers after each of the two reads; here both are issued and then awaited together,
// which is the difference between serialising the two DRAM round trips and overlapping them. Same
// arguments, so it is a drop-in for a benchmark that wants the dataflow out of the way.
//   Args: 0=a_addr, 1=b_addr, 2=n, 3=a_start, 4=b_start, 5=stride (tile i is at start+i*stride,
//   so a core can walk the S axis of an (S, C, 32, 32) tensor with stride C).

#include <cstdint>

void kernel_main() {
    const uint32_t a_addr  = get_arg_val<uint32_t>(0);
    const uint32_t b_addr  = get_arg_val<uint32_t>(1);
    const uint32_t n       = get_arg_val<uint32_t>(2);
    const uint32_t a_start = get_arg_val<uint32_t>(3);
    const uint32_t b_start = get_arg_val<uint32_t>(4);
    const uint32_t stride  = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1;
    const uint32_t ts = get_tile_size(cb_a);
    const InterleavedAddrGenFast<true> a = {.bank_base_address = a_addr, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> b = {.bank_base_address = b_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n; ++i) {
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);
        noc_async_read_tile(a_start + i * stride, a, get_write_ptr(cb_a));
        noc_async_read_tile(b_start + i * stride, b, get_write_ptr(cb_b));
        noc_async_read_barrier();                      // both reads in flight, one wait
        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }
}
