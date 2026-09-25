// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// General two-operand reader: A from a_addr at (a_start+i), B from b_addr at
// (b_start+i), for i in [0,n). Independent A/B tile offsets (unlike reader_modmul,
// which reads both at the same id). A -> c_0, B -> c_1.
// Args: 0=a_addr, 1=b_addr, 2=n, 3=a_start, 4=b_start.

#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    const uint32_t a_start = get_arg_val<uint32_t>(3);
    const uint32_t b_start = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1;
    const uint32_t ts = get_tile_size(cb_a);
    const InterleavedAddrGenFast<true> a = {.bank_base_address = a_addr, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> b = {.bank_base_address = b_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n; ++i) {
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);
        noc_async_read_tile(a_start + i, a, get_write_ptr(cb_a));
        noc_async_read_tile(b_start + i, b, get_write_ptr(cb_b));
        noc_async_read_barrier();
        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }
}
