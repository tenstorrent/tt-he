// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Strided modmul writer (from c_16). args: 0=c,1=n,2=start,3=stride,4=group.
#include <cstdint>

void kernel_main() {
    uint32_t c_addr = get_arg_val<uint32_t>(0);
    uint32_t n = get_arg_val<uint32_t>(1);
    uint32_t start = get_arg_val<uint32_t>(2);
    uint32_t stride = get_arg_val<uint32_t>(3);
    uint32_t group = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb = tt::CBIndex::c_16;
    const uint32_t ts = get_tile_size(cb);
    const InterleavedAddrGenFast<true> c = {.bank_base_address = c_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t base = (start + i * stride) * group;
        for (uint32_t j = 0; j < group; ++j) {
            cb_wait_front(cb, 1);
            noc_async_write_tile(base + j, c, get_read_ptr(cb));
            noc_async_write_barrier();
            cb_pop_front(cb, 1);
        }
    }
}
