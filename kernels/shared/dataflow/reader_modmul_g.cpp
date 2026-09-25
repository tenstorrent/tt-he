// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Strided modmul reader (A,B at same tile id). args: 0=a,1=b,2=n,3=start,4=stride,5=group.
#include <cstdint>

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t n = get_arg_val<uint32_t>(2);
    uint32_t start = get_arg_val<uint32_t>(3);
    uint32_t stride = get_arg_val<uint32_t>(4);
    uint32_t group = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb0 = tt::CBIndex::c_0, cb1 = tt::CBIndex::c_1;
    const uint32_t ts = get_tile_size(cb0);
    const InterleavedAddrGenFast<true> a = {.bank_base_address = a_addr, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> b = {.bank_base_address = b_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t base = (start + i * stride) * group;
        for (uint32_t j = 0; j < group; ++j) {
            const uint32_t id = base + j;
            cb_reserve_back(cb0, 1); cb_reserve_back(cb1, 1);
            noc_async_read_tile(id, a, get_write_ptr(cb0));
            noc_async_read_tile(id, b, get_write_ptr(cb1));
            noc_async_read_barrier();
            cb_push_back(cb0, 1); cb_push_back(cb1, 1);
        }
    }
}
