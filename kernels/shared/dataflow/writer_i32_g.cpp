// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Strided/grouped int32 writer (from c_16).  Runtime args: 0=addr,1=n,2=start,3=stride,4=group.
#include <cstdint>

void kernel_main() {
    uint32_t addr = get_arg_val<uint32_t>(0);
    uint32_t n = get_arg_val<uint32_t>(1);
    uint32_t start = get_arg_val<uint32_t>(2);
    uint32_t stride = get_arg_val<uint32_t>(3);
    uint32_t group = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t ts = get_tile_size(cb_out);
    const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t base = (start + i * stride) * group;
        for (uint32_t j = 0; j < group; ++j) {
            cb_wait_front(cb_out, 1);
            noc_async_write_tile(base + j, s, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
