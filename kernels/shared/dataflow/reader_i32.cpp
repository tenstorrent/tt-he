// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reads n int32 tiles from DRAM into CB c_0.
#include <cstdint>

void kernel_main() {
    uint32_t addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t ts = get_tile_size(cb_in);
    const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_reserve_back(cb_in, 1);
        noc_async_read_tile(start + i, s, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
