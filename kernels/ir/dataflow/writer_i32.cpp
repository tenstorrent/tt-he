// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writes n int32 tiles from CB c_16 to DRAM.
#include <cstdint>

void kernel_main() {
    uint32_t addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t ts = get_tile_size(cb_out);
    const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(start + i, s, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
