// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writes the uint32 output vector (C) tile-by-tile from CB back to DRAM.
#include <cstdint>

void kernel_main() {
    uint32_t c_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);

    const InterleavedAddrGenFast<true> c = {
        .bank_base_address = c_addr, .page_size = tile_size_bytes, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t tile_id = start_tile + i;
        cb_wait_front(cb_out0, 1);
        noc_async_write_tile(tile_id, c, get_read_ptr(cb_out0));
        noc_async_write_barrier();
        cb_pop_front(cb_out0, 1);
    }
}
