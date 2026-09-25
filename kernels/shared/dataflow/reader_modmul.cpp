// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reads two uint32 input vectors (A, B) tile-by-tile from DRAM into CBs.
#include <cstdint>

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);
    uint32_t start_tile = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    const InterleavedAddrGenFast<true> a = {
        .bank_base_address = a_addr, .page_size = tile_size_bytes, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> b = {
        .bank_base_address = b_addr, .page_size = tile_size_bytes, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t tile_id = start_tile + i;
        cb_reserve_back(cb_in0, 1);
        cb_reserve_back(cb_in1, 1);
        noc_async_read_tile(tile_id, a, get_write_ptr(cb_in0));
        noc_async_read_tile(tile_id, b, get_write_ptr(cb_in1));
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
        cb_push_back(cb_in1, 1);
    }
}
