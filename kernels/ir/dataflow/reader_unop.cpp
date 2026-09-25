// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// 1-operand reader: reads `in` at tile (start+i) -> c_0, for i in [0, n_tiles).
// Args: 0 = in_addr, 1 = n_tiles, 2 = start_tile.
#include <cstdint>

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t ts = get_tile_size(cb_in);
    const InterleavedAddrGenFast<true> a = {
        .bank_base_address = in_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_reserve_back(cb_in, 1);
        noc_async_read_tile(start_tile + i, a, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
