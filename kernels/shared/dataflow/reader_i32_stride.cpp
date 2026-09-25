// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Strided int32 reader -> CB c_0.  Reads n_tiles tiles at ids (off + i*stride), i=0..n_tiles-1.
// Used to gather one tile-position across a batch of contiguously-stored 2-tile polys for a
// buffer-resident modadd_reduce (no host round-trip).
//   Runtime args: 0 = addr, 1 = n_tiles, 2 = stride, 3 = off.
#include <cstdint>

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t off = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t ts = get_tile_size(cb_in);
    const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_reserve_back(cb_in, 1);
        noc_async_read_tile(off + i * stride, s, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
