// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reads the KLIMBS radix-2^28 limb streams for gadget_decompose_signed into CB c_0, interleaved
// per coefficient-tile: (X0[t], X1[t], ...) for t = 0..n_tiles-1.  KLIMBS DRAM buffers, int32.
// K=2 (~56-bit q) is the default; K=3 (~84-bit q) reads a third (mid) limb buffer.
#include <cstdint>

#ifndef GADGET_KLIMBS
#define GADGET_KLIMBS 2
#endif

void kernel_main() {
    constexpr uint32_t KLIMBS = GADGET_KLIMBS;
    uint32_t addr[GADGET_KLIMBS];
    for (uint32_t j = 0; j < KLIMBS; ++j) addr[j] = get_arg_val<uint32_t>(j);
    uint32_t n_tiles = get_arg_val<uint32_t>(KLIMBS);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t ts = get_tile_size(cb_in);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        for (uint32_t j = 0; j < KLIMBS; ++j) {
            const InterleavedAddrGenFast<true> s = {
                .bank_base_address = addr[j], .page_size = ts, .data_format = DataFormat::UInt32};
            cb_reserve_back(cb_in, 1);
            noc_async_read_tile(t, s, get_write_ptr(cb_in));
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
}
