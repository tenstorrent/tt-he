// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reads the two radix-2^28 limb streams for gadget_decompose_signed into CB c_0, interleaved
// per tile: (X0[t], X1[t]) for t = 0..n_tiles-1.  Two DRAM buffers (limb 0, limb 1), int32.
#include <cstdint>

void kernel_main() {
    uint32_t addr0 = get_arg_val<uint32_t>(0);   // X0 (low limb) buffer
    uint32_t addr1 = get_arg_val<uint32_t>(1);   // X1 (high limb) buffer
    uint32_t n_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t ts = get_tile_size(cb_in);
    const InterleavedAddrGenFast<true> s0 = {.bank_base_address = addr0, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> s1 = {.bank_base_address = addr1, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_reserve_back(cb_in, 1);
        noc_async_read_tile(t, s0, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);

        cb_reserve_back(cb_in, 1);
        noc_async_read_tile(t, s1, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
