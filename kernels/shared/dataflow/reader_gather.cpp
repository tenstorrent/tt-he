// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Gathers 1 tile from each of n separate source buffers into CB c_0 (resident FBC).
// Replaces the host stack+tilize+upload: each source limb lives in its own 1-tile
// DRAM buffer; this reads tile 0 of each, in order, so c_0 holds [src0,src1,...].
// Runtime args: 0 = n, 1..n = source tile base addresses.
#include <cstdint>

void kernel_main() {
    uint32_t n = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t ts = get_tile_size(cb_in);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t addr = get_arg_val<uint32_t>(1 + i);
        const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};
        cb_reserve_back(cb_in, 1);
        noc_async_read_tile(0, s, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
