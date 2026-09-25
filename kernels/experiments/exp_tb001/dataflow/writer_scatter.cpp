// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Scatters the first n_out tiles of CB c_16 to n_out separate dest buffers (1 tile
// each); pops all n_total tiles (padding tiles past n_out are discarded). Replaces
// the host readback+untilize+extract of the resident FBC.
// Runtime args: 0 = n_out, 1 = n_total, 2..(1+n_out) = dest tile base addresses.
#include <cstdint>

void kernel_main() {
    uint32_t n_out = get_arg_val<uint32_t>(0);
    uint32_t n_total = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t ts = get_tile_size(cb_out);
    for (uint32_t i = 0; i < n_total; ++i) {
        cb_wait_front(cb_out, 1);
        if (i < n_out) {
            uint32_t addr = get_arg_val<uint32_t>(2 + i);
            const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};
            noc_async_write_tile(0, s, get_read_ptr(cb_out));
            noc_async_write_barrier();
        }
        cb_pop_front(cb_out, 1);
    }
}
