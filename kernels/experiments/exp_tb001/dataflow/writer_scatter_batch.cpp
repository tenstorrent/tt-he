// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Batched variant of writer_scatter.cpp — same semantics, ONE barrier per group of B tiles.
//
// WHY THIS EXISTS. writer_scatter.cpp barriers after every tile, so it keeps one write in flight.
// bconv_one at Ld=46 therefore pays 46 serialised write round trips per core. See
// reader_gather_batch.cpp for the measurements behind this and for the B == CB depth requirement
// (get_read_ptr() + j*tile_size does not wrap, so a group must not straddle the buffer end).
//
// Runtime args: 0 = n_out, 1 = n_total, 2 = B, 3..(2+n_out) = dest tile base addresses.
// Tiles past n_out are padding: still popped, never written -- as in the unbatched version.
#include <cstdint>

void kernel_main() {
    const uint32_t n_out = get_arg_val<uint32_t>(0);
    const uint32_t n_total = get_arg_val<uint32_t>(1);
    const uint32_t B = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t ts = get_tile_size(cb_out);

    for (uint32_t i = 0; i < n_total; i += B) {
        const uint32_t k = (n_total - i < B) ? (n_total - i) : B;
        cb_wait_front(cb_out, k);
        const uint32_t base = get_read_ptr(cb_out);
        bool any = false;
        for (uint32_t j = 0; j < k; ++j) {                  // all k writes in flight
            if (i + j >= n_out) continue;
            const uint32_t addr = get_arg_val<uint32_t>(3 + i + j);
            const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};
            noc_async_write_tile(0, s, base + j * ts);
            any = true;
        }
        if (any) noc_async_write_barrier();                 // ...then ONE barrier
        cb_pop_front(cb_out, k);
    }
}
