// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the fused reduce+twiddle kernel (compute/modadd_twiddle.cpp).
//   A: n int32 partial tiles from one buffer -> c_0
//   B: 1 int32 twiddle tile                  -> c_1
// All n+1 reads are independent, so they are issued together behind a single barrier
// (see reader_fused_batch.cpp for why barrier-per-tile costs ~0.5 us/tile).
//
// Runtime args: 0 = A base address, 1 = n, 2 = B base address.

#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t b_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;   // partials
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;   // twiddle
    const uint32_t ts = get_tile_size(cb_in0);

    const InterleavedAddrGenFast<true> a = {
        .bank_base_address = a_addr, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> b = {
        .bank_base_address = b_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb_in0, n_tiles);
    cb_reserve_back(cb_in1, 1);
    const uint32_t a_ptr = get_write_ptr(cb_in0);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        noc_async_read_tile(i, a, a_ptr + i * ts);
    }
    noc_async_read_tile(0, b, get_write_ptr(cb_in1));
    noc_async_read_barrier();

    cb_push_back(cb_in0, n_tiles);
    cb_push_back(cb_in1, 1);
}
