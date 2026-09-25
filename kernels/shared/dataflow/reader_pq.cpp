// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reads two 2-tiles-per-group int32 streams for crt_limb3: bP -> c_0 (P0,P1), bQ -> c_1 (Q0,Q1).
#include <cstdint>

void kernel_main() {
    uint32_t addrP = get_arg_val<uint32_t>(0);
    uint32_t addrQ = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_p = tt::CBIndex::c_0, cb_q = tt::CBIndex::c_1;
    const uint32_t ts = get_tile_size(cb_p);
    const InterleavedAddrGenFast<true> sp = {.bank_base_address = addrP, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> sq = {.bank_base_address = addrQ, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_reserve_back(cb_p, 2);
        uint32_t wp = get_write_ptr(cb_p);
        noc_async_read_tile(2 * t + 0, sp, wp);
        noc_async_read_tile(2 * t + 1, sp, wp + ts);
        cb_reserve_back(cb_q, 2);
        uint32_t wq = get_write_ptr(cb_q);
        noc_async_read_tile(2 * t + 0, sq, wq);
        noc_async_read_tile(2 * t + 1, sq, wq + ts);
        noc_async_read_barrier();
        cb_push_back(cb_p, 2);
        cb_push_back(cb_q, 2);
    }
}
