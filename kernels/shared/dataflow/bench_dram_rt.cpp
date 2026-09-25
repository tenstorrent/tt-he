// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Microbench: DRAM round-trip reshuffle. Each core writes G tiles to DRAM then
// reads G tiles back, R times (= the write+read a DRAM-based transpose pays).
// Args: 0=my_id, 1=G, 2=R, 3=dram_addr.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t my_id = get_arg_val<uint32_t>(0);
    const uint32_t G     = get_arg_val<uint32_t>(1);
    const uint32_t R     = get_arg_val<uint32_t>(2);
    const uint32_t addr  = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_s = tt::CBIndex::c_0, cb_d = tt::CBIndex::c_1;
    const uint32_t ts = get_tile_size(cb_s);
    cb_reserve_back(cb_s, G);
    cb_reserve_back(cb_d, G);
    const uint32_t src = get_write_ptr(cb_s);
    const uint32_t dpt = get_write_ptr(cb_d);
    const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};
    for (uint32_t r = 0; r < R; ++r) {
        for (uint32_t j = 0; j < G; ++j) noc_async_write_tile(my_id * G + j, s, src + j * ts);
        noc_async_write_barrier();
        for (uint32_t j = 0; j < G; ++j) noc_async_read_tile(my_id * G + j, s, dpt + j * ts);
        noc_async_read_barrier();
    }
}
