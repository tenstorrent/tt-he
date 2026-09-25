// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// reader_ab_deep — the A/B feed for exp_roofline001, with every DRAM rule in docs/he/dram_guide.md
// applied, because the whole point of the experiment is that the memory side must not be the
// accidental limit:
//
//   * DEPTH — CHUNK tile-pairs (2*CHUNK reads) in flight behind ONE barrier, not one read then a
//     barrier. §3.3 measures this as the difference between ~43 and ~82 GB/s.
//   * ALL EIGHT CHANNELS — the host lays A and B out as page = tile, so a core reading a consecutive
//     run of tiles walks bank = page % 8 across all eight. A stride that is a multiple of 8, or a
//     single-page buffer, would sit on one channel (that cost exp_Tg018 a 1.79x).
//   * BOTH NoCs — this half rides NCRISC/NOC_1; writer_deep.cpp rides BRISC/NOC_0.
//
// The caller must give c_0 and c_1 a depth of at least CHUNK. Tiles are handed to the compute kernel
// one pair at a time, so a chunk of reads overlaps the previous chunk's arithmetic.
//
// Runtime args: 0 = a_addr, 1 = b_addr, 2 = n_tiles, 3 = start_tile.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb0 = tt::CBIndex::c_0, cb1 = tt::CBIndex::c_1, CHUNK = get_compile_time_arg_val(0);
    const uint32_t ts = get_tile_size(cb0);

    const InterleavedAddrGenFast<true> a = {.bank_base_address = a_addr,
                                            .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> b = {.bank_base_address = b_addr,
                                            .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n; i += CHUNK) {
        const uint32_t k = (n - i < CHUNK) ? (n - i) : CHUNK;
        cb_reserve_back(cb0, k);
        cb_reserve_back(cb1, k);
        const uint32_t p0 = get_write_ptr(cb0), p1 = get_write_ptr(cb1);
        for (uint32_t j = 0; j < k; ++j) {
            noc_async_read_tile(start + i + j, a, p0 + j * ts);
            noc_async_read_tile(start + i + j, b, p1 + j * ts);
        }
        noc_async_read_barrier();
        cb_push_back(cb0, k);
        cb_push_back(cb1, k);
    }
}
