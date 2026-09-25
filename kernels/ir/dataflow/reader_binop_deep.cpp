// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// reader_binop_deep -- reader_binop with the DRAM read-depth rule applied (docs/he/dram_guide.md).
// Drop-in: same runtime args, same CBs, same push order.  reader_binop.cpp is RETAINED.
//
//   reader_binop:      reserve 1+1 -> read 2 -> BARRIER -> push 1+1, repeated n_tiles times (depth 2).
//   reader_binop_deep: reserve C+C -> read 2C -> BARRIER -> push C+C           (depth 2*CHUNK).
//
// The caller must give c_0 and c_1 a depth of at least CHUNK; scan_add_pass allocates 16.
//
//   Runtime args: 0 = a_addr, 1 = b_addr, 2 = n_tiles, 3 = start_tile.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_tile = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0, cb_in1 = tt::CBIndex::c_1, CHUNK = 8;
    const uint32_t ts = get_tile_size(cb_in0);

    const InterleavedAddrGenFast<true> a = {
        .bank_base_address = a_addr, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> b = {
        .bank_base_address = b_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n_tiles; i += CHUNK) {
        const uint32_t n = (n_tiles - i < CHUNK) ? (n_tiles - i) : CHUNK;
        cb_reserve_back(cb_in0, n);
        cb_reserve_back(cb_in1, n);
        const uint32_t p0 = get_write_ptr(cb_in0), p1 = get_write_ptr(cb_in1);
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t tile_id = start_tile + i + j;
            noc_async_read_tile(tile_id, a, p0 + j * ts);
            noc_async_read_tile(tile_id, b, p1 + j * ts);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0, n);
        cb_push_back(cb_in1, n);
    }
}
