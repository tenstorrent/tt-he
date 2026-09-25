// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// writer_i32_deep -- writer_i32 with the write-depth rule applied (docs/he/dram_guide.md).
// Drop-in: same runtime args, same CB.  writer_i32.cpp is RETAINED and unchanged.
//
//   writer_i32:      wait 1 -> write 1 -> BARRIER -> pop 1, repeated n_tiles times (depth 1).
//   writer_i32_deep: wait C -> write C -> BARRIER -> pop C                        (depth CHUNK).
//
// The barrier is what serialises: at depth 1 every tile pays a full round trip that nothing overlaps.
// The caller must give c_16 a depth of at least CHUNK, or the wait below can never be satisfied --
// scan_add_pass allocates 16.  A tail shorter than CHUNK is handled by the min.
//
//   Runtime args: 0 = addr, 1 = n_tiles, 2 = start_tile.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16, CHUNK = 8;
    const uint32_t ts = get_tile_size(cb_out);
    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n_tiles; i += CHUNK) {
        const uint32_t n = (n_tiles - i < CHUNK) ? (n_tiles - i) : CHUNK;
        cb_wait_front(cb_out, n);
        const uint32_t p = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < n; ++j) noc_async_write_tile(start + i + j, s, p + j * ts);
        noc_async_write_barrier();
        cb_pop_front(cb_out, n);
    }
}
