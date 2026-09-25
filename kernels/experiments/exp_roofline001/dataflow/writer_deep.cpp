// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// writer_deep — the result drain for exp_roofline001. BRISC/NOC_0, so the writes ride the other NoC
// from reader_ab_deep.cpp's reads, and CHUNK tiles are in flight behind one barrier.
//
// The caller must give c_16 a depth of at least CHUNK.
//
// Runtime args: 0 = out_addr, 1 = n_tiles, 2 = start_tile.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16, CHUNK = get_compile_time_arg_val(0);
    const uint32_t ts = get_tile_size(cb_out);
    const InterleavedAddrGenFast<true> s = {.bank_base_address = addr,
                                            .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n; i += CHUNK) {
        const uint32_t k = (n - i < CHUNK) ? (n - i) : CHUNK;
        cb_wait_front(cb_out, k);
        const uint32_t p = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < k; ++j) noc_async_write_tile(start + i + j, s, p + j * ts);
        noc_async_write_barrier();
        cb_pop_front(cb_out, k);
    }
}
