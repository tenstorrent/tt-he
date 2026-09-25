// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writes n consecutive int32 tiles from c_16 with all of them in flight, then one barrier.
// Runtime args: 0 = addr, 1 = n_tiles, 2 = start_page.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t ts = get_tile_size(cb_out);
    const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};

    cb_wait_front(cb_out, n);
    const uint32_t p = get_read_ptr(cb_out);
    for (uint32_t i = 0; i < n; ++i) noc_async_write_tile(start + i, s, p + i * ts);
    noc_async_write_barrier();
    cb_pop_front(cb_out, n);
}
