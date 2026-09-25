// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_modmul001 writer: n int32 tiles from c_16 to DRAM. Identical for all three compute kernels.
// Runtime args: 0 = out_addr, 1 = n_tiles.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t ts = get_tile_size(cb_out);
    const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(i, s, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
