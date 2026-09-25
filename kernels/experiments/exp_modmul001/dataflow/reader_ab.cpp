// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_modmul001 reader: n tile-pairs, a -> c_0 and b -> c_1, one pair in flight.
// Deliberately the simplest possible feed, and IDENTICAL for all three compute kernels, so that no
// part of the measured difference can come from the dataflow side.
// Runtime args: 0 = a_addr, 1 = b_addr, 2 = n_tiles.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1;
    const uint32_t ts = get_tile_size(cb_a);
    const InterleavedAddrGenFast<true> sa = {.bank_base_address = a_addr, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> sb = {.bank_base_address = b_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n; ++i) {
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);
        noc_async_read_tile(i, sa, get_write_ptr(cb_a));
        noc_async_read_tile(i, sb, get_write_ptr(cb_b));
        noc_async_read_barrier();
        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }
}
