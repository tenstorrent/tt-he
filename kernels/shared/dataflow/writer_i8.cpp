// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writes n int8 tiles from CB c_16 to DRAM (TensorAccessor: correct addressing
// past NUM_DRAM_BANKS, unlike InterleavedAddrGenFast with a 1KB int8 page).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = 16;
    constexpr auto args = TensorAccessorArgs<0>();
    const auto d = TensorAccessor(args, addr);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(start + i, d, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
