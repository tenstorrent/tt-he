// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Microbench: NoC core-to-core all-to-all. Each core writes G tiles from its L1
// (c_0) directly into each of G peer cores' L1 (c_1, slot = my_id), R times.
// Pure NoC L1->L1 (no DRAM). Sender-side barrier only (no semaphores -> no deadlock).
// Args: 0=my_id, 1=G, 2=R, 3..(3+2G-1) = (noc_x,noc_y) per destination core.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t my_id = get_arg_val<uint32_t>(0);
    const uint32_t G     = get_arg_val<uint32_t>(1);
    const uint32_t R     = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_s = tt::CBIndex::c_0, cb_d = tt::CBIndex::c_1;
    const uint32_t ts = get_tile_size(cb_s);
    cb_reserve_back(cb_s, G);
    cb_reserve_back(cb_d, G);
    const uint32_t src = get_write_ptr(cb_s);
    const uint32_t dbase = get_write_ptr(cb_d);   // same L1 offset on every core (identical CB alloc)
    for (uint32_t r = 0; r < R; ++r) {
        for (uint32_t j = 0; j < G; ++j) {
            const uint32_t x = get_arg_val<uint32_t>(3 + 2 * j);
            const uint32_t y = get_arg_val<uint32_t>(3 + 2 * j + 1);
            const uint64_t d = get_noc_addr(x, y, dbase + my_id * ts);   // peer j, slot my_id
            noc_async_write(src + j * ts, d, ts);
        }
        noc_async_write_barrier();
    }
}
