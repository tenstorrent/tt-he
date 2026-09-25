// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for recombine_rns: 4 int32 plane tiles -> c_0, then 5 int32 per-column
// param tiles -> c_1. Args: 0 = planes_addr, 1 = par_addr.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t planes_addr = get_arg_val<uint32_t>(0);
    const uint32_t par_addr = get_arg_val<uint32_t>(1);
    const uint32_t nct = get_arg_val<uint32_t>(2);   // coeff-tiles (4 planes each)

    constexpr uint32_t cb_in = tt::CBIndex::c_0, cb_par = tt::CBIndex::c_1;
    const uint32_t ts = get_tile_size(cb_in);
    const InterleavedAddrGenFast<true> pl = {.bank_base_address = planes_addr, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> pr = {.bank_base_address = par_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    // 5 per-column param tiles first (consumed once, kept resident by compute).
    for (uint32_t i = 0; i < 5; ++i) {
        cb_reserve_back(cb_par, 1);
        noc_async_read_tile(i, pr, get_write_ptr(cb_par));
        noc_async_read_barrier();
        cb_push_back(cb_par, 1);
    }
    // 4 planes per coeff-tile, streamed.
    for (uint32_t t = 0; t < 4 * nct; ++t) {
        cb_reserve_back(cb_in, 1);
        noc_async_read_tile(t, pl, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
