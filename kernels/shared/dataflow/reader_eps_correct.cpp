// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for eps_correct_crns (P3). Loads the 2 per-column param tiles [QMODP, P] -> c_1
// once, then per super-block: L y_i tiles -> c_0 and 1 combine tile -> c_2.
// All int32 (4096-byte pages) so InterleavedAddrGenFast is fine.
// Args: 0=y_addr, 1=par_addr, 2=cmb_addr, 3=nsb, 4=L.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr   = get_arg_val<uint32_t>(0);
    const uint32_t par_addr = get_arg_val<uint32_t>(1);
    const uint32_t cmb_addr = get_arg_val<uint32_t>(2);
    const uint32_t nsb      = get_arg_val<uint32_t>(3);
    const uint32_t L        = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_y = tt::CBIndex::c_0, cb_par = tt::CBIndex::c_1, cb_cmb = tt::CBIndex::c_2;
    const uint32_t ts = get_tile_size(cb_y);
    const InterleavedAddrGenFast<true> yb = {.bank_base_address = y_addr,   .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> pr = {.bank_base_address = par_addr, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> cb = {.bank_base_address = cmb_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < 2; ++i) {
        cb_reserve_back(cb_par, 1);
        noc_async_read_tile(i, pr, get_write_ptr(cb_par));
        noc_async_read_barrier();
        cb_push_back(cb_par, 1);
    }
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        cb_reserve_back(cb_y, L);
        uint32_t wp = get_write_ptr(cb_y);
        for (uint32_t i = 0; i < L; ++i) {
            noc_async_read_tile(sb * L + i, yb, wp + i * ts);
        }
        noc_async_read_barrier();
        cb_push_back(cb_y, L);

        cb_reserve_back(cb_cmb, 1);
        noc_async_read_tile(sb, cb, get_write_ptr(cb_cmb));
        noc_async_read_barrier();
        cb_push_back(cb_cmb, 1);
    }
}
