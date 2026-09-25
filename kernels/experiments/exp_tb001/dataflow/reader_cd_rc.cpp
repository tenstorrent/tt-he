// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for tensor2_fused: streams n tiles from 3 DRAM buffers into c_0 (c),
// c_1 (d), c_3 (Rc). Runtime args: 0=c_addr, 1=d_addr, 2=rc_addr, 3=n.
#include <cstdint>

void kernel_main() {
    const uint32_t c_addr  = get_arg_val<uint32_t>(0);
    const uint32_t d_addr  = get_arg_val<uint32_t>(1);
    const uint32_t rc_addr = get_arg_val<uint32_t>(2);
    const uint32_t n       = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_c = tt::CBIndex::c_0, cb_d = tt::CBIndex::c_1, cb_rc = tt::CBIndex::c_3;
    const uint32_t ts = get_tile_size(cb_c);
    const InterleavedAddrGenFast<true> gc  = {.bank_base_address = c_addr,  .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> gd  = {.bank_base_address = d_addr,  .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> grc = {.bank_base_address = rc_addr, .page_size = ts, .data_format = DataFormat::UInt32};
    for (uint32_t t = 0; t < n; ++t) {
        cb_reserve_back(cb_c, 1);  noc_async_read_tile(t, gc,  get_write_ptr(cb_c));  noc_async_read_barrier(); cb_push_back(cb_c, 1);
        cb_reserve_back(cb_d, 1);  noc_async_read_tile(t, gd,  get_write_ptr(cb_d));  noc_async_read_barrier(); cb_push_back(cb_d, 1);
        cb_reserve_back(cb_rc, 1); noc_async_read_tile(t, grc, get_write_ptr(cb_rc)); noc_async_read_barrier(); cb_push_back(cb_rc, 1);
    }
}
