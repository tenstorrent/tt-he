// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for crt_mul_split: loads 2 param tiles (q1_low16, q1_high16) into c_2 once, then
// streams t[i] -> c_0 and r1[i] -> c_1 for i in [0, n).
// Args: 0=t_addr, 1=r1_addr, 2=par_addr, 3=n_tiles.
#include <cstdint>

void kernel_main() {
    const uint32_t t_addr = get_arg_val<uint32_t>(0);
    const uint32_t r1_addr = get_arg_val<uint32_t>(1);
    const uint32_t par_addr = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_t = tt::CBIndex::c_0, cb_r1 = tt::CBIndex::c_1, cb_par = tt::CBIndex::c_2;
    const uint32_t ts = get_tile_size(cb_t);
    const InterleavedAddrGenFast<true> st = {.bank_base_address = t_addr, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> sr = {.bank_base_address = r1_addr, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> sp = {.bank_base_address = par_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb_par, 2);
    uint32_t pw = get_write_ptr(cb_par);
    noc_async_read_tile(0, sp, pw);
    noc_async_read_tile(1, sp, pw + ts);
    noc_async_read_barrier();
    cb_push_back(cb_par, 2);

    for (uint32_t i = 0; i < n; ++i) {
        cb_reserve_back(cb_t, 1);
        cb_reserve_back(cb_r1, 1);
        noc_async_read_tile(i, st, get_write_ptr(cb_t));
        noc_async_read_tile(i, sr, get_write_ptr(cb_r1));
        noc_async_read_barrier();
        cb_push_back(cb_t, 1);
        cb_push_back(cb_r1, 1);
    }
}
