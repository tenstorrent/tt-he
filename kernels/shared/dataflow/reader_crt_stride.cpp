// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for crt_mul_split with a STRIDED t input — lets the K=3 tail feed w0 / w1 (which land
// interleaved [w0,w1,w0,w1,...] in one buffer) as separate single-tile t streams.
//   t[i] = tile (i*t_stride + t_off) of t_addr;  r1[i] = tile (r1_off + i) of r1_addr;  par into c_2.
// r1_off lets r1 be one poly inside a multi-poly buffer (the buffer-resident key-switch reads the
// ct's component-1 residues straight out of the ciphertext buffer) — pass 0 for the standalone case.
// Args: 0=t_addr, 1=t_stride, 2=t_off, 3=r1_addr, 4=par_addr, 5=n_tiles, 6=r1_off.
#include <cstdint>

void kernel_main() {
    const uint32_t t_addr   = get_arg_val<uint32_t>(0);
    const uint32_t t_stride = get_arg_val<uint32_t>(1);
    const uint32_t t_off    = get_arg_val<uint32_t>(2);
    const uint32_t r1_addr  = get_arg_val<uint32_t>(3);
    const uint32_t par_addr = get_arg_val<uint32_t>(4);
    const uint32_t n        = get_arg_val<uint32_t>(5);
    const uint32_t r1_off   = get_arg_val<uint32_t>(6);

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
        noc_async_read_tile(i * t_stride + t_off, st, get_write_ptr(cb_t));
        noc_async_read_tile(r1_off + i, sr, get_write_ptr(cb_r1));
        noc_async_read_barrier();
        cb_push_back(cb_t, 1);
        cb_push_back(cb_r1, 1);
    }
}
