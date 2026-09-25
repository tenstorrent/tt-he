// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// reader_fbc_tr_pg — reader_fbc_tr.cpp for a pipeline that stores each limb in its OWN buffer.
//
// This is the one thing that kept the matrix-engine FBC out of the exp_Tg018 multiply. exp_Tg018r
// lays every limb of a polynomial in a single interleaved buffer at page = our_tile*L + limb, so its
// reader does
//     base = y.get_noc_addr(ot * L + i);
// exp_Tg018 stores L SEPARATE 64-page buffers (`B64`, page = tile), so that address does not exist.
// Here the L source base addresses arrive as runtime args instead and each limb is read from its own
// buffer at page `ot`. Nothing else changes -- same transpose, same CB contract, same push order.
//
// (tt-metal allows 4096 runtime args per Tensix kernel, so L addresses is not close to a limit.)
//
// Runtime args: 0=bt_addr, 1=par_addr, 2=LS, 3=ngroups, 4=ot_start, 5.. = LS source base addresses.
// Compile args: TensorAccessorArgs for the folded constant matrix (as reader_fbc_tr.cpp).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t bt_addr = get_arg_val<uint32_t>(0);
    const uint32_t par_addr = get_arg_val<uint32_t>(1);
    const uint32_t LS = get_arg_val<uint32_t>(2);
    const uint32_t ngroups = get_arg_val<uint32_t>(3);
    const uint32_t ot_start = get_arg_val<uint32_t>(4);
    constexpr uint32_t A0 = 5;                    // first per-limb source base address

    constexpr uint32_t cb_y = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1, cb_par = tt::CBIndex::c_4;
    constexpr uint32_t NRNP = 16;                 // NR radix planes x NP radix limbs
    constexpr uint32_t CHUNK = 16 * 4;            // 16 int32 = one contiguous face-row run

    const uint32_t yts = get_tile_size(cb_y), bts = get_tile_size(cb_b);
    const InterleavedAddrGenFast<true> pr = {.bank_base_address = par_addr, .page_size = yts, .data_format = DataFormat::UInt32};
    constexpr auto b_args = TensorAccessorArgs<0>();
    const auto b = TensorAccessor(b_args, bt_addr, bts);

    cb_reserve_back(cb_par, 5);
    { const uint32_t p = get_write_ptr(cb_par);
      for (uint32_t i = 0; i < 5; ++i) noc_async_read_tile(i, pr, p + i * yts);
      noc_async_read_barrier(); }
    cb_push_back(cb_par, 5);

    cb_reserve_back(cb_b, NRNP);
    { const uint32_t p = get_write_ptr(cb_b);
      for (uint32_t i = 0; i < NRNP; ++i) noc_async_read_tile(i, b, p + i * bts);
      noc_async_read_barrier(); }
    cb_push_back(cb_b, NRNP);

    for (uint32_t g = 0; g < ngroups; ++g) {
        const uint32_t ot = ot_start + g / 32, r = g % 32;
        const uint32_t src_lo = (((r / 16) * 2) * 256 + (r % 16) * 16) * 4;   // c = 0..15
        const uint32_t src_hi = src_lo + 256 * 4;                              // c = 16..31
        cb_reserve_back(cb_y, 1);
        const uint32_t p = get_write_ptr(cb_y);
        for (uint32_t i = 0; i < LS; ++i) {
            const uint32_t dst_lo = (((i / 16) * 2) * 256 + (i % 16) * 16) * 4;
            const InterleavedAddrGenFast<true> yi = {.bank_base_address = get_arg_val<uint32_t>(A0 + i),
                                                    .page_size = yts, .data_format = DataFormat::UInt32};
            const uint64_t base = yi.get_noc_addr(ot);
            noc_async_read(base + src_lo, p + dst_lo, CHUNK);
            noc_async_read(base + src_hi, p + dst_lo + 256 * 4, CHUNK);
        }
        noc_async_read_barrier();
        cb_push_back(cb_y, 1);
    }
}
