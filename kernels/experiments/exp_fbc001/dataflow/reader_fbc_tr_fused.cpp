// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// THE LIMB -> COEFFICIENT TRANSPOSE, done as NoC addressing on the way in.  Feeds fbc_fpu_tr.cpp.
// exp_fbc001 variant of reader_fbc_tr.cpp: the C1 scale is fused into the compute kernel, so this
// gathers the transpose of X (not Y) and there is no y buffer in DRAM at all. It also carries 10
// parameter tiles instead of 5 -- the extra 5 are the SOURCE-side (mod q_i, per row i) constants the
// fused scale needs, mirroring the 5 destination-side ones the recombine already used.
//
// A tile in exp_Tg018 is 1024 coefficients of ONE residue, tilized as four 16x16 faces:
//     element (r, c)  ->  offset ((r/16)*2 + c/16)*256 + (r%16)*16 + (c%16)
// so tile ROW r is exactly TWO contiguous runs of 16 elements (c = 0..15 and c = 16..31).  The
// matrix engine wants Y'[i][c] = y_i[r*32 + c] -- rows = source residue, columns = coefficient --
// and row i of Y' is likewise two contiguous 16-runs.  So the whole transpose is 2 copies of 64
// bytes per (group, source): no transpose op, no extra pass, no scratch buffer.
//
// Group g covers our-tile (ot_start + g/32) and tile row r = g%32.  Reads are issued for all LS
// sources before the barrier so the DRAM sees LS requests in flight.
//
// Also pushes, once: the NR*NP int8 Bt tiles (c_1) and the 5 recombine param tiles (c_4), both
// resident for the whole dispatch.  Bt goes through TensorAccessor because a 1024-byte int8 page
// is mis-addressed past bank 12 by InterleavedAddrGenFast.
//
// Compile args: TensorAccessorArgs(Bt).
// Runtime args: 0 = x_addr (was y_addr: the C1 scale is now FUSED, so this reads X directly), 1 = bt_addr, 2 = par_addr, 3 = LS, 4 = ngroups, 5 = ot_start, 6 = L
//               (L = the source count the y buffer is laid out with, page = ot*L + i).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t bt_addr = get_arg_val<uint32_t>(1);
    const uint32_t par_addr = get_arg_val<uint32_t>(2);
    const uint32_t LS = get_arg_val<uint32_t>(3);
    const uint32_t ngroups = get_arg_val<uint32_t>(4);
    const uint32_t ot_start = get_arg_val<uint32_t>(5);
    const uint32_t L = get_arg_val<uint32_t>(6);
    const uint32_t NPAR = get_arg_val<uint32_t>(7);   // 10: 5 P-side (output rows) + 5 Q-side (source rows)

    constexpr uint32_t cb_y = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1, cb_par = tt::CBIndex::c_4;
    constexpr uint32_t NRNP = 16;                 // NR radix planes x NP radix limbs
    constexpr uint32_t CHUNK = 16 * 4;            // 16 int32 = one contiguous face-row run

    const uint32_t yts = get_tile_size(cb_y), bts = get_tile_size(cb_b);
    const InterleavedAddrGenFast<true> y = {.bank_base_address = y_addr, .page_size = yts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> pr = {.bank_base_address = par_addr, .page_size = yts, .data_format = DataFormat::UInt32};
    constexpr auto b_args = TensorAccessorArgs<0>();
    const auto b = TensorAccessor(b_args, bt_addr, bts);

    cb_reserve_back(cb_par, NPAR);
    { const uint32_t p = get_write_ptr(cb_par);
      for (uint32_t i = 0; i < NPAR; ++i) noc_async_read_tile(i, pr, p + i * yts);
      noc_async_read_barrier(); }
    cb_push_back(cb_par, NPAR);

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
            const uint64_t base = y.get_noc_addr(ot * L + i);
            noc_async_read(base + src_lo, p + dst_lo, CHUNK);
            noc_async_read(base + src_hi, p + dst_lo + 256 * 4, CHUNK);
        }
        noc_async_read_barrier();
        cb_push_back(cb_y, 1);
    }
}
