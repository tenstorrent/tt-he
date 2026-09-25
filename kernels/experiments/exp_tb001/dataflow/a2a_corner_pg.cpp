// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// All-to-all corner turn over a SINGLE 64-page buffer, for exp_Tg018m.
//
//   out[s][n1][t] = in[n1][s][t]     (swap the tile axis n1 with the row axis s, keep the column t)
//
// WHY THIS EXISTS, and why it is not just a2a_corner_bufs.cpp with different arguments.
// tt-metal maps interleaved page p to DRAM bank p % NUM_DRAM_BANKS, so a buffer holding ONE page
// holds it on bank 0 -- always. a2a_corner_bufs.cpp takes NT separate single-tile buffers and reads
// page 0 of each, which means all NT reads, from every core, hit bank 0. Measured on exp_Tg018 that
// is 24.6 GB/s against 53.9 for one saturated channel and 433 for the chip: the corner turn was
// never slow, it was single-banked.
//
// Here the 64 tiles are pages of one buffer, so tile t lives on bank t % 8 and a 32-tile read set is
// spread over all eight channels. The page for tile index n1 in group kh is n1*2 + kh, which the
// kernel computes -- so this also drops the argument count from 4+NT+n_out to 7.
//
// Tile face layout (32x32 = 2x2 of 16x16 faces): element (r,c) sits at uint32 offset
//   ((r/16)*2 + (c/16))*256 + (r%16)*16 + (c%16)
//
// Runtime args: 0 = NT, 1 = n_out, 2 = out_start, 3 = out_stride, 4 = kh,
//               5 = in_addr, 6 = out_addr
// Each core stages all NT input tiles in L1 (NT=32 -> 128 KB) and assembles the outputs it owns.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t TELEM = 1024;   // uint32 per 32x32 tile
}  // namespace

void kernel_main() {
    const uint32_t NT         = get_arg_val<uint32_t>(0);
    const uint32_t n_out      = get_arg_val<uint32_t>(1);
    const uint32_t out_start  = get_arg_val<uint32_t>(2);
    const uint32_t out_stride = get_arg_val<uint32_t>(3);
    const uint32_t kh         = get_arg_val<uint32_t>(4);
    const uint32_t in_addr    = get_arg_val<uint32_t>(5);
    const uint32_t out_addr   = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb = tt::CBIndex::c_0;          // NT input tiles + 1 output scratch
    const uint32_t tbytes = get_tile_size(cb);

    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,
                                              .page_size = tbytes, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr,
                                              .page_size = tbytes, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb, NT + 1);
    const uint32_t l1 = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* L = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);

    for (uint32_t n1 = 0; n1 < NT; ++n1) {             // page n1*2+kh -> bank (n1*2+kh) % 8
        noc_async_read_tile(n1 * 2 + kh, in, l1 + n1 * tbytes);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* outp = L + NT * TELEM;
    for (uint32_t i = 0; i < n_out; ++i) {
        const uint32_t s = out_start + i * out_stride;                       // the input ROW this output takes
        const uint32_t in0 = ((s >> 4) * 2) * 256 + (s & 15) * 16;           // input face-col 0
        const uint32_t in1 = ((s >> 4) * 2 + 1) * 256 + (s & 15) * 16;       // input face-col 1
        for (uint32_t n1 = 0; n1 < NT; ++n1) {
            volatile tt_l1_ptr uint32_t* intile = L + n1 * TELEM;
            const uint32_t o0 = ((n1 >> 4) * 2) * 256 + (n1 & 15) * 16;      // output face-col 0
            const uint32_t o1 = ((n1 >> 4) * 2 + 1) * 256 + (n1 & 15) * 16;  // output face-col 1
            for (uint32_t k = 0; k < 16; ++k) outp[o0 + k] = intile[in0 + k];
            for (uint32_t k = 0; k < 16; ++k) outp[o1 + k] = intile[in1 + k];
        }
        noc_async_write_tile(s * 2 + kh, out, reinterpret_cast<uint32_t>(outp));
        noc_async_write_barrier();
    }

    cb_push_back(cb, NT + 1);
}
