// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage-convert all-to-all (corner turn) for the (32,32,32) NTT.
//
//   D[s][n1][t] = Cb[n1][s][t]      (swap the tile axis n1 with the row axis s,
//                                    column t preserved)
//
// This moves sub-tile rows ACROSS tiles, so it cannot be a whole-tile permutation.
// We stage all NT input tiles in L1, then assemble each output tile by remapping
// (out row n1 = in tile n1's row s) directly in L1, and DMA each output tile out.
// Single core, compute-free (pure data movement + L1 copy). NT = 32 here.
//
// Tile face layout (32x32 = 2x2 of 16x16 faces): element (r,c) lives at
//   idx(r,c) = ((r/16)*2 + (c/16))*256 + (r%16)*16 + (c%16)   (uint32 offset).
//
// Runtime args: 0=in_addr, 1=out_addr, 2=NT.   Both buffers: NT int32 tiles.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t TELEM = 1024;   // uint32 per 32x32 tile
inline uint32_t faceidx(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}
}  // namespace

void kernel_main() {
    const uint32_t in_addr  = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t NT       = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb = tt::CBIndex::c_0;          // scratch: NT input tiles + 1 output tile
    const uint32_t tbytes = get_tile_size(cb);

    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,  .page_size = tbytes, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr, .page_size = tbytes, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb, NT + 1);
    const uint32_t l1 = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* L = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);

    // stage all input tiles in L1
    for (uint32_t n1 = 0; n1 < NT; ++n1) {
        noc_async_read_tile(n1, in, l1 + n1 * tbytes);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* outp = L + NT * TELEM;     // scratch slot for one output tile
    for (uint32_t s = 0; s < NT; ++s) {
        for (uint32_t n1 = 0; n1 < NT; ++n1) {
            volatile tt_l1_ptr uint32_t* intile = L + n1 * TELEM;
            for (uint32_t t = 0; t < 32; ++t) {
                outp[faceidx(n1, t)] = intile[faceidx(s, t)];   // D[s](n1,t) = Cb[n1](s,t)
            }
        }
        noc_async_write_tile(s, out, reinterpret_cast<uint32_t>(outp));
        noc_async_write_barrier();
    }

    cb_push_back(cb, NT + 1);
}
