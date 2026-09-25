// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Optimized stage-convert all-to-all (corner turn) for the (32,32,32) NTT:
//   D[s][n1][t] = Cb[n1][s][t]   (swap tile axis n1 with row axis s, col t kept)
//
// Two optimizations over all_to_all_corner.cpp (which was single-core, per-element):
//   1. MULTI-CORE: the 32 output tiles are independent. Each core builds the output
//      tiles assigned to it (out_start + i*out_stride, i<n_out); the work divides by C.
//   2. ROW-CHUNK COPY: a tile row = two contiguous 16-element face-chunks. We copy
//      16 words at a time and hoist the face-index math out of the inner loop, instead
//      of recomputing faceidx() per element. ~16x fewer index computations.
//
// Each core stages all NT input tiles in L1 (read-only), then assembles its outputs.
//
// Tile face layout (32x32 = 2x2 of 16x16 faces): element (r,c) at uint32 offset
//   ((r/16)*2 + (c/16))*256 + (r%16)*16 + (c%16);  a row's 16 cols of one face-col
//   are contiguous (base = ((r/16)*2 + fc)*256 + (r%16)*16, then +0..15).
//
// Runtime args: 0=in_addr, 1=out_addr, 2=NT, 3=out_start, 4=out_stride, 5=n_out.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t TELEM = 1024;   // uint32 per 32x32 tile
}  // namespace

void kernel_main() {
    const uint32_t in_addr   = get_arg_val<uint32_t>(0);
    const uint32_t out_addr  = get_arg_val<uint32_t>(1);
    const uint32_t NT        = get_arg_val<uint32_t>(2);
    const uint32_t out_start = get_arg_val<uint32_t>(3);
    const uint32_t out_stride= get_arg_val<uint32_t>(4);
    const uint32_t n_out     = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb = tt::CBIndex::c_0;          // NT input tiles + 1 output scratch
    const uint32_t tbytes = get_tile_size(cb);

    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,  .page_size = tbytes, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr, .page_size = tbytes, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb, NT + 1);
    const uint32_t l1 = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* L = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);

    for (uint32_t n1 = 0; n1 < NT; ++n1) noc_async_read_tile(n1, in, l1 + n1 * tbytes);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* outp = L + NT * TELEM;
    for (uint32_t i = 0; i < n_out; ++i) {
        const uint32_t s = out_start + i * out_stride;
        // input row s -> output row n1; chunk bases (per face-col) hoisted out of the col loop
        const uint32_t in0 = ((s >> 4) * 2) * 256 + (s & 15) * 16;          // input face-col 0
        const uint32_t in1 = ((s >> 4) * 2 + 1) * 256 + (s & 15) * 16;      // input face-col 1
        for (uint32_t n1 = 0; n1 < NT; ++n1) {
            volatile tt_l1_ptr uint32_t* intile = L + n1 * TELEM;
            const uint32_t o0 = ((n1 >> 4) * 2) * 256 + (n1 & 15) * 16;     // output face-col 0
            const uint32_t o1 = ((n1 >> 4) * 2 + 1) * 256 + (n1 & 15) * 16; // output face-col 1
            for (uint32_t k = 0; k < 16; ++k) outp[o0 + k] = intile[in0 + k];   // cols 0..15
            for (uint32_t k = 0; k < 16; ++k) outp[o1 + k] = intile[in1 + k];   // cols 16..31
        }
        noc_async_write_tile(s, out, reinterpret_cast<uint32_t>(outp));
        noc_async_write_barrier();
    }

    cb_push_back(cb, NT + 1);
}
