// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// All-to-all corner turn over PER-TILE BUFFERS, for the 3-level NTT-65536 (exp_Tg018).
//
//   out[s][n1][t] = in[n1][s][t]     (swap the tile axis n1 with the row axis s, keep the column t)
//
// WHY A NEW KERNEL. kernels/shared/dataflow/all_to_all_corner_mc.cpp does exactly this arithmetic, but it
// addresses the NT input tiles as pages 0..NT-1 of ONE buffer. Every other stage in the exp_Tb001
// tree takes each tile as its own single-tile buffer (that is what reader_i32 / reader_fused_acc /
// writer_i32 expect), and a TensorAccessor cannot be pointed at "tile t of a larger buffer" by
// address arithmetic -- DRAM interleaving puts page i on bank (i mod nbanks). So either every matmul
// reader learns a page offset, or the corner turn learns per-tile addresses. This is the second, and
// it is the smaller change: one new file, no edit to any kernel the other experiments dispatch.
//
// The row-chunk copy is kept verbatim from all_to_all_corner_mc.cpp: a tile row is two contiguous
// 16-element face chunks, so the face-index arithmetic is hoisted out of the column loop and 16
// words move at a time.
//
// Tile face layout (32x32 = 2x2 of 16x16 faces): element (r,c) sits at uint32 offset
//   ((r/16)*2 + (c/16))*256 + (r%16)*16 + (c%16)
//
// Runtime args: 0 = NT, 1 = n_out, 2 = out_start, 3 = out_stride,
//               4 .. 4+NT-1        = the NT input tile addresses, indexed by n1
//               4+NT .. 4+NT+n_out = this core's n_out output tile addresses, in order
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
    const uint32_t A_IN = 4, A_OUT = 4 + NT;

    constexpr uint32_t cb = tt::CBIndex::c_0;          // NT input tiles + 1 output scratch
    const uint32_t tbytes = get_tile_size(cb);

    cb_reserve_back(cb, NT + 1);
    const uint32_t l1 = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* L = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);

    for (uint32_t n1 = 0; n1 < NT; ++n1) {
        const InterleavedAddrGenFast<true> in = {.bank_base_address = get_arg_val<uint32_t>(A_IN + n1),
                                                 .page_size = tbytes, .data_format = DataFormat::UInt32};
        noc_async_read_tile(0, in, l1 + n1 * tbytes);
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
        const InterleavedAddrGenFast<true> out = {.bank_base_address = get_arg_val<uint32_t>(A_OUT + i),
                                                 .page_size = tbytes, .data_format = DataFormat::UInt32};
        noc_async_write_tile(0, out, reinterpret_cast<uint32_t>(outp));
        noc_async_write_barrier();
    }

    cb_push_back(cb, NT + 1);
}
