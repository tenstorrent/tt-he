// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Corner-turn ASSEMBLE + WRITE HALF, for exp_Tg018m: BRISC/NOC_0 fetches the SECOND NT/2 input
// tiles on the other NoC, waits for a2a_corner_rd.cpp's half on c_0, assembles the outputs it owns
// and writes them. See a2a_corner_rd.cpp for why the reads are split.
//
//   out[s][n1][t] = in[n1][s][t]     (swap the tile axis n1 with the row axis s, keep the column t)
//
// L1 layout: c_0 holds input tiles 0..half-1 (produced by NCRISC), c_1 holds half..NT-1 (produced
// here), c_2 is the one-tile output scratch. c_1 is never pushed -- this kernel is both its producer
// and its only consumer, so the reserve + write pointer is all the ownership it needs.
//
// Tile face layout (32x32 = 2x2 of 16x16 faces): element (r,c) sits at uint32 offset
//   ((r/16)*2 + (c/16))*256 + (r%16)*16 + (c%16)
//
// Runtime args: identical vector to a2a_corner_rd.cpp / a2a_corner_pg.cpp.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr uint32_t TELEM = 1024;   // uint32 per 32x32 tile
}  // namespace

void kernel_main() {
    // The corner turn is DATAFLOW-ONLY -- no compute kernel, so without a named zone its
    // dispatch window emits no name and tools/traces/dispatch_csv.py labels it with whichever
    // NEIGHBOURING dispatch's zone happens to start inside its bounds. That put 152 a2a
    // dispatches under modmul_fused3 and the three matmuls in the 54.1 ms census.
    DeviceZoneScopedN("a2a_corner");
    const uint32_t NT         = get_arg_val<uint32_t>(0);
    const uint32_t n_out      = get_arg_val<uint32_t>(1);
    const uint32_t out_start  = get_arg_val<uint32_t>(2);
    const uint32_t out_stride = get_arg_val<uint32_t>(3);
    const uint32_t kh         = get_arg_val<uint32_t>(4);
    const uint32_t in_addr    = get_arg_val<uint32_t>(5);
    const uint32_t out_addr   = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_lo = tt::CBIndex::c_0;   // tiles 0..half-1, from NCRISC
    constexpr uint32_t cb_hi = tt::CBIndex::c_1;   // tiles half..NT-1, read here
    constexpr uint32_t cb_o  = tt::CBIndex::c_2;   // output scratch, 1 tile
    const uint32_t tbytes = get_tile_size(cb_lo);
    const uint32_t half = NT >> 1;

    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,
                                              .page_size = tbytes, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr,
                                              .page_size = tbytes, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb_hi, half);
    const uint32_t hi = get_write_ptr(cb_hi);
    for (uint32_t n1 = half; n1 < NT; ++n1) {          // this half rides NOC_0
        noc_async_read_tile(n1 * 2 + kh, in, hi + (n1 - half) * tbytes);
    }
    noc_async_read_barrier();

    cb_wait_front(cb_lo, half);                        // ...and NCRISC's half has landed
    const uint32_t lo = get_read_ptr(cb_lo);
    cb_reserve_back(cb_o, 1);
    volatile tt_l1_ptr uint32_t* outp =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_o));

    for (uint32_t i = 0; i < n_out; ++i) {
        const uint32_t s = out_start + i * out_stride;                       // the input ROW this output takes
        const uint32_t in0 = ((s >> 4) * 2) * 256 + (s & 15) * 16;           // input face-col 0
        const uint32_t in1 = ((s >> 4) * 2 + 1) * 256 + (s & 15) * 16;       // input face-col 1
        for (uint32_t n1 = 0; n1 < NT; ++n1) {
            const uint32_t base = (n1 < half) ? (lo + n1 * tbytes) : (hi + (n1 - half) * tbytes);
            volatile tt_l1_ptr uint32_t* intile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
            const uint32_t o0 = ((n1 >> 4) * 2) * 256 + (n1 & 15) * 16;      // output face-col 0
            const uint32_t o1 = ((n1 >> 4) * 2 + 1) * 256 + (n1 & 15) * 16;  // output face-col 1
            for (uint32_t k = 0; k < 16; ++k) outp[o0 + k] = intile[in0 + k];
            for (uint32_t k = 0; k < 16; ++k) outp[o1 + k] = intile[in1 + k];
        }
        noc_async_write_tile(s * 2 + kh, out, reinterpret_cast<uint32_t>(outp));
        noc_async_write_barrier();
    }
    cb_pop_front(cb_lo, half);
    cb_push_back(cb_o, 1);
}
