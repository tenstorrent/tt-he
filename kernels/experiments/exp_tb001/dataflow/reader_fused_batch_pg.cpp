// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Batched variant of reader_fused.cpp — same inputs, same CB contents, ONE barrier.
//   Xin      (1 int32 tile)                  -> c_0
//   folded W (NB*NB int8 tiles, (p,r) order) -> c_1
//
// WHY THIS EXISTS. reader_fused.cpp issues `read; barrier; push` per tile, so each of the 17 reads
// pays full DRAM latency in series and nothing pipelines: measured 8.77 us to deliver 17 tiles
// (~0.5 us/tile, ~2 GB/s on a core that can do far more). The compute kernel consumes W at
// 0.43 us/tile in its matmul phase, so it tracks the reader rather than the matrix engine --
// ph_matmul was 41.5% of fused_matmul_a while doing only 16 int8 tile-matmuls.
//
// The 17 reads are mutually independent, so they can all be in flight at once. One barrier at the
// end is the only ordering the consumer needs. REQUIRES c_1 to be sized for all NB*NB tiles
// (reader_fused.cpp works with a depth-2 streaming CB; this one does not) -- which is why it is a
// separate file rather than an edit to the shared reader.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t xin_addr = get_arg_val<uint32_t>(0);   // Xin (1 int32 tile)
    const uint32_t w_addr = get_arg_val<uint32_t>(1);     // folded W (NB*NB int8 tiles)
    const uint32_t x_page = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = 0;   // Xin, Int32
    constexpr uint32_t cb_w = 1;    // folded W, Int8
    constexpr uint32_t NB = 4;
    constexpr uint32_t NW = NB * NB;

    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, xin_addr);
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto w = TensorAccessor(w_args, w_addr);

    cb_reserve_back(cb_in, 1);
    cb_reserve_back(cb_w, NW);
    const uint32_t x_ptr = get_write_ptr(cb_in);
    const uint32_t w_ptr = get_write_ptr(cb_w);
    const uint32_t wts = get_tile_size(cb_w);

    // all 17 reads in flight together, then a single barrier
    noc_async_read_tile(x_page, x, x_ptr);
    for (uint32_t i = 0; i < NW; ++i) {
        noc_async_read_tile(i, w, w_ptr + i * wts);
    }
    noc_async_read_barrier();

    cb_push_back(cb_in, 1);
    cb_push_back(cb_w, NW);
}
