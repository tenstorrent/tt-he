// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// reader_mbatch_deep -- reader_mbatch with the DRAM read-depth rule applied (docs/he/dram_guide.md).
//
// Drop-in for kernels/ir/dataflow/reader_mbatch.cpp: same runtime args, same CBs, same push order.
// The ONLY change is how many reads are in flight at once.
//
//   reader_mbatch:      reserve 1 -> read 1 -> BARRIER -> push 1,  repeated M + M*16 times.
//   reader_mbatch_deep: reserve C -> read C -> BARRIER -> push C.
//
// This is the single hottest feed in the PIR server: the initial-dimension scan streams the WHOLE
// folded database past the cores on every query -- 6.0 GB at N0=512 x rest=128 -- and it is the
// foldW loop below that moves all of it.  At depth 1 each 1 KB tile pays a full DRAM round trip that
// nothing overlaps, which is why the scan measured 34 GB/s against the ~250 GB/s the guide records.
//
// CHUNK is half of cb_w's 32-tile depth on purpose.  Reserving the whole CB would put 32 reads in
// flight but leave the matmul idle until every one of them landed; at 16 the reader can reserve and
// fill the second half while the matmul consumes the first, so the depth AND the overlap are there.
// cb_in is batched whole because the compute kernel's phase 1 waits for all M tiles anyway.
//
//   Runtime args: 0 = xin_addr, 1 = w_addr, 2 = x_base (tile), 3 = w_base (tile), 4 = M.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t xin_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr   = get_arg_val<uint32_t>(1);
    const uint32_t x_base   = get_arg_val<uint32_t>(2);
    const uint32_t w_base   = get_arg_val<uint32_t>(3);
    const uint32_t M        = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in = 0, cb_w = 1, NB = 4, CHUNK = 16;
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, xin_addr);
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto w = TensorAccessor(w_args, w_addr);

    const uint32_t xts = get_tile_size(cb_in);
    cb_reserve_back(cb_in, M);
    { const uint32_t p = get_write_ptr(cb_in);
      for (uint32_t m = 0; m < M; ++m) noc_async_read_tile(x_base + m, x, p + m * xts); }
    noc_async_read_barrier();
    cb_push_back(cb_in, M);

    const uint32_t wts = get_tile_size(cb_w);
    const uint32_t WT = M * NB * NB;
    for (uint32_t i = 0; i < WT; i += CHUNK) {
        const uint32_t n = (WT - i < CHUNK) ? (WT - i) : CHUNK;
        cb_reserve_back(cb_w, n);
        const uint32_t p = get_write_ptr(cb_w);
        for (uint32_t j = 0; j < n; ++j) noc_async_read_tile(w_base + i + j, w, p + j * wts);
        noc_async_read_barrier();
        cb_push_back(cb_w, n);
    }
}
