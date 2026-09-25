// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// reader_fused_batch_g — reader_fused_batch.cpp with a TILE INDEX for the dynamic operand, so a grid
// of cores can each feed a fused_matmul_{a,c}_1pass kernel with a different tile.
//
//   Xin tile `ti`  (1 int32 tile)             -> c_0
//   folded W       (NB*NB int8 tiles, (p,r))  -> c_1
//
// WHY THIS EXISTS. reader_fused_batch.cpp hardcodes `noc_async_read_tile(0, x, ...)`: it was written
// for exp_Tg018m, where each dispatch's input buffer holds exactly the one tile that dispatch wants.
// The NTT-32768 pipeline instead keeps all 32 cube tiles in ONE buffer and gives core c tile c, so a
// tile index is required. Same reason kernels/shared/dataflow/reader_fused_off.cpp exists for the PIR path;
// that one is offset-capable but barriers per tile, so it does not carry the depth rule.
//
// Both properties at once, which neither existing sibling has:
//   * ALL 17 reads in flight behind ONE barrier (docs/he/dram_guide.md 3.3; reader_fused.cpp took
//     8.77 us to deliver these 17 tiles at ~2 GB/s, and one barrier took ph_matmul 6.82 -> 1.25 us);
//   * a tile index for the dynamic operand.
//
// A NEW FILE, not an edit: reader_fused_batch.cpp has callers in host/experiments/exp_tg018m.cpp, and TT-Metal
// silently ignores surplus runtime args, so adding an argument would leave them reading garbage.
//
// REQUIRES c_1 sized for all NB*NB tiles (a depth-2 streaming CB cannot hold the batch).
//
// Runtime args: 0=xin_addr, 1=w_addr, 2=ti (which int32 tile of xin this core owns).
// Compile args: TensorAccessorArgs(xin) then TensorAccessorArgs(w).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t xin_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr   = get_arg_val<uint32_t>(1);
    const uint32_t ti       = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = 0;   // Xin, Int32
    constexpr uint32_t cb_w  = 1;   // folded W, Int8
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

    noc_async_read_tile(ti, x, x_ptr);                  // the only line that differs from _batch
    for (uint32_t i = 0; i < NW; ++i) {
        noc_async_read_tile(i, w, w_ptr + i * wts);
    }
    noc_async_read_barrier();                           // one barrier for all 17

    cb_push_back(cb_in, 1);
    cb_push_back(cb_w, NW);
}
