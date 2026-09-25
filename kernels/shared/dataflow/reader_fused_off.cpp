// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Offset-capable variant of reader_fused.cpp, for the PIR buffer-resident datapath.
//
// WHY A SIBLING KERNEL INSTEAD OF AN EXTRA ARG ON reader_fused:
// reader_fused hardcodes noc_async_read_tile(0, ...), so a transform can only ever read block 0 of its
// input buffer. That blocks the resident chain, which needs a poly's two blocks PACKED in one buffer so
// the MAC can consume it in a single dispatch. But reader_fused has 20+ call sites across device/ntt,
// device/he (including the freshly Blackhole-validated he_mul_cross) and device/pir, and TT runtime args
// are POSITIONAL -- adding arg 2 there would make every caller that does not pass it read garbage. So
// this copy takes the offset and only the PIR resident path uses it. Identical otherwise.
//
//   Xin      (1 int32 tile, at tile index x_off) -> c_0
//   folded W (NB*NB int8 tiles, (p,r) order)     -> c_1

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t xin_addr = get_arg_val<uint32_t>(0);   // Xin base
    const uint32_t w_addr = get_arg_val<uint32_t>(1);     // folded W (NB*NB int8 tiles)
    const uint32_t x_off = get_arg_val<uint32_t>(2);      // Xin TILE index to read (0 == reader_fused)

    constexpr uint32_t cb_in = 0;   // Xin, Int32
    constexpr uint32_t cb_w = 1;    // folded W, Int8
    constexpr uint32_t NB = 4;

    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, xin_addr);
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto w = TensorAccessor(w_args, w_addr);

    cb_reserve_back(cb_in, 1);
    noc_async_read_tile(x_off, x, get_write_ptr(cb_in));
    noc_async_read_barrier();
    cb_push_back(cb_in, 1);

    for (uint32_t i = 0; i < NB * NB; ++i) {
        cb_reserve_back(cb_w, 1);
        noc_async_read_tile(i, w, get_write_ptr(cb_w));
        noc_async_read_barrier();
        cb_push_back(cb_w, 1);
    }
}
