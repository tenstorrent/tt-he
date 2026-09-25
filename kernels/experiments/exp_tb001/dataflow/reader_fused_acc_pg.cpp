// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Feed for the ACCUMULATING fused matmul (compute/fused_matmul_{a,c}_acc.cpp): NACC partials of the
// block contraction in one dispatch, so NACC input tiles and NACC folded-W sets.
//
//   NACC int32 tiles, one from each input buffer                -> c_0
//   NACC * NB*NB int8 tiles, one W set per partial, a-major     -> c_1
//
// Same shape as reader_fused_batch.cpp -- every read is independent, so all NACC*(1+16) of them go
// in flight and there is ONE barrier. Inputs are issued before the W sets: the compute kernel's
// phase 1 only needs c_0, so the (much larger) W traffic overlaps the decompose.
//
// Runtime args: 0 = NACC, 1 = x_page (partial a is page x_page + a), 2..1+NACC = input buffer addresses,
//               2+NACC..1+2*NACC = W set addresses.  PAGE VARIANT for exp_Tg018m.
// Compile args: the input TensorAccessor's, then the W TensorAccessor's (same order as the
// prototypes appended by build_mm_acc on the host).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t NACC = get_arg_val<uint32_t>(0);
    const uint32_t x_page = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = 0;   // inputs, Int32
    constexpr uint32_t cb_w = 1;    // folded W, Int8
    constexpr uint32_t NB = 4;
    constexpr uint32_t NW = NB * NB;

    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    cb_reserve_back(cb_in, NACC);
    cb_reserve_back(cb_w, NACC * NW);
    const uint32_t x_ptr = get_write_ptr(cb_in);
    const uint32_t w_ptr = get_write_ptr(cb_w);
    const uint32_t xts = get_tile_size(cb_in);
    const uint32_t wts = get_tile_size(cb_w);

    for (uint32_t a = 0; a < NACC; ++a) {
        const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(2 + a));
        noc_async_read_tile(x_page + a, x, x_ptr + a * xts);   // partials are consecutive pages
    }
    for (uint32_t a = 0; a < NACC; ++a) {
        const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(2 + NACC + a));
        for (uint32_t i = 0; i < NW; ++i) {
            noc_async_read_tile(i, w, w_ptr + (a * NW + i) * wts);
        }
    }
    noc_async_read_barrier();

    cb_push_back(cb_in, NACC);
    cb_push_back(cb_w, NACC * NW);
}
