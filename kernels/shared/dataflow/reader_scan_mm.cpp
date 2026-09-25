// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the multi-core scan matmul (compute/scan_mm.cpp).  This core computes output tiles
// [o0, o0+n_out).  For each output tile o it streams KB (A,B) int8 tile pairs:
//   A[k] = DB tile ((o0+o)*KB + k)   — unique per output tile (the streamed DB, bandwidth-dominant)
//   B[k] = query tile (k)            — shared across all output tiles
// A and B live in DRAM; page size = one int8 32×32 tile (1024 B).
//
//   Runtime args: 0 = a_addr, 1 = b_addr, 2 = o0 (first output tile), 3 = n_out, 4 = KB.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t o0     = get_arg_val<uint32_t>(2);
    const uint32_t n_out  = get_arg_val<uint32_t>(3);
    const uint32_t KB     = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in0 = 0, cb_in1 = 1;
    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    for (uint32_t o = 0; o < n_out; ++o) {
        const uint32_t abase = (o0 + o) * KB;
        for (uint32_t k = 0; k < KB; ++k) {
            cb_reserve_back(cb_in0, 1);
            noc_async_read_tile(abase + k, a, get_write_ptr(cb_in0));
            cb_reserve_back(cb_in1, 1);
            noc_async_read_tile(k, b, get_write_ptr(cb_in1));
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
            cb_push_back(cb_in1, 1);
        }
    }
}
