// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Multi-tile plane-matmul writer: for each cube tile ti = start + i*stride, writes
// the NB int32 plane tiles (popped from c_16 in bj order) to plane indices
// [ti*NB .. ti*NB+NB-1], so recombine can read each tile's 4 planes contiguously.
//
// Runtime args: 0=planebase, 1=n_tiles, 2=start, 3=stride.
// Compile args: TensorAccessorArgs(planes).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t planebase = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles   = get_arg_val<uint32_t>(1);
    const uint32_t start     = get_arg_val<uint32_t>(2);
    const uint32_t stride    = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = 16;
    constexpr uint32_t NB = 4;

    constexpr auto d_args = TensorAccessorArgs<0>();
    const auto d = TensorAccessor(d_args, planebase);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t base = (start + i * stride) * NB;
        for (uint32_t bj = 0; bj < NB; ++bj) {
            cb_wait_front(cb_out, 1);
            noc_async_write_tile(base + bj, d, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
