// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_Bw001 streaming writer — drains c_16 to the C tensor with the same strided/wrapped addressing
// the reader uses, so output tile (i mod W, tile0) always receives A*B for that same row.  That is
// what keeps the correctness check meaningful once n_iter > W: every one of the W resident output
// rows ends up holding the product of the A,B rows it was computed from, no matter how many times
// the loop wrapped over it.
//
// Batched for the same reason as the reader: BATCH writes in flight, one barrier.
//
//   Runtime args: 0=c_addr, 1=n_iter, 2=tile0, 3=C, 4=W, 5=BATCH
//   Compile args: TensorAccessorArgs for C.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t c_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_iter = get_arg_val<uint32_t>(1);
    const uint32_t tile0  = get_arg_val<uint32_t>(2);
    const uint32_t C      = get_arg_val<uint32_t>(3);
    const uint32_t W      = get_arg_val<uint32_t>(4);
    const uint32_t BATCH  = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, c_addr);

    const uint32_t ts = get_tile_size(cb_out);

    uint32_t id = tile0;
    uint32_t row = 0;

    for (uint32_t i = 0; i < n_iter; i += BATCH) {
        cb_wait_front(cb_out, BATCH);
        const uint32_t p = get_read_ptr(cb_out);

        for (uint32_t j = 0; j < BATCH; ++j) {
            noc_async_write_tile(id, c, p + j * ts);
            ++row;
            id += C;
            if (row == W) { row = 0; id = tile0; }
        }
        noc_async_write_barrier();

        cb_pop_front(cb_out, BATCH);
    }
}
