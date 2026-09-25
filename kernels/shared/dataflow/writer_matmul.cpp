// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writes the NB int32 product tiles P_bj from c_16 to DRAM, in bj order
// (output tile index == bj). Host then recombines + reduces mod q.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_out = 16;
    constexpr uint32_t NB = 4;   // 32 / 8

    constexpr auto d_args = TensorAccessorArgs<0>();
    const auto d = TensorAccessor(d_args, dst_addr);

    for (uint32_t t = 0; t < NB; ++t) {
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(t, d, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
