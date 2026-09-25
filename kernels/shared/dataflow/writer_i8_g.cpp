// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Strided/grouped int8 writer (from c_16, TensorAccessor). args: 0=addr,1=n,2=start,3=stride,4=group.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    const uint32_t group = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out = 16;
    constexpr auto args = TensorAccessorArgs<0>();
    const auto d = TensorAccessor(args, addr);

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t base = (start + i * stride) * group;
        for (uint32_t j = 0; j < group; ++j) {
            cb_wait_front(cb_out, 1);
            noc_async_write_tile(base + j, d, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
