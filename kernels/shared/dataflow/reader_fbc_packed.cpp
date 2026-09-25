// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for matmul_fbc_packed. Streams, per super-block sb and output limb r,
// the 4 (A, B) block pairs:  A_{sb*4+bl} -> c_0,  B_{bl,r} -> c_1  (bl = 0..3).
// A is re-streamed for each r. TensorAccessor for both (int8 buffers span > 12
// DRAM banks). Compile args: TensorAccessorArgs(A) then TensorAccessorArgs(B).
// Runtime args: 0 = a_addr, 1 = b_addr, 2 = nsb.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t nsb = get_arg_val<uint32_t>(2);
    const uint32_t nbl = get_arg_val<uint32_t>(3);   // blocks packed per tile = 32/L

    constexpr uint32_t cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1;
    constexpr uint32_t NR = 4;

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr, get_tile_size(cb_a));
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr, get_tile_size(cb_b));

    for (uint32_t sb = 0; sb < nsb; ++sb) {
        for (uint32_t r = 0; r < NR; ++r) {
            for (uint32_t bl = 0; bl < nbl; ++bl) {
                cb_reserve_back(cb_a, 1);
                noc_async_read_tile(sb * nbl + bl, a, get_write_ptr(cb_a));   // A_{sb*nbl+bl}
                noc_async_read_barrier();
                cb_push_back(cb_a, 1);
                cb_reserve_back(cb_b, 1);
                noc_async_read_tile(bl * NR + r, b, get_write_ptr(cb_b));     // B_{bl,r}
                noc_async_read_barrier();
                cb_push_back(cb_b, 1);
            }
        }
    }
}
