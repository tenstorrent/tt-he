// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the plane-formulation matmul.  C_r = sum_p a_p @ b_{p,r}.
//   LHS a:  NB tiles, a_p at index p              (limb p of operand A)
//   RHS b:  NB*NB tiles, b_{p,r} at index p*NB+r  (limb r of (128^p * B mod q))
// matmul_compute loops  for r(outer) for p(inner): matmul_tiles(a,b) accumulate.
// So per (r, p) feed a[p] and b[p*NB+r].

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t NB = 4;

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    for (uint32_t r = 0; r < NB; ++r) {
        for (uint32_t p = 0; p < NB; ++p) {
            cb_reserve_back(cb_in0, 1);
            noc_async_read_tile(p, a, get_write_ptr(cb_in0));
            cb_reserve_back(cb_in1, 1);
            noc_async_read_tile(p * NB + r, b, get_write_ptr(cb_in1));
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
            cb_push_back(cb_in1, 1);
        }
    }
}
