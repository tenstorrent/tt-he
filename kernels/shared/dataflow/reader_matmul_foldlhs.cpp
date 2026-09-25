// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Matmul reader for the "fold into the constant LHS" orientation, used by stage A
// of the NTT (G = W @ Xin) so the runtime fold is eliminated:
//
//   a (LHS) = folded constant W:  NB*NB tiles, a_{p,r} = limb_p(128^r * W mod q)  [baked offline]
//   b (RHS) = Xin limbs:          NB tiles,      b_r   = limb_r(Xin)              [plain decompose]
//
// Compute does  C_p = sum_r a[p*NB+r] @ b[r]  (matmul_compute loops bj=p outer, bk=r inner),
// so feed, per (p, r):  cb_in0 <- a[p*NB+r]   (LHS, 16 tiles),  cb_in1 <- b[r]  (RHS, 4 tiles).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);   // folded W (NB*NB tiles)
    const uint32_t b_addr = get_arg_val<uint32_t>(1);   // Xin limbs (NB tiles)

    constexpr uint32_t cb_in0 = 0;   // LHS, Int8
    constexpr uint32_t cb_in1 = 1;   // RHS, Int8
    constexpr uint32_t NB = 4;

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    for (uint32_t p = 0; p < NB; ++p) {
        for (uint32_t r = 0; r < NB; ++r) {
            cb_reserve_back(cb_in0, 1);
            noc_async_read_tile(p * NB + r, a, get_write_ptr(cb_in0));   // LHS = folded W
            cb_reserve_back(cb_in1, 1);
            noc_async_read_tile(r, b, get_write_ptr(cb_in1));            // RHS = Xin limb r
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
            cb_push_back(cb_in1, 1);
        }
    }
}
