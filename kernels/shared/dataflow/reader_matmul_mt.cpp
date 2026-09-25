// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Multi-tile plane-matmul reader. The 32-pt NTT matrix W is the shared folded
// constant (NB*NB = 16 int8 tiles at wbase); the dynamic limbs are 4 int8 tiles
// per cube tile (at limbbase, tile ti uses limbs [ti*4 .. ti*4+3]). This core
// processes cube tiles  ti = start + i*stride,  i in [0, n_tiles).
//
// Two orientations (matmul_compute_mt consumes (bj,bk) order):
//   mode 0 (fold-LHS, W on the left):  C_p = sum_r W[p*NB+r] @ limb[r]
//          bj=p outer, bk=r inner  ->  c0 = W[p*NB+r],   c1 = limb[ti*4+r]
//   mode 1 (fold-RHS, W on the right): C_r = sum_p limb[p] @ W[p*NB+r]
//          bj=r outer, bk=p inner  ->  c0 = limb[ti*4+p], c1 = W[p*NB+r]
//
// Runtime args: 0=wbase, 1=limbbase, 2=n_tiles, 3=start, 4=stride, 5=mode.
// Compile args: TensorAccessorArgs(W) then TensorAccessorArgs(limbs).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t wbase    = get_arg_val<uint32_t>(0);
    const uint32_t limbbase = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles  = get_arg_val<uint32_t>(2);
    const uint32_t start    = get_arg_val<uint32_t>(3);
    const uint32_t stride   = get_arg_val<uint32_t>(4);
    const uint32_t mode     = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t NB = 4;

    constexpr auto w_args = TensorAccessorArgs<0>();
    const auto W = TensorAccessor(w_args, wbase);
    constexpr auto l_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto L = TensorAccessor(l_args, limbbase);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t ti = start + i * stride;
        const uint32_t lb = ti * NB;
        for (uint32_t a = 0; a < NB; ++a) {        // bj
            for (uint32_t b = 0; b < NB; ++b) {    // bk
                uint32_t w_idx, l_idx;
                if (mode == 0) { w_idx = a * NB + b; l_idx = lb + b; }   // c0=W[p*NB+r], c1=limb[r]
                else           { l_idx = lb + b;     w_idx = b * NB + a; } // c0=limb[p], c1=W[p*NB+r]
                cb_reserve_back(cb_in0, 1);
                cb_reserve_back(cb_in1, 1);
                if (mode == 0) {
                    noc_async_read_tile(w_idx, W, get_write_ptr(cb_in0));
                    noc_async_read_tile(l_idx, L, get_write_ptr(cb_in1));
                } else {
                    noc_async_read_tile(l_idx, L, get_write_ptr(cb_in0));
                    noc_async_read_tile(w_idx, W, get_write_ptr(cb_in1));
                }
                noc_async_read_barrier();
                cb_push_back(cb_in0, 1);
                cb_push_back(cb_in1, 1);
            }
        }
    }
}
