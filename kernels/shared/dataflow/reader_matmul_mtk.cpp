// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// K-looped multi-tile plane-matmul reader (large-shape modular matmul). The folded
// DFT matrix is the constant (W accessor); the decomposed data is the dynamic limbs
// (L accessor, laid out [tile][limb] = tile*NB + limb by decompose_limbs).
//
// This core builds output tiles oi = out_start + i*out_stride (i<n_out) of the Mt*Nt
// grid (oi = mt*Nt + nt). For each output plane and each (limb, kt) it feeds one
// (LHS,RHS) pair, matching matmul_compute_mtk's (plane, K_acc) consumption.
//
//   mode 1 (fold-RHS, C = A @ Wconst): plane=r, inner=p
//       LHS c0 = dyn limb_p(A)[mt][kt]    at (mt*Kt+kt)*NB + p
//       RHS c1 = const Wrhs_{p,r}[kt][nt] at ((p*NB+r)*Kt+kt)*Nt + nt
//   mode 0 (fold-LHS, C = Wconst @ B):   plane=p, inner=r
//       LHS c0 = const Wfold_{p,r}[mt][kt] at ((p*NB+r)*Mt+mt)*Kt + kt
//       RHS c1 = dyn limb_r(B)[kt][nt]     at (kt*Nt+nt)*NB + r
//
// Runtime args: 0=wbase,1=limbbase,2=mode,3=Mt,4=Kt,5=Nt,6=n_out,7=out_start,8=out_stride.
// Compile args: TensorAccessorArgs(Wconst) then TensorAccessorArgs(limbs).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t wbase     = get_arg_val<uint32_t>(0);
    const uint32_t limbbase  = get_arg_val<uint32_t>(1);
    const uint32_t mode      = get_arg_val<uint32_t>(2);
    const uint32_t Mt        = get_arg_val<uint32_t>(3);
    const uint32_t Kt        = get_arg_val<uint32_t>(4);
    const uint32_t Nt        = get_arg_val<uint32_t>(5);
    const uint32_t n_out     = get_arg_val<uint32_t>(6);
    const uint32_t out_start = get_arg_val<uint32_t>(7);
    const uint32_t out_stride= get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t NB = 4;

    constexpr auto w_args = TensorAccessorArgs<0>();
    const auto W = TensorAccessor(w_args, wbase);
    constexpr auto l_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto L = TensorAccessor(l_args, limbbase);

    for (uint32_t i = 0; i < n_out; ++i) {
        const uint32_t oi = out_start + i * out_stride;
        const uint32_t mt = oi / Nt;
        const uint32_t nt = oi % Nt;
        for (uint32_t plane = 0; plane < NB; ++plane) {
            for (uint32_t inner = 0; inner < NB; ++inner) {
                for (uint32_t kt = 0; kt < Kt; ++kt) {
                    uint32_t c0_idx, c1_idx;
                    bool c0_is_w;
                    if (mode == 1) {                       // fold-RHS: p=inner, r=plane
                        const uint32_t p = inner, r = plane;
                        c0_idx = (mt * Kt + kt) * NB + p;            // dyn -> c0 (LHS)
                        c1_idx = ((p * NB + r) * Kt + kt) * Nt + nt; // const -> c1 (RHS)
                        c0_is_w = false;
                    } else {                               // fold-LHS: p=plane, r=inner
                        const uint32_t p = plane, r = inner;
                        c0_idx = ((p * NB + r) * Mt + mt) * Kt + kt; // const -> c0 (LHS)
                        c1_idx = (kt * Nt + nt) * NB + r;            // dyn -> c1 (RHS)
                        c0_is_w = true;
                    }
                    cb_reserve_back(cb_in0, 1);
                    cb_reserve_back(cb_in1, 1);
                    if (c0_is_w) {
                        noc_async_read_tile(c0_idx, W, get_write_ptr(cb_in0));
                        noc_async_read_tile(c1_idx, L, get_write_ptr(cb_in1));
                    } else {
                        noc_async_read_tile(c0_idx, L, get_write_ptr(cb_in0));
                        noc_async_read_tile(c1_idx, W, get_write_ptr(cb_in1));
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_in0, 1);
                    cb_push_back(cb_in1, 1);
                }
            }
        }
    }
}
