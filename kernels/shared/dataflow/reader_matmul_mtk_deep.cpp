// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// reader_matmul_mtk_deep — reader_matmul_mtk with the DRAM depth rule applied.
//
// WHY.  reader_matmul_mtk.cpp puts its `noc_async_read_barrier()` inside the innermost `kt` loop, so
// exactly TWO tiles (one LHS, one RHS) are ever in flight and every pair pays full DRAM round-trip
// latency in series.  docs/matmul/exp_ntt001.md 2a measured the consequence: the 2-factor NTT-32768
// matmul stages run at 3.4-3.8 GB/s per core, ~115 GB/s aggregate over 32 cores -- 26% of the
// 436 GB/s ceiling in docs/he/dram_guide.md, and the stage time tracks TRAFFIC (2.889x) rather than
// MAC count (4.000x), which is the signature of a feed-bound kernel.
//
// dram_guide 3.3 measures a single core at 42.6 GB/s with 96 pages outstanding and 81.5 at 384.
// This kernel issues one whole plane's worth of pairs -- K_acc = NB*Kt pairs = 2*NB*Kt tiles -- behind
// ONE barrier: 64 tiles in flight at Kt=8, against 2.
//
// This is the same fix reader_fused_batch.cpp already made for the HE path (17 reads behind one
// barrier instead of 17 barriers: ph_matmul 6.82 -> 1.25 us, -82%; docs/he/blackhole_port.md 5).
//
// A NEW FILE, not an edit: reader_matmul_mtk.cpp has other callers (host/tests/ntt/ntt32768_2factor_dev.cpp)
// and TT-Metal silently ignores surplus runtime args, so a shared edit would fail silently elsewhere.
//
// ORDER IS PRESERVED EXACTLY.  matmul_compute_mtk consumes K_acc pairs per plane, one at a time, in
// (inner, kt) order; this kernel pushes the same K_acc pairs in the same order, just in one batch.
// Consuming 1 from a CB that was pushed N at a time is ordinary CB semantics.
//
// HOST REQUIREMENT: cb_in0/cb_in1 must each hold at least K_acc = NB*Kt pages, not 2.
//
// Runtime args: identical to reader_matmul_mtk.cpp --
//   0=wbase,1=limbbase,2=mode,3=Mt,4=Kt,5=Nt,6=n_out,7=out_start,8=out_stride.
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

    const uint32_t ts0 = get_tile_size(cb_in0);
    const uint32_t ts1 = get_tile_size(cb_in1);
    const uint32_t K_acc = NB * Kt;                 // pairs consumed per output plane

    for (uint32_t i = 0; i < n_out; ++i) {
        const uint32_t oi = out_start + i * out_stride;
        const uint32_t mt = oi / Nt;
        const uint32_t nt = oi % Nt;
        for (uint32_t plane = 0; plane < NB; ++plane) {
            cb_reserve_back(cb_in0, K_acc);
            cb_reserve_back(cb_in1, K_acc);
            const uint32_t p0 = get_write_ptr(cb_in0);
            const uint32_t p1 = get_write_ptr(cb_in1);
            uint32_t slot = 0;
            for (uint32_t inner = 0; inner < NB; ++inner) {
                for (uint32_t kt = 0; kt < Kt; ++kt, ++slot) {
                    uint32_t c0_idx, c1_idx;
                    bool c0_is_w;
                    if (mode == 1) {                       // fold-RHS: p=inner, r=plane
                        const uint32_t p = inner, r = plane;
                        c0_idx = (mt * Kt + kt) * NB + p;
                        c1_idx = ((p * NB + r) * Kt + kt) * Nt + nt;
                        c0_is_w = false;
                    } else {                               // fold-LHS: p=plane, r=inner
                        const uint32_t p = plane, r = inner;
                        c0_idx = ((p * NB + r) * Mt + mt) * Kt + kt;
                        c1_idx = (kt * Nt + nt) * NB + r;
                        c0_is_w = true;
                    }
                    if (c0_is_w) {                          // ALL of them in flight; one barrier below
                        noc_async_read_tile(c0_idx, W, p0 + slot * ts0);
                        noc_async_read_tile(c1_idx, L, p1 + slot * ts1);
                    } else {
                        noc_async_read_tile(c0_idx, L, p0 + slot * ts0);
                        noc_async_read_tile(c1_idx, W, p1 + slot * ts1);
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_in0, K_acc);
            cb_push_back(cb_in1, K_acc);
        }
    }
}
