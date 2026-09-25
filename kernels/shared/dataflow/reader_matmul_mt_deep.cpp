// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// reader_matmul_mt_deep — reader_matmul_mt with the DRAM depth rule applied.
//
// Sibling of reader_matmul_mtk_deep.cpp; same reasoning, applied to the K=1 (3-factor) reader.
// reader_matmul_mt.cpp barriers inside the innermost `bk` loop, so two tiles are in flight at a time.
// This issues one whole output plane -- NB pairs = 2*NB tiles -- behind one barrier.
//
// The depth here is necessarily modest (8 tiles at NB=4, against the 64 the K-looped reader reaches
// at Kt=8) because a K=1 matmul simply has less to prefetch per plane. That asymmetry is itself the
// point of measuring both: docs/matmul/exp_ntt001.md 2a predicts the 2-factor gains far more from
// this change, and applying it to only one variant would break the like-for-like comparison the
// ranking depends on.
//
// A NEW FILE, not an edit: reader_matmul_mt.cpp has other callers (host/tests/ntt/ntt32768_dev.cpp).
//
// HOST REQUIREMENT: cb_in0/cb_in1 must each hold at least NB pages, not 2.
//
// Runtime args: identical to reader_matmul_mt.cpp --
//   0=wbase, 1=limbbase, 2=n_tiles, 3=start, 4=stride, 5=mode.
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

    const uint32_t ts0 = get_tile_size(cb_in0);
    const uint32_t ts1 = get_tile_size(cb_in1);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t ti = start + i * stride;
        const uint32_t lb = ti * NB;
        for (uint32_t a = 0; a < NB; ++a) {             // bj — one output plane
            cb_reserve_back(cb_in0, NB);
            cb_reserve_back(cb_in1, NB);
            const uint32_t p0 = get_write_ptr(cb_in0);
            const uint32_t p1 = get_write_ptr(cb_in1);
            for (uint32_t b = 0; b < NB; ++b) {         // bk — all NB pairs in flight
                uint32_t w_idx, l_idx;
                if (mode == 0) { w_idx = a * NB + b; l_idx = lb + b; }
                else           { l_idx = lb + b;     w_idx = b * NB + a; }
                if (mode == 0) {
                    noc_async_read_tile(w_idx, W, p0 + b * ts0);
                    noc_async_read_tile(l_idx, L, p1 + b * ts1);
                } else {
                    noc_async_read_tile(l_idx, L, p0 + b * ts0);
                    noc_async_read_tile(w_idx, W, p1 + b * ts1);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_in0, NB);
            cb_push_back(cb_in1, NB);
        }
    }
}
