// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reads n_groups groups of `gsize` CONSECUTIVE int32 tiles, the groups themselves strided.
//   group g, element j  ->  tile  base + g*gstride + j
//
// Why: recombine_bigK consumes groups of 4 consecutive plane tiles, but scan_gemm writes 8 planes per
// output tile (`[o][d]`), so group A is tiles o*8+0..3 and group B is o*8+4..7. reader_i32_stride reads one
// tile per stride step, not a run of them, hence this variant. It is what lets the 7-plane mod-q recombine
// reuse recombine_bigK twice instead of needing a new compute kernel (see latency_optimization.md §4f C1).
//
//   Runtime args: 0=addr, 1=n_groups, 2=gsize, 3=gstride, 4=base

#include <cstdint>

void kernel_main() {
    const uint32_t addr     = get_arg_val<uint32_t>(0);
    const uint32_t n_groups = get_arg_val<uint32_t>(1);
    const uint32_t gsize    = get_arg_val<uint32_t>(2);
    const uint32_t gstride  = get_arg_val<uint32_t>(3);
    const uint32_t base     = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t ts = get_tile_size(cb_in);
    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t g = 0; g < n_groups; ++g) {
        for (uint32_t j = 0; j < gsize; ++j) {
            cb_reserve_back(cb_in, 1);
            noc_async_read_tile(base + g * gstride + j, s, get_write_ptr(cb_in));
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
}
