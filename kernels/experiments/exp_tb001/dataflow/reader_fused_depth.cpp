// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Feed for compute/fused_matmul_a_loop.cpp: n work tiles per core, D of them in flight, for
// exp_Tg018dram.
//
//   Xin      (n int32 tiles, one per work tile)             -> c_0
//   folded W (NB*NB int8 tiles, (p,r) order)                -> c_1
//
// TWO W POLICIES, because the fused matmul's DRAM traffic is dominated by W, not by the data:
// one work tile is 4 KB of input and 16 KB of folded weights.
//
//   w_share = 0   re-read all NB*NB W tiles for every work tile. This is what the tree does today,
//                 where a dispatch carries exactly one work tile, so scaling that pattern to n tiles
//                 per core means n copies of the same 16 KB.
//   w_share = 1   read W ONCE at kernel entry and leave it resident in c_1 for all n work tiles.
//                 Legal whenever the whole core runs against one prime, which is exactly how the
//                 transform passes are scheduled -- a dispatch is a limb.
//
// The difference is the point of the measurement: at n = 32 it is 20 KB/tile against 4.5 KB/tile,
// so if W traffic is what the DRAM path is spending itself on, w_share moves the number and nothing
// else needs to change.
//
// Reads within a group are mutually independent, so a group issues D*(1 + w_reads) reads and one
// barrier. D=1 with w_share=0 reproduces reader_fused_batch.cpp exactly: 1 + 16 reads, one barrier.
// The host sizes c_0 (and c_1, when not shared) at a multiple of D so a group never straddles the
// CB wrap -- see reader_ab_depth.cpp for why that rule exists.
//
// ARITHMETIC INTENSITY KNOB (`extra`): E further 4 KB page reads per work tile, into a scratch CB
// nothing consumes. Compute and results are untouched -- the golden check still has to pass -- so E
// moves ONLY the bytes per tile, the `s` in C = max(C_compute, G*s/B). See reader_ab_depth.cpp for
// the two rules the destinations and page ids have to follow.
//
// Runtime args:  0 = x_addr, 1 = w_addr, 2 = n, 3 = start_tile, 4 = D, 5 = w_share,
//                6 = extra, 7 = n_total, 8 = chan_stride, 9 = chan_off.
// Compile args:  the Xin TensorAccessor's, then the W TensorAccessor's (the order build_mm uses).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3);
    const uint32_t D = get_arg_val<uint32_t>(4);
    const uint32_t w_share = get_arg_val<uint32_t>(5);
    const uint32_t extra = get_arg_val<uint32_t>(6);
    const uint32_t n_total = get_arg_val<uint32_t>(7);
    // CHANNEL MODE. tt-metal maps interleaved page p to DRAM bank (p % NUM_DRAM_BANKS), so walking
    // pages with stride == the bank count keeps every read on ONE channel. chan_stride = 0 is the
    // normal consecutive walk (all channels round-robin); chan_stride = NUM_DRAM_BANKS with a
    // per-core chan_off pins each core to a channel of its own, which is what measures a single
    // channel's bandwidth and how many channels the part actually has.
    const uint32_t chan_stride = get_arg_val<uint32_t>(8);
    const uint32_t chan_off = get_arg_val<uint32_t>(9);
    uint32_t pc = chan_off;

    constexpr uint32_t cb_in = tt::CBIndex::c_0;   // Xin, Int32
    constexpr uint32_t cb_w = tt::CBIndex::c_1;    // folded W, Int8
    constexpr uint32_t cb_junk = tt::CBIndex::c_4; // scratch sink for the extra traffic
    constexpr uint32_t JUNKP = 32;                 // distinct L1 destinations -- see reader_ab_depth
    constexpr uint32_t NB = 4;
    constexpr uint32_t NW = NB * NB;

    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, x_addr);
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto w = TensorAccessor(w_args, w_addr);

    const uint32_t xts = get_tile_size(cb_in);
    const uint32_t wts = get_tile_size(cb_w);
    const uint32_t junk = extra ? get_write_ptr(cb_junk) : 0;

    if (w_share) {   // one copy of W for the whole kernel; compute never pops c_1
        cb_reserve_back(cb_w, NW);
        const uint32_t wp = get_write_ptr(cb_w);
        for (uint32_t i = 0; i < NW; ++i) noc_async_read_tile(i, w, wp + i * wts);
        noc_async_read_barrier();
        cb_push_back(cb_w, NW);
    }

    for (uint32_t i = 0; i < n; i += D) {
        const uint32_t k = (n - i < D) ? (n - i) : D;
        cb_reserve_back(cb_in, k);
        const uint32_t xp = get_write_ptr(cb_in);
        for (uint32_t j = 0; j < k; ++j) {
            const uint32_t id = start + i + j;
            noc_async_read_tile(id, x, xp + j * xts);
            for (uint32_t e = 0; e < extra; ++e)           // arithmetic-intensity knob
                {
                    uint32_t pg;
                    if (chan_stride) { pg = pc % n_total; pc += chan_stride; }
                    else             { pg = (id + 1u + e) % n_total; }
                    noc_async_read_tile(pg, x, junk + (e % JUNKP) * xts);
                }
        }

        if (!w_share) {
            cb_reserve_back(cb_w, k * NW);
            const uint32_t wp = get_write_ptr(cb_w);
            for (uint32_t j = 0; j < k; ++j)
                for (uint32_t t = 0; t < NW; ++t) noc_async_read_tile(t, w, wp + (j * NW + t) * wts);
        }
        noc_async_read_barrier();                          // ONE barrier for the whole group

        cb_push_back(cb_in, k);
        if (!w_share) cb_push_back(cb_w, k * NW);
    }
}
