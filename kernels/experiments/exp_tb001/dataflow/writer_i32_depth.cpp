// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// int32 writer with an EXPLICIT IN-FLIGHT DEPTH, the write-side twin of reader_ab_depth.cpp.
//
// Drains c_16 to out[start+i] for i in [0,n), keeping D writes in flight against one barrier.
// D = 1 IS writer_i32.cpp / writer_modmul.cpp -- wait one tile, write, barrier, pop.
//
// Writes are worth a depth knob for the same reason reads are: a write that is barriered
// immediately holds the core for a full DRAM round trip while exactly one 4 KB transaction is
// outstanding. The wrap rule is the reader's -- the host sizes c_16 at a multiple of D, so a group
// of D never straddles the buffer end.
//
// SECOND-NOC READ KNOB (`extra`, `src_addr`). The reader runs on NCRISC/NOC_1 and the writer on
// BRISC/NOC_0, so a kernel whose only reads are in the reader drives exactly ONE of the chip's two
// NoCs. Blackhole has two, each 64 B/cycle per link, routing in opposite directions -- so half the
// fabric sits idle no matter how the cores are placed. That is the standing candidate for why this
// experiment plateaus at ~245 GB/s against a 520 GB/s DRAM specification.
//
// This makes the writer issue `extra` reads of its own per work tile, on NOC_0, into a scratch CB
// nothing consumes. Same two rules as reader_ab_depth: consecutive page ids, distinct L1 slots. The
// writes are unaffected and the golden check still has to pass.
//
// Runtime args: 0 = out_addr, 1 = n, 2 = start_tile, 3 = D, 4 = extra, 5 = n_total, 6 = src_addr.
#include <cstdint>

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t D = get_arg_val<uint32_t>(3);
    const uint32_t extra = get_arg_val<uint32_t>(4);
    const uint32_t n_total = get_arg_val<uint32_t>(5);
    const uint32_t src_addr = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_junk2 = tt::CBIndex::c_5;   // NOC_0 read sink
    constexpr uint32_t JUNKP = 32;
    const uint32_t ts = get_tile_size(cb_out);
    const uint32_t junk2 = extra ? get_write_ptr(cb_junk2) : 0;
    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> src = {
        .bank_base_address = src_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n; i += D) {
        const uint32_t k = (n - i < D) ? (n - i) : D;
        cb_wait_front(cb_out, k);                          // the whole group, then all k writes
        const uint32_t base = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < k; ++j) {
            noc_async_write_tile(start + i + j, s, base + j * ts);
            for (uint32_t e = 0; e < extra; ++e)       // second-NoC read traffic
                noc_async_read_tile((start + i + j + 1u + e) % n_total, src, junk2 + (e % JUNKP) * ts);
        }
        if (extra) noc_async_read_barrier();
        noc_async_write_barrier();
        cb_pop_front(cb_out, k);
    }
}
