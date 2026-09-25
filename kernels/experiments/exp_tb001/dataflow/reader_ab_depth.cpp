// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Two-operand reader with an EXPLICIT IN-FLIGHT DEPTH, for exp_Tg018dram.
//
// Reads A[start+i] and B[start+i] for i in [0,n) into c_0 / c_1, keeping D tile PAIRS -- 2D NOC
// reads -- in flight against a single barrier.
//
// WHY A DEPTH ARGUMENT. Utilisation of the DRAM path tracks BYTES OUTSTANDING per core, not bytes
// requested: measured on this card at a fixed 64 cores, 2 tiles in flight reaches 16% of achievable
// bandwidth, 17 tiles 25%, 64 tiles 94% (docs/he/exp_tg018.md). Every reader in the tree today keeps
// exactly one tile pair in flight, so D is the axis the sweep has to be able to move.
//
// D = 1 IS reader_modmul.cpp: reserve one slot on each CB, issue both reads, one barrier, push. That
// is deliberate -- the D=1 row of the sweep must be the production kernel, not an approximation of it.
//
// GROUPS MUST NOT STRADDLE THE CB WRAP. get_write_ptr() returns the write cursor and tile j of a
// group is written at cursor + j*tile_size with no wrap handling. The host therefore sizes both CBs
// at an INTEGER MULTIPLE of D (it uses 2D, for double buffering), which makes every group start on a
// D-aligned slot and end at or before the buffer end. Same rule as reader_gather_batch.cpp.
//
// ARITHMETIC INTENSITY KNOB (`extra`). Both kernels in exp_Tg018dram are compute-bound at every core
// count this card has, so the memory-bound half of the roofline is unreachable by sweeping G alone.
// `extra` adds E further 4 KB page reads per work tile into a scratch CB nothing consumes. Compute is
// untouched and the result is unchanged -- the golden check still has to pass -- so E moves ONLY the
// bytes per tile, which is exactly the `s` in C = max(C_compute, G*s/B). Sweeping E walks the knee
// across the core grid.
//
// TWO RULES, and getting either wrong measures the harness instead of the machine:
//
//  1. CONSECUTIVE PAGE IDS. tt-metal maps interleaved page ids round-robin over the DRAM channels, so
//     ids p, p+1, p+2, ... land on successive channels and a group of E reads spreads across all of
//     them. An earlier version used a prime stride, which covers the channels but not in the clean
//     round-robin the hardware is designed for.
//  2. SEPARATE L1 DESTINATIONS. Every outstanding read needs its own landing area. An earlier version
//     pointed all E reads at ONE L1 address; distinct DRAM pages, but a single destination stream,
//     which throttles at the destination rather than exercising the channels. JUNKP slots fix it.
//
// Rule 2 is also why a single core cannot use the whole machine: its inbound NoC port is ~64 B/cycle
// (86.4 GB/s), reached after roughly three concurrent channel streams, so aggregate DRAM bandwidth
// needs many cores no matter how deep one core's request queue is.
//
// Runtime args: 0 = a_addr, 1 = b_addr, 2 = n, 3 = start_tile, 4 = D, 5 = extra, 6 = n_total, 7 = chan_stride, 8 = chan_off.
#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3);
    const uint32_t D = get_arg_val<uint32_t>(4);
    const uint32_t extra = get_arg_val<uint32_t>(5);
    const uint32_t n_total = get_arg_val<uint32_t>(6);
    // CHANNEL MODE. tt-metal maps interleaved page p to DRAM bank (p % NUM_DRAM_BANKS), so walking
    // pages with stride == the bank count keeps every read on ONE channel. chan_stride = 0 is the
    // normal consecutive walk (all channels round-robin); chan_stride = NUM_DRAM_BANKS with a
    // per-core chan_off pins each core to a channel of its own, which is what measures a single
    // channel's bandwidth and how many channels the part actually has.
    const uint32_t chan_stride = get_arg_val<uint32_t>(7);
    const uint32_t chan_off = get_arg_val<uint32_t>(8);
    uint32_t pc = chan_off;

    constexpr uint32_t cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_junk = tt::CBIndex::c_4;   // scratch sink for the extra traffic
    constexpr uint32_t JUNKP = 32;                   // distinct L1 destinations, one per outstanding read
    const uint32_t ts = get_tile_size(cb_a);
    const uint32_t junk = extra ? get_write_ptr(cb_junk) : 0;
    const InterleavedAddrGenFast<true> a = {
        .bank_base_address = a_addr, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> b = {
        .bank_base_address = b_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n; i += D) {
        const uint32_t k = (n - i < D) ? (n - i) : D;
        cb_reserve_back(cb_a, k);
        cb_reserve_back(cb_b, k);
        const uint32_t pa = get_write_ptr(cb_a), pb = get_write_ptr(cb_b);
        for (uint32_t j = 0; j < k; ++j) {                 // 2k reads in flight...
            const uint32_t id = start + i + j;
            noc_async_read_tile(id, a, pa + j * ts);
            noc_async_read_tile(id, b, pb + j * ts);
            for (uint32_t e = 0; e < extra; ++e)           // ...plus E consecutive-page dummies each
                {
                    uint32_t pg;
                    if (chan_stride) { pg = pc % n_total; pc += chan_stride; }
                    else             { pg = (id + 1u + e) % n_total; }
                    noc_async_read_tile(pg, a, junk + (e % JUNKP) * ts);
                }
        }
        noc_async_read_barrier();                          // ...then ONE barrier
        cb_push_back(cb_a, k);
        cb_push_back(cb_b, k);
    }
}
