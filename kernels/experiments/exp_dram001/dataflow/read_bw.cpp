// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// read_bw — a pure DRAM READ bandwidth kernel for exp_dram001, built to test whether the repo's
// 436 GB/s figure is the part's ceiling or only the ceiling of the access pattern that produced it.
//
// Four things are knobs here, because each is a claim in docs/he/dram_guide.md that a p150 study
// disputes:
//
//   PKT    bytes per NoC read.  NOC_MAX_BURST_SIZE on Blackhole is 256 words x 64 B = 16384, so a
//          16 KB read is ONE burst.  The repo has only ever issued 4 KB (tile-sized) reads.
//   NTRID  reads in flight, as a TRUE SLIDING WINDOW via transaction ids -- not a batch.  This is the
//          distinction exp_roofline001 §5b found mattered: `reserve N, issue N, barrier, push N` is a
//          batch, and deeper batches LOSE overlap.  Here read i is tagged trid (i % NTRID) and the
//          window only waits for the read NTRID ago, so NTRID reads are always outstanding and the
//          consumer never waits on a whole group.
//   stride pages between successive reads.  stride=1 walks all 8 banks (bank = page % 8); stride=8
//          pins this core to ONE bank, which is what "place readers adjacent to their banks" needs.
//   noc    which NoC -- set by the host via the processor/noc this kernel is created on.
//
// THE CHECKSUM IS NOT DECORATION.  A bandwidth kernel that reads the wrong size, or whose barrier
// does not actually wait, reports a *higher* number -- the failure mode is silent and flattering. So
// every packet is XOR-folded (first and last word, after its own barrier) into a per-core accumulator
// that the host verifies. Two words per 16 KB is free; it proves the full extent landed and that the
// trid barrier really ordered against it.
//
// Runtime args: 0=src_addr, 1=n_pkts, 2=start_pkt, 3=stride_pkts, 4=out_addr, 5=my_slot
// Compile args: 0=PKT bytes, 1=NTRID

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t n        = get_arg_val<uint32_t>(1);
    const uint32_t start    = get_arg_val<uint32_t>(2);
    const uint32_t stride   = get_arg_val<uint32_t>(3);
    const uint32_t out_addr = get_arg_val<uint32_t>(4);
    const uint32_t slot     = get_arg_val<uint32_t>(5);

    constexpr uint32_t PKT   = get_compile_time_arg_val(0);
    constexpr uint32_t NTRID = get_compile_time_arg_val(1);
    // Each NoC's reader needs its OWN landing buffer. Sharing one CB between the NCRISC and BRISC
    // readers made them overwrite each other's slots: the reads all still happened (so the bandwidth
    // looked right -- 99.7% of 436) but the data was corrupt. The checksum is what caught it.
    constexpr uint32_t cb    = get_compile_time_arg_val(2);

    const InterleavedAddrGen<true> s = {.bank_base_address = src_addr, .page_size = PKT};

    cb_reserve_back(cb, NTRID);
    const uint32_t base = get_write_ptr(cb);          // NTRID landing slots, one per trid
    uint32_t acc = 0;

    auto fold = [&](uint32_t t) {
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + t * PKT);
        acc ^= p[0] ^ p[PKT / 4 - 1];                 // first and last word: proves the extent landed
    };

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t t = i % NTRID;
        if (i >= NTRID) { noc_async_read_barrier_with_trid(t); fold(t); }   // retire the read NTRID ago
        noc_async_read_set_trid(t);
        noc_async_read(get_noc_addr(start + i * stride, s), base + t * PKT, PKT);
    }
    // drain: the still-outstanding reads are the last min(n, NTRID) issued, in issue order
    for (uint32_t i = (n > NTRID) ? (n - NTRID) : 0; i < n; ++i) {
        const uint32_t t = i % NTRID;
        noc_async_read_barrier_with_trid(t);
        fold(t);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    o[0] = acc;
    const InterleavedAddrGen<true> od = {.bank_base_address = out_addr, .page_size = 4096};
    noc_async_write(base, get_noc_addr(slot, od), 4);
    noc_async_write_barrier();
}
