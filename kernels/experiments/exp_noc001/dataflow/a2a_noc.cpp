// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// a2a_noc — the same all-to-all as a CORE-TO-CORE NoC EXCHANGE, no DRAM in the exchange at all.  Route C.
//
// Generalised from all_to_all_corner_x.cpp, which took the NTT corner turn from 23.97 to 4.44 us
// (docs/matmul/exp_ntt001.md 2b-sexies).  That kernel is hard-wired to the tile geometry: 32 rows means
// exactly 32 participants sending 64 B each.  Here the block is the unit instead of the row, so G and
// the message size sweep independently:
//
//     G participants, P tiles each, BLK = P*4096/G bytes per message
//     G=4,P=1 -> 1024 B      G=64,P=1 -> 64 B      G=32,P=4 -> 512 B
//
// Message size is the axis that matters: 64 B is a quarter of the NoC's 256 B packet and a sixty-fourth
// of a DRAM page, so the small-G end and the large-G end stress completely different things.
//
// AS IN THE CORNER TURN, THE SENDER WRITES INTO THE RECEIVER'S FINAL OFFSET.  My block always lands at
// offset me*BLK in every peer, so the destination offset is loop-invariant and the receiver assembles
// nothing.  Removing the receive-side gather was worth more than removing the DRAM traffic in the
// corner turn (8.74 us of 24), and it is only possible because the sender knows the layout.
//
// ACCOUNTING (this is the part that decides whether the comparison is fair).  Three zones:
//   a2a     — the exchange proper.  The initial read of MY OWN P tiles is deliberately OUTSIDE it:
//             in the real pipeline that data is already in L1 from the previous stage, which is the
//             actual reason the exchange wins.  Routes A and B have no equivalent to exclude -- their
//             read IS the exchange -- so this zone alone would flatter C.
//   a2a_ld  — that setup read, measured, so it can be added back.
//   a2a_wb  — the final writeback, identical in all three routes.
// exp_noc001 reports a2a AND (a2a_ld + a2a + a2a_wb).  The second is the like-for-like number; the
// first is the one that transfers to a fused pipeline.
//
// A core that never sends hangs every other core, so the host MUST launch exactly G cores.
//
// -DA2A_NO_PAYLOAD (NOC1_ROUTE=syncprobe) DELETES THE PAYLOAD WRITES and keeps the barrier, so
// a2a_sync can be priced on its own.  It is needed because a2a_sync is NOT pure barrier overhead:
// a2a_put ends when MY writes are flushed, while peers' writes may still be in flight, so a2a_sync
// absorbs the slowest peer's tail as well as the 2G barrier messages.  With no payload there is no
// tail, and what is left is the barrier.  Like all_to_all_corner_probe.cpp this computes a WRONG
// answer on purpose -- a syncprobe run MUST fail its correctness gate, and that failure is the
// evidence the probe was actually the kernel that ran.
//
//
// DRAM CHANNEL CONTROL.  An interleaved DRAM buffer round-robins page i onto bank i%8, so a kernel
// that walks pages consecutively always uses all 8 channels and the channel count is not a free
// variable.  To make it one, logical tile i is placed at PHYSICAL page
//
//     PAGE(i) = (i/CH)*8 + (i%CH)          CH = channels to use, 1|2|4|8
//
// every one of which has page%8 < CH, so exactly CH banks are ever touched (CH=8 reduces to the
// identity). The buffer is padded to suit; the wasted pages are never read.  The host places and
// recovers the data through the same map, so the permutation and its golden are unchanged.
//
// Runtime args: 0=in_addr, 1=out_addr, 2=G, 3=P, 4=me, 5=CH, 5=sem_id, 6.. = 2*G physical (x,y)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr= get_arg_val<uint32_t>(1);
    const uint32_t G       = get_arg_val<uint32_t>(2);
    const uint32_t P       = get_arg_val<uint32_t>(3);
    const uint32_t me      = get_arg_val<uint32_t>(4);
    const uint32_t CH      = get_arg_val<uint32_t>(6);
    const uint32_t sem2_id = get_arg_val<uint32_t>(7);
    const uint32_t sem_id  = get_arg_val<uint32_t>(5);

    const auto PAGE = [CH](uint32_t i) { return (i / CH) * 8 + (i % CH); };
    constexpr uint32_t cb_in = tt::CBIndex::c_0, cb_out = tt::CBIndex::c_1;
    const uint32_t ts  = get_tile_size(cb_in);
    const uint32_t BLK = P * ts / G;
    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,  .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb_in, P);
    const uint32_t src = get_write_ptr(cb_in);
    cb_reserve_back(cb_out, P);
    const uint32_t dst = get_write_ptr(cb_out);          // same L1 offset on every core: identical program
    volatile tt_l1_ptr uint32_t* sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    {   // setup: my own P tiles, contiguous pages -- the shape DRAM likes.  Outside the exchange zone.
        DeviceZoneScopedN("a2a_ld");
        for (uint32_t p = 0; p < P; ++p) noc_async_read_tile(PAGE(me * P + p), in, src + p * ts);
        noc_async_read_barrier();
    }
#ifdef A2A_ALIGN
    {   // ALIGN THE START OF THE TIMED ZONE.  Without this, a2a_sync silently charges the exchange
        // for skew created BEFORE it: the setup load is a DRAM read, and at one channel the cores
        // finish it at very different times, so noc_semaphore_wait -- which waits for the SLOWEST
        // peer -- absorbs that spread.  The exchange then appears to depend on DRAM channel count
        // while moving no DRAM traffic at all.  This barrier parks the skew OUTSIDE the measurement,
        // which is also the honest model of a fused pipeline, where the previous stage's output
        // arrives in rough lockstep rather than staggered by memory contention.
        DeviceZoneScopedN("a2a_align");
        for (uint32_t s = 0; s < G; ++s)
            noc_semaphore_inc(::get_noc_addr(get_arg_val<uint32_t>(8 + 2 * s),
                                             get_arg_val<uint32_t>(8 + 2 * s + 1),
                                             get_semaphore(sem2_id)), 1);
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem2_id)), G);
    }
#endif
    {   // exchange: block s of mine -> offset me*BLK of peer s.  Destination offset loop-invariant.
        // Split in two, because "is the exchange fabric-bound or barrier-bound?" is the question the
        // route-B comparison raises and it is not answerable from the total: a2a_put is G payload
        // writes plus the flush, a2a_sync is the 2G-message barrier that carries no payload at all.
        DeviceZoneScopedN("a2a");
        {   DeviceZoneScopedN("a2a_put");
#ifdef A2A_NO_PAYLOAD
            // One sentinel word so the probe's output is DEFINITIVELY wrong.  Without it the probe
            // once passed its gate at (G=32,P=4): the CBs sit at the same L1 addresses in every run,
            // so the receive buffer still held the CORRECT answer left by the previous real run, and
            // the writeback shipped it to DRAM.  A probe that "computes nothing" does not thereby
            // compute garbage -- it computes whatever was there last time.
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst)[0] = 0xDEADBEEFu;
#else
            for (uint32_t s = 0; s < G; ++s) {
                const uint32_t px = get_arg_val<uint32_t>(8 + 2 * s);
                const uint32_t py = get_arg_val<uint32_t>(8 + 2 * s + 1);
                noc_async_write(src + s * BLK, ::get_noc_addr(px, py, dst + me * BLK), BLK);
            }
            noc_async_write_barrier();                   // flush before announcing
#endif
        }
        {   DeviceZoneScopedN("a2a_sync");
            for (uint32_t s = 0; s < G; ++s) {
                const uint32_t px = get_arg_val<uint32_t>(8 + 2 * s);
                const uint32_t py = get_arg_val<uint32_t>(8 + 2 * s + 1);
                noc_semaphore_inc(::get_noc_addr(px, py, get_semaphore(sem_id)), 1);
            }
            noc_semaphore_wait(sem, G);                  // all G peers have delivered
        }
    }
    {   DeviceZoneScopedN("a2a_wb");
        for (uint32_t p = 0; p < P; ++p) noc_async_write_tile(PAGE(me * P + p), out, dst + p * ts);
        noc_async_write_barrier();
    }
    cb_push_back(cb_in, P); cb_push_back(cb_out, P);
}
