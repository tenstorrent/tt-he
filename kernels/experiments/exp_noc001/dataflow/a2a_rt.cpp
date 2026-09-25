// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// a2a_rt — the all-to-all as a DRAM ROUND TRIP, scoped the way the application actually poses it:
// THE DATA IS ALREADY IN L1 AND THE RESULT MUST END IN L1.
//
// exp_noc001's first three routes started with the data in DRAM, so they paid only the READ half of
// the round trip -- the host had put the data there for free.  That is not the situation any stage
// of a real pipeline is in.  A previous kernel left its output in L1; the transpose has to happen;
// the next kernel wants its input in L1.  Under that scope the DRAM route costs
//
//     write my P tiles out   ->   GLOBAL BARRIER   ->   read my G blocks back
//
// and the NoC route costs a put plus the same barrier.  Both now start in L1 and end in L1, both
// pay one barrier of identical construction, and neither gets a phase donated by the host.
//
// THE BARRIER IS NOT OPTIONAL AND WAS THE OTHER ASYMMETRY.  Core s cannot read core n's block until
// core n has written it.  The earlier DRAM routes had no barrier only because the host had already
// staged everything -- i.e. the synchronisation had been performed off-device, for free, before
// timing began, while route C paid for its own.  Here both routes run the SAME semaphore protocol
// (flush, increment every peer, wait for G), so the barrier cancels in the comparison instead of
// being charged to one side.
//
// The scratch buffer is SEPARATE from the input buffer on purpose. Writing the round trip back into
// the buffer it was read from would leave the read able to succeed even if the write never happened,
// and the correctness gate would not notice a missing phase.
//
// RT_BULK selects whole-page reads (read all G*P tiles, keep 1/G of each -- the O(G^2) shape);
// otherwise each core issues G strided reads of exactly its own block.
//
// Zones: a2a_ld  setup, OUTSIDE the timed zone -- models "the previous stage left it in L1"
//        a2a     rt_wr + rt_bar + rt_rd, the whole round trip: THE NUMBER
//        a2a_wb  verification writeback, OUTSIDE the timed zone
//
// Runtime args: 0=in, 1=scratch, 2=out, 3=G, 4=P, 5=me, 6=sem, 7=CH, 8.. = 2*G physical (x,y)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t in_addr  = get_arg_val<uint32_t>(0);
    const uint32_t scr_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t G        = get_arg_val<uint32_t>(3);
    const uint32_t P        = get_arg_val<uint32_t>(4);
    const uint32_t me       = get_arg_val<uint32_t>(5);
    const uint32_t sem_id   = get_arg_val<uint32_t>(6);
    const uint32_t CH       = get_arg_val<uint32_t>(7);

    const auto PAGE = [CH](uint32_t i) { return (i / CH) * 8 + (i % CH); };
    constexpr uint32_t cb_in = tt::CBIndex::c_0, cb_out = tt::CBIndex::c_1, cb_stg = tt::CBIndex::c_2;
    const uint32_t ts  = get_tile_size(cb_in);
    const uint32_t BLK = P * ts / G;
    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,  .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> scr = {.bank_base_address = scr_addr, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb_in, P);  const uint32_t src = get_write_ptr(cb_in);
    cb_reserve_back(cb_out, P); const uint32_t dst = get_write_ptr(cb_out);
    volatile tt_l1_ptr uint32_t* sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    {   DeviceZoneScopedN("a2a_ld");        // the previous stage's output; outside the measurement
        for (uint32_t p = 0; p < P; ++p) noc_async_read_tile(PAGE(me * P + p), in, src + p * ts);
        noc_async_read_barrier();
    }
    {   DeviceZoneScopedN("a2a");
        {   DeviceZoneScopedN("rt_wr");     // L1 -> DRAM: the half the earlier routes never paid
            for (uint32_t p = 0; p < P; ++p) noc_async_write_tile(PAGE(me * P + p), scr, src + p * ts);
            noc_async_write_barrier();
        }
        {   DeviceZoneScopedN("rt_bar");    // identical protocol to a2a_noc.cpp, so it cancels
            for (uint32_t s = 0; s < G; ++s)
                noc_semaphore_inc(::get_noc_addr(get_arg_val<uint32_t>(8 + 2 * s),
                                                 get_arg_val<uint32_t>(8 + 2 * s + 1),
                                                 get_semaphore(sem_id)), 1);
            noc_semaphore_wait(sem, G);
        }
        {   DeviceZoneScopedN("rt_rd");     // DRAM -> L1
#ifdef RT_BULK
            cb_reserve_back(cb_stg, G * P);
            const uint32_t stage = get_write_ptr(cb_stg);
            for (uint32_t t = 0; t < G * P; ++t) noc_async_read_tile(PAGE(t), scr, stage + t * ts);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
            for (uint32_t n = 0; n < G; ++n) {
                volatile tt_l1_ptr uint32_t* s2 =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + n * P * ts + me * BLK);
                for (uint32_t w = 0; w < BLK / 4; ++w) d[n * (BLK / 4) + w] = s2[w];
            }
            cb_push_back(cb_stg, G * P);
#else
            const uint32_t pg = (me * BLK) / ts, off = (me * BLK) % ts;
            for (uint32_t n = 0; n < G; ++n)
                noc_async_read(scr.get_noc_addr(PAGE(n * P + pg), off), dst + n * BLK, BLK);
            noc_async_read_barrier();
#endif
        }
    }
    {   DeviceZoneScopedN("a2a_wb");        // verification only; outside the measurement
        for (uint32_t p = 0; p < P; ++p) noc_async_write_tile(PAGE(me * P + p), out, dst + p * ts);
        noc_async_write_barrier();
    }
    cb_push_back(cb_in, P); cb_push_back(cb_out, P);
}
