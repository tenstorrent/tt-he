// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// all_to_all_corner_x — the corner turn as a CORE-TO-CORE NoC EXCHANGE instead of a DRAM round trip.
//
//   D[s][n1][t] = Cb[n1][s][t]      (swap the tile axis n1 with the row axis s, keep the column t)
//
// WHY.  all_to_all_corner_{mc,rd,wr}.cpp all have every core read ALL NT input tiles (128 KB at NT=32)
// out of DRAM in order to take ONE row from each.  docs/matmul/exp_ntt001.md 2b-quinquies measured that
// stage as 15.48 us of read+write+launch against 8.74 us of L1 assembly -- i.e. READ-bound, 64%.
// Core n1 already holds Cb[n1] though: it is the tile that core's own Block-1 just produced.  So the
// data every other core wants is on-chip, and fetching it from DRAM is redundant.
//
// Here each core reads only ITS OWN tile (4 KB) and then pushes each row straight into the peer that
// needs it.  DRAM traffic for the stage: 128 KB -> 4 KB per core, 16x less.
//
// AND IT REMOVES THE ASSEMBLY TOO, which the DRAM versions cannot.  Those must gather 32 scattered
// rows out of 32 staged tiles into face-tiled order (1024 word copies).  Here the SENDER knows exactly
// where its row belongs in the receiver's tile, so it writes into the final face-layout offsets
// directly and the receiver assembles nothing:
//
//   a 32x32 tile is 2x2 faces of 16x16; element (r,t) sits at uint32 offset
//       ((r/16)*2 + (t/16))*256 + (r%16)*16 + (t%16)
//   so row r is TWO runs of 16 contiguous words, at ((r>>4)*2 + {0,1})*256 + (r&15)*16.
//
// My destination row is always n1 = my own index, so the destination offsets are LOOP-INVARIANT; only
// the source row moves. Two 64-byte writes per peer, 64 per core.
//
// THE TRADE, stated because it may not pay: this replaces 32 sequential 4 KB DRAM page reads -- which
// docs/he/dram_guide.md 3.3/3.4 say is the shape DRAM likes -- with 64 sixty-four-byte NoC writes,
// which is the shape it does not. The DRAM reads are also shared across cores, so the fabric may
// already be serving them efficiently. Whether the exchange wins is a measurement, not a deduction.
//
// SYNCHRONISATION.  This is the FIRST core-to-core barrier in this repo -- nothing else here uses
// noc_semaphore_*; every other stage boundary goes through DRAM. The protocol:
//   1. every core writes its 32 rows to its 32 peers, then flushes;
//   2. every core then increments EVERY peer's semaphore once (so an increment always follows the data
//      it announces -- the flush in step 1 is what makes that ordering meaningful);
//   3. every core waits for its own semaphore to reach NT, i.e. all NT peers have delivered;
//   4. only then does it write its assembled tile to DRAM.
// A core that never sends hangs all the others, so this REQUIRES the full NT cores to be in the range:
// the host must run C == NT. The peer L1 address is taken from this core's own c_2 write pointer, valid
// because every core runs an identical program and therefore has an identical CB layout.
//
// Runtime args: 0=in_addr, 1=out_addr, 2=NT, 3=my_index, 4=sem_id,
//               5.. = 2*NT physical NoC coords, (x,y) per core index.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    DeviceZoneScopedN("a2a_corner");

    const uint32_t in_addr  = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t NT       = get_arg_val<uint32_t>(2);
    const uint32_t me       = get_arg_val<uint32_t>(3);
    const uint32_t sem_id   = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;    // my own input tile
    constexpr uint32_t cb_o  = tt::CBIndex::c_2;    // my output tile == every peer's write target
    const uint32_t tbytes = get_tile_size(cb_in);

    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,
                                              .page_size = tbytes, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr,
                                              .page_size = tbytes, .data_format = DataFormat::UInt32};

    // ---- 1. my own tile only: 4 KB, not 128 ----
    cb_reserve_back(cb_in, 1);
    const uint32_t src = get_write_ptr(cb_in);
    noc_async_read_tile(me, in, src);
    noc_async_read_barrier();

    cb_reserve_back(cb_o, 1);
    const uint32_t dst_l1 = get_write_ptr(cb_o);          // same L1 offset on every core
    volatile tt_l1_ptr uint32_t* sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    // destination row is always `me`, so these are loop-invariant
    const uint32_t o0 = (((me >> 4) * 2) * 256 + (me & 15) * 16) * 4;       // byte offsets
    const uint32_t o1 = (((me >> 4) * 2 + 1) * 256 + (me & 15) * 16) * 4;
    constexpr uint32_t RUN = 16 * 4;                                        // 16 words = 64 B

    // ---- 2. scatter: row s of my tile -> row `me` of peer s's tile, in final face layout ----
    for (uint32_t s = 0; s < NT; ++s) {
        const uint32_t px = get_arg_val<uint32_t>(5 + 2 * s);
        const uint32_t py = get_arg_val<uint32_t>(5 + 2 * s + 1);
        const uint32_t i0 = (((s >> 4) * 2) * 256 + (s & 15) * 16) * 4;
        const uint32_t i1 = (((s >> 4) * 2 + 1) * 256 + (s & 15) * 16) * 4;
        noc_async_write(src + i0, ::get_noc_addr(px, py, dst_l1 + o0), RUN);   // cols 0..15
        noc_async_write(src + i1, ::get_noc_addr(px, py, dst_l1 + o1), RUN);   // cols 16..31
    }
    noc_async_write_barrier();                       // my data has landed everywhere...

    // ---- 3. ...so announcing it is now truthful ----
    for (uint32_t s = 0; s < NT; ++s) {
        const uint32_t px = get_arg_val<uint32_t>(5 + 2 * s);
        const uint32_t py = get_arg_val<uint32_t>(5 + 2 * s + 1);
        noc_semaphore_inc(::get_noc_addr(px, py, reinterpret_cast<uint32_t>(sem)), 1);
    }
    noc_semaphore_wait(sem, NT);                     // all NT peers delivered

    // ---- 4. my tile is complete: every word came from some peer's row ----
    noc_async_write_tile(me, out, dst_l1);
    noc_async_write_barrier();

    cb_push_back(cb_in, 1);
    cb_push_back(cb_o, 1);
}
