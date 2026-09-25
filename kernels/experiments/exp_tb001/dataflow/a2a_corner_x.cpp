// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// a2a_corner_x — the exp_Tg018 corner turn as a CORE-TO-CORE NoC EXCHANGE instead of a DRAM
// round trip.  Port of dataflow/all_to_all_corner_x.cpp (docs/matmul/exp_ntt001.md §2b-sexies,
// measured 5.40x there) to exp_Tg018's indexing.  a2a_corner_{rd,wr}.cpp are RETAINED unchanged.
//
//   out[s][n1][t] = in[n1][s][t]        (swap the tile axis n1 with the row axis s, keep column t)
//
// WHY.  a2a_corner_{rd,wr}.cpp have every core read ALL NT input tiles -- 128 KB at NT=32 -- to take
// K rows out of each.  exp_ntt001 §2b-quinquies measured that shape as READ-bound: 64% of the stage
// is read+write+launch, only 36% is the L1 assembly.  But the rows every core wants are already
// on-chip, in some other core's L1, so the DRAM traffic is redundant.  Here each core reads only the
// NPC tiles it OWNS and pushes each row straight into the peer that needs it.
//
// AND IT REMOVES THE ASSEMBLY, which the DRAM version cannot.  There, a core must gather NT scattered
// rows into face-tiled order.  Here the SENDER knows where its row lands, so it writes directly into
// the receiver's final face layout and the receiver assembles nothing:
//
//   a 32x32 tile is 2x2 faces of 16x16; element (r,t) sits at uint32 offset
//       ((r/16)*2 + (t/16))*256 + (r%16)*16 + (t%16)
//   so row r is TWO runs of 16 contiguous words at ((r>>4)*2 + {0,1})*256 + (r&15)*16.
//
// WHAT IS DIFFERENT FROM all_to_all_corner_x.cpp, and why this is not just a copy:
//
//   * PAGE STRIDE.  exp_Tg018 interleaves the two half-transforms in one 64-page buffer, so input
//     tile n1 is page n1*2+kh and output row s is page s*2+kh -- not page n1.  (Note this touches
//     only 4 of the 8 DRAM banks, since bank = page % 8 and the stride is 2.  It costs the DRAM
//     version dearly and costs this one almost nothing, because this one reads NPC pages, not NT.)
//   * NPC > 1.  all_to_all_corner_x assumes one tile per core and therefore NT cores per transform.
//     exp_Tg018m gives each core K=2 output tiles (its measured optimum -- 32 reads amortised over
//     two writes), i.e. P = NT/K cores per transform.  Generalising to NPC = NT/P tiles per core
//     keeps the dispatch count and the row assignment IDENTICAL to a2a_corner_wr.cpp's
//     (out_start=me, out_stride=P), so this is a like-for-like swap rather than a reshaped stage.
//
// A core owns input tiles AND output rows {me, me+P, me+2P, ...} -- the same index set for both,
// which is what makes the exchange symmetric: everyone sends to everyone exactly once per own tile.
//
// SYNCHRONISATION.  Core-to-core, so it needs a real barrier (see all_to_all_corner_x.cpp -- the
// repo's first).  The protocol:
//   1. every core writes all its rows to all P peers, then FLUSHES;
//   2. it then increments every peer's semaphore once, so an announcement always follows the data it
//      announces -- the flush in step 1 is what makes that ordering meaningful;
//   3. it waits for its own semaphore to reach P, i.e. every peer in its block has delivered;
//   4. only then does it write its NPC assembled tiles to DRAM.
// A core that never sends hangs its whole block, so THE HOST MUST DISPATCH WHOLE BLOCKS OF P CORES.
// exp_Tg018m's chunking (`base += NC`) can split a block across two dispatches, which would deadlock;
// exp_tg018ntt chunks in whole blocks instead.  The peer L1 address is this core's own c_2 write
// pointer, valid because every core runs an identical program and so has an identical CB layout.
//
// Runtime args: 0=NT, 1=P, 2=me (index within block), 3=kh, 4=in_addr, 5=out_addr, 6=sem_id,
//               7.. = 2*P physical NoC coords (x,y) of this core's block, peer-index order.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    // Dataflow-only stage: without a named zone its dispatch window emits no name and
    // tools/traces/dispatch_csv.py labels it with a neighbouring dispatch's zone. Same name as
    // a2a_corner_{rd,wr}.cpp so the census line is directly comparable.
    DeviceZoneScopedN("a2a_corner");

    const uint32_t NT       = get_arg_val<uint32_t>(0);
    const uint32_t P        = get_arg_val<uint32_t>(1);
    const uint32_t me       = get_arg_val<uint32_t>(2);
    const uint32_t kh       = get_arg_val<uint32_t>(3);
    const uint32_t in_addr  = get_arg_val<uint32_t>(4);
    const uint32_t out_addr = get_arg_val<uint32_t>(5);
    const uint32_t sem_id   = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;   // the NPC tiles I own
    constexpr uint32_t cb_o  = tt::CBIndex::c_2;   // the NPC tiles I assemble == every peer's target
    const uint32_t tbytes = get_tile_size(cb_in);
    const uint32_t NPC = NT / P;

    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,
                                              .page_size = tbytes, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr,
                                              .page_size = tbytes, .data_format = DataFormat::UInt32};

    // ---- 1. my own tiles only: NPC pages, not NT ----
    cb_reserve_back(cb_in, NPC);
    const uint32_t src0 = get_write_ptr(cb_in);
    for (uint32_t li = 0; li < NPC; ++li) {
        noc_async_read_tile((me + li * P) * 2 + kh, in, src0 + li * tbytes);
    }
    noc_async_read_barrier();

    cb_reserve_back(cb_o, NPC);
    const uint32_t dst_l1 = get_write_ptr(cb_o);          // same L1 offset on every core
    volatile tt_l1_ptr uint32_t* sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    constexpr uint32_t RUN = 16 * 4;                      // 16 words = 64 B, one face-row run

    // ---- 2. scatter: row s of my tile n1 -> row n1 of the core that owns output row s ----
    for (uint32_t li = 0; li < NPC; ++li) {
        const uint32_t n1 = me + li * P;
        const uint32_t src = src0 + li * tbytes;
        // destination row is n1 -- my own tile index -- so these are loop-invariant in s
        const uint32_t d0 = (((n1 >> 4) * 2) * 256 + (n1 & 15) * 16) * 4;
        const uint32_t d1 = (((n1 >> 4) * 2 + 1) * 256 + (n1 & 15) * 16) * 4;
        for (uint32_t s = 0; s < NT; ++s) {
            const uint32_t peer = s % P, slot = s / P;                  // who owns output row s, where
            const uint32_t px = get_arg_val<uint32_t>(7 + 2 * peer);
            const uint32_t py = get_arg_val<uint32_t>(7 + 2 * peer + 1);
            const uint32_t s0 = (((s >> 4) * 2) * 256 + (s & 15) * 16) * 4;
            const uint32_t s1 = (((s >> 4) * 2 + 1) * 256 + (s & 15) * 16) * 4;
            const uint32_t base = dst_l1 + slot * tbytes;
            noc_async_write(src + s0, ::get_noc_addr(px, py, base + d0), RUN);   // cols 0..15
            noc_async_write(src + s1, ::get_noc_addr(px, py, base + d1), RUN);   // cols 16..31
        }
    }
    noc_async_write_barrier();                       // my data has landed everywhere...

    // ---- 3. ...so announcing it is now truthful ----
    for (uint32_t peer = 0; peer < P; ++peer) {
        const uint32_t px = get_arg_val<uint32_t>(7 + 2 * peer);
        const uint32_t py = get_arg_val<uint32_t>(7 + 2 * peer + 1);
        noc_semaphore_inc(::get_noc_addr(px, py, reinterpret_cast<uint32_t>(sem)), 1);
    }
    noc_semaphore_wait(sem, P);                      // every peer in my block delivered

    // ---- 4. my tiles are complete: every word came from some peer's row ----
    for (uint32_t li = 0; li < NPC; ++li) {
        noc_async_write_tile((me + li * P) * 2 + kh, out, dst_l1 + li * tbytes);
    }
    noc_async_write_barrier();

    cb_push_back(cb_in, NPC);
    cb_push_back(cb_o, NPC);
}
