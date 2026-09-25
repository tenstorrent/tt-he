// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ntt64_fused_wr — BRISC/NOC_0 half of ntt64_fused.cpp.  It does two jobs, in order:
//
//   1. THE CORNER TURN, entirely core-to-core, while the compute kernel is parked on cb_d.
//      It waits for the compute kernel's four twiddled tiles on cb_x, scatters their rows straight
//      into the peers that need them -- in final face layout, so nobody assembles anything -- and
//      pushes cb_d once the whole 16-core block has delivered.  NOTHING here touches DRAM: this is
//      the difference from a2a_corner_x.cpp, which still had to read its input and write its output.
//   2. THE ONLY DRAM WRITE OF THE WHOLE NTT: the four output tiles, after the compute kernel's last
//      pass. Pages {2me, 2me+1, 2me+32, 2me+33} = two runs of two consecutive pages.
//
// THE SHUFFLE, per half kh (the two halves are independent and are both done here):
//   out[s][n1] = in[n1][s], within the 32 tiles of half kh.  This core owns n1 in {me, me+P} and
//   output rows s in the same set, so it sends row s of its tile n1 to the core owning row s --
//   peer s%P, its local slot (s/P)*2 + kh -- landing at row n1 of that tile.  Destination offsets
//   are loop-invariant in s because the destination row is always this core's own n1.
//
//   a 32x32 tile is 2x2 faces of 16x16; element (r,t) is at uint32 offset
//       ((r/16)*2 + (t/16))*256 + (r%16)*16 + (t%16)
//   so row r is TWO runs of 16 contiguous words at ((r>>4)*2 + {0,1})*256 + (r&15)*16.
//
// SYNCHRONISATION: scatter, FLUSH, then increment every peer's semaphore (an announcement must follow
// the data it announces), wait for P, then push cb_d. A core that never reaches its scatter hangs its
// whole block AND that block's compute kernels, so the host must dispatch whole blocks of P.
//
// Runtime args: 0=me, 1=out_addr, 2=sem_id, 3.. = 2*P physical NoC coords of this block, peer order.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t me       = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t sem_id   = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_x = tt::CBIndex::c_9, cb_d = tt::CBIndex::c_10, cb_out = tt::CBIndex::c_16;
    constexpr uint32_t NL = 4, P = 16, NT = 32, RUN = 16 * 4;
    const uint32_t tb = get_tile_size(cb_x);

    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr,
                                              .page_size = tb, .data_format = DataFormat::UInt32};

    // ---- 1. the corner turn ----
    cb_wait_front(cb_x, NL);                       // compute has finished the twiddle
    const uint32_t src0 = get_read_ptr(cb_x);
    cb_reserve_back(cb_d, NL);
    const uint32_t dst_l1 = get_write_ptr(cb_d);   // identical L1 offset on every core
    volatile tt_l1_ptr uint32_t* sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    for (uint32_t j = 0; j < 2; ++j) {             // my two n1 values: me, me+P
        const uint32_t n1 = me + j * P;
        // destination row is n1, loop-invariant
        const uint32_t d0 = (((n1 >> 4) * 2) * 256 + (n1 & 15) * 16) * 4;
        const uint32_t d1 = (((n1 >> 4) * 2 + 1) * 256 + (n1 & 15) * 16) * 4;
        for (uint32_t kh = 0; kh < 2; ++kh) {
            const uint32_t src = src0 + (j * 2 + kh) * tb;
            for (uint32_t s = 0; s < NT; ++s) {
                const uint32_t peer = s % P, pslot = (s / P) * 2 + kh;
                const uint32_t px = get_arg_val<uint32_t>(3 + 2 * peer);
                const uint32_t py = get_arg_val<uint32_t>(3 + 2 * peer + 1);
                const uint32_t s0 = (((s >> 4) * 2) * 256 + (s & 15) * 16) * 4;
                const uint32_t s1 = (((s >> 4) * 2 + 1) * 256 + (s & 15) * 16) * 4;
                const uint32_t base = dst_l1 + pslot * tb;
                noc_async_write(src + s0, ::get_noc_addr(px, py, base + d0), RUN);
                noc_async_write(src + s1, ::get_noc_addr(px, py, base + d1), RUN);
            }
        }
    }
    noc_async_write_barrier();                     // my rows have landed everywhere...

    for (uint32_t peer = 0; peer < P; ++peer) {    // ...so announcing them is truthful
        const uint32_t px = get_arg_val<uint32_t>(3 + 2 * peer);
        const uint32_t py = get_arg_val<uint32_t>(3 + 2 * peer + 1);
        noc_semaphore_inc(::get_noc_addr(px, py, reinterpret_cast<uint32_t>(sem)), 1);
    }
    noc_semaphore_wait(sem, P);

    cb_pop_front(cb_x, NL);
    cb_push_back(cb_d, NL);                        // compute un-parks here

    // ---- 2. the NTT's only DRAM write ----
    cb_wait_front(cb_out, NL);
    { const uint32_t p = get_read_ptr(cb_out);
      for (uint32_t L = 0; L < NL; ++L)
          noc_async_write_tile((me + (L >> 1) * P) * 2 + (L & 1), out, p + L * tb); }
    noc_async_write_barrier();
    cb_pop_front(cb_out, NL);
}
