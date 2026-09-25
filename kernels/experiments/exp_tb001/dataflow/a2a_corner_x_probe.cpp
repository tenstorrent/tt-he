// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// a2a_corner_x_probe — a2a_corner_x.cpp with the DRAM READ and the DRAM WRITE deleted, to price the
// question "what would a corner turn with NO DRAM access at all be worth?"
//
// It COMPUTES A DELIBERATELY WRONG ANSWER: it scatters whatever happens to be in L1 and writes one
// sentinel word instead of its assembled tiles. **The correctness gate must FAIL for the measurement
// to be valid.** exp_ntt001 §2b-quinquies threw away a first probe that corrupted its output and
// still passed 0/32768 -- a probe that cannot be right and reports right is a broken instrument, and
// the gate is what catches it.
//
// (real - probe) is therefore the cost of the DRAM half of the exchange: NPC page reads in and NPC
// page writes out per core. What remains in the probe is the NoC scatter, the semaphore barrier and
// the launch -- i.e. the part that a fully DRAM-free corner turn would still have to pay.
//
// Runtime args: identical to a2a_corner_x.cpp, so the host builds one argument vector for both.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    DeviceZoneScopedN("a2a_corner");

    const uint32_t NT       = get_arg_val<uint32_t>(0);
    const uint32_t P        = get_arg_val<uint32_t>(1);
    const uint32_t me       = get_arg_val<uint32_t>(2);
    const uint32_t kh       = get_arg_val<uint32_t>(3);
    const uint32_t out_addr = get_arg_val<uint32_t>(5);
    const uint32_t sem_id   = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_o  = tt::CBIndex::c_2;
    const uint32_t tbytes = get_tile_size(cb_in);
    const uint32_t NPC = NT / P;

    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr,
                                              .page_size = tbytes, .data_format = DataFormat::UInt32};

    // ---- 1. THE READ IS GONE. Scatter whatever is already in L1. ----
    cb_reserve_back(cb_in, NPC);
    const uint32_t src0 = get_write_ptr(cb_in);

    cb_reserve_back(cb_o, NPC);
    const uint32_t dst_l1 = get_write_ptr(cb_o);
    volatile tt_l1_ptr uint32_t* sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    constexpr uint32_t RUN = 16 * 4;

    // ---- 2. scatter: unchanged, so the NoC traffic and the barrier are priced exactly ----
    for (uint32_t li = 0; li < NPC; ++li) {
        const uint32_t n1 = me + li * P;
        const uint32_t src = src0 + li * tbytes;
        const uint32_t d0 = (((n1 >> 4) * 2) * 256 + (n1 & 15) * 16) * 4;
        const uint32_t d1 = (((n1 >> 4) * 2 + 1) * 256 + (n1 & 15) * 16) * 4;
        for (uint32_t s = 0; s < NT; ++s) {
            const uint32_t peer = s % P, slot = s / P;
            const uint32_t px = get_arg_val<uint32_t>(7 + 2 * peer);
            const uint32_t py = get_arg_val<uint32_t>(7 + 2 * peer + 1);
            const uint32_t s0 = (((s >> 4) * 2) * 256 + (s & 15) * 16) * 4;
            const uint32_t s1 = (((s >> 4) * 2 + 1) * 256 + (s & 15) * 16) * 4;
            const uint32_t base = dst_l1 + slot * tbytes;
            noc_async_write(src + s0, ::get_noc_addr(px, py, base + d0), RUN);
            noc_async_write(src + s1, ::get_noc_addr(px, py, base + d1), RUN);
        }
    }
    noc_async_write_barrier();

    // ---- 3. barrier: unchanged ----
    for (uint32_t peer = 0; peer < P; ++peer) {
        const uint32_t px = get_arg_val<uint32_t>(7 + 2 * peer);
        const uint32_t py = get_arg_val<uint32_t>(7 + 2 * peer + 1);
        noc_semaphore_inc(::get_noc_addr(px, py, reinterpret_cast<uint32_t>(sem)), 1);
    }
    noc_semaphore_wait(sem, P);

    // ---- 4. THE WRITE IS GONE. One sentinel word, so the gate fails as it must. ----
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1);
    p[0] = 0xDEADBEEFu;
    noc_async_write(dst_l1, ::get_noc_addr(0, 0, out_addr), 4);
    noc_async_write_barrier();

    cb_push_back(cb_in, NPC);
    cb_push_back(cb_o, NPC);
}
