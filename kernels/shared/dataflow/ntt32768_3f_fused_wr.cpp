// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ntt32768_3f_fused_wr — BRISC half of the single-dispatch NTT-32768. It performs the corner turn
// IN THE MIDDLE of the compute kernel's execution and then writes the result.
//
//   1. wait for compute to hand over Cb on c_16;
//   2. all-to-all exchange (the all_to_all_corner_x.cpp protocol, verbatim in behaviour): scatter row
//      s of my tile into peer s's c_6, in final face layout; flush; increment every peer's semaphore;
//      wait for NT; push c_6 -- which UNBLOCKS the compute kernel's D triple;
//   3. wait for E on c_17 and write it to DRAM.
//
// So a compute kernel is parked on cb_d while this RISC runs a 32-core barrier. Every core must reach
// step 1 or all of them hang -- see all_to_all_corner_x.cpp for the protocol's ordering argument.
//
// Runtime args: 0=out_addr, 1=NT, 2=me, 3=sem_id, 4.. = 2*NT physical NoC coords (x,y) per core.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t NT       = get_arg_val<uint32_t>(1);
    const uint32_t me       = get_arg_val<uint32_t>(2);
    const uint32_t sem_id   = get_arg_val<uint32_t>(3);

    constexpr auto cb_d  = tt::CBIndex::c_6;
    constexpr auto cb_cb = tt::CBIndex::c_16;
    constexpr auto cb_e  = tt::CBIndex::c_17;
    const uint32_t tbytes = get_tile_size(cb_cb);
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr,
                                              .page_size = tbytes, .data_format = DataFormat::UInt32};

    cb_wait_front(cb_cb, 1);                                  // compute finished Block 1
    const uint32_t src = get_read_ptr(cb_cb);
    cb_reserve_back(cb_d, 1);
    const uint32_t dst_l1 = get_write_ptr(cb_d);              // identical L1 offset on every core
    volatile tt_l1_ptr uint32_t* sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    const uint32_t o0 = (((me >> 4) * 2) * 256 + (me & 15) * 16) * 4;   // destination row is always me
    const uint32_t o1 = (((me >> 4) * 2 + 1) * 256 + (me & 15) * 16) * 4;
    constexpr uint32_t RUN = 16 * 4;

    for (uint32_t s = 0; s < NT; ++s) {
        const uint32_t px = get_arg_val<uint32_t>(4 + 2 * s), py = get_arg_val<uint32_t>(4 + 2 * s + 1);
        const uint32_t i0 = (((s >> 4) * 2) * 256 + (s & 15) * 16) * 4;
        const uint32_t i1 = (((s >> 4) * 2 + 1) * 256 + (s & 15) * 16) * 4;
        noc_async_write(src + i0, ::get_noc_addr(px, py, dst_l1 + o0), RUN);
        noc_async_write(src + i1, ::get_noc_addr(px, py, dst_l1 + o1), RUN);
    }
    noc_async_write_barrier();
    for (uint32_t s = 0; s < NT; ++s) {
        const uint32_t px = get_arg_val<uint32_t>(4 + 2 * s), py = get_arg_val<uint32_t>(4 + 2 * s + 1);
        noc_semaphore_inc(::get_noc_addr(px, py, reinterpret_cast<uint32_t>(sem)), 1);
    }
    noc_semaphore_wait(sem, NT);
    cb_pop_front(cb_cb, 1);
    cb_push_back(cb_d, 1);                                    // unblocks the compute kernel's D triple

    cb_wait_front(cb_e, 1);
    noc_async_write_tile(me, out, get_read_ptr(cb_e));
    noc_async_write_barrier();
    cb_pop_front(cb_e, 1);
}
