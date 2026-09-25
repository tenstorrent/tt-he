// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ntt32768_2f_fused_xw — BRISC half of the fully SRAM-resident 2-factor NTT-32768: TWO group
// exchanges and the final write. The 2-factor needs two because BOTH its matmuls are non-tile-local:
//
//   M1  output (mt,nt) needs A[mt][kt]  kt=0..7  -> the 8-core mt GROUP (cores mt*8 + j)
//   M2  output (mt,nt) needs S'[kt][nt] kt=0..3  -> the 4-core nt GROUP (cores j*8 + nt)
//
// where the 3-factor needed exactly ONE (its triples are tile-local; only the corner turn crosses
// cores). That asymmetry is the factorization's, not the implementation's -- see
// docs/matmul/exp_ntt001.md. Each core BROADCASTS its own 4 limb tiles to its group and receives the
// group's, so after each exchange every core holds every dynamic operand its matmul needs, in L1.
//
// Landing slots are the matmul's own indexing, so the compute kernel gathers nothing:
//   exchange 1: my limb p -> peer's c_5 slot (kt_me*NB + p),  kt_me = me % 8
//   exchange 2: my limb r -> peer's c_7 slot (kt_me*NB + r),  kt_me = me / 8
//
// Same protocol as all_to_all_corner_x.cpp: scatter, flush, then increment every group member's
// semaphore, then wait for the group size. Requires the full grid.
//
// Runtime args: 0=out_addr, 1=me, 2=sem1, 3=sem2, 4.. = 64 physical coords (x,y) by core index.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    DeviceZoneScopedN("x2f_total");
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t me       = get_arg_val<uint32_t>(1);
    const uint32_t sem1_id  = get_arg_val<uint32_t>(2);
    const uint32_t sem2_id  = get_arg_val<uint32_t>(3);
    constexpr uint32_t NB = 4;
    const uint32_t mt = me >> 3, nt = me & 7;

    constexpr auto cb_my1 = tt::CBIndex::c_4;   // my A limbs        (4 int8)
    constexpr auto cb_g1  = tt::CBIndex::c_5;   // gathered, 8*NB    (32 int8)
    constexpr auto cb_my2 = tt::CBIndex::c_6;   // my S' limbs       (4 int8)
    constexpr auto cb_g2  = tt::CBIndex::c_7;   // gathered, 4*NB    (16 int8)
    constexpr auto cb_r   = tt::CBIndex::c_16;  // result
    const uint32_t i8 = get_tile_size(cb_my1);
    volatile tt_l1_ptr uint32_t* s1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem1_id));
    volatile tt_l1_ptr uint32_t* s2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem2_id));
    auto px = [&](uint32_t c) { return get_arg_val<uint32_t>(4 + 2*c); };
    auto py = [&](uint32_t c) { return get_arg_val<uint32_t>(4 + 2*c + 1); };

    // ---- exchange 1: broadcast my A limbs to the 8-core mt group ----
    { DeviceZoneScopedN("x2f_exch1");
    cb_wait_front(cb_my1, NB);
    const uint32_t src = get_read_ptr(cb_my1);
    cb_reserve_back(cb_g1, 8*NB);
    const uint32_t dst = get_write_ptr(cb_g1);
    const uint32_t slot = (me & 7) * NB;                       // kt_me
    for (uint32_t j = 0; j < 8; ++j) {
        const uint32_t peer = mt*8 + j;
        for (uint32_t p = 0; p < NB; ++p)
            noc_async_write(src + p*i8, ::get_noc_addr(px(peer), py(peer), dst + (slot + p)*i8), i8);
    }
    noc_async_write_barrier();
    for (uint32_t j = 0; j < 8; ++j) { const uint32_t peer = mt*8 + j;
        noc_semaphore_inc(::get_noc_addr(px(peer), py(peer), reinterpret_cast<uint32_t>(s1)), 1); }
    noc_semaphore_wait(s1, 8);
    cb_pop_front(cb_my1, NB);
    cb_push_back(cb_g1, 8*NB); }

    // ---- exchange 2: broadcast my S' limbs to the 4-core nt group ----
    { DeviceZoneScopedN("x2f_exch2");
    cb_wait_front(cb_my2, NB);
    const uint32_t src2 = get_read_ptr(cb_my2);
    cb_reserve_back(cb_g2, 4*NB);
    const uint32_t dst2 = get_write_ptr(cb_g2);
    const uint32_t slot2 = (me >> 3) * NB;                     // kt_me
    for (uint32_t j = 0; j < 4; ++j) {
        const uint32_t peer = j*8 + nt;
        for (uint32_t r = 0; r < NB; ++r)
            noc_async_write(src2 + r*i8, ::get_noc_addr(px(peer), py(peer), dst2 + (slot2 + r)*i8), i8);
    }
    noc_async_write_barrier();
    for (uint32_t j = 0; j < 4; ++j) { const uint32_t peer = j*8 + nt;
        noc_semaphore_inc(::get_noc_addr(px(peer), py(peer), reinterpret_cast<uint32_t>(s2)), 1); }
    noc_semaphore_wait(s2, 4);
    cb_pop_front(cb_my2, NB);
    cb_push_back(cb_g2, 4*NB); }

    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr,
                                              .page_size = get_tile_size(cb_r), .data_format = DataFormat::UInt32};
    cb_wait_front(cb_r, 1);
    noc_async_write_tile(me, out, get_read_ptr(cb_r));
    noc_async_write_barrier();
    cb_pop_front(cb_r, 1);
}
