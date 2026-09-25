// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ntt32768_3f_fused_rd — NCRISC feed for the single-dispatch NTT-32768 (compute/ntt32768_3f_fused.cpp).
// Everything the whole NTT needs, read ONCE, all in flight behind one barrier (dram_guide 3.3):
//   Xin[me] -> c_0 | WfoldA 16 int8 -> c_1 | Wrhs 16 int8 -> c_2 | Ta[me] -> c_3 | Tb[me] -> c_4
// The two W sets are resident for the whole kernel -- three triples read them without re-fetching,
// which is the traffic the six-dispatch version paid three times over.
//
// Runtime args: 0=x,1=wa,2=wc,3=ta,4=tb,5=me
// Compile args: TensorAccessorArgs for x, wa, wc, ta, tb in that order.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t xa = get_arg_val<uint32_t>(0), wa = get_arg_val<uint32_t>(1);
    const uint32_t wc = get_arg_val<uint32_t>(2), ta = get_arg_val<uint32_t>(3);
    const uint32_t tb = get_arg_val<uint32_t>(4), me = get_arg_val<uint32_t>(5);
    constexpr uint32_t NW = 16;

    constexpr auto a0 = TensorAccessorArgs<0>();                       const auto X  = TensorAccessor(a0, xa);
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>(); const auto WA = TensorAccessor(a1, wa);
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>(); const auto WC = TensorAccessor(a2, wc);
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>(); const auto TA = TensorAccessor(a3, ta);
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>(); const auto TB = TensorAccessor(a4, tb);

    // STAGED in need-order, matching ntt32768_2f_fused_rd.cpp so neither variant is favoured by its
    // reader. Far less to hide here (40 KB against the 2-factor's 196), but the treatment is identical.
    //   1. Xin + WfoldA -> the A triple can start        (20 KB)
    //   2. Ta + Wrhs + Tb -> B twiddle, C triple, TB     (24 KB); WfoldA stays resident for D
    const uint32_t i8 = get_tile_size(tt::CBIndex::c_1);

    cb_reserve_back(tt::CBIndex::c_0, 1);
    cb_reserve_back(tt::CBIndex::c_1, NW);
    const uint32_t px = get_write_ptr(tt::CBIndex::c_0);
    const uint32_t p1 = get_write_ptr(tt::CBIndex::c_1);
    noc_async_read_tile(me, X, px);
    for (uint32_t i = 0; i < NW; ++i) noc_async_read_tile(i, WA, p1 + i * i8);
    noc_async_read_barrier();
    cb_push_back(tt::CBIndex::c_0, 1); cb_push_back(tt::CBIndex::c_1, NW);   // A triple may now run

    cb_reserve_back(tt::CBIndex::c_2, NW);
    cb_reserve_back(tt::CBIndex::c_3, 1);
    cb_reserve_back(tt::CBIndex::c_4, 1);
    const uint32_t p2 = get_write_ptr(tt::CBIndex::c_2);
    const uint32_t p3 = get_write_ptr(tt::CBIndex::c_3);
    const uint32_t p4 = get_write_ptr(tt::CBIndex::c_4);
    for (uint32_t i = 0; i < NW; ++i) noc_async_read_tile(i, WC, p2 + i * i8);
    noc_async_read_tile(me, TA, p3);
    noc_async_read_tile(me, TB, p4);
    noc_async_read_barrier();
    cb_push_back(tt::CBIndex::c_2, NW); cb_push_back(tt::CBIndex::c_3, 1);
    cb_push_back(tt::CBIndex::c_4, 1);
}
