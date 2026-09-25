// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ntt32768_2f_fused_rd — NCRISC feed for the fully SRAM-resident 2-factor NTT-32768.
// Reads, ONCE, everything the whole transform needs, all in flight behind one barrier:
//   A[me]          (1 int32)                        -> c_0
//   W256 slice     (NB*NB*Kt1 = 128 int8, nt fixed) -> c_1     Wrhs256[(p*NB+r)*Kt1+kt]*Nt1 + nt
//   W128 slice     (NB*NB*Kt2 =  64 int8, mt fixed) -> c_2     Wfold128[((p*NB+r)*Mt2+mt)*Kt2+kt]
//   Tb[me]         (1 int32)                        -> c_3
// After this there is NO DRAM read for the rest of the NTT: the dynamic operands both matmuls need
// arrive from peer cores over the NoC (ntt32768_2f_fused_xw.cpp), not from memory.
//
// The two slices are why the 2-factor needs 196 KB of resident constants against the 3-factor's 40:
// a folded f x f DFT costs f^2 * NB^2 bytes, so f=256 is 1 MB whole and 128 KB per nt-slice.
//
// Runtime args: 0=a,1=w256,2=w128,3=tb,4=me
// Compile args: TensorAccessorArgs for a, w256, w128, tb in that order.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t aa = get_arg_val<uint32_t>(0), w2 = get_arg_val<uint32_t>(1);
    const uint32_t w1 = get_arg_val<uint32_t>(2), tb = get_arg_val<uint32_t>(3);
    const uint32_t me = get_arg_val<uint32_t>(4);
    constexpr uint32_t NB = 4, Kt1 = 8, Nt1 = 8, Mt2 = 4, Kt2 = 4;
    const uint32_t mt = me >> 3, nt = me & 7;

    constexpr auto s0 = TensorAccessorArgs<0>();                                   const auto A  = TensorAccessor(s0, aa);
    constexpr auto s1 = TensorAccessorArgs<s0.next_compile_time_args_offset()>();  const auto W2 = TensorAccessor(s1, w2);
    constexpr auto s2 = TensorAccessorArgs<s1.next_compile_time_args_offset()>();  const auto W1 = TensorAccessor(s2, w1);
    constexpr auto s3 = TensorAccessorArgs<s2.next_compile_time_args_offset()>();  const auto TB = TensorAccessor(s3, tb);

    // STAGED, not one barrier for everything. A single barrier over all 196 KB would keep compute
    // idle until the last constant landed -- measured 36.63 us of NCRISC-KERNEL fully in front of the
    // transform (docs/matmul/exp_ntt001.md). Each stage is pushed as soon as it lands, in the order
    // the compute kernel asks for it, so the big slices stream underneath work that has already begun:
    //   1. A[me]      -> decompose + exchange 1 can start immediately   (4 KB)
    //   2. W256 + Tb  -> M1 can start                                   (132 KB)
    //   3. W128       -> only M2 needs it, so it streams under M1        (64 KB)
    // The trade is issue depth: a stage's reads are only in flight with each other, not with the next
    // stage's. Each stage here is 64-128 KB, far past the ~192-page knee of dram_guide.md 3.3, so the
    // depth rule is still satisfied within a stage.
    const uint32_t i8 = get_tile_size(tt::CBIndex::c_1);

    cb_reserve_back(tt::CBIndex::c_0, 1);
    const uint32_t p0 = get_write_ptr(tt::CBIndex::c_0);
    noc_async_read_tile(me, A, p0);
    noc_async_read_barrier();
    cb_push_back(tt::CBIndex::c_0, 1);                       // decompose may now run

    cb_reserve_back(tt::CBIndex::c_1, NB*NB*Kt1);
    cb_reserve_back(tt::CBIndex::c_3, 1);
    const uint32_t p1 = get_write_ptr(tt::CBIndex::c_1);
    const uint32_t p3 = get_write_ptr(tt::CBIndex::c_3);
    for (uint32_t pr = 0; pr < NB*NB; ++pr)                       // W256: nt fixed, stride Nt1
        for (uint32_t kt = 0; kt < Kt1; ++kt)
            noc_async_read_tile((pr*Kt1 + kt)*Nt1 + nt, W2, p1 + (pr*Kt1 + kt)*i8);
    noc_async_read_tile(me, TB, p3);
    noc_async_read_barrier();
    cb_push_back(tt::CBIndex::c_1, NB*NB*Kt1);
    cb_push_back(tt::CBIndex::c_3, 1);                       // M1 may now run

    cb_reserve_back(tt::CBIndex::c_2, NB*NB*Kt2);
    const uint32_t p2 = get_write_ptr(tt::CBIndex::c_2);
    for (uint32_t pr = 0; pr < NB*NB; ++pr)                       // W128: mt fixed -- only M2 needs it
        for (uint32_t kt = 0; kt < Kt2; ++kt)
            noc_async_read_tile((pr*Mt2 + mt)*Kt2 + kt, W1, p2 + (pr*Kt2 + kt)*i8);
    noc_async_read_barrier();
    cb_push_back(tt::CBIndex::c_2, NB*NB*Kt2);
}
