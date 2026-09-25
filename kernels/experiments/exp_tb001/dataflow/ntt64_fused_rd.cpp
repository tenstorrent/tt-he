// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ntt64_fused_rd — NCRISC/NOC_1 feed for ntt64_fused.cpp.  Reads, ONCE per dispatch, everything the
// whole six-pass NTT needs: this core's four input tiles and every folded constant.
//
// The four tiles are pages {2me, 2me+1, 2me+32, 2me+33} in local-slot order L = j*2 + kh, i.e.
// page(L) = (me + (L>>1)*16)*2 + (L&1).  See ntt64_fused.cpp for why a core owns exactly these.
//
// Constants, all resident for the kernel's lifetime (~112 KB, against 1.5 MB of L1):
//   cb_w64  4 x 16 int8   the P1 folds, in the order the kernel indexes them: (bo*2+a)
//   cb_wa      16 int8    w32a, stage C
//   cb_wb      16 int8    w32b, stage A
//   cb_ta       2 int32   twiddles indexed by kh
//   cb_tb       4 int32   twiddles for THIS core's four tiles, pushed in slot order
//
// Every read of a group is in flight before its barrier (docs/he/dram_guide.md, the depth rule).
//
// Runtime args: 0=me, 1=in_addr, 2..5=w64[0..3], 6=w32a, 7=w32b, 8=ta_addr, 9=tb_addr

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t me      = get_arg_val<uint32_t>(0);
    const uint32_t in_addr = get_arg_val<uint32_t>(1);
    const uint32_t ta_addr = get_arg_val<uint32_t>(8);
    const uint32_t tb_addr = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_in = tt::CBIndex::c_0, cb_w64 = tt::CBIndex::c_1;
    constexpr uint32_t cb_wa = tt::CBIndex::c_2, cb_wb = tt::CBIndex::c_3;
    constexpr uint32_t cb_ta = tt::CBIndex::c_4, cb_tb = tt::CBIndex::c_5;
    constexpr uint32_t NB = 4, NW = NB * NB, NL = 4, P = 16;

    const uint32_t i32b = get_tile_size(cb_in);
    const uint32_t i8b  = get_tile_size(cb_w64);

    // The W folds are allocated by mkbufW, which puts them in L1 unless TG18NTT_W_DRAM is set -- so
    // they MUST be addressed through a TensorAccessor carrying the buffer's real config, not through
    // a hardcoded InterleavedAddrGenFast<true>. (That mistake reads a DRAM address for an L1 buffer
    // and returns garbage silently; it cost a debugging cycle here.) The input and twiddle buffers
    // come from mkbuf and are DRAM, so those keep the plain generator.
    constexpr auto w_args = TensorAccessorArgs<0>();

    const InterleavedAddrGenFast<true> in = {.bank_base_address = in_addr,
                                             .page_size = i32b, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> ta = {.bank_base_address = ta_addr,
                                             .page_size = i32b, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> tb = {.bank_base_address = tb_addr,
                                             .page_size = i32b, .data_format = DataFormat::UInt32};

    // ---- the four input tiles, in local-slot order ----
    cb_reserve_back(cb_in, NL);
    { const uint32_t p = get_write_ptr(cb_in);
      for (uint32_t L = 0; L < NL; ++L)
          noc_async_read_tile((me + (L >> 1) * P) * 2 + (L & 1), in, p + L * i32b); }

    // ---- the P1 folds: four buffers of NW, laid out so the kernel can index (bo*2+a)*NW + p*NB+r
    cb_reserve_back(cb_w64, 4 * NW);
    { const uint32_t p = get_write_ptr(cb_w64);
      for (uint32_t k = 0; k < 4; ++k) {
          const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(2 + k), i8b);
          for (uint32_t i = 0; i < NW; ++i) noc_async_read_tile(i, w, p + (k * NW + i) * i8b); } }

    // ---- the two 32-point folds ----
    cb_reserve_back(cb_wa, NW);
    { const uint32_t p = get_write_ptr(cb_wa);
      const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(6), i8b);
      for (uint32_t i = 0; i < NW; ++i) noc_async_read_tile(i, w, p + i * i8b); }
    cb_reserve_back(cb_wb, NW);
    { const uint32_t p = get_write_ptr(cb_wb);
      const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(7), i8b);
      for (uint32_t i = 0; i < NW; ++i) noc_async_read_tile(i, w, p + i * i8b); }

    // ---- twiddles ----
    cb_reserve_back(cb_ta, 2);
    { const uint32_t p = get_write_ptr(cb_ta);
      for (uint32_t i = 0; i < 2; ++i) noc_async_read_tile(i, ta, p + i * i32b); }
    cb_reserve_back(cb_tb, NL);
    { const uint32_t p = get_write_ptr(cb_tb);
      for (uint32_t L = 0; L < NL; ++L)
          noc_async_read_tile((me + (L >> 1) * P) * 2 + (L & 1), tb, p + L * i32b); }

    noc_async_read_barrier();

    cb_push_back(cb_in, NL);
    cb_push_back(cb_w64, 4 * NW);
    cb_push_back(cb_wa, NW);
    cb_push_back(cb_wb, NW);
    cb_push_back(cb_ta, 2);
    cb_push_back(cb_tb, NL);
}
