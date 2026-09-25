// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Multi-core slot→poly reshuffle — the REVERSE of reshuffle_gather_mc, for the scan/fold OUTPUT
// (slot-major matmul result → poly-major for the inverse NTT).  Each core owns a shard of the output
// poly-tiles; it holds only its outputs and STREAMS the input slot-tiles, reading a FIXED intra-tile
// position (row,col) from every slot and writing it to that slot's position in the output poly:
//   out_poly[p'] (slot j) = in_slot[j] (row,col)    where p' = row*COLS + col   (row=p'/COLS, col=p'%COLS)
// The output poly index p' (drives the read position) is kept SEPARATE from the DRAM write-tile index
// (= p'*out_stride + out_off) so n>1024 works: a poly spans multiple tiles, one scatter pass per half
// with in_base selecting the slot range and out_stride/out_off the poly's sub-tile.
//   Runtime args: 0=in,1=out,2=S_local,3=in_base,4=COLS,5=poly_start,6=n_out,7=out_stride,8=out_off.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t TELEM = 1024;
inline uint32_t faceidx(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }
}  // namespace

void kernel_main() {
    const uint32_t in_addr    = get_arg_val<uint32_t>(0);
    const uint32_t out_addr   = get_arg_val<uint32_t>(1);
    const uint32_t S_local    = get_arg_val<uint32_t>(2);   // input slot-tiles this pass (1024 per half)
    const uint32_t in_base    = get_arg_val<uint32_t>(3);   // input slot-tile offset (half*1024)
    const uint32_t COLS       = get_arg_val<uint32_t>(4);
    const uint32_t poly_start = get_arg_val<uint32_t>(5);   // first output poly p' for this core
    const uint32_t n_out      = get_arg_val<uint32_t>(6);
    const uint32_t out_stride = get_arg_val<uint32_t>(7);   // DRAM tile = p'*out_stride + out_off
    const uint32_t out_off    = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb = tt::CBIndex::c_0;
    const uint32_t tbytes = get_tile_size(cb);
    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,  .page_size = tbytes, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr, .page_size = tbytes, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb, n_out + 1);
    const uint32_t l1 = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* L = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < n_out * TELEM; ++i) L[i] = 0;
    volatile tt_l1_ptr uint32_t* inbuf = L + n_out * TELEM;

    for (uint32_t j = 0; j < S_local; ++j) {                              // stream slot-tiles of this half
        noc_async_read_tile(in_base + j, in, reinterpret_cast<uint32_t>(inbuf));
        noc_async_read_barrier();
        const uint32_t wpos = faceidx(j >> 5, j & 31);                    // local slot j's position in a poly tile
        for (uint32_t jl = 0; jl < n_out; ++jl) {
            const uint32_t pp = poly_start + jl;                          // output poly p'
            L[jl * TELEM + wpos] = inbuf[faceidx(pp / COLS, pp % COLS)];
        }
    }
    for (uint32_t jl = 0; jl < n_out; ++jl) {
        noc_async_write_tile((poly_start + jl) * out_stride + out_off, out, l1 + jl * tbytes);
        noc_async_write_barrier();
    }
    cb_push_back(cb, n_out + 1);
}
