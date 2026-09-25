// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Multi-core poly→slot reshuffle (the scan/fold all-to-all), scaling past one core's L1.
// Each core owns a shard of the S output slot-tiles [out_start, out_start+n_out).  It holds only its
// n_out output tiles in L1 and STREAMS the P input poly-tiles from DRAM one at a time, scattering each
// input's slot-values into its output tiles — so large P never has to sit in L1.
//   out[j] (row,col) = in[p = row*SR + col*SC] (slot j)      for row<ROWS, col<COLS   (P = ROWS·COLS)
// The (SR,SC,ROWS,COLS) parameterization covers both scan operands with the SAME kernel:
//   DB  A_j[r][k] = DB_ntt[k*rest+r][j]   → ROWS=rest, COLS=N0, SR=1,   SC=rest
//   Q   B_j[k][c] = Q_ntt[k*2+c][j]       → ROWS=N0,   COLS=2,  SR=2,   SC=1
// Reads each input once per core (C·P total); the NoC-distributed a2a that reads inputs once and pushes
// them to peers (bench_noc_a2a, ~3.8× @ G=64) is the further bandwidth optimization on top of this.
// slot_start/n_out select this core's slot shard; out_tile_base is the DRAM tile offset of this block
// (e.g. k-tile kt writes to sX[kt*sOT + slot], so out_tile_base = kt*sOT) — kept SEPARATE from the slot
// index, which drives faceidx (else a nonzero base corrupts the intra-tile position).
// For n>1024 a poly spans multiple tiles: run once per half, with in_stride/in_off selecting the poly's
// sub-tile (n=2048: in_stride=2, in_off=half∈{0,1}); slot_start is the LOCAL slot within that half.
//   Runtime args: 0=in,1=out,2=ROWS,3=COLS,4=SR,5=SC,6=slot_start,7=n_out,8=out_tile_base,
//                 9=in_stride,10=in_off.  int32 tiles.  (in_stride=1,in_off=0 for n=1024.)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t TELEM = 1024;
inline uint32_t faceidx(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }
}  // namespace

void kernel_main() {
    const uint32_t in_addr   = get_arg_val<uint32_t>(0);
    const uint32_t out_addr  = get_arg_val<uint32_t>(1);
    const uint32_t ROWS      = get_arg_val<uint32_t>(2);
    const uint32_t COLS      = get_arg_val<uint32_t>(3);
    const uint32_t SR        = get_arg_val<uint32_t>(4);
    const uint32_t SC        = get_arg_val<uint32_t>(5);
    const uint32_t slot_start = get_arg_val<uint32_t>(6);
    const uint32_t n_out       = get_arg_val<uint32_t>(7);
    const uint32_t out_tile_base = get_arg_val<uint32_t>(8);
    const uint32_t in_stride = get_arg_val<uint32_t>(9);
    const uint32_t in_off    = get_arg_val<uint32_t>(10);

    constexpr uint32_t cb = tt::CBIndex::c_0;
    const uint32_t tbytes = get_tile_size(cb);
    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,  .page_size = tbytes, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr, .page_size = tbytes, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb, n_out + 1);
    const uint32_t l1 = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* L = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < n_out * TELEM; ++i) L[i] = 0;               // cols k≥N0 / rows r≥rest stay 0
    volatile tt_l1_ptr uint32_t* inbuf = L + n_out * TELEM;

    for (uint32_t row = 0; row < ROWS; ++row) for (uint32_t col = 0; col < COLS; ++col) {  // stream inputs
        const uint32_t p = (row * SR + col * SC) * in_stride + in_off;    // input tile (poly's sub-tile)
        noc_async_read_tile(p, in, reinterpret_cast<uint32_t>(inbuf));
        noc_async_read_barrier();
        const uint32_t dst = faceidx(row, col);
        for (uint32_t jl = 0; jl < n_out; ++jl) {
            const uint32_t j = slot_start + jl;                          // slot index (drives faceidx)
            L[jl * TELEM + dst] = inbuf[faceidx(j >> 5, j & 31)];        // gather slot j of this poly
        }
    }
    for (uint32_t jl = 0; jl < n_out; ++jl) {
        noc_async_write_tile(out_tile_base + slot_start + jl, out, l1 + jl * tbytes);   // DRAM tile = base + slot
        noc_async_write_barrier();
    }
    cb_push_back(cb, n_out + 1);
}
