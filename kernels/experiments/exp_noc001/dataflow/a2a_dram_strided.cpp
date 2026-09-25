// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// a2a_dram_strided — the same all-to-all through DRAM, but reading ONLY the bytes it needs.  Route B.
//
// Route A (a2a_dram_bulk.cpp) reads whole 4 KB pages to use BLK = P*4096/G bytes of each.  That is not
// a property of DRAM, it is a property of noc_async_read_tile: the address generator's unit is a page.
// addrgen.get_noc_addr(page, byte_offset) gives an arbitrary offset inside a page, so a core can fetch
// exactly its own block from each peer:
//
//     chip-wide traffic = G * (P tiles) = O(G)     -- identical to the NoC route (C)
//
// So B and C move THE SAME NUMBER OF BYTES and differ only in the medium: B goes DRAM->L1 through the
// memory controller, C goes L1->L1 across the fabric.  That is the fair "same bytes, different medium"
// comparison; A vs B is "same medium, different access pattern".  Splitting the two effects is the
// whole point of exp_noc001 -- quoting a single DRAM-vs-NoC ratio conflates them.
//
// The cost B pays instead is REQUEST SHAPE.  docs/he/dram_guide.md 3.4: DRAM wants few large
// transfers, and G small reads of BLK bytes (64 B at G=64,P=1) is the shape it likes least.  Whether
// trading G^2 bytes for G small requests wins is exactly what gets measured.
//
// BLK divides the tile size for every (G,P) in the sweep (G/P integral), so each block lies wholly
// inside one page -- no read straddles a page boundary.
//
//
// DRAM CHANNEL CONTROL.  An interleaved DRAM buffer round-robins page i onto bank i%8, so a kernel
// that walks pages consecutively always uses all 8 channels and the channel count is not a free
// variable.  To make it one, logical tile i is placed at PHYSICAL page
//
//     PAGE(i) = (i/CH)*8 + (i%CH)          CH = channels to use, 1|2|4|8
//
// every one of which has page%8 < CH, so exactly CH banks are ever touched (CH=8 reduces to the
// identity). The buffer is padded to suit; the wasted pages are never read.  The host places and
// recovers the data through the same map, so the permutation and its golden are unchanged.
//
// Runtime args: 0=in_addr, 1=out_addr, 2=G, 3=P, 4=me, 5=CH, 5=CH
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr= get_arg_val<uint32_t>(1);
    const uint32_t G       = get_arg_val<uint32_t>(2);
    const uint32_t P       = get_arg_val<uint32_t>(3);
    const uint32_t me      = get_arg_val<uint32_t>(4);
    const uint32_t CH      = get_arg_val<uint32_t>(5);

    const auto PAGE = [CH](uint32_t i) { return (i / CH) * 8 + (i % CH); };
    constexpr uint32_t cb_out = tt::CBIndex::c_1;
    const uint32_t ts  = get_tile_size(cb_out);
    const uint32_t BLK = P * ts / G;
    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,  .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb_out, P);
    const uint32_t dst = get_write_ptr(cb_out);

    // my block sits at the same byte offset inside every peer's P-tile run: page pg, offset off
    const uint32_t pg  = (me * BLK) / ts;
    const uint32_t off = (me * BLK) % ts;

    {   // exchange proper: G small strided reads, all in flight, one barrier
        DeviceZoneScopedN("a2a");
        for (uint32_t n = 0; n < G; ++n)
            noc_async_read(in.get_noc_addr(PAGE(n * P + pg), off), dst + n * BLK, BLK);
        noc_async_read_barrier();
    }
    {   // writeback, counted separately so route B and route C are compared on equal accounting
        DeviceZoneScopedN("a2a_wb");
        for (uint32_t p = 0; p < P; ++p) noc_async_write_tile(PAGE(me * P + p), out, dst + p * ts);
        noc_async_write_barrier();
    }
    cb_push_back(cb_out, P);
}
