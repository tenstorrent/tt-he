// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// a2a_dram_bulk — personalised all-to-all THROUGH DRAM, reading whole pages.  exp_noc001 route A.
//
// This is the shape all_to_all_corner_mc.cpp uses: to get the one block it needs from each peer, a
// core reads that peer's WHOLE tile and throws the rest away.  DRAM's transfer granularity is a page,
// so "read a 1/G slice" costs the same as "read the page" -- and every one of the G cores does it for
// every one of the G peers.
//
//     chip-wide traffic = G * (G*P tiles)  =  O(G^2)     to move G*P tiles of actual data
//
// That quadratic term is the thing exp_noc001 exists to measure.  Route B (a2a_dram_strided.cpp)
// removes it while staying on DRAM; route C (a2a_noc.cpp) removes the DRAM round trip as well.
//
// Layout (byte-granular, no tile-face addressing -- this is a communication benchmark, so the
// permutation is defined on blocks and the golden is a plain byte permutation):
//   in  : page = n*P + p        core n owns P tiles
//   block(n,s) = the s-th 1/G slice of core n's P tiles, BLK = P*4096/G bytes
//   out : core s holds block(n,s) at offset n*BLK, for all n
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
    constexpr uint32_t cb_in = tt::CBIndex::c_0, cb_out = tt::CBIndex::c_1;
    const uint32_t ts = get_tile_size(cb_in);
    const uint32_t BLK = P * ts / G;                       // bytes each peer owes each peer
    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,  .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb_in, G * P);
    const uint32_t stage = get_write_ptr(cb_in);
    cb_reserve_back(cb_out, P);
    const uint32_t dst = get_write_ptr(cb_out);

    {   // exchange proper.  Read EVERY peer's tiles IN FULL -- all in flight, one barrier
        // (dram_guide 3.3), then keep only my 1/G slice of each and discard the rest.
        // That discarded fraction IS the O(G^2): (G-1)/G of every byte read is thrown away.
        DeviceZoneScopedN("a2a");
        for (uint32_t t = 0; t < G * P; ++t) noc_async_read_tile(PAGE(t), in, stage + t * ts);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
        for (uint32_t n = 0; n < G; ++n) {
            volatile tt_l1_ptr uint32_t* s =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + n * P * ts + me * BLK);
            for (uint32_t w = 0; w < BLK / 4; ++w) d[n * (BLK / 4) + w] = s[w];
        }
    }
    {   DeviceZoneScopedN("a2a_wb");
        for (uint32_t p = 0; p < P; ++p) noc_async_write_tile(PAGE(me * P + p), out, dst + p * ts);
        noc_async_write_barrier();
    }
    cb_push_back(cb_in, G * P); cb_push_back(cb_out, P);
}
