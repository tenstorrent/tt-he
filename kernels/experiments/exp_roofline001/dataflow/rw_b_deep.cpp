// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// rw_b_deep — BRISC/NOC_0 half of the SPLIT-READ feed (ROOF_SPLIT=1): reads operand B **and** writes
// the result, both on NOC_0.
//
// A NoC carries reads and writes at once -- docs/he/dram_guide.md §3.1's 433 GB/s configuration
// already has the writer's RISC doing both -- and a read's payload travels DRAM->core while a write's
// travels core->DRAM, so on a full-duplex link they need not contend. That is the hypothesis this
// kernel exists to test: does moving B here (making each NoC 4 B/op inbound instead of 8 and 0) lift
// the memory roof from 334.6 toward the ~445 GB/s exp_dram001 measured for reads on both NoCs?
//
// THE INTERLEAVE, and why it cannot deadlock. Per chunk: read B, push it, then drain the PREVIOUS
// output chunk. The one-chunk lag is what makes it safe -- compute already holds the A and B it needs
// for the chunk being drained, so it can never be waiting on this RISC for input while this RISC waits
// on it for output. c_1 and c_16 are 2*CHUNK deep so the producer can run a chunk ahead of the drain.
//
// Runtime args: 0 = b_addr, 1 = n_tiles, 2 = start_tile, 3 = out_addr.  Compile args: 0 = CHUNK.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t b_addr   = get_arg_val<uint32_t>(0);
    const uint32_t n        = get_arg_val<uint32_t>(1);
    const uint32_t start    = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb1 = tt::CBIndex::c_1, cb_out = tt::CBIndex::c_16;
    constexpr uint32_t CHUNK = get_compile_time_arg_val(0);
    const uint32_t ts = get_tile_size(cb1);

    const InterleavedAddrGenFast<true> b = {.bank_base_address = b_addr,
                                            .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> o = {.bank_base_address = out_addr,
                                            .page_size = ts, .data_format = DataFormat::UInt32};

    uint32_t done = 0;                                  // output tiles already written
    auto drain = [&]() {
        const uint32_t w = (n - done < CHUNK) ? (n - done) : CHUNK;
        cb_wait_front(cb_out, w);
        const uint32_t q = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < w; ++j) noc_async_write_tile(start + done + j, o, q + j * ts);
        noc_async_write_barrier();
        cb_pop_front(cb_out, w);
        done += w;
    };

    for (uint32_t i = 0; i < n; i += CHUNK) {
        const uint32_t k = (n - i < CHUNK) ? (n - i) : CHUNK;
        cb_reserve_back(cb1, k);
        const uint32_t p = get_write_ptr(cb1);
        for (uint32_t j = 0; j < k; ++j) noc_async_read_tile(start + i + j, b, p + j * ts);
        noc_async_read_barrier();
        cb_push_back(cb1, k);
        if (i > 0) drain();                             // one chunk behind -- see the header
    }
    while (done < n) drain();
}
