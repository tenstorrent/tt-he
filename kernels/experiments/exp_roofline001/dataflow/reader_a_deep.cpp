// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// reader_a_deep — NCRISC/NOC_1 half of the SPLIT-READ feed (ROOF_SPLIT=1). Reads ONLY operand A.
//
// The default feed (reader_ab_deep.cpp) puts both operand reads on this one RISC, so NOC_1 carries
// 8 of the 12 bytes/op and saturates at ~238 GB/s while NOC_0 idles carrying just the 4-byte write
// (docs/vector/exp_roofline001.md §5b, docs/he/exp_dram001.md §2). Here A stays on NOC_1 and B moves
// to BRISC/NOC_0 alongside the write, so each NoC carries 4 B/op INBOUND.
//
// Runtime args: 0 = a_addr, 1 = n_tiles, 2 = start_tile.  Compile args: 0 = CHUNK.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb0 = tt::CBIndex::c_0, CHUNK = get_compile_time_arg_val(0);
    const uint32_t ts = get_tile_size(cb0);
    const InterleavedAddrGenFast<true> a = {.bank_base_address = a_addr,
                                            .page_size = ts, .data_format = DataFormat::UInt32};
    for (uint32_t i = 0; i < n; i += CHUNK) {
        const uint32_t k = (n - i < CHUNK) ? (n - i) : CHUNK;
        cb_reserve_back(cb0, k);
        const uint32_t p = get_write_ptr(cb0);
        for (uint32_t j = 0; j < k; ++j) noc_async_read_tile(start + i + j, a, p + j * ts);
        noc_async_read_barrier();
        cb_push_back(cb0, k);
    }
}
