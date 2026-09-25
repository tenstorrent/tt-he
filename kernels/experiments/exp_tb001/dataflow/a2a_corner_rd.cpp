// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Corner-turn READ HALF, for exp_Tg018m: NCRISC/NOC_1 fetches the first NT/2 input tiles.
//
// WHY SPLIT IT. a2a_corner_pg.cpp does everything on one RISC, so all 32 of its input reads ride
// NOC_1 and NOC_0 sits idle. The card has two NoCs, each 64 B/cycle per link, and driving only one
// is the single largest factor measured anywhere in docs/he/dram_guide.md (§3.1, 1.79x). The corner
// turn is read-dominated -- 128 KB in against 4 KB out per core -- so splitting the READS is what
// matters; the write stays with the assembler.
//
// This half only reads. Its partner a2a_corner_wr.cpp (BRISC/NOC_0) fetches the other half, waits on
// c_0, assembles and writes. c_0 is a plain producer/consumer CB across the two RISCs.
//
// Runtime args: 0 = NT, 1 = n_out, 2 = out_start, 3 = out_stride, 4 = kh, 5 = in_addr, 6 = out_addr
// (the same argument vector as a2a_corner_pg.cpp, so the host builds one list for both kernels).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    // The corner turn is DATAFLOW-ONLY -- no compute kernel, so without a named zone its
    // dispatch window emits no name and tools/traces/dispatch_csv.py labels it with whichever
    // NEIGHBOURING dispatch's zone happens to start inside its bounds. That put 152 a2a
    // dispatches under modmul_fused3 and the three matmuls in the 54.1 ms census.
    DeviceZoneScopedN("a2a_corner");
    const uint32_t NT      = get_arg_val<uint32_t>(0);
    const uint32_t kh      = get_arg_val<uint32_t>(4);
    const uint32_t in_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb = tt::CBIndex::c_0;
    const uint32_t tbytes = get_tile_size(cb);
    const uint32_t half = NT >> 1;

    const InterleavedAddrGenFast<true> in = {.bank_base_address = in_addr,
                                             .page_size = tbytes, .data_format = DataFormat::UInt32};
    cb_reserve_back(cb, half);
    const uint32_t l1 = get_write_ptr(cb);
    for (uint32_t n1 = 0; n1 < half; ++n1) {          // page n1*2+kh -> bank (n1*2+kh) % 8
        noc_async_read_tile(n1 * 2 + kh, in, l1 + n1 * tbytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb, half);
}
