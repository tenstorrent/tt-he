// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// all_to_all_corner_rd — corner-turn READ HALF for the (32,32,32) NTT-32768.  NCRISC/NOC_1 fetches
// the first NT/2 input tiles; its partner all_to_all_corner_wr.cpp fetches the other half on NOC_0,
// then assembles and writes.
//
// WHY SPLIT IT.  all_to_all_corner_mc.cpp does everything in ONE kernel on RISCV_1, so all of its
// reads AND its writes ride NOC_1 and NOC_0 sits completely idle -- the only stage in the NTT
// pipeline with no NOC_0 traffic at all.  The corner turn is read-dominated (128 KB in against 4 KB
// out per core at C=32), and driving only one NoC is the single largest factor measured anywhere in
// docs/he/dram_guide.md (3.1, 1.79x).  Measured share: the corner turn is 32.24 us, 26.2% of the
// 3-factor's critical path at stock dataflow and 29.4% after the deep-reader work
// (docs/matmul/exp_ntt001.md 2b).
//
// This is a PORT of kernels/experiments/exp_tb001/dataflow/a2a_corner_{rd,wr}.cpp, which already does exactly
// this split for exp_Tg018m.  The transform is identical -- out[s][n1][t] = in[n1][s][t] -- and only
// the page indexing differs: exp_Tg018m interleaves two half-planes as `n1*2 + kh`, the NTT's tiles
// are plain `n1`.  all_to_all_corner_mc.cpp is RETAINED and unchanged; this is a sibling.
//
// The row-chunk copy and hoisted face-index arithmetic of all_to_all_corner_mc.cpp are preserved in
// the _wr half; this change is about the fabric only, not about the assembly loop.
//
// Runtime args (the SAME vector as all_to_all_corner_mc.cpp and as the _wr half, so the host builds
// one argument list for both kernels):
//   0=in_addr, 1=out_addr, 2=NT, 3=out_start, 4=out_stride, 5=n_out
// This half uses only in_addr and NT.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    // Dataflow-only dispatch: without a named zone tools/traces/dispatch_csv.py labels this window
    // with whichever NEIGHBOURING dispatch's zone starts inside it. That is the bug that made 152
    // a2a dispatches invisible in the exp_Tg018m census.
    DeviceZoneScopedN("a2a_corner");

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t NT      = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_lo = tt::CBIndex::c_0;      // tiles 0..half-1, handed to the _wr half
    const uint32_t tbytes = get_tile_size(cb_lo);
    const uint32_t half = NT >> 1;

    const InterleavedAddrGenFast<true> in = {.bank_base_address = in_addr,
                                             .page_size = tbytes, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb_lo, half);
    const uint32_t l1 = get_write_ptr(cb_lo);
    for (uint32_t n1 = 0; n1 < half; ++n1) {          // all in flight, one barrier (dram_guide 3.3)
        noc_async_read_tile(n1, in, l1 + n1 * tbytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_lo, half);                        // the _wr half waits on this
}
