// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// reader_seq_batch — Reads n consecutive tiles with ALL of them in flight behind one barrier (the DRAM depth rule).
//
// MEASURED: read depth is the difference between ~50 and ~250 GB/s.  docs/he/dram_guide.md
//
// OPTIMISED variant promoted into the shared kernel tree; the kernel it derives from
// (kernels/experiments/exp_tb001/dataflow/reader_seq_batch.cpp) is RETAINED and unchanged.
// See kernels/OPTIMISED.md.
//
// Original header follows.
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: Apache-2.0
//
// Reads n consecutive int32 tiles into c_0 with ALL of them in flight, then one barrier -- the
// depth that exp_Tg018dram measured as the difference between ~50 and ~250 GB/s (dram_guide.md).
// Consecutive page ids also put consecutive tiles on consecutive DRAM banks, which is why every
// buffer in exp_Tg018r is laid out page = our_tile * L + limb rather than limb * NOT + our_tile.
//
// Runtime args: 0 = addr, 1 = n_tiles, 2 = start_page.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t ts = get_tile_size(cb_in);
    const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb_in, n);
    const uint32_t p = get_write_ptr(cb_in);
    for (uint32_t i = 0; i < n; ++i) noc_async_read_tile(start + i, s, p + i * ts);
    noc_async_read_barrier();
    cb_push_back(cb_in, n);
}
