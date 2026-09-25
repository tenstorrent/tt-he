// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// writer_fbc_tr_pg — writer_fbc_tr.cpp for per-limb destination buffers. See reader_fbc_tr_pg.cpp for
// why: exp_Tg018 keeps each limb in its own 64-page buffer, so the LD destination base addresses come
// in as runtime args and limb j is written to its own buffer at page `ot`.
//
// Runtime args: 0=LD, 1=ngroups, 2=ot_start, 3.. = LD destination base addresses.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t LD = get_arg_val<uint32_t>(0);
    const uint32_t ngroups = get_arg_val<uint32_t>(1);
    const uint32_t ot_start = get_arg_val<uint32_t>(2);
    constexpr uint32_t A0 = 3;                    // first per-limb destination base address

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t CHUNK = 16 * 4;
    const uint32_t ts = get_tile_size(cb_out);

    for (uint32_t g = 0; g < ngroups; ++g) {
        const uint32_t ot = ot_start + g / 32, r = g % 32;
        const uint32_t dst_lo = (((r / 16) * 2) * 256 + (r % 16) * 16) * 4;
        cb_wait_front(cb_out, 1);
        const uint32_t p = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < LD; ++j) {
            const uint32_t src_lo = (((j / 16) * 2) * 256 + (j % 16) * 16) * 4;
            const InterleavedAddrGenFast<true> oj = {.bank_base_address = get_arg_val<uint32_t>(A0 + j),
                                                    .page_size = ts, .data_format = DataFormat::UInt32};
            const uint64_t base = oj.get_noc_addr(ot);
            noc_async_write(p + src_lo, base + dst_lo, CHUNK);
            noc_async_write(p + src_lo + 256 * 4, base + dst_lo + 256 * 4, CHUNK);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
