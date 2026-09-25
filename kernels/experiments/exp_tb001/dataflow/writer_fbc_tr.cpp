// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// THE COEFFICIENT -> LIMB TRANSPOSE, done as NoC addressing on the way out.  Mirror of
// reader_fbc_tr.cpp: fbc_fpu_tr emits out'[j][c] (rows = destination modulus, columns =
// coefficient), and limb-major storage wants O_j[r][c].  Row j of out' and row r of O_j are each
// two contiguous 16-element runs, so it is 2 writes of 64 bytes per (group, destination).
//
// The 32 groups of one our-tile write disjoint 64-byte regions that exactly tile each output page,
// so no read-modify-write and no cross-core overlap.
//
// Runtime args: 0 = out_addr, 1 = LD, 2 = ngroups, 3 = ot_start, 4 = L (page = ot*L + j).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t LD = get_arg_val<uint32_t>(1);
    const uint32_t ngroups = get_arg_val<uint32_t>(2);
    const uint32_t ot_start = get_arg_val<uint32_t>(3);
    const uint32_t L = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t CHUNK = 16 * 4;
    const uint32_t ts = get_tile_size(cb_out);
    const InterleavedAddrGenFast<true> o = {.bank_base_address = out_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t g = 0; g < ngroups; ++g) {
        const uint32_t ot = ot_start + g / 32, r = g % 32;
        const uint32_t dst_lo = (((r / 16) * 2) * 256 + (r % 16) * 16) * 4;
        cb_wait_front(cb_out, 1);
        const uint32_t p = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < LD; ++j) {
            const uint32_t src_lo = (((j / 16) * 2) * 256 + (j % 16) * 16) * 4;
            const uint64_t base = o.get_noc_addr(ot * L + j);
            noc_async_write(p + src_lo, base + dst_lo, CHUNK);
            noc_async_write(p + src_lo + 256 * 4, base + dst_lo + 256 * 4, CHUNK);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
