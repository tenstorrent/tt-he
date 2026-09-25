// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// writer_probe — writer_deep.cpp with the DRAM write DELETED, to separate two candidate causes of
// the 334.6 GB/s plateau: is it the read side (NoC/DRAM read limit) or the cost of MIXING reads and
// writes on the bus?
//
// It drains c_16 exactly as the real writer does -- so the compute kernel is not stalled and the READ
// side is unchanged -- but writes a single sentinel tile instead of n.  The kernel therefore computes
// a WRONG answer and the correctness gate MUST fail for the timing to be valid (exp_ntt001
// §2b-quinquies: a probe that corrupts the output and still passes is a broken instrument).
//
// (real - probe) is the write's cost; probe alone gives the read-only bandwidth of this exact feed.
//
// Runtime args: identical to writer_deep.cpp.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16, CHUNK = get_compile_time_arg_val(0);
    const uint32_t ts = get_tile_size(cb_out);
    const InterleavedAddrGenFast<true> s = {.bank_base_address = addr,
                                            .page_size = ts, .data_format = DataFormat::UInt32};
    bool once = true;
    for (uint32_t i = 0; i < n; i += CHUNK) {
        const uint32_t k = (n - i < CHUNK) ? (n - i) : CHUNK;
        cb_wait_front(cb_out, k);
        if (once) { noc_async_write_tile(start, s, get_read_ptr(cb_out));
                    noc_async_write_barrier(); once = false; }
        cb_pop_front(cb_out, k);
    }
}
