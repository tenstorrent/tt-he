// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// all_to_all_corner_probe — MEASUREMENT ONLY.  Computes a WRONG answer on purpose.
//
// all_to_all_corner_wr.cpp does three things: read NT tiles, assemble one output tile out of one row
// from each, write it.  docs/matmul/exp_ntt001.md 2b-bis split those terms by solving two equations
// (one-NoC vs both-NoC) and got ~15.4 us of reads against ~16.6 us of assembly -- but that split does
// not reconcile with the machine: 128 KB in 7.7 us is 17 GB/s per core, and x32 cores that exceeds the
// 436 GB/s ceiling of docs/he/dram_guide.md 1.  So the model is wrong somewhere, and the question it
// was being used to answer -- is a core-to-core NoC exchange worth building? -- deserves a direct
// measurement instead.
//
// This kernel is all_to_all_corner_wr.cpp with the ASSEMBLY LOOP DELETED: same reads on both NoCs,
// same single write, no row copying.  The stage time it produces is therefore
//     read + write + launch,
// and (real stage - this stage) is the assembly cost.  That decides whether replacing the DRAM read
// with a NoC exchange can pay: if the reads are a small fraction, T3's whole premise is wrong.
//
// IT DOES NOT COMPUTE THE CORNER TURN.  The output tile is whatever was left in the scratch CB, so
// any run using this kernel MUST fail its correctness gate -- that is the expected result and the
// proof the kernel was actually swapped in.  Never wire it into a path whose output is trusted.
//
// Runtime args: identical to all_to_all_corner_{rd,wr}.cpp.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    DeviceZoneScopedN("a2a_corner");

    const uint32_t in_addr    = get_arg_val<uint32_t>(0);
    const uint32_t out_addr   = get_arg_val<uint32_t>(1);
    const uint32_t NT         = get_arg_val<uint32_t>(2);
    const uint32_t out_start  = get_arg_val<uint32_t>(3);
    const uint32_t out_stride = get_arg_val<uint32_t>(4);
    const uint32_t n_out      = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_lo = tt::CBIndex::c_0;
    constexpr uint32_t cb_hi = tt::CBIndex::c_1;
    constexpr uint32_t cb_o  = tt::CBIndex::c_2;
    const uint32_t tbytes = get_tile_size(cb_lo);
    const uint32_t half = NT >> 1;

    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,
                                              .page_size = tbytes, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr,
                                              .page_size = tbytes, .data_format = DataFormat::UInt32};

    cb_reserve_back(cb_hi, half);
    const uint32_t hi = get_write_ptr(cb_hi);
    for (uint32_t n1 = half; n1 < NT; ++n1) {
        noc_async_read_tile(n1, in, hi + (n1 - half) * tbytes);
    }
    noc_async_read_barrier();

    cb_wait_front(cb_lo, half);                        // the partner's half, same as the real kernel

    cb_reserve_back(cb_o, 1);
    volatile tt_l1_ptr uint32_t* outp =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_o));
    for (uint32_t i = 0; i < n_out; ++i) {
        // ---- ASSEMBLY DELETED HERE (the 32 x 32-word row copies) ----
        // ONE sentinel word, not a fill: the real assembly copies 32 tiles x 32 words = 1024 words,
        // so filling 1024 words would leave the same L1 write count in place and isolate nothing.
        // One word is enough to prove the write lands (and to force the correctness gate to fail).
        outp[0] = 0xDEADBEEFu;
        const uint32_t s = out_start + i * out_stride;
        noc_async_write_tile(s, out, reinterpret_cast<uint32_t>(outp));
        noc_async_write_barrier();
    }
    cb_pop_front(cb_lo, half);
    cb_push_back(cb_o, 1);
}
