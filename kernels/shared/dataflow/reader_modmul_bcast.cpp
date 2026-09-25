// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// `reader_modmul.cpp` with a BROADCAST second operand: a streams, b is one tile reused for every
// tile of a.
//
// reader_modmul applies a single `start_tile` to BOTH operands, which is right for a batched MAC
// (dN and the KSK indexed alike) but wrong for the NTT twiddle stage, where all M polys multiply by
// the SAME twiddle tile. The workaround was to replicate the twiddle M× in DRAM — which silently
// caps M at the replication count and reads out of bounds past it (the bug this kernel fixes: any
// core whose tile index exceeded the 32-copy buffer got garbage). Broadcasting instead removes the
// replication buffer and the cap together.
//
// b_cycle generalises "one tile" to "one repeating block": b tile = b_base + ((a_start+i) % b_cycle).
//   b_cycle = 1 -> a constant operand (NTT twiddle, Garner inverse)
//   b_cycle = 2 -> a 2-tile poly reused across a batch (the KSK key in a level-wide MAC)
// Indexing uses the GLOBAL tile index (a_start+i) so the phase stays correct on every shard.
//
// Runtime args: 0 = a_addr, 1 = b_addr, 2 = n_tiles, 3 = a_start, 4 = b_base, 5 = b_cycle.

#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);
    const uint32_t a_start = get_arg_val<uint32_t>(3);
    const uint32_t b_base = get_arg_val<uint32_t>(4);
    const uint32_t b_cycle = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0, cb_in1 = tt::CBIndex::c_1;
    const uint32_t ts = get_tile_size(cb_in0);
    const InterleavedAddrGenFast<true> a = {.bank_base_address = a_addr, .page_size = ts, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> b = {.bank_base_address = b_addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_reserve_back(cb_in0, 1);
        cb_reserve_back(cb_in1, 1);
        noc_async_read_tile(a_start + i, a, get_write_ptr(cb_in0));
        noc_async_read_tile(b_base + (a_start + i) % b_cycle, b, get_write_ptr(cb_in1));
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
        cb_push_back(cb_in1, 1);
    }
}
