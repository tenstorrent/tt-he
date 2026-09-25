// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for compute/scan_gemm.cpp. This core computes output tiles [o0, o0+n_out); for each it streams
// KB k-blocks, each block being NL=4 int8 limb tiles of A and of B.
//
//   A limb tile index = ((o0+o)*KB + kt)*NL + l     -- unique per output tile (the streamed DB)
//   B limb tile index = (kt)*NL + l                 -- shared across this core's output tiles
//
// NOTE on the B index: sharing B across output tiles is correct only when those output tiles belong to
// the SAME coefficient slot (they differ in the m/n tile, not the slot). That is the fix relative to
// reader_scan_mm.cpp, which shares one B across ALL output tiles -- valid only for constant-polynomial
// plaintexts (the n=1024 demo), not the packed 128 MB cell. Callers must therefore launch per slot, or
// pass a per-slot B base address.
//
// SLOT BATCHING: output tiles enumerate (slot, m-tile) pairs, so global tile index gidx = slot*MTS + mt.
// A is laid out [slot][mt][kt][limb], which makes its index simply (gidx*KB + kt)*NL + l. B is laid out
// [slot][kt][limb] and must advance once per SLOT, not per output tile -- hence MTS. Passing MTS = n_out
// with a single slot reproduces the previous single-slot behaviour.
//
//   Runtime args: 0 = a_addr, 1 = b_addr, 2 = o0, 3 = n_out, 4 = KB, 5 = MTS (m-tiles per slot)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t o0     = get_arg_val<uint32_t>(2);
    const uint32_t n_out  = get_arg_val<uint32_t>(3);
    const uint32_t KB     = get_arg_val<uint32_t>(4);
    const uint32_t MTS    = get_arg_val<uint32_t>(5);   // m-tiles per slot; B advances per slot

    constexpr uint32_t cb_a = 0, cb_b = 1;
    constexpr uint32_t NL = 4;

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    const uint32_t tsa = get_tile_size(cb_a);
    const uint32_t tsb = get_tile_size(cb_b);

    // B IS LOADED ONCE PER SLOT, not once per output tile. B is shared by all MTS m-tiles of a slot, so
    // re-reading it per output tile cost 4x more B traffic than necessary: at 2048 output tiles that was
    // 128 MB of B against 134 MB of A (~262 MB ≈ 32 GB/s, i.e. bandwidth-bound on self-inflicted traffic,
    // which is why 64 cores measured no faster than 32). Loading per slot cuts B traffic to 32 MB.
    // The compute kernel therefore holds the whole slot's B resident in L1 (KB*NL int8 tiles = 64 KB at
    // KB=16) and indexes it without popping until the slot is done.
    const uint32_t BPS = KB * NL;                       // B tiles per slot
    uint32_t last_slot = 0xFFFFFFFFu;
    for (uint32_t o = 0; o < n_out; ++o) {
        const uint32_t gidx = o0 + o;
        const uint32_t slot = (MTS > 0) ? (gidx / MTS) : 0;
        const uint32_t abase = (gidx * KB) * NL;
        if (slot != last_slot) {
            cb_reserve_back(cb_b, BPS);
            const uint32_t wb = get_write_ptr(cb_b);
            const uint32_t bbase = (slot * KB) * NL;
            for (uint32_t t = 0; t < BPS; ++t)
                noc_async_read_tile(bbase + t, b, wb + t * tsb);
            noc_async_read_barrier();
            cb_push_back(cb_b, BPS);
            last_slot = slot;
        }
        for (uint32_t kt = 0; kt < KB; ++kt) {
            cb_reserve_back(cb_a, NL);
            const uint32_t wa = get_write_ptr(cb_a);
            for (uint32_t l = 0; l < NL; ++l)
                noc_async_read_tile(abase + kt * NL + l, a, wa + l * tsa);
            noc_async_read_barrier();
            cb_push_back(cb_a, NL);
        }
    }
}
