// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Batched variant of reader_gather.cpp — same CB contents, ONE barrier per group of B tiles.
//
// WHY THIS EXISTS. reader_gather.cpp issues `reserve; read; barrier; push` per tile, so it keeps
// exactly ONE read in flight and pays a full DRAM round trip for every tile. Under contention that
// round trip is the dominant cost: at G=64 it measures ~11 us, against ~2.3 us at G=8. Utilisation
// tracks bytes-in-flight per core (measured on this card, same G, same interleaving: 2 tiles -> 16%
// of achievable, 17 tiles -> 25%, 64 tiles -> 94%), so a single outstanding read is the worst case.
//
// The n reads are mutually independent, so B of them can be in flight at once. This is the same fix
// reader_fused_batch.cpp applied to the matmul feed, and the same idea as the transaction-id
// double-buffering in tt-metal's tech_reports/Saturating_DRAM_bandwidth.
//
// B MUST EQUAL THE CB DEPTH. get_write_ptr() returns the write cursor and the B tiles are filled at
// cursor + j*tile_size with NO wrap handling, so a group must never straddle the buffer end. With
// B == depth every group starts at the buffer base, which is the only arrangement that guarantees it.
// (Same hazard the FBC hit at a mismatched depth -- see docs/he/exp_tb001.md.)
//
// Runtime args: 0 = n, 1 = B, 2 = page, 3..(2+n) = source buffer base addresses.
// PAGE VARIANT for exp_Tg018m: the sources are 64-page buffers and the tile wanted is `page` of
// each, not page 0 -- see a2a_corner_pg.cpp for why single-page buffers all land on bank 0.
#include <cstdint>

void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t B = get_arg_val<uint32_t>(1);
    const uint32_t page = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t ts = get_tile_size(cb_in);

    for (uint32_t i = 0; i < n; i += B) {
        const uint32_t k = (n - i < B) ? (n - i) : B;
        cb_reserve_back(cb_in, k);
        const uint32_t base = get_write_ptr(cb_in);
        for (uint32_t j = 0; j < k; ++j) {                 // all k reads in flight
            const uint32_t addr = get_arg_val<uint32_t>(3 + i + j);
            const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};
            noc_async_read_tile(page, s, base + j * ts);
        }
        noc_async_read_barrier();                          // ...then ONE barrier
        cb_push_back(cb_in, k);
    }
}
