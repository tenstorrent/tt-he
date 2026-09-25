// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// `reader_gadget.cpp` for an INTERLEAVED limb buffer — the buffer-resident CRT→gadget hand-off.
//
// The Garner CRT tail (`crt_mul_split`/`crt_limb3`) writes its K radix-2^28 limbs into ONE buffer,
// tile-position-major: limb `lb` at tiles `base + t*KLIMBS + lb`. `reader_gadget` instead expects K
// separate buffers, which is why the host used to read the CRT output back, de-interleave it, and
// write K buffers. Reading the interleaved buffer directly removes that round-trip entirely.
//
// Push order is identical to reader_gadget — (t, then j) — so `gadget_decompose_signed` is unchanged.
//
// Runtime args: 0 = addr, 1 = n_tiles, 2 = base.  KLIMBS from the GADGET_KLIMBS define.

#include <cstdint>

#ifndef GADGET_KLIMBS
#define GADGET_KLIMBS 2
#endif

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t base = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = tt::CBIndex::c_0, KLIMBS = GADGET_KLIMBS;
    const uint32_t ts = get_tile_size(cb_in);
    const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t t = 0; t < n_tiles; ++t) {
        for (uint32_t j = 0; j < KLIMBS; ++j) {
            cb_reserve_back(cb_in, 1);
            noc_async_read_tile(base + t * KLIMBS + j, s, get_write_ptr(cb_in));
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
}
