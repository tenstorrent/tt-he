// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Strided int32 tile writer: tile i goes to (start + i*stride).  A separate file rather than an extra
// argument on the shared `writer_i32`, because adding a runtime arg to a shared dataflow kernel is not
// a compile error at its other call sites — it silently reads uninitialised memory there.
//   Args: 0=addr, 1=n_tiles, 2=start, 3=stride.
#include <cstdint>

void kernel_main() {
    const uint32_t addr    = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start   = get_arg_val<uint32_t>(2);
    const uint32_t stride  = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t ts = get_tile_size(cb_out);
    const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(start + i * stride, s, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
