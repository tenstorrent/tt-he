// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Two-level strided reader for `modadd_reduce_groups`: feeds G groups of n_in tiles.
//
// NOTE ON THE NAME: the `wh` branch called this `reader_i32_groups.cpp`, but this tree already has a
// DIFFERENT kernel by that name (groups of CONSECUTIVE tiles, args {addr,n_groups,gsize,gstride,base})
// used by the 7-plane mod-q recombine. The two are not interchangeable and the arg lists collide
// silently -- surplus runtime args are ignored, so the mismatch produces wrong values, not an error.
// Hence the `_2l` (two-level) suffix. Keep both.
//
//   group g starts at   base(g) = (g / gdiv) * gstride + (g % gdiv)
//   within a group      tile i  = base(g) + i * istride
//
// The level-wide MAC needs exactly this shape.  Products land in [node][digit][half] order, tile
// b*2ell + 2k + t, and the reduction runs over the digit axis k.  With one group per (node, half) —
// g = 2b + t, so gdiv=2, gstride=2*ell, istride=2 — group g reads the ell tiles belonging to that
// node/half, and the G outputs come out as tile 2b+t, which is precisely the layout the batched
// inverse NTT reads (poly b at tiles 2b, 2b+1).  No repacking anywhere.
//
// g_start shifts the group index so the groups can be sharded across cores.
// Runtime args: 0 = addr, 1 = G, 2 = n_in, 3 = gdiv, 4 = gstride, 5 = istride, 6 = g_start.

#include <cstdint>

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t G = get_arg_val<uint32_t>(1);
    const uint32_t n_in = get_arg_val<uint32_t>(2);
    const uint32_t gdiv = get_arg_val<uint32_t>(3);
    const uint32_t gstride = get_arg_val<uint32_t>(4);
    const uint32_t istride = get_arg_val<uint32_t>(5);
    const uint32_t g_start = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t ts = get_tile_size(cb_in);
    const InterleavedAddrGenFast<true> s = {.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32};

    for (uint32_t gl = 0; gl < G; ++gl) {
        const uint32_t g = g_start + gl;
        const uint32_t base = (g / gdiv) * gstride + (g % gdiv);
        for (uint32_t i = 0; i < n_in; ++i) {
            cb_reserve_back(cb_in, 1);
            noc_async_read_tile(base + i * istride, s, get_write_ptr(cb_in));
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
}
