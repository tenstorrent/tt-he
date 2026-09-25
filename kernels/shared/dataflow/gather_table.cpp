// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Table-driven index-permutation gather: out[d] = in[tbl[d]], offsets in the poly's own physical
// (face-tiled) layout, so an arbitrary fixed coefficient permutation costs one dataflow program and
// no compute engine.
//
// This is what makes the four-step NTT buffer-resident.  `fwd2048`/`inv2048` bracket their device
// stages with a host reindex (`x -> blk0/blk1` on the way in, `blk -> X` on the way out) — a FIXED
// permutation of [0,n), independent of the data and the modulus.  Baking it into a table turns each
// of those host round-trips into a device gather, so an NTT becomes buffer-in / buffer-out.
// (`permute_poly.cpp` is the same idea with the table computed in-kernel from (A,Bc) plus a sign.)
//
// The table is NT tiles of uint32 built once on the host at Engine construction: tbl[phys(dest)] =
// phys(src).  in_base/in_stride and out_base/out_stride select one poly out of an interleaved
// multi-poly buffer (e.g. `digit_to_rns` writes residue ch at tiles ch, K+ch → in_base=ch,
// in_stride=K), so no repacking program is needed between stages.
//
// NPOLY polys are permuted per program (the table is read once and replayed), poly m living at
// in_base + m*in_pstride + t*in_stride and out_base + m*out_pstride + t*out_stride — so a whole batch
// of transforms needs ONE gather, not one per poly.
//
// Runtime args: 0=in, 1=out, 2=tbl, 3=NT (tiles/poly), 4=in_base, 5=in_stride, 6=out_base,
//               7=out_stride, 8=NPOLY, 9=in_pstride, 10=out_pstride.  All buffers UInt32 tiles.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t in_addr    = get_arg_val<uint32_t>(0);
    const uint32_t out_addr   = get_arg_val<uint32_t>(1);
    const uint32_t tbl_addr   = get_arg_val<uint32_t>(2);
    const uint32_t NT         = get_arg_val<uint32_t>(3);
    const uint32_t in_base    = get_arg_val<uint32_t>(4);
    const uint32_t in_stride  = get_arg_val<uint32_t>(5);
    const uint32_t out_base   = get_arg_val<uint32_t>(6);
    const uint32_t out_stride = get_arg_val<uint32_t>(7);
    const uint32_t NPOLY      = get_arg_val<uint32_t>(8);
    const uint32_t in_pstride = get_arg_val<uint32_t>(9);
    const uint32_t out_pstride= get_arg_val<uint32_t>(10);

    constexpr uint32_t cb = tt::CBIndex::c_0, TELEM = 1024;
    const uint32_t tbytes = get_tile_size(cb);
    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,  .page_size = tbytes, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr, .page_size = tbytes, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> tb  = {.bank_base_address = tbl_addr, .page_size = tbytes, .data_format = DataFormat::UInt32};

    const uint32_t n = NT * TELEM;
    cb_reserve_back(cb, 3 * NT);                       // [table | input poly | output poly]
    const uint32_t l1 = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    volatile tt_l1_ptr uint32_t* inb = tab + n;
    volatile tt_l1_ptr uint32_t* outb = inb + n;

    for (uint32_t t = 0; t < NT; ++t)                                    // table read once, replayed
        noc_async_read_tile(t, tb, reinterpret_cast<uint32_t>(tab) + t * tbytes);
    noc_async_read_barrier();

    for (uint32_t m = 0; m < NPOLY; ++m) {
        for (uint32_t t = 0; t < NT; ++t)
            noc_async_read_tile(in_base + m * in_pstride + t * in_stride, in,
                                reinterpret_cast<uint32_t>(inb) + t * tbytes);
        noc_async_read_barrier();
        for (uint32_t d = 0; d < n; ++d) outb[d] = inb[tab[d]];
        for (uint32_t t = 0; t < NT; ++t)
            noc_async_write_tile(out_base + m * out_pstride + t * out_stride, out,
                                 reinterpret_cast<uint32_t>(outb) + t * tbytes);
        noc_async_write_barrier();
    }
    cb_push_back(cb, 3 * NT);
}
