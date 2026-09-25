// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SIGNED PERMUTATION on device — the Galois automorphism (#14) and monomial multiply (#15).
//
//   automorph:  r[(i·j) mod 2N] = ±a[i]        (negate when the index wraps past N)
//   monomul:    r[(i+k) mod 2N] = ±a[i]
//
// Both are the same primitive: a fixed permutation of the N coefficients with a per-output sign. These were
// the last host operations in the Subs tree, and they forced a readback per key-switch output (~79% of
// expansion's remaining time), so they must be on device.
//
// KEY POINT: this needs no element-granular DMA. A dataflow kernel is ordinary C++ on a RISC-V core with
// direct L1 pointer access, so once the poly is in L1 (tile-granular DMA, fast) the permute is just an
// indexed copy: dst[m] = ±src[idx[m]]. One RISC, no compute engine, no CB producer/consumer sync — so no
// exposure to the engine-switch deadlock of docs/he/blackhole_port.md.
//
// The index table is precomputed on host **in tilized coordinates** (so it composes the tilize layout with
// the permutation) and is FIXED per (j or k, N): one table serves every node at a Subs level.
//   table[m] = source_tilized_position | (sign << 31)
//
// CBs are used purely as L1 scratch (no push/pop): c_0 = input, c_1 = index table, c_16 = output.
//   Runtime args: 0=in_addr, 1=idx_addr, 2=out_addr, 3=n_out_tiles, 4=q, 5=in_tile_off, 6=in_tile_stride,
//                 7=n_in_tiles (0 => same as n_out)
// n_in may differ from n_out: the scan tail gathers rows from 64 transposed tiles into 2 output tiles, with
// the row-gather composed into the same index table as the inv-input layout. Table size is n_out*1024;
// entries index the n_in*1024 input space.
// in_tile_off/stride let the input tiles be gathered from inside a larger buffer (e.g. residue (k,ch) sits
// at tiles k*2K+ch and k*2K+K+ch of the digit_rns output, i.e. stride K) -- so the producer need not read back.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t in_addr  = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t nt       = get_arg_val<uint32_t>(3);   // OUTPUT tiles (2 for a poly at n=2048)
    const uint32_t q        = get_arg_val<uint32_t>(4);
    const uint32_t in_off   = get_arg_val<uint32_t>(5);   // 0 for a plain 2-tile poly
    const uint32_t in_str   = get_arg_val<uint32_t>(6);   // 1 for contiguous tiles
    const uint32_t nt_in_a  = get_arg_val<uint32_t>(7);
    const uint32_t nt_in    = nt_in_a ? nt_in_a : nt;      // 0 => same as output (the common case)

    constexpr uint32_t cb_in = 0, cb_idx = 1, cb_out = 16;

    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(in_args, in_addr);
    constexpr auto idx_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto idxa = TensorAccessor(idx_args, idx_addr);
    constexpr auto out_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);

    const uint32_t ts    = get_tile_size(cb_in);
    const uint32_t p_in  = get_write_ptr(cb_in);
    const uint32_t p_idx = get_write_ptr(cb_idx);
    const uint32_t p_out = get_write_ptr(cb_out);

    for (uint32_t t = 0; t < nt_in; ++t)
        noc_async_read_tile(in_off + t * (in_str ? in_str : 1), in, p_in + t * ts);
    for (uint32_t t = 0; t < nt; ++t)
        noc_async_read_tile(t, idxa, p_idx + t * ts);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p_in);
    volatile tt_l1_ptr uint32_t* ix  = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p_idx);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p_out);

    const uint32_t n = nt * 1024;        // output elements; idx entries index nt_in*1024 inputs
    for (uint32_t m = 0; m < n; ++m) {
        const uint32_t e = ix[m];
        const uint32_t sgn = e >> 31, from = e & 0x7FFFFFFFu;
        const uint32_t v = src[from];
        dst[m] = (sgn && v) ? (q - v) : v;      // v==0 must stay 0, not q
    }

    for (uint32_t t = 0; t < nt; ++t) noc_async_write_tile(t, out, p_out + t * ts);
    noc_async_write_barrier();
}
