// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PIR #14/#15 on device: the signed negacyclic coefficient permutations — Galois automorphism
// σ_j (x -> x^j) and monomial multiply (x^k) — as an L1 gather, no SFPU/FPU.  Replaces the host
// reindex that `pir_permute.hpp` did between device stages (the last host arithmetic left in the
// Subs tree of query expansion).
//
// Both ops are the SAME map once written destination-driven.  Scatter form (the definition):
//     automorph σ_j:  out[±(i·j mod 2n)] = ± in[i]      monomial x^k: out[±(i+k mod 2n)] = ± in[i]
// j odd ⇒ i ↦ i·j is a bijection mod 2n, so each output is written exactly once and the map can be
// inverted.  Gather form (what this kernel runs), for output coefficient o ∈ [0,n):
//     m = (o·A + Bc) mod 2n ;  m < n → out[o] = in[m] ;  m ≥ n → out[o] = q − in[m−n]   (x^n = −1)
//     automorph σ_j : A = j⁻¹ mod 2n, Bc = 0        monomial x^k : A = 1, Bc = (−k) mod 2n
// One source per output ⇒ a pure gather.  2n is a power of two, so "mod 2n" is a mask.
//
// The (source offset, sign) table depends only on (A,Bc,n) — NOT on the data or the modulus — so it
// is built ONCE per invocation into L1 and then replayed over every poly this core owns.  Polys are
// replayed over every poly this invocation owns; each is held in L1 one at a time.
//
// Addressing is strided so the same kernel serves both ciphertext layouts: poly m tile t lives at
// in_base + m*in_pstride + t*in_tstride (likewise out).  The per-ct layout [comp*K+ch] uses
// pstride=NT, tstride=1, qdiv=1; the level-batch layout [comp][ch][half][node] is walked one
// (comp,ch) group at a time with pstride=1, tstride=B, NQ=1.  The modulus for poly m is
// q[((qoff+m)/qdiv) % NQ] — qoff carries the poly index this shard starts at, so a core whose shard
// does not begin on a multiple of K still picks the right channel.
//
// Runtime args: 0=in, 1=out, 2=NT (tiles/poly, n = NT*1024), 3=A, 4=Bc, 5=npoly, 6=NQ, 7=qdiv,
//               8=in_base, 9=in_pstride, 10=in_tstride, 11=out_base, 12=out_pstride, 13=out_tstride,
//               14=qoff, 15..15+NQ-1 = the NQ moduli.  int32 (UInt32) tiles, canonical values in [0,q).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t TELEM = 1024, MAXQ = 4, SIGN = 0x80000000u;
// natural coefficient index -> offset in the poly's contiguous NT-tile L1 block (32x32 face layout)
inline uint32_t phys(uint32_t m) {
    const uint32_t t = m >> 10, i = m & 1023, r = i >> 5, c = i & 31;
    return t * TELEM + ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}
}  // namespace

void kernel_main() {
    const uint32_t in_addr    = get_arg_val<uint32_t>(0);
    const uint32_t out_addr   = get_arg_val<uint32_t>(1);
    const uint32_t NT         = get_arg_val<uint32_t>(2);
    const uint32_t A          = get_arg_val<uint32_t>(3);
    const uint32_t Bc         = get_arg_val<uint32_t>(4);
    const uint32_t npoly      = get_arg_val<uint32_t>(5);
    const uint32_t NQ         = get_arg_val<uint32_t>(6);
    const uint32_t qdiv       = get_arg_val<uint32_t>(7);
    const uint32_t in_base    = get_arg_val<uint32_t>(8);
    const uint32_t in_pstride = get_arg_val<uint32_t>(9);
    const uint32_t in_tstride = get_arg_val<uint32_t>(10);
    const uint32_t out_base   = get_arg_val<uint32_t>(11);
    const uint32_t out_pstride= get_arg_val<uint32_t>(12);
    const uint32_t out_tstride= get_arg_val<uint32_t>(13);
    const uint32_t qoff       = get_arg_val<uint32_t>(14);
    uint32_t qv[MAXQ];
    for (uint32_t i = 0; i < NQ && i < MAXQ; ++i) qv[i] = get_arg_val<uint32_t>(15 + i);

    const uint32_t n = NT * TELEM, mask = 2 * n - 1;

    constexpr uint32_t cb = tt::CBIndex::c_0;
    const uint32_t tbytes = get_tile_size(cb);
    const InterleavedAddrGenFast<true> in  = {.bank_base_address = in_addr,  .page_size = tbytes, .data_format = DataFormat::UInt32};
    const InterleavedAddrGenFast<true> out = {.bank_base_address = out_addr, .page_size = tbytes, .data_format = DataFormat::UInt32};

    // L1: [0,NT) = gather table, [NT,2NT) = input poly, [2NT,3NT) = output poly
    cb_reserve_back(cb, 3 * NT);
    const uint32_t l1 = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    volatile tt_l1_ptr uint32_t* inb = tab + n;
    volatile tt_l1_ptr uint32_t* outb = inb + n;

    for (uint32_t o = 0; o < n; ++o) {                       // build (source offset | sign) once
        const uint32_t m = (o * A + Bc) & mask;
        tab[phys(o)] = (m < n) ? phys(m) : (phys(m - n) | SIGN);
    }

    for (uint32_t m = 0; m < npoly; ++m) {
        const uint32_t q = qv[(NQ == 1) ? 0 : (((qoff + m) / qdiv) % NQ)];
        for (uint32_t t = 0; t < NT; ++t)
            noc_async_read_tile(in_base + m * in_pstride + t * in_tstride, in,
                                reinterpret_cast<uint32_t>(inb) + t * tbytes);
        noc_async_read_barrier();
        for (uint32_t d = 0; d < n; ++d) {
            const uint32_t e = tab[d], v = inb[e & ~SIGN];
            outb[d] = (e & SIGN) ? (v ? q - v : 0u) : v;     // negate mod q, keeping 0 -> 0
        }
        for (uint32_t t = 0; t < NT; ++t)
            noc_async_write_tile(out_base + m * out_pstride + t * out_tstride, out,
                                 reinterpret_cast<uint32_t>(outb) + t * tbytes);
        noc_async_write_barrier();
    }
    cb_push_back(cb, 3 * NT);
}
