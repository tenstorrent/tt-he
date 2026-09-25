// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PIR first-dimension SCAN as a MODULAR batched GEMM on the int8 matrix engine.
//
//   C[mt][nt] = Σ_kt  A[mt][kt] @ B[kt][nt]   (mod q),   per coefficient slot, per RNS channel
//
// This is the batched-GEMM formulation of the first dimension (docs/pir/latency_optimization.md §4d):
// the coefficient slot sits on the BATCH axis, so both operands are slot-local and there is no
// cross-slot product to discard. m = rest, k = N0 (contraction), n = ct components.
//
// WHY BOTH OPERANDS ARRIVE PRE-DECOMPOSED (and why that is free):
// a 28-bit modular product lowers as  x·y = Σ_{p,s} 128^(p+s)·limb(x,p)·limb(y,s)  (mod q), validated
// by cpu/pir/check_scan_gemm.py. Storing 4 int8 limbs per element costs exactly 4 bytes -- identical to
// one uint32 -- so pre-decomposing offline costs NO extra storage or bandwidth, and it keeps this kernel
// free of an on-core SFPU decompose phase. That matters: mixing SFPU and matmul phases in one kernel is
// what caused the DEST-sync deadlock in docs/he/blackhole_port.md. Here the kernel is matmul-only; the
// mod-q recombine is a separate program (the SFPU tail).
//
// Products sharing p+s share a radix weight, so 16 limb-pairs accumulate into 7 int32 planes
// (p+s = 0..6) rather than 16 -- 7 tiles fits the 8-tile DEST budget with fp32_dest_acc_en.
// Accumulator bound: 127·127·32·KB per plane-pair, ~1.5% of int32 max at KB=16. No overflow.
//
//   c_0 = A limb tiles (int8, 4 per k-block), c_1 = B limb tiles (int8, 4 per k-block)
//   c_16 = 7 int32 plane tiles per output tile
//   Runtime args: 0 = n_out (output tiles for this core), 1 = KB (contraction tiles)

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "hostdevcommon/kernel_structs.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    DeviceZoneScopedN("scan_gemm");
    const uint32_t n_out = get_arg_val<uint32_t>(0);
    const uint32_t KB    = get_arg_val<uint32_t>(1);
    const uint32_t MTS   = get_arg_val<uint32_t>(2);   // m-tiles per slot; B is resident for that long

    constexpr uint32_t cb_a   = tt::CBIndex::c_0;    // A limbs (int8)
    constexpr uint32_t cb_b   = tt::CBIndex::c_1;    // B limbs (int8)
    constexpr uint32_t cb_out = tt::CBIndex::c_16;   // int32 planes
    constexpr uint32_t NL = 4;                       // radix-128 limbs per operand
    // 8 planes, not 7: p+s spans [0,6] but packing a zero 8th plane makes the recombine consume two groups
    // of FOUR CONTIGUOUS tiles, which is exactly what recombine_bigK takes -- so the mod-q tail needs no new
    // kernel (2x recombine_bigK + one modmul by mont(128^4 mod q) + one mod-add). See §4f C1.
    constexpr uint32_t NPLANE = 2 * NL;              // p+s ∈ [0,6] used; plane 7 packed as zero

    mm_init(cb_a, cb_b, cb_out);

    // B is L1-resident for a whole slot (BPS tiles): waited once, indexed as kt*NL+s, popped only when
    // the slot's MTS output tiles are done. That is what removes the 4x B re-read (see reader).
    const uint32_t BPS = KB * NL;
    uint32_t mt_in_slot = 0;

    for (uint32_t o = 0; o < n_out; ++o) {
        if (mt_in_slot == 0) cb_wait_front(cb_b, BPS);          // this slot's B arrives once
        tile_regs_acquire();                          // planes start zeroed, accumulate across kt
        for (uint32_t kt = 0; kt < KB; ++kt) {
            cb_wait_front(cb_a, NL);
            for (uint32_t p = 0; p < NL; ++p)
                for (uint32_t s = 0; s < NL; ++s)
                    matmul_tiles(cb_a, cb_b, p, kt * NL + s, p + s);   // -> DEST plane p+s
            cb_pop_front(cb_a, NL);
        }
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out, NPLANE);
        for (uint32_t d = 0; d < NPLANE; ++d) pack_tile(d, cb_out);   // d=7 is never written -> zero
        cb_push_back(cb_out, NPLANE);
        tile_regs_release();

        if (MTS > 0 && ++mt_in_slot == MTS) { cb_pop_front(cb_b, BPS); mt_in_slot = 0; }
    }
}
