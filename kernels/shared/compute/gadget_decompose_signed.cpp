// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Signed gadget decomposition (PIR kernel #9), on-device (SFPI).
//   Input : per coefficient a 56-bit value X in [0, q), q = q1*q2 (~56 bit), supplied as two
//           radix-2^28 limbs  X = X0 + X1*2^28  (X0,X1 in [0, 2^28)) — the form CRT-combine
//           (#13) emits.  Two int32 tiles per coefficient-tile on CB c_0: (X0_tile, X1_tile).
//   Output: ELL signed base-B digit tiles on CB c_16 (int32, two's complement), each digit in
//           [-B/2, B/2), with  Σ_k digit_k · B^k ≡ X (mod q).  Balanced digits => low ext-prod noise.
//
// B must be a power of two (B = 2^LOG2B, LOG2B < 28) so decomposition is pure shifts/masks.
// Balanced base-B needs B^ELL >= 2q (one extra digit vs unsigned), see onionpir_ref.py.
//
// Register budget (fp32_dest_acc => 8 DST slots): ELL digit slots + 2 input slots.  So this
// single-pass kernel supports ELL <= 6 (the L_EP external-product config).  Larger ELL
// (expansion/key-switch, L_KEY/L_KS = 8..10) will chunk over passes — a follow-up.
//
// The whole per-coefficient decomposition (all ELL digits, with balanced carry propagated
// across the two 28-bit limbs) is one custom SFPI register pass — the running value lives in
// LOCAL vUInt registers across the digit loop, so it costs no DST slots.
//
// Compile-time: GADGET_B, GADGET_LOG2B, GADGET_ELL.   Runtime arg: 0 = n_tiles (X-tiles).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifndef GADGET_B
#define GADGET_B 1024
#endif
#ifndef GADGET_LOG2B
#define GADGET_LOG2B 10
#endif
#ifndef GADGET_ELL
#define GADGET_ELL 6
#endif
// Digits emitted per register pass. DST holds CHUNK output + 2 input ≤ 8 slots → CHUNK ≤ 6.
// ELL>6 (expansion key-switch, L_KEY/L_KS=8) runs ceil(ELL/CHUNK) passes, recomputing the running
// value each pass and storing only that pass's digit window. ELL≤6 → 1 pass (identical to before).
#ifndef GADGET_CHUNK
#define GADGET_CHUNK 6
#endif
// Number of radix-2^28 input limbs = RNS limb count K.  K=2 (~56-bit q), K=3 (~84-bit q).
// DST budget = CHUNK output + KLIMBS input ≤ 8, so CHUNK ≤ 8-KLIMBS (K=2→6, K=3→5).
#ifndef GADGET_KLIMBS
#define GADGET_KLIMBS 2
#endif

namespace {
constexpr auto FMT = DataFormat::UInt32;
constexpr uint32_t B      = GADGET_B;
constexpr uint32_t LOG2B  = GADGET_LOG2B;
constexpr uint32_t ELL    = GADGET_ELL;
constexpr uint32_t KLIMBS = GADGET_KLIMBS;
constexpr uint32_t HALFB  = B >> 1;
constexpr uint32_t MASKB  = B - 1;
constexpr uint32_t SH     = 28u - LOG2B;      // hi-limb bits shifted into the low limb on >>B
constexpr uint32_t P28    = 1u << 28;         // limb radix 2^28
constexpr uint32_t CAP    = 8u - KLIMBS;      // max digit outputs per pass (DST = CAP + KLIMBS ≤ 8)
constexpr uint32_t CM     = (GADGET_CHUNK < CAP) ? GADGET_CHUNK : CAP;
constexpr uint32_t CHUNK  = (CM < ELL) ? CM : ELL;                       // digits per pass
constexpr uint32_t NPASS  = (ELL + CHUNK - 1) / CHUNK;                    // passes over the digit range
constexpr uint32_t IN0    = CHUNK;            // input limbs land at slots CHUNK .. CHUNK+KLIMBS-1
}  // namespace

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;

// X (KLIMBS radix-2^28 limbs at slots IN0..IN0+KLIMBS-1) -> this pass's digit window
// [k_lo, k_lo+CHUNK) in slots 0..CHUNK-1. Running value in registers across all ELL digits; balanced
// carry propagates across all KLIMBS limbs. (k_lo forwarded as the LLK helper's 2nd positional arg.)
inline void face_gadget(uint32_t base, uint32_t k_lo, uint32_t /*unused*/) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt lo = sfpi::dst_reg[(base + 0) * NV + i];
#if GADGET_KLIMBS >= 3
        sfpi::vUInt mid = sfpi::dst_reg[(base + 1) * NV + i];
        sfpi::vUInt hi  = sfpi::dst_reg[(base + 2) * NV + i];
#else
        sfpi::vUInt hi  = sfpi::dst_reg[(base + 1) * NV + i];
#endif
        for (uint32_t k = 0; k < ELL; k++) {
            sfpi::vUInt raw = lo & MASKB;                              // low LOG2B bits = candidate digit
#if GADGET_KLIMBS >= 3
            sfpi::vUInt ylo  = (lo >> LOG2B)  | ((mid & MASKB) << SH); // (X >> LOG2B), 3-limb chain
            sfpi::vUInt ymid = (mid >> LOG2B) | ((hi & MASKB) << SH);
            sfpi::vUInt yhi  = hi >> LOG2B;
#else
            sfpi::vUInt ylo  = (lo >> LOG2B)  | ((hi & MASKB) << SH);
            sfpi::vUInt yhi  = hi >> LOG2B;
#endif
            sfpi::vInt d = sfpi::reinterpret<sfpi::vInt>(raw);
            sfpi::vInt diff = d - (int)HALFB;
            v_if(diff < 0) {
                // raw < B/2: digit = raw, no carry
            }
            v_else {
                d = d - (int)B;                                       // balanced digit
                ylo = ylo + 1;                                        // carry +1 into the shifted value
                sfpi::vInt yl = sfpi::reinterpret<sfpi::vInt>(ylo) - (int)P28;
                v_if(yl < 0) {
                    // no overflow out of the low limb
                }
                v_else {
                    ylo = sfpi::reinterpret<sfpi::vUInt>(yl);         // ylo -= 2^28
#if GADGET_KLIMBS >= 3
                    ymid = ymid + 1;                                 // carry into mid
                    sfpi::vInt ym = sfpi::reinterpret<sfpi::vInt>(ymid) - (int)P28;
                    v_if(ym < 0) {
                        // no overflow out of mid
                    }
                    v_else {
                        ymid = sfpi::reinterpret<sfpi::vUInt>(ym);    // ymid -= 2^28
                        yhi = yhi + 1;                               // carry into hi
                    }
                    v_endif;
#else
                    yhi = yhi + 1;
#endif
                }
                v_endif;
            }
            v_endif;
            if (k >= k_lo && (k - k_lo) < CHUNK)                       // store only this pass's window
                sfpi::dst_reg[(k - k_lo) * NV + i] = sfpi::reinterpret<sfpi::vUInt>(d);
            lo = ylo;
#if GADGET_KLIMBS >= 3
            mid = ymid;
#endif
            hi = yhi;
        }
    }
}
}  // namespace
#endif

inline void S_gadget(uint32_t base, uint32_t k_lo) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_gadget, base, k_lo, 0, ckernel::VectorMode::RC));
}

void kernel_main() {
    DeviceZoneScopedN("gadget_decompose");
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in = tt::CBIndex::c_0;      // KLIMBS radix-2^28 limb tiles per coefficient-tile
    constexpr auto cb_out = tt::CBIndex::c_16;    // ELL digit tiles per coefficient-tile

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in, KLIMBS);
        // ceil(ELL/CHUNK) passes: pass p emits digits [p·CHUNK, p·CHUNK+chunk) — pushed in order, so the
        // writer sees digits 0..ELL-1 contiguous. Each pass re-derives the running value from X in registers.
        for (uint32_t p = 0; p < NPASS; ++p) {
            const uint32_t k_lo = p * CHUNK;
            const uint32_t chunk = (ELL - k_lo < CHUNK) ? (ELL - k_lo) : CHUNK;
            tile_regs_acquire();
            copy_tile_init(cb_in);
            for (uint32_t j = 0; j < KLIMBS; ++j) copy_tile(cb_in, j, IN0 + j);
            S_gadget(IN0, k_lo);
            tile_regs_commit();

            tile_regs_wait();
            cb_reserve_back(cb_out, chunk);
            for (uint32_t j = 0; j < chunk; ++j) pack_tile(j, cb_out, j);
            cb_push_back(cb_out, chunk);
            tile_regs_release();
        }
        cb_pop_front(cb_in, KLIMBS);
    }
}
