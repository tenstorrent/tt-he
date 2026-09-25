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

namespace {
constexpr auto FMT = DataFormat::UInt32;
constexpr uint32_t B      = GADGET_B;
constexpr uint32_t LOG2B  = GADGET_LOG2B;
constexpr uint32_t ELL    = GADGET_ELL;
constexpr uint32_t HALFB  = B >> 1;
constexpr uint32_t MASKB  = B - 1;
constexpr uint32_t SH     = 28u - LOG2B;      // hi-limb bits shifted into the low limb on >>B
constexpr uint32_t P28    = 1u << 28;         // limb radix 2^28
constexpr uint32_t LO_SLOT = ELL;             // input X0 lands here (slot after the ELL digits)
constexpr uint32_t HI_SLOT = ELL + 1;         // input X1
}  // namespace

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;

// X (radix-2^28 limbs in slots los,his) -> ELL signed digits in slots 0..ELL-1.
// (third arg unused — the LLK params helper always forwards 3 positional slot args.)
inline void face_gadget(uint32_t los, uint32_t his, uint32_t /*unused*/) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt lo = sfpi::dst_reg[los * NV + i];
        sfpi::vUInt hi = sfpi::dst_reg[his * NV + i];
        for (uint32_t k = 0; k < ELL; k++) {
            sfpi::vUInt raw = lo & MASKB;                              // low LOG2B bits
            sfpi::vUInt ylo = (lo >> LOG2B) | ((hi & MASKB) << SH);    // (X >> LOG2B) low limb
            sfpi::vUInt yhi = hi >> LOG2B;                             //           high limb
            sfpi::vInt d = sfpi::reinterpret<sfpi::vInt>(raw);
            // balanced: if raw >= B/2, digit = raw - B and carry +1 into the shifted value
            sfpi::vInt diff = d - (int)HALFB;
            v_if(diff < 0) {
                // raw < B/2: digit = raw, no carry (nothing to do)
            }
            v_else {
                d = d - (int)B;
                ylo = ylo + 1;
                sfpi::vInt yl = sfpi::reinterpret<sfpi::vInt>(ylo) - (int)P28;
                v_if(yl < 0) {
                    // no limb overflow
                }
                v_else {
                    ylo = sfpi::reinterpret<sfpi::vUInt>(yl);          // ylo -= 2^28
                    yhi = yhi + 1;
                }
                v_endif;
            }
            v_endif;
            sfpi::dst_reg[k * NV + i] = sfpi::reinterpret<sfpi::vUInt>(d);  // signed digit bits
            lo = ylo;
            hi = yhi;
        }
    }
}
}  // namespace
#endif

inline void S_gadget(uint32_t los, uint32_t his) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_gadget, los, his, 0, ckernel::VectorMode::RC));
}

void kernel_main() {
    DeviceZoneScopedN("gadget_decompose");
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in = tt::CBIndex::c_0;      // pairs (X0_tile, X1_tile)
    constexpr auto cb_out = tt::CBIndex::c_16;    // ELL digit tiles per input tile

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in, 2);

        tile_regs_acquire();
        copy_tile_init(cb_in);
        copy_tile(cb_in, 0, LO_SLOT);
        copy_tile(cb_in, 1, HI_SLOT);
        S_gadget(LO_SLOT, HI_SLOT);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out, ELL);
        for (uint32_t k = 0; k < ELL; ++k) pack_tile(k, cb_out, k);
        cb_push_back(cb_out, ELL);
        tile_regs_release();

        cb_pop_front(cb_in, 2);
    }
}
