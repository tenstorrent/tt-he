// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_modmul001 / modmul_sfpi_hybrid
//
// One of THREE Montgomery modular multiply kernels compared head to head on ONE core by
// host/experiments/exp_modmul001.cpp.  All three are self-contained copies in kernels/experiments/exp_modmul001/ so that
// nothing about this comparison depends on another experiment's kernel tree.
//
// The three are scaffolded IDENTICALLY -- same CBs (c_0 = a, c_1 = b, c_16 = out), same runtime
// args, same per-tile loop, same copy_tile/pack_tile, and **no DeviceZoneScopedN anywhere**.
// That last point is why these are fresh copies: kernels/experiments/exp_tb001/compute/modmul_sfpi_hybrid.cpp
// carries ~8 profiler zones inside its per-tile loop while modmul_sfpi_full.cpp carries one at
// kernel entry, so any timing that compared those two files was comparing different amounts of
// instrumentation as well as different arithmetic.  Timing here comes only from tt-metal's own
// firmware `*-KERNEL` envelope, which is identical for all three.
//
// MULTIPLY: mul_int_tile -- the tt-metal tile-op API -- for every product.  All the NON-multiply
// arithmetic is hand-written SFPI kept in vector registers, so the kernel runs 4 SFPI passes where
// modmul_sfpi_full runs 7 and avoids the DST round-trips between them.
// OUTPUT: LAZY, in [0, 2q) -- the final canonical subtract is deliberately skipped so results chain
// back as lazy operands.  The bench therefore scores it BOTH ways: against the canonical golden and
// against the lazy contract (out == gold or out == gold + q).
//
// Runtime args: 0=n_tiles, 1=q, 2=q_inv_32, 3=q_low16, 4=q_high16.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/mul_int_sfpu.h"

namespace {
constexpr auto FMT = DataFormat::UInt32;
// init programs persistent SFPU state (mul: LREG12/13 consts + SFPLOADMACRO
// templates; fill: its store template). It only needs to run once before a run
// of the same op -- not before every call. We re-init when the op kind changes,
// since the intervening op may have reprogrammed that shared SFPU state.
inline void MUL_INIT() { mul_int_tile_init<FMT>(); }
inline void MUL(uint32_t a, uint32_t b, uint32_t o) { mul_int_tile<FMT>(a, b, o); }
inline void FILL_INIT() { fill_tile_init(); }
inline void FILL(uint32_t d, uint32_t v) { fill_tile_int<FMT>(d, v); }
}  // namespace

// ---- custom SFPI face functions (MATH core only). Each is one face of 8 vectors;
//      the params wrapper iterates all 4 faces. Tiles addressed as dst_reg[idx*32+i]. ----
#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;  // dst_reg slots per tile (n_vector_in_tile)

// in: a, b  -> out: aL, aH, bL, bH   (16-bit limbs)
inline void face_limbs(uint32_t a, uint32_t b, uint32_t aL, uint32_t aH, uint32_t bL, uint32_t bH) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt va = sfpi::dst_reg[a * NV + i];
        sfpi::vUInt vb = sfpi::dst_reg[b * NV + i];
        sfpi::dst_reg[aL * NV + i] = va & 0xFFFFu;
        sfpi::dst_reg[aH * NV + i] = va >> 16u;
        sfpi::dst_reg[bL * NV + i] = vb & 0xFFFFu;
        sfpi::dst_reg[bH * NV + i] = vb >> 16u;
    }
}

// in: ll, hh, c1=aL*bH, c2=aH*bL  -> out: lo, hi   (mul_hi_lo assembly)
inline void face_lohi(uint32_t ll, uint32_t hh, uint32_t c1, uint32_t c2, uint32_t lo, uint32_t hi) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vll = sfpi::dst_reg[ll * NV + i];
        sfpi::vUInt vhh = sfpi::dst_reg[hh * NV + i];
        sfpi::vUInt cross = sfpi::vUInt(sfpi::dst_reg[c1 * NV + i]) + sfpi::vUInt(sfpi::dst_reg[c2 * NV + i]);
        sfpi::vUInt mid = (vll >> 16u) + (cross & 0xFFFFu);
        sfpi::dst_reg[lo * NV + i] = (vll & 0xFFFFu) | (mid << 16u);
        sfpi::dst_reg[hi * NV + i] = vhh + (cross >> 16u) + (mid >> 16u);
    }
}

// in: t  -> out: tL, tH
inline void face_split(uint32_t t, uint32_t tL, uint32_t tH) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vt = sfpi::dst_reg[t * NV + i];
        sfpi::dst_reg[tL * NV + i] = vt & 0xFFFFu;
        sfpi::dst_reg[tH * NV + i] = vt >> 16u;
    }
}

// in: pHH, pMH, pML, pLL, hi, q  -> out: (a*b) mod q, lazy in [0, 2q) (no finalize)
inline void face_reduce(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL,
                        uint32_t hi, uint32_t qIdx, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vpMH = sfpi::dst_reg[pMH * NV + i];
        sfpi::vUInt vpML = sfpi::dst_reg[pML * NV + i];
        sfpi::vUInt vpLL = sfpi::dst_reg[pLL * NV + i];
        sfpi::vUInt mid_low = (vpMH & 0xFFFFu) + (vpML & 0xFFFFu) + (vpLL >> 16u);
        sfpi::vUInt mid_high = (vpMH >> 16u) + (vpML >> 16u) + (mid_low >> 16u);
        sfpi::vUInt t_final = sfpi::vUInt(sfpi::dst_reg[pHH * NV + i]) + mid_high;
        sfpi::vUInt q = sfpi::dst_reg[qIdx * NV + i];
        // Lazy output in [0, 2q): hi + q - t_final. The conditional subtract that
        // would canonicalize to [0, q) is skipped -- operands chain back in lazy
        // form, so the finalize is deferred to the very end of the pipeline.
        sfpi::dst_reg[out * NV + i] = sfpi::vUInt(sfpi::dst_reg[hi * NV + i]) + q - t_final;
    }
}
}  // namespace
#endif

// Wrappers (MATH() expands to nothing off the MATH core). Pass VectorMode::RC
// explicitly since extra tile indices follow it in the variadic wrapper.
inline void S_limbs(uint32_t a, uint32_t b, uint32_t aL, uint32_t aH, uint32_t bL, uint32_t bH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_limbs, a, b, aL, ckernel::VectorMode::RC, aH, bL, bH));
}
inline void S_lohi(uint32_t ll, uint32_t hh, uint32_t c1, uint32_t c2, uint32_t lo, uint32_t hi) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_lohi, ll, hh, c1, ckernel::VectorMode::RC, c2, lo, hi));
}
inline void S_split(uint32_t t, uint32_t tL, uint32_t tH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_split, t, tL, tH, ckernel::VectorMode::RC));
}
inline void S_reduce(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL,
                     uint32_t hi, uint32_t qIdx, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_reduce, pHH, pMH, pML, ckernel::VectorMode::RC, pLL, hi, qIdx, out));
}

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t Q = get_arg_val<uint32_t>(1);
    const uint32_t QINV = get_arg_val<uint32_t>(2);
    const uint32_t QLOW = get_arg_val<uint32_t>(3);
    const uint32_t QHIGH = get_arg_val<uint32_t>(4);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in0, cb_out);

    // mul_int and fill each program persistent SFPU state once; nothing in the
    // loop (SFPI passes, copy_tile, the other op) reprograms it, so a single
    // init per op for the whole kernel is sufficient and bit-exact (verified vs
    // golden at N=4K..64K). NOTE: if you insert a different SFPU op between the
    // muls/fills, re-init the affected op after it.
    MUL_INIT();
    FILL_INIT();

    for (uint32_t t = 0; t < n_tiles; ++t) {

        {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);
            cb_reserve_back(cb_out, 1);
        }

        tile_regs_acquire();

        {
            copy_tile_init(cb_in0); copy_tile(cb_in0, 0, 0);  // d0 = a
            copy_tile_init(cb_in1); copy_tile(cb_in1, 0, 1);  // d1 = b
        }

        {
            // One START/END pair per tile, recorded per RISC. Read it on the
            // TRISC_1 (MATH) row -> end-start = cycles for this block (= ns at 1 GHz).
            // With n_tiles == num_cores (1 tile/core) you get a clean per-tile number.

            {
                S_limbs(0, 1, 2, 3, 4, 5);  // aL=d2, aH=d3, bL=d4, bH=d5
            }
            {
                MUL(2, 4, 0);  // ll  = aL*bL -> d0
                MUL(3, 5, 1);  // hh  = aH*bH -> d1
                MUL(2, 5, 6);  // c1  = aL*bH -> d6
                MUL(3, 4, 7);  // c2  = aH*bL -> d7
            }
            {
                S_lohi(0, 1, 6, 7, 2, 3);  // lo=d2, hi=d3
            }

            {
                FILL(0, QINV);   // d0 = q_inv_32
                MUL(2, 0, 2);    // t = lo*q_inv_32 -> d2
                S_split(2, 4, 5);  // tL=d4, tH=d5
            }

            {
                FILL(0, QHIGH);  // d0 = q_high
                FILL(1, QLOW);   // d1 = q_low
                MUL(5, 0, 6);    // pHH = tH*q_high -> d6
                MUL(5, 1, 5);    // pMH = tH*q_low  -> d5
                MUL(4, 0, 0);    // pML = tL*q_high -> d0
                MUL(4, 1, 1);    // pLL = tL*q_low  -> d1
                FILL(2, Q);      // d2 = q
            }
            {
                S_reduce(6, 5, 0, 1, 3, 2, 4);  // out = (a*b) mod q -> d4
            }
        }

        {
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(4, cb_out);
            tile_regs_release();
        }

        {
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }
    }
}
