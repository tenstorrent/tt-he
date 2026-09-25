// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// compute/fused_matmul_a.cpp with a LOOP over n_tiles, for exp_Tg018dram.
//
// Identical arithmetic, identical three phases -- decompose (SFPU) -> int8 matmul (matrix engine) ->
// recombine mod q (SFPU). The only change is that one dispatch now carries n work tiles instead of
// one, which is what lets the sweep vary work-per-core independently of core count.
//
// THE ONE THING THAT IS NOT A MECHANICAL WRAP. fused_matmul_a.cpp opens with
// unary_op_init_common(cb_in, cb_limbs), and that call re-runs the DEST-sync handshake
// (MATH(llk_math_pack_sync_init) + PACK(llk_pack_dest_init)). Re-running it mid-kernel deadlocks
// MATH against PACK on Blackhole -- the reason the original file carries a header comment about it.
// So the init is HOISTED OUT of the loop and runs exactly once, and every per-iteration engine
// switch is light-weight only: reconfig_data_format* / pack_reconfig_data_format for the operand
// and output formats, matmul_init for the matrix engine, copy_tile_init / mul_int_tile_init /
// fill_tile_init to re-arm the SFPU math path. Phase 3 -> phase 1 of the next iteration is exactly
// the same kind of switch as phase 2 -> phase 3, which the original already does.
//
// W INDEXING. The batched reader pushes W as NB*NB tiles in (p,r) order, so foldW[p,r] is tile
// p*NB+r counted from the CB front. The original streamed them one at a time and always read index
// 0; here they are all resident, which is the same contraction with the addressing moved from the
// CB cursor into the index. With w_share the whole kernel runs off one resident copy and never pops.
//
// Runtime args: 0=q, 1=q_inv_32, 2=q_low16, 3=q_high16, 4=K14 (= mont(2^14)), 5=n_tiles, 6=w_share.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/mul_int_sfpu.h"

// NO DeviceZoneScopedN HERE, deliberately. exp_Tg018dram measures this kernel through the per-RISC
// `*-KERNEL` envelopes tt-metal emits for every dispatch, so a named zone would buy nothing -- and
// it costs: tt-metal hashes zone source locations into 16 bits, and adding one more zone to a tree
// that already registers dozens collided ("Source location hashes are colliding") and aborted the
// device at close.

namespace {
constexpr auto FMT = DataFormat::UInt32;
inline void MUL_INIT() { mul_int_tile_init<FMT>(); }
inline void MUL(uint32_t a, uint32_t b, uint32_t o) { mul_int_tile<FMT>(a, b, o); }
inline void FILL_INIT() { fill_tile_init(); }
inline void FILL(uint32_t d, uint32_t v) { fill_tile_int<FMT>(d, v); }
}  // namespace

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;

inline void face_decompose(uint32_t v, uint32_t l0, uint32_t l1, uint32_t l2, uint32_t l3) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt x = sfpi::dst_reg[v * NV + i];
        sfpi::dst_reg[l0 * NV + i] = x & 127u;
        sfpi::dst_reg[l1 * NV + i] = (x >> 7u) & 127u;
        sfpi::dst_reg[l2 * NV + i] = (x >> 14u) & 127u;
        sfpi::dst_reg[l3 * NV + i] = (x >> 21u) & 127u;
    }
}
inline void face_build_hi(uint32_t C2, uint32_t C3, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt c2 = sfpi::dst_reg[C2 * NV + i];
        sfpi::vUInt c3 = sfpi::dst_reg[C3 * NV + i];
        sfpi::dst_reg[out * NV + i] = c2 + (c3 << 7u);
    }
}
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
inline void face_split(uint32_t t, uint32_t tL, uint32_t tH) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vt = sfpi::dst_reg[t * NV + i];
        sfpi::dst_reg[tL * NV + i] = vt & 0xFFFFu;
        sfpi::dst_reg[tH * NV + i] = vt >> 16u;
    }
}
inline void face_reduce_fin(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL,
                            uint32_t hi, uint32_t qIdx, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vpMH = sfpi::dst_reg[pMH * NV + i];
        sfpi::vUInt vpML = sfpi::dst_reg[pML * NV + i];
        sfpi::vUInt vpLL = sfpi::dst_reg[pLL * NV + i];
        sfpi::vUInt mid_low = (vpMH & 0xFFFFu) + (vpML & 0xFFFFu) + (vpLL >> 16u);
        sfpi::vUInt mid_high = (vpMH >> 16u) + (vpML >> 16u) + (mid_low >> 16u);
        sfpi::vUInt t_final = sfpi::vUInt(sfpi::dst_reg[pHH * NV + i]) + mid_high;
        sfpi::vUInt q = sfpi::dst_reg[qIdx * NV + i];
        sfpi::vUInt b = sfpi::vUInt(sfpi::dst_reg[hi * NV + i]) + q - t_final;
        sfpi::vUInt bq = b - q;
        sfpi::vUInt res = bq;
        v_if(sfpi::reinterpret<sfpi::vInt>(bq) < 0) { res = b; }
        v_endif;
        sfpi::dst_reg[out * NV + i] = res;
    }
}
inline void face_final(uint32_t C0, uint32_t C1, uint32_t hi14, uint32_t qIdx, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt c0 = sfpi::dst_reg[C0 * NV + i];
        sfpi::vUInt c1 = sfpi::dst_reg[C1 * NV + i];
        sfpi::vUInt q = sfpi::dst_reg[qIdx * NV + i];
        sfpi::vUInt low = c0 + (c1 << 7u);
        sfpi::vUInt r = low + sfpi::vUInt(sfpi::dst_reg[hi14 * NV + i]);
        sfpi::vUInt rq = r - q;
        sfpi::vUInt res = rq;
        v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; }
        v_endif;
        r = res;
        rq = r - q;
        res = rq;
        v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; }
        v_endif;
        sfpi::dst_reg[out * NV + i] = res;
    }
}
}  // namespace
#endif

inline void S_decompose(uint32_t v, uint32_t l0, uint32_t l1, uint32_t l2, uint32_t l3) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_decompose, v, l0, l1, ckernel::VectorMode::RC, l2, l3));
}
inline void S_build_hi(uint32_t C2, uint32_t C3, uint32_t o) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_build_hi, C2, C3, o, ckernel::VectorMode::RC));
}
inline void S_limbs(uint32_t a, uint32_t b, uint32_t aL, uint32_t aH, uint32_t bL, uint32_t bH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_limbs, a, b, aL, ckernel::VectorMode::RC, aH, bL, bH));
}
inline void S_lohi(uint32_t ll, uint32_t hh, uint32_t c1, uint32_t c2, uint32_t lo, uint32_t hi) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_lohi, ll, hh, c1, ckernel::VectorMode::RC, c2, lo, hi));
}
inline void S_split(uint32_t t, uint32_t tL, uint32_t tH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_split, t, tL, tH, ckernel::VectorMode::RC));
}
inline void S_reduce_fin(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL,
                         uint32_t hi, uint32_t qIdx, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_reduce_fin, pHH, pMH, pML, ckernel::VectorMode::RC, pLL, hi, qIdx, out));
}
inline void S_final(uint32_t C0, uint32_t C1, uint32_t hi14, uint32_t qIdx, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_final, C0, C1, hi14, ckernel::VectorMode::RC, qIdx, out));
}

void kernel_main() {
    const uint32_t Q = get_arg_val<uint32_t>(0);
    const uint32_t QINV = get_arg_val<uint32_t>(1);
    const uint32_t QLOW = get_arg_val<uint32_t>(2);
    const uint32_t QHIGH = get_arg_val<uint32_t>(3);
    const uint32_t K14 = get_arg_val<uint32_t>(4);
    const uint32_t n_tiles = get_arg_val<uint32_t>(5);
    const uint32_t w_share = get_arg_val<uint32_t>(6);

    constexpr auto cb_in = tt::CBIndex::c_0;       // Xin int32
    constexpr auto cb_w = tt::CBIndex::c_1;        // folded W int8
    constexpr auto cb_limbs = tt::CBIndex::c_2;    // int8 limbs (intermediate)
    constexpr auto cb_planes = tt::CBIndex::c_3;   // int32 planes (intermediate)
    constexpr auto cb_out = tt::CBIndex::c_16;     // int32 result
    constexpr uint32_t NB = 4;
    constexpr uint32_t NW = NB * NB;

    // ONCE, never inside the loop -- see the header note on the DEST-sync handshake.
    unary_op_init_common(cb_in, cb_limbs);
    if (w_share) cb_wait_front(cb_w, NW);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        // ===== phase 1: decompose Xin -> 4 int8 limbs =====
        reconfig_data_format_srca(cb_in);
        pack_reconfig_data_format(cb_limbs);
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile_init(cb_in); copy_tile(cb_in, 0, 4);   // d4 = Xin
        S_decompose(4, 0, 1, 2, 3);                       // d0..d3 = limb 0..3
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_limbs, NB);
        pack_tile(0, cb_limbs, 0);
        pack_tile(1, cb_limbs, 1);
        pack_tile(2, cb_limbs, 2);
        pack_tile(3, cb_limbs, 3);
        cb_push_back(cb_limbs, NB);
        tile_regs_release();
        cb_pop_front(cb_in, 1);

        // ===== phase 2: matmul  C_p = sum_r foldW[p,r] @ limb_r  -> 4 int32 planes =====
        reconfig_data_format(cb_limbs, cb_w);
        pack_reconfig_data_format(cb_planes);
        matmul_init(cb_w, cb_limbs, 0 /*transpose*/);
        cb_wait_front(cb_limbs, NB);
        if (!w_share) cb_wait_front(cb_w, NW);
        for (uint32_t p = 0; p < NB; ++p) {
            tile_regs_acquire();
            for (uint32_t r = 0; r < NB; ++r) {
                matmul_tiles(cb_w, cb_limbs, p * NB + r, r, 0);   // foldW[p,r] @ limb_r, accumulate
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_planes, 1);
            pack_tile(0, cb_planes);
            cb_push_back(cb_planes, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_limbs, NB);
        if (!w_share) cb_pop_front(cb_w, NW);

        // ===== phase 3: recombine 4 planes -> R mod q =====
        reconfig_data_format_srca(cb_planes);
        pack_reconfig_data_format(cb_out);
        MUL_INIT();
        FILL_INIT();
        cb_wait_front(cb_planes, NB);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_tile_init(cb_planes);
        copy_tile(cb_planes, 2, 2);          // d2 = plane2
        copy_tile(cb_planes, 3, 3);          // d3 = plane3
        S_build_hi(2, 3, 0);                 // d0 = hi = plane2 + plane3<<7
        FILL(1, K14);
        S_limbs(0, 1, 2, 3, 4, 5);
        MUL(2, 4, 0); MUL(3, 5, 1); MUL(2, 5, 6); MUL(3, 4, 7);
        S_lohi(0, 1, 6, 7, 2, 3);
        FILL(0, QINV); MUL(2, 0, 2);
        S_split(2, 4, 5);
        FILL(0, QHIGH); FILL(1, QLOW);
        MUL(5, 0, 6); MUL(5, 1, 5); MUL(4, 0, 0); MUL(4, 1, 1);
        FILL(2, Q);
        S_reduce_fin(6, 5, 0, 1, 3, 2, 4);   // d4 = hi14
        copy_tile(cb_planes, 0, 0);          // d0 = plane0
        copy_tile(cb_planes, 1, 1);          // d1 = plane1
        S_final(0, 1, 4, 2, 5);              // d5 = R
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(5, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_planes, NB);
    }
}
