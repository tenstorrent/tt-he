// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// matmul_modq_mbatch_1pass -- matmul_modq_mbatch with the recombine collapsed to ONE SFPI pass and
// the 4 int32 planes kept in DST.  Drop-in: same runtime args (q, q_inv, q_lo, q_hi, K14, M), same
// CBs, same output.  kernels/ir/compute/matmul_modq_mbatch.cpp is RETAINED and unchanged.
//
// This is the compute kernel of the OnionPIR initial-dimension scan -- the stage that touches every
// row of the database on every query -- so it is where the three techniques in kernels/OPTIMISED.md
// have the most to work with.  All three apply:
//
//   1. sfpmul24 instead of the fp32 byte-split   -- 9 mul_int_tile -> 22 sfpmul24 in registers
//   2. collapse the SFPI passes                  -- 7 passes / 53 DST accesses -> 1
//   3. broadcast scalars in register             -- all 5 fill_tile_int deleted
//
// and, because 1 frees the registers for 2, phases 2 and 3 also fuse.  The original packed each of
// the 4 planes out to cb_planes and copied all 4 straight back into DST; accumulating plane p into
// DST tile p inside ONE tile_regs_acquire removes, PER OUTPUT TILE, 4 pack_tile + 4 copy_tile
// (eight whole-tile moves through L1) and 3 of 4 acquire/commit/wait/release cycles.  At M=32 that
// is 256 tile moves per dispatch that no longer happen, and cb_planes goes unused entirely.
//
// DST use is 4 planes + 1 output = 5 of 8, independent of M, because M is stepped one tile at a
// time.  No srcA reconfig between the matmul and the recombine: the SFPI pass only reads and writes
// dst_reg, it unpacks nothing.  Only the PACK format has to move, once, to cb_out.
//
// BLACKHOLE ONLY: sfpi::fractional_mul is behind __riscv_xtttensixbh.  On Wormhole use
// matmul_modq_mbatch.cpp, which is unchanged and still correct.
//
// Original header follows.
// ----------------------------------------------------------------------------
// M-BATCHED modular matmul -- PHASE-MAJOR: batch M independent output tiles with only THREE engine-mode
// switches TOTAL (all decomposes, then all matmuls, then all recombines), NOT per-tile.  This is the
// safe batching pattern (single engine mode per phase, loop tiles); the failing slot-loop kernel
// switched modes per tile.  Each tile is an independent contract-32 modular matmul.
//
//   phase 1  decompose: Xin(int32, c_0) -> 4 int8 limbs (c_2)         [SFPU]
//   phase 2  matmul:     C_p = sum_r foldW[p,r](c_1) @ limb_r(c_2)     [matrix engine]
//   phase 3  recombine:  R = (sum_k 128^k * plane_k) mod q             [SFPU]
//
// Engine switch between phases must NOT be a full re-init: unary_op_init_common and mm_init both
// re-run the DEST-sync handshake, and re-running it mid-kernel deadlocks MATH against PACK on
// Blackhole.  One full init at entry, then light-weight per-phase switches only.
// (Fusing 2 and 3 does NOT reintroduce that: there is no init between them at all -- the SFPI pass
// runs inside the same acquire as the matmuls that produced its inputs.)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/mul_int_sfpu.h"
#include "tools/profiler/kernel_profiler.hpp"

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

// Exact 23x23 -> 46-bit product; both operands are silently masked to 23 bits, so every caller
// below feeds it 16-bit limbs.  docs/he/exp_tg018s.md.
inline sfpi::vUInt mul16(sfpi::vUInt x, sfpi::vUInt y) {
    using namespace sfpi;
    vUInt lo = vUInt(fractional_mul(x, y, FractionalHalf::Low));    // product bits [22:0]
    vUInt hi = vUInt(fractional_mul(x, y, FractionalHalf::High));   // product bits [45:23]
    return lo + (hi << 23u);
}

// The whole of phase 3 in one pass: R = (C0 + C1<<7 + C2<<14 + C3<<21) mod q, via one Montgomery
// reduction of hi14 = (C2 + C3<<7) * 2^14.  Same arithmetic, same constants, same canonical output
// as the seven-pass original -- the partial products simply never reach DST.
inline void face_recomb(uint32_t C0, uint32_t C1, uint32_t C2, uint32_t C3, uint32_t out,
                        uint32_t K14, uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt vhi = vUInt(dst_reg[C2 * NV + i]) + (vUInt(dst_reg[C3 * NV + i]) << 7u);
        // full 32x32 product vhi * K14 -> lo, hiP
        vUInt aL = vhi & 0xFFFFu, aH = vhi >> 16u;
        vUInt bL = vUInt(K14 & 0xFFFFu), bH = vUInt(K14 >> 16u);
        vUInt ll = mul16(aL, bL);
        vUInt cross = mul16(aL, bH) + mul16(aH, bL);
        vUInt hh = mul16(aH, bH);
        vUInt mid = (ll >> 16u) + (cross & 0xFFFFu);
        vUInt lo = (ll & 0xFFFFu) | (mid << 16u);
        vUInt hiP = hh + (cross >> 16u) + (mid >> 16u);
        // t = (lo * q_inv_32) mod 2^32
        vUInt qiL = vUInt(QINV & 0xFFFFu), qiH = vUInt(QINV >> 16u);
        vUInt lL = lo & 0xFFFFu, lH = lo >> 16u;
        vUInt t = mul16(lL, qiL) + ((mul16(lL, qiH) + mul16(lH, qiL)) << 16u);
        vUInt tL = t & 0xFFFFu, tH = t >> 16u;
        // the four t*q products and the reduce -> hi14
        vUInt vqh = vUInt(QHIGH), vql = vUInt(QLOW), vq = vUInt(Q);
        vUInt pHH = mul16(tH, vqh), pMH = mul16(tH, vql);
        vUInt pML = mul16(tL, vqh), pLL = mul16(tL, vql);
        vUInt mid_low  = (pMH & 0xFFFFu) + (pML & 0xFFFFu) + (pLL >> 16u);
        vUInt mid_high = (pMH >> 16u) + (pML >> 16u) + (mid_low >> 16u);
        vUInt t_final = pHH + mid_high;
        vUInt b1 = hiP + vq - t_final;
        vUInt bq1 = b1 - vq;
        vUInt hi14 = bq1;
        v_if(sfpi::reinterpret<sfpi::vInt>(bq1) < 0) { hi14 = b1; }
        v_endif;
        // R = (C0 + C1<<7 + hi14) mod q -- two conditional subtracts, as face_final did
        vUInt low = vUInt(dst_reg[C0 * NV + i]) + (vUInt(dst_reg[C1 * NV + i]) << 7u);
        vUInt r = low + hi14;
        vUInt rq = r - vq;
        vUInt res = rq;
        v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; }
        v_endif;
        r = res; rq = r - vq; res = rq;
        v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; }
        v_endif;
        dst_reg[out * NV + i] = res;
    }
}
}  // namespace
#endif

inline void S_decompose(uint32_t v, uint32_t l0, uint32_t l1, uint32_t l2, uint32_t l3) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_decompose, v, l0, l1, ckernel::VectorMode::RC, l2, l3));
}
inline void S_recomb(uint32_t C0, uint32_t C1, uint32_t C2, uint32_t C3, uint32_t out,
                     uint32_t K14, uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_recomb, C0, C1, C2, ckernel::VectorMode::RC,
                                               C3, out, K14, QINV, QHIGH, QLOW, Q));
}

void kernel_main() {
    const uint32_t Q = get_arg_val<uint32_t>(0);
    const uint32_t QINV = get_arg_val<uint32_t>(1);
    const uint32_t QLOW = get_arg_val<uint32_t>(2);
    const uint32_t QHIGH = get_arg_val<uint32_t>(3);
    const uint32_t K14 = get_arg_val<uint32_t>(4);
    const uint32_t M = get_arg_val<uint32_t>(5);   // BATCH: M independent output tiles

    constexpr auto cb_in = tt::CBIndex::c_0;       // Xin int32 (M)
    constexpr auto cb_w = tt::CBIndex::c_1;        // folded W int8 (M*16), streamed in matmul phase
    constexpr auto cb_limbs = tt::CBIndex::c_2;    // int8 limbs (M*4 resident)
    constexpr auto cb_out = tt::CBIndex::c_16;     // int32 result (M)
    constexpr uint32_t NB = 4;                     // c_3 (planes) is deliberately unused

    DeviceZoneScopedN("mbatch_total");

    // ===== phase 1: decompose all M Xin -> M*4 int8 limbs (one unary init) =====
    unary_op_init_common(cb_in, cb_limbs);
    cb_wait_front(cb_in, M);
    cb_reserve_back(cb_limbs, M * NB);
    for (uint32_t m = 0; m < M; ++m) {
        tile_regs_acquire();
        copy_tile_init(cb_in); copy_tile(cb_in, m, 4);
        S_decompose(4, 0, 1, 2, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_limbs, m * NB + 0);
        pack_tile(1, cb_limbs, m * NB + 1);
        pack_tile(2, cb_limbs, m * NB + 2);
        pack_tile(3, cb_limbs, m * NB + 3);
        tile_regs_release();
    }
    cb_push_back(cb_limbs, M * NB);
    cb_pop_front(cb_in, M);

    // ===== phases 2+3 FUSED: matmul -> recombine, planes never leave DST =====
    reconfig_data_format(cb_limbs, cb_w);
    matmul_init(cb_w, cb_limbs, 0 /*transpose*/);
    pack_reconfig_data_format(cb_out);
    cb_wait_front(cb_limbs, M * NB);
    cb_reserve_back(cb_out, M);
    for (uint32_t m = 0; m < M; ++m) {
        // ONE cb_wait_front/cb_pop_front for this tile's whole 4x4 foldW block, not sixteen.  The
        // original synchronised with the reader per tile -- 512 wait/pop pairs per dispatch at M=32 --
        // and each pair is a semaphore round trip that stalls MATH.  Waiting for the block once and
        // indexing into it costs nothing extra: the tiles are already contiguous in the CB, and
        // reader_mbatch_deep delivers them 16 at a time anyway.
        cb_wait_front(cb_w, NB * NB);
        tile_regs_acquire();
        for (uint32_t p = 0; p < NB; ++p) {
            for (uint32_t r = 0; r < NB; ++r) {
                matmul_tiles(cb_w, cb_limbs, p * NB + r, m * NB + r, p);  // foldW[m][p,r] @ limb[m][r] -> DST p
            }
        }
        S_recomb(0, 1, 2, 3, 4, K14, QINV, QHIGH, QLOW, Q);       // planes read in place from d0..d3
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(4, cb_out, m);
        tile_regs_release();
        cb_pop_front(cb_w, NB * NB);
    }
    cb_push_back(cb_out, M);
    cb_pop_front(cb_limbs, M * NB);
}
