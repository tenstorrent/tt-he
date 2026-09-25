// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ntt64_fused — the ENTIRE exp_Tg018 NTT (64 tiles, 3-level 32x32x64) as ONE compute kernel, one
// dispatch, with the corner turn done CORE-TO-CORE so no intermediate ever reaches DRAM.
//
//   fwd:  P1 (acc mma, w64) -> TW(ta) -> MM_C(w32a) -> TW(tb) -> <<a2a>> -> MM_A(w32b)
//   inv:  MM_A(w32b) -> <<a2a>> -> TW(tb) -> MM_C(w32a) -> TW(ta) -> P1 (acc mma, w64)
//
// Six passes that were six sets of dispatches, each round-tripping all 64 tiles through DRAM. Here
// the kernel reads its input tiles and the folded constants ONCE and writes its outputs ONCE; the
// twiddled/matmul'd intermediates and the corner turn all stay in L1.
//
// THE DECOMPOSITION THAT MAKES IT FIT.  Everything except the corner turn is tile-local, and the
// corner turn's ownership is a fixed point:
//
//   * a block is P=16 cores and carries ONE transform; core `me` owns n1 in {me, me+16}
//   * tile index t = n1*2 + kh, so a core holds FOUR tiles: pages {2me, 2me+1, 2me+32, 2me+33},
//     addressed here by local slot L = j*2 + kh  (j = which n1, kh = which half)
//   * P1 pairs pages 2n1 and 2n1+1 -- the two HALVES of the same n1 -- which is why a core must own
//     both halves, and why it then owns all four tiles for free
//   * the a2a maps input tile n1 -> output row n1 within a half, so a core's tile INDICES are
//     unchanged across it: slot L holds page(L) before and after
//
// 16 cores x 4 tiles = 64 = one transform, and 8 blocks = 128 cores = 8 transforms per dispatch.
//
// THE ENGINE-INIT RULE, which is what makes this kernel dangerous. `unary_op_init_common` and
// `mm_init` both re-run the DEST-sync handshake, and re-running it mid-kernel deadlocks MATH against
// PACK on Blackhole -- the failure that "hung every NTT-based binary" (docs/he/blackhole_port.md §2).
// This kernel alternates SFPU <-> matrix engine TEN times, so the rule is absolute:
//   * exactly ONE full init, at entry;
//   * per phase only matmul_init / reconfig_data_format* / copy_tile_init / pack_reconfig_data_format.
//
// THE MID-KERNEL WAIT. After the fourth phase this kernel blocks on cb_d, which BRISC fills only
// after a 16-core barrier. If any core in the block fails to reach its cb_x push, every core in that
// block hangs -- the all-or-nothing property of a2a_corner_x.cpp, now with a compute kernel inside
// the loop. The host must dispatch whole blocks.
//
// Runtime args: 0=Q, 1=QINV, 2=QLOW, 3=QHIGH, 4=K14 (= mont(2^14)), 5=fwd (1/0).

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

// NTT64_PHASE_ZONES: split this kernel's time into the SFPU passes and the matrix-engine matmuls.
// OFF by default -- it multiplies the marker volume by ~35 per dispatch and the per-RISC buffer will
// drop markers without a drained capture (kernels/experiments/exp_tb001/README.md). Build with
// -DNTT64_PHASE_ZONES and capture with TG18NTT_DRAIN_ONLY=1.
#ifdef NTT64_PHASE_ZONES
#define PH_ZONE(name) DeviceZoneScopedN(name)
#else
#define PH_ZONE(name)
#endif

namespace {
constexpr uint32_t NB = 4, NL = 4;             // NB = radix blocks, NL = local tiles per core
}

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;

// radix-128 split, from fused_matmul_a.cpp
inline void face_decompose(uint32_t v, uint32_t l0, uint32_t l1, uint32_t l2, uint32_t l3) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt x = sfpi::dst_reg[v * NV + i];
        sfpi::dst_reg[l0 * NV + i] = x & 127u;
        sfpi::dst_reg[l1 * NV + i] = (x >> 7u) & 127u;
        sfpi::dst_reg[l2 * NV + i] = (x >> 14u) & 127u;
        sfpi::dst_reg[l3 * NV + i] = (x >> 21u) & 127u;
    }
}

inline sfpi::vUInt mul16(sfpi::vUInt x, sfpi::vUInt y) {
    using namespace sfpi;
    vUInt lo = vUInt(fractional_mul(x, y, FractionalHalf::Low));
    vUInt hi = vUInt(fractional_mul(x, y, FractionalHalf::High));
    return lo + (hi << 23u);
}

// the whole recombine in one SFPI pass, from fused_matmul_a.cpp (unchanged)
inline void face_recomb(uint32_t C0, uint32_t C1, uint32_t C2, uint32_t C3, uint32_t out,
                        uint32_t K14, uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt vhi = vUInt(dst_reg[C2 * NV + i]) + (vUInt(dst_reg[C3 * NV + i]) << 7u);
        vUInt aL = vhi & 0xFFFFu, aH = vhi >> 16u;
        vUInt bL = vUInt(K14 & 0xFFFFu), bH = vUInt(K14 >> 16u);
        vUInt ll = mul16(aL, bL);
        vUInt cross = mul16(aL, bH) + mul16(aH, bL);
        vUInt hh = mul16(aH, bH);
        vUInt mid = (ll >> 16u) + (cross & 0xFFFFu);
        vUInt lo = (ll & 0xFFFFu) | (mid << 16u);
        vUInt hiP = hh + (cross >> 16u) + (mid >> 16u);
        vUInt qiL = vUInt(QINV & 0xFFFFu), qiH = vUInt(QINV >> 16u);
        vUInt lL = lo & 0xFFFFu, lH = lo >> 16u;
        vUInt t = mul16(lL, qiL) + ((mul16(lL, qiH) + mul16(lH, qiL)) << 16u);
        vUInt tL = t & 0xFFFFu, tH = t >> 16u;
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

// the whole Montgomery modmul in one SFPI pass, from modmul_fused3.cpp (unchanged)
inline void face_all(uint32_t A, uint32_t B_, uint32_t out,
                     uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt a = dst_reg[A * NV + i], b = dst_reg[B_ * NV + i];
        vUInt aL = a & 0xFFFFu, aH = a >> 16u, bL = b & 0xFFFFu, bH = b >> 16u;
        vUInt ll = mul16(aL, bL);
        vUInt cross = mul16(aL, bH) + mul16(aH, bL);
        vUInt hh = mul16(aH, bH);
        vUInt mid = (ll >> 16u) + (cross & 0xFFFFu);
        vUInt lo = (ll & 0xFFFFu) | (mid << 16u);
        vUInt hi = hh + (cross >> 16u) + (mid >> 16u);
        vUInt qiL = vUInt(QINV & 0xFFFFu), qiH = vUInt(QINV >> 16u);
        vUInt lL = lo & 0xFFFFu, lH = lo >> 16u;
        vUInt t = mul16(lL, qiL) + ((mul16(lL, qiH) + mul16(lH, qiL)) << 16u);
        vUInt tL = t & 0xFFFFu, tH = t >> 16u;
        vUInt vqh = vUInt(QHIGH), vql = vUInt(QLOW), vq = vUInt(Q);
        vUInt pHH = mul16(tH, vqh), pMH = mul16(tH, vql);
        vUInt pML = mul16(tL, vqh), pLL = mul16(tL, vql);
        vUInt mid_low  = (pMH & 0xFFFFu) + (pML & 0xFFFFu) + (pLL >> 16u);
        vUInt mid_high = (pMH >> 16u) + (pML >> 16u) + (mid_low >> 16u);
        vUInt t_final = pHH + mid_high;
        vUInt bo = hi + vq - t_final;
        vUInt bq = bo - vq;
        vUInt res = bq;
        v_if(sfpi::reinterpret<sfpi::vInt>(bq) < 0) { res = bo; }
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
inline void S_all(uint32_t A, uint32_t B_, uint32_t out,
                  uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_all, A, B_, out, ckernel::VectorMode::RC,
                                               QINV, QHIGH, QLOW, Q));
}

namespace {
constexpr auto cb_in    = tt::CBIndex::c_0;    // 4 int32 input tiles
constexpr auto cb_w64   = tt::CBIndex::c_1;    // 4 x 16 int8, the P1 folds
constexpr auto cb_wa    = tt::CBIndex::c_2;    // 16 int8, w32a (stage C, foldR)
constexpr auto cb_wb    = tt::CBIndex::c_3;    // 16 int8, w32b (stage A, foldL)
constexpr auto cb_ta    = tt::CBIndex::c_4;    // 2 int32 twiddles, indexed by kh
constexpr auto cb_tb    = tt::CBIndex::c_5;    // 4 int32 twiddles, one per local slot
constexpr auto cb_limbs = tt::CBIndex::c_6;    // 8 int8 (NACC=2 x NB)
constexpr auto cb_p     = tt::CBIndex::c_7;    // working
constexpr auto cb_q     = tt::CBIndex::c_8;    // working
constexpr auto cb_x     = tt::CBIndex::c_9;    // a2a SOURCE   -> BRISC
constexpr auto cb_d     = tt::CBIndex::c_10;   // a2a RESULT   <- BRISC / peers
constexpr auto cb_out   = tt::CBIndex::c_16;   // 4 int32 results -> BRISC writer
}  // namespace

void kernel_main() {
    DeviceZoneScopedN("ntt64_fused");
    const uint32_t Q     = get_arg_val<uint32_t>(0);
    const uint32_t QINV  = get_arg_val<uint32_t>(1);
    const uint32_t QLOW  = get_arg_val<uint32_t>(2);
    const uint32_t QHIGH = get_arg_val<uint32_t>(3);
    const uint32_t K14   = get_arg_val<uint32_t>(4);
    const bool     FWD   = get_arg_val<uint32_t>(5) != 0;
    const uint32_t DBG   = get_arg_val<uint32_t>(6);   // 1 = corner turn ONLY, for bisecting

    // THE one full init. Everything after this is a light-weight switch.
    unary_op_init_common(cb_in, cb_out);

    // ---- decompose src[si] -> NB int8 limbs at limb slot `ls` ----
    auto decompose_into = [&](uint32_t src, uint32_t si, uint32_t ls) {
        PH_ZONE("ph_sfpu");
        tile_regs_acquire();
        reconfig_data_format_srca(src);
        copy_tile_init(src); copy_tile(src, si, 4);
        S_decompose(4, 0, 1, 2, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_limbs);
        pack_tile(0, cb_limbs, ls * NB + 0);
        pack_tile(1, cb_limbs, ls * NB + 1);
        pack_tile(2, cb_limbs, ls * NB + 2);
        pack_tile(3, cb_limbs, ls * NB + 3);
        tile_regs_release();
    };

    // ---- ONE tile: decompose -> matmul(w, resident) -> recombine -> dst[di] ----
    // foldA: C_p = sum_r W[p,r] @ limb_r     foldC: C_r = sum_p limb_p @ W[p,r]
    auto pass_mm = [&](uint32_t src, uint32_t w, bool foldA, uint32_t dst) {
        cb_wait_front(src, NL);
        cb_wait_front(w, NB * NB);
        cb_reserve_back(dst, NL);
        for (uint32_t L = 0; L < NL; ++L) {
            cb_reserve_back(cb_limbs, NB);
            decompose_into(src, L, 0);
            cb_push_back(cb_limbs, NB);

            cb_wait_front(cb_limbs, NB);
            reconfig_data_format(cb_limbs, w);
            matmul_init(w, cb_limbs, 0);
            tile_regs_acquire();
            { PH_ZONE("ph_matmul");
              for (uint32_t p = 0; p < NB; ++p)
                  for (uint32_t r = 0; r < NB; ++r) {
                      if (foldA) matmul_tiles(w, cb_limbs, p * NB + r, r, p);
                      else       matmul_tiles(cb_limbs, w, p, p * NB + r, r);
                  } }
            { PH_ZONE("ph_sfpu"); S_recomb(0, 1, 2, 3, 4, K14, QINV, QHIGH, QLOW, Q); }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(dst);
            pack_tile(4, dst, L);
            tile_regs_release();
            cb_pop_front(cb_limbs, NB);
        }
        cb_push_back(dst, NL);
        cb_pop_front(src, NL);
    };

    // ---- P1: the ACCUMULATING pass. out[j*2+bo] = mma(in[j*2+0], in[j*2+1]; w64[2bo], w64[2bo+1])
    // The two inputs are the two HALVES of the same n1, both owned by this core (see the header).
    auto pass_p1 = [&](uint32_t src, uint32_t dst) {
        cb_wait_front(src, NL);
        cb_wait_front(cb_w64, 4 * NB * NB);
        cb_reserve_back(dst, NL);
        for (uint32_t j = 0; j < 2; ++j) {
            // decompose both partials once; they are reused for bo = 0 and bo = 1
            cb_reserve_back(cb_limbs, 2 * NB);
            decompose_into(src, j * 2 + 0, 0);
            decompose_into(src, j * 2 + 1, 1);
            cb_push_back(cb_limbs, 2 * NB);
            cb_wait_front(cb_limbs, 2 * NB);
            for (uint32_t bo = 0; bo < 2; ++bo) {
                reconfig_data_format(cb_limbs, cb_w64);
                matmul_init(cb_w64, cb_limbs, 0);
                tile_regs_acquire();
                { PH_ZONE("ph_matmul");
                  for (uint32_t p = 0; p < NB; ++p)
                      for (uint32_t a = 0; a < 2; ++a)      // accumulate over BOTH partials
                          for (uint32_t r = 0; r < NB; ++r)
                              matmul_tiles(cb_w64, cb_limbs,
                                           (bo * 2 + a) * NB * NB + p * NB + r, a * NB + r, p); }
                { PH_ZONE("ph_sfpu"); S_recomb(0, 1, 2, 3, 4, K14, QINV, QHIGH, QLOW, Q); }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(dst);
                pack_tile(4, dst, j * 2 + bo);
                tile_regs_release();
            }
            cb_pop_front(cb_limbs, 2 * NB);
        }
        cb_push_back(dst, NL);
        cb_pop_front(src, NL);
    };

    // ---- twiddle: out[L] = a[L] (.) tw[twi(L)], Montgomery, one SFPI pass ----
    // ta is indexed by kh (= L & 1); tb has one tile per local slot, so it is indexed by L.
    auto pass_tw = [&](uint32_t a, uint32_t tw, bool use_ta, uint32_t out) {
        cb_wait_front(a, NL);
        cb_wait_front(tw, use_ta ? 2 : NL);
        cb_reserve_back(out, NL);
        for (uint32_t L = 0; L < NL; ++L) {
            PH_ZONE("ph_sfpu");
            tile_regs_acquire();
            reconfig_data_format_srca(a);
            copy_tile_init(a);  copy_tile(a, L, 0);
            reconfig_data_format_srca(tw);
            copy_tile_init(tw); copy_tile(tw, use_ta ? (L & 1) : L, 1);
            S_all(0, 1, 2, QINV, QHIGH, QLOW, Q);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(out);
            pack_tile(2, out, L);
            tile_regs_release();
        }
        cb_push_back(out, NL);
        cb_pop_front(a, NL);
        // tw is NOT popped: resident, and this kernel uses each twiddle set once anyway.
    };

    // DBG=1: skip every arithmetic phase, so the dispatch is plumbing + the on-chip corner turn and
    // nothing else. Its output must equal the DRAM corner turn's, which separates "the exchange or the
    // page mapping is wrong" from "a compute phase is wrong".
    auto pass_copy = [&](uint32_t src, uint32_t dst) {
        cb_wait_front(src, NL);
        cb_reserve_back(dst, NL);
        for (uint32_t L = 0; L < NL; ++L) {
            tile_regs_acquire();
            reconfig_data_format_srca(src);
            copy_tile_init(src); copy_tile(src, L, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(dst);
            pack_tile(0, dst, L);
            tile_regs_release();
        }
        cb_push_back(dst, NL);
        cb_pop_front(src, NL);
    };
    // DBG=k>=2: run the first (k-1) PRE-corner-turn phases, then hand the result to BOTH the
    // exchange (so BRISC never blocks) and the writer (so the host can read that phase out and
    // compare it against the unfused chain's scratch buffer). cb_d is deliberately left unread.
    auto pass_copy2 = [&](uint32_t src, uint32_t d1, uint32_t d2) {
        cb_wait_front(src, NL);
        cb_reserve_back(d1, NL); cb_reserve_back(d2, NL);
        for (uint32_t L = 0; L < NL; ++L) {
            tile_regs_acquire();
            reconfig_data_format_srca(src);
            copy_tile_init(src); copy_tile(src, L, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(d1); pack_tile(0, d1, L);
            pack_reconfig_data_format(d2); pack_tile(0, d2, L);
            tile_regs_release();
        }
        cb_push_back(d1, NL); cb_push_back(d2, NL);
        cb_pop_front(src, NL);
    };
    if (DBG == 1) { pass_copy(cb_in, cb_x); pass_copy(cb_d, cb_out); return; }
    if (DBG >= 2 && FWD) {
        pass_p1(cb_in, cb_p);                                   if (DBG == 2) { pass_copy2(cb_p, cb_x, cb_out); return; }
        pass_tw(cb_p, cb_ta, true,  cb_q);                      if (DBG == 3) { pass_copy2(cb_q, cb_x, cb_out); return; }
        pass_mm(cb_q, cb_wa, false, cb_p);                      if (DBG == 4) { pass_copy2(cb_p, cb_x, cb_out); return; }
        pass_tw(cb_p, cb_tb, false, cb_x);
        pass_copy(cb_d, cb_out); return;                        // DBG == 5: through the corner turn
    }

    if (FWD) {
        pass_p1(cb_in, cb_p);                       // P1   : acc mma over w64
        pass_tw(cb_p, cb_ta, true,  cb_q);          // TW   : (.) ta[kh]
        pass_mm(cb_q, cb_wa, false, cb_p);          // MM_C : foldR
        pass_tw(cb_p, cb_tb, false, cb_x);          // TW   : (.) tb[L]   -> BRISC takes it
        // <<< BRISC runs the 16-core corner turn and pushes cb_d >>>
        pass_mm(cb_d, cb_wb, true,  cb_out);        // MM_A : foldL
    } else {
        pass_mm(cb_in, cb_wb, true,  cb_x);         // MM_A : foldL       -> BRISC takes it
        // <<< BRISC runs the 16-core corner turn and pushes cb_d >>>
        pass_tw(cb_d, cb_tb, false, cb_p);          // TW   : (.) tb[L]
        pass_mm(cb_p, cb_wa, false, cb_q);          // MM_C : foldR
        pass_tw(cb_q, cb_ta, true,  cb_p);          // TW   : (.) ta[kh]
        pass_p1(cb_p, cb_out);                      // P1   : acc mma over w64
    }
}
