// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ntt32768_3f_fused — the ENTIRE 3-factor NTT-32768 as ONE compute kernel, one dispatch.
//
//   A triple:  decompose -> matmul(WfoldA, fold-LHS) -> recombine   = G
//   B twiddle: G  (.) Ta                                            = G'
//   C triple:  decompose -> matmul(Wrhs,  fold-RHS) -> recombine    = H
//   TB twiddle:H  (.) Tb                                            = Cb   -> handed to BRISC
//   <<< BRISC performs the all-to-all corner turn and hands back D >>>
//   D triple:  decompose -> matmul(WfoldA, fold-LHS) -> recombine   = E
//
// EVERY intermediate stays in L1.  The limbs (int8), the four int32 planes, and G/G'/H/Cb/D never
// touch DRAM: the kernel reads Xin + the folded constants once and writes E once.  See
// docs/matmul/exp_ntt001.md -- the six-dispatch version still round-tripped a tile through DRAM at
// each of its five stage boundaries (~40 KB/core).
//
// THE ENGINE-INIT RULE, which is what makes this kernel dangerous.  `unary_op_init_common` and
// `mm_init` both re-run the DEST-sync handshake, and re-running that mid-kernel deadlocks MATH
// against PACK on Blackhole -- the failure that "hung every NTT-based binary" (diagnosis in
// docs/he/blackhole_port.md 2).  fused_matmul_a_1pass alternates engines ONCE and already has to obey
// this; this kernel alternates SFPU <-> matrix engine EIGHT times, so the rule is absolute:
//   * exactly ONE full init, at entry;
//   * per phase only the light-weight switches -- matmul_init, reconfig_data_format*,
//     copy_tile_init, pack_reconfig_data_format. Never a second full init.
//
// THE MID-KERNEL WAIT.  After TB this kernel blocks on cb_d, which BRISC fills only after a 32-core
// barrier. So a compute kernel here waits on a global exchange. If any core fails to reach its
// cb_cb push, every core hangs -- the same all-or-nothing property all_to_all_corner_x.cpp already
// has, now with a compute kernel inside the loop.
//
// Runtime args: 0=Q, 1=QINV, 2=QLOW, 3=QHIGH, 4=K14 (= mont(2^14)).

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

// NTT_PHASE_ZONES: optional inner zones so decompose / matmul+recombine / twiddle can be attributed
// separately inside this one dispatch. OFF by default -- kernels/experiments/exp_tb001/README.md records that the
// extra marker volume can overflow the profiler's per-RISC DRAM buffer and break the 1-1 dispatch map.
// At C=32 with one dispatch the volume is small, but the flag stays opt-in for that reason.
#ifdef NTT_PHASE_ZONES
#define PHASE_ZONE(name) DeviceZoneScopedN(name)
#else
#define PHASE_ZONE(name)
#endif

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;

inline sfpi::vUInt mul16(sfpi::vUInt x, sfpi::vUInt y) {
    using namespace sfpi;
    vUInt lo = vUInt(fractional_mul(x, y, FractionalHalf::Low));
    vUInt hi = vUInt(fractional_mul(x, y, FractionalHalf::High));
    return lo + (hi << 23u);
}

// radix-128 split, from fused_matmul_a_1pass.cpp
inline void face_decompose(uint32_t v, uint32_t l0, uint32_t l1, uint32_t l2, uint32_t l3) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt x = sfpi::dst_reg[v * NV + i];
        sfpi::dst_reg[l0 * NV + i] = x & 127u;
        sfpi::dst_reg[l1 * NV + i] = (x >> 7u) & 127u;
        sfpi::dst_reg[l2 * NV + i] = (x >> 14u) & 127u;
        sfpi::dst_reg[l3 * NV + i] = (x >> 21u) & 127u;
    }
}

// the whole recombine in one SFPI pass, from fused_matmul_a_1pass.cpp (unchanged)
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

// the whole Montgomery modmul in one SFPI pass, from modmul_sfpi_1pass.cpp (unchanged)
inline void face_all(uint32_t a, uint32_t b, uint32_t out,
                     uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt va = dst_reg[a * NV + i], vb = dst_reg[b * NV + i];
        vUInt aL = va & 0xFFFFu, aH = va >> 16u;
        vUInt bL = vb & 0xFFFFu, bH = vb >> 16u;
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
        vUInt bb = hi + vq - t_final;
        vUInt bq = bb - vq;
        vUInt res = bq;
        v_if(sfpi::reinterpret<sfpi::vInt>(bq) < 0) { res = bb; }
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
inline void S_all(uint32_t a, uint32_t b, uint32_t out,
                  uint32_t QINV, uint32_t QHIGH, uint32_t QLOW, uint32_t Q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_all, a, b, out, ckernel::VectorMode::RC,
                                               QINV, QHIGH, QLOW, Q));
}

namespace {
constexpr uint32_t NB = 4;
constexpr auto cb_x     = tt::CBIndex::c_0;    // Xin (my cube tile)
constexpr auto cb_wa    = tt::CBIndex::c_1;    // WfoldA, 16 int8, RESIDENT (never popped)
constexpr auto cb_wc    = tt::CBIndex::c_2;    // Wrhs,   16 int8, RESIDENT
constexpr auto cb_ta    = tt::CBIndex::c_3;    // twiddle a
constexpr auto cb_tb    = tt::CBIndex::c_4;    // twiddle b
constexpr auto cb_limbs = tt::CBIndex::c_5;    // 4 int8 limbs, scratch, reused by all three triples
constexpr auto cb_d     = tt::CBIndex::c_6;    // D, filled by BRISC after the corner turn
constexpr auto cb_cb    = tt::CBIndex::c_16;   // Cb, handed to BRISC
constexpr auto cb_e     = tt::CBIndex::c_17;   // E, the result
constexpr auto cb_g     = tt::CBIndex::c_18;   // internal: G
constexpr auto cb_gp    = tt::CBIndex::c_19;   // internal: G'
constexpr auto cb_h     = tt::CBIndex::c_20;   // internal: H
}  // namespace

void kernel_main() {
    DeviceZoneScopedN("ntt3f_fused");
    const uint32_t Q     = get_arg_val<uint32_t>(0);
    const uint32_t QINV  = get_arg_val<uint32_t>(1);
    const uint32_t QLOW  = get_arg_val<uint32_t>(2);
    const uint32_t QHIGH = get_arg_val<uint32_t>(3);
    const uint32_t K14   = get_arg_val<uint32_t>(4);

    // THE one full init. Everything after this is a light-weight switch.
    unary_op_init_common(cb_x, cb_cb);

    // --- decompose src -> 4 int8 limbs in cb_limbs ---
    auto phase_decompose = [&](uint32_t src) {
        cb_wait_front(src, 1);            // park (on the reader, or on BRISC's exchange) is OUTSIDE
        PHASE_ZONE("ph_decompose");       // the zone, so this row measures decomposing only
        tile_regs_acquire();
        reconfig_data_format_srca(src);
        copy_tile_init(src); copy_tile(src, 0, 4);
        S_decompose(4, 0, 1, 2, 3);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_limbs, NB);
        pack_reconfig_data_format(cb_limbs);
        pack_tile(0, cb_limbs, 0); pack_tile(1, cb_limbs, 1);
        pack_tile(2, cb_limbs, 2); pack_tile(3, cb_limbs, 3);
        cb_push_back(cb_limbs, NB);
        tile_regs_release();
        cb_pop_front(src, 1);
    };

    // --- matmul(W resident) + recombine, planes never leaving DST -> dstcb ---
    // foldA: C_p = sum_r W[p*NB+r] @ limb_r     foldC: C_r = sum_p limb_p @ W[p*NB+r]
    auto phase_mm_recomb = [&](uint32_t w, bool foldA, uint32_t dstcb) {
        PHASE_ZONE("ph_matmul_recomb");
        cb_wait_front(cb_limbs, NB);
        cb_wait_front(w, NB * NB);
        reconfig_data_format(cb_limbs, w);
        matmul_init(w, cb_limbs, 0);
        cb_reserve_back(dstcb, 1);
        tile_regs_acquire();
        { PHASE_ZONE("ph_matmul");
          for (uint32_t p = 0; p < NB; ++p) {
              for (uint32_t r = 0; r < NB; ++r) {
                  if (foldA) matmul_tiles(w, cb_limbs, p * NB + r, r, p);
                  else       matmul_tiles(cb_limbs, w, p, p * NB + r, r);
              }
          } }
        { PHASE_ZONE("ph_recombine");
          S_recomb(0, 1, 2, 3, 4, K14, QINV, QHIGH, QLOW, Q); }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(dstcb);
        pack_tile(4, dstcb);
        cb_push_back(dstcb, 1);
        tile_regs_release();
        cb_pop_front(cb_limbs, NB);              // W is NOT popped: resident for the next triple
    };

    // --- twiddle: out = a (.) tw, Montgomery, one SFPI pass ---
    auto phase_twiddle = [&](uint32_t a, uint32_t tw, uint32_t out) {
        PHASE_ZONE("ph_twiddle");
        cb_wait_front(a, 1);
        cb_wait_front(tw, 1);
        cb_reserve_back(out, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(a);
        copy_tile_init(a);  copy_tile(a, 0, 0);
        reconfig_data_format_srca(tw);
        copy_tile_init(tw); copy_tile(tw, 0, 1);
        S_all(0, 1, 2, QINV, QHIGH, QLOW, Q);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(out);
        pack_tile(2, out);
        cb_push_back(out, 1);
        tile_regs_release();
        cb_pop_front(a, 1);
        cb_pop_front(tw, 1);
    };

    phase_decompose(cb_x);                       // ---- A ----
    phase_mm_recomb(cb_wa, true,  cb_g);
    phase_twiddle(cb_g, cb_ta, cb_gp);           // ---- B ----
    phase_decompose(cb_gp);                      // ---- C ----
    phase_mm_recomb(cb_wc, false, cb_h);
    phase_twiddle(cb_h, cb_tb, cb_cb);           // ---- TB ----  BRISC takes it from here

    // BRISC now runs the all-to-all corner turn across all 32 cores and pushes cb_d.
    phase_decompose(cb_d);                       // ---- D ----
    phase_mm_recomb(cb_wa, true,  cb_e);
}
