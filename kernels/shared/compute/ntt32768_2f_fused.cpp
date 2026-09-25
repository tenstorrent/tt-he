// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ntt32768_2f_fused — the ENTIRE 2-factor (128,256) NTT-32768 in ONE compute kernel, one dispatch,
// with NO DRAM ACCESS DURING THE TRANSFORM. The counterpart of ntt32768_3f_fused.cpp, built so the
// two factorizations can be compared under the same rule: read inputs and constants once at entry,
// write the result once at exit, nothing to memory in between.
//
//   dec       A[me]                       -> 4 int8 limbs        -> handed to BRISC
//   <<< exchange 1: the 8-core mt group swaps limbs >>>
//   M1        C_r = sum_p limb_p @ W256[p,r]      K_acc = NB*Kt1 = 32   -> recombine (bigK) = S
//   twiddle   S (.) Tb                                                   = S'
//   dec       S'                          -> 4 int8 limbs        -> handed to BRISC
//   <<< exchange 2: the 4-core nt group swaps limbs >>>
//   M2        C_p = sum_r W128[p,r] @ limb_r      K_acc = NB*Kt2 = 16   -> recombine (bigK) = R
//
// TWO parks, not one. The 3-factor's triples are tile-local and it parks once, for the corner turn;
// BOTH of the 2-factor's matmuls are non-tile-local (M1 needs 8 input tiles, M2 needs 4 from other
// cores), so this kernel blocks twice on BRISC. That is the factorization's structure, not a
// deficiency of the port -- and it is the honest reason the 2-factor is harder to keep on chip.
//
// Both recombines are the bigK variant (five conditional subtracts): K=256 and K=128 push the plane
// sums to ~2^22, where the 3-factor's K=32 needs only two.
//
// ENGINE-INIT RULE, as in ntt32768_3f_fused.cpp: ONE full init at entry, then only matmul_init /
// reconfig_data_format* / copy_tile_init / pack_reconfig_data_format. A second full init mid-kernel
// deadlocks MATH against PACK on Blackhole (docs/he/blackhole_port.md 2).
//
// Runtime args: 0=Q, 1=QINV, 2=QLOW, 3=QHIGH, 4=K14.

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
        // bigK: both matmuls contract large K (256, 128), so low reaches ~3.2q and R ~4.2q --
        // FIVE conditional subtracts, per recombine_bigK.cpp.
        vUInt low = vUInt(dst_reg[C0 * NV + i]) + (vUInt(dst_reg[C1 * NV + i]) << 7u);
        vUInt r = low + hi14;
        vUInt res = r;
        for (uint32_t k5 = 0; k5 < 5; ++k5) {
            vUInt rq = r - vq; res = rq;
            v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; }
            v_endif;
            r = res;
        }
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
constexpr uint32_t NB = 4, Kt1 = 8, Kt2 = 4;
constexpr auto cb_a   = tt::CBIndex::c_0;    // my A tile
constexpr auto cb_w2  = tt::CBIndex::c_1;    // W256 slice, 128 int8, resident
constexpr auto cb_w1  = tt::CBIndex::c_2;    // W128 slice,  64 int8, resident
constexpr auto cb_tb  = tt::CBIndex::c_3;    // twiddle
constexpr auto cb_my1 = tt::CBIndex::c_4;    // my A limbs   -> BRISC
constexpr auto cb_g1  = tt::CBIndex::c_5;    // gathered     <- BRISC
constexpr auto cb_my2 = tt::CBIndex::c_6;    // my S' limbs  -> BRISC
constexpr auto cb_g2  = tt::CBIndex::c_7;    // gathered     <- BRISC
constexpr auto cb_r   = tt::CBIndex::c_16;   // result
constexpr auto cb_s   = tt::CBIndex::c_18;   // S  (internal)
constexpr auto cb_sp  = tt::CBIndex::c_19;   // S' (internal)
}  // namespace

void kernel_main() {
    DeviceZoneScopedN("ntt2f_fused");
    const uint32_t Q     = get_arg_val<uint32_t>(0);
    const uint32_t QINV  = get_arg_val<uint32_t>(1);
    const uint32_t QLOW  = get_arg_val<uint32_t>(2);
    const uint32_t QHIGH = get_arg_val<uint32_t>(3);
    const uint32_t K14   = get_arg_val<uint32_t>(4);

    unary_op_init_common(cb_a, cb_r);           // THE one full init

    auto decompose_to = [&](uint32_t src, uint32_t dstcb) {
        cb_wait_front(src, 1);            // park outside the zone, matching ntt32768_3f_fused.cpp
        PHASE_ZONE("ph_decompose");
        tile_regs_acquire();
        reconfig_data_format_srca(src);
        copy_tile_init(src); copy_tile(src, 0, 4);
        S_decompose(4, 0, 1, 2, 3);
        tile_regs_commit(); tile_regs_wait();
        cb_reserve_back(dstcb, NB);
        pack_reconfig_data_format(dstcb);
        pack_tile(0, dstcb, 0); pack_tile(1, dstcb, 1);
        pack_tile(2, dstcb, 2); pack_tile(3, dstcb, 3);
        cb_push_back(dstcb, NB);
        tile_regs_release();
        cb_pop_front(src, 1);
    };

    decompose_to(cb_a, cb_my1);                 // ---- dec A ----

    // ---- M1: fold-RHS, plane r accumulates; limbs gathered from the mt group ----
    cb_wait_front(cb_g1, 8 * NB);               // parks here until BRISC's exchange 1 completes
    cb_wait_front(cb_w2, NB * NB * Kt1);
    reconfig_data_format(cb_g1, cb_w2);
    matmul_init(cb_g1, cb_w2, 0);
    cb_reserve_back(cb_s, 1);
    tile_regs_acquire();
    { PHASE_ZONE("ph_matmul");
      for (uint32_t r = 0; r < NB; ++r)
          for (uint32_t p = 0; p < NB; ++p)
              for (uint32_t kt = 0; kt < Kt1; ++kt)
                  matmul_tiles(cb_g1, cb_w2, kt * NB + p, (p * NB + r) * Kt1 + kt, r); }
    { PHASE_ZONE("ph_recombine");
      S_recomb(0, 1, 2, 3, 4, K14, QINV, QHIGH, QLOW, Q); }
    tile_regs_commit(); tile_regs_wait();
    pack_reconfig_data_format(cb_s); pack_tile(4, cb_s);
    cb_push_back(cb_s, 1); tile_regs_release();

    // ---- twiddle ----
    { PHASE_ZONE("ph_twiddle");
    cb_wait_front(cb_s, 1); cb_wait_front(cb_tb, 1);
    cb_reserve_back(cb_sp, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_s);  copy_tile_init(cb_s);  copy_tile(cb_s, 0, 0);
    reconfig_data_format_srca(cb_tb); copy_tile_init(cb_tb); copy_tile(cb_tb, 0, 1);
    S_all(0, 1, 2, QINV, QHIGH, QLOW, Q);
    tile_regs_commit(); tile_regs_wait();
    pack_reconfig_data_format(cb_sp); pack_tile(2, cb_sp);
    cb_push_back(cb_sp, 1); tile_regs_release();
    cb_pop_front(cb_s, 1); cb_pop_front(cb_tb, 1); }

    decompose_to(cb_sp, cb_my2);                // ---- dec S' ----

    // ---- M2: fold-LHS, plane p accumulates; limbs gathered from the nt group ----
    cb_wait_front(cb_g2, 4 * NB);               // parks here until BRISC's exchange 2 completes
    cb_wait_front(cb_w1, NB * NB * Kt2);
    reconfig_data_format(cb_g2, cb_w1);
    matmul_init(cb_w1, cb_g2, 0);
    cb_reserve_back(cb_r, 1);
    tile_regs_acquire();
    { PHASE_ZONE("ph_matmul");
      for (uint32_t p = 0; p < NB; ++p)
          for (uint32_t r = 0; r < NB; ++r)
              for (uint32_t kt = 0; kt < Kt2; ++kt)
                  matmul_tiles(cb_w1, cb_g2, (p * NB + r) * Kt2 + kt, kt * NB + r, p); }
    { PHASE_ZONE("ph_recombine");
      S_recomb(0, 1, 2, 3, 4, K14, QINV, QHIGH, QLOW, Q); }
    tile_regs_commit(); tile_regs_wait();
    pack_reconfig_data_format(cb_r); pack_tile(4, cb_r);
    cb_push_back(cb_r, 1); tile_regs_release();
}
