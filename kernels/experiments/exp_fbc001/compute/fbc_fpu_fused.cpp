// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// fbc_fpu_fused — the WHOLE approximate FBC on the MATRIX ENGINE in ONE dispatch, no DRAM access
// during the computation.  exp_fbc001; derived from exp_tb001/compute/fbc_fpu_tr.cpp (retained).
//
// WHAT CHANGED vs fbc_fpu_tr.  That kernel started from Y' -- i.e. stage 1 (y_i = x_i.C1[i] mod q_i)
// ran as a SEPARATE dispatch and handed Y over through DRAM: Ls tiles out and Ls tiles back per core,
// 128 KB each way at Ls=32.  Here stage 1 is phase 0 of this kernel, so the y buffer does not exist.
//
// The one difficulty, and why it is cheap to solve.  The transpose is free only because it is fused
// into the READ (reader_fbc_tr_fused.cpp gathers it as 2 x 64 B NoC copies per (group, limb)), so
// anything before the transpose would have to precede the reader -- impossible in one dispatch.  So
// the scale runs AFTER the transpose, where its constants vary along the ROW axis (row i = source
// limb) instead of being uniform per tile.  That is exactly the shape the recombine in phase 3
// already had ("PER-ROW modulus"), and it reads all of q, qinv, qhigh, qlow from parameter TILES.
// Phase 0 therefore reuses that machinery verbatim with a second set of 5 tiles for the SOURCE
// moduli, and cb_par grows 5 -> 10.
//
// Phase 0 and phase 1 share ONE tile_regs_acquire: the mulmod leaves y in DST 4 and S_decompose reads
// DST 4, so y never even reaches a CB.
//
// Original header follows.
// ----------------------------------------------------------------------------
// FBC on the MATRIX ENGINE with the limb<->coefficient transpose FUSED INTO THE I/O.  exp_Tg018r.
//
// The matrix engine contracts only WITHIN a tile, and the axis being contracted here is the source-
// limb axis -- which in exp_Tg018's limb-major storage is a TILE axis.  The usual reading of that is
// "you need an on-device all-to-all transpose first".  You do not, if you transpose the OPERANDS
// instead of the data:
//
//     limb-major:   plane [c][j] = SUM_lane A[c][lane] . B[lane][j]      needs A coefficient-major
//     this kernel:  plane'[j][c] = SUM_lane Bt[j][lane] . At[lane][c]    needs At LIMB-major
//
// and At limb-major is exactly what a strided read of the existing tiles produces: row `lane` of At
// is a 32-coefficient run of one source residue.  So the transpose is nothing but NoC addressing --
// reader_fbc_tr.cpp gathers it on the way in, writer_fbc_tr.cpp scatters it on the way out, and no
// transpose op, no extra pass and no DRAM round-trip appear anywhere.  The planes stay in CBs.
//
//   phase 0  scale:      Y'[i][c] = X'[i][c] . C1[i] mod q_i    (per-ROW source modulus)   [SFPU]
//   phase 1  decompose:  Y'[i][c] (int32) -> 4 int8 radix-128 limb planes At[p][i][c]   [SFPU]
//   phase 2  matmul:     plane'[rr] = SUM_p Bt[rr][p] @ At[p]                    [matrix engine]
//   phase 3  recombine:  out'[j][c] = (SUM_rr 128^rr . plane'[rr]) mod p_j              [SFPU]
//
// LANE ORDER is p-major: lane = p*32 + i, so At[p] is limb_p of Y' elementwise -- no lane
// permutation anywhere.  It costs 4 A-tiles even when Ls < 32 (rows >= Ls are ignored because the
// host zeroes the matching Bt columns); at Ls=32 that is exactly optimal.
//
// PER-ROW modulus: j is the ROW index of the output here, not the column, so the 5 recombine
// parameter tiles are built row-wise on the host.  The kernel is recombine_rns's, unchanged.
//
// Engine switching follows fused_matmul_a.cpp: ONE full init at entry, then only light-weight
// matmul_init / reconfig_data_format per phase.  Re-running unary_op_init_common or mm_init inside
// the loop re-does the DEST-sync handshake and deadlocks MATH against PACK on Blackhole.
//
// c_0  X' int32, one tile per coefficient-group      c_1  Bt int8, NR*4 tiles, resident
// c_2  At int8, 4 tiles                              c_3  plane' int32, 4 tiles
// c_4  10 param tiles, resident: 0-4 destination (mod p_j, per output row j) as before,
//      5-9 source (C1, QSINV, QSHIGH, QSLOW, QS; mod q_i, per input row i) for phase 0
// c_16 out' int32, one tile per group
// Runtime args: 0 = ngroups (32 per 1024-coefficient tile).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/mul_int_sfpu.h"
#include "hostdevcommon/kernel_structs.h"

namespace {
constexpr auto FMT = DataFormat::UInt32;
inline void MUL_INIT() { mul_int_tile_init<FMT>(); }
inline void MUL(uint32_t a, uint32_t b, uint32_t o) { mul_int_tile<FMT>(a, b, o); }
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
    for (uint32_t i = 0; i < 8; i++)
        sfpi::dst_reg[out * NV + i] = sfpi::vUInt(sfpi::dst_reg[C2 * NV + i]) + (sfpi::vUInt(sfpi::dst_reg[C3 * NV + i]) << 7u);
}
inline void face_limbs(uint32_t a, uint32_t b, uint32_t aL, uint32_t aH, uint32_t bL, uint32_t bH) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt va = sfpi::dst_reg[a * NV + i], vb = sfpi::dst_reg[b * NV + i];
        sfpi::dst_reg[aL * NV + i] = va & 0xFFFFu; sfpi::dst_reg[aH * NV + i] = va >> 16u;
        sfpi::dst_reg[bL * NV + i] = vb & 0xFFFFu; sfpi::dst_reg[bH * NV + i] = vb >> 16u;
    }
}
inline void face_lohi(uint32_t ll, uint32_t hh, uint32_t c1, uint32_t c2, uint32_t lo, uint32_t hi) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vll = sfpi::dst_reg[ll * NV + i], vhh = sfpi::dst_reg[hh * NV + i];
        sfpi::vUInt cross = sfpi::vUInt(sfpi::dst_reg[c1 * NV + i]) + sfpi::vUInt(sfpi::dst_reg[c2 * NV + i]);
        sfpi::vUInt mid = (vll >> 16u) + (cross & 0xFFFFu);
        sfpi::dst_reg[lo * NV + i] = (vll & 0xFFFFu) | (mid << 16u);
        sfpi::dst_reg[hi * NV + i] = vhh + (cross >> 16u) + (mid >> 16u);
    }
}
inline void face_split(uint32_t t, uint32_t tL, uint32_t tH) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vt = sfpi::dst_reg[t * NV + i];
        sfpi::dst_reg[tL * NV + i] = vt & 0xFFFFu; sfpi::dst_reg[tH * NV + i] = vt >> 16u;
    }
}
inline void face_reduce_fin(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL, uint32_t hi, uint32_t qIdx, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt vpMH = sfpi::dst_reg[pMH * NV + i], vpML = sfpi::dst_reg[pML * NV + i], vpLL = sfpi::dst_reg[pLL * NV + i];
        sfpi::vUInt mid_low = (vpMH & 0xFFFFu) + (vpML & 0xFFFFu) + (vpLL >> 16u);
        sfpi::vUInt mid_high = (vpMH >> 16u) + (vpML >> 16u) + (mid_low >> 16u);
        sfpi::vUInt t_final = sfpi::vUInt(sfpi::dst_reg[pHH * NV + i]) + mid_high;
        sfpi::vUInt q = sfpi::dst_reg[qIdx * NV + i];
        sfpi::vUInt b = sfpi::vUInt(sfpi::dst_reg[hi * NV + i]) + q - t_final;
        sfpi::vUInt bq = b - q, res = bq;
        v_if(sfpi::reinterpret<sfpi::vInt>(bq) < 0) { res = b; } v_endif;
        sfpi::dst_reg[out * NV + i] = res;
    }
}
inline void face_final(uint32_t C0, uint32_t C1, uint32_t hi14, uint32_t qIdx, uint32_t out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vUInt q = sfpi::dst_reg[qIdx * NV + i];
        sfpi::vUInt low = sfpi::vUInt(sfpi::dst_reg[C0 * NV + i]) + (sfpi::vUInt(sfpi::dst_reg[C1 * NV + i]) << 7u);
        sfpi::vUInt r = low + sfpi::vUInt(sfpi::dst_reg[hi14 * NV + i]);
        sfpi::vUInt rq = r - q, res = rq;
        v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; } v_endif;
        r = res; rq = r - q; res = rq;
        v_if(sfpi::reinterpret<sfpi::vInt>(rq) < 0) { res = r; } v_endif;
        sfpi::dst_reg[out * NV + i] = res;
    }
}
}  // namespace
#endif

inline void S_decompose(uint32_t v, uint32_t l0, uint32_t l1, uint32_t l2, uint32_t l3) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_decompose, v, l0, l1, ckernel::VectorMode::RC, l2, l3)); }
inline void S_build_hi(uint32_t C2, uint32_t C3, uint32_t o) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_build_hi, C2, C3, o, ckernel::VectorMode::RC)); }
inline void S_limbs(uint32_t a, uint32_t b, uint32_t aL, uint32_t aH, uint32_t bL, uint32_t bH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_limbs, a, b, aL, ckernel::VectorMode::RC, aH, bL, bH)); }
inline void S_lohi(uint32_t ll, uint32_t hh, uint32_t c1, uint32_t c2, uint32_t lo, uint32_t hi) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_lohi, ll, hh, c1, ckernel::VectorMode::RC, c2, lo, hi)); }
inline void S_split(uint32_t t, uint32_t tL, uint32_t tH) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_split, t, tL, tH, ckernel::VectorMode::RC)); }
inline void S_reduce_fin(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL, uint32_t hi, uint32_t qIdx, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_reduce_fin, pHH, pMH, pML, ckernel::VectorMode::RC, pLL, hi, qIdx, out)); }
inline void S_final(uint32_t C0, uint32_t C1, uint32_t hi14, uint32_t qIdx, uint32_t out) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_final, C0, C1, hi14, ckernel::VectorMode::RC, qIdx, out)); }

void kernel_main() {
    const uint32_t ngroups = get_arg_val<uint32_t>(0);

    constexpr auto cb_x = tt::CBIndex::c_0;      // X'[i][c], int32, one per coefficient-group
    constexpr auto cb_b = tt::CBIndex::c_1;      // Bt, int8, NR*NP tiles, resident
    constexpr auto cb_a = tt::CBIndex::c_2;      // At, int8, NP tiles
    constexpr auto cb_pl = tt::CBIndex::c_3;     // plane', int32, NR tiles
    constexpr auto cb_par = tt::CBIndex::c_4;    // K14,QINV,QHIGH,QLOW,Q  (per ROW j)
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t NR = 4, NP = 4;           // radix planes out, radix limbs in
    constexpr uint32_t PK14 = 0, PQINV = 1, PQHIGH = 2, PQLOW = 3, PQ = 4;         // destination, per row j
    constexpr uint32_t PC1 = 5, PSQINV = 6, PSQHIGH = 7, PSQLOW = 8, PSQ = 9;      // source, per row i
    constexpr uint32_t NPAR = 10;

    unary_op_init_common(cb_x, cb_out);          // the ONE full init -- see header
    MUL_INIT();

    cb_wait_front(cb_par, NPAR);
    cb_wait_front(cb_b, NR * NP);

    for (uint32_t g = 0; g < ngroups; ++g) {
        // ---- phase 0 + 1 FUSED: Y' = X'.C1 mod q_i (per-ROW), then decompose ----
        // The mulmod is phase 3's sequence with the SOURCE parameter tiles; it leaves y in DST 4,
        // which is exactly where S_decompose reads from, so y never reaches a CB.
        reconfig_data_format_srca(cb_x);
        pack_reconfig_data_format(cb_a);
        MUL_INIT();
        cb_wait_front(cb_x, 1);
        tile_regs_acquire();
        copy_tile_init(cb_x);   copy_tile(cb_x, 0, 0);            // d0 = X'
        copy_tile_init(cb_par); copy_tile(cb_par, PC1, 1);        // d1 = mont(C1[i]), per row
        S_limbs(0, 1, 2, 3, 4, 5);
        MUL(2, 4, 0); MUL(3, 5, 1); MUL(2, 5, 6); MUL(3, 4, 7);
        S_lohi(0, 1, 6, 7, 2, 3);                                 // d2 = lo, d3 = hi
        copy_tile_init(cb_par); copy_tile(cb_par, PSQINV, 0); MUL(2, 0, 2);   // t = lo . qinv_i
        S_split(2, 4, 5);
        copy_tile(cb_par, PSQHIGH, 0); copy_tile(cb_par, PSQLOW, 1);
        MUL(5, 0, 6); MUL(5, 1, 5); MUL(4, 0, 0); MUL(4, 1, 1);
        copy_tile(cb_par, PSQ, 2);
        S_reduce_fin(6, 5, 0, 1, 3, 2, 4);                        // d4 = y, canonical in [0, q_i)
        S_decompose(4, 0, 1, 2, 3);                               // d0..d3 = radix-128 limbs of y
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_a, NP);
        pack_tile(0, cb_a, 0); pack_tile(1, cb_a, 1); pack_tile(2, cb_a, 2); pack_tile(3, cb_a, 3);
        cb_push_back(cb_a, NP);
        tile_regs_release();
        cb_pop_front(cb_x, 1);

        // ---- phase 2: plane'[rr] = SUM_p Bt[rr*NP+p] @ At[p] ----
        reconfig_data_format(cb_a, cb_b);
        pack_reconfig_data_format(cb_pl);
        matmul_init(cb_b, cb_a, 0 /*transpose*/);
        cb_wait_front(cb_a, NP);
        cb_reserve_back(cb_pl, NR);
        for (uint32_t rr = 0; rr < NR; ++rr) {
            tile_regs_acquire();
            for (uint32_t p = 0; p < NP; ++p) matmul_tiles(cb_b, cb_a, rr * NP + p, p, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_pl, rr);
            tile_regs_release();
        }
        cb_push_back(cb_pl, NR);
        cb_pop_front(cb_a, NP);

        // ---- phase 3: recombine, per-ROW modulus ----
        reconfig_data_format_srca(cb_pl);
        pack_reconfig_data_format(cb_out);
        MUL_INIT();
        cb_wait_front(cb_pl, NR);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        copy_tile_init(cb_pl); copy_tile(cb_pl, 2, 2); copy_tile(cb_pl, 3, 3);
        S_build_hi(2, 3, 0);
        copy_tile_init(cb_par); copy_tile(cb_par, PK14, 1);
        S_limbs(0, 1, 2, 3, 4, 5);
        MUL(2, 4, 0); MUL(3, 5, 1); MUL(2, 5, 6); MUL(3, 4, 7);
        S_lohi(0, 1, 6, 7, 2, 3);
        copy_tile_init(cb_par); copy_tile(cb_par, PQINV, 0); MUL(2, 0, 2);
        S_split(2, 4, 5);
        copy_tile(cb_par, PQHIGH, 0); copy_tile(cb_par, PQLOW, 1);
        MUL(5, 0, 6); MUL(5, 1, 5); MUL(4, 0, 0); MUL(4, 1, 1);
        copy_tile(cb_par, PQ, 2);
        S_reduce_fin(6, 5, 0, 1, 3, 2, 4);
        copy_tile_init(cb_pl); copy_tile(cb_pl, 0, 0); copy_tile(cb_pl, 1, 1);
        S_final(0, 1, 4, 2, 5);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(5, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_pl, NR);
    }
}
