// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FUSED tensor op: out = c * (d * Rc) mod q, i.e. montify d by Rc then multiply by c,
// in ONE kernel with the intermediate dm staged in an L1 CB (c_2) — NO DRAM round-trip
// (was two separate modmul dispatches, dm written to / read from DRAM between them).
// Reuses the modmul_sfpi_full Montgomery machinery verbatim as a callable do_modmul().
//   phase 1:  dm = d * Rc mod q   (c_1 x c_3) -> c_2
//   phase 2:  out = c * dm mod q  (c_0 x c_2) -> c_16
// Runtime args: 0=n_tiles, 1=q, 2=q_inv_32, 3=q_low16, 4=q_high16.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr auto FMT = DataFormat::UInt32;
inline void FILL(uint32_t d, uint32_t v) { fill_tile_init(); fill_tile_int<FMT>(d, v); }
}  // namespace

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;
inline sfpi::vUInt mul16(sfpi::vUInt x, sfpi::vUInt y) {
    using namespace sfpi;
    vUInt x0 = x & 0xFFu, x1 = x >> 8u, y0 = y & 0xFFu, y1 = y >> 8u;
    vFloat fx0 = int32_to_float(reinterpret<vInt>(x0), RoundMode::NearestEven);
    vFloat fx1 = int32_to_float(reinterpret<vInt>(x1), RoundMode::NearestEven);
    vFloat fy0 = int32_to_float(reinterpret<vInt>(y0), RoundMode::NearestEven);
    vFloat fy1 = int32_to_float(reinterpret<vInt>(y1), RoundMode::NearestEven);
    vUInt p00 = float_to_uint16(fx0 * fy0, RoundMode::NearestEven);
    vUInt p01 = float_to_uint16(fx0 * fy1, RoundMode::NearestEven);
    vUInt p10 = float_to_uint16(fx1 * fy0, RoundMode::NearestEven);
    vUInt p11 = float_to_uint16(fx1 * fy1, RoundMode::NearestEven);
    vUInt mid = p01 + p10;
    return p00 + (mid << 8u) + (p11 << 16u);
}
inline void face_mul(uint32_t a, uint32_t b, uint32_t o) {
    using namespace sfpi;
    for (uint32_t i=0;i<8;i++) dst_reg[o*NV+i] = mul16(vUInt(dst_reg[a*NV+i]), vUInt(dst_reg[b*NV+i]));
}
inline void face_mullo_a(uint32_t x, uint32_t y, uint32_t s) {
    using namespace sfpi;
    for (uint32_t i=0;i<8;i++){ vUInt vx=dst_reg[x*NV+i], vy=dst_reg[y*NV+i];
        dst_reg[s*NV+i] = mul16(vx&0xFFFFu, vy>>16u) + mul16(vx>>16u, vy&0xFFFFu); }
}
inline void face_mullo_b(uint32_t x, uint32_t y, uint32_t s, uint32_t o) {
    using namespace sfpi;
    for (uint32_t i=0;i<8;i++){ vUInt lolo=mul16(vUInt(dst_reg[x*NV+i])&0xFFFFu, vUInt(dst_reg[y*NV+i])&0xFFFFu);
        dst_reg[o*NV+i] = lolo + (vUInt(dst_reg[s*NV+i])<<16u); }
}
inline void face_limbs(uint32_t a, uint32_t b, uint32_t aL, uint32_t aH, uint32_t bL, uint32_t bH) {
    for (uint32_t i=0;i<8;i++){ sfpi::vUInt va=sfpi::dst_reg[a*NV+i], vb=sfpi::dst_reg[b*NV+i];
        sfpi::dst_reg[aL*NV+i]=va&0xFFFFu; sfpi::dst_reg[aH*NV+i]=va>>16u;
        sfpi::dst_reg[bL*NV+i]=vb&0xFFFFu; sfpi::dst_reg[bH*NV+i]=vb>>16u; }
}
inline void face_lohi(uint32_t ll, uint32_t hh, uint32_t c1, uint32_t c2, uint32_t lo, uint32_t hi) {
    for (uint32_t i=0;i<8;i++){ sfpi::vUInt vll=sfpi::dst_reg[ll*NV+i], vhh=sfpi::dst_reg[hh*NV+i];
        sfpi::vUInt cross=sfpi::vUInt(sfpi::dst_reg[c1*NV+i])+sfpi::vUInt(sfpi::dst_reg[c2*NV+i]);
        sfpi::vUInt mid=(vll>>16u)+(cross&0xFFFFu);
        sfpi::dst_reg[lo*NV+i]=(vll&0xFFFFu)|(mid<<16u);
        sfpi::dst_reg[hi*NV+i]=vhh+(cross>>16u)+(mid>>16u); }
}
inline void face_split(uint32_t t, uint32_t tL, uint32_t tH) {
    for (uint32_t i=0;i<8;i++){ sfpi::vUInt vt=sfpi::dst_reg[t*NV+i];
        sfpi::dst_reg[tL*NV+i]=vt&0xFFFFu; sfpi::dst_reg[tH*NV+i]=vt>>16u; }
}
inline void face_reduce(uint32_t pHH, uint32_t pMH, uint32_t pML, uint32_t pLL, uint32_t hi, uint32_t qIdx, uint32_t out) {
    for (uint32_t i=0;i<8;i++){ sfpi::vUInt vpMH=sfpi::dst_reg[pMH*NV+i], vpML=sfpi::dst_reg[pML*NV+i], vpLL=sfpi::dst_reg[pLL*NV+i];
        sfpi::vUInt mid_low=(vpMH&0xFFFFu)+(vpML&0xFFFFu)+(vpLL>>16u);
        sfpi::vUInt mid_high=(vpMH>>16u)+(vpML>>16u)+(mid_low>>16u);
        sfpi::vUInt t_final=sfpi::vUInt(sfpi::dst_reg[pHH*NV+i])+mid_high;
        sfpi::vUInt q=sfpi::dst_reg[qIdx*NV+i];
        sfpi::vUInt b=sfpi::vUInt(sfpi::dst_reg[hi*NV+i])+q-t_final;
        sfpi::vUInt bq=b-q, res=bq;
        v_if(sfpi::reinterpret<sfpi::vInt>(bq)<0){ res=b; } v_endif;
        sfpi::dst_reg[out*NV+i]=res; }
}
}  // namespace
#endif

inline void S_mul(uint32_t a,uint32_t b,uint32_t o){ MATH(_llk_math_eltwise_binary_sfpu_params_(face_mul,a,b,o,ckernel::VectorMode::RC)); }
inline void S_mullo_a(uint32_t x,uint32_t y,uint32_t s){ MATH(_llk_math_eltwise_binary_sfpu_params_(face_mullo_a,x,y,s,ckernel::VectorMode::RC)); }
inline void S_mullo_b(uint32_t x,uint32_t y,uint32_t s,uint32_t o){ MATH(_llk_math_eltwise_binary_sfpu_params_(face_mullo_b,x,y,s,ckernel::VectorMode::RC,o)); }
inline void S_limbs(uint32_t a,uint32_t b,uint32_t aL,uint32_t aH,uint32_t bL,uint32_t bH){ MATH(_llk_math_eltwise_binary_sfpu_params_(face_limbs,a,b,aL,ckernel::VectorMode::RC,aH,bL,bH)); }
inline void S_lohi(uint32_t ll,uint32_t hh,uint32_t c1,uint32_t c2,uint32_t lo,uint32_t hi){ MATH(_llk_math_eltwise_binary_sfpu_params_(face_lohi,ll,hh,c1,ckernel::VectorMode::RC,c2,lo,hi)); }
inline void S_split(uint32_t t,uint32_t tL,uint32_t tH){ MATH(_llk_math_eltwise_binary_sfpu_params_(face_split,t,tL,tH,ckernel::VectorMode::RC)); }
inline void S_reduce(uint32_t pHH,uint32_t pMH,uint32_t pML,uint32_t pLL,uint32_t hi,uint32_t qIdx,uint32_t out){ MATH(_llk_math_eltwise_binary_sfpu_params_(face_reduce,pHH,pMH,pML,ckernel::VectorMode::RC,pLL,hi,qIdx,out)); }

// a in d0, b in d1 -> result (a*b mod q) in d4. Clobbers d0..d7.
inline void do_modmul(uint32_t Q, uint32_t QINV, uint32_t QLOW, uint32_t QHIGH) {
    S_limbs(0,1,2,3,4,5);
    S_mul(2,4,0); S_mul(3,5,1); S_mul(2,5,6); S_mul(3,4,7);
    S_lohi(0,1,6,7,2,3);
    FILL(0,QINV); S_mullo_a(2,0,4); S_mullo_b(2,0,4,2); S_split(2,4,5);
    FILL(0,QHIGH); FILL(1,QLOW);
    S_mul(5,0,6); S_mul(5,1,5); S_mul(4,0,0); S_mul(4,1,1);
    FILL(2,Q); S_reduce(6,5,0,1,3,2,4);
}

void kernel_main() {
    DeviceZoneScopedN("tensor2_fused");
    const uint32_t n = get_arg_val<uint32_t>(0), Q = get_arg_val<uint32_t>(1),
                   QINV = get_arg_val<uint32_t>(2), QLOW = get_arg_val<uint32_t>(3), QHIGH = get_arg_val<uint32_t>(4);
    constexpr auto c_c = tt::CBIndex::c_0, c_d = tt::CBIndex::c_1, c_dm = tt::CBIndex::c_2,
                   c_rc = tt::CBIndex::c_3, c_out = tt::CBIndex::c_16;
    unary_op_init_common(c_c, c_out);
    for (uint32_t t = 0; t < n; ++t) {
        // phase 1: dm = d * Rc -> c_dm (stays in L1)
        cb_wait_front(c_d, 1); cb_wait_front(c_rc, 1); cb_reserve_back(c_dm, 1);
        tile_regs_acquire();
        copy_tile_init(c_d);  copy_tile(c_d, 0, 0);      // d0 = d
        copy_tile_init(c_rc); copy_tile(c_rc, 0, 1);     // d1 = Rc
        do_modmul(Q, QINV, QLOW, QHIGH);                 // d4 = d*Rc mod q
        tile_regs_commit(); tile_regs_wait(); pack_tile(4, c_dm); tile_regs_release();
        cb_push_back(c_dm, 1); cb_pop_front(c_d, 1); cb_pop_front(c_rc, 1);
        // phase 2: out = c * dm -> c_out
        cb_wait_front(c_c, 1); cb_wait_front(c_dm, 1); cb_reserve_back(c_out, 1);
        tile_regs_acquire();
        copy_tile_init(c_c);  copy_tile(c_c, 0, 0);      // d0 = c
        copy_tile_init(c_dm); copy_tile(c_dm, 0, 1);     // d1 = dm
        do_modmul(Q, QINV, QLOW, QHIGH);                 // d4 = c*dm mod q
        tile_regs_commit(); tile_regs_wait(); pack_tile(4, c_out); tile_regs_release();
        cb_push_back(c_out, 1); cb_pop_front(c_c, 1); cb_pop_front(c_dm, 1);
    }
}
