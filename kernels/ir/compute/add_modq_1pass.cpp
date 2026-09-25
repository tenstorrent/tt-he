// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// add_modq_1pass -- add_modq with the whole modular add in ONE SFPI pass and q broadcast in register.
// Drop-in: same runtime args (n_tiles, q), same CBs, same output.  kernels/ir/compute/add_modq.cpp is
// RETAINED and unchanged.
//
// This kernel is the accumulate half of the OnionPIR scan: it runs once per (m-tile, channel, k-tile)
// and folds one partial into the running sum, so at N0=512 it moves ~4.6 GB per query.  Two of the
// three techniques in kernels/OPTIMISED.md apply (there is no multiply here for sfpmul24 to improve):
//
//   2. collapse the SFPI passes  -- 3 tile ops (add/sub/min) -> 1 pass, partials never reach DST
//   3. broadcast scalars in register -- the per-tile fill_tile_int wrote a WHOLE DST tile to hand
//      every lane the same q, 32 times per dispatch; vUInt(Q) costs one instruction
//
// Arithmetic is unchanged.  The original computed min_u32(a+b, a+b-q); with a,b < q < 2^28 the sum is
// under 2^29, so the sign test below selects the same branch: if a+b < q then a+b-q wraps high (which
// is negative read as signed) and a+b is kept, otherwise a+b-q is in [0,q).  This is the same idiom
// the verified face_final/face_recomb passes already use.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifdef TRISC_MATH
namespace {
constexpr uint32_t NV = 32;
inline void face_addmod(uint32_t A, uint32_t B, uint32_t out, uint32_t Q) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt s = vUInt(dst_reg[A * NV + i]) + vUInt(dst_reg[B * NV + i]);
        vUInt q = vUInt(Q);
        vUInt sq = s - q;
        vUInt res = sq;
        v_if(sfpi::reinterpret<sfpi::vInt>(sq) < 0) { res = s; }
        v_endif;
        dst_reg[out * NV + i] = res;
    }
}
}  // namespace
#endif

inline void S_addmod(uint32_t A, uint32_t B, uint32_t out, uint32_t Q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(face_addmod, A, B, out, ckernel::VectorMode::RC, Q));
}

void kernel_main() {
    DeviceZoneScopedN("add_modq");
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t Q = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in0, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_tile_init(cb_in0); copy_tile(cb_in0, 0, 0);     // d0 = a
        copy_tile_init(cb_in1); copy_tile(cb_in1, 0, 1);     // d1 = b
        S_addmod(0, 1, 2, Q);                                // d2 = (a + b) mod q
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(2, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
