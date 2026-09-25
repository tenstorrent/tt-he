// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// noc_bw_probe — what can ONE core actually push across the fabric, and what limits it?
//
// exp_noc001's exchange reached ~5.9 GB/s per core, which is ~14% of a Tensix NoC port's ~32 B/clk
// injection rate.  That gap is the thing to explain: either the fabric cannot be driven harder from
// one RISC, or the exchange is paying per-REQUEST cost rather than per-BYTE cost.
//
// This separates them.  Each core writes a FIXED total (TOT bytes) into one peer's L1 in chunks of
// MSG bytes, in a ring (me -> me+1), so there is no hotspot, no barrier, and no DRAM.  Sweeping MSG
// at constant TOT changes ONLY the request count.  If time is flat in MSG the fabric is the limit;
// if time falls as MSG grows, the cost is per-request and the model is
//
//     time = (TOT/MSG) * issue_cycles  +  TOT / wire_bytes_per_cycle
//
// whose intercept is the issue overhead we are paying 32 times per exchange.
//
// This computes no permutation and is never gated for correctness -- it is a machine-limit probe,
// the NoC analogue of docs/vector/exp_bw001.md.  Its output is a rate, not a result.
//
// Source and destination are the CB write pointers, not runtime args: every core runs an identical
// program, so my c_1 pointer IS the peer's c_1 pointer.
//
// Runtime args: 0,1 unused, 2=peer_x, 3=peer_y, 4=TOT bytes, 5=MSG bytes
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t px   = get_arg_val<uint32_t>(2);
    const uint32_t py   = get_arg_val<uint32_t>(3);
    const uint32_t TOT  = get_arg_val<uint32_t>(4);
    const uint32_t MSG  = get_arg_val<uint32_t>(5);
    const uint32_t N    = TOT / MSG;

    cb_reserve_back(tt::CBIndex::c_0, 1); const uint32_t src = get_write_ptr(tt::CBIndex::c_0);
    cb_reserve_back(tt::CBIndex::c_1, 1); const uint32_t dst = get_write_ptr(tt::CBIndex::c_1);

    {   DeviceZoneScopedN("bw");
#ifdef BW_STATEFUL
        // Destination coords and size are constant here, so they can be programmed into the command
        // buffer ONCE and each write reduced to two address stores. This bounds how much of the
        // per-request cost is unavoidable. It does NOT transfer to the a2a exchange, where every
        // message goes to a DIFFERENT peer and the state would have to be reloaded each time.
        noc_async_write_one_packet_set_state(::get_noc_addr(px, py, dst), MSG);
        for (uint32_t i = 0; i < N; ++i)
            noc_async_write_one_packet_with_state(src + i * MSG, dst + i * MSG);
#else
        for (uint32_t i = 0; i < N; ++i)
            noc_async_write(src + i * MSG, ::get_noc_addr(px, py, dst + i * MSG), MSG);
#endif
        noc_async_write_barrier();
    }
}
