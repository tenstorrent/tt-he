// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_Bw001 streaming reader — feeds one core's column of the (2^n, C, 32, 32) int32 elementwise
// multiply.  The tensor is laid out row-major in tiles: row i, column c is tile id (i*C + c).  This
// core owns exactly ONE column (`tile0`), so its work is the strided sequence
//
//     tile0, tile0 + C, tile0 + 2C, ...      (one tile per sequential iteration)
//
// wrapped every W rows so the DRAM working set stays bounded while the traffic stays real (there is
// no DRAM-side cache on Blackhole, so a wrapped re-read costs a full DRAM read).
//
// BATCHING IS THE WHOLE POINT.  A read-barrier-push loop per tile pays full DRAM latency in series
// and nothing pipelines — the same trap documented in dataflow/reader_fused_batch.cpp.  Here BATCH
// iterations' worth of A and B reads (2*BATCH tiles) are all put in flight and retired by ONE
// barrier, so latency is overlapped and the core tracks DRAM bandwidth rather than DRAM latency.
//
// CB WRAP HAZARD: this writes the batch as `base + j*ts`, which is only contiguous if the reserved
// run never straddles the circular buffer's wrap point.  The host guarantees that by sizing every
// CB to a multiple of BATCH and by only ever reserving/pushing BATCH at a time (n_iter is a power
// of two and BATCH <= 4 is too, so the last batch is never partial).
//
//   Runtime args: 0=a_addr, 1=b_addr, 2=n_iter (2^n), 3=tile0, 4=C, 5=W, 6=BATCH
//   Compile args: TensorAccessorArgs for A then B.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_iter = get_arg_val<uint32_t>(2);
    const uint32_t tile0  = get_arg_val<uint32_t>(3);
    const uint32_t C      = get_arg_val<uint32_t>(4);
    const uint32_t W      = get_arg_val<uint32_t>(5);
    const uint32_t BATCH  = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb0 = 0, cb1 = 1;

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    const uint32_t ts = get_tile_size(cb0);

    uint32_t id = tile0;   // current tile id  (row*C + tile0)
    uint32_t row = 0;      // row counter, wraps at W

    for (uint32_t i = 0; i < n_iter; i += BATCH) {
        cb_reserve_back(cb0, BATCH);
        cb_reserve_back(cb1, BATCH);
        const uint32_t p0 = get_write_ptr(cb0);
        const uint32_t p1 = get_write_ptr(cb1);

        // all 2*BATCH reads in flight together
        for (uint32_t j = 0; j < BATCH; ++j) {
            noc_async_read_tile(id, a, p0 + j * ts);
            noc_async_read_tile(id, b, p1 + j * ts);
            ++row;
            id += C;
            if (row == W) { row = 0; id = tile0; }
        }
        noc_async_read_barrier();

        cb_push_back(cb0, BATCH);
        cb_push_back(cb1, BATCH);
    }
}
