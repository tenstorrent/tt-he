// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Unit test for the on-device recombine path:
//   result = (sum_r 128^r * C_r) mod q,  computed as
//   term_r = modmul(C_r, montgomery(128^r))  [reuse modmul kernel]   then
//   result = modadd_reduce(term_0..term_3) mod q.
// The intermediate term buffer stays in DRAM (no host read between the two
// device programs) -- also exercises on-device chaining.

#include <cstdint>
#include <random>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    bool pass = true;
    try {
        constexpr uint32_t ELEMS = 1024, TB = ELEMS * 4;  // int32 tile = 4096 B
        const uint32_t q = 167772161;
        mont::MontgomeryContext ctx(q);
        const uint64_t POW[4] = {1, 128, 16384, 2097152};

        std::mt19937 rng(3);
        std::uniform_int_distribution<uint32_t> dist(0, (1u << 21) - 1);  // C_r < 2^21
        std::vector<uint32_t> C(4 * ELEMS), K(4 * ELEMS), expected(ELEMS);
        for (uint32_t r = 0; r < 4; ++r) {
            uint32_t kr = ctx.to_montgomery((uint32_t)(POW[r] % q));
            for (uint32_t i = 0; i < ELEMS; ++i) { C[r * ELEMS + i] = dist(rng); K[r * ELEMS + i] = kr; }
        }
        for (uint32_t i = 0; i < ELEMS; ++i) {
            unsigned __int128 v = 0;
            for (uint32_t r = 0; r < 4; ++r) v += (unsigned __int128)C[r * ELEMS + i] * POW[r];
            expected[i] = (uint32_t)(v % q);
        }

        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        distributed::DeviceLocalBufferConfig dram{.page_size = TB, .buffer_type = BufferType::DRAM};
        auto Cb = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = 4 * TB}, dram, mesh.get());
        auto Kb = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = 4 * TB}, dram, mesh.get());
        auto Tb = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = 4 * TB}, dram, mesh.get());  // term
        auto Rb = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = TB}, dram, mesh.get());      // result

        auto cb = [&](Program& p, CBIndex idx, uint32_t depth) {
            CircularBufferConfig c(depth * TB, {{idx, DataFormat::UInt32}}); c.set_page_size(idx, TB);
            CreateCircularBuffer(p, core, c);
        };

        // ---- program 1: modmul (C .* K -> term) ----
        Program p1 = CreateProgram();
        cb(p1, CBIndex::c_0, 2); cb(p1, CBIndex::c_1, 2); cb(p1, CBIndex::c_16, 2);
        auto r1 = CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto w1 = CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto c1 = CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
        SetRuntimeArgs(p1, r1, core, {(uint32_t)Cb->address(), (uint32_t)Kb->address(), 4u, 0u});
        SetRuntimeArgs(p1, w1, core, {(uint32_t)Tb->address(), 4u, 0u});
        SetRuntimeArgs(p1, c1, core, {4u, q, ctx.q_inv_32(), q & 0xFFFF, q >> 16});
        distributed::MeshWorkload wl1; wl1.add_program(dr, std::move(p1));

        // ---- program 2: modadd_reduce (term -> result) ----
        Program p2 = CreateProgram();
        cb(p2, CBIndex::c_0, 4); cb(p2, CBIndex::c_16, 2);
        auto r2 = CreateKernel(p2, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto w2 = CreateKernel(p2, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto c2 = CreateKernel(p2, OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
        SetRuntimeArgs(p2, r2, core, {(uint32_t)Tb->address(), 4u, 0u});
        SetRuntimeArgs(p2, w2, core, {(uint32_t)Rb->address(), 1u, 0u});
        SetRuntimeArgs(p2, c2, core, {4u, q});
        distributed::MeshWorkload wl2; wl2.add_program(dr, std::move(p2));

        distributed::EnqueueWriteMeshBuffer(cq, Cb, C, false);
        distributed::EnqueueWriteMeshBuffer(cq, Kb, K, false);
        distributed::EnqueueMeshWorkload(cq, wl1, false);   // modmul -> term (stays in DRAM)
        distributed::EnqueueMeshWorkload(cq, wl2, false);   // modadd -> result
        std::vector<uint32_t> result;
        distributed::EnqueueReadMeshBuffer(cq, result, Rb, true);

        uint32_t mism = 0;
        for (uint32_t i = 0; i < ELEMS; ++i)
            if (result[i] != expected[i]) { if (mism < 8) fmt::print("  i={} got={} exp={}\n", i, result[i], expected[i]); ++mism; }
        fmt::print("recombine: mismatches = {}/{}\n", mism, ELEMS);
        pass = (mism == 0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
