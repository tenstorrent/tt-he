// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Unit test for recombine_bigK.cpp: R = (C0 + C1<<7 + C2<<14 + C3<<21) mod q,
// with large-K planes C_r in [0, 2^22) (the K=256 matmul case).
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
        constexpr uint32_t ELEMS = 1024, TB = ELEMS * 4;
        const uint32_t q = 167772161;
        mont::MontgomeryContext ctx(q);
        const uint64_t POW[4] = {1, 128, 16384, 2097152};
        const uint32_t K14 = ctx.to_montgomery(16384u);

        std::mt19937 rng(11);
        std::uniform_int_distribution<uint32_t> dist(0, (1u << 22) - 1);   // C_r < 2^22 (K=256)
        std::vector<uint32_t> C(4 * ELEMS), expected(ELEMS);
        for (uint32_t r = 0; r < 4; ++r)
            for (uint32_t i = 0; i < ELEMS; ++i) C[r * ELEMS + i] = dist(rng);
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
        auto Rb = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = TB}, dram, mesh.get());

        Program p = CreateProgram();
        auto cb = [&](CBIndex idx, uint32_t depth) {
            CircularBufferConfig c(depth * TB, {{idx, DataFormat::UInt32}}); c.set_page_size(idx, TB);
            CreateCircularBuffer(p, core, c);
        };
        cb(CBIndex::c_0, 4); cb(CBIndex::c_16, 2);
        auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/recombine_bigK.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                          .dst_full_sync_en = true, .math_approx_mode = false});
        SetRuntimeArgs(p, rd, core, {(uint32_t)Cb->address(), 4u, 0u});
        SetRuntimeArgs(p, wr, core, {(uint32_t)Rb->address(), 1u, 0u});
        SetRuntimeArgs(p, cp, core, {1u, q, ctx.q_inv_32(), q & 0xFFFF, q >> 16, K14});

        distributed::MeshWorkload wl; wl.add_program(dr, std::move(p));
        distributed::EnqueueWriteMeshBuffer(cq, Cb, C, false);
        distributed::EnqueueMeshWorkload(cq, wl, false);
        std::vector<uint32_t> result;
        distributed::EnqueueReadMeshBuffer(cq, result, Rb, true);

        uint32_t mism = 0;
        for (uint32_t i = 0; i < ELEMS; ++i)
            if (result[i] != expected[i]) { if (mism < 8) fmt::print("  i={} got={} exp={}\n", i, result[i], expected[i]); ++mism; }
        fmt::print("recombine-bigK (C_r<2^22): mismatches = {}/{}\n", mism, ELEMS);
        pass = (mism == 0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
