// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Unit test: on-device radix-128 limb decomposition (decompose_limbs.cpp).
// Verifies int32 -> 4x int8 (limb_p = (v>>7p)&127) and that the packer narrows
// int32 dest -> Int8 CB correctly.

#include <cstdint>
#include <random>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    bool pass = true;
    try {
        constexpr uint32_t ELEMS = 1024, NT = 6;  // NT input tiles -> 4*NT output limb tiles
        constexpr uint32_t i32_tile = ELEMS * 4, i8_tile = ELEMS * 1;
        std::mt19937 rng(7);
        std::uniform_int_distribution<uint32_t> dist(0, (1u << 28) - 1);
        std::vector<uint32_t> in(NT * ELEMS);
        for (auto& v : in) v = dist(rng);

        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        Program prog = CreateProgram();
        constexpr CoreCoord core = {0, 0};

        distributed::DeviceLocalBufferConfig i32d{.page_size = i32_tile, .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig i8d{.page_size = i8_tile, .buffer_type = BufferType::DRAM};
        auto inb = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = NT * i32_tile}, i32d, mesh.get());
        auto outb = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = 4 * NT * i8_tile}, i8d, mesh.get());

        CircularBufferConfig c0(NT * i32_tile, {{CBIndex::c_0, DataFormat::UInt32}}); c0.set_page_size(CBIndex::c_0, i32_tile);
        CreateCircularBuffer(prog, core, c0);
        CircularBufferConfig c16(8 * i8_tile, {{CBIndex::c_16, DataFormat::Int8}}); c16.set_page_size(CBIndex::c_16, i8_tile);
        CreateCircularBuffer(prog, core, c16);

        auto rd = CreateKernel(prog, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        std::vector<uint32_t> wc; TensorAccessorArgs(*outb).append_to(wc);
        auto wr = CreateKernel(prog, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i8.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = wc});
        auto cp = CreateKernel(prog, OVERRIDE_KERNEL_PREFIX "compute/decompose_limbs.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
        SetRuntimeArgs(prog, rd, core, {(uint32_t)inb->address(), NT, 0u});
        SetRuntimeArgs(prog, wr, core, {(uint32_t)outb->address(), 4u * NT, 0u});
        SetRuntimeArgs(prog, cp, core, {NT});  // n_tiles for the compute kernel

        std::vector<int8_t> zeros(4 * NT * ELEMS, 0);
        distributed::MeshWorkload wl; wl.add_program(dr, std::move(prog));
        distributed::EnqueueWriteMeshBuffer(cq, outb, zeros, false);   // zero output first
        distributed::EnqueueWriteMeshBuffer(cq, inb, in, false);
        distributed::EnqueueMeshWorkload(cq, wl, false);
        std::vector<int8_t> out;
        distributed::EnqueueReadMeshBuffer(cq, out, outb, true);  // 4*NT*1024 int8

        uint32_t mism = 0;
        for (uint32_t t = 0; t < NT; ++t) {
            uint32_t tm = 0;
            for (uint32_t p = 0; p < 4; ++p)
                for (uint32_t i = 0; i < ELEMS; ++i) {
                    int8_t got = out[(t * 4 + p) * ELEMS + i];
                    int8_t exp = (int8_t)((in[t * ELEMS + i] >> (7 * p)) & 127);
                    if (got != exp) ++tm;
                }
            fmt::print("  input tile t={}: mism={}/{}\n", t, tm, 4 * ELEMS);
            mism += tm;
        }
        fmt::print("decompose: mismatches = {}/{}\n", mism, 4 * NT * ELEMS);
        pass = (mism == 0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
