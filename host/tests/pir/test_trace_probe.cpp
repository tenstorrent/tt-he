// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Minimal MeshTrace probe — learn the capture/replay constraints on a 2-program workload before
// wiring trace into the PIR pipeline, where a mistake hangs the device for ten minutes.
//
// Answers three things:
//   (a) does a trace region have to be requested?  (DEFAULT_TRACE_REGION_SIZE is 0)
//   (b) can a program be BUILT inside the capture?  (no: "Writes are not supported during trace
//       capture" — CreateKernel uploads binaries, which is a write.  So program reuse is a hard
//       PREREQUISITE for trace, not an independent optimisation.)
//   (c) does replay actually re-execute, i.e. does the output change when the input changes?
//
//   ./build/test_trace_probe

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    constexpr uint32_t ELEMS = 1024, I32 = ELEMS * 4, Q = 167772161;
    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 64u << 20);
        auto& cq = mesh->mesh_command_queue();
        distributed::MeshCoordinateRange dr(mesh->shape());
        constexpr CoreCoord core{0, 0};

        auto mkbuf = [&](uint32_t nt) {
            distributed::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * I32}, d, mesh.get()); };
        auto ba = mkbuf(1), bb = mkbuf(1), bo = mkbuf(1);

        std::vector<uint32_t> A(ELEMS), B(ELEMS);
        for (uint32_t i = 0; i < ELEMS; ++i) { A[i] = i % Q; B[i] = (3 * i + 1) % Q; }
        distributed::EnqueueWriteMeshBuffer(cq, ba, A, false);
        distributed::EnqueueWriteMeshBuffer(cq, bb, B, false);

        // keep the workloads alive: the trace may reference them after capture
        std::vector<std::shared_ptr<distributed::MeshWorkload>> keep;
        auto add_prog = [&]() {
            Program p = CreateProgram();
            for (auto idx : {CBIndex::c_0, CBIndex::c_1, CBIndex::c_16}) {
                CircularBufferConfig c(2 * I32, {{idx, tt::DataFormat::UInt32}}); c.set_page_size(idx, I32);
                CreateCircularBuffer(p, core, c); }
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modadd_map.cpp", core,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)ba->address(), (uint32_t)bb->address(), 1u, 0u, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)bo->address(), 1u, 0u});
            SetRuntimeArgs(p, cp, core, {1u, Q});
            auto wl = std::make_shared<distributed::MeshWorkload>();
            wl->add_program(dr, std::move(p));
            distributed::EnqueueMeshWorkload(cq, *wl, false);
            keep.push_back(wl); };

        auto rdb = [&](auto b) { std::vector<uint32_t> v; distributed::EnqueueReadMeshBuffer(cq, v, b, true); return v; };

        fmt::print("1. plain enqueue (no trace)... ");
        add_prog();
        distributed::Finish(cq);
        auto got = rdb(bo);
        uint32_t mism = 0;
        for (uint32_t i = 0; i < ELEMS; ++i) if (got[i] != (A[i] + B[i]) % Q) ++mism;
        fmt::print("mismatches={}/{}\n", mism, ELEMS); pass &= (mism == 0);

        fmt::print("2. capture, RE-ENQUEUING the already-built workload (no CreateKernel inside)... ");
        fflush(stdout);
        auto tid = mesh->begin_mesh_trace(0);
        distributed::EnqueueMeshWorkload(cq, *keep[0], false);      // same object: nothing to upload
        mesh->end_mesh_trace(0, tid);
        fmt::print("captured\n");

        fmt::print("3. replay... "); fflush(stdout);
        std::vector<uint32_t> Z(ELEMS, 0);
        distributed::EnqueueWriteMeshBuffer(cq, bo, Z, false);        // clobber the output first
        mesh->replay_mesh_trace(0, tid, true);
        distributed::Finish(cq);
        got = rdb(bo); mism = 0;
        for (uint32_t i = 0; i < ELEMS; ++i) if (got[i] != (A[i] + B[i]) % Q) ++mism;
        fmt::print("mismatches={}/{}\n", mism, ELEMS); pass &= (mism == 0);

        fmt::print("4. replay picks up NEW input (proves it re-executes)... "); fflush(stdout);
        for (uint32_t i = 0; i < ELEMS; ++i) A[i] = (7 * i + 5) % Q;
        distributed::EnqueueWriteMeshBuffer(cq, ba, A, false);
        mesh->replay_mesh_trace(0, tid, true);
        distributed::Finish(cq);
        got = rdb(bo); mism = 0;
        for (uint32_t i = 0; i < ELEMS; ++i) if (got[i] != (A[i] + B[i]) % Q) ++mism;
        fmt::print("mismatches={}/{}\n", mism, ELEMS); pass &= (mism == 0);

        mesh->release_mesh_trace(tid);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
