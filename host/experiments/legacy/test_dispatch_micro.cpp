// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PIR dispatch-cost micro-benchmark — the T1 experiment of docs/he/history/OPTIMIZATION_REPORT.md, at PIR's
// program shape (single-tile modmul, 2 tiles/poly).  The answer pipeline is ~84% EnqueueMeshWorkload
// (see pir_engine2048.hpp prof:: split), and the HE report's finding is that this cost is *fresh-program
// serialization*, not dispatch count: re-enqueueing the SAME Program object collapses it (~0.65 -> 0.04
// ms/call on n150).  This measures that ratio on Blackhole for the PIR program shape, so the engine
// refactor (workload cache + buffer pool) is sized against a real number instead of a projection.
//
// Three regimes, identical device work each iteration:
//   A  build-fresh   : CreateProgram + CreateKernel x3 + SetRuntimeArgs + Enqueue   (what the engine does today)
//   B  persistent    : build ONCE, re-enqueue the same MeshWorkload                 (the T2 lever)
//   C  traced replay : capture B's re-enqueue as a MeshTrace, then replay_mesh_trace (the T2.5 lever)
//
//   ./build/test_dispatch_micro [n_dispatch=64] [iters=20]

#include <chrono>
#include <cstdint>
#include <vector>
#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
namespace Dp = tt::tt_metal::distributed;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
using MeshBuf = std::shared_ptr<Dp::MeshBuffer>;
using hrc = std::chrono::high_resolution_clock;
static double msd(hrc::time_point t) { return std::chrono::duration<double, std::milli>(hrc::now() - t).count(); }

constexpr uint32_t ELEMS = 1024, I32 = ELEMS * 4;

int main(int argc, char** argv) {
    const uint32_t ND = (argc > 1) ? std::stoul(argv[1]) : 64;    // dispatches per iteration
    const uint32_t IT = (argc > 2) ? std::stoul(argv[2]) : 20;    // timed iterations
    bool pass = true;
    try {
        // 256 MB trace region, as HE's T1 does (trace replay needs it reserved at init).
        auto mesh = Dp::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 256u << 20);
        auto& cq = mesh->mesh_command_queue();
        Dp::MeshCoordinateRange dr(mesh->shape());
        CoreCoord core{0, 0};

        const uint32_t q = 268369921;
        mont::MontgomeryContext ctx(q);

        auto mkbuf = [&](uint32_t nt) {
            Dp::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return Dp::MeshBuffer::create(Dp::ReplicatedBufferConfig{.size = nt * I32}, d, mesh.get()); };

        // All buffers allocated UP FRONT (fixed addresses; trace bakes addresses in, and allocating
        // while a trace is alive warns -- OPTIMIZATION_REPORT.md T1 caveat 2).
        auto bA = mkbuf(2), bB = mkbuf(2), bO = mkbuf(2);
        std::vector<uint32_t> host(2 * ELEMS);
        for (uint32_t i = 0; i < 2 * ELEMS; ++i) host[i] = ctx.to_montgomery(i % q);
        Dp::EnqueueWriteMeshBuffer(cq, bA, host, false);
        Dp::EnqueueWriteMeshBuffer(cq, bB, host, false);

        // One modmul program: the engine's most-dispatched shape (Engine::mmul, 2 tiles).
        auto build = [&]() {
            Program p = CreateProgram();
            auto cb = [&](CBIndex idx) { CircularBufferConfig c(2 * I32, {{idx, tt::DataFormat::UInt32}});
                c.set_page_size(idx, I32); CreateCircularBuffer(p, core, c); };
            cb(CBIndex::c_0); cb(CBIndex::c_1); cb(CBIndex::c_16);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp", core,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)bA->address(), (uint32_t)bB->address(), 2u, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)bO->address(), 2u, 0u});
            SetRuntimeArgs(p, cp, core, {2u, q, ctx.q_inv_32(), q & 0xFFFF, q >> 16});
            Dp::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        auto rd_out = [&]() { std::vector<uint32_t> v; Dp::EnqueueReadMeshBuffer(cq, v, bO, true); return v; };

        // ---- A: build-fresh + enqueue (status quo) ----
        { auto wl = build(); Dp::EnqueueMeshWorkload(cq, wl, false); Dp::Finish(cq); }   // warm JIT
        auto refv = rd_out();
        double tA;
        { auto t0 = hrc::now();
          for (uint32_t i = 0; i < IT; ++i) { for (uint32_t d = 0; d < ND; ++d) { auto wl = build();
                Dp::EnqueueMeshWorkload(cq, wl, false); } Dp::Finish(cq); }
          tA = msd(t0) / IT; }

        // ---- B: persistent — build ND workloads ONCE, re-enqueue ----
        std::vector<Dp::MeshWorkload> WC; WC.reserve(ND);
        for (uint32_t d = 0; d < ND; ++d) WC.push_back(build());
        for (auto& w : WC) Dp::EnqueueMeshWorkload(cq, w, false);       // WARM: first enqueue writes binaries
        Dp::Finish(cq);                                                 // (trace forbids writes during capture)
        double tB;
        { auto t0 = hrc::now();
          for (uint32_t i = 0; i < IT; ++i) { for (auto& w : WC) Dp::EnqueueMeshWorkload(cq, w, false);
                Dp::Finish(cq); }
          tB = msd(t0) / IT; }
        auto gotB = rd_out();
        uint32_t mmB = 0; for (size_t i = 0; i < refv.size(); ++i) if (gotB[i] != refv[i]) ++mmB;

        // ---- C: traced replay of the same re-enqueue sequence ----
        double tC = -1; uint32_t mmC = 0;
        auto tid = mesh->begin_mesh_trace(0);
        for (auto& w : WC) Dp::EnqueueMeshWorkload(cq, w, false);
        mesh->end_mesh_trace(0, tid);
        Dp::Finish(cq);
        mesh->replay_mesh_trace(0, tid, true);                          // warm replay
        { auto t0 = hrc::now();
          for (uint32_t i = 0; i < IT; ++i) mesh->replay_mesh_trace(0, tid, true);   // blocking: wall ~= device-busy span
          tC = msd(t0) / IT; }
        { auto gotC = rd_out(); for (size_t i = 0; i < refv.size(); ++i) if (gotC[i] != refv[i]) ++mmC; }

        // ---- D: CO-SCHEDULED — ONE program, ND cores, one modmul each (disjoint output offsets) ----
        // The decisive question for query expansion: its Subs tree has 2^a INDEPENDENT key-switches per
        // level, so the win should come from running many independent single-core ops at once. But separate
        // programs SERIALIZE on the single command queue no matter which core they target -- parallelism
        // requires one program with disjoint core ranges. This measures that difference directly.
        double tD = -1; uint32_t mmD = 0;
        {
            auto grid = mesh->compute_with_storage_grid_size();
            const uint32_t GX = std::min(ND, (uint32_t)grid.x), GY = (ND + GX - 1) / GX;
            if (GY <= (uint32_t)grid.y) {
                auto b_multi = mkbuf(2 * ND);
                Program pp = CreateProgram();
                CoreRange crs(CoreCoord{0, 0}, CoreCoord{GX - 1, GY - 1});
                auto cbm = [&](CBIndex idx) { CircularBufferConfig c(2 * I32, {{idx, tt::DataFormat::UInt32}});
                    c.set_page_size(idx, I32); CreateCircularBuffer(pp, crs, c); };
                cbm(CBIndex::c_0); cbm(CBIndex::c_1); cbm(CBIndex::c_16);
                auto rd2 = CreateKernel(pp, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", crs,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
                auto wr2 = CreateKernel(pp, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", crs,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
                auto cp2 = CreateKernel(pp, OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp", crs,
                    ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
                for (uint32_t i = 0; i < ND; ++i) {
                    CoreCoord cc{i % GX, i / GX};
                    SetRuntimeArgs(pp, rd2, cc, {(uint32_t)bA->address(), (uint32_t)bB->address(), 2u, 0u});
                    SetRuntimeArgs(pp, wr2, cc, {(uint32_t)b_multi->address(), 2u, 2 * i});
                    SetRuntimeArgs(pp, cp2, cc, {2u, q, ctx.q_inv_32(), q & 0xFFFF, q >> 16});
                }
                Dp::MeshWorkload wlm; wlm.add_program(dr, std::move(pp));
                Dp::EnqueueMeshWorkload(cq, wlm, false); Dp::Finish(cq);          // warm
                auto t0 = hrc::now();
                for (uint32_t i = 0; i < IT; ++i) { Dp::EnqueueMeshWorkload(cq, wlm, false); Dp::Finish(cq); }
                tD = msd(t0) / IT;
                std::vector<uint32_t> vm; Dp::EnqueueReadMeshBuffer(cq, vm, b_multi, true);
                for (uint32_t i = 0; i < ND; ++i)
                    for (size_t x = 0; x < refv.size(); ++x)
                        if (vm[i * 2 * ELEMS + x] != refv[x]) ++mmD;
            }
        }

        fmt::print("\nPIR dispatch cost — {} dispatches/iter, {} iters, single-tile modmul (2 tiles/poly)\n", ND, IT);
        fmt::print(" regime                  |  ms/iter |  ms/dispatch | speedup\n");
        fmt::print("-------------------------+----------+--------------+--------\n");
        fmt::print(" A build-fresh + enqueue | {:8.3f} | {:12.4f} | {:5.2f}x\n", tA, tA / ND, 1.0);
        fmt::print(" B persistent re-enqueue | {:8.3f} | {:12.4f} | {:5.2f}x\n", tB, tB / ND, tA / tB);
        fmt::print(" C traced replay         | {:8.3f} | {:12.4f} | {:5.2f}x\n", tC, tC / ND, tA / tC);
        if (tD > 0)
            fmt::print(" D co-sched 1 prog/{:2} cores | {:8.3f} | {:12.4f} | {:5.2f}x\n", ND, tD, tD / ND, tA / tD);
        fmt::print("\n correctness: persistent {}/{}  traced {}/{}  co-sched {}/{}  (vs build-fresh)\n",
                   mmB, refv.size(), mmC, refv.size(), mmD, refv.size() * ND);
        pass = (mmB == 0 && mmC == 0 && mmD == 0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
    return pass ? 0 : 1;
}
