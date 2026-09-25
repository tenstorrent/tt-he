// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// int32 elementwise multiply microbenchmark — ONE dispatch, C cores, 2^n tiles each.
//
// Question it answers: can a single dispatch cover a (2^n, C, 32, 32) elementwise op, and what rate
// does it reach?  Answer to the first half is yes: one program holds C cores, each looping 2^n tiles,
// so the whole (2^n, C, 32, 32) volume is one EnqueueMeshWorkload.  The sweep over n then shows where
// per-dispatch overhead stops mattering and the steady-state rate begins.
//
// What is measured: MeshTrace replay of that single program, so no host program build and no host
// dispatch is in the number — device time only.  Every host-side step from the OnionPIR work is
// applied: buffers allocated once (trace bakes addresses), program built and enqueued once before the
// capture (a program cannot be built inside one), then captured and replayed R times.
//
// The operation is `mul_int_tile`, which on Wormhole is EMULATED (no native integer multiplier — it
// goes through fp32 limb products and keeps the low 32 bits). So this is the emulated-int32 rate, not
// a hardware int32 ALU rate.
//
//   ./int32_mul_bench [max_n] [cores] [reps] [core_major]   # default 10 56 20 0
//
// Correctness is checked against a host a*b (mod 2^32) at every n, not just once: the whole point of
// a rate number is that the kernel actually did the work.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
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

namespace {
constexpr uint32_t ELEMS = 1024, I32 = ELEMS * 4;        // one 32x32 int32 tile = 4096 B
using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }
}  // namespace

int main(int argc, char** argv) {
    const uint32_t max_n = (argc > 1) ? (uint32_t)std::atoi(argv[1]) : 10;
    const uint32_t CORES = (argc > 2) ? (uint32_t)std::atoi(argv[2]) : 56;
    const uint32_t REPS  = (argc > 3) ? (uint32_t)std::atoi(argv[3]) : 20;
    const bool CORE_MAJOR = (argc > 4) && std::atoi(argv[4]) != 0;   // compare the other layout
    const uint32_t MMAXT = 1u << max_n;

    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 64u << 20);
        auto& cq = mesh->mesh_command_queue();
        distributed::MeshCoordinateRange dr(mesh->shape());
        auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX = grid.x, GY = grid.y;
        if (CORES > GX * GY) { fmt::print(stderr, "only {}x{}={} cores available\n", GX, GY, GX*GY); return 2; }
        fmt::print("grid {}x{} = {} cores available; using {}\n", GX, GY, GX*GY, CORES);
        fmt::print("int32 elementwise multiply over (S={}, C={}, 32, 32), one dispatch, {} layout\n"
                   "every tile distinct: read once, one multiply, written once\n\n",
                   "2^n", CORES, CORE_MAJOR ? "(C,S) core-major" : "(S,C) as specified");

        // first C cores, row-major (a full rectangle when C is a multiple of GX; 56 = 8x7)
        auto core_set = [&](uint32_t C) { std::vector<CoreRange> v; const uint32_t full = C / GX, rem = C % GX;
            if (full) v.push_back(CoreRange(CoreCoord{0, 0}, CoreCoord{GX - 1, full - 1}));
            if (rem)  v.push_back(CoreRange(CoreCoord{0, full}, CoreCoord{rem - 1, full}));
            return CoreRangeSet(v); };
        auto core_at = [&](uint32_t i) { return CoreCoord{i % GX, i / GX}; };

        // buffers sized for the LARGEST n and reused by every config: trace bakes addresses, and
        // reallocating per config would also mean re-measuring allocation instead of the kernel
        const size_t NT = (size_t)CORES * MMAXT;
        distributed::DeviceLocalBufferConfig dram{.page_size = I32, .buffer_type = BufferType::DRAM};
        auto mk = [&](size_t nt) { return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = nt * I32}, dram, mesh.get()); };
        fmt::print("largest config: 3 x {} tiles = {:.2f} GB\n\n", NT, 3.0 * NT * I32 / (1ull << 30));

        std::mt19937 rng(7);
        std::vector<uint32_t> A((size_t)NT * ELEMS), B((size_t)NT * ELEMS);
        for (auto& v : A) v = rng();
        for (auto& v : B) v = rng();

        fmt::print("{:>4} {:>8} {:>10} {:>12} {:>12} {:>10} {:>9}\n",
                   "n", "S", "tiles", "traced ms", "Gelem/s", "GB/s", "check");
        for (uint32_t n = 0; n <= max_n; ++n) {
            const uint32_t M = 1u << n;                       // tiles per core, sequential
            const size_t tiles = (size_t)CORES * M;
            auto cr = core_set(CORES);
            // sized per config: each config gets its own trace, so addresses only need to be stable
            // within one, and this keeps the verification read-back proportional to the work
            auto bA = mk(tiles), bB = mk(tiles), bO = mk(tiles);
            { std::vector<uint32_t> a(A.begin(), A.begin() + tiles * ELEMS),
                                    b(B.begin(), B.begin() + tiles * ELEMS);
              distributed::EnqueueWriteMeshBuffer(cq, bA, a, false);
              distributed::EnqueueWriteMeshBuffer(cq, bB, b, false);
              distributed::Finish(cq); }

            // ---- ONE program: C cores x M tiles = the whole (M, C, 32, 32) volume ----
            Program p = CreateProgram();
            const uint32_t depth = std::min(M, 4u);            // double/quad buffer; streaming, so small
            auto cb = [&](CBIndex idx, uint32_t d) { CircularBufferConfig c(d * I32, {{idx, tt::DataFormat::UInt32}});
                c.set_page_size(idx, I32); CreateCircularBuffer(p, cr, c); };
            cb(CBIndex::c_0, depth); cb(CBIndex::c_1, depth); cb(CBIndex::c_16, depth);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab_2.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32_s.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/mul_int32.cpp", cr,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            // The tensor is (S, C, 32, 32) with S = M: tile (s, core c) is at index s*C + c, so core c
            // starts at c and strides by C.  Every tile is distinct — read once, multiplied once,
            // written once, no reuse anywhere.  CORE_MAJOR flips to the (C, S) alternative (core c owns
            // the contiguous run [c*S, (c+1)*S)) purely to see what the access pattern costs.
            const uint32_t stride = CORE_MAJOR ? 1u : CORES;
            for (uint32_t i = 0; i < CORES; ++i) { const auto c = core_at(i);
                const uint32_t base = CORE_MAJOR ? i * M : i;
                SetRuntimeArgs(p, rd, c, {(uint32_t)bA->address(), (uint32_t)bB->address(), M, base, base, stride});
                SetRuntimeArgs(p, wr, c, {(uint32_t)bO->address(), M, base, stride});
                SetRuntimeArgs(p, cp, c, {M}); }

            auto wl = std::make_shared<distributed::MeshWorkload>();
            wl->add_program(dr, std::move(p));
            distributed::EnqueueMeshWorkload(cq, *wl, false);   // build + run once, outside any capture
            distributed::Finish(cq);

            // ---- correctness, every n ----
            std::vector<uint32_t> O;
            distributed::EnqueueReadMeshBuffer(cq, O, bO, true);
            size_t bad = 0;
            for (size_t i = 0; i < tiles * ELEMS; ++i)     // every tile distinct: out = a*b, once
                if (O[i] != (uint32_t)(A[i] * B[i])) ++bad;

            // ---- trace: replay the SAME workload, no host build, no host dispatch ----
            auto tid = mesh->begin_mesh_trace(0);
            distributed::EnqueueMeshWorkload(cq, *wl, false);
            mesh->end_mesh_trace(0, tid);
            distributed::Finish(cq);
            mesh->replay_mesh_trace(0, tid, true);              // warm
            distributed::Finish(cq);
            auto t0 = clk::now();
            for (uint32_t r = 0; r < REPS; ++r) mesh->replay_mesh_trace(0, tid, false);
            distributed::Finish(cq);
            const double ms = ms_since(t0) / REPS;
            mesh->release_mesh_trace(tid);

            const double elems = (double)tiles * ELEMS;
            const double bytes = (double)tiles * I32 * 3;       // 2 tiles in + 1 out
            fmt::print("{:>4} {:>8} {:>10} {:>12.4f} {:>12.3f} {:>10.1f} {:>9}\n",
                       n, M, tiles, ms, elems / (ms * 1e6), bytes / (ms * 1e6), bad ? "FAIL" : "ok");
            pass &= (bad == 0); }

        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
    return pass ? 0 : 1;
}
