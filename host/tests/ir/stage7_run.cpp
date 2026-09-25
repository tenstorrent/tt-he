// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// stage7_run — codegen/dispatch of the compiler's multi-core schedule, on real hardware.
// Runs W independent tile-ops of a chosen kernel, split across exactly P cores in ONE program
// (each core gets ~W/P tiles), and measures steady-state makespan. This is the executable form
// of schedule.py's placement for the parallel (bag-of-independent-ops) part of an HE workload.
//
// Env:  S7_TILES (W, default 512)   S7_CORES (P, default 8)   S7_OP (modmul|modadd, default modmul)
//       S7_ITERS (timed iters, default 100)
//   ./build/stage7_run
//
// "Same workload, different cores": vary S7_CORES.   "Different workload": vary S7_OP / S7_TILES.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#error "OVERRIDE_KERNEL_PREFIX must be set by CMake"
#endif

namespace {
uint32_t envu(const char* n, uint32_t d) { const char* v = std::getenv(n); return v ? (uint32_t)std::stoul(v) : d; }
std::string envs(const char* n, const char* d) { const char* v = std::getenv(n); return v ? v : d; }
constexpr uint32_t ELEMS = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT, TB = ELEMS * 4;
}  // namespace

int main() {
    const uint32_t W = envu("S7_TILES", 512), P = envu("S7_CORES", 8), iters = envu("S7_ITERS", 100);
    const std::string op = envs("S7_OP", "modmul");
    const uint32_t prime = 251658241u;
    const bool is_mul = (op == "modmul");
    mont::MontgomeryContext ctx(prime);
    const uint32_t N = W * ELEMS;

    std::vector<uint32_t> a(N), b(N);
    std::mt19937 rng(7);
    std::uniform_int_distribution<uint32_t> dx(0, prime - 1), coin(0, 1);
    for (uint32_t i = 0; i < N; ++i)
        if (is_mul) { a[i] = ctx.to_montgomery(dx(rng)) + (coin(rng) ? prime : 0); b[i] = ctx.to_montgomery(dx(rng)) + (coin(rng) ? prime : 0); }
        else        { a[i] = dx(rng); b[i] = dx(rng); }

    auto md = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange dr(md->shape());
    auto grid = md->compute_with_storage_grid_size();
    const uint32_t maxP = grid.x * grid.y, Pc = std::min(P, maxP);

    distributed::DeviceLocalBufferConfig dram{.page_size = TB, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig bc{.size = (uint64_t)N * 4};
    auto ba = distributed::MeshBuffer::create(bc, dram, md.get());
    auto bb = distributed::MeshBuffer::create(bc, dram, md.get());
    auto bo = distributed::MeshBuffer::create(bc, dram, md.get());
    EnqueueWriteMeshBuffer(cq, ba, a, false); EnqueueWriteMeshBuffer(cq, bb, b, false);

    Program prog = CreateProgram();
    // P cores in row-major order
    std::vector<CoreCoord> cores;
    for (uint32_t i = 0; i < Pc; ++i) cores.push_back({i % grid.x, i / grid.x});
    std::set<CoreRange> crset; for (auto c : cores) crset.insert(CoreRange(c, c));
    CoreRangeSet all(crset);
    auto mkcb = [&](CBIndex idx) { CircularBufferConfig c(2 * TB, {{idx, DataFormat::UInt32}}); c.set_page_size(idx, TB); CreateCircularBuffer(prog, all, c); };
    mkcb(CBIndex::c_0); mkcb(CBIndex::c_1); mkcb(CBIndex::c_16);
    auto rd = CreateKernel(prog, OVERRIDE_KERNEL_PREFIX "dataflow/reader_binop.cpp", all,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto wr = CreateKernel(prog, OVERRIDE_KERNEL_PREFIX "dataflow/writer_out.cpp", all,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto cp = CreateKernel(prog, OVERRIDE_KERNEL_PREFIX + std::string(is_mul ? "compute/mul_modq.cpp" : "compute/add_modq.cpp"), all,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});

    // split W tiles across the Pc cores (even + remainder to the first cores)
    const uint32_t base = W / Pc, rem = W % Pc;
    uint32_t start = 0;
    const uint32_t aa = ba->address(), bbb = bb->address(), oo = bo->address();
    for (uint32_t i = 0; i < Pc; ++i) {
        uint32_t work = base + (i < rem ? 1 : 0);
        SetRuntimeArgs(prog, rd, cores[i], {aa, bbb, work, start});
        SetRuntimeArgs(prog, wr, cores[i], {oo, work, start});
        if (is_mul) SetRuntimeArgs(prog, cp, cores[i], {work, prime, ctx.q_inv_32(), prime & 0xFFFFu, prime >> 16});
        else        SetRuntimeArgs(prog, cp, cores[i], {work, prime});
        start += work;
    }
    distributed::MeshWorkload wl; wl.add_program(dr, std::move(prog));
    distributed::EnqueueMeshWorkload(cq, wl, false); distributed::Finish(cq);   // warm (JIT)

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t it = 0; it < iters; ++it) distributed::EnqueueMeshWorkload(cq, wl, false);
    distributed::Finish(cq);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e3 / iters;

    // correctness (modmul only, quick)
    std::vector<uint32_t> out; EnqueueReadMeshBuffer(cq, out, bo, true);
    size_t bad = 0;
    for (uint32_t i = 0; i < N; ++i) {
        bool ok = is_mul ? (out[i] < 2u * prime && ctx.finalize(out[i]) == ctx.mulmod(a[i], b[i]))
                         : (out[i] == (uint32_t)(((uint64_t)a[i] + b[i]) % prime));
        bad += !ok;
    }
    md->close();
    fmt::print("op={} W={} P={}/{} : {:.1f} us/iter  ({} tiles/core avg)  {}\n",
               op, W, Pc, maxP, us, W / Pc, bad ? "BAD" : "ok");
    return bad ? 1 : 0;
}
