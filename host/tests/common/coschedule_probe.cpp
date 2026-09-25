// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Co-scheduling feasibility probe (docs/ir/formalization.md §1, pre-build item #1).
//
// Question: on Wormhole, can ONE dispatched program run two DIFFERENT compute
// kernels on two DISJOINT core ranges concurrently, and is that faster than
// dispatching the two ops as separate programs serially?  This is the premise of
// the whole "spatial co-scheduling" mapping design (e.g. "NTT on A || modmul on B").
//
// Setup: op A = Montgomery modmul (modmul_compute.cpp) on core A; op B = modular
// add (modadd_map.cpp, a genuinely different kernel) on core B; each processes
// W independent 1024-elem tiles.
//   - SERIAL: two programs (op A on core A) then (op B on core B), dispatched back
//             to back on one command queue -> execute sequentially.
//   - FUSED : one program with op A on core A AND op B on core B -> one dispatch,
//             concurrent execution.
// We verify both outputs bit-exact and compare us/iter.
//
// Env: COSCHED_TILES (W, default 256), COSCHED_ITERS (default 200).
// Build/run: cmake --build build --target coschedule_probe && ./build/coschedule_probe

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
#error "OVERRIDE_KERNEL_PREFIX must be set by CMake to this project's kernels/ dir"
#endif

namespace {
uint32_t env_u32(const char* name, uint32_t def) {
    const char* v = std::getenv(name);
    return v ? static_cast<uint32_t>(std::stoul(v)) : def;
}

constexpr uint32_t ELEMS = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;  // 1024
constexpr uint32_t TILE_BYTES = sizeof(uint32_t) * ELEMS;                           // 4096

struct Op {                 // one op's buffers
    std::shared_ptr<distributed::MeshBuffer> a, b, c;
};

CircularBufferConfig cb_cfg(CBIndex idx) {
    return CircularBufferConfig(2 * TILE_BYTES, {{idx, DataFormat::UInt32}}).set_page_size(idx, TILE_BYTES);
}
ComputeConfig compute_cfg() {
    return ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                         .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false};
}

// Add op A (modmul) to `program` on `core`.
void add_modmul(Program& program, const CoreCoord& core, const Op& op, uint32_t W,
                uint32_t prime, const mont::MontgomeryContext& ctx) {
    CoreRange cr(core, core);
    CreateCircularBuffer(program, cr, cb_cfg(CBIndex::c_0));
    CreateCircularBuffer(program, cr, cb_cfg(CBIndex::c_1));
    CreateCircularBuffer(program, cr, cb_cfg(CBIndex::c_16));
    auto rd = CreateKernel(program, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", cr,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto wr = CreateKernel(program, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", cr,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto cp = CreateKernel(program, OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp", cr, compute_cfg());
    SetRuntimeArgs(program, rd, core, {op.a->address(), op.b->address(), W, 0});
    SetRuntimeArgs(program, wr, core, {op.c->address(), W, 0});
    SetRuntimeArgs(program, cp, core, {W, prime, ctx.q_inv_32(), prime & 0xFFFFu, prime >> 16});
}

// Add op B (modadd, a different kernel) to `program` on `core`.
void add_modadd(Program& program, const CoreCoord& core, const Op& op, uint32_t W, uint32_t prime) {
    CoreRange cr(core, core);
    CreateCircularBuffer(program, cr, cb_cfg(CBIndex::c_0));
    CreateCircularBuffer(program, cr, cb_cfg(CBIndex::c_1));
    CreateCircularBuffer(program, cr, cb_cfg(CBIndex::c_16));
    auto rd = CreateKernel(program, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", cr,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto wr = CreateKernel(program, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", cr,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto cp = CreateKernel(program, OVERRIDE_KERNEL_PREFIX "compute/modadd_map.cpp", cr, compute_cfg());
    SetRuntimeArgs(program, rd, core, {op.a->address(), op.b->address(), W, 0});
    SetRuntimeArgs(program, wr, core, {op.c->address(), W, 0});
    SetRuntimeArgs(program, cp, core, {W, prime});
}

double time_workload(distributed::MeshCommandQueue& cq, std::vector<distributed::MeshWorkload*> wls, int iters) {
    for (auto* wl : wls) distributed::EnqueueMeshWorkload(cq, *wl, false);  // warm up (JIT + first run)
    distributed::Finish(cq);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it)
        for (auto* wl : wls) distributed::EnqueueMeshWorkload(cq, *wl, false);
    distributed::Finish(cq);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e3 / iters;
}
}  // namespace

int main() {
    const uint32_t W = env_u32("COSCHED_TILES", 256u);
    const int iters = (int)env_u32("COSCHED_ITERS", 200u);
    const uint32_t prime = 251658241u;  // ~27.9-bit NTT prime
    mont::MontgomeryContext ctx(prime);

    fmt::print("coschedule_probe: W={} tiles/op ({} elems), iters={}, prime={}\n", W, W * ELEMS, iters, prime);

    // ---- inputs: modmul ops use Montgomery lazy [0,2q); modadd op uses [0,q) ----
    const uint32_t N = W * ELEMS;
    std::vector<uint32_t> aA(N), bA(N), expA(N), aB(N), bB(N), expB(N), aC(N), bC(N), expC(N);
    std::mt19937 rng(2024);
    std::uniform_int_distribution<uint32_t> dx(0, prime - 1), coin(0, 1);
    auto mont_pair = [&](uint32_t& a, uint32_t& b, uint32_t& e) {
        uint32_t x = dx(rng), y = dx(rng);
        a = ctx.to_montgomery(x) + (coin(rng) ? prime : 0);
        b = ctx.to_montgomery(y) + (coin(rng) ? prime : 0);
        e = ctx.mulmod(a, b);
    };
    for (uint32_t i = 0; i < N; ++i) {
        mont_pair(aA[i], bA[i], expA[i]);                 // op A: modmul
        aB[i] = dx(rng); bB[i] = dx(rng);                 // op B: modadd ([0,q))
        expB[i] = (uint32_t)(((uint64_t)aB[i] + bB[i]) % prime);
        mont_pair(aC[i], bC[i], expC[i]);                 // op C: modmul (for the equal-cost pair)
    }

    // ---- device + buffers ----
    auto md = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange device_range(md->shape());
    distributed::DeviceLocalBufferConfig dram{.page_size = TILE_BYTES, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig bc{.size = (uint64_t)N * sizeof(uint32_t)};
    auto mk = [&] { return distributed::MeshBuffer::create(bc, dram, md.get()); };
    Op opA{mk(), mk(), mk()}, opB{mk(), mk(), mk()}, opC{mk(), mk(), mk()};

    EnqueueWriteMeshBuffer(cq, opA.a, aA, false); EnqueueWriteMeshBuffer(cq, opA.b, bA, false);
    EnqueueWriteMeshBuffer(cq, opB.a, aB, false); EnqueueWriteMeshBuffer(cq, opB.b, bB, false);
    EnqueueWriteMeshBuffer(cq, opC.a, aC, false); EnqueueWriteMeshBuffer(cq, opC.b, bC, false);

    const CoreCoord coreA{0, 0}, coreB{1, 0};   // disjoint single cores

    // ===== Pairing 1: HETEROGENEOUS (modmul || modadd) — proves different kernels co-run =====
    Program hA = CreateProgram(); add_modmul(hA, coreA, opA, W, prime, ctx);
    Program hB = CreateProgram(); add_modadd(hB, coreB, opB, W, prime);
    distributed::MeshWorkload het_wlA, het_wlB;
    het_wlA.add_program(device_range, std::move(hA));
    het_wlB.add_program(device_range, std::move(hB));
    double het_serial = time_workload(cq, {&het_wlA, &het_wlB}, iters);
    Program hF = CreateProgram();
    add_modmul(hF, coreA, opA, W, prime, ctx); add_modadd(hF, coreB, opB, W, prime);
    distributed::MeshWorkload het_wlF; het_wlF.add_program(device_range, std::move(hF));
    double het_fused = time_workload(cq, {&het_wlF}, iters);

    // ===== Pairing 2: EQUAL-COST (modmul || modmul) — fair test of concurrency =====
    Program eA = CreateProgram(); add_modmul(eA, coreA, opA, W, prime, ctx);
    Program eB = CreateProgram(); add_modmul(eB, coreB, opC, W, prime, ctx);
    distributed::MeshWorkload eq_wlA, eq_wlB;
    eq_wlA.add_program(device_range, std::move(eA));
    eq_wlB.add_program(device_range, std::move(eB));
    double eq_serial = time_workload(cq, {&eq_wlA, &eq_wlB}, iters);
    Program eF = CreateProgram();
    add_modmul(eF, coreA, opA, W, prime, ctx); add_modmul(eF, coreB, opC, W, prime, ctx);
    distributed::MeshWorkload eq_wlF; eq_wlF.add_program(device_range, std::move(eF));
    double eq_fused = time_workload(cq, {&eq_wlF}, iters);

    // ---- verify (read back after the runs) ----
    std::vector<uint32_t> rA, rB, rC;
    EnqueueReadMeshBuffer(cq, rA, opA.c, true);
    EnqueueReadMeshBuffer(cq, rB, opB.c, true);
    EnqueueReadMeshBuffer(cq, rC, opC.c, true);
    size_t badA = 0, badB = 0, badC = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (!(rA[i] < 2u * prime && ctx.finalize(rA[i]) == expA[i])) ++badA;
        if (rB[i] != expB[i]) ++badB;
        if (!(rC[i] < 2u * prime && ctx.finalize(rC[i]) == expC[i])) ++badC;
    }
    md->close();

    fmt::print("\n=== Heterogeneous: modmul@(0,0) || modadd@(1,0) (different kernels, one program) ===\n");
    fmt::print("  correctness: modmul {}  modadd {}\n", badA == 0 ? "OK" : "FAIL", badB == 0 ? "OK" : "FAIL");
    fmt::print("  serial (2 programs): {:.1f} us   fused (1 program): {:.1f} us   speedup {:.2f}x\n",
               het_serial, het_fused, het_serial / het_fused);
    fmt::print("  (small speedup expected: modadd << modmul, so fusing saves only modadd's time)\n");

    fmt::print("\n=== Equal-cost: modmul@(0,0) || modmul@(1,0) (fair concurrency test) ===\n");
    fmt::print("  correctness: modmul(A) {}  modmul(C) {}\n", badA == 0 ? "OK" : "FAIL", badC == 0 ? "OK" : "FAIL");
    fmt::print("  serial (2 programs): {:.1f} us   fused (1 program): {:.1f} us   speedup {:.2f}x\n",
               eq_serial, eq_fused, eq_serial / eq_fused);

    if (badA || badB || badC) { fmt::print("FAIL: correctness\n"); return 1; }
    return 0;
}
