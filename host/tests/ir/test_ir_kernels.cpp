// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// test_ir_kernels — validate the kernels_ir library (uniform contract) bit-exact on device.
// Dispatches mul_modq / add_modq / sub_modq over W tiles on one core, checks vs the C++ golden
// (same exact semantics as reference/ir/reference.py). Env: IRK_TILES (default 64).
//   cmake --build build --target test_ir_kernels && ./build/test_ir_kernels

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
#error "OVERRIDE_KERNEL_PREFIX must be set by CMake to this project's kernels_ir/ dir"
#endif

namespace {
uint32_t env_u32(const char* n, uint32_t d) { const char* v = std::getenv(n); return v ? (uint32_t)std::stoul(v) : d; }
constexpr uint32_t ELEMS = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;   // 1024
constexpr uint32_t TILE_BYTES = sizeof(uint32_t) * ELEMS;                            // 4096

// Dispatch one binary IR op (reader_binop -> compute -> writer_out) on core (0,0), W tiles.
std::vector<uint32_t> run_binop(const std::shared_ptr<distributed::MeshDevice>& md,
                                distributed::MeshCommandQueue& cq,
                                const std::string& compute_rel, uint32_t W,
                                const std::vector<uint32_t>& a, const std::vector<uint32_t>& b,
                                const std::vector<uint32_t>& compute_args) {
    distributed::MeshCoordinateRange dr(md->shape());
    distributed::DeviceLocalBufferConfig dram{.page_size = TILE_BYTES, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig bc{.size = (uint64_t)W * ELEMS * sizeof(uint32_t)};
    auto ab = distributed::MeshBuffer::create(bc, dram, md.get());
    auto bb = distributed::MeshBuffer::create(bc, dram, md.get());
    auto cb = distributed::MeshBuffer::create(bc, dram, md.get());
    EnqueueWriteMeshBuffer(cq, ab, a, false);
    EnqueueWriteMeshBuffer(cq, bb, b, false);

    Program p = CreateProgram();
    CoreCoord core{0, 0}; CoreRange cr(core, core);
    auto mkcb = [&](CBIndex i) {
        return CircularBufferConfig(2 * TILE_BYTES, {{i, DataFormat::UInt32}}).set_page_size(i, TILE_BYTES);
    };
    CreateCircularBuffer(p, cr, mkcb(CBIndex::c_0));
    CreateCircularBuffer(p, cr, mkcb(CBIndex::c_1));
    CreateCircularBuffer(p, cr, mkcb(CBIndex::c_16));
    auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_binop.cpp", cr,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_out.cpp", cr,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + compute_rel, cr,
                           ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                                         .dst_full_sync_en = true, .math_approx_mode = false});
    SetRuntimeArgs(p, rd, core, {(uint32_t)ab->address(), (uint32_t)bb->address(), W, 0});
    SetRuntimeArgs(p, wr, core, {(uint32_t)cb->address(), W, 0});
    SetRuntimeArgs(p, cp, core, compute_args);

    distributed::MeshWorkload wl; wl.add_program(dr, std::move(p));
    distributed::EnqueueMeshWorkload(cq, wl, false);
    std::vector<uint32_t> out;
    EnqueueReadMeshBuffer(cq, out, cb, true);
    return out;
}
}  // namespace

int main() {
    const uint32_t W = env_u32("IRK_TILES", 64u);
    const uint32_t prime = 251658241u;
    const uint32_t N = W * ELEMS;
    mont::MontgomeryContext ctx(prime);
    fmt::print("test_ir_kernels: W={} tiles ({} elems), prime={}\n", W, N, prime);

    std::mt19937 rng(99);
    std::uniform_int_distribution<uint32_t> dx(0, prime - 1), coin(0, 1);

    auto md = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = md->mesh_command_queue();
    int fails = 0;

    // ---- mul_modq (Montgomery domain) ----
    {
        std::vector<uint32_t> a(N), b(N), exp(N);
        for (uint32_t i = 0; i < N; ++i) {
            a[i] = ctx.to_montgomery(dx(rng)) + (coin(rng) ? prime : 0);   // lazy [0,2q)
            b[i] = ctx.to_montgomery(dx(rng)) + (coin(rng) ? prime : 0);
            exp[i] = ctx.mulmod(a[i], b[i]);                               // (a*b) mod q, canonical
        }
        auto r = run_binop(md, cq, "compute/mul_modq.cpp", W, a, b,
                           {W, prime, ctx.q_inv_32(), prime & 0xFFFFu, prime >> 16});
        size_t bad = 0;
        for (uint32_t i = 0; i < N; ++i) if (!(r[i] < 2u * prime && ctx.finalize(r[i]) == exp[i])) ++bad;
        fmt::print("  mul_modq : {}  ({} / {} bad)\n", bad ? "FAIL" : "OK", bad, N); fails += !!bad;
    }
    // ---- add_modq / sub_modq ([0,q)) ----
    for (const char* op : {"add", "sub"}) {
        std::vector<uint32_t> a(N), b(N), exp(N);
        for (uint32_t i = 0; i < N; ++i) {
            a[i] = dx(rng); b[i] = dx(rng);
            exp[i] = (op[0] == 'a') ? (uint32_t)(((uint64_t)a[i] + b[i]) % prime)
                                    : (uint32_t)((a[i] + prime - b[i]) % prime);
        }
        std::string rel = std::string("compute/") + op + "_modq.cpp";
        auto r = run_binop(md, cq, rel, W, a, b, {W, prime});
        size_t bad = 0;
        for (uint32_t i = 0; i < N; ++i) if (r[i] != exp[i]) ++bad;
        fmt::print("  {}_modq : {}  ({} / {} bad)\n", op, bad ? "FAIL" : "OK", bad, N); fails += !!bad;
    }

    md->close();
    if (fails) fmt::print("\nFAIL: {} op(s) mismatched\n", fails);
    else       fmt::print("\nALL IR KERNELS OK\n");
    return fails ? 1 : 0;
}
