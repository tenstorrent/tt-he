// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// RNS (multi-modulus) vector modular multiplication on Wormhole B0.
//
// Up to 16 vectors, each up to 2^16 elements, EACH WITH ITS OWN PRIME (the
// residue-number-system layout common in HE). Data is laid out vector-major:
// vector v occupies tiles [v*Lt, (v+1)*Lt). Per tile, the compute kernel selects
// that vector's modulus and computes C[i] = (A[i]*B[i]) mod q_v in Montgomery form.
//
// Same Montgomery algorithm as cpu/montgomery.hpp; multi-prime hybrid SFPI
// kernel. Env knobs:
//   MODMUL_KERNEL    kernel rel-path        (default compute/modmul_rns_sfpi.cpp)
//   MODMUL_NVEC      number of vectors V    (default 16, max 16)
//   MODMUL_VEC_TILES tiles per vector Lt    (default 64 -> 2^16 elems/vector)
//
// Build:  cmake -B build -G Ninja && cmake --build build
// Run:    ./build/vector_modmul_rns

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
#include <tt-metalium/work_split.hpp>

#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#error "OVERRIDE_KERNEL_PREFIX must be set by CMake to this project's kernels/ dir"
#endif

namespace {
std::string env_or(const char* n, const std::string& d) { const char* v = std::getenv(n); return v ? v : d; }
uint32_t env_u32(const char* n, uint32_t d) { const char* v = std::getenv(n); return v ? (uint32_t)std::stoul(v) : d; }

bool is_prime(uint32_t n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (uint64_t d = 3; d * d <= n; d += 2)
        if (n % d == 0) return false;
    return true;
}
// V distinct 28-bit primes, spread across the range.
std::vector<uint32_t> pick_primes(uint32_t V) {
    std::vector<uint32_t> ps;
    for (uint32_t k = 0; k < V; ++k) {
        uint32_t anchor = 268000000u - k * 15000000u;  // spaced ~15M apart, all < 2^28
        uint32_t c = anchor | 1u;                        // odd
        while (!is_prime(c)) c -= 2;
        ps.push_back(c);
    }
    return ps;
}
}  // namespace

int main() {
    const std::string kernel_rel = env_or("MODMUL_KERNEL", "compute/modmul_rns_sfpi.cpp");
    const uint32_t V = std::min<uint32_t>(env_u32("MODMUL_NVEC", 16u), 16u);
    const uint32_t Lt = env_u32("MODMUL_VEC_TILES", 64u);  // tiles per vector (64 -> 65536 elems)

    constexpr uint32_t ept = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;  // 1024
    constexpr uint32_t tile_bytes = sizeof(uint32_t) * ept;
    const uint32_t total_tiles = V * Lt;
    const uint32_t N = total_tiles * ept;

    auto primes = pick_primes(V);
    std::vector<mont::MontgomeryContext> ctx;
    std::vector<uint32_t> Qv, QIv, QLv, QHv;
    for (uint32_t v = 0; v < V; ++v) {
        ctx.emplace_back(primes[v]);
        Qv.push_back(primes[v]);
        QIv.push_back(ctx[v].q_inv_32());
        QLv.push_back(primes[v] & 0xFFFFu);
        QHv.push_back(primes[v] >> 16);
    }
    fmt::print("RNS modmul: kernel={} V={} tiles/vec={} total_tiles={} N={}\n",
               kernel_rel, V, Lt, total_tiles, N);
    fmt::print("primes: ");
    for (auto p : primes) fmt::print("{} ", p);
    fmt::print("\n");

    // Inputs, vector-major: tile gid = v*Lt + j  -> vector v -> prime[v].
    std::vector<uint32_t> a_vec(N), b_vec(N), expected(N);
    std::mt19937 rng(2024);
    for (uint32_t v = 0; v < V; ++v) {
        std::uniform_int_distribution<uint32_t> dx(0, primes[v] - 1);
        std::uniform_int_distribution<uint32_t> coin(0, 1);
        for (uint32_t j = 0; j < Lt; ++j) {
            uint32_t base = (v * Lt + j) * ept;
            for (uint32_t e = 0; e < ept; ++e) {
                uint32_t am = ctx[v].to_montgomery(dx(rng)) + (coin(rng) ? primes[v] : 0);
                uint32_t bm = ctx[v].to_montgomery(dx(rng)) + (coin(rng) ? primes[v] : 0);
                a_vec[base + e] = am;
                b_vec[base + e] = bm;
                expected[base + e] = ctx[v].mulmod(am, bm);
            }
        }
    }

    // ---- device ----
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    Program program = CreateProgram();

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_bytes, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buf_config{.size = (uint64_t)N * sizeof(uint32_t)};
    auto a_buf = distributed::MeshBuffer::create(buf_config, dram_config, mesh_device.get());
    auto b_buf = distributed::MeshBuffer::create(buf_config, dram_config, mesh_device.get());
    auto c_buf = distributed::MeshBuffer::create(buf_config, dram_config, mesh_device.get());

    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto all_device_cores = CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1});
    auto [num_cores, all_cores, cg1, cg2, tpc1, tpc2] =
        tt::tt_metal::split_work_to_cores(core_grid, total_tiles, /*row_major=*/true);
    fmt::print("using {} core(s) of {}x{} grid\n", num_cores, core_grid.x, core_grid.y);

    constexpr uint32_t cb_depth = 2;
    auto make_cb = [&](CBIndex idx) {
        return CircularBufferConfig(cb_depth * tile_bytes, {{idx, DataFormat::UInt32}}).set_page_size(idx, tile_bytes);
    };
    CreateCircularBuffer(program, all_cores, make_cb(CBIndex::c_0));
    CreateCircularBuffer(program, all_cores, make_cb(CBIndex::c_1));
    CreateCircularBuffer(program, all_cores, make_cb(CBIndex::c_16));

    KernelHandle reader = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    KernelHandle writer = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    KernelHandle compute = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX + kernel_rel, all_cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                      .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});

    const uint32_t a_addr = a_buf->address(), b_addr = b_buf->address(), c_addr = c_buf->address();
    uint32_t start_tile = 0;
    for (const auto& [group, work] : {std::make_pair(cg1, tpc1), std::make_pair(cg2, tpc2)}) {
        for (const auto& range : group.ranges()) {
            for (const auto& c : range) {
                SetRuntimeArgs(program, reader, c, {a_addr, b_addr, work, start_tile});
                SetRuntimeArgs(program, writer, c, {c_addr, work, start_tile});
                // compute args: work, start_tile, Lt, V, then Q[], QINV[], QLOW[], QHIGH[]
                std::vector<uint32_t> cargs{work, start_tile, Lt, V};
                cargs.insert(cargs.end(), Qv.begin(), Qv.end());
                cargs.insert(cargs.end(), QIv.begin(), QIv.end());
                cargs.insert(cargs.end(), QLv.begin(), QLv.end());
                cargs.insert(cargs.end(), QHv.begin(), QHv.end());
                SetRuntimeArgs(program, compute, c, cargs);
                start_tile += work;
            }
        }
    }

    EnqueueWriteMeshBuffer(cq, a_buf, a_vec, false);
    EnqueueWriteMeshBuffer(cq, b_buf, b_vec, false);
    workload.add_program(device_range, std::move(program));

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    constexpr int iters = 50;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e3 / iters;
    fmt::print("timing: {:.1f} us/iter ({} elems -> {:.1f} M modmul/s)\n", us, N, N / us);

    std::vector<uint32_t> result;
    distributed::EnqueueReadMeshBuffer(cq, result, c_buf, true);

    size_t mism = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (result[i] != expected[i]) {
            if (mism < 8) {
                uint32_t v = (i / ept) / Lt;
                fmt::print(stderr, "mismatch [{}] vec={} prime={}: a={} b={} got={} exp={}\n",
                           i, v, primes[v], a_vec[i], b_vec[i], result[i], expected[i]);
            }
            ++mism;
        }
    }
    mesh_device->close();
    if (mism == 0) { fmt::print("OK: all {} elements match ({} vectors x {} primes)\n", N, V, V); return 0; }
    fmt::print("FAIL: {} / {} mismatched\n", mism, N);
    return 1;
}
