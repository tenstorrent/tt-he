// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Standalone tt-metalium program: element-wise vector modular multiplication
// under a fixed 28-bit prime, using the Montgomery algorithm from
// cpu/montgomery.hpp. Targets Wormhole B0 (n150).
//
// The compute kernel, the validation MODE, the prime, and the tile count are
// all selected by environment variables so the kernel can be brought up in
// stages (raw multiply -> mul_hi_lo -> full modmul) and validated on-device
// against the CPU golden without rebuilding the host:
//
//   MODMUL_KERNEL  kernel rel-path under kernels/shared/  (default compute/modmul_compute.cpp)
//   MODMUL_MODE    passthrough|mullo|mulhi|modmul  (default modmul)
//   MODMUL_PRIME   28-bit prime                    (default 251658241)
//   MODMUL_NTILES  number of 1024-elem tiles       (default 4  -> N=4096)
//
// Build:  cmake -B build -G Ninja && cmake --build build
// Run:    ./build/vector_modmul

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

std::string env_or(const char* name, const std::string& def) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : def;
}
uint32_t env_u32(const char* name, uint32_t def) {
    const char* v = std::getenv(name);
    return v ? static_cast<uint32_t>(std::stoul(v)) : def;
}

enum class Mode { Passthrough, MulLo, MulHi, Modmul };

Mode parse_mode(const std::string& s) {
    if (s == "passthrough") return Mode::Passthrough;
    if (s == "mullo") return Mode::MulLo;
    if (s == "mulhi") return Mode::MulHi;
    return Mode::Modmul;
}

// Host golden for one element, matching what each kernel stage should produce.
uint32_t golden(Mode mode, const mont::MontgomeryContext& ctx, uint32_t a, uint32_t b) {
    switch (mode) {
        case Mode::Passthrough: return a;
        case Mode::MulLo:       return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) & 0xFFFFFFFFu);
        case Mode::MulHi:       return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
        case Mode::Modmul:      return ctx.mulmod(a, b);  // (a*b) mod q, canonical [0,q)
    }
    return 0;
}

}  // namespace

int main() {
    const std::string kernel_rel = env_or("MODMUL_KERNEL", "compute/modmul_compute.cpp");
    const Mode mode = parse_mode(env_or("MODMUL_MODE", "modmul"));
    const uint32_t prime = env_u32("MODMUL_PRIME", 251658241u);  // 15*2^24+1, ~27.9 bits
    const uint32_t n_tiles = env_u32("MODMUL_NTILES", 4u);

    mont::MontgomeryContext ctx(prime);
    const uint32_t q_low = prime & 0xFFFFu;
    const uint32_t q_high = prime >> 16;

    constexpr uint32_t elems_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;  // 1024
    const uint32_t N = n_tiles * elems_per_tile;
    constexpr uint32_t tile_bytes = sizeof(uint32_t) * elems_per_tile;  // 4096

    fmt::print("vector_modmul: kernel={} mode={} prime={} n_tiles={} N={}\n",
               kernel_rel, env_or("MODMUL_MODE", "modmul"), prime, n_tiles, N);

    // ---- build inputs (Montgomery-domain residues; lazy in [0, 2q)) ----
    std::vector<uint32_t> a_vec(N), b_vec(N), expected(N);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> dist_x(0, prime - 1);
    std::uniform_int_distribution<uint32_t> coin(0, 1);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t x = dist_x(rng), y = dist_x(rng);
        uint32_t am = ctx.to_montgomery(x);
        uint32_t bm = ctx.to_montgomery(y);
        a_vec[i] = am + (coin(rng) ? prime : 0);  // lazy [0, 2q)
        b_vec[i] = bm + (coin(rng) ? prime : 0);
        expected[i] = golden(mode, ctx, a_vec[i], b_vec[i]);
    }

    // ---- device setup ----
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    Program program = CreateProgram();

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_bytes, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buf_config{.size = static_cast<uint64_t>(N) * sizeof(uint32_t)};

    auto a_buf = distributed::MeshBuffer::create(buf_config, dram_config, mesh_device.get());
    auto b_buf = distributed::MeshBuffer::create(buf_config, dram_config, mesh_device.get());
    auto c_buf = distributed::MeshBuffer::create(buf_config, dram_config, mesh_device.get());

    // ---- multi-core work split: distribute the n_tiles across the core grid ----
    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto all_device_cores = CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1});
    auto [num_cores, all_cores, core_group_1, core_group_2,
          tiles_per_core_1, tiles_per_core_2] =
        tt::tt_metal::split_work_to_cores(core_grid, n_tiles, /*row_major=*/true);
    fmt::print("using {} core(s) of {}x{} grid\n", num_cores, core_grid.x, core_grid.y);

    // CBs: depth-2 double buffered, UInt32, on every used core.
    constexpr uint32_t cb_depth = 2;
    auto make_cb = [&](CBIndex idx) {
        return CircularBufferConfig(cb_depth * tile_bytes, {{idx, DataFormat::UInt32}})
            .set_page_size(idx, tile_bytes);
    };
    CreateCircularBuffer(program, all_cores, make_cb(CBIndex::c_0));   // a
    CreateCircularBuffer(program, all_cores, make_cb(CBIndex::c_1));   // b
    CreateCircularBuffer(program, all_cores, make_cb(CBIndex::c_16));  // out

    KernelHandle reader = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    KernelHandle writer = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    KernelHandle compute = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX + kernel_rel, all_cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                      .fp32_dest_acc_en = true,   // required: int32 needs 32-bit dest
                      .dst_full_sync_en = env_u32("MODMUL_FULL_SYNC", 1u) != 0,  // 8 tiles (full) vs 4 (half)
                      .math_approx_mode = false});

    // Per-core runtime args: each core handles its own [start_tile, start_tile+work) range.
    const uint32_t a_addr = a_buf->address(), b_addr = b_buf->address(), c_addr = c_buf->address();
    uint32_t start_tile = 0;
    for (const auto& [group, work] : {std::make_pair(core_group_1, tiles_per_core_1),
                                      std::make_pair(core_group_2, tiles_per_core_2)}) {
        for (const auto& range : group.ranges()) {
            for (const auto& c : range) {
                SetRuntimeArgs(program, reader, c, {a_addr, b_addr, work, start_tile});
                SetRuntimeArgs(program, writer, c, {c_addr, work, start_tile});
                SetRuntimeArgs(program, compute, c, {work, prime, ctx.q_inv_32(), q_low, q_high});
                start_tile += work;
            }
        }
    }

    EnqueueWriteMeshBuffer(cq, a_buf, a_vec, false);
    EnqueueWriteMeshBuffer(cq, b_buf, b_vec, false);

    workload.add_program(device_range, std::move(program));

    // Warm up (JIT compile + first dispatch), then time steady-state enqueues.
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    const int iters = (int)env_u32("MODMUL_ITERS", 50u);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    distributed::Finish(cq);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e3 / iters;
    fmt::print("timing: {:.1f} us/iter ({} elems -> {:.1f} M modmul/s)\n",
               us, N, N / us);

    // Device profiler trace (only meaningful when TT_METAL_DEVICE_PROFILER=1; the
    // installed lib is a profiler build). Dumps per-core/per-RISC kernel timestamps
    // to $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv.
    if (std::getenv("TT_METAL_DEVICE_PROFILER")) {
        ReadMeshDeviceProfilerResults(*mesh_device);
    }

    std::vector<uint32_t> result;
    distributed::EnqueueReadMeshBuffer(cq, result, c_buf, true);

    // ---- verify ----
    // Modmul kernels may emit a lazy result in [0, 2q): canonicalize before
    // comparing against the [0, q) golden, and check the lazy range. This is a
    // no-op for kernels that already return canonical [0, q) values.
    size_t mismatches = 0;
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t got = result[i];
        bool ok;
        if (mode == Mode::Modmul) {
            ok = (result[i] < 2u * prime) && (ctx.finalize(result[i]) == expected[i]);
        } else {
            ok = (result[i] == expected[i]);
        }
        if (!ok) {
            if (mismatches < 8) {
                fmt::print(stderr, "mismatch [{}]: a={} b={} got={} expected={}\n",
                           i, a_vec[i], b_vec[i], got, expected[i]);
            }
            ++mismatches;
        }
    }
    mesh_device->close();

    if (mismatches == 0) {
        fmt::print("OK: all {} elements match\n", N);
        return 0;
    }
    fmt::print("FAIL: {} / {} mismatched\n", mismatches, N);
    return 1;
}
