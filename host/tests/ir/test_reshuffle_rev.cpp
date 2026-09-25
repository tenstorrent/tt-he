// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Validate the on-device slot→poly reshuffle (reshuffle_scatter_mc.cpp, the scan output reshuffle)
// bit-exact vs a host gather.  S slot-tiles → P'=rest·COLS poly-tiles;
//   out_poly[p'](slot j) = in_slot[j](p'/COLS, p'%COLS).
//   ./build/test_reshuffle_rev [rest] [COLS] [seed]
#include <cstdint>
#include <random>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#error "OVERRIDE_KERNEL_PREFIX must be set by CMake"
#endif
namespace {
constexpr uint32_t T = 32, ELEMS = 1024, I32 = ELEMS * 4;
inline uint32_t faceidx(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }
}  // namespace

int main(int argc, char** argv) {
    const uint32_t rest = (argc > 1) ? std::stoul(argv[1]) : 32;
    const uint32_t COLS = (argc > 2) ? std::stoul(argv[2]) : 2;
    const uint32_t seed = (argc > 3) ? std::stoul(argv[3]) : 1;
    const uint32_t S = ELEMS, Pp = rest * COLS;      // slot-tiles in, poly-tiles out
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> di(0, 0xFFFFFFu);

    std::vector<uint32_t> Xin((size_t)S * ELEMS);
    for (auto& v : Xin) v = di(rng);
    // golden: out_poly[p'] at faceidx(j/32,j%32) = slot_j at faceidx(p'/COLS, p'%COLS)
    std::vector<uint32_t> gold((size_t)Pp * ELEMS, 0);
    for (uint32_t pp = 0; pp < Pp; ++pp) for (uint32_t j = 0; j < S; ++j)
        gold[(size_t)pp * ELEMS + faceidx(j >> 5, j & 31)] = Xin[(size_t)j * ELEMS + faceidx(pp / COLS, pp % COLS)];

    bool pass = true;
    try {
        auto md = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = md->mesh_command_queue();
        distributed::MeshCoordinateRange dr(md->shape());
        auto grid = md->compute_with_storage_grid_size();
        const uint32_t GX = grid.x, GY = grid.y, C = GX * GY;
        auto mkbuf = [&](size_t n) { distributed::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = n * I32}, d, md.get()); };
        auto b_in = mkbuf(S), b_out = mkbuf(Pp);
        distributed::EnqueueWriteMeshBuffer(cq, b_in, Xin, false);
        const uint32_t per = (Pp + C - 1) / C;

        Program p = CreateProgram();
        CoreRange cores(CoreCoord{0, 0}, CoreCoord{GX - 1, GY - 1});
        CircularBufferConfig c((per + 1) * I32, {{CBIndex::c_0, DataFormat::UInt32}}); c.set_page_size(CBIndex::c_0, I32);
        CreateCircularBuffer(p, cores, c);
        auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reshuffle_scatter_mc.cpp", cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        for (uint32_t gy = 0; gy < GY; ++gy) for (uint32_t gx = 0; gx < GX; ++gx) {
            uint32_t idx = gy * GX + gx, out_start = idx * per;
            uint32_t n_out = (out_start >= Pp) ? 0 : std::min(per, Pp - out_start);
            SetRuntimeArgs(p, rd, CoreCoord{gx, gy}, {(uint32_t)b_in->address(), (uint32_t)b_out->address(), S, 0u, COLS, out_start, n_out, 1u, 0u});
        }
        distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false);
        distributed::Finish(cq);

        std::vector<uint32_t> out; distributed::EnqueueReadMeshBuffer(cq, out, b_out, true);
        uint32_t mism = 0;
        for (size_t i = 0; i < (size_t)Pp * ELEMS; ++i) if (out[i] != gold[i]) ++mism;
        fmt::print("slot→poly reshuffle  rest={} COLS={} S={} P'={}  validation: {}  ({} bad)\n", rest, COLS, S, Pp, mism == 0 ? "OK" : "FAILED", mism);
        pass &= (mism == 0); pass &= md->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
