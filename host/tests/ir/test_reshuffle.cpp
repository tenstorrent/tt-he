// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Validate the on-device poly→slot reshuffle (reshuffle_gather_mc.cpp) bit-exact vs a host gather.
// P = rest·N0 poly tiles (each 32×32 holding S=1024 slots) → S slot tiles, out[j](r,k)=poly[k*rest+r][j].
//   ./build/test_reshuffle [rest] [N0] [seed]
#include <cstdint>
#include <random>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
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
    const uint32_t rest   = (argc > 1) ? std::stoul(argv[1]) : 8;
    const uint32_t N0     = (argc > 2) ? std::stoul(argv[2]) : 32;
    const uint32_t HALVES = (argc > 3) ? std::stoul(argv[3]) : 1;   // 1 → n=1024, 2 → n=2048 (2 tiles/poly)
    const uint32_t seed   = (argc > 4) ? std::stoul(argv[4]) : 1;
    const uint32_t P = rest * N0, S = HALVES * ELEMS;              // S output slot-tiles; input = P*HALVES tiles
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> di(0, 0xFFFFFFu);

    // poly[p][s] random; poly p's slot s lives in input tile p*HALVES + s/1024 at faceidx((s%1024)/32,(s%1024)%32)
    std::vector<std::vector<uint32_t>> poly(P, std::vector<uint32_t>(S));
    std::vector<uint32_t> Xin((size_t)P * HALVES * ELEMS, 0);
    for (uint32_t p = 0; p < P; ++p) for (uint32_t s = 0; s < S; ++s) { poly[p][s] = di(rng);
        Xin[((size_t)p * HALVES + s / ELEMS) * ELEMS + faceidx((s % ELEMS) >> 5, (s % ELEMS) & 31)] = poly[p][s]; }
    // golden: out slot-tile j has (r,k) at faceidx(r,k) = poly[k*rest+r][j]
    std::vector<uint32_t> gold((size_t)S * ELEMS, 0);
    for (uint32_t j = 0; j < S; ++j) for (uint32_t k = 0; k < N0; ++k) for (uint32_t r = 0; r < rest; ++r)
        gold[(size_t)j * ELEMS + faceidx(r, k)] = poly[k * rest + r][j];

    bool pass = true;
    try {
        auto md = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = md->mesh_command_queue();
        distributed::MeshCoordinateRange dr(md->shape());
        constexpr CoreCoord core{0, 0};
        auto mkbuf = [&](size_t n) { distributed::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = n * I32}, d, md.get()); };
        auto b_in = mkbuf((size_t)P * HALVES), b_out = mkbuf(S);
        distributed::EnqueueWriteMeshBuffer(cq, b_in, Xin, false);
        auto grid = md->compute_with_storage_grid_size();
        const uint32_t GX = grid.x, GY = grid.y, C = GX * GY;
        const uint32_t per = (ELEMS + C - 1) / C;             // slot-tiles per core, per half (1024 slots/half)

        // DB mapping: out[j](r,k)=poly[k*rest+r][j] → ROWS=rest,COLS=N0,SR=1,SC=rest.  One pass per half h:
        //   in_stride=HALVES, in_off=h (select poly's sub-tile); out_tile_base=h*1024; slot_start local 0..1023.
        for (uint32_t h = 0; h < HALVES; ++h) {
            Program p = CreateProgram();
            CoreRange cores(CoreCoord{0, 0}, CoreCoord{GX - 1, GY - 1});
            CircularBufferConfig c((per + 1) * I32, {{CBIndex::c_0, DataFormat::UInt32}}); c.set_page_size(CBIndex::c_0, I32);
            CreateCircularBuffer(p, cores, c);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reshuffle_gather_mc.cpp", cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            for (uint32_t gy = 0; gy < GY; ++gy) for (uint32_t gx = 0; gx < GX; ++gx) {
                uint32_t idx = gy * GX + gx, slot_start = idx * per;
                uint32_t n_out = (slot_start >= ELEMS) ? 0 : std::min(per, ELEMS - slot_start);
                SetRuntimeArgs(p, rd, CoreCoord{gx, gy}, {(uint32_t)b_in->address(), (uint32_t)b_out->address(),
                    rest, N0, 1u, rest, slot_start, n_out, h * ELEMS, HALVES, h});
            }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false);
        }
        distributed::Finish(cq);

        std::vector<uint32_t> out; distributed::EnqueueReadMeshBuffer(cq, out, b_out, true);
        uint32_t mism = 0;
        for (size_t i = 0; i < (size_t)S * ELEMS; ++i) if (out[i] != gold[i]) { ++mism; }
        fmt::print("poly→slot reshuffle  rest={} N0={} P={} S={}  validation: {}  ({} bad)\n", rest, N0, P, S, mism == 0 ? "OK" : "FAILED", mism);
        pass &= (mism == 0); pass &= md->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
