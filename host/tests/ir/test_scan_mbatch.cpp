// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// M-BATCHED multi-core modular-matmul scan.  Each core runs matmul_modq_mbatch on M output tiles in
// ONE program (phase-major: all decomposes, then all matmuls, then all recombines — 3 engine-mode
// switches total, NOT per-tile), so a round covers C·M tiles and the OT/(C·M) dispatch rounds shrink
// M-fold vs the 1-tile-per-core path.  Each tile is an independent query-batched contract-32 modular
// matmul C_o = (DBᵀ_o @ Q_o) mod q.  Validated bit-exact vs a host golden; sweeps cores × M.
//
//   ./build/test_scan_mbatch [OT] [M] [q] [seed]

#include <chrono>
#include <cstdint>
#include <random>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#error "OVERRIDE_KERNEL_PREFIX must be set by CMake"
#endif

namespace {
constexpr uint32_t T = 32, NB = 4, ELEMS = 1024, I8 = ELEMS, I32 = ELEMS * 4;
const uint64_t POW[4] = {1, 128, 16384, 2097152};
template <typename E> std::vector<E> tilize(const std::vector<E>& in) {
    std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = in[r * T + c];
    return o; }
template <typename E> std::vector<E> untilize(const std::vector<E>& in) {
    std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[r * T + c] = in[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)];
    return o; }
}  // namespace

int main(int argc, char** argv) {
    const uint32_t OT = (argc > 1) ? std::stoul(argv[1]) : 2048;   // output tiles
    const uint32_t M  = (argc > 2) ? std::stoul(argv[2]) : 16;     // tiles per core per program
    const uint64_t q  = (argc > 3) ? std::stoull(argv[3]) : 167772161ULL;
    const uint32_t seed = (argc > 4) ? std::stoul(argv[4]) : 1;
    mont::MontgomeryContext ctx((uint32_t)q);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> di(0, (uint32_t)q - 1);
    auto limb = [](uint32_t v, uint32_t p) -> int8_t { return (int8_t)((v >> (7 * p)) & 127); };

    std::vector<int8_t> Wfold((size_t)OT * NB * NB * ELEMS);
    std::vector<uint32_t> Xin((size_t)OT * ELEMS);
    std::vector<std::vector<uint32_t>> Cg(OT, std::vector<uint32_t>(ELEMS));
    for (uint32_t o = 0; o < OT; ++o) {
        std::vector<uint32_t> A(ELEMS), B(ELEMS);
        for (auto& v : A) v = di(rng); for (auto& v : B) v = di(rng);
        for (uint32_t i = 0; i < T; ++i) for (uint32_t j = 0; j < T; ++j) {
            unsigned __int128 a = 0; for (uint32_t k = 0; k < T; ++k) a += (unsigned __int128)A[i * T + k] * B[k * T + j];
            Cg[o][i * T + j] = (uint32_t)(a % q); }
        auto Bt = tilize(B); std::copy(Bt.begin(), Bt.end(), Xin.begin() + (size_t)o * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t rr = 0; rr < NB; ++rr) {
            std::vector<int8_t> l(ELEMS);
            for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[rr] * A[i]) % q), p);
            auto t = tilize(l);
            std::copy(t.begin(), t.end(), Wfold.begin() + ((size_t)o * NB * NB + p * NB + rr) * ELEMS); }
    }

    bool pass = true;
    try {
        auto md = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = md->mesh_command_queue();
        distributed::MeshCoordinateRange dr(md->shape());
        auto grid = md->compute_with_storage_grid_size();
        const uint32_t GX = grid.x, GY = grid.y;
        fmt::print("grid {}x{} = {} cores; OT={} M={} q={}\n", GX, GY, GX * GY, OT, M, q);
        auto mkbuf = [&](uint32_t n, uint32_t page) {
            distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = n * page}, d, md.get()); };
        auto b_X = mkbuf(OT, I32), b_W = mkbuf(OT * NB * NB, I8), b_C = mkbuf(OT, I32);
        distributed::EnqueueWriteMeshBuffer(cq, b_X, Xin, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_W, Wfold, false);

        // Each core does M tiles per program; a round covers C·M tiles.
        auto run = [&](uint32_t gx, uint32_t gy, uint32_t Mb) -> double {
            const uint32_t C = gx * gy, per = C * Mb;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t base = 0; base < OT; base += per) {
                Program p = CreateProgram();
                CoreRange cores(CoreCoord{0, 0}, CoreCoord{gx - 1, gy - 1});
                auto cb = [&](CBIndex idx, DataFormat fmt, uint32_t depth, uint32_t pg) {
                    CircularBufferConfig c(depth * pg, {{idx, fmt}}); c.set_page_size(idx, pg); CreateCircularBuffer(p, cores, c); };
                cb(CBIndex::c_0, DataFormat::UInt32, Mb, I32); cb(CBIndex::c_1, DataFormat::Int8, 32, I8);
                cb(CBIndex::c_2, DataFormat::Int8, Mb * NB, I8); cb(CBIndex::c_3, DataFormat::Int32, Mb * NB, I32);
                cb(CBIndex::c_16, DataFormat::UInt32, Mb, I32);
                std::vector<uint32_t> rc; TensorAccessorArgs(*b_X).append_to(rc); TensorAccessorArgs(*b_W).append_to(rc);
                auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_mbatch.cpp", cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
                auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
                auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/matmul_modq_mbatch.cpp", cores,
                    ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
                for (uint32_t gyi = 0; gyi < gy; ++gyi) for (uint32_t gxi = 0; gxi < gx; ++gxi) {
                    uint32_t idx = gyi * gx + gxi; uint32_t x_base = base + idx * Mb;
                    if (x_base >= OT) continue;
                    uint32_t m_here = std::min(Mb, OT - x_base);
                    CoreCoord c{gxi, gyi};
                    SetRuntimeArgs(p, rd, c, {(uint32_t)b_X->address(), (uint32_t)b_W->address(), x_base, x_base * NB * NB, m_here});
                    SetRuntimeArgs(p, wr, c, {(uint32_t)b_C->address(), m_here, x_base});
                    SetRuntimeArgs(p, cp, c, {(uint32_t)q, ctx.q_inv_32(), (uint32_t)(q & 0xFFFF), (uint32_t)(q >> 16), ctx.to_montgomery(16384u), m_here});
                }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false);
            }
            distributed::Finish(cq);
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        run(1, 1, M);  // warm JIT
        std::vector<uint32_t> out; distributed::EnqueueReadMeshBuffer(cq, out, b_C, true);
        uint32_t mism = 0;
        for (uint32_t o = 0; o < OT; ++o) { auto Cm = untilize(std::vector<uint32_t>(out.begin() + (size_t)o * ELEMS, out.begin() + (size_t)(o + 1) * ELEMS));
            for (uint32_t i = 0; i < ELEMS; ++i) if (Cm[i] != Cg[o][i]) ++mism; }
        fmt::print("mbatch modular matmul validation: {}  ({} bad)\n", mism == 0 ? "OK" : "FAILED", mism);
        pass &= (mism == 0);

        fmt::print("\n cores | gx×gy | M  | time(ms) | tiles/s  | speedup\n-------+-------+----+----------+----------+--------\n");
        struct Cfg { uint32_t gx, gy; }; std::vector<Cfg> cfgs = {{1,1},{2,1},{4,1},{8,1},{8,2},{8,4},{8,8}};
        double t1c = 0;
        for (auto cf : cfgs) { if (cf.gx > GX || cf.gy > GY) continue; uint32_t C = cf.gx * cf.gy;
            run(cf.gx, cf.gy, M); double best = 1e18; for (int it = 0; it < 3; ++it) best = std::min(best, run(cf.gx, cf.gy, M));
            if (C == 1) t1c = best;
            fmt::print(" {:4}  | {:2}×{:<2} | {:2} | {:8.2f} | {:8.0f} | {:5.2f}×\n", C, cf.gx, cf.gy, M, best, OT / best * 1e3, t1c > 0 ? t1c / best : 1.0); }
        pass &= md->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
    return pass ? 0 : 1;
}
