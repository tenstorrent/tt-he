// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PIR first-dimension SCAN throughput benchmark — the dominant, DB-linear OnionPIR op, run as an
// int8 matmul on the Tensix matrix engine and SHARDED across 1..32 cores.  Measures how fast TT
// streams+contracts the database (the "why TT for private retrieval" headline).
//
// Workload: OT output plane-tiles, each C_o = Σ_{k<KB} A[o,k] @ B[k]  (int8 32×32 -> int32,
// K-accumulated).  A = DB radix-128 limb tiles (unique per output tile — the streamed DB), B =
// query limb tiles (shared).  DB bytes streamed = OT·KB·1024.  The radix-128 recombine + mod-q
// reduction is the separately-validated SFPU tail (matmul_fbc/recombine_rns) — the scan is
// matmul/DB-bandwidth bound, which is what this measures.  Correctness of the int8 matmul-accumulate
// is validated vs a CPU golden on a sample of tiles.
//
//   ./build/test_scan_mc [OT] [KB]     (default OT=8192 tiles, KB=16 -> 512-wide contraction)

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t T = 32, ELEMS = 1024, I8 = ELEMS, I32 = ELEMS * 4;
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
    const uint32_t OT = (argc > 1) ? std::stoul(argv[1]) : 8192;   // total output plane-tiles
    const uint32_t KB = (argc > 2) ? std::stoul(argv[2]) : 16;     // contraction tiles (KB*32 wide)
    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GXmax = grid.x, GYmax = grid.y;
        fmt::print("compute grid: {}x{} = {} cores; OT={} KB={} (contraction={} )\n",
                   GXmax, GYmax, GXmax * GYmax, OT, KB, KB * 32);

        auto mkbuf = [&](uint32_t n, uint32_t page) {
            distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = n * page}, d, mesh.get()); };

        // ---- build A (OT*KB DB tiles) + B (KB query tiles), int8 in [0,63], tilized ----
        std::mt19937 rng(0);
        std::uniform_int_distribution<int> di(0, 63);
        std::vector<int8_t> Ablob((size_t)OT * KB * ELEMS), Bblob((size_t)KB * ELEMS);
        for (size_t t = 0; t < (size_t)OT * KB; ++t) { std::vector<int8_t> nat(ELEMS);
            for (uint32_t i = 0; i < ELEMS; ++i) nat[i] = (int8_t)di(rng); auto tl = tilize(nat);
            std::copy(tl.begin(), tl.end(), Ablob.begin() + t * ELEMS); }
        for (uint32_t t = 0; t < KB; ++t) { std::vector<int8_t> nat(ELEMS);
            for (uint32_t i = 0; i < ELEMS; ++i) nat[i] = (int8_t)di(rng); auto tl = tilize(nat);
            std::copy(tl.begin(), tl.end(), Bblob.begin() + t * ELEMS); }
        auto b_A = mkbuf(OT * KB, I8), b_B = mkbuf(KB, I8), b_out = mkbuf(OT, I32);
        distributed::EnqueueWriteMeshBuffer(cq, b_A, Ablob, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_B, Bblob, false);

        // one scan run over C cores; returns wall-clock seconds (device drained).
        auto run = [&](uint32_t GX, uint32_t GY, bool time_it) -> double {
            const uint32_t C = GX * GY, n_out = OT / C;
            Program p = CreateProgram();
            CoreRange cores(CoreCoord{0, 0}, CoreCoord{GX - 1, GY - 1});
            CircularBufferConfig c0(4 * I8, {{CBIndex::c_0, tt::DataFormat::Int8}});  c0.set_page_size(CBIndex::c_0, I8);
            CircularBufferConfig c1(4 * I8, {{CBIndex::c_1, tt::DataFormat::Int8}});  c1.set_page_size(CBIndex::c_1, I8);
            CircularBufferConfig c16(4 * I32, {{CBIndex::c_16, tt::DataFormat::Int32}}); c16.set_page_size(CBIndex::c_16, I32);
            CreateCircularBuffer(p, cores, c0); CreateCircularBuffer(p, cores, c1); CreateCircularBuffer(p, cores, c16);
            std::vector<uint32_t> rc; TensorAccessorArgs(*b_A).append_to(rc); TensorAccessorArgs(*b_B).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_scan_mm.cpp", cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/scan_mm.cpp", cores,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true});
            for (uint32_t gy = 0; gy < GY; ++gy) for (uint32_t gx = 0; gx < GX; ++gx) {
                uint32_t idx = gy * GX + gx, o0 = idx * n_out; CoreCoord c{gx, gy};
                SetRuntimeArgs(p, rd, c, {(uint32_t)b_A->address(), (uint32_t)b_B->address(), o0, n_out, KB});
                SetRuntimeArgs(p, wr, c, {(uint32_t)b_out->address(), n_out, o0});
                SetRuntimeArgs(p, cp, c, {n_out, KB});
            }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p));
            auto t0 = std::chrono::high_resolution_clock::now();
            distributed::EnqueueMeshWorkload(cq, wl, false);
            distributed::Finish(cq);
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double>(t1 - t0).count();
        };

        // ---- validate int8 matmul-accumulate vs CPU golden (C=1, sample of output tiles) ----
        run(1, 1, false);
        std::vector<uint32_t> out; distributed::EnqueueReadMeshBuffer(cq, out, b_out, true);
        uint32_t mism = 0; const uint32_t NS = std::min<uint32_t>(OT, 32);
        for (uint32_t o = 0; o < NS; ++o) {
            std::vector<int64_t> C(ELEMS, 0);
            for (uint32_t k = 0; k < KB; ++k) {
                const int8_t* A = Ablob.data() + ((size_t)o * KB + k) * ELEMS;
                const int8_t* B = Bblob.data() + (size_t)k * ELEMS;
                std::vector<int8_t> An(ELEMS), Bn(ELEMS);
                { std::vector<int8_t> a(A, A + ELEMS), b(B, B + ELEMS); An = untilize(a); Bn = untilize(b); }
                for (uint32_t i = 0; i < T; ++i) for (uint32_t j = 0; j < T; ++j) { int64_t s = 0;
                    for (uint32_t m = 0; m < T; ++m) s += (int64_t)An[i * T + m] * Bn[m * T + j]; C[i * T + j] += s; }
            }
            std::vector<uint32_t> raw(out.begin() + (size_t)o * ELEMS, out.begin() + (size_t)(o + 1) * ELEMS);
            auto got = untilize(raw);
            for (uint32_t i = 0; i < ELEMS; ++i) if ((int32_t)got[i] != (int32_t)C[i]) { if (mism < 6)
                fmt::print("  MISMATCH tile{} [{}]: got {} exp {}\n", o, i, (int32_t)got[i], (int32_t)C[i]); ++mism; }
        }
        fmt::print("int8 matmul-accumulate validation ({} tiles): {}\n", NS, mism == 0 ? "OK" : "FAILED");
        pass &= (mism == 0);

        // ---- throughput scaling: 1,2,4,8,16,32 cores (rectangular tiles within the grid) ----
        const double db_bytes = (double)OT * KB * ELEMS;               // int8 DB bytes streamed
        const double macs = (double)OT * KB * T * T * T;               // int8 multiply-accumulates
        fmt::print("\n cores |  GX×GY |   time(ms) |   DB GB/s |  int8 GMAC/s | speedup\n");
        fmt::print("-------+--------+------------+-----------+--------------+--------\n");
        struct Cfg { uint32_t gx, gy; };
        std::vector<Cfg> cfgs = {{1,1},{2,1},{4,1},{8,1},{8,2},{8,4}};
        double t1core = 0;
        for (auto cf : cfgs) {
            if (cf.gx > GXmax || cf.gy > GYmax) continue;
            uint32_t C = cf.gx * cf.gy;
            if (OT % C != 0) continue;
            run(cf.gx, cf.gy, false);                                   // warm-up (JIT + caches)
            double best = 1e9; for (int it = 0; it < 3; ++it) best = std::min(best, run(cf.gx, cf.gy, true));
            if (C == 1) t1core = best;
            fmt::print(" {:4}  | {:2}×{:<2}  | {:9.3f}  | {:8.2f}  | {:11.2f}  | {:5.2f}×\n",
                       C, cf.gx, cf.gy, best * 1e3, db_bytes / best / 1e9, macs / best / 1e9,
                       t1core > 0 ? t1core / best : 1.0);
        }
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
    return pass ? 0 : 1;
}
