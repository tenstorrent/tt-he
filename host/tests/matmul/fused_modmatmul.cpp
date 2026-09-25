// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Standalone single-core 32x32 modular matmul  C = (A @ B) mod q, FULLY FUSED on
// device: one program does decompose(B) -> matmul(folded A) -> recombine, with the
// int8 limb tiles and int32 plane tiles kept in on-core L1 CBs (no DRAM round-trip
// between phases, one dispatch). Validates against an exact __int128 reference and
// reports steady-state device latency.
//
// Orientation (matches the fused stage-A kernel): A is the constant LHS, folded
// offline into 16 int8 tiles  Afold[p*NB+r] = limb_p(128^r * A mod q); B is the
// dynamic operand, written once as one int32 tile and decomposed on device.
//
//   ./build/fused_modmatmul [q] [seed]
//
// NOTE: the recombine's lazy reduction is tuned for a 28-bit prime near 2^28
// (e.g. 167772161). Smaller primes need the dual-mulmod recombine (see notes).

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
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t T = 32, NB = 4, ELEMS = 1024;
constexpr uint32_t I8 = ELEMS * 1, I32 = ELEMS * 4;
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
    bool pass = true;
    try {
        const uint64_t q = (argc > 1) ? std::stoull(argv[1]) : 167772161ULL;
        const uint32_t seed = (argc > 2) ? std::stoul(argv[2]) : 0;
        mont::MontgomeryContext ctx((uint32_t)q);
        const uint32_t QINV = ctx.q_inv_32(), QLO = q & 0xFFFFu, QHI = q >> 16;
        const uint32_t K14 = ctx.to_montgomery(16384u);     // mont(2^14) for the recombine

        // ---- inputs A, B in [0, q) ----
        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)q - 1);
        std::vector<uint32_t> A(ELEMS), B(ELEMS);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);

        // ---- exact golden  C = (A @ B) mod q ----
        std::vector<uint32_t> Cref(ELEMS);
        for (uint32_t i = 0; i < T; ++i)
            for (uint32_t j = 0; j < T; ++j) {
                unsigned __int128 acc = 0;
                for (uint32_t k = 0; k < T; ++k)
                    acc += (unsigned __int128)A[i * T + k] * B[k * T + j];
                Cref[i * T + j] = (uint32_t)(acc % q);
            }

        // ---- fold A into the constant LHS: Afold[p*NB+r] = limb_p(128^r * A mod q) ----
        auto limb = [](uint32_t v, uint32_t p) -> int8_t { return (int8_t)((v >> (7 * p)) & 127); };
        std::vector<int8_t> Afold(NB * NB * ELEMS);
        for (uint32_t p = 0; p < NB; ++p)
            for (uint32_t r = 0; r < NB; ++r) {
                std::vector<int8_t> l(ELEMS);
                for (uint32_t i = 0; i < ELEMS; ++i)
                    l[i] = limb((uint32_t)((POW[r] * A[i]) % q), p);
                auto t = tilize(l);
                std::copy(t.begin(), t.end(), Afold.begin() + (p * NB + r) * ELEMS);
            }
        std::vector<uint32_t> Bt = tilize(B);   // B is decomposed on device

        // ---- device ----
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};

        auto mkbuf = [&](uint32_t ntiles, uint32_t page) {
            distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = ntiles * page}, d, mesh.get());
        };
        auto b_B = mkbuf(1, I32), b_Afold = mkbuf(NB * NB, I8), b_C = mkbuf(1, I32);

        auto mkcb = [&](Program& p, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
            CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, core, c); };

        Program prog = CreateProgram();
        mkcb(prog, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32);   // B (dynamic, decomposed on device)
        mkcb(prog, CBIndex::c_1, tt::DataFormat::Int8, 2, I8);      // folded A (streamed)
        mkcb(prog, CBIndex::c_2, tt::DataFormat::Int8, NB, I8);     // limbs (intermediate, L1)
        mkcb(prog, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);   // planes (intermediate, L1)
        mkcb(prog, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);  // result
        std::vector<uint32_t> rc;
        TensorAccessorArgs(*b_B).append_to(rc);
        TensorAccessorArgs(*b_Afold).append_to(rc);
        auto rd = CreateKernel(prog, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
        auto wr = CreateKernel(prog, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto cp = CreateKernel(prog, OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
        SetRuntimeArgs(prog, rd, core, {(uint32_t)b_B->address(), (uint32_t)b_Afold->address()});
        SetRuntimeArgs(prog, wr, core, {(uint32_t)b_C->address(), 1u, 0u});
        SetRuntimeArgs(prog, cp, core, {(uint32_t)q, QINV, QLO, QHI, K14});

        distributed::MeshWorkload wl;
        wl.add_program(dr, std::move(prog));

        // ---- write inputs once, run, read back ----
        distributed::EnqueueWriteMeshBuffer(cq, b_B, Bt, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Afold, Afold, false);
        distributed::EnqueueMeshWorkload(cq, wl, false);
        std::vector<uint32_t> Craw;
        distributed::EnqueueReadMeshBuffer(cq, Craw, b_C, true);
        auto C = untilize(Craw);

        // ---- validate ----
        uint32_t mism = 0;
        for (uint32_t i = 0; i < ELEMS; ++i)
            if (C[i] != Cref[i]) { if (mism < 8) fmt::print("  ({},{}) got={} exp={}\n", i / T, i % T, C[i], Cref[i]); ++mism; }
        fmt::print("fused 32x32 modular matmul  q={}  seed={}  mismatches={}/{}\n", q, seed, mism, ELEMS);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: fused modular matmul WORKS.\n");
        else fmt::print("RESULT: WRONG.\n");

        // ---- steady-state timing (100 iters, one Finish) ----
        if (pass) {
            distributed::EnqueueMeshWorkload(cq, wl, false);
            distributed::Finish(cq);  // warm
            const int iters = 100;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it) distributed::EnqueueMeshWorkload(cq, wl, false);
            distributed::Finish(cq);
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e3 / iters;
            fmt::print("timing: {:.2f} us / 32x32 modular matmul (fused, single core)\n", us);
        }
        pass &= mesh->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Exception:\n{}\n", e.what());
        return 2;
    }
    if (pass) fmt::print("Test Passed\n");
    else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
