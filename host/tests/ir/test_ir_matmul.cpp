// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// test_ir_matmul — validate kernels_ir matmul_modq (int8, param-folded) and transpose on device.
//   matmul_modq: C = (A @ B) mod q, A const folded to 16 int8 tiles, B decomposed on device.
//   transpose:   per-tile 32x32 transpose.
// Bit-exact vs an exact __int128 / logical golden.  ./build/test_ir_matmul [q] [seed]

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

CircularBufferConfig cb(CBIndex i, DataFormat f, uint32_t depth, uint32_t page) {
    CircularBufferConfig c(depth * page, {{i, f}}); c.set_page_size(i, page); return c; }
}  // namespace

int main(int argc, char** argv) {
    const uint64_t q = (argc > 1) ? std::stoull(argv[1]) : 167772161ULL;   // fused-recombine-tuned prime
    const uint32_t seed = (argc > 2) ? std::stoul(argv[2]) : 1;
    mont::MontgomeryContext ctx((uint32_t)q);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)q - 1);

    auto md = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange dr(md->shape());
    constexpr CoreCoord core{0, 0};
    auto mkbuf = [&](uint32_t n, uint32_t page) {
        distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = n * page}, d, md.get()); };
    int fails = 0;

    // ================= matmul_modq =================
    {
        std::vector<uint32_t> A(ELEMS), B(ELEMS);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
        std::vector<uint32_t> Cref(ELEMS);
        for (uint32_t i = 0; i < T; ++i) for (uint32_t j = 0; j < T; ++j) {
            unsigned __int128 acc = 0;
            for (uint32_t k = 0; k < T; ++k) acc += (unsigned __int128)A[i * T + k] * B[k * T + j];
            Cref[i * T + j] = (uint32_t)(acc % q); }
        auto limb = [](uint32_t v, uint32_t p) -> int8_t { return (int8_t)((v >> (7 * p)) & 127); };
        std::vector<int8_t> Afold(NB * NB * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
            std::vector<int8_t> l(ELEMS);
            for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[r] * A[i]) % q), p);
            auto t = tilize(l); std::copy(t.begin(), t.end(), Afold.begin() + (p * NB + r) * ELEMS); }
        auto Bt = tilize(B);

        auto b_B = mkbuf(1, I32), b_Af = mkbuf(NB * NB, I8), b_C = mkbuf(1, I32);
        Program p = CreateProgram();
        CreateCircularBuffer(p, core, cb(CBIndex::c_0, DataFormat::UInt32, 2, I32));
        CreateCircularBuffer(p, core, cb(CBIndex::c_1, DataFormat::Int8, 2, I8));
        CreateCircularBuffer(p, core, cb(CBIndex::c_2, DataFormat::Int8, NB, I8));
        CreateCircularBuffer(p, core, cb(CBIndex::c_3, DataFormat::Int32, NB, I32));
        CreateCircularBuffer(p, core, cb(CBIndex::c_16, DataFormat::UInt32, 2, I32));
        std::vector<uint32_t> rc;
        TensorAccessorArgs(*b_B).append_to(rc); TensorAccessorArgs(*b_Af).append_to(rc);
        auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
        auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/matmul_modq.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
        SetRuntimeArgs(p, rd, core, {(uint32_t)b_B->address(), (uint32_t)b_Af->address()});
        SetRuntimeArgs(p, wr, core, {(uint32_t)b_C->address(), 1u, 0u});
        SetRuntimeArgs(p, cp, core, {(uint32_t)q, ctx.q_inv_32(), (uint32_t)(q & 0xFFFF), (uint32_t)(q >> 16), ctx.to_montgomery(16384u)});
        distributed::MeshWorkload wl; wl.add_program(dr, std::move(p));
        distributed::EnqueueWriteMeshBuffer(cq, b_B, Bt, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Af, Afold, false);
        distributed::EnqueueMeshWorkload(cq, wl, false);
        std::vector<uint32_t> Craw; distributed::EnqueueReadMeshBuffer(cq, Craw, b_C, true);
        auto C = untilize(Craw);
        uint32_t bad = 0; for (uint32_t i = 0; i < ELEMS; ++i) bad += (C[i] != Cref[i]);
        fmt::print("  matmul_modq : {}  ({} / {} bad)\n", bad ? "FAIL" : "OK", bad, ELEMS); fails += !!bad;
    }

    // ================= transpose =================
    {
        const uint32_t W = 4;
        std::vector<uint32_t> in(W * ELEMS), exp(W * ELEMS);
        for (uint32_t t = 0; t < W; ++t) {
            std::vector<uint32_t> M(ELEMS), Mt(ELEMS);
            for (auto& v : M) v = dist(rng);
            for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c) Mt[r * T + c] = M[c * T + r];
            auto ti = tilize(M), te = tilize(Mt);
            std::copy(ti.begin(), ti.end(), in.begin() + t * ELEMS);
            std::copy(te.begin(), te.end(), exp.begin() + t * ELEMS);
        }
        auto b_in = mkbuf(W, I32), b_out = mkbuf(W, I32);
        Program p = CreateProgram();
        CreateCircularBuffer(p, core, cb(CBIndex::c_0, DataFormat::UInt32, 2, I32));
        CreateCircularBuffer(p, core, cb(CBIndex::c_16, DataFormat::UInt32, 2, I32));
        auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_unop.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_out.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/transpose.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
        SetRuntimeArgs(p, rd, core, {(uint32_t)b_in->address(), W, 0u});
        SetRuntimeArgs(p, wr, core, {(uint32_t)b_out->address(), W, 0u});
        SetRuntimeArgs(p, cp, core, {W});
        distributed::MeshWorkload wl; wl.add_program(dr, std::move(p));
        distributed::EnqueueWriteMeshBuffer(cq, b_in, in, false);
        distributed::EnqueueMeshWorkload(cq, wl, false);
        std::vector<uint32_t> out; distributed::EnqueueReadMeshBuffer(cq, out, b_out, true);
        uint32_t bad = 0; for (uint32_t i = 0; i < W * ELEMS; ++i) bad += (out[i] != exp[i]);
        fmt::print("  transpose   : {}  ({} / {} bad)\n", bad ? "FAIL" : "OK", bad, W * ELEMS); fails += !!bad;
    }

    md->close();
    if (fails) fmt::print("\nFAIL: {} op(s) mismatched\n", fails);
    else       fmt::print("\nMATMUL+TRANSPOSE OK\n");
    return fails ? 1 : 0;
}
