// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// test_ir_rns — validate kernels_ir crt_combine (garner_diff -> mul_modq -> crt_mul_split)
// and gadget_decompose on device, self-contained (own C++ golden, no .bin).
// Primes q1>q2 (28-bit); q = q1*q2 (~56-bit). GADGET_B=1024, ELL=6 (kernel defaults).
//   ./build/test_ir_rns [G_tiles]
//
// crt_combine golden: x -> (X0=x&(2^28-1), X1=x>>28) via CRT of (x%q1, x%q2).
// gadget golden:      balanced base-B signed digits of X, Σ d_k B^k = X.

#include <cstdint>
#include <random>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#error "OVERRIDE_KERNEL_PREFIX must be set by CMake"
#endif

namespace {
constexpr uint32_t T = 32, ELEMS = 1024, I32 = ELEMS * 4;
constexpr uint32_t GB = 1024, LOG2B = 10, ELL = 6;     // must match gadget_decompose.cpp defaults
constexpr uint64_t q1 = 251658241ULL, q2 = 201326593ULL;   // 28-bit NTT primes, q1 > q2

std::vector<uint32_t> tilize(const std::vector<uint32_t>& v) {
    std::vector<uint32_t> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = v[r * T + c];
    return o; }
uint32_t untidx(uint32_t r, uint32_t c) { return ((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16); }

CircularBufferConfig cbc(CBIndex i, uint32_t depth) {
    CircularBufferConfig c(depth * I32, {{i, DataFormat::UInt32}}); c.set_page_size(i, I32); return c; }

using Buf = std::shared_ptr<distributed::MeshBuffer>;
}  // namespace

int main(int argc, char** argv) {
    const uint32_t G = (argc > 1) ? std::stoul(argv[1]) : 2u;   // # coefficient-tiles
    const uint32_t N = G * ELEMS;
    std::mt19937_64 rng(5);
    std::uniform_int_distribution<uint64_t> dq(0, q1 * q2 - 1);

    // ---- host inputs + goldens ----
    std::vector<uint64_t> x(N);
    std::vector<uint32_t> res1(N), res2(N), gX0(N), gX1(N);
    std::vector<std::vector<int32_t>> gdig(ELL, std::vector<int32_t>(N));
    for (uint32_t i = 0; i < N; ++i) {
        uint64_t xi = dq(rng); x[i] = xi;
        res1[i] = (uint32_t)(xi % q1); res2[i] = (uint32_t)(xi % q2);
        gX0[i] = (uint32_t)(xi & ((1u << 28) - 1)); gX1[i] = (uint32_t)(xi >> 28);
        uint64_t xx = xi;                                    // balanced base-B digits
        for (uint32_t k = 0; k < ELL; ++k) {
            int64_t raw = (int64_t)(xx & (GB - 1));
            int64_t d = (raw >= (int64_t)(GB / 2)) ? raw - (int64_t)GB : raw;
            xx = (uint64_t)(((int64_t)xx - d) >> LOG2B);
            gdig[k][i] = (int32_t)d;
        }
    }

    mont::MontgomeryContext ctx2((uint32_t)q2);
    const uint32_t inv_mont = ctx2.to_montgomery((uint32_t)mont::modular_inverse((uint32_t)q1, (uint32_t)q2));
    const uint32_t q1L = q1 & 0xFFFFu, q1H = q1 >> 16;

    // tilized device buffers
    auto tilize_into = [&](const std::vector<uint32_t>& src) {
        std::vector<uint32_t> o(N);
        for (uint32_t g = 0; g < G; ++g) {
            std::vector<uint32_t> s(src.begin() + g * ELEMS, src.begin() + (g + 1) * ELEMS);
            auto t = tilize(s); std::copy(t.begin(), t.end(), o.begin() + g * ELEMS);
        }
        return o; };
    auto in_r1 = tilize_into(res1), in_r2 = tilize_into(res2);
    std::vector<uint32_t> in_inv(N, inv_mont), par(2 * ELEMS);
    for (uint32_t i = 0; i < ELEMS; ++i) { par[i] = q1L; par[ELEMS + i] = q1H; }
    // gadget input: X0,X1 buffers (tilized)
    auto in_X0 = tilize_into(gX0), in_X1 = tilize_into(gX1);

    auto md = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange dr(md->shape());
    constexpr CoreCoord core{0, 0};
    distributed::DeviceLocalBufferConfig dram{.page_size = I32, .buffer_type = BufferType::DRAM};
    auto mk = [&](uint32_t n) { return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = n * I32}, dram, md.get()); };
    int fails = 0;

    auto DM1 = [&](KernelHandle) {};  // noop
    auto rd_binop = [&](Program& p, Buf a, Buf b) {
        auto k = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_binop.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        SetRuntimeArgs(p, k, core, {(uint32_t)a->address(), (uint32_t)b->address(), G, 0u}); };
    auto wr_out = [&](Program& p, Buf o, uint32_t nt) {
        auto k = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(p, k, core, {(uint32_t)o->address(), nt, 0u}); };
    auto cfg = ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false};
    auto run = [&](Program&& p) { distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false); };

    // ================= crt_combine (3-stage pipeline) =================
    {
        auto bR1 = mk(G), bR2 = mk(G), bInv = mk(G), bPar = mk(2), bDiff = mk(G), bT = mk(G), bOut = mk(2 * G);
        distributed::EnqueueWriteMeshBuffer(cq, bR1, in_r1, false);
        distributed::EnqueueWriteMeshBuffer(cq, bR2, in_r2, false);
        distributed::EnqueueWriteMeshBuffer(cq, bInv, in_inv, false);
        distributed::EnqueueWriteMeshBuffer(cq, bPar, par, false);

        // A: garner_diff (r1->c_0, r2->c_1) args {q2, G}
        { Program p = CreateProgram();
          CreateCircularBuffer(p, core, cbc(CBIndex::c_0, 2)); CreateCircularBuffer(p, core, cbc(CBIndex::c_1, 2)); CreateCircularBuffer(p, core, cbc(CBIndex::c_16, 2));
          rd_binop(p, bR1, bR2); wr_out(p, bDiff, G);
          auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/garner_diff.cpp", core, cfg);
          SetRuntimeArgs(p, cp, core, {(uint32_t)q2, G}); run(std::move(p)); }
        // B: mul_modq  t = diff * inv_mont mod q2  (one operand Montgomery -> plain product)
        { Program p = CreateProgram();
          CreateCircularBuffer(p, core, cbc(CBIndex::c_0, 2)); CreateCircularBuffer(p, core, cbc(CBIndex::c_1, 2)); CreateCircularBuffer(p, core, cbc(CBIndex::c_16, 2));
          rd_binop(p, bDiff, bInv); wr_out(p, bT, G);
          auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/mul_modq.cpp", core, cfg);
          SetRuntimeArgs(p, cp, core, {G, (uint32_t)q2, ctx2.q_inv_32(), (uint32_t)(q2 & 0xFFFF), (uint32_t)(q2 >> 16)}); run(std::move(p)); }
        // C: crt_mul_split (t->c_0, r1->c_1, par->c_2) args {G} -> (X0,X1)
        { Program p = CreateProgram();
          CreateCircularBuffer(p, core, cbc(CBIndex::c_0, 2)); CreateCircularBuffer(p, core, cbc(CBIndex::c_1, 2));
          CreateCircularBuffer(p, core, cbc(CBIndex::c_2, 2)); CreateCircularBuffer(p, core, cbc(CBIndex::c_16, 4));
          auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_crt.cpp", core,
              DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
          SetRuntimeArgs(p, rd, core, {(uint32_t)bT->address(), (uint32_t)bR1->address(), (uint32_t)bPar->address(), G});
          wr_out(p, bOut, 2 * G);
          auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/crt_mul_split.cpp", core, cfg);
          SetRuntimeArgs(p, cp, core, {G}); run(std::move(p)); }

        std::vector<uint32_t> out; distributed::EnqueueReadMeshBuffer(cq, out, bOut, true);
        uint32_t bad = 0;
        for (uint32_t g = 0; g < G; ++g) for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c) {
            uint32_t g0 = out[(2 * g + 0) * ELEMS + untidx(r, c)], g1 = out[(2 * g + 1) * ELEMS + untidx(r, c)];
            uint32_t i = g * ELEMS + r * T + c;
            bad += (g0 != gX0[i] || g1 != gX1[i]);
        }
        fmt::print("  crt_combine      : {}  ({} / {} bad)\n", bad ? "FAIL" : "OK", bad, N); fails += !!bad;
    }

    // ================= gadget_decompose =================
    {
        auto b0 = mk(G), b1 = mk(G), bDig = mk(G * ELL);
        distributed::EnqueueWriteMeshBuffer(cq, b0, in_X0, false);
        distributed::EnqueueWriteMeshBuffer(cq, b1, in_X1, false);
        Program p = CreateProgram();
        CreateCircularBuffer(p, core, cbc(CBIndex::c_0, 4));
        CreateCircularBuffer(p, core, cbc(CBIndex::c_16, 2 * ELL));
        auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_gadget.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        SetRuntimeArgs(p, rd, core, {(uint32_t)b0->address(), (uint32_t)b1->address(), G});
        wr_out(p, bDig, G * ELL);
        auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/gadget_decompose.cpp", core, cfg);
        SetRuntimeArgs(p, cp, core, {G}); run(std::move(p));

        std::vector<uint32_t> out; distributed::EnqueueReadMeshBuffer(cq, out, bDig, true);
        uint32_t bad = 0;
        for (uint32_t g = 0; g < G; ++g) for (uint32_t k = 0; k < ELL; ++k)
            for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c) {
                int32_t got = (int32_t)out[(g * ELL + k) * ELEMS + untidx(r, c)];
                bad += (got != gdig[k][g * ELEMS + r * T + c]);
            }
        fmt::print("  gadget_decompose : {}  ({} / {} bad)\n", bad ? "FAIL" : "OK", bad, N * ELL); fails += !!bad;
    }

    md->close();
    if (fails) fmt::print("\nFAIL: {} op(s) mismatched\n", fails);
    else       fmt::print("\nCRT + GADGET OK\n");
    return fails ? 1 : 0;
}
