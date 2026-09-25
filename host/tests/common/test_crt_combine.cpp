// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Unit test: on-device CRT-combine (#13), 3-stage pipeline through DRAM.
//   res1=x mod q1, res2=x mod q2  --A: garner_diff-->  diff = (res2 - res1) mod q2
//                                 --B: modmul_compute-> t = diff · q1^{-1} mod q2
//                                 --C: crt_mul_split--> X0=X&(2^28-1), X1=X>>28,  X = q1·t + res1
// Validated vs the golden radix-2^28 limbs (lb0,lb1) from reference/pir/gen_gadget_case.py.
//
//   ./build/test_crt_combine [case.bin]    (default reference/pir/gadget_k2_l8.bin)

#include <cstdint>
#include <cstdio>
#include <fstream>
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
constexpr uint32_t T = 32, ELEMS = 1024;
std::vector<uint32_t> tilize(const uint32_t* v) {
    std::vector<uint32_t> o(ELEMS, 0);
    for (uint32_t r = 0; r < T; ++r)
        for (uint32_t c = 0; c < T; ++c)
            o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = v[r * T + c];
    return o;
}
uint32_t untidx(uint32_t r, uint32_t c) { return ((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16); }

// one compute-program stage: reader_ab (A->c_0, B->c_1) + compute + writer_i32.
using MeshBuf = std::shared_ptr<distributed::MeshBuffer>;
}  // namespace

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "reference/pir/gadget_k2_l8.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
    auto ru = [&](uint32_t& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    uint32_t magic, ver, n, ell, nvec, B, K, log2b;
    ru(magic); ru(ver); ru(n); ru(ell); ru(nvec); ru(B); ru(K); ru(log2b);
    if (magic != 0x47414447u || ver != 2u || K != 2u) { fmt::print(stderr, "need V2 K=2 case\n"); return 2; }
    uint32_t q1, q2; ru(q1); ru(q2);                       // K=2 primes
    const uint32_t NT = n / ELEMS, G = nvec * NT;
    fmt::print("case {}: n={} nvec={} q1={} q2={} (NT={}, G={})\n", path, n, nvec, q1, q2, NT, G);

    std::vector<uint32_t> res1(nvec * n), res2(nvec * n), lb0(nvec * n), lb1(nvec * n);
    for (uint32_t v = 0; v < nvec; ++v) {
        f.read(reinterpret_cast<char*>(res1.data() + v * n), 4 * n);
        f.read(reinterpret_cast<char*>(res2.data() + v * n), 4 * n);
        f.read(reinterpret_cast<char*>(lb0.data() + v * n), 4 * n);
        f.read(reinterpret_cast<char*>(lb1.data() + v * n), 4 * n);
        f.seekg(4 * ell * n, std::ios::cur);   // skip digits
    }

    // constants
    mont::MontgomeryContext ctx2(q2);
    const uint32_t qinv2 = ctx2.q_inv_32(), qlow2 = q2 & 0xFFFFu, qhigh2 = q2 >> 16;
    const uint32_t inv = (uint32_t)mont::modular_inverse(q1, q2);       // q1^{-1} mod q2
    const uint32_t inv_mont = ctx2.to_montgomery(inv);                  // so modmul gives diff·inv
    const uint32_t q1L = q1 & 0xFFFFu, q1H = q1 >> 16;

    // tilized inputs
    std::vector<uint32_t> in_r1(G * ELEMS), in_r2(G * ELEMS), in_inv(G * ELEMS), par(2 * ELEMS);
    for (uint32_t v = 0; v < nvec; ++v)
        for (uint32_t tt = 0; tt < NT; ++tt) {
            uint32_t g = v * NT + tt;
            auto a = tilize(res1.data() + v * n + tt * ELEMS);
            auto b = tilize(res2.data() + v * n + tt * ELEMS);
            std::copy(a.begin(), a.end(), in_r1.begin() + g * ELEMS);
            std::copy(b.begin(), b.end(), in_r2.begin() + g * ELEMS);
        }
    for (auto& x : in_inv) x = inv_mont;              // constant tile, filled everywhere
    for (uint32_t i = 0; i < ELEMS; ++i) { par[i] = q1L; par[ELEMS + i] = q1H; }

    bool pass = true;
    try {
        constexpr uint32_t i32 = ELEMS * 4;
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto range = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        distributed::DeviceLocalBufferConfig dram{.page_size = i32, .buffer_type = BufferType::DRAM};
        auto mk = [&](uint32_t ntiles) {
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = ntiles * i32}, dram, mesh.get());
        };
        auto bR1 = mk(G), bR2 = mk(G), bInv = mk(G), bPar = mk(2), bDiff = mk(G), bT = mk(G), bOut = mk(2 * G);

        distributed::EnqueueWriteMeshBuffer(cq, bR1, in_r1, false);
        distributed::EnqueueWriteMeshBuffer(cq, bR2, in_r2, false);
        distributed::EnqueueWriteMeshBuffer(cq, bInv, in_inv, false);
        distributed::EnqueueWriteMeshBuffer(cq, bPar, par, false);

        auto mkcb = [&](Program& p, CBIndex idx, uint32_t depth) {
            CircularBufferConfig c(depth * i32, {{idx, DataFormat::UInt32}}); c.set_page_size(idx, i32);
            CreateCircularBuffer(p, core, c);
        };

        // ---- Stage A: garner_diff ----
        {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, 2); mkcb(p, CBIndex::c_1, 2); mkcb(p, CBIndex::c_16, 2);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/garner_diff.cpp", core,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)bR1->address(), (uint32_t)bR2->address(), G, 0u, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)bDiff->address(), G, 0u});
            SetRuntimeArgs(p, cp, core, {q2, G});
            distributed::MeshWorkload wl; wl.add_program(range, std::move(p));
            distributed::EnqueueMeshWorkload(cq, wl, false);
        }
        // ---- Stage B: modmul_compute (t = diff · inv_mont mod q2) ----
        {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, 2); mkcb(p, CBIndex::c_1, 2); mkcb(p, CBIndex::c_16, 2);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp", core,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)bDiff->address(), (uint32_t)bInv->address(), G, 0u, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)bT->address(), G, 0u});
            SetRuntimeArgs(p, cp, core, {G, q2, qinv2, qlow2, qhigh2});
            distributed::MeshWorkload wl; wl.add_program(range, std::move(p));
            distributed::EnqueueMeshWorkload(cq, wl, false);
        }
        // ---- Stage C: crt_mul_split (X = q1·t + r1 -> X0,X1) ----
        {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, 2); mkcb(p, CBIndex::c_1, 2); mkcb(p, CBIndex::c_2, 2); mkcb(p, CBIndex::c_16, 4);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_crt.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/crt_mul_split.cpp", core,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)bT->address(), (uint32_t)bR1->address(), (uint32_t)bPar->address(), G});
            SetRuntimeArgs(p, wr, core, {(uint32_t)bOut->address(), 2 * G, 0u});
            SetRuntimeArgs(p, cp, core, {G});
            distributed::MeshWorkload wl; wl.add_program(range, std::move(p));
            distributed::EnqueueMeshWorkload(cq, wl, false);
        }

        std::vector<uint32_t> out;
        distributed::EnqueueReadMeshBuffer(cq, out, bOut, true);   // 2*G int32 tiles: (X0,X1) per g

        uint32_t mism = 0, checked = 0;
        for (uint32_t v = 0; v < nvec; ++v)
            for (uint32_t tt = 0; tt < NT; ++tt) {
                uint32_t g = v * NT + tt;
                const uint32_t* X0 = out.data() + (2 * g + 0) * ELEMS;
                const uint32_t* X1 = out.data() + (2 * g + 1) * ELEMS;
                for (uint32_t r = 0; r < T; ++r)
                    for (uint32_t c = 0; c < T; ++c) {
                        uint32_t g0 = X0[untidx(r, c)], g1 = X1[untidx(r, c)];
                        uint32_t e0 = lb0[v * n + tt * ELEMS + r * T + c];
                        uint32_t e1 = lb1[v * n + tt * ELEMS + r * T + c];
                        if (g0 != e0 || g1 != e1) {
                            if (mism < 8)
                                fmt::print("  MISMATCH v{} tile{} ({},{}): X0 got {} exp {} | X1 got {} exp {}\n",
                                           v, tt, r, c, g0, e0, g1, e1);
                            ++mism;
                        }
                        ++checked;
                    }
            }
        fmt::print("crt_combine: mismatches = {}/{}\n", mism, checked);
        pass = (mism == 0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
