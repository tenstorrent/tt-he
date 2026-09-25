// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Unit test: on-device CRT-combine (#13) at K=3 (~84-bit q), device-resident through DRAM.
// Reconstructs X = x in [0,q), q=q0·q1·q2, from residues (r0,r1,r2) via Garner's nested form
//   X = v0 + q0·(v1 + q1·v2),  v0=r0,
//     v1 = (r1 - r0)·q0^{-1} mod q1,
//     v2 = ((r2 - r0)·q0^{-1} - v1)·q1^{-1} mod q2,
// computed as reuses of the VALIDATED K=2 kernels (garner_diff, modmul_compute, crt_mul_split)
// plus the crt_limb3 tail:
//   w = q1·v2 + v1   = crt_mul_split(t=v2, r1=v1, par=q1)          -> (w0,w1)
//   P = q0·w0 + v0   = crt_mul_split(t=w0, r1=r0, par=q0)          -> (P0,P1)
//   Q = q0·w1        = crt_mul_split(t=w1, r1=0,  par=q0)          -> (Q0,Q1)
//   X = P + Q·2^28   = crt_limb3(P,Q)                              -> (X0,X1,X2)
// Validated vs the golden radix-2^28 limbs from cpu/pir/gen_gadget_case.py --K 3.
//
//   ./build/test_crt_combine_k3 [reference/pir/gadget_k3_l8.bin]

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
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
using MeshBuf = std::shared_ptr<distributed::MeshBuffer>;
}  // namespace

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "reference/pir/gadget_k3_l8.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
    auto ru = [&](uint32_t& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    uint32_t magic, ver, n, ell, nvec, B, K, log2b;
    ru(magic); ru(ver); ru(n); ru(ell); ru(nvec); ru(B); ru(K); ru(log2b);
    if (magic != 0x47414447u || ver != 2u || K != 3u) { fmt::print(stderr, "need V2 K=3 case\n"); return 2; }
    uint32_t q0, q1, q2; ru(q0); ru(q1); ru(q2);
    const uint32_t NT = n / ELEMS, G = nvec * NT;
    fmt::print("case {}: n={} nvec={} q0={} q1={} q2={} (NT={}, G={})\n", path, n, nvec, q0, q1, q2, NT, G);

    // per vec: 3 residues, 3 limbs, ell digits (natural order)
    std::vector<std::vector<uint32_t>> res(3, std::vector<uint32_t>(nvec * n)),
                                       lb(3, std::vector<uint32_t>(nvec * n));
    for (uint32_t v = 0; v < nvec; ++v) {
        for (uint32_t j = 0; j < 3; ++j) f.read(reinterpret_cast<char*>(res[j].data() + v * n), 4 * n);
        for (uint32_t j = 0; j < 3; ++j) f.read(reinterpret_cast<char*>(lb[j].data() + v * n), 4 * n);
        f.seekg(4 * ell * n, std::ios::cur);   // skip digits
    }

    // Garner constants
    mont::MontgomeryContext ctx1(q1), ctx2(q2);
    const uint32_t inv01m = ctx1.to_montgomery((uint32_t)mont::modular_inverse(q0, q1));  // q0^{-1} mod q1
    const uint32_t inv02m = ctx2.to_montgomery((uint32_t)mont::modular_inverse(q0, q2));  // q0^{-1} mod q2
    const uint32_t inv12m = ctx2.to_montgomery((uint32_t)mont::modular_inverse(q1, q2));  // q1^{-1} mod q2

    // tilized residues
    auto tilize_stream = [&](const std::vector<uint32_t>& src) {
        std::vector<uint32_t> out(G * ELEMS);
        for (uint32_t v = 0; v < nvec; ++v)
            for (uint32_t tt = 0; tt < NT; ++tt) {
                auto t = tilize(src.data() + v * n + tt * ELEMS);
                std::copy(t.begin(), t.end(), out.begin() + (v * NT + tt) * ELEMS);
            }
        return out;
    };

    bool pass = true;
    try {
        constexpr uint32_t i32 = ELEMS * 4;
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto range = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        distributed::DeviceLocalBufferConfig dram{.page_size = i32, .buffer_type = BufferType::DRAM};
        auto mk = [&](uint32_t nt) {
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * i32}, dram, mesh.get());
        };
        auto wrb = [&](MeshBuf b, const std::vector<uint32_t>& v) { distributed::EnqueueWriteMeshBuffer(cq, b, v, false); };
        auto mkcb = [&](Program& p, CBIndex idx, uint32_t depth) {
            CircularBufferConfig c(depth * i32, {{idx, DataFormat::UInt32}}); c.set_page_size(idx, i32);
            CreateCircularBuffer(p, core, c);
        };

        // input residue + constant buffers
        auto bR0 = mk(G), bR1 = mk(G), bR2 = mk(G), bZero = mk(G);
        wrb(bR0, tilize_stream(res[0])); wrb(bR1, tilize_stream(res[1])); wrb(bR2, tilize_stream(res[2]));
        wrb(bZero, std::vector<uint32_t>(G * ELEMS, 0));
        auto cbuf = [&](uint32_t val) { auto b = mk(G); wrb(b, std::vector<uint32_t>(G * ELEMS, val)); return b; };
        auto bInv01 = cbuf(inv01m), bInv02 = cbuf(inv02m), bInv12 = cbuf(inv12m);
        auto parbuf = [&](uint32_t q) { auto b = mk(2); std::vector<uint32_t> par(2 * ELEMS);
            for (uint32_t i = 0; i < ELEMS; ++i) { par[i] = q & 0xFFFFu; par[ELEMS + i] = q >> 16; } wrb(b, par); return b; };
        auto bParQ0 = parbuf(q0), bParQ1 = parbuf(q1);

        // ---- generic stage runners (each is one Program enqueued on the shared queue) ----
        // A·B -> out via a compute kernel fed by reader_ab (A->c_0, B->c_1), n tiles out.
        auto run_ab = [&](const char* kernel, MeshBuf bA, MeshBuf bB, MeshBuf bOut,
                          const std::vector<uint32_t>& cargs) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, 2); mkcb(p, CBIndex::c_1, 2); mkcb(p, CBIndex::c_16, 2);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + std::string(kernel), core,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)bA->address(), (uint32_t)bB->address(), G, 0u, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)bOut->address(), G, 0u});
            SetRuntimeArgs(p, cp, core, cargs);
            distributed::MeshWorkload wl; wl.add_program(range, std::move(p));
            distributed::EnqueueMeshWorkload(cq, wl, false);
        };
        // crt_mul_split with strided t: X = q·t + r1 -> (X0,X1) interleaved (2G tiles).
        auto run_crt = [&](MeshBuf bT, uint32_t stride, uint32_t off, MeshBuf bR1, MeshBuf bPar, MeshBuf bOut) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, 2); mkcb(p, CBIndex::c_1, 2); mkcb(p, CBIndex::c_2, 2); mkcb(p, CBIndex::c_16, 4);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_crt_stride.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/crt_mul_split.cpp", core,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)bT->address(), stride, off, (uint32_t)bR1->address(),
                                         (uint32_t)bPar->address(), G, 0u});   // arg6 = r1_off
            SetRuntimeArgs(p, wr, core, {(uint32_t)bOut->address(), 2 * G, 0u});
            SetRuntimeArgs(p, cp, core, {G});
            distributed::MeshWorkload wl; wl.add_program(range, std::move(p));
            distributed::EnqueueMeshWorkload(cq, wl, false);
        };

        // Garner v1 = (r1 - r0)·inv01 mod q1
        auto bD01 = mk(G), bV1 = mk(G);
        run_ab("compute/garner_diff.cpp", bR0, bR1, bD01, {q1, G});                   // c0=r0(sub), c1=r1(min)
        run_ab("compute/modmul_compute.cpp", bD01, bInv01, bV1, {G, q1, ctx1.q_inv_32(), q1 & 0xFFFFu, q1 >> 16});
        // Garner v2 = ((r2 - r0)·inv02 - v1)·inv12 mod q2
        auto bA = mk(G), bBt = mk(G), bCC = mk(G), bV2 = mk(G);
        run_ab("compute/garner_diff.cpp", bR0, bR2, bA, {q2, G});
        run_ab("compute/modmul_compute.cpp", bA, bInv02, bBt, {G, q2, ctx2.q_inv_32(), q2 & 0xFFFFu, q2 >> 16});
        run_ab("compute/garner_diff.cpp", bV1, bBt, bCC, {q2, G});                     // (b - v1) mod q2
        run_ab("compute/modmul_compute.cpp", bCC, bInv12, bV2, {G, q2, ctx2.q_inv_32(), q2 & 0xFFFFu, q2 >> 16});
        // assembly
        auto bW = mk(2 * G), bP = mk(2 * G), bQ = mk(2 * G), bOut = mk(3 * G);
        run_crt(bV2, 1, 0, bV1, bParQ1, bW);        // w = q1·v2 + v1
        run_crt(bW, 2, 0, bR0, bParQ0, bP);         // P = q0·w0 + r0
        run_crt(bW, 2, 1, bZero, bParQ0, bQ);       // Q = q0·w1 + 0
        {   // crt_limb3: X = P + Q·2^28 -> (X0,X1,X2)
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, 4); mkcb(p, CBIndex::c_1, 4); mkcb(p, CBIndex::c_16, 6);  // c_16 = 2 groups of 3
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_pq.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/crt_limb3.cpp", core,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)bP->address(), (uint32_t)bQ->address(), G});
            SetRuntimeArgs(p, wr, core, {(uint32_t)bOut->address(), 3 * G, 0u});
            SetRuntimeArgs(p, cp, core, {G});
            distributed::MeshWorkload wl; wl.add_program(range, std::move(p));
            distributed::EnqueueMeshWorkload(cq, wl, false);
        }

        if (getenv("DBG")) {
            std::vector<uint32_t> w, P, Q, v1, v2;
            distributed::EnqueueReadMeshBuffer(cq, w, bW, true);
            distributed::EnqueueReadMeshBuffer(cq, P, bP, true);
            distributed::EnqueueReadMeshBuffer(cq, Q, bQ, true);
            distributed::EnqueueReadMeshBuffer(cq, v1, bV1, true);
            distributed::EnqueueReadMeshBuffer(cq, v2, bV2, true);
            uint32_t e = untidx(0, 0);
            for (uint32_t g = 0; g < 2 && g < G; ++g)
                fmt::print("  g{}: v1={} v2={} w0={} w1={} P0={} P1={} Q0={} Q1={}\n", g,
                           v1[g * ELEMS + e], v2[g * ELEMS + e],
                           w[(2 * g) * ELEMS + e], w[(2 * g + 1) * ELEMS + e],
                           P[(2 * g) * ELEMS + e], P[(2 * g + 1) * ELEMS + e],
                           Q[(2 * g) * ELEMS + e], Q[(2 * g + 1) * ELEMS + e]);
        }

        std::vector<uint32_t> out;
        distributed::EnqueueReadMeshBuffer(cq, out, bOut, true);   // 3*G int32 tiles: (X0,X1,X2) per g

        uint32_t mism = 0, checked = 0;
        for (uint32_t v = 0; v < nvec; ++v)
            for (uint32_t tt = 0; tt < NT; ++tt) {
                uint32_t g = v * NT + tt;
                const uint32_t* X[3] = {out.data() + (3 * g + 0) * ELEMS,
                                        out.data() + (3 * g + 1) * ELEMS,
                                        out.data() + (3 * g + 2) * ELEMS};
                for (uint32_t r = 0; r < T; ++r)
                    for (uint32_t c = 0; c < T; ++c) {
                        bool bad = false;
                        for (uint32_t j = 0; j < 3; ++j)
                            if (X[j][untidx(r, c)] != lb[j][v * n + tt * ELEMS + r * T + c]) bad = true;
                        if (bad) {
                            if (mism < 8) {
                                uint32_t idx = v * n + tt * ELEMS + r * T + c;
                                fmt::print("  MISMATCH v{} tile{} ({},{}): got [{},{},{}] exp [{},{},{}]\n",
                                           v, tt, r, c, X[0][untidx(r, c)], X[1][untidx(r, c)], X[2][untidx(r, c)],
                                           lb[0][idx], lb[1][idx], lb[2][idx]);
                            }
                            ++mism;
                        }
                        ++checked;
                    }
            }
        fmt::print("crt_combine_k3: mismatches = {}/{}\n", mism, checked);
        pass = (mism == 0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
