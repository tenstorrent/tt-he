// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// he_run — device runtime for compiler/devgen.py's spec v2: pointwise ops AND MATMULS.
//
// codegen_run.cpp executes the pointwise subset only, because codegen.py's spec cannot express a
// matmul (no shape, no contraction axis, no per-op constant). That was the ONLY reason the compiler's
// matmuls never reached the device -- kernels/ir/compute/matmul_modq.cpp has been there all along,
// device-validated by host/tests/ir/test_ir_matmul.cpp. This runtime adds opkind 3 and drives it.
//
// matmul_modq is a fused decompose + int8 matmul + recombine over one 32x32 tile, with the weight
// pre-folded into 16 int8 tiles by the host. Two properties make it drop straight into the existing
// wave scheduler:
//   * data is TILIZED throughout (devgen.py tilizes inputs, golden and weights in Python), and
//     elementwise ops are permutation-invariant, so both op families share one layout;
//   * a plain-constant matmul is LINEAR in its data operand, so it preserves the Montgomery domain:
//     A @ mont(B) = (A@B)*R = mont(A@B). No conversion, no special-casing in the chain.
//
// Dispatch is unchanged: dependency waves, one Program per wave, ready ops on DISTINCT cores. Each
// core in a wave carries exactly one op, so its CB set is chosen per op kind.
//
//   ./build/he_run [spec.bin]

#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <set>
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
constexpr uint32_t ELEMS = 1024, TB = ELEMS * 4, I8 = ELEMS, NB = 4;
constexpr uint32_t MAGIC = 0xC0DE6E02u;
constexpr uint32_t K_MUL = 0, K_ADD = 1, K_SUB = 2, K_MM = 3;
struct Op { uint32_t kind, in0, in1, out, core; };
using Buf = std::shared_ptr<distributed::MeshBuffer>;

CircularBufferConfig cbc(CBIndex i, DataFormat f, uint32_t depth, uint32_t page) {
    CircularBufferConfig c(depth * page, {{i, f}});
    c.set_page_size(i, page);
    return c;
}
}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "compiler/generated/he_spec.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
    auto ru = [&] { uint32_t x; f.read((char*)&x, 4); return x; };
    uint32_t magic = ru(), N = ru(), q = ru(), nbuf = ru(), nop = ru(), nw = ru();
    if (magic != MAGIC || N != ELEMS) { fmt::print(stderr, "bad spec (magic {:#x}, N {})\n", magic, N); return 2; }

    std::vector<uint32_t> kind(nbuf);
    for (auto& k : kind) k = ru();
    std::vector<Op> ops(nop);
    for (auto& o : ops) { o.kind = ru(); o.in0 = ru(); o.in1 = ru(); o.out = ru(); o.core = ru(); }
    std::vector<std::vector<uint32_t>> in(nbuf), gold(nbuf);
    for (uint32_t i = 0; i < nbuf; ++i) if (kind[i] == 1) { in[i].resize(N); f.read((char*)in[i].data(), 4 * N); }
    for (uint32_t i = 0; i < nbuf; ++i) if (kind[i] == 2) { gold[i].resize(N); f.read((char*)gold[i].data(), 4 * N); }
    std::vector<std::vector<int8_t>> wdat(nw);
    for (uint32_t i = 0; i < nw; ++i) { wdat[i].resize(NB * NB * I8); f.read((char*)wdat[i].data(), NB * NB * I8); }

    mont::MontgomeryContext ctx(q);
    auto md = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange dr(md->shape());
    auto grid = md->compute_with_storage_grid_size();
    distributed::DeviceLocalBufferConfig dram32{.page_size = TB, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig dram8{.page_size = I8, .buffer_type = BufferType::DRAM};

    std::vector<Buf> B(nbuf);
    for (uint32_t i = 0; i < nbuf; ++i)
        B[i] = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = TB}, dram32, md.get());
    std::vector<Buf> W(nw);
    for (uint32_t i = 0; i < nw; ++i)
        W[i] = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = NB * NB * I8}, dram8, md.get());

    // inputs -> Montgomery (the matmul weight stays PLAIN: it is what makes the domain pass through)
    for (uint32_t i = 0; i < nbuf; ++i) if (kind[i] == 1) {
        std::vector<uint32_t> m(N);
        for (uint32_t j = 0; j < N; ++j) m[j] = ctx.to_montgomery(in[i][j]);
        distributed::EnqueueWriteMeshBuffer(cq, B[i], m, false);
    }
    for (uint32_t i = 0; i < nw; ++i) distributed::EnqueueWriteMeshBuffer(cq, W[i], wdat[i], false);

    std::vector<int> producer(nbuf, -1);
    for (uint32_t i = 0; i < nop; ++i) producer[ops[i].out] = (int)i;

    std::vector<char> done(nop, 0);
    std::vector<std::unique_ptr<distributed::MeshWorkload>> wls;
    uint32_t remaining = nop, waves = 0, mm_run = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    while (remaining) {
        Program p = CreateProgram();
        std::set<uint32_t> used;
        std::vector<uint32_t> wave;
        for (uint32_t i = 0; i < nop; ++i) {
            if (done[i]) continue;
            const Op& o = ops[i];
            bool ready = (producer[o.in0] < 0 || done[producer[o.in0]]) &&
                         (o.kind == K_MM || producer[o.in1] < 0 || done[producer[o.in1]]);
            if (ready && !used.count(o.core)) { wave.push_back(i); used.insert(o.core); }
        }
        for (uint32_t i : wave) {
            const Op& o = ops[i];
            CoreCoord core{o.core % grid.x, o.core / grid.x};
            CoreRange cr(core, core);
            if (o.kind == K_MM) {
                CreateCircularBuffer(p, cr, cbc(CBIndex::c_0, DataFormat::UInt32, 2, TB));
                CreateCircularBuffer(p, cr, cbc(CBIndex::c_1, DataFormat::Int8, 2, I8));
                CreateCircularBuffer(p, cr, cbc(CBIndex::c_2, DataFormat::Int8, NB, I8));
                CreateCircularBuffer(p, cr, cbc(CBIndex::c_3, DataFormat::Int32, NB, TB));
                CreateCircularBuffer(p, cr, cbc(CBIndex::c_16, DataFormat::UInt32, 2, TB));
                std::vector<uint32_t> rc;
                TensorAccessorArgs(*B[o.in0]).append_to(rc);
                TensorAccessorArgs(*W[o.in1]).append_to(rc);
                auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul.cpp", cr,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                       .noc = NOC::RISCV_1_default, .compile_args = rc});
                auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", cr,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
                auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/matmul_modq.cpp", cr,
                    ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                                  .dst_full_sync_en = true, .math_approx_mode = false});
                SetRuntimeArgs(p, rd, core, {(uint32_t)B[o.in0]->address(), (uint32_t)W[o.in1]->address()});
                SetRuntimeArgs(p, wr, core, {(uint32_t)B[o.out]->address(), 1u, 0u});
                SetRuntimeArgs(p, cp, core, {q, ctx.q_inv_32(), q & 0xFFFFu, q >> 16,
                                             ctx.to_montgomery(16384u)});
                ++mm_run;
            } else {
                CreateCircularBuffer(p, cr, cbc(CBIndex::c_0, DataFormat::UInt32, 2, TB));
                CreateCircularBuffer(p, cr, cbc(CBIndex::c_1, DataFormat::UInt32, 2, TB));
                CreateCircularBuffer(p, cr, cbc(CBIndex::c_16, DataFormat::UInt32, 2, TB));
                auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_binop.cpp", cr,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
                auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_out.cpp", cr,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
                const char* k = o.kind == K_MUL ? "compute/mul_modq.cpp"
                              : o.kind == K_ADD ? "compute/add_modq.cpp" : "compute/sub_modq.cpp";
                auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + std::string(k), cr,
                    ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                                  .dst_full_sync_en = true, .math_approx_mode = false});
                SetRuntimeArgs(p, rd, core, {(uint32_t)B[o.in0]->address(), (uint32_t)B[o.in1]->address(), 1u, 0u});
                SetRuntimeArgs(p, wr, core, {(uint32_t)B[o.out]->address(), 1u, 0u});
                if (o.kind == K_MUL) SetRuntimeArgs(p, cp, core, {1u, q, ctx.q_inv_32(), q & 0xFFFFu, q >> 16});
                else                 SetRuntimeArgs(p, cp, core, {1u, q});
            }
        }
        wls.push_back(std::make_unique<distributed::MeshWorkload>());
        wls.back()->add_program(dr, std::move(p));
        distributed::EnqueueMeshWorkload(cq, *wls.back(), false);   // in-order CQ => deps respected
        for (uint32_t i : wave) done[i] = 1;
        remaining -= wave.size();
        ++waves;
    }
    distributed::Finish(cq);
    double us = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - t0).count() / 1e3;

    const int iters = 20;
    auto s0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it)
        for (auto& w : wls) distributed::EnqueueMeshWorkload(cq, *w, false);
    distributed::Finish(cq);
    double dev_us = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - s0).count() / 1e3 / iters;

    uint32_t bad = 0, checked = 0;
    for (uint32_t i = 0; i < nbuf; ++i) if (kind[i] == 2) {
        std::vector<uint32_t> out;
        distributed::EnqueueReadMeshBuffer(cq, out, B[i], true);
        for (uint32_t j = 0; j < N; ++j) { ++checked; if (ctx.mulmod(out[j], 1u) != gold[i][j]) ++bad; }
    }
    md->close();
    fmt::print("he_run: {} ops ({} matmul) in {} waves, {} buffers, {} folded weights\n",
               nop, mm_run, waves, nbuf, nw);
    fmt::print("  cold build+dispatch {:.0f} us (host-bound)   steady-state device {:.0f} us/DAG\n", us, dev_us);
    fmt::print("  bit-exact vs golden: {}  ({} / {} bad)\n", bad ? "FAIL" : "OK", bad, checked);
    return bad ? 1 : 0;
}
