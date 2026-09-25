// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// codegen_run — stage 7 runtime: execute a compiler-emitted spec on device, golden-checked.
// Reads compiler/artifacts/codegen_spec.bin (from codegen.py): a DRAM buffer per Value + a placed DAG
// of SFPU tile ops. Dispatches in dependency WAVES — each wave is one Program holding the ready
// ops on DISTINCT cores (co-scheduled, heterogeneous). Data is unified Montgomery: inputs are
// encoded, the output decoded and compared bit-exact to the spec's logical golden.
//   ./build/codegen_run [spec.bin]

#include <chrono>
#include <cstdint>
#include <fstream>
#include <vector>
#include <set>
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
constexpr uint32_t ELEMS = 1024, TB = ELEMS * 4;
struct Op { uint32_t kind, in0, in1, out, core; };
using Buf = std::shared_ptr<distributed::MeshBuffer>;
}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "compiler/artifacts/codegen_spec.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
    auto ru = [&] { uint32_t x; f.read((char*)&x, 4); return x; };
    uint32_t magic = ru(), N = ru(), q = ru(), nbuf = ru(), nop = ru();
    if (magic != 0xC0DE6E00u || N != ELEMS) { fmt::print(stderr, "bad spec\n"); return 2; }
    std::vector<uint32_t> kind(nbuf); for (auto& k : kind) k = ru();
    std::vector<Op> ops(nop); for (auto& o : ops) { o.kind = ru(); o.in0 = ru(); o.in1 = ru(); o.out = ru(); o.core = ru(); }
    std::vector<std::vector<uint32_t>> in(nbuf), gold(nbuf);
    for (uint32_t i = 0; i < nbuf; ++i) if (kind[i] == 1) { in[i].resize(N); f.read((char*)in[i].data(), 4 * N); }
    for (uint32_t i = 0; i < nbuf; ++i) if (kind[i] == 2) { gold[i].resize(N); f.read((char*)gold[i].data(), 4 * N); }

    mont::MontgomeryContext ctx(q);
    auto md = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange dr(md->shape());
    auto grid = md->compute_with_storage_grid_size();
    distributed::DeviceLocalBufferConfig dram{.page_size = TB, .buffer_type = BufferType::DRAM};
    std::vector<Buf> B(nbuf);
    for (uint32_t i = 0; i < nbuf; ++i) B[i] = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = TB}, dram, md.get());
    // inputs -> Montgomery, write
    for (uint32_t i = 0; i < nbuf; ++i) if (kind[i] == 1) {
        std::vector<uint32_t> m(N); for (uint32_t j = 0; j < N; ++j) m[j] = ctx.to_montgomery(in[i][j]);
        EnqueueWriteMeshBuffer(cq, B[i], m, false);
    }

    // producer[buf] = op that writes it (-1 = input/none)
    std::vector<int> producer(nbuf, -1);
    for (uint32_t i = 0; i < nop; ++i) producer[ops[i].out] = (int)i;

    auto mkcb = [&](Program& p, const CoreCoord& c, CBIndex idx) {
        CircularBufferConfig cfg(2 * TB, {{idx, DataFormat::UInt32}}); cfg.set_page_size(idx, TB);
        CreateCircularBuffer(p, CoreRange(c, c), cfg);
    };

    std::vector<char> done(nop, 0);
    std::vector<std::unique_ptr<distributed::MeshWorkload>> wls;   // keep workloads alive until Finish
    uint32_t remaining = nop, waves = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    while (remaining) {
        Program p = CreateProgram();
        std::set<uint32_t> used;
        std::vector<uint32_t> wave;
        for (uint32_t i = 0; i < nop; ++i) {
            if (done[i]) continue;
            bool ready = (producer[ops[i].in0] < 0 || done[producer[ops[i].in0]]) &&
                         (producer[ops[i].in1] < 0 || done[producer[ops[i].in1]]);
            if (ready && !used.count(ops[i].core)) { wave.push_back(i); used.insert(ops[i].core); }
        }
        for (uint32_t i : wave) {
            const Op& o = ops[i];
            CoreCoord core{o.core % grid.x, o.core / grid.x};
            mkcb(p, core, CBIndex::c_0); mkcb(p, core, CBIndex::c_1); mkcb(p, core, CBIndex::c_16);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_binop.cpp", CoreRange(core, core),
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_out.cpp", CoreRange(core, core),
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            const char* k = o.kind == 0 ? "compute/mul_modq.cpp" : o.kind == 1 ? "compute/add_modq.cpp" : "compute/sub_modq.cpp";
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + std::string(k), CoreRange(core, core),
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)B[o.in0]->address(), (uint32_t)B[o.in1]->address(), 1u, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)B[o.out]->address(), 1u, 0u});
            if (o.kind == 0) SetRuntimeArgs(p, cp, core, {1u, q, ctx.q_inv_32(), q & 0xFFFFu, q >> 16});
            else             SetRuntimeArgs(p, cp, core, {1u, q});
        }
        wls.push_back(std::make_unique<distributed::MeshWorkload>());
        wls.back()->add_program(dr, std::move(p));
        distributed::EnqueueMeshWorkload(cq, *wls.back(), false);   // in-order CQ => deps respected, no per-wave sync
        for (uint32_t i : wave) done[i] = 1;
        remaining -= wave.size(); ++waves;
    }
    distributed::Finish(cq);
    double us = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - t0).count() / 1e3;

    // steady-state device time (build-once, re-dispatch all waves) — excludes host construction
    distributed::Finish(cq);
    const int iters = 50;
    auto s0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it)
        for (auto& w : wls) distributed::EnqueueMeshWorkload(cq, *w, false);
    distributed::Finish(cq);
    double dev_us = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - s0).count() / 1e3 / iters;

    // read outputs, decode Montgomery, compare to golden
    uint32_t bad = 0, checked = 0;
    for (uint32_t i = 0; i < nbuf; ++i) if (kind[i] == 2) {
        std::vector<uint32_t> out; EnqueueReadMeshBuffer(cq, out, B[i], true);
        for (uint32_t j = 0; j < N; ++j) { ++checked; if (ctx.mulmod(out[j], 1u) != gold[i][j]) ++bad; }
    }
    md->close();
    fmt::print("codegen_run: {} ops in {} waves, {} buffers\n", nop, waves, nbuf);
    fmt::print("  cold build+dispatch {:.0f} us (host-bound)   steady-state device {:.0f} us/DAG\n", us, dev_us);
    fmt::print("  bit-exact vs golden: {}  ({} / {} bad)\n", bad ? "FAIL" : "OK", bad, checked);
    return bad ? 1 : 0;
}
