// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Unit test: on-device signed gadget decomposition (compute/gadget_decompose_signed.cpp, #9).
// Reads a golden case from cpu/pir/gen_gadget_case.py, feeds the two radix-2^28 limbs, and
// checks the ELL signed int32 digit tiles bit-exact against the golden.
//
//   ./build/test_gadget_decompose [case.bin]      (default reference/pir/gadget_ep.bin)
//
// The compiled GADGET_B/GADGET_ELL must match the case file (asserted from its header).

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <string>
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
#ifndef GADGET_B
#define GADGET_B 1024
#endif
#ifndef GADGET_ELL
#define GADGET_ELL 6
#endif
#ifndef GADGET_LOG2B
#define GADGET_LOG2B 10
#endif
#ifndef GADGET_KLIMBS
#define GADGET_KLIMBS 2
#endif

namespace {
constexpr uint32_t T = 32, ELEMS = 1024;

// 32x32 face-tiled layout (matches gen_ctmul_case.py / the device tile order).
std::vector<uint32_t> tilize(const uint32_t* v) {
    std::vector<uint32_t> o(ELEMS, 0);
    for (uint32_t r = 0; r < T; ++r)
        for (uint32_t c = 0; c < T; ++c)
            o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = v[r * T + c];
    return o;
}
uint32_t untilize_idx(uint32_t r, uint32_t c) {
    return ((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16);
}
}  // namespace

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "reference/pir/gadget_ep.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open case file {}\n", path); return 2; }

    auto ru32 = [&](uint32_t& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    uint32_t magic, version, n, ell, nvec, B, K, log2b;
    ru32(magic); ru32(version); ru32(n); ru32(ell); ru32(nvec); ru32(B); ru32(K); ru32(log2b);
    if (magic != 0x47414447u || version != 2u) { fmt::print(stderr, "bad magic/version\n"); return 2; }
    constexpr uint32_t KL = GADGET_KLIMBS;
    if (ell != GADGET_ELL || B != GADGET_B || K != KL) {
        fmt::print(stderr, "case (B={},ell={},K={}) != compiled (B={},ell={},K={}); rebuild to match\n",
                   B, ell, K, (uint32_t)GADGET_B, (uint32_t)GADGET_ELL, KL);
        return 2;
    }
    std::vector<uint32_t> primes(K);
    f.read(reinterpret_cast<char*>(primes.data()), 4 * K);
    const uint32_t NT = n / ELEMS;                 // tiles per poly
    const uint32_t G = nvec * NT;                  // total coefficient-tiles

    // read all vectors: K limb streams (uint32) + digits (int32); skip the K residue streams.
    std::vector<std::vector<uint32_t>> limb(K, std::vector<uint32_t>(nvec * n));
    std::vector<int32_t> golden(nvec * ell * n);
    std::vector<uint32_t> scratch(n);
    for (uint32_t v = 0; v < nvec; ++v) {
        for (uint32_t j = 0; j < K; ++j)                                  // residues (unused here)
            f.read(reinterpret_cast<char*>(scratch.data()), 4 * n);
        for (uint32_t j = 0; j < K; ++j)                                  // radix-2^28 limbs
            f.read(reinterpret_cast<char*>(limb[j].data() + v * n), 4 * n);
        f.read(reinterpret_cast<char*>(golden.data() + v * ell * n), 4 * ell * n);
    }
    fmt::print("case {}: n={} ell={} nvec={} B={} K={} (NT={}, G={} X-tiles)\n",
               path, n, ell, nvec, B, K, NT, G);

    // tilize each of the K limb streams into device order
    std::vector<std::vector<uint32_t>> in(K, std::vector<uint32_t>(G * ELEMS));
    for (uint32_t v = 0; v < nvec; ++v)
        for (uint32_t tt = 0; tt < NT; ++tt) {
            uint32_t g = v * NT + tt;
            for (uint32_t j = 0; j < K; ++j) {
                auto tj = tilize(limb[j].data() + v * n + tt * ELEMS);
                std::copy(tj.begin(), tj.end(), in[j].begin() + g * ELEMS);
            }
        }

    bool pass = true;
    try {
        constexpr uint32_t i32_tile = ELEMS * 4;
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        Program prog = CreateProgram();
        constexpr CoreCoord core = {0, 0};

        distributed::DeviceLocalBufferConfig i32d{.page_size = i32_tile, .buffer_type = BufferType::DRAM};
        std::vector<std::shared_ptr<distributed::MeshBuffer>> bufs(K);
        for (uint32_t j = 0; j < K; ++j)
            bufs[j] = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = G * i32_tile}, i32d, mesh.get());
        auto ob = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = G * ell * i32_tile}, i32d, mesh.get());

        // c_0 holds an integral number of KLIMBS-tile groups (2 groups = clean double-buffer, no mid-group wrap)
        CircularBufferConfig c0(2 * K * i32_tile, {{CBIndex::c_0, DataFormat::UInt32}}); c0.set_page_size(CBIndex::c_0, i32_tile);
        CreateCircularBuffer(prog, core, c0);
        CircularBufferConfig c16(2 * ell * i32_tile, {{CBIndex::c_16, DataFormat::UInt32}}); c16.set_page_size(CBIndex::c_16, i32_tile);
        CreateCircularBuffer(prog, core, c16);

        std::map<std::string, std::string> kdefs = {{"GADGET_KLIMBS", std::to_string(KL)}};
        auto rd = CreateKernel(prog, OVERRIDE_KERNEL_PREFIX "dataflow/reader_gadget.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = kdefs});
        std::vector<uint32_t> wc; TensorAccessorArgs(*ob).append_to(wc);
        auto wr = CreateKernel(prog, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto cp = CreateKernel(prog, OVERRIDE_KERNEL_PREFIX "compute/gadget_decompose_signed.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false,
                          .defines = {{"GADGET_B", std::to_string((uint32_t)GADGET_B)},
                                      {"GADGET_LOG2B", std::to_string((uint32_t)GADGET_LOG2B)},
                                      {"GADGET_ELL", std::to_string((uint32_t)GADGET_ELL)},
                                      {"GADGET_KLIMBS", std::to_string(KL)}}});
        std::vector<uint32_t> rargs(K);
        for (uint32_t j = 0; j < K; ++j) rargs[j] = (uint32_t)bufs[j]->address();
        rargs.push_back(G);
        SetRuntimeArgs(prog, rd, core, rargs);
        SetRuntimeArgs(prog, wr, core, {(uint32_t)ob->address(), G * ell, 0u});
        SetRuntimeArgs(prog, cp, core, {G});

        distributed::MeshWorkload wl; wl.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(prog));
        for (uint32_t j = 0; j < K; ++j)
            distributed::EnqueueWriteMeshBuffer(cq, bufs[j], in[j], false);
        distributed::EnqueueMeshWorkload(cq, wl, false);
        std::vector<uint32_t> out;
        distributed::EnqueueReadMeshBuffer(cq, out, ob, true);   // G*ell int32 tiles

        uint32_t mism = 0, checked = 0;
        for (uint32_t v = 0; v < nvec; ++v)
            for (uint32_t tt = 0; tt < NT; ++tt) {
                uint32_t g = v * NT + tt;
                for (uint32_t k = 0; k < ell; ++k) {
                    const uint32_t* tile = out.data() + (g * ell + k) * ELEMS;
                    for (uint32_t r = 0; r < T; ++r)
                        for (uint32_t c = 0; c < T; ++c) {
                            int32_t got = (int32_t)tile[untilize_idx(r, c)];
                            int32_t exp = golden[v * ell * n + k * n + tt * ELEMS + r * T + c];
                            if (got != exp) {
                                if (mism < 8)
                                    fmt::print("  MISMATCH v{} tile{} digit{} ({},{}): got {} exp {}\n",
                                               v, tt, k, r, c, got, exp);
                                ++mism;
                            }
                            ++checked;
                        }
                }
            }
        fmt::print("gadget_decompose_signed: mismatches = {}/{}\n", mism, checked);
        pass = (mism == 0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
