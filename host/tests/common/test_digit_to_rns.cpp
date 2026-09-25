// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Unit test: on-device digit->RNS split (compute/digit_to_rns.cpp, external-product step 4).
// Uses the golden signed digits from cpu/pir/gen_gadget_case.py (VERSION 2, K-parameterized) as
// input; expects (d mod q_j) for j=0..K-1 per digit.
//   ./build/test_digit_to_rns [reference/pir/gadget_k2_l8.bin]   (K from the file must == DR_K)

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
#ifndef DR_K
#define DR_K 2
#endif

namespace {
constexpr uint32_t T = 32, ELEMS = 1024;
std::vector<uint32_t> tilize(const int32_t* v) {
    std::vector<uint32_t> o(ELEMS, 0);
    for (uint32_t r = 0; r < T; ++r)
        for (uint32_t c = 0; c < T; ++c)
            o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = (uint32_t)v[r * T + c];
    return o;
}
uint32_t untidx(uint32_t r, uint32_t c) { return ((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16); }
uint32_t modq(int32_t d, uint32_t q) { return (uint32_t)(d >= 0 ? d : d + (int32_t)q); }
}  // namespace

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "reference/pir/gadget_k2_l8.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
    auto ru = [&](uint32_t& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    uint32_t magic, ver, n, ell, nvec, B, K, log2b;
    ru(magic); ru(ver); ru(n); ru(ell); ru(nvec); ru(B); ru(K); ru(log2b);
    if (magic != 0x47414447u || ver != 2u) { fmt::print(stderr, "bad magic/version\n"); return 2; }
    constexpr uint32_t DK = DR_K;
    if (K != DK) { fmt::print(stderr, "case K={} != compiled DR_K={}; rebuild to match\n", K, DK); return 2; }
    std::vector<uint32_t> primes(K);
    f.read(reinterpret_cast<char*>(primes.data()), 4 * K);
    const uint32_t NT = n / ELEMS;
    const uint32_t NDIG = nvec * ell * NT;          // total digit-tiles

    // read digits only (skip the K residue + K limb arrays per vec)
    std::vector<int32_t> dig(nvec * ell * n);
    for (uint32_t v = 0; v < nvec; ++v) {
        f.seekg(2 * K * 4 * n, std::ios::cur);
        f.read(reinterpret_cast<char*>(dig.data() + v * ell * n), 4 * ell * n);
    }
    fmt::print("case {}: n={} ell={} nvec={} K={} primes[0..]={},{}{} (NDIG={})\n", path, n, ell, nvec, K,
               primes[0], primes[1], (K >= 3 ? fmt::format(",{}", primes[2]) : ""), NDIG);

    std::vector<uint32_t> in(NDIG * ELEMS);
    uint32_t di = 0;
    for (uint32_t v = 0; v < nvec; ++v)
        for (uint32_t k = 0; k < ell; ++k)
            for (uint32_t tt = 0; tt < NT; ++tt) {
                auto t = tilize(dig.data() + v * ell * n + k * n + tt * ELEMS);
                std::copy(t.begin(), t.end(), in.begin() + (di++) * ELEMS);
            }

    bool pass = true;
    try {
        constexpr uint32_t i32 = ELEMS * 4;
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto range = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        distributed::DeviceLocalBufferConfig dram{.page_size = i32, .buffer_type = BufferType::DRAM};
        auto ib = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = NDIG * i32}, dram, mesh.get());
        auto ob = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = K * NDIG * i32}, dram, mesh.get());

        Program p = CreateProgram();
        CircularBufferConfig c0(4 * i32, {{CBIndex::c_0, DataFormat::UInt32}}); c0.set_page_size(CBIndex::c_0, i32);
        CreateCircularBuffer(p, core, c0);
        CircularBufferConfig c16(2 * K * i32, {{CBIndex::c_16, DataFormat::UInt32}}); c16.set_page_size(CBIndex::c_16, i32);
        CreateCircularBuffer(p, core, c16);
        auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/digit_to_rns.cpp", core,
            ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false,
                          .defines = {{"DR_K", std::to_string(DK)}}});
        SetRuntimeArgs(p, rd, core, {(uint32_t)ib->address(), NDIG, 0u});
        SetRuntimeArgs(p, wr, core, {(uint32_t)ob->address(), K * NDIG, 0u});
        std::vector<uint32_t> cargs = {NDIG};
        for (uint32_t j = 0; j < K; ++j) cargs.push_back(primes[j]);
        SetRuntimeArgs(p, cp, core, cargs);
        distributed::MeshWorkload wl; wl.add_program(range, std::move(p));
        distributed::EnqueueWriteMeshBuffer(cq, ib, in, false);
        distributed::EnqueueMeshWorkload(cq, wl, false);
        std::vector<uint32_t> out;
        distributed::EnqueueReadMeshBuffer(cq, out, ob, true);

        uint32_t mism = 0, checked = 0;
        di = 0;
        for (uint32_t v = 0; v < nvec; ++v)
            for (uint32_t k = 0; k < ell; ++k)
                for (uint32_t tt = 0; tt < NT; ++tt) {
                    const int32_t* g = dig.data() + v * ell * n + k * n + tt * ELEMS;
                    for (uint32_t r = 0; r < T; ++r)
                        for (uint32_t c = 0; c < T; ++c) {
                            bool bad = false;
                            for (uint32_t j = 0; j < K; ++j) {
                                const uint32_t* oj = out.data() + (K * di + j) * ELEMS;
                                if (oj[untidx(r, c)] != modq(g[r * T + c], primes[j])) bad = true;
                            }
                            if (bad) {
                                if (mism < 8) fmt::print("  MISMATCH v{} k{} ({},{}) d={}\n", v, k, r, c, g[r * T + c]);
                                ++mism;
                            }
                            ++checked;
                        }
                    ++di;
                }
        fmt::print("digit_to_rns: mismatches = {}/{}\n", mism, checked);
        pass = (mism == 0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
