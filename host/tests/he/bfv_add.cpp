// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BFV prototype — HE ciphertext addition (ct_add), single Tensix core, N=1024.
// See docs/he/bfv_prototype_plan.md. Ciphertexts are RNS polynomials over the
// Q base (4 limbs, ~28-bit NTT-friendly primes), all limbs in Montgomery form.
//
//   ct_add((c0,c1),(d0,d1)) = (c0+d0, c1+d1)   per-limb vec_modadd.
//
// Montgomery is preserved by modular add: mont(a)+mont(b) = mont((a+b) mod q),
// so per-limb modadd on Montgomery limbs gives the Montgomery result directly —
// no constant needed, no new kernel (reuses modadd_reduce with n_in=2).
//
//   ./build/bfv_add [seed]

#include <cstdint>
#include <chrono>
#include <random>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t T = 32, N = 1024, ELEMS = 1024, I32 = ELEMS * 4;
constexpr uint32_t LQ = 4;              // Q base: 4 limbs
constexpr uint32_t NPOLY = 2;           // fresh ciphertext: (c0, c1)
using u128 = unsigned __int128;

bool is_prime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) return false;
    return true; }

// NTT-friendly ~28-bit primes:  p = k*1024 + 1 < 2^28, prime.  Return `cnt` of them (descending).
std::vector<uint64_t> ntt_primes(uint32_t cnt) {
    std::vector<uint64_t> out;
    for (uint64_t k = ((1ull << 28) - 1) / 1024; out.size() < cnt && k > 0; --k) {
        uint64_t p = k * 1024 + 1;
        if (p < (1ull << 28) && is_prime(p)) out.push_back(p);
    }
    return out; }

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
        const uint32_t seed = (argc > 1) ? std::stoul(argv[1]) : 0;

        // ---- params: Q base = 4 NTT-friendly ~28-bit primes ----
        auto primes = ntt_primes(LQ);
        std::vector<mont::MontgomeryContext> ctxQ;
        for (uint32_t i = 0; i < LQ; ++i) ctxQ.emplace_back((uint32_t)primes[i]);

        // ---- two ciphertexts, each (c0,c1); RNS over Q, coeffs random in [0,q_i) ----
        std::mt19937_64 rng(seed);
        // plain[which][poly][limb][coeff]   which: 0=c, 1=d
        auto gen = [&](std::vector<std::vector<std::vector<uint32_t>>>& ct) {
            ct.assign(NPOLY, std::vector<std::vector<uint32_t>>(LQ, std::vector<uint32_t>(N)));
            for (uint32_t k = 0; k < NPOLY; ++k)
                for (uint32_t i = 0; i < LQ; ++i) {
                    std::uniform_int_distribution<uint32_t> d(0, (uint32_t)primes[i] - 1);
                    for (uint32_t n = 0; n < N; ++n) ct[k][i][n] = d(rng);
                }
        };
        std::vector<std::vector<std::vector<uint32_t>>> C, D;
        gen(C); gen(D);

        // ---- CPU reference: e_k = (c_k + d_k) mod q, then Montgomery-encode ----
        // (Montgomery preserved by add, so ref = mont((c+d) mod q).)
        std::vector<std::vector<std::vector<uint32_t>>> Eref(NPOLY,
            std::vector<std::vector<uint32_t>>(LQ, std::vector<uint32_t>(N)));
        for (uint32_t k = 0; k < NPOLY; ++k)
            for (uint32_t i = 0; i < LQ; ++i)
                for (uint32_t n = 0; n < N; ++n) {
                    uint32_t s = (uint32_t)(((u128)C[k][i][n] + D[k][i][n]) % primes[i]);
                    Eref[k][i][n] = ctxQ[i].to_montgomery(s);
                }

        // ---- device buffers: interleave (c_ki, d_ki) so modadd reads 2 contiguous tiles ----
        // layout: pair p = k*LQ + i  ->  tiles [2p]=mont(c_ki), [2p+1]=mont(d_ki)
        const uint32_t NPAIR = NPOLY * LQ;               // 8
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        auto mkbuf = [&](uint32_t nt) {
            distributed::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * I32}, d, mesh.get()); };
        auto b_in = mkbuf(2 * NPAIR), b_out = mkbuf(NPAIR);

        std::vector<uint32_t> inblob(2 * NPAIR * ELEMS);
        for (uint32_t k = 0; k < NPOLY; ++k)
            for (uint32_t i = 0; i < LQ; ++i) {
                uint32_t p = k * LQ + i;
                std::vector<uint32_t> rc(N), rd(N);
                for (uint32_t n = 0; n < N; ++n) { rc[n] = ctxQ[i].to_montgomery(C[k][i][n]); rd[n] = ctxQ[i].to_montgomery(D[k][i][n]); }
                auto tc = tilize(rc), td = tilize(rd);
                std::copy(tc.begin(), tc.end(), inblob.begin() + (2 * p) * ELEMS);
                std::copy(td.begin(), td.end(), inblob.begin() + (2 * p + 1) * ELEMS);
            }
        distributed::EnqueueWriteMeshBuffer(cq, b_in, inblob, false);

        // ---- one modadd program per output limb (n_in=2), reusing modadd_reduce ----
        auto mkcb = [&](Program& pr, CBIndex idx, uint32_t depth) {
            CircularBufferConfig c(depth * I32, {{idx, tt::DataFormat::UInt32}});
            c.set_page_size(idx, I32); CreateCircularBuffer(pr, core, c); };
        std::vector<distributed::MeshWorkload> wls; wls.reserve(NPAIR);
        for (uint32_t p = 0; p < NPAIR; ++p) {
            uint32_t qi = primes[p % LQ];
            Program pr = CreateProgram();
            mkcb(pr, CBIndex::c_0, 2); mkcb(pr, CBIndex::c_16, 2);
            auto rdk = CreateKernel(pr, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wrk = CreateKernel(pr, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cpk = CreateKernel(pr, OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                              .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(pr, rdk, core, {(uint32_t)b_in->address(), 2u, 2 * p});   // read pair
            SetRuntimeArgs(pr, wrk, core, {(uint32_t)b_out->address(), 1u, p});      // write out[p]
            SetRuntimeArgs(pr, cpk, core, {2u, qi});
            wls.emplace_back(); wls.back().add_program(dr, std::move(pr));
        }
        for (auto& wl : wls) distributed::EnqueueMeshWorkload(cq, wl, false);

        // ---- timing: steady-state host wall-clock (matches ntt1024_dev.cpp) ----
        {
            distributed::Finish(cq);  // warm
            const int iters = 100;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it)
                for (auto& wl : wls) distributed::EnqueueMeshWorkload(cq, wl, false);
            distributed::Finish(cq);
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e3 / iters;
            fmt::print("TIMING: bfv_add steady-state = {:.1f} us/op  ({} device dispatches)\n", us, wls.size());
        }

        std::vector<uint32_t> outblob;
        distributed::EnqueueReadMeshBuffer(cq, outblob, b_out, true);

        // ---- validate ----
        uint32_t mism = 0;
        for (uint32_t k = 0; k < NPOLY; ++k)
            for (uint32_t i = 0; i < LQ; ++i) {
                uint32_t p = k * LQ + i;
                auto got = untilize(std::vector<uint32_t>(outblob.begin() + p * ELEMS, outblob.begin() + (p + 1) * ELEMS));
                for (uint32_t n = 0; n < N; ++n)
                    if (got[n] != Eref[k][i][n]) { if (mism < 8) fmt::print("  poly{} limb{} n{} got={} exp={}\n", k, i, n, got[n], Eref[k][i][n]); ++mism; }
            }
        fmt::print("BFV ct_add  N=1024 Q(4)  seed={}  mismatches={}/{}\n", seed, mism, NPOLY * LQ * N);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: BFV ct_add WORKS.\n"); else fmt::print("RESULT: WRONG.\n");
        pass &= mesh->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Exception:\n{}\n", e.what());
        return 2;
    }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
