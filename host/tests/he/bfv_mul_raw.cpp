// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BFV prototype — raw ciphertext multiplication (tensor product), NTT/evaluation
// domain, single core, N=1024, Q=4 limbs, Montgomery. See docs/he/bfv_prototype_plan.md.
//
//   ct_mul_raw((c0,c1),(d0,d1)) -> (e0,e1,e2):
//     e0 = c0·d0 ;  e1 = c0·d1 + c1·d0 ;  e2 = c1·d1
//
// In the (negacyclic-)NTT domain, poly multiplication is POINTWISE, so each `·` is a
// per-limb elementwise vec_modmul (Montgomery-preserving) and the e1 sum is a modadd.
// This is the tensor product proper; converting coeff<->NTT is the negacyclic transform
// (ψ-folded NTT/INTT — the remaining coeff-domain converter, not exercised here).
//
// Reuses modmul_sfpi_hybrid (pointwise tile×tile modmul) + modadd_reduce.  No new kernel.
//   ./build/bfv_mul_raw [seed]

#include <chrono>
#include <cstdint>
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
constexpr uint32_t LQ = 4;
using u128 = unsigned __int128;

bool is_prime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) return false;
    return true; }
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
        auto primes = ntt_primes(LQ);
        std::vector<mont::MontgomeryContext> ctxQ;
        for (uint32_t i = 0; i < LQ; ++i) ctxQ.emplace_back((uint32_t)primes[i]);

        // NTT-domain ciphertext polys c0,c1,d0,d1, coeffs random in [0,q_i)
        std::mt19937_64 rng(seed);
        auto genpoly = [&](std::vector<std::vector<uint32_t>>& P) {
            P.assign(LQ, std::vector<uint32_t>(N));
            for (uint32_t i = 0; i < LQ; ++i) {
                std::uniform_int_distribution<uint32_t> d(0, (uint32_t)primes[i] - 1);
                for (uint32_t n = 0; n < N; ++n) P[i][n] = d(rng);
            } };
        std::vector<std::vector<uint32_t>> c0, c1, d0, d1;
        genpoly(c0); genpoly(c1); genpoly(d0); genpoly(d1);

        // CPU reference (pointwise), Montgomery-encoded
        auto mm = [](u128 a, u128 b, uint64_t q){ return (uint32_t)(a * b % q); };
        std::vector<std::vector<uint32_t>> re0(LQ, std::vector<uint32_t>(N)),
            re1(LQ, std::vector<uint32_t>(N)), re2(LQ, std::vector<uint32_t>(N));
        for (uint32_t i = 0; i < LQ; ++i)
            for (uint32_t n = 0; n < N; ++n) {
                uint64_t q = primes[i];
                re0[i][n] = ctxQ[i].to_montgomery(mm(c0[i][n], d0[i][n], q));
                uint32_t t0 = mm(c0[i][n], d1[i][n], q), t1 = mm(c1[i][n], d0[i][n], q);
                re1[i][n] = ctxQ[i].to_montgomery((uint32_t)(((u128)t0 + t1) % q));
                re2[i][n] = ctxQ[i].to_montgomery(mm(c1[i][n], d1[i][n], q));
            }

        // ---- device ----
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        auto mkbuf = [&](uint32_t nt) {
            distributed::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * I32}, d, mesh.get()); };
        // 4 input polys (4 limb-tiles each), Montgomery-encoded + tilized
        auto b_c0 = mkbuf(LQ), b_c1 = mkbuf(LQ), b_d0 = mkbuf(LQ), b_d1 = mkbuf(LQ);
        auto put = [&](std::shared_ptr<distributed::MeshBuffer>& buf, std::vector<std::vector<uint32_t>>& P) {
            std::vector<uint32_t> blob(LQ * ELEMS);
            for (uint32_t i = 0; i < LQ; ++i) {
                std::vector<uint32_t> row(N);
                for (uint32_t n = 0; n < N; ++n) row[n] = ctxQ[i].to_montgomery(P[i][n]);
                auto t = tilize(row); std::copy(t.begin(), t.end(), blob.begin() + i * ELEMS);
            }
            distributed::EnqueueWriteMeshBuffer(cq, buf, blob, false); };
        put(b_c0, c0); put(b_c1, c1); put(b_d0, d0); put(b_d1, d1);

        auto b_e0 = mkbuf(LQ), b_e1 = mkbuf(LQ), b_e2 = mkbuf(LQ);
        auto b_t = mkbuf(2 * LQ);   // e1 temps: [t0_i, t1_i] interleaved per limb

        auto mkcb = [&](Program& pr, CBIndex idx, uint32_t depth) {
            CircularBufferConfig c(depth * I32, {{idx, tt::DataFormat::UInt32}});
            c.set_page_size(idx, I32); CreateCircularBuffer(pr, core, c); };
        std::vector<distributed::MeshWorkload> wls; wls.reserve(64);

        // pointwise modmul: out[wid] = (a[aid] ⊙ b[aid]) mod q_i   (reader_modmul: a,b same tile id)
        auto modmul = [&](std::shared_ptr<distributed::MeshBuffer>& a, std::shared_ptr<distributed::MeshBuffer>& b,
                          std::shared_ptr<distributed::MeshBuffer>& out, uint32_t tid, uint32_t wid, uint32_t i) {
            Program pr = CreateProgram();
            mkcb(pr, CBIndex::c_0, 2); mkcb(pr, CBIndex::c_1, 2); mkcb(pr, CBIndex::c_16, 2);
            auto rd = CreateKernel(pr, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(pr, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(pr, OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_full.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                              .dst_full_sync_en = true, .math_approx_mode = false});
            const uint32_t q = primes[i];
            SetRuntimeArgs(pr, rd, core, {(uint32_t)a->address(), (uint32_t)b->address(), 1u, tid});
            SetRuntimeArgs(pr, wr, core, {(uint32_t)out->address(), 1u, wid});
            SetRuntimeArgs(pr, cp, core, {1u, q, ctxQ[i].q_inv_32(), q & 0xFFFFu, q >> 16});
            wls.emplace_back(); wls.back().add_program(dr, std::move(pr));
            distributed::EnqueueMeshWorkload(cq, wls.back(), false); };
        // modadd: out[wid] = (in[2*rid] + in[2*rid+1]) mod q_i
        auto modadd = [&](std::shared_ptr<distributed::MeshBuffer>& in, std::shared_ptr<distributed::MeshBuffer>& out,
                          uint32_t rid, uint32_t wid, uint32_t i) {
            Program pr = CreateProgram();
            mkcb(pr, CBIndex::c_0, 2); mkcb(pr, CBIndex::c_16, 2);
            auto rd = CreateKernel(pr, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(pr, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(pr, OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                              .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(pr, rd, core, {(uint32_t)in->address(), 2u, 2 * rid});
            SetRuntimeArgs(pr, wr, core, {(uint32_t)out->address(), 1u, wid});
            SetRuntimeArgs(pr, cp, core, {2u, (uint32_t)primes[i]});
            wls.emplace_back(); wls.back().add_program(dr, std::move(pr));
            distributed::EnqueueMeshWorkload(cq, wls.back(), false); };

        for (uint32_t i = 0; i < LQ; ++i) {
            modmul(b_c0, b_d0, b_e0, i, i, i);          // e0 = c0·d0
            modmul(b_c1, b_d1, b_e2, i, i, i);          // e2 = c1·d1
            modmul(b_c0, b_d1, b_t, i, 2 * i, i);       // t0 = c0·d1  -> b_t[2i]
            modmul(b_c1, b_d0, b_t, i, 2 * i + 1, i);   // t1 = c1·d0  -> b_t[2i+1]
            modadd(b_t, b_e1, i, i, i);                 // e1 = t0 + t1
        }

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
            fmt::print("TIMING: bfv_mul_raw steady-state = {:.1f} us/op  ({} device dispatches)\n", us, wls.size());
        }

        std::vector<uint32_t> o0, o1, o2;
        distributed::EnqueueReadMeshBuffer(cq, o0, b_e0, true);
        distributed::EnqueueReadMeshBuffer(cq, o1, b_e1, true);
        distributed::EnqueueReadMeshBuffer(cq, o2, b_e2, true);

        auto check = [&](std::vector<uint32_t>& got, std::vector<std::vector<uint32_t>>& ref, const char* nm) {
            uint32_t m = 0;
            for (uint32_t i = 0; i < LQ; ++i) {
                auto g = untilize(std::vector<uint32_t>(got.begin() + i * ELEMS, got.begin() + (i + 1) * ELEMS));
                for (uint32_t n = 0; n < N; ++n) if (g[n] != ref[i][n]) { if (m < 4) fmt::print("  {} limb{} n{} got={} exp={}\n", nm, i, n, g[n], ref[i][n]); ++m; }
            }
            return m; };
        uint32_t mism = check(o0, re0, "e0") + check(o1, re1, "e1") + check(o2, re2, "e2");
        fmt::print("BFV ct_mul_raw (NTT-domain tensor)  N=1024 Q(4)  seed={}  mismatches={}/{}\n", seed, mism, 3 * LQ * N);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: BFV ct_mul_raw (tensor) WORKS.\n"); else fmt::print("RESULT: WRONG.\n");
        pass &= mesh->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Exception:\n{}\n", e.what());
        return 2;
    }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
