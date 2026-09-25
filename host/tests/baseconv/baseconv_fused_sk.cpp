// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FUSED + EXACT RNS base conversion  x mod Q -> x mod P  (4->4) in ONE program /
// compute kernel (baseconv_fbc_sk.cpp), with the CRT overflow ε recovered FULLY ON
// DEVICE via the integer Shenoy-Kumaresan sentinel m_sk (no floating point, one
// dispatch). Same Montgomery I/O as baseconv_fused.cpp; the on-device single-dispatch
// counterpart of the multi-dispatch baseconv_sk.cpp.
//
//   ./build/baseconv_fused_sk [seed]

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
constexpr uint32_t T = 32, NB = 4, ELEMS = 1024;
constexpr uint32_t I32 = ELEMS * 4;
using u128 = unsigned __int128;

uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) {
    u128 r = 1, bb = b % m;
    while (e) { if (e & 1) r = r * bb % m; bb = bb * bb % m; e >>= 1; }
    return (uint64_t)r; }
bool is_prime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) return false;
    return true; }
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

        // 9 distinct primes just below 2^28: 4 = Q, 4 = P, 1 = sentinel m_sk.
        uint64_t Qm[NB], Pm[NB], Msk;
        { std::vector<uint64_t> pr;
          for (uint64_t c = (1ull << 28) - 1; pr.size() < 2 * NB + 1; c -= 2) if (is_prime(c)) pr.push_back(c);
          for (uint32_t i = 0; i < NB; ++i) { Qm[i] = pr[i]; Pm[i] = pr[NB + i]; }
          Msk = pr[2 * NB]; }
        std::vector<mont::MontgomeryContext> ctxQ, ctxP;
        for (uint32_t i = 0; i < NB; ++i) ctxQ.emplace_back((uint32_t)Qm[i]);
        for (uint32_t j = 0; j < NB; ++j) ctxP.emplace_back((uint32_t)Pm[j]);
        mont::MontgomeryContext ctxM((uint32_t)Msk);

        u128 Q = 1; for (uint32_t i = 0; i < NB; ++i) Q *= Qm[i];
        uint64_t qhat_inv[NB], qhat_mod_p[NB][NB], qhat_mod_sk[NB];
        for (uint32_t i = 0; i < NB; ++i) {
            u128 qhat = Q / Qm[i];
            qhat_inv[i] = modpow((uint64_t)(qhat % Qm[i]), Qm[i] - 2, Qm[i]);
            for (uint32_t j = 0; j < NB; ++j) qhat_mod_p[i][j] = (uint64_t)(qhat % Pm[j]);
            qhat_mod_sk[i] = (uint64_t)(qhat % Msk);
        }
        uint64_t Qinv_sk = modpow((uint64_t)(Q % Msk), Msk - 2, Msk);   // Q^{-1} mod m_sk

        // FULL Montgomery I/O (all inputs incl. sentinel x_sk are Montgomery; R folded into
        // the bconv constants, kernel unchanged): C1 plain, C2/Ecorr/Dsk R^2, QinvM plain.
        //   Dsk = R^2 so A_sk = Σ yᵢ(plain)·Dsk comes out Montgomery, matching x̃_sk = x·R;
        //   QinvM plain so ε = modmul(diff_mont, QinvM) = εQ·R·Q⁻¹·R⁻¹ = ε (plain, for correction).
        uint32_t C1[NB], C2[NB][NB], Dsk[NB], Ecorr[NB], QinvM;
        for (uint32_t i = 0; i < NB; ++i) {
            C1[i] = (uint32_t)qhat_inv[i];                                                 // plain
            Dsk[i] = ctxM.to_montgomery(ctxM.to_montgomery((uint32_t)qhat_mod_sk[i]));      // R^2
            for (uint32_t j = 0; j < NB; ++j)
                C2[i][j] = ctxP[j].to_montgomery(ctxP[j].to_montgomery((uint32_t)qhat_mod_p[i][j]));  // R^2
        }
        for (uint32_t j = 0; j < NB; ++j)
            Ecorr[j] = ctxP[j].to_montgomery(ctxP[j].to_montgomery((uint32_t)(Q % Pm[j]))); // R^2
        QinvM = (uint32_t)Qinv_sk;                                                          // plain

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(0, (1ull << 60) - 1);
        std::vector<u128> val(ELEMS);
        for (uint32_t k = 0; k < ELEMS; ++k) val[k] = dist(rng);

        // input: n tiles x̃_i = mont_{q_i}(x_i) then 1 tile x̃_sk = mont_{m_sk}(x mod m_sk)
        // — ALL Montgomery, matching the domain the previous HE kernel emits.
        std::vector<uint32_t> xblob((NB + 1) * ELEMS);
        for (uint32_t i = 0; i < NB; ++i) {
            std::vector<uint32_t> row(ELEMS);
            for (uint32_t k = 0; k < ELEMS; ++k) row[k] = ctxQ[i].to_montgomery((uint32_t)(val[k] % Qm[i]));
            auto t = tilize(row); std::copy(t.begin(), t.end(), xblob.begin() + i * ELEMS);
        }
        { std::vector<uint32_t> row(ELEMS);
          for (uint32_t k = 0; k < ELEMS; ++k) row[k] = ctxM.to_montgomery((uint32_t)(val[k] % Msk));
          auto t = tilize(row); std::copy(t.begin(), t.end(), xblob.begin() + NB * ELEMS); }

        // reference: EXACT x mod p_j, then Montgomery-encode under p_j
        std::vector<std::vector<uint32_t>> ref(NB, std::vector<uint32_t>(ELEMS));
        for (uint32_t k = 0; k < ELEMS; ++k)
            for (uint32_t j = 0; j < NB; ++j) ref[j][k] = ctxP[j].to_montgomery((uint32_t)(val[k] % Pm[j]));

        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        auto mkbuf = [&](uint32_t n) {
            distributed::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = n * I32}, d, mesh.get()); };
        auto b_x = mkbuf(NB + 1), b_out = mkbuf(NB);

        Program p = CreateProgram();
        auto mkcb = [&](CBIndex idx, uint32_t depth) {
            CircularBufferConfig c(depth * I32, {{idx, tt::DataFormat::UInt32}});
            c.set_page_size(idx, I32); CreateCircularBuffer(p, core, c); };
        mkcb(CBIndex::c_0, NB + 2); mkcb(CBIndex::c_2, 8); mkcb(CBIndex::c_3, 8);
        mkcb(CBIndex::c_4, 2); mkcb(CBIndex::c_5, 2); mkcb(CBIndex::c_16, 8);

        auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/baseconv_fbc_sk.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                          .dst_full_sync_en = true, .math_approx_mode = false});
        SetRuntimeArgs(p, rd, core, {(uint32_t)b_x->address(), NB + 1, 0u});
        SetRuntimeArgs(p, wr, core, {(uint32_t)b_out->address(), NB, 0u});
        std::vector<uint32_t> cargs;
        cargs.push_back(NB);                                             // arg 0 = n
        for (uint32_t i = 0; i < NB; ++i) cargs.push_back(C1[i]);
        for (uint32_t i = 0; i < NB; ++i) for (uint32_t j = 0; j < NB; ++j) cargs.push_back(C2[i][j]);
        for (uint32_t i = 0; i < NB; ++i) cargs.push_back((uint32_t)Qm[i]);
        for (uint32_t i = 0; i < NB; ++i) cargs.push_back(ctxQ[i].q_inv_32());
        for (uint32_t j = 0; j < NB; ++j) cargs.push_back((uint32_t)Pm[j]);
        for (uint32_t j = 0; j < NB; ++j) cargs.push_back(ctxP[j].q_inv_32());
        for (uint32_t i = 0; i < NB; ++i) cargs.push_back(Dsk[i]);
        for (uint32_t j = 0; j < NB; ++j) cargs.push_back(Ecorr[j]);
        cargs.push_back((uint32_t)Msk);
        cargs.push_back(ctxM.q_inv_32());
        cargs.push_back(QinvM);
        SetRuntimeArgs(p, cp, core, cargs);

        distributed::MeshWorkload wl; wl.add_program(dr, std::move(p));
        distributed::EnqueueWriteMeshBuffer(cq, b_x, xblob, false);
        distributed::EnqueueMeshWorkload(cq, wl, false);
        std::vector<uint32_t> outblob;
        distributed::EnqueueReadMeshBuffer(cq, outblob, b_out, true);

        uint32_t mism = 0;
        for (uint32_t j = 0; j < NB; ++j) {
            auto got = untilize(std::vector<uint32_t>(outblob.begin() + j * ELEMS, outblob.begin() + (j + 1) * ELEMS));
            for (uint32_t k = 0; k < ELEMS; ++k)
                if (got[k] != ref[j][k]) { if (mism < 8) fmt::print("  j={} k={} got={} exp={}\n", j, k, got[k], ref[j][k]); ++mism; }
        }
        fmt::print("fused EXACT base-convert (SK, ε on device)  {}->{}  N=1024  seed={}  m_sk={}  mismatches={}/{}\n",
                   NB, NB, seed, Msk, mism, NB * ELEMS);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: fused exact FBC (on-device ε, one kernel) WORKS.\n");
        else fmt::print("RESULT: WRONG.\n");

        if (pass) {
            distributed::EnqueueMeshWorkload(cq, wl, false); distributed::Finish(cq);  // warm
            const int iters = 100;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it) distributed::EnqueueMeshWorkload(cq, wl, false);
            distributed::Finish(cq);
            auto t1 = std::chrono::high_resolution_clock::now();
            fmt::print("timing: {:.2f} us / base-convert (4->4 exact SK fused, 1024 coeffs, 1 core)\n",
                       std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e3 / iters);
        }

        pass &= mesh->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Exception:\n{}\n", e.what());
        return 2;
    }
    if (pass) fmt::print("Test Passed\n");
    else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
