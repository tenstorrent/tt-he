// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FUSED RNS base conversion in ONE program/compute kernel (baseconv_fbc.cpp):
// approximate FBC  x mod Q -> (x+εQ) mod P  for L source -> L dest moduli, on
// N=1024 coefficients (one 32x32 tile per limb), one core. y_i intermediates stay
// in an on-core L1 CB (no DRAM round-trips, one dispatch). L = 4 or 8 via argv[2].
//
//   ./build/baseconv_fused [seed] [L]      (L in {4,8}, default 4)

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
constexpr uint32_t T = 32, ELEMS = 1024;
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
        const uint32_t L = (argc > 2) ? std::stoul(argv[2]) : 4;   // moduli per base

        std::vector<uint64_t> Qm(L), Pm(L);
        { std::vector<uint64_t> pr;
          for (uint64_t c = (1ull << 28) - 1; pr.size() < 2 * L; c -= 2) if (is_prime(c)) pr.push_back(c);
          for (uint32_t i = 0; i < L; ++i) { Qm[i] = pr[i]; Pm[i] = pr[L + i]; } }
        std::vector<mont::MontgomeryContext> ctxQ, ctxP;
        for (uint32_t i = 0; i < L; ++i) ctxQ.emplace_back((uint32_t)Qm[i]);
        for (uint32_t j = 0; j < L; ++j) ctxP.emplace_back((uint32_t)Pm[j]);

        u128 Q = 1; for (uint32_t i = 0; i < L; ++i) Q *= Qm[i];
        std::vector<uint64_t> qhat_inv(L);
        std::vector<std::vector<uint64_t>> qhat_mod_p(L, std::vector<uint64_t>(L));
        for (uint32_t i = 0; i < L; ++i) {
            u128 qhat = Q / Qm[i];
            qhat_inv[i] = modpow((uint64_t)(qhat % Qm[i]), Qm[i] - 2, Qm[i]);
            for (uint32_t j = 0; j < L; ++j) qhat_mod_p[i][j] = (uint64_t)(qhat % Pm[j]);
        }
        // MONTGOMERY I/O: input x_i and output out_j are both in Montgomery form
        // (ṽ = v·R mod m). The device modmul is REDC(a·b)=a·b·R⁻¹, so we merge the
        // R factors into the baked constants (kernel unchanged):
        //   C1 = q̂_i⁻¹        (PLAIN):   modmul(x̃_i, C1) = x_i·R·q̂⁻¹·R⁻¹ = y_i  (plain)
        //   C2 = c_ij·R²   (DOUBLE-mont): modmul(y_i, C2) = y_i·c_ij·R  → Σ = out_j·R = õut_j
        // y_i stays PLAIN/canonical internally (required for the cross-modulus combine).
        std::vector<uint32_t> C1(L);
        std::vector<std::vector<uint32_t>> C2(L, std::vector<uint32_t>(L));
        for (uint32_t i = 0; i < L; ++i) {
            C1[i] = (uint32_t)qhat_inv[i];                                   // plain (no mont)
            for (uint32_t j = 0; j < L; ++j)
                C2[i][j] = ctxP[j].to_montgomery(ctxP[j].to_montgomery((uint32_t)qhat_mod_p[i][j]));  // R^2
        }

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(0, (1ull << 60) - 1);
        std::vector<u128> val(ELEMS);
        for (uint32_t k = 0; k < ELEMS; ++k) val[k] = dist(rng);

        // input tiles: x̃_i = mont_{q_i}(x_i) = x_i·R mod q_i
        std::vector<uint32_t> xblob(L * ELEMS);
        for (uint32_t i = 0; i < L; ++i) {
            std::vector<uint32_t> row(ELEMS);
            for (uint32_t k = 0; k < ELEMS; ++k) row[k] = ctxQ[i].to_montgomery((uint32_t)(val[k] % Qm[i]));
            auto t = tilize(row); std::copy(t.begin(), t.end(), xblob.begin() + i * ELEMS);
        }
        // reference: approximate FBC out_j, then Montgomery-encode under p_j
        std::vector<std::vector<uint32_t>> ref(L, std::vector<uint32_t>(ELEMS));
        for (uint32_t k = 0; k < ELEMS; ++k) {
            std::vector<uint64_t> y(L);
            for (uint32_t i = 0; i < L; ++i) y[i] = (uint64_t)(((u128)(val[k] % Qm[i]) * qhat_inv[i]) % Qm[i]);
            for (uint32_t j = 0; j < L; ++j) {
                u128 acc = 0; for (uint32_t i = 0; i < L; ++i) acc += (u128)y[i] * qhat_mod_p[i][j];
                uint32_t out_plain = (uint32_t)(acc % Pm[j]);
                ref[j][k] = ctxP[j].to_montgomery(out_plain);               // õut_j = out_j·R mod p_j
            }
        }

        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        auto mkbuf = [&](uint32_t n) {
            distributed::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = n * I32}, d, mesh.get()); };
        auto b_x = mkbuf(L), b_out = mkbuf(L);

        Program p = CreateProgram();
        auto mkcb = [&](CBIndex idx, uint32_t depth) {
            CircularBufferConfig c(depth * I32, {{idx, tt::DataFormat::UInt32}});
            c.set_page_size(idx, I32); CreateCircularBuffer(p, core, c); };
        mkcb(CBIndex::c_0, 8); mkcb(CBIndex::c_2, 8); mkcb(CBIndex::c_3, 8); mkcb(CBIndex::c_16, 8);

        auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/baseconv_fbc.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                          .dst_full_sync_en = true, .math_approx_mode = false});
        SetRuntimeArgs(p, rd, core, {(uint32_t)b_x->address(), L, 0u});
        SetRuntimeArgs(p, wr, core, {(uint32_t)b_out->address(), L, 0u});
        std::vector<uint32_t> cargs;
        cargs.push_back(L);                                          // arg 0 = n
        for (uint32_t i = 0; i < L; ++i) cargs.push_back(C1[i]);
        for (uint32_t i = 0; i < L; ++i) for (uint32_t j = 0; j < L; ++j) cargs.push_back(C2[i][j]);
        for (uint32_t i = 0; i < L; ++i) cargs.push_back((uint32_t)Qm[i]);
        for (uint32_t i = 0; i < L; ++i) cargs.push_back(ctxQ[i].q_inv_32());
        for (uint32_t j = 0; j < L; ++j) cargs.push_back((uint32_t)Pm[j]);
        for (uint32_t j = 0; j < L; ++j) cargs.push_back(ctxP[j].q_inv_32());
        SetRuntimeArgs(p, cp, core, cargs);

        distributed::MeshWorkload wl; wl.add_program(dr, std::move(p));
        distributed::EnqueueWriteMeshBuffer(cq, b_x, xblob, false);
        distributed::EnqueueMeshWorkload(cq, wl, false);
        std::vector<uint32_t> outblob;
        distributed::EnqueueReadMeshBuffer(cq, outblob, b_out, true);

        uint32_t mism = 0;
        for (uint32_t j = 0; j < L; ++j) {
            auto got = untilize(std::vector<uint32_t>(outblob.begin() + j * ELEMS, outblob.begin() + (j + 1) * ELEMS));
            for (uint32_t k = 0; k < ELEMS; ++k)
                if (got[k] != ref[j][k]) { if (mism < 8) fmt::print("  j={} k={} got={} exp={}\n", j, k, got[k], ref[j][k]); ++mism; }
        }
        fmt::print("fused base-convert (montgomery I/O) {}->{}  N=1024  seed={}  mismatches={}/{}\n", L, L, seed, mism, L * ELEMS);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: fused approximate FBC (one kernel) WORKS.\n");
        else fmt::print("RESULT: WRONG.\n");

        if (pass) {
            distributed::EnqueueMeshWorkload(cq, wl, false); distributed::Finish(cq);  // warm
            const int iters = 100;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it) distributed::EnqueueMeshWorkload(cq, wl, false);
            distributed::Finish(cq);
            auto t1 = std::chrono::high_resolution_clock::now();
            fmt::print("timing: {:.2f} us / base-convert ({}->{} fused, 1024 coeffs, 1 core)\n",
                       std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e3 / iters, L, L);
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
