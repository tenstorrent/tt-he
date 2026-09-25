// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// EXACT RNS base conversion x mod Q -> x mod P on one Tensix core, with the CRT
// overflow ε computed FULLY ON DEVICE via the Shenoy-Kumaresan integer method
// (no floating point — see docs/he/bfv_build_plan.md §1). Companion to
// baseconv.cpp (which computes ε on host).
//
// Carry x redundantly: x mod q_i (i<NB) AND x_sk = x mod m_sk for a sentinel
// prime m_sk (coprime to Q,P; m_sk > NB so ε ∈ {0..NB-1} is uniquely recovered).
//
//   y_i   = (x_i · q̂_i^{-1}) mod q_i                      [stage 1, canonical]
//   A_sk  = (Σ_i y_i · (q̂_i mod m_sk)) mod m_sk = (x+εQ) mod m_sk
//   ε     = ((A_sk − x_sk) · Q^{-1}) mod m_sk             [exact integer ε, on device]
//   A_j   = (Σ_i y_i · (q̂_i mod p_j)) mod p_j  = (x+εQ) mod p_j   [stage 2, approx]
//   out_j = (A_j − ε·(Q mod p_j)) mod p_j       = x mod p_j        [exact correction]
//
// All multiplies are modmul_sfpi_full (canonical (a·mont(c)) mod m); subtractions
// are modsub_reduce; sums are modadd_reduce. reader_ab gives independent A/B tile
// offsets so every operand pairing lines up. ./build/baseconv_sk [seed]

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
    return (uint64_t)r;
}
bool is_prime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) return false;
    return true;
}
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
        {
            std::vector<uint64_t> pr;
            for (uint64_t c = (1ull << 28) - 1; pr.size() < 2 * NB + 1; c -= 2)
                if (is_prime(c)) pr.push_back(c);
            for (uint32_t i = 0; i < NB; ++i) { Qm[i] = pr[i]; Pm[i] = pr[NB + i]; }
            Msk = pr[2 * NB];
        }
        std::vector<mont::MontgomeryContext> ctxQ, ctxP;
        for (uint32_t i = 0; i < NB; ++i) ctxQ.emplace_back((uint32_t)Qm[i]);
        for (uint32_t j = 0; j < NB; ++j) ctxP.emplace_back((uint32_t)Pm[j]);
        mont::MontgomeryContext ctxM((uint32_t)Msk);

        // ---- CRT / Montgomery constants ----
        u128 Q = 1; for (uint32_t i = 0; i < NB; ++i) Q *= Qm[i];
        uint64_t qhat_inv[NB], qhat_mod_p[NB][NB], qhat_mod_sk[NB];
        for (uint32_t i = 0; i < NB; ++i) {
            u128 qhat = Q / Qm[i];
            qhat_inv[i] = modpow((uint64_t)(qhat % Qm[i]), Qm[i] - 2, Qm[i]);
            for (uint32_t j = 0; j < NB; ++j) qhat_mod_p[i][j] = (uint64_t)(qhat % Pm[j]);
            qhat_mod_sk[i] = (uint64_t)(qhat % Msk);
        }
        uint64_t Qinv_sk = modpow((uint64_t)(Q % Msk), Msk - 2, Msk);   // Q^{-1} mod m_sk
        // Montgomery-form constants:  modmul(a, mont(c)) = a·c mod m
        uint32_t C1[NB], C2[NB][NB], Dsk[NB], Ecorr[NB], QinvM;
        for (uint32_t i = 0; i < NB; ++i) {
            C1[i] = ctxQ[i].to_montgomery((uint32_t)qhat_inv[i]);
            Dsk[i] = ctxM.to_montgomery((uint32_t)qhat_mod_sk[i]);
            for (uint32_t j = 0; j < NB; ++j) C2[i][j] = ctxP[j].to_montgomery((uint32_t)qhat_mod_p[i][j]);
        }
        for (uint32_t j = 0; j < NB; ++j) Ecorr[j] = ctxP[j].to_montgomery((uint32_t)(Q % Pm[j]));
        QinvM = ctxM.to_montgomery((uint32_t)Qinv_sk);

        // ---- inputs: 1024 integers < 2^60 (< Q) ----
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(0, (1ull << 60) - 1);
        std::vector<u128> val(ELEMS);
        for (uint32_t k = 0; k < ELEMS; ++k) val[k] = dist(rng);

        std::vector<std::vector<uint32_t>> xT(NB);
        for (uint32_t i = 0; i < NB; ++i) {
            std::vector<uint32_t> row(ELEMS);
            for (uint32_t k = 0; k < ELEMS; ++k) row[k] = (uint32_t)(val[k] % Qm[i]);
            xT[i] = tilize(row);
        }
        std::vector<uint32_t> xskRow(ELEMS);
        for (uint32_t k = 0; k < ELEMS; ++k) xskRow[k] = (uint32_t)(val[k] % Msk);
        auto xskT = tilize(xskRow);

        // ---- exact reference  x mod p_j ----
        std::vector<std::vector<uint32_t>> ref_exact(NB, std::vector<uint32_t>(ELEMS));
        for (uint32_t k = 0; k < ELEMS; ++k)
            for (uint32_t j = 0; j < NB; ++j) ref_exact[j][k] = (uint32_t)(val[k] % Pm[j]);

        // ---- device ----
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};

        auto mkbuf = [&](uint32_t ntiles) {
            distributed::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = ntiles * I32}, d, mesh.get());
        };
        auto b_x = mkbuf(NB), b_c1 = mkbuf(NB), b_y = mkbuf(NB);
        auto b_xsk = mkbuf(1), b_D = mkbuf(NB), b_prodsk = mkbuf(NB);
        auto b_Ask = mkbuf(1), b_diff = mkbuf(1), b_QinvM = mkbuf(1), b_eps = mkbuf(1);
        std::vector<std::shared_ptr<distributed::MeshBuffer>> b_c2(NB);
        for (uint32_t j = 0; j < NB; ++j) b_c2[j] = mkbuf(NB);
        auto b_prod = mkbuf(NB), b_approx = mkbuf(NB);
        auto b_Ecorr = mkbuf(NB), b_corr = mkbuf(NB), b_exact = mkbuf(NB);

        auto mkcb = [&](Program& p, CBIndex idx, uint32_t depth) {
            CircularBufferConfig c(depth * I32, {{idx, tt::DataFormat::UInt32}});
            c.set_page_size(idx, I32); CreateCircularBuffer(p, core, c); };

        // modmul: out[w_start..] = (a[a_start..] · b[b_start..]) mod m, b pre-Montgomery'd
        auto build_modmul = [&](auto a, auto b, auto out, uint32_t n, uint32_t a_start,
                                uint32_t b_start, uint32_t w_start, const mont::MontgomeryContext& ctx) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, 2); mkcb(p, CBIndex::c_1, 2); mkcb(p, CBIndex::c_16, 2);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_full.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                              .dst_full_sync_en = true, .math_approx_mode = false});
            const uint32_t q = ctx.prime(), qi = ctx.q_inv_32();
            SetRuntimeArgs(p, rd, core, {(uint32_t)a->address(), (uint32_t)b->address(), n, a_start, b_start});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), n, w_start});
            SetRuntimeArgs(p, cp, core, {n, q, qi, q & 0xFFFFu, q >> 16});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // modsub: out[w_start] = (a[a_start] − b[b_start]) mod m
        auto build_modsub = [&](auto a, auto b, auto out, uint32_t a_start, uint32_t b_start,
                                uint32_t w_start, uint64_t m) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, 2); mkcb(p, CBIndex::c_1, 2); mkcb(p, CBIndex::c_16, 2);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modsub_reduce.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                              .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)a->address(), (uint32_t)b->address(), 1u, a_start, b_start});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, w_start});
            SetRuntimeArgs(p, cp, core, {(uint32_t)m});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // modadd: out[w_start] = (Σ_{i<n} in[i]) mod m  (contiguous from id 0)
        auto build_modadd = [&](auto in, auto out, uint32_t n, uint32_t w_start, uint64_t m) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, NB); mkcb(p, CBIndex::c_16, 2);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                              .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)in->address(), n, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, w_start});
            SetRuntimeArgs(p, cp, core, {n, (uint32_t)m});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        auto fill_buf = [&](auto buf, uint32_t ntiles, auto gen) {
            std::vector<uint32_t> blob(ntiles * ELEMS);
            for (uint32_t t = 0; t < ntiles; ++t) { auto tile = gen(t); std::copy(tile.begin(), tile.end(), blob.begin() + t * ELEMS); }
            distributed::EnqueueWriteMeshBuffer(cq, buf, blob, false); };
        auto const_tile = [&](uint32_t v) { return std::vector<uint32_t>(ELEMS, v); };

        // ---- upload inputs + constants ----
        fill_buf(b_x, NB, [&](uint32_t i){ return xT[i]; });
        distributed::EnqueueWriteMeshBuffer(cq, b_xsk, xskT, false);
        fill_buf(b_c1, NB, [&](uint32_t i){ return const_tile(C1[i]); });
        fill_buf(b_D, NB, [&](uint32_t i){ return const_tile(Dsk[i]); });
        distributed::EnqueueWriteMeshBuffer(cq, b_QinvM, const_tile(QinvM), false);
        for (uint32_t j = 0; j < NB; ++j) fill_buf(b_c2[j], NB, [&](uint32_t i){ return const_tile(C2[i][j]); });
        fill_buf(b_Ecorr, NB, [&](uint32_t j){ return const_tile(Ecorr[j]); });

        // workloads must outlive their (non-blocking) enqueue; keep them alive here.
        std::vector<distributed::MeshWorkload> wls; wls.reserve(64);
        auto run = [&](distributed::MeshWorkload&& w) {
            wls.push_back(std::move(w));
            distributed::EnqueueMeshWorkload(cq, wls.back(), false); };

        // ---- stage 1: y_i = (x_i · C1[i]) mod q_i ----
        for (uint32_t i = 0; i < NB; ++i)
            run(build_modmul(b_x, b_c1, b_y, 1, i, i, i, ctxQ[i]));

        // ---- SK ε: A_sk = Σ_i y_i·D[i] mod m_sk ; ε = ((A_sk − x_sk)·Q^{-1}) mod m_sk ----
        run(build_modmul(b_y, b_D, b_prodsk, NB, 0, 0, 0, ctxM));
        run(build_modadd(b_prodsk, b_Ask, NB, 0, Msk));
        run(build_modsub(b_Ask, b_xsk, b_diff, 0, 0, 0, Msk));
        run(build_modmul(b_diff, b_QinvM, b_eps, 1, 0, 0, 0, ctxM));

        // ---- stage 2: A_j = Σ_i y_i·C2[i][j] mod p_j  -> b_approx[j] ----
        for (uint32_t j = 0; j < NB; ++j) {
            run(build_modmul(b_y, b_c2[j], b_prod, NB, 0, 0, 0, ctxP[j]));
            run(build_modadd(b_prod, b_approx, NB, j, Pm[j]));
        }
        // ---- correction: corr_j = ε·Ecorr[j] mod p_j ; out_j = (A_j − corr_j) mod p_j ----
        for (uint32_t j = 0; j < NB; ++j) {
            run(build_modmul(b_eps, b_Ecorr, b_corr, 1, 0, j, j, ctxP[j]));
            run(build_modsub(b_approx, b_corr, b_exact, j, j, j, Pm[j]));
        }

        std::vector<uint32_t> exactblob;
        distributed::EnqueueReadMeshBuffer(cq, exactblob, b_exact, true);

        uint32_t mism = 0;
        for (uint32_t j = 0; j < NB; ++j) {
            auto ge = untilize(std::vector<uint32_t>(exactblob.begin() + j * ELEMS, exactblob.begin() + (j + 1) * ELEMS));
            for (uint32_t k = 0; k < ELEMS; ++k)
                if (ge[k] != ref_exact[j][k]) { if (mism < 8) fmt::print("  j={} k={} got={} exp={}\n", j, k, ge[k], ref_exact[j][k]); ++mism; }
        }
        fmt::print("base-convert (Shenoy-Kumaresan, ε on device)  seed={}  m_sk={}\n", seed, Msk);
        fmt::print("  exact: mismatches={}/{} (vs x mod p)\n", mism, NB * ELEMS);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: exact RNS base conversion (on-device ε) WORKS.\n");
        else fmt::print("RESULT: WRONG.\n");

        pass &= mesh->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Exception:\n{}\n", e.what());
        return 2;
    }
    if (pass) fmt::print("Test Passed\n");
    else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
