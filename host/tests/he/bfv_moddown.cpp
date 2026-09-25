// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BFV prototype — mod-down QP -> Q (the last block before relinearization).
// Single core, N=1024, Q=4 / P=4 limbs, Montgomery I/O. See docs/he/bfv_prototype_plan.md.
//
// Given x in QP (Q-part x_Q, P-part x_P), produce y in Q with y ≈ X·P⁻¹ mod Q:
//   x_P→Q = FBC(P→Q)(x_P)              (approximate FBC, error → noise; reuse baseconv_fbc)
//   y_j   = (x_Q,j − x_P→Q,j) · P⁻¹ mod q_j     (modsub + mul_const)
//
// Reuses baseconv_fbc (fused FBC, P→Q) + modsub_reduce + modmul_sfpi_full. No new kernel.
// Validated bit-exact vs a big-int reference using the SAME approximate FBC (standard for
// hybrid key-switch mod-down).
//   ./build/bfv_moddown [seed]

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
#ifndef CFG_LQ
#define CFG_LQ 4
#endif
#ifndef CFG_LP
#define CFG_LP 4
#endif
constexpr uint32_t LQ = CFG_LQ, LP = CFG_LP;      // Q base, P base
using u128 = unsigned __int128;

uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) { u128 r = 1, bb = b % m;
    while (e) { if (e & 1) r = r * bb % m; bb = bb * bb % m; e >>= 1; } return (uint64_t)r; }
bool is_prime(uint64_t n) { if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) return false; return true; }
std::vector<uint64_t> ntt_primes(uint32_t cnt) { std::vector<uint64_t> out;
    for (uint64_t k = ((1ull << 28) - 1) / 1024; out.size() < cnt && k > 0; --k) {
        uint64_t p = k * 1024 + 1; if (p < (1ull << 28) && is_prime(p)) out.push_back(p); } return out; }
template <typename E> std::vector<E> tilize(const std::vector<E>& in) { std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = in[r * T + c]; return o; }
template <typename E> std::vector<E> untilize(const std::vector<E>& in) { std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[r * T + c] = in[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)]; return o; }
}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const uint32_t seed = (argc > 1) ? std::stoul(argv[1]) : 0;
        auto pr = ntt_primes(LQ + LP);
        std::vector<uint64_t> Qm(pr.begin(), pr.begin() + LQ), Pm(pr.begin() + LQ, pr.begin() + LQ + LP);
        std::vector<mont::MontgomeryContext> ctxQ, ctxP;
        for (auto q : Qm) ctxQ.emplace_back((uint32_t)q);
        for (auto p : Pm) ctxP.emplace_back((uint32_t)p);

        // ---- FBC(P->Q) constants (source=P, dest=Q), Montgomery I/O ----
        u128 Pfull = 1; for (auto p : Pm) Pfull *= p;
        std::vector<uint64_t> phat_inv(LP);                 // (P/p_i)^{-1} mod p_i
        std::vector<std::vector<uint64_t>> phat_mod_q(LP, std::vector<uint64_t>(LQ));   // (P/p_i) mod q_j
        for (uint32_t i = 0; i < LP; ++i) {
            u128 phat = Pfull / Pm[i];
            phat_inv[i] = modpow((uint64_t)(phat % Pm[i]), Pm[i] - 2, Pm[i]);
            for (uint32_t j = 0; j < LQ; ++j) phat_mod_q[i][j] = (uint64_t)(phat % Qm[j]);
        }
        std::vector<uint32_t> C1(LP);
        std::vector<std::vector<uint32_t>> C2(LP, std::vector<uint32_t>(LQ));
        for (uint32_t i = 0; i < LP; ++i) {
            C1[i] = (uint32_t)phat_inv[i];                                            // plain
            for (uint32_t j = 0; j < LQ; ++j)
                C2[i][j] = ctxQ[j].to_montgomery(ctxQ[j].to_montgomery((uint32_t)phat_mod_q[i][j]));  // R^2
        }
        // P^{-1} mod q_j, Montgomery-encoded (for the ×P^{-1} mul_const)
        std::vector<uint32_t> PinvMont(LQ);
        for (uint32_t j = 0; j < LQ; ++j) PinvMont[j] = ctxQ[j].to_montgomery((uint32_t)modpow((uint64_t)(Pfull % Qm[j]), Qm[j] - 2, Qm[j]));

        // ---- input X (< 2^110, fits u128) → residues x_Q, x_P ----
        std::mt19937_64 rng(seed);
        std::vector<u128> X(N);
        for (uint32_t n = 0; n < N; ++n) { u128 v = ((u128)rng() << 46) ^ (u128)rng(); X[n] = v % (((u128)1) << 110); }

        // ---- CPU reference (same approximate FBC), Montgomery-encoded ----
        std::vector<std::vector<uint32_t>> yref(LQ, std::vector<uint32_t>(N));
        for (uint32_t n = 0; n < N; ++n) {
            uint64_t xp[LP];
            for (uint32_t i = 0; i < LP; ++i) xp[i] = (uint64_t)(X[n] % Pm[i]);
            uint64_t yi[LP];
            for (uint32_t i = 0; i < LP; ++i) yi[i] = (uint64_t)(((u128)xp[i] * phat_inv[i]) % Pm[i]);
            for (uint32_t j = 0; j < LQ; ++j) {
                u128 acc = 0; for (uint32_t i = 0; i < LP; ++i) acc += (u128)yi[i] * phat_mod_q[i][j];
                uint32_t fbc = (uint32_t)(acc % Qm[j]);                       // approxFBC = (X+εP) mod q_j
                uint32_t xq = (uint32_t)(X[n] % Qm[j]);
                uint32_t diff = (uint32_t)(((u128)xq + Qm[j] - fbc) % Qm[j]);
                uint32_t y = (uint32_t)(((u128)diff * modpow((uint64_t)(Pfull % Qm[j]), Qm[j] - 2, Qm[j])) % Qm[j]);
                yref[j][n] = ctxQ[j].to_montgomery(y);
            }
        }

        // ---- device ----
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        auto mkbuf = [&](uint32_t nt) {
            distributed::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * I32}, d, mesh.get()); };
        auto b_xP = mkbuf(LP), b_xQ = mkbuf(LQ), b_fbc = mkbuf(LQ), b_diff = mkbuf(LQ), b_Pinv = mkbuf(LQ), b_y = mkbuf(LQ);

        // upload x_P (mont_pi), x_Q (mont_qj), Pinv tiles (mont_qj broadcast)
        { std::vector<uint32_t> blob(LP * ELEMS);
          for (uint32_t i = 0; i < LP; ++i) { std::vector<uint32_t> row(N);
            for (uint32_t n = 0; n < N; ++n) row[n] = ctxP[i].to_montgomery((uint32_t)(X[n] % Pm[i]));
            auto t = tilize(row); std::copy(t.begin(), t.end(), blob.begin() + i * ELEMS); }
          distributed::EnqueueWriteMeshBuffer(cq, b_xP, blob, false); }
        { std::vector<uint32_t> blob(LQ * ELEMS);
          for (uint32_t j = 0; j < LQ; ++j) { std::vector<uint32_t> row(N);
            for (uint32_t n = 0; n < N; ++n) row[n] = ctxQ[j].to_montgomery((uint32_t)(X[n] % Qm[j]));
            auto t = tilize(row); std::copy(t.begin(), t.end(), blob.begin() + j * ELEMS); }
          distributed::EnqueueWriteMeshBuffer(cq, b_xQ, blob, false); }
        { std::vector<uint32_t> blob(LQ * ELEMS);
          for (uint32_t j = 0; j < LQ; ++j) { std::vector<uint32_t> row(N, PinvMont[j]);
            auto t = tilize(row); std::copy(t.begin(), t.end(), blob.begin() + j * ELEMS); }
          distributed::EnqueueWriteMeshBuffer(cq, b_Pinv, blob, false); }

        auto mkcb = [&](Program& p, CBIndex idx, uint32_t depth) {
            CircularBufferConfig c(depth * I32, {{idx, tt::DataFormat::UInt32}}); c.set_page_size(idx, I32); CreateCircularBuffer(p, core, c); };
        std::vector<distributed::MeshWorkload> wls; wls.reserve(16);
        auto run = [&](Program&& p){ wls.emplace_back(); wls.back().add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wls.back(), false); };

        // program 1: FBC(P->Q)  x_P (b_xP) -> x_P→Q (b_fbc)   [fused baseconv_fbc, L=LP=LQ=4]
        { Program p = CreateProgram();
          mkcb(p, CBIndex::c_0, 8); mkcb(p, CBIndex::c_2, 8); mkcb(p, CBIndex::c_3, 8); mkcb(p, CBIndex::c_16, 8);
          auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core,
              DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
          auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
              DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
          auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/baseconv_fbc.cpp", core,
              ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
          SetRuntimeArgs(p, rd, core, {(uint32_t)b_xP->address(), LP, 0u});
          SetRuntimeArgs(p, wr, core, {(uint32_t)b_fbc->address(), LQ, 0u});
          std::vector<uint32_t> ca; ca.push_back(LP);                              // n = 4
          for (uint32_t i = 0; i < LP; ++i) ca.push_back(C1[i]);
          for (uint32_t i = 0; i < LP; ++i) for (uint32_t j = 0; j < LQ; ++j) ca.push_back(C2[i][j]);
          for (uint32_t i = 0; i < LP; ++i) ca.push_back((uint32_t)Pm[i]);         // source moduli = P
          for (uint32_t i = 0; i < LP; ++i) ca.push_back(ctxP[i].q_inv_32());
          for (uint32_t j = 0; j < LQ; ++j) ca.push_back((uint32_t)Qm[j]);         // dest moduli = Q
          for (uint32_t j = 0; j < LQ; ++j) ca.push_back(ctxQ[j].q_inv_32());
          SetRuntimeArgs(p, cp, core, ca);
          run(std::move(p)); }

        // program 2 (per limb j): diff = (x_Q,j − x_P→Q,j) mod q_j   [modsub_reduce, reader_ab]
        for (uint32_t j = 0; j < LQ; ++j) {
          Program p = CreateProgram();
          mkcb(p, CBIndex::c_0, 2); mkcb(p, CBIndex::c_1, 2); mkcb(p, CBIndex::c_16, 2);
          auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp", core,
              DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
          auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core,
              DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
          auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modsub_reduce.cpp", core,
              ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
          SetRuntimeArgs(p, rd, core, {(uint32_t)b_xQ->address(), (uint32_t)b_fbc->address(), 1u, j, j});
          SetRuntimeArgs(p, wr, core, {(uint32_t)b_diff->address(), 1u, j});
          SetRuntimeArgs(p, cp, core, {(uint32_t)Qm[j]});
          run(std::move(p)); }

        // program 3 (per limb j): y = diff · P^{-1} mod q_j   [modmul_sfpi_full, reader_modmul]
        for (uint32_t j = 0; j < LQ; ++j) {
          Program p = CreateProgram();
          mkcb(p, CBIndex::c_0, 2); mkcb(p, CBIndex::c_1, 2); mkcb(p, CBIndex::c_16, 2);
          auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", core,
              DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
          auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core,
              DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
          auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_full.cpp", core,
              ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
          const uint32_t q = Qm[j];
          SetRuntimeArgs(p, rd, core, {(uint32_t)b_diff->address(), (uint32_t)b_Pinv->address(), 1u, j});
          SetRuntimeArgs(p, wr, core, {(uint32_t)b_y->address(), 1u, j});
          SetRuntimeArgs(p, cp, core, {1u, q, ctxQ[j].q_inv_32(), q & 0xFFFFu, q >> 16});
          run(std::move(p)); }

        std::vector<uint32_t> outblob;
        distributed::EnqueueReadMeshBuffer(cq, outblob, b_y, true);

        uint32_t mism = 0;
        for (uint32_t j = 0; j < LQ; ++j) {
            auto got = untilize(std::vector<uint32_t>(outblob.begin() + j * ELEMS, outblob.begin() + (j + 1) * ELEMS));
            uint32_t lm = 0;
            for (uint32_t n = 0; n < N; ++n) if (got[n] != yref[j][n]) { ++lm; ++mism; }
            fmt::print("  limb{} mism={}/{}\n", j, lm, N);
        }
        fmt::print("BFV mod-down QP->Q  N=1024 Q(4)/P(4)  seed={}  mismatches={}/{}\n", seed, mism, LQ * N);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: BFV mod-down WORKS.\n"); else fmt::print("RESULT: WRONG.\n");
        pass &= mesh->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Exception:\n{}\n", e.what());
        return 2;
    }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
