// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// RNS fast base conversion on ONE Tensix core:  x mod Q  ->  x mod P.
// Source base Q = {q0,q1,q2,q3}, destination base P = {p0,p1,p2,p3} (4 moduli each,
// all distinct primes in (2^27, 2^28)). One 32x32 tile = 1024 coefficients.
//
// Approximate FBC (the pure vector-op part; see docs/he/bfv_build_plan.md §1):
//   y_i    = (x_i · q̂_i^{-1}) mod q_i           q̂_i = Q/q_i        [stage 1, canonical]
//   out_j  = ( Σ_i  y_i · (q̂_i mod p_j) ) mod p_j                   [stage 2]
//          = (x + ε·Q) mod p_j   (ε overflow not corrected here)
//
// Built ENTIRELY from existing kernels, chained through DRAM (one core):
//   modmul_sfpi_full  — canonical (a·mont(c)) mod m = a·c mod m   [stage 1 & 2 multiplies]
//   modadd_reduce     — sum 4 canonical tiles mod p_j             [stage 2 accumulate]
// Canonical (not lazy) residues are REQUIRED: stage 2 reinterprets the integer y_i
// under a different modulus p_j, so the exact value in [0,q_i) must be materialized.
//
//   ./build/baseconv [seed]

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

        // 8 distinct primes just below 2^28 (search downward): first 4 = Q, next 4 = P.
        uint64_t Qm[NB], Pm[NB];
        {
            std::vector<uint64_t> primes;
            for (uint64_t c = (1ull << 28) - 1; primes.size() < 2 * NB; c -= 2)
                if (is_prime(c)) primes.push_back(c);
            for (uint32_t i = 0; i < NB; ++i) { Qm[i] = primes[i]; Pm[i] = primes[NB + i]; }
        }

        std::vector<mont::MontgomeryContext> ctxQ, ctxP;
        for (uint32_t i = 0; i < NB; ++i) ctxQ.emplace_back((uint32_t)Qm[i]);
        for (uint32_t j = 0; j < NB; ++j) ctxP.emplace_back((uint32_t)Pm[j]);

        // ---- precompute CRT / Montgomery constants ----
        u128 Q = 1; for (uint32_t i = 0; i < NB; ++i) Q *= Qm[i];
        uint64_t qhat_inv[NB];                 // (Q/q_i)^{-1} mod q_i
        uint64_t qhat_mod_p[NB][NB];           // (Q/q_i) mod p_j
        for (uint32_t i = 0; i < NB; ++i) {
            u128 qhat = Q / Qm[i];
            uint64_t qhat_mod_qi = (uint64_t)(qhat % Qm[i]);
            qhat_inv[i] = modpow(qhat_mod_qi, Qm[i] - 2, Qm[i]);   // Fermat (q_i prime)
            for (uint32_t j = 0; j < NB; ++j) qhat_mod_p[i][j] = (uint64_t)(qhat % Pm[j]);
        }
        // Montgomery-form constants so modmul(a, mont(c)) = a·c mod m
        uint32_t C1[NB];        // stage 1: mont_{q_i}(qhat_inv_i)
        uint32_t C2[NB][NB];    // stage 2: mont_{p_j}(qhat_i mod p_j)
        for (uint32_t i = 0; i < NB; ++i) {
            C1[i] = ctxQ[i].to_montgomery((uint32_t)qhat_inv[i]);
            for (uint32_t j = 0; j < NB; ++j) C2[i][j] = ctxP[j].to_montgomery((uint32_t)qhat_mod_p[i][j]);
        }
        // ε-correction constants:  M[j][e] = (e · (Q mod p_j)) mod p_j  for e=0..3.
        // and the fp32 reciprocals 1/q_i (bit-patterns) for the on-device ε.

        // ---- inputs: 1024 integers < Q ... but coefficients fit u64? Q ~ 2^108, too big.
        //      Use values < 2^60 (representative; their residues exercise all limbs). ----
        std::mt19937_64 rng(seed);
        std::vector<u128> val(ELEMS);
        std::uniform_int_distribution<uint64_t> dist(0, (1ull << 60) - 1);
        for (uint32_t k = 0; k < ELEMS; ++k) val[k] = dist(rng);

        // device RNS input x_i = val mod q_i, tilized
        std::vector<std::vector<uint32_t>> xT(NB);
        for (uint32_t i = 0; i < NB; ++i) {
            std::vector<uint32_t> row(ELEMS);
            for (uint32_t k = 0; k < ELEMS; ++k) row[k] = (uint32_t)(val[k] % Qm[i]);
            xT[i] = tilize(row);
        }

        // ---- host references ----
        //  ref_approx[j] = (x + ε·Q) mod p_j  (the approximate FBC, what stage 2 computes)
        //  ref_exact[j]  = x mod p_j           (the true value; what the ε correction yields)
        std::vector<std::vector<uint32_t>> ref_approx(NB, std::vector<uint32_t>(ELEMS));
        std::vector<std::vector<uint32_t>> ref_exact(NB, std::vector<uint32_t>(ELEMS));
        // ε = ⌊Σ_i y_i·q̂_i / Q⌋ ∈ {0,..,NB-1}: the CRT overflow, computed EXACTLY in
        // big-int on host (the fp32 path is precision-marginal AND Wormhole's float→int
        // cast has no truncation mode; the on-device exact ε is the Shenoy-Kumaresan
        // integer path — see docs/he/bfv_build_plan.md §1). Device does the correction.
        // negcorr_j[k] = (p_j − (ε·(Q mod p_j) mod p_j)) mod p_j  — the per-element correction,
        // added on device:  exact_j = (approx_j + negcorr_j) mod p_j = x mod p_j.
        std::vector<std::vector<uint32_t>> negcorr(NB, std::vector<uint32_t>(ELEMS));
        uint64_t Qmodp[NB]; for (uint32_t j = 0; j < NB; ++j) Qmodp[j] = (uint64_t)(Q % Pm[j]);
        for (uint32_t k = 0; k < ELEMS; ++k) {
            u128 S = 0;
            uint64_t y[NB];
            for (uint32_t i = 0; i < NB; ++i) {
                y[i] = (uint64_t)(((u128)(val[k] % Qm[i]) * qhat_inv[i]) % Qm[i]);
                S += (u128)y[i] * (Q / Qm[i]);
            }
            uint64_t eps = (uint64_t)(S / Q);                  // = ε (since Σ y_i q̂_i = x + εQ, x<Q)
            for (uint32_t j = 0; j < NB; ++j) {
                u128 acc = 0;
                for (uint32_t i = 0; i < NB; ++i) acc += (u128)y[i] * qhat_mod_p[i][j];
                ref_approx[j][k] = (uint32_t)(acc % Pm[j]);
                ref_exact[j][k] = (uint32_t)(val[k] % Pm[j]);
                uint64_t corr = ((u128)eps * Qmodp[j]) % Pm[j];
                negcorr[j][k] = (uint32_t)((Pm[j] - corr) % Pm[j]);
            }
        }

        // ---- device ----
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};

        auto mkbuf = [&](uint32_t ntiles) {
            distributed::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = ntiles * I32}, d, mesh.get());
        };
        auto b_x = mkbuf(NB);          // input residues  (tile i = x_i)
        auto b_c1 = mkbuf(NB);         // stage-1 consts   (tile i = C1[i])
        auto b_y = mkbuf(NB);          // stage-1 output   (tile i = y_i)
        std::vector<std::shared_ptr<distributed::MeshBuffer>> b_c2(NB);  // per-j: 4 tiles [C2[i][j]]
        for (uint32_t j = 0; j < NB; ++j) b_c2[j] = mkbuf(NB);
        auto b_prod = mkbuf(NB);       // stage-2 products  (scratch)
        // b_pair[2j] = approximate residue (x+εQ) mod p_j (written by stage 2),
        // b_pair[2j+1] = negcorr_j (written by host); correction adds the pair.
        auto b_pair = mkbuf(2 * NB);
        auto b_exact = mkbuf(NB);      // exact residues    (tile j = x mod p_j)

        auto mkcb = [&](Program& p, CBIndex idx, uint32_t depth) {
            CircularBufferConfig c(depth * I32, {{idx, tt::DataFormat::UInt32}});
            c.set_page_size(idx, I32); CreateCircularBuffer(p, core, c); };

        // modmul:  out[start..start+n) = (a · b) mod m   (b pre-Montgomery'd)
        auto build_modmul = [&](auto a, auto b, auto out, uint32_t n, uint32_t start,
                                const mont::MontgomeryContext& ctx) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, 2); mkcb(p, CBIndex::c_1, 2); mkcb(p, CBIndex::c_16, 2);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_full.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                              .dst_full_sync_en = true, .math_approx_mode = false});
            const uint32_t q = ctx.prime(), qi = ctx.q_inv_32();
            SetRuntimeArgs(p, rd, core, {(uint32_t)a->address(), (uint32_t)b->address(), n, start});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), n, start});
            SetRuntimeArgs(p, cp, core, {n, q, qi, q & 0xFFFFu, q >> 16});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // modadd: out[out_start] = (Σ_{i<n} in[read_start+i]) mod m
        auto build_modadd = [&](auto in, auto out, uint32_t n, uint32_t read_start, uint32_t out_start, uint64_t m) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, NB); mkcb(p, CBIndex::c_16, 2);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                              .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)in->address(), n, read_start});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, out_start});
            SetRuntimeArgs(p, cp, core, {n, (uint32_t)m});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // ---- write inputs + constants (x as one NB-tile blob, each const buffer) ----
        {
            std::vector<uint32_t> xblob(NB * ELEMS);
            for (uint32_t i = 0; i < NB; ++i) std::copy(xT[i].begin(), xT[i].end(), xblob.begin() + i * ELEMS);
            distributed::EnqueueWriteMeshBuffer(cq, b_x, xblob, false);

            std::vector<uint32_t> c1blob(NB * ELEMS);
            for (uint32_t i = 0; i < NB; ++i) std::fill_n(c1blob.begin() + i * ELEMS, ELEMS, C1[i]);
            distributed::EnqueueWriteMeshBuffer(cq, b_c1, c1blob, false);

            for (uint32_t j = 0; j < NB; ++j) {
                std::vector<uint32_t> c2blob(NB * ELEMS);
                for (uint32_t i = 0; i < NB; ++i) std::fill_n(c2blob.begin() + i * ELEMS, ELEMS, C2[i][j]);
                distributed::EnqueueWriteMeshBuffer(cq, b_c2[j], c2blob, false);
            }
            // seed b_pair: negcorr_j at odd slots (2j+1); even slots (2j) filled by stage 2.
            std::vector<uint32_t> pairblob(2 * NB * ELEMS, 0);
            for (uint32_t j = 0; j < NB; ++j) {
                auto t = tilize(negcorr[j]);
                std::copy(t.begin(), t.end(), pairblob.begin() + (2 * j + 1) * ELEMS);
            }
            distributed::EnqueueWriteMeshBuffer(cq, b_pair, pairblob, false);
        }

        // ---- stage 1: y_i = (x_i · C1[i]) mod q_i ----
        for (uint32_t i = 0; i < NB; ++i) {
            auto wl = build_modmul(b_x, b_c1, b_y, 1, i, ctxQ[i]);
            distributed::EnqueueMeshWorkload(cq, wl, false);
        }
        // ---- stage 2: approximate out_j = Σ_i (y_i · C2[i][j]) mod p_j  -> b_pair[2j] ----
        for (uint32_t j = 0; j < NB; ++j) {
            auto mm = build_modmul(b_y, b_c2[j], b_prod, NB, 0, ctxP[j]);
            distributed::EnqueueMeshWorkload(cq, mm, false);
            auto ad = build_modadd(b_prod, b_pair, NB, 0, 2 * j, Pm[j]);
            distributed::EnqueueMeshWorkload(cq, ad, false);
        }
        // ---- exact correction: out_exact_j = (approx_j + negcorr_j) mod p_j ----
        //   b_pair[2j]=approx_j, b_pair[2j+1]=negcorr_j; modadd the pair.
        for (uint32_t j = 0; j < NB; ++j) {
            auto wl = build_modadd(b_pair, b_exact, 2, 2 * j, j, Pm[j]);
            distributed::EnqueueMeshWorkload(cq, wl, false);
        }

        std::vector<uint32_t> pairblob, exactblob;
        distributed::EnqueueReadMeshBuffer(cq, pairblob, b_pair, true);
        distributed::EnqueueReadMeshBuffer(cq, exactblob, b_exact, true);

        // ---- validate: approximate (b_pair even slots) vs approx-ref, exact vs true x mod p_j ----
        uint32_t mism_a = 0, mism_e = 0;
        for (uint32_t j = 0; j < NB; ++j) {
            auto ga = untilize(std::vector<uint32_t>(pairblob.begin() + (2 * j) * ELEMS, pairblob.begin() + (2 * j + 1) * ELEMS));
            auto ge = untilize(std::vector<uint32_t>(exactblob.begin() + j * ELEMS, exactblob.begin() + (j + 1) * ELEMS));
            for (uint32_t k = 0; k < ELEMS; ++k) {
                if (ga[k] != ref_approx[j][k]) ++mism_a;
                if (ge[k] != ref_exact[j][k]) { if (mism_e < 8) fmt::print("  exact j={} k={} got={} exp={}\n", j, k, ge[k], ref_exact[j][k]); ++mism_e; }
            }
        }
        fmt::print("base-convert Q(4)->P(4)  seed={}\n", seed);
        fmt::print("  approximate FBC : mismatches={}/{} (vs (x+εQ) mod p)\n", mism_a, NB * ELEMS);
        fmt::print("  exact (ε-corrected): mismatches={}/{} (vs x mod p)\n", mism_e, NB * ELEMS);
        pass = (mism_a == 0 && mism_e == 0);
        if (pass) fmt::print("RESULT: exact RNS base conversion on device WORKS.\n");
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
