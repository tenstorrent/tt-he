// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// RNS base conversion whose CRT combine runs on the INT8 MATRIX ENGINE, fully on
// device. L source -> L dest moduli (L = 4 or 8 via argv[2]), N = 1024 coefficients
// = 32 coefficient-tiles (32 rows each), one Tensix core.
//
//   y[c][i] = (x[c][i] · q̂_i^{-1}) mod q_i                     [stage 1, host prep]
//   out_j[c] = (Σ_i y[c][i] · (q̂_i mod p_j)) mod p_j  = (x[c]+εQ) mod p_j   [approx FBC]
//
// Per coefficient-tile: L src × 4 radix-128 limbs contraction lanes; the matrix
// engine (matmul_fbc.cpp) produces 4 int32 radix planes; the per-column-modulus
// recombine (recombine_rns.cpp) reduces each output column j mod p_j on the SFPU.
// ε correction not applied (approximate FBC; see baseconv*.cpp for exact).
//
//   ./build/baseconv_matmul [seed] [L]      (L in {4,8}, default 8)

#include <chrono>
#include <cstdint>
#include <random>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t T = 32, NR = 4, ELEMS = 1024, N = 1024, NCT = N / 32;  // 32 coeff-tiles
const uint64_t POW[NR] = {1, 128, 16384, 2097152};
using u128 = unsigned __int128;

uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) {
    u128 r = 1, bb = b % m;
    while (e) { if (e & 1) r = r * bb % m; bb = bb * bb % m; e >>= 1; }
    return (uint64_t)r; }
bool is_prime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) return false;
    return true; }
int8_t limb(uint64_t v, uint32_t p) { return (int8_t)((v / POW[p]) % 128); }

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
        const uint32_t L = (argc > 2) ? std::stoul(argv[2]) : 8;   // moduli per base

        std::vector<uint64_t> Qm(L), Pm(L);
        { std::vector<uint64_t> pr;
          for (uint64_t c = (1ull << 28) - 1; pr.size() < 2 * L; c -= 2) if (is_prime(c)) pr.push_back(c);
          for (uint32_t i = 0; i < L; ++i) { Qm[i] = pr[i]; Pm[i] = pr[L + i]; } }
        std::vector<mont::MontgomeryContext> ctxP;
        for (uint32_t j = 0; j < L; ++j) ctxP.emplace_back((uint32_t)Pm[j]);

        u128 Q = 1; for (uint32_t i = 0; i < L; ++i) Q *= Qm[i];
        std::vector<uint64_t> qhat_inv(L);
        std::vector<std::vector<uint64_t>> qhat_mod_p(L, std::vector<uint64_t>(L));
        for (uint32_t i = 0; i < L; ++i) {
            u128 qhat = Q / Qm[i];
            qhat_inv[i] = modpow((uint64_t)(qhat % Qm[i]), Qm[i] - 2, Qm[i]);
            for (uint32_t j = 0; j < L; ++j) qhat_mod_p[i][j] = (uint64_t)(qhat % Pm[j]);
        }

        // 5 per-column recombine param tiles (column col -> p_{col<L?col:0}). order K14,QINV,QHIGH,QLOW,Q.
        std::vector<uint32_t> parblob(5 * ELEMS);
        for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c) {
            uint32_t j = (c < L) ? c : 0, at = r * T + c;
            parblob[0 * ELEMS + at] = ctxP[j].to_montgomery(16384u);
            parblob[1 * ELEMS + at] = ctxP[j].q_inv_32();
            parblob[2 * ELEMS + at] = (uint32_t)(Pm[j] >> 16);
            parblob[3 * ELEMS + at] = (uint32_t)(Pm[j] & 0xFFFFu);
            parblob[4 * ELEMS + at] = (uint32_t)Pm[j];
        }
        for (uint32_t t = 0; t < 5; ++t) {
            std::vector<uint32_t> rm(parblob.begin() + t * ELEMS, parblob.begin() + (t + 1) * ELEMS);
            auto til = tilize(rm); std::copy(til.begin(), til.end(), parblob.begin() + t * ELEMS);
        }

        // inputs: N coefficients < Q
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(0, (1ull << 60) - 1);
        std::vector<u128> val(N);
        for (uint32_t c = 0; c < N; ++c) val[c] = dist(rng);

        // y[c][i] (stage 1, host)
        std::vector<std::vector<uint64_t>> y(N, std::vector<uint64_t>(L));
        for (uint32_t c = 0; c < N; ++c)
            for (uint32_t i = 0; i < L; ++i)
                y[c][i] = (uint64_t)(((u128)(val[c] % Qm[i]) * qhat_inv[i]) % Qm[i]);

        // LHS A: NCT tiles, A_t[c'][i*4+p] = limb_p(y[t*32+c'][i])
        std::vector<int8_t> Ablob(NCT * ELEMS, 0);
        for (uint32_t t = 0; t < NCT; ++t) {
            std::vector<int8_t> Arow(ELEMS, 0);
            for (uint32_t cp = 0; cp < T; ++cp)
                for (uint32_t i = 0; i < L; ++i)
                    for (uint32_t p = 0; p < NR; ++p)
                        Arow[cp * T + (i * NR + p)] = limb(y[t * T + cp][i], p);
            auto til = tilize(Arow); std::copy(til.begin(), til.end(), Ablob.begin() + t * ELEMS);
        }

        // MONTGOMERY I/O: fold R into the CRT-fold constant so the output is Montgomery
        // (õut_j = out_j·R mod p_j) — pure constant merge, recombine kernel unchanged.
        // Input-Montgomery is absorbed by the host stage-1 (y_i is the same plain value).
        //   B_r[(i*4+p)][j] = limb_r( (128^p · (q̂_i mod p_j) · R) mod p_j )
        std::vector<int8_t> Bblob(NR * ELEMS, 0);
        for (uint32_t r = 0; r < NR; ++r) {
            std::vector<int8_t> Brow(ELEMS, 0);
            for (uint32_t i = 0; i < L; ++i)
                for (uint32_t p = 0; p < NR; ++p)
                    for (uint32_t j = 0; j < L; ++j) {
                        uint64_t Bp = (uint64_t)(((u128)POW[p] * qhat_mod_p[i][j]) % Pm[j]);
                        Bp = ctxP[j].to_montgomery((uint32_t)Bp);          // ×R (mod p_j)
                        Brow[(i * NR + p) * T + j] = limb(Bp, r);
                    }
            auto til = tilize(Brow); std::copy(til.begin(), til.end(), Bblob.begin() + r * ELEMS);
        }

        // reference: approximate FBC out_j, then Montgomery-encode under p_j
        std::vector<std::vector<uint32_t>> ref(L, std::vector<uint32_t>(N));
        for (uint32_t c = 0; c < N; ++c)
            for (uint32_t j = 0; j < L; ++j) {
                u128 acc = 0; for (uint32_t i = 0; i < L; ++i) acc += (u128)y[c][i] * qhat_mod_p[i][j];
                ref[j][c] = ctxP[j].to_montgomery((uint32_t)(acc % Pm[j]));  // õut_j = out_j·R mod p_j
            }

        // ---- device ----
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        constexpr uint32_t I8 = ELEMS, I32 = ELEMS * 4;

        auto mkbuf = [&](uint32_t n, uint32_t page) {
            distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = n * page}, d, mesh.get()); };
        auto b_A = mkbuf(NCT, I8), b_B = mkbuf(NR, I8), b_planes = mkbuf(NR * NCT, I32);
        auto b_par = mkbuf(5, I32), b_out = mkbuf(NCT, I32);

        auto mkcb = [&](Program& p, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
            CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, core, c); };

        // program 1: int8 matmul over NCT coeff-tiles -> 4*NCT int32 planes
        Program p1 = CreateProgram();
        mkcb(p1, CBIndex::c_0, tt::DataFormat::Int8, 4, I8);
        mkcb(p1, CBIndex::c_1, tt::DataFormat::Int8, NR, I8);
        mkcb(p1, CBIndex::c_16, tt::DataFormat::Int32, 4, I32);
        std::vector<uint32_t> rc1; TensorAccessorArgs(*b_A).append_to(rc1); TensorAccessorArgs(*b_B).append_to(rc1);
        auto rd1 = CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fbc.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc1});
        auto wr1 = CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto cp1 = CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "compute/matmul_fbc.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false});
        SetRuntimeArgs(p1, rd1, core, {(uint32_t)b_A->address(), (uint32_t)b_B->address(), NCT});
        SetRuntimeArgs(p1, wr1, core, {(uint32_t)b_planes->address(), NR * NCT, 0u});
        SetRuntimeArgs(p1, cp1, core, {NCT});

        // program 2: per-column-modulus recombine over NCT coeff-tiles
        Program p2 = CreateProgram();
        mkcb(p2, CBIndex::c_0, tt::DataFormat::UInt32, NR, I32);
        mkcb(p2, CBIndex::c_1, tt::DataFormat::UInt32, 5, I32);
        mkcb(p2, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
        auto rd2 = CreateKernel(p2, OVERRIDE_KERNEL_PREFIX "dataflow/reader_recombine_rns.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto wr2 = CreateKernel(p2, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto cp2 = CreateKernel(p2, OVERRIDE_KERNEL_PREFIX "compute/recombine_rns.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                          .dst_full_sync_en = true, .math_approx_mode = false});
        SetRuntimeArgs(p2, rd2, core, {(uint32_t)b_planes->address(), (uint32_t)b_par->address(), NCT});
        SetRuntimeArgs(p2, wr2, core, {(uint32_t)b_out->address(), NCT, 0u});
        SetRuntimeArgs(p2, cp2, core, {NCT});

        distributed::MeshWorkload w1; w1.add_program(dr, std::move(p1));
        distributed::MeshWorkload w2; w2.add_program(dr, std::move(p2));
        distributed::EnqueueWriteMeshBuffer(cq, b_A, Ablob, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_B, Bblob, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_par, parblob, false);
        distributed::EnqueueMeshWorkload(cq, w1, false);
        distributed::EnqueueMeshWorkload(cq, w2, false);
        std::vector<uint32_t> outblob;
        distributed::EnqueueReadMeshBuffer(cq, outblob, b_out, true);

        uint32_t mism = 0;
        for (uint32_t t = 0; t < NCT; ++t) {
            auto out = untilize(std::vector<uint32_t>(outblob.begin() + t * ELEMS, outblob.begin() + (t + 1) * ELEMS));
            for (uint32_t cp = 0; cp < T; ++cp)
                for (uint32_t j = 0; j < L; ++j) {
                    uint32_t got = out[cp * T + j], exp = ref[j][t * T + cp];
                    if (got != exp) { if (mism < 8) fmt::print("  t={} cp={} j={} got={} exp={}\n", t, cp, j, got, exp); ++mism; }
                }
        }
        fmt::print("matmul base-convert {}->{}  N=1024  seed={}  mismatches={}/{}\n", L, L, seed, mism, N * L);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: matrix-engine FBC (fully on-device) WORKS.\n");
        else fmt::print("RESULT: WRONG.\n");

        if (pass) {
            distributed::EnqueueMeshWorkload(cq, w1, false); distributed::EnqueueMeshWorkload(cq, w2, false);
            distributed::Finish(cq);  // warm
            const int iters = 100;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it) {
                distributed::EnqueueMeshWorkload(cq, w1, false);
                distributed::EnqueueMeshWorkload(cq, w2, false);
            }
            distributed::Finish(cq);
            auto t1 = std::chrono::high_resolution_clock::now();
            fmt::print("timing: {:.2f} us / base-convert ({}->{} matmul+recombine, 1024 coeffs, 1 core)\n",
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
