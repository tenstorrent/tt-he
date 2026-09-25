// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// NAIVE two-factor (128,256) NTT-32768 baseline — see docs/matmul/history/NTT_32768_2FACTOR_PLAN.md.
// Textbook four-step, NO NTT-level tiling/fold/corner-turn: just two large int8 modular
// matmuls + one large modular twiddle. Compared against the hand-tuned 3-factor.
//
//   A[n1][n2] = x[n1 + 128*n2]   (128x256)
//   M1  S  = A @ W256     (128x256)@(256x256), contract n2 (K=256)
//   TW  S' = S (.) Tb,  Tb[n1][u]=wN^{n1 u}
//   M2  R  = W128 @ S'    (128x128)@(128x256), contract n1 (K=128)
//   out[u + 256*v] = R[v][u]
//
// W256/W128 folded into the constant offline (no runtime fold). The K=256 matmul makes
// the radix-128 planes reach ~2^22, so recombine uses recombine_bigK. Same C=1..32
// core-count sweep as ntt32768_dev.cpp.

#include <chrono>
#include <cstdint>
#include <random>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t T = 32, NB = 4, ELEMS = 1024;
constexpr uint32_t I8 = ELEMS * 1, I32 = ELEMS * 4;
constexpr uint32_t N = 32768, N1 = 128, N2 = 256, NTILE = 32;   // output tiles per stage
const uint64_t POW[4] = {1, 128, 16384, 2097152};

uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) { uint64_t r = 1; b %= m;
    while (e) { if (e & 1) r = (__uint128_t)r * b % m; b = (__uint128_t)b * b % m; e >>= 1; } return r; }
uint64_t prim_root(uint64_t q) { uint64_t phi = q - 1, n = phi; std::vector<uint64_t> f;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) { f.push_back(d); while (n % d == 0) n /= d; }
    if (n > 1) f.push_back(n);
    for (uint64_t g = 2; g < q; ++g) { bool ok = true; for (uint64_t p : f) if (modpow(g, phi / p, q) == 1) ok = false;
        if (ok) return g; } throw std::runtime_error("no root"); }

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
        const uint64_t q = (argc > 1) ? std::stoull(argv[1]) : 167772161ULL;
        const uint32_t seed = (argc > 2) ? std::stoul(argv[2]) : 0;
        mont::MontgomeryContext ctx((uint32_t)q);
        const uint32_t QINV = ctx.q_inv_32(), QLO = q & 0xFFFF, QHI = q >> 16;
        const uint32_t K14 = ctx.to_montgomery(16384u);
        auto limb = [](uint32_t v, uint32_t p) -> int8_t { return (int8_t)((v >> (7 * p)) & 127); };

        // ---- roots & DFT matrices ----
        uint64_t g = prim_root(q), wN = modpow(g, (q - 1) / N, q);
        uint64_t w256 = modpow(wN, N / N2, q), w128 = modpow(wN, N / N1, q);
        std::vector<uint64_t> W256(N2 * N2), W128(N1 * N1);
        for (uint32_t a = 0; a < N2; ++a) for (uint32_t b = 0; b < N2; ++b) W256[a * N2 + b] = modpow(w256, (uint64_t)a * b % N2, q);
        for (uint32_t a = 0; a < N1; ++a) for (uint32_t b = 0; b < N1; ++b) W128[a * N1 + b] = modpow(w128, (uint64_t)a * b % N1, q);

        // ---- folded constants (offline) ----
        // M1 fold-RHS:  Wrhs256_{p,r}[kt][nt] = limb_r(128^p * W256_block)   layout (((p*4+r)*Kt+kt)*Nt+nt), Kt=Nt=8
        const uint32_t M1_Kt = 8, M1_Nt = 8;
        std::vector<int8_t> Wrhs256(NB * NB * M1_Kt * M1_Nt * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r)
            for (uint32_t kt = 0; kt < M1_Kt; ++kt) for (uint32_t nt = 0; nt < M1_Nt; ++nt) {
                std::vector<int8_t> blk(ELEMS);
                for (uint32_t i = 0; i < T; ++i) for (uint32_t j = 0; j < T; ++j)
                    blk[i * T + j] = limb((uint32_t)((POW[p] * W256[(kt * T + i) * N2 + (nt * T + j)]) % q), r);
                auto tt = tilize(blk);
                std::copy(tt.begin(), tt.end(), Wrhs256.begin() + (((p * NB + r) * M1_Kt + kt) * M1_Nt + nt) * ELEMS);
            }
        // M2 fold-LHS:  Wfold128_{p,r}[mt][kt] = limb_p(128^r * W128_block)   layout (((p*4+r)*Mt+mt)*Kt+kt), Mt=Kt=4
        const uint32_t M2_Mt = 4, M2_Kt = 4;
        std::vector<int8_t> Wfold128(NB * NB * M2_Mt * M2_Kt * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r)
            for (uint32_t mt = 0; mt < M2_Mt; ++mt) for (uint32_t kt = 0; kt < M2_Kt; ++kt) {
                std::vector<int8_t> blk(ELEMS);
                for (uint32_t i = 0; i < T; ++i) for (uint32_t j = 0; j < T; ++j)
                    blk[i * T + j] = limb((uint32_t)((POW[r] * W128[(mt * T + i) * N1 + (kt * T + j)]) % q), p);
                auto tt = tilize(blk);
                std::copy(tt.begin(), tt.end(), Wfold128.begin() + (((p * NB + r) * M2_Mt + mt) * M2_Kt + kt) * ELEMS);
            }

        // ---- twiddle Tb[n1][u] = wN^{n1 u}, Montgomery; 32 tiles indexed (n1blk)*8 + (ublk) ----
        std::vector<uint32_t> TbTiles(NTILE * ELEMS);
        for (uint32_t mt = 0; mt < 4; ++mt) for (uint32_t nt = 0; nt < 8; ++nt) {
            std::vector<uint32_t> blk(ELEMS);
            for (uint32_t i = 0; i < T; ++i) for (uint32_t j = 0; j < T; ++j) {
                uint32_t n1 = mt * T + i, u = nt * T + j;
                blk[i * T + j] = ctx.to_montgomery((uint32_t)modpow(wN, (uint64_t)n1 * u % N, q));
            }
            auto tt = tilize(blk); std::copy(tt.begin(), tt.end(), TbTiles.begin() + (mt * 8 + nt) * ELEMS);
        }

        // ---- input A[n1][n2]=x[n1+128 n2]; 32 tiles indexed (n1blk)*8 + (n2blk) ----
        std::mt19937 rng(seed); std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)q - 1);
        std::vector<uint32_t> x(N); for (auto& v : x) v = dist(rng);
        std::vector<uint32_t> ATiles(NTILE * ELEMS);
        for (uint32_t mt = 0; mt < 4; ++mt) for (uint32_t kt = 0; kt < 8; ++kt) {
            std::vector<uint32_t> blk(ELEMS);
            for (uint32_t i = 0; i < T; ++i) for (uint32_t j = 0; j < T; ++j) {
                uint32_t n1 = mt * T + i, n2 = kt * T + j;
                blk[i * T + j] = x[n1 + N1 * n2];
            }
            auto tt = tilize(blk); std::copy(tt.begin(), tt.end(), ATiles.begin() + (mt * 8 + kt) * ELEMS);
        }

        // ---- host golden (2-factor) + direct-DFT spot check ----
        auto mm = [&](const std::vector<uint64_t>& a, uint32_t am, uint32_t ak,
                      const std::vector<uint64_t>& b, uint32_t bn) {
            std::vector<uint64_t> o(am * bn, 0);
            for (uint32_t i = 0; i < am; ++i) for (uint32_t k = 0; k < ak; ++k) { uint64_t aik = a[i * ak + k];
                if (aik) for (uint32_t j = 0; j < bn; ++j) o[i * bn + j] = (o[i * bn + j] + aik * b[k * bn + j]) % q; }
            return o; };
        std::vector<uint64_t> A(N1 * N2);
        for (uint32_t n1 = 0; n1 < N1; ++n1) for (uint32_t n2 = 0; n2 < N2; ++n2) A[n1 * N2 + n2] = x[n1 + N1 * n2];
        auto Sg = mm(A, N1, N2, W256, N2);                                   // S = A @ W256  (128x256)
        for (uint32_t n1 = 0; n1 < N1; ++n1) for (uint32_t u = 0; u < N2; ++u)
            Sg[n1 * N2 + u] = Sg[n1 * N2 + u] * modpow(wN, (uint64_t)n1 * u % N, q) % q;  // twiddle
        auto Rg = mm(W128, N1, N1, Sg, N2);                                  // R = W128 @ S'  (128x256)
        std::vector<uint32_t> Xref(N);
        for (uint32_t v = 0; v < N1; ++v) for (uint32_t u = 0; u < N2; ++u) Xref[u + N2 * v] = (uint32_t)Rg[v * N2 + u];
        { std::mt19937 r2(123); std::uniform_int_distribution<uint32_t> di(0, N - 1); uint32_t bad = 0;
          for (int it = 0; it < 24; ++it) { uint32_t k = di(r2); unsigned __int128 acc = 0;
              for (uint32_t n = 0; n < N; ++n) acc += (unsigned __int128)x[n] * modpow(wN, (uint64_t)n * k % N, q);
              if ((uint32_t)(acc % q) != Xref[k]) ++bad; }
          if (bad) { fmt::print("GOLDEN SELF-CHECK FAILED ({} bad)\n", bad); return 3; } }

        // ---- device ----
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        auto mkbuf = [&](uint32_t ntiles, uint32_t page) {
            distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = ntiles * page}, d, mesh.get()); };

        auto b_A = mkbuf(NTILE, I32), b_Tb = mkbuf(NTILE, I32);
        auto b_Wrhs256 = mkbuf(NB * NB * M1_Kt * M1_Nt, I8), b_Wfold128 = mkbuf(NB * NB * M2_Mt * M2_Kt, I8);
        auto b_Alimbs = mkbuf(NTILE * NB, I8), b_Splanes = mkbuf(NTILE * NB, I32), b_S = mkbuf(NTILE, I32);
        auto b_Sp = mkbuf(NTILE, I32), b_Splimbs = mkbuf(NTILE * NB, I8), b_Rplanes = mkbuf(NTILE * NB, I32), b_R = mkbuf(NTILE, I32);

        auto crange = [](uint32_t C) { uint32_t w = C < 8 ? C : 8, h = (C + 7) / 8; return CoreRange(CoreCoord{0, 0}, CoreCoord{w - 1, h - 1}); };
        auto cat = [](uint32_t c) { return CoreCoord{c % 8, c / 8}; };
        auto mkcbr = [&](Program& p, CoreRange cr, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
            CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, cr, c); };

        auto build_decompose = [&](auto in, auto out, uint32_t C) {
            Program p = CreateProgram(); CoreRange cr = crange(C); const uint32_t npc = NTILE / C;
            mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::Int8, 4, I8);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            std::vector<uint32_t> wc; TensorAccessorArgs(*out).append_to(wc);
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i8_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = wc});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/decompose_limbs.cpp", cr,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true});
            for (uint32_t c = 0; c < C; ++c) { auto co = cat(c);
                SetRuntimeArgs(p, rd, co, {(uint32_t)in->address(), npc, c, C, 1u});
                SetRuntimeArgs(p, wr, co, {(uint32_t)out->address(), npc, c, C, 4u});
                SetRuntimeArgs(p, cp, co, {npc}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        auto build_matmul = [&](auto wconst, auto dyn, auto out, uint32_t C, uint32_t mode, uint32_t Mt, uint32_t Kt, uint32_t Nt) {
            Program p = CreateProgram(); CoreRange cr = crange(C); const uint32_t npc = NTILE / C;
            mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::Int8, 2, I8); mkcbr(p, cr, CBIndex::c_1, tt::DataFormat::Int8, 2, I8);
            mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::Int32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*wconst).append_to(rc); TensorAccessorArgs(*dyn).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul_mtk.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            std::vector<uint32_t> wc; TensorAccessorArgs(*out).append_to(wc);
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_matmul_mt.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = wc});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/matmul_compute_mtk.cpp", cr,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
            for (uint32_t c = 0; c < C; ++c) { auto co = cat(c);
                SetRuntimeArgs(p, rd, co, {(uint32_t)wconst->address(), (uint32_t)dyn->address(), mode, Mt, Kt, Nt, npc, c, C});
                SetRuntimeArgs(p, wr, co, {(uint32_t)out->address(), npc, c, C});
                SetRuntimeArgs(p, cp, co, {npc, NB * Kt}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        auto build_recombine = [&](auto in, auto out, uint32_t C) {
            Program p = CreateProgram(); CoreRange cr = crange(C); const uint32_t npc = NTILE / C;
            mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::UInt32, 8, I32); mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/recombine_bigK.cpp", cr,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true});
            for (uint32_t c = 0; c < C; ++c) { auto co = cat(c);
                SetRuntimeArgs(p, rd, co, {(uint32_t)in->address(), npc, c, C, 4u});
                SetRuntimeArgs(p, wr, co, {(uint32_t)out->address(), npc, c, C, 1u});
                SetRuntimeArgs(p, cp, co, {npc, (uint32_t)q, QINV, QLO, QHI, K14}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        auto build_modmul = [&](auto a, auto b, auto out, uint32_t C) {
            Program p = CreateProgram(); CoreRange cr = crange(C); const uint32_t npc = NTILE / C;
            mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcbr(p, cr, CBIndex::c_1, tt::DataFormat::UInt32, 2, I32);
            mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_hybrid.cpp", cr,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true});
            for (uint32_t c = 0; c < C; ++c) { auto co = cat(c);
                SetRuntimeArgs(p, rd, co, {(uint32_t)a->address(), (uint32_t)b->address(), npc, c, C, 1u});
                SetRuntimeArgs(p, wr, co, {(uint32_t)out->address(), npc, c, C, 1u});
                SetRuntimeArgs(p, cp, co, {npc, (uint32_t)q, QINV, QLO, QHI}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        distributed::EnqueueWriteMeshBuffer(cq, b_A, ATiles, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Wrhs256, Wrhs256, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Wfold128, Wfold128, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Tb, TbTiles, false);

        fmt::print("on-device NTT-32768 (128,256) 2-factor  q={}  seed={}  core-count sweep\n", q, seed);
        fmt::print("  cores  waves   mism      us   speedup   eff\n");
        double t_c1 = 0.0;
        for (uint32_t C : {1u, 2u, 4u, 8u, 16u, 32u}) {
            auto S0 = build_decompose(b_A, b_Alimbs, C);
            auto S1 = build_matmul(b_Wrhs256, b_Alimbs, b_Splanes, C, 1u, 4u, M1_Kt, M1_Nt);   // M1: A@W256, fold-RHS
            auto S2 = build_recombine(b_Splanes, b_S, C);
            auto S3 = build_modmul(b_S, b_Tb, b_Sp, C);
            auto S4 = build_decompose(b_Sp, b_Splimbs, C);
            auto S5 = build_matmul(b_Wfold128, b_Splimbs, b_Rplanes, C, 0u, M2_Mt, M2_Kt, 8u); // M2: W128@S', fold-LHS
            auto S6 = build_recombine(b_Rplanes, b_R, C);
            std::vector<distributed::MeshWorkload*> wls = {&S0, &S1, &S2, &S3, &S4, &S5, &S6};

            for (auto* wl : wls) distributed::EnqueueMeshWorkload(cq, *wl, false);
            std::vector<uint32_t> Rraw; distributed::EnqueueReadMeshBuffer(cq, Rraw, b_R, true);
            std::vector<uint32_t> out(N);
            for (uint32_t mt = 0; mt < 4; ++mt) for (uint32_t nt = 0; nt < 8; ++nt) {
                std::vector<uint32_t> tile(Rraw.begin() + (mt * 8 + nt) * ELEMS, Rraw.begin() + (mt * 8 + nt + 1) * ELEMS);
                auto blk = untilize(tile);
                for (uint32_t i = 0; i < T; ++i) for (uint32_t j = 0; j < T; ++j) {
                    uint32_t v = mt * T + i, u = nt * T + j; out[u + N2 * v] = blk[i * T + j]; } }
            uint32_t mism = 0; for (uint32_t k = 0; k < N; ++k) if (out[k] != Xref[k]) ++mism;
            pass &= (mism == 0);

            distributed::Finish(cq);
            const int iters = 20;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it) for (auto* wl : wls) distributed::EnqueueMeshWorkload(cq, *wl, false);
            distributed::Finish(cq);
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e3 / iters;
            if (C == 1) t_c1 = us;
            fmt::print("  {:5d}  {:5d}  {:5d}  {:7.1f}   {:5.2f}x  {:4.0f}%\n",
                       C, NTILE / C, mism, us, t_c1 / us, 100.0 * (t_c1 / us) / C);

            if (C == 32) {
                const char* nm[] = {"dec A", "M1 mm K=8", "rec1 bigK", "twiddle", "dec S'", "M2 mm K=4", "rec2 bigK"};
                for (size_t s = 0; s < wls.size(); ++s) {
                    distributed::EnqueueMeshWorkload(cq, *wls[s], false); distributed::Finish(cq);
                    auto s0 = std::chrono::high_resolution_clock::now();
                    for (int it = 0; it < iters; ++it) distributed::EnqueueMeshWorkload(cq, *wls[s], false);
                    distributed::Finish(cq);
                    auto s1 = std::chrono::high_resolution_clock::now();
                    fmt::print("      [C32] {:10s}: {:6.1f} us\n", nm[s],
                               std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count() / 1e3 / iters);
                }
            }
        }
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
