// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FULLY ON-DEVICE size-32768 NTT, decomposed (32,32,32). See
// docs/matmul/history/NTT_32768_3FACTOR_PLAN.md. The 32768 values are a 32x32x32 cube =
// 32 tiles of 32x32; global index n = n1 + 32*n2 + 1024*n3, and tile n1 holds
// Xin[n1][n3][n2] = x[n1 + 32*n2 + 1024*n3].
//
// Pipeline (one all-to-all; passes 1+2 folded into an on-tile NTT-1024):
//   Block 1  (per tile = the verified NTT-1024):
//     A1 decompose(Xin) -> A2 matmul(W@, foldLHS) -> A3 recombine = G
//     B  twiddle a (w1024) -> C1 decompose -> C2 matmul(@W, foldRHS) -> C3 recombine = H
//   TB twiddle b (wN):  Cb = H (.) Tb
//   A2A all-to-all corner turn (n1<->s):  D[s][n1][t] = Cb[n1][s][t]
//   Block 2 (Pass 3, contract n1):
//     D1 decompose(D) -> D2 matmul(W@, foldLHS) -> D3 recombine = E
//   out[s + 32*t + 1024*v] = E[s][v][t]   (host reindex of the single read)
//
// Core-count sweep: each stage runs on C cores, core c handling tiles {c,c+C,...}
// (NTILE/C each), for C in {1,2,4,8,16,32}. The all-to-all stays single-core (a
// global reshuffle). No host<->device traffic between stages. Prints a
// speedup/efficiency table vs the C=1 baseline.

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
constexpr uint32_t T = 32, NB = 4, ELEMS = 1024, NTILE = 32;
constexpr uint32_t I8 = ELEMS * 1, I32 = ELEMS * 4;
constexpr uint32_t N = 32768;
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

// 32x32 matmul mod q (row-major a,b -> row-major out)
std::vector<uint64_t> mm(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b, uint64_t q) {
    std::vector<uint64_t> o(ELEMS, 0);
    for (uint32_t i = 0; i < T; ++i) for (uint32_t k = 0; k < T; ++k) { uint64_t aik = a[i * T + k];
        if (!aik) continue; for (uint32_t j = 0; j < T; ++j) o[i * T + j] = (o[i * T + j] + aik * b[k * T + j]) % q; }
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

        // ---- roots ----
        uint64_t g = prim_root(q);
        uint64_t wN = modpow(g, (q - 1) / N, q);          // ω_N
        uint64_t w1024 = modpow(wN, N / 1024, q);          // ω_1024 = wN^32
        uint64_t w32 = modpow(wN, N / T, q);               // ω_32   = wN^1024

        // ---- W (32-pt NTT matrix), folded constants (shared) ----
        std::vector<uint32_t> W(ELEMS);
        for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b) W[a * T + b] = (uint32_t)modpow(w32, (uint64_t)a * b, q);
        auto limb = [](uint32_t v, uint32_t p) -> int8_t { return (int8_t)((v >> (7 * p)) & 127); };
        std::vector<int8_t> WfoldA(NB * NB * ELEMS), Wrhs(NB * NB * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
            std::vector<int8_t> la(ELEMS), lc(ELEMS);
            for (uint32_t i = 0; i < ELEMS; ++i) {
                la[i] = limb((uint32_t)((POW[r] * W[i]) % q), p);   // limb_p(128^r W)  (LHS fold)
                lc[i] = limb((uint32_t)((POW[p] * W[i]) % q), r);   // limb_r(128^p W)  (RHS fold)
            }
            auto ta = tilize(la); std::copy(ta.begin(), ta.end(), WfoldA.begin() + (p * NB + r) * ELEMS);
            auto tc = tilize(lc); std::copy(tc.begin(), tc.end(), Wrhs.begin() + (p * NB + r) * ELEMS);
        }

        // ---- twiddle a (w1024), replicated to 32 tiles; twiddle b (wN), per tile ----
        std::vector<uint32_t> TaTiles(NTILE * ELEMS), TbTiles(NTILE * ELEMS);
        { std::vector<uint32_t> ta(ELEMS);
          for (uint32_t s = 0; s < T; ++s) for (uint32_t n2 = 0; n2 < T; ++n2)
              ta[s * T + n2] = ctx.to_montgomery((uint32_t)modpow(w1024, (uint64_t)s * n2 % 1024, q));
          auto tt = tilize(ta);
          for (uint32_t n1 = 0; n1 < NTILE; ++n1) std::copy(tt.begin(), tt.end(), TaTiles.begin() + n1 * ELEMS);
        }
        for (uint32_t n1 = 0; n1 < NTILE; ++n1) { std::vector<uint32_t> tb(ELEMS);
            for (uint32_t s = 0; s < T; ++s) for (uint32_t t = 0; t < T; ++t)
                tb[s * T + t] = ctx.to_montgomery((uint32_t)modpow(wN, (uint64_t)n1 * ((s + T * t) % N) % N, q));
            auto tt = tilize(tb); std::copy(tt.begin(), tt.end(), TbTiles.begin() + n1 * ELEMS);
        }

        // ---- input cube: tile n1 = Xin[n1][n3][n2] = x[n1 + 32 n2 + 1024 n3] ----
        std::mt19937 rng(seed); std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)q - 1);
        std::vector<uint32_t> x(N); for (auto& v : x) v = dist(rng);
        std::vector<uint32_t> XinTiles(NTILE * ELEMS);
        for (uint32_t n1 = 0; n1 < NTILE; ++n1) { std::vector<uint32_t> slab(ELEMS);
            for (uint32_t n3 = 0; n3 < T; ++n3) for (uint32_t n2 = 0; n2 < T; ++n2)
                slab[n3 * T + n2] = x[n1 + T * n2 + 1024 * n3];
            auto tt = tilize(slab); std::copy(tt.begin(), tt.end(), XinTiles.begin() + n1 * ELEMS);
        }

        // ---- host golden via the 3-pass algorithm (mirrors oracle_32768.py) ----
        auto col_w = [&](uint64_t root) { std::vector<uint64_t> M(ELEMS);
            for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b) M[a * T + b] = modpow(root, (uint64_t)a * b, q); return M; };
        std::vector<uint64_t> Wm = col_w(w32);
        std::vector<std::vector<uint64_t>> Cgold(NTILE), Cbgold(NTILE);
        for (uint32_t n1 = 0; n1 < NTILE; ++n1) {
            std::vector<uint64_t> cube(ELEMS);
            for (uint32_t n3 = 0; n3 < T; ++n3) for (uint32_t n2 = 0; n2 < T; ++n2) cube[n3 * T + n2] = x[n1 + T * n2 + 1024 * n3];
            auto B = mm(Wm, cube, q);                                   // W @ cube  (contract n3)
            for (uint32_t s = 0; s < T; ++s) for (uint32_t n2 = 0; n2 < T; ++n2)
                B[s * T + n2] = B[s * T + n2] * modpow(w1024, (uint64_t)s * n2 % 1024, q) % q;  // twiddle a
            auto C = mm(B, Wm, q);                                      // @ W (contract n2) -> H
            Cgold[n1] = C;
            std::vector<uint64_t> Cb(ELEMS);
            for (uint32_t s = 0; s < T; ++s) for (uint32_t t = 0; t < T; ++t)
                Cb[s * T + t] = C[s * T + t] * modpow(wN, (uint64_t)n1 * ((s + T * t) % N) % N, q) % q;  // twiddle b
            Cbgold[n1] = Cb;
        }
        std::vector<std::vector<uint64_t>> Dgold(NTILE), Egold(NTILE);
        for (uint32_t s = 0; s < T; ++s) { std::vector<uint64_t> D(ELEMS);
            for (uint32_t n1 = 0; n1 < T; ++n1) for (uint32_t t = 0; t < T; ++t) D[n1 * T + t] = Cbgold[n1][s * T + t];  // corner turn
            Dgold[s] = D; Egold[s] = mm(Wm, D, q);                       // W @ D (contract n1)
        }
        std::vector<uint32_t> Xref(N);
        for (uint32_t s = 0; s < T; ++s) for (uint32_t v = 0; v < T; ++v) for (uint32_t t = 0; t < T; ++t)
            Xref[s + T * t + 1024 * v] = (uint32_t)Egold[s][v * T + t];
        // spot-check the algorithm golden against the DIRECT DFT at a few indices
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

        auto b_Xin = mkbuf(NTILE, I32), b_WfoldA = mkbuf(NB * NB, I8), b_Wrhs = mkbuf(NB * NB, I8);
        auto b_Ta = mkbuf(NTILE, I32), b_Tb = mkbuf(NTILE, I32);
        auto b_xlimbs = mkbuf(NTILE * NB, I8), b_CrA = mkbuf(NTILE * NB, I32), b_G = mkbuf(NTILE, I32);
        auto b_Gp = mkbuf(NTILE, I32), b_aC = mkbuf(NTILE * NB, I8), b_CrC = mkbuf(NTILE * NB, I32), b_H = mkbuf(NTILE, I32);
        auto b_Cb = mkbuf(NTILE, I32), b_D = mkbuf(NTILE, I32);
        auto b_Dlimbs = mkbuf(NTILE * NB, I8), b_CrD = mkbuf(NTILE * NB, I32), b_E = mkbuf(NTILE, I32);

        // CoreRange for C cores laid out row-major in a width-8 grid (C in {1,2,4,8,16,32}
        // -> exact rectangles 1x1,2x1,4x1,8x1,8x2,8x4). core index c -> {c%8, c/8}.
        auto crange = [](uint32_t C) {
            uint32_t w = C < 8 ? C : 8, h = (C + 7) / 8;
            return CoreRange(CoreCoord{0, 0}, CoreCoord{w - 1, h - 1}); };
        auto cat = [](uint32_t c) { return CoreCoord{c % 8, c / 8}; };
        auto mkcbr = [&](Program& p, CoreRange cr, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
            CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, cr, c); };

        // each stage: program over C cores; core c handles tiles {c, c+C, ...} = NTILE/C tiles.
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

        auto build_matmul = [&](auto w, auto limbs, auto out, uint32_t C, uint32_t mode) {
            Program p = CreateProgram(); CoreRange cr = crange(C); const uint32_t npc = NTILE / C;
            mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::Int8, 2, I8); mkcbr(p, cr, CBIndex::c_1, tt::DataFormat::Int8, 2, I8);
            mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::Int32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*w).append_to(rc); TensorAccessorArgs(*limbs).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul_mt.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            std::vector<uint32_t> wc; TensorAccessorArgs(*out).append_to(wc);
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_matmul_mt.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = wc});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/matmul_compute_mt.cpp", cr,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
            for (uint32_t c = 0; c < C; ++c) { auto co = cat(c);
                SetRuntimeArgs(p, rd, co, {(uint32_t)w->address(), (uint32_t)limbs->address(), npc, c, C, mode});
                SetRuntimeArgs(p, wr, co, {(uint32_t)out->address(), npc, c, C});
                SetRuntimeArgs(p, cp, co, {npc}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        auto build_recombine = [&](auto in, auto out, uint32_t C) {
            Program p = CreateProgram(); CoreRange cr = crange(C); const uint32_t npc = NTILE / C;
            mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::UInt32, 8, I32); mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/recombine_compute.cpp", cr,
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

        // all-to-all corner turn across C cores: each core builds NTILE/C output tiles
        // {c, c+C, ...}, staging all input tiles in L1 and copying row-chunks.
        auto build_a2a = [&](auto in, auto out, uint32_t C) {
            Program p = CreateProgram(); CoreRange cr = crange(C); const uint32_t npc = NTILE / C;
            mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::UInt32, NTILE + 1, I32);
            auto k = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/all_to_all_corner_mc.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            for (uint32_t c = 0; c < C; ++c)
                SetRuntimeArgs(p, k, cat(c), {(uint32_t)in->address(), (uint32_t)out->address(), NTILE, c, C, npc});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // write input + constants ONCE (read-only across all core-count runs)
        distributed::EnqueueWriteMeshBuffer(cq, b_Xin, XinTiles, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_WfoldA, WfoldA, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Wrhs, Wrhs, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Ta, TaTiles, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Tb, TbTiles, false);

        fmt::print("on-device NTT-32768 (32,32,32)  q={}  seed={}  core-count sweep\n", q, seed);
        fmt::print("  cores  waves   mism      us   speedup   eff\n");
        double t_c1 = 0.0;
        for (uint32_t C : {1u, 2u, 4u, 8u, 16u, 32u}) {
            auto A1 = build_decompose(b_Xin, b_xlimbs, C);
            auto A2 = build_matmul(b_WfoldA, b_xlimbs, b_CrA, C, 0u);
            auto A3 = build_recombine(b_CrA, b_G, C);
            auto B  = build_modmul(b_G, b_Ta, b_Gp, C);
            auto C1 = build_decompose(b_Gp, b_aC, C);
            auto C2 = build_matmul(b_Wrhs, b_aC, b_CrC, C, 1u);
            auto C3 = build_recombine(b_CrC, b_H, C);
            auto TB = build_modmul(b_H, b_Tb, b_Cb, C);
            auto A2A = build_a2a(b_Cb, b_D, C);
            auto D1 = build_decompose(b_D, b_Dlimbs, C);
            auto D2 = build_matmul(b_WfoldA, b_Dlimbs, b_CrD, C, 0u);
            auto D3 = build_recombine(b_CrD, b_E, C);
            std::vector<distributed::MeshWorkload*> wls = {&A1, &A2, &A3, &B, &C1, &C2, &C3, &TB, &A2A, &D1, &D2, &D3};

            for (auto* wl : wls) distributed::EnqueueMeshWorkload(cq, *wl, false);
            std::vector<uint32_t> Eraw; distributed::EnqueueReadMeshBuffer(cq, Eraw, b_E, true);
            std::vector<uint32_t> out(N);
            for (uint32_t s = 0; s < T; ++s) { std::vector<uint32_t> tile(Eraw.begin() + s * ELEMS, Eraw.begin() + (s + 1) * ELEMS);
                auto Es = untilize(tile);
                for (uint32_t v = 0; v < T; ++v) for (uint32_t t = 0; t < T; ++t) out[s + T * t + 1024 * v] = Es[v * T + t]; }
            uint32_t mism = 0; for (uint32_t j = 0; j < N; ++j) if (out[j] != Xref[j]) ++mism;
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

            if (C == 32) {  // attribute the C=32 cost: per-stage steady-state time
                const char* nm[] = {"A1 dec", "A2 mm", "A3 rec", "B  tw", "C1 dec", "C2 mm",
                                    "C3 rec", "TB tw", "A2A!!", "D1 dec", "D2 mm", "D3 rec"};
                for (size_t s = 0; s < wls.size(); ++s) {
                    distributed::EnqueueMeshWorkload(cq, *wls[s], false); distributed::Finish(cq);
                    auto s0 = std::chrono::high_resolution_clock::now();
                    for (int it = 0; it < iters; ++it) distributed::EnqueueMeshWorkload(cq, *wls[s], false);
                    distributed::Finish(cq);
                    auto s1 = std::chrono::high_resolution_clock::now();
                    fmt::print("      [C32] {:8s}: {:6.1f} us\n", nm[s],
                               std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count() / 1e3 / iters);
                }
            }
        }
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
