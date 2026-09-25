// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// MULTI-CORE variant of the fully on-device size-1024 NTT (ntt1024_dev.cpp).
// Everything is identical EXCEPT the int8 plane-matmul is split across 4 Tensix
// cores: the matmul produces 4 int32 result planes C_0..C_3, so we give each
// plane its own core. Core `oi` computes  C_oi = sum_k a_k @ b_k  (NB int8
// matmuls accumulated locally = the per-core "add"), and writes its one tile to
// DRAM. A single core then gathers the 4 planes and recombines them.
//
//   A: G  = W @ Xin    decompose(Xin) -> matmul[4 cores] -> recombine[1 core]
//   B: G' = G (.) T     twiddle modmul (1 core)
//   C: H  = G' @ W      decompose(G') -> matmul[4 cores] -> recombine[1 core]
//   out[j2+32 j1] = H[j2,j1]   (host reindex of the single output read)
//
// The fold is baked into the constant W offline (no runtime fold modmul). The
// matmul orientation differs per stage, selected by `mode` in reader_matmul_mc:
//   mode 0 (A): a = folded W (16 tiles), b = Xin limbs (4)  -> C_p = sum_r a[p*NB+r] @ b[r]
//   mode 1 (C): a = G' limbs (4),  b = folded W (16 tiles)  -> C_r = sum_p a[p] @ b[p*NB+r]

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
        const uint32_t N = 1024;
        mont::MontgomeryContext ctx((uint32_t)q);
        const uint32_t QINV = ctx.q_inv_32(), QLO = q & 0xFFFF, QHI = q >> 16;

        // ---- roots and baked constants ----
        uint64_t g = prim_root(q), w = modpow(g, (q - 1) / N, q), w32 = modpow(g, (q - 1) / T, q);
        std::vector<uint32_t> W(ELEMS), Tw(ELEMS), x(N);
        for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b) W[a * T + b] = (uint32_t)modpow(w32, (uint64_t)a * b, q);
        for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b) Tw[a * T + b] = (uint32_t)modpow(w, (uint64_t)a * b, q);
        std::mt19937 rng(seed); std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)q - 1);
        for (auto& v : x) v = dist(rng);
        std::vector<uint32_t> Xin(ELEMS);
        for (uint32_t n2 = 0; n2 < T; ++n2) for (uint32_t n1 = 0; n1 < T; ++n1) Xin[n2 * T + n1] = x[n1 + T * n2];

        // golden direct NTT
        std::vector<uint32_t> Xref(N);
        for (uint32_t j = 0; j < N; ++j) { unsigned __int128 a = 0;
            for (uint32_t n = 0; n < N; ++n) a += (unsigned __int128)x[n] * modpow(w, (uint64_t)n * j, q);
            Xref[j] = (uint32_t)(a % q); }

        // host-side limb / montgomery prep for the baked tiles (W folded into the CONSTANT)
        auto limb = [](uint32_t v, uint32_t p) -> int8_t { return (int8_t)((v >> (7 * p)) & 127); };
        std::vector<int8_t> WfoldA(NB * NB * ELEMS), Wrhs(NB * NB * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
            std::vector<int8_t> la(ELEMS), lc(ELEMS);
            for (uint32_t i = 0; i < ELEMS; ++i) {
                la[i] = limb((uint32_t)((POW[r] * W[i]) % q), p);    // limb_p(128^r * W)
                lc[i] = limb((uint32_t)((POW[p] * W[i]) % q), r);    // limb_r(128^p * W)
            }
            auto ta = tilize(la); std::copy(ta.begin(), ta.end(), WfoldA.begin() + (p * NB + r) * ELEMS);
            auto tc = tilize(lc); std::copy(tc.begin(), tc.end(), Wrhs.begin() + (p * NB + r) * ELEMS);
        }
        const uint32_t K14 = ctx.to_montgomery(16384u);
        std::vector<uint32_t> XinT = tilize(Xin);
        std::vector<uint32_t> Tmont(ELEMS); { std::vector<uint32_t> tm(ELEMS);
            for (uint32_t i = 0; i < ELEMS; ++i) tm[i] = ctx.to_montgomery(Tw[i]); Tmont = tilize(tm); }

        // ---- device ----
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};

        auto mkbuf = [&](uint32_t ntiles, uint32_t page) {
            distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = ntiles * page}, d, mesh.get());
        };
        auto b_Xin = mkbuf(1, I32), b_xlimbs = mkbuf(NB, I8), b_WfoldA = mkbuf(NB * NB, I8);
        auto b_CrA = mkbuf(NB, I32), b_G = mkbuf(1, I32), b_Tm = mkbuf(1, I32), b_Gp = mkbuf(1, I32);
        auto b_aC = mkbuf(NB, I8), b_Wrhs = mkbuf(NB * NB, I8), b_CrC = mkbuf(NB, I32), b_H = mkbuf(1, I32);

        auto mkcb = [&](Program& p, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
            CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, core, c); };

        auto build_modmul = [&](auto a, auto b, auto out, uint32_t n) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::UInt32, 2, I32);
            mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)a->address(), (uint32_t)b->address(), n, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), n, 0u});
            SetRuntimeArgs(p, cp, core, {n, (uint32_t)q, QINV, QLO, QHI});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        auto build_decompose = [&](auto in, auto out, uint32_t n_in) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_16, tt::DataFormat::Int8, 4 * n_in, I8);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            std::vector<uint32_t> wc; TensorAccessorArgs(*out).append_to(wc);
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i8.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = wc});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/decompose_limbs.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)in->address(), n_in, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 4 * n_in, 0u});
            SetRuntimeArgs(p, cp, core, {n_in});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // MULTI-CORE int8 plane-matmul: 4 cores, core `oi` computes result plane C_oi.
        //   mode 0 -> a=folded W (16), b=limbs (4):  C_p = sum_r a[p*NB+r] @ b[r]
        //   mode 1 -> a=limbs (4),  b=folded W (16):  C_r = sum_p a[p] @ b[p*NB+r]
        auto build_matmul_mc = [&](auto a, auto b, auto out, uint32_t mode) {
            Program p = CreateProgram();
            CoreRange cores(CoreCoord{0, 0}, CoreCoord{NB - 1, 0});   // 4 cores, one per output plane
            auto mkcbr = [&](CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
                CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, cores, c); };
            mkcbr(CBIndex::c_0, tt::DataFormat::Int8, 2, I8); mkcbr(CBIndex::c_1, tt::DataFormat::Int8, 2, I8);
            mkcbr(CBIndex::c_16, tt::DataFormat::Int32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*a).append_to(rc); TensorAccessorArgs(*b).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul_mc.cpp", cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/matmul_plane.cpp", cores,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false});
            for (uint32_t oi = 0; oi < NB; ++oi) {
                CoreCoord c{oi, 0};
                SetRuntimeArgs(p, rd, c, {(uint32_t)a->address(), (uint32_t)b->address(), oi, mode});
                SetRuntimeArgs(p, wr, c, {(uint32_t)out->address(), 1u, oi});   // write plane oi -> tile oi
            }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // single-kernel recombine on ONE core: 4 limb planes C_r -> 1 tile (Σ_r 128^r C_r) mod q
        auto build_recombine = [&](auto in, auto out) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, NB, I32); mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/recombine_compute.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)in->address(), NB, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, 0u});
            SetRuntimeArgs(p, cp, core, {1u, (uint32_t)q, ctx.q_inv_32(), q & 0xFFFF, q >> 16, K14});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // build all stages (fold baked into W; matmul on 4 cores)
        auto A1 = build_decompose(b_Xin, b_xlimbs, 1);              // limb_r(Xin)
        auto A2 = build_matmul_mc(b_WfoldA, b_xlimbs, b_CrA, 0u);   // C_p on 4 cores (fold-LHS)
        auto A3 = build_recombine(b_CrA, b_G);                      // G = recombine(C_p)
        auto B = build_modmul(b_G, b_Tm, b_Gp, 1);                 // G' = G (.) T
        auto C1 = build_decompose(b_Gp, b_aC, 1);                  // limb_p(G')
        auto C2 = build_matmul_mc(b_aC, b_Wrhs, b_CrC, 1u);        // C_r on 4 cores (fold-RHS)
        auto C3 = build_recombine(b_CrC, b_H);                     // H = recombine(C_r)

        // ---- write input + constants ONCE ----
        distributed::EnqueueWriteMeshBuffer(cq, b_Xin, XinT, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_WfoldA, WfoldA, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Wrhs, Wrhs, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Tm, Tmont, false);

        // ---- run the 7 stages, no host reads between ----
        for (auto* wl : {&A1, &A2, &A3, &B, &C1, &C2, &C3})
            distributed::EnqueueMeshWorkload(cq, *wl, false);

        std::vector<uint32_t> Hraw;
        distributed::EnqueueReadMeshBuffer(cq, Hraw, b_H, true);   // single output read
        auto H = untilize(Hraw);

        std::vector<uint32_t> out(N);
        for (uint32_t j1 = 0; j1 < T; ++j1) for (uint32_t j2 = 0; j2 < T; ++j2) out[j2 + T * j1] = H[j2 * T + j1];

        uint32_t mism = 0;
        for (uint32_t j = 0; j < N; ++j) if (out[j] != Xref[j]) { if (mism < 8) fmt::print("  X[{}] got={} exp={}\n", j, out[j], Xref[j]); ++mism; }
        fmt::print("multi-core on-device NTT-1024  q={}  seed={}  mismatches={}/{}\n", q, seed, mism, N);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: multi-core on-device NTT WORKS.\n"); else fmt::print("RESULT: WRONG.\n");

        // ---- timing ----
        {
            std::vector<distributed::MeshWorkload*> wls = {&A1, &A2, &A3, &B, &C1, &C2, &C3};
            for (auto* wl : wls) distributed::EnqueueMeshWorkload(cq, *wl, false);
            distributed::Finish(cq);  // warm
            const int iters = 100;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it) for (auto* wl : wls) distributed::EnqueueMeshWorkload(cq, *wl, false);
            distributed::Finish(cq);
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e3 / iters;
            fmt::print("timing: {:.2f} us / NTT-1024 (device, 7 chained stages, matmul on 4 cores)\n", us);

            const char* nm[] = {"A1 decompose(Xin)", "A2 matmul(4-core)", "A3 recombine",
                                "B  twiddle(modmul)", "C1 decompose", "C2 matmul(4-core)", "C3 recombine"};
            for (size_t s = 0; s < wls.size(); ++s) {
                distributed::EnqueueMeshWorkload(cq, *wls[s], false); distributed::Finish(cq);  // warm
                auto s0 = std::chrono::high_resolution_clock::now();
                for (int it = 0; it < iters; ++it) distributed::EnqueueMeshWorkload(cq, *wls[s], false);
                distributed::Finish(cq);
                auto s1 = std::chrono::high_resolution_clock::now();
                fmt::print("    {:<20}: {:6.2f} us\n", nm[s],
                           std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count() / 1e3 / iters);
            }
        }
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
