// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// n=2048 NTT round-trip: builds fwd2048 / inv2048 as composable lambdas (the exact 32x64 four-step
// from nntt2048_dev / nintt2048_dev) and checks inv2048(fwd2048(x)) == x on device.  This packages
// the transforms as the reusable helpers the n=2048 external product needs (ONION_MATCH_PLAN §10b).
//
//   ./build/test_ntt2048_rt [q] [seed]      (q ≡ 1 mod 4096; default 268369921)

#include <cstdint>
#include <cstdio>
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
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t T = 32, NB = 4, N1 = 32, N2 = 64, N = 2048, ELEMS = 1024, I8 = ELEMS, I32 = ELEMS * 4;
const uint64_t POW[4] = {1, 128, 16384, 2097152};
using u128 = unsigned __int128;
uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) { uint64_t r = 1; b %= m;
    while (e) { if (e & 1) r = (u128)r * b % m; b = (u128)b * b % m; e >>= 1; } return r; }
uint64_t prim_root(uint64_t q) { uint64_t phi = q - 1, n = phi; std::vector<uint64_t> f;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) { f.push_back(d); while (n % d == 0) n /= d; }
    if (n > 1) f.push_back(n);
    for (uint64_t g = 2; g < q; ++g) { bool ok = true; for (uint64_t p : f) if (modpow(g, phi / p, q) == 1) ok = false;
        if (ok) return g; } throw std::runtime_error("no root"); }
int8_t limb(uint32_t v, uint32_t p) { return (int8_t)((v >> (7 * p)) & 127); }
template <typename E> std::vector<E> tilize32(const std::vector<E>& in) {
    std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = in[r * T + c];
    return o; }
template <typename E> std::vector<E> untilize32(const std::vector<E>& in) {
    std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[r * T + c] = in[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)];
    return o; }
}  // namespace

int main(int argc, char** argv) {
    const uint64_t q = (argc > 1) ? std::stoull(argv[1]) : 268369921ULL;
    const uint32_t seed = (argc > 2) ? std::stoul(argv[2]) : 0;
    bool pass = true;
    try {
        mont::MontgomeryContext ctx((uint32_t)q);
        const uint32_t QINV = ctx.q_inv_32(), QLO = q & 0xFFFF, QHI = q >> 16, K14 = ctx.to_montgomery(16384u);
        uint64_t g = prim_root(q);
        // forward roots
        uint64_t wF = modpow(g, (q - 1) / N, q), psiF = modpow(g, (q - 1) / (2 * N), q);
        uint64_t w64F = modpow(wF, N1, q), w32F = modpow(wF, N2, q);
        // inverse roots
        uint64_t wI = mont::modular_inverse(wF, q), psiI = mont::modular_inverse(psiF, q), Ninv = mont::modular_inverse(N, q);
        uint64_t w64I = mont::modular_inverse(modpow(wF, N1, q), q), w32I = mont::modular_inverse(modpow(wF, N2, q), q);

        // ---- FWD matrices/folds ----
        std::vector<uint32_t> WA(N2 * N2), TwF(N2 * N1), WCt(N1 * N1);
        for (uint32_t k2 = 0; k2 < N2; ++k2) for (uint32_t n2 = 0; n2 < N2; ++n2)
            WA[k2 * N2 + n2] = (uint32_t)((u128)modpow(w64F, (uint64_t)k2 * n2, q) * modpow(psiF, 32ull * n2, q) % q);
        for (uint32_t k2 = 0; k2 < N2; ++k2) for (uint32_t n1 = 0; n1 < N1; ++n1)
            TwF[k2 * N1 + n1] = (uint32_t)((u128)modpow(psiF, n1, q) * modpow(wF, (uint64_t)n1 * k2, q) % q);
        for (uint32_t n1 = 0; n1 < N1; ++n1) for (uint32_t k1 = 0; k1 < N1; ++k1)
            WCt[n1 * N1 + k1] = (uint32_t)modpow(w32F, (uint64_t)k1 * n1, q);
        std::vector<std::vector<int8_t>> WAfold(4, std::vector<int8_t>(NB * NB * ELEMS));
        for (uint32_t bk2 = 0; bk2 < 2; ++bk2) for (uint32_t bn2 = 0; bn2 < 2; ++bn2) {
            std::vector<uint32_t> blk(ELEMS);
            for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b) blk[a * T + b] = WA[(bk2 * T + a) * N2 + (bn2 * T + b)];
            auto& dst = WAfold[bk2 * 2 + bn2];
            for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
                std::vector<int8_t> l(ELEMS); for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[r] * blk[i]) % q), p);
                auto tl = tilize32(l); std::copy(tl.begin(), tl.end(), dst.begin() + (p * NB + r) * ELEMS); } }
        std::vector<int8_t> WCtfold(NB * NB * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
            std::vector<int8_t> l(ELEMS); for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[p] * WCt[i]) % q), r);
            auto tl = tilize32(l); std::copy(tl.begin(), tl.end(), WCtfold.begin() + (p * NB + r) * ELEMS); }

        // ---- INV matrices/folds ----
        std::vector<uint32_t> Winv32(N1 * N1), TwI(N1 * N2), Winv64(N2 * N2);
        for (uint32_t n1 = 0; n1 < N1; ++n1) for (uint32_t k1 = 0; k1 < N1; ++k1)
            Winv32[n1 * N1 + k1] = (uint32_t)((u128)modpow(w32I, (uint64_t)n1 * k1, q) * Ninv % q);
        for (uint32_t n1 = 0; n1 < N1; ++n1) for (uint32_t k2 = 0; k2 < N2; ++k2)
            TwI[n1 * N2 + k2] = (uint32_t)((u128)modpow(wI, (uint64_t)n1 * k2, q) * modpow(psiI, n1, q) % q);
        for (uint32_t k2 = 0; k2 < N2; ++k2) for (uint32_t n2 = 0; n2 < N2; ++n2)
            Winv64[k2 * N2 + n2] = (uint32_t)((u128)modpow(w64I, (uint64_t)k2 * n2, q) * modpow(psiI, 32ull * n2, q) % q);
        std::vector<int8_t> Winv32fold(NB * NB * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
            std::vector<int8_t> l(ELEMS); for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[r] * Winv32[i]) % q), p);
            auto tl = tilize32(l); std::copy(tl.begin(), tl.end(), Winv32fold.begin() + (p * NB + r) * ELEMS); }
        std::vector<std::vector<int8_t>> Winv64fold(4, std::vector<int8_t>(NB * NB * ELEMS));
        for (uint32_t bk2 = 0; bk2 < 2; ++bk2) for (uint32_t bn2 = 0; bn2 < 2; ++bn2) {
            std::vector<uint32_t> blk(ELEMS);
            for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b) blk[a * T + b] = Winv64[(bk2 * T + a) * N2 + (bn2 * T + b)];
            auto& dst = Winv64fold[bk2 * 2 + bn2];
            for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
                std::vector<int8_t> l(ELEMS); for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[p] * blk[i]) % q), r);
                auto tl = tilize32(l); std::copy(tl.begin(), tl.end(), dst.begin() + (p * NB + r) * ELEMS); } }

        // ---- device ----
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        auto mkbuf = [&](uint32_t nt, uint32_t pg) { distributed::DeviceLocalBufferConfig d{.page_size = pg, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * pg}, d, mesh.get()); };
        auto mkcb = [&](Program& p, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t pg) {
            CircularBufferConfig c(depth * pg, {{idx, fmt}}); c.set_page_size(idx, pg); CreateCircularBuffer(p, core, c); };
        auto wrb = [&](auto buf, const auto& v) { distributed::EnqueueWriteMeshBuffer(cq, buf, v, false); };

        // matmul/modadd/modmul program builders (out_start lets fused write partial tile 0 or 1)
        auto fusedA = [&](auto xin, auto w, auto out, uint32_t out_start) { Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::Int8, 2, I8);
            mkcb(p, CBIndex::c_2, tt::DataFormat::Int8, NB, I8); mkcb(p, CBIndex::c_3, tt::DataFormat::Int32, NB, I32); mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*xin).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a.cpp", core, ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)xin->address(), (uint32_t)w->address()}); SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, out_start});
            SetRuntimeArgs(p, cp, core, {(uint32_t)q, QINV, QLO, QHI, K14});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false); };
        auto fusedC = [&](auto gp, auto w, auto out, uint32_t out_start) { Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::Int8, NB * NB, I8);
            mkcb(p, CBIndex::c_2, tt::DataFormat::Int8, NB, I8); mkcb(p, CBIndex::c_3, tt::DataFormat::Int32, NB, I32); mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*gp).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c.cpp", core, ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)gp->address(), (uint32_t)w->address()}); SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, out_start});
            SetRuntimeArgs(p, cp, core, {(uint32_t)q, QINV, QLO, QHI, K14});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false); };
        auto modadd2 = [&](auto in, auto out) { Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp", core, ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)in->address(), 2u, 0u}); SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, 0u}); SetRuntimeArgs(p, cp, core, {2u, (uint32_t)q});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false); };
        auto modmul = [&](auto a, auto b, auto out) { Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp", core, ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)a->address(), (uint32_t)b->address(), 1u, 0u}); SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, 0u}); SetRuntimeArgs(p, cp, core, {1u, (uint32_t)q, QINV, QLO, QHI});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false); };

        // fold buffers (write once)
        std::vector<std::shared_ptr<distributed::MeshBuffer>> bWA(4), bWinv64(4);
        for (uint32_t i = 0; i < 4; ++i) { bWA[i] = mkbuf(NB*NB, I8); wrb(bWA[i], WAfold[i]); bWinv64[i] = mkbuf(NB*NB, I8); wrb(bWinv64[i], Winv64fold[i]); }
        auto bWCt = mkbuf(NB*NB, I8); wrb(bWCt, WCtfold);
        auto bWinv32 = mkbuf(NB*NB, I8); wrb(bWinv32, Winv32fold);
        std::vector<std::shared_ptr<distributed::MeshBuffer>> bTwF(2), bTwI(2);
        for (uint32_t bk2 = 0; bk2 < 2; ++bk2) {
            std::vector<uint32_t> bf(ELEMS), bi(ELEMS);
            for (uint32_t a = 0; a < T; ++a) for (uint32_t n1 = 0; n1 < N1; ++n1) bf[a*T+n1] = ctx.to_montgomery(TwF[(bk2*T+a)*N1+n1]);   // TwF[k2][n1]
            for (uint32_t n1 = 0; n1 < N1; ++n1) for (uint32_t k2loc = 0; k2loc < T; ++k2loc) bi[n1*T+k2loc] = ctx.to_montgomery(TwI[n1*N2+(bk2*T+k2loc)]);
            bTwF[bk2] = mkbuf(1, I32); wrb(bTwF[bk2], tilize32(bf)); bTwI[bk2] = mkbuf(1, I32); wrb(bTwI[bk2], tilize32(bi)); }

        // working buffers
        std::vector<std::shared_ptr<distributed::MeshBuffer>> bIn(2), bAcc(2), bSum(2), bB(2), bOut(2);
        for (uint32_t i = 0; i < 2; ++i) { bIn[i]=mkbuf(1,I32); bAcc[i]=mkbuf(2,I32); bSum[i]=mkbuf(1,I32); bB[i]=mkbuf(1,I32); bOut[i]=mkbuf(1,I32); }

        // fwd2048: x[N] natural -> X[N] natural (freq)
        auto fwd2048 = [&](const std::vector<uint32_t>& x) {
            for (uint32_t bn2 = 0; bn2 < 2; ++bn2) { std::vector<uint32_t> blk(ELEMS);
                for (uint32_t b = 0; b < T; ++b) for (uint32_t n1 = 0; n1 < N1; ++n1) blk[b*T+n1] = x[n1 + 32*(bn2*T+b)];
                wrb(bIn[bn2], tilize32(blk)); }
            for (uint32_t bk2 = 0; bk2 < 2; ++bk2) {
                fusedA(bIn[0], bWA[bk2*2+0], bAcc[bk2], 0); fusedA(bIn[1], bWA[bk2*2+1], bAcc[bk2], 1);
                modadd2(bAcc[bk2], bSum[bk2]); modmul(bSum[bk2], bTwF[bk2], bB[bk2]); fusedC(bB[bk2], bWCt, bOut[bk2], 0); }
            std::vector<uint32_t> X(N);
            for (uint32_t bk2 = 0; bk2 < 2; ++bk2) { std::vector<uint32_t> raw; distributed::EnqueueReadMeshBuffer(cq, raw, bOut[bk2], true);
                auto blk = untilize32(raw);
                for (uint32_t k2loc = 0; k2loc < T; ++k2loc) for (uint32_t k1 = 0; k1 < N1; ++k1) X[(bk2*T+k2loc) + N2*k1] = blk[k2loc*T+k1]; }
            return X; };

        // inv2048: X[N] natural (freq) -> x[N] natural
        auto inv2048 = [&](const std::vector<uint32_t>& X) {
            for (uint32_t bk2 = 0; bk2 < 2; ++bk2) { std::vector<uint32_t> blk(ELEMS);
                for (uint32_t k1 = 0; k1 < N1; ++k1) for (uint32_t k2loc = 0; k2loc < T; ++k2loc) blk[k1*T+k2loc] = X[(bk2*T+k2loc) + N2*k1];
                wrb(bIn[bk2], tilize32(blk)); }
            for (uint32_t bk2 = 0; bk2 < 2; ++bk2) { fusedA(bIn[bk2], bWinv32, bSum[bk2], 0); modmul(bSum[bk2], bTwI[bk2], bB[bk2]); }  // stage1: P then Q(=bB)
            for (uint32_t bn2 = 0; bn2 < 2; ++bn2) {
                fusedC(bB[0], bWinv64[0*2+bn2], bAcc[bn2], 0); fusedC(bB[1], bWinv64[1*2+bn2], bAcc[bn2], 1); modadd2(bAcc[bn2], bOut[bn2]); }
            std::vector<uint32_t> x(N);
            for (uint32_t bn2 = 0; bn2 < 2; ++bn2) { std::vector<uint32_t> raw; distributed::EnqueueReadMeshBuffer(cq, raw, bOut[bn2], true);
                auto blk = untilize32(raw);
                for (uint32_t n1 = 0; n1 < N1; ++n1) for (uint32_t n2loc = 0; n2loc < T; ++n2loc) x[n1 + 32*(bn2*T+n2loc)] = blk[n1*T+n2loc]; }
            return x; };

        // ---- round-trip check ----
        std::mt19937 rng(seed); std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)q - 1);
        std::vector<uint32_t> x(N); for (auto& v : x) v = dist(rng);
        auto X = fwd2048(x);
        auto xrt = inv2048(X);
        uint32_t mism = 0;
        for (uint32_t n = 0; n < N; ++n) if (xrt[n] != x[n]) { if (mism < 8) fmt::print("  x[{}] got={} exp={}\n", n, xrt[n], x[n]); ++mism; }
        fmt::print("ntt2048 round-trip inv(fwd(x))==x  q={}  mismatches={}/{}\n", q, mism, N);
        pass = (mism == 0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
