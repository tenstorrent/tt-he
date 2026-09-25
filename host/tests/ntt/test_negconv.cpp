// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Validate the external-product core primitive on device: negacyclic convolution via the NTT,
//   c = a ⊛ b (mod x^N+1)  ==  niNTT( nNTT(a) ⊙ nNTT(b) ),  N=1024.
// This is the domain-matching integration that the RGSW MAC (#16 step 6) relies on: chain
// nNTT -> pointwise Montgomery modmul -> niNTT and check vs a schoolbook negacyclic multiply.
// Reuses the validated nntt1024/nintt1024 program chains (fused_matmul_a + modmul + fused_matmul_c)
// as device subroutines; host does only the reshape/reindex the standalone NTTs already do.
//
//   ./build/test_negconv [q] [seed]     (q ≡ 1 mod 2048; default 268369921)

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
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t T = 32, NB = 4, N = 1024, ELEMS = 1024, I8 = ELEMS, I32 = ELEMS * 4;
const uint64_t POW[4] = {1, 128, 16384, 2097152};
using u128 = unsigned __int128;
uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) { uint64_t r = 1; b %= m;
    while (e) { if (e & 1) r = (u128)r * b % m; b = (u128)b * b % m; e >>= 1; } return r; }
uint64_t prim_root(uint64_t q) { uint64_t phi = q - 1, n = phi; std::vector<uint64_t> f;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) { f.push_back(d); while (n % d == 0) n /= d; }
    if (n > 1) f.push_back(n);
    for (uint64_t g = 2; g < q; ++g) { bool ok = true; for (uint64_t p : f) if (modpow(g, phi / p, q) == 1) ok = false;
        if (ok) return g; } throw std::runtime_error("no root"); }
std::vector<uint32_t> tilize(const std::vector<uint32_t>& in) {
    std::vector<uint32_t> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = in[r * T + c];
    return o; }
std::vector<uint32_t> untilize(const std::vector<uint32_t>& in) {
    std::vector<uint32_t> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[r * T + c] = in[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)];
    return o; }
int8_t limb(uint32_t v, uint32_t p) { return (int8_t)((v >> (7 * p)) & 127); }
}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const uint64_t q = (argc > 1) ? std::stoull(argv[1]) : 268369921ULL;
        const uint32_t seed = (argc > 2) ? std::stoul(argv[2]) : 0;
        if ((q - 1) % (2 * N) != 0) { fmt::print(stderr, "q must be ≡ 1 mod {}\n", 2 * N); return 2; }
        mont::MontgomeryContext ctx((uint32_t)q);
        const uint32_t QINV = ctx.q_inv_32(), QLO = q & 0xFFFF, QHI = q >> 16, K14 = ctx.to_montgomery(16384u);

        // ---- bake nNTT and niNTT constants (as in nntt1024_dev / nintt1024_dev) ----
        uint64_t g = prim_root(q);
        // forward
        uint64_t w = modpow(g, (q - 1) / N, q), w32 = modpow(g, (q - 1) / T, q), psi = modpow(g, (q - 1) / (2 * N), q);
        std::vector<uint32_t> Wc(ELEMS), Wa(ELEMS), Tw(ELEMS);
        for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b) {
            uint32_t wv = (uint32_t)modpow(w32, (uint64_t)a * b, q);
            Wc[a * T + b] = wv; Wa[a * T + b] = (uint32_t)((u128)wv * modpow(psi, 32ull * b, q) % q);
        }
        for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b)
            Tw[a * T + b] = (uint32_t)((u128)modpow(w, (uint64_t)a * b, q) * modpow(psi, b, q) % q);
        // inverse
        uint64_t wi = mont::modular_inverse(w, q), w32i = mont::modular_inverse(w32, q);
        uint64_t psii = mont::modular_inverse(psi, q), Ninv = mont::modular_inverse(N, q);
        std::vector<uint32_t> Wci(ELEMS), Wai(ELEMS), Twi(ELEMS);
        for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b) {
            uint32_t wv = (uint32_t)modpow(w32i, (uint64_t)a * b, q);
            Wci[a * T + b] = (uint32_t)((u128)wv * modpow(psii, 32ull * b, q) % q);
            Wai[a * T + b] = (uint32_t)((u128)wv * Ninv % q);
        }
        for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b)
            Twi[a * T + b] = (uint32_t)((u128)modpow(wi, (uint64_t)a * b, q) * modpow(psii, a, q) % q);

        auto foldA = [&](const std::vector<uint32_t>& Wm) {          // limb_p(128^r · Wm) at [p*NB+r]
            std::vector<int8_t> o(NB * NB * ELEMS);
            for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
                std::vector<uint32_t> l(ELEMS);
                for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[r] * Wm[i]) % q), p);
                std::vector<uint32_t> t(ELEMS); { auto tl = tilize(l); for (uint32_t i = 0; i < ELEMS; ++i) t[i] = tl[i]; }
                for (uint32_t i = 0; i < ELEMS; ++i) o[(p * NB + r) * ELEMS + i] = (int8_t)t[i];
            } return o; };
        auto foldC = [&](const std::vector<uint32_t>& Wm) {          // limb_r(128^p · Wm) at [p*NB+r]
            std::vector<int8_t> o(NB * NB * ELEMS);
            for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
                std::vector<uint32_t> l(ELEMS);
                for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[p] * Wm[i]) % q), r);
                auto tl = tilize(l);
                for (uint32_t i = 0; i < ELEMS; ++i) o[(p * NB + r) * ELEMS + i] = (int8_t)tl[i];
            } return o; };
        auto montile = [&](const std::vector<uint32_t>& Wm) {
            std::vector<uint32_t> m(ELEMS); for (uint32_t i = 0; i < ELEMS; ++i) m[i] = ctx.to_montgomery(Wm[i]);
            return tilize(m); };

        std::vector<int8_t> WfoldA = foldA(Wa), Wrhs = foldC(Wc), WfoldAi = foldA(Wai), Wrhsi = foldC(Wci);
        std::vector<uint32_t> Tmont = montile(Tw), Tmonti = montile(Twi);

        // ---- device ----
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        auto mkbuf = [&](uint32_t nt, uint32_t pg) { distributed::DeviceLocalBufferConfig d{.page_size = pg, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * pg}, d, mesh.get()); };
        auto mkcb = [&](Program& p, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t pg) {
            CircularBufferConfig c(depth * pg, {{idx, fmt}}); c.set_page_size(idx, pg); CreateCircularBuffer(p, core, c); };

        // persistent constant buffers
        auto b_WfA = mkbuf(NB * NB, I8), b_Wr = mkbuf(NB * NB, I8), b_Tm = mkbuf(1, I32);
        auto b_WfAi = mkbuf(NB * NB, I8), b_Wri = mkbuf(NB * NB, I8), b_Tmi = mkbuf(1, I32);
        distributed::EnqueueWriteMeshBuffer(cq, b_WfA, WfoldA, false); distributed::EnqueueWriteMeshBuffer(cq, b_Wr, Wrhs, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Tm, Tmont, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_WfAi, WfoldAi, false); distributed::EnqueueWriteMeshBuffer(cq, b_Wri, Wrhsi, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Tmi, Tmonti, false);

        auto fusedA = [&](auto xin, auto w, auto out) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::Int8, 2, I8);
            mkcb(p, CBIndex::c_2, tt::DataFormat::Int8, NB, I8); mkcb(p, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);
            mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*xin).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp", core, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a.cpp", core, ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)xin->address(), (uint32_t)w->address()}); SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, 0u});
            SetRuntimeArgs(p, cp, core, {(uint32_t)q, QINV, QLO, QHI, K14});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false); };
        auto fusedC = [&](auto gp, auto w, auto out) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::Int8, NB * NB, I8);
            mkcb(p, CBIndex::c_2, tt::DataFormat::Int8, NB, I8); mkcb(p, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);
            mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*gp).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp", core, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c.cpp", core, ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)gp->address(), (uint32_t)w->address()}); SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, 0u});
            SetRuntimeArgs(p, cp, core, {(uint32_t)q, QINV, QLO, QHI, K14});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false); };
        auto modmul = [&](auto a, auto b, auto out) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", core, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp", core, ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)a->address(), (uint32_t)b->address(), 1u, 0u}); SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, 0u});
            SetRuntimeArgs(p, cp, core, {1u, (uint32_t)q, QINV, QLO, QHI});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false); };

        auto b_in = mkbuf(1, I32), b_g = mkbuf(1, I32), b_gp = mkbuf(1, I32), b_h = mkbuf(1, I32);
        // nNTT(x_natural) -> X_natural : reshape, chain, reindex
        auto fwd = [&](const std::vector<uint32_t>& x) {
            std::vector<uint32_t> Xin(ELEMS);
            for (uint32_t n2 = 0; n2 < T; ++n2) for (uint32_t n1 = 0; n1 < T; ++n1) Xin[n2 * T + n1] = x[n1 + T * n2];
            distributed::EnqueueWriteMeshBuffer(cq, b_in, tilize(Xin), false);
            fusedA(b_in, b_WfA, b_g); modmul(b_g, b_Tm, b_gp); fusedC(b_gp, b_Wr, b_h);
            std::vector<uint32_t> raw; distributed::EnqueueReadMeshBuffer(cq, raw, b_h, true); auto H = untilize(raw);
            std::vector<uint32_t> X(N); for (uint32_t j1 = 0; j1 < T; ++j1) for (uint32_t j2 = 0; j2 < T; ++j2) X[j2 + T * j1] = H[j2 * T + j1];
            return X; };
        auto inv = [&](const std::vector<uint32_t>& X) {
            std::vector<uint32_t> Xin(ELEMS);
            for (uint32_t n2 = 0; n2 < T; ++n2) for (uint32_t n1 = 0; n1 < T; ++n1) Xin[n2 * T + n1] = X[n1 + T * n2];
            distributed::EnqueueWriteMeshBuffer(cq, b_in, tilize(Xin), false);
            fusedA(b_in, b_WfAi, b_g); modmul(b_g, b_Tmi, b_gp); fusedC(b_gp, b_Wri, b_h);
            std::vector<uint32_t> raw; distributed::EnqueueReadMeshBuffer(cq, raw, b_h, true); auto H = untilize(raw);
            std::vector<uint32_t> x(N); for (uint32_t j1 = 0; j1 < T; ++j1) for (uint32_t j2 = 0; j2 < T; ++j2) x[j2 + T * j1] = H[j2 * T + j1];
            return x; };

        // ---- inputs + golden ----
        std::mt19937 rng(seed); std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)q - 1);
        std::vector<uint32_t> a(N), b(N); for (auto& v : a) v = dist(rng); for (auto& v : b) v = dist(rng);
        std::vector<uint32_t> cref(N, 0);
        for (uint32_t i = 0; i < N; ++i) for (uint32_t j = 0; j < N; ++j) {
            u128 v = (u128)a[i] * b[j] % q; uint32_t k = i + j;
            if (k < N) cref[k] = (uint32_t)((cref[k] + v) % q); else cref[k - N] = (uint32_t)((cref[k - N] + q - v) % q);
        }

        // ---- device: c = niNTT( nNTT(a) ⊙ nNTT(b) ) ----
        auto A = fwd(a), Bf = fwd(b);
        std::vector<uint32_t> Amont(ELEMS), Bt(ELEMS);
        { std::vector<uint32_t> am(N); for (uint32_t i = 0; i < N; ++i) am[i] = ctx.to_montgomery(A[i]); Amont = tilize(am); Bt = tilize(Bf); }
        auto b_A = mkbuf(1, I32), b_B = mkbuf(1, I32), b_P = mkbuf(1, I32);
        distributed::EnqueueWriteMeshBuffer(cq, b_A, Amont, false); distributed::EnqueueWriteMeshBuffer(cq, b_B, Bt, false);
        modmul(b_A, b_B, b_P);                                   // P = A ⊙ B  (A in Montgomery -> plain product)
        std::vector<uint32_t> Praw; distributed::EnqueueReadMeshBuffer(cq, Praw, b_P, true);
        auto Pun = untilize(Praw); std::vector<uint32_t> P(N); for (uint32_t i = 0; i < N; ++i) P[i] = Pun[i];
        auto c = inv(P);

        uint32_t mism = 0;
        for (uint32_t i = 0; i < N; ++i) if (c[i] != cref[i]) { if (mism < 8) fmt::print("  c[{}] got={} exp={}\n", i, c[i], cref[i]); ++mism; }
        fmt::print("device negacyclic conv via NTT  q={}  seed={}  mismatches={}/{}\n", q, seed, mism, N);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: nNTT⊙niNTT = negacyclic conv WORKS.\n"); else fmt::print("RESULT: WRONG.\n");
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
