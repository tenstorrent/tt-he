// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FULLY ON-DEVICE negacyclic INVERSE NTT-8192 (N = 32x256 four-step) on a single Tensix core.
// Mirror of nntt8192_dev with inverse roots + N⁻¹ folded into the stage-1 matrix; generalizes
// nintt2048_dev (32x64) to N2=256 → NBLK=8. Reuses fused_matmul_a/c, modadd_reduce, modmul_compute.
//
// Layout (n = n1 + 32*n2, k = k2 + 256*k1):
//   X reshape: XT[bk2] tile [k1][k2loc] = X[(bk2*32 + k2loc) + 256*k1]     (NBLK=8 tiles)
//   stage 1 (32-DFT + N⁻¹): P[bk2] = fused_a(X[bk2], Winv32)              Winv32 32x32 LHS
//   twiddle:                Q[bk2] = P[bk2] ⊙ Tw[bk2]                     Tw 32x256 = 8 tiles (Mont)
//   stage 2 (256-DFT):      R[n2] = Σ_k2 Q[k2]·Winv256[k2][n2]           Winv256 256x256 = 8x8 blocks
//       R[bn2] = Σ_bk2 fused_c(Q[bk2], Winv256fold[bk2][bn2])  (8 partials) -> modadd(8)
//   reindex:  x[n1 + 32*n2] = R[bn2][n1][n2loc]  (n2 = bn2*32 + n2loc)
//
//   ./build/nintt8192_dev [q] [seed]   (q ≡ 1 mod 2N=16384)

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
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t T = 32, NB = 4, N1 = 32, N2 = 256, N = 8192, ELEMS = 1024;
constexpr uint32_t NBLK = N2 / T;           // 8
constexpr uint32_t I8 = ELEMS, I32 = ELEMS * 4;
const uint64_t POW[4] = {1, 128, 16384, 2097152};
using u128 = unsigned __int128;

uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) { uint64_t r = 1; b %= m;
    while (e) { if (e & 1) r = (u128)r * b % m; b = (u128)b * b % m; e >>= 1; } return r; }
bool is_prime(uint64_t n) { if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) return false; return true; }
uint64_t prim_root(uint64_t q) { uint64_t phi = q - 1, n = phi; std::vector<uint64_t> f;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) { f.push_back(d); while (n % d == 0) n /= d; }
    if (n > 1) f.push_back(n);
    for (uint64_t g = 2; g < q; ++g) { bool ok = true; for (uint64_t p : f) if (modpow(g, phi / p, q) == 1) ok = false;
        if (ok) return g; } throw std::runtime_error("no root"); }
uint64_t first_ntt_prime() {
    for (uint64_t k = ((1ull << 28) - 1) / (2 * N); k > 0; --k) {
        uint64_t p = k * 2 * N + 1; if (p < (1ull << 28) && is_prime(p)) return p; }
    throw std::runtime_error("no prime"); }

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
int8_t limb(uint32_t v, uint32_t p) { return (int8_t)((v >> (7 * p)) & 127); }
}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const uint64_t q = (argc > 1) ? std::stoull(argv[1]) : first_ntt_prime();
        const uint32_t seed = (argc > 2) ? std::stoul(argv[2]) : 0;
        if ((q - 1) % (2 * N) != 0) { fmt::print(stderr, "q must be ≡ 1 mod {}\n", 2 * N); return 2; }
        mont::MontgomeryContext ctx((uint32_t)q);
        const uint32_t QINV = ctx.q_inv_32(), QLO = q & 0xFFFF, QHI = q >> 16;
        const uint32_t K14 = ctx.to_montgomery(16384u);

        // inverse roots
        uint64_t g = prim_root(q);
        uint64_t wf = modpow(g, (q - 1) / N, q);
        uint64_t w = mont::modular_inverse(wf, q);                     // ω⁻¹
        uint64_t w256 = mont::modular_inverse(modpow(wf, N1, q), q);   // ω_256⁻¹
        uint64_t w32 = mont::modular_inverse(modpow(wf, N2, q), q);    // ω_32⁻¹
        uint64_t psi = modpow(g, (q - 1) / (2 * N), q);
        uint64_t psinv = mont::modular_inverse(psi, q);                // ψ⁻¹
        uint64_t Ninv = mont::modular_inverse(N, q);

        // constant matrices
        std::vector<uint32_t> Winv32(N1 * N1);                         // Winv32[n1][k1] = ω_32⁻ⁿ¹ᵏ¹·N⁻¹
        for (uint32_t n1 = 0; n1 < N1; ++n1) for (uint32_t k1 = 0; k1 < N1; ++k1)
            Winv32[n1 * N1 + k1] = (uint32_t)((u128)modpow(w32, (uint64_t)n1 * k1, q) * Ninv % q);
        std::vector<uint32_t> Tw(N1 * N2);                             // Tw[n1][k2] = ω⁻ⁿ¹ᵏ²·ψ⁻ⁿ¹
        for (uint32_t n1 = 0; n1 < N1; ++n1) for (uint32_t k2 = 0; k2 < N2; ++k2)
            Tw[n1 * N2 + k2] = (uint32_t)((u128)modpow(w, (uint64_t)n1 * k2, q) * modpow(psinv, n1, q) % q);
        std::vector<uint32_t> Winv256(N2 * N2);                        // Winv256[k2][n2] = ω_256⁻ᵏ²ⁿ²·ψ⁻³²ⁿ²
        for (uint32_t k2 = 0; k2 < N2; ++k2) for (uint32_t n2 = 0; n2 < N2; ++n2)
            Winv256[k2 * N2 + n2] = (uint32_t)((u128)modpow(w256, (uint64_t)k2 * n2, q) * modpow(psinv, 32ull * n2, q) % q);

        // stage-1 fold (LHS): Winv32fold[p*NB+r] = tilize(limb_p(128^r · Winv32))
        std::vector<int8_t> Winv32fold(NB * NB * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
            std::vector<int8_t> l(ELEMS);
            for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[r] * Winv32[i]) % q), p);
            auto tl = tilize32(l); std::copy(tl.begin(), tl.end(), Winv32fold.begin() + (p * NB + r) * ELEMS);
        }
        // stage-2 fold (RHS): Winv256fold[bk2*NBLK+bn2][p*NB+r] = tilize(limb_r(128^p · block))
        std::vector<std::vector<int8_t>> Winv256fold(NBLK * NBLK, std::vector<int8_t>(NB * NB * ELEMS));
        for (uint32_t bk2 = 0; bk2 < NBLK; ++bk2) for (uint32_t bn2 = 0; bn2 < NBLK; ++bn2) {
            std::vector<uint32_t> blk(ELEMS);
            for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b)
                blk[a * T + b] = Winv256[(bk2 * T + a) * N2 + (bn2 * T + b)];
            auto& dst = Winv256fold[bk2 * NBLK + bn2];
            for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
                std::vector<int8_t> l(ELEMS);
                for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[p] * blk[i]) % q), r);
                auto tl = tilize32(l); std::copy(tl.begin(), tl.end(), dst.begin() + (p * NB + r) * ELEMS);
            }
        }

        // input (frequency domain) + golden
        std::mt19937 rng(seed); std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)q - 1);
        std::vector<uint32_t> X(N); for (auto& v : X) v = dist(rng);
        std::vector<std::vector<uint32_t>> XT(NBLK, std::vector<uint32_t>(ELEMS));
        for (uint32_t bk2 = 0; bk2 < NBLK; ++bk2) {
            std::vector<uint32_t> blk(ELEMS);
            for (uint32_t k1 = 0; k1 < N1; ++k1) for (uint32_t k2loc = 0; k2loc < T; ++k2loc)
                blk[k1 * T + k2loc] = X[(bk2 * T + k2loc) + N2 * k1];
            XT[bk2] = tilize32(blk);
        }
        std::vector<std::vector<uint32_t>> TwT(NBLK, std::vector<uint32_t>(ELEMS));
        for (uint32_t bk2 = 0; bk2 < NBLK; ++bk2) {
            std::vector<uint32_t> blk(ELEMS);
            for (uint32_t n1 = 0; n1 < N1; ++n1) for (uint32_t k2loc = 0; k2loc < T; ++k2loc)
                blk[n1 * T + k2loc] = ctx.to_montgomery(Tw[n1 * N2 + (bk2 * T + k2loc)]);
            TwT[bk2] = tilize32(blk);
        }
        std::vector<uint32_t> xref(N);                                 // x[n] = N⁻¹·ψ⁻ⁿ·Σ_k X[k]·ω⁻ⁿᵏ
        for (uint32_t n = 0; n < N; ++n) { u128 acc = 0;
            for (uint32_t k = 0; k < N; ++k) acc += (u128)X[k] * modpow(w, (uint64_t)n * k, q);
            acc %= q; acc = acc * (u128)Ninv % q; acc = acc * (u128)modpow(psinv, n, q) % q;
            xref[n] = (uint32_t)acc; }

        // device
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        auto mkbuf = [&](uint32_t ntiles, uint32_t page) {
            distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = ntiles * page}, d, mesh.get()); };
        auto mkcb = [&](Program& p, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
            CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, core, c); };

        std::vector<std::shared_ptr<distributed::MeshBuffer>> b_X(NBLK), b_Tw(NBLK), b_P(NBLK), b_Q(NBLK),
            b_Rp(NBLK), b_R(NBLK);
        for (uint32_t i = 0; i < NBLK; ++i) { b_X[i] = mkbuf(1, I32); b_Tw[i] = mkbuf(1, I32); b_P[i] = mkbuf(1, I32);
            b_Q[i] = mkbuf(1, I32); b_Rp[i] = mkbuf(NBLK, I32); b_R[i] = mkbuf(1, I32); }
        std::vector<std::shared_ptr<distributed::MeshBuffer>> b_W256(NBLK * NBLK);
        for (uint32_t i = 0; i < NBLK * NBLK; ++i) b_W256[i] = mkbuf(NB * NB, I8);
        auto b_W32 = mkbuf(NB * NB, I8);

        auto build_fusedA = [&](auto xin, auto w, auto out) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::Int8, NB * NB, I8);
            mkcb(p, CBIndex::c_2, tt::DataFormat::Int8, NB, I8); mkcb(p, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);
            mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*xin).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)xin->address(), (uint32_t)w->address()});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, 0u});
            SetRuntimeArgs(p, cp, core, {(uint32_t)q, QINV, QLO, QHI, K14});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        auto build_fusedC = [&](auto gp, auto w, auto out, uint32_t out_start) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::Int8, NB * NB, I8);
            mkcb(p, CBIndex::c_2, tt::DataFormat::Int8, NB, I8); mkcb(p, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);
            mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*gp).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)gp->address(), (uint32_t)w->address()});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, out_start});
            SetRuntimeArgs(p, cp, core, {(uint32_t)q, QINV, QLO, QHI, K14});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        auto build_modadd = [&](auto in, auto out, uint32_t n_in) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, n_in, I32); mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)in->address(), n_in, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, 0u});
            SetRuntimeArgs(p, cp, core, {n_in, (uint32_t)q});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        auto build_modmul = [&](auto a, auto b, auto out) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::UInt32, 2, I32);
            mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)a->address(), (uint32_t)b->address(), 1u, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, 0u});
            SetRuntimeArgs(p, cp, core, {1u, (uint32_t)q, QINV, QLO, QHI});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // write constants + input
        for (uint32_t i = 0; i < NBLK; ++i) { distributed::EnqueueWriteMeshBuffer(cq, b_X[i], XT[i], false);
            distributed::EnqueueWriteMeshBuffer(cq, b_Tw[i], TwT[i], false); }
        distributed::EnqueueWriteMeshBuffer(cq, b_W32, Winv32fold, false);
        for (uint32_t i = 0; i < NBLK * NBLK; ++i) distributed::EnqueueWriteMeshBuffer(cq, b_W256[i], Winv256fold[i], false);

        // program chain
        std::vector<distributed::MeshWorkload> prog;
        for (uint32_t bk2 = 0; bk2 < NBLK; ++bk2) {
            prog.push_back(build_fusedA(b_X[bk2], b_W32, b_P[bk2]));
            prog.push_back(build_modmul(b_P[bk2], b_Tw[bk2], b_Q[bk2]));
        }
        for (uint32_t bn2 = 0; bn2 < NBLK; ++bn2) {
            for (uint32_t bk2 = 0; bk2 < NBLK; ++bk2)
                prog.push_back(build_fusedC(b_Q[bk2], b_W256[bk2 * NBLK + bn2], b_Rp[bn2], bk2));
            prog.push_back(build_modadd(b_Rp[bn2], b_R[bn2], NBLK));
        }
        for (auto& wl : prog) distributed::EnqueueMeshWorkload(cq, wl, false);

        // read + reindex
        std::vector<uint32_t> out(N);
        for (uint32_t bn2 = 0; bn2 < NBLK; ++bn2) {
            std::vector<uint32_t> raw; distributed::EnqueueReadMeshBuffer(cq, raw, b_R[bn2], true);
            auto Rblk = untilize32(raw);
            for (uint32_t n1 = 0; n1 < N1; ++n1) for (uint32_t n2loc = 0; n2loc < T; ++n2loc) {
                uint32_t n2 = bn2 * T + n2loc;
                out[n1 + 32 * n2] = Rblk[n1 * T + n2loc];
            }
        }

        uint32_t mism = 0;
        for (uint32_t n = 0; n < N; ++n)
            if (out[n] != xref[n]) { if (mism < 8) fmt::print("  x[{}] got={} exp={}\n", n, out[n], xref[n]); ++mism; }
        fmt::print("on-device negacyclic INVERSE NTT-8192 (32x256)  q={}  seed={}  mismatches={}/{}\n", q, seed, mism, N);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: on-device negacyclic INVERSE NTT-8192 WORKS.\n"); else fmt::print("RESULT: WRONG.\n");
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
