// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PACKED matrix-engine base conversion, EXACT via CRNS fixed-point ε (no m_sk sentinel).
// L->L moduli, N=1024, one Tensix core. The combine (P1) + per-column recombine (P2) are
// the EXACT SAME kernels as baseconv_matmul_packed (plain-domain B here); a third program
// (P3, eps_correct_crns) computes the CRT overflow ε with a fixed-point truncated MAC on
// the SFPU and applies the exact correction:
//
//   P1  matmul_fbc_packed :  planes  (Σ over radix-128 limbs of y_i · (q̂_i mod p_j))
//   P2  recombine_rns     :  combine[cp][col] = (x + εQ) mod p_j          (plain)
//   P3  eps_correct_crns  :  ε = (Σ_i (y_i>>d)·gs_i + off) >> sh          (SFPI, no wide acc)
//                            out = (combine − ε·(Q mod p_j)) mod p_j = x mod p_j   (EXACT)
//
// ε is exact when the input is bounded a full limb below Q (the BFV rescale case), which
// this harness uses (val < Q/max(q_i)). gs_i = round(2^32/q_i); d=18, sh=14, off=2^13
// from the exactness analysis (reference/he/bfv_bconv_crns.py, docs/bconv §2.5).
//
//   ./build/baseconv_matmul_crns [seed] [L]   (L in {4,8}, default 8)

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
constexpr uint32_t T = 32, NR = 4, ELEMS = 1024, N = 1024;
constexpr uint32_t NCT = N / 32;          // 32 coeff-blocks
const uint64_t POW[NR] = {1, 128, 16384, 2097152};
using u128 = unsigned __int128;
// CRNS ε fixed-point params (u=40): ε = (Σ (y_i>>d)·gs_i + off) >> sh, gs_i=round(2^40/q_i).
// u=40 gives gs≈4096 (rounding error ~0.01%) so α=0.5 is robust; acc ≤ 2^28 (< 2^31). See docs/bconv §2.5.
constexpr uint32_t EPS_D = 16, EPS_SH = 24, EPS_OFF = 1u << 23;

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
        const uint32_t L = (argc > 2) ? std::stoul(argv[2]) : 4;
        if (L * 28 > 120) {   // Q = ∏q_i must fit u128 for a real x-mod-p_j reference
            fmt::print("L={} needs a >128-bit Q on host (harness limit). The BFV pipeline uses L≤4; "
                       "run L=4. The kernel itself is L-agnostic.\n", L);
            return 0;
        }
        const uint32_t NBL = 32 / L;          // blocks packed per 32-wide tile
        const uint32_t NSB = NCT / NBL;       // super-blocks

        std::vector<uint64_t> Qm(L), Pm(L);
        { std::vector<uint64_t> pr;
          for (uint64_t c = (1ull << 28) - 1; pr.size() < 2 * L; c -= 2) if (is_prime(c)) pr.push_back(c);
          for (uint32_t i = 0; i < L; ++i) { Qm[i] = pr[i]; Pm[i] = pr[L + i]; } }
        std::vector<mont::MontgomeryContext> ctxP;
        for (uint32_t j = 0; j < L; ++j) ctxP.emplace_back((uint32_t)Pm[j]);

        u128 Q = 1; for (uint32_t i = 0; i < L; ++i) Q *= Qm[i];
        uint64_t maxq = 0; for (auto q : Qm) maxq = std::max(maxq, q);
        std::vector<uint64_t> qhat_inv(L), gs(L), Qmodp(L);
        std::vector<std::vector<uint64_t>> qhat_mod_p(L, std::vector<uint64_t>(L));
        for (uint32_t i = 0; i < L; ++i) {
            u128 qhat = Q / Qm[i];
            qhat_inv[i] = modpow((uint64_t)(qhat % Qm[i]), Qm[i] - 2, Qm[i]);
            gs[i] = (((uint64_t)1 << 40) + Qm[i] / 2) / Qm[i];        // round(2^40/q_i)
            for (uint32_t j = 0; j < L; ++j) qhat_mod_p[i][j] = (uint64_t)(qhat % Pm[j]);
        }
        for (uint32_t j = 0; j < L; ++j) Qmodp[j] = (uint64_t)(Q % Pm[j]);

        // P2 recombine params: column c carries p_{c%L}. order K14,QINV,QHIGH,QLOW,Q.
        std::vector<uint32_t> parblob(5 * ELEMS);
        for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c) {
            uint32_t j = c % L, at = r * T + c;
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

        // P3 params: column c -> QMODP=Q mod p_{c%L}, P=p_{c%L}
        std::vector<uint32_t> par2blob(2 * ELEMS);
        for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c) {
            uint32_t j = c % L, at = r * T + c;
            par2blob[0 * ELEMS + at] = (uint32_t)Qmodp[j];
            par2blob[1 * ELEMS + at] = (uint32_t)Pm[j];
        }
        for (uint32_t t = 0; t < 2; ++t) {
            std::vector<uint32_t> rm(par2blob.begin() + t * ELEMS, par2blob.begin() + (t + 1) * ELEMS);
            auto til = tilize(rm); std::copy(til.begin(), til.end(), par2blob.begin() + t * ELEMS);
        }

        // BOUNDED inputs (BFV rescale case: a full spare limb below Q) so CRNS ε is exact.
        std::mt19937_64 rng(seed);
        u128 bound = Q / maxq;
        std::vector<u128> val(N);
        for (uint32_t c = 0; c < N; ++c) {
            u128 v = ((u128)rng() << 64) | rng();
            val[c] = v % bound;
        }

        // stage 1 (host): y[c][i]
        std::vector<std::vector<uint64_t>> y(N, std::vector<uint64_t>(L));
        for (uint32_t c = 0; c < N; ++c)
            for (uint32_t i = 0; i < L; ++i)
                y[c][i] = (uint64_t)(((u128)(val[c] % Qm[i]) * qhat_inv[i]) % Qm[i]);

        // A: NCT tiles, A_t[cp][i*4+p] = limb_p(y[t*32+cp][i])
        std::vector<int8_t> Ablob(NCT * ELEMS, 0);
        for (uint32_t t = 0; t < NCT; ++t) {
            std::vector<int8_t> Arow(ELEMS, 0);
            for (uint32_t cp = 0; cp < T; ++cp)
                for (uint32_t i = 0; i < L; ++i)
                    for (uint32_t p = 0; p < NR; ++p)
                        Arow[cp * T + (i * NR + p)] = limb(y[t * T + cp][i], p);
            auto til = tilize(Arow); std::copy(til.begin(), til.end(), Ablob.begin() + t * ELEMS);
        }

        // B: NBL*NR tiles, PLAIN (no ·R) — combine output is plain (x+εQ) mod p_j.
        std::vector<int8_t> Bblob(NBL * NR * ELEMS, 0);
        for (uint32_t bl = 0; bl < NBL; ++bl)
            for (uint32_t r = 0; r < NR; ++r) {
                std::vector<int8_t> Brow(ELEMS, 0);
                for (uint32_t i = 0; i < L; ++i)
                    for (uint32_t p = 0; p < NR; ++p)
                        for (uint32_t j = 0; j < L; ++j) {
                            uint64_t Bp = (uint64_t)(((u128)POW[p] * qhat_mod_p[i][j]) % Pm[j]);
                            Brow[(i * NR + p) * T + (bl * L + j)] = limb(Bp, r);
                        }
                auto til = tilize(Brow); std::copy(til.begin(), til.end(), Bblob.begin() + (bl * NR + r) * ELEMS);
            }

        // Y_i tiles for P3: Yblob[(sb*L+i)] tile[cp][bl*L+j] = y[coeff(sb,bl,cp)][i]  (replicated over j)
        std::vector<uint32_t> Yblob(NSB * L * ELEMS, 0);
        for (uint32_t sb = 0; sb < NSB; ++sb)
            for (uint32_t i = 0; i < L; ++i) {
                std::vector<uint32_t> Yrow(ELEMS, 0);
                for (uint32_t cp = 0; cp < T; ++cp)
                    for (uint32_t bl = 0; bl < NBL; ++bl) {
                        uint32_t coeff = (sb * NBL + bl) * T + cp;
                        for (uint32_t j = 0; j < L; ++j)
                            Yrow[cp * T + (bl * L + j)] = (uint32_t)y[coeff][i];
                    }
                auto til = tilize(Yrow); std::copy(til.begin(), til.end(), Yblob.begin() + (sb * L + i) * ELEMS);
            }

        // reference: EXACT plain x mod p_j
        std::vector<std::vector<uint32_t>> ref(L, std::vector<uint32_t>(N));
        for (uint32_t c = 0; c < N; ++c)
            for (uint32_t j = 0; j < L; ++j)
                ref[j][c] = (uint32_t)(val[c] % Pm[j]);

        // ---- device ----
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        constexpr uint32_t I8 = ELEMS, I32 = ELEMS * 4;

        auto mkbuf = [&](uint32_t n, uint32_t page) {
            distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = n * page}, d, mesh.get()); };
        auto b_A = mkbuf(NCT, I8), b_B = mkbuf(NBL * NR, I8), b_planes = mkbuf(NR * NSB, I32);
        auto b_par = mkbuf(5, I32), b_cmb = mkbuf(NSB, I32);
        auto b_Y = mkbuf(NSB * L, I32), b_par2 = mkbuf(2, I32), b_out = mkbuf(NSB, I32);

        auto mkcb = [&](Program& p, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
            CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, core, c); };

        // program 1: packed int8 matmul (plain B) -> NR*NSB int32 packed planes
        Program p1 = CreateProgram();
        mkcb(p1, CBIndex::c_0, tt::DataFormat::Int8, 4, I8);
        mkcb(p1, CBIndex::c_1, tt::DataFormat::Int8, 4, I8);
        mkcb(p1, CBIndex::c_16, tt::DataFormat::Int32, 4, I32);
        std::vector<uint32_t> rc1; TensorAccessorArgs(*b_A).append_to(rc1); TensorAccessorArgs(*b_B).append_to(rc1);
        auto rd1 = CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fbc_packed.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc1});
        auto wr1 = CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto cp1 = CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "compute/matmul_fbc_packed.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false});
        SetRuntimeArgs(p1, rd1, core, {(uint32_t)b_A->address(), (uint32_t)b_B->address(), NSB, NBL});
        SetRuntimeArgs(p1, wr1, core, {(uint32_t)b_planes->address(), NR * NSB, 0u});
        SetRuntimeArgs(p1, cp1, core, {NSB, NBL});

        // program 2: per-column recombine -> combine (plain (x+εQ) mod p_j)
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
        SetRuntimeArgs(p2, rd2, core, {(uint32_t)b_planes->address(), (uint32_t)b_par->address(), NSB});
        SetRuntimeArgs(p2, wr2, core, {(uint32_t)b_cmb->address(), NSB, 0u});
        SetRuntimeArgs(p2, cp2, core, {NSB});

        // program 3: CRNS ε (SFPI fixed-point MAC) + exact correction
        Program p3 = CreateProgram();
        mkcb(p3, CBIndex::c_0, tt::DataFormat::UInt32, 2 * L, I32);
        mkcb(p3, CBIndex::c_1, tt::DataFormat::UInt32, 2, I32);
        mkcb(p3, CBIndex::c_2, tt::DataFormat::UInt32, 2, I32);
        mkcb(p3, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
        auto rd3 = CreateKernel(p3, OVERRIDE_KERNEL_PREFIX "dataflow/reader_eps_correct.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto wr3 = CreateKernel(p3, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto cp3 = CreateKernel(p3, OVERRIDE_KERNEL_PREFIX "compute/eps_correct_crns.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                          .dst_full_sync_en = true, .math_approx_mode = false});
        SetRuntimeArgs(p3, rd3, core, {(uint32_t)b_Y->address(), (uint32_t)b_par2->address(), (uint32_t)b_cmb->address(), NSB, L});
        SetRuntimeArgs(p3, wr3, core, {(uint32_t)b_out->address(), NSB, 0u});
        std::vector<uint32_t> ra3 = {NSB, L, EPS_D, EPS_SH, EPS_OFF};
        for (uint32_t i = 0; i < L; ++i) ra3.push_back((uint32_t)gs[i]);
        SetRuntimeArgs(p3, cp3, core, ra3);

        distributed::MeshWorkload w1; w1.add_program(dr, std::move(p1));
        distributed::MeshWorkload w2; w2.add_program(dr, std::move(p2));
        distributed::MeshWorkload w3; w3.add_program(dr, std::move(p3));
        distributed::EnqueueWriteMeshBuffer(cq, b_A, Ablob, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_B, Bblob, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_par, parblob, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Y, Yblob, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_par2, par2blob, false);
        distributed::EnqueueMeshWorkload(cq, w1, false);
        distributed::EnqueueMeshWorkload(cq, w2, false);
        distributed::EnqueueMeshWorkload(cq, w3, false);
        std::vector<uint32_t> outblob;
        distributed::EnqueueReadMeshBuffer(cq, outblob, b_out, true);

        uint32_t mism = 0;
        for (uint32_t sb = 0; sb < NSB; ++sb) {
            auto out = untilize(std::vector<uint32_t>(outblob.begin() + sb * ELEMS, outblob.begin() + (sb + 1) * ELEMS));
            for (uint32_t cp = 0; cp < T; ++cp)
                for (uint32_t bl = 0; bl < NBL; ++bl)
                    for (uint32_t j = 0; j < L; ++j) {
                        uint32_t got = out[cp * T + (bl * L + j)];
                        uint32_t coeff = (sb * NBL + bl) * T + cp;
                        if (got != ref[j][coeff]) { if (mism < 8) fmt::print("  sb={} bl={} cp={} j={} got={} exp={}\n", sb, bl, cp, j, got, ref[j][coeff]); ++mism; }
                    }
        }
        fmt::print("CRNS exact base-convert {}->{}  N=1024  seed={}  mismatches={}/{}\n", L, L, seed, mism, N * L);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: CRNS exact matrix-engine FBC (fully on-device, no m_sk) WORKS.\n");
        else fmt::print("RESULT: WRONG.\n");

        if (pass) {
            distributed::EnqueueMeshWorkload(cq, w1, false); distributed::EnqueueMeshWorkload(cq, w2, false);
            distributed::EnqueueMeshWorkload(cq, w3, false); distributed::Finish(cq);
            const int iters = 100;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it) {
                distributed::EnqueueMeshWorkload(cq, w1, false);
                distributed::EnqueueMeshWorkload(cq, w2, false);
                distributed::EnqueueMeshWorkload(cq, w3, false);
            }
            distributed::Finish(cq);
            auto t1 = std::chrono::high_resolution_clock::now();
            fmt::print("timing: {:.2f} us / base-convert ({}->{} CRNS exact packed matmul+recombine+ε, 1024 coeffs, 1 core)\n",
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
