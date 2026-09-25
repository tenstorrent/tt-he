// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FULLY ON-DEVICE size-1024 NTT (N = 32x32 four-step) on a single Tensix core.
// Input: (32,32) int32 (= x reshaped, Xin[n2,n1]=x[n1+32 n2]).
// Output: (32,32) int32 NTT result (natural order after host reindex of the read).
//
// There is NO host<->device data movement between stages: the host writes the
// input + baked constants once, issues 7 device programs that chain through
// DRAM buffers, and reads the result once.
//
// The radix fold is ALWAYS baked into the constant W offline (never computed on the
// dynamic data) -> no runtime fold modmul in either stage:
//   A: G  = W @ Xin    decompose(Xin) -> matmul(folded W) -> recombine
//   B: G' = G (.) T     twiddle modmul (T pre-converted to Montgomery)
//   C: H  = G' @ W      decompose(G') -> matmul(folded W) -> recombine
//   out[j2+32 j1] = H[j2,j1]   (host reindex of the single output read)
//
// Per-stage matmul: C_r = sum_p limb_p(A) @ limb_r(128^p*B mod q), 4 int32 planes.
// recombine R = (C0 + C1<<7 + C2<<14 + C3<<21) mod q, ONE kernel (recombine_compute.cpp)
// with a single Montgomery reduction:  low=C0+(C1<<7); hi=C2+(C3<<7);
// R = (low + (hi*2^14 mod q)) mod q.

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
using u128 = unsigned __int128;

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

        // ---- NEGACYCLIC roots and baked constants ----
        // Negacyclic NTT X̂[k] = Σ_n x[n]·ψ^n·ω^{nk}  (ψ = 2N-th root, ψ²=ω) — for the ring
        // x^N+1, so pointwise-multiply in this domain = negacyclic convolution. The ψ^n factor
        // folds into the four-step constants (zero extra passes): with n = n1+32·n2,
        //   ψ^n = ψ^{n1}·ψ^{32·n2}  →  ψ^{32·n2} into stage-A W columns, ψ^{n1} into twiddle columns.
        uint64_t g = prim_root(q), w = modpow(g, (q - 1) / N, q), w32 = modpow(g, (q - 1) / T, q);
        uint64_t psi = modpow(g, (q - 1) / (2 * N), q);          // primitive 2N-th root, ψ²=ω
        std::vector<uint32_t> Wc(ELEMS), Wa(ELEMS), Tw(ELEMS), x(N);
        for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b) {
            uint32_t wv = (uint32_t)modpow(w32, (uint64_t)a * b, q);
            Wc[a * T + b] = wv;                                                            // stage-C W (unscaled)
            Wa[a * T + b] = (uint32_t)(((u128)wv * modpow(psi, 32ull * b, q)) % q);         // stage-A W · ψ^{32b}
        }
        for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b)                   // twiddle · ψ^{n1=b}
            Tw[a * T + b] = (uint32_t)(((u128)modpow(w, (uint64_t)a * b, q) * modpow(psi, (uint64_t)b, q)) % q);
        std::mt19937 rng(seed); std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)q - 1);
        for (auto& v : x) v = dist(rng);
        std::vector<uint32_t> Xin(ELEMS);
        for (uint32_t n2 = 0; n2 < T; ++n2) for (uint32_t n1 = 0; n1 < T; ++n1) Xin[n2 * T + n1] = x[n1 + T * n2];

        // golden negacyclic NTT:  Xref[k] = Σ_n x[n]·ψ^n·ω^{nk} mod q
        std::vector<uint32_t> Xref(N);
        for (uint32_t k = 0; k < N; ++k) { u128 a = 0;
            for (uint32_t n = 0; n < N; ++n) a += (u128)x[n] * ((u128)modpow(psi, n, q) * modpow(w, (uint64_t)n * k, q) % q);
            Xref[k] = (uint32_t)(a % q); }

        // host-side limb / montgomery prep for the baked tiles
        auto limb = [](uint32_t v, uint32_t p) -> int8_t { return (int8_t)((v >> (7 * p)) & 127); };
        //   stage A: WfoldA uses Wa (= ψ^{32b}·W)   ;   stage C: Wrhs uses Wc (= W, unscaled)
        std::vector<int8_t> WfoldA(NB * NB * ELEMS), Wrhs(NB * NB * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
            std::vector<int8_t> la(ELEMS), lc(ELEMS);
            for (uint32_t i = 0; i < ELEMS; ++i) {
                la[i] = limb((uint32_t)((POW[r] * Wa[i]) % q), p);   // limb_p(128^r · ψ^{32b}·W)
                lc[i] = limb((uint32_t)((POW[p] * Wc[i]) % q), r);   // limb_r(128^p · W)
            }
            auto ta = tilize(la); std::copy(ta.begin(), ta.end(), WfoldA.begin() + (p * NB + r) * ELEMS);
            auto tc = tilize(lc); std::copy(tc.begin(), tc.end(), Wrhs.begin() + (p * NB + r) * ELEMS);
        }
        const uint32_t K14 = ctx.to_montgomery(16384u);              // mont(2^14) for the recombine
        // MONTG_IO=1: prove Montgomery-transparent I/O — feed mont(x), expect mont(Xref). The NTT is
        // Z_q-linear so mont(x) in -> mont(NTT(x)) out with NO kernel/constant change (mont(x)<q still
        // decomposes into 4 int8 limbs). Unified-Montgomery data format for the tensor chain.
        const bool montg = getenv("MONTG_IO") != nullptr;
        if (montg) for (auto& v : Xin) v = ctx.to_montgomery(v);
        std::vector<uint32_t> XinT = tilize(Xin);                    // Xin as one tile (decomposed on device)
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
        // buffers
        auto b_Xin = mkbuf(1, I32), b_xlimbs = mkbuf(NB, I8), b_WfoldA = mkbuf(NB * NB, I8);
        auto b_CrA = mkbuf(NB, I32), b_G = mkbuf(1, I32), b_Tm = mkbuf(1, I32), b_Gp = mkbuf(1, I32);
        auto b_aC = mkbuf(NB, I8), b_Wrhs = mkbuf(NB * NB, I8), b_CrC = mkbuf(NB, I32), b_H = mkbuf(1, I32);

        auto mkcb = [&](Program& p, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
            CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, core, c); };

        // ---- program builders (each returns a MeshWorkload, built once) ----
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

        auto build_matmul = [&](auto a, auto b, auto out) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::Int8, 2, I8); mkcb(p, CBIndex::c_1, tt::DataFormat::Int8, 2, I8);
            mkcb(p, CBIndex::c_16, tt::DataFormat::Int32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*a).append_to(rc); TensorAccessorArgs(*b).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul_planes.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            std::vector<uint32_t> wc; TensorAccessorArgs(*out).append_to(wc);
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_matmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = wc});
            CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/matmul_compute.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)a->address(), (uint32_t)b->address()});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address()});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // matmul with the folded constant on the LHS (16 tiles) and dynamic limbs on the RHS (4):
        // C_p = sum_r a[p*NB+r] @ b[r]  (stage A, fold-into-W).
        auto build_matmul_foldlhs = [&](auto a, auto b, auto out) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::Int8, 2, I8); mkcb(p, CBIndex::c_1, tt::DataFormat::Int8, 2, I8);
            mkcb(p, CBIndex::c_16, tt::DataFormat::Int32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*a).append_to(rc); TensorAccessorArgs(*b).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul_foldlhs.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            std::vector<uint32_t> wc; TensorAccessorArgs(*out).append_to(wc);
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_matmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = wc});
            CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/matmul_compute.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)a->address(), (uint32_t)b->address()});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address()});
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

        // single-kernel recombine: 4 limb tiles C_r -> 1 tile (Σ_r 128^r C_r) mod q
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

        // Fused stage A: decompose + matmul(foldW) + recombine in ONE program/kernel,
        // limb & plane intermediates stay in on-core CBs (no DRAM round-trip).
        auto build_fusedA = [&](auto xin, auto w, auto out) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32);   // Xin
            mkcb(p, CBIndex::c_1, tt::DataFormat::Int8, 2, I8);      // folded W (streamed)
            mkcb(p, CBIndex::c_2, tt::DataFormat::Int8, NB, I8);     // limbs (intermediate, resident)
            mkcb(p, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);   // planes (intermediate)
            mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);  // result
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

        // Fused stage C: decompose(G') + matmul(Wrhs) + recombine in ONE program.
        // Dynamic limbs are the LHS; all 16 Wrhs tiles are resident (RHS), indexed p*NB+r.
        auto build_fusedC = [&](auto gp, auto w, auto out) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32);       // G'
            mkcb(p, CBIndex::c_1, tt::DataFormat::Int8, NB * NB, I8);    // Wrhs (16 resident)
            mkcb(p, CBIndex::c_2, tt::DataFormat::Int8, NB, I8);         // limbs
            mkcb(p, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);       // planes
            mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);      // result
            std::vector<uint32_t> rc; TensorAccessorArgs(*gp).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)gp->address(), (uint32_t)w->address()});
            SetRuntimeArgs(p, wr, core, {(uint32_t)out->address(), 1u, 0u});
            SetRuntimeArgs(p, cp, core, {(uint32_t)q, QINV, QLO, QHI, K14});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // build all stages (programs persist for the run); fold is baked into W (no runtime fold)
        auto A = build_fusedA(b_Xin, b_WfoldA, b_G);           // G = recombine(W @ Xin), fully fused
        auto B = build_modmul(b_G, b_Tm, b_Gp, 1);             // G' = G (.) T
        auto C = build_fusedC(b_Gp, b_Wrhs, b_H);              // H = recombine(G' @ W), fully fused

        // ---- write input + constants ONCE ----
        distributed::EnqueueWriteMeshBuffer(cq, b_Xin, XinT, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_WfoldA, WfoldA, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Wrhs, Wrhs, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Tm, Tmont, false);

        // ---- run the stages, no host reads between (A is fused: decompose+matmul+recombine) ----
        for (auto* wl : {&A, &B, &C})
            distributed::EnqueueMeshWorkload(cq, *wl, false);

        std::vector<uint32_t> Hraw;
        distributed::EnqueueReadMeshBuffer(cq, Hraw, b_H, true);   // single output read
        auto H = untilize(Hraw);

        std::vector<uint32_t> out(N);
        for (uint32_t j1 = 0; j1 < T; ++j1) for (uint32_t j2 = 0; j2 < T; ++j2) out[j2 + T * j1] = H[j2 * T + j1];

        uint32_t mism = 0;
        for (uint32_t j = 0; j < N; ++j) { uint32_t exp = montg ? ctx.to_montgomery(Xref[j]) : Xref[j];
            if (out[j] != exp) { if (mism < 8) fmt::print("  X[{}] got={} exp={}\n", j, out[j], exp); ++mism; } }
        fmt::print("fully-on-device negacyclic-NTT-1024  q={}  seed={}  I/O={}  mismatches={}/{}\n", q, seed, montg?"MONTG":"plain", mism, N);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: on-device negacyclic NTT WORKS.\n"); else fmt::print("RESULT: WRONG.\n");

        // ---- timing: full on-device NTT (10 enqueues), no host work between, one final Finish ----
        {
            std::vector<distributed::MeshWorkload*> wls = {&A, &B, &C};
            for (auto* wl : wls) distributed::EnqueueMeshWorkload(cq, *wl, false);
            distributed::Finish(cq);  // warm
            const int iters = 100;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it) for (auto* wl : wls) distributed::EnqueueMeshWorkload(cq, *wl, false);
            distributed::Finish(cq);
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e3 / iters;
            fmt::print("timing: {:.2f} us / nNTT-1024 (device, single core)\n", us);

            // per-stage breakdown (steady-state enqueue->Finish of each program alone)
            const char* nm[] = {"A  fused matmul", "B  twiddle(modmul)", "C  fused matmul"};
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
