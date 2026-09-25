// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BFV prototype — RELINEARIZATION / key-switch of c2, fully on device (all arithmetic),
// single core, N=1024, Q=4 / P=4 limbs (QP=8), dnum=1. See docs/he/bfv_prototype_plan.md
// and reference/he/bfv_relin_ref.py (validated golden).
//
//   relinearize(c2) -> (k0,k1):
//     1. mod-up   c2 : Q -> QP        FBC(Q->P) [baseconv_fbc], concat
//     2. nNTT     c2_qp (8 limbs)      [negacyclic NTT, per limb]
//     3. acc_i = c2_ntt (.) evk_i      [pointwise modmul, evk in nNTT/Montgomery form]  (dnum=1)
//     4. niNTT    acc0, acc1           [negacyclic INTT, per limb]
//     5. mod-down acc0,acc1 : QP -> Q  [FBC(P->Q) + modsub + ×P^-1]  -> k0, k1
//
// PLAIN domain (matches the validated transforms); constants carry the Montgomery R.
// Host does ONLY tilize/untilize + the NTT reindex (permutation, no arithmetic).
// Validated per-limb (RNS) against a C++ replica of the golden relinearize.
//   ./build/bfv_relin [seed]

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
constexpr uint32_t T = 32, NB = 4, N = 1024, ELEMS = 1024, I8 = ELEMS, I32 = ELEMS * 4;
// RNS limb counts are compile-time configurable (default Q=4/P=4 preserves the validated
// binary; -DCFG_LQ=3 -DCFG_LP=3 gives the Q=3 build used by the full-multiply integration).
#ifndef CFG_LQ
#define CFG_LQ 4
#endif
#ifndef CFG_LP
#define CFG_LP 4
#endif
constexpr uint32_t LQ = CFG_LQ, LP = CFG_LP, LQP = CFG_LQ + CFG_LP;
const uint64_t POW[4] = {1, 128, 16384, 2097152};
using u128 = unsigned __int128;

uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) { u128 r = 1, bb = b % m;
    while (e) { if (e & 1) r = r * bb % m; bb = bb * bb % m; e >>= 1; } return (uint64_t)r; }
bool is_prime(uint64_t n) { if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) return false; return true; }
uint64_t prim_root(uint64_t q) { uint64_t phi = q - 1, n = phi; std::vector<uint64_t> f;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) { f.push_back(d); while (n % d == 0) n /= d; }
    if (n > 1) f.push_back(n);
    for (uint64_t g = 2; g < q; ++g) { bool ok = true; for (uint64_t p : f) if (modpow(g, phi / p, q) == 1) ok = false;
        if (ok) return g; } throw std::runtime_error("no root"); }
std::vector<uint64_t> ntt_primes(uint32_t cnt) { std::vector<uint64_t> out;   // p = k·2N+1 (ψ exists)
    for (uint64_t k = ((1ull << 28) - 1) / (2 * N); out.size() < cnt && k > 0; --k) {
        uint64_t p = k * 2 * N + 1; if (p < (1ull << 28) && is_prime(p)) out.push_back(p); } return out; }
int8_t limb(uint64_t v, uint32_t p) { return (int8_t)((v >> (7 * p)) & 127); }

template <typename E> std::vector<E> tilize(const std::vector<E>& in) { std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = in[r * T + c]; return o; }
template <typename E> std::vector<E> untilize(const std::vector<E>& in) { std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[r * T + c] = in[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)]; return o; }
// NTT output reindex: natural[j2+32·j1] = Hgrid[j2·32+j1]
std::vector<uint32_t> reindex(const std::vector<uint32_t>& H) { std::vector<uint32_t> o(N);
    for (uint32_t j1 = 0; j1 < T; ++j1) for (uint32_t j2 = 0; j2 < T; ++j2) o[j2 + T * j1] = H[j2 * T + j1];
    return o; }

// ---- CPU reference (per-limb RNS) : replica of reference/he/bfv_relin_ref.py ----
std::vector<uint32_t> negmul(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, uint64_t q) {
    std::vector<uint32_t> c(N, 0);
    for (uint32_t i = 0; i < N; ++i) { if (!a[i]) continue; u128 ai = a[i];
        for (uint32_t j = 0; j < N; ++j) { uint32_t k = i + j; uint64_t v = (uint64_t)(ai * b[j] % q);
            if (k < N) c[k] = (uint32_t)(((u128)c[k] + v) % q); else c[k - N] = (uint32_t)(((u128)c[k - N] + q - v) % q); } }
    return c; }
}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const uint32_t seed = (argc > 1) ? std::stoul(argv[1]) : 0;
        auto pr = ntt_primes(LQP);
        std::vector<uint64_t> Qm(pr.begin(), pr.begin() + LQ), Pm(pr.begin() + LQ, pr.end());
        std::vector<uint64_t> QP = pr;                       // 8 QP primes
        std::vector<mont::MontgomeryContext> ctx;
        for (auto m : QP) ctx.emplace_back((uint32_t)m);
        u128 Qprod = 1; for (auto q : Qm) Qprod *= q;
        u128 Pprod = 1; for (auto p : Pm) Pprod *= p;

        std::mt19937_64 rng(seed);
        auto small = [&](){ std::vector<int64_t> v(N); for (auto& x : v) x = (int64_t)(rng() % 3) - 1; return v; };
        std::vector<int64_t> s = small();

        // ---- evk (key-switch key for s^2 -> s), coeff domain, per QP limb ----
        //   evk0 = −a·s + e + P·s²,  evk1 = a.   e = ONE small poly (per-limb-consistent!).
        std::vector<int64_t> e_int(N); for (auto& x : e_int) x = (int64_t)(rng() % 5) - 2;
        std::vector<std::vector<uint32_t>> evk0(LQP), evk1(LQP), s2m(LQP);
        // s² as integer poly (small), reduced per limb
        {   std::vector<int64_t> s2(2 * N, 0);
            for (uint32_t i = 0; i < N; ++i) for (uint32_t j = 0; j < N; ++j) { uint32_t k = i + j;
                if (k < N) s2[k] += s[i] * s[j]; else s2[k] -= s[i] * s[j]; }         // negacyclic s²
            for (uint32_t li = 0; li < LQP; ++li) { uint64_t m = QP[li]; std::vector<uint32_t> r(N);
                for (uint32_t n = 0; n < N; ++n) r[n] = (uint32_t)(((s2[n] % (int64_t)m) + m) % m); s2m[li] = r; } }
        for (uint32_t li = 0; li < LQP; ++li) {
            uint64_t m = QP[li];
            std::vector<uint32_t> a(N), sl(N), el(N);
            for (uint32_t n = 0; n < N; ++n) { a[n] = (uint32_t)(rng() % m); sl[n] = (uint32_t)(((s[n] % (int64_t)m) + m) % m); el[n] = (uint32_t)(((e_int[n] % (int64_t)m) + m) % m); }
            auto as_ = negmul(a, sl, m);
            uint64_t Pm_ = (uint64_t)(Pprod % m);
            std::vector<uint32_t> k0(N);
            for (uint32_t n = 0; n < N; ++n) k0[n] = (uint32_t)(((u128)m + el[n] + (u128)Pm_ * s2m[li][n] - as_[n]) % m);
            evk0[li] = k0; evk1[li] = a;
        }

        // c2 : random poly in base Q (coeff domain)
        std::vector<std::vector<uint32_t>> c2(LQ, std::vector<uint32_t>(N));
        for (uint32_t i = 0; i < LQ; ++i) { uint64_t q = Qm[i]; for (uint32_t n = 0; n < N; ++n) c2[i][n] = (uint32_t)(rng() % q); }

        // =========================================================================
        // CPU REFERENCE relinearize (per-limb) -> kref0, kref1  (base Q)
        // =========================================================================
        auto fbc_approx = [&](const std::vector<std::vector<uint32_t>>& x, const std::vector<uint64_t>& src,
                              const std::vector<uint64_t>& dst) {
            uint32_t Ls = src.size(), Ld = dst.size(); u128 S = 1; for (auto s_ : src) S *= s_;
            std::vector<std::vector<uint32_t>> out(Ld, std::vector<uint32_t>(N));
            for (uint32_t n = 0; n < N; ++n) {
                std::vector<uint64_t> y(Ls);
                for (uint32_t i = 0; i < Ls; ++i) y[i] = (uint64_t)(((u128)x[i][n] * modpow((uint64_t)((S / src[i]) % src[i]), src[i] - 2, src[i])) % src[i]);
                for (uint32_t j = 0; j < Ld; ++j) { u128 acc = 0; for (uint32_t i = 0; i < Ls; ++i) acc += (u128)y[i] * ((S / src[i]) % dst[j]); out[j][n] = (uint32_t)(acc % dst[j]); }
            } return out; };
        std::vector<std::vector<uint32_t>> kref0(LQ), kref1(LQ);
        {   // mod-up
            std::vector<std::vector<uint32_t>> c2qp(LQP);
            for (uint32_t i = 0; i < LQ; ++i) c2qp[i] = c2[i];
            auto pP = fbc_approx(c2, Qm, Pm); for (uint32_t j = 0; j < LP; ++j) c2qp[LQ + j] = pP[j];
            // poly-mult
            std::vector<std::vector<uint32_t>> a0(LQP), a1(LQP);
            for (uint32_t li = 0; li < LQP; ++li) { a0[li] = negmul(c2qp[li], evk0[li], QP[li]); a1[li] = negmul(c2qp[li], evk1[li], QP[li]); }
            // mod-down
            auto moddown = [&](std::vector<std::vector<uint32_t>>& acc) {
                std::vector<std::vector<uint32_t>> aP(acc.begin() + LQ, acc.end());
                std::vector<std::vector<uint32_t>> out(LQ, std::vector<uint32_t>(N));
                auto ptoq = fbc_approx(aP, Pm, Qm);
                for (uint32_t j = 0; j < LQ; ++j) { uint64_t q = Qm[j]; uint64_t Pinv = modpow((uint64_t)(Pprod % q), q - 2, q);
                    for (uint32_t n = 0; n < N; ++n) out[j][n] = (uint32_t)((((u128)acc[j][n] + q - ptoq[j][n]) % q) * Pinv % q); }
                return out; };
            kref0 = moddown(a0); kref1 = moddown(a1);
        }

        // =========================================================================
        // DEVICE relinearize
        // =========================================================================
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        auto mkbuf = [&](uint32_t nt, uint32_t page) {
            distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * page}, d, mesh.get()); };
        auto mkcb = [&](Program& p, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
            CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, core, c); };
        std::vector<distributed::MeshWorkload> hold; hold.reserve(256);
        auto run = [&](Program&& p){ hold.emplace_back(); hold.back().add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, hold.back(), false); };

        // scratch buffers reused across transforms
        auto b_Xin = mkbuf(1, I32), b_WfoldA = mkbuf(NB * NB, I8), b_G = mkbuf(1, I32),
             b_Tm = mkbuf(1, I32), b_Gp = mkbuf(1, I32), b_Wrhs = mkbuf(NB * NB, I8), b_H = mkbuf(1, I32);

        // negacyclic transform of a natural-order host vector for prime QP[li], forward or inverse.
        auto transform = [&](const std::vector<uint32_t>& in_nat, uint32_t li, bool fwd) -> std::vector<uint32_t> {
            uint64_t q = QP[li]; auto& C = ctx[li];
            uint32_t QINV = C.q_inv_32(), QLO = q & 0xFFFFu, QHI = q >> 16, K14 = C.to_montgomery(16384u);
            uint64_t g = prim_root(q);
            uint64_t wf = modpow(g, (q - 1) / N, q), w32f = modpow(g, (q - 1) / T, q), psif = modpow(g, (q - 1) / (2 * N), q);
            uint64_t w = fwd ? wf : mont::modular_inverse(wf, q);
            uint64_t w32 = fwd ? w32f : mont::modular_inverse(w32f, q);
            uint64_t psi = fwd ? psif : mont::modular_inverse(psif, q);
            uint64_t Ninv = mont::modular_inverse(N, q);
            std::vector<uint32_t> Wa(ELEMS), Wc(ELEMS), Tw(ELEMS);
            for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b) {
                uint32_t wv = (uint32_t)modpow(w32, (uint64_t)a * b, q);
                if (fwd) { Wa[a*T+b] = (uint32_t)(((u128)wv * modpow(psi, 32ull*b, q)) % q); Wc[a*T+b] = wv; }
                else     { Wa[a*T+b] = (uint32_t)(((u128)wv * Ninv) % q);                    Wc[a*T+b] = (uint32_t)(((u128)wv * modpow(psi, 32ull*b, q)) % q); }
            }
            for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b) {
                uint64_t base = modpow(w, (uint64_t)a * b, q);
                Tw[a*T+b] = (uint32_t)(((u128)base * (fwd ? modpow(psi, (uint64_t)b, q) : modpow(psi, (uint64_t)a, q))) % q);
            }
            std::vector<int8_t> WfoldA(NB*NB*ELEMS), Wrhs(NB*NB*ELEMS);
            for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
                std::vector<int8_t> la(ELEMS), lc(ELEMS);
                for (uint32_t i = 0; i < ELEMS; ++i) { la[i] = limb((uint64_t)((POW[r]*Wa[i]) % q), p); lc[i] = limb((uint64_t)((POW[p]*Wc[i]) % q), r); }
                auto ta = tilize(la); std::copy(ta.begin(), ta.end(), WfoldA.begin() + (p*NB+r)*ELEMS);
                auto tc = tilize(lc); std::copy(tc.begin(), tc.end(), Wrhs.begin() + (p*NB+r)*ELEMS);
            }
            std::vector<uint32_t> Tm(ELEMS); { std::vector<uint32_t> tm(ELEMS); for (uint32_t i = 0; i < ELEMS; ++i) tm[i] = C.to_montgomery(Tw[i]); Tm = tilize(tm); }
            distributed::EnqueueWriteMeshBuffer(cq, b_Xin, tilize(in_nat), false);
            distributed::EnqueueWriteMeshBuffer(cq, b_WfoldA, WfoldA, false);
            distributed::EnqueueWriteMeshBuffer(cq, b_Wrhs, Wrhs, false);
            distributed::EnqueueWriteMeshBuffer(cq, b_Tm, Tm, false);
            // fusedA
            { Program p = CreateProgram();
              mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::Int8, 2, I8);
              mkcb(p, CBIndex::c_2, tt::DataFormat::Int8, NB, I8); mkcb(p, CBIndex::c_3, tt::DataFormat::Int32, NB, I32); mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
              std::vector<uint32_t> rc; TensorAccessorArgs(*b_Xin).append_to(rc); TensorAccessorArgs(*b_WfoldA).append_to(rc);
              auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
              auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a.cpp", core, ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              SetRuntimeArgs(p, rd, core, {(uint32_t)b_Xin->address(), (uint32_t)b_WfoldA->address()});
              SetRuntimeArgs(p, wr, core, {(uint32_t)b_G->address(), 1u, 0u});
              SetRuntimeArgs(p, cp, core, {(uint32_t)q, QINV, QLO, QHI, K14}); run(std::move(p)); }
            // twiddle: G' = G ⊙ Tmont
            { Program p = CreateProgram();
              mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
              auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp", core, ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              SetRuntimeArgs(p, rd, core, {(uint32_t)b_G->address(), (uint32_t)b_Tm->address(), 1u, 0u});
              SetRuntimeArgs(p, wr, core, {(uint32_t)b_Gp->address(), 1u, 0u});
              SetRuntimeArgs(p, cp, core, {1u, (uint32_t)q, QINV, QLO, QHI}); run(std::move(p)); }
            // fusedC
            { Program p = CreateProgram();
              mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::Int8, NB*NB, I8);
              mkcb(p, CBIndex::c_2, tt::DataFormat::Int8, NB, I8); mkcb(p, CBIndex::c_3, tt::DataFormat::Int32, NB, I32); mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
              std::vector<uint32_t> rc; TensorAccessorArgs(*b_Gp).append_to(rc); TensorAccessorArgs(*b_Wrhs).append_to(rc);
              auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
              auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c.cpp", core, ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              SetRuntimeArgs(p, rd, core, {(uint32_t)b_Gp->address(), (uint32_t)b_Wrhs->address()});
              SetRuntimeArgs(p, wr, core, {(uint32_t)b_H->address(), 1u, 0u});
              SetRuntimeArgs(p, cp, core, {(uint32_t)q, QINV, QLO, QHI, K14}); run(std::move(p)); }
            std::vector<uint32_t> Hraw; distributed::EnqueueReadMeshBuffer(cq, Hraw, b_H, true);
            return reindex(untilize(Hraw));
        };

        // pointwise modmul on device: out = a ⊙ b_mont  (plain·mont = plain), for prime QP[li]
        auto b_a = mkbuf(1, I32), b_b = mkbuf(1, I32), b_o = mkbuf(1, I32);
        auto pointwise = [&](const std::vector<uint32_t>& a_nat, const std::vector<uint32_t>& bmont_nat, uint32_t li) {
            uint64_t q = QP[li];
            distributed::EnqueueWriteMeshBuffer(cq, b_a, tilize(a_nat), false);
            distributed::EnqueueWriteMeshBuffer(cq, b_b, tilize(bmont_nat), false);
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_full.cpp", core, ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)b_a->address(), (uint32_t)b_b->address(), 1u, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)b_o->address(), 1u, 0u});
            SetRuntimeArgs(p, cp, core, {1u, (uint32_t)q, ctx[li].q_inv_32(), q & 0xFFFFu, q >> 16}); run(std::move(p));
            std::vector<uint32_t> o; distributed::EnqueueReadMeshBuffer(cq, o, b_o, true); return untilize(o); };

        // ---- fused approximate FBC {x mod src} -> {(x+εS) mod dst}, PADDED to width 4 ----
        // The fused baseconv_fbc kernel is only correct when the modulus count divides the CB
        // depth (n∈{2,4,8}); at n=3 the first (n-2) dst limbs read stale CB data (verified:
        // bfv_moddown_q{3,5,6} fail last-2-correct). Plan §1's blank-limb pad fixes it: run at
        // width FBCW=4 with inert padding — pad src limbs get C1=0 (⇒ y=0, contribute nothing);
        // pad dst limbs get a dummy prime and their output is discarded. src/dst give ctx indices.
        constexpr uint32_t FBCW = 4;
        auto fbc_fused = [&](const std::vector<std::vector<uint32_t>>& xsrc,
                             const std::vector<uint32_t>& src, const std::vector<uint32_t>& dst)
            -> std::vector<std::vector<uint32_t>> {
            uint32_t Ls = src.size(), Ld = dst.size();
            u128 S = 1; for (uint32_t i=0;i<Ls;++i) S *= QP[src[i]];
            std::vector<uint32_t> C1(FBCW,0); std::vector<std::vector<uint32_t>> C2(FBCW, std::vector<uint32_t>(FBCW,0));
            std::vector<uint32_t> Sm(FBCW), SmI(FBCW), Dm(FBCW), DmI(FBCW);
            for (uint32_t i=0;i<FBCW;++i){ uint32_t si = (i<Ls)?src[i]:src[0];   // dummy modulus = src[0]
                uint64_t qi=QP[si]; Sm[i]=(uint32_t)qi; SmI[i]=ctx[si].q_inv_32();
                if (i<Ls) C1[i]=ctx[si].to_montgomery((uint32_t)modpow((uint64_t)((S/qi)%qi), qi-2, qi)); }  // pad ⇒ C1=0
            for (uint32_t j=0;j<FBCW;++j){ uint32_t dj = (j<Ld)?dst[j]:dst[0];   // dummy dst prime = dst[0]
                uint64_t pj=QP[dj]; Dm[j]=(uint32_t)pj; DmI[j]=ctx[dj].q_inv_32();
                if (j<Ld) for (uint32_t i=0;i<Ls;++i) C2[i][j]=ctx[dj].to_montgomery((uint32_t)((S/QP[src[i]])%pj)); }
            auto b_x = mkbuf(FBCW, I32), b_out = mkbuf(FBCW, I32);
            std::vector<uint32_t> xb(FBCW*ELEMS, 0);
            for (uint32_t i=0;i<Ls;++i){ auto t=tilize(xsrc[i]); std::copy(t.begin(),t.end(),xb.begin()+i*ELEMS); }
            distributed::EnqueueWriteMeshBuffer(cq, b_x, xb, false);
            std::vector<uint32_t> ca; ca.push_back(FBCW);
            for (uint32_t i=0;i<FBCW;++i) ca.push_back(C1[i]);
            for (uint32_t i=0;i<FBCW;++i) for (uint32_t j=0;j<FBCW;++j) ca.push_back(C2[i][j]);
            for (uint32_t i=0;i<FBCW;++i) ca.push_back(Sm[i]);
            for (uint32_t i=0;i<FBCW;++i) ca.push_back(SmI[i]);
            for (uint32_t j=0;j<FBCW;++j) ca.push_back(Dm[j]);
            for (uint32_t j=0;j<FBCW;++j) ca.push_back(DmI[j]);
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 8, I32); mkcb(p, CBIndex::c_2, tt::DataFormat::UInt32, 8, I32); mkcb(p, CBIndex::c_3, tt::DataFormat::UInt32, 8, I32); mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 8, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/baseconv_fbc.cpp", core, ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)b_x->address(), FBCW, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)b_out->address(), FBCW, 0u});
            SetRuntimeArgs(p, cp, core, ca); run(std::move(p));
            std::vector<uint32_t> ob; distributed::EnqueueReadMeshBuffer(cq, ob, b_out, true);
            std::vector<std::vector<uint32_t>> out(Ld);
            for (uint32_t j=0;j<Ld;++j) out[j] = untilize(std::vector<uint32_t>(ob.begin()+j*ELEMS, ob.begin()+(j+1)*ELEMS));
            return out; };

        // ---- 1. mod-up: FBC(Q->P)(c2) on device (plain scheme), padded to width 4 ----
        std::vector<std::vector<uint32_t>> c2qp(LQP);
        for (uint32_t i = 0; i < LQ; ++i) c2qp[i] = c2[i];
        {   std::vector<uint32_t> srcIdx(LQ), dstIdx(LP);
            for (uint32_t i=0;i<LQ;++i) srcIdx[i]=i;
            for (uint32_t j=0;j<LP;++j) dstIdx[j]=LQ+j;
            auto pP = fbc_fused(c2, srcIdx, dstIdx);
            for (uint32_t j=0;j<LP;++j) c2qp[LQ+j] = pP[j]; }

        // ---- 2-4. per QP limb: nNTT, ⊙evk0/evk1, niNTT  -> acc0,acc1 (coeff, QP) ----
        std::vector<std::vector<uint32_t>> acc0(LQP), acc1(LQP);
        for (uint32_t li = 0; li < LQP; ++li) {
            uint64_t m = QP[li];
            auto c2f = transform(c2qp[li], li, true);                    // nNTT
            // evk in freq domain (CPU nNTT of evk_coeff), Montgomery-encoded
            auto nntt_cpu = [&](const std::vector<uint32_t>& v){ std::vector<uint32_t> X(N);
                uint64_t g=prim_root(m), w=modpow(g,(m-1)/N,m), psi=modpow(g,(m-1)/(2*N),m);
                for (uint32_t k=0;k<N;++k){ u128 acc=0; for (uint32_t n=0;n<N;++n) acc += (u128)v[n]*((u128)modpow(psi,n,m)*modpow(w,(uint64_t)n*k,m)%m); X[k]=(uint32_t)(acc%m);} return X; };
            auto e0f = nntt_cpu(evk0[li]); auto e1f = nntt_cpu(evk1[li]);
            std::vector<uint32_t> e0m(N), e1m(N); for (uint32_t n=0;n<N;++n){ e0m[n]=ctx[li].to_montgomery(e0f[n]); e1m[n]=ctx[li].to_montgomery(e1f[n]); }
            auto a0f = pointwise(c2f, e0m, li);
            auto a1f = pointwise(c2f, e1m, li);
            acc0[li] = transform(a0f, li, false);                        // niNTT
            acc1[li] = transform(a1f, li, false);
        }

        // ---- 5. mod-down acc0,acc1 : QP->Q  -> k0,k1 (device: FBC(P->Q)+modsub+×P^-1) ----
        auto moddown_dev = [&](std::vector<std::vector<uint32_t>>& acc) {
            // FBC(P->Q) of the P-part, padded to width 4 (src=P limbs, dst=Q limbs)
            std::vector<std::vector<uint32_t>> xP(LP);
            std::vector<uint32_t> srcIdx(LP), dstIdx(LQ);
            for (uint32_t i=0;i<LP;++i){ xP[i]=acc[LQ+i]; srcIdx[i]=LQ+i; }
            for (uint32_t j=0;j<LQ;++j) dstIdx[j]=j;
            auto fbcR = fbc_fused(xP, srcIdx, dstIdx);
            std::vector<std::vector<uint32_t>> out(LQ, std::vector<uint32_t>(N));
            for (uint32_t j=0;j<LQ;++j){ uint64_t q=Qm[j]; uint64_t Pinv=modpow((uint64_t)(Pprod%q),q-2,q);
                for (uint32_t n=0;n<N;++n){ uint32_t diff=(uint32_t)(((u128)acc[j][n]+q-fbcR[j][n])%q); out[j][n]=(uint32_t)((u128)diff*Pinv%q);} }
            return out; };
        auto k0 = moddown_dev(acc0), k1 = moddown_dev(acc1);

        // ---- validate device (k0,k1) vs CPU reference (kref0,kref1) per limb ----
        uint32_t mism = 0;
        for (uint32_t j=0;j<LQ;++j) for (uint32_t n=0;n<N;++n) {
            if (k0[j][n]!=kref0[j][n]){ if(mism<8) fmt::print("  k0 limb{} n{} got={} exp={}\n",j,n,k0[j][n],kref0[j][n]); ++mism; }
            if (k1[j][n]!=kref1[j][n]){ if(mism<8) fmt::print("  k1 limb{} n{} got={} exp={}\n",j,n,k1[j][n],kref1[j][n]); ++mism; }
        }
        fmt::print("BFV relinearize (device)  N=1024 Q({})/P({})  seed={}  mismatches={}/{}\n", LQ, LP, seed, mism, 2*LQ*N);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: BFV relinearization on device WORKS (matches golden).\n"); else fmt::print("RESULT: WRONG.\n");
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
