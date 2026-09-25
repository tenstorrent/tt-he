// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BFV RELINEARIZATION / hybrid key-switch of c2 with dnum>1 DIGIT DECOMPOSITION, fully on
// device (all heavy modular arithmetic), single core, N=1024.  Small easy-divide workload:
// Q=4 limbs, P=2 limbs, dnum=2 -> alpha=2 limbs/digit.  BConv = CRNS-family combine
// (baseconv_fbc.cpp; the key-switch tolerates the approximate/ε-uncorrected form — exact ε
// recovery would need a >32-bit accumulator and is only needed in rescale).  Golden:
// reference/he/bfv_relin_dnum_ref.py (key-switch identity, all seeds).
//
//   relinearize_dnum(c2) -> (k0,k1):
//     for each digit j (group of alpha Q-limbs):
//       1. mod-up  v_j = [c2]_{Q_j} : Q_j -> QP      (own limbs kept, rest via CRNS FBC)  [baseconv_fbc]
//       2. per QP limb: nNTT(v_j) ⊙ evk_j            (evk_j freq/Montgomery, one pair per digit)
//       3. accumulate the freq products over digits
//     4. niNTT the accumulators -> acc0,acc1 (coeff, QP)
//     5. mod-down acc0,acc1 : QP -> Q               (CRNS FBC(P->Q) + modsub + ×P^-1)  -> k0,k1
//
// The switching key for digit j encrypts P·ĝ_j·s²,  ĝ_j = (Q/Q_j)·[(Q/Q_j)^{-1} mod Q_j] (mod Q),
// the CRT idempotent so Σ_j v_j·ĝ_j ≡ c2 (mod Q).  dnum=1 collapses to bfv_relin.cpp.
// Validated per-limb (RNS) bit-exact against an in-file CPU replica, plus the key-switch identity.
//   ./build/bfv_relin_dnum [seed]

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
// Small easy-divide workload: Q=4, P=2, dnum=2 (alpha = LQ/dnum = 2 limbs per digit).
#ifndef CFG_LQ
#define CFG_LQ 4
#endif
#ifndef CFG_LP
#define CFG_LP 2
#endif
#ifndef CFG_DNUM
#define CFG_DNUM 2
#endif
constexpr uint32_t LQ = CFG_LQ, LP = CFG_LP, LQP = CFG_LQ + CFG_LP, DNUM = CFG_DNUM, ALPHA = LQ / DNUM;
static_assert(LQ % DNUM == 0, "dnum must divide LQ");
const uint64_t POW[4] = {1, 128, 16384, 2097152};
using u128 = unsigned __int128;
using i128 = __int128;

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

// modular inverse over a (possibly composite) modulus via extended Euclid (i128 headroom)
i128 egcd(i128 a, i128 b, i128& x, i128& y) { if (b == 0) { x = 1; y = 0; return a; }
    i128 x1, y1; i128 g = egcd(b, a % b, x1, y1); x = y1; y = x1 - (a / b) * y1; return g; }
u128 modinv128(u128 a, u128 m) { i128 x, y; egcd((i128)(a % m), (i128)m, x, y);
    i128 r = x % (i128)m; if (r < 0) r += (i128)m; return (u128)r; }

template <typename E> std::vector<E> tilize(const std::vector<E>& in) { std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = in[r * T + c]; return o; }
template <typename E> std::vector<E> untilize(const std::vector<E>& in) { std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[r * T + c] = in[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)]; return o; }
std::vector<uint32_t> reindex(const std::vector<uint32_t>& H) { std::vector<uint32_t> o(N);
    for (uint32_t j1 = 0; j1 < T; ++j1) for (uint32_t j2 = 0; j2 < T; ++j2) o[j2 + T * j1] = H[j2 * T + j1];
    return o; }

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
        std::vector<uint64_t> QP = pr;
        std::vector<mont::MontgomeryContext> ctx;
        for (auto m : QP) ctx.emplace_back((uint32_t)m);
        u128 Qprod = 1; for (auto q : Qm) Qprod *= q;
        u128 Pprod = 1; for (auto p : Pm) Pprod *= p;

        // digit groups: group j owns Q-limb indices [j*ALPHA .. j*ALPHA+ALPHA-1]
        std::vector<std::vector<uint32_t>> group(DNUM);
        for (uint32_t j = 0; j < DNUM; ++j) for (uint32_t k = 0; k < ALPHA; ++k) group[j].push_back(j * ALPHA + k);
        // ĝ_j (mod Q) as u128, and its per-limb reductions
        std::vector<u128> ghat(DNUM);
        for (uint32_t j = 0; j < DNUM; ++j) {
            u128 Qj = 1; for (auto i : group[j]) Qj *= Qm[i];
            u128 Qhat = Qprod / Qj;
            u128 inv = modinv128(Qhat % Qj, Qj);
            ghat[j] = Qhat % Qprod * inv % Qprod;              // Qhat, inv < 2^56 -> product < 2^112 (fits u128)
        }

        std::mt19937_64 rng(seed);
        auto small = [&](){ std::vector<int64_t> v(N); for (auto& x : v) x = (int64_t)(rng() % 3) - 1; return v; };
        std::vector<int64_t> s = small();

        // s² as integer poly (small), reduced per limb
        std::vector<std::vector<uint32_t>> s2m(LQP);
        {   std::vector<int64_t> s2(2 * N, 0);
            for (uint32_t i = 0; i < N; ++i) for (uint32_t j = 0; j < N; ++j) { uint32_t k = i + j;
                if (k < N) s2[k] += s[i] * s[j]; else s2[k] -= s[i] * s[j]; }
            for (uint32_t li = 0; li < LQP; ++li) { uint64_t m = QP[li]; std::vector<uint32_t> r(N);
                for (uint32_t n = 0; n < N; ++n) r[n] = (uint32_t)(((s2[n] % (int64_t)m) + m) % m); s2m[li] = r; } }

        // ---- evk PER DIGIT: evk_j0 = −a_j·s + e_j + P·ĝ_j·s²,  evk_j1 = a_j  (coeff, per QP limb) ----
        std::vector<std::vector<std::vector<uint32_t>>> evk0(DNUM, std::vector<std::vector<uint32_t>>(LQP)),
                                                        evk1(DNUM, std::vector<std::vector<uint32_t>>(LQP));
        for (uint32_t j = 0; j < DNUM; ++j) {
            std::vector<int64_t> e_int(N); for (auto& x : e_int) x = (int64_t)(rng() % 5) - 2;
            for (uint32_t li = 0; li < LQP; ++li) {
                uint64_t m = QP[li];
                std::vector<uint32_t> a(N), sl(N), el(N);
                for (uint32_t n = 0; n < N; ++n) { a[n] = (uint32_t)(rng() % m);
                    sl[n] = (uint32_t)(((s[n] % (int64_t)m) + m) % m); el[n] = (uint32_t)(((e_int[n] % (int64_t)m) + m) % m); }
                auto as_ = negmul(a, sl, m);
                uint64_t Pghat = (uint64_t)((Pprod % m) * (ghat[j] % m) % m);   // < 2^28
                std::vector<uint32_t> k0(N);
                for (uint32_t n = 0; n < N; ++n) k0[n] = (uint32_t)(((u128)m + el[n] + (u128)Pghat * s2m[li][n] - as_[n]) % m);
                evk0[j][li] = k0; evk1[j][li] = a;
            }
        }

        // c2 : random poly in base Q (coeff domain)
        std::vector<std::vector<uint32_t>> c2(LQ, std::vector<uint32_t>(N));
        for (uint32_t i = 0; i < LQ; ++i) { uint64_t q = Qm[i]; for (uint32_t n = 0; n < N; ++n) c2[i][n] = (uint32_t)(rng() % q); }

        // complement QP-limb indices for each digit (all QP limbs NOT in the digit's group)
        std::vector<std::vector<uint32_t>> comp(DNUM);
        for (uint32_t j = 0; j < DNUM; ++j) for (uint32_t li = 0; li < LQP; ++li) {
            bool own = false; for (auto g : group[j]) if (g == li) own = true; if (!own) comp[j].push_back(li); }

        // =========================================================================
        // CPU REFERENCE relinearize_dnum (per-limb) -> kref0, kref1  (base Q)
        // =========================================================================
        auto fbc_approx = [&](const std::vector<std::vector<uint32_t>>& x, const std::vector<uint64_t>& src,
                              const std::vector<uint64_t>& dst) {
            uint32_t Ls = src.size(), Ld = dst.size(); u128 S = 1; for (auto s_ : src) S *= s_;
            std::vector<std::vector<uint32_t>> out(Ld, std::vector<uint32_t>(N));
            for (uint32_t n = 0; n < N; ++n) {
                std::vector<uint64_t> y(Ls);
                for (uint32_t i = 0; i < Ls; ++i) y[i] = (uint64_t)(((u128)x[i][n] * modpow((uint64_t)((S / src[i]) % src[i]), src[i] - 2, src[i])) % src[i]);
                for (uint32_t jj = 0; jj < Ld; ++jj) { u128 acc = 0; for (uint32_t i = 0; i < Ls; ++i) acc += (u128)y[i] * ((S / src[i]) % dst[jj]); out[jj][n] = (uint32_t)(acc % dst[jj]); }
            } return out; };
        auto moddown_cpu = [&](std::vector<std::vector<uint32_t>>& acc) {
            std::vector<std::vector<uint32_t>> aP(acc.begin() + LQ, acc.end());
            std::vector<std::vector<uint32_t>> out(LQ, std::vector<uint32_t>(N));
            auto ptoq = fbc_approx(aP, Pm, Qm);
            for (uint32_t j = 0; j < LQ; ++j) { uint64_t q = Qm[j]; uint64_t Pinv = modpow((uint64_t)(Pprod % q), q - 2, q);
                for (uint32_t n = 0; n < N; ++n) out[j][n] = (uint32_t)((((u128)acc[j][n] + q - ptoq[j][n]) % q) * Pinv % q); }
            return out; };
        std::vector<std::vector<uint32_t>> kref0, kref1;
        {   std::vector<std::vector<uint32_t>> acc0(LQP, std::vector<uint32_t>(N, 0)), acc1(LQP, std::vector<uint32_t>(N, 0));
            for (uint32_t j = 0; j < DNUM; ++j) {
                // mod-up digit j
                std::vector<std::vector<uint32_t>> vj(LQP);
                for (auto i : group[j]) vj[i] = c2[i];
                std::vector<std::vector<uint32_t>> xsrc; std::vector<uint64_t> srcMod, dstMod;
                for (auto i : group[j]) { xsrc.push_back(c2[i]); srcMod.push_back(Qm[i]); }
                for (auto li : comp[j]) dstMod.push_back(QP[li]);
                auto out = fbc_approx(xsrc, srcMod, dstMod);
                for (uint32_t k = 0; k < comp[j].size(); ++k) vj[comp[j][k]] = out[k];
                // ×evk accumulate (coeff domain)
                for (uint32_t li = 0; li < LQP; ++li) { uint64_t m = QP[li];
                    auto p0 = negmul(vj[li], evk0[j][li], m), p1 = negmul(vj[li], evk1[j][li], m);
                    for (uint32_t n = 0; n < N; ++n) { acc0[li][n] = (uint32_t)(((u128)acc0[li][n] + p0[n]) % m);
                        acc1[li][n] = (uint32_t)(((u128)acc1[li][n] + p1[n]) % m); } }
            }
            kref0 = moddown_cpu(acc0); kref1 = moddown_cpu(acc1);
        }

        // =========================================================================
        // DEVICE relinearize_dnum
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
        std::vector<distributed::MeshWorkload> hold; hold.reserve(512);
        auto run = [&](Program&& p){ hold.emplace_back(); hold.back().add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, hold.back(), false); };

        auto b_Xin = mkbuf(1, I32), b_WfoldA = mkbuf(NB * NB, I8), b_G = mkbuf(1, I32),
             b_Tm = mkbuf(1, I32), b_Gp = mkbuf(1, I32), b_Wrhs = mkbuf(NB * NB, I8), b_H = mkbuf(1, I32);

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
            { Program p = CreateProgram();
              mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_1, tt::DataFormat::UInt32, 2, I32); mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
              auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp", core, ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              SetRuntimeArgs(p, rd, core, {(uint32_t)b_G->address(), (uint32_t)b_Tm->address(), 1u, 0u});
              SetRuntimeArgs(p, wr, core, {(uint32_t)b_Gp->address(), 1u, 0u});
              SetRuntimeArgs(p, cp, core, {1u, (uint32_t)q, QINV, QLO, QHI}); run(std::move(p)); }
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

        // ---- CRNS-family FBC (baseconv_fbc combine), PADDED to width 4. src/dst are ctx(QP) indices. ----
        constexpr uint32_t FBCW = 4;
        auto fbc_fused = [&](const std::vector<std::vector<uint32_t>>& xsrc,
                             const std::vector<uint32_t>& src, const std::vector<uint32_t>& dst)
            -> std::vector<std::vector<uint32_t>> {
            uint32_t Ls = src.size(), Ld = dst.size();
            u128 S = 1; for (uint32_t i=0;i<Ls;++i) S *= QP[src[i]];
            std::vector<uint32_t> C1(FBCW,0); std::vector<std::vector<uint32_t>> C2(FBCW, std::vector<uint32_t>(FBCW,0));
            std::vector<uint32_t> Sm(FBCW), SmI(FBCW), Dm(FBCW), DmI(FBCW);
            for (uint32_t i=0;i<FBCW;++i){ uint32_t si = (i<Ls)?src[i]:src[0];
                uint64_t qi=QP[si]; Sm[i]=(uint32_t)qi; SmI[i]=ctx[si].q_inv_32();
                if (i<Ls) C1[i]=ctx[si].to_montgomery((uint32_t)modpow((uint64_t)((S/qi)%qi), qi-2, qi)); }
            for (uint32_t j=0;j<FBCW;++j){ uint32_t dj = (j<Ld)?dst[j]:dst[0];
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

        // ---- per-digit mod-up (CRNS FBC) -> vj[digit][QP limb] ----
        std::vector<std::vector<std::vector<uint32_t>>> vj(DNUM, std::vector<std::vector<uint32_t>>(LQP));
        for (uint32_t j = 0; j < DNUM; ++j) {
            for (auto i : group[j]) vj[j][i] = c2[i];                 // own limbs kept
            std::vector<std::vector<uint32_t>> xsrc;
            for (auto i : group[j]) xsrc.push_back(c2[i]);
            auto out = fbc_fused(xsrc, group[j], comp[j]);            // src=digit Q-limbs, dst=complement QP
            for (uint32_t k = 0; k < comp[j].size(); ++k) vj[j][comp[j][k]] = out[k];
        }

        // ---- per QP limb: Σ_digit niNTT(nNTT(vj)⊙evk_j) accumulated in freq domain -> acc0,acc1 ----
        auto nntt_cpu = [&](const std::vector<uint32_t>& v, uint64_t m){ std::vector<uint32_t> X(N);
            uint64_t g=prim_root(m), w=modpow(g,(m-1)/N,m), psi=modpow(g,(m-1)/(2*N),m);
            for (uint32_t k=0;k<N;++k){ u128 acc=0; for (uint32_t n=0;n<N;++n) acc += (u128)v[n]*((u128)modpow(psi,n,m)*modpow(w,(uint64_t)n*k,m)%m); X[k]=(uint32_t)(acc%m);} return X; };
        std::vector<std::vector<uint32_t>> acc0(LQP), acc1(LQP);
        for (uint32_t li = 0; li < LQP; ++li) {
            uint64_t m = QP[li];
            std::vector<uint32_t> f0(N, 0), f1(N, 0);                 // freq-domain accumulators (natural order)
            for (uint32_t j = 0; j < DNUM; ++j) {
                auto c2f = transform(vj[j][li], li, true);           // nNTT(v_j)
                auto e0f = nntt_cpu(evk0[j][li], m), e1f = nntt_cpu(evk1[j][li], m);
                std::vector<uint32_t> e0m(N), e1m(N);
                for (uint32_t n=0;n<N;++n){ e0m[n]=ctx[li].to_montgomery(e0f[n]); e1m[n]=ctx[li].to_montgomery(e1f[n]); }
                auto p0 = pointwise(c2f, e0m, li), p1 = pointwise(c2f, e1m, li);
                for (uint32_t n=0;n<N;++n){ f0[n]=(uint32_t)(((u128)f0[n]+p0[n])%m); f1[n]=(uint32_t)(((u128)f1[n]+p1[n])%m); }
            }
            acc0[li] = transform(f0, li, false);                     // niNTT
            acc1[li] = transform(f1, li, false);
        }

        // ---- mod-down acc0,acc1 : QP->Q  (CRNS FBC(P->Q) + modsub + ×P^-1) ----
        auto moddown_dev = [&](std::vector<std::vector<uint32_t>>& acc) {
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
        fmt::print("BFV relin dnum (device)  N=1024 Q({})/P({}) dnum={} alpha={}  seed={}  mismatches={}/{}\n",
                   LQ, LP, DNUM, ALPHA, seed, mism, 2*LQ*N);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: dnum-decomposed key-switch on device WORKS (bit-exact vs CPU reference).\n");
        else      fmt::print("RESULT: WRONG.\n");
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
