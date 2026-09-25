// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Set-B HE-multiply benchmark (CROSS Table VIII parameters) — N=2^13=8192, L=8 limbs (28-bit),
// dnum=2 (alpha=4), P=4 -> QP=12.  docs/he/cross_comparison.md.
//
// M1 stage 1/several: the REUSABLE per-prime multi-core transform8192 (fwd + inv), the block the
// whole op graph calls ~60x.  Both are the grid-parallel fused-dispatch four-step (nntt8192_mc /
// nintt8192 chains) refactored into callables over a SHARED scratch-buffer pool, parameterized by
// prime index.  This file validates the transform (round-trip INTT(NTT(x))==x + NTT vs golden) and
// times it; tensor / CKKS-drop-prime rescale / dnum=2 key-switch are layered on top next.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <set>
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
constexpr uint32_t T = 32, NB = 4, N1 = 32, N2 = 256, N = 8192, ELEMS = 1024;
constexpr uint32_t NBLK = N2 / T;                 // 8
constexpr uint32_t I8 = ELEMS, I32 = ELEMS * 4;
constexpr uint32_t LQ = 8, LP = 4, LQP = LQ + LP; // Set-B: Q=8, P=4, QP=12
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
std::vector<uint64_t> ntt_primes(uint32_t cnt) {   // cnt distinct 28-bit primes p = k*2N+1
    std::vector<uint64_t> out;
    for (uint64_t k = ((1ull << 28) - 1) / (2 * N); out.size() < cnt && k > 0; --k) {
        uint64_t p = k * 2 * N + 1; if (p < (1ull << 28) && is_prime(p)) out.push_back(p); }
    return out; }

template <typename E> std::vector<E> tilize32(const std::vector<E>& in) { std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = in[r * T + c]; return o; }
template <typename E> std::vector<E> untilize32(const std::vector<E>& in) { std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[r * T + c] = in[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)]; return o; }
int8_t limb(uint32_t v, uint32_t p) { return (int8_t)((v >> (7 * p)) & 127); }
}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const uint32_t seed = (argc > 1) ? std::stoul(argv[1]) : 0;
        auto QP = ntt_primes(LQP);
        std::vector<mont::MontgomeryContext> ctx; for (auto m : QP) ctx.emplace_back((uint32_t)m);

        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX = grid.x, GY = grid.y, NCORES = GX * GY;
        auto CC = [&](uint32_t i){ return CoreCoord{i % GX, i / GX}; };
        fmt::print("Set-B HE-mul  N={} LQ={} LP={} QP={}  grid {}x{}={} cores\n", N, LQ, LP, LQP, GX, GY, NCORES);

        auto mkbuf = [&](uint32_t nt, uint32_t page) {
            distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * page}, d, mesh.get()); };
        auto mkcb = [&](Program& p, const CoreRangeSet& crs, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
            CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, crs, c); };
        auto crs_of = [&](uint32_t cnt){ std::set<CoreRange> s; for (uint32_t i=0;i<cnt;++i) s.insert(CoreRange(CC(i))); return CoreRangeSet(s); };

        // ---- shared scratch buffer pool (reused across every transform, rewritten per prime) ----
        std::vector<std::shared_ptr<distributed::MeshBuffer>> in8(NBLK), tw8(NBLK), mid8(NBLK),
            midB8(NBLK), out8(NBLK), part8(NBLK);
        for (uint32_t i = 0; i < NBLK; ++i) { in8[i]=mkbuf(1,I32); tw8[i]=mkbuf(1,I32); mid8[i]=mkbuf(1,I32);
            midB8[i]=mkbuf(1,I32); out8[i]=mkbuf(1,I32); part8[i]=mkbuf(NBLK,I32); }
        std::vector<std::shared_ptr<distributed::MeshBuffer>> bigW(NBLK*NBLK);
        for (uint32_t i=0;i<NBLK*NBLK;++i) bigW[i]=mkbuf(NB*NB,I8);
        auto smallW = mkbuf(NB*NB, I8);

        // ---- generic multi-core stage builders (reused by fwd & inv) ----
        // 64 "big-matmul" placements: fused_matmul (c_1 depth cdep) core i does out=w@in-block.
        struct MM { std::shared_ptr<distributed::MeshBuffer> in, w, out; uint32_t out_tile; };
        auto build_mm = [&](const std::vector<MM>& jobs, uint32_t base, uint32_t cnt, uint32_t cdep,
                            const char* kern, const std::shared_ptr<distributed::MeshBuffer>& inproto,
                            const std::shared_ptr<distributed::MeshBuffer>& wproto, uint32_t q, uint32_t li) {
            Program p = CreateProgram(); auto crs = crs_of(cnt);
            mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, crs, CBIndex::c_1, tt::DataFormat::Int8, cdep, I8);
            mkcb(p, crs, CBIndex::c_2, tt::DataFormat::Int8, NB, I8); mkcb(p, crs, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);
            mkcb(p, crs, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*inproto).append_to(rc); TensorAccessorArgs(*wproto).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + std::string(kern), crs,
                ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            for (uint32_t i=0;i<cnt;++i){ auto& j=jobs[base+i]; auto c=CC(i);
                SetRuntimeArgs(p, rd, c, {(uint32_t)j.in->address(), (uint32_t)j.w->address()});
                SetRuntimeArgs(p, wr, c, {(uint32_t)j.out->address(), 1u, j.out_tile});
                SetRuntimeArgs(p, cp, c, {q, ctx[li].q_inv_32(), (uint32_t)(QP[li]&0xFFFF), (uint32_t)(QP[li]>>16), ctx[li].to_montgomery(16384u)}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        auto build_modadd = [&](const std::vector<std::shared_ptr<distributed::MeshBuffer>>& ins,
                                const std::vector<std::shared_ptr<distributed::MeshBuffer>>& outs, uint32_t q) {
            Program p = CreateProgram(); auto crs = crs_of(NBLK);
            mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, NBLK, I32); mkcb(p, crs, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp", crs,
                ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            for (uint32_t i=0;i<NBLK;++i){ auto c=CC(i);
                SetRuntimeArgs(p, rd, c, {(uint32_t)ins[i]->address(), NBLK, 0u});
                SetRuntimeArgs(p, wr, c, {(uint32_t)outs[i]->address(), 1u, 0u});
                SetRuntimeArgs(p, cp, c, {NBLK, q}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        auto build_twiddle = [&](const std::vector<std::shared_ptr<distributed::MeshBuffer>>& a,
                                 const std::vector<std::shared_ptr<distributed::MeshBuffer>>& b,
                                 const std::vector<std::shared_ptr<distributed::MeshBuffer>>& o, uint32_t q, uint32_t li) {
            Program p = CreateProgram(); auto crs = crs_of(NBLK);
            mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, crs, CBIndex::c_1, tt::DataFormat::UInt32, 2, I32);
            mkcb(p, crs, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp", crs,
                ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            for (uint32_t i=0;i<NBLK;++i){ auto c=CC(i);
                SetRuntimeArgs(p, rd, c, {(uint32_t)a[i]->address(), (uint32_t)b[i]->address(), 1u, 0u});
                SetRuntimeArgs(p, wr, c, {(uint32_t)o[i]->address(), 1u, 0u});
                SetRuntimeArgs(p, cp, c, {1u, q, ctx[li].q_inv_32(), (uint32_t)(QP[li]&0xFFFF), (uint32_t)(QP[li]>>16)}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // ---------- FORWARD transform: X = NTT(x), x/X natural order (8192) ----------
        auto fwd8192 = [&](const std::vector<uint32_t>& x, uint32_t li) -> std::vector<uint32_t> {
            uint64_t q = QP[li]; auto& C = ctx[li];
            uint64_t g = prim_root(q), w = modpow(g,(q-1)/N,q), psi = modpow(g,(q-1)/(2*N),q);
            uint64_t w256 = modpow(w,N1,q), w32 = modpow(w,N2,q);
            // constants
            std::vector<uint32_t> WA(N2*N2), Tw(N2*N1), WCt(N1*N1);
            for (uint32_t k2=0;k2<N2;++k2) for (uint32_t n2=0;n2<N2;++n2)
                WA[k2*N2+n2]=(uint32_t)((u128)modpow(w256,(uint64_t)k2*n2,q)*modpow(psi,32ull*n2,q)%q);
            for (uint32_t k2=0;k2<N2;++k2) for (uint32_t n1=0;n1<N1;++n1)
                Tw[k2*N1+n1]=(uint32_t)((u128)modpow(psi,n1,q)*modpow(w,(uint64_t)n1*k2,q)%q);
            for (uint32_t n1=0;n1<N1;++n1) for (uint32_t k1=0;k1<N1;++k1)
                WCt[n1*N1+k1]=(uint32_t)modpow(w32,(uint64_t)k1*n1,q);
            // fold + write constants
            for (uint32_t bk2=0;bk2<NBLK;++bk2) for (uint32_t bn2=0;bn2<NBLK;++bn2){
                std::vector<uint32_t> blk(ELEMS);
                for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b) blk[a*T+b]=WA[(bk2*T+a)*N2+(bn2*T+b)];
                std::vector<int8_t> fold(NB*NB*ELEMS);
                for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<int8_t> l(ELEMS);
                    for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[r]*blk[i])%q),p);
                    auto tl=tilize32(l); std::copy(tl.begin(),tl.end(),fold.begin()+(p*NB+r)*ELEMS); }
                distributed::EnqueueWriteMeshBuffer(cq, bigW[bk2*NBLK+bn2], fold, false); }
            { std::vector<int8_t> fold(NB*NB*ELEMS);
              for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<int8_t> l(ELEMS);
                  for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[p]*WCt[i])%q),r);
                  auto tl=tilize32(l); std::copy(tl.begin(),tl.end(),fold.begin()+(p*NB+r)*ELEMS); }
              distributed::EnqueueWriteMeshBuffer(cq, smallW, fold, false); }
            for (uint32_t bn2=0;bn2<NBLK;++bn2){ std::vector<uint32_t> blk(ELEMS);
                for (uint32_t b=0;b<T;++b) for (uint32_t n1=0;n1<N1;++n1) blk[b*T+n1]=x[n1+32*(bn2*T+b)];
                distributed::EnqueueWriteMeshBuffer(cq, in8[bn2], tilize32(blk), false); }
            for (uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> blk(ELEMS);
                for (uint32_t a=0;a<T;++a) for (uint32_t n1=0;n1<N1;++n1) blk[a*T+n1]=C.to_montgomery(Tw[(bk2*T+a)*N1+n1]);
                distributed::EnqueueWriteMeshBuffer(cq, tw8[bk2], tilize32(blk), false); }
            // chain
            std::vector<MM> s1;
            for (uint32_t bk2=0;bk2<NBLK;++bk2) for (uint32_t bn2=0;bn2<NBLK;++bn2)
                s1.push_back({in8[bn2], bigW[bk2*NBLK+bn2], part8[bk2], bn2});
            std::vector<distributed::MeshWorkload> wls;
            for (uint32_t base=0;base<s1.size();base+=NCORES)
                wls.push_back(build_mm(s1, base, std::min<uint32_t>(NCORES,s1.size()-base), 2,
                                       "compute/fused_matmul_a.cpp", in8[0], bigW[0], (uint32_t)q, li));
            wls.push_back(build_modadd(part8, mid8, (uint32_t)q));
            wls.push_back(build_twiddle(mid8, tw8, midB8, (uint32_t)q, li));
            std::vector<MM> s2; for (uint32_t bk2=0;bk2<NBLK;++bk2) s2.push_back({midB8[bk2], smallW, out8[bk2], 0});
            wls.push_back(build_mm(s2, 0, NBLK, NB*NB, "compute/fused_matmul_c.cpp", midB8[0], smallW, (uint32_t)q, li));
            for (auto& wl : wls) distributed::EnqueueMeshWorkload(cq, wl, false);
            distributed::Finish(cq);
            std::vector<uint32_t> X(N);
            for (uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> raw; distributed::EnqueueReadMeshBuffer(cq, raw, out8[bk2], true);
                auto blk=untilize32(raw); for (uint32_t k2l=0;k2l<T;++k2l) for (uint32_t k1=0;k1<N1;++k1) X[(bk2*T+k2l)+N2*k1]=blk[k2l*T+k1]; }
            return X; };

        // ---------- INVERSE transform: x = INTT(X) ----------
        auto inv8192 = [&](const std::vector<uint32_t>& X, uint32_t li) -> std::vector<uint32_t> {
            uint64_t q = QP[li]; auto& C = ctx[li];
            uint64_t g = prim_root(q), wf = modpow(g,(q-1)/N,q);
            uint64_t w = mont::modular_inverse(wf,q), w256 = mont::modular_inverse(modpow(wf,N1,q),q), w32 = mont::modular_inverse(modpow(wf,N2,q),q);
            uint64_t psi = modpow(g,(q-1)/(2*N),q), psinv = mont::modular_inverse(psi,q), Ninv = mont::modular_inverse(N,q);
            std::vector<uint32_t> Winv32(N1*N1), Tw(N1*N2), Winv256(N2*N2);
            for (uint32_t n1=0;n1<N1;++n1) for (uint32_t k1=0;k1<N1;++k1)
                Winv32[n1*N1+k1]=(uint32_t)((u128)modpow(w32,(uint64_t)n1*k1,q)*Ninv%q);
            for (uint32_t n1=0;n1<N1;++n1) for (uint32_t k2=0;k2<N2;++k2)
                Tw[n1*N2+k2]=(uint32_t)((u128)modpow(w,(uint64_t)n1*k2,q)*modpow(psinv,n1,q)%q);
            for (uint32_t k2=0;k2<N2;++k2) for (uint32_t n2=0;n2<N2;++n2)
                Winv256[k2*N2+n2]=(uint32_t)((u128)modpow(w256,(uint64_t)k2*n2,q)*modpow(psinv,32ull*n2,q)%q);
            { std::vector<int8_t> fold(NB*NB*ELEMS);
              for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<int8_t> l(ELEMS);
                  for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[r]*Winv32[i])%q),p);
                  auto tl=tilize32(l); std::copy(tl.begin(),tl.end(),fold.begin()+(p*NB+r)*ELEMS); }
              distributed::EnqueueWriteMeshBuffer(cq, smallW, fold, false); }
            for (uint32_t bk2=0;bk2<NBLK;++bk2) for (uint32_t bn2=0;bn2<NBLK;++bn2){
                std::vector<uint32_t> blk(ELEMS);
                for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b) blk[a*T+b]=Winv256[(bk2*T+a)*N2+(bn2*T+b)];
                std::vector<int8_t> fold(NB*NB*ELEMS);
                for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<int8_t> l(ELEMS);
                    for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[p]*blk[i])%q),r);
                    auto tl=tilize32(l); std::copy(tl.begin(),tl.end(),fold.begin()+(p*NB+r)*ELEMS); }
                distributed::EnqueueWriteMeshBuffer(cq, bigW[bk2*NBLK+bn2], fold, false); }
            for (uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> blk(ELEMS);
                for (uint32_t k1=0;k1<N1;++k1) for (uint32_t k2l=0;k2l<T;++k2l) blk[k1*T+k2l]=X[(bk2*T+k2l)+N2*k1];
                distributed::EnqueueWriteMeshBuffer(cq, in8[bk2], tilize32(blk), false); }
            for (uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> blk(ELEMS);
                for (uint32_t n1=0;n1<N1;++n1) for (uint32_t k2l=0;k2l<T;++k2l) blk[n1*T+k2l]=C.to_montgomery(Tw[n1*N2+(bk2*T+k2l)]);
                distributed::EnqueueWriteMeshBuffer(cq, tw8[bk2], tilize32(blk), false); }
            std::vector<distributed::MeshWorkload> wls;
            std::vector<MM> s1; for (uint32_t bk2=0;bk2<NBLK;++bk2) s1.push_back({in8[bk2], smallW, mid8[bk2], 0});
            wls.push_back(build_mm(s1, 0, NBLK, NB*NB, "compute/fused_matmul_a.cpp", in8[0], smallW, (uint32_t)q, li));
            wls.push_back(build_twiddle(mid8, tw8, midB8, (uint32_t)q, li));
            std::vector<MM> s2;
            for (uint32_t bn2=0;bn2<NBLK;++bn2) for (uint32_t bk2=0;bk2<NBLK;++bk2)
                s2.push_back({midB8[bk2], bigW[bk2*NBLK+bn2], part8[bn2], bk2});
            for (uint32_t base=0;base<s2.size();base+=NCORES)
                wls.push_back(build_mm(s2, base, std::min<uint32_t>(NCORES,s2.size()-base), NB*NB,
                                       "compute/fused_matmul_c.cpp", midB8[0], bigW[0], (uint32_t)q, li));
            wls.push_back(build_modadd(part8, out8, (uint32_t)q));
            for (auto& wl : wls) distributed::EnqueueMeshWorkload(cq, wl, false);
            distributed::Finish(cq);
            std::vector<uint32_t> xo(N);
            for (uint32_t bn2=0;bn2<NBLK;++bn2){ std::vector<uint32_t> raw; distributed::EnqueueReadMeshBuffer(cq, raw, out8[bn2], true);
                auto blk=untilize32(raw); for (uint32_t n1=0;n1<N1;++n1) for (uint32_t n2l=0;n2l<T;++n2l) xo[n1+32*(bn2*T+n2l)]=blk[n1*T+n2l]; }
            return xo; };

        // ================= extra scratch buffers for pointwise + FBC =================
        constexpr uint32_t FBCW = 8;   // FBC width (L=8 keyswitch: mod-up 4->8, mod-down 4->8; n|8 valid)
        auto pa = mkbuf(NBLK, I32), pb = mkbuf(NBLK, I32), po = mkbuf(NBLK, I32);
        std::vector<std::shared_ptr<distributed::MeshBuffer>> fin(NBLK), fout(NBLK);
        for (uint32_t i = 0; i < NBLK; ++i) { fin[i] = mkbuf(FBCW, I32); fout[i] = mkbuf(FBCW, I32); }

        // ---- pointwise8192: out = a ⊙ b (mod q), 8 tiles on 8 cores (b Montgomery-encoded host-side) ----
        auto pointwise8192 = [&](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, uint32_t li) {
            uint64_t q = QP[li]; auto& C = ctx[li];
            std::vector<uint32_t> at(NBLK*ELEMS), bt(NBLK*ELEMS);
            for (uint32_t t = 0; t < NBLK; ++t) {
                std::vector<uint32_t> as(ELEMS), bs(ELEMS);
                for (uint32_t i = 0; i < ELEMS; ++i) { as[i] = a[t*ELEMS+i]; bs[i] = C.to_montgomery(b[t*ELEMS+i]); }
                auto ta = tilize32(as), tb = tilize32(bs);
                std::copy(ta.begin(), ta.end(), at.begin()+t*ELEMS); std::copy(tb.begin(), tb.end(), bt.begin()+t*ELEMS);
            }
            distributed::EnqueueWriteMeshBuffer(cq, pa, at, false); distributed::EnqueueWriteMeshBuffer(cq, pb, bt, false);
            Program p = CreateProgram(); auto crs = crs_of(NBLK);
            mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, crs, CBIndex::c_1, tt::DataFormat::UInt32, 2, I32);
            mkcb(p, crs, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_full.cpp", crs,
                ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            for (uint32_t t = 0; t < NBLK; ++t) { auto c = CC(t);
                SetRuntimeArgs(p, rd, c, {(uint32_t)pa->address(), (uint32_t)pb->address(), 1u, t});
                SetRuntimeArgs(p, wr, c, {(uint32_t)po->address(), 1u, t});
                SetRuntimeArgs(p, cp, c, {1u, (uint32_t)q, C.q_inv_32(), (uint32_t)(q&0xFFFF), (uint32_t)(q>>16)}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false);
            distributed::Finish(cq);
            std::vector<uint32_t> raw; distributed::EnqueueReadMeshBuffer(cq, raw, po, true);
            std::vector<uint32_t> out(N);
            for (uint32_t t = 0; t < NBLK; ++t) { auto u = untilize32(std::vector<uint32_t>(raw.begin()+t*ELEMS, raw.begin()+(t+1)*ELEMS));
                std::copy(u.begin(), u.end(), out.begin()+t*ELEMS); }
            return out; };

        // ---- fbc8192: approximate CRNS-combine FBC (src_idx -> dst_idx primes), 8 tiles on 8 cores ----
        auto fbc8192 = [&](const std::vector<std::vector<uint32_t>>& xsrc,
                           const std::vector<uint32_t>& srcIdx, const std::vector<uint32_t>& dstIdx) {
            uint32_t Ls = srcIdx.size(), Ld = dstIdx.size();
            u128 S = 1; for (uint32_t i=0;i<Ls;++i) S *= QP[srcIdx[i]];
            std::vector<uint32_t> C1(FBCW,0), Sm(FBCW), SmI(FBCW), Dm(FBCW), DmI(FBCW);
            std::vector<std::vector<uint32_t>> C2(FBCW, std::vector<uint32_t>(FBCW,0));
            for (uint32_t i=0;i<FBCW;++i){ uint32_t si=(i<Ls)?srcIdx[i]:srcIdx[0]; uint64_t qi=QP[si];
                Sm[i]=(uint32_t)qi; SmI[i]=ctx[si].q_inv_32();
                if (i<Ls) C1[i]=ctx[si].to_montgomery((uint32_t)modpow((uint64_t)((S/qi)%qi), qi-2, qi)); }
            for (uint32_t j=0;j<FBCW;++j){ uint32_t dj=(j<Ld)?dstIdx[j]:dstIdx[0]; uint64_t pj=QP[dj];
                Dm[j]=(uint32_t)pj; DmI[j]=ctx[dj].q_inv_32();
                if (j<Ld) for (uint32_t i=0;i<Ls;++i) C2[i][j]=ctx[dj].to_montgomery((uint32_t)((S/QP[srcIdx[i]])%pj)); }
            std::vector<uint32_t> ca; ca.push_back(FBCW);
            for (uint32_t i=0;i<FBCW;++i) ca.push_back(C1[i]);
            for (uint32_t i=0;i<FBCW;++i) for (uint32_t j=0;j<FBCW;++j) ca.push_back(C2[i][j]);
            for (uint32_t i=0;i<FBCW;++i) ca.push_back(Sm[i]);
            for (uint32_t i=0;i<FBCW;++i) ca.push_back(SmI[i]);
            for (uint32_t j=0;j<FBCW;++j) ca.push_back(Dm[j]);
            for (uint32_t j=0;j<FBCW;++j) ca.push_back(DmI[j]);
            for (uint32_t t=0;t<NBLK;++t){ std::vector<uint32_t> xb(FBCW*ELEMS,0);
                for (uint32_t i=0;i<Ls;++i){ std::vector<uint32_t> sl(xsrc[i].begin()+t*ELEMS, xsrc[i].begin()+(t+1)*ELEMS);
                    auto tl=tilize32(sl); std::copy(tl.begin(),tl.end(),xb.begin()+i*ELEMS); }
                distributed::EnqueueWriteMeshBuffer(cq, fin[t], xb, false); }
            Program p = CreateProgram(); auto crs = crs_of(NBLK);
            mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, 8, I32); mkcb(p, crs, CBIndex::c_2, tt::DataFormat::UInt32, 8, I32);
            mkcb(p, crs, CBIndex::c_3, tt::DataFormat::UInt32, 8, I32); mkcb(p, crs, CBIndex::c_16, tt::DataFormat::UInt32, 8, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/baseconv_fbc.cpp", crs,
                ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            for (uint32_t t=0;t<NBLK;++t){ auto c=CC(t);
                SetRuntimeArgs(p, rd, c, {(uint32_t)fin[t]->address(), FBCW, 0u});
                SetRuntimeArgs(p, wr, c, {(uint32_t)fout[t]->address(), FBCW, 0u});
                SetRuntimeArgs(p, cp, c, ca); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false);
            distributed::Finish(cq);
            std::vector<std::vector<uint32_t>> out(Ld, std::vector<uint32_t>(N));
            for (uint32_t t=0;t<NBLK;++t){ std::vector<uint32_t> raw; distributed::EnqueueReadMeshBuffer(cq, raw, fout[t], true);
                for (uint32_t j=0;j<Ld;++j){ auto u=untilize32(std::vector<uint32_t>(raw.begin()+j*ELEMS, raw.begin()+(j+1)*ELEMS));
                    std::copy(u.begin(), u.end(), out[j].begin()+t*ELEMS); } }
            return out; };

        // ================= workload data + evk (placeholder; latency is evk-independent) =================
        // Digit groups for dnum=2 (alpha=4): g0={0,1,2,3}, g1={4,5,6,7}; complement = other QP limbs.
        constexpr uint32_t DNUM=2, ALPHA=LQ/DNUM;
        std::vector<std::vector<uint32_t>> group(DNUM), comp(DNUM);
        for (uint32_t j=0;j<DNUM;++j){ for (uint32_t k=0;k<ALPHA;++k) group[j].push_back(j*ALPHA+k);
            for (uint32_t li=0;li<LQP;++li){ bool own=false; for (auto gg:group[j]) if(gg==li) own=true; if(!own) comp[j].push_back(li);} }
        u128 Pprod=1; for (uint32_t i=0;i<LP;++i) Pprod*=QP[LQ+i];

        std::mt19937_64 rng(seed);
        auto randpoly=[&](uint64_t q){ std::vector<uint32_t> v(N); for (auto& x:v) x=(uint32_t)(rng()%q); return v; };
        std::vector<std::vector<uint32_t>> c0(LQ),c1(LQ),d0(LQ),d1(LQ);
        for (uint32_t i=0;i<LQ;++i){ c0[i]=randpoly(QP[i]); c1[i]=randpoly(QP[i]); d0[i]=randpoly(QP[i]); d1[i]=randpoly(QP[i]); }
        // evk (coeff, per digit per QP limb) — placeholder random; CPU ref + device use the SAME evk.
        std::vector<std::vector<std::vector<uint32_t>>> evk0(DNUM, std::vector<std::vector<uint32_t>>(LQP)),
                                                        evk1(DNUM, std::vector<std::vector<uint32_t>>(LQP));
        for (uint32_t j=0;j<DNUM;++j) for (uint32_t li=0;li<LQP;++li){ evk0[j][li]=randpoly(QP[li]); evk1[j][li]=randpoly(QP[li]); }

        // ---- CPU helpers ----
        auto negmul=[&](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, uint64_t q){
            std::vector<uint32_t> c(N,0);
            for (uint32_t i=0;i<N;++i){ if(!a[i]) continue; u128 ai=a[i];
                for (uint32_t jj=0;jj<N;++jj){ uint32_t k=i+jj; uint64_t v=(uint64_t)(ai*b[jj]%q);
                    if(k<N) c[k]=(uint32_t)(((u128)c[k]+v)%q); else c[k-N]=(uint32_t)(((u128)c[k-N]+q-v)%q); } }
            return c; };
        auto fbc_approx=[&](const std::vector<std::vector<uint32_t>>& x, const std::vector<uint32_t>& srcIdx, const std::vector<uint32_t>& dstIdx){
            uint32_t Ls=srcIdx.size(), Ld=dstIdx.size(); u128 S=1; for (uint32_t i=0;i<Ls;++i) S*=QP[srcIdx[i]];
            std::vector<std::vector<uint32_t>> out(Ld, std::vector<uint32_t>(N));
            for (uint32_t n=0;n<N;++n){ std::vector<uint64_t> y(Ls);
                for (uint32_t i=0;i<Ls;++i){ uint64_t si=QP[srcIdx[i]]; y[i]=(uint64_t)(((u128)x[i][n]*modpow((uint64_t)((S/si)%si),si-2,si))%si); }
                for (uint32_t j=0;j<Ld;++j){ uint64_t pj=QP[dstIdx[j]]; u128 acc=0; for (uint32_t i=0;i<Ls;++i) acc+=(u128)y[i]*((S/QP[srcIdx[i]])%pj); out[j][n]=(uint32_t)(acc%pj); } }
            return out; };

        // ================= CPU oracle: tensor -> keyswitch(dnum=2) -> combine -> rescale =================
        auto cpu_hemul=[&](){
            std::vector<std::vector<uint32_t>> e0(LQ),e1(LQ),e2(LQ);
            for (uint32_t li=0;li<LQ;++li){ uint64_t q=QP[li];
                e0[li]=negmul(c0[li],d0[li],q); e2[li]=negmul(c1[li],d1[li],q);
                auto t01=negmul(c0[li],d1[li],q), t10=negmul(c1[li],d0[li],q);
                e1[li].resize(N); for (uint32_t n=0;n<N;++n) e1[li][n]=(uint32_t)(((u128)t01[n]+t10[n])%q); }
            // keyswitch e2
            std::vector<std::vector<uint32_t>> acc0(LQP, std::vector<uint32_t>(N,0)), acc1(LQP, std::vector<uint32_t>(N,0));
            for (uint32_t j=0;j<DNUM;++j){ std::vector<std::vector<uint32_t>> vj(LQP);
                for (auto i:group[j]) vj[i]=e2[i];
                std::vector<std::vector<uint32_t>> xs; for (auto i:group[j]) xs.push_back(e2[i]);
                auto o=fbc_approx(xs, group[j], comp[j]); for (uint32_t k=0;k<comp[j].size();++k) vj[comp[j][k]]=o[k];
                for (uint32_t li=0;li<LQP;++li){ uint64_t q=QP[li];
                    auto p0=negmul(vj[li],evk0[j][li],q), p1=negmul(vj[li],evk1[j][li],q);
                    for (uint32_t n=0;n<N;++n){ acc0[li][n]=(uint32_t)(((u128)acc0[li][n]+p0[n])%q); acc1[li][n]=(uint32_t)(((u128)acc1[li][n]+p1[n])%q);} } }
            auto moddown=[&](std::vector<std::vector<uint32_t>>& acc){ std::vector<std::vector<uint32_t>> xP; std::vector<uint32_t> sI,dI;
                for (uint32_t i=0;i<LP;++i){ xP.push_back(acc[LQ+i]); sI.push_back(LQ+i);} for (uint32_t j=0;j<LQ;++j) dI.push_back(j);
                auto f=fbc_approx(xP,sI,dI); std::vector<std::vector<uint32_t>> out(LQ, std::vector<uint32_t>(N));
                for (uint32_t j=0;j<LQ;++j){ uint64_t q=QP[j]; uint64_t Pinv=modpow((uint64_t)(Pprod%q),q-2,q);
                    for (uint32_t n=0;n<N;++n) out[j][n]=(uint32_t)((((u128)acc[j][n]+q-f[j][n])%q)*Pinv%q); } return out; };
            auto k0=moddown(acc0), k1=moddown(acc1);
            // combine + rescale (drop limb LQ-1)
            uint64_t qL=QP[LQ-1];
            std::vector<std::vector<uint32_t>> R0(LQ-1), R1(LQ-1);
            std::vector<uint32_t> r07(N), r17(N);
            for (uint32_t n=0;n<N;++n){ r07[n]=(uint32_t)(((u128)e0[LQ-1][n]+k0[LQ-1][n])%qL); r17[n]=(uint32_t)(((u128)e1[LQ-1][n]+k1[LQ-1][n])%qL); }
            for (uint32_t j=0;j<LQ-1;++j){ uint64_t q=QP[j]; uint64_t qLinv=modpow(qL%q,q-2,q);
                R0[j].resize(N); R1[j].resize(N);
                for (uint32_t n=0;n<N;++n){ uint32_t a0=(uint32_t)(((u128)e0[j][n]+k0[j][n])%q), a1=(uint32_t)(((u128)e1[j][n]+k1[j][n])%q);
                    R0[j][n]=(uint32_t)((((u128)a0+q-(r07[n]%q))%q)*qLinv%q); R1[j][n]=(uint32_t)((((u128)a1+q-(r17[n]%q))%q)*qLinv%q); } }
            std::vector<std::vector<uint32_t>> out; for (auto&v:R0) out.push_back(v); for (auto&v:R1) out.push_back(v); return out; };

        // ================= DEVICE HE-mul (same graph via transform8192 / pointwise8192 / fbc8192) =======
        // evk in freq domain (device NTT of the coeff evk), reused across timed iters.
        std::vector<std::vector<std::vector<uint32_t>>> evk0f(DNUM, std::vector<std::vector<uint32_t>>(LQP)),
                                                        evk1f(DNUM, std::vector<std::vector<uint32_t>>(LQP));
        for (uint32_t j=0;j<DNUM;++j) for (uint32_t li=0;li<LQP;++li){ evk0f[j][li]=fwd8192(evk0[j][li],li); evk1f[j][li]=fwd8192(evk1[j][li],li); }

        auto dev_hemul=[&](){
            std::vector<std::vector<uint32_t>> e0(LQ),e1(LQ),e2(LQ);
            for (uint32_t li=0;li<LQ;++li){ uint64_t q=QP[li];
                auto c0f=fwd8192(c0[li],li), c1f=fwd8192(c1[li],li), d0f=fwd8192(d0[li],li), d1f=fwd8192(d1[li],li);
                auto e0f=pointwise8192(c0f,d0f,li), e2f=pointwise8192(c1f,d1f,li);
                auto t01=pointwise8192(c0f,d1f,li), t10=pointwise8192(c1f,d0f,li);
                std::vector<uint32_t> e1f(N); for (uint32_t n=0;n<N;++n) e1f[n]=(uint32_t)(((u128)t01[n]+t10[n])%q);
                e0[li]=inv8192(e0f,li); e1[li]=inv8192(e1f,li); e2[li]=inv8192(e2f,li); }
            // keyswitch e2
            std::vector<std::vector<uint32_t>> acc0(LQP), acc1(LQP);
            std::vector<std::vector<std::vector<uint32_t>>> vj(DNUM, std::vector<std::vector<uint32_t>>(LQP));
            for (uint32_t j=0;j<DNUM;++j){ for (auto i:group[j]) vj[j][i]=e2[i];
                std::vector<std::vector<uint32_t>> xs; for (auto i:group[j]) xs.push_back(e2[i]);
                auto o=fbc8192(xs, group[j], comp[j]); for (uint32_t k=0;k<comp[j].size();++k) vj[j][comp[j][k]]=o[k]; }
            for (uint32_t li=0;li<LQP;++li){ uint64_t q=QP[li]; std::vector<uint32_t> f0(N,0),f1(N,0);
                for (uint32_t j=0;j<DNUM;++j){ auto vjf=fwd8192(vj[j][li],li);
                    auto p0=pointwise8192(vjf,evk0f[j][li],li), p1=pointwise8192(vjf,evk1f[j][li],li);
                    for (uint32_t n=0;n<N;++n){ f0[n]=(uint32_t)(((u128)f0[n]+p0[n])%q); f1[n]=(uint32_t)(((u128)f1[n]+p1[n])%q);} }
                acc0[li]=inv8192(f0,li); acc1[li]=inv8192(f1,li); }
            auto moddown=[&](std::vector<std::vector<uint32_t>>& acc){ std::vector<std::vector<uint32_t>> xP; std::vector<uint32_t> sI,dI;
                for (uint32_t i=0;i<LP;++i){ xP.push_back(acc[LQ+i]); sI.push_back(LQ+i);} for (uint32_t j=0;j<LQ;++j) dI.push_back(j);
                auto f=fbc8192(xP,sI,dI); std::vector<std::vector<uint32_t>> out(LQ, std::vector<uint32_t>(N));
                for (uint32_t j=0;j<LQ;++j){ uint64_t q=QP[j]; uint64_t Pinv=modpow((uint64_t)(Pprod%q),q-2,q);
                    for (uint32_t n=0;n<N;++n) out[j][n]=(uint32_t)((((u128)acc[j][n]+q-f[j][n])%q)*Pinv%q); } return out; };
            auto k0=moddown(acc0), k1=moddown(acc1);
            uint64_t qL=QP[LQ-1];
            std::vector<std::vector<uint32_t>> R0(LQ-1), R1(LQ-1); std::vector<uint32_t> r07(N),r17(N);
            for (uint32_t n=0;n<N;++n){ r07[n]=(uint32_t)(((u128)e0[LQ-1][n]+k0[LQ-1][n])%qL); r17[n]=(uint32_t)(((u128)e1[LQ-1][n]+k1[LQ-1][n])%qL); }
            for (uint32_t j=0;j<LQ-1;++j){ uint64_t q=QP[j]; uint64_t qLinv=modpow(qL%q,q-2,q); R0[j].resize(N); R1[j].resize(N);
                for (uint32_t n=0;n<N;++n){ uint32_t a0=(uint32_t)(((u128)e0[j][n]+k0[j][n])%q), a1=(uint32_t)(((u128)e1[j][n]+k1[j][n])%q);
                    R0[j][n]=(uint32_t)((((u128)a0+q-(r07[n]%q))%q)*qLinv%q); R1[j][n]=(uint32_t)((((u128)a1+q-(r17[n]%q))%q)*qLinv%q); } }
            std::vector<std::vector<uint32_t>> out; for (auto&v:R0) out.push_back(v); for (auto&v:R1) out.push_back(v); return out; };

        // ================= validate device vs CPU oracle (bit-exact) =================
        fmt::print("computing CPU oracle...\n");
        auto ref = cpu_hemul();
        fmt::print("running device HE-mul...\n");
        auto got = dev_hemul();
        uint32_t mism=0, total=0;
        for (uint32_t r=0;r<ref.size();++r) for (uint32_t n=0;n<N;++n){ ++total; if (got[r][n]!=ref[r][n]){ if(mism<8) fmt::print("  row{} n{} got={} exp={}\n",r,n,got[r][n],ref[r][n]); ++mism; } }
        bool ok = (mism==0);
        fmt::print("Set-B HE-mul  device-vs-oracle mismatches = {}/{}  [{}]\n", mism, total, ok?"PASS":"FAIL");

        // ================= timing =================
        int iters = 3; if (const char* e=std::getenv("HE_ITERS")){ int v=std::atoi(e); if(v>0) iters=v; }
        dev_hemul();   // warm
        auto t0=std::chrono::high_resolution_clock::now();
        for (int i=0;i<iters;++i) dev_hemul();
        auto t1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/1e6/iters;
        fmt::print("TIMING: Set-B HE-mul (N={}, L={}, dnum={})  {:.3f} ms/mul  ({} cores, {} iters)\n",
                   N, LQ, DNUM, ms, NCORES, iters);
        fmt::print("RESULT: Set-B HE-mul {} end-to-end (tensor+keyswitch+rescale, bit-exact vs oracle)\n", ok?"WORKS":"WRONG");

        pass = ok;
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
