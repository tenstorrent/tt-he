// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// CROSS-ALIGNED Set-B HE-multiply: eval-resident, tensor=pointwise (0 NTT), drop-prime rescale, dnum keyswitch.
// dnum=2 (alpha=4), P=4 -> QP=12.  docs/he/cross_comparison.md.
//
// Matches CROSS's HE-Mult sub-kernel DAG (only the NTT INTERNALS differ): ciphertexts kept in the
// eval/NTT domain, so the TENSOR is pure pointwise (0 NTT) exactly like CROSS's polynomial_mult;
// rescale is a real drop-prime done first (INTT-last-limb + NTT-correction, ~4 INTT + 4(L-1) NTT);
// key-switch is dnum digit-decomposed, eval-in/eval-out (own limbs reuse the eval e2, only the
// mod-up complement is NTT'd). Reuses transform8192 / pointwise8192 / fbc8192. Validated bit-exact
// vs a CPU oracle of the identical RNS arithmetic (evk placeholder-random; latency is evk-independent).

#include <chrono>
#include <utility>
#include <functional>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <set>
#include <map>
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

// ---- host-side phase profiling (idle-cause attribution) -----------------------------
namespace prof {
  double g_build=0, g_wr=0, g_rd=0, g_enq=0, g_fin=0, g_replay=0;   // ms accumulated per host phase
  long   n_build=0, n_wr=0, n_rd=0, n_enq=0, n_fin=0, n_replay=0;   // call counts
  using hrc = std::chrono::high_resolution_clock;
  inline double msd(hrc::time_point t){ return std::chrono::duration<double,std::milli>(hrc::now()-t).count(); }
  inline void reset(){ g_build=g_wr=g_rd=g_enq=g_fin=g_replay=0; n_build=n_wr=n_rd=n_enq=n_fin=n_replay=0; }
  std::function<void()> seg_hook;   // called at every host boundary (t_wr/t_rd) to break trace segments
  std::function<void()> drain_fn;   // periodic profiler drain (Finish + ReadMeshDeviceProfilerResults)
  long drain_every=0, drain_ctr=0;  // drain the device profiler buffer every N dispatches (0 = off)
}
namespace Dp = tt::tt_metal::distributed;
template<class...A> void t_wr (A&&...a){ if(prof::seg_hook) prof::seg_hook(); auto t=prof::hrc::now(); Dp::EnqueueWriteMeshBuffer(std::forward<A>(a)...); prof::g_wr +=prof::msd(t); ++prof::n_wr;  }
template<class...A> void t_rd (A&&...a){ if(prof::seg_hook) prof::seg_hook(); auto t=prof::hrc::now(); Dp::EnqueueReadMeshBuffer (std::forward<A>(a)...); prof::g_rd +=prof::msd(t); ++prof::n_rd;  }
template<class...A> void t_enq(A&&...a){ auto t=prof::hrc::now(); Dp::EnqueueMeshWorkload   (std::forward<A>(a)...); prof::g_enq+=prof::msd(t); ++prof::n_enq;
    if(prof::drain_every && (++prof::drain_ctr % prof::drain_every)==0 && prof::drain_fn) prof::drain_fn(); }   // drain profiler before it overflows
template<class Q>   void t_fin(Q&& cq) { if(prof::seg_hook) prof::seg_hook();   /* close open trace before Finish (event-sync forbidden in capture) */
                                        auto t=prof::hrc::now(); Dp::Finish(cq);                                    prof::g_fin+=prof::msd(t); ++prof::n_fin; }

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

        auto mesh = distributed::MeshDevice::create_unit_mesh(
            0, DEFAULT_L1_SMALL_SIZE, /*trace_region_size=*/256u*1024u*1024u);   // reserve DRAM for Metal Trace
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX = grid.x, GY = grid.y;
        uint32_t want = GX * GY; if (const char* e = std::getenv("NTT_CORES")) { int v = std::atoi(e); if (v > 0) want = v; }
        const uint32_t NCORES = std::min<uint32_t>(want, GX * GY);
        const uint32_t EWC = std::min<uint32_t>(NBLK, NCORES);   // cores for the 8-tile-block ops (2 blocks/core when <8)
        auto CC = [&](uint32_t i){ return CoreCoord{i % GX, i / GX}; };
        fmt::print("Set-B HE-mul  N={} LQ={} LP={} QP={}  grid {}x{}={} cores\n", N, LQ, LP, LQP, GX, GY, NCORES);

        auto mkbuf = [&](uint32_t nt, uint32_t page) {
            distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * page}, d, mesh.get()); };
        auto mkcb = [&](Program& p, const CoreRangeSet& crs, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
            CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, crs, c); };
        auto crs_of = [&](uint32_t cnt){ std::set<CoreRange> s; for (uint32_t i=0;i<cnt;++i) s.insert(CoreRange(CC(i))); return CoreRangeSet(s); };

        // ===== NOC_BENCH: DRAM round-trip vs NoC core-to-core all-to-all, for a G^2-tile reshuffle =====
        if (std::getenv("NOC_BENCH")) {
            uint32_t G=8; if(const char*e=std::getenv("NOC_G")){int v=std::atoi(e); if(v>0)G=v;}
            uint32_t R=200; if(const char*e=std::getenv("NOC_R")){int v=std::atoi(e); if(v>0)R=v;}
            auto crsG = crs_of(G);
            double tile_bytes = 32.0*32.0*4.0;                       // one int32 tile = 4096 B
            double a2a_bytes = (double)G*G*tile_bytes;               // one reshuffle moves G^2 tiles
            // ---- (1) DRAM round-trip ----
            auto dbuf = mkbuf(G*G, I32);
            { Program p=CreateProgram();
              mkcb(p,crsG,CBIndex::c_0,tt::DataFormat::UInt32,G,I32); mkcb(p,crsG,CBIndex::c_1,tt::DataFormat::UInt32,G,I32);
              auto k=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/bench_dram_rt.cpp",crsG,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              for(uint32_t i=0;i<G;++i) SetRuntimeArgs(p,k,CC(i),{i,G,R,(uint32_t)dbuf->address()});
              distributed::MeshWorkload wl; wl.add_program(dr,std::move(p));
              distributed::EnqueueMeshWorkload(cq,wl,false); distributed::Finish(cq);   // warm
              auto t0=std::chrono::high_resolution_clock::now();
              distributed::EnqueueMeshWorkload(cq,wl,false); distributed::Finish(cq);
              auto t1=std::chrono::high_resolution_clock::now();
              double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
              double per=ms/R;   // ms per reshuffle (write G^2 + read G^2)
              fmt::print("NOC_BENCH DRAM round-trip: {:.4f} ms/reshuffle  ({:.1f} GB/s incl. write+read)\n",
                         per, (2.0*a2a_bytes)/(per*1e-3)/1e9); }
            // ---- (2) NoC core-to-core all-to-all ----
            { Program p=CreateProgram();
              mkcb(p,crsG,CBIndex::c_0,tt::DataFormat::UInt32,G,I32); mkcb(p,crsG,CBIndex::c_1,tt::DataFormat::UInt32,G,I32);
              auto k=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/bench_noc_a2a.cpp",crsG,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              std::vector<uint32_t> coords; for(uint32_t j=0;j<G;++j){ auto w=mesh->worker_core_from_logical_core(CC(j)); coords.push_back((uint32_t)w.x); coords.push_back((uint32_t)w.y); }
              for(uint32_t i=0;i<G;++i){ std::vector<uint32_t> a={i,G,R}; for(auto c:coords) a.push_back(c); SetRuntimeArgs(p,k,CC(i),a); }
              distributed::MeshWorkload wl; wl.add_program(dr,std::move(p));
              distributed::EnqueueMeshWorkload(cq,wl,false); distributed::Finish(cq);   // warm
              auto t0=std::chrono::high_resolution_clock::now();
              distributed::EnqueueMeshWorkload(cq,wl,false); distributed::Finish(cq);
              auto t1=std::chrono::high_resolution_clock::now();
              double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
              double per=ms/R;   // ms per reshuffle (G^2 one-way L1->L1)
              fmt::print("NOC_BENCH NoC all-to-all : {:.4f} ms/reshuffle  ({:.1f} GB/s one-way)\n",
                         per, a2a_bytes/(per*1e-3)/1e9); }
            fmt::print("NOC_BENCH: G={} cores, R={} iters, {:.0f} KB/reshuffle (G^2 tiles)\n", G, R, a2a_bytes/1024.0);
            bool ok=true; ok&=mesh->close(); fmt::print("{}\n", ok?"Test Passed":"Test FAILED"); return ok?0:1;
        }

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
            auto __tb=prof::hrc::now(); Program p = CreateProgram(); auto crs = crs_of(cnt);
            mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, crs, CBIndex::c_1, tt::DataFormat::Int8, cdep, I8);
            mkcb(p, crs, CBIndex::c_2, tt::DataFormat::Int8, NB, I8); mkcb(p, crs, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);
            mkcb(p, crs, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*inproto).append_to(rc); TensorAccessorArgs(*wproto).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused_batch.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + std::string(kern), crs,
                ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            for (uint32_t i=0;i<cnt;++i){ auto& j=jobs[base+i]; auto c=CC(i);
                SetRuntimeArgs(p, rd, c, {(uint32_t)j.in->address(), (uint32_t)j.w->address()});
                SetRuntimeArgs(p, wr, c, {(uint32_t)j.out->address(), 1u, j.out_tile});
                SetRuntimeArgs(p, cp, c, {q, ctx[li].q_inv_32(), (uint32_t)(QP[li]&0xFFFF), (uint32_t)(QP[li]>>16), ctx[li].to_montgomery(16384u)}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); prof::g_build+=prof::msd(__tb); ++prof::n_build; return wl; };

        auto build_modadd = [&](const std::vector<std::shared_ptr<distributed::MeshBuffer>>& ins,
                                const std::vector<std::shared_ptr<distributed::MeshBuffer>>& outs, uint32_t q) {
            auto __tb=prof::hrc::now(); Program p = CreateProgram(); auto crs = crs_of(NBLK);
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
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); prof::g_build+=prof::msd(__tb); ++prof::n_build; return wl; };

        auto build_twiddle = [&](const std::vector<std::shared_ptr<distributed::MeshBuffer>>& a,
                                 const std::vector<std::shared_ptr<distributed::MeshBuffer>>& b,
                                 const std::vector<std::shared_ptr<distributed::MeshBuffer>>& o, uint32_t q, uint32_t li) {
            auto __tb=prof::hrc::now(); Program p = CreateProgram(); auto crs = crs_of(NBLK);
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
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); prof::g_build+=prof::msd(__tb); ++prof::n_build; return wl; };

        // ===== paramgen HOISTED: precompute every (prime,dir) folded twiddle/matrix set ONCE =====
        // (was regenerated on every transform call — the dominant ~50 ms host gap between kernels).
        struct CS { std::vector<std::vector<int8_t>> big{std::vector<std::vector<int8_t>>(NBLK*NBLK)};
                    std::vector<int8_t> small; std::vector<std::vector<uint32_t>> tw{std::vector<std::vector<uint32_t>>(NBLK)}; };
        std::map<int,CS> KC;                                    // key = li*2 + fwd
        auto gen_consts = [&](uint32_t li, int fwd) {
            uint64_t q=QP[li]; auto& C=ctx[li]; CS cs; uint64_t g=prim_root(q);
            if (fwd) {
                uint64_t w=modpow(g,(q-1)/N,q), psi=modpow(g,(q-1)/(2*N),q), w256=modpow(w,N1,q), w32=modpow(w,N2,q);
                std::vector<uint32_t> WA(N2*N2),Tw(N2*N1),WCt(N1*N1);
                for (uint32_t k2=0;k2<N2;++k2) for (uint32_t n2=0;n2<N2;++n2)
                    WA[k2*N2+n2]=(uint32_t)((u128)modpow(w256,(uint64_t)k2*n2,q)*modpow(psi,32ull*n2,q)%q);
                for (uint32_t k2=0;k2<N2;++k2) for (uint32_t n1=0;n1<N1;++n1)
                    Tw[k2*N1+n1]=(uint32_t)((u128)modpow(psi,n1,q)*modpow(w,(uint64_t)n1*k2,q)%q);
                for (uint32_t n1=0;n1<N1;++n1) for (uint32_t k1=0;k1<N1;++k1)
                    WCt[n1*N1+k1]=(uint32_t)modpow(w32,(uint64_t)k1*n1,q);
                for (uint32_t bk2=0;bk2<NBLK;++bk2) for (uint32_t bn2=0;bn2<NBLK;++bn2){
                    std::vector<uint32_t> blk(ELEMS);
                    for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b) blk[a*T+b]=WA[(bk2*T+a)*N2+(bn2*T+b)];
                    std::vector<int8_t> fold(NB*NB*ELEMS);
                    for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<int8_t> l(ELEMS);
                        for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[r]*blk[i])%q),p);
                        auto tl=tilize32(l); std::copy(tl.begin(),tl.end(),fold.begin()+(p*NB+r)*ELEMS); }
                    cs.big[bk2*NBLK+bn2]=std::move(fold); }
                { std::vector<int8_t> fold(NB*NB*ELEMS);
                  for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<int8_t> l(ELEMS);
                      for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[p]*WCt[i])%q),r);
                      auto tl=tilize32(l); std::copy(tl.begin(),tl.end(),fold.begin()+(p*NB+r)*ELEMS); }
                  cs.small=std::move(fold); }
                for (uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> blk(ELEMS);
                    for (uint32_t a=0;a<T;++a) for (uint32_t n1=0;n1<N1;++n1) blk[a*T+n1]=C.to_montgomery(Tw[(bk2*T+a)*N1+n1]);
                    cs.tw[bk2]=tilize32(blk); }
            } else {
                uint64_t wf=modpow(g,(q-1)/N,q);
                uint64_t w=mont::modular_inverse(wf,q), w256=mont::modular_inverse(modpow(wf,N1,q),q), w32=mont::modular_inverse(modpow(wf,N2,q),q);
                uint64_t psi=modpow(g,(q-1)/(2*N),q), psinv=mont::modular_inverse(psi,q), Ninv=mont::modular_inverse(N,q);
                std::vector<uint32_t> Winv32(N1*N1),Tw(N1*N2),Winv256(N2*N2);
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
                  cs.small=std::move(fold); }
                for (uint32_t bk2=0;bk2<NBLK;++bk2) for (uint32_t bn2=0;bn2<NBLK;++bn2){
                    std::vector<uint32_t> blk(ELEMS);
                    for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b) blk[a*T+b]=Winv256[(bk2*T+a)*N2+(bn2*T+b)];
                    std::vector<int8_t> fold(NB*NB*ELEMS);
                    for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<int8_t> l(ELEMS);
                        for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[p]*blk[i])%q),r);
                        auto tl=tilize32(l); std::copy(tl.begin(),tl.end(),fold.begin()+(p*NB+r)*ELEMS); }
                    cs.big[bk2*NBLK+bn2]=std::move(fold); }
                for (uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> blk(ELEMS);
                    for (uint32_t n1=0;n1<N1;++n1) for (uint32_t k2l=0;k2l<T;++k2l) blk[n1*T+k2l]=C.to_montgomery(Tw[n1*N2+(bk2*T+k2l)]);
                    cs.tw[bk2]=tilize32(blk); }
            }
            return cs; };
        for (uint32_t li=0; li<LQP; ++li) { KC[li*2+1]=gen_consts(li,1); KC[li*2+0]=gen_consts(li,0); }

        // ===== constant RESIDENCY: upload every (prime,dir) folded set to its own DRAM buffer ONCE =====
        // (was re-uploaded on every transform call). Transforms then reference these — no per-call const upload.
        std::map<int,std::vector<std::shared_ptr<distributed::MeshBuffer>>> bigWres, twres;
        std::map<int,std::shared_ptr<distributed::MeshBuffer>> smallWres;
        for (auto& kv : KC) { int key=kv.first; const CS& cs=kv.second;
            bigWres[key].resize(NBLK*NBLK); twres[key].resize(NBLK);
            for (uint32_t i=0;i<NBLK*NBLK;++i){ bigWres[key][i]=mkbuf(NB*NB,I8); t_wr(cq, bigWres[key][i], cs.big[i], false); }
            smallWres[key]=mkbuf(NB*NB,I8); t_wr(cq, smallWres[key], cs.small, false);
            for (uint32_t bk2=0;bk2<NBLK;++bk2){ twres[key][bk2]=mkbuf(1,I32); t_wr(cq, twres[key][bk2], cs.tw[bk2], false); } }
        t_fin(cq);

        // ---------- FORWARD transform: X = NTT(x), x/X natural order (8192) ----------
        auto fwd8192 = [&](const std::vector<uint32_t>& x, uint32_t li) -> std::vector<uint32_t> {
            uint64_t q = QP[li]; int key=li*2+1;
            auto& big=bigWres[key]; auto sm=smallWres[key]; auto& tw=twres[key];   // RESIDENT constants
            for (uint32_t bn2=0;bn2<NBLK;++bn2){ std::vector<uint32_t> blk(ELEMS);
                for (uint32_t b=0;b<T;++b) for (uint32_t n1=0;n1<N1;++n1) blk[b*T+n1]=x[n1+32*(bn2*T+b)];
                t_wr(cq, in8[bn2], tilize32(blk), false); }
            // chain
            std::vector<MM> s1;
            for (uint32_t bk2=0;bk2<NBLK;++bk2) for (uint32_t bn2=0;bn2<NBLK;++bn2)
                s1.push_back({in8[bn2], big[bk2*NBLK+bn2], part8[bk2], bn2});
            std::vector<distributed::MeshWorkload> wls;
            for (uint32_t base=0;base<s1.size();base+=NCORES)
                wls.push_back(build_mm(s1, base, std::min<uint32_t>(NCORES,s1.size()-base), NB*NB,
                                       "compute/fused_matmul_a.cpp", in8[0], big[0], (uint32_t)q, li));
            wls.push_back(build_modadd(part8, mid8, (uint32_t)q));
            wls.push_back(build_twiddle(mid8, tw, midB8, (uint32_t)q, li));
            std::vector<MM> s2; for (uint32_t bk2=0;bk2<NBLK;++bk2) s2.push_back({midB8[bk2], sm, out8[bk2], 0});
            wls.push_back(build_mm(s2, 0, NBLK, NB*NB, "compute/fused_matmul_c.cpp", midB8[0], sm, (uint32_t)q, li));
            for (auto& wl : wls) t_enq(cq, wl, false);
            t_fin(cq);
            std::vector<uint32_t> X(N);
            for (uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> raw; t_rd(cq, raw, out8[bk2], true);
                auto blk=untilize32(raw); for (uint32_t k2l=0;k2l<T;++k2l) for (uint32_t k1=0;k1<N1;++k1) X[(bk2*T+k2l)+N2*k1]=blk[k2l*T+k1]; }
            return X; };

        // ---------- INVERSE transform: x = INTT(X) ----------
        auto inv8192 = [&](const std::vector<uint32_t>& X, uint32_t li) -> std::vector<uint32_t> {
            uint64_t q = QP[li]; int key=li*2+0;
            auto& big=bigWres[key]; auto sm=smallWres[key]; auto& tw=twres[key];   // RESIDENT constants
            for (uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> blk(ELEMS);
                for (uint32_t k1=0;k1<N1;++k1) for (uint32_t k2l=0;k2l<T;++k2l) blk[k1*T+k2l]=X[(bk2*T+k2l)+N2*k1];
                t_wr(cq, in8[bk2], tilize32(blk), false); }
            std::vector<distributed::MeshWorkload> wls;
            std::vector<MM> s1; for (uint32_t bk2=0;bk2<NBLK;++bk2) s1.push_back({in8[bk2], sm, mid8[bk2], 0});
            wls.push_back(build_mm(s1, 0, NBLK, NB*NB, "compute/fused_matmul_a.cpp", in8[0], sm, (uint32_t)q, li));
            wls.push_back(build_twiddle(mid8, tw, midB8, (uint32_t)q, li));
            std::vector<MM> s2;
            for (uint32_t bn2=0;bn2<NBLK;++bn2) for (uint32_t bk2=0;bk2<NBLK;++bk2)
                s2.push_back({midB8[bk2], big[bk2*NBLK+bn2], part8[bn2], bk2});
            for (uint32_t base=0;base<s2.size();base+=NCORES)
                wls.push_back(build_mm(s2, base, std::min<uint32_t>(NCORES,s2.size()-base), NB*NB,
                                       "compute/fused_matmul_c.cpp", midB8[0], big[0], (uint32_t)q, li));
            wls.push_back(build_modadd(part8, out8, (uint32_t)q));
            for (auto& wl : wls) t_enq(cq, wl, false);
            t_fin(cq);
            std::vector<uint32_t> xo(N);
            for (uint32_t bn2=0;bn2<NBLK;++bn2){ std::vector<uint32_t> raw; t_rd(cq, raw, out8[bn2], true);
                auto blk=untilize32(raw); for (uint32_t n1=0;n1<N1;++n1) for (uint32_t n2l=0;n2l<T;++n2l) xo[n1+32*(bn2*T+n2l)]=blk[n1*T+n2l]; }
            return xo; };

        // ===== Step 3 brick: BUFFER-RESIDENT transform (in DRAM -> out DRAM, no host round-trip) =====
        std::vector<distributed::MeshWorkload> hold3; hold3.reserve(8192);   // keep workloads alive until Finish
        // ---- T2: persistent workload cache (build ONCE on mul#1, re-enqueue after) ----
        // emit(): CACHE off -> build+enqueue (original). CACHE on: REC -> build+cache; replay -> re-enqueue.
        bool CACHE_ON=false, CACHE_REC=false; std::vector<distributed::MeshWorkload> WC; size_t WI=0;
        // ---- T2.5: Metal Trace on the cached path ----
        // TR_PHASE: 0=none(REC/plain cache), 1=CAPTURE(wrap device-op runs in traces), 2=REPLAY.
        // A "segment" = a maximal run of emit() with no host boundary (t_wr/t_rd). seg_hook breaks them.
        int TR_PHASE=0; std::vector<distributed::MeshTraceId> SEG; bool seg_open=false, seg_started=false;
        size_t seg_idx=0; bool MEASURE_DEV=false;
        auto emit=[&](auto&& build){
            if(!CACHE_ON){ auto wl=build(); t_enq(cq,(hold3.emplace_back(std::move(wl)),hold3.back()),false); return; }
            if(TR_PHASE==2){   // REPLAY: on first emit of a segment, replay that segment's trace
                if(!seg_started){ if(seg_idx<SEG.size()){ auto rt=prof::hrc::now();
                        mesh->replay_mesh_trace(0, SEG[seg_idx], MEASURE_DEV);
                        if(MEASURE_DEV){ prof::g_replay+=prof::msd(rt); ++prof::n_replay; } }
                    ++seg_idx; seg_started=true; }
                ++WI; return; }
            if(CACHE_REC) WC.emplace_back(build());
            if(TR_PHASE==1 && !seg_open){ SEG.push_back(mesh->begin_mesh_trace(0)); seg_open=true; }   // open a trace
            t_enq(cq, WC[WI++], false); };
        auto seg_break=[&](){   // host boundary: close the open capture trace / arm next replay segment
            if(!CACHE_ON) return;
            if(TR_PHASE==1){ if(seg_open){ mesh->end_mesh_trace(0, SEG.back()); seg_open=false; } }
            else if(TR_PHASE==2){ seg_started=false; } };
        prof::seg_hook = seg_break;
        auto transpose_dev = [&](std::shared_ptr<distributed::MeshBuffer> in, std::shared_ptr<distributed::MeshBuffer> out){
            emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(1);
            mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto w=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto c=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/transpose_tile.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,r,CC(0),{(uint32_t)in->address(),1u,0u}); SetRuntimeArgs(p,w,CC(0),{(uint32_t)out->address(),1u,0u}); SetRuntimeArgs(p,c,CC(0),{});
            prof::g_build+=prof::msd(__tb); ++prof::n_build; distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); };
        // resident chunked mod-add-reduce + twiddle (EWC cores, 2 tile-blocks/core when EWC<8)
        auto rmodadd=[&](std::vector<std::shared_ptr<distributed::MeshBuffer>>& ins, std::vector<std::shared_ptr<distributed::MeshBuffer>>& outs, uint32_t q){
          for(uint32_t base=0;base<NBLK;base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,NBLK-base);
            emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(cnt);
              mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,NBLK,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
              auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              for(uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                SetRuntimeArgs(p,rd,c,{(uint32_t)ins[t]->address(),NBLK,0u}); SetRuntimeArgs(p,wr,c,{(uint32_t)outs[t]->address(),1u,0u}); SetRuntimeArgs(p,cp,c,{NBLK,q}); }
              prof::g_build+=prof::msd(__tb); ++prof::n_build; distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); } };
        // FUSED reduce+twiddle: the rmodadd->rtwiddle pair as ONE program. The 8:1 reduction result
        // stays in DEST instead of round-tripping through DRAM to feed the twiddle multiply, so this
        // drops one dispatch and one 8-tile write + 8-tile read per forward NTT.
        auto rmodadd_tw=[&](std::vector<std::shared_ptr<distributed::MeshBuffer>>& ins, std::vector<std::shared_ptr<distributed::MeshBuffer>>& tw, std::vector<std::shared_ptr<distributed::MeshBuffer>>& outs, uint32_t q, uint32_t li){
          for(uint32_t base=0;base<NBLK;base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,NBLK-base);
            emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(cnt);
              mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,NBLK,I32); mkcb(p,crs,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
              auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_reduce_tw.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_twiddle.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              for(uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                SetRuntimeArgs(p,rd,c,{(uint32_t)ins[t]->address(),NBLK,(uint32_t)tw[t]->address()});
                SetRuntimeArgs(p,wr,c,{(uint32_t)outs[t]->address(),1u,0u});
                SetRuntimeArgs(p,cp,c,{NBLK,q,ctx[li].q_inv_32(),(uint32_t)(QP[li]&0xFFFF),(uint32_t)(QP[li]>>16)}); }
              prof::g_build+=prof::msd(__tb); ++prof::n_build; distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); } };
        auto rtwiddle=[&](std::vector<std::shared_ptr<distributed::MeshBuffer>>& a, std::vector<std::shared_ptr<distributed::MeshBuffer>>& b, std::vector<std::shared_ptr<distributed::MeshBuffer>>& o, uint32_t q, uint32_t li){
          for(uint32_t base=0;base<NBLK;base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,NBLK-base);
            emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(cnt);
              mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
              auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              for(uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                SetRuntimeArgs(p,rd,c,{(uint32_t)a[t]->address(),(uint32_t)b[t]->address(),1u,0u}); SetRuntimeArgs(p,wr,c,{(uint32_t)o[t]->address(),1u,0u}); SetRuntimeArgs(p,cp,c,{1u,q,ctx[li].q_inv_32(),(uint32_t)(QP[li]&0xFFFF),(uint32_t)(QP[li]>>16)}); }
              prof::g_build+=prof::msd(__tb); ++prof::n_build; distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); } };
        // fwd chain, input from DRAM inb[8], raw output to DRAM outb[8]; NO readback
        auto fwd_buf = [&](std::vector<std::shared_ptr<distributed::MeshBuffer>>& inb,
                           std::vector<std::shared_ptr<distributed::MeshBuffer>>& outb, uint32_t li){
            uint64_t q=QP[li]; int key=li*2+1; auto& big=bigWres[key]; auto sm=smallWres[key]; auto& tw=twres[key];
            std::vector<MM> s1; for(uint32_t bk2=0;bk2<NBLK;++bk2) for(uint32_t bn2=0;bn2<NBLK;++bn2) s1.push_back({inb[bn2],big[bk2*NBLK+bn2],part8[bk2],bn2});
            for(uint32_t base=0;base<s1.size();base+=NCORES){ uint32_t cnt=std::min<uint32_t>(NCORES,s1.size()-base); emit([&]{ return build_mm(s1,base,cnt,NB*NB,"compute/fused_matmul_a.cpp",inb[0],big[0],(uint32_t)q,li); }); }
            rmodadd_tw(part8,tw,midB8,(uint32_t)q,li);   // was: rmodadd(part8,mid8) -> rtwiddle(mid8,tw,midB8)
            std::vector<MM> s2; for(uint32_t bk2=0;bk2<NBLK;++bk2) s2.push_back({midB8[bk2],sm,outb[bk2],0});
            for(uint32_t base=0;base<s2.size();base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,s2.size()-base); emit([&]{ return build_mm(s2,base,cnt,NB*NB,"compute/fused_matmul_c.cpp",midB8[0],sm,(uint32_t)q,li); }); } };
        // inv chain, input from DRAM inb[8] (already transposed), output to DRAM outb[8]; NO readback
        auto inv_buf = [&](std::vector<std::shared_ptr<distributed::MeshBuffer>>& inb,
                           std::vector<std::shared_ptr<distributed::MeshBuffer>>& outb, uint32_t li){
            uint64_t q=QP[li]; int key=li*2+0; auto& big=bigWres[key]; auto sm=smallWres[key]; auto& tw=twres[key];
            std::vector<MM> s1; for(uint32_t bk2=0;bk2<NBLK;++bk2) s1.push_back({inb[bk2],sm,mid8[bk2],0});
            for(uint32_t base=0;base<s1.size();base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,s1.size()-base); emit([&]{ return build_mm(s1,base,cnt,NB*NB,"compute/fused_matmul_a.cpp",inb[0],sm,(uint32_t)q,li); }); }
            rtwiddle(mid8,tw,midB8,(uint32_t)q,li);
            std::vector<MM> s2; for(uint32_t bn2=0;bn2<NBLK;++bn2) for(uint32_t bk2=0;bk2<NBLK;++bk2) s2.push_back({midB8[bk2],big[bk2*NBLK+bn2],part8[bn2],bk2});
            for(uint32_t base=0;base<s2.size();base+=NCORES){ uint32_t cnt=std::min<uint32_t>(NCORES,s2.size()-base); emit([&]{ return build_mm(s2,base,cnt,NB*NB,"compute/fused_matmul_c.cpp",midB8[0],big[0],(uint32_t)q,li); }); }
            rmodadd(part8,outb,(uint32_t)q); };

        if (std::getenv("HE_RT")) {   // buffer-resident fwd->transpose->inv round-trip, one prime, no host round-trip
            uint32_t li=0; uint64_t q=QP[li]; std::mt19937 rr(7); std::uniform_int_distribution<uint32_t> d(0,(uint32_t)q-1);
            std::vector<uint32_t> x(N); for(auto&v:x) v=d(rr);
            std::vector<std::shared_ptr<distributed::MeshBuffer>> ini(NBLK), fo(NBLK), tin(NBLK), io(NBLK);
            for(uint32_t i=0;i<NBLK;++i){ ini[i]=mkbuf(1,I32); fo[i]=mkbuf(1,I32); tin[i]=mkbuf(1,I32); io[i]=mkbuf(1,I32); }
            for(uint32_t bn2=0;bn2<NBLK;++bn2){ std::vector<uint32_t> blk(ELEMS);
                for(uint32_t b=0;b<T;++b) for(uint32_t n1=0;n1<N1;++n1) blk[b*T+n1]=x[n1+32*(bn2*T+b)];
                t_wr(cq, ini[bn2], tilize32(blk), false); }
            fwd_buf(ini, fo, li);
            for(uint32_t bk2=0;bk2<NBLK;++bk2) transpose_dev(fo[bk2], tin[bk2]);   // on-device reindex
            inv_buf(tin, io, li);
            t_fin(cq);                                               // ONE finish, at the end
            std::vector<uint32_t> xo(N);
            for(uint32_t bn2=0;bn2<NBLK;++bn2){ std::vector<uint32_t> raw; t_rd(cq, raw, io[bn2], true);
                auto blk=untilize32(raw); for(uint32_t n1=0;n1<N1;++n1) for(uint32_t n2l=0;n2l<T;++n2l) xo[n1+32*(bn2*T+n2l)]=blk[n1*T+n2l]; }
            uint32_t mm=0; for(uint32_t n=0;n<N;++n) if(xo[n]!=x[n]) ++mm;
            fmt::print("HE_RT buffer-resident roundtrip INTT(NTT(x))==x : mism={}/{} [{}]\n", mm, N, mm==0?"PASS":"FAIL");
            pass=(mm==0); pass&=mesh->close();
            if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n"); return pass?0:1;
        }

        // ===== T1: PERSISTENT PROGRAMS + Metal Trace (jax.jit-style capture+replay) =====
        // Trace forbids writes during capture, and a fresh program's first enqueue writes its binary.
        // So: build the chain's workloads ONCE (persistent), WARM them (loads binaries), then capture
        // the RE-ENQUEUE of the same objects. Replay per call w/ fresh inputs in the SAME fixed buffers.
        if (std::getenv("HE_TRACE")) {
            uint32_t li=0; std::mt19937 rr(7); std::uniform_int_distribution<uint32_t> d(0,(uint32_t)QP[li]-1);
            std::vector<uint32_t> x(N); for(auto&v:x) v=d(rr);
            std::vector<std::shared_ptr<distributed::MeshBuffer>> ini(NBLK), fo(NBLK), tin(NBLK), io(NBLK);
            for(uint32_t i=0;i<NBLK;++i){ ini[i]=mkbuf(1,I32); fo[i]=mkbuf(1,I32); tin[i]=mkbuf(1,I32); io[i]=mkbuf(1,I32); }
            auto upload=[&](){ for(uint32_t bn2=0;bn2<NBLK;++bn2){ std::vector<uint32_t> blk(ELEMS);
                for(uint32_t b=0;b<T;++b) for(uint32_t n1=0;n1<N1;++n1) blk[b*T+n1]=x[n1+32*(bn2*T+b)];
                t_wr(cq, ini[bn2], tilize32(blk), false); } };
            auto readout=[&](){ std::vector<uint32_t> xo(N);
                for(uint32_t bn2=0;bn2<NBLK;++bn2){ std::vector<uint32_t> raw; t_rd(cq, raw, io[bn2], true);
                    auto blk=untilize32(raw); for(uint32_t n1=0;n1<N1;++n1) for(uint32_t n2l=0;n2l<T;++n2l) xo[n1+32*(bn2*T+n2l)]=blk[n1*T+n2l]; } return xo; };
            // transpose workload builder (returns wl; mirrors transpose_dev)
            auto build_tr=[&](std::shared_ptr<distributed::MeshBuffer> in, std::shared_ptr<distributed::MeshBuffer> out){
                Program p=CreateProgram(); auto crs=crs_of(1);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto w=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto c=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/transpose_tile.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                SetRuntimeArgs(p,r,CC(0),{(uint32_t)in->address(),1u,0u}); SetRuntimeArgs(p,w,CC(0),{(uint32_t)out->address(),1u,0u}); SetRuntimeArgs(p,c,CC(0),{});
                distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; };
            // build the FULL chain (fwd + 8 transpose + inv) as a persistent ordered workload list
            uint64_t q=QP[li];
            auto build_chain=[&](){ std::vector<distributed::MeshWorkload> P;
                { int key=li*2+1; auto& big=bigWres[key]; auto sm=smallWres[key]; auto& tw=twres[key];   // FWD
                  std::vector<MM> s1; for(uint32_t bk2=0;bk2<NBLK;++bk2) for(uint32_t bn2=0;bn2<NBLK;++bn2) s1.push_back({ini[bn2],big[bk2*NBLK+bn2],part8[bk2],bn2});
                  for(uint32_t b=0;b<s1.size();b+=NCORES) P.push_back(build_mm(s1,b,std::min<uint32_t>(NCORES,s1.size()-b),NB*NB,"compute/fused_matmul_a.cpp",ini[0],big[0],(uint32_t)q,li));
                  P.push_back(build_modadd(part8,mid8,(uint32_t)q)); P.push_back(build_twiddle(mid8,tw,midB8,(uint32_t)q,li));
                  std::vector<MM> s2; for(uint32_t bk2=0;bk2<NBLK;++bk2) s2.push_back({midB8[bk2],sm,fo[bk2],0});
                  P.push_back(build_mm(s2,0,NBLK,NB*NB,"compute/fused_matmul_c.cpp",midB8[0],sm,(uint32_t)q,li)); }
                for(uint32_t t=0;t<NBLK;++t) P.push_back(build_tr(fo[t],tin[t]));                          // TRANSPOSE
                { int key=li*2+0; auto& big=bigWres[key]; auto sm=smallWres[key]; auto& tw=twres[key];   // INV
                  std::vector<MM> s1; for(uint32_t bk2=0;bk2<NBLK;++bk2) s1.push_back({tin[bk2],sm,mid8[bk2],0});
                  P.push_back(build_mm(s1,0,NBLK,NB*NB,"compute/fused_matmul_a.cpp",tin[0],sm,(uint32_t)q,li));
                  P.push_back(build_twiddle(mid8,tw,midB8,(uint32_t)q,li));
                  std::vector<MM> s2; for(uint32_t bn2=0;bn2<NBLK;++bn2) for(uint32_t bk2=0;bk2<NBLK;++bk2) s2.push_back({midB8[bk2],big[bk2*NBLK+bn2],part8[bn2],bk2});
                  for(uint32_t b=0;b<s2.size();b+=NCORES) P.push_back(build_mm(s2,b,std::min<uint32_t>(NCORES,s2.size()-b),NB*NB,"compute/fused_matmul_c.cpp",midB8[0],big[0],(uint32_t)q,li));
                  P.push_back(build_modadd(part8,io,(uint32_t)q)); }
                return P; };

            auto progs = build_chain();                                   // build ONCE (persistent)
            fmt::print("HE_TRACE: chain = {} persistent workloads\n", progs.size());
            auto enq_all=[&](){ for(auto& wl:progs) t_enq(cq,wl,false); };

            // (a) reference via persistent re-enqueue (also WARMS the binaries)
            upload(); enq_all(); t_fin(cq); auto ref=readout();
            uint32_t m0=0; for(uint32_t n=0;n<N;++n) if(ref[n]!=x[n]) ++m0;
            // (b) CAPTURE the re-enqueue of the SAME warm programs (no writes inside)
            auto tid = mesh->begin_mesh_trace(0); enq_all(); mesh->end_mesh_trace(0, tid);
            // (c) REPLAY: fresh input into fixed buffers, then replay the compiled trace
            upload(); mesh->replay_mesh_trace(0, tid, true); auto got=readout();
            uint32_t m1=0,m2=0; for(uint32_t n=0;n<N;++n){ if(got[n]!=x[n]) ++m1; if(got[n]!=ref[n]) ++m2; }
            bool ok=(m0==0&&m1==0&&m2==0);
            fmt::print("HE_TRACE T1: persistent_rt={}/{}  traced_rt={}/{}  traced_vs_persistent={}/{}  [{}]\n",
                       m0,N,m1,N,m2,N, ok?"PASS":"FAIL");
            // (d) timing: build-fresh vs persistent-reenqueue vs traced-replay
            int it=20; if(const char* e=std::getenv("HE_ITERS")){int v=std::atoi(e); if(v>0) it=v;}
            prof::reset(); auto t0=prof::hrc::now();
            for(int i=0;i<it;++i){ auto pp=build_chain(); for(auto& wl:pp) t_enq(cq,wl,false); t_fin(cq); } double bf=prof::msd(t0)/it;
            double bfb=prof::g_build/it, bfe=prof::g_enq/it;
            auto t1b=prof::hrc::now(); for(int i=0;i<it;++i){ enq_all(); t_fin(cq); } double pe2=prof::msd(t1b)/it;
            auto t2=prof::hrc::now(); for(int i=0;i<it;++i) mesh->replay_mesh_trace(0,tid,true); double tr=prof::msd(t2)/it;
            fmt::print("HE_TRACE T1 timing (1 transform-chain = {} dispatches, {} iters):\n", progs.size(), it);
            fmt::print("  build-fresh + enqueue : {:.3f} ms/iter  (build {:.3f} + enqueue {:.3f})\n", bf, bfb, bfe);
            fmt::print("  persistent re-enqueue : {:.3f} ms/iter  (no rebuild)              {:.1f}x\n", pe2, bf/pe2);
            fmt::print("  TRACED replay         : {:.3f} ms/iter  ({} device commands)      {:.1f}x\n", tr, 1, tr>0?bf/tr:0.0);
            pass=ok; pass&=mesh->close();
            if(pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n"); return pass?0:1;
        }

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
            t_wr(cq, pa, at, false); t_wr(cq, pb, bt, false);
            auto __tb=prof::hrc::now(); Program p = CreateProgram(); auto crs = crs_of(NBLK);
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
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); t_enq(cq, wl, false);
            t_fin(cq);
            std::vector<uint32_t> raw; t_rd(cq, raw, po, true);
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
                t_wr(cq, fin[t], xb, false); }
            auto __tb=prof::hrc::now(); Program p = CreateProgram(); auto crs = crs_of(NBLK);
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
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); t_enq(cq, wl, false);
            t_fin(cq);
            std::vector<std::vector<uint32_t>> out(Ld, std::vector<uint32_t>(N));
            for (uint32_t t=0;t<NBLK;++t){ std::vector<uint32_t> raw; t_rd(cq, raw, fout[t], true);
                for (uint32_t j=0;j<Ld;++j){ auto u=untilize32(std::vector<uint32_t>(raw.begin()+j*ELEMS, raw.begin()+(j+1)*ELEMS));
                    std::copy(u.begin(), u.end(), out[j].begin()+t*ELEMS); } }
            return out; };

        // ============ CROSS-aligned Set-B HE-mul: eval-resident, tensor pointwise, drop-prime rescale ============
        // Prime-index sets (index 7 = the Q prime dropped by rescale; P = 8..11).
        constexpr uint32_t DNUM=2;
        std::vector<uint32_t> qin={0,1,2,3,4,5,6,7};          // 8 input Q limbs
        const uint32_t dropIdx=7;                              // rescale drops this prime
        std::vector<uint32_t> qidx={0,1,2,3,4,5,6};            // 7 working Q limbs
        std::vector<uint32_t> pidx={8,9,10,11};                // 4 P limbs
        std::vector<uint32_t> qp={0,1,2,3,4,5,6,8,9,10,11};    // 11 QP limbs (keyswitch)
        std::vector<std::vector<uint32_t>> group={{0,1,2,3},{4,5,6}};   // dnum=2 digits (sizes 4,3)
        std::vector<std::vector<uint32_t>> comp(DNUM);
        for (uint32_t j=0;j<DNUM;++j) for (auto li:qp){ bool own=false; for(auto g:group[j]) if(g==li) own=true; if(!own) comp[j].push_back(li); }
        u128 Pprod=1; for (auto p:pidx) Pprod*=QP[p];

        std::mt19937_64 rng(seed);
        auto randpoly=[&](uint64_t q){ std::vector<uint32_t> v(N); for(auto&x:v) x=(uint32_t)(rng()%q); return v; };
        std::vector<std::vector<uint32_t>> c0(LQP),c1(LQP),d0(LQP),d1(LQP);   // input coeff, limbs 0..7
        for (auto i:qin){ c0[i]=randpoly(QP[i]); c1[i]=randpoly(QP[i]); d0[i]=randpoly(QP[i]); d1[i]=randpoly(QP[i]); }
        std::vector<std::vector<std::vector<uint32_t>>> evk0(DNUM,std::vector<std::vector<uint32_t>>(LQP)),
                                                        evk1(DNUM,std::vector<std::vector<uint32_t>>(LQP));
        for (uint32_t j=0;j<DNUM;++j) for (auto li:qp){ evk0[j][li]=randpoly(QP[li]); evk1[j][li]=randpoly(QP[li]); }

        // ---- CPU helpers ----
        auto negmul=[&](const std::vector<uint32_t>& a,const std::vector<uint32_t>& b,uint64_t q){ std::vector<uint32_t> c(N,0);
            for (uint32_t i=0;i<N;++i){ if(!a[i]) continue; u128 ai=a[i];
                for (uint32_t jj=0;jj<N;++jj){ uint32_t k=i+jj; uint64_t v=(uint64_t)(ai*b[jj]%q);
                    if(k<N) c[k]=(uint32_t)(((u128)c[k]+v)%q); else c[k-N]=(uint32_t)(((u128)c[k-N]+q-v)%q); } } return c; };
        auto fbc_approx=[&](const std::vector<std::vector<uint32_t>>& x,const std::vector<uint32_t>& sI,const std::vector<uint32_t>& dI){
            uint32_t Ls=sI.size(),Ld=dI.size(); u128 S=1; for(uint32_t i=0;i<Ls;++i) S*=QP[sI[i]];
            std::vector<std::vector<uint32_t>> out(Ld,std::vector<uint32_t>(N));
            for (uint32_t n=0;n<N;++n){ std::vector<uint64_t> y(Ls);
                for (uint32_t i=0;i<Ls;++i){ uint64_t si=QP[sI[i]]; y[i]=(uint64_t)(((u128)x[i][n]*modpow((uint64_t)((S/si)%si),si-2,si))%si); }
                for (uint32_t j=0;j<Ld;++j){ uint64_t pj=QP[dI[j]]; u128 acc=0; for(uint32_t i=0;i<Ls;++i) acc+=(u128)y[i]*((S/QP[sI[i]])%pj); out[j][n]=(uint32_t)(acc%pj);} }
            return out; };

        // ================= CPU oracle (coeff): rescale inputs -> tensor -> keyswitch(dnum) -> combine =====
        auto cpu_hemul=[&](){
            auto rescale=[&](const std::vector<std::vector<uint32_t>>& poly){ uint64_t qL=QP[dropIdx];
                std::vector<std::vector<uint32_t>> out(LQP);
                for (auto k:qidx){ uint64_t q=QP[k]; uint64_t qLinv=modpow(qL%q,q-2,q); out[k].resize(N);
                    for (uint32_t n=0;n<N;++n){ uint32_t corr=(uint32_t)(poly[dropIdx][n]%q);
                        out[k][n]=(uint32_t)((((u128)poly[k][n]+q-corr)%q)*qLinv%q); } } return out; };
            auto c0r=rescale(c0),c1r=rescale(c1),d0r=rescale(d0),d1r=rescale(d1);
            std::vector<std::vector<uint32_t>> e0(LQP),e1(LQP),e2(LQP);
            for (auto li:qidx){ uint64_t q=QP[li]; e0[li]=negmul(c0r[li],d0r[li],q); e2[li]=negmul(c1r[li],d1r[li],q);
                auto t01=negmul(c0r[li],d1r[li],q),t10=negmul(c1r[li],d0r[li],q);
                e1[li].resize(N); for(uint32_t n=0;n<N;++n) e1[li][n]=(uint32_t)(((u128)t01[n]+t10[n])%q); }
            std::vector<std::vector<uint32_t>> acc0(LQP,std::vector<uint32_t>(N,0)),acc1(LQP,std::vector<uint32_t>(N,0));
            for (uint32_t j=0;j<DNUM;++j){ std::vector<std::vector<uint32_t>> vj(LQP);
                for (auto li:group[j]) vj[li]=e2[li];
                std::vector<std::vector<uint32_t>> xs; for(auto li:group[j]) xs.push_back(e2[li]);
                auto o=fbc_approx(xs,group[j],comp[j]); for(uint32_t k=0;k<comp[j].size();++k) vj[comp[j][k]]=o[k];
                for (auto li:qp){ uint64_t q=QP[li]; auto p0=negmul(vj[li],evk0[j][li],q),p1=negmul(vj[li],evk1[j][li],q);
                    for(uint32_t n=0;n<N;++n){ acc0[li][n]=(uint32_t)(((u128)acc0[li][n]+p0[n])%q); acc1[li][n]=(uint32_t)(((u128)acc1[li][n]+p1[n])%q);} } }
            auto moddown=[&](std::vector<std::vector<uint32_t>>& acc){ std::vector<std::vector<uint32_t>> aP; for(auto p:pidx) aP.push_back(acc[p]);
                auto f=fbc_approx(aP,pidx,qidx); std::vector<std::vector<uint32_t>> out(LQP);
                for (uint32_t k=0;k<qidx.size();++k){ uint32_t li=qidx[k]; uint64_t q=QP[li]; uint64_t Pinv=modpow((uint64_t)(Pprod%q),q-2,q); out[li].resize(N);
                    for(uint32_t n=0;n<N;++n) out[li][n]=(uint32_t)((((u128)acc[li][n]+q-f[k][n])%q)*Pinv%q); } return out; };
            auto k0=moddown(acc0),k1=moddown(acc1);
            std::vector<std::vector<uint32_t>> out;
            for (auto li:qidx){ uint64_t q=QP[li]; std::vector<uint32_t> r(N); for(uint32_t n=0;n<N;++n) r[n]=(uint32_t)(((u128)e0[li][n]+k0[li][n])%q); out.push_back(r); }
            for (auto li:qidx){ uint64_t q=QP[li]; std::vector<uint32_t> r(N); for(uint32_t n=0;n<N;++n) r[n]=(uint32_t)(((u128)e1[li][n]+k1[li][n])%q); out.push_back(r); }
            return out; };

        // ================= DEVICE (eval-resident): NTT inputs (setup) -> rescale -> tensor -> ks -> combine =====
        // evk freq (device NTT of coeff evk), reused across iters.
        // resident-only paths (dev_hemul_res) use the resident NTT (NTT_CORES wide) + Montgomery evk
        // (evk0m), so the 8-core non-resident fwd8192 evk/input prep below is UNUSED there — skip it,
        // else its 8-core zones pollute the profiler capture with cores the mul itself never touches.
        bool resident_only = std::getenv("HE_PROF_DRAIN")||std::getenv("HE_TRACE2")||std::getenv("HE_RES_CACHE")||std::getenv("HE_RES");
        std::vector<std::vector<std::vector<uint32_t>>> evk0f(DNUM,std::vector<std::vector<uint32_t>>(LQP)),
                                                        evk1f(DNUM,std::vector<std::vector<uint32_t>>(LQP));
        for (uint32_t j=0;j<DNUM;++j) for (auto li:qp){ evk0f[j][li]=fwd8192(evk0[j][li],li); evk1f[j][li]=fwd8192(evk1[j][li],li); }  // needed: evk0m built from this
        // eval-resident inputs (encryption-side NTT; not part of the timed mul) — only the non-resident dev_hemul uses these
        std::vector<std::vector<uint32_t>> c0e(LQP),c1e(LQP),d0e(LQP),d1e(LQP);
        if(!resident_only) for (auto i:qin){ c0e[i]=fwd8192(c0[i],i); c1e[i]=fwd8192(c1[i],i); d0e[i]=fwd8192(d0[i],i); d1e[i]=fwd8192(d1[i],i); }

        auto rescale_dev=[&](const std::vector<std::vector<uint32_t>>& pe){   // pe eval, limbs 0..7 -> eval, qidx
            uint64_t qL=QP[dropIdx]; auto lastc=inv8192(pe[dropIdx],dropIdx);     // 1 INTT
            std::vector<std::vector<uint32_t>> out(LQP);
            for (auto k:qidx){ uint64_t q=QP[k]; uint64_t qLinv=modpow(qL%q,q-2,q);
                std::vector<uint32_t> corr(N); for(uint32_t n=0;n<N;++n) corr[n]=(uint32_t)(lastc[n]%q);
                auto ce=fwd8192(corr,k);                                          // 1 NTT
                out[k].resize(N); for(uint32_t n=0;n<N;++n) out[k][n]=(uint32_t)((((u128)pe[k][n]+q-ce[n])%q)*qLinv%q); }
            return out; };

        // =====================================================================================
        // STEP 3: buffer-resident dev_hemul_res — intermediates stay in DRAM, one final readback.
        // Layout: eval = E (fwd_buf raw), coeff = C'' (inv_buf out); transposes bracket transforms.
        // FBC kept at a host boundary (~8 readbacks/mul) to avoid an on-device gather. Gated HE_RES.
        // =====================================================================================
        using Buf = std::shared_ptr<distributed::MeshBuffer>;
        using Buf8 = std::vector<Buf>;
        // buffer POOL: when CACHE_ON, hand out fixed buffers in deterministic order (reset PI each mul)
        // so addresses are STABLE across muls -> cached workloads' baked addresses stay valid.
        std::vector<Buf8> POOL; size_t PI=0;
        auto mkBuf8 = [&]() -> Buf8 {
            if(!CACHE_ON){ Buf8 b(NBLK); for(uint32_t i=0;i<NBLK;++i) b[i]=mkbuf(1,I32); return b; }
            if(PI>=POOL.size()){ Buf8 b(NBLK); for(uint32_t i=0;i<NBLK;++i) b[i]=mkbuf(1,I32); POOL.push_back(b); }
            return POOL[PI++]; };
        auto fill8 = [&](uint32_t v){ Buf8 b=mkBuf8(); std::vector<uint32_t> t(ELEMS,v); auto tl=tilize32(t);
            for(uint32_t i=0;i<NBLK;++i) t_wr(cq,b[i],tl,false); return b; };
        auto up_natC = [&](const std::vector<uint32_t>& x){ Buf8 b=mkBuf8();               // natural coeff -> C'' ([n1][n2l])
            for(uint32_t bn2=0;bn2<NBLK;++bn2){ std::vector<uint32_t> blk(ELEMS);
                for(uint32_t n1=0;n1<N1;++n1) for(uint32_t n2l=0;n2l<T;++n2l) blk[n1*T+n2l]=x[n1+32*(bn2*T+n2l)];
                t_wr(cq,b[bn2],tilize32(blk),false);} return b; };
        auto up_natE = [&](const std::vector<uint32_t>& x, bool mont, uint32_t li){ Buf8 b=mkBuf8();  // natural eval -> E ([k2l][k1])
            for(uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> blk(ELEMS);
                for(uint32_t k2l=0;k2l<T;++k2l) for(uint32_t k1=0;k1<N1;++k1){ uint32_t v=x[(bk2*T+k2l)+N2*k1]; blk[k2l*T+k1]=mont?ctx[li].to_montgomery(v):v; }
                t_wr(cq,b[bk2],tilize32(blk),false);} return b; };
        auto rd_natE = [&](Buf8& b){ std::vector<uint32_t> x(N);                            // E -> natural eval
            for(uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> raw; t_rd(cq,raw,b[bk2],true);
                auto blk=untilize32(raw); for(uint32_t k2l=0;k2l<T;++k2l) for(uint32_t k1=0;k1<N1;++k1) x[(bk2*T+k2l)+N2*k1]=blk[k2l*T+k1]; } return x; };
        auto rd_natC = [&](Buf8& b){ std::vector<uint32_t> x(N);                            // C'' -> natural coeff
            for(uint32_t bn2=0;bn2<NBLK;++bn2){ std::vector<uint32_t> raw; t_rd(cq,raw,b[bn2],true);
                auto blk=untilize32(raw); for(uint32_t n1=0;n1<N1;++n1) for(uint32_t n2l=0;n2l<T;++n2l) x[n1+32*(bn2*T+n2l)]=blk[n1*T+n2l]; } return x; };

        // resident elementwise (8 tiles -> 8 cores), no readback
        auto ew = [&](const char* rd, const char* cp, Buf8& a, Buf8& b, Buf8& out, std::vector<uint32_t> cargs, bool modmul){
          for(uint32_t base=0;base<NBLK;base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,NBLK-base);
            emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(cnt);
            mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX+std::string(rd),crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto w=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto c=CreateKernel(p,OVERRIDE_KERNEL_PREFIX+std::string(cp),crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            for(uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto cc=CC(k);
                if(modmul) SetRuntimeArgs(p,r,cc,{(uint32_t)a[t]->address(),(uint32_t)b[t]->address(),1u,0u});
                else       SetRuntimeArgs(p,r,cc,{(uint32_t)a[t]->address(),(uint32_t)b[t]->address(),1u,0u,0u});
                SetRuntimeArgs(p,w,cc,{(uint32_t)out[t]->address(),1u,0u});
                std::vector<uint32_t> ca=cargs; SetRuntimeArgs(p,c,cc,ca); }
            prof::g_build+=prof::msd(__tb); ++prof::n_build; distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); } };
        auto pw_buf   = [&](Buf8& a, Buf8& b, Buf8& o, uint32_t li){ ew("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp",a,b,o,{1u,(uint32_t)QP[li],ctx[li].q_inv_32(),(uint32_t)(QP[li]&0xFFFF),(uint32_t)(QP[li]>>16)},true); };
        auto madd_buf = [&](Buf8& a, Buf8& b, Buf8& o, uint32_t li){ ew("dataflow/reader_ab.cpp","compute/modadd_map.cpp",a,b,o,{1u,(uint32_t)QP[li]},false); };
        auto msub_buf = [&](Buf8& a, Buf8& b, Buf8& o, uint32_t li){ ew("dataflow/reader_ab.cpp","compute/modsub_reduce.cpp",a,b,o,{(uint32_t)QP[li]},false); };
        // FUSED tensor op: out = c * (d * rc) mod q in ONE kernel (dm stays in L1, no DRAM round-trip)
        auto pw2_buf = [&](Buf8& c, Buf8& d, Buf8& rc, Buf8& out, uint32_t li){
            emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(NBLK);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_1,tt::DataFormat::UInt32,2,I32);
                mkcb(p,crs,CBIndex::c_2,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_3,tt::DataFormat::UInt32,2,I32);
                mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_cd_rc.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto w=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/tensor2_fused.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for(uint32_t t=0;t<NBLK;++t){ auto cc=CC(t);
                    SetRuntimeArgs(p,r,cc,{(uint32_t)c[t]->address(),(uint32_t)d[t]->address(),(uint32_t)rc[t]->address(),1u});
                    SetRuntimeArgs(p,w,cc,{(uint32_t)out[t]->address(),1u,0u});
                    SetRuntimeArgs(p,cp,cc,{1u,(uint32_t)QP[li],ctx[li].q_inv_32(),(uint32_t)(QP[li]&0xFFFF),(uint32_t)(QP[li]>>16)}); }
                prof::g_build+=prof::msd(__tb); ++prof::n_build; distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); };
        Buf8 ttmp=mkBuf8();
        // batched reindex: all 8 tiles transposed in ONE program across 8 cores (was 8x single-tile/1-core)
        auto transpose8_dev = [&](Buf8& in, Buf8& out){
          for(uint32_t base=0;base<NBLK;base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,NBLK-base);
            emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(cnt);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto w=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto c=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/transpose_tile.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for(uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto cc=CC(k); SetRuntimeArgs(p,r,cc,{(uint32_t)in[t]->address(),1u,0u}); SetRuntimeArgs(p,w,cc,{(uint32_t)out[t]->address(),1u,0u}); SetRuntimeArgs(p,c,cc,{}); }
                prof::g_build+=prof::msd(__tb); ++prof::n_build; distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); } };
        auto NTT_res = [&](Buf8& inC, Buf8& outE, uint32_t li){ transpose8_dev(inC,ttmp); fwd_buf(ttmp,outE,li); };
        auto INTT_res= [&](Buf8& inE, Buf8& outC, uint32_t li){ transpose8_dev(inE,ttmp); inv_buf(ttmp,outC,li); };

        // T3: buffer-resident FBC. src limbs (C'' Buf8) -> baseconv (per-element, layout-agnostic) -> dst
        // limbs (C'' Buf8). On-device gather/scatter replace rd_natC/up_natC — no host round-trip. Same
        // baseconv_fbc kernel + constants as fbc8192, so bit-identical; only the data plumbing is resident.
        Buf8 zt = fill8(0);   // persistent zero tile-group for FBC source padding (allocated with CACHE off)
        auto fbc8192_res = [&](const std::vector<Buf8>& src, const std::vector<uint32_t>& srcIdx,
                               const std::vector<uint32_t>& dstIdx) -> std::vector<Buf8> {
            uint32_t Ls=srcIdx.size(), Ld=dstIdx.size();
            u128 S=1; for(uint32_t i=0;i<Ls;++i) S*=QP[srcIdx[i]];
            std::vector<uint32_t> C1(FBCW,0),Sm(FBCW),SmI(FBCW),Dm(FBCW),DmI(FBCW);
            std::vector<std::vector<uint32_t>> C2(FBCW,std::vector<uint32_t>(FBCW,0));
            for(uint32_t i=0;i<FBCW;++i){ uint32_t si=(i<Ls)?srcIdx[i]:srcIdx[0]; uint64_t qi=QP[si];
                Sm[i]=(uint32_t)qi; SmI[i]=ctx[si].q_inv_32();
                if(i<Ls) C1[i]=ctx[si].to_montgomery((uint32_t)modpow((uint64_t)((S/qi)%qi),qi-2,qi)); }
            for(uint32_t j=0;j<FBCW;++j){ uint32_t dj=(j<Ld)?dstIdx[j]:dstIdx[0]; uint64_t pj=QP[dj];
                Dm[j]=(uint32_t)pj; DmI[j]=ctx[dj].q_inv_32();
                if(j<Ld) for(uint32_t i=0;i<Ls;++i) C2[i][j]=ctx[dj].to_montgomery((uint32_t)((S/QP[srcIdx[i]])%pj)); }
            std::vector<uint32_t> ca; ca.push_back(FBCW);
            for(uint32_t i=0;i<FBCW;++i) ca.push_back(C1[i]);
            for(uint32_t i=0;i<FBCW;++i) for(uint32_t j=0;j<FBCW;++j) ca.push_back(C2[i][j]);
            for(uint32_t i=0;i<FBCW;++i) ca.push_back(Sm[i]);
            for(uint32_t i=0;i<FBCW;++i) ca.push_back(SmI[i]);
            for(uint32_t j=0;j<FBCW;++j) ca.push_back(Dm[j]);
            for(uint32_t j=0;j<FBCW;++j) ca.push_back(DmI[j]);
            std::vector<Buf8> dst(Ld); for(uint32_t j=0;j<Ld;++j) dst[j]=mkBuf8();
            for(uint32_t base=0;base<NBLK;base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,NBLK-base);
            emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(cnt);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,8,I32); mkcb(p,crs,CBIndex::c_2,tt::DataFormat::UInt32,8,I32);
                mkcb(p,crs,CBIndex::c_3,tt::DataFormat::UInt32,8,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,8,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_gather.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_scatter.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/baseconv_fbc.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for(uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                    std::vector<uint32_t> ra; ra.push_back(FBCW);
                    for(uint32_t i=0;i<FBCW;++i) ra.push_back((uint32_t)((i<Ls?src[i][t]:zt[t])->address()));
                    SetRuntimeArgs(p,rd,c,ra);
                    std::vector<uint32_t> wa; wa.push_back(Ld); wa.push_back(FBCW);
                    for(uint32_t j=0;j<Ld;++j) wa.push_back((uint32_t)(dst[j][t]->address()));
                    SetRuntimeArgs(p,wr,c,wa);
                    SetRuntimeArgs(p,cp,c,ca); }
                prof::g_build+=prof::msd(__tb); ++prof::n_build; distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); }
            return dst; };

        // resident constants
        std::map<uint32_t,Buf8> RConst, qLinvC, PinvC;
        for (auto li : qp){ RConst[li]=fill8(ctx[li].to_montgomery(ctx[li].to_montgomery(1u))); }
        for (auto k : qidx){ uint64_t q=QP[k],qL=QP[dropIdx]; RConst[k]=RConst.count(k)?RConst[k]:fill8(ctx[k].to_montgomery(ctx[k].to_montgomery(1u)));
            qLinvC[k]=fill8(ctx[k].to_montgomery((uint32_t)modpow(qL%q,q-2,q))); PinvC[k]=fill8(ctx[k].to_montgomery((uint32_t)modpow((uint64_t)(Pprod%q),q-2,q))); }
        std::map<int,Buf8> evk0m,evk1m;   // evk in E layout, Montgomery
        for (uint32_t j=0;j<DNUM;++j) for (auto li:qp){ evk0m[j*100+li]=up_natE(evk0f[j][li],true,li); evk1m[j*100+li]=up_natE(evk1f[j][li],true,li); }

        auto dev_hemul_res=[&](){   // fully buffer-resident; returns (r0,r1) natural eval (7 Q-limbs)
            hold3.clear(); WI=0; PI=0; seg_idx=0; seg_started=false;   // reset cache/pool/segment indices each mul
            // inputs coeff -> C'' -> NTT -> eval E
            std::map<int,Buf8> ce;   // ce[poly*100+li] eval E, poly 0=c0 1=c1 2=d0 3=d1
            auto NTTin=[&](std::vector<std::vector<uint32_t>>& src, int poly){ for(auto li:qin){ Buf8 cc=up_natC(src[li]); Buf8 e=mkBuf8(); NTT_res(cc,e,li); ce[poly*100+li]=e; } };
            NTTin(c0,0); NTTin(c1,1); NTTin(d0,2); NTTin(d1,3);
            // rescale each (drop dropIdx): lastc=INTT(pe[7]); corr=fbc({7}->qidx); out=( pe - NTT(corr) )*qLinv
            auto rescale=[&](int poly){ std::map<int,Buf8> out;
                Buf8 lastc=mkBuf8(); INTT_res(ce[poly*100+dropIdx], lastc, dropIdx);
                auto corr=fbc8192_res({lastc}, std::vector<uint32_t>{dropIdx}, qidx);           // resident FBC (no host round-trip)
                for(uint32_t k=0;k<qidx.size();++k){ uint32_t li=qidx[k];
                    Buf8 ceb=mkBuf8(); NTT_res(corr[k],ceb,li);                                  // NTT(corr) eval E
                    Buf8 diff=mkBuf8(); msub_buf(ce[poly*100+li], ceb, diff, li);
                    Buf8 o=mkBuf8(); pw_buf(diff, qLinvC[li], o, li); out[li]=o; } return out; };
            auto c0r=rescale(0),c1r=rescale(1),d0r=rescale(2),d1r=rescale(3);
            // tensor (eval E): e0=c0r*d0r, e2=c1r*d1r, e1=c0r*d1r+c1r*d0r  (montify one operand)
            std::map<int,Buf8> e0,e1,e2;
            // NOTE: tensor2_fused (pw2_buf) works bit-exact but is ~neutral/slower here — montify recompute
            // (4x vs 2x) + already-resident small intermediates. Fusion pays on the NTT reshuffle, not this.
            // Reverted to dm-computed-once. `pw2_buf`/`tensor2_fused.cpp` kept as a proven fused-kernel primitive.
            for(auto li:qidx){ Buf8 dm0=mkBuf8(),dm1=mkBuf8(); pw_buf(d0r[li],RConst[li],dm0,li); pw_buf(d1r[li],RConst[li],dm1,li);
                Buf8 E0=mkBuf8(),E2=mkBuf8(),t01=mkBuf8(),t10=mkBuf8(),E1=mkBuf8();
                pw_buf(c0r[li],dm0,E0,li); pw_buf(c1r[li],dm1,E2,li); pw_buf(c0r[li],dm1,t01,li); pw_buf(c1r[li],dm0,t10,li); madd_buf(t01,t10,E1,li);
                e0[li]=E0; e1[li]=E1; e2[li]=E2; }
            // keyswitch e2 -> k0,k1 (eval E)
            std::map<int,Buf8> e2c;                                                            // e2 coeff C''
            for(auto li:qidx){ Buf8 c=mkBuf8(); INTT_res(e2[li],c,li); e2c[li]=c; }
            std::map<int,Buf8> acc0,acc1; for(auto li:qp){ acc0[li]=fill8(0); acc1[li]=fill8(0); }
            for(uint32_t j=0;j<DNUM;++j){
                std::vector<Buf8> xs; for(auto li:group[j]) xs.push_back(e2c[li]);              // resident FBC (no host round-trip)
                auto o=fbc8192_res(xs,group[j],comp[j]);
                std::map<int,Buf8> vje; for(auto li:group[j]) vje[li]=e2[li];                   // own = e2 eval E
                for(uint32_t k=0;k<comp[j].size();++k){ uint32_t li=comp[j][k]; Buf8 ve=mkBuf8(); NTT_res(o[k],ve,li); vje[li]=ve; }
                for(auto li:qp){ Buf8 p0=mkBuf8(),p1=mkBuf8(),n0=mkBuf8(),n1=mkBuf8();
                    pw_buf(vje[li],evk0m[j*100+li],p0,li); pw_buf(vje[li],evk1m[j*100+li],p1,li);
                    madd_buf(acc0[li],p0,n0,li); madd_buf(acc1[li],p1,n1,li); acc0[li]=n0; acc1[li]=n1; } }
            auto moddown=[&](std::map<int,Buf8>& acc){ std::map<int,Buf8> ac; for(auto li:qp){ Buf8 c=mkBuf8(); INTT_res(acc[li],c,li); ac[li]=c; }
                std::vector<Buf8> aP; for(auto pp:pidx) aP.push_back(ac[pp]);                   // resident FBC (no host round-trip)
                auto f=fbc8192_res(aP,pidx,qidx); std::map<int,Buf8> ke;
                for(uint32_t k=0;k<qidx.size();++k){ uint32_t li=qidx[k]; Buf8 diff=mkBuf8(),kc=mkBuf8(),keb=mkBuf8();
                    msub_buf(ac[li],f[k],diff,li); pw_buf(diff,PinvC[li],kc,li); NTT_res(kc,keb,li); ke[li]=keb; } return ke; };
            auto k0=moddown(acc0),k1=moddown(acc1);
            std::vector<std::vector<uint32_t>> out;
            for(auto li:qidx){ Buf8 r=mkBuf8(); madd_buf(e0[li],k0[li],r,li); t_fin(cq); out.push_back(rd_natE(r)); }
            for(auto li:qidx){ Buf8 r=mkBuf8(); madd_buf(e1[li],k1[li],r,li); t_fin(cq); out.push_back(rd_natE(r)); }
            return out; };

        // HE_PROF_DRAIN: capture the FULL device trace of ONE ct_mul with NO dropped markers, by draining
        // the profiler buffer every K dispatches (tt-metal: "run read device profiler more often").
        if (std::getenv("HE_PROF_DRAIN")) {
            fmt::print("computing CPU oracle...\n"); auto ref=cpu_hemul();
            (void)ref;   // (correctness validated by HE_TRACE2; here we only want a clean trace)
            CACHE_ON=true; CACHE_REC=true;
            // NCORES, not a hardcoded "4-core": this label arrived from the CROSS package assuming the
            // drain capture is always run at NTT_CORES=4, so it announced "4-core" while running 16 --
            // directly contradicting the grid line printed above it.
            fmt::print("REC: warm+cache workloads ({}-core mul, NO 8-core inv8192)...\n", NCORES);
            auto g0=dev_hemul_res(); (void)g0; CACHE_REC=false;
            bool ok=true;   // NOTE: skip the 8-core inv8192 validation so the capture holds ONLY mul zones
            // clean the profiler buffer + CSV, then a DRAINED capture of one cached mul
            std::string csv = std::string(std::getenv("TT_METAL_HOME")) + "/generated/profiler/.logs/profile_log_device.csv";
            distributed::Finish(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh); std::remove(csv.c_str());
            int K=100; if(const char* e=std::getenv("HE_DRAIN_K")){ int v=std::atoi(e); if(v>0) K=v; }
            prof::drain_ctr=0; prof::drain_every=K;
            prof::drain_fn=[&]{ distributed::Finish(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh); };   // drain before overflow
            fmt::print("DRAINED capture of ONE cached mul (drain every {} dispatches)...\n", K);
            dev_hemul_res();
            prof::drain_every=0; distributed::Finish(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);   // final drain
            fmt::print("HE_PROF_DRAIN: FULL trace (no dropped markers) -> {}\n", csv);
            pass=ok; pass&=mesh->close(); if(pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n"); return pass?0:1;
        }

        // T2.5: Metal Trace on the cached path. REC (warm+cache) -> CAP (record traces) -> REP (replay).
        // End-to-end LATENCY = device-busy span = sum of blocking segment-replay walls (host FBC/tiling excluded).
        if (std::getenv("HE_TRACE2")) {
            fmt::print("computing CPU oracle...\n"); auto ref=cpu_hemul();
            CACHE_ON=true;
            CACHE_REC=true;  TR_PHASE=0; fmt::print("REC: warm+cache workloads...\n"); auto g0=dev_hemul_res(); (void)g0; CACHE_REC=false;
            TR_PHASE=1; t_fin(cq); fmt::print("CAP: recording traces...\n"); auto g1=dev_hemul_res(); (void)g1;
            if(seg_open){ mesh->end_mesh_trace(0, SEG.back()); seg_open=false; } t_fin(cq);
            fmt::print("captured {} trace segments (from {} cached workloads)\n", SEG.size(), WC.size());
            TR_PHASE=2; auto gote=dev_hemul_res();   // replay -> validate
            std::vector<std::vector<uint32_t>> got; for(uint32_t r=0;r<gote.size();++r){ uint32_t li=qidx[r%qidx.size()]; got.push_back(inv8192(gote[r],li)); }
            uint32_t mism=0,total=0; for(uint32_t r=0;r<ref.size();++r) for(uint32_t n=0;n<N;++n){ ++total; if(got[r][n]!=ref[r][n]){ if(mism<8) fmt::print("  row{} n{} got={} exp={}\n",r,n,got[r][n],ref[r][n]); ++mism; } }
            bool ok=(mism==0); fmt::print("TRACED RESIDENT device-vs-oracle = {}/{}  [{}]\n",mism,total,ok?"PASS":"FAIL");
            int it=5; if(const char* e=std::getenv("HE_ITERS")){int v=std::atoi(e); if(v>0) it=v;}
            // (1) host wall (non-blocking replays)
            dev_hemul_res(); prof::reset(); auto t0=std::chrono::high_resolution_clock::now();
            for(int i=0;i<it;++i) dev_hemul_res(); auto t1=std::chrono::high_resolution_clock::now();
            double wall=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/1e6/it;
            // (2) device-busy span: blocking segment replays, sum their walls (host FBC/tiling NOT counted)
            MEASURE_DEV=true; prof::g_replay=0; prof::n_replay=0; for(int i=0;i<it;++i) dev_hemul_res(); MEASURE_DEV=false;
            double devbusy=prof::g_replay/it;
            fmt::print("TIMING: TRACED RESIDENT Set-B HE-mul\n");
            fmt::print("  host wall            : {:.3f} ms/mul  ({} cores, {} iters)\n", wall, NCORES, it);
            fmt::print("  DEVICE-BUSY span     : {:.3f} ms/mul  ({} segment replays)  <-- end-to-end latency (tiling+host-FBC excluded)\n", devbusy, (double)prof::n_replay/it);
            if(std::getenv("HE_PROF2")){   // profile ONE replay: pure-device-busy (ZONE union) + device-span (first->last)
                t_fin(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);
                std::string csv = std::string(std::getenv("TT_METAL_HOME")) + "/generated/profiler/.logs/profile_log_device.csv";
                std::remove(csv.c_str());
                dev_hemul_res();                                  // one traced-replay mul
                t_fin(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);
                fmt::print("HE_PROF2: profiled ONE traced replay ({} cores) -> {}\n", NCORES, csv);
            }
            pass=ok; pass&=mesh->close(); if(pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n"); return pass?0:1;
        }

        // T2: persistent-program CACHED resident path — build workloads on mul#1, re-enqueue after.
        if (std::getenv("HE_RES_CACHE")) {
            fmt::print("computing CPU oracle...\n"); auto ref=cpu_hemul();
            CACHE_ON=true; CACHE_REC=true;
            fmt::print("recording persistent workloads (mul #1)...\n"); auto g1=dev_hemul_res(); (void)g1;
            CACHE_REC=false;
            fmt::print("cached {} workloads, pool = {} buf-groups\n", WC.size(), POOL.size());
            auto gote=dev_hemul_res();   // replay (re-enqueue cached workloads)
            std::vector<std::vector<uint32_t>> got; for(uint32_t r=0;r<gote.size();++r){ uint32_t li=qidx[r%qidx.size()]; got.push_back(inv8192(gote[r],li)); }
            uint32_t mism=0,total=0; for(uint32_t r=0;r<ref.size();++r) for(uint32_t n=0;n<N;++n){ ++total; if(got[r][n]!=ref[r][n]){ if(mism<8) fmt::print("  row{} n{} got={} exp={}\n",r,n,got[r][n],ref[r][n]); ++mism; } }
            bool ok=(mism==0); fmt::print("CACHED RESIDENT device-vs-oracle = {}/{}  [{}]\n",mism,total,ok?"PASS":"FAIL");
            int it=3; if(const char* e=std::getenv("HE_ITERS")){int v=std::atoi(e); if(v>0) it=v;}
            dev_hemul_res(); prof::reset(); auto t0=std::chrono::high_resolution_clock::now();
            for(int i=0;i<it;++i) dev_hemul_res(); auto t1=std::chrono::high_resolution_clock::now();
            double wall=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/1e6/it;
            fmt::print("TIMING: CACHED RESIDENT Set-B HE-mul  {:.3f} ms/mul  ({} cores, {} iters)\n", wall, NCORES, it);
            auto pc=[&](const char*nm,double g,long n){ fmt::print("  {:<24} {:8.2f} ms/mul  {:5.1f}%   {:6.1f} calls  ({:.3f} ms/call)\n",
                       nm, g/it, 100*g/it/wall, (double)n/it, n?g/(double)n:0.0); };
            fmt::print("PHASE BREAKDOWN (host wall):\n");
            pc("program build",    prof::g_build, prof::n_build); pc("enqueue (dispatch)",prof::g_enq,prof::n_enq);
            pc("upload  (H->D wr)", prof::g_wr,    prof::n_wr);    pc("readback(D->H rd)", prof::g_rd, prof::n_rd);
            pc("Finish  (drain)",   prof::g_fin,   prof::n_fin);
            double acct=(prof::g_build+prof::g_enq+prof::g_wr+prof::g_rd+prof::g_fin)/it;
            fmt::print("  {:<24} {:8.2f} ms/mul  {:5.1f}%  (host misc: tilize/CPU FBC/vector ops)\n","unaccounted", wall-acct, 100*(wall-acct)/wall);
            pass=ok; pass&=mesh->close(); if(pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n"); return pass?0:1;
        }

        if (std::getenv("HE_RES")) {
            fmt::print("computing CPU oracle...\n"); auto ref=cpu_hemul();
            fmt::print("running RESIDENT dev_hemul_res...\n"); auto gote=dev_hemul_res();
            std::vector<std::vector<uint32_t>> got; for(uint32_t r=0;r<gote.size();++r){ uint32_t li=qidx[r%qidx.size()]; got.push_back(inv8192(gote[r],li)); }
            uint32_t mism=0,total=0; for(uint32_t r=0;r<ref.size();++r) for(uint32_t n=0;n<N;++n){ ++total; if(got[r][n]!=ref[r][n]){ if(mism<8) fmt::print("  row{} n{} got={} exp={}\n",r,n,got[r][n],ref[r][n]); ++mism; } }
            bool ok=(mism==0); fmt::print("RESIDENT device-vs-oracle = {}/{}  [{}]\n",mism,total,ok?"PASS":"FAIL");
            int it=1; if(const char* e=std::getenv("HE_ITERS")){int v=std::atoi(e); if(v>0) it=v;}
            dev_hemul_res(); prof::reset(); auto t0=std::chrono::high_resolution_clock::now();
            for(int i=0;i<it;++i) dev_hemul_res(); auto t1=std::chrono::high_resolution_clock::now();
            double wall=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/1e6/it;
            fmt::print("TIMING: RESIDENT Set-B HE-mul  {:.3f} ms/mul  ({} cores, {} iters)\n", wall, NCORES, it);
            auto pc=[&](const char*nm,double g,long n){ fmt::print("  {:<24} {:8.2f} ms/mul  {:5.1f}%   {:6.1f} calls  ({:.3f} ms/call)\n",
                       nm, g/it, 100*g/it/wall, (double)n/it, n?g/(double)n:0.0); };
            fmt::print("PHASE BREAKDOWN (host wall, idle-cause attribution):\n");
            pc("program build",    prof::g_build, prof::n_build);
            pc("enqueue (dispatch)",prof::g_enq,   prof::n_enq);
            pc("upload  (H->D wr)", prof::g_wr,    prof::n_wr);
            pc("readback(D->H rd)", prof::g_rd,    prof::n_rd);
            pc("Finish  (drain)",   prof::g_fin,   prof::n_fin);
            double acct=(prof::g_build+prof::g_enq+prof::g_wr+prof::g_rd+prof::g_fin)/it;
            fmt::print("  {:<24} {:8.2f} ms/mul  {:5.1f}%  (host misc: tilize/CPU FBC/vector ops)\n",
                       "unaccounted", wall-acct, 100*(wall-acct)/wall);
            pass=ok; pass&=mesh->close(); if(pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n"); return pass?0:1;
        }

        auto dev_hemul=[&](){   // returns (r0,r1) EVAL, 7 Q-limbs (CROSS keeps result in eval)
            auto c0r=rescale_dev(c0e),c1r=rescale_dev(c1e),d0r=rescale_dev(d0e),d1r=rescale_dev(d1e);
            std::vector<std::vector<uint32_t>> e0(LQP),e1(LQP),e2(LQP);           // tensor: pointwise, 0 NTT
            for (auto li:qidx){ uint64_t q=QP[li]; e0[li]=pointwise8192(c0r[li],d0r[li],li); e2[li]=pointwise8192(c1r[li],d1r[li],li);
                auto t01=pointwise8192(c0r[li],d1r[li],li),t10=pointwise8192(c1r[li],d0r[li],li);
                e1[li].resize(N); for(uint32_t n=0;n<N;++n) e1[li][n]=(uint32_t)(((u128)t01[n]+t10[n])%q); }
            // keyswitch e2 (eval->eval)
            std::vector<std::vector<uint32_t>> e2c(LQP); for (auto li:qidx) e2c[li]=inv8192(e2[li],li);    // 7 INTT
            std::vector<std::vector<uint32_t>> acc0(LQP,std::vector<uint32_t>(N,0)),acc1(LQP,std::vector<uint32_t>(N,0));
            for (uint32_t j=0;j<DNUM;++j){ std::vector<std::vector<uint32_t>> xs; for(auto li:group[j]) xs.push_back(e2c[li]);
                auto o=fbc8192(xs,group[j],comp[j]);                              // mod-up BConv (coeff)
                std::vector<std::vector<uint32_t>> vje(LQP);
                for (auto li:group[j]) vje[li]=e2[li];                            // own limbs: reuse e2 eval
                for (uint32_t k=0;k<comp[j].size();++k) vje[comp[j][k]]=fwd8192(o[k],comp[j][k]);   // |comp| NTT
                for (auto li:qp){ uint64_t q=QP[li]; auto p0=pointwise8192(vje[li],evk0f[j][li],li),p1=pointwise8192(vje[li],evk1f[j][li],li);
                    for(uint32_t n=0;n<N;++n){ acc0[li][n]=(uint32_t)(((u128)acc0[li][n]+p0[n])%q); acc1[li][n]=(uint32_t)(((u128)acc1[li][n]+p1[n])%q);} } }
            auto moddown=[&](std::vector<std::vector<uint32_t>>& acc){
                std::vector<std::vector<uint32_t>> ac(LQP); for (auto li:qp) ac[li]=inv8192(acc[li],li);   // 11 INTT
                std::vector<std::vector<uint32_t>> aP; for(auto p:pidx) aP.push_back(ac[p]);
                auto f=fbc8192(aP,pidx,qidx);
                std::vector<std::vector<uint32_t>> ke(LQP);
                for (uint32_t k=0;k<qidx.size();++k){ uint32_t li=qidx[k]; uint64_t q=QP[li]; uint64_t Pinv=modpow((uint64_t)(Pprod%q),q-2,q);
                    std::vector<uint32_t> kc(N); for(uint32_t n=0;n<N;++n) kc[n]=(uint32_t)((((u128)ac[li][n]+q-f[k][n])%q)*Pinv%q);
                    ke[li]=fwd8192(kc,li); }                                       // 7 NTT (back to eval)
                return ke; };
            auto k0=moddown(acc0),k1=moddown(acc1);
            std::vector<std::vector<uint32_t>> out;
            for (auto li:qidx){ uint64_t q=QP[li]; std::vector<uint32_t> r(N); for(uint32_t n=0;n<N;++n) r[n]=(uint32_t)(((u128)e0[li][n]+k0[li][n])%q); out.push_back(r); }
            for (auto li:qidx){ uint64_t q=QP[li]; std::vector<uint32_t> r(N); for(uint32_t n=0;n<N;++n) r[n]=(uint32_t)(((u128)e1[li][n]+k1[li][n])%q); out.push_back(r); }
            return out; };   // eval, rows = [r0 over qidx][r1 over qidx]

        // ================= HE_PROFILE: isolate ONE dev_hemul for on-device profiling =================
        // Flush keygen/setup profiler data + clear the CSV, then profile exactly one mul.
        if (std::getenv("HE_PROFILE")) {
            t_fin(cq);
            tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);           // dump keygen/setup, clear device buffer
            std::string csv = std::string(std::getenv("TT_METAL_HOME")) + "/generated/profiler/.logs/profile_log_device.csv";
            std::remove(csv.c_str());                                     // discard keygen prefix
            fmt::print("HE_PROFILE: profiling ONE dev_hemul ({} cores)...\n", NCORES);
            auto pr = dev_hemul();
            t_fin(cq);
            tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);           // CSV now = one mul only
            fmt::print("HE_PROFILE done -> {}\n", csv);
            pass = true; pass &= mesh->close();
            if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
            return pass ? 0 : 1;
        }

        // ================= validate (device eval -> coeff) vs CPU oracle =================
        fmt::print("computing CPU oracle...\n");
        auto ref = cpu_hemul();
        fmt::print("running device HE-mul (CROSS-aligned)...\n");
        auto gote = dev_hemul();
        // INTT device eval result back to coeff for comparison (decrypt-side; not part of the mul)
        std::vector<std::vector<uint32_t>> got;
        for (uint32_t r=0;r<gote.size();++r){ uint32_t li=qidx[r%qidx.size()]; got.push_back(inv8192(gote[r],li)); }
        uint32_t mism=0,total=0;
        for (uint32_t r=0;r<ref.size();++r) for (uint32_t n=0;n<N;++n){ ++total; if(got[r][n]!=ref[r][n]){ if(mism<8) fmt::print("  row{} n{} got={} exp={}\n",r,n,got[r][n],ref[r][n]); ++mism; } }
        bool ok=(mism==0);
        fmt::print("CROSS-aligned Set-B HE-mul  device-vs-oracle = {}/{}  [{}]\n", mism, total, ok?"PASS":"FAIL");

        int iters=3; if(const char* e=std::getenv("HE_ITERS")){ int v=std::atoi(e); if(v>0) iters=v; }
        dev_hemul();   // warm
        prof::reset();
        auto t0=std::chrono::high_resolution_clock::now();
        for (int i=0;i<iters;++i) dev_hemul();
        auto t1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/1e6/iters;
        fmt::print("TIMING: CROSS-aligned Set-B HE-mul (eval-resident, tensor=pointwise, dnum={})  {:.3f} ms/mul  ({} cores, {} iters)\n", DNUM, ms, NCORES, iters);
        {   auto pc=[&](const char*nm,double g,long n){ fmt::print("  {:<24} {:8.2f} ms/mul  {:5.1f}%   {:6.1f} calls  ({:.3f} ms/call)\n",
                       nm, g/iters, 100*g/iters/ms, (double)n/iters, n?g/(double)n:0.0); };
            fmt::print("PHASE BREAKDOWN (host wall, idle-cause attribution):\n");
            pc("program build",    prof::g_build, prof::n_build);
            pc("enqueue (dispatch)",prof::g_enq,   prof::n_enq);
            pc("upload  (H->D wr)", prof::g_wr,    prof::n_wr);
            pc("readback(D->H rd)", prof::g_rd,    prof::n_rd);
            pc("Finish  (drain)",   prof::g_fin,   prof::n_fin);
            double acct=(prof::g_build+prof::g_enq+prof::g_wr+prof::g_rd+prof::g_fin)/iters;
            fmt::print("  {:<24} {:8.2f} ms/mul  {:5.1f}%  (host misc: tilize/CPU FBC/vector ops)\n",
                       "unaccounted", ms-acct, 100*(ms-acct)/ms); }
        fmt::print("RESULT: CROSS-aligned Set-B HE-mul {} (same sub-kernel workload: tensor 0 NTT, drop-prime rescale, dnum keyswitch)\n", ok?"WORKS":"WRONG");
        pass = ok;
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
