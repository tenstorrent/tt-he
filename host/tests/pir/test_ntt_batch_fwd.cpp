// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BATCHED forward NTT — the online NTT batched phase-major, same pattern as the mbatch scan matmul.
// A length-N negacyclic NTT = reshape → stage-A modular-matmul(WfA) → twiddle-mul(Tm) → stage-C
// modular-matmul(Wr) → dereindex.  The two matmul stages ARE modular matmuls (fused_matmul_a/_c);
// with the twiddle W SHARED across all polys they batch exactly like the scan:
//   stage A : matmul_modq_mbatch      (W = WfA, replicated per poly — LHS streamed)
//   twiddle : modmul_compute          (loops tiles; Tm replicated per poly)
//   stage C : matmul_modq_mbatch_c    (W = Wr, 16 tiles SHARED/resident across all polys)
// So NB polys cost 3 device programs per core-round instead of 3·NB.  Validated bit-exact vs the
// per-poly device fwd; times batched vs per-poly.  n=1024 (1 tile/poly), single RNS channel.
//
//   ./build/test_ntt_batch_fwd [NP] [M] [q] [seed]      (NP polys, M per core/round)

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
#include "pir_permute.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
#ifndef OVERRIDE_KERNEL_PREFIX_IR
#define OVERRIDE_KERNEL_PREFIX_IR ""
#endif
#define KIR OVERRIDE_KERNEL_PREFIX_IR

namespace {
constexpr uint32_t T = 32, NB = 4, N = 1024, ELEMS = 1024, I8 = ELEMS, I32 = ELEMS * 4;
const uint64_t POW[4] = {1, 128, 16384, 2097152};
using u128 = unsigned __int128;
using Buf = std::shared_ptr<distributed::MeshBuffer>;
uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) { uint64_t r = 1; b %= m;
    while (e) { if (e & 1) r = (u128)r * b % m; b = (u128)b * b % m; e >>= 1; } return r; }
uint64_t prim_root(uint64_t q) { uint64_t phi = q - 1, n = phi; std::vector<uint64_t> f;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) { f.push_back(d); while (n % d == 0) n /= d; }
    if (n > 1) f.push_back(n);
    for (uint64_t g = 2; g < q; ++g) { bool ok = true; for (uint64_t p : f) if (modpow(g, phi / p, q) == 1) ok = false;
        if (ok) return g; } throw std::runtime_error("no root"); }
int8_t limb8(uint32_t v, uint32_t p) { return (int8_t)((v >> (7 * p)) & 127); }
std::vector<uint32_t> til(const std::vector<uint32_t>& v) { return pir::tilize_poly(v, 1); }
std::vector<uint32_t> unt(const std::vector<uint32_t>& v) { return pir::untilize_poly(v, 1); }
}  // namespace

int main(int argc, char** argv) {
    const uint32_t NP = (argc > 1) ? std::stoul(argv[1]) : 2048;  // polys to transform
    const uint32_t M  = (argc > 2) ? std::stoul(argv[2]) : 32;    // polys per core per program
    const uint64_t q  = (argc > 3) ? std::stoull(argv[3]) : 167772161ULL;
    const uint32_t seed = (argc > 4) ? std::stoul(argv[4]) : 1;
    mont::MontgomeryContext ctx((uint32_t)q);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> di(0, (uint32_t)q - 1);

    std::vector<std::vector<uint32_t>> X(NP, std::vector<uint32_t>(N));   // input polys (coeff)
    for (auto& p : X) for (auto& v : p) v = di(rng);

    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX = grid.x, GY = grid.y;
        constexpr CoreCoord core = {0, 0};
        auto mkbuf = [&](size_t nt, uint32_t pg) { distributed::DeviceLocalBufferConfig d{.page_size = pg, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * pg}, d, mesh.get()); };
        auto mkcb = [&](Program& p, CoreRange cr, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t pg) {
            CircularBufferConfig c(depth * pg, {{idx, fmt}}); c.set_page_size(idx, pg); CreateCircularBuffer(p, cr, c); };
        auto wrbuf = [&](auto buf, const std::vector<uint32_t>& v) { distributed::EnqueueWriteMeshBuffer(cq, buf, v, false); };
        auto rdbuf = [&](auto buf) { std::vector<uint32_t> v; distributed::EnqueueReadMeshBuffer(cq, v, buf, true); return v; };

        // ---- twiddles / folds (from test_initdim) ----
        uint64_t g=prim_root(q), w=modpow(g,(q-1)/N,q), w32=modpow(g,(q-1)/T,q), psi=modpow(g,(q-1)/(2*N),q);
        std::vector<uint32_t> Wc(ELEMS),Wa(ELEMS),Tw(ELEMS);
        for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b){ uint32_t wv=(uint32_t)modpow(w32,(uint64_t)a*b,q); Wc[a*T+b]=wv; Wa[a*T+b]=(uint32_t)((u128)wv*modpow(psi,32ull*b,q)%q); }
        for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b) Tw[a*T+b]=(uint32_t)((u128)modpow(w,(uint64_t)a*b,q)*modpow(psi,b,q)%q);
        uint32_t Q=(uint32_t)q, QINV=ctx.q_inv_32(), QLO=Q&0xFFFF, QHI=Q>>16, K14=ctx.to_montgomery(16384u);
        auto foldA=[&](const std::vector<uint32_t>& Wm){ std::vector<int8_t> o(NB*NB*ELEMS);
            for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<uint32_t> l(ELEMS);
                for (uint32_t i=0;i<ELEMS;++i) l[i]=limb8((uint32_t)((POW[r]*Wm[i])%Q),p); auto t=til(l);
                for (uint32_t i=0;i<ELEMS;++i) o[(p*NB+r)*ELEMS+i]=(int8_t)t[i]; } return o; };
        auto foldC=[&](const std::vector<uint32_t>& Wm){ std::vector<int8_t> o(NB*NB*ELEMS);
            for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<uint32_t> l(ELEMS);
                for (uint32_t i=0;i<ELEMS;++i) l[i]=limb8((uint32_t)((POW[p]*Wm[i])%Q),r); auto t=til(l);
                for (uint32_t i=0;i<ELEMS;++i) o[(p*NB+r)*ELEMS+i]=(int8_t)t[i]; } return o; };
        auto mont_til=[&](const std::vector<uint32_t>& v){ std::vector<uint32_t> m(ELEMS); for(uint32_t i=0;i<ELEMS;++i) m[i]=ctx.to_montgomery(v[i]); return til(m); };
        auto WfA_h=foldA(Wa), Wr_h=foldC(Wc); auto Tm_h=mont_til(Tw);
        auto reshape=[&](const std::vector<uint32_t>& x){ std::vector<uint32_t> R(ELEMS); for (uint32_t n2=0;n2<T;++n2) for (uint32_t n1=0;n1<T;++n1) R[n2*T+n1]=x[n1+T*n2]; return til(R); };
        auto dereindex=[&](const std::vector<uint32_t>& raw){ auto H=unt(raw); std::vector<uint32_t> Xo(N); for (uint32_t j1=0;j1<T;++j1) for (uint32_t j2=0;j2<T;++j2) Xo[j2+T*j1]=H[j2*T+j1]; return Xo; };

        // ======== per-poly device fwd (golden) ========
        auto b_in=mkbuf(1,I32), b_g=mkbuf(1,I32), b_gp=mkbuf(1,I32), b_h=mkbuf(1,I32);
        auto bWfA=mkbuf(NB*NB,I8), bWr=mkbuf(NB*NB,I8), bTm=mkbuf(1,I32);
        distributed::EnqueueWriteMeshBuffer(cq,bWfA,WfA_h,false); distributed::EnqueueWriteMeshBuffer(cq,bWr,Wr_h,false); wrbuf(bTm,Tm_h);
        auto fusedA1=[&](auto xin,auto wb,auto out){ Program p=CreateProgram();
            mkcb(p,core,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,core,CBIndex::c_1,tt::DataFormat::Int8,2,I8); mkcb(p,core,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcb(p,core,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,core,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*xin).append_to(rc); TensorAccessorArgs(*wb).append_to(rc);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)xin->address(),(uint32_t)wb->address()}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{Q,QINV,QLO,QHI,K14});
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };
        auto fusedC1=[&](auto gp,auto wb,auto out){ Program p=CreateProgram();
            mkcb(p,core,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,core,CBIndex::c_1,tt::DataFormat::Int8,NB*NB,I8); mkcb(p,core,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcb(p,core,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,core,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*gp).append_to(rc); TensorAccessorArgs(*wb).append_to(rc);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)gp->address(),(uint32_t)wb->address()}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{Q,QINV,QLO,QHI,K14});
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };
        auto modmul1=[&](auto a,auto b,auto out){ Program p=CreateProgram();
            mkcb(p,core,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,core,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,core,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)a->address(),(uint32_t)b->address(),1u,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{1u,Q,QINV,QLO,QHI});
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };
        auto fwd1=[&](const std::vector<uint32_t>& x){ wrbuf(b_in,reshape(x)); fusedA1(b_in,bWfA,b_g); modmul1(b_g,bTm,b_gp); fusedC1(b_gp,bWr,b_h); return dereindex(rdbuf(b_h)); };

        std::vector<std::vector<uint32_t>> gold(NP);
        for (uint32_t i = 0; i < NP; ++i) gold[i] = fwd1(X[i]);

        // ======== batched fwd (mbatch A + modmul + mbatch_c C) ========
        // shared/replicated twiddle buffers; batched Xin (reshaped), Tm replicated per poly.
        auto bWfA_rep = mkbuf((size_t)M * NB * NB, I8);   // WfA replicated M× (stage A streams M·16)
        { std::vector<int8_t> rep((size_t)M * NB * NB * ELEMS); for (uint32_t m=0;m<M;++m) std::copy(WfA_h.begin(),WfA_h.end(),rep.begin()+(size_t)m*NB*NB*ELEMS);
          distributed::EnqueueWriteMeshBuffer(cq,bWfA_rep,rep,false); }
        auto bWr_share = bWr;                            // stage C shares the 16 Wr tiles
        auto bXin = mkbuf(NP, I32), bG = mkbuf(NP, I32), bGp = mkbuf(NP, I32), bH = mkbuf(NP, I32);
        auto bTm_rep = mkbuf(NP, I32);
        { std::vector<uint32_t> xb((size_t)NP*ELEMS), tb((size_t)NP*ELEMS);
          for (uint32_t i=0;i<NP;++i){ auto r=reshape(X[i]); std::copy(r.begin(),r.end(),xb.begin()+(size_t)i*ELEMS); std::copy(Tm_h.begin(),Tm_h.end(),tb.begin()+(size_t)i*ELEMS); }
          wrbuf(bXin,xb); wrbuf(bTm_rep,tb); }

        auto stageA=[&](uint32_t gx,uint32_t gy,uint32_t Mb, Buf in, Buf w, Buf out){ const uint32_t C=gx*gy, per=C*Mb;
            for (uint32_t base=0;base<NP;base+=per){ Program p=CreateProgram(); CoreRange cores(CoreCoord{0,0},CoreCoord{gx-1,gy-1});
                mkcb(p,cores,CBIndex::c_0,tt::DataFormat::UInt32,Mb,I32); mkcb(p,cores,CBIndex::c_1,tt::DataFormat::Int8,32,I8); mkcb(p,cores,CBIndex::c_2,tt::DataFormat::Int8,Mb*NB,I8); mkcb(p,cores,CBIndex::c_3,tt::DataFormat::Int32,Mb*NB,I32); mkcb(p,cores,CBIndex::c_16,tt::DataFormat::UInt32,Mb,I32);
                std::vector<uint32_t> rc; TensorAccessorArgs(*in).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
                auto rd=CreateKernel(p,KIR "dataflow/reader_mbatch.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
                auto wr=CreateKernel(p,KIR "dataflow/writer_i32.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,KIR "compute/matmul_modq_mbatch.cpp",cores,ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t gyi=0;gyi<gy;++gyi) for (uint32_t gxi=0;gxi<gx;++gxi){ uint32_t idx=gyi*gx+gxi, o=base+idx*Mb; if (o>=NP) continue; uint32_t mh=std::min(Mb,NP-o); CoreCoord cc{gxi,gyi};
                    SetRuntimeArgs(p,rd,cc,{(uint32_t)in->address(),(uint32_t)w->address(),o,0u,mh});   // w_base=0: replicated block reused per round
                    SetRuntimeArgs(p,wr,cc,{(uint32_t)out->address(),mh,o}); SetRuntimeArgs(p,cp,cc,{Q,QINV,QLO,QHI,K14,mh}); }
                distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); } };
        auto stageC=[&](uint32_t gx,uint32_t gy,uint32_t Mb, Buf in, Buf w, Buf out){ const uint32_t C=gx*gy, per=C*Mb;
            for (uint32_t base=0;base<NP;base+=per){ Program p=CreateProgram(); CoreRange cores(CoreCoord{0,0},CoreCoord{gx-1,gy-1});
                mkcb(p,cores,CBIndex::c_0,tt::DataFormat::UInt32,Mb,I32); mkcb(p,cores,CBIndex::c_1,tt::DataFormat::Int8,NB*NB,I8); mkcb(p,cores,CBIndex::c_2,tt::DataFormat::Int8,Mb*NB,I8); mkcb(p,cores,CBIndex::c_3,tt::DataFormat::Int32,Mb*NB,I32); mkcb(p,cores,CBIndex::c_16,tt::DataFormat::UInt32,Mb,I32);
                std::vector<uint32_t> rc; TensorAccessorArgs(*in).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
                auto rd=CreateKernel(p,KIR "dataflow/reader_mbatch_c.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
                auto wr=CreateKernel(p,KIR "dataflow/writer_i32.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,KIR "compute/matmul_modq_mbatch_c.cpp",cores,ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t gyi=0;gyi<gy;++gyi) for (uint32_t gxi=0;gxi<gx;++gxi){ uint32_t idx=gyi*gx+gxi, o=base+idx*Mb; if (o>=NP) continue; uint32_t mh=std::min(Mb,NP-o); CoreCoord cc{gxi,gyi};
                    SetRuntimeArgs(p,rd,cc,{(uint32_t)in->address(),(uint32_t)w->address(),o,0u,mh}); SetRuntimeArgs(p,wr,cc,{(uint32_t)out->address(),mh,o}); SetRuntimeArgs(p,cp,cc,{Q,QINV,QLO,QHI,K14,mh}); }
                distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); } };
        auto modmulB=[&](uint32_t gx,uint32_t gy,uint32_t Mb, Buf in, Buf tm, Buf out){ const uint32_t C=gx*gy, per=C*Mb;   // G ⊙ Tm → Gp, batched
            for (uint32_t base=0;base<NP;base+=per){ Program p=CreateProgram(); CoreRange cores(CoreCoord{0,0},CoreCoord{gx-1,gy-1});
                mkcb(p,cores,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,cores,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,cores,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp",cores,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t gyi=0;gyi<gy;++gyi) for (uint32_t gxi=0;gxi<gx;++gxi){ uint32_t idx=gyi*gx+gxi, o=base+idx*Mb; if (o>=NP) continue; uint32_t mh=std::min(Mb,NP-o); CoreCoord cc{gxi,gyi};
                    SetRuntimeArgs(p,rd,cc,{(uint32_t)in->address(),(uint32_t)tm->address(),mh,o}); SetRuntimeArgs(p,wr,cc,{(uint32_t)out->address(),mh,o}); SetRuntimeArgs(p,cp,cc,{mh,Q,QINV,QLO,QHI}); }
                distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); } };

        auto run_fwd=[&](uint32_t gx,uint32_t gy)->double{ auto t0=std::chrono::high_resolution_clock::now();
            stageA(gx,gy,M,bXin,bWfA_rep,bG); modmulB(gx,gy,M,bG,bTm_rep,bGp); stageC(gx,gy,M,bGp,bWr_share,bH); distributed::Finish(cq);
            auto t1=std::chrono::high_resolution_clock::now(); return std::chrono::duration<double,std::milli>(t1-t0).count(); };

        run_fwd(1,1);
        auto Hraw = rdbuf(bH);
        std::vector<std::vector<uint32_t>> Fout(NP);
        uint32_t mism=0;
        for (uint32_t i=0;i<NP;++i){ Fout[i]=dereindex(std::vector<uint32_t>(Hraw.begin()+(size_t)i*ELEMS,Hraw.begin()+(size_t)(i+1)*ELEMS));
            for (uint32_t j=0;j<N;++j) if (Fout[i][j]!=gold[i][j]) { if (mism<8) fmt::print("  poly{} [{}] got={} exp={}\n",i,j,Fout[i][j],gold[i][j]); ++mism; } }
        fmt::print("batched fwd NTT validation: {}  ({} bad)\n", mism==0?"OK":"FAILED", mism);
        pass &= (mism==0);

        // ======== batched INVERSE NTT (symmetric: same kernels, inverse fold buffers) ========
        // round-trip check: batched inv(batched fwd(X)) == X.
        uint64_t wi=mont::modular_inverse(w,Q),w32i=mont::modular_inverse(w32,Q),psii=mont::modular_inverse(psi,Q),Ninv=mont::modular_inverse(N,Q);
        std::vector<uint32_t> Wci(ELEMS),Wai(ELEMS),Twi(ELEMS);
        for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b){ uint32_t wv=(uint32_t)modpow(w32i,(uint64_t)a*b,Q); Wci[a*T+b]=(uint32_t)((u128)wv*modpow(psii,32ull*b,Q)%Q); Wai[a*T+b]=(uint32_t)((u128)wv*Ninv%Q); }
        for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b) Twi[a*T+b]=(uint32_t)((u128)modpow(wi,(uint64_t)a*b,Q)*modpow(psii,a,Q)%Q);
        auto WfAi_h=foldA(Wai), Wri_h=foldC(Wci); auto Tmi_h=mont_til(Twi);
        auto bWfAi_rep=mkbuf((size_t)M*NB*NB,I8);
        { std::vector<int8_t> rep((size_t)M*NB*NB*ELEMS); for (uint32_t m=0;m<M;++m) std::copy(WfAi_h.begin(),WfAi_h.end(),rep.begin()+(size_t)m*NB*NB*ELEMS); distributed::EnqueueWriteMeshBuffer(cq,bWfAi_rep,rep,false); }
        auto bWri=mkbuf(NB*NB,I8); distributed::EnqueueWriteMeshBuffer(cq,bWri,Wri_h,false);
        auto bXin2=mkbuf(NP,I32), bTmi_rep=mkbuf(NP,I32), bH2=mkbuf(NP,I32);
        { std::vector<uint32_t> xb((size_t)NP*ELEMS), tb((size_t)NP*ELEMS);
          for (uint32_t i=0;i<NP;++i){ auto r=reshape(Fout[i]); std::copy(r.begin(),r.end(),xb.begin()+(size_t)i*ELEMS); std::copy(Tmi_h.begin(),Tmi_h.end(),tb.begin()+(size_t)i*ELEMS); }
          wrbuf(bXin2,xb); wrbuf(bTmi_rep,tb); }
        stageA(GX,GY,M,bXin2,bWfAi_rep,bG); modmulB(GX,GY,M,bG,bTmi_rep,bGp); stageC(GX,GY,M,bGp,bWri,bH2); distributed::Finish(cq);
        auto H2raw=rdbuf(bH2); uint32_t rmism=0;
        for (uint32_t i=0;i<NP;++i){ auto rt=dereindex(std::vector<uint32_t>(H2raw.begin()+(size_t)i*ELEMS,H2raw.begin()+(size_t)(i+1)*ELEMS));
            for (uint32_t j=0;j<N;++j) if (rt[j]!=X[i][j]) { if (rmism<8) fmt::print("  RT poly{} [{}] got={} exp={}\n",i,j,rt[j],X[i][j]); ++rmism; } }
        fmt::print("batched inv NTT round-trip (inv(fwd(x))==x): {}  ({} bad)\n", rmism==0?"OK":"FAILED", rmism);
        pass &= (rmism==0);
        auto run_batched=[&](uint32_t gx,uint32_t gy){ return run_fwd(gx,gy); };

        fmt::print("\n cores | gx×gy | time(ms) | NTTs/s   | speedup\n-------+-------+----------+----------+--------\n");
        struct Cfg{uint32_t gx,gy;}; std::vector<Cfg> cfgs={{1,1},{8,1},{8,2},{8,4},{8,8}}; double t1c=0;
        for (auto cf:cfgs){ if (cf.gx>GX||cf.gy>GY) continue; uint32_t Cc=cf.gx*cf.gy;
            run_batched(cf.gx,cf.gy); double best=1e18; for(int it=0;it<3;++it) best=std::min(best,run_batched(cf.gx,cf.gy));
            if (Cc==1) t1c=best;
            fmt::print(" {:4}  | {:2}×{:<2} | {:8.2f} | {:8.0f} | {:5.2f}×\n", Cc, cf.gx, cf.gy, best, NP/best*1e3, t1c>0?t1c/best:1.0); }
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
    return pass ? 0 : 1;
}
