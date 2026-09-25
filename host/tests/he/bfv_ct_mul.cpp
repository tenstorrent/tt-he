// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BFV FULL MULTIPLY, device-chained (correctness mode) — plan docs/he/BFV_new_plan.md step 7.
//   ct_mul = tensor -> rescale ⌊t·e/Q⌉ -> relinearize -> add
// N=1024, Q=3 / P=3 (QP=6), aux S=4, sentinel m_sk, t=12289, single Tensix (n150).
//
// The tensor is threaded in from the host golden (§3 correctness mode) — its on-device
// 8-limb NTT + INTT wiring is the remaining perf-mode gap (§5). Everything downstream is
// on device: rescale x3 (reuses bfv_rescale sequence) -> relin(e~2) (reuses bfv_relin
// machinery, padded FBC) -> per-limb modadd. Validated bit-exact vs the golden (out0,out1)
// from reference/he/gen_ctmul_case.py, whose decrypt(out)==m1·m2 is checked in Python.
//   ./build/bfv_ct_mul [reference/he/ctmul_case.bin]      (env INJECT_GOLDEN=1 feeds golden e~2 into relin)
//
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
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
constexpr uint32_t T = 32, N = 1024, ELEMS = 1024, I8 = ELEMS, I32 = ELEMS * 4;
constexpr uint32_t LQ = 3, LP = 3, LQP = 6, LS = 4, NB = 4, FBCW = 4;
const uint64_t POW[4] = {1, 128, 16384, 2097152};
using u128 = unsigned __int128;
uint64_t modpow(uint64_t b, uint64_t e, uint64_t m){ u128 r=1,x=b%m; while(e){ if(e&1) r=r*x%m; x=x*x%m; e>>=1;} return (uint64_t)r; }
uint64_t modinv(uint64_t a, uint64_t m){ return modpow(a%m, m-2, m); }
bool is_prime(uint64_t n){ if(n<2) return false; for(uint64_t d=2;d*d<=n;++d) if(n%d==0) return false; return true; }
uint64_t prim_root(uint64_t q){ uint64_t phi=q-1,n=phi; std::vector<uint64_t> f;
    for(uint64_t d=2;d*d<=n;++d) if(n%d==0){ f.push_back(d); while(n%d==0) n/=d; } if(n>1) f.push_back(n);
    for(uint64_t g=2;g<q;++g){ bool ok=true; for(uint64_t p:f) if(modpow(g,phi/p,q)==1) ok=false; if(ok) return g; } throw std::runtime_error("no root"); }
int8_t limb(uint64_t v, uint32_t p){ return (int8_t)((v>>(7*p))&127); }
template <typename E> std::vector<E> tilize(const std::vector<E>& in){ std::vector<E> o(ELEMS);
    for(uint32_t r=0;r<T;++r) for(uint32_t c=0;c<T;++c) o[((r/16)*2+c/16)*256+(r%16)*16+(c%16)]=in[r*T+c]; return o; }
template <typename E> std::vector<E> untilize(const std::vector<E>& in){ std::vector<E> o(ELEMS);
    for(uint32_t r=0;r<T;++r) for(uint32_t c=0;c<T;++c) o[r*T+c]=in[((r/16)*2+c/16)*256+(r%16)*16+(c%16)]; return o; }
std::vector<uint32_t> reindex(const std::vector<uint32_t>& H){ std::vector<uint32_t> o(N);
    for(uint32_t j1=0;j1<T;++j1) for(uint32_t j2=0;j2<T;++j2) o[j2+T*j1]=H[j2*T+j1]; return o; }
}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const char* path = (argc > 1) ? argv[1] : "reference/he/ctmul_case.bin";
        bool inject = getenv("INJECT_GOLDEN") != nullptr;
        std::ifstream f(path, std::ios::binary);
        if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
        uint32_t Nf, t; f.read((char*)&Nf,4); f.read((char*)&t,4);
        uint32_t pr[11]; f.read((char*)pr,44);
        std::vector<uint64_t> Q(pr,pr+3), P(pr+3,pr+6), S(pr+6,pr+10); uint64_t Msk=pr[10];
        std::vector<uint64_t> QP = Q; QP.insert(QP.end(), P.begin(), P.end());
        auto rd_rows = [&](uint32_t n){ std::vector<std::vector<uint32_t>> v(n, std::vector<uint32_t>(ELEMS));
            for(uint32_t i=0;i<n;++i) f.read((char*)v[i].data(), ELEMS*4); return v; };   // tilized rows
        auto evk0 = rd_rows(LQP), evk1 = rd_rows(LQP);
        constexpr uint32_t LT = LQ+LS+1;                        // tensor base = Q(3)|S(4)|msk(1) = 8
        auto c0i=rd_rows(LT), c1i=rd_rows(LT), d0i=rd_rows(LT), d1i=rd_rows(LT);  // ct residues (tilized, plain)
        std::vector<std::vector<std::vector<uint32_t>>> eQ(3), eS(3), eM(3);
        for (uint32_t k=0;k<3;++k){ eQ[k]=rd_rows(LQ); eS[k]=rd_rows(LS); eM[k]=rd_rows(1); }
        std::vector<std::vector<std::vector<uint32_t>>> gTilde(3);
        for (uint32_t k=0;k<3;++k) gTilde[k]=rd_rows(LQ);
        auto gK0 = rd_rows(LQ), gK1 = rd_rows(LQ), gOut0 = rd_rows(LQ), gOut1 = rd_rows(LQ);

        std::vector<mont::MontgomeryContext> cQ, cS, cQP; mont::MontgomeryContext cM((uint32_t)Msk);
        for (auto q:Q) cQ.emplace_back((uint32_t)q);
        for (auto s:S) cS.emplace_back((uint32_t)s);
        for (auto q:QP) cQP.emplace_back((uint32_t)q);
        u128 Qp=1; for(auto q:Q) Qp*=q;  u128 Sp=1; for(auto s:S) Sp*=s;  u128 Pp=1; for(auto p:P) Pp*=p;
        std::vector<uint64_t> P8 = Q; P8.insert(P8.end(),S.begin(),S.end()); P8.push_back(Msk);   // tensor base (8)
        std::vector<mont::MontgomeryContext> C8; for(auto p:P8) C8.emplace_back((uint32_t)p);

        // ================= device setup =================
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core={0,0};
        auto mkbuf=[&](uint32_t n, uint32_t page=I32){ distributed::DeviceLocalBufferConfig d{.page_size=page,.buffer_type=BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size=n*page}, d, mesh.get()); };
        std::vector<distributed::MeshWorkload> wls; wls.reserve(2048);
        std::vector<std::shared_ptr<distributed::MeshBuffer>> keep;
        auto run=[&](Program&& p){ wls.emplace_back(); wls.back().add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wls.back(),false); };
        auto CB=[&](Program& p, CBIndex i, uint32_t d, tt::DataFormat fmt=tt::DataFormat::UInt32, uint32_t page=I32){ CircularBufferConfig c(d*page,{{i,fmt}}); c.set_page_size(i,page); CreateCircularBuffer(p,core,c); };
        auto RD=[&](Program& p,const char* k){ return CreateKernel(p, std::string(OVERRIDE_KERNEL_PREFIX)+k, core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default}); };
        auto WR=[&](Program& p,const char* k){ return CreateKernel(p, std::string(OVERRIDE_KERNEL_PREFIX)+k, core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default}); };
        auto CC=[&](Program& p,const char* k){ return CreateKernel(p, std::string(OVERRIDE_KERNEL_PREFIX)+k, core, ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false}); };
        auto kbuf=[&](std::vector<uint32_t> vals){ auto b=mkbuf((uint32_t)vals.size());
            std::vector<uint32_t> blob(vals.size()*ELEMS); for(size_t i=0;i<vals.size();++i) for(uint32_t e=0;e<ELEMS;++e) blob[i*ELEMS+e]=vals[i];
            distributed::EnqueueWriteMeshBuffer(cq,b,blob,false); keep.push_back(b); return b; };
        auto put_rows=[&](const std::vector<std::vector<uint32_t>>& rows){ auto b=mkbuf((uint32_t)rows.size());
            std::vector<uint32_t> blob(rows.size()*ELEMS); for(size_t i=0;i<rows.size();++i) std::copy(rows[i].begin(),rows[i].end(),blob.begin()+i*ELEMS);
            distributed::EnqueueWriteMeshBuffer(cq,b,blob,false); keep.push_back(b); return b; };

        // per-op programs (from bfv_rescale.cpp)
        auto modmul=[&](std::shared_ptr<distributed::MeshBuffer> a,uint32_t as, std::shared_ptr<distributed::MeshBuffer> b,uint32_t bs,
                        std::shared_ptr<distributed::MeshBuffer> o,uint32_t os, uint32_t n, const mont::MontgomeryContext& ctx){
            Program p=CreateProgram(); CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,2);CB(p,CBIndex::c_16,2);
            auto r=RD(p,"dataflow/reader_ab.cpp"); auto w=WR(p,"dataflow/writer_modmul.cpp"); auto c=CC(p,"compute/modmul_sfpi_full.cpp");
            uint32_t q=ctx.prime();
            SetRuntimeArgs(p,r,core,{(uint32_t)a->address(),(uint32_t)b->address(),n,as,bs});
            SetRuntimeArgs(p,w,core,{(uint32_t)o->address(),n,os});
            SetRuntimeArgs(p,c,core,{n,q,ctx.q_inv_32(),q&0xFFFFu,q>>16}); run(std::move(p)); };
        auto modsub=[&](std::shared_ptr<distributed::MeshBuffer> a,uint32_t as, std::shared_ptr<distributed::MeshBuffer> b,uint32_t bs,
                        std::shared_ptr<distributed::MeshBuffer> o,uint32_t os, uint64_t mmod){
            Program p=CreateProgram(); CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,2);CB(p,CBIndex::c_16,2);
            auto r=RD(p,"dataflow/reader_ab.cpp"); auto w=WR(p,"dataflow/writer_modmul.cpp"); auto c=CC(p,"compute/modsub_reduce.cpp");
            SetRuntimeArgs(p,r,core,{(uint32_t)a->address(),(uint32_t)b->address(),1u,as,bs});
            SetRuntimeArgs(p,w,core,{(uint32_t)o->address(),1u,os});
            SetRuntimeArgs(p,c,core,{(uint32_t)mmod}); run(std::move(p)); };
        auto modadd_sum=[&](std::shared_ptr<distributed::MeshBuffer> in, std::shared_ptr<distributed::MeshBuffer> o,uint32_t os, uint32_t n, uint64_t mmod){
            Program p=CreateProgram(); CB(p,CBIndex::c_0,LS+1);CB(p,CBIndex::c_16,2);
            auto r=RD(p,"dataflow/reader_i32.cpp"); auto w=WR(p,"dataflow/writer_i32.cpp"); auto c=CC(p,"compute/modadd_reduce.cpp");
            SetRuntimeArgs(p,r,core,{(uint32_t)in->address(),n,0u});
            SetRuntimeArgs(p,w,core,{(uint32_t)o->address(),1u,os});
            SetRuntimeArgs(p,c,core,{n,(uint32_t)mmod}); run(std::move(p)); };
        // pairwise modadd: o = (a[as] + b[bs]) mod q     (reuse reader_ab + modadd over 2 tiles is awkward; use reader_i32 pair)
        auto modadd2=[&](std::shared_ptr<distributed::MeshBuffer> pairbuf, std::shared_ptr<distributed::MeshBuffer> o,uint32_t os, uint64_t mmod){
            Program p=CreateProgram(); CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_16,2);
            auto r=RD(p,"dataflow/reader_i32.cpp"); auto w=WR(p,"dataflow/writer_i32.cpp"); auto c=CC(p,"compute/modadd_reduce.cpp");
            SetRuntimeArgs(p,r,core,{(uint32_t)pairbuf->address(),2u,0u});
            SetRuntimeArgs(p,w,core,{(uint32_t)o->address(),1u,os});
            SetRuntimeArgs(p,c,core,{2u,(uint32_t)mmod}); run(std::move(p)); };

        // ===================================================================
        // RESCALE (device-faithful, from bfv_rescale.cpp) : e_res -> e~ base Q
        // constants depend only on primes -> compute once.
        // ===================================================================
        uint32_t C1a[LQ], C2a[LS][LQ], C2aM[LQ], TS[LS], TM, QiS[LS], QiM, C1e[LS], Dsk[LS], QM, C2e[LQ][LS], Ecorr[LQ];
        for (uint32_t i=0;i<LQ;++i){ u128 qh=Qp/Q[i]; uint64_t qhi=modinv((uint64_t)(qh%Q[i]),Q[i]);
            C1a[i]=cQ[i].to_montgomery((uint32_t)(((u128)t*qhi)%Q[i]));
            for(uint32_t j=0;j<LS;++j) C2a[j][i]=cS[j].to_montgomery((uint32_t)(qh%S[j]));
            C2aM[i]=cM.to_montgomery((uint32_t)(qh%Msk)); }
        for (uint32_t j=0;j<LS;++j){ TS[j]=cS[j].to_montgomery(t); QiS[j]=cS[j].to_montgomery((uint32_t)modinv((uint64_t)(Qp%S[j]),S[j])); }
        TM=cM.to_montgomery(t); QiM=cM.to_montgomery((uint32_t)modinv((uint64_t)(Qp%Msk),Msk));
        for (uint32_t j=0;j<LS;++j){ u128 sh=Sp/S[j]; uint64_t shi=modinv((uint64_t)(sh%S[j]),S[j]);
            C1e[j]=cS[j].to_montgomery((uint32_t)shi); Dsk[j]=cM.to_montgomery((uint32_t)(sh%Msk));
            for(uint32_t i=0;i<LQ;++i) C2e[i][j]=cQ[i].to_montgomery((uint32_t)(sh%Q[i])); }
        QM=cM.to_montgomery((uint32_t)modinv((uint64_t)(Sp%Msk),Msk));
        for (uint32_t i=0;i<LQ;++i) Ecorr[i]=cQ[i].to_montgomery((uint32_t)(Sp%Q[i]));
        auto kC1a=kbuf({C1a[0],C1a[1],C1a[2]});
        std::vector<std::shared_ptr<distributed::MeshBuffer>> kC2a(LS); for(uint32_t j=0;j<LS;++j) kC2a[j]=kbuf({C2a[j][0],C2a[j][1],C2a[j][2]});
        auto kC2aM=kbuf({C2aM[0],C2aM[1],C2aM[2]});
        auto kTS=kbuf({TS[0],TS[1],TS[2],TS[3]}); auto kTM=kbuf({TM});
        auto kQiS=kbuf({QiS[0],QiS[1],QiS[2],QiS[3]}); auto kQiM=kbuf({QiM});
        auto kC1e=kbuf({C1e[0],C1e[1],C1e[2],C1e[3]}); auto kDsk=kbuf({Dsk[0],Dsk[1],Dsk[2],Dsk[3]}); auto kQM=kbuf({QM});
        auto kEc=kbuf({Ecorr[0],Ecorr[1],Ecorr[2]});
        std::vector<std::shared_ptr<distributed::MeshBuffer>> kC2e(LQ); for(uint32_t i=0;i<LQ;++i) kC2e[i]=kbuf({C2e[i][0],C2e[i][1],C2e[i][2],C2e[i][3]});
        // RAW variants of the input-facing t·e constants — used when e is fed in Montgomery form
        // (device tensor output). The leading modmul then decodes R⁻¹ for free (zero extra ops).
        uint32_t C1aR[LQ]; for(uint32_t i=0;i<LQ;++i){ u128 qh=Qp/Q[i]; uint64_t qhi=modinv((uint64_t)(qh%Q[i]),Q[i]); C1aR[i]=(uint32_t)(((u128)t*qhi)%Q[i]); }
        auto kC1aR=kbuf({C1aR[0],C1aR[1],C1aR[2]});
        auto kTSR=kbuf({(uint32_t)(t%S[0]),(uint32_t)(t%S[1]),(uint32_t)(t%S[2]),(uint32_t)(t%S[3])});
        auto kTMR=kbuf({(uint32_t)(t%Msk)});

        auto rescale = [&](const std::vector<std::vector<uint32_t>>& erQ, const std::vector<std::vector<uint32_t>>& erS,
                           const std::vector<std::vector<uint32_t>>& erM, bool mont_in) -> std::vector<std::vector<uint32_t>> {
            auto C1aB = mont_in?kC1aR:kC1a; auto TSB = mont_in?kTSR:kTS; auto TMB = mont_in?kTMR:kTM;
            auto b_eQ=put_rows(erQ), b_eS=put_rows(erS), b_eMsk=put_rows(erM);
            auto b_yQ=mkbuf(LQ), b_prod=mkbuf(LS), b_fbcS=mkbuf(LS), b_fbcM=mkbuf(1), b_tfS=mkbuf(LS), b_tfM=mkbuf(1);
            auto b_dS=mkbuf(LS), b_tS=mkbuf(LS), b_dM=mkbuf(1), b_tM=mkbuf(1), b_yS=mkbuf(LS), b_Ask=mkbuf(1);
            auto b_diff=mkbuf(1), b_eps=mkbuf(1), b_apx=mkbuf(LQ), b_corr=mkbuf(LQ), b_out=mkbuf(LQ);
            keep.insert(keep.end(),{b_yQ,b_prod,b_fbcS,b_fbcM,b_tfS,b_tfM,b_dS,b_tS,b_dM,b_tM,b_yS,b_Ask,b_diff,b_eps,b_apx,b_corr,b_out});
            for(uint32_t i=0;i<LQ;++i) modmul(b_eQ,i,C1aB,i,b_yQ,i,1,cQ[i]);                      // 1
            for(uint32_t j=0;j<LS;++j){ modmul(b_yQ,0,kC2a[j],0,b_prod,0,LQ,cS[j]); modadd_sum(b_prod,b_fbcS,j,LQ,S[j]); } // 2
            { modmul(b_yQ,0,kC2aM,0,b_prod,0,LQ,cM); modadd_sum(b_prod,b_fbcM,0,LQ,Msk); }
            for(uint32_t j=0;j<LS;++j) modmul(b_eS,j,TSB,j,b_tfS,j,1,cS[j]);                      // 3
            { modmul(b_eMsk,0,TMB,0,b_tfM,0,1,cM); }
            for(uint32_t j=0;j<LS;++j){ modsub(b_tfS,j,b_fbcS,j,b_dS,j,S[j]); modmul(b_dS,j,kQiS,j,b_tS,j,1,cS[j]); }  // 4
            { modsub(b_tfM,0,b_fbcM,0,b_dM,0,Msk); modmul(b_dM,0,kQiM,0,b_tM,0,1,cM); }
            for(uint32_t j=0;j<LS;++j) modmul(b_tS,j,kC1e,j,b_yS,j,1,cS[j]);                     // 5 exact S->Q
            { modmul(b_yS,0,kDsk,0,b_prod,0,LS,cM); modadd_sum(b_prod,b_Ask,0,LS,Msk); }
            { modsub(b_Ask,0,b_tM,0,b_diff,0,Msk); modmul(b_diff,0,kQM,0,b_eps,0,1,cM); }
            for(uint32_t i=0;i<LQ;++i){ modmul(b_yS,0,kC2e[i],0,b_prod,0,LS,cQ[i]); modadd_sum(b_prod,b_apx,i,LS,Q[i]);
                modmul(b_eps,0,kEc,i,b_corr,i,1,cQ[i]); modsub(b_apx,i,b_corr,i,b_out,i,Q[i]); }
            std::vector<uint32_t> ob; distributed::EnqueueReadMeshBuffer(cq, ob, b_out, true);
            std::vector<std::vector<uint32_t>> res(LQ);
            for(uint32_t i=0;i<LQ;++i) res[i]=std::vector<uint32_t>(ob.begin()+i*ELEMS, ob.begin()+(i+1)*ELEMS);  // tilized
            return res; };

        // ===================================================================
        // RELIN (device, from bfv_relin.cpp) : c2 base Q + evk -> (k0,k1) base Q
        // ===================================================================
        auto b_Xin=mkbuf(1), b_WfoldA=mkbuf(NB*NB,I8), b_G=mkbuf(1), b_Tm=mkbuf(1), b_Gp=mkbuf(1), b_Wrhs=mkbuf(NB*NB,I8), b_H=mkbuf(1);
        keep.insert(keep.end(),{b_Xin,b_WfoldA,b_G,b_Tm,b_Gp,b_Wrhs,b_H});
        auto transform=[&](const std::vector<uint32_t>& in_nat, uint64_t q, const mont::MontgomeryContext& C, bool fwd, bool in_tiled=false)->std::vector<uint32_t>{
            uint32_t QINV=C.q_inv_32(), QLO=q&0xFFFFu, QHI=q>>16, K14=C.to_montgomery(16384u);
            uint64_t g=prim_root(q);
            uint64_t wf=modpow(g,(q-1)/N,q), w32f=modpow(g,(q-1)/T,q), psif=modpow(g,(q-1)/(2*N),q);
            uint64_t w=fwd?wf:mont::modular_inverse(wf,q), w32=fwd?w32f:mont::modular_inverse(w32f,q), psi=fwd?psif:mont::modular_inverse(psif,q);
            uint64_t Ninv=mont::modular_inverse(N,q);
            std::vector<uint32_t> Wa(ELEMS),Wc(ELEMS),Tw(ELEMS);
            for(uint32_t a=0;a<T;++a) for(uint32_t b=0;b<T;++b){ uint32_t wv=(uint32_t)modpow(w32,(uint64_t)a*b,q);
                if(fwd){ Wa[a*T+b]=(uint32_t)(((u128)wv*modpow(psi,32ull*b,q))%q); Wc[a*T+b]=wv; }
                else   { Wa[a*T+b]=(uint32_t)(((u128)wv*Ninv)%q);                    Wc[a*T+b]=(uint32_t)(((u128)wv*modpow(psi,32ull*b,q))%q); } }
            for(uint32_t a=0;a<T;++a) for(uint32_t b=0;b<T;++b){ uint64_t base=modpow(w,(uint64_t)a*b,q);
                Tw[a*T+b]=(uint32_t)(((u128)base*(fwd?modpow(psi,(uint64_t)b,q):modpow(psi,(uint64_t)a,q)))%q); }
            std::vector<int8_t> WfoldA(NB*NB*ELEMS), Wrhs(NB*NB*ELEMS);
            for(uint32_t p=0;p<NB;++p) for(uint32_t r=0;r<NB;++r){ std::vector<int8_t> la(ELEMS),lc(ELEMS);
                for(uint32_t i=0;i<ELEMS;++i){ la[i]=limb((uint64_t)((POW[r]*Wa[i])%q),p); lc[i]=limb((uint64_t)((POW[p]*Wc[i])%q),r); }
                auto ta=tilize(la); std::copy(ta.begin(),ta.end(),WfoldA.begin()+(p*NB+r)*ELEMS);
                auto tc=tilize(lc); std::copy(tc.begin(),tc.end(),Wrhs.begin()+(p*NB+r)*ELEMS); }
            std::vector<uint32_t> Tm(ELEMS); { std::vector<uint32_t> tm(ELEMS); for(uint32_t i=0;i<ELEMS;++i) tm[i]=C.to_montgomery(Tw[i]); Tm=tilize(tm); }
            distributed::EnqueueWriteMeshBuffer(cq,b_Xin, in_tiled? in_nat : tilize(in_nat), false);
            distributed::EnqueueWriteMeshBuffer(cq,b_WfoldA,WfoldA,false);
            distributed::EnqueueWriteMeshBuffer(cq,b_Wrhs,Wrhs,false);
            distributed::EnqueueWriteMeshBuffer(cq,b_Tm,Tm,false);
            { Program p=CreateProgram();
              CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,2,tt::DataFormat::Int8,I8);CB(p,CBIndex::c_2,NB,tt::DataFormat::Int8,I8);CB(p,CBIndex::c_3,NB,tt::DataFormat::Int32,I32);CB(p,CBIndex::c_16,2);
              std::vector<uint32_t> rc; TensorAccessorArgs(*b_Xin).append_to(rc); TensorAccessorArgs(*b_WfoldA).append_to(rc);
              auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
              auto w=WR(p,"dataflow/writer_i32.cpp"); auto c=CC(p,"compute/fused_matmul_a.cpp");
              SetRuntimeArgs(p,r,core,{(uint32_t)b_Xin->address(),(uint32_t)b_WfoldA->address()});
              SetRuntimeArgs(p,w,core,{(uint32_t)b_G->address(),1u,0u});
              SetRuntimeArgs(p,c,core,{(uint32_t)q,QINV,QLO,QHI,K14}); run(std::move(p)); }
            { Program p=CreateProgram(); CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,2);CB(p,CBIndex::c_16,2);
              auto r=RD(p,"dataflow/reader_modmul.cpp"); auto w=WR(p,"dataflow/writer_modmul.cpp"); auto c=CC(p,"compute/modmul_compute.cpp");
              SetRuntimeArgs(p,r,core,{(uint32_t)b_G->address(),(uint32_t)b_Tm->address(),1u,0u});
              SetRuntimeArgs(p,w,core,{(uint32_t)b_Gp->address(),1u,0u});
              SetRuntimeArgs(p,c,core,{1u,(uint32_t)q,QINV,QLO,QHI}); run(std::move(p)); }
            { Program p=CreateProgram();
              CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,NB*NB,tt::DataFormat::Int8,I8);CB(p,CBIndex::c_2,NB,tt::DataFormat::Int8,I8);CB(p,CBIndex::c_3,NB,tt::DataFormat::Int32,I32);CB(p,CBIndex::c_16,2);
              std::vector<uint32_t> rc; TensorAccessorArgs(*b_Gp).append_to(rc); TensorAccessorArgs(*b_Wrhs).append_to(rc);
              auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
              auto w=WR(p,"dataflow/writer_i32.cpp"); auto c=CC(p,"compute/fused_matmul_c.cpp");
              SetRuntimeArgs(p,r,core,{(uint32_t)b_Gp->address(),(uint32_t)b_Wrhs->address()});
              SetRuntimeArgs(p,w,core,{(uint32_t)b_H->address(),1u,0u});
              SetRuntimeArgs(p,c,core,{(uint32_t)q,QINV,QLO,QHI,K14}); run(std::move(p)); }
            std::vector<uint32_t> Hraw; distributed::EnqueueReadMeshBuffer(cq,Hraw,b_H,true);
            return reindex(untilize(Hraw)); };

        auto b_a=mkbuf(1), b_b=mkbuf(1), b_o=mkbuf(1); keep.insert(keep.end(),{b_a,b_b,b_o});
        auto pointwise=[&](const std::vector<uint32_t>& a_nat, const std::vector<uint32_t>& bmont_nat, uint64_t q, const mont::MontgomeryContext& C){
            distributed::EnqueueWriteMeshBuffer(cq,b_a,tilize(a_nat),false);
            distributed::EnqueueWriteMeshBuffer(cq,b_b,tilize(bmont_nat),false);
            Program p=CreateProgram(); CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,2);CB(p,CBIndex::c_16,2);
            auto r=RD(p,"dataflow/reader_modmul.cpp"); auto w=WR(p,"dataflow/writer_modmul.cpp"); auto c=CC(p,"compute/modmul_sfpi_full.cpp");
            SetRuntimeArgs(p,r,core,{(uint32_t)b_a->address(),(uint32_t)b_b->address(),1u,0u});
            SetRuntimeArgs(p,w,core,{(uint32_t)b_o->address(),1u,0u});
            SetRuntimeArgs(p,c,core,{1u,(uint32_t)q,C.q_inv_32(),q&0xFFFFu,q>>16}); run(std::move(p));
            std::vector<uint32_t> o; distributed::EnqueueReadMeshBuffer(cq,o,b_o,true); return untilize(o); };

        // ---- BUFFER-RESIDENT transform (NTT/INTT): reindex on device (transpose_tile); leaves
        //      the reindexed result in out_buf with NO host readback. in_buf/out_buf are 1-tile DRAM. ----
        auto transpose_dev=[&](std::shared_ptr<distributed::MeshBuffer> in, std::shared_ptr<distributed::MeshBuffer> out){
            Program p=CreateProgram(); CB(p,CBIndex::c_0,2); CB(p,CBIndex::c_16,2);
            auto r=RD(p,"dataflow/reader_i32.cpp"); auto w=WR(p,"dataflow/writer_i32.cpp"); auto c=CC(p,"compute/transpose_tile.cpp");
            SetRuntimeArgs(p,r,core,{(uint32_t)in->address(),1u,0u});
            SetRuntimeArgs(p,w,core,{(uint32_t)out->address(),1u,0u});
            SetRuntimeArgs(p,c,core,{}); run(std::move(p)); };
        auto transform_buf=[&](std::shared_ptr<distributed::MeshBuffer> in_buf, std::shared_ptr<distributed::MeshBuffer> out_buf,
                               uint64_t q, const mont::MontgomeryContext& C, bool fwd){
            uint32_t QINV=C.q_inv_32(), QLO=q&0xFFFFu, QHI=q>>16, K14=C.to_montgomery(16384u);
            uint64_t g=prim_root(q);
            uint64_t wf=modpow(g,(q-1)/N,q), w32f=modpow(g,(q-1)/T,q), psif=modpow(g,(q-1)/(2*N),q);
            uint64_t w=fwd?wf:mont::modular_inverse(wf,q), w32=fwd?w32f:mont::modular_inverse(w32f,q), psi=fwd?psif:mont::modular_inverse(psif,q);
            uint64_t Ninv=mont::modular_inverse(N,q);
            std::vector<uint32_t> Wa(ELEMS),Wc(ELEMS),Tw(ELEMS);
            for(uint32_t a=0;a<T;++a) for(uint32_t b=0;b<T;++b){ uint32_t wv=(uint32_t)modpow(w32,(uint64_t)a*b,q);
                if(fwd){ Wa[a*T+b]=(uint32_t)(((u128)wv*modpow(psi,32ull*b,q))%q); Wc[a*T+b]=wv; }
                else   { Wa[a*T+b]=(uint32_t)(((u128)wv*Ninv)%q);                    Wc[a*T+b]=(uint32_t)(((u128)wv*modpow(psi,32ull*b,q))%q); } }
            for(uint32_t a=0;a<T;++a) for(uint32_t b=0;b<T;++b){ uint64_t base=modpow(w,(uint64_t)a*b,q);
                Tw[a*T+b]=(uint32_t)(((u128)base*(fwd?modpow(psi,(uint64_t)b,q):modpow(psi,(uint64_t)a,q)))%q); }
            std::vector<int8_t> WfoldA(NB*NB*ELEMS), Wrhs(NB*NB*ELEMS);
            for(uint32_t p=0;p<NB;++p) for(uint32_t r=0;r<NB;++r){ std::vector<int8_t> la(ELEMS),lc(ELEMS);
                for(uint32_t i=0;i<ELEMS;++i){ la[i]=limb((uint64_t)((POW[r]*Wa[i])%q),p); lc[i]=limb((uint64_t)((POW[p]*Wc[i])%q),r); }
                auto ta=tilize(la); std::copy(ta.begin(),ta.end(),WfoldA.begin()+(p*NB+r)*ELEMS);
                auto tc=tilize(lc); std::copy(tc.begin(),tc.end(),Wrhs.begin()+(p*NB+r)*ELEMS); }
            std::vector<uint32_t> Tm(ELEMS); { std::vector<uint32_t> tm(ELEMS); for(uint32_t i=0;i<ELEMS;++i) tm[i]=C.to_montgomery(Tw[i]); Tm=tilize(tm); }
            distributed::EnqueueWriteMeshBuffer(cq,b_WfoldA,WfoldA,false);
            distributed::EnqueueWriteMeshBuffer(cq,b_Wrhs,Wrhs,false);
            distributed::EnqueueWriteMeshBuffer(cq,b_Tm,Tm,false);
            { Program p=CreateProgram();
              CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,2,tt::DataFormat::Int8,I8);CB(p,CBIndex::c_2,NB,tt::DataFormat::Int8,I8);CB(p,CBIndex::c_3,NB,tt::DataFormat::Int32,I32);CB(p,CBIndex::c_16,2);
              std::vector<uint32_t> rc; TensorAccessorArgs(*in_buf).append_to(rc); TensorAccessorArgs(*b_WfoldA).append_to(rc);
              auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
              auto w=WR(p,"dataflow/writer_i32.cpp"); auto c=CC(p,"compute/fused_matmul_a.cpp");
              SetRuntimeArgs(p,r,core,{(uint32_t)in_buf->address(),(uint32_t)b_WfoldA->address()});
              SetRuntimeArgs(p,w,core,{(uint32_t)b_G->address(),1u,0u});
              SetRuntimeArgs(p,c,core,{(uint32_t)q,QINV,QLO,QHI,K14}); run(std::move(p)); }
            { Program p=CreateProgram(); CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,2);CB(p,CBIndex::c_16,2);
              auto r=RD(p,"dataflow/reader_modmul.cpp"); auto w=WR(p,"dataflow/writer_modmul.cpp"); auto c=CC(p,"compute/modmul_compute.cpp");
              SetRuntimeArgs(p,r,core,{(uint32_t)b_G->address(),(uint32_t)b_Tm->address(),1u,0u});
              SetRuntimeArgs(p,w,core,{(uint32_t)b_Gp->address(),1u,0u});
              SetRuntimeArgs(p,c,core,{1u,(uint32_t)q,QINV,QLO,QHI}); run(std::move(p)); }
            { Program p=CreateProgram();
              CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,NB*NB,tt::DataFormat::Int8,I8);CB(p,CBIndex::c_2,NB,tt::DataFormat::Int8,I8);CB(p,CBIndex::c_3,NB,tt::DataFormat::Int32,I32);CB(p,CBIndex::c_16,2);
              std::vector<uint32_t> rc; TensorAccessorArgs(*b_Gp).append_to(rc); TensorAccessorArgs(*b_Wrhs).append_to(rc);
              auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
              auto w=WR(p,"dataflow/writer_i32.cpp"); auto c=CC(p,"compute/fused_matmul_c.cpp");
              SetRuntimeArgs(p,r,core,{(uint32_t)b_Gp->address(),(uint32_t)b_Wrhs->address()});
              SetRuntimeArgs(p,w,core,{(uint32_t)b_H->address(),1u,0u});
              SetRuntimeArgs(p,c,core,{(uint32_t)q,QINV,QLO,QHI,K14}); run(std::move(p)); }
            transpose_dev(b_H, out_buf); };   // on-device reindex -> out_buf (tiled, resident)

        // padded fused FBC (width 4) — from bfv_relin.cpp fbc_fused
        auto fbc_fused=[&](const std::vector<std::vector<uint32_t>>& xsrc, const std::vector<uint32_t>& src, const std::vector<uint32_t>& dst)
            -> std::vector<std::vector<uint32_t>> {
            uint32_t Ls=src.size(), Ld=dst.size();
            u128 Spr=1; for(uint32_t i=0;i<Ls;++i) Spr*=QP[src[i]];
            std::vector<uint32_t> C1(FBCW,0); std::vector<std::vector<uint32_t>> C2(FBCW, std::vector<uint32_t>(FBCW,0));
            std::vector<uint32_t> Sm(FBCW),SmI(FBCW),Dm(FBCW),DmI(FBCW);
            for(uint32_t i=0;i<FBCW;++i){ uint32_t si=(i<Ls)?src[i]:src[0]; uint64_t qi=QP[si]; Sm[i]=(uint32_t)qi; SmI[i]=cQP[si].q_inv_32();
                if(i<Ls) C1[i]=cQP[si].to_montgomery((uint32_t)modpow((uint64_t)((Spr/qi)%qi), qi-2, qi)); }
            for(uint32_t j=0;j<FBCW;++j){ uint32_t dj=(j<Ld)?dst[j]:dst[0]; uint64_t pj=QP[dj]; Dm[j]=(uint32_t)pj; DmI[j]=cQP[dj].q_inv_32();
                if(j<Ld) for(uint32_t i=0;i<Ls;++i) C2[i][j]=cQP[dj].to_montgomery((uint32_t)((Spr/QP[src[i]])%pj)); }
            auto b_x=mkbuf(FBCW), b_out=mkbuf(FBCW); keep.insert(keep.end(),{b_x,b_out});
            std::vector<uint32_t> xb(FBCW*ELEMS,0);
            for(uint32_t i=0;i<Ls;++i) std::copy(xsrc[i].begin(),xsrc[i].end(),xb.begin()+i*ELEMS);  // xsrc already tilized
            distributed::EnqueueWriteMeshBuffer(cq,b_x,xb,false);
            std::vector<uint32_t> ca; ca.push_back(FBCW);
            for(uint32_t i=0;i<FBCW;++i) ca.push_back(C1[i]);
            for(uint32_t i=0;i<FBCW;++i) for(uint32_t j=0;j<FBCW;++j) ca.push_back(C2[i][j]);
            for(uint32_t i=0;i<FBCW;++i) ca.push_back(Sm[i]);
            for(uint32_t i=0;i<FBCW;++i) ca.push_back(SmI[i]);
            for(uint32_t j=0;j<FBCW;++j) ca.push_back(Dm[j]);
            for(uint32_t j=0;j<FBCW;++j) ca.push_back(DmI[j]);
            Program p=CreateProgram(); CB(p,CBIndex::c_0,8);CB(p,CBIndex::c_2,8);CB(p,CBIndex::c_3,8);CB(p,CBIndex::c_16,8);
            auto r=RD(p,"dataflow/reader_i32.cpp"); auto w=WR(p,"dataflow/writer_i32.cpp"); auto c=CC(p,"compute/baseconv_fbc.cpp");
            SetRuntimeArgs(p,r,core,{(uint32_t)b_x->address(),FBCW,0u});
            SetRuntimeArgs(p,w,core,{(uint32_t)b_out->address(),FBCW,0u});
            SetRuntimeArgs(p,c,core,ca); run(std::move(p));
            std::vector<uint32_t> ob; distributed::EnqueueReadMeshBuffer(cq,ob,b_out,true);
            std::vector<std::vector<uint32_t>> out(Ld);
            for(uint32_t j=0;j<Ld;++j) out[j]=untilize(std::vector<uint32_t>(ob.begin()+j*ELEMS, ob.begin()+(j+1)*ELEMS));
            return out; };

        // relin(c2 base-Q natural-order rows, evk coeff QP tilized) -> (k0,k1) natural-order base Q
        auto relin=[&](const std::vector<std::vector<uint32_t>>& c2nat)->std::pair<std::vector<std::vector<uint32_t>>,std::vector<std::vector<uint32_t>>>{
            // mod-up Q->P (padded FBC). c2nat given natural order; FBC needs tilized src.
            std::vector<std::vector<uint32_t>> c2qp(LQP);
            for(uint32_t i=0;i<LQ;++i) c2qp[i]=c2nat[i];
            { std::vector<std::vector<uint32_t>> xt(LQ); std::vector<uint32_t> srcIdx(LQ),dstIdx(LP);
              for(uint32_t i=0;i<LQ;++i){ xt[i]=tilize(c2nat[i]); srcIdx[i]=i; }
              for(uint32_t j=0;j<LP;++j) dstIdx[j]=LQ+j;
              auto pP=fbc_fused(xt,srcIdx,dstIdx); for(uint32_t j=0;j<LP;++j) c2qp[LQ+j]=pP[j]; }  // pP natural order
            std::vector<std::vector<uint32_t>> acc0(LQP), acc1(LQP);
            // buffer-resident per-limb: fwd NTT -> ×evk -> inv NTT stays in DRAM (device reindex);
            // only acc0,acc1 read back (2/limb, was 5). c2f plain ⊙ mont(evk) -> plain (relin convention).
            auto rb_in=mkbuf(1), rb_c2f=mkbuf(1), rb_e0=mkbuf(1), rb_e1=mkbuf(1), rb_a0=mkbuf(1), rb_a1=mkbuf(1), rb_acc0=mkbuf(1), rb_acc1=mkbuf(1);
            keep.insert(keep.end(),{rb_in,rb_c2f,rb_e0,rb_e1,rb_a0,rb_a1,rb_acc0,rb_acc1});
            for(uint32_t li=0;li<LQP;++li){ uint64_t m=QP[li];
                auto nntt_cpu=[&](const std::vector<uint32_t>& v){ std::vector<uint32_t> X(N);
                    uint64_t g=prim_root(m),w=modpow(g,(m-1)/N,m),psi=modpow(g,(m-1)/(2*N),m);
                    for(uint32_t k=0;k<N;++k){ u128 acc=0; for(uint32_t n=0;n<N;++n) acc+=(u128)v[n]*((u128)modpow(psi,n,m)*modpow(w,(uint64_t)n*k,m)%m); X[k]=(uint32_t)(acc%m);} return X; };
                auto e0f=nntt_cpu(untilize(evk0[li])), e1f=nntt_cpu(untilize(evk1[li]));  // evk stored tilized coeff
                std::vector<uint32_t> e0m(N),e1m(N); for(uint32_t n=0;n<N;++n){ e0m[n]=cQP[li].to_montgomery(e0f[n]); e1m[n]=cQP[li].to_montgomery(e1f[n]); }
                distributed::EnqueueWriteMeshBuffer(cq,rb_in,tilize(c2qp[li]),false);
                transform_buf(rb_in,rb_c2f,QP[li],cQP[li],true);
                distributed::EnqueueWriteMeshBuffer(cq,rb_e0,tilize(e0m),false);
                distributed::EnqueueWriteMeshBuffer(cq,rb_e1,tilize(e1m),false);
                modmul(rb_c2f,0,rb_e0,0,rb_a0,0,1,cQP[li]);   // c2f ⊙ evk0
                modmul(rb_c2f,0,rb_e1,0,rb_a1,0,1,cQP[li]);   // c2f ⊙ evk1
                transform_buf(rb_a0,rb_acc0,QP[li],cQP[li],false);
                transform_buf(rb_a1,rb_acc1,QP[li],cQP[li],false);
                std::vector<uint32_t> r0,r1; distributed::EnqueueReadMeshBuffer(cq,r0,rb_acc0,true); distributed::EnqueueReadMeshBuffer(cq,r1,rb_acc1,true);
                acc0[li]=untilize(r0); acc1[li]=untilize(r1); }
            auto moddown=[&](std::vector<std::vector<uint32_t>>& acc){
                std::vector<std::vector<uint32_t>> xP(LP); std::vector<uint32_t> srcIdx(LP),dstIdx(LQ);
                for(uint32_t i=0;i<LP;++i){ xP[i]=tilize(acc[LQ+i]); srcIdx[i]=LQ+i; }
                for(uint32_t j=0;j<LQ;++j) dstIdx[j]=j;
                auto fbcR=fbc_fused(xP,srcIdx,dstIdx);   // natural order
                std::vector<std::vector<uint32_t>> out(LQ, std::vector<uint32_t>(N));
                for(uint32_t j=0;j<LQ;++j){ uint64_t q=Q[j]; uint64_t Pinv=modpow((uint64_t)(Pp%q),q-2,q);
                    for(uint32_t n=0;n<N;++n){ uint32_t diff=(uint32_t)(((u128)acc[j][n]+q-fbcR[j][n])%q); out[j][n]=(uint32_t)((u128)diff*Pinv%q);} }
                return out; };
            return { moddown(acc0), moddown(acc1) }; };

        // ================= run pipeline =================
        auto cmp_tiled = [&](const std::vector<std::vector<uint32_t>>& got_t, const std::vector<std::vector<uint32_t>>& gold_t, const char* nm){
            uint32_t mism=0; for(uint32_t i=0;i<got_t.size();++i) for(uint32_t e=0;e<ELEMS;++e) if(got_t[i][e]!=gold_t[i][e]) ++mism;
            fmt::print("  [{}] mism={}/{}\n", nm, mism, got_t.size()*ELEMS); return mism; };
        // ---- device TENSOR over 8 limbs (Q|S|msk): NTT -> pointwise -> INTT, all Montgomery ----
        // per limb: fwd nNTT the 4 ct polys (mont in -> mont freq), pointwise tensor (mont⊙mont->mont),
        // inv nNTT -> mont-form coeff residue of e_k. Output feeds rescale in Montgomery (mont_in).
        std::vector<std::vector<std::vector<uint32_t>>> eMont(3, std::vector<std::vector<uint32_t>>(LT));  // [k][limb] mont-coeff TILED
        if (!inject) {
            // BUFFER-RESIDENT tensor: data stays in DRAM through fwd-NTT -> pointwise -> inv-NTT;
            // reindex is on-device (transpose_dev), so only the 3 final e_k tiles/limb read back.
            auto bc0=mkbuf(1),bc1=mkbuf(1),bd0=mkbuf(1),bd1=mkbuf(1);
            auto bc0f=mkbuf(1),bc1f=mkbuf(1),bd0f=mkbuf(1),bd1f=mkbuf(1);
            auto bE0=mkbuf(1),bE2=mkbuf(1),bT=mkbuf(2),bE1=mkbuf(1),be0=mkbuf(1),be1=mkbuf(1),be2=mkbuf(1);
            keep.insert(keep.end(),{bc0,bc1,bd0,bd1,bc0f,bc1f,bd0f,bd1f,bE0,bE2,bT,bE1,be0,be1,be2});
            for(uint32_t li=0; li<LT; ++li){ uint64_t q=P8[li]; auto& C=C8[li];
                // encode ct in TILE order (to_montgomery is per-element) and write straight to DRAM
                auto encw=[&](std::shared_ptr<distributed::MeshBuffer> buf, const std::vector<uint32_t>& trow){
                    auto v=trow; for(auto&x:v) x=C.to_montgomery(x); distributed::EnqueueWriteMeshBuffer(cq,buf,v,false); };
                encw(bc0,c0i[li]); encw(bc1,c1i[li]); encw(bd0,d0i[li]); encw(bd1,d1i[li]);
                transform_buf(bc0,bc0f,q,C,true); transform_buf(bc1,bc1f,q,C,true);
                transform_buf(bd0,bd0f,q,C,true); transform_buf(bd1,bd1f,q,C,true);
                modmul(bc0f,0,bd0f,0,bE0,0,1,C);   // e0 = c0·d0
                modmul(bc1f,0,bd1f,0,bE2,0,1,C);   // e2 = c1·d1
                modmul(bc0f,0,bd1f,0,bT,0,1,C);    // t0 = c0·d1  -> bT[0]
                modmul(bc1f,0,bd0f,0,bT,1,1,C);    // t1 = c1·d0  -> bT[1]
                modadd_sum(bT,bE1,0,2,q);          // e1 = t0 + t1
                transform_buf(bE0,be0,q,C,false); transform_buf(bE1,be1,q,C,false); transform_buf(bE2,be2,q,C,false);
                distributed::EnqueueReadMeshBuffer(cq,eMont[0][li],be0,true);   // 3 reads/limb (was 12)
                distributed::EnqueueReadMeshBuffer(cq,eMont[1][li],be1,true);
                distributed::EnqueueReadMeshBuffer(cq,eMont[2][li],be2,true); }
            // (optional) validate device tensor vs golden e — DBG only (eMont is TILED mont)
            if (getenv("DBG_TENSOR")) { uint32_t tmism=0;
                for(uint32_t k=0;k<3;++k) for(uint32_t li=0; li<LT; ++li){
                    auto pl=eMont[k][li]; for(auto&v:pl) v=C8[li].from_montgomery(v);
                    const std::vector<uint32_t>& gold = (li<LQ)? eQ[k][li] : (li<LQ+LS? eS[k][li-LQ] : eM[k][0]);
                    for(uint32_t e=0;e<ELEMS;++e) if(pl[e]!=gold[e]) ++tmism; }
                fmt::print("  [tensor e] mism={}/{}\n", tmism, 3*LT*ELEMS); }
        }

        // rescale x3.  device tensor -> Montgomery e (mont_in=true); INJECT_GOLDEN -> golden plain e.
        std::vector<std::vector<std::vector<uint32_t>>> tildeDev(3);
        uint32_t rescale_mism=0;
        for(uint32_t k=0;k<3;++k){
            std::vector<std::vector<uint32_t>> erQ(LQ), erS(LS), erM(1);
            if(!inject){ for(uint32_t i=0;i<LQ;++i) erQ[i]=eMont[k][i];         // eMont already TILED
                         for(uint32_t j=0;j<LS;++j) erS[j]=eMont[k][LQ+j];
                         erM[0]=eMont[k][LQ+LS]; }
            else { erQ=eQ[k]; erS=eS[k]; erM=eM[k]; }
            tildeDev[k]=rescale(erQ,erS,erM, !inject);
            rescale_mism+=cmp_tiled(tildeDev[k],gTilde[k], (k==0?"rescale e~0":k==1?"rescale e~1":"rescale e~2")); }

        // relin on e~2 (natural order needed). Use device tilde or golden if INJECT_GOLDEN.
        std::vector<std::vector<uint32_t>> c2src(LQ);
        auto& srcTilde = inject ? gTilde[2] : tildeDev[2];
        for(uint32_t i=0;i<LQ;++i) c2src[i]=untilize(srcTilde[i]);
        auto kk = relin(c2src);
        // compare k (natural order) vs golden (tilized) -> tilize device k
        std::vector<std::vector<uint32_t>> k0t(LQ),k1t(LQ);
        for(uint32_t i=0;i<LQ;++i){ k0t[i]=tilize(kk.first[i]); k1t[i]=tilize(kk.second[i]); }
        uint32_t relin_mism = cmp_tiled(k0t,gK0,"relin k0") + cmp_tiled(k1t,gK1,"relin k1");

        // final add: out_k = (tilde_k + k_k) mod Q per limb (device modadd2). Use device values.
        auto add_pair=[&](const std::vector<std::vector<uint32_t>>& A_t, const std::vector<std::vector<uint32_t>>& K_t)->std::vector<std::vector<uint32_t>>{
            std::vector<std::vector<uint32_t>> out(LQ);
            for(uint32_t i=0;i<LQ;++i){ auto pb=mkbuf(2); keep.push_back(pb);
                std::vector<uint32_t> blob(2*ELEMS); std::copy(A_t[i].begin(),A_t[i].end(),blob.begin()); std::copy(K_t[i].begin(),K_t[i].end(),blob.begin()+ELEMS);
                distributed::EnqueueWriteMeshBuffer(cq,pb,blob,false);
                auto ob=mkbuf(1); keep.push_back(ob); modadd2(pb,ob,0,Q[i]);
                std::vector<uint32_t> o; distributed::EnqueueReadMeshBuffer(cq,o,ob,true); out[i]=o; }
            return out; };
        auto& tildeForAdd0 = inject ? gTilde[0] : tildeDev[0];
        auto& tildeForAdd1 = inject ? gTilde[1] : tildeDev[1];
        auto out0 = add_pair(tildeForAdd0, k0t);
        auto out1 = add_pair(tildeForAdd1, k1t);
        uint32_t out_mism = cmp_tiled(out0,gOut0,"out0") + cmp_tiled(out1,gOut1,"out1");

        fmt::print("BFV full ct_mul ({})  rescale={}  relin={}  final={}\n",
                   inject?"INJECT_GOLDEN: golden tensor, device rescale+relin+add":"FULL device: tensor+rescale+relin+add",
                   rescale_mism, relin_mism, out_mism);
        pass = (rescale_mism==0 && relin_mism==0 && out_mism==0);
        fmt::print("{}", pass ? "RESULT: full BFV multiply on device MATCHES golden (decrypts to m1·m2).\n" : "RESULT: WRONG.\n");

        // ---- device-active steady-state timing (build-once, re-enqueue all workloads; TIME=1) ----
        // All device programs are collected in `wls` (via run()); re-enqueue them with no host work
        // or blocking readbacks between, one Finish at the end. Measures device dispatch+exec per
        // ct_mul (host permutation/readback excluded — that's the buffer-resident path's payoff).
        if (getenv("TIME")) {
            distributed::Finish(cq);                                   // warm
            const int iters = 20;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it=0; it<iters; ++it) for (auto& wl : wls) distributed::EnqueueMeshWorkload(cq, wl, false);
            distributed::Finish(cq);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/1e6/iters;
            fmt::print("TIMING: device-active steady-state = {:.2f} ms/ct_mul  ({} dispatches)\n", ms, wls.size());
        }
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr,"Exception:\n{}\n",e.what()); return 2; }
    fmt::print("{}\n", pass?"Test Passed":"Test FAILED");
    return pass?0:1;
}
