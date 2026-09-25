// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// CORRECT on-device BFV rescale  ẽ = ⌊t·e/Q⌋  at Q=3, aux S=4, sentinel m_sk (plan §1/§2).
// Plain-domain multi-dispatch, matches reference/he/bfv_mul_ref.py:rns_scale EXACTLY:
//   1. yQ_i = (t·e_i·q̂_i⁻¹) mod q_i                          [t folded into C1a]
//   2. Q->{S,msk} approx FBC:  fbc_x = Σ_i yQ_i·(Q/q_i mod x)  (drops ε; noise-absorbed)
//   3. tf_x = t·e_x ;  ẽ_x = (tf_x − fbc_x)·Q⁻¹  mod x         (x over S and m_sk)
//   4. S->Q exact FBC via m_sk (Shenoy-Kumaresan):  out_i = ẽ mod q_i
// Validated vs a Python golden (gen_rescale_case.py -> rescale_case.bin): same 8 primes below
// 2^28, plain residues. All values PLAIN; consts are Montgomery only as modmul's b-operand
// (modmul_sfpi_full computes a·mont(c)=a·c).   ./build/bfv_rescale [reference/he/rescale_case.bin]

#include <cstdint>
#include <fstream>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t ELEMS = 1024, I32 = ELEMS * 4, LQ = 3, LS = 4;
using u128 = unsigned __int128;
uint64_t modpow(uint64_t b, uint64_t e, uint64_t m){ u128 r=1,x=b%m; while(e){ if(e&1) r=r*x%m; x=x*x%m; e>>=1;} return (uint64_t)r; }
uint64_t modinv(uint64_t a, uint64_t m){ return modpow(a%m, m-2, m); }
}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const char* path = (argc > 1) ? argv[1] : "reference/he/rescale_case.bin";
        std::ifstream f(path, std::ios::binary);
        if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
        uint32_t N, t; f.read((char*)&N,4); f.read((char*)&t,4);
        uint32_t pr[8]; f.read((char*)pr,32);
        uint64_t Q[LQ], S[LS], Msk;
        for (uint32_t i=0;i<LQ;++i) Q[i]=pr[i];
        for (uint32_t j=0;j<LS;++j) S[j]=pr[LQ+j];
        Msk=pr[LQ+LS];
        auto rd=[&](uint32_t n){ std::vector<uint32_t> v(n*ELEMS); f.read((char*)v.data(), (std::streamsize)n*ELEMS*4); return v; };
        auto eQ=rd(LQ), eS=rd(LS), eMsk=rd(1), gold=rd(LQ);

        std::vector<mont::MontgomeryContext> cQ, cS; mont::MontgomeryContext cM((uint32_t)Msk);
        for (uint32_t i=0;i<LQ;++i) cQ.emplace_back((uint32_t)Q[i]);
        for (uint32_t j=0;j<LS;++j) cS.emplace_back((uint32_t)S[j]);
        u128 Qp=1; for(uint32_t i=0;i<LQ;++i) Qp*=Q[i];
        u128 Sp=1; for(uint32_t j=0;j<LS;++j) Sp*=S[j];

        // MONTG_IN=1: e is fed in MONTGOMERY form (the on-device tensor's output convention). The
        // rescale's leading t·e modmuls (C1a on Q, TS/TM on S/msk) then use RAW constants so the
        // Montgomery R⁻¹ decode folds into that existing multiply — plain internals + plain ẽ output
        // unchanged, ZERO extra device ops. (FBC needs plain CRT coeffs; decode happens at the edge.)
        const bool montg = getenv("MONTG_IN") != nullptr;
        if (montg) {
            for (uint32_t i=0;i<LQ;++i) for (uint32_t e=0;e<ELEMS;++e) eQ[i*ELEMS+e]=cQ[i].to_montgomery(eQ[i*ELEMS+e]);
            for (uint32_t j=0;j<LS;++j) for (uint32_t e=0;e<ELEMS;++e) eS[j*ELEMS+e]=cS[j].to_montgomery(eS[j*ELEMS+e]);
            for (uint32_t e=0;e<ELEMS;++e) eMsk[e]=cM.to_montgomery(eMsk[e]);
        }

        // ---- Montgomery b-operand constants ----
        uint32_t C1a[LQ], C2a[LS][LQ], C2aM[LQ], TS[LS], TM, QiS[LS], QiM,
                 C1e[LS], Dsk[LS], QM, C2e[LQ][LS], Ecorr[LQ];
        for (uint32_t i=0;i<LQ;++i){
            u128 qh=Qp/Q[i]; uint64_t qhi=modinv((uint64_t)(qh%Q[i]),Q[i]);
            uint32_t c1a=(uint32_t)(((u128)t*qhi)%Q[i]);
            C1a[i]= montg ? c1a : cQ[i].to_montgomery(c1a);   // raw when input is Montgomery (decodes)
            for(uint32_t j=0;j<LS;++j) C2a[j][i]=cS[j].to_montgomery((uint32_t)(qh%S[j]));
            C2aM[i]=cM.to_montgomery((uint32_t)(qh%Msk));
        }
        for (uint32_t j=0;j<LS;++j){ TS[j]= montg ? (uint32_t)(t%S[j]) : cS[j].to_montgomery(t); QiS[j]=cS[j].to_montgomery((uint32_t)modinv((uint64_t)(Qp%S[j]),S[j])); }
        TM= montg ? (uint32_t)(t%Msk) : cM.to_montgomery(t); QiM=cM.to_montgomery((uint32_t)modinv((uint64_t)(Qp%Msk),Msk));
        for (uint32_t j=0;j<LS;++j){
            u128 sh=Sp/S[j]; uint64_t shi=modinv((uint64_t)(sh%S[j]),S[j]);
            C1e[j]=cS[j].to_montgomery((uint32_t)shi);
            Dsk[j]=cM.to_montgomery((uint32_t)(sh%Msk));
            for(uint32_t i=0;i<LQ;++i) C2e[i][j]=cQ[i].to_montgomery((uint32_t)(sh%Q[i]));
        }
        QM=cM.to_montgomery((uint32_t)modinv((uint64_t)(Sp%Msk),Msk));
        for (uint32_t i=0;i<LQ;++i) Ecorr[i]=cQ[i].to_montgomery((uint32_t)(Sp%Q[i]));

        // ---- device ----
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core={0,0};
        auto mkbuf=[&](uint32_t n){ distributed::DeviceLocalBufferConfig d{.page_size=I32,.buffer_type=BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size=n*I32}, d, mesh.get()); };
        auto b_eQ=mkbuf(LQ), b_eS=mkbuf(LS), b_eMsk=mkbuf(1), b_yQ=mkbuf(LQ), b_prod=mkbuf(LS);
        auto b_fbcS=mkbuf(LS), b_fbcM=mkbuf(1), b_tfS=mkbuf(LS), b_tfM=mkbuf(1), b_dS=mkbuf(LS);
        auto b_tS=mkbuf(LS), b_dM=mkbuf(1), b_tM=mkbuf(1), b_yS=mkbuf(LS), b_Ask=mkbuf(1);
        auto b_diff=mkbuf(1), b_eps=mkbuf(1), b_apx=mkbuf(LQ), b_corr=mkbuf(LQ), b_out=mkbuf(LQ);
        distributed::EnqueueWriteMeshBuffer(cq,b_eQ,eQ,false);
        distributed::EnqueueWriteMeshBuffer(cq,b_eS,eS,false);
        distributed::EnqueueWriteMeshBuffer(cq,b_eMsk,eMsk,false);

        std::vector<distributed::MeshWorkload> wls; wls.reserve(256);
        std::vector<std::shared_ptr<distributed::MeshBuffer>> cbufs;   // const buffers (kept alive)
        auto CB=[&](Program& p, CBIndex i, uint32_t d){ CircularBufferConfig c(d*I32,{{i,tt::DataFormat::UInt32}}); c.set_page_size(i,I32); CreateCircularBuffer(p,core,c); };
        auto run=[&](Program&& p){ wls.emplace_back(); wls.back().add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wls.back(),false); };
        auto RD=[&](Program& p,const char* k){ return CreateKernel(p, std::string(OVERRIDE_KERNEL_PREFIX)+k, core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default}); };
        auto WR=[&](Program& p,const char* k){ return CreateKernel(p, std::string(OVERRIDE_KERNEL_PREFIX)+k, core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default}); };
        auto CC=[&](Program& p,const char* k){ return CreateKernel(p, std::string(OVERRIDE_KERNEL_PREFIX)+k, core, ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false}); };
        // const buffer filled with n mont values
        auto kbuf=[&](std::vector<uint32_t> vals){ auto b=mkbuf((uint32_t)vals.size());
            std::vector<uint32_t> blob(vals.size()*ELEMS); for(size_t i=0;i<vals.size();++i) for(uint32_t e=0;e<ELEMS;++e) blob[i*ELEMS+e]=vals[i];
            distributed::EnqueueWriteMeshBuffer(cq,b,blob,false); cbufs.push_back(b); return b; };

        auto modmul=[&](auto a,uint32_t as, auto b,uint32_t bs, auto o,uint32_t os, uint32_t n, const mont::MontgomeryContext& ctx){
            Program p=CreateProgram(); CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,2);CB(p,CBIndex::c_16,2);
            auto r=RD(p,"dataflow/reader_ab.cpp"); auto w=WR(p,"dataflow/writer_modmul.cpp"); auto c=CC(p,"compute/modmul_sfpi_full.cpp");
            uint32_t q=ctx.prime();
            SetRuntimeArgs(p,r,core,{(uint32_t)a->address(),(uint32_t)b->address(),n,as,bs});
            SetRuntimeArgs(p,w,core,{(uint32_t)o->address(),n,os});
            SetRuntimeArgs(p,c,core,{n,q,ctx.q_inv_32(),q&0xFFFFu,q>>16}); run(std::move(p)); };
        auto modsub=[&](auto a,uint32_t as, auto b,uint32_t bs, auto o,uint32_t os, uint64_t m){
            Program p=CreateProgram(); CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,2);CB(p,CBIndex::c_16,2);
            auto r=RD(p,"dataflow/reader_ab.cpp"); auto w=WR(p,"dataflow/writer_modmul.cpp"); auto c=CC(p,"compute/modsub_reduce.cpp");
            SetRuntimeArgs(p,r,core,{(uint32_t)a->address(),(uint32_t)b->address(),1u,as,bs});
            SetRuntimeArgs(p,w,core,{(uint32_t)o->address(),1u,os});
            SetRuntimeArgs(p,c,core,{(uint32_t)m}); run(std::move(p)); };
        auto modadd=[&](auto in, auto o,uint32_t os, uint32_t n, uint64_t m){
            Program p=CreateProgram(); CB(p,CBIndex::c_0,LS);CB(p,CBIndex::c_16,2);
            auto r=RD(p,"dataflow/reader_i32.cpp"); auto w=WR(p,"dataflow/writer_i32.cpp"); auto c=CC(p,"compute/modadd_reduce.cpp");
            SetRuntimeArgs(p,r,core,{(uint32_t)in->address(),n,0u});
            SetRuntimeArgs(p,w,core,{(uint32_t)o->address(),1u,os});
            SetRuntimeArgs(p,c,core,{n,(uint32_t)m}); run(std::move(p)); };

        // 1. yQ_i
        auto kC1a=kbuf({C1a[0],C1a[1],C1a[2]});
        for(uint32_t i=0;i<LQ;++i) modmul(b_eQ,i,kC1a,i,b_yQ,i,1,cQ[i]);
        // 2. fbcS_j = Σ yQ·C2a[j] ; fbcM
        for(uint32_t j=0;j<LS;++j){ auto k=kbuf({C2a[j][0],C2a[j][1],C2a[j][2]});
            modmul(b_yQ,0,k,0,b_prod,0,LQ,cS[j]); modadd(b_prod,b_fbcS,j,LQ,S[j]); }
        { auto k=kbuf({C2aM[0],C2aM[1],C2aM[2]}); modmul(b_yQ,0,k,0,b_prod,0,LQ,cM); modadd(b_prod,b_fbcM,0,LQ,Msk); }
        // 3. tf
        auto kTS=kbuf({TS[0],TS[1],TS[2],TS[3]});
        for(uint32_t j=0;j<LS;++j) modmul(b_eS,j,kTS,j,b_tfS,j,1,cS[j]);
        { auto k=kbuf({TM}); modmul(b_eMsk,0,k,0,b_tfM,0,1,cM); }
        // 4. ẽ_x = (tf-fbc)·Q^{-1}
        auto kQiS=kbuf({QiS[0],QiS[1],QiS[2],QiS[3]});
        for(uint32_t j=0;j<LS;++j){ modsub(b_tfS,j,b_fbcS,j,b_dS,j,S[j]); modmul(b_dS,j,kQiS,j,b_tS,j,1,cS[j]); }
        { modsub(b_tfM,0,b_fbcM,0,b_dM,0,Msk); auto k=kbuf({QiM}); modmul(b_dM,0,k,0,b_tM,0,1,cM); }
        // 5. exact S->Q
        auto kC1e=kbuf({C1e[0],C1e[1],C1e[2],C1e[3]});
        for(uint32_t j=0;j<LS;++j) modmul(b_tS,j,kC1e,j,b_yS,j,1,cS[j]);
        { auto k=kbuf({Dsk[0],Dsk[1],Dsk[2],Dsk[3]}); modmul(b_yS,0,k,0,b_prod,0,LS,cM); modadd(b_prod,b_Ask,0,LS,Msk); }
        { modsub(b_Ask,0,b_tM,0,b_diff,0,Msk); auto k=kbuf({QM}); modmul(b_diff,0,k,0,b_eps,0,1,cM); }
        auto kEc=kbuf({Ecorr[0],Ecorr[1],Ecorr[2]});
        for(uint32_t i=0;i<LQ;++i){ auto k=kbuf({C2e[i][0],C2e[i][1],C2e[i][2],C2e[i][3]});
            modmul(b_yS,0,k,0,b_prod,0,LS,cQ[i]); modadd(b_prod,b_apx,i,LS,Q[i]);
            modmul(b_eps,0,kEc,i,b_corr,i,1,cQ[i]); modsub(b_apx,i,b_corr,i,b_out,i,Q[i]); }

        std::vector<uint32_t> ob; distributed::EnqueueReadMeshBuffer(cq, ob, b_out, true);
        uint32_t mism=0;
        for(uint32_t i=0;i<LQ;++i) for(uint32_t e=0;e<ELEMS;++e)
            if(ob[i*ELEMS+e]!=gold[i*ELEMS+e]){ if(mism<8) fmt::print("  i={} e={} got={} exp={}\n",i,e,ob[i*ELEMS+e],gold[i*ELEMS+e]); ++mism; }
        fmt::print("bfv_rescale (on device, Q=3 S=4 msk)  mismatches vs golden ẽ = {}/{}\n", mism, LQ*ELEMS);
        pass = (mism==0);
        fmt::print("{}", pass ? "RESULT: on-device rescale WORKS.\n" : "RESULT: WRONG.\n");
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr,"Exception:\n{}\n",e.what()); return 2; }
    fmt::print("{}\n", pass?"Test Passed":"Test FAILED");
    return pass?0:1;
}
