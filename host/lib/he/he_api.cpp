// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Implementation of he_api.hpp — all TT-Metalium device control lives here.
// Wraps the validated BFV evaluator: HE add (per-limb modadd) and HE multiply
// (device tensor -> rescale -> relinearize), plus host-side keygen/encrypt/decrypt.
// Params: N=1024, Q=3 / P=3 (QP=6), scale aux S=4, sentinel m_sk, t=12289 — the exact
// configuration validated bit-exact vs the golden in docs/he/BFV_MULTIPLY_PERF.md.
#include "he_api.hpp"
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
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
constexpr uint32_t T=32, N=1024, ELEMS=1024, I8=ELEMS, I32=ELEMS*4;
constexpr uint32_t LQ=3, LP=3, LQP=6, LS=4, LT=LQ+LS+1, NB=4, FBCW=4;
const uint64_t POW[4]={1,128,16384,2097152};
using u128 = unsigned __int128;
uint64_t modpow(uint64_t b,uint64_t e,uint64_t m){ u128 r=1,x=b%m; while(e){ if(e&1) r=r*x%m; x=x*x%m; e>>=1;} return (uint64_t)r; }
uint64_t modinv(uint64_t a,uint64_t m){ return modpow(a%m,m-2,m); }
bool is_prime(uint64_t n){ if(n<2)return false; for(uint64_t d=2;d*d<=n;++d) if(n%d==0)return false; return true; }
uint64_t prim_root(uint64_t q){ uint64_t phi=q-1,n=phi; std::vector<uint64_t> f;
    for(uint64_t d=2;d*d<=n;++d) if(n%d==0){ f.push_back(d); while(n%d==0)n/=d; } if(n>1) f.push_back(n);
    for(uint64_t g=2;g<q;++g){ bool ok=true; for(uint64_t p:f) if(modpow(g,phi/p,q)==1) ok=false; if(ok) return g; } throw std::runtime_error("no root"); }
std::vector<uint64_t> ntt_primes(uint32_t cnt){ std::vector<uint64_t> out;    // p = k·2N+1 (ψ exists), <2^28
    for(uint64_t k=((1ull<<28)-1)/(2*N); out.size()<cnt && k>0; --k){ uint64_t p=k*2*N+1; if(p<(1ull<<28)&&is_prime(p)) out.push_back(p);} return out; }
int8_t limb(uint64_t v,uint32_t p){ return (int8_t)((v>>(7*p))&127); }
std::vector<uint32_t> tilize(const std::vector<uint32_t>& in){ std::vector<uint32_t> o(ELEMS);
    for(uint32_t r=0;r<T;++r) for(uint32_t c=0;c<T;++c) o[((r/16)*2+c/16)*256+(r%16)*16+(c%16)]=in[r*T+c]; return o; }
std::vector<uint32_t> untilize(const std::vector<uint32_t>& in){ std::vector<uint32_t> o(ELEMS);
    for(uint32_t r=0;r<T;++r) for(uint32_t c=0;c<T;++c) o[r*T+c]=in[((r/16)*2+c/16)*256+(r%16)*16+(c%16)]; return o; }
// per-limb negacyclic multiply a*b in Z_q[x]/(x^N+1)
std::vector<uint32_t> negmul(const std::vector<uint32_t>& a,const std::vector<uint32_t>& b,uint64_t q){
    std::vector<uint32_t> c(N,0);
    for(uint32_t i=0;i<N;++i){ if(!a[i])continue; u128 ai=a[i];
        for(uint32_t j=0;j<N;++j){ uint32_t k=i+j; uint64_t v=(uint64_t)(ai*b[j]%q);
            if(k<N) c[k]=(uint32_t)(((u128)c[k]+v)%q); else c[k-N]=(uint32_t)(((u128)c[k-N]+q-v)%q); } }
    return c; }
}  // namespace

namespace he {

struct HEContext::Impl {
    uint32_t t = 12289;
    std::vector<uint64_t> Q, P, S, QP, P8; uint64_t Msk;
    std::vector<mont::MontgomeryContext> cQ, cS, cQP, C8; std::unique_ptr<mont::MontgomeryContext> cM;
    u128 Qp=1, Sp=1, Pp=1;
    std::vector<int64_t> s;                                  // secret key (ternary int poly)
    std::vector<std::vector<uint32_t>> evk0, evk1;           // relin key, QP limbs, TILIZED coeff
    std::mt19937_64 rng;
    std::shared_ptr<distributed::MeshDevice> mesh;

    // ---- residue helpers ----
    std::vector<std::vector<uint32_t>> rowsQ(const std::vector<int64_t>& poly) {   // base-Q residues (natural)
        std::vector<std::vector<uint32_t>> r(LQ, std::vector<uint32_t>(N));
        for (uint32_t i=0;i<LQ;++i){ int64_t q=(int64_t)Q[i]; for(uint32_t n=0;n<N;++n) r[i][n]=(uint32_t)(((poly[n]%q)+q)%q); }
        return r; }

    Impl(uint64_t seed) : rng(seed) {
        auto pr = ntt_primes(LQ+LP+LS+1);
        Q.assign(pr.begin(), pr.begin()+LQ);
        P.assign(pr.begin()+LQ, pr.begin()+LQ+LP);
        S.assign(pr.begin()+LQ+LP, pr.begin()+LQ+LP+LS);
        Msk = pr[LQ+LP+LS];
        QP = Q; QP.insert(QP.end(), P.begin(), P.end());
        P8 = Q; P8.insert(P8.end(), S.begin(), S.end()); P8.push_back(Msk);
        for (auto q:Q) cQ.emplace_back((uint32_t)q);
        for (auto q:S) cS.emplace_back((uint32_t)q);
        for (auto q:QP) cQP.emplace_back((uint32_t)q);
        for (auto q:P8) C8.emplace_back((uint32_t)q);
        cM = std::make_unique<mont::MontgomeryContext>((uint32_t)Msk);
        for (auto q:Q) Qp*=q; for (auto q:S) Sp*=q; for (auto q:P) Pp*=q;
        keygen();
        mesh = distributed::MeshDevice::create_unit_mesh(0);
    }
    ~Impl(){ if (mesh) mesh->close(); }

    // ---- keygen: secret s (ternary) + relin key evk over QP (coeff, tilized) ----
    void keygen() {
        s.resize(N); for (auto& x:s) x=(int64_t)(rng()%3)-1;
        std::vector<int64_t> e_int(N); for (auto& x:e_int) x=(int64_t)(rng()%5)-2;
        std::vector<int64_t> s2(2*N,0);
        for (uint32_t i=0;i<N;++i) for (uint32_t j=0;j<N;++j){ uint32_t k=i+j; if(k<N) s2[k]+=s[i]*s[j]; else s2[k-N]-=s[i]*s[j]; }  // negacyclic wrap
        evk0.assign(LQP, {}); evk1.assign(LQP, {});
        for (uint32_t li=0; li<LQP; ++li) {
            uint64_t m=QP[li]; uint64_t Pm=(uint64_t)(Pp%m);
            std::vector<uint32_t> a(N), sl(N), el(N), s2l(N);
            for (uint32_t n=0;n<N;++n){ a[n]=(uint32_t)(rng()%m); sl[n]=(uint32_t)(((s[n]%(int64_t)m)+m)%m);
                el[n]=(uint32_t)(((e_int[n]%(int64_t)m)+m)%m); s2l[n]=(uint32_t)(((s2[n]%(int64_t)m)+m)%m); }
            auto as_=negmul(a,sl,m);
            std::vector<uint32_t> k0(N);
            for (uint32_t n=0;n<N;++n) k0[n]=(uint32_t)(((u128)m+el[n]+(u128)Pm*s2l[n]-as_[n])%m);
            evk0[li]=tilize(k0); evk1[li]=tilize(a);
        }
    }

    // ---- encrypt one integer as a constant plaintext poly ----
    Ciphertext encrypt(int64_t value) {
        uint32_t m = (uint32_t)(((value % t) + t) % t);
        std::vector<int64_t> e_int(N); for (auto& x:e_int) x=(int64_t)(rng()%5)-2;
        std::vector<int64_t> a_int(N);                                    // one integer mask poly (per-limb reduced)
        Ciphertext ct; ct.c0.assign(LQ,std::vector<uint32_t>(N)); ct.c1.assign(LQ,std::vector<uint32_t>(N));
        // build a as base-Q residues from one random integer poly in [0,Qprod)
        std::vector<std::vector<uint32_t>> aQ(LQ, std::vector<uint32_t>(N));
        for (uint32_t n=0;n<N;++n){ u128 av=((u128)rng()<<64|rng()) % Qp; for(uint32_t i=0;i<LQ;++i) aQ[i][n]=(uint32_t)(av%Q[i]); }
        for (uint32_t i=0;i<LQ;++i){
            uint64_t q=Q[i];
            std::vector<uint32_t> sl(N); for(uint32_t n=0;n<N;++n) sl[n]=(uint32_t)(((s[n]%(int64_t)q)+q)%q);
            auto as_=negmul(aQ[i], sl, q);
            uint64_t Delta_i=(uint64_t)((Qp/t)%q);
            for (uint32_t n=0;n<N;++n){ uint32_t el=(uint32_t)(((e_int[n]%(int64_t)q)+q)%q);
                uint64_t base=(n==0)?((u128)Delta_i*m%q):0;
                ct.c0[i][n]=(uint32_t)(((u128)q+base+el-as_[n])%q);
                ct.c1[i][n]=aQ[i][n]; }
        }
        return ct;
    }

    // ---- decrypt: message = round(t/Q · [c0 + c1·s]_Q) mod t, coefficient 0 ----
    int64_t decrypt(const Ciphertext& ct) {
        // v0 residue per Q-limb (coefficient 0 of c0 + c1*s, negacyclic)
        std::vector<uint64_t> v0(LQ);
        for (uint32_t i=0;i<LQ;++i){ uint64_t q=Q[i];
            std::vector<uint32_t> sl(N); for(uint32_t n=0;n<N;++n) sl[n]=(uint32_t)(((s[n]%(int64_t)q)+q)%q);
            u128 acc = ct.c0[i][0];
            acc += (u128)ct.c1[i][0]*sl[0] % q;                               // k=0 term
            for (uint32_t k=1;k<N;++k) acc += q - ((u128)ct.c1[i][k]*sl[N-k] % q);  // negacyclic: -c1[k]*s[N-k]
            v0[i]=(uint64_t)(acc % q);
        }
        // CRT reconstruct v0 (base Q) -> integer in [0, Qprod)
        u128 v=0; for (uint32_t i=0;i<LQ;++i){ u128 qh=Qp/Q[i]; uint64_t qhi=modinv((uint64_t)(qh%Q[i]),Q[i]);
            v += (u128)v0[i]*qhi % Q[i] * qh; } v %= Qp;
        // center, then round(t*v / Q)
        u128 half=Qp/2; bool neg = v>half;
        u128 mag = neg ? (Qp - v) : v;
        u128 num = (u128)t*mag; u128 r = (num + Qp/2) / Qp;   // round
        int64_t res = (int64_t)(r % t); if (neg) res = ((t - res) % t);
        return res;
    }

    // ================= device evaluator =================
    Ciphertext add(const Ciphertext& a, const Ciphertext& b);
    Ciphertext multiply(const Ciphertext& a, const Ciphertext& b);
};

// --------------------------------------------------------------------------- //
// HE ADD (device): per-limb (c0+d0, c1+d1) mod Q
// --------------------------------------------------------------------------- //
Ciphertext HEContext::Impl::add(const Ciphertext& a, const Ciphertext& b) {
    auto& cq = mesh->mesh_command_queue();
    auto dr = distributed::MeshCoordinateRange(mesh->shape());
    constexpr CoreCoord core={0,0};
    auto mkbuf=[&](uint32_t n){ distributed::DeviceLocalBufferConfig d{.page_size=I32,.buffer_type=BufferType::DRAM};
        return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size=n*I32}, d, mesh.get()); };
    std::vector<distributed::MeshWorkload> wls; wls.reserve(16);
    auto modadd=[&](const std::vector<uint32_t>& x_nat,const std::vector<uint32_t>& y_nat,uint64_t q){
        auto pb=mkbuf(2), ob=mkbuf(1);
        std::vector<uint32_t> blob(2*ELEMS); auto tx=tilize(x_nat), ty=tilize(y_nat);
        std::copy(tx.begin(),tx.end(),blob.begin()); std::copy(ty.begin(),ty.end(),blob.begin()+ELEMS);
        distributed::EnqueueWriteMeshBuffer(cq,pb,blob,false);
        Program p=CreateProgram();
        auto CB=[&](CBIndex i,uint32_t d){ CircularBufferConfig c(d*I32,{{i,tt::DataFormat::UInt32}}); c.set_page_size(i,I32); CreateCircularBuffer(p,core,c); };
        CB(CBIndex::c_0,2); CB(CBIndex::c_16,2);
        auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto w=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto c=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp",core,ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,r,core,{(uint32_t)pb->address(),2u,0u});
        SetRuntimeArgs(p,w,core,{(uint32_t)ob->address(),1u,0u});
        SetRuntimeArgs(p,c,core,{2u,(uint32_t)q});
        wls.emplace_back(); wls.back().add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wls.back(),false);
        std::vector<uint32_t> o; distributed::EnqueueReadMeshBuffer(cq,o,ob,true); return untilize(o); };
    Ciphertext out; out.c0.resize(LQ); out.c1.resize(LQ);
    for (uint32_t i=0;i<LQ;++i){ out.c0[i]=modadd(a.c0[i],b.c0[i],Q[i]); out.c1[i]=modadd(a.c1[i],b.c1[i],Q[i]); }
    return out;
}

// --------------------------------------------------------------------------- //
// HE MULTIPLY (device): tensor (8-limb NTT->pointwise->INTT, Montgomery) -> rescale
// (mont-in) -> relinearize (padded FBC) -> add.  Buffer-resident; reindex on device.
// (Lifted verbatim from the validated device/he/bfv_ct_mul.cpp pipeline.)
// --------------------------------------------------------------------------- //
Ciphertext HEContext::Impl::multiply(const Ciphertext& A, const Ciphertext& B) {
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
    auto modmul=[&](std::shared_ptr<distributed::MeshBuffer> a,uint32_t as, std::shared_ptr<distributed::MeshBuffer> b,uint32_t bs,
                    std::shared_ptr<distributed::MeshBuffer> o,uint32_t os, uint32_t n, const mont::MontgomeryContext& ctx){
        Program p=CreateProgram(); CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,2);CB(p,CBIndex::c_16,2);
        auto r=RD(p,"dataflow/reader_ab.cpp"); auto w=WR(p,"dataflow/writer_modmul.cpp"); auto c=CC(p,"compute/modmul_sfpi_1pass.cpp");
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
    auto modadd2=[&](std::shared_ptr<distributed::MeshBuffer> pairbuf, std::shared_ptr<distributed::MeshBuffer> o,uint32_t os, uint64_t mmod){
        Program p=CreateProgram(); CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_16,2);
        auto r=RD(p,"dataflow/reader_i32.cpp"); auto w=WR(p,"dataflow/writer_i32.cpp"); auto c=CC(p,"compute/modadd_reduce.cpp");
        SetRuntimeArgs(p,r,core,{(uint32_t)pairbuf->address(),2u,0u});
        SetRuntimeArgs(p,w,core,{(uint32_t)o->address(),1u,os});
        SetRuntimeArgs(p,c,core,{2u,(uint32_t)mmod}); run(std::move(p)); };

    // rescale constants (depend only on primes)
    uint32_t C1a[LQ],C2a[LS][LQ],C2aM[LQ],TS[LS],TM,QiS[LS],QiM,C1e[LS],Dsk[LS],QM,C2e[LQ][LS],Ecorr[LQ];
    for (uint32_t i=0;i<LQ;++i){ u128 qh=Qp/Q[i]; uint64_t qhi=modinv((uint64_t)(qh%Q[i]),Q[i]);
        C1a[i]=cQ[i].to_montgomery((uint32_t)(((u128)t*qhi)%Q[i]));
        for(uint32_t j=0;j<LS;++j) C2a[j][i]=cS[j].to_montgomery((uint32_t)(qh%S[j]));
        C2aM[i]=cM->to_montgomery((uint32_t)(qh%Msk)); }
    for (uint32_t j=0;j<LS;++j){ TS[j]=cS[j].to_montgomery(t); QiS[j]=cS[j].to_montgomery((uint32_t)modinv((uint64_t)(Qp%S[j]),S[j])); }
    TM=cM->to_montgomery(t); QiM=cM->to_montgomery((uint32_t)modinv((uint64_t)(Qp%Msk),Msk));
    for (uint32_t j=0;j<LS;++j){ u128 sh=Sp/S[j]; uint64_t shi=modinv((uint64_t)(sh%S[j]),S[j]);
        C1e[j]=cS[j].to_montgomery((uint32_t)shi); Dsk[j]=cM->to_montgomery((uint32_t)(sh%Msk));
        for(uint32_t i=0;i<LQ;++i) C2e[i][j]=cQ[i].to_montgomery((uint32_t)(sh%Q[i])); }
    QM=cM->to_montgomery((uint32_t)modinv((uint64_t)(Sp%Msk),Msk));
    for (uint32_t i=0;i<LQ;++i) Ecorr[i]=cQ[i].to_montgomery((uint32_t)(Sp%Q[i]));
    uint32_t C1aR[LQ]; for(uint32_t i=0;i<LQ;++i){ u128 qh=Qp/Q[i]; uint64_t qhi=modinv((uint64_t)(qh%Q[i]),Q[i]); C1aR[i]=(uint32_t)(((u128)t*qhi)%Q[i]); }
    auto kC1aR=kbuf({C1aR[0],C1aR[1],C1aR[2]});
    auto kTSR=kbuf({(uint32_t)(t%S[0]),(uint32_t)(t%S[1]),(uint32_t)(t%S[2]),(uint32_t)(t%S[3])});
    auto kTMR=kbuf({(uint32_t)(t%Msk)});
    std::vector<std::shared_ptr<distributed::MeshBuffer>> kC2a(LS); for(uint32_t j=0;j<LS;++j) kC2a[j]=kbuf({C2a[j][0],C2a[j][1],C2a[j][2]});
    auto kC2aM=kbuf({C2aM[0],C2aM[1],C2aM[2]});
    auto kQiS=kbuf({QiS[0],QiS[1],QiS[2],QiS[3]}); auto kQiM=kbuf({QiM});
    auto kC1e=kbuf({C1e[0],C1e[1],C1e[2],C1e[3]}); auto kDsk=kbuf({Dsk[0],Dsk[1],Dsk[2],Dsk[3]}); auto kQM=kbuf({QM});
    auto kEc=kbuf({Ecorr[0],Ecorr[1],Ecorr[2]});
    std::vector<std::shared_ptr<distributed::MeshBuffer>> kC2e(LQ); for(uint32_t i=0;i<LQ;++i) kC2e[i]=kbuf({C2e[i][0],C2e[i][1],C2e[i][2],C2e[i][3]});

    auto rescale=[&](const std::vector<std::vector<uint32_t>>& erQ,const std::vector<std::vector<uint32_t>>& erS,
                     const std::vector<std::vector<uint32_t>>& erM)->std::vector<std::vector<uint32_t>>{
        auto b_eQ=put_rows(erQ), b_eS=put_rows(erS), b_eMsk=put_rows(erM);
        auto b_yQ=mkbuf(LQ),b_prod=mkbuf(LS),b_fbcS=mkbuf(LS),b_fbcM=mkbuf(1),b_tfS=mkbuf(LS),b_tfM=mkbuf(1);
        auto b_dS=mkbuf(LS),b_tS=mkbuf(LS),b_dM=mkbuf(1),b_tM=mkbuf(1),b_yS=mkbuf(LS),b_Ask=mkbuf(1);
        auto b_diff=mkbuf(1),b_eps=mkbuf(1),b_apx=mkbuf(LQ),b_corr=mkbuf(LQ),b_out=mkbuf(LQ);
        keep.insert(keep.end(),{b_yQ,b_prod,b_fbcS,b_fbcM,b_tfS,b_tfM,b_dS,b_tS,b_dM,b_tM,b_yS,b_Ask,b_diff,b_eps,b_apx,b_corr,b_out});
        for(uint32_t i=0;i<LQ;++i) modmul(b_eQ,i,kC1aR,i,b_yQ,i,1,cQ[i]);
        for(uint32_t j=0;j<LS;++j){ modmul(b_yQ,0,kC2a[j],0,b_prod,0,LQ,cS[j]); modadd_sum(b_prod,b_fbcS,j,LQ,S[j]); }
        { modmul(b_yQ,0,kC2aM,0,b_prod,0,LQ,*cM); modadd_sum(b_prod,b_fbcM,0,LQ,Msk); }
        for(uint32_t j=0;j<LS;++j) modmul(b_eS,j,kTSR,j,b_tfS,j,1,cS[j]);
        { modmul(b_eMsk,0,kTMR,0,b_tfM,0,1,*cM); }
        for(uint32_t j=0;j<LS;++j){ modsub(b_tfS,j,b_fbcS,j,b_dS,j,S[j]); modmul(b_dS,j,kQiS,j,b_tS,j,1,cS[j]); }
        { modsub(b_tfM,0,b_fbcM,0,b_dM,0,Msk); modmul(b_dM,0,kQiM,0,b_tM,0,1,*cM); }
        for(uint32_t j=0;j<LS;++j) modmul(b_tS,j,kC1e,j,b_yS,j,1,cS[j]);
        { modmul(b_yS,0,kDsk,0,b_prod,0,LS,*cM); modadd_sum(b_prod,b_Ask,0,LS,Msk); }
        { modsub(b_Ask,0,b_tM,0,b_diff,0,Msk); modmul(b_diff,0,kQM,0,b_eps,0,1,*cM); }
        for(uint32_t i=0;i<LQ;++i){ modmul(b_yS,0,kC2e[i],0,b_prod,0,LS,cQ[i]); modadd_sum(b_prod,b_apx,i,LS,Q[i]);
            modmul(b_eps,0,kEc,i,b_corr,i,1,cQ[i]); modsub(b_apx,i,b_corr,i,b_out,i,Q[i]); }
        std::vector<uint32_t> ob; distributed::EnqueueReadMeshBuffer(cq,ob,b_out,true);
        std::vector<std::vector<uint32_t>> res(LQ);
        for(uint32_t i=0;i<LQ;++i) res[i]=std::vector<uint32_t>(ob.begin()+i*ELEMS, ob.begin()+(i+1)*ELEMS);
        return res; };

    // transforms + FBC (buffer-resident NTT via on-device transpose)
    auto b_WfoldA=mkbuf(NB*NB,I8), b_G=mkbuf(1), b_Tm=mkbuf(1), b_Gp=mkbuf(1), b_Wrhs=mkbuf(NB*NB,I8), b_H=mkbuf(1);
    keep.insert(keep.end(),{b_WfoldA,b_G,b_Tm,b_Gp,b_Wrhs,b_H});
    auto transpose_dev=[&](std::shared_ptr<distributed::MeshBuffer> in, std::shared_ptr<distributed::MeshBuffer> out){
        Program p=CreateProgram(); CB(p,CBIndex::c_0,2); CB(p,CBIndex::c_16,2);
        auto r=RD(p,"dataflow/reader_i32.cpp"); auto w=WR(p,"dataflow/writer_i32.cpp"); auto c=CC(p,"compute/transpose_tile.cpp");
        SetRuntimeArgs(p,r,core,{(uint32_t)in->address(),1u,0u}); SetRuntimeArgs(p,w,core,{(uint32_t)out->address(),1u,0u});
        SetRuntimeArgs(p,c,core,{}); run(std::move(p)); };
    auto transform_buf=[&](std::shared_ptr<distributed::MeshBuffer> in_buf, std::shared_ptr<distributed::MeshBuffer> out_buf,
                           uint64_t q, const mont::MontgomeryContext& C, bool fwd){
        uint32_t QINV=C.q_inv_32(),QLO=q&0xFFFFu,QHI=q>>16,K14=C.to_montgomery(16384u);
        uint64_t g=prim_root(q), wf=modpow(g,(q-1)/N,q), w32f=modpow(g,(q-1)/T,q), psif=modpow(g,(q-1)/(2*N),q);
        uint64_t w=fwd?wf:mont::modular_inverse(wf,q), w32=fwd?w32f:mont::modular_inverse(w32f,q), psi=fwd?psif:mont::modular_inverse(psif,q);
        uint64_t Ninv=mont::modular_inverse(N,q);
        std::vector<uint32_t> Wa(ELEMS),Wc(ELEMS),Tw(ELEMS);
        for(uint32_t a=0;a<T;++a) for(uint32_t b=0;b<T;++b){ uint32_t wv=(uint32_t)modpow(w32,(uint64_t)a*b,q);
            if(fwd){ Wa[a*T+b]=(uint32_t)(((u128)wv*modpow(psi,32ull*b,q))%q); Wc[a*T+b]=wv; }
            else   { Wa[a*T+b]=(uint32_t)(((u128)wv*Ninv)%q);                    Wc[a*T+b]=(uint32_t)(((u128)wv*modpow(psi,32ull*b,q))%q); } }
        for(uint32_t a=0;a<T;++a) for(uint32_t b=0;b<T;++b){ uint64_t base=modpow(w,(uint64_t)a*b,q);
            Tw[a*T+b]=(uint32_t)(((u128)base*(fwd?modpow(psi,(uint64_t)b,q):modpow(psi,(uint64_t)a,q)))%q); }
        std::vector<int8_t> WfoldA(NB*NB*ELEMS),Wrhs(NB*NB*ELEMS);   // int8 W-fold tiles (manual int8 tilize)
        for(uint32_t p=0;p<NB;++p) for(uint32_t r=0;r<NB;++r){ std::vector<int8_t> la(ELEMS),lc(ELEMS);
            for(uint32_t i=0;i<ELEMS;++i){ la[i]=limb((uint64_t)((POW[r]*Wa[i])%q),p); lc[i]=limb((uint64_t)((POW[p]*Wc[i])%q),r); }
            for(uint32_t rr=0;rr<T;++rr) for(uint32_t cc=0;cc<T;++cc){ uint32_t di=((rr/16)*2+cc/16)*256+(rr%16)*16+(cc%16);
                WfoldA[(p*NB+r)*ELEMS+di]=la[rr*T+cc]; Wrhs[(p*NB+r)*ELEMS+di]=lc[rr*T+cc]; } }
        std::vector<uint32_t> Tm(ELEMS); { std::vector<uint32_t> tm(ELEMS); for(uint32_t i=0;i<ELEMS;++i) tm[i]=C.to_montgomery(Tw[i]); Tm=tilize(tm); }
        distributed::EnqueueWriteMeshBuffer(cq,b_WfoldA,WfoldA,false);
        distributed::EnqueueWriteMeshBuffer(cq,b_Wrhs,Wrhs,false);
        distributed::EnqueueWriteMeshBuffer(cq,b_Tm,Tm,false);
        { Program p=CreateProgram();
          CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,2,tt::DataFormat::Int8,I8);CB(p,CBIndex::c_2,NB,tt::DataFormat::Int8,I8);CB(p,CBIndex::c_3,NB,tt::DataFormat::Int32,I32);CB(p,CBIndex::c_16,2);
          std::vector<uint32_t> rc; TensorAccessorArgs(*in_buf).append_to(rc); TensorAccessorArgs(*b_WfoldA).append_to(rc);
          auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
          auto w=WR(p,"dataflow/writer_i32.cpp"); auto c=CC(p,"compute/fused_matmul_a_1pass.cpp");
          SetRuntimeArgs(p,r,core,{(uint32_t)in_buf->address(),(uint32_t)b_WfoldA->address()});
          SetRuntimeArgs(p,w,core,{(uint32_t)b_G->address(),1u,0u});
          SetRuntimeArgs(p,c,core,{(uint32_t)q,QINV,QLO,QHI,K14}); run(std::move(p)); }
        { Program p=CreateProgram(); CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,2);CB(p,CBIndex::c_16,2);
          auto r=RD(p,"dataflow/reader_modmul.cpp"); auto w=WR(p,"dataflow/writer_modmul.cpp"); auto c=CC(p,"compute/modmul_sfpi_1pass.cpp");
          SetRuntimeArgs(p,r,core,{(uint32_t)b_G->address(),(uint32_t)b_Tm->address(),1u,0u});
          SetRuntimeArgs(p,w,core,{(uint32_t)b_Gp->address(),1u,0u});
          SetRuntimeArgs(p,c,core,{1u,(uint32_t)q,QINV,QLO,QHI}); run(std::move(p)); }
        { Program p=CreateProgram();
          CB(p,CBIndex::c_0,2);CB(p,CBIndex::c_1,NB*NB,tt::DataFormat::Int8,I8);CB(p,CBIndex::c_2,NB,tt::DataFormat::Int8,I8);CB(p,CBIndex::c_3,NB,tt::DataFormat::Int32,I32);CB(p,CBIndex::c_16,2);
          std::vector<uint32_t> rc; TensorAccessorArgs(*b_Gp).append_to(rc); TensorAccessorArgs(*b_Wrhs).append_to(rc);
          auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
          auto w=WR(p,"dataflow/writer_i32.cpp"); auto c=CC(p,"compute/fused_matmul_c_1pass.cpp");
          SetRuntimeArgs(p,r,core,{(uint32_t)b_Gp->address(),(uint32_t)b_Wrhs->address()});
          SetRuntimeArgs(p,w,core,{(uint32_t)b_H->address(),1u,0u});
          SetRuntimeArgs(p,c,core,{(uint32_t)q,QINV,QLO,QHI,K14}); run(std::move(p)); }
        transpose_dev(b_H,out_buf); };

    constexpr uint32_t FW=FBCW;
    auto fbc_fused=[&](const std::vector<std::vector<uint32_t>>& xsrc,const std::vector<uint32_t>& src,const std::vector<uint32_t>& dst)
        ->std::vector<std::vector<uint32_t>>{
        uint32_t Ls=src.size(),Ld=dst.size(); u128 Spr=1; for(uint32_t i=0;i<Ls;++i) Spr*=QP[src[i]];
        std::vector<uint32_t> C1(FW,0); std::vector<std::vector<uint32_t>> C2(FW,std::vector<uint32_t>(FW,0));
        std::vector<uint32_t> Sm(FW),SmI(FW),Dm(FW),DmI(FW);
        for(uint32_t i=0;i<FW;++i){ uint32_t si=(i<Ls)?src[i]:src[0]; uint64_t qi=QP[si]; Sm[i]=(uint32_t)qi; SmI[i]=cQP[si].q_inv_32();
            if(i<Ls) C1[i]=cQP[si].to_montgomery((uint32_t)modpow((uint64_t)((Spr/qi)%qi),qi-2,qi)); }
        for(uint32_t j=0;j<FW;++j){ uint32_t dj=(j<Ld)?dst[j]:dst[0]; uint64_t pj=QP[dj]; Dm[j]=(uint32_t)pj; DmI[j]=cQP[dj].q_inv_32();
            if(j<Ld) for(uint32_t i=0;i<Ls;++i) C2[i][j]=cQP[dj].to_montgomery((uint32_t)((Spr/QP[src[i]])%pj)); }
        auto b_x=mkbuf(FW), b_out=mkbuf(FW); keep.insert(keep.end(),{b_x,b_out});
        std::vector<uint32_t> xb(FW*ELEMS,0); for(uint32_t i=0;i<Ls;++i) std::copy(xsrc[i].begin(),xsrc[i].end(),xb.begin()+i*ELEMS);
        distributed::EnqueueWriteMeshBuffer(cq,b_x,xb,false);
        std::vector<uint32_t> ca; ca.push_back(FW);
        for(uint32_t i=0;i<FW;++i) ca.push_back(C1[i]);
        for(uint32_t i=0;i<FW;++i) for(uint32_t j=0;j<FW;++j) ca.push_back(C2[i][j]);
        for(uint32_t i=0;i<FW;++i) ca.push_back(Sm[i]);
        for(uint32_t i=0;i<FW;++i) ca.push_back(SmI[i]);
        for(uint32_t j=0;j<FW;++j) ca.push_back(Dm[j]);
        for(uint32_t j=0;j<FW;++j) ca.push_back(DmI[j]);
        Program p=CreateProgram(); CB(p,CBIndex::c_0,8);CB(p,CBIndex::c_2,8);CB(p,CBIndex::c_3,8);CB(p,CBIndex::c_16,8);
        auto r=RD(p,"dataflow/reader_i32.cpp"); auto w=WR(p,"dataflow/writer_i32.cpp"); auto c=CC(p,"compute/baseconv_fbc.cpp");
        SetRuntimeArgs(p,r,core,{(uint32_t)b_x->address(),FW,0u}); SetRuntimeArgs(p,w,core,{(uint32_t)b_out->address(),FW,0u});
        SetRuntimeArgs(p,c,core,ca); run(std::move(p));
        std::vector<uint32_t> ob; distributed::EnqueueReadMeshBuffer(cq,ob,b_out,true);
        std::vector<std::vector<uint32_t>> out(Ld);
        for(uint32_t j=0;j<Ld;++j) out[j]=untilize(std::vector<uint32_t>(ob.begin()+j*ELEMS, ob.begin()+(j+1)*ELEMS));
        return out; };

    auto relin=[&](const std::vector<std::vector<uint32_t>>& c2nat)->std::pair<std::vector<std::vector<uint32_t>>,std::vector<std::vector<uint32_t>>>{
        std::vector<std::vector<uint32_t>> c2qp(LQP);
        for(uint32_t i=0;i<LQ;++i) c2qp[i]=c2nat[i];
        { std::vector<std::vector<uint32_t>> xt(LQ); std::vector<uint32_t> srcIdx(LQ),dstIdx(LP);
          for(uint32_t i=0;i<LQ;++i){ xt[i]=tilize(c2nat[i]); srcIdx[i]=i; } for(uint32_t j=0;j<LP;++j) dstIdx[j]=LQ+j;
          auto pP=fbc_fused(xt,srcIdx,dstIdx); for(uint32_t j=0;j<LP;++j) c2qp[LQ+j]=pP[j]; }
        std::vector<std::vector<uint32_t>> acc0(LQP),acc1(LQP);
        auto rb_in=mkbuf(1),rb_c2f=mkbuf(1),rb_e0=mkbuf(1),rb_e1=mkbuf(1),rb_a0=mkbuf(1),rb_a1=mkbuf(1),rb_acc0=mkbuf(1),rb_acc1=mkbuf(1);
        keep.insert(keep.end(),{rb_in,rb_c2f,rb_e0,rb_e1,rb_a0,rb_a1,rb_acc0,rb_acc1});
        for(uint32_t li=0;li<LQP;++li){ uint64_t m=QP[li];
            auto nntt_cpu=[&](const std::vector<uint32_t>& v){ std::vector<uint32_t> X(N);
                uint64_t g=prim_root(m),w=modpow(g,(m-1)/N,m),psi=modpow(g,(m-1)/(2*N),m);
                for(uint32_t k=0;k<N;++k){ u128 acc=0; for(uint32_t n=0;n<N;++n) acc+=(u128)v[n]*((u128)modpow(psi,n,m)*modpow(w,(uint64_t)n*k,m)%m); X[k]=(uint32_t)(acc%m);} return X; };
            auto e0f=nntt_cpu(untilize(evk0[li])), e1f=nntt_cpu(untilize(evk1[li]));
            std::vector<uint32_t> e0m(N),e1m(N); for(uint32_t n=0;n<N;++n){ e0m[n]=cQP[li].to_montgomery(e0f[n]); e1m[n]=cQP[li].to_montgomery(e1f[n]); }
            distributed::EnqueueWriteMeshBuffer(cq,rb_in,tilize(c2qp[li]),false);
            transform_buf(rb_in,rb_c2f,QP[li],cQP[li],true);
            distributed::EnqueueWriteMeshBuffer(cq,rb_e0,tilize(e0m),false);
            distributed::EnqueueWriteMeshBuffer(cq,rb_e1,tilize(e1m),false);
            modmul(rb_c2f,0,rb_e0,0,rb_a0,0,1,cQP[li]); modmul(rb_c2f,0,rb_e1,0,rb_a1,0,1,cQP[li]);
            transform_buf(rb_a0,rb_acc0,QP[li],cQP[li],false); transform_buf(rb_a1,rb_acc1,QP[li],cQP[li],false);
            std::vector<uint32_t> r0,r1; distributed::EnqueueReadMeshBuffer(cq,r0,rb_acc0,true); distributed::EnqueueReadMeshBuffer(cq,r1,rb_acc1,true);
            acc0[li]=untilize(r0); acc1[li]=untilize(r1); }
        auto moddown=[&](std::vector<std::vector<uint32_t>>& acc){
            std::vector<std::vector<uint32_t>> xP(LP); std::vector<uint32_t> srcIdx(LP),dstIdx(LQ);
            for(uint32_t i=0;i<LP;++i){ xP[i]=tilize(acc[LQ+i]); srcIdx[i]=LQ+i; } for(uint32_t j=0;j<LQ;++j) dstIdx[j]=j;
            auto fbcR=fbc_fused(xP,srcIdx,dstIdx);
            std::vector<std::vector<uint32_t>> out(LQ,std::vector<uint32_t>(N));
            for(uint32_t j=0;j<LQ;++j){ uint64_t q=Q[j]; uint64_t Pinv=modpow((uint64_t)(Pp%q),q-2,q);
                for(uint32_t n=0;n<N;++n){ uint32_t diff=(uint32_t)(((u128)acc[j][n]+q-fbcR[j][n])%q); out[j][n]=(uint32_t)((u128)diff*Pinv%q);} }
            return out; };
        return { moddown(acc0), moddown(acc1) }; };

    // ---- reconstruct integer ct coeffs and build 8-limb tensor inputs (Q|S|msk) ----
    auto to8=[&](const std::vector<std::vector<uint32_t>>& cQrows)->std::vector<std::vector<uint32_t>>{  // -> LT tilized rows
        std::vector<std::vector<uint32_t>> out(LT);
        std::vector<u128> coeff(N);
        for(uint32_t n=0;n<N;++n){ u128 v=0; for(uint32_t i=0;i<LQ;++i){ u128 qh=Qp/Q[i]; uint64_t qhi=modinv((uint64_t)(qh%Q[i]),Q[i]); v+=(u128)cQrows[i][n]*qhi%Q[i]*qh; } coeff[n]=v%Qp; }
        for(uint32_t li=0; li<LT; ++li){ uint64_t p=P8[li]; std::vector<uint32_t> row(N);
            for(uint32_t n=0;n<N;++n) row[n]=(uint32_t)(coeff[n]%p); out[li]=tilize(row); }
        return out; };
    auto c0i=to8(A.c0), c1i=to8(A.c1), d0i=to8(B.c0), d1i=to8(B.c1);

    // ---- TENSOR (buffer-resident) ----
    std::vector<std::vector<std::vector<uint32_t>>> eMont(3, std::vector<std::vector<uint32_t>>(LT));
    {
        auto bc0=mkbuf(1),bc1=mkbuf(1),bd0=mkbuf(1),bd1=mkbuf(1);
        auto bc0f=mkbuf(1),bc1f=mkbuf(1),bd0f=mkbuf(1),bd1f=mkbuf(1);
        auto bE0=mkbuf(1),bE2=mkbuf(1),bT=mkbuf(2),bE1=mkbuf(1),be0=mkbuf(1),be1=mkbuf(1),be2=mkbuf(1);
        keep.insert(keep.end(),{bc0,bc1,bd0,bd1,bc0f,bc1f,bd0f,bd1f,bE0,bE2,bT,bE1,be0,be1,be2});
        for(uint32_t li=0; li<LT; ++li){ uint64_t q=P8[li]; auto& C=C8[li];
            auto encw=[&](std::shared_ptr<distributed::MeshBuffer> buf, const std::vector<uint32_t>& trow){
                auto v=trow; for(auto&x:v) x=C.to_montgomery(x); distributed::EnqueueWriteMeshBuffer(cq,buf,v,false); };
            encw(bc0,c0i[li]); encw(bc1,c1i[li]); encw(bd0,d0i[li]); encw(bd1,d1i[li]);
            transform_buf(bc0,bc0f,q,C,true); transform_buf(bc1,bc1f,q,C,true);
            transform_buf(bd0,bd0f,q,C,true); transform_buf(bd1,bd1f,q,C,true);
            modmul(bc0f,0,bd0f,0,bE0,0,1,C); modmul(bc1f,0,bd1f,0,bE2,0,1,C);
            modmul(bc0f,0,bd1f,0,bT,0,1,C);  modmul(bc1f,0,bd0f,0,bT,1,1,C);
            modadd_sum(bT,bE1,0,2,q);
            transform_buf(bE0,be0,q,C,false); transform_buf(bE1,be1,q,C,false); transform_buf(bE2,be2,q,C,false);
            distributed::EnqueueReadMeshBuffer(cq,eMont[0][li],be0,true);
            distributed::EnqueueReadMeshBuffer(cq,eMont[1][li],be1,true);
            distributed::EnqueueReadMeshBuffer(cq,eMont[2][li],be2,true); }
    }

    // ---- rescale each poly, relin e~2, out=(e~0+k0, e~1+k1) ----
    std::vector<std::vector<std::vector<uint32_t>>> tilde(3);
    for(uint32_t k=0;k<3;++k){
        std::vector<std::vector<uint32_t>> erQ(LQ),erS(LS),erM(1);
        for(uint32_t i=0;i<LQ;++i) erQ[i]=eMont[k][i];
        for(uint32_t j=0;j<LS;++j) erS[j]=eMont[k][LQ+j];
        erM[0]=eMont[k][LQ+LS];
        tilde[k]=rescale(erQ,erS,erM);
    }
    std::vector<std::vector<uint32_t>> c2src(LQ); for(uint32_t i=0;i<LQ;++i) c2src[i]=untilize(tilde[2][i]);
    auto kk=relin(c2src);
    std::vector<std::vector<uint32_t>> k0t(LQ),k1t(LQ); for(uint32_t i=0;i<LQ;++i){ k0t[i]=tilize(kk.first[i]); k1t[i]=tilize(kk.second[i]); }
    auto add_pair=[&](const std::vector<std::vector<uint32_t>>& A_t,const std::vector<std::vector<uint32_t>>& K_t)->std::vector<std::vector<uint32_t>>{
        std::vector<std::vector<uint32_t>> out(LQ);
        for(uint32_t i=0;i<LQ;++i){ auto pb=mkbuf(2); keep.push_back(pb);
            std::vector<uint32_t> blob(2*ELEMS); std::copy(A_t[i].begin(),A_t[i].end(),blob.begin()); std::copy(K_t[i].begin(),K_t[i].end(),blob.begin()+ELEMS);
            distributed::EnqueueWriteMeshBuffer(cq,pb,blob,false);
            auto ob=mkbuf(1); keep.push_back(ob); modadd2(pb,ob,0,Q[i]);
            std::vector<uint32_t> o; distributed::EnqueueReadMeshBuffer(cq,o,ob,true); out[i]=untilize(o); }
        return out; };
    Ciphertext out; out.c0=add_pair(tilde[0],k0t); out.c1=add_pair(tilde[1],k1t);
    return out;
}

// --------------------------------------------------------------------------- //
// public wrappers
// --------------------------------------------------------------------------- //
HEContext::HEContext(uint64_t seed) : p_(std::make_unique<Impl>(seed)) {}
HEContext::~HEContext() = default;
HEContext::HEContext(HEContext&&) noexcept = default;
Ciphertext HEContext::encrypt(int64_t v){ return p_->encrypt(v); }
int64_t    HEContext::decrypt(const Ciphertext& c){ return p_->decrypt(c); }
Ciphertext HEContext::add(const Ciphertext& a,const Ciphertext& b){ return p_->add(a,b); }
Ciphertext HEContext::multiply(const Ciphertext& a,const Ciphertext& b){ return p_->multiply(a,b); }
uint32_t   HEContext::plaintext_modulus() const { return p_->t; }

}  // namespace he
