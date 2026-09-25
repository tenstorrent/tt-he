// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PERFORMANCE harness for the BFV rescale  ẽ = round(t·e/Q)  assembled from existing
// kernels at 4->4 (NON-fused, per the build plan). Numbers are NOT correct here (4->4
// undersizes Bsk; the correct path needs Bsk=5 and L_src->L_dst FBC) — correctness is
// checked separately with the HOST scale (reference/he/bfv_mul_ref.py). This program
// exists only to MEASURE the device scale's cost: it runs the real kernel sequence with
// the right tile counts, so the timing is representative (a lower bound: the correct
// 5-limb scale does ~1 limb more work).
//
// Per polynomial, the scale chains:
//   S1  approx FBC  Q->B      (baseconv_fbc)                 [t folded into C1]
//   S2  t-fold      tf_b = t·e_B          (per-limb modmul)
//   S3  finalize    d   = tf_b - fbc      (per-limb modsub)
//   S4  finalize    ẽ_B = d · Qinv_B      (per-limb modmul)
//   S5  exact FBC   B->Q      (baseconv_fbc_sk)
// The tensor product e (in Q,B,m_sk) is random input here (values irrelevant to timing).
//
//   ./build/bfv_scale [seed]

#include <chrono>
#include <cstdint>
#include <random>
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
constexpr uint32_t NB = 4, ELEMS = 1024, I32 = ELEMS * 4;
using u128 = unsigned __int128;
bool is_prime(uint64_t n){ if(n<2)return false; for(uint64_t d=2;d*d<=n;++d) if(n%d==0)return false; return true; }
}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const uint32_t seed = (argc > 1) ? std::stoul(argv[1]) : 0;
        const uint32_t NPOLY = 3;                         // (e0,e1,e2)

        // 9 primes < 2^28: Q(4), B(4), m_sk(1)
        uint64_t Qm[NB], Bm[NB], Msk;
        { std::vector<uint64_t> pr;
          for (uint64_t c=(1ull<<28)-1; pr.size()<2*NB+1; c-=2) if(is_prime(c)) pr.push_back(c);
          for (uint32_t i=0;i<NB;++i){ Qm[i]=pr[i]; Bm[i]=pr[NB+i]; } Msk=pr[2*NB]; }
        std::vector<mont::MontgomeryContext> ctxQ, ctxB; mont::MontgomeryContext ctxM((uint32_t)Msk);
        for (uint32_t i=0;i<NB;++i){ ctxQ.emplace_back((uint32_t)Qm[i]); ctxB.emplace_back((uint32_t)Bm[i]); }

        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        auto mkbuf = [&](uint32_t n){ distributed::DeviceLocalBufferConfig d{.page_size=I32,.buffer_type=BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size=n*I32}, d, mesh.get()); };

        auto b_eQ=mkbuf(NB), b_eB=mkbuf(NB), b_eMsk=mkbuf(1), b_tconst=mkbuf(NB);
        auto b_fbc=mkbuf(NB), b_tfb=mkbuf(NB), b_d=mkbuf(NB), b_tilde=mkbuf(NB+1), b_out=mkbuf(NB);
        auto b_qinv=mkbuf(NB);

        // random fill (values irrelevant to timing)
        std::mt19937_64 rng(seed);
        auto fill=[&](auto buf,uint32_t n,uint64_t m){ std::vector<uint32_t> v(n*ELEMS);
            for(auto& x:v) x=(uint32_t)(rng()%m); distributed::EnqueueWriteMeshBuffer(cq,buf,v,false); };
        fill(b_eQ,NB,Qm[0]); fill(b_eB,NB,Bm[0]); fill(b_eMsk,1,Msk); fill(b_tconst,NB,Bm[0]);
        fill(b_tilde,NB+1,Qm[0]); fill(b_qinv,NB,Bm[0]);

        std::vector<distributed::MeshWorkload> hold; hold.reserve(256);
        auto run=[&](Program&& p){ hold.emplace_back(); hold.back().add_program(dr,std::move(p));
            distributed::EnqueueMeshWorkload(cq,hold.back(),false); };
        auto mkcb=[&](Program& p, CBIndex idx, uint32_t depth){ CircularBufferConfig c(depth*I32,{{idx,tt::DataFormat::UInt32}});
            c.set_page_size(idx,I32); CreateCircularBuffer(p,core,c); };
        auto CC=[&](Program& p,const char* k){ return CreateKernel(p, std::string(OVERRIDE_KERNEL_PREFIX)+k, core,
            ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false}); };
        auto RD=[&](Program& p,const char* k){ return CreateKernel(p, std::string(OVERRIDE_KERNEL_PREFIX)+k, core,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default}); };
        auto WR=[&](Program& p,const char* k){ return CreateKernel(p, std::string(OVERRIDE_KERNEL_PREFIX)+k, core,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default}); };

        // ---- one FBC dispatch (baseconv_fbc, 4->4). arbitrary constants (timing-only). ----
        auto fbc_approx=[&](std::shared_ptr<distributed::MeshBuffer> in, std::shared_ptr<distributed::MeshBuffer> out){
            Program p=CreateProgram(); mkcb(p,CBIndex::c_0,8); mkcb(p,CBIndex::c_2,8); mkcb(p,CBIndex::c_3,8); mkcb(p,CBIndex::c_16,8);
            auto rd=RD(p,"dataflow/reader_i32.cpp"); auto wr=WR(p,"dataflow/writer_i32.cpp"); auto cp=CC(p,"compute/baseconv_fbc.cpp");
            SetRuntimeArgs(p,rd,core,{(uint32_t)in->address(),NB,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),NB,0u});
            std::vector<uint32_t> ca; ca.push_back(NB);
            for(uint32_t i=0;i<NB;++i) ca.push_back(1u);                       // C1
            for(uint32_t i=0;i<NB*NB;++i) ca.push_back(1u);                    // C2
            for(uint32_t i=0;i<NB;++i) ca.push_back((uint32_t)Bm[i]);          // dst modulus q_i role = B
            for(uint32_t i=0;i<NB;++i) ca.push_back(ctxB[i].q_inv_32());
            for(uint32_t j=0;j<NB;++j) ca.push_back((uint32_t)Bm[j]);
            for(uint32_t j=0;j<NB;++j) ca.push_back(ctxB[j].q_inv_32());
            SetRuntimeArgs(p,cp,core,ca); run(std::move(p)); };

        auto fbc_exact=[&](std::shared_ptr<distributed::MeshBuffer> in, std::shared_ptr<distributed::MeshBuffer> out){
            Program p=CreateProgram(); mkcb(p,CBIndex::c_0,NB+2); mkcb(p,CBIndex::c_2,8); mkcb(p,CBIndex::c_3,8);
            mkcb(p,CBIndex::c_4,2); mkcb(p,CBIndex::c_5,2); mkcb(p,CBIndex::c_16,8);
            auto rd=RD(p,"dataflow/reader_i32.cpp"); auto wr=WR(p,"dataflow/writer_i32.cpp"); auto cp=CC(p,"compute/baseconv_fbc_sk.cpp");
            SetRuntimeArgs(p,rd,core,{(uint32_t)in->address(),NB+1,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),NB,0u});
            std::vector<uint32_t> ca; ca.push_back(NB);
            for(uint32_t i=0;i<NB;++i) ca.push_back(1u);                       // C1
            for(uint32_t i=0;i<NB*NB;++i) ca.push_back(1u);                    // C2
            for(uint32_t i=0;i<NB;++i) ca.push_back((uint32_t)Bm[i]);          // src modulus role = B
            for(uint32_t i=0;i<NB;++i) ca.push_back(ctxB[i].q_inv_32());
            for(uint32_t j=0;j<NB;++j) ca.push_back((uint32_t)Qm[j]);          // dst = Q
            for(uint32_t j=0;j<NB;++j) ca.push_back(ctxQ[j].q_inv_32());
            for(uint32_t i=0;i<NB;++i) ca.push_back(1u);                       // Dsk
            for(uint32_t j=0;j<NB;++j) ca.push_back(1u);                       // Ecorr
            ca.push_back((uint32_t)Msk); ca.push_back(ctxM.q_inv_32()); ca.push_back(1u);   // MSK,MSKINV,QINVM
            SetRuntimeArgs(p,cp,core,ca); run(std::move(p)); };

        // ---- per-limb modmul(a·b mod q) and modsub(a-b mod q) ----
        auto modop=[&](const char* comp, std::shared_ptr<distributed::MeshBuffer> a, std::shared_ptr<distributed::MeshBuffer> bb,
                       std::shared_ptr<distributed::MeshBuffer> out, uint32_t idx, uint64_t q, uint32_t qinv, bool sub){
            Program p=CreateProgram(); mkcb(p,CBIndex::c_0,2); mkcb(p,CBIndex::c_1,2); mkcb(p,CBIndex::c_16,2);
            auto rd=RD(p,"dataflow/reader_ab.cpp"); auto wr=WR(p,"dataflow/writer_modmul.cpp"); auto cp=CC(p,comp);
            SetRuntimeArgs(p,rd,core,{(uint32_t)a->address(),(uint32_t)bb->address(),1u,idx,idx});
            SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,idx});
            if(sub) SetRuntimeArgs(p,cp,core,{(uint32_t)q});
            else    SetRuntimeArgs(p,cp,core,{1u,(uint32_t)q,qinv,(uint32_t)(q&0xFFFF),(uint32_t)(q>>16)});
            run(std::move(p)); };

        // ---------- the scale, one polynomial ----------
        auto scale_one=[&](){
            fbc_approx(b_eQ, b_fbc);                                          // S1: Q->B approx
            for(uint32_t j=0;j<NB;++j) modop("compute/modmul_sfpi_full.cpp", b_eB,b_tconst,b_tfb,j,Bm[j],ctxB[j].q_inv_32(),false); // S2 t-fold
            for(uint32_t j=0;j<NB;++j) modop("compute/modsub_reduce.cpp",   b_tfb,b_fbc,b_d,j,Bm[j],0,true);                        // S3 subtract
            for(uint32_t j=0;j<NB;++j) modop("compute/modmul_sfpi_full.cpp", b_d,b_qinv,b_tilde,j,Bm[j],ctxB[j].q_inv_32(),false);  // S4 xQinv
            fbc_exact(b_tilde, b_out);                                        // S5: B->Q exact
        };

        // BUILD the workloads ONCE (NPOLY scales), then time re-enqueuing the pre-built set.
        for(uint32_t r=0;r<NPOLY;++r) scale_one();     // populates `hold` with all workloads
        distributed::Finish(cq);                        // warm
        const int iters=50;
        auto t0=std::chrono::high_resolution_clock::now();
        for(int it=0;it<iters;++it)
            for(auto& wl : hold) distributed::EnqueueMeshWorkload(cq, wl, false);
        distributed::Finish(cq);
        auto t1=std::chrono::high_resolution_clock::now();
        double us=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/1e3/iters;
        fmt::print("SCALE (4->4 assembled, NON-fused, {} polys/mul): {:.1f} us/mul  ({:.1f} us/poly)\n",
                   NPOLY, us, us/NPOLY);
        fmt::print("NOTE: numbers are WRONG at 4->4 (perf-only); correct scale needs Bsk=5 + L_src->L_dst FBC.\n");
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr,"Exception:\n{}\n",e.what()); return 2; }
    fmt::print("{}\n", pass?"Test Passed":"Test FAILED");
    return pass?0:1;
}
