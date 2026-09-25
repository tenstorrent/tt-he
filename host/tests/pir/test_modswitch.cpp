// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Modulus switch (#11) on device — runs ONCE on the final response ct.  c'_i = round(q'·c_i/q) mod q'.
// Device CRT-combine (validated) reconstructs the 56-bit value C per coefficient; the scale-round
// finalize (round(q'·C/q)) is a once-per-response host step. Validated vs cpu/pir/gen_modswitch_case.py.
//
//   ./build/test_modswitch [reference/pir/modswitch_case.bin]

#include <cstdint>
#include <cstdio>
#include <fstream>
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
namespace { constexpr uint32_t N = 1024, ELEMS = 1024, I32 = ELEMS * 4; using u128 = unsigned __int128;
std::vector<uint32_t> til(const std::vector<uint32_t>& v){ return pir::tilize_poly(v,1);}
std::vector<uint32_t> unt(const std::vector<uint32_t>& v){ return pir::untilize_poly(v,1);} }

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "reference/pir/modswitch_case.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
    auto ru = [&](uint32_t& x){ f.read(reinterpret_cast<char*>(&x),4); };
    uint32_t magic,n,q1,q2,qp; ru(magic);ru(n);ru(q1);ru(q2);ru(qp);
    if (magic!=0x4D535754u||n!=N){ fmt::print(stderr,"bad case\n"); return 2; }
    const uint64_t q=(uint64_t)q1*q2;
    auto rdvec=[&](){ std::vector<uint32_t> v(N); f.read(reinterpret_cast<char*>(v.data()),4*N); return v; };
    std::vector<std::vector<uint32_t>> c(4); for (int i=0;i<4;++i) c[i]=rdvec();   // c0q1,c0q2,c1q1,c1q2
    std::vector<std::vector<uint32_t>> gold(2); for (int i=0;i<2;++i) gold[i]=rdvec();  // c0',c1'
    fmt::print("case {}: n={} q1={} q2={} qprime={}\n", path, n, q1, q2, qp);

    bool pass=true;
    try {
        auto mesh=distributed::MeshDevice::create_unit_mesh(0); auto& cq=mesh->mesh_command_queue();
        auto dr=distributed::MeshCoordinateRange(mesh->shape()); constexpr CoreCoord core={0,0};
        auto mkbuf=[&](uint32_t nt){ distributed::DeviceLocalBufferConfig d{.page_size=I32,.buffer_type=BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size=nt*I32}, d, mesh.get()); };
        auto mkcb=[&](Program& p, CBIndex idx, uint32_t depth){ CircularBufferConfig cc(depth*I32,{{idx,tt::DataFormat::UInt32}}); cc.set_page_size(idx,I32); CreateCircularBuffer(p,core,cc); };
        auto wr=[&](auto b,const std::vector<uint32_t>& v){ distributed::EnqueueWriteMeshBuffer(cq,b,v,false); };
        auto rd=[&](auto b){ std::vector<uint32_t> v; distributed::EnqueueReadMeshBuffer(cq,v,b,true); return v; };

        mont::MontgomeryContext ctx2(q2);
        const uint32_t qinv2=ctx2.q_inv_32(), qlo2=q2&0xFFFF, qhi2=q2>>16;
        const uint32_t inv_mont=ctx2.to_montgomery((uint32_t)mont::modular_inverse(q1,q2)), q1L=q1&0xFFFF, q1H=q1>>16;
        auto bpar=mkbuf(2); { std::vector<uint32_t> par(2*ELEMS); auto tL=til(std::vector<uint32_t>(ELEMS,q1L)); auto tH=til(std::vector<uint32_t>(ELEMS,q1H));
            std::copy(tL.begin(),tL.end(),par.begin()); std::copy(tH.begin(),tH.end(),par.begin()+ELEMS); wr(bpar,par); }
        auto bimont=mkbuf(1); wr(bimont, til(std::vector<uint32_t>(N,inv_mont)));

        // device CRT-combine one component (cq1,cq2 natural) -> C (natural 56-bit host values)
        auto crt=[&](const std::vector<uint32_t>& cq1,const std::vector<uint32_t>& cq2){
            auto ba=mkbuf(1),bb=mkbuf(1),bdiff=mkbuf(1),bt=mkbuf(1),bX=mkbuf(2); wr(ba,til(cq1)); wr(bb,til(cq2));
            { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,2); mkcb(p,CBIndex::c_1,2); mkcb(p,CBIndex::c_16,2);
              auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto w=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/garner_diff.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              SetRuntimeArgs(p,r,core,{(uint32_t)ba->address(),(uint32_t)bb->address(),1u,0u,0u}); SetRuntimeArgs(p,w,core,{(uint32_t)bdiff->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{q2,1u});
              distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); }
            { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,2); mkcb(p,CBIndex::c_1,2); mkcb(p,CBIndex::c_16,2);
              auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto w=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              SetRuntimeArgs(p,r,core,{(uint32_t)bdiff->address(),(uint32_t)bimont->address(),1u,0u}); SetRuntimeArgs(p,w,core,{(uint32_t)bt->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{1u,q2,qinv2,qlo2,qhi2});
              distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); }
            { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,2); mkcb(p,CBIndex::c_1,2); mkcb(p,CBIndex::c_2,2); mkcb(p,CBIndex::c_16,4);
              auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_crt.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto w=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/crt_mul_split.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              SetRuntimeArgs(p,r,core,{(uint32_t)bt->address(),(uint32_t)ba->address(),(uint32_t)bpar->address(),1u}); SetRuntimeArgs(p,w,core,{(uint32_t)bX->address(),2u,0u}); SetRuntimeArgs(p,cp,core,{1u});
              distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); }
            auto raw=rd(bX); std::vector<uint32_t> x0(raw.begin(),raw.begin()+ELEMS), x1(raw.begin()+ELEMS,raw.end());
            auto X0=unt(x0), X1=unt(x1); std::vector<uint64_t> C(N);
            for (uint32_t i=0;i<N;++i) C[i]=(uint64_t)X0[i] | ((uint64_t)X1[i]<<28); return C; };

        // mod-switch finalize (once, host): c'_i = round(q'·C_i/q) mod q'
        auto scale=[&](const std::vector<uint64_t>& C){ std::vector<uint32_t> o(N);
            for (uint32_t i=0;i<N;++i){ u128 num=(u128)qp*C[i]; o[i]=(uint32_t)(((num + q/2)/q) % qp); } return o; };

        auto c0p=scale(crt(c[0],c[1])), c1p=scale(crt(c[2],c[3]));
        uint32_t mism=0; for (uint32_t i=0;i<N;++i){ if (c0p[i]!=gold[0][i]||c1p[i]!=gold[1][i]) ++mism; }
        fmt::print("mod-switch (device CRT + finalize)  mismatches={}/{}\n", mism, 2*N);
        pass=(mism==0); pass &= mesh->close();
    } catch (const std::exception& e){ fmt::print(stderr,"Exception:\n{}\n",e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass?0:1;
}
