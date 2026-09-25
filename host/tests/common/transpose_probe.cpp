// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Validate compute/transpose_tile.cpp (transpose_wh_tile) == the NTT `reindex` permutation.
// Random 32x32 uint32 tile -> tilize -> device transpose -> untilize -> compare to host transpose.
//   ./build/transpose_probe [seed]
#include <cstdint>
#include <random>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
namespace {
constexpr uint32_t T=32, ELEMS=1024, I32=ELEMS*4;
std::vector<uint32_t> tilize(const std::vector<uint32_t>& in){ std::vector<uint32_t> o(ELEMS);
    for(uint32_t r=0;r<T;++r) for(uint32_t c=0;c<T;++c) o[((r/16)*2+c/16)*256+(r%16)*16+(c%16)]=in[r*T+c]; return o; }
std::vector<uint32_t> untilize(const std::vector<uint32_t>& in){ std::vector<uint32_t> o(ELEMS);
    for(uint32_t r=0;r<T;++r) for(uint32_t c=0;c<T;++c) o[r*T+c]=in[((r/16)*2+c/16)*256+(r%16)*16+(c%16)]; return o; }
}
int main(int argc, char** argv){
    bool pass=true;
    try{
        uint32_t seed=(argc>1)?std::stoul(argv[1]):0;
        std::mt19937 rng(seed); std::uniform_int_distribution<uint32_t> d(0, 268369920u);
        std::vector<uint32_t> x(ELEMS); for(auto&v:x) v=d(rng);
        // host transpose (= reindex): out[c*32+r] = x[r*32+c]
        std::vector<uint32_t> href(ELEMS);
        for(uint32_t r=0;r<T;++r) for(uint32_t c=0;c<T;++c) href[c*T+r]=x[r*T+c];

        auto mesh=distributed::MeshDevice::create_unit_mesh(0);
        auto& cq=mesh->mesh_command_queue();
        auto dr=distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core={0,0};
        auto mkbuf=[&](uint32_t n){ distributed::DeviceLocalBufferConfig dd{.page_size=I32,.buffer_type=BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size=n*I32}, dd, mesh.get()); };
        auto b_in=mkbuf(1), b_out=mkbuf(1);
        distributed::EnqueueWriteMeshBuffer(cq,b_in,tilize(x),false);
        Program p=CreateProgram();
        auto CB=[&](CBIndex i){ CircularBufferConfig c(2*I32,{{i,tt::DataFormat::UInt32}}); c.set_page_size(i,I32); CreateCircularBuffer(p,core,c); };
        CB(CBIndex::c_0); CB(CBIndex::c_16);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/transpose_tile.cpp",core,ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)b_in->address(),1u,0u});
        SetRuntimeArgs(p,wr,core,{(uint32_t)b_out->address(),1u,0u});
        SetRuntimeArgs(p,cp,core,{});
        distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false);
        std::vector<uint32_t> ob; distributed::EnqueueReadMeshBuffer(cq,ob,b_out,true);
        auto got=untilize(ob);
        uint32_t mism=0; for(uint32_t i=0;i<ELEMS;++i) if(got[i]!=href[i]){ if(mism<6) fmt::print("  i={} got={} exp={}\n",i,got[i],href[i]); ++mism; }
        fmt::print("transpose_tile (device) vs host transpose  seed={}  mismatches={}/{}\n", seed, mism, ELEMS);
        pass=(mism==0); pass&=mesh->close();
    }catch(const std::exception& e){ fmt::print(stderr,"Exception:\n{}\n",e.what()); return 2; }
    fmt::print("{}\n", pass?"Test Passed":"Test FAILED");
    return pass?0:1;
}
