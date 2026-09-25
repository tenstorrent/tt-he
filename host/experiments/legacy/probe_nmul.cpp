// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// timing probe driver: cycles per mul16 as the slope over NEXTRA
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <set>
#include <vector>
#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include "common/montgomery.hpp"
using namespace tt; using namespace tt::tt_metal; namespace Dp = tt::tt_metal::distributed;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
namespace { constexpr uint32_t ELEMS=1024, I32=ELEMS*4; using hrc=std::chrono::high_resolution_clock;
double ms(hrc::time_point t){return std::chrono::duration<double,std::milli>(hrc::now()-t).count();}
bool ip(uint64_t n){if(n<2||(n%2==0&&n!=2))return false;for(uint64_t d=3;d*d<=n;d+=2)if(n%d==0)return false;return true;} }
int main(int argc,char**argv){
  const uint32_t T=(argc>1)?std::stoul(argv[1]):64, ITERS=(argc>2)?std::stoul(argv[2]):7;
  uint64_t q=(1ull<<28)-1; while(!(q&1)||!ip(q))--q; const mont::MontgomeryContext C((uint32_t)q);
  try{
    auto mesh=Dp::MeshDevice::create_unit_mesh(0,DEFAULT_L1_SMALL_SIZE); auto&cq=mesh->mesh_command_queue();
    auto dr=Dp::MeshCoordinateRange(mesh->shape()); constexpr CoreCoord core={0,0};
    const CoreRangeSet crs(std::set<CoreRange>{CoreRange(core)});
    auto mk=[&](uint64_t n){Dp::DeviceLocalBufferConfig d{.page_size=I32,.buffer_type=BufferType::DRAM};
      return Dp::MeshBuffer::create(Dp::ReplicatedBufferConfig{.size=n*I32},d,mesh.get());};
    auto bA=mk(T),bB=mk(T),bO=mk(T);
    std::vector<uint32_t> hA(T*ELEMS),hB(T*ELEMS); std::mt19937 rng(7);
    for(auto&v:hA)v=(uint32_t)(rng()%q); for(auto&v:hB)v=(uint32_t)(rng()%q);
    Dp::EnqueueWriteMeshBuffer(cq,bA,hA,false); Dp::EnqueueWriteMeshBuffer(cq,bB,hB,false); Dp::Finish(cq);
    fmt::print("\ncycles per mul16, chained standalone (T={}, one core, 1350 MHz)\n\n",T);
    fmt::print("  {:>7}{:>12}{:>14}\n","N mul16","ms/replay","us/tile");
    std::vector<std::pair<uint32_t,double>> pts;
    for(uint32_t nx : {2u,6u,10u,14u}){
      Program p=CreateProgram();
      auto cb=[&](CBIndex i,uint32_t d){CircularBufferConfig c(d*I32,{{i,tt::DataFormat::UInt32}});c.set_page_size(i,I32);CreateCircularBuffer(p,crs,c);};
      cb(CBIndex::c_0,2); cb(CBIndex::c_1,2); cb(CBIndex::c_16,2);
      auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
      auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
      auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/probe_mulchain.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
      SetRuntimeArgs(p,rd,core,{(uint32_t)bA->address(),(uint32_t)bB->address(),T,0u,0u});
      SetRuntimeArgs(p,wr,core,{(uint32_t)bO->address(),T,0u});
      SetRuntimeArgs(p,cp,core,{T,nx});
      Dp::MeshWorkload wl; wl.add_program(dr,std::move(p));
      Dp::EnqueueMeshWorkload(cq,wl,false); Dp::Finish(cq);
      double b=1e18; for(uint32_t i=0;i<ITERS;++i){auto t0=hrc::now();Dp::EnqueueMeshWorkload(cq,wl,false);Dp::Finish(cq);b=std::min(b,ms(t0));}
      pts.push_back({nx,b}); fmt::print("  {:>7}{:>12.4f}{:>14.3f}\n",nx,b,b*1000.0/T);
    }
    const double slope=((pts[3].second-pts[0].second)*1000.0/T)/12.0;   // us per extra mul16 per tile
    const double cyc=slope*1350.0/32.0;                                  // cycles per mul16 per face-row
    fmt::print("\n  slope: {:.4f} us/tile per extra mul16  ->  {:.2f} cycles per mul16 per face-row\n",slope,cyc);
    fmt::print("  the kernel has 11 mul16 -> multiply is ~{:.0f} of ~130 cycles/face-row = {:.0f}%\n",
               11*cyc, 100.0*11*cyc/130.0);
    mesh->close();
  }catch(const std::exception&e){fmt::print(stderr,"Exception:\n{}\n",e.what());return 2;}
  return 0;
}
