// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_Tg018k — the two compute kernels that are left.  docs/he/exp_tg018k.md
//
// WHY. exp_Tg018m took the CKKS ct_mul from 238.7 ms to 64.9 ms by fixing memory and occupancy
// (docs/he/exp_tg018m.md). What survives is arithmetic, measured with a verified 1-1 dispatch map:
//
//     modmul_sfpi_full   1088 calls   20.357 ms   32.2%
//     baseconv_fbc         15 calls   11.897 ms   18.8%     <- 51% of the multiply between them
//
// Neither responds to scheduling any more. modmul sits 50% above its own single-core floor and the
// FBC is 792 us per dispatch of pure compute. So this experiment goes at the arithmetic.
//
// PART A — modmul. The tree has FIVE implementations of the same Montgomery multiply and uses one of
// them. They differ in how the 32x32-bit product is formed, which is the expensive part:
//
//   modmul_sfpi_full    fp32 BYTE-SPLIT in SFPI. Its header says why: "Wormhole SFPI has no integer
//                       multiply", so each product is four fp32 multiplies of 8-bit halves,
//                       reassembled. Written for Wormhole. This is what exp_Tg018m runs.
//   modmul_compute      the same algorithm on `mul_int_tile` -- the NATIVE integer multiply, which
//                       Blackhole has and which fused_matmul_a already uses for its recombine.
//   modmul_sfpi_hybrid  SFPI passes in LREGs, no DST round trips, mul_int_tile for the products.
//   modmul_sfpi_half    scheduled into 4 DST tiles so it can run HALF-SYNC (dst_full_sync_en=false),
//                       which lets math and pack overlap instead of alternating.
//   modmul_mullo        a fifth variant; included so the shootout is complete rather than curated.
//
// Every one takes the same runtime args and the same CBs, so they are swappable in place. The only
// difference the host must know about is that _half wants half-sync.
//
// PART B — the FBC. `baseconv_fbc` does the CRT combine on the SFPU: for each element, Ls source
// residues are each multiplied by Ld constants and accumulated, so the work is Ls x Ld modular
// multiply-accumulates per element -- 128 of them at the shape exp_Tg018m runs (Ls=16, Ld=8). This
// part measures the cost against (Ls, Ld) to establish whether it really is Ls*Ld-bound, which is
// what decides whether the matrix-engine route (kernels/shared/compute/matmul_fbc.cpp, the same int8 trick
// that makes the modular matmul fast) is worth integrating.
//
// PART C — the SAME FBC on the MATRIX ENGINE, at our shape. Part B measures the SFPU. This part
// measures the FPU alternative so the two can be compared on one card on one run, instead of
// extrapolating from host/tests/baseconv/baseconv_matmul.cpp (which is Ls=Ld=8, one core, N=1024).
//
// The matrix engine cannot multiply 28-bit residues, so `matmul_fbc` decomposes both operands into
// radix-128 limbs: y_i into 4 limbs (LHS lanes = Ls*4) and the folded CRT constant into 4 limbs
// (the NR=4 output planes). One int8 matmul per plane; the planes are then recombined
// (Σ_r 128^r · plane_r) mod p_j by `recombine_rns` on the SFPU, one Montgomery reduction per
// coefficient-tile with a PER-COLUMN modulus.
//
// At our shape Ls=16 the contraction is 16*4 = 64 lanes = TWO tiles, so each plane is the sum of
// NA=2 tile-matmuls accumulated in DST -- hence matmul_fbc_ls16.cpp rather than matmul_fbc.cpp.
//
// LAYOUT. This is the honest catch and it is reported, not hidden. The SFPU FBC is LIMB-MAJOR: a
// tile is 1024 coefficients of ONE residue. The matrix engine needs COEFFICIENT-MAJOR: a tile is 32
// coefficients x 32 lanes. So one of our 1024-coefficient tiles is 32 coefficient-tiles, the
// destinations occupy Ld=8 of 32 output columns, and getting there needs a limb-major ->
// coefficient-major transpose plus the radix-128 decomposition. Part C builds A on the HOST, so the
// number it reports is matmul + recombine ONLY. The transpose is excluded and is not free.
//
// Correctness is checked at every point against a host golden, so a variant that is fast because it
// computes something else cannot be reported as a win.
//
//   ./build/exp_tg018k [G=130] [T=8] [ITERS=3]
//
//   TGK_PART=a|b|c|both|all   which parts to run ("both" = a+b, "all" = a+b+c)
//   TGK_G, TGK_T          cores, tiles per core
//   TGK_LS, TGK_LD        FBC shape list for part B, e.g. TGK_LS=4,8,16 TGK_LD=2,4,8
//   TGK_TC                tiles/core for part C (default = TGK_T; part C's blobs are large)
//   TGK_LDC               destination moduli for part C (default 8 = our shape); Ls stays 16
//   TGK_LDMAX             part B destination-buffer pool (default 8); Ls=16 admits Ld<=11
//
// Device timings come from tt-metal's own per-RISC `*-KERNEL` zones, parsed after mesh->close();
// no zone is added to any kernel. Same method as exp_Tg018dram -- see docs/he/dram_guide.md §7.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <fmt/base.h>
#include <fmt/ranges.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
namespace Dp = tt::tt_metal::distributed;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t ELEMS = 1024, I32 = ELEMS * 4;
constexpr double CLK_MHZ = 1350.0;
using hrc = std::chrono::high_resolution_clock;
double msd(hrc::time_point t) { return std::chrono::duration<double, std::milli>(hrc::now() - t).count(); }

bool is_prime(uint64_t n) { if (n < 2 || (n % 2 == 0 && n != 2)) return false;
    for (uint64_t d = 3; d * d <= n; d += 2) if (n % d == 0) return false; return true; }

std::vector<uint32_t> parse_list(const char* env, std::vector<uint32_t> dflt) {
    const char* e = std::getenv(env); if (!e) return dflt;
    std::vector<uint32_t> v; std::stringstream ss(e); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back((uint32_t)std::stoul(t));
    return v.empty() ? dflt : v; }

// ---- device profiler readback: every `*-KERNEL` dispatch group, in issue order (see dram_guide §7)
struct KStat { bool ok=false; uint32_t ncore=0; double span_us=0, risc_us[5]={0,0,0,0,0}; };
const char* RISC_NAME[5] = {"BRISC","NCRISC","TRISC_0","TRISC_1","TRISC_2"};
std::vector<KStat> parse_profiler(const std::string& path) {
    std::vector<KStat> out; std::ifstream f(path); if (!f) return out;
    std::string line; if (!std::getline(f,line)) return out; if (!std::getline(f,line)) return out;
    std::vector<std::string> hdr; { std::stringstream ss(line); std::string t;
        while (std::getline(ss,t,',')) { size_t a=t.find_first_not_of(" \t\r\n"), b=t.find_last_not_of(" \t\r\n");
            hdr.push_back(a==std::string::npos?"":t.substr(a,b-a+1)); } }
    auto col=[&](const char* n){ for (size_t i=0;i<hdr.size();++i) if (hdr[i]==n) return (int)i; return -1; };
    const int cx=col("core_x"),cy=col("core_y"),cr=col("RISC processor type"),
              ct=col("time[cycles since reset]"),cz=col("zone name"),cp=col("type");
    if (cx<0||cy<0||cr<0||ct<0||cz<0||cp<0) return out;
    struct Iv { uint64_t s,e; uint32_t core,risc; };
    std::vector<Iv> iv; std::map<std::tuple<uint32_t,uint32_t,std::string>,std::vector<uint64_t>> open;
    while (std::getline(f,line)) {
        std::vector<std::string> x; { std::stringstream ss(line); std::string t;
            while (std::getline(ss,t,',')) { size_t a=t.find_first_not_of(" \t\r\n"), b=t.find_last_not_of(" \t\r\n");
                x.push_back(a==std::string::npos?"":t.substr(a,b-a+1)); } }
        if ((int)x.size() <= std::max({cx,cy,cr,ct,cz,cp})) continue;
        const std::string& z = x[cz];
        if (z.size()<7 || z.compare(z.size()-7,7,"-KERNEL")!=0) continue;
        int ri=-1; for (int i=0;i<5;++i) if (x[cr]==RISC_NAME[i]) ri=i;
        if (ri<0) continue;
        const uint32_t core=(uint32_t)std::stoul(x[cx])*1000u+(uint32_t)std::stoul(x[cy]);
        auto key=std::make_tuple(core,(uint32_t)ri,z); const uint64_t t=std::stoull(x[ct]);
        if (x[cp]=="ZONE_START") open[key].push_back(t);
        else if (x[cp]=="ZONE_END" && !open[key].empty()) { iv.push_back({open[key].back(),t,core,(uint32_t)ri}); open[key].pop_back(); }
    }
    if (iv.empty()) return out;
    std::sort(iv.begin(),iv.end(),[](const Iv&a,const Iv&b){return a.s<b.s;});
    auto emit=[&](size_t lo,size_t hi){ std::map<uint32_t,std::pair<uint64_t,uint64_t>> per; KStat s;
        for (size_t i=lo;i<hi;++i){ auto it=per.find(iv[i].core);
            if (it==per.end()) per.emplace(iv[i].core,std::make_pair(iv[i].s,iv[i].e));
            else { it->second.first=std::min(it->second.first,iv[i].s); it->second.second=std::max(it->second.second,iv[i].e); }
            s.risc_us[iv[i].risc]=std::max(s.risc_us[iv[i].risc],(double)(iv[i].e-iv[i].s)/CLK_MHZ); }
        for (auto& [c,p] : per) s.span_us=std::max(s.span_us,(double)(p.second-p.first)/CLK_MHZ);
        s.ncore=(uint32_t)per.size(); s.ok=true; out.push_back(s); };
    size_t g0=0; uint64_t ge=iv[0].e;
    for (size_t i=1;i<iv.size();++i){ if (iv[i].s<=ge) ge=std::max(ge,iv[i].e); else { emit(g0,i); g0=i; ge=iv[i].e; } }
    emit(g0,iv.size()); return out; }

struct Row { std::string what; uint32_t a=0,b=0; double host_ms=0; size_t d0=0,nd=0; KStat k; uint64_t mism=0; bool ran=true; };

// ---- part C: radix-128 limbs and the 32x32 tile order the matrix engine wants
using u128 = unsigned __int128;
constexpr uint32_t TDIM = 32, NR = 4;               // tile edge, radix-128 output planes
const uint64_t POW[NR] = {1, 128, 16384, 2097152};
int8_t limb8(uint64_t v, uint32_t p) { return (int8_t)((v / POW[p]) % 128); }
template <typename E> std::vector<E> tilize(const std::vector<E>& in) {
    std::vector<E> o(ELEMS);
    for (uint32_t r=0;r<TDIM;++r) for (uint32_t c=0;c<TDIM;++c)
        o[((r/16)*2 + c/16)*256 + (r%16)*16 + (c%16)] = in[r*TDIM + c];
    return o; }
template <typename E> std::vector<E> untilize(const E* in) {
    std::vector<E> o(ELEMS);
    for (uint32_t r=0;r<TDIM;++r) for (uint32_t c=0;c<TDIM;++c)
        o[r*TDIM + c] = in[((r/16)*2 + c/16)*256 + (r%16)*16 + (c%16)];
    return o; }
}  // namespace

int main(int argc, char** argv) {
    uint32_t G = (argc>1)?(uint32_t)std::stoul(argv[1]):130;
    uint32_t T = (argc>2)?(uint32_t)std::stoul(argv[2]):8;
    uint32_t ITERS = (argc>3)?(uint32_t)std::stoul(argv[3]):3;
    if (const char* e=std::getenv("TGK_G")) G=(uint32_t)std::atoi(e);
    if (const char* e=std::getenv("TGK_T")) T=(uint32_t)std::atoi(e);
    const std::string PART = std::getenv("TGK_PART") ? std::getenv("TGK_PART") : "both";

    uint64_t q = (1ull<<28)-1; while (!(q&1) || !is_prime(q)) --q;
    const mont::MontgomeryContext C((uint32_t)q);

    bool pass = true;
    try {
        auto mesh = Dp::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE);
        auto& cq = mesh->mesh_command_queue();
        auto dr = Dp::MeshCoordinateRange(mesh->shape());
        auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX=grid.x, GMAX=grid.x*grid.y;
        G = std::min(G, GMAX);
        auto CC=[&](uint32_t i){ return CoreCoord{i%GX, i/GX}; };
        auto crs_of=[&](uint32_t c){ std::set<CoreRange> ss; for (uint32_t i=0;i<c;++i) ss.insert(CoreRange(CC(i))); return CoreRangeSet(ss); };
        auto mkbufp=[&](uint64_t nt, uint32_t page){ Dp::DeviceLocalBufferConfig d{.page_size=page,.buffer_type=BufferType::DRAM};
            return Dp::MeshBuffer::create(Dp::ReplicatedBufferConfig{.size=nt*page}, d, mesh.get()); };
        auto mkbuf=[&](uint64_t nt){ return mkbufp(nt, I32); };
        auto mkcb=[&](Program& p, const CoreRangeSet& crs, CBIndex idx, uint32_t depth){
            CircularBufferConfig c(depth*I32, {{idx, tt::DataFormat::UInt32}}); c.set_page_size(idx, I32);
            CreateCircularBuffer(p, crs, c); };

        // Part B's buffer pool. LDMAX is raised past 8 only to measure the SFPU's SLOPE in Ld
        // (part C shows the FPU route is flat in Ld, so the slope is what decides the crossover);
        // the kernel's runtime-arg budget caps Ls=16 at Ld<=11.
        const uint32_t LSMAX = 16;
        uint32_t LDMAX = 8; if (const char* e=std::getenv("TGK_LDMAX")) LDMAX=(uint32_t)std::atoi(e);
        const uint64_t NT = (uint64_t)G*T;
        fmt::print("\nexp_Tg018k — the two compute kernels exp_Tg018m could not schedule away\n");
        fmt::print("grid {}x{}={}, using G={}, T={} tiles/core, q={} (28-bit), {} timed replays\n",
                   grid.x, grid.y, GMAX, G, T, q, ITERS);

        // buffers: LSMAX sources, LDMAX destinations, all NT tiles
        std::vector<std::shared_ptr<Dp::MeshBuffer>> SRC(LSMAX), DST(LDMAX);
        for (auto& b : SRC) b = mkbuf(NT);
        for (auto& b : DST) b = mkbuf(NT);
        auto bB = mkbuf(NT);

        std::vector<std::vector<uint32_t>> hS(LSMAX, std::vector<uint32_t>(NT*ELEMS));
        std::vector<uint32_t> hB(NT*ELEMS), gMM(NT*ELEMS), poison(NT*ELEMS, 0xDEADBEEFu), got;
        { std::mt19937 rng(31337);
          for (uint32_t i=0;i<LSMAX;++i) for (auto& v : hS[i]) v = (uint32_t)(rng()%q);
          for (auto& v : hB) v = (uint32_t)(rng()%q);
          for (size_t e=0;e<NT*ELEMS;++e) gMM[e] = C.mulmod(hS[0][e], hB[e]); }
        for (uint32_t i=0;i<LSMAX;++i) Dp::EnqueueWriteMeshBuffer(cq, SRC[i], hS[i], false);
        Dp::EnqueueWriteMeshBuffer(cq, bB, hB, false);
        Dp::Finish(cq);

        const char* home = std::getenv("TT_METAL_HOME");
        const std::string PCSV = home ? std::string(home)+"/generated/profiler/.logs/profile_log_device.csv" : "";
        const bool PROF = std::getenv("TT_METAL_DEVICE_PROFILER") && !PCSV.empty();
        if (!PROF) fmt::print("\nNOTE: profiler off -- device columns blank."
                              "  re-run as TT_METAL_DEVICE_PROFILER=1 ...\n");

        std::vector<Row> rows; size_t ndisp = 0;
        auto run = [&](Row r, Dp::MeshWorkload& wl, const std::function<uint64_t()>& check) {
            r.d0 = ndisp;
            Dp::EnqueueWriteMeshBuffer(cq, DST[0], poison, false); Dp::Finish(cq);
            Dp::EnqueueMeshWorkload(cq, wl, false); Dp::Finish(cq); ++ndisp;
            // A mismatch here is a RESULT, not a failure: the point of part A is to find out which
            // of the five implementations actually compute the same function. Part B's shapes must
            // all be exact, so only those fail the run.
            r.mism = check(); if (r.mism && r.a != 0) pass = false;
            double mn = 1e18;
            for (uint32_t i=0;i<ITERS;++i){ auto t0=hrc::now(); Dp::EnqueueMeshWorkload(cq,wl,false); Dp::Finish(cq); mn=std::min(mn,msd(t0)); }
            ndisp += ITERS; r.host_ms = mn; r.nd = ndisp - r.d0;
            if (PROF) { Dp::Finish(cq); ReadMeshDeviceProfilerResults(*mesh); }
            fmt::print("  {:<22} {:>8.3f} ms host   mism {}\n", r.what, r.host_ms, r.mism);
            rows.push_back(r); };

        // ================= PART A: the five modmul implementations =================
        if (PART=="a" || PART=="both" || PART=="all") {
            fmt::print("\n=== PART A — modmul: five implementations of the same Montgomery multiply ===\n");
            struct MV { const char* name; const char* path; bool full_sync; };
            const std::vector<MV> V = {
                {"sfpi_full (in use)", "compute/modmul_sfpi_full.cpp", true},
                {"compute (mul_int)",  "compute/modmul_compute.cpp",   true},
                {"sfpi_hybrid",        "compute/modmul_sfpi_hybrid.cpp", true},
                {"sfpi_half",          "compute/modmul_sfpi_half.cpp", false},
                {"mullo",              "compute/modmul_mullo.cpp",     true},
                {"mul24 (sfpmul24)",   "compute/modmul_mul24.cpp",     true},
                {"mul24 + half-sync",  "compute/modmul_mul24_half.cpp", false},
            };
            for (auto& v : V) {
                Program p = CreateProgram(); auto crs = crs_of(G);
                mkcb(p,crs,CBIndex::c_0,2); mkcb(p,crs,CBIndex::c_1,2); mkcb(p,crs,CBIndex::c_16,2);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab_depth.cpp",crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32_depth.cpp",crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                KernelHandle cp;
                try { cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX+std::string(v.path), crs,
                          ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=v.full_sync,.math_approx_mode=false}); }
                catch (const std::exception& e) { fmt::print("  {:<22} SKIPPED — {}\n", v.name,
                          std::string(e.what()).substr(0,60)); continue; }
                for (uint32_t k=0;k<G;++k){ auto c=CC(k); const uint32_t st=k*T;
                    SetRuntimeArgs(p,rd,c,{(uint32_t)SRC[0]->address(),(uint32_t)bB->address(),T,st,1u,0u,(uint32_t)NT});
                    SetRuntimeArgs(p,wr,c,{(uint32_t)DST[0]->address(),T,st,1u,0u,(uint32_t)NT,(uint32_t)SRC[0]->address()});
                    SetRuntimeArgs(p,cp,c,{T,(uint32_t)q,C.q_inv_32(),(uint32_t)(q&0xFFFF),(uint32_t)(q>>16)}); }
                Dp::MeshWorkload wl; wl.add_program(dr, std::move(p));
                Row r; r.what = v.name;
                run(r, wl, [&]() -> uint64_t { Dp::EnqueueReadMeshBuffer(cq, got, DST[0], true);
                    uint64_t m=0; for (uint64_t e=0;e<NT*ELEMS;++e) if (got[e]!=gMM[e]) ++m; return m; });
            }
        }

        // ================= PART B: what the FBC's cost is made of =================
        // The kernel does Ls x Ld modular multiply-accumulates per element. If its cost really is
        // proportional to that product, the matrix-engine route (kernels/shared/compute/matmul_fbc.cpp) is
        // attacking the right term; if not, the cost is elsewhere and that integration is wasted.
        if (PART=="b" || PART=="both" || PART=="all") {
            fmt::print("\n=== PART B — baseconv_fbc: is the cost really Ls x Ld? ===\n");
            auto Lsl = parse_list("TGK_LS", {4,8,16});
            auto Ldl = parse_list("TGK_LD", {2,4,8});
            std::vector<uint64_t> P(LDMAX); { uint64_t p2=q-2; for (uint32_t j=0;j<LDMAX;++j){ while(!(p2&1)||!is_prime(p2)) --p2; P[j]=p2; p2-=2; } }
            std::vector<mont::MontgomeryContext> PC; for (uint32_t j=0;j<LDMAX;++j) PC.emplace_back((uint32_t)P[j]);
            for (uint32_t Ls : Lsl) for (uint32_t Ld : Ldl) {
                std::vector<uint32_t> C1(Ls), C2(Ls*Ld);
                { std::mt19937 rng(7+Ls*31+Ld); for (auto& v : C1) v = C.to_montgomery((uint32_t)(rng()%q));
                  for (uint32_t i=0;i<Ls;++i) for (uint32_t j=0;j<Ld;++j) C2[i*Ld+j]=PC[j].to_montgomery((uint32_t)(rng()%P[j])); }
                std::vector<uint32_t> ca{Ls, Ld};
                for (uint32_t i=0;i<Ls;++i) ca.push_back(C1[i]);
                for (uint32_t i=0;i<Ls;++i) for (uint32_t j=0;j<Ld;++j) ca.push_back(C2[i*Ld+j]);
                for (uint32_t i=0;i<Ls;++i) ca.push_back((uint32_t)q);
                for (uint32_t i=0;i<Ls;++i) ca.push_back(C.q_inv_32());
                for (uint32_t j=0;j<Ld;++j) ca.push_back((uint32_t)P[j]);
                for (uint32_t j=0;j<Ld;++j) ca.push_back(PC[j].q_inv_32());
                if (ca.size() > 250) { fmt::print("  Ls={:<3} Ld={:<3} SKIPPED — {} runtime args exceeds the budget\n", Ls, Ld, ca.size()); continue; }

                Program p = CreateProgram(); auto crs = crs_of(G);
                mkcb(p,crs,CBIndex::c_0,Ls); mkcb(p,crs,CBIndex::c_2,Ls); mkcb(p,crs,CBIndex::c_3,Ls); mkcb(p,crs,CBIndex::c_16,Ld);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_gather_batch_pg.cpp",crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_scatter_batch_pg.cpp",crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/baseconv_fbc.cpp",crs,
                    ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t k=0;k<G;++k){ auto c=CC(k);
                    std::vector<uint32_t> ra{Ls, Ls, k*T};
                    for (uint32_t i=0;i<Ls;++i) ra.push_back((uint32_t)SRC[i]->address());
                    SetRuntimeArgs(p,rd,c,ra);
                    std::vector<uint32_t> wa{Ld, Ld, Ld, k*T};
                    for (uint32_t j=0;j<Ld;++j) wa.push_back((uint32_t)DST[j]->address());
                    SetRuntimeArgs(p,wr,c,wa);
                    SetRuntimeArgs(p,cp,c,ca); }
                Dp::MeshWorkload wl; wl.add_program(dr, std::move(p));
                Row r; r.what = fmt::format("Ls={} Ld={}", Ls, Ld); r.a=Ls; r.b=Ld;
                // golden: y_i = mont(x_i, C1_i) mod q ; out_j = sum_i mont(y_i, C2_ij) mod p_j
                run(r, wl, [&]() -> uint64_t {
                    Dp::EnqueueReadMeshBuffer(cq, got, DST[0], true);
                    uint64_t m=0;
                    for (uint32_t k=0;k<G;++k) for (uint32_t e=0;e<ELEMS;++e) {
                        const size_t off=(size_t)k*T*ELEMS+e; uint64_t acc=0;
                        for (uint32_t i=0;i<Ls;++i) { const uint32_t y=C.mulmod(hS[i][off], C1[i]);
                            acc = (acc + PC[0].mulmod(y, C2[i*Ld+0])) % P[0]; }
                        if (got[off] != (uint32_t)acc) ++m; }
                    return m; });
            }
        }

        // ================= PART C: the same FBC on the matrix engine =================
        // Ls=16 sources -> Ld=8 destinations, the shape exp_Tg018m actually runs. Two programs:
        //   1. matmul_fbc_ls16   NA=2 int8 matmuls per plane, NR=4 planes per coefficient-tile
        //   2. recombine_rns     (Σ_r 128^r · plane_r) mod p_j, per-column modulus, on the SFPU
        // Each is timed in its own Finish'd loop so the profiler sees one dispatch group per replay
        // and the two costs are separable. Verified against a host golden at every output.
        if (PART=="c" || PART=="all") {
            // Ld is a knob here because the FPU route's cost is nearly INDEPENDENT of it (the
            // matmul and the recombine both run on full 32-wide tiles either way) while the SFPU's
            // is linear in it -- so Ld is where the two curves cross. Ls stays 16, our shape.
            uint32_t LdC = 8; if (const char* e=std::getenv("TGK_LDC")) LdC=(uint32_t)std::atoi(e);
            const uint32_t LsC = 16, NA = LsC*NR/TDIM, NCT = ELEMS/TDIM, I8 = ELEMS;
            uint32_t TC = T; if (const char* e=std::getenv("TGK_TC")) TC=(uint32_t)std::atoi(e);
            const uint64_t NOT = (uint64_t)G*TC;          // our-tiles (1024 coefficients each)
            const uint64_t NCTT = NOT*NCT;                // coefficient-tiles
            fmt::print("\n=== PART C — the same FBC on the MATRIX ENGINE (Ls={}, Ld={}) ===\n", LsC, LdC);
            fmt::print("  {} our-tiles x {} coefficient-tiles, NA={} A-tiles/coeff-tile, NR={} planes\n",
                       NOT, NCT, NA, NR);

            // 16 source + 8 destination moduli, all 28-bit, taken the same way as the rest of the file
            std::vector<uint64_t> Qm(LsC), Pm(LdC);
            { std::vector<uint64_t> pr; for (uint64_t c=(1ull<<28)-1; pr.size()<LsC+LdC; c-=2) if (is_prime(c)) pr.push_back(c);
              for (uint32_t i=0;i<LsC;++i) Qm[i]=pr[i];
              for (uint32_t j=0;j<LdC;++j) Pm[j]=pr[LsC+j]; }
            std::vector<mont::MontgomeryContext> ctxP; for (uint32_t j=0;j<LdC;++j) ctxP.emplace_back((uint32_t)Pm[j]);

            // The CRT fold constants. Their VALUES do not matter to a timing or an exactness check --
            // only their range does (< p_j, same as a real q̂_i mod p_j), so they are drawn at random
            // rather than derived from a 448-bit Q that would not fit a u128.
            std::vector<std::vector<uint64_t>> qhat_mod_p(LsC, std::vector<uint64_t>(LdC));
            std::mt19937_64 rngC(20250803);
            for (uint32_t i=0;i<LsC;++i) for (uint32_t j=0;j<LdC;++j) qhat_mod_p[i][j] = rngC()%Pm[j];

            // par: 5 per-column tiles, column c -> p_{c<Ld?c:0}. order K14,QINV,QHIGH,QLOW,Q.
            std::vector<uint32_t> parblob(5*ELEMS);
            for (uint32_t r=0;r<TDIM;++r) for (uint32_t c=0;c<TDIM;++c) {
                const uint32_t j=(c<LdC)?c:0, at=r*TDIM+c;
                parblob[0*ELEMS+at]=ctxP[j].to_montgomery(16384u);
                parblob[1*ELEMS+at]=ctxP[j].q_inv_32();
                parblob[2*ELEMS+at]=(uint32_t)(Pm[j]>>16);
                parblob[3*ELEMS+at]=(uint32_t)(Pm[j]&0xFFFFu);
                parblob[4*ELEMS+at]=(uint32_t)Pm[j]; }
            for (uint32_t t=0;t<5;++t) { std::vector<uint32_t> rm(parblob.begin()+t*ELEMS, parblob.begin()+(t+1)*ELEMS);
                auto til=tilize(rm); std::copy(til.begin(), til.end(), parblob.begin()+t*ELEMS); }

            // B: B_r[(i*4+p)][j] = limb_r( (128^p · q̂_ij · R) mod p_j ), R folded in so the output is
            // Montgomery-encoded (pure constant merge; the recombine kernel is unchanged).
            std::vector<int8_t> Bblob(NR*NA*I8, 0);
            for (uint32_t r=0;r<NR;++r) {
                std::vector<int8_t> Brow(NA*ELEMS, 0);
                for (uint32_t i=0;i<LsC;++i) for (uint32_t p=0;p<NR;++p) for (uint32_t j=0;j<LdC;++j) {
                    uint64_t Bp = (uint64_t)(((u128)POW[p]*qhat_mod_p[i][j]) % Pm[j]);
                    Bp = ctxP[j].to_montgomery((uint32_t)Bp);
                    const uint32_t lane = i*NR+p;                       // 0..LsC*NR-1
                    Brow[(lane/TDIM)*ELEMS + (lane%TDIM)*TDIM + j] = limb8(Bp, r); }
                for (uint32_t s=0;s<NA;++s) { std::vector<int8_t> one(Brow.begin()+s*ELEMS, Brow.begin()+(s+1)*ELEMS);
                    auto til=tilize(one); std::copy(til.begin(), til.end(), Bblob.begin()+(r*NA+s)*ELEMS); } }

            // A: A_ct[c'][i*4+p] = limb_p(y[ct*32+c'][i]).  y drawn per (coefficient, source).
            fmt::print("  building host blobs ({:.0f} MB A, {:.0f} MB planes) ...\n",
                       (double)NCTT*NA*I8/1e6, (double)NCTT*NR*I32/1e6);
            std::vector<int8_t> Ablob((size_t)NCTT*NA*I8, 0);
            std::vector<uint32_t> refC((size_t)NCTT*TDIM*LdC);
            {
                std::vector<int8_t> Arow(NA*ELEMS);
                std::vector<uint64_t> yy(TDIM*LsC);
                for (uint64_t ct=0; ct<NCTT; ++ct) {
                    for (uint32_t cp=0; cp<TDIM; ++cp) for (uint32_t i=0;i<LsC;++i) yy[cp*LsC+i] = rngC()%Qm[i];
                    std::fill(Arow.begin(), Arow.end(), (int8_t)0);
                    for (uint32_t cp=0; cp<TDIM; ++cp) for (uint32_t i=0;i<LsC;++i) for (uint32_t p=0;p<NR;++p) {
                        const uint32_t lane=i*NR+p;
                        Arow[(lane/TDIM)*ELEMS + cp*TDIM + (lane%TDIM)] = limb8(yy[cp*LsC+i], p); }
                    for (uint32_t s=0;s<NA;++s) { std::vector<int8_t> one(Arow.begin()+s*ELEMS, Arow.begin()+(s+1)*ELEMS);
                        auto til=tilize(one); std::copy(til.begin(), til.end(), Ablob.begin()+(size_t)(ct*NA+s)*ELEMS); }
                    for (uint32_t cp=0; cp<TDIM; ++cp) for (uint32_t j=0;j<LdC;++j) {
                        u128 acc=0; for (uint32_t i=0;i<LsC;++i) acc += (u128)yy[cp*LsC+i]*qhat_mod_p[i][j];
                        refC[(size_t)(ct*TDIM+cp)*LdC+j] = ctxP[j].to_montgomery((uint32_t)(acc % Pm[j])); }
                }
            }

            auto bA = mkbufp(NCTT*NA, I8), bB = mkbufp(NR*NA, I8);
            auto bPl = mkbufp(NCTT*NR, I32), bPar = mkbufp(5, I32), bOut = mkbufp(NCTT, I32);
            Dp::EnqueueWriteMeshBuffer(cq, bA, Ablob, false);
            Dp::EnqueueWriteMeshBuffer(cq, bB, Bblob, false);
            Dp::EnqueueWriteMeshBuffer(cq, bPar, parblob, false);
            Dp::Finish(cq);

            auto mkcbf=[&](Program& p, const CoreRangeSet& crs, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page){
                CircularBufferConfig c(depth*page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, crs, c); };
            auto crs = crs_of(G);
            const uint32_t nct_core = TC*NCT;

            Program p1 = CreateProgram();
            mkcbf(p1, crs, CBIndex::c_0, tt::DataFormat::Int8, 4*NA, I8);
            mkcbf(p1, crs, CBIndex::c_1, tt::DataFormat::Int8, NR*NA, I8);
            mkcbf(p1, crs, CBIndex::c_16, tt::DataFormat::Int32, 8, I32);
            std::vector<uint32_t> rc1; TensorAccessorArgs(*bA).append_to(rc1); TensorAccessorArgs(*bB).append_to(rc1);
            auto rd1=CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fbc_ls16.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc1});
            auto wr1=CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp1=CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "compute/matmul_fbc_ls16.cpp", crs,
                ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.math_approx_mode=false});
            for (uint32_t k=0;k<G;++k){ auto c=CC(k); const uint32_t ct0=k*nct_core;
                SetRuntimeArgs(p1, rd1, c, {(uint32_t)bA->address(),(uint32_t)bB->address(),nct_core,NA,ct0});
                SetRuntimeArgs(p1, wr1, c, {(uint32_t)bPl->address(), NR*nct_core, NR*ct0});
                SetRuntimeArgs(p1, cp1, c, {nct_core, NA}); }

            Program p2 = CreateProgram();
            mkcbf(p2, crs, CBIndex::c_0, tt::DataFormat::UInt32, 2*NR, I32);
            mkcbf(p2, crs, CBIndex::c_1, tt::DataFormat::UInt32, 5, I32);
            mkcbf(p2, crs, CBIndex::c_16, tt::DataFormat::UInt32, 4, I32);
            auto rd2=CreateKernel(p2, OVERRIDE_KERNEL_PREFIX "dataflow/reader_recombine_rns_off.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr2=CreateKernel(p2, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp2=CreateKernel(p2, OVERRIDE_KERNEL_PREFIX "compute/recombine_rns.cpp", crs,
                ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,
                              .dst_full_sync_en=true,.math_approx_mode=false});
            for (uint32_t k=0;k<G;++k){ auto c=CC(k); const uint32_t ct0=k*nct_core;
                SetRuntimeArgs(p2, rd2, c, {(uint32_t)bPl->address(),(uint32_t)bPar->address(),nct_core,ct0});
                SetRuntimeArgs(p2, wr2, c, {(uint32_t)bOut->address(), nct_core, ct0});
                SetRuntimeArgs(p2, cp2, c, {nct_core}); }

            Dp::MeshWorkload w1; w1.add_program(dr, std::move(p1));
            Dp::MeshWorkload w2; w2.add_program(dr, std::move(p2));

            auto stage=[&](const char* name, Dp::MeshWorkload& wl) -> Row {
                Row r; r.what = name; r.a = 100; r.d0 = ndisp;
                Dp::EnqueueMeshWorkload(cq, wl, false); Dp::Finish(cq); ++ndisp;
                double mn=1e18;
                for (uint32_t i=0;i<ITERS;++i){ auto t0=hrc::now(); Dp::EnqueueMeshWorkload(cq,wl,false); Dp::Finish(cq); mn=std::min(mn,msd(t0)); }
                ndisp += ITERS; r.host_ms = mn; r.nd = ndisp - r.d0;
                if (PROF) { Dp::Finish(cq); ReadMeshDeviceProfilerResults(*mesh); }
                fmt::print("  {:<22} {:>8.3f} ms host\n", r.what, r.host_ms);
                return r; };
            Row rm = stage("C matmul (FPU)", w1);
            Row rr = stage("C recombine (SFPU)", w2);

            std::vector<uint32_t> outC;
            Dp::EnqueueReadMeshBuffer(cq, outC, bOut, true);
            uint64_t mmC=0;
            for (uint64_t ct=0; ct<NCTT; ++ct) {
                auto o = untilize<uint32_t>(outC.data() + ct*ELEMS);
                for (uint32_t cp=0; cp<TDIM; ++cp) for (uint32_t j=0;j<LdC;++j)
                    if (o[cp*TDIM+j] != refC[(size_t)(ct*TDIM+cp)*LdC+j]) ++mmC; }
            fmt::print("  verify: {} mismatches / {} outputs\n", mmC, NCTT*TDIM*LdC);
            if (mmC) pass = false;
            rm.mism = mmC; rr.mism = mmC; rm.b = TC; rr.b = TC;
            rows.push_back(rm); rows.push_back(rr);
        }

        pass &= mesh->close();

        if (PROF) {
            auto disp = parse_profiler(PCSV);
            fmt::print("\nprofiler: {} dispatch groups; host issued {}\n", disp.size(), ndisp);
            if (disp.size() == ndisp)
                for (auto& r : rows) { const KStat* best=nullptr;
                    for (size_t i=r.d0+1;i<r.d0+r.nd;++i) if (!best || disp[i].span_us<best->span_us) best=&disp[i];
                    if (best) r.k=*best; }
            else fmt::print("  MISMATCH -- device columns dropped rather than misattributed.\n");
        }

        auto table=[&](const char* title, const std::function<bool(const Row&)>& sel, bool fbc){
            std::vector<const Row*> rs; for (auto& r : rows) if (sel(r)) rs.push_back(&r);
            if (rs.empty()) return;
            fmt::print("\n{}\n", title);
            fmt::print("  {:<22}{:>10}{:>11}{:>9}{:>9}{:>9}{:>9}{:>8}\n",
                       "variant","kern us","us/tile","unpack","math","pack","vs best","mism");
            double best=1e18; for (auto* r : rs) if (r->k.ok && r->k.span_us>0) best=std::min(best,r->k.span_us);
            for (auto* r : rs) {
                const double s=r->k.ok?r->k.span_us:0.0;
                fmt::print("  {:<22}{:>10.2f}{:>11.3f}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}{:>8}\n",
                           r->what, s, s/T, r->k.risc_us[2], r->k.risc_us[3], r->k.risc_us[4],
                           (best<1e17&&s>0)?s/best:0.0, r->mism);
            }
            if (fbc) { fmt::print("\n  cost per Ls*Ld unit (us/tile / (Ls*Ld)) -- flat means the cost IS the MAC count:\n");
                for (auto* r : rs) if (r->k.ok) fmt::print("    Ls={:<3} Ld={:<3} {:>8.4f} us per MAC-column\n",
                                                           r->a, r->b, r->k.span_us/T/(r->a*r->b)); }
        };
        table("PART A — modmul variants", [](const Row& r){ return r.a==0; }, false);
        table("PART B — baseconv_fbc against its (Ls, Ld) shape", [](const Row& r){ return r.a!=0 && r.a<100; }, true);

        // PART C is not a variant shootout, so it gets its own report: the two FPU-route stages,
        // their sum, and the SFPU measurement of the SAME (Ls=16, Ld=8) conversion from part B.
        {
            std::vector<const Row*> cs; for (auto& r : rows) if (r.a==100) cs.push_back(&r);
            if (!cs.empty()) {
                fmt::print("\nPART C — matrix-engine FBC at Ls=16, Ld=8\n");
                fmt::print("  {:<22}{:>10}{:>12}{:>9}{:>9}{:>9}\n","stage","kern us","us/our-tile","unpack","math","pack");
                double tot=0; uint32_t TCu=cs[0]->b?cs[0]->b:1;
                for (auto* r : cs) { const double s=r->k.ok?r->k.span_us:0.0; tot += s/TCu;
                    fmt::print("  {:<22}{:>10.2f}{:>12.3f}{:>9.2f}{:>9.2f}{:>9.2f}\n",
                               r->what, s, s/TCu, r->k.risc_us[2], r->k.risc_us[3], r->k.risc_us[4]); }
                fmt::print("  {:<22}{:>10}{:>12.3f}\n","FPU route total","",tot);
                const Row* sfpu=nullptr; for (auto& r : rows) if (r.a==16 && r.b==8) sfpu=&r;
                if (sfpu && sfpu->k.ok) fmt::print("  {:<22}{:>10.2f}{:>12.3f}   <- part B, same conversion\n",
                                                  "SFPU baseconv_fbc", sfpu->k.span_us, sfpu->k.span_us/T);
                if (sfpu && sfpu->k.ok && tot>0) fmt::print("\n  FPU route is {:.2f}x the SFPU at this shape.\n", tot/(sfpu->k.span_us/T));
                fmt::print("  EXCLUDED from the FPU number: the limb-major -> coefficient-major transpose and the\n"
                           "  radix-128 decomposition of y, both built on the host here. Also excluded: stage 1\n"
                           "  (y_i = x_i·C1_i mod q_i), which exp_Tg018m runs as a separate fbc_scale pass and\n"
                           "  which part B's SFPU number DOES include.\n");
            }
        }
        fmt::print("\nEvery row is ONE dispatch of G={} cores x T={} tiles, verified against a host golden.\n", G, T);
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
    return pass ? 0 : 1;
}
