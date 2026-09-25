// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_Tg018r — SFPU vs MATRIX ENGINE for the RNS base conversion, on equal footing.
// docs/he/exp_tg018r.md
//
// exp_Tg018k part C measured the matrix-engine FBC without its layout transform and only at
// Ls=16, Ld=8, because two things blocked anything wider:
//
//   * the SFPU kernel copies the Ls x Ld constant block into STACK ARRAYS (`uint32_t C2[16][12]`),
//     so Ls=32 could not run at all.  tt-metal allows 4096 runtime args per Tensix kernel
//     (max_runtime_args_tensix); the 256 "budget" was self-imposed.  baseconv_fbc_wide.cpp reads the
//     same args in place, no stack, no cap.
//   * the matrix engine needs the source-limb axis INSIDE a tile, and exp_Tg018 stores it as a tile
//     axis.  The standard answer is an on-device all-to-all transpose pass.  It is not needed:
//     transpose the OPERANDS instead of the data.
//
//         limb-major form:   plane [c][j] = SUM_lane A[c][lane] . B[lane][j]
//         this experiment:   plane'[j][c] = SUM_lane Bt[j][lane] . At[lane][c]
//
//     Both contract over the limb axis, but the second wants At LIMB-major -- which is what a
//     strided read of the tiles we already store produces, because a tilized row is two contiguous
//     16-element runs.  So the transpose becomes 2 NoC copies of 64 bytes per (group, limb) on the
//     way in (reader_fbc_tr.cpp) and the same on the way out (writer_fbc_tr.cpp), fused into the one
//     dispatch, with the radix planes never leaving L1.
//
// WHAT IS COMPARED.  Both routes compute the same function, from the same buffers, verified against
// the same host golden:
//
//     y_i   = (x_i . C1[i]) mod q_i                       [stage 1]
//     out_j = SUM_i (y_i . C2[i][j]) mod p_j              [approximate FBC, no eps correction]
//
//     SFPU:  1 dispatch   reader_seq_batch -> baseconv_fbc_wide -> writer_seq_batch
//     FPU:   2 dispatches reader_seq_batch -> fbc_scale_wide    -> writer_seq_batch
//                         reader_fbc_tr    -> fbc_fpu_tr        -> writer_fbc_tr
//
// The FPU total INCLUDES stage 1 and BOTH transposes.  Nothing is on the host but the constants.
//
// DRAM.  Every buffer is laid out page = our_tile * L + limb, so the L pages a core touches for one
// our-tile are consecutive and therefore land on consecutive banks; readers put the whole batch in
// flight before the barrier.  Both are from docs/he/dram_guide.md and both apply to both routes.
//
//   ./build/exp_tg018r [G=64] [ITERS=3]
//
//   TGR_G        cores (default 64)
//   TGR_N        ring degree (default 65536 -> 64 tiles of 1024 coefficients)
//   TGR_LS, TGR_LD   grid lists (default 8,16,32 each)
//   TGR_ROUTE    sfpu | fpu | both
//   TGR_CSV      write the grid to this path
//
// Device timings are tt-metal's own per-RISC `*-KERNEL` zones, parsed after mesh->close(); no zone
// is added to any kernel.  Same method as exp_Tg018dram -- docs/he/dram_guide.md section 7.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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
constexpr uint32_t ELEMS = 1024, I32 = ELEMS * 4, I8 = ELEMS, TDIM = 32, NR = 4, NP = 4;
constexpr double CLK_MHZ = 1350.0;
using hrc = std::chrono::high_resolution_clock;
using u128 = unsigned __int128;
double msd(hrc::time_point t) { return std::chrono::duration<double, std::milli>(hrc::now() - t).count(); }

bool is_prime(uint64_t n) { if (n < 2 || (n % 2 == 0 && n != 2)) return false;
    for (uint64_t d = 3; d * d <= n; d += 2) if (n % d == 0) return false; return true; }
std::vector<uint32_t> parse_list(const char* env, std::vector<uint32_t> d) {
    const char* e = std::getenv(env); if (!e) return d;
    std::vector<uint32_t> v; std::stringstream ss(e); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back((uint32_t)std::stoul(t));
    return v.empty() ? d : v; }

const uint64_t POW[NR] = {1, 128, 16384, 2097152};
int8_t limb8(uint64_t v, uint32_t p) { return (int8_t)((v / POW[p]) % 128); }
template <typename E> std::vector<E> tilize(const std::vector<E>& in) {
    std::vector<E> o(ELEMS);
    for (uint32_t r=0;r<TDIM;++r) for (uint32_t c=0;c<TDIM;++c)
        o[((r/16)*2 + c/16)*256 + (r%16)*16 + (c%16)] = in[r*TDIM + c];
    return o; }

// ---- device profiler readback: every `*-KERNEL` dispatch group, in issue order
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

struct Cell { uint32_t ls=0, ld=0; double sfpu_us=0, fpu_scale_us=0, fpu_tr_us=0;
              double sfpu_host=0, fpu_host=0; uint64_t sfpu_mism=0, fpu_mism=0; bool ran=false; };
}  // namespace

int main(int argc, char** argv) {
    uint32_t G = (argc>1)?(uint32_t)std::stoul(argv[1]):64;
    uint32_t ITERS = (argc>2)?(uint32_t)std::stoul(argv[2]):3;
    uint32_t N = 65536;
    if (const char* e=std::getenv("TGR_G")) G=(uint32_t)std::atoi(e);
    if (const char* e=std::getenv("TGR_N")) N=(uint32_t)std::atoi(e);
    const std::string ROUTE = std::getenv("TGR_ROUTE") ? std::getenv("TGR_ROUTE") : "both";
    const auto LSL = parse_list("TGR_LS", {8,16,32});
    const auto LDL = parse_list("TGR_LD", {8,16,32});
    const uint32_t NOT = N / ELEMS;                      // our-tiles (1024 coefficients each)

    bool pass = true;
    try {
        auto mesh = Dp::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE);
        auto& cq = mesh->mesh_command_queue();
        auto dr = Dp::MeshCoordinateRange(mesh->shape());
        auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX=grid.x, GMAX=grid.x*grid.y;
        G = std::min(G, std::min(GMAX, NOT));
        const uint32_t T = NOT / G;                      // our-tiles per core
        // baseconv_fbc_wide and fbc_scale_wide each convert ONE our-tile per dispatch (they
        // cb_wait_front(LS) once), so one core owns exactly one our-tile here.
        if (T != 1) { fmt::print(stderr, "G must equal the our-tile count {} (got G={})\n", NOT, G); return 2; }
        auto CC=[&](uint32_t i){ return CoreCoord{i%GX, i/GX}; };
        std::set<CoreRange> ss; for (uint32_t i=0;i<G;++i) ss.insert(CoreRange(CC(i)));
        const CoreRangeSet crs(ss);

        auto mkbuf=[&](uint64_t nt, uint32_t page){ Dp::DeviceLocalBufferConfig d{.page_size=page,.buffer_type=BufferType::DRAM};
            return Dp::MeshBuffer::create(Dp::ReplicatedBufferConfig{.size=nt*page}, d, mesh.get()); };
        auto mkcb=[&](Program& p, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page){
            CircularBufferConfig c(depth*page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, crs, c); };

        const uint32_t LSMAX = *std::max_element(LSL.begin(), LSL.end());
        const uint32_t LDMAX = *std::max_element(LDL.begin(), LDL.end());
        fmt::print("\nexp_Tg018r — SFPU vs matrix engine for the FBC, equal footing\n");
        fmt::print("N={} ({} tiles of {} coefficients), grid {}x{}, G={} cores, T={} tile(s)/core, {} replays\n",
                   N, NOT, ELEMS, grid.x, grid.y, G, T, ITERS);
        fmt::print("Ls in {}, Ld in {}\n", fmt::join(LSL, ","), fmt::join(LDL, ","));

        // buffers, all page = our_tile * L + limb so a core's pages are consecutive (dram_guide)
        auto bX = mkbuf((uint64_t)NOT*LSMAX, I32);
        auto bY = mkbuf((uint64_t)NOT*LSMAX, I32);
        auto bOs = mkbuf((uint64_t)NOT*LDMAX, I32);
        auto bOf = mkbuf((uint64_t)NOT*LDMAX, I32);
        auto bBt = mkbuf(NR*NP, I8);
        auto bPar = mkbuf(5, I32);

        // 64 distinct 28-bit primes: LSMAX sources then LDMAX destinations
        std::vector<uint64_t> Qm(LSMAX), Pm(LDMAX);
        { std::vector<uint64_t> pr; for (uint64_t c=(1ull<<28)-1; pr.size()<LSMAX+LDMAX; c-=2) if (is_prime(c)) pr.push_back(c);
          for (uint32_t i=0;i<LSMAX;++i) Qm[i]=pr[i];
          for (uint32_t j=0;j<LDMAX;++j) Pm[j]=pr[LSMAX+j]; }
        std::vector<mont::MontgomeryContext> CQ, CP;
        for (uint32_t i=0;i<LSMAX;++i) CQ.emplace_back((uint32_t)Qm[i]);
        for (uint32_t j=0;j<LDMAX;++j) CP.emplace_back((uint32_t)Pm[j]);

        // x_i, and the CRT constants. c1_i / v_ij are drawn in range: their VALUES cannot change a
        // timing or an exactness check, only their magnitude can, and that is fixed by the modulus.
        std::mt19937_64 rng(918273645);
        std::vector<uint32_t> hX((size_t)NOT*LSMAX*ELEMS);
        std::vector<uint32_t> c1(LSMAX), v(LSMAX*LDMAX);
        for (uint32_t i=0;i<LSMAX;++i) c1[i] = (uint32_t)(rng()%Qm[i]);
        for (uint32_t i=0;i<LSMAX;++i) for (uint32_t j=0;j<LDMAX;++j) v[i*LDMAX+j] = (uint32_t)(rng()%Pm[j]);
        for (uint32_t ot=0; ot<NOT; ++ot) for (uint32_t i=0;i<LSMAX;++i) {
            const size_t base = ((size_t)ot*LSMAX + i)*ELEMS;
            for (uint32_t e=0;e<ELEMS;++e) hX[base+e] = (uint32_t)(rng()%Qm[i]); }
        Dp::EnqueueWriteMeshBuffer(cq, bX, hX, false); Dp::Finish(cq);

        const char* home = std::getenv("TT_METAL_HOME");
        const std::string PCSV = home ? std::string(home)+"/generated/profiler/.logs/profile_log_device.csv" : "";
        const bool PROF = std::getenv("TT_METAL_DEVICE_PROFILER") && !PCSV.empty();
        if (!PROF) fmt::print("\nNOTE: profiler off -- device columns blank. re-run as TT_METAL_DEVICE_PROFILER=1 ...\n");

        size_t ndisp = 0;
        struct Slot { size_t d0, nd; double* out; };
        std::vector<Slot> slots;
        auto timed=[&](Dp::MeshWorkload& wl, double* dev, double* host){
            const size_t d0 = ndisp;
            Dp::EnqueueMeshWorkload(cq, wl, false); Dp::Finish(cq); ++ndisp;
            double mn=1e18;
            for (uint32_t i=0;i<ITERS;++i){ auto t0=hrc::now(); Dp::EnqueueMeshWorkload(cq,wl,false); Dp::Finish(cq); mn=std::min(mn,msd(t0)); }
            ndisp += ITERS;
            if (host) *host = mn;
            slots.push_back({d0, ndisp-d0, dev});
            if (PROF) { Dp::Finish(cq); ReadMeshDeviceProfilerResults(*mesh); } };

        std::vector<Cell> cells;
        // reserved and pushed BEFORE the run, because `slots` holds pointers into these cells and a
        // reallocation (or a copy of a loop-local) would leave the profiler writing into dead storage.
        cells.reserve(LSL.size() * LDL.size());
        std::vector<uint32_t> got;

        for (uint32_t Ls : LSL) for (uint32_t Ld : LDL) {
            cells.push_back(Cell{}); Cell& cell = cells.back();
            cell.ls=Ls; cell.ld=Ld; cell.ran=true;
            fmt::print("\n--- Ls={} Ld={} ---\n", Ls, Ld);

            // ---------- host golden ----------
            // y_i = x_i*c1_i mod q_i ; out_j = sum_i y_i*v_ij mod p_j
            std::vector<uint32_t> gold((size_t)NOT*Ld*ELEMS);
            {
                std::vector<uint64_t> y(Ls);
                for (uint32_t ot=0; ot<NOT; ++ot) for (uint32_t e=0;e<ELEMS;++e) {
                    for (uint32_t i=0;i<Ls;++i)
                        y[i] = ((u128)hX[((size_t)ot*LSMAX + i)*ELEMS + e] * c1[i]) % Qm[i];
                    for (uint32_t j=0;j<Ld;++j) {
                        u128 acc=0; for (uint32_t i=0;i<Ls;++i) acc += (u128)y[i]*v[i*LDMAX+j];
                        gold[((size_t)ot*Ld + j)*ELEMS + e] = (uint32_t)(acc % Pm[j]); }
                }
            }
            auto verify=[&](std::shared_ptr<Dp::MeshBuffer>& b)->uint64_t {
                Dp::EnqueueReadMeshBuffer(cq, got, b, true);
                uint64_t m=0;
                for (uint32_t ot=0; ot<NOT; ++ot) for (uint32_t j=0;j<Ld;++j) {
                    const size_t db=((size_t)ot*LDMAX + j)*ELEMS, gb=((size_t)ot*Ld + j)*ELEMS;
                    for (uint32_t e=0;e<ELEMS;++e) if (got[db+e]!=gold[gb+e]) ++m; }
                return m; };

            // ---------- constants, in the two forms the two routes want ----------
            // SFPU: scalars, Montgomery-encoded so mulmod(y, mont(c)) = y*c mod m
            std::vector<uint32_t> sa{Ls, Ld};
            for (uint32_t i=0;i<Ls;++i) sa.push_back(CQ[i].to_montgomery(c1[i]));
            for (uint32_t i=0;i<Ls;++i) for (uint32_t j=0;j<Ld;++j) sa.push_back(CP[j].to_montgomery(v[i*LDMAX+j]));
            for (uint32_t i=0;i<Ls;++i) sa.push_back((uint32_t)Qm[i]);
            for (uint32_t i=0;i<Ls;++i) sa.push_back(CQ[i].q_inv_32());
            for (uint32_t j=0;j<Ld;++j) sa.push_back((uint32_t)Pm[j]);
            for (uint32_t j=0;j<Ld;++j) sa.push_back(CP[j].q_inv_32());
            std::vector<uint32_t> sca{Ls, Ld};
            for (uint32_t i=0;i<Ls;++i) sca.push_back(CQ[i].to_montgomery(c1[i]));
            for (uint32_t i=0;i<Ls;++i) sca.push_back((uint32_t)Qm[i]);
            for (uint32_t i=0;i<Ls;++i) sca.push_back(CQ[i].q_inv_32());

            // FPU: Bt[rr*NP+p][j][i] = limb_rr( (128^p * v_ij) mod p_j ), zero outside (Ls, Ld).
            // NO Montgomery fold here -- recombine_rns's reduction already returns the plain
            // sum_rr 128^rr * plane_rr mod p_j, so the output matches the SFPU's bit for bit.
            std::vector<int8_t> Bt((size_t)NR*NP*I8, 0);
            for (uint32_t rr=0; rr<NR; ++rr) for (uint32_t p=0;p<NP;++p) {
                std::vector<int8_t> row(ELEMS, 0);
                for (uint32_t j=0;j<Ld;++j) for (uint32_t i=0;i<Ls;++i) {
                    const uint64_t Bp = (uint64_t)(((u128)POW[p]*v[i*LDMAX+j]) % Pm[j]);
                    row[j*TDIM + i] = limb8(Bp, rr); }
                auto til = tilize(row); std::copy(til.begin(), til.end(), Bt.begin()+(size_t)(rr*NP+p)*I8); }
            // recombine params, per ROW j (j is the output's row index in the transposed form)
            std::vector<uint32_t> par(5*ELEMS);
            for (uint32_t r=0;r<TDIM;++r) for (uint32_t c=0;c<TDIM;++c) {
                const uint32_t j=(r<Ld)?r:0, at=r*TDIM+c;
                par[0*ELEMS+at]=CP[j].to_montgomery(16384u);
                par[1*ELEMS+at]=CP[j].q_inv_32();
                par[2*ELEMS+at]=(uint32_t)(Pm[j]>>16);
                par[3*ELEMS+at]=(uint32_t)(Pm[j]&0xFFFFu);
                par[4*ELEMS+at]=(uint32_t)Pm[j]; }
            for (uint32_t t=0;t<5;++t) { std::vector<uint32_t> rm(par.begin()+t*ELEMS, par.begin()+(t+1)*ELEMS);
                auto til=tilize(rm); std::copy(til.begin(), til.end(), par.begin()+t*ELEMS); }
            Dp::EnqueueWriteMeshBuffer(cq, bBt, Bt, false);
            Dp::EnqueueWriteMeshBuffer(cq, bPar, par, false);
            Dp::Finish(cq);

            // ---------- route 1: SFPU, one dispatch ----------
            if (ROUTE=="sfpu" || ROUTE=="both") {
                Program p = CreateProgram();
                mkcb(p, CBIndex::c_0, tt::DataFormat::UInt32, Ls, I32);
                mkcb(p, CBIndex::c_2, tt::DataFormat::UInt32, Ls, I32);
                mkcb(p, CBIndex::c_3, tt::DataFormat::UInt32, Ls, I32);
                mkcb(p, CBIndex::c_16, tt::DataFormat::UInt32, Ld, I32);
                auto rd=CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_seq_batch.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_seq_batch.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/baseconv_fbc_wide.cpp", crs,
                    ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t k=0;k<G;++k){ auto c=CC(k); const uint32_t ot=k*T;
                    SetRuntimeArgs(p,rd,c,{(uint32_t)bX->address(), Ls*T, ot*LSMAX});
                    SetRuntimeArgs(p,wr,c,{(uint32_t)bOs->address(), Ld*T, ot*LDMAX});
                    SetRuntimeArgs(p,cp,c,sa); }
                Dp::MeshWorkload wl; wl.add_program(dr, std::move(p));
                timed(wl, &cell.sfpu_us, &cell.sfpu_host);
                cell.sfpu_mism = verify(bOs);
                fmt::print("  SFPU   {:>8.3f} ms host   mism {}\n", cell.sfpu_host, cell.sfpu_mism);
                if (cell.sfpu_mism) pass = false;
            }

            // ---------- route 2: matrix engine, scale + fused transpose/matmul/recombine ----------
            if (ROUTE=="fpu" || ROUTE=="both") {
                Program p1 = CreateProgram();
                mkcb(p1, CBIndex::c_0, tt::DataFormat::UInt32, Ls, I32);
                mkcb(p1, CBIndex::c_16, tt::DataFormat::UInt32, Ls, I32);
                auto rd1=CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "dataflow/reader_seq_batch.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr1=CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "dataflow/writer_seq_batch.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp1=CreateKernel(p1, OVERRIDE_KERNEL_PREFIX "compute/fbc_scale_wide.cpp", crs,
                    ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t k=0;k<G;++k){ auto c=CC(k); const uint32_t ot=k*T;
                    SetRuntimeArgs(p1,rd1,c,{(uint32_t)bX->address(), Ls*T, ot*LSMAX});
                    SetRuntimeArgs(p1,wr1,c,{(uint32_t)bY->address(), Ls*T, ot*LSMAX});
                    SetRuntimeArgs(p1,cp1,c,sca); }
                Dp::MeshWorkload w1; w1.add_program(dr, std::move(p1));

                Program p2 = CreateProgram();
                mkcb(p2, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32);
                mkcb(p2, CBIndex::c_1, tt::DataFormat::Int8, NR*NP, I8);
                mkcb(p2, CBIndex::c_2, tt::DataFormat::Int8, NP, I8);
                mkcb(p2, CBIndex::c_3, tt::DataFormat::UInt32, NR, I32);
                mkcb(p2, CBIndex::c_4, tt::DataFormat::UInt32, 5, I32);
                mkcb(p2, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
                std::vector<uint32_t> rc; TensorAccessorArgs(*bBt).append_to(rc);
                auto rd2=CreateKernel(p2, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fbc_tr.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
                auto wr2=CreateKernel(p2, OVERRIDE_KERNEL_PREFIX "dataflow/writer_fbc_tr.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp2=CreateKernel(p2, OVERRIDE_KERNEL_PREFIX "compute/fbc_fpu_tr.cpp", crs,
                    ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,
                                  .dst_full_sync_en=true,.math_approx_mode=false});
                const uint32_t NGRP = T*(ELEMS/TDIM);
                for (uint32_t k=0;k<G;++k){ auto c=CC(k); const uint32_t ot=k*T;
                    SetRuntimeArgs(p2,rd2,c,{(uint32_t)bY->address(),(uint32_t)bBt->address(),
                                             (uint32_t)bPar->address(), Ls, NGRP, ot, LSMAX});
                    SetRuntimeArgs(p2,wr2,c,{(uint32_t)bOf->address(), Ld, NGRP, ot, LDMAX});
                    SetRuntimeArgs(p2,cp2,c,{NGRP}); }
                Dp::MeshWorkload w2; w2.add_program(dr, std::move(p2));

                double h1=0, h2=0;
                timed(w1, &cell.fpu_scale_us, &h1);
                timed(w2, &cell.fpu_tr_us, &h2);
                cell.fpu_host = h1 + h2;
                cell.fpu_mism = verify(bOf);
                fmt::print("  FPU    {:>8.3f} ms host   mism {}\n", cell.fpu_host, cell.fpu_mism);
                if (cell.fpu_mism) pass = false;
            }
        }

        pass &= mesh->close();

        if (PROF) {
            auto disp = parse_profiler(PCSV);
            fmt::print("\nprofiler: {} dispatch groups; host issued {}\n", disp.size(), ndisp);
            if (disp.size() == ndisp) {
                for (auto& s : slots) { const KStat* best=nullptr;
                    for (size_t i=s.d0+1;i<s.d0+s.nd;++i) if (!best || disp[i].span_us<best->span_us) best=&disp[i];
                    if (best) *s.out = best->span_us; }
            } else fmt::print("  MISMATCH -- device columns dropped rather than misattributed.\n");
        }

        // ---------------- the grid ----------------
        auto tot=[&](const Cell& c){ return c.fpu_scale_us + c.fpu_tr_us; };
        fmt::print("\n=== per conversion of N={} coefficients, {} cores, device us ===\n", N, G);
        fmt::print("  {:<6}{:<6}{:>10}{:>10}{:>10}{:>10}{:>9}{:>9}{:>9}\n",
                   "Ls","Ld","SFPU","FPU scale","FPU main","FPU tot","FPU/SFPU","mismS","mismF");
        for (auto& c : cells)
            fmt::print("  {:<6}{:<6}{:>10.1f}{:>10.1f}{:>10.1f}{:>10.1f}{:>9.2f}{:>9}{:>9}\n",
                       c.ls, c.ld, c.sfpu_us, c.fpu_scale_us, c.fpu_tr_us, tot(c),
                       c.sfpu_us>0?tot(c)/c.sfpu_us:0.0, c.sfpu_mism, c.fpu_mism);

        fmt::print("\n=== the same, as a grid of FPU/SFPU (< 1.00 means the matrix engine wins) ===\n");
        fmt::print("  {:<8}", "Ls\\Ld"); for (uint32_t ld : LDL) fmt::print("{:>10}", ld); fmt::print("\n");
        for (uint32_t ls : LSL) { fmt::print("  {:<8}", ls);
            for (uint32_t ld : LDL) { double r=0; for (auto& c : cells) if (c.ls==ls && c.ld==ld && c.sfpu_us>0) r=tot(c)/c.sfpu_us;
                fmt::print("{:>10.2f}", r); } fmt::print("\n"); }

        if (const char* f = std::getenv("TGR_CSV")) {
            std::ofstream o(f);
            o << "# exp_Tg018r: SFPU vs matrix-engine FBC, equal footing. N=" << N << ", G=" << G
              << ", T=" << T << " our-tile(s)/core, " << ITERS << " replays (min).\n"
              << "# Device us = per-RISC *-KERNEL zone envelope for ONE conversion of all N coefficients.\n"
              << "# FPU total = scale + fused(transpose-in, decompose, matmul, recombine, transpose-out).\n"
              << "# Both routes verified against the same host golden; mism = differing outputs.\n"
              << "ls,ld,sfpu_us,fpu_scale_us,fpu_main_us,fpu_total_us,ratio,sfpu_mism,fpu_mism,sfpu_host_ms,fpu_host_ms\n";
            for (auto& c : cells)
                o << c.ls << "," << c.ld << "," << c.sfpu_us << "," << c.fpu_scale_us << "," << c.fpu_tr_us
                  << "," << tot(c) << "," << (c.sfpu_us>0?tot(c)/c.sfpu_us:0.0) << "," << c.sfpu_mism << ","
                  << c.fpu_mism << "," << c.sfpu_host << "," << c.fpu_host << "\n";
            fmt::print("\nwrote {}\n", f);
        }
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
    return pass ? 0 : 1;
}
