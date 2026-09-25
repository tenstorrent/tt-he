// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_modmul002 — BARRETT vs MONTGOMERY, both written to the same rules.
// docs/he/exp_modmul002.md
//
// exp_modmul001 established the fastest Montgomery modular multiply we have: one SFPI pass, every
// product an sfpmul24, no fill_tile_int, 3.086 us/tile.  The obvious question it does not answer is
// whether MONTGOMERY is the right reduction at all -- Barrett needs no Montgomery domain and is what
// a from-scratch implementation would reach for.
//
// So this compares the two ALGORITHMS, not two levels of effort.  The Barrett kernel is built to the
// identical rules (kernels/OPTIMISED.md): one SFPI pass, sfpmul24 for every product, scalar
// constants broadcast in register.  Same CBs, same runtime-arg shape, same reader and writer, no
// profiler zone in either.  Per-tile cost is FITTED from span(T) rather than divided, so the fixed
// dispatch cost is removed instead of being folded in.
//
// THE ONE THING THAT IS NOT IDENTICAL, and it matters:
//
//   Montgomery computes  a*b*R^-1 mod q   on Montgomery-domain residues
//   Barrett    computes  a*b      mod q   on plain residues
//
// These are different functions, so each is checked against its OWN host golden.  For a per-multiply
// LATENCY comparison that is fair -- both take two 28-bit residues and return one, and a pipeline
// that stays in one domain never pays a conversion.  It is NOT fair as a claim about which is better
// end to end: Montgomery's cost is amortised only if the whole pipeline stays in the domain (ours
// does), while Barrett is free to mix with plain-domain data.  That trade is a pipeline question,
// not a kernel question, and this bench deliberately does not answer it.
//
//   ./build/exp_modmul002 [ITERS=5]
//
//   MM2_T      tile counts to sweep (default 1,2,4,8,16,32,64,128)
//   MM2_CORE   which core, as "x,y" (default 0,0)
//   MM2_CSV    write the table here
//   MM2_STAT   run the PAIRED statistics protocol instead: N repetitions at a single tile count,
//              the two kernels INTERLEAVED within each repetition so drift cancels.  Reports
//              mean/sd/min/max per kernel and, more usefully, the paired per-repetition difference.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <map>
#include <tuple>

#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
namespace Dp = tt::tt_metal::distributed;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t ELEMS = 1024, I32 = ELEMS * 4;
using hrc = std::chrono::high_resolution_clock;
using u128 = unsigned __int128;
double msd(hrc::time_point t) { return std::chrono::duration<double, std::milli>(hrc::now() - t).count(); }
bool is_prime(uint64_t n) { if (n < 2 || (n % 2 == 0 && n != 2)) return false;
    for (uint64_t d = 3; d * d <= n; d += 2) if (n % d == 0) return false; return true; }
std::vector<uint32_t> parse_list(const char* e, std::vector<uint32_t> d) {
    const char* v = std::getenv(e); if (!v) return d;
    std::vector<uint32_t> o; std::stringstream ss(v); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) o.push_back((uint32_t)std::stoul(t));
    return o.empty() ? d : o; }

// ---- device profiler readback: per-dispatch, per-RISC `*-KERNEL` spans (same method as
// docs/he/exp_modmul001.md and docs/he/dram_guide.md section 7).  TRISC_1 is MATH: its span is the
// compute kernel's own execution, excluding the host enqueue, the command queue, and
// wait_for_go_message(), which the firmware zone does not cover.
constexpr double CLK_MHZ = 1350.0;
const char* RISC_NAME[5] = {"BRISC","NCRISC","TRISC_0","TRISC_1","TRISC_2"};
struct KStat { double span_us = 0, risc_us[5] = {0,0,0,0,0}; };
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
    struct Iv { uint64_t s,e; uint32_t risc; };
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
        else if (x[cp]=="ZONE_END" && !open[key].empty()) { iv.push_back({open[key].back(),t,(uint32_t)ri}); open[key].pop_back(); }
    }
    if (iv.empty()) return out;
    std::sort(iv.begin(),iv.end(),[](const Iv&a,const Iv&b){return a.s<b.s;});
    auto emit=[&](size_t lo,size_t hi){ KStat k; uint64_t s=iv[lo].s,e=iv[lo].e;
        for (size_t i=lo;i<hi;++i){ s=std::min(s,iv[i].s); e=std::max(e,iv[i].e);
            k.risc_us[iv[i].risc]=std::max(k.risc_us[iv[i].risc],(double)(iv[i].e-iv[i].s)/CLK_MHZ); }
        k.span_us=(double)(e-s)/CLK_MHZ; out.push_back(k); };
    size_t g0=0; uint64_t ge=iv[0].e;
    for (size_t i=1;i<iv.size();++i){ if (iv[i].s<=ge) ge=std::max(ge,iv[i].e); else { emit(g0,i); g0=i; ge=iv[i].e; } }
    emit(g0,iv.size()); return out; }

struct Pt { uint32_t T; double ms = 0; uint64_t mism = 0;
            size_t d0 = 0, nd = 0; double math_us = 0, span_us = 0; };
struct Var { const char* name; const char* path; const char* fn; bool barrett; std::vector<Pt> pts; };
}  // namespace

int main(int argc, char** argv) {
    const uint32_t ITERS = (argc > 1) ? (uint32_t)std::stoul(argv[1]) : 5;
    const auto TL = parse_list("MM2_T", {1,2,4,8,16,32,64,128});
    uint32_t CX = 0, CY = 0;
    if (const char* e = std::getenv("MM2_CORE")) std::sscanf(e, "%u,%u", &CX, &CY);
    const uint32_t TMAX = *std::max_element(TL.begin(), TL.end());

    uint64_t q = (1ull << 28) - 1; while (!(q & 1) || !is_prime(q)) --q;
    const mont::MontgomeryContext C((uint32_t)q);
    const uint32_t MU = (uint32_t)(((u128)1 << 56) / q);          // floor(2^56 / q), < 2^29
    // barrett_1sub is exact with ONE conditional subtract iff 2^27 + s < q, where s = 2^56 - mu*q.
    // Asserted, not sampled: a sweep can disprove a bound but never establish one.
    const uint64_t S1 = (uint64_t)(((u128)1 << 56) - (u128)MU * q);
    const bool ONE_SUB_OK = ((uint64_t)1 << 27) + S1 < q;

    bool pass = true;
    try {
        auto mesh = Dp::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE);
        auto& cq = mesh->mesh_command_queue();
        auto dr = Dp::MeshCoordinateRange(mesh->shape());
        const CoreCoord core{CX, CY};
        const CoreRangeSet crs(std::set<CoreRange>{CoreRange(core)});

        fmt::print("\nexp_modmul002 — Barrett vs Montgomery, both one SFPI pass, both sfpmul24\n");
        fmt::print("one core ({},{}), q = {} (28-bit), mu = floor(2^56/q) = {}\n", CX, CY, q, MU);
        fmt::print("{} timed replays, min taken; per-tile cost FITTED from span(T), not divided\n", ITERS);
        fmt::print("one-subtract precondition 2^27 + s < q:  s = {}, 2^27+s = {}, q = {}  -> {}\n\n",
                   S1, ((uint64_t)1<<27)+S1, q, ONE_SUB_OK ? "HOLDS" : "FAILS (barrett_1sub invalid)");

        auto mkbuf = [&](uint64_t n) {
            Dp::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return Dp::MeshBuffer::create(Dp::ReplicatedBufferConfig{.size = n * I32}, d, mesh.get()); };
        auto bA = mkbuf(TMAX), bB = mkbuf(TMAX), bO = mkbuf(TMAX);

        // one input set; two goldens, because the two kernels compute different functions
        std::vector<uint32_t> hA(TMAX*ELEMS), hB(TMAX*ELEMS),
                              gMont(TMAX*ELEMS), gBar(TMAX*ELEMS), got;
        { std::mt19937 rng(20260817);
          for (auto& v : hA) v = (uint32_t)(rng() % q);
          for (auto& v : hB) v = (uint32_t)(rng() % q);
          for (size_t i = 0; i < hA.size(); ++i) {
              gMont[i] = C.mulmod(hA[i], hB[i]);                       // a*b*R^-1 mod q
              gBar[i]  = (uint32_t)(((u128)hA[i] * hB[i]) % q); } }    // a*b       mod q
        Dp::EnqueueWriteMeshBuffer(cq, bA, hA, false);
        Dp::EnqueueWriteMeshBuffer(cq, bB, hB, false);
        Dp::Finish(cq);

        std::vector<Var> V = {
            {"montgomery_1pass", "compute/modmul_montgomery_1pass.cpp", "a*b*R^-1 mod q", false, {}},
            // modmul_barrett_2sub.cpp is NOT in this list: it is the general two-subtract form, is
            // ~7% slower, and is dominated by modmul_barrett.cpp wherever the one-subtract
            // precondition holds. It is kept in the tree as the fallback for moduli where it does not.
            {"barrett",          "compute/modmul_barrett.cpp",          "a*b      mod q", true,  {}},
        };

        // ---------------- paired statistics mode ----------------
        // One tile count, N repetitions, the kernels INTERLEAVED. Interleaving matters: an earlier
        // version of this bench ran each kernel's whole sweep in turn and the RESULT CHANGED SIGN
        // when a third variant was removed, purely because that moved one kernel's position in the
        // dispatch sequence. Pairing each repetition cancels drift that position exposes.
        if (const char* e = std::getenv("MM2_STAT")) {
            const uint32_t REPS = std::max(2, std::atoi(e));
            const uint32_t T = TL[0];
            fmt::print("PAIRED protocol: T={} tiles in the kernel loop, {} repetitions, "
                       "kernels interleaved,\n  each repetition = min over {} inner replays.\n\n",
                       T, REPS, ITERS);

            std::vector<Dp::MeshWorkload> wls;
            for (auto& v : V) {
                Program p = CreateProgram();
                auto mkcb = [&](CBIndex i, uint32_t depth) {
                    CircularBufferConfig c(depth*I32, {{i, tt::DataFormat::UInt32}});
                    c.set_page_size(i, I32); CreateCircularBuffer(p, crs, c); };
                mkcb(CBIndex::c_0, 2); mkcb(CBIndex::c_1, 2); mkcb(CBIndex::c_16, 2);
                auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + std::string(v.path), crs,
                    ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                SetRuntimeArgs(p, rd, core, {(uint32_t)bA->address(), (uint32_t)bB->address(), T});
                SetRuntimeArgs(p, wr, core, {(uint32_t)bO->address(), T});
                if (v.barrett) SetRuntimeArgs(p, cp, core, {T, (uint32_t)q, MU});
                else           SetRuntimeArgs(p, cp, core, {T, (uint32_t)q, C.q_inv_32(),
                                                            (uint32_t)(q & 0xFFFF), (uint32_t)(q >> 16)});
                Dp::MeshWorkload w; w.add_program(dr, std::move(p)); wls.push_back(std::move(w));
            }
            // verify once per kernel, then time
            for (size_t k = 0; k < V.size(); ++k) {
                Dp::EnqueueMeshWorkload(cq, wls[k], false); Dp::Finish(cq);
                Dp::EnqueueReadMeshBuffer(cq, got, bO, true);
                const auto& gold = V[k].barrett ? gBar : gMont;
                uint64_t m = 0;
                for (uint64_t x = 0; x < (uint64_t)T*ELEMS; ++x) if (got[x] != gold[x]) ++m;
                if (m) pass = false;
                fmt::print("  {:<20} verify: {} mismatches / {}\n", V[k].name, m, (uint64_t)T*ELEMS);
            }

            std::vector<std::vector<double>> obs(V.size());
            for (uint32_t r = 0; r < REPS; ++r)
                for (size_t k = 0; k < V.size(); ++k) {
                    double mn = 1e18;
                    for (uint32_t i = 0; i < ITERS; ++i) { auto t0 = hrc::now();
                        Dp::EnqueueMeshWorkload(cq, wls[k], false); Dp::Finish(cq);
                        mn = std::min(mn, msd(t0)); }
                    obs[k].push_back(mn * 1000.0 / T);          // us per tile, fixed cost INCLUDED
                }

            auto stats = [](const std::vector<double>& x) {
                double s=0; for (double v : x) s+=v; const double mu=s/x.size();
                double q2=0; for (double v : x) q2+=(v-mu)*(v-mu);
                const double sd = std::sqrt(q2/(x.size()-1));
                return std::tuple<double,double,double,double>{mu, sd,
                    *std::min_element(x.begin(),x.end()), *std::max_element(x.begin(),x.end())}; };

            fmt::print("\n  {:<20}{:>10}{:>9}{:>10}{:>10}{:>9}\n",
                       "kernel","mean us","sd","min","max","sd %");
            for (size_t k = 0; k < V.size(); ++k) {
                auto [mu, sd, lo, hi] = stats(obs[k]);
                fmt::print("  {:<20}{:>10.4f}{:>9.4f}{:>10.4f}{:>10.4f}{:>8.2f}%\n",
                           V[k].name, mu, sd, lo, hi, 100*sd/mu); }

            if (V.size() == 2) {
                std::vector<double> d(REPS), rr(REPS);
                for (uint32_t r = 0; r < REPS; ++r) {
                    d[r]  = obs[1][r] - obs[0][r];
                    rr[r] = obs[1][r] / obs[0][r]; }
                auto [dm, dsd, dlo, dhi] = stats(d);
                auto [rm, rsd, rlo, rhi] = stats(rr);
                const double se = dsd / std::sqrt((double)REPS);
                fmt::print("\n  PAIRED difference ({} - {}), one value per repetition:\n",
                           V[1].name, V[0].name);
                fmt::print("    mean {:+.4f} us   sd {:.4f}   sem {:.4f}   range [{:+.4f}, {:+.4f}]\n",
                           dm, dsd, se, dlo, dhi);
                fmt::print("    mean ratio {:.4f}   sd {:.4f}   range [{:.4f}, {:.4f}]\n",
                           rm, rsd, rlo, rhi);
                fmt::print("    |mean| / sem = {:.2f}   ->  {}\n", std::abs(dm)/se,
                           std::abs(dm) > 2*se
                             ? "difference is resolved at this sample size"
                             : "NOT resolved -- the difference is inside the noise");
                const int nb = (int)std::count_if(d.begin(), d.end(), [](double x){return x>0;});
                fmt::print("    sign test: {} of {} repetitions have {} slower\n",
                           nb, REPS, V[1].name);
            }
            pass &= mesh->close();
            if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
            return pass ? 0 : 1;
        }

        size_t ndisp = 0;
        for (auto& v : V) {
            fmt::print("--- {} ({}) ---\n", v.name, v.fn);
            for (uint32_t T : TL) {
                Program p = CreateProgram();
                auto mkcb = [&](CBIndex i, uint32_t depth) {
                    CircularBufferConfig c(depth*I32, {{i, tt::DataFormat::UInt32}});
                    c.set_page_size(i, I32); CreateCircularBuffer(p, crs, c); };
                mkcb(CBIndex::c_0, 2); mkcb(CBIndex::c_1, 2); mkcb(CBIndex::c_16, 2);
                auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + std::string(v.path), crs,
                    ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                SetRuntimeArgs(p, rd, core, {(uint32_t)bA->address(), (uint32_t)bB->address(), T});
                SetRuntimeArgs(p, wr, core, {(uint32_t)bO->address(), T});
                if (v.barrett) SetRuntimeArgs(p, cp, core, {T, (uint32_t)q, MU});
                else           SetRuntimeArgs(p, cp, core, {T, (uint32_t)q, C.q_inv_32(),
                                                            (uint32_t)(q & 0xFFFF), (uint32_t)(q >> 16)});
                Dp::MeshWorkload wl; wl.add_program(dr, std::move(p));

                Pt pt; pt.T = T; pt.d0 = ndisp;
                Dp::EnqueueMeshWorkload(cq, wl, false); Dp::Finish(cq); ++ndisp;
                Dp::EnqueueReadMeshBuffer(cq, got, bO, true);
                const auto& gold = v.barrett ? gBar : gMont;
                for (uint64_t e = 0; e < (uint64_t)T*ELEMS; ++e) if (got[e] != gold[e]) ++pt.mism;
                if (pt.mism) pass = false;

                double mn = 1e18;
                for (uint32_t i = 0; i < ITERS; ++i) { auto t0 = hrc::now();
                    Dp::EnqueueMeshWorkload(cq, wl, false); Dp::Finish(cq); mn = std::min(mn, msd(t0)); }
                ndisp += ITERS; pt.nd = ndisp - pt.d0; pt.ms = mn; v.pts.push_back(pt);
                fmt::print("  T={:<4} {:>8.4f} ms   mism {}\n", T, pt.ms, pt.mism);
            }
        }

        pass &= mesh->close();

        // ---- pure computation latency, from the device trace ----
        {
            const char* home = std::getenv("TT_METAL_HOME");
            const std::string PCSV = home ? std::string(home)+"/generated/profiler/.logs/profile_log_device.csv" : "";
            auto disp = std::getenv("TT_METAL_DEVICE_PROFILER") ? parse_profiler(PCSV) : std::vector<KStat>{};
            if (disp.size() != ndisp) {
                fmt::print("\nTRACE: {} dispatch groups vs {} issued -- device columns dropped rather "
                           "than misattributed.\n", disp.size(), ndisp);
            } else {
                for (auto& v : V) for (auto& p : v.pts) {
                    const KStat* best = nullptr;
                    for (size_t i = p.d0+1; i < p.d0+p.nd; ++i)
                        if (!best || disp[i].risc_us[3] < best->risc_us[3]) best = &disp[i];
                    if (best) { p.math_us = best->risc_us[3]; p.span_us = best->span_us; } }

                fmt::print("\n=== PURE COMPUTATION LATENCY, from the device *-KERNEL trace ===\n");
                fmt::print("MATH (TRISC_1) span per dispatch; excludes host enqueue, the command queue\n"
                           "and wait_for_go_message(), which sit outside the firmware zone.\n\n");
                fmt::print("  {:<20}{:>10}", "kernel", "T=");
                for (auto& p : V[0].pts) fmt::print("{:>9}", p.T); fmt::print("\n");
                for (auto& v : V) { fmt::print("  {:<20}{:>10}", v.name, "");
                    for (auto& p : v.pts) fmt::print("{:>9.2f}", p.math_us); fmt::print("\n"); }

                fmt::print("\n  {:<20}{:>13}{:>11}{:>12}{:>10}\n",
                           "kernel", "per_tile us", "fixed us", "max resid", "vs mont");
                double base = 0;
                for (auto& v : V) {
                    double sx=0,sy=0,sxx=0,sxy=0; const double n=(double)v.pts.size();
                    for (auto& p : v.pts) { sx+=p.T; sy+=p.math_us; sxx+=(double)p.T*p.T; sxy+=(double)p.T*p.math_us; }
                    const double den=n*sxx-sx*sx;
                    const double b=den?(n*sxy-sx*sy)/den:0.0, a=den?(sy-b*sx)/n:0.0;
                    double resid=0; for (auto& p : v.pts) resid=std::max(resid,std::abs(p.math_us-(a+b*p.T)));
                    if (!base) base=b;
                    fmt::print("  {:<20}{:>13.4f}{:>11.3f}{:>12.4f}{:>10.4f}\n", v.name, b, a, resid, b/base); }
                fmt::print("\n  This is the number to quote: the fit removes the ~1 us of in-kernel\n"
                           "  setup, leaving the marginal cost of one more tile of arithmetic.\n");
            }
        }

        fmt::print("\n=== fit  ms(T) = fixed + T x per_tile   (NOT ms/T) ===\n");
        fmt::print("  {:<20}{:>12}{:>11}{:>11}{:>10}{:>8}   {}\n",
                   "kernel", "per_tile us", "fixed ms", "max resid", "vs mont", "mism", "computes");
        double base = 0;
        for (auto& v : V) {
            double sx=0, sy=0, sxx=0, sxy=0; const double n = (double)v.pts.size();
            for (auto& p : v.pts) { sx+=p.T; sy+=p.ms; sxx+=(double)p.T*p.T; sxy+=(double)p.T*p.ms; }
            const double den = n*sxx - sx*sx;
            const double b = den ? (n*sxy - sx*sy)/den : 0.0, a = den ? (sy - b*sx)/n : 0.0;
            double resid = 0; for (auto& p : v.pts) resid = std::max(resid, std::abs(p.ms - (a + b*p.T)));
            const double per = b*1000.0;
            if (!base) base = per;
            uint64_t m = 0; for (auto& p : v.pts) m += p.mism;
            fmt::print("  {:<20}{:>12.3f}{:>11.4f}{:>11.4f}{:>10.2f}{:>8}   {}\n",
                       v.name, per, a, resid, per/base, m, v.fn);
        }
        fmt::print("\n  Both kernels issue 11 mul16 = 22 sfpmul24. They differ in shifts and in the\n"
                   "  number of final conditional subtracts (Montgomery 1, Barrett 2), not in\n"
                   "  multiply count -- so this measures the reduction, not the multiply.\n");

        if (const char* f = std::getenv("MM2_CSV")) {
            std::ofstream o(f);
            o << "# exp_modmul002: Barrett vs Montgomery, one core, q=" << q << ", mu=" << MU << "\n"
              << "# Each kernel checked against its OWN golden: montgomery a*b*R^-1 mod q, barrett a*b mod q.\n"
              << "kernel,computes,T,ms,mismatches\n";
            for (auto& v : V) for (auto& p : v.pts)
                o << v.name << "," << v.fn << "," << p.T << "," << p.ms << "," << p.mism << "\n";
            fmt::print("\nwrote {}\n", f);
        }
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
    return pass ? 0 : 1;
}
