// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_modmul001 — three Montgomery modmul kernels, one core, head to head, from scratch.
// docs/he/exp_modmul001.md
//
// This experiment exists because the earlier comparisons were not fair and could not be trusted:
//
//   * `kernels/experiments/exp_tb001/compute/modmul_sfpi_hybrid.cpp` carries ~8 `DeviceZoneScopedN` markers
//     INSIDE its per-tile loop; `modmul_sfpi_full.cpp` carries one at kernel entry. Timing those two
//     files against each other compared instrumentation as well as arithmetic.
//   * per-tile figures were obtained by dividing a dispatch span by T, which silently assumes the
//     kernel loops T times and that the dispatch has no fixed cost. One earlier experiment was wrong
//     for exactly that reason.
//
// So: three self-contained kernels in kernels/experiments/exp_modmul001/, scaffolded identically, with **zero**
// profiler zones in any of them, fed by one reader and drained by one writer that are also identical.
// The only differences are the arithmetic under test.
//
// THE THREE COMPARED ARE ALL CANONICAL -- each returns a residue in [0, q), so they are
// interchangeable at any call site and no result here carries a contract caveat:
//
//   modmul_sfpi_full        fp32 byte-split multiply in SFPI, 7 passes
//   modmul_sfpi_full_bh     _full with mul16() swapped for sfpi::fractional_mul (SFPMUL24),
//                           7 passes. Differs from _full ONLY in that function.
//   modmul_sfpi_hybrid_canon  mul_int_tile for the products, SFPI in registers for the rest,
//                           4 passes, with the canonical finalize restored.
//
// The LAZY kernel (modmul_sfpi_hybrid, output in [0, 2q)) is available with MM1_LAZY=1 for
// reference, but is not in the default comparison: ct_mul cannot consume lazy operands -- its
// radix-128 decompose covers exactly 28 bits and modadd_map does exactly one conditional
// subtract. See docs/he/exp_modmul001.md 3b.
//
// METHOD. One core. For each kernel the tile count T is swept, and the per-tile cost is obtained by
// LEAST-SQUARES FIT of span(T) = fixed + T * per_tile, not by dividing. The fit also reports the
// dispatch's fixed cost and the residual, so a kernel that does not actually scale with T is visible
// rather than hidden. Device time is tt-metal's own firmware `*-KERNEL` envelope -- the same
// envelope for all three, since none of them adds a zone.
//
// CORRECTNESS is scored two ways, because the three do not share an output contract:
//   canonical : out == mont_mul(a,b)                    -- what _full and _full_bh must satisfy
//   lazy      : out == that, or that + q, and out < 2q  -- _hybrid's documented contract
// A kernel is only credited with a speed if it is exact under its own stated contract.
//
//   ./build/exp_modmul001 [ITERS=5]
//
//   MM1_T      tile counts to sweep (default 1,2,4,8,16,32,64,128)
//   MM1_LAZY   also run the lazy kernel, for reference
//
// NOT in the list: kernels/experiments/exp_modmul001/compute/modmul_fused3_fp32.cpp. It is the same full fusion
// with the fp32 byte-split multiply put back, and it DOES NOT COMPILE -- the SFPI compiler reports
// "internal compiler error: cannot store sfpu register (register spill)". It is kept as the control
// that proves the fusion depends on sfpmul24's low register cost. See docs/he/exp_modmul001.md 2b.
//   MM1_CORE   which core, as "x,y" (default 0,0)
//   MM1_CSV    write the table here

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

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
constexpr double CLK_MHZ = 1350.0;
using hrc = std::chrono::high_resolution_clock;
double msd(hrc::time_point t) { return std::chrono::duration<double, std::milli>(hrc::now() - t).count(); }

bool is_prime(uint64_t n) { if (n < 2 || (n % 2 == 0 && n != 2)) return false;
    for (uint64_t d = 3; d * d <= n; d += 2) if (n % d == 0) return false; return true; }
std::vector<uint32_t> parse_list(const char* e, std::vector<uint32_t> d) {
    const char* v = std::getenv(e); if (!v) return d;
    std::vector<uint32_t> o; std::stringstream ss(v); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) o.push_back((uint32_t)std::stoul(t));
    return o.empty() ? d : o; }

// ---- device profiler readback: every `*-KERNEL` dispatch group, in issue order
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
    size_t g0=0; uint64_t ge=iv[0].e;
    auto emit=[&](size_t lo,size_t hi){ KStat k; uint64_t s=iv[lo].s,e=iv[lo].e;
        for (size_t i=lo;i<hi;++i){ s=std::min(s,iv[i].s); e=std::max(e,iv[i].e);
            k.risc_us[iv[i].risc]=std::max(k.risc_us[iv[i].risc],(double)(iv[i].e-iv[i].s)/CLK_MHZ); }
        k.span_us=(double)(e-s)/CLK_MHZ; out.push_back(k); };
    for (size_t i=1;i<iv.size();++i){ if (iv[i].s<=ge) ge=std::max(ge,iv[i].e); else { emit(g0,i); g0=i; ge=iv[i].e; } }
    emit(g0,iv.size()); return out; }

struct Pt { uint32_t T; double span_us = 0, math_us = 0, host_ms = 0; size_t d0 = 0, nd = 0;
            uint64_t mism_canon = 0, mism_lazy = 0; };
struct Var { const char* name; const char* path; const char* contract; std::vector<Pt> pts; };
}  // namespace

int main(int argc, char** argv) {
    uint32_t ITERS = (argc > 1) ? (uint32_t)std::stoul(argv[1]) : 5;
    const auto TL = parse_list("MM1_T", {1,2,4,8,16,32,64,128});
    uint32_t CX = 0, CY = 0;
    if (const char* e = std::getenv("MM1_CORE")) { std::sscanf(e, "%u,%u", &CX, &CY); }
    const uint32_t TMAX = *std::max_element(TL.begin(), TL.end());

    uint64_t q = (1ull << 28) - 1; while (!(q & 1) || !is_prime(q)) --q;
    const mont::MontgomeryContext C((uint32_t)q);

    bool pass = true;
    try {
        auto mesh = Dp::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE);
        auto& cq = mesh->mesh_command_queue();
        auto dr = Dp::MeshCoordinateRange(mesh->shape());
        const CoreCoord core{CX, CY};
        const CoreRangeSet crs(std::set<CoreRange>{CoreRange(core)});

        fmt::print("\nexp_modmul001 — three modmul kernels, ONE core ({},{}), q = {} (28-bit)\n", CX, CY, q);
        fmt::print("T swept over {}, {} timed replays each, min taken.\n",
                   [&]{ std::string s; for (size_t i=0;i<TL.size();++i) s += (i?",":"") + std::to_string(TL[i]); return s; }(), ITERS);
        fmt::print("No kernel contains a profiler zone; device time is the firmware *-KERNEL envelope.\n");

        auto mkbuf=[&](uint64_t nt){ Dp::DeviceLocalBufferConfig d{.page_size=I32,.buffer_type=BufferType::DRAM};
            return Dp::MeshBuffer::create(Dp::ReplicatedBufferConfig{.size=nt*I32}, d, mesh.get()); };
        auto bA = mkbuf(TMAX), bB = mkbuf(TMAX), bO = mkbuf(TMAX);

        // canonical inputs in [0, q); golden is the canonical Montgomery product
        std::vector<uint32_t> hA(TMAX*ELEMS), hB(TMAX*ELEMS), gold(TMAX*ELEMS), poison(TMAX*ELEMS, 0xDEADBEEFu), got;
        { std::mt19937 rng(20260816);
          for (auto& v : hA) v = (uint32_t)(rng() % q);
          for (auto& v : hB) v = (uint32_t)(rng() % q);
          for (size_t i = 0; i < hA.size(); ++i) gold[i] = C.mulmod(hA[i], hB[i]); }
        Dp::EnqueueWriteMeshBuffer(cq, bA, hA, false);
        Dp::EnqueueWriteMeshBuffer(cq, bB, hB, false);
        Dp::Finish(cq);

        const char* home = std::getenv("TT_METAL_HOME");
        const std::string PCSV = home ? std::string(home)+"/generated/profiler/.logs/profile_log_device.csv" : "";
        const bool PROF = std::getenv("TT_METAL_DEVICE_PROFILER") && !PCSV.empty();
        if (!PROF) fmt::print("\nNOTE: profiler off -- device columns blank. Re-run with TT_METAL_DEVICE_PROFILER=1.\n");

        // DEFAULT SET IS CANONICAL ONLY. All three return a residue in [0, q), so they are
        // interchangeable at any call site and the comparison needs no contract caveat. The lazy
        // kernel is kept behind MM1_LAZY=1 for reference only -- ct_mul cannot consume its output
        // (docs/he/exp_modmul001.md 3b), so it does not belong in the headline comparison.
        std::vector<Var> V = {
            {"sfpi_full",    "compute/modmul_sfpi_full.cpp",         "canonical [0,q)"},
            {"full_nofill",  "compute/modmul_full_nofill.cpp",       "canonical [0,q)"},
            {"sfpi_full_bh", "compute/modmul_sfpi_full_bh.cpp",      "canonical [0,q)"},
            {"hybrid_canon", "compute/modmul_sfpi_hybrid_canon.cpp", "canonical [0,q)"},
            {"fused_bh",     "compute/modmul_fused_bh.cpp",          "canonical [0,q)"},
            {"fused2_bh",    "compute/modmul_fused2_bh.cpp",         "canonical [0,q)"},
            {"fused3_bh",    "compute/modmul_fused3_bh.cpp",         "canonical [0,q)"},
        };
        if (std::getenv("MM1_LAZY"))
            V.push_back({"sfpi_hybrid", "compute/modmul_sfpi_hybrid.cpp", "lazy [0,2q)"});

        size_t ndisp = 0;
        for (auto& v : V) {
            fmt::print("\n--- {} ({}) ---\n", v.name, v.contract);
            for (uint32_t T : TL) {
                Program p = CreateProgram();
                auto mkcb=[&](CBIndex idx, uint32_t depth){ CircularBufferConfig c(depth*I32, {{idx, tt::DataFormat::UInt32}});
                    c.set_page_size(idx, I32); CreateCircularBuffer(p, crs, c); };
                mkcb(CBIndex::c_0, 2); mkcb(CBIndex::c_1, 2); mkcb(CBIndex::c_16, 2);
                auto rd=CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p, OVERRIDE_KERNEL_PREFIX+std::string(v.path), crs,
                    ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                SetRuntimeArgs(p, rd, core, {(uint32_t)bA->address(), (uint32_t)bB->address(), T});
                SetRuntimeArgs(p, wr, core, {(uint32_t)bO->address(), T});
                SetRuntimeArgs(p, cp, core, {T, (uint32_t)q, C.q_inv_32(),
                                             (uint32_t)(q & 0xFFFF), (uint32_t)(q >> 16)});
                Dp::MeshWorkload wl; wl.add_program(dr, std::move(p));

                Pt pt; pt.T = T; pt.d0 = ndisp;
                Dp::EnqueueWriteMeshBuffer(cq, bO, poison, false); Dp::Finish(cq);
                Dp::EnqueueMeshWorkload(cq, wl, false); Dp::Finish(cq); ++ndisp;

                Dp::EnqueueReadMeshBuffer(cq, got, bO, true);
                for (uint64_t e = 0; e < (uint64_t)T*ELEMS; ++e) {
                    const uint32_t g = gold[e];
                    if (got[e] != g) { ++pt.mism_canon;
                        if (!(got[e] == g + (uint32_t)q && got[e] < 2*q)) ++pt.mism_lazy; } }

                double mn = 1e18;
                for (uint32_t i = 0; i < ITERS; ++i) { auto t0 = hrc::now();
                    Dp::EnqueueMeshWorkload(cq, wl, false); Dp::Finish(cq); mn = std::min(mn, msd(t0)); }
                ndisp += ITERS; pt.host_ms = mn; pt.nd = ndisp - pt.d0;
                if (PROF) { Dp::Finish(cq); ReadMeshDeviceProfilerResults(*mesh); }
                fmt::print("  T={:<3} host {:>7.4f} ms   canon-mism {:<8} lazy-mism {}\n",
                           T, pt.host_ms, pt.mism_canon, pt.mism_lazy);
                v.pts.push_back(pt);
            }
        }

        pass &= mesh->close();

        if (PROF) {
            auto disp = parse_profiler(PCSV);
            fmt::print("\nprofiler: {} dispatch groups; host issued {}\n", disp.size(), ndisp);
            if (disp.size() == ndisp) {
                for (auto& v : V) for (auto& pt : v.pts) { const KStat* best = nullptr;
                    for (size_t i = pt.d0+1; i < pt.d0+pt.nd; ++i)
                        if (!best || disp[i].risc_us[3] < best->risc_us[3]) best = &disp[i];
                    if (best) { pt.span_us = best->span_us; pt.math_us = best->risc_us[3]; } }
            } else fmt::print("  MISMATCH -- device columns dropped rather than misattributed.\n");
        }

        // ---- per-tile cost by least-squares fit of span(T) = fixed + T*per_tile ----
        fmt::print("\n=== device kernel span (us) against T, one core ===\n");
        fmt::print("  {:<16}", "T ="); for (uint32_t T : TL) fmt::print("{:>10}", T); fmt::print("\n");
        for (auto& v : V) { fmt::print("  {:<16}", v.name);
            for (auto& pt : v.pts) fmt::print("{:>10.2f}", pt.span_us); fmt::print("\n"); }

        // ---- PURE COMPUTATION LATENCY: the MATH (TRISC_1) span, fitted (docs/guide/profiling.md 4b) ----
        if (V[0].pts.size() && V[0].pts[0].math_us > 0) {
            fmt::print("\n=== PURE COMPUTATION LATENCY — MATH (TRISC_1) span, fitted ===\n");
            fmt::print("  {:<16}{:>13}{:>11}{:>12}{:>10}{:>8}   {}\n",
                       "kernel","per_tile us","fixed us","max resid","vs full","WH?","output");
            double base = 0;
            for (auto& v : V) {
                double sx=0,sy=0,sxx=0,sxy=0; const double n=(double)v.pts.size();
                for (auto& p : v.pts) { sx+=p.T; sy+=p.math_us; sxx+=(double)p.T*p.T; sxy+=(double)p.T*p.math_us; }
                const double den=n*sxx-sx*sx;
                const double b=den?(n*sxy-sx*sy)/den:0.0, a=den?(sy-b*sx)/n:0.0;
                double resid=0; for (auto& p : v.pts) resid=std::max(resid,std::abs(p.math_us-(a+b*p.T)));
                if (!base) base=b;
                // Wormhole portability: fractional_mul (SFPMUL24) is behind #if __riscv_xtttensixbh,
                // and the pass fusion does not compile without it (register spill).
                const std::string nm = v.name;
                const bool wh = (nm.find("bh") == std::string::npos) && (nm.find("fused") == std::string::npos);
                fmt::print("  {:<16}{:>13.4f}{:>11.3f}{:>12.4f}{:>10.4f}{:>8}   {}\n",
                           v.name, b, a, resid, b/base, wh ? "yes" : "no", v.contract); }
            fmt::print("\n  'WH?' = runs on Wormhole: no sfpmul24, and the fusion will not compile without it.\n");
        }

        fmt::print("\n=== fit  span(T) = fixed + T x per_tile  (envelope, max over cores) ===\n");
        fmt::print("  {:<16}{:>12}{:>12}{:>11}{:>10}{:>13}{:>11}\n",
                   "kernel","per_tile us","fixed us","max resid","vs full","contract","exact?");
        double base = 0;
        std::vector<double> per(V.size(), 0);
        for (size_t k = 0; k < V.size(); ++k) {
            double sx=0, sy=0, sxx=0, sxy=0; const double n=(double)V[k].pts.size();
            for (auto& pt : V[k].pts) { sx+=pt.T; sy+=pt.span_us; sxx+=(double)pt.T*pt.T; sxy+=(double)pt.T*pt.span_us; }
            const double den = n*sxx - sx*sx;
            const double b = den!=0 ? (n*sxy - sx*sy)/den : 0.0;
            const double a = den!=0 ? (sy - b*sx)/n : 0.0;
            double resid = 0; for (auto& pt : V[k].pts) resid = std::max(resid, std::abs(pt.span_us - (a + b*pt.T)));
            per[k] = b; if (k == 0) base = b;
            uint64_t mc=0, ml=0; for (auto& pt : V[k].pts) { mc += pt.mism_canon; ml += pt.mism_lazy; }
            const bool ok = (std::string(V[k].contract)[0]=='c') ? (mc==0) : (ml==0);
            if (!ok) pass = false;
            fmt::print("  {:<16}{:>12.3f}{:>12.2f}{:>11.2f}{:>10.2f}{:>13}{:>11}\n",
                       V[k].name, b, a, resid, base>0?b/base:0.0, V[k].contract, ok?"yes":"NO");
        }
        fmt::print("\n  'exact?' is against the kernel's OWN contract: canonical kernels must match the\n"
                   "  golden bit for bit; the lazy kernel must match it or the golden + q, and stay < 2q.\n");

        if (const char* f = std::getenv("MM1_CSV")) {
            std::ofstream o(f);
            o << "# exp_modmul001: three modmul kernels, one core (" << CX << "," << CY << "), q=" << q
              << ", " << ITERS << " replays (min).\n"
              << "# No kernel contains a profiler zone. Device us = firmware *-KERNEL envelope.\n"
              << "# canon_mism vs the canonical golden; lazy_mism allows golden + q (the hybrid's contract).\n"
              << "kernel,contract,T,span_us,host_ms,canon_mism,lazy_mism\n";
            for (auto& v : V) for (auto& pt : v.pts)
                o << v.name << "," << v.contract << "," << pt.T << "," << pt.span_us << ","
                  << pt.host_ms << "," << pt.mism_canon << "," << pt.mism_lazy << "\n";
            fmt::print("\nwrote {}\n", f);
        }
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
    return pass ? 0 : 1;
}
