// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_Tg018dram — the DRAM sweep for exp_Tg018's two dominant kernels.  docs/he/exp_tg018dram.md
//
// WHY THIS EXPERIMENT EXISTS
//
//   exp_Tg018's CKKS ct_mul spends 236.6 ms of critical-path busy, and two kernels own 69% of it:
//   modmul_sfpi_full (48.4 ms) and the fused int8 matmul family (43.8 ms across _a / _a_acc / _c).
//   The investigation in docs/he/exp_tg018.md concluded that neither is compute-bound and neither is
//   bandwidth-bound: at G=64 the whole multiply moves DRAM at ~16% of what the same cards reach on a
//   deep-request benchmark, and the packer sits idle 0.57 us out of a 16.5 us dispatch. What limits
//   it is OUTSTANDING REQUEST DEPTH -- each core has 8 KB in flight against a machine that needs
//   ~256 KB per core to saturate, and there is no compute to hide the wait behind.
//
//   That conclusion came from three separate measurements stitched together (a NOC benchmark, a
//   per-RISC breakdown, a per-core table across two experiments). This experiment measures it
//   directly instead, on the actual kernels, with the two things that were confounded in exp_Tg018
//   pulled apart into independent axes:
//
//       T   tiles of work per core     -- what the tree fixes at 1, and the only way to deepen
//                                         requests without changing the schedule
//       G   cores occupied             -- what the tree varies, and what shares the DRAM path
//
//   Every point is ONE program dispatch: G cores in a single CoreRangeSet, T tiles looped INSIDE
//   each core's kernel. So dispatch cost is O(1) across the whole sweep and cannot be what moves the
//   numbers -- the same structural choice exp_Bw001 makes, and for the same reason.
//
// WHAT IS REPORTED, AND WHY EACH ONE IS NEEDED
//
//   host ms      wall of EnqueueMeshWorkload + Finish, min over ITERS. What a caller pays.
//   kern us      the dispatch's device span: max over cores of (last marker - first marker),
//                from the device profiler. This is `critical path busy` for a single dispatch.
//   rd/cp/wr us  the same span split per RISC -- NCRISC is the reader, BRISC the writer, TRISC_*
//                the unpack/math/pack pipeline. This split is what makes the answer falsifiable:
//                a bandwidth story and a compute story predict different columns.
//   GB/s         bytes actually moved to and from DRAM, over the KERNEL span (not the host wall --
//                dividing by a host wall measures the dispatch path, not the memory system).
//   us/tile      kern us / T, the per-core per-tile latency the exp_Tg018 per-core table used.
//
// THE THREE KNOBS THE SWEEP TURNS
//
//   depth mode   seq   D=1: one tile (pair) in flight, one barrier per tile. This is EXACTLY the
//                      production pattern -- reader_ab_depth at D=1 is reader_modmul.cpp, and
//                      reader_fused_depth at D=1 is reader_fused_batch.cpp. The T>1 rows of `seq`
//                      are therefore "the current kernel, run T times", the honest baseline.
//                batch D = min(T, L1 cap): the whole group in flight against one barrier.
//                      seq vs batch at the same (G,T) isolates request depth from everything else.
//   w_share      fused matmul only. A work tile is 4 KB of data and 16 KB of folded weights, and
//                every tile of a dispatch runs against the SAME prime, hence the same W. w_share=1
//                reads W once per core instead of once per tile: 4.5 KB/tile at T=32 against 20 KB.
//   G, T         the two axes above.
//
// CORRECTNESS IS CHECKED AT EVERY POINT, not sampled. The output buffer is poisoned before each
// configuration and the full G*T tile range is compared against a host golden afterwards, so a
// timing row can never be a measurement of a kernel that quietly did nothing. modmul's golden is
// mont::MontgomeryContext::mulmod (bit-identical to what the kernel computes); the fused matmul's is
// (W @ X) mod q evaluated in u64.
//
//   ./build/exp_tg018dram [ITERS=5]
//
//   TGD_G=1,8,64,130     core counts to sweep          TGD_T=1,4,16,64   tiles per core
//   TGD_KERNEL=modmul|fused|both                       TGD_MODE=seq|batch|both
//   TGD_WSHARE=0|1|both  W policy for the fused matmul TGD_ITERS=N       timed repetitions
//   TGD_CSV=path         where to write the row-per-configuration CSV
//   TGD_NOVERIFY=1       skip the golden check (for a quick shape-of-the-curve run)
//
// Kernel-side columns need the device profiler:  TT_METAL_DEVICE_PROFILER=1 ./build/exp_tg018dram
// Without it the sweep still runs and reports host wall; the device columns are left blank. Note
// that the profiler's markers cost a little on every dispatch, so the host wall under it is very
// slightly pessimistic -- run once without for the cleanest host numbers.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <functional>
#include <sstream>
#include <string>
#include <thread>
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

constexpr uint32_t T32 = 32, NB = 4, NW = NB * NB, ELEMS = 1024;
constexpr uint32_t I8 = ELEMS, I32 = ELEMS * 4;
constexpr double CLK_MHZ = 1350.0;          // Blackhole AICLK under load; see docs/he/exp_tb001.md
constexpr uint64_t L1_BUDGET = 1000000;     // of the 1.46 MB L1, what the CBs may claim

using hrc = std::chrono::high_resolution_clock;
double msd(hrc::time_point t) { return std::chrono::duration<double, std::milli>(hrc::now() - t).count(); }

template <typename E> std::vector<E> tilize32(const std::vector<E>& in) {
    std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T32; ++r) for (uint32_t c = 0; c < T32; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = in[r * T32 + c];
    return o;
}

template <class F> void par_for(uint32_t n, F&& f) {
    unsigned h = std::thread::hardware_concurrency();
    const uint32_t NTH = std::min<uint32_t>(h ? h : 1, n);
    if (NTH <= 1) { for (uint32_t i = 0; i < n; ++i) f(i); return; }
    std::vector<std::thread> th;
    std::atomic<uint32_t> next{0};
    for (uint32_t t = 0; t < NTH; ++t)
        th.emplace_back([&]{ for (;;) { uint32_t i = next.fetch_add(1); if (i >= n) break; f(i); } });
    for (auto& x : th) x.join();
}

bool is_prime(uint64_t n) {
    if (n < 2 || (n % 2 == 0 && n != 2)) return false;
    for (uint64_t d = 3; d * d <= n; d += 2) if (n % d == 0) return false;
    return true;
}

std::vector<uint32_t> parse_list(const char* env, std::vector<uint32_t> dflt) {
    const char* e = std::getenv(env);
    if (!e) return dflt;
    std::vector<uint32_t> v;
    std::stringstream ss(e);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back((uint32_t)std::stoul(tok));
    return v.empty() ? dflt : v;
}

// ---------------------------------------------------------------- device profiler readback
//
// The profiler CSV is a flat marker log: one row per zone boundary, tagged with core, RISC and a
// cycle stamp. `*-KERNEL` zones are the per-RISC kernel envelopes tt-metal emits for every dispatch,
// which is what we want -- no zone has to be added to a dataflow kernel to see the reader and writer.
//
// WHEN THE FILE CAN BE READ. tt-metal clears the log directory at device init and only writes this
// CSV out when the device closes -- ReadMeshDeviceProfilerResults pulls markers off the device into
// HOST memory, it does not flush them to disk. So the sweep cannot read its own timings while it
// runs. It calls ReadMeshDeviceProfilerResults after every configuration anyway, because that drain
// is what stops the per-RISC on-device marker buffer from overflowing and silently dropping markers
// (the failure that forced exp_Tg018 onto a census estimate), and then parses the whole log ONCE,
// after mesh->close(), while it is still the same process. One binary, complete tables, no
// post-processing step.
//
// THE JOIN IS VERIFIED, NOT ASSUMED. Dispatch groups appear in the log in issue order, and the sweep
// records one label per EnqueueMeshWorkload in that same order, so group i is label i. If the two
// counts disagree the device columns are dropped with a diagnostic rather than misattributed -- a
// shifted join would silently assign every configuration its neighbour's numbers.
struct KStat {
    bool ok = false;
    uint32_t ncore = 0;
    double span_us = 0;                 // max over cores of (core's last marker - core's first)
    double span_mean_us = 0;            // MEAN over cores of the same -- the gap to span_us is the
    double span_min_us = 0;             // load imbalance, which max-over-cores folds into the answer
    double risc_us[5] = {0, 0, 0, 0, 0};  // BRISC, NCRISC, TRISC_0, TRISC_1, TRISC_2 (max over cores)
};
const char* RISC_NAME[5] = {"BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"};

// Every `*-KERNEL` dispatch group in the log, in issue order.
std::vector<KStat> parse_profiler(const std::string& path) {
    std::vector<KStat> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string line;
    if (!std::getline(f, line)) return out;            // tt-metal's banner line
    if (!std::getline(f, line)) return out;            // the header
    std::vector<std::string> hdr;
    { std::stringstream ss(line); std::string t;
      while (std::getline(ss, t, ',')) {
          size_t a = t.find_first_not_of(" \t\r\n"), b = t.find_last_not_of(" \t\r\n");
          hdr.push_back(a == std::string::npos ? "" : t.substr(a, b - a + 1)); } }
    auto col = [&](const char* n) -> int {
        for (size_t i = 0; i < hdr.size(); ++i) if (hdr[i] == n) return (int)i;
        return -1; };
    const int cx = col("core_x"), cy = col("core_y"), cr = col("RISC processor type"),
              ct = col("time[cycles since reset]"), cz = col("zone name"), cp = col("type");
    if (cx < 0 || cy < 0 || cr < 0 || ct < 0 || cz < 0 || cp < 0) return out;

    struct Iv { uint64_t s, e; uint32_t core, risc; };
    std::vector<Iv> iv;
    std::map<std::tuple<uint32_t, uint32_t, std::string>, std::vector<uint64_t>> open;
    while (std::getline(f, line)) {
        std::vector<std::string> x;
        { std::stringstream ss(line); std::string t;
          while (std::getline(ss, t, ',')) {
              size_t a = t.find_first_not_of(" \t\r\n"), b = t.find_last_not_of(" \t\r\n");
              x.push_back(a == std::string::npos ? "" : t.substr(a, b - a + 1)); } }
        if ((int)x.size() <= std::max({cx, cy, cr, ct, cz, cp})) continue;
        const std::string& z = x[cz];
        if (z.size() < 7 || z.compare(z.size() - 7, 7, "-KERNEL") != 0) continue;
        int ri = -1;
        for (int i = 0; i < 5; ++i) if (x[cr] == RISC_NAME[i]) ri = i;
        if (ri < 0) continue;
        const uint32_t core = (uint32_t)std::stoul(x[cx]) * 1000u + (uint32_t)std::stoul(x[cy]);
        auto key = std::make_tuple(core, (uint32_t)ri, z);
        const uint64_t t = std::stoull(x[ct]);
        if (x[cp] == "ZONE_START") open[key].push_back(t);
        else if (x[cp] == "ZONE_END" && !open[key].empty()) {
            iv.push_back({open[key].back(), t, core, (uint32_t)ri});
            open[key].pop_back();
        }
    }
    if (iv.empty()) return out;
    std::sort(iv.begin(), iv.end(), [](const Iv& a, const Iv& b){ return a.s < b.s; });

    // Group by overlap -- the rule tools/traces/dispatch_csv.py uses. Dispatches are serialized on
    // the command queue (exp_Tg018 measured 0 overlapping adjacent pairs out of 6412), so an overlap
    // in start/end times means "same dispatch" and a gap means "the next one".
    auto emit = [&](size_t lo, size_t hi) {
        std::map<uint32_t, std::pair<uint64_t, uint64_t>> per;   // core -> (min start, max end)
        KStat s;
        for (size_t i = lo; i < hi; ++i) {
            auto it = per.find(iv[i].core);
            if (it == per.end()) per.emplace(iv[i].core, std::make_pair(iv[i].s, iv[i].e));
            else { it->second.first = std::min(it->second.first, iv[i].s);
                   it->second.second = std::max(it->second.second, iv[i].e); }
            s.risc_us[iv[i].risc] = std::max(s.risc_us[iv[i].risc], (double)(iv[i].e - iv[i].s) / CLK_MHZ);
        }
        double sum = 0, mn = 1e300;
        for (auto& [c, p] : per) {
            const double d = (double)(p.second - p.first) / CLK_MHZ;
            s.span_us = std::max(s.span_us, d); sum += d; mn = std::min(mn, d);
        }
        s.span_mean_us = per.empty() ? 0 : sum / per.size();
        s.span_min_us = per.empty() ? 0 : mn;
        s.ncore = (uint32_t)per.size();
        s.ok = true;
        out.push_back(s);
    };
    size_t gstart = 0;
    uint64_t gend = iv[0].e;
    for (size_t i = 1; i < iv.size(); ++i) {
        if (iv[i].s <= gend) { gend = std::max(gend, iv[i].e); }
        else { emit(gstart, i); gstart = i; gend = iv[i].e; }
    }
    emit(gstart, iv.size());
    return out;
}

struct Row {
    std::string kern, mode;
    uint32_t G, T, D, wshare, extra;
    uint64_t tiles, bytes;
    double host_ms;
    size_t disp0 = 0, ndisp = 0;   // this row's slice of the dispatch log (verify + timed)
    KStat k;
    uint64_t mism;
};

}  // namespace

int main(int argc, char** argv) {
    uint32_t ITERS = (argc > 1) ? (uint32_t)std::stoul(argv[1]) : 5;
    if (const char* e = std::getenv("TGD_ITERS")) { int v = std::atoi(e); if (v > 0) ITERS = (uint32_t)v; }
    const bool VERIFY = std::getenv("TGD_NOVERIFY") == nullptr;
    const std::string WHICH = std::getenv("TGD_KERNEL") ? std::getenv("TGD_KERNEL") : "both";
    const std::string MODES = std::getenv("TGD_MODE") ? std::getenv("TGD_MODE") : "both";
    const std::string WSH = std::getenv("TGD_WSHARE") ? std::getenv("TGD_WSHARE") : "both";
    const std::string CSVOUT = std::getenv("TGD_CSV") ? std::getenv("TGD_CSV") : "docs/he/data/tg018dram_sweep.csv";

    auto Tlist = parse_list("TGD_T", {1, 2, 4, 8, 16, 32, 64});
    auto Glist = parse_list("TGD_G", {1, 2, 4, 8, 16, 32, 64, 104, 130});
    std::sort(Tlist.begin(), Tlist.end());
    std::sort(Glist.begin(), Glist.end());

    // one 28-bit prime for the whole sweep; the arithmetic is the same at every prime and this is a
    // performance experiment, so a second one would only cost time
    uint64_t q = (1ull << 28) - 1;
    while (!(q & 1) || !is_prime(q)) --q;
    const mont::MontgomeryContext C((uint32_t)q);

    bool pass = true;
    try {
        auto mesh = Dp::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE);
        auto& cq = mesh->mesh_command_queue();
        auto dr = Dp::MeshCoordinateRange(mesh->shape());
        auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX = grid.x, GY = grid.y, GMAX = grid.x * grid.y;
        // Core PLACEMENT, not core count. Row-major packs the first G cores into as few grid ROWS as
        // possible (G<=13 is one row); column-major spreads them down columns, so the same G spans as
        // many rows as it can. The two differ only in which NoC links the cores share on the way to
        // DRAM, so comparing them at fixed G separates "the memory system is saturated" from "these
        // particular cores share a route".
        // TGD_BLOCKW=W lays the G cores out as a W-wide block: W=13 is row-major (as few grid rows as
        // possible), W=1 is one core per row. Sweeping W at fixed G changes the SHAPE of the core set
        // and nothing else, which is what tests whether the contended resource is per-row.
        const char* PLACE = std::getenv("TGD_PLACE");
        const std::string PL = PLACE ? PLACE : "row";
        uint32_t BW = GX;
        if (const char* e = std::getenv("TGD_BLOCKW")) { int v = std::atoi(e); if (v > 0) BW = (uint32_t)v; }
        // DIAGONAL: {i%GX, i%GY}. GX=13 and GY=10 are coprime, so by CRT the 130 pairs are all
        // distinct and the first 10 cores land on 10 distinct ROWS and 10 distinct COLUMNS. Column
        // placement gives each core its own row but stacks them all in column 0; this is the test
        // that separates "the row was the constraint" from "the row and the column both were".
        auto CC = [&](uint32_t i) -> CoreCoord {
            if (PL == "diag") return CoreCoord{i % GX, i % GY};
            if (PL == "col")  return CoreCoord{i / GY, i % GY};
            return CoreCoord{i % BW, i / BW};
        };
        auto crs_of = [&](uint32_t c){ std::set<CoreRange> ss;
            for (uint32_t i = 0; i < c; ++i) ss.insert(CoreRange(CC(i))); return CoreRangeSet(ss); };

        Glist.erase(std::remove_if(Glist.begin(), Glist.end(), [&](uint32_t g){ return g == 0 || g > GMAX; }),
                    Glist.end());
        if (Glist.empty()) { fmt::print(stderr, "no usable core counts\n"); return 2; }

        const uint32_t TMAX = Tlist.back(), GTOP = Glist.back();
        const uint64_t NT = (uint64_t)TMAX * GTOP;                 // tiles per operand buffer

        fmt::print("\nexp_Tg018dram — DRAM sweep over modmul_sfpi_full and the fused int8 matmul\n");
        fmt::print("compute grid {}x{} = {} cores;  q = {} (28-bit)\n", grid.x, grid.y, GMAX, q);
        fmt::print("sweep  T = tiles/core in {}\n       G = cores      in {}\n",
                   fmt::join(Tlist, ","), fmt::join(Glist, ","));
        fmt::print("buffers: {} tiles/operand = {:.1f} MB each;  ONE dispatch per point;  {} timed replays\n",
                   NT, (double)NT * I32 / (1 << 20), ITERS);

        auto mkbuf = [&](uint64_t nt, uint32_t page) {
            Dp::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return Dp::MeshBuffer::create(Dp::ReplicatedBufferConfig{.size = nt * page}, d, mesh.get()); };
        auto mkcb = [&](Program& p, const CoreRangeSet& crs, CBIndex idx, tt::DataFormat fmt,
                        uint32_t depth, uint32_t page) {
            CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page);
            CreateCircularBuffer(p, crs, c); };

        auto bA = mkbuf(NT, I32), bB = mkbuf(NT, I32), bC = mkbuf(NT, I32);
        auto bW = mkbuf(NW, I8);

        // ---- host data and goldens ----
        fmt::print("\nbuilding inputs and host goldens ({} tiles)...\n", NT);
        auto tgen = hrc::now();
        std::vector<uint32_t> hA(NT * ELEMS), hB(NT * ELEMS), gMM(NT * ELEMS), gFM(NT * ELEMS);
        std::vector<uint32_t> hW(ELEMS), fW(NW * ELEMS);
        {
            std::mt19937 rng(20260803);
            for (uint32_t i = 0; i < ELEMS; ++i) hW[i] = (uint32_t)(rng() % q);
            // LHS fold, verbatim from exp_Tg018's DevNTT::foldL: l[i] = limb((POW[r]*W[i]) mod q, p)
            const uint64_t POW[4] = {1, 128, 16384, 2097152};
            std::vector<int8_t> f8(NW * ELEMS);
            for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
                std::vector<int8_t> l(ELEMS);
                for (uint32_t i = 0; i < ELEMS; ++i) {
                    const uint32_t v = (uint32_t)((POW[r] * hW[i]) % q);
                    l[i] = (int8_t)((v >> (7 * p)) & 127);
                }
                auto tl = tilize32(l);
                std::copy(tl.begin(), tl.end(), f8.begin() + (p * NB + r) * ELEMS);
            }
            std::memcpy(fW.data(), f8.data(), f8.size());
        }
        {
            std::vector<uint32_t> seeds(NT);
            std::mt19937 sr(7);
            for (uint64_t t = 0; t < NT; ++t) seeds[t] = sr();
            par_for((uint32_t)NT, [&](uint32_t t) {
                std::mt19937 rng(seeds[t]);
                std::vector<uint32_t> xrm(ELEMS), brm(ELEMS), crm(ELEMS);
                for (uint32_t i = 0; i < ELEMS; ++i) { xrm[i] = (uint32_t)(rng() % q); brm[i] = (uint32_t)(rng() % q); }
                // A holds the TILIZED matrix (the fused matmul needs a matrix; modmul is layout-agnostic)
                auto at = tilize32(xrm);
                std::copy(at.begin(), at.end(), hA.begin() + (size_t)t * ELEMS);
                std::copy(brm.begin(), brm.end(), hB.begin() + (size_t)t * ELEMS);
                for (uint32_t i = 0; i < ELEMS; ++i) gMM[(size_t)t * ELEMS + i] = C.mulmod(at[i], brm[i]);
                for (uint32_t r = 0; r < T32; ++r) for (uint32_t c2 = 0; c2 < T32; ++c2) {
                    uint64_t acc = 0;
                    for (uint32_t k = 0; k < T32; ++k) acc += (uint64_t)hW[r * T32 + k] * xrm[k * T32 + c2] % q;
                    crm[r * T32 + c2] = (uint32_t)(acc % q);
                }
                auto ct = tilize32(crm);
                std::copy(ct.begin(), ct.end(), gFM.begin() + (size_t)t * ELEMS);
            });
        }
        fmt::print("  goldens built in {:.1f} s\n", msd(tgen) / 1000.0);

        Dp::EnqueueWriteMeshBuffer(cq, bA, hA, false);
        Dp::EnqueueWriteMeshBuffer(cq, bB, hB, false);
        Dp::EnqueueWriteMeshBuffer(cq, bW, fW, false);
        Dp::Finish(cq);

        // Poisoned before every configuration, so a row can never be a timing of a kernel that did
        // nothing: the golden check below would have to pass against 0xDEADBEEF.
        std::vector<uint32_t> poison(NT * ELEMS, 0xDEADBEEFu);

        // TGD_EXTRA_W: the same knob on the WRITER, which owns the other NoC. Reader traffic rides
        // NOC_1, writer traffic NOC_0, so this is what lets a configuration drive both.
        uint32_t EW = 0;
        if (const char* e = std::getenv("TGD_EXTRA_W")) { int v = std::atoi(e); if (v > 0) EW = (uint32_t)v; }
        // TGD_NBANK>0 pins each core to ONE DRAM channel: page stride = the bank count, offset =
        // (core index + TGD_CHANOFF) mod bank count. TGD_CHANOFF alone selects which channel a
        // single-core run measures.
        uint32_t NBANK = 0, CHANOFF = 0;
        if (const char* e = std::getenv("TGD_NBANK"))   { int v = std::atoi(e); if (v > 0) NBANK = (uint32_t)v; }
        if (const char* e = std::getenv("TGD_CHANOFF")) { int v = std::atoi(e); if (v >= 0) CHANOFF = (uint32_t)v; }
        // TGD_CHANMOD: how many DISTINCT channels the cores spread over. Default = NBANK (one core
        // per channel); 1 puts every core on the same channel, which is what separates "this is the
        // channel's limit" from "this is one core's reach into a channel".
        uint32_t CHANMOD = 0;
        if (const char* e = std::getenv("TGD_CHANMOD")) { int v = std::atoi(e); if (v > 0) CHANMOD = (uint32_t)v; }
        auto chan_of = [&](uint32_t k){
            if (!NBANK) return 0u;
            const uint32_t m = CHANMOD ? CHANMOD : NBANK;
            return (k % m + CHANOFF) % NBANK; };

        // ---- L1 budget -> how deep a group may be ----
        auto l1_bytes = [&](bool fused, uint32_t D, uint32_t wshare) -> uint64_t {
            if (!fused) return 3ull * 2 * D * I32;                       // c_0, c_1, c_16
            return (uint64_t)2 * D * I32 * 2                             // c_0 + c_16
                 + (uint64_t)(wshare ? NW : 2 * D * NW) * I8             // c_1
                 + (uint64_t)NB * I8 + (uint64_t)NB * I32;               // c_2 + c_3
        };
        auto depth_for = [&](bool fused, uint32_t T, uint32_t wshare) {
            uint32_t D = T;
            while (D > 1 && l1_bytes(fused, D, wshare) > L1_BUDGET) D >>= 1;
            return D;
        };

        // ---- program builders. Every point is ONE program: G cores, T tiles looped in-kernel. ----
        auto build_modmul = [&](uint32_t G, uint32_t T, uint32_t D, uint32_t extra) {
            Program p = CreateProgram(); auto crs = crs_of(G);
            const uint32_t depth = 2 * D;                    // multiple of D => a group never wraps
            mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, depth, I32);
            mkcb(p, crs, CBIndex::c_1, tt::DataFormat::UInt32, depth, I32);
            mkcb(p, crs, CBIndex::c_16, tt::DataFormat::UInt32, depth, I32);
            if (extra) mkcb(p, crs, CBIndex::c_4, tt::DataFormat::UInt32, 32, I32);  // traffic sink: 32 distinct L1 slots
            if (EW) mkcb(p, crs, CBIndex::c_5, tt::DataFormat::UInt32, 32, I32);     // NOC_0 sink
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab_depth.cpp", crs,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32_depth.cpp", crs,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_full.cpp", crs,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            for (uint32_t k = 0; k < G; ++k) {
                const uint32_t st = k * T;
                SetRuntimeArgs(p, rd, CC(k), {(uint32_t)bA->address(), (uint32_t)bB->address(), T, st, D,
                                              extra, (uint32_t)NT, NBANK, chan_of(k)});
                SetRuntimeArgs(p, wr, CC(k), {(uint32_t)bC->address(), T, st, D, EW, (uint32_t)NT, (uint32_t)bA->address()});
                SetRuntimeArgs(p, cp, CC(k), {T, (uint32_t)q, C.q_inv_32(),
                                              (uint32_t)(q & 0xFFFF), (uint32_t)(q >> 16)});
            }
            Dp::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl;
        };

        auto build_fused = [&](uint32_t G, uint32_t T, uint32_t D, uint32_t wshare, uint32_t extra) {
            Program p = CreateProgram(); auto crs = crs_of(G);
            const uint32_t depth = 2 * D;
            mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, depth, I32);
            mkcb(p, crs, CBIndex::c_1, tt::DataFormat::Int8, wshare ? NW : depth * NW, I8);
            mkcb(p, crs, CBIndex::c_2, tt::DataFormat::Int8, NB, I8);
            mkcb(p, crs, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);
            mkcb(p, crs, CBIndex::c_16, tt::DataFormat::UInt32, depth, I32);
            if (extra) mkcb(p, crs, CBIndex::c_4, tt::DataFormat::UInt32, 32, I32);  // traffic sink: 32 distinct L1 slots
            if (EW) mkcb(p, crs, CBIndex::c_5, tt::DataFormat::UInt32, 32, I32);     // NOC_0 sink
            std::vector<uint32_t> rc;
            TensorAccessorArgs(*bA).append_to(rc);
            TensorAccessorArgs(*bW).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused_depth.cpp", crs,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                                   .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32_depth.cpp", crs,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a_loop.cpp", crs,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            for (uint32_t k = 0; k < G; ++k) {
                const uint32_t st = k * T;
                SetRuntimeArgs(p, rd, CC(k), {(uint32_t)bA->address(), (uint32_t)bW->address(), T, st, D,
                                              wshare, extra, (uint32_t)NT, NBANK, chan_of(k)});
                SetRuntimeArgs(p, wr, CC(k), {(uint32_t)bC->address(), T, st, D, EW, (uint32_t)NT, (uint32_t)bA->address()});
                SetRuntimeArgs(p, cp, CC(k), {(uint32_t)q, C.q_inv_32(), (uint32_t)(q & 0xFFFF),
                                              (uint32_t)(q >> 16), C.to_montgomery(16384u), T, wshare});
            }
            Dp::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl;
        };

        const char* home = std::getenv("TT_METAL_HOME");
        const std::string PCSV = home ? std::string(home) + "/generated/profiler/.logs/profile_log_device.csv" : "";
        const bool PROF = std::getenv("TT_METAL_DEVICE_PROFILER") != nullptr && !PCSV.empty();
        if (!PROF)
            fmt::print("\nNOTE: device profiler is OFF -- kernel-side columns will be blank.\n"
                       "      re-run as  TT_METAL_DEVICE_PROFILER=1 ./build/exp_tg018dram\n");

        std::vector<Row> rows;
        std::vector<uint32_t> got;
        size_t ndispatch = 0;                    // running count of EnqueueMeshWorkload calls

        auto run_point = [&](const char* kname, const char* mode, uint32_t G, uint32_t T, uint32_t D,
                             uint32_t wshare, uint32_t extra, Dp::MeshWorkload& wl, uint64_t bytes) {
            Row r;
            r.kern = kname; r.mode = mode; r.G = G; r.T = T; r.D = D; r.wshare = wshare; r.extra = extra;
            r.tiles = (uint64_t)G * T; r.bytes = bytes; r.mism = 0;
            r.disp0 = ndispatch;

            if (VERIFY) { Dp::EnqueueWriteMeshBuffer(cq, bC, poison, false); Dp::Finish(cq); }
            Dp::EnqueueMeshWorkload(cq, wl, false);
            Dp::Finish(cq);
            ++ndispatch;
            if (VERIFY) {
                Dp::EnqueueReadMeshBuffer(cq, got, bC, true);
                const std::vector<uint32_t>& gold = (std::strcmp(kname, "modmul_sfpi_full") == 0) ? gMM : gFM;
                for (uint64_t e = 0; e < r.tiles * ELEMS; ++e) if (got[e] != gold[e]) ++r.mism;
                if (r.mism) pass = false;
            }

            double mn = 1e18;
            for (uint32_t i = 0; i < ITERS; ++i) {
                auto t0 = hrc::now();
                Dp::EnqueueMeshWorkload(cq, wl, false);
                Dp::Finish(cq);
                mn = std::min(mn, msd(t0));
            }
            ndispatch += ITERS;
            r.host_ms = mn;
            r.ndisp = ndispatch - r.disp0;

            // Drain the device-side marker buffers into host memory. Nothing reaches the CSV until
            // close, but without this the per-RISC on-device buffer overflows and tt-metal silently
            // drops markers -- which would shift the group<->label join for everything after it.
            if (PROF) { Dp::Finish(cq); ReadMeshDeviceProfilerResults(*mesh); }

            rows.push_back(r);
            fmt::print("  {:<20} {:<5} G={:<4} T={:<3} D={:<3} ws={} E={:<3} | host {:8.3f} ms |"
                       " {:7.1f} MB | mism {}\n",
                       kname, mode, G, T, D, wshare, extra, r.host_ms, (double)bytes / (1 << 20), r.mism);
        };

        const bool do_mm = (WHICH == "both" || WHICH == "modmul");
        const bool do_fm = (WHICH == "both" || WHICH == "fused");
        std::vector<std::string> modes;
        if (MODES == "both") modes = {"seq", "batch"}; else modes = {MODES};
        std::vector<uint32_t> wshares;
        if (WSH == "both") wshares = {0, 1}; else wshares = {(uint32_t)std::stoul(WSH)};
        // E = extra 4 KB reads per work tile. The arithmetic-intensity axis: it changes `s` in
        // C = max(C_compute, G*s/B) and nothing else, so it walks the knee across the core grid.
        auto Elist = parse_list("TGD_EXTRA", {0});

        if (do_mm) {
            fmt::print("\n=== modmul_sfpi_full — 3 tiles of DRAM traffic per work tile (A, B, C) ===\n");
            for (uint32_t E : Elist) for (const auto& m : modes) for (uint32_t T : Tlist) {
                const uint32_t D = (m == "seq") ? 1u : depth_for(false, T, 0);
                for (uint32_t G : Glist) {
                    auto wl = build_modmul(G, T, D, E);
                    run_point("modmul_sfpi_full", m.c_str(), G, T, D, 0, E, wl,
                              (uint64_t)G * T * (3ull + E + EW) * I32);
                }
            }
        }
        if (do_fm) {
            fmt::print("\n=== fused_matmul_a_loop — 1 input tile + {} folded-W int8 tiles + 1 output ===\n", NW);
            for (uint32_t E : Elist) for (uint32_t ws : wshares) for (const auto& m : modes)
              for (uint32_t T : Tlist) {
                const uint32_t D = (m == "seq") ? 1u : depth_for(true, T, ws);
                for (uint32_t G : Glist) {
                    auto wl = build_fused(G, T, D, ws, E);
                    const uint64_t bytes = ws
                        ? (uint64_t)G * ((uint64_t)T * (2 + E + EW) * I32 + (uint64_t)NW * I8)
                        : (uint64_t)G * T * ((uint64_t)(2 + E + EW) * I32 + (uint64_t)NW * I8);
                    run_point("fused_matmul_a_loop", m.c_str(), G, T, D, ws, E, wl, bytes);
                }
            }
        }

        // ---- close the device FIRST: that is when tt-metal writes the profiler log out ----
        pass &= mesh->close();

        if (PROF) {
            auto disp = parse_profiler(PCSV);
            fmt::print("\nprofiler log: {} dispatch groups; host issued {}\n", disp.size(), ndispatch);
            if (disp.size() != ndispatch) {
                fmt::print("  MISMATCH -- the group<->configuration join is not 1-1, so device columns\n"
                           "  are left blank rather than misattributed. Markers were probably dropped;\n"
                           "  re-run a narrower sweep (fewer G, fewer T, TGD_ITERS=1).\n");
            } else {
                for (auto& r : rows) {
                    // min over the TIMED dispatches, matching the host's min-of-ITERS. The verify
                    // dispatch (r.disp0) is excluded: it is the first enqueue of that workload and
                    // carries the kernel-binary write.
                    const KStat* best = nullptr;
                    for (size_t i = r.disp0 + 1; i < r.disp0 + r.ndisp; ++i)
                        if (!best || disp[i].span_us < best->span_us) best = &disp[i];
                    if (best) r.k = *best;
                    if (r.k.ok && r.k.ncore != r.G)
                        fmt::print("  [warn] G={} T={} {}: profiler saw {} cores\n",
                                   r.G, r.T, r.mode, r.k.ncore);
                }
            }
        }

        // ---------------- report ----------------
        auto table = [&](const char* title, const std::function<bool(const Row&)>& sel) {
            std::vector<const Row*> rs;
            for (auto& r : rows) if (sel(r)) rs.push_back(&r);
            if (rs.empty()) return;
            fmt::print("\n{}\n", title);
            fmt::print("   read = NCRISC, unpk/math/pack = TRISC_0/1/2, write = BRISC; each is a"
                       " max over cores\n");
            fmt::print("  mode |   G |  T |  D |    tiles |   MB |  host ms |   kern us |   read |   unpk |"
                       "   math |   pack |  write |  GB/s | us/tile | mism\n");
            fmt::print("  -----+-----+----+----+----------+------+----------+-----------+--------+--------+"
                       "--------+--------+--------+-------+---------+------\n");
            for (auto* r : rs) {
                const double gbs = (r->k.ok && r->k.span_us > 0) ? (double)r->bytes / (r->k.span_us * 1e-6) / 1e9 : 0.0;
                fmt::print("  {:<5}|{:5}|{:4}|{:4}|{:9}|{:6.1f}|{:9.3f}|{:10.2f}|{:8.2f}|{:8.2f}|{:8.2f}|"
                           "{:8.2f}|{:8.2f}|{:6.1f}|{:8.3f}|{:5}\n",
                           r->mode, r->G, r->T, r->D, r->tiles, (double)r->bytes / (1 << 20), r->host_ms,
                           r->k.ok ? r->k.span_us : 0.0,
                           r->k.ok ? r->k.risc_us[1] : 0.0,     // NCRISC = reader
                           r->k.ok ? r->k.risc_us[2] : 0.0,     // TRISC_0 = unpack
                           r->k.ok ? r->k.risc_us[3] : 0.0,     // TRISC_1 = math
                           r->k.ok ? r->k.risc_us[4] : 0.0,     // TRISC_2 = pack
                           r->k.ok ? r->k.risc_us[0] : 0.0,     // BRISC  = writer
                           gbs, r->k.ok ? r->k.span_us / r->T : 0.0, r->mism);
            }
        };
        table("modmul_sfpi_full", [](const Row& r){ return r.kern[0] == 'm'; });
        table("fused_matmul_a_loop, w_share=0 (W re-read per work tile)",
              [](const Row& r){ return r.kern[0] == 'f' && r.wshare == 0; });
        table("fused_matmul_a_loop, w_share=1 (W read once per core)",
              [](const Row& r){ return r.kern[0] == 'f' && r.wshare == 1; });

        {
            std::ofstream f(CSVOUT);
            f << "kernel,mode,G,T,depth,w_share,extra,tiles,bytes,host_ms,kern_us,read_us,unpack_us,math_us,"
                 "pack_us,write_us,gbs_kern,gbs_host,us_per_tile,kern_mean_us,kern_min_us,ncore,mismatches\n";
            for (auto& r : rows) {
                const double gk = (r.k.ok && r.k.span_us > 0) ? (double)r.bytes / (r.k.span_us * 1e-6) / 1e9 : 0.0;
                const double gh = (r.host_ms > 0) ? (double)r.bytes / (r.host_ms * 1e-3) / 1e9 : 0.0;
                f << r.kern << ',' << r.mode << ',' << r.G << ',' << r.T << ',' << r.D << ',' << r.wshare
                  << ',' << r.extra << ',' << r.tiles << ',' << r.bytes << ',' << r.host_ms << ','
                  << (r.k.ok ? r.k.span_us : 0.0) << ',' << (r.k.ok ? r.k.risc_us[1] : 0.0) << ','
                  << (r.k.ok ? r.k.risc_us[2] : 0.0) << ',' << (r.k.ok ? r.k.risc_us[3] : 0.0) << ','
                  << (r.k.ok ? r.k.risc_us[4] : 0.0) << ',' << (r.k.ok ? r.k.risc_us[0] : 0.0) << ','
                  << gk << ',' << gh << ',' << (r.k.ok ? r.k.span_us / r.T : 0.0) << ','
                  << (r.k.ok ? r.k.span_mean_us : 0.0) << ',' << (r.k.ok ? r.k.span_min_us : 0.0) << ','
                  << r.k.ncore << ',' << r.mism << '\n';
            }
            fmt::print("\nwrote {} : {} rows\n", CSVOUT, rows.size());
        }

        fmt::print("\nEvery row is ONE dispatch: the T axis is a kernel-internal loop and the G axis is a\n"
                   "CoreRangeSet in a single program, so dispatch cost is constant across the sweep.\n");
        if (!PROF)
            fmt::print("Kernel-side columns are blank because the device profiler was off.\n");
    } catch (const std::exception& e) {
        fmt::print(stderr, "Exception:\n{}\n", e.what());
        return 2;
    }
    if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
    return pass ? 0 : 1;
}
