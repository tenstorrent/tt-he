// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_noc001 — DRAM versus NoC as the medium for an all-to-all, and what each costs as it scales.
// See docs/noc/exp_noc001.md.
//
// THE APPLICATION is a personalised all-to-all (a block transpose): G cores, core n owning P tiles,
// conceptually a G x G matrix of blocks with block (n,s) on core n; afterwards core s holds
// block (n,s) for every n.  This is the communication skeleton of the NTT corner turn
// (D[s][n1][t] = Cb[n1][s][t], docs/matmul/exp_ntt001.md 2b) with the tile ROW generalised to a
// BLOCK, so that G and the message size become free parameters instead of both being pinned at 32
// by the tile geometry.  It is not the corner turn's exact face-level permutation -- it is the
// traffic pattern, which is what a communication benchmark should isolate.
//
// THREE ROUTES, and the reason there are three rather than two.  It is tempting to measure
// "through DRAM" against "over the NoC" and quote the ratio.  That ratio would conflate two
// independent effects, because the DRAM implementation we actually had was also doing far more
// work:
//
//   A  dram_bulk      through DRAM, reading whole 4 KB pages to use BLK = P*4096/G bytes of each.
//                     Chip-wide traffic O(G^2).  This is the shape all_to_all_corner_mc.cpp has.
//   B  dram_strided   through DRAM, reading exactly the needed bytes via addrgen.get_noc_addr(page,
//                     offset).  Traffic O(G) -- THE SAME BYTES AS ROUTE C.  Pays in request shape
//                     instead: G reads of BLK bytes, down to 64 B, which dram_guide 3.4 says is
//                     the access pattern DRAM likes least.
//   C  noc            core-to-core L1->L1, sender writing into the receiver's final offset so the
//                     receiver assembles nothing.  Generalised from all_to_all_corner_x.cpp.
//
//   A vs B  = same medium, different access pattern   -> isolates the O(G^2) waste
//   B vs C  = SAME BYTES, different medium            -> isolates DRAM-vs-fabric
//
// FAIRNESS.  All three compute the identical permutation and every cell is checked against the same
// host golden before its timing is reported; a route that is fast because it moved the wrong bytes
// cannot be quoted.  All three end by writing the result to DRAM (zone a2a_wb) so the accounting is
// identical.  The one asymmetry is deliberate and is reported both ways: route C must first have its
// own P tiles in L1 (zone a2a_ld), which routes A and B have no equivalent of because their read IS
// the exchange.  In the pipeline this models, that data is already resident from the previous stage
// -- which is the actual source of the win -- so `a2a` alone is the number that transfers, and
// `a2a_ld + a2a + a2a_wb` is the like-for-like number.  Both are printed.
//
// SCALING.  Two axes, chosen so each moves one thing:
//   axis A   G in {4,8,16,32,64} at P=1   -- more cores, more total data, message 1024 -> 64 B
//   axis B   P in {1,2,4} at G=32         -- more data per core at fixed core count, message 128 -> 512 B
// Message size falls out of (G,P) as P*4096/G, so the sweep also covers the transfer-size range
// where the fabric and the memory controller have opposite preferences.
//
//   ./build/exp_noc001
//   env: NOC1_G=32  NOC1_P=1  NOC1_ITERS=50  NOC1_ROUTE=bulk|strided|noc|all  NOC1_PROF=0|1
//        NOC1_CSV=<path>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <map>
#include <string>
#include <vector>

#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t ELEMS = 1024, I32 = ELEMS * 4;      // one 32x32 uint32 tile = 4096 B

using hrc = std::chrono::high_resolution_clock;
double us_since(hrc::time_point t) {
    return std::chrono::duration<double, std::micro>(hrc::now() - t).count(); }
const char* env_s(const char* k, const char* d) { const char* v = std::getenv(k); return v ? v : d; }
uint32_t env_u(const char* k, uint32_t d) { const char* v = std::getenv(k); return v ? (uint32_t)std::stoul(v) : d; }

struct Result { const char* name; uint32_t mism; double wall_us; double gbps; };
}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const uint32_t G = env_u("NOC1_G", 32);
        const uint32_t P = env_u("NOC1_P", 1);
        const int iters  = (int)env_u("NOC1_ITERS", 50);
        const bool prof  = env_u("NOC1_PROF", 0) != 0;
        const std::string route = env_s("NOC1_ROUTE", "all");
        const uint32_t CH = env_u("NOC1_CH", 8);          // DRAM channels to use: 1|2|4|8
        const uint32_t TILES = G * P;                   // total tiles in play
        const uint32_t WPB   = P * ELEMS / G;            // uint32 words per block
        const uint32_t BLK   = WPB * 4;                  // bytes per message

        // The block must be a whole number of words and must not straddle a page, or the strided
        // route's single read per peer is not expressible. Both hold iff G divides P*1024 and
        // BLK divides the tile -- true for every (G,P) with G/P integral, which the sweep respects.
        if (WPB == 0 || P * ELEMS % G != 0 || (G >= P && I32 % BLK != 0)) {
            fmt::print("bad (G={},P={}): block must divide the tile\n", G, P); return 2; }
        if (G > 64) { fmt::print("G<=64 (8x8 grid)\n"); return 2; }
        if (CH != 1 && CH != 2 && CH != 4 && CH != 8) { fmt::print("NOC1_CH must be 1|2|4|8\n"); return 2; }
        // logical tile i -> physical page PAGE(i); every used page has page%8 < CH, so exactly CH
        // banks are touched. The buffer is padded to the highest page the map produces.
        auto PAGE = [CH](uint32_t i) { return (i / CH) * 8 + (i % CH); };
        const uint32_t NPAGES = PAGE(TILES - 1) + 1;

        fmt::print("exp_noc001  G={} cores  P={} tiles/core  block {} B  total {} KB\n",
                   G, P, BLK, TILES * I32 / 1024);
        fmt::print("            DRAM channels {}   ({} physical pages allocated)\n", CH, NPAGES);

        // ---- data + golden: out[s][n] = in[n][s], blocks of WPB words ----
        std::mt19937 rng(12345);
        std::vector<uint32_t> in(TILES * ELEMS), golden(TILES * ELEMS);
        for (auto& v : in) v = rng();
        for (uint32_t n = 0; n < G; ++n)
            for (uint32_t s = 0; s < G; ++s)
                for (uint32_t w = 0; w < WPB; ++w)
                    golden[s * P * ELEMS + n * WPB + w] = in[n * P * ELEMS + s * WPB + w];

        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        auto mkbuf = [&](uint32_t nt) {
            distributed::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * I32}, d, mesh.get()); };
        auto crange = [](uint32_t c) { uint32_t w = c < 8 ? c : 8, h = (c + 7) / 8;
            return CoreRange(CoreCoord{0, 0}, CoreCoord{w - 1, h - 1}); };
        auto cat = [](uint32_t c) { return CoreCoord{c % 8, c / 8}; };
        auto mkcbr = [&](Program& p, CoreRange cr, CBIndex idx, uint32_t depth) {
            CircularBufferConfig cc(depth * I32, {{idx, tt::DataFormat::UInt32}});
            cc.set_page_size(idx, I32); CreateCircularBuffer(p, cr, cc); };

        auto b_in = mkbuf(NPAGES), b_out = mkbuf(NPAGES), b_scr = mkbuf(NPAGES);
        std::vector<uint32_t> in_phys(NPAGES * ELEMS, 0);
        for (uint32_t i = 0; i < TILES; ++i)
            std::copy(in.begin() + i * ELEMS, in.begin() + (i + 1) * ELEMS,
                      in_phys.begin() + PAGE(i) * ELEMS);
        distributed::EnqueueWriteMeshBuffer(cq, b_in, in_phys, false);

        const CoreRange cr = crange(G);

        // ---- route A: whole-page DRAM reads, O(G^2) ----
        auto build_bulk = [&]() {
            Program p = CreateProgram();
            mkcbr(p, cr, CBIndex::c_0, TILES);      // stage every peer's tiles: the O(G^2) buffer
            mkcbr(p, cr, CBIndex::c_1, P);
            auto k = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/a2a_dram_bulk.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            for (uint32_t c = 0; c < G; ++c)
                SetRuntimeArgs(p, k, cat(c), {(uint32_t)b_in->address(), (uint32_t)b_out->address(), G, P, c, CH});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // ---- route B: strided DRAM reads, O(G) bytes, G small requests ----
        auto build_strided = [&]() {
            Program p = CreateProgram();
            mkcbr(p, cr, CBIndex::c_1, P);
            auto k = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/a2a_dram_strided.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            for (uint32_t c = 0; c < G; ++c)
                SetRuntimeArgs(p, k, cat(c), {(uint32_t)b_in->address(), (uint32_t)b_out->address(), G, P, c, CH});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // ---- route C: core-to-core exchange, no DRAM in the exchange ----
        auto build_noc = [&](bool syncprobe) {
            Program p = CreateProgram();
            mkcbr(p, cr, CBIndex::c_0, P); mkcbr(p, cr, CBIndex::c_1, P);
            const uint32_t sem = CreateSemaphore(p, cr, 0), sem2 = CreateSemaphore(p, cr, 0);
            std::map<std::string, std::string> def;
            if (syncprobe) def["A2A_NO_PAYLOAD"] = "1";   // barrier only: MUST fail the gate
            if (std::getenv("NOC1_ALIGN")) def["A2A_ALIGN"] = "1";
            auto k = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/a2a_noc.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                   .noc = NOC::RISCV_1_default, .defines = def});
            std::vector<uint32_t> coords;               // peers' PHYSICAL coords: only the host knows them
            for (uint32_t j = 0; j < G; ++j) { auto ph = mesh->worker_core_from_logical_core(cat(j));
                coords.push_back((uint32_t)ph.x); coords.push_back((uint32_t)ph.y); }
            for (uint32_t c = 0; c < G; ++c) {
                std::vector<uint32_t> av = {(uint32_t)b_in->address(), (uint32_t)b_out->address(), G, P, c, sem, CH, sem2};
                av.insert(av.end(), coords.begin(), coords.end());
                SetRuntimeArgs(p, k, cat(c), av); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // ---- measure: gate on the golden FIRST, then time ----
        std::vector<Result> results;
        auto run = [&](const char* name, distributed::MeshWorkload& wl) {
            std::vector<uint32_t> phys(NPAGES * ELEMS, 0);
            distributed::EnqueueMeshWorkload(cq, wl, false);
            distributed::EnqueueReadMeshBuffer(cq, phys, b_out, true);
            uint32_t mism = 0;
            for (uint32_t i = 0; i < TILES; ++i)          // recover through the same map
                for (uint32_t w = 0; w < ELEMS; ++w)
                    if (phys[PAGE(i) * ELEMS + w] != golden[i * ELEMS + w]) ++mism;
            if (mism) pass = false;

            double wall = 0.0;
            if (!prof) {                                 // steady-state chain, then drain
                distributed::EnqueueMeshWorkload(cq, wl, false); distributed::Finish(cq);
                auto t0 = hrc::now();
                for (int it = 0; it < iters; ++it) distributed::EnqueueMeshWorkload(cq, wl, false);
                distributed::Finish(cq);
                wall = us_since(t0) / iters; }
            // useful bytes moved = the whole dataset, once, regardless of route
            const double gbps = wall > 0 ? (double)TILES * I32 / (wall * 1e-6) / 1e9 : 0.0;
            results.push_back({name, mism, wall, gbps});
            fmt::print("  {:<12} mism {:7d}/{:<8d} wall {:8.2f} us   {:6.1f} GB/s useful\n",
                       name, mism, TILES * ELEMS, wall, gbps);
            std::fflush(stdout); };

        const bool doA = (route == "all" || route == "bulk");
        const bool doB = (route == "all" || route == "strided");
        const bool doC = (route == "all" || route == "noc");
        if (doA) { auto w = build_bulk();    run("dram_bulk", w); }
        if (doB) { auto w = build_strided(); run("dram_strided", w); }
        if (doC) { auto w = build_noc(false); run("noc", w); }
        // ---- routes A'/B': the DRAM round trip, L1 in and L1 out (the application's real scope) ----
        auto build_rt = [&](bool bulk) {
            Program p = CreateProgram();
            mkcbr(p, cr, CBIndex::c_0, P); mkcbr(p, cr, CBIndex::c_1, P);
            if (bulk) mkcbr(p, cr, CBIndex::c_2, TILES);
            const uint32_t sem = CreateSemaphore(p, cr, 0);
            std::map<std::string, std::string> rd;
            if (bulk) rd["RT_BULK"] = "1";
            auto k = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/a2a_rt.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                   .noc = NOC::RISCV_1_default, .defines = rd});
            std::vector<uint32_t> coords;
            for (uint32_t j = 0; j < G; ++j) { auto ph = mesh->worker_core_from_logical_core(cat(j));
                coords.push_back((uint32_t)ph.x); coords.push_back((uint32_t)ph.y); }
            for (uint32_t c = 0; c < G; ++c) {
                std::vector<uint32_t> av = {(uint32_t)b_in->address(), (uint32_t)b_scr->address(),
                                            (uint32_t)b_out->address(), G, P, c, sem, CH};
                av.insert(av.end(), coords.begin(), coords.end());
                SetRuntimeArgs(p, k, cat(c), av); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };
        if (route == "rt" || route == "rt_bulk")    { auto w = build_rt(true);  run("rt_bulk", w); }
        if (route == "rt" || route == "rt_strided") { auto w = build_rt(false); run("rt_strided", w); }
        if (route == "rt") { auto w = build_noc(false); run("noc", w); }

        // ---- bandwidth probe: fixed bytes per core, sweep the message size ----
        // Answers "is the exchange fabric-bound or issue-bound?" and is not gated: it computes no
        // permutation. TOT is held at P tiles so the per-core payload matches the exchange's.
        if (route == "bwprobe") {
            const uint32_t TOT = P * I32;
            for (uint32_t MSG : {64u, 128u, 256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u}) {
                if (MSG > TOT) continue;
                Program p = CreateProgram();
                mkcbr(p, cr, CBIndex::c_0, P); mkcbr(p, cr, CBIndex::c_1, P);
                std::map<std::string, std::string> bd;
                if (std::getenv("NOC1_STATEFUL")) bd["BW_STATEFUL"] = "1";
                auto k = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/noc_bw_probe.cpp", cr,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                       .noc = NOC::RISCV_1_default, .defines = bd});
                for (uint32_t c = 0; c < G; ++c) {
                    auto ph = mesh->worker_core_from_logical_core(cat((c + 1) % G));   // ring: no hotspot
                    SetRuntimeArgs(p, k, cat(c), {0, 0, (uint32_t)ph.x, (uint32_t)ph.y, TOT, MSG}); }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p));
                distributed::EnqueueMeshWorkload(cq, wl, false); distributed::Finish(cq);
                fmt::print("  bwprobe MSG {:5d} B  x {:4d} req  = {} KB/core\n", MSG, TOT / MSG, TOT / 1024);
                std::fflush(stdout); }
            mesh->close(); fmt::print("\nTest PASSED\n"); return 0; }
        if (route == "syncprobe") {   // prices the barrier alone; deliberately wrong, gate must fail
            auto w = build_noc(true); run("syncprobe", w);
            fmt::print("  (syncprobe computes a WRONG answer by design -- a nonzero mism above is the\n"
                       "   evidence the no-payload kernel ran. Read a2a_sync, ignore the verdict.)\n"); }

        // traffic accounting, so the measured ratios can be read against the byte counts
        const double MB = 1024.0 * 1024.0;
        fmt::print("\n  chip-wide DRAM read traffic:  bulk {:.2f} MB (O(G^2))   strided {:.2f} MB   noc {:.2f} MB\n",
                   (double)G * TILES * I32 / MB, (double)TILES * I32 / MB, (double)TILES * I32 / MB);
        fmt::print("  requests per core:            bulk {} x 4096 B   strided {} x {} B   noc {} x {} B (L1->L1)\n",
                   TILES, G, BLK, G, BLK);
        if (prof) fmt::print("\n  profiler armed: one dispatch per route. Zones a2a / a2a_ld / a2a_wb.\n"
                             "  python3 tools/traces/zone_phases.py <profile.csv>\n");
        if (const char* csv = std::getenv("NOC1_CSV")) {
            FILE* f = fopen(csv, "a");
            if (f) { for (auto& r : results)
                        fmt::print(f, "{},{},{},{},{},{:.2f},{:.2f}\n", G, P, BLK, r.name, r.mism, r.wall_us, r.gbps);
                     fclose(f); } }

        mesh->close();
    } catch (const std::exception& e) { fmt::print("EXCEPTION: {}\n", e.what()); pass = false; }
    if (pass) fmt::print("\nTest PASSED\n"); else fmt::print("\nTest FAILED\n");
    return pass ? 0 : 1;
}
