// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_dram001 — can the 436 GB/s DRAM figure be beaten?
//
// docs/he/dram_guide.md reports 436 GB/s (8 channels x ~54) and 433 achieved, and calls 436 "the right
// denominator for any bandwidth claim on this part". But that number was produced by ONE access
// pattern: 4 KB tile-sized reads, batch barriers, no bank affinity. A p150 study suggests six changes,
// three of which contradict this repo's own conclusions. Each is a knob here, so each is testable:
//
//   1. LARGE PACKETS.  NOC_MAX_BURST_SIZE on Blackhole is 256 x 64 B = 16384, so a 16 KB read is one
//      burst. The repo has only ever issued 4 KB.                                     -> DR1_PKT
//   2. IN-FLIGHT DEPTH AS A SLIDING WINDOW, not a batch. exp_roofline001 §5b found deeper *batches*
//      LOSE bandwidth because the consumer waits on the whole group; transaction ids retire the read
//      NTRID-ago instead, so the window never drains.                                 -> DR1_TRID
//   3. ~4 REQUESTERS PER CHANNEL. dram_guide §1 already says "one core cannot saturate one channel
//      (42.6 of 53.9) -- it takes about four".                                        -> DR1_PER_BANK
//   4. READERS ADJACENT TO THEIR BANK, using the NoC's own bank->coord mapping. dram_guide's null
//      table says pinning cores to channels is WORSE (189 vs 235 GB/s) -- but that was pinning
//      WITHOUT proximity. This separates the two.                                     -> DR1_MODE=pinadj
//   5. BALANCE ACROSS ALL EIGHT BANKS, and verify each bank separately rather than trusting the sum.
//                                                                                     -> DR1_PERBANK=1
//   6. BOTH NoCs. dram_guide §3.1 measures one NoC at 234.8 against 433 for both -- the largest single
//      factor in the repo. For a pure read stream this means splitting the READS.     -> DR1_NOCS=2
//
// WHAT IS MEASURED: pure DRAM reads. No compute, no writes (beyond a 4-byte checksum per core), so the
// number is the read path alone and is directly comparable to dram_guide §1's ladder.
//
// PROFILING (docs/guide/profiling.md): first execution discarded (JIT); every point is the MIN
// of DR1_ITERS traced replays with the spread printed; Metal Trace so host enqueue is outside the
// timed region. And every point is CHECKSUM-GATED -- a bandwidth kernel that reads the wrong size or
// whose barrier does not wait reports a *higher* number, so an ungated sweep would reward its own bugs.
//
//   ./build/exp_dram001                    the sweeps
//   DR1_PERBANK=1 ./build/exp_dram001      per-bank verification
//
// env: DR1_MB (512), DR1_PKT, DR1_TRID, DR1_CORES, DR1_MODE, DR1_NOCS, DR1_ITERS (5), DR1_CSV

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;
namespace Dp = tt::tt_metal::distributed;
using MBuf = std::shared_ptr<Dp::MeshBuffer>;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
using hrc = std::chrono::high_resolution_clock;
double ms_of(hrc::time_point t) { return std::chrono::duration<double, std::milli>(hrc::now() - t).count(); }
uint32_t envu(const char* k, uint32_t d) { const char* e = std::getenv(k); return e ? (uint32_t)std::atoi(e) : d; }
constexpr uint32_t NB = 8;                     // DRAM banks; bank = page % NUM_DRAM_BANKS
}  // namespace

int main(int argc, char** argv) {
    const uint64_t MB    = envu("DR1_MB", 512);
    const uint32_t ITERS = envu("DR1_ITERS", 5);
    const bool PERBANK   = envu("DR1_PERBANK", 0) != 0;

    bool pass = true;
    try {
        auto mesh = Dp::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 64u << 20);
        auto& cq = mesh->mesh_command_queue();
        auto dr = Dp::MeshCoordinateRange(mesh->shape());
        auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX = grid.x, GFULL = grid.x * grid.y;

        // ---- the physical map: where each DRAM bank sits, and where each worker sits ----
        std::vector<CoreCoord> bank_xy(NB);
        for (uint32_t b = 0; b < NB; ++b)
            bank_xy[b] = mesh->virtual_core_from_logical_core(
                mesh->logical_core_from_dram_channel(b), CoreType::DRAM);
        std::vector<CoreCoord> wk(GFULL), wk_phys(GFULL);
        for (uint32_t i = 0; i < GFULL; ++i) {
            wk[i] = CoreCoord{i % GX, i / GX};
            wk_phys[i] = mesh->worker_core_from_logical_core(wk[i]);
        }
        fmt::print("\nexp_dram001 — pure DRAM read bandwidth, {} banks, {} worker cores\n", NB, GFULL);
        fmt::print("  bank NoC coords:");
        for (uint32_t b = 0; b < NB; ++b) fmt::print(" b{}=({},{})", b, bank_xy[b].x, bank_xy[b].y);
        fmt::print("\n  NOC_MAX_BURST_SIZE = 16384 B, so PKT=16384 is one burst\n\n");

        // workers sorted by NoC distance to each bank -- for DR1_MODE=pinadj
        std::vector<std::vector<uint32_t>> near(NB);
        for (uint32_t b = 0; b < NB; ++b) {
            near[b].resize(GFULL);
            for (uint32_t i = 0; i < GFULL; ++i) near[b][i] = i;
            std::sort(near[b].begin(), near[b].end(), [&](uint32_t a, uint32_t c) {
                auto d = [&](uint32_t k){ return std::abs((int)wk_phys[k].x - (int)bank_xy[b].x)
                                               + std::abs((int)wk_phys[k].y - (int)bank_xy[b].y); };
                return d(a) < d(c); });
        }

        const uint64_t TOT = MB << 20;
        auto mkbuf = [&](uint64_t bytes, uint32_t page) {
            Dp::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return Dp::MeshBuffer::create(Dp::ReplicatedBufferConfig{.size = bytes}, d, mesh.get()); };

        // one source buffer per packet size (page size == packet size, so bank = page % 8)
        std::vector<uint32_t> host;                       // raw words of the source
        auto bOut = mkbuf(4096ull * GFULL * 2, 4096);   // 2 slots per core: one per NoC

        // ---- one traced measurement ----
        // mode: 0 interleave (each core walks a contiguous run -> all 8 banks)
        //       1 pin        (core -> one bank, stride 8, placement row-major)
        //       2 pinadj     (as pin, but the core is the nearest worker to that bank)
        struct Res { double ms, spread; uint64_t bad; };
        auto run = [&](uint32_t PKT, uint32_t TRID, uint32_t cores, int mode, uint32_t nocs,
                       int only_bank, const std::vector<uint32_t>& src_words,
                       const MBuf& src, uint32_t npkt) -> Res {
            // which logical worker index does slot i use, and which pages does it read?
            // Worker assignment must be a BIJECTION. The first version took near[b][i/NB], which lets
            // two banks claim the same physically-close worker -- SetRuntimeArgs then overwrote one
            // slot's work and that slot's reads never happened. The checksum caught it (BAD), which is
            // exactly why a bandwidth kernel needs one: the bug made it look FASTER.
            // A bank-pinned mode partitions each bank's pages among cores/NB cores, so `cores` MUST
            // be a multiple of NB -- otherwise the last cores get rank == cores/NB and address past
            // the end of the buffer. (That read OOB on the host golden and segfaulted.)
            if (mode != 0 && only_bank < 0 && (cores % NB) != 0) return {0, 0, ~0ull};
            std::vector<uint32_t> use(cores);
            { std::vector<char> taken(GFULL, 0);
              for (uint32_t i = 0; i < cores; ++i) {
                  if (mode != 2) { use[i] = i; taken[i] = 1; continue; }
                  const uint32_t b = (only_bank >= 0) ? (uint32_t)only_bank : (i % NB);
                  uint32_t pick = GFULL;
                  for (uint32_t k = 0; k < GFULL; ++k)
                      if (!taken[near[b][k]]) { pick = near[b][k]; break; }
                  if (pick == GFULL) return {0, 0, ~0ull};        // cannot place: report, do not crash
                  taken[pick] = 1; use[i] = pick; } }
            std::set<CoreRange> rs; for (auto u : use) rs.insert(CoreRange(wk[u]));
            CoreRangeSet crs(rs);

            Program p = CreateProgram();
            auto mkcb = [&](CBIndex idx){
                CircularBufferConfig c(TRID * PKT, {{idx, tt::DataFormat::UInt32}});
                c.set_page_size(idx, PKT); CreateCircularBuffer(p, crs, c); };
            mkcb(CBIndex::c_0);
            if (nocs == 2) mkcb(CBIndex::c_1);                    // BRISC lands in its own buffer
            auto k1 = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/read_bw.cpp", crs,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                   .noc = NOC::RISCV_1_default,
                                   .compile_args = {PKT, TRID, (uint32_t)CBIndex::c_0}});
            KernelHandle k0 = 0;
            if (nocs == 2) k0 = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/read_bw.cpp", crs,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                   .noc = NOC::RISCV_0_default,
                                   .compile_args = {PKT, TRID, (uint32_t)CBIndex::c_1}});

            // page assignment, and the host-side golden for each core's XOR fold
            std::vector<uint32_t> gold(2 * cores, 0);
            const uint32_t W = PKT / 4;
            auto assign = [&](uint32_t i, uint32_t& start, uint32_t& stride, uint32_t& n) {
                if (mode == 0) { const uint32_t per = npkt / cores; start = i * per; stride = 1; n = per; }
                else { const uint32_t b = (only_bank >= 0) ? (uint32_t)only_bank : (i % NB);
                       const uint32_t nb_cores = (only_bank >= 0) ? cores : (cores / NB);
                       const uint32_t rank = (only_bank >= 0) ? i : (i / NB);
                       const uint32_t tot_b = npkt / NB;                 // pages on bank b
                       const uint32_t per = tot_b / nb_cores;
                       start = b + (rank * per) * NB; stride = NB; n = per; } };
            for (uint32_t i = 0; i < cores; ++i) {
                uint32_t st, sd, n; assign(i, st, sd, n);
                for (uint32_t half = 0; half < (uint32_t)nocs; ++half) {
                    const uint32_t n_h = n / nocs, st_h = st + half * n_h * sd;
                    uint32_t acc = 0;
                    for (uint32_t j = 0; j < n_h; ++j) {
                        const uint64_t base = (uint64_t)(st_h + j * sd) * W;
                        acc ^= src_words[base] ^ src_words[base + W - 1]; }
                    gold[half * cores + i] = acc;
                    const uint32_t slot = half * cores + i;
                    SetRuntimeArgs(p, half ? k0 : k1, CoreCoord(wk[use[i]]),
                        {(uint32_t)src->address(), n_h, st_h, sd, (uint32_t)bOut->address(), slot}); }
            }
            Dp::MeshWorkload wl; wl.add_program(dr, std::move(p));
            Dp::EnqueueMeshWorkload(cq, wl, false); Dp::Finish(cq);          // rule 2: discard the JIT pass

            auto tid = mesh->begin_mesh_trace(0);
            Dp::EnqueueMeshWorkload(cq, wl, false);
            mesh->end_mesh_trace(0, tid); Dp::Finish(cq);
            double lo = 1e18, hi = 0;
            for (uint32_t r = 0; r < ITERS; ++r) {
                auto t0 = hrc::now(); mesh->replay_mesh_trace(0, tid, true);
                const double d = ms_of(t0); lo = std::min(lo, d); hi = std::max(hi, d); }
            mesh->release_mesh_trace(tid);

            std::vector<uint32_t> got; Dp::EnqueueReadMeshBuffer(cq, got, bOut, true);
            uint64_t bad = 0;
            for (uint32_t i = 0; i < (uint32_t)nocs * cores; ++i)
                if (got[(size_t)i * 1024] != gold[i]) ++bad;
            return {lo, (hi - lo) / lo * 100.0, bad};
        };

        auto make_src = [&](uint32_t PKT) {
            const uint32_t npkt = (uint32_t)(TOT / PKT);
            auto b = mkbuf((uint64_t)npkt * PKT, PKT);
            host.assign((size_t)npkt * PKT / 4, 0);
            std::mt19937 rng(1234); for (auto& v : host) v = rng();
            Dp::EnqueueWriteMeshBuffer(cq, b, host, false); Dp::Finish(cq);
            return std::make_pair(b, npkt); };

        FILE* csv = nullptr;
        if (const char* cp = std::getenv("DR1_CSV")) { csv = std::fopen(cp, "w");
            if (csv) std::fprintf(csv, "test,pkt,trid,cores,mode,nocs,ms,GBps,pct436,bad\n"); }
        auto report = [&](const char* tag, uint32_t PKT, uint32_t TRID, uint32_t cores, int mode,
                          uint32_t nocs, Res r, uint64_t bytes) {
            const double gbps = (double)bytes / (r.ms * 1e-3) / 1e9;
            fmt::print("  {:<10}{:>7}{:>6}{:>7}{:>9}{:>6}{:>10.4f}{:>8.1f}{:>10.1f}{:>8}\n",
                       tag, PKT, TRID, cores, mode == 0 ? "interlv" : (mode == 1 ? "pin" : "pinadj"),
                       nocs, r.ms, gbps, gbps / 436.0 * 100.0, r.bad ? "BAD" : "ok");
            if (r.bad) pass = false;
            if (csv) std::fprintf(csv, "%s,%u,%u,%u,%d,%u,%.5f,%.2f,%.2f,%llu\n",
                                  tag, PKT, TRID, cores, mode, nocs, r.ms, gbps,
                                  gbps / 436.0 * 100.0, (unsigned long long)r.bad);
            return gbps; };

        constexpr const char* HDR = "  {:<10}{:>7}{:>6}{:>7}{:>9}{:>6}{:>10}{:>8}{:>10}{:>8}\n";
        fmt::print(HDR, "test", "PKT", "TRID", "cores", "mode", "NoCs", "ms", "GB/s", "%of436", "chk");

        // ---- 1. packet size x in-flight depth, all cores, interleaved, one NoC ----
        double best = 0; uint32_t bestPkt = 4096, bestTrid = 2;
        for (uint32_t PKT : {4096u, 16384u}) {
            auto [src, npkt] = make_src(PKT);
            for (uint32_t TRID : {1u, 2u, 4u}) {
                if ((uint64_t)TRID * PKT > (64u << 10)) continue;        // keep L1 and host staging sane
                auto r = run(PKT, TRID, GFULL, 0, 1, -1, host, src, npkt);
                const double g = report("pkt x trid", PKT, TRID, GFULL, 0, 1, r, TOT);
                if (g > best && !r.bad) { best = g; bestPkt = PKT; bestTrid = TRID; } }
        }
        fmt::print("\n  best so far: PKT={} TRID={} -> {:.1f} GB/s\n\n", bestPkt, bestTrid, best);

        // ---- 2. placement: interleaved vs pinned vs pinned-and-adjacent ----
        { auto [src, npkt] = make_src(bestPkt);
          for (int mode : {0, 1, 2})
              for (uint32_t cores : {(uint32_t)(NB * 4), (uint32_t)(NB * 8), GFULL}) {
                  const uint32_t c = (mode == 0) ? cores : (cores / NB) * NB;   // whole banks
                  if (c == 0 || c > GFULL) continue;
                  auto r = run(bestPkt, bestTrid, c, mode, 1, -1, host, src, npkt);
                  report("placement", bestPkt, bestTrid, c, mode, 1, r, TOT); }
          fmt::print("\n");

          // ---- 3. both NoCs reading ----
          for (uint32_t nocs : {1u, 2u})
              for (int mode : {0, 2}) {
                  fmt::print("  [both-noc mode={} nocs={}]\n", mode, nocs); std::fflush(stdout);
                  const uint32_t c = (mode == 0) ? GFULL : (GFULL / NB) * NB;
                  auto r = run(bestPkt, bestTrid, c, mode, nocs, -1, host, src, npkt);
                  report("both-noc", bestPkt, bestTrid, c, mode, nocs, r, TOT); }
          fmt::print("\n");

          // ---- 4. per-bank verification: is any bank carrying less than the others? ----
          if (PERBANK) {
              fmt::print("  per-bank (4 adjacent readers each, one bank at a time):\n");
              for (int b = 0; b < (int)NB; ++b) {
                  auto r = run(bestPkt, bestTrid, 4, 2, 1, b, host, src, npkt);
                  const double gb = (double)(TOT / NB) / (r.ms * 1e-3) / 1e9;
                  fmt::print("    bank {}  {:.4f} ms  {:>6.1f} GB/s   {}\n", b, r.ms, gb,
                             r.bad ? "BAD" : "ok");
                  if (r.bad) pass = false; }
              fmt::print("\n"); }
        }

        if (csv) std::fclose(csv);
        fmt::print("  reference: docs/he/dram_guide.md — one channel 53.9, 8 summed 436.0,\n"
                   "             whole chip achieved 433 GB/s; one NoC only 234.8.\n\n");
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    fmt::print("{}\n", pass ? "Test Passed" : "Test FAILED");
    return pass ? 0 : 1;
}
