// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_roofline001 — a roofline for the three arithmetic primitives this repo actually issues, on
// Blackhole p150: the Montgomery modular multiply, a bare multiply, and a non-multiply op.
//
// THE SIX POINTS ASKED FOR, and the curve between them.  Each op is measured at both ends of the
// roofline, and the sweep that connects them is the same kernel with one runtime argument changed:
//
//   R = 1      read A, read B, do ONE op, write C.  No data reuse at all -- about the worst
//              arithmetic intensity an operation can have, and the regime every streaming HE kernel
//              is actually in.
//   R large    read A, read B, do R ops on the SAME operands in DST, write C.  The memory traffic is
//              amortised to nothing, so what is left is the SFPU's throughput on that op: the
//              ARITHMETIC BOUND, "data always reused".
//
//   arithmetic intensity = R * 1024 ops / (3 * 4096 bytes) = R/12 ops/byte
//
// so R is literally the x-axis. One kernel per op, one code path, nothing swapped between the two
// regimes -- which is what makes the two ends comparable.
//
// THE MEMORY SIDE IS FULLY OPTIMISED ON PURPOSE.  A roofline is only honest if the memory ceiling is
// the machine's and not the harness's, so every rule in docs/he/dram_guide.md is applied:
// page = tile so a core's run walks all EIGHT banks (bank = page % 8); CHUNK=8 tile-pairs in flight
// behind one barrier (the depth rule); reads on NCRISC/NOC_1 and writes on BRISC/NOC_0 (both NoCs);
// one dispatch over G cores, not G dispatches.  ROOF_SWEEP_CORES=1 sweeps G to show the ceiling is
// reached rather than asserted.
//
// PROFILING DISCIPLINE (docs/guide/profiling.md):
//   * rule 2 -- the first execution is discarded; it JIT-compiles.
//   * rule 3 -- never time a single pass. Every point is the MIN of ITERS traced replays, and the
//     spread is printed beside it so the detection floor is visible rather than assumed.
//   * Metal Trace, so host enqueue (~30 us) is outside the timed region. At R=1 a point is ~1 ms, so
//     an untraced enqueue would be a ~3% error on exactly the memory-bound point that matters most.
//   * rule 1 -- the metric is named: device time for ONE traced replay of the whole T-tile workload.
//   * every point is gated bit-exact against a host golden. Because the R loop recomputes the same
//     result from the same operands, the golden does NOT depend on R -- so correctness is checked at
//     all 21 points, not just the ends.
//   * the linearity guard: time(R) = fixed + R*slope is FITTED per op and the max residual printed.
//     If the compiler had collapsed the R identical passes, the fit would break; a high ceiling with
//     a bad fit is not quotable. The slope is also how the arithmetic bound is reported, since it
//     removes the memory/launch term instead of hoping it is small.
//
//   ./build/exp_roofline001
//   ROOF_SWEEP_CORES=1 ./build/exp_roofline001
//
// env: ROOF_TILES (default 32768), ROOF_CORES (default full grid), ROOF_ITERS (5), ROOF_CSV

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
double ms_of(hrc::time_point t) { return std::chrono::duration<double, std::milli>(hrc::now() - t).count(); }
uint32_t envu(const char* k, uint32_t d) { const char* e = std::getenv(k); return e ? (uint32_t)std::atoi(e) : d; }
bool is_prime(uint64_t n) { if (n < 2 || (n % 2 == 0 && n != 2)) return false;
    for (uint64_t d = 3; d * d <= n; d += 2) if (n % d == 0) return false; return true; }

struct Op { const char* name; const char* kernel; const char* what; };
const std::vector<Op> OPS = {
    {"modmul", "compute/roof_modmul.cpp", "Montgomery modular multiply (11 mul16 = 22 sfpmul24)"},
    {"mul",    "compute/roof_mul.cpp",    "32x32 -> low 32 multiply  ( 3 mul16 =  6 sfpmul24)"},
    {"add",    "compute/roof_add.cpp",    "modular add               ( 0 multiplies)"},
    {"addp",   "compute/roof_addp.cpp",   "PLAIN 32-bit add          ( 0 multiplies, no reduce)"},
};

struct Pt { uint32_t R; double ms, spread; uint64_t mism; };
}  // namespace

int main(int argc, char** argv) {
    const uint32_t T     = envu("ROOF_TILES", 32768);
    const uint32_t ITERS = envu("ROOF_ITERS", 5);
    const bool SWEEP     = envu("ROOF_SWEEP_CORES", 0) != 0;
    const bool DSWEEP    = envu("ROOF_SWEEP_DEPTH", 0) != 0;
    const uint32_t CHUNK0= envu("ROOF_CHUNK", 8);
    const bool NOWRITE   = envu("ROOF_NOWRITE", 0) != 0;   // probe: reads only, WRONG answer
    // ROOF_SPLIT=1: put operand B's reads on BRISC/NOC_0 alongside the write, so each NoC carries
    // 4 B/op inbound instead of 8 on one and 0 on the other. See §5b -- the default feed's memory roof
    // is the one-NoC wall, not the machine's.
    const bool SPLIT     = envu("ROOF_SPLIT", 0) != 0;
    const std::vector<uint32_t> RS = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    uint64_t q = (1ull << 28) - 1; while (!(q & 1) || !is_prime(q)) --q;
    const mont::MontgomeryContext C((uint32_t)q);

    bool pass = true;
    try {
        auto mesh = Dp::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 64u << 20);
        auto& cq = mesh->mesh_command_queue();
        auto dr = Dp::MeshCoordinateRange(mesh->shape());
        auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX = grid.x, GFULL = grid.x * grid.y;
        const uint32_t G = std::min(envu("ROOF_CORES", GFULL), GFULL);

        auto CC = [&](uint32_t i){ return CoreCoord{i % GX, i / GX}; };
        auto crs_of = [&](uint32_t n){ std::set<CoreRange> s;
            for (uint32_t i = 0; i < n; ++i) s.insert(CoreRange(CC(i))); return CoreRangeSet(s); };
        auto mkbuf = [&](uint32_t nt){
            Dp::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return Dp::MeshBuffer::create(Dp::ReplicatedBufferConfig{.size = (uint64_t)nt * I32}, d, mesh.get()); };

        fmt::print("\nexp_roofline001 — roofline for modmul / mul / add on p150\n");
        fmt::print("  q = {} (28-bit), {} tiles = {} elements, {:.0f} MB moved per pass, G = {} of {} cores\n",
                   q, T, (uint64_t)T * ELEMS, (double)T * 3 * I32 / 1e6, G, GFULL);
        fmt::print("  ops/pass = R x {} ; bytes/pass = {} ; arithmetic intensity = R/12 ops/byte\n\n",
                   (uint64_t)T * ELEMS, (uint64_t)T * 3 * I32);

        auto bA = mkbuf(T), bB = mkbuf(T), bC = mkbuf(T);
        std::vector<uint32_t> hA((size_t)T * ELEMS), hB((size_t)T * ELEMS), got;
        { std::mt19937 rng(20260818);
          for (auto& v : hA) v = (uint32_t)(rng() % q);
          for (auto& v : hB) v = (uint32_t)(rng() % q); }
        Dp::EnqueueWriteMeshBuffer(cq, bA, hA, false);
        Dp::EnqueueWriteMeshBuffer(cq, bB, hB, false);
        Dp::Finish(cq);

        // goldens: independent of R, because the R loop recomputes the same value
        std::vector<std::vector<uint32_t>> gold(OPS.size(), std::vector<uint32_t>((size_t)T * ELEMS));
        for (size_t i = 0; i < hA.size(); ++i) {
            gold[0][i] = C.mulmod(hA[i], hB[i]);                                  // modmul
            gold[1][i] = (uint32_t)((uint64_t)hA[i] * hB[i]);                      // low-32 multiply
            gold[2][i] = (uint32_t)(((uint64_t)hA[i] + hB[i]) % q);                // modular add
            gold[3][i] = (uint32_t)(hA[i] + hB[i]);                                // plain add (wraps)
        }

        // ---- one traced measurement of (op, R, cores) ----
        uint32_t CHUNK = CHUNK0;
        auto run = [&](const Op& op, uint32_t R, uint32_t cores, uint64_t* mism) -> std::pair<double,double> {
            const uint32_t per = (T + cores - 1) / cores;
            Program p = CreateProgram(); auto crs = crs_of(cores);
            auto mkcb = [&](CBIndex i, uint32_t depth){
                CircularBufferConfig c(depth * I32, {{i, tt::DataFormat::UInt32}});
                c.set_page_size(i, I32); CreateCircularBuffer(p, crs, c); };
            mkcb(CBIndex::c_0, 2*CHUNK); mkcb(CBIndex::c_1, 2*CHUNK); mkcb(CBIndex::c_16, 2*CHUNK);
            auto rd = CreateKernel(p, SPLIT ? OVERRIDE_KERNEL_PREFIX "dataflow/reader_a_deep.cpp"
                                            : OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab_deep.cpp", crs,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                                   .compile_args = {CHUNK}});
            auto wr = CreateKernel(p, SPLIT ? OVERRIDE_KERNEL_PREFIX "dataflow/rw_b_deep.cpp"
                                   : (NOWRITE ? OVERRIDE_KERNEL_PREFIX "dataflow/writer_probe.cpp"
                                              : OVERRIDE_KERNEL_PREFIX "dataflow/writer_deep.cpp"), crs,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                                   .compile_args = {CHUNK}});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + std::string(op.kernel), crs,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            for (uint32_t c = 0; c < cores; ++c) {
                const uint32_t st = c * per; if (st >= T) continue;
                const uint32_t n = std::min(per, T - st);
                if (SPLIT) {
                    SetRuntimeArgs(p, rd, CC(c), {(uint32_t)bA->address(), n, st});
                    SetRuntimeArgs(p, wr, CC(c), {(uint32_t)bB->address(), n, st, (uint32_t)bC->address()});
                } else {
                    SetRuntimeArgs(p, rd, CC(c), {(uint32_t)bA->address(), (uint32_t)bB->address(), n, st});
                    SetRuntimeArgs(p, wr, CC(c), {(uint32_t)bC->address(), n, st});
                }
                SetRuntimeArgs(p, cp, CC(c), {n, R, (uint32_t)q, C.q_inv_32(),
                                              (uint32_t)(q & 0xFFFF), (uint32_t)(q >> 16)});
            }
            Dp::MeshWorkload wl; wl.add_program(dr, std::move(p));

            // rule 2: the first execution JIT-compiles -- run it and throw it away.
            Dp::EnqueueMeshWorkload(cq, wl, false); Dp::Finish(cq);

            auto tid = mesh->begin_mesh_trace(0);
            Dp::EnqueueMeshWorkload(cq, wl, false);
            mesh->end_mesh_trace(0, tid);
            Dp::Finish(cq);

            double lo = 1e18, hi = 0;                       // rule 3: ITERS replays, min AND spread
            for (uint32_t i = 0; i < ITERS; ++i) {
                auto t0 = hrc::now();
                mesh->replay_mesh_trace(0, tid, true);
                const double d = ms_of(t0);
                lo = std::min(lo, d); hi = std::max(hi, d);
            }
            mesh->release_mesh_trace(tid);

            if (mism) { Dp::EnqueueReadMeshBuffer(cq, got, bC, true);
                const auto& g = gold[(&op - &OPS[0])];
                uint64_t m = 0;
                for (size_t e = 0; e < g.size(); ++e) if (got[e] != g[e]) ++m;
                *mism = m; }
            return {lo, (hi - lo) / lo * 100.0};
        };

        // ================= the sweep =================
        std::vector<std::vector<Pt>> all(OPS.size());
        std::vector<double> ceilings(OPS.size(), 0.0);
        FILE* csv = nullptr;
        if (const char* cp = std::getenv("ROOF_CSV")) { csv = std::fopen(cp, "w");
            if (csv) std::fprintf(csv, "op,R,ai_ops_per_byte,ms,spread_pct,Gops,GBps,mismatches\n"); }

        for (size_t oi = 0; oi < OPS.size(); ++oi) {
            const Op& op = OPS[oi];
            fmt::print("=== {} — {} ===\n", op.name, op.what);
            fmt::print("  {:>4}{:>12}{:>11}{:>9}{:>12}{:>10}{:>10}\n",
                       "R", "AI op/B", "ms", "spread%", "Gop/s", "GB/s", "mism");
            for (uint32_t R : RS) {
                uint64_t m = 0;
                auto [ms, sp] = run(op, R, G, &m);
                const double ops = (double)T * ELEMS * R, bytes = (double)T * 3 * I32;
                const double gops = ops / (ms * 1e-3) / 1e9, gbps = bytes / (ms * 1e-3) / 1e9;
                fmt::print("  {:>4}{:>12.3f}{:>11.4f}{:>9.1f}{:>12.2f}{:>10.1f}{:>10}\n",
                           R, R / 12.0, ms, sp, gops, gbps, m);
                if (m) pass = false;
                all[oi].push_back({R, ms, sp, m});
                if (csv) std::fprintf(csv, "%s,%u,%.4f,%.5f,%.2f,%.3f,%.2f,%llu\n",
                                      op.name, R, R / 12.0, ms, sp, gops, gbps, (unsigned long long)m);
            }
            // ---- the arithmetic bound ----
            // The MARGINAL cost between the two largest R is what to quote: both points are deep in
            // the compute-bound regime, so differencing them cancels the fixed term AND any residual
            // memory time. A least-squares fit over the WHOLE sweep would be wrong here -- at low R
            // the op is memory-bound, so ms(R) is max(memory, compute), not a line, and including
            // those points tilts the slope. (Fitting all nine gave `add` 706 Gop/s against a true
            // ~529, with an 8.3% residual that was the tell.)
            const size_t L = all[oi].size();
            const double dR = (double)(all[oi][L-1].R - all[oi][L-2].R);
            const double slope = (all[oi][L-1].ms - all[oi][L-2].ms) / dR;
            const double ceil_gops = (double)T * ELEMS / (slope * 1e-3) / 1e9;
            // linearity guard: the same slope taken one step earlier must agree, or the R loop is
            // not doing R times the work (e.g. the compiler collapsed it) and nothing here is quotable.
            const double slope_prev = (all[oi][L-2].ms - all[oi][L-3].ms)
                                    / (double)(all[oi][L-2].R - all[oi][L-3].R);
            fmt::print("  marginal ms/R over R={}..{}: {:.5f}   (previous step {:.5f}, {:+.2f}%)\n",
                       all[oi][L-2].R, all[oi][L-1].R, slope, slope_prev,
                       (slope / slope_prev - 1.0) * 100.0);
            fmt::print("  -> ARITHMETIC BOUND {:.2f} Gop/s  ({:.3f} ns per op per chip, {} cores)\n\n",
                       ceil_gops, 1e9 / (ceil_gops * 1e9) * 1e9, G);
            ceilings[oi] = ceil_gops;
        }

        if (DSWEEP) {  // does read DEPTH set the memory roof? dram_guide.md §3.3 says a core needs
                       // ~192-384 pages outstanding; CHUNK tile-pairs is 2*CHUNK pages.
            fmt::print("=== read-depth sweep at R=1, op = add (the memory-bound point) ===\n");
            fmt::print("  {:>7}{:>9}{:>11}{:>10}{:>12}\n", "CHUNK", "pages", "ms", "GB/s", "% of 436");
            for (uint32_t c : {4u, 8u, 16u, 32u, 64u}) {
                CHUNK = c;
                uint64_t m = 0; auto [ms, sp] = run(OPS[2], 1, G, &m);
                const double gbps = (double)T * 3 * I32 / (ms * 1e-3) / 1e9;
                fmt::print("  {:>7}{:>9}{:>11.4f}{:>10.1f}{:>12.1f}{}\n",
                           c, 2 * c, ms, gbps, gbps / 436.0 * 100.0, m ? "  MISMATCH" : "");
                if (m) pass = false;
            }
            CHUNK = CHUNK0;
            fmt::print("\n");
        }
        if (SWEEP) {   // shows the DRAM ceiling is reached, not asserted (R=1 = the memory-bound end)
            fmt::print("=== core sweep at R=1 (the memory-bound point), op = add ===\n");
            fmt::print("  {:>6}{:>11}{:>10}{:>12}\n", "cores", "ms", "GB/s", "GB/s/core");
            for (uint32_t c : {1u, 2u, 4u, 8u, 16u, 32u, 64u, G}) {
                if (c > G) continue;
                auto [ms, sp] = run(OPS[2], 1, c, nullptr);
                const double gbps = (double)T * 3 * I32 / (ms * 1e-3) / 1e9;
                fmt::print("  {:>6}{:>11.4f}{:>10.1f}{:>12.2f}\n", c, ms, gbps, gbps / c);
            }
            fmt::print("\n");
        }

        // ---- the roofline itself ----
        double bw = 0;                                  // the memory ceiling, measured here
        for (auto& v : all) for (auto& pt : v)
            bw = std::max(bw, (double)T * 3 * I32 / (pt.ms * 1e-3) / 1e9);
        fmt::print("=== THE ROOFLINE ===\n");
        fmt::print("  memory ceiling (measured here, best of all points): {:.1f} GB/s\n", bw);
        fmt::print("  at R=1 the intensity is 1/12 op/byte, so the memory roof allows "
                   "{:.2f} Gop/s\n\n", bw / 12.0);
        fmt::print("  {:<9}{:>14}{:>13}{:>11}{:>11}{:>10}\n",
                   "op", "arith bound", "R=1 (no", "% of mem", "% of its", "ridge");
        fmt::print("  {:<9}{:>14}{:>13}{:>11}{:>11}{:>10}\n",
                   "", "Gop/s", "reuse)", "roof", "own roof", "R*");
        for (size_t oi = 0; oi < OPS.size(); ++oi) {
            const double g1 = (double)T * ELEMS / (all[oi][0].ms * 1e-3) / 1e9;
            fmt::print("  {:<9}{:>14.2f}{:>13.2f}{:>11.1f}{:>11.1f}{:>10.1f}\n",
                       OPS[oi].name, ceilings[oi], g1, g1 / (bw / 12.0) * 100.0,
                       g1 / ceilings[oi] * 100.0, 12.0 * ceilings[oi] / bw);
        }
        fmt::print("\n  ridge R* is where compute time equals memory time: below it the op is\n"
                   "  MEMORY bound, above it ARITHMETIC bound.\n\n");
        if (csv) std::fclose(csv);
        fmt::print("Reference ceilings from docs/vector/exp_bw001.md [M-p150b]:\n"
                   "  SFPU (mul_int_tile, L1-resident) 517.5 Gmul/s   DRAM streaming 322.3 GB/s\n\n");
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    fmt::print("{}\n", pass ? "Test Passed" : "Test FAILED");
    return pass ? 0 : 1;
}
