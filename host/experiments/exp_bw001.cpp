// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_Bw001 — how fast can Blackhole do a plain int32 ELEMENTWISE MULTIPLY, and what limits it?
//
// Workload:  C = A * B  over a (2^n, NC, 32, 32) int32 tensor.
//   * NC cores (default 112 of the p150's 130), one full 32x32 tile per core per iteration
//   * the 2^n iterations run SEQUENTIALLY INSIDE each core's kernel
//   * ONE dispatch covers the whole tensor -- see "one dispatch" below
//   * n is swept 0..NMAX so the scaling is visible, not asserted
//
// ONE DISPATCH.  Yes, the entire (2^n, NC, 32, 32) op is a single program launch, and this is the
// experiment's structural point.  Two things make it so:
//   (a) the 2^n sequential axis is a runtime arg, so the loop is inside the kernel -- the host does
//       not re-dispatch per iteration;
//   (b) the NC axis is a CoreRangeSet in ONE program with per-core runtime args, not NC programs.
//       Separate programs serialize on the single command queue no matter which core they target
//       (host/experiments/test_dispatch_micro.cpp regime D measures exactly this), so co-scheduling REQUIRES
//       one program with disjoint tile assignments. NC=112 is not a rectangle in a 13x10 grid, so
//       it is expressed as a full 13x8 block plus a partial row.
// The consequence is that dispatch cost is O(1) in the tensor size, and after tracing it is ~0.
//
// TWO REGIMES, because a single number cannot say what the limit is:
//   A  STREAM  every iteration reads a fresh (A,B) tile pair from DRAM and writes C  -> GB/s
//   B  L1      one tile pair is unpacked into DST once, then 2^n multiplies with no memory traffic
//              at all                                                                -> Gmul/s
// A is min(SFPU rate, DRAM bandwidth); B is the SFPU rate alone. The ratio is the roofline: it says
// how much of the multiply rate the memory system is actually costing at this arithmetic intensity
// (1 multiply per 12 bytes moved -- elementwise multiply is about as memory-bound as an op gets).
//
// OPTIMIZATIONS APPLIED (all of them measurable in the output):
//   * batched NOC reads/writes: BATCH iterations in flight, one barrier   (dataflow/*_stream.cpp)
//   * BATCH tiles per DST acquire, amortizing the tile_regs handshake     (compute/mul_i32_stream)
//   * deep CBs (4*BATCH slots) so reader / compute / writer pipeline
//   * reader and writer on separate RISCs and separate NOCs
//   * fp32_dest_acc_en + dst_full_sync_en -> 8 int32 tiles of DST, i.e. BATCH=4 pairs
//   * one program over all NC cores; persistent MeshWorkload (no rebuild per measurement)
//   * Metal Trace capture/replay, which is what makes the reported number a KERNEL time
//
// WHY TRACE.  An untraced enqueue wall = host serialization + dispatch + device execution. On this
// repo's PIR pipeline that serialization was ~84% of host time (docs/pir/latency_optimization.md).
// A traced replay re-runs pre-serialized commands, so a blocking replay's wall is device execution
// plus a fixed few-microsecond launch. Both are reported side by side: the gap IS the dispatch cost,
// and at large n the two converge, which is the check that the kernel time is real.
//
// Correctness is checked against a CPU golden -- the raw wrapping 32-bit product, since the kernel
// is bare `mul_int32` with no modular reduction.
//
//   ./build/exp_bw001 [NMAX=12] [NCORES=112] [W=64] [ITERS=5] [SWEEP_CORES=0]
//
// SWEEP_CORES=1 adds a core-count scaling table at a fixed n, which is what locates the bandwidth
// knee -- the n sweep can only tell you how fast NCORES cores are, not whether NCORES was enough.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>
#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
namespace Dp = tt::tt_metal::distributed;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using hrc = std::chrono::high_resolution_clock;
static double msd(hrc::time_point t) { return std::chrono::duration<double, std::milli>(hrc::now() - t).count(); }

namespace {
constexpr uint32_t ELEMS = 1024;          // 32x32
constexpr uint32_t I32 = ELEMS * 4;       // bytes per int32 tile
constexpr uint32_t DST_PAIRS = 4;         // DST holds 8 int32 tiles = 4 (a,b) pairs

// The first n cores of the grid in row-major order. 112 is not a rectangle in 13x10, so this is a
// full block of complete rows plus a partial row -- CoreRangeSet handles both uniformly.
CoreRangeSet first_n_cores(uint32_t n, CoreCoord grid) {
    std::vector<CoreRange> rs;
    const uint32_t full = n / grid.x, rem = n % grid.x;
    if (full) rs.emplace_back(CoreCoord{0, 0}, CoreCoord{grid.x - 1, full - 1});
    if (rem) rs.emplace_back(CoreCoord{0, full}, CoreCoord{rem - 1, full});
    return CoreRangeSet(rs);
}
CoreCoord core_at(uint32_t k, CoreCoord grid) { return CoreCoord{k % grid.x, k / grid.x}; }
}  // namespace

int main(int argc, char** argv) {
    const uint32_t NMAX = (argc > 1) ? std::stoul(argv[1]) : 12;      // sweep n = 0..NMAX
    const uint32_t NCMAX = (argc > 2) ? std::stoul(argv[2]) : 112;    // cores == tiles per iteration
    const uint32_t WMAX = (argc > 3) ? std::stoul(argv[3]) : 64;      // DRAM working set, in rows, at NCMAX
    const uint32_t ITERS = (argc > 4) ? std::stoul(argv[4]) : 5;      // timed repetitions
    const uint32_t SWEEP_CORES = (argc > 5) ? std::stoul(argv[5]) : 0;  // also sweep core count at fixed n

    // Mutable so the core-count sweep can re-target the same buffers at a narrower grid.
    //
    // W scales INVERSELY with NC on purpose, so W*NC -- the DRAM footprint actually streamed -- is
    // the same at every core count. Holding W fixed instead would give the small-core points a tiny
    // working set (at NC=1, W=64, just 256 KB), which re-reads the same few DRAM pages 2^n times and
    // reports a per-core bandwidth no real stream could reach. That inflated the low end of the
    // scaling curve badly enough to fake a knee.
    uint32_t NC = NCMAX;
    uint32_t W = WMAX;

    bool pass = true;
    try {
        // Trace replay needs its region reserved at device init; buffers must all be allocated
        // before any capture, because a trace bakes in buffer addresses.
        auto mesh = Dp::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 256u << 20);
        auto& cq = mesh->mesh_command_queue();
        Dp::MeshCoordinateRange dr(mesh->shape());
        auto grid = mesh->compute_with_storage_grid_size();

        if (NC > grid.x * grid.y) {
            fmt::print(stderr, "NCORES={} exceeds the compute grid {}x{} = {}\n", NC, grid.x, grid.y,
                       grid.x * grid.y);
            return 2;
        }
        auto cores = first_n_cores(NC, grid);
        const size_t NT = (size_t)WMAX * NCMAX;        // tiles per buffer, sized for the widest grid
        const double buf_mb = (double)NT * I32 / (1 << 20);

        fmt::print("\nexp_Bw001 — int32 elementwise multiply, (2^n, {}, 32, 32), ONE dispatch\n", NC);
        fmt::print("compute grid {}x{} = {} cores; using {} cores ({} core ranges), 1 tile/core/iter\n",
                   grid.x, grid.y, grid.x * grid.y, NC, cores.ranges().size());
        fmt::print("DRAM working set: W={} rows x {} tiles = {} tiles = {:.1f} MB per operand ({:.1f} MB total)\n",
                   WMAX, NCMAX, NT, buf_mb, 3 * buf_mb);
        fmt::print("sweep n = 0..{}, {} timed replays each\n", NMAX, ITERS);

        auto mkbuf = [&](size_t nt) {
            Dp::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return Dp::MeshBuffer::create(Dp::ReplicatedBufferConfig{.size = nt * I32}, d, mesh.get());
        };
        auto bA = mkbuf(NT), bB = mkbuf(NT), bC = mkbuf(NT), bCl1 = mkbuf(NCMAX);

        // ---- inputs + CPU golden. Elementwise is layout-agnostic (A, B and C all use the same tile
        // layout), so no tilize step is needed and the golden is a flat product. ----
        std::mt19937 rng(1234);
        std::vector<uint32_t> hA(NT * ELEMS), hB(NT * ELEMS), gold(NT * ELEMS);
        for (size_t i = 0; i < NT * ELEMS; ++i) {
            hA[i] = rng();
            hB[i] = rng();
            gold[i] = hA[i] * hB[i];   // wrapping low-32 product, exactly what mul_int32 computes
        }
        std::vector<uint32_t> zeros(NCMAX * ELEMS, 0);
        Dp::EnqueueWriteMeshBuffer(cq, bA, hA, false);
        Dp::EnqueueWriteMeshBuffer(cq, bB, hB, false);
        Dp::EnqueueWriteMeshBuffer(cq, bCl1, zeros, false);
        Dp::Finish(cq);

        auto mkcb = [&](Program& p, CBIndex idx, uint32_t slots) {
            CircularBufferConfig c(slots * I32, {{idx, tt::DataFormat::Int32}});
            c.set_page_size(idx, I32);
            CreateCircularBuffer(p, cores, c);
        };

        // ---- regime A: streaming program. n_iter and BATCH are runtime args, so every n reuses the
        // same JIT-compiled binaries; only the args (and the CB depth) differ. ----
        auto build_stream = [&](uint32_t n_iter, uint32_t BATCH) {
            Program p = CreateProgram();
            const uint32_t slots = 4 * BATCH;   // multiple of BATCH => a reserved run never straddles
            mkcb(p, CBIndex::c_0, slots);       //                      the CB wrap (see reader)
            mkcb(p, CBIndex::c_1, slots);
            mkcb(p, CBIndex::c_16, slots);

            std::vector<uint32_t> rc;
            TensorAccessorArgs(*bA).append_to(rc);
            TensorAccessorArgs(*bB).append_to(rc);
            std::vector<uint32_t> wc;
            TensorAccessorArgs(*bC).append_to(wc);

            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_mul_stream.cpp", cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                                   .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_mul_stream.cpp", cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                                   .compile_args = wc});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/mul_i32_stream.cpp", cores,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});

            for (uint32_t k = 0; k < NC; ++k) {
                CoreCoord c = core_at(k, grid);
                SetRuntimeArgs(p, rd, c, {(uint32_t)bA->address(), (uint32_t)bB->address(), n_iter, k, NC, W, BATCH});
                SetRuntimeArgs(p, wr, c, {(uint32_t)bC->address(), n_iter, k, NC, W, BATCH});
                SetRuntimeArgs(p, cp, c, {n_iter, BATCH});
            }
            Dp::MeshWorkload wl;
            wl.add_program(dr, std::move(p));
            return wl;
        };

        // ---- regime B: L1-resident. Same reader/writer, but they move exactly one tile each; the
        // 2^n loop is entirely inside the compute kernel with no memory traffic. ----
        auto build_l1 = [&](uint32_t n_iter) {
            Program p = CreateProgram();
            mkcb(p, CBIndex::c_0, 2);
            mkcb(p, CBIndex::c_1, 2);
            mkcb(p, CBIndex::c_16, 2);

            std::vector<uint32_t> rc;
            TensorAccessorArgs(*bA).append_to(rc);
            TensorAccessorArgs(*bB).append_to(rc);
            std::vector<uint32_t> wc;
            TensorAccessorArgs(*bCl1).append_to(wc);

            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_mul_stream.cpp", cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                                   .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_mul_stream.cpp", cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                                   .compile_args = wc});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/mul_i32_l1.cpp", cores,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});

            for (uint32_t k = 0; k < NC; ++k) {
                CoreCoord c = core_at(k, grid);
                SetRuntimeArgs(p, rd, c, {(uint32_t)bA->address(), (uint32_t)bB->address(), 1u, k, NC, 1u, 1u});
                SetRuntimeArgs(p, wr, c, {(uint32_t)bCl1->address(), 1u, k, NC, 1u, 1u});
                SetRuntimeArgs(p, cp, c, {n_iter});
            }
            Dp::MeshWorkload wl;
            wl.add_program(dr, std::move(p));
            return wl;
        };

        // min-of-ITERS for an untraced enqueue (host serialization + dispatch + device).
        auto time_enq = [&](Dp::MeshWorkload& wl) {
            double mn = 1e18;
            for (uint32_t i = 0; i < ITERS; ++i) {
                auto t0 = hrc::now();
                Dp::EnqueueMeshWorkload(cq, wl, false);
                Dp::Finish(cq);
                mn = std::min(mn, msd(t0));
            }
            return mn;
        };
        // min-of-ITERS for a blocking traced replay: pre-serialized commands, so this is the kernel.
        struct Stat { double mn, mean, mx; };
        auto time_trace = [&](Dp::MeshTraceId tid) {
            mesh->replay_mesh_trace(0, tid, true);   // warm
            Stat s{1e18, 0, 0};
            for (uint32_t i = 0; i < ITERS; ++i) {
                auto t0 = hrc::now();
                mesh->replay_mesh_trace(0, tid, true);
                double d = msd(t0);
                s.mn = std::min(s.mn, d);
                s.mean += d;
                s.mx = std::max(s.mx, d);
            }
            s.mean /= ITERS;
            return s;
        };

        // Measure ONE workload end to end, then tear it down completely before the next is built.
        //
        // This ordering is not incidental. A program's circular buffers ARE L1 allocations, so
        // holding all 2*(NMAX+1) programs alive at once would ask ~2.4 MB/core of a 1.46 MB L1; and
        // allocating anything while a trace is alive is unsafe, because a trace bakes in addresses
        // ("Allocating device buffers is unsafe due to the existence of an active trace"). Building,
        // capturing, replaying and then RELEASING each trace before the next build keeps exactly one
        // program resident and leaves no trace active across an allocation.
        //
        // Within one workload the order is forced too: the first enqueue writes the kernel binaries,
        // and trace capture forbids writes, so the warm enqueue must precede the capture.
        auto measure = [&](Dp::MeshWorkload& wl, double& t_enq, Stat& t_tr, const std::function<uint64_t()>& check) {
            Dp::EnqueueMeshWorkload(cq, wl, false);
            Dp::Finish(cq);
            const uint64_t mism = check();
            t_enq = time_enq(wl);
            auto tid = mesh->begin_mesh_trace(0);
            Dp::EnqueueMeshWorkload(cq, wl, false);   // capture RECORDS but does not EXECUTE
            mesh->end_mesh_trace(0, tid);
            Dp::Finish(cq);
            t_tr = time_trace(tid);
            mesh->release_mesh_trace(tid);
            return mism;
        };

        struct Row { uint32_t n; uint64_t R; double enq, tr_mn, tr_mean, tr_mx, l1_enq, l1_mn; uint64_t mism_s, mism_l; };
        std::vector<Row> rows;

        // One (n, NC) configuration, both regimes. NC and `cores` are read from the enclosing scope,
        // so the core-count sweep just reassigns them and calls this again.
        auto run_config = [&](uint32_t n) {
            const uint64_t R = 1ull << n;
            const uint32_t BATCH = (uint32_t)std::min<uint64_t>(DST_PAIRS, R);   // R is 2^n, so BATCH | R
            Row row{n, R, 0, 0, 0, 0, 0, 0, 0, 0};
            Stat sS{}, sL{};

            {   // ---- A: streaming ----
                auto wlS = build_stream((uint32_t)R, BATCH);
                row.mism_s = measure(wlS, row.enq, sS, [&]() -> uint64_t {
                    std::vector<uint32_t> got;
                    Dp::EnqueueReadMeshBuffer(cq, got, bC, true);
                    const uint64_t chk = std::min<uint64_t>(R, W) * NC;   // rows beyond R were never written
                    uint64_t m = 0;
                    for (uint64_t t = 0; t < chk; ++t)
                        for (uint32_t e = 0; e < ELEMS; ++e)
                            if (got[t * ELEMS + e] != gold[t * ELEMS + e]) ++m;
                    return m;
                });
                row.tr_mn = sS.mn; row.tr_mean = sS.mean; row.tr_mx = sS.mx;
            }
            {   // ---- B: L1-resident ----
                auto wlL = build_l1((uint32_t)R);
                row.mism_l = measure(wlL, row.l1_enq, sL, [&]() -> uint64_t {
                    std::vector<uint32_t> got;
                    Dp::EnqueueReadMeshBuffer(cq, got, bCl1, true);
                    uint64_t m = 0;
                    for (uint64_t t = 0; t < NC; ++t)
                        for (uint32_t e = 0; e < ELEMS; ++e)
                            if (got[t * ELEMS + e] != gold[t * ELEMS + e]) ++m;
                    return m;
                });
                row.l1_mn = sL.mn;
            }

            pass &= (row.mism_s == 0 && row.mism_l == 0);
            return row;
        };

        for (uint32_t n = 0; n <= NMAX; ++n) {
            rows.push_back(run_config(n));
            auto& r = rows.back();
            fmt::print("  n={:2}  stream {:9.3f} ms   L1 {:9.3f} ms   mism {}/{}\n",
                       r.n, r.tr_mn, r.l1_mn, r.mism_s, r.mism_l);
        }

        // ---- core-count sweep at a fixed n, to locate the bandwidth knee ----
        // The n sweep says how fast NC cores go; it cannot say whether NC is the right NC. This runs
        // the same configuration over a range of core counts so the saturation point is visible: if
        // GB/s is flat from some C upward, every core past C is idle bandwidth-wise.
        struct CRow { uint32_t nc; double tr, l1; uint64_t mism; };
        std::vector<CRow> crows;
        if (SWEEP_CORES) {
            const uint32_t nfix = std::min(NMAX, 10u);   // deep enough to be past the trace-launch floor
            fmt::print("\ncore-count sweep at n={} ({} iterations/core)\n", nfix, 1u << nfix);
            for (uint32_t c : {1u, 2u, 4u, 8u, 16u, 32u, 56u, 64u, 96u, 104u, 112u, 120u, 130u}) {
                if (c > NCMAX) continue;
                NC = c;
                cores = first_n_cores(NC, grid);
                W = (uint32_t)(NT / NC);   // constant DRAM footprint across the sweep (see above)
                auto r = run_config(nfix);
                crows.push_back({c, r.tr_mn, r.l1_mn, r.mism_s + r.mism_l});
                fmt::print("  cores={:3}  stream {:9.3f} ms   L1 {:9.3f} ms   mism {}\n",
                           c, r.tr_mn, r.l1_mn, r.mism_s + r.mism_l);
            }
            NC = NCMAX;
            cores = first_n_cores(NC, grid);
            W = WMAX;
        }

        // ---------------- report ----------------
        const double GRID_MUL = (double)NC * ELEMS;   // multiplies per sequential iteration, all cores

        fmt::print("\nA — STREAMING: every iteration reads a fresh (A,B) tile pair from DRAM, writes C\n");
        fmt::print("  n |     2^n |    tiles |    bytes | enqueue ms | trace ms (min/mean/max)    |   GB/s |  Gmul/s | mism\n");
        fmt::print("----+---------+----------+----------+------------+----------------------------+--------+---------+------\n");
        for (auto& r : rows) {
            const double tiles = (double)r.R * NC;
            const double bytes = tiles * I32 * 3.0;               // read A + read B + write C
            const double s = r.tr_mn / 1e3;
            fmt::print("  {:2}| {:8}| {:9.0f}| {:7.1f}MB| {:11.4f}| {:8.4f} /{:8.4f} /{:8.4f} | {:7.1f}| {:8.2f}| {:5}\n",
                       r.n, r.R, tiles, bytes / (1 << 20), r.enq, r.tr_mn, r.tr_mean, r.tr_mx,
                       bytes / s / 1e9, (double)r.R * GRID_MUL / s / 1e9, r.mism_s);
        }

        fmt::print("\nB — L1-RESIDENT: one tile pair per core in DST, 2^n multiplies, zero memory traffic\n");
        fmt::print("  n |     2^n |  enqueue ms |   trace ms |  Gmul/s | mul/cycle/core | mism\n");
        fmt::print("----+---------+-------------+------------+---------+----------------+------\n");
        const double CLK = 1.35e9;   // Blackhole Tensix clock, for the per-core issue rate
        for (auto& r : rows) {
            const double s = r.l1_mn / 1e3;
            const double gmul = (double)r.R * GRID_MUL / s;
            fmt::print("  {:2}| {:8}| {:12.4f}| {:11.4f}| {:8.2f}| {:15.1f}| {:5}\n",
                       r.n, r.R, r.l1_enq, r.l1_mn, gmul / 1e9, gmul / NC / CLK, r.mism_l);
        }

        if (!crows.empty()) {
            fmt::print("\nC — CORE SCALING at n={} (streaming; each core still does 1 tile/iteration)\n",
                       std::min(NMAX, 10u));
            fmt::print(" cores |  trace ms |   GB/s | GB/s/core | scaling | ideal |  L1 Gmul/s | mism\n");
            fmt::print("-------+-----------+--------+-----------+---------+-------+------------+------\n");
            const double R0 = (double)(1u << std::min(NMAX, 10u));
            const double gbs0 = R0 * crows[0].nc * I32 * 3.0 / (crows[0].tr / 1e3) / 1e9;
            for (auto& c : crows) {
                // Both axes are THROUGHPUT: more cores also means more tiles per iteration, so a time
                // ratio would read as a slowdown. `scaling` is GB/s relative to one core; `ideal` is
                // the core count, i.e. what perfect scaling would give.
                const double gbs = R0 * c.nc * I32 * 3.0 / (c.tr / 1e3) / 1e9;
                fmt::print(" {:6}| {:10.4f}| {:7.1f}| {:10.2f}| {:6.1f}x| {:4}x| {:11.2f}| {:5}\n",
                           c.nc, c.tr, gbs, gbs / c.nc, gbs / gbs0, c.nc,
                           R0 * c.nc * ELEMS / (c.l1 / 1e3) / 1e9, c.mism);
            }
        }

        if (!rows.empty()) {
            auto& t = rows.back();
            const double ss = t.tr_mn / 1e3, sl = t.l1_mn / 1e3;
            const double stream_g = (double)t.R * GRID_MUL / ss / 1e9;
            const double l1_g = (double)t.R * GRID_MUL / sl / 1e9;
            fmt::print("\nROOFLINE at n={} ({} sequential iterations, {} cores):\n", t.n, t.R, NC);
            fmt::print("  SFPU ceiling (L1-resident)      {:8.2f} Gmul/s ({:.1f} mul/cycle/core)\n",
                       l1_g, l1_g * 1e9 / NC / CLK);
            fmt::print("  achieved streaming from DRAM    {:8.2f} Gmul/s  = {:.1f}% of ceiling\n",
                       stream_g, 100.0 * stream_g / l1_g);
            fmt::print("  DRAM traffic to sustain it      {:8.1f} GB/s   (12 B moved per multiply)\n",
                       (double)t.R * NC * I32 * 3.0 / ss / 1e9);

            // Dispatch cost is a CONSTANT (~tens of microseconds), so it is only measurable where the
            // kernel is short. Quoting it at the largest n would report run-to-run jitter instead --
            // there the honest statement is that the two have converged.
            const auto& sm = rows[std::min<size_t>(2, rows.size() - 1)];
            fmt::print("  dispatch cost, removed by trace {:8.4f} ms    (at n={}: enqueue {:.4f} -> trace {:.4f})\n",
                       sm.enq - sm.tr_mn, sm.n, sm.enq, sm.tr_mn);
            fmt::print("  ...and at n={} it is already lost in the noise: enqueue {:.4f} vs trace {:.4f} ms,\n"
                       "     i.e. the traced wall IS the kernel time.\n", t.n, t.enq, t.tr_mn);
        }

        fmt::print("\nEvery row above is ONE dispatch: the 2^n axis is a kernel-internal loop and the\n"
                   "{}-core axis is a CoreRangeSet in a single program.\n", NC);

        pass &= mesh->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Exception:\n{}\n", e.what());
        return 2;
    }
    if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
    return pass ? 0 : 1;
}
