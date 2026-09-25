// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ctmul_fast_kernels — the optimised ct_mul kernels, on device, verified, side by side with the
// kernels they replace.  See kernels/OPTIMISED.md.
//
// exp_Tg018m took a CKKS ct_mul (L_Q=48, dnum=3, N=65536) from 238.663 ms to 50.920 ms on a
// Blackhole p150, bit-identical throughout.  Most of that is scheduling and memory layout specific
// to that pipeline, but three COMPUTE-kernel techniques came out of it that apply to any SFPU kernel
// in this repo.  This example runs the kernels that carry them, against the originals, on the same
// data, so the win is reproducible in about twenty seconds rather than inferred from a document:
//
//   1. sfpmul24 instead of the fp32 byte-split       (Blackhole has an integer multiply)
//   2. collapse SFPI passes, keep intermediates in registers  (only possible because of 1)
//   3. broadcast scalars in register, not via fill_tile_int
//
// It also uses the DRAM feed rule from docs/he/dram_guide.md -- every read of a batch in flight
// before the barrier, consecutive page ids -- via reader_seq_batch / writer_seq_batch.
//
// WHAT IS AND IS NOT SHOWN.  This runs the modular multiply that dominates ct_mul, not a whole
// ct_mul: the full pipeline needs the NTT, the corner turn, the base conversion and their
// scheduling, which is host/experiments/exp_tg018m.cpp (~1800 lines) and is the reference implementation.
// What is shown here is the part you would reuse in your own kernel.
//
//   ./build/ctmul_fast_kernels [T=32] [ITERS=5]
//
// Every variant is checked against the same host golden; a variant that is fast because it computes
// something else cannot be reported as a win.
//
// TIMING METHOD.  Each variant is run at TWO tile counts and the per-tile cost is taken as the
// MARGINAL cost between them, (ms_T - ms_T/4) / (T - T/4).  Dividing a single wall-clock by T would
// fold in the per-dispatch overhead -- ~68 us per replay here -- which inflates both kernels and the
// FASTER one proportionally more, reporting 3.1x for a 4.0x kernel.  The marginal cost cancels the
// fixed term and reproduces the device-side figure in docs/he/exp_modmul001.md without needing the
// profiler.  Both raw wall-clocks are printed so the fixed cost is visible rather than hidden.

#include <algorithm>
#include <chrono>
#include <cstdint>
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
double ms(hrc::time_point t) { return std::chrono::duration<double, std::milli>(hrc::now() - t).count(); }
bool is_prime(uint64_t n) { if (n < 2 || (n % 2 == 0 && n != 2)) return false;
    for (uint64_t d = 3; d * d <= n; d += 2) if (n % d == 0) return false; return true; }
}  // namespace

int main(int argc, char** argv) {
    const uint32_t T = (argc > 1) ? (uint32_t)std::stoul(argv[1]) : 32;      // tiles per dispatch
    const uint32_t ITERS = (argc > 2) ? (uint32_t)std::stoul(argv[2]) : 5;

    uint64_t q = (1ull << 28) - 1; while (!(q & 1) || !is_prime(q)) --q;
    const mont::MontgomeryContext C((uint32_t)q);

    bool pass = true;
    try {
        auto mesh = Dp::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE);
        auto& cq = mesh->mesh_command_queue();
        auto dr = Dp::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        const CoreRangeSet crs(std::set<CoreRange>{CoreRange(core)});

        fmt::print("\nctmul_fast_kernels — the optimised ct_mul kernels  (kernels/OPTIMISED.md)\n");
        fmt::print("one core, q = {} (28-bit), {} tiles/dispatch, {} timed replays, min taken\n\n", q, T, ITERS);

        auto mkbuf = [&](uint64_t n) {
            Dp::DeviceLocalBufferConfig d{.page_size = I32, .buffer_type = BufferType::DRAM};
            return Dp::MeshBuffer::create(Dp::ReplicatedBufferConfig{.size = n * I32}, d, mesh.get()); };
        auto bA = mkbuf(T), bB = mkbuf(T), bO = mkbuf(T);

        // Montgomery-domain operands in [0, q); golden is the canonical Montgomery product.
        std::vector<uint32_t> hA(T * ELEMS), hB(T * ELEMS), gold(T * ELEMS), got;
        { std::mt19937 rng(20260817);
          for (auto& v : hA) v = (uint32_t)(rng() % q);
          for (auto& v : hB) v = (uint32_t)(rng() % q);
          for (size_t i = 0; i < hA.size(); ++i) gold[i] = C.mulmod(hA[i], hB[i]); }
        Dp::EnqueueWriteMeshBuffer(cq, bA, hA, false);
        Dp::EnqueueWriteMeshBuffer(cq, bB, hB, false);
        Dp::Finish(cq);

        struct V { const char* name; const char* path; const char* note; };
        const std::vector<V> variants = {
            {"modmul_sfpi_full",  "compute/modmul_sfpi_full.cpp",
             "the original: fp32 byte-split, 14 SFPI passes, 4 fill_tile_int"},
            {"modmul_sfpi_1pass", "compute/modmul_sfpi_1pass.cpp",
             "sfpmul24 + ONE pass + no fills  <-- kernels/OPTIMISED.md"},
        };

        // Two tile counts per variant; the per-tile cost is the marginal difference (see header).
        const uint32_t Tlo = (T >= 8) ? T / 4 : 1;
        auto run_at = [&](const char* path, uint32_t nt, uint64_t* mism_out) -> double {
            Program p = CreateProgram();
            auto mkcb = [&](CBIndex i, uint32_t depth) {
                CircularBufferConfig c(depth * I32, {{i, tt::DataFormat::UInt32}});
                c.set_page_size(i, I32); CreateCircularBuffer(p, crs, c); };
            mkcb(CBIndex::c_0, 2); mkcb(CBIndex::c_1, 2); mkcb(CBIndex::c_16, 2);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp", crs,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + std::string(path), crs,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)bA->address(), (uint32_t)bB->address(), nt, 0u, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)bO->address(), nt, 0u});
            SetRuntimeArgs(p, cp, core, {nt, (uint32_t)q, C.q_inv_32(),
                                         (uint32_t)(q & 0xFFFF), (uint32_t)(q >> 16)});
            Dp::MeshWorkload wl; wl.add_program(dr, std::move(p));
            Dp::EnqueueMeshWorkload(cq, wl, false); Dp::Finish(cq);
            if (mism_out) {
                Dp::EnqueueReadMeshBuffer(cq, got, bO, true);
                uint64_t m = 0;
                for (uint64_t e = 0; e < (uint64_t)nt * ELEMS; ++e) if (got[e] != gold[e]) ++m;
                *mism_out = m; }
            double best = 1e18;
            for (uint32_t i = 0; i < ITERS; ++i) {
                auto t0 = hrc::now(); Dp::EnqueueMeshWorkload(cq, wl, false); Dp::Finish(cq);
                best = std::min(best, ms(t0)); }
            return best; };

        fmt::print("  {:<20}{:>10}{:>10}{:>13}{:>9}{:>7}   {}\n",
                   "kernel", "ms@T", "ms@T/4", "us/tile", "vs full", "mism", "what it is");
        double base = 0;
        for (const auto& v : variants) {
            uint64_t mism = 0;
            const double hi = run_at(v.path, T, &mism);
            const double lo = run_at(v.path, Tlo, nullptr);
            const double per = (hi - lo) * 1000.0 / (double)(T - Tlo);   // marginal us/tile
            if (mism) pass = false;
            if (!base) base = per;
            fmt::print("  {:<20}{:>10.4f}{:>10.4f}{:>13.3f}{:>9.2f}{:>7}   {}\n",
                       v.name, hi, lo, per, base / per, mism, v.note);
        }
        fmt::print("\n  us/tile is the MARGINAL cost between the two tile counts, so the per-dispatch\n"
                   "  overhead visible in the two ms columns is cancelled rather than divided into it.\n"
                   "  Compare docs/he/exp_modmul001.md, which fits span(T) over six points on the\n"
                   "  device-side *-KERNEL zone: 12.430 and 3.086 us/tile, 4.03x.\n");

        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    fmt::print("\n{}\n", pass ? "Test Passed" : "Test FAILED");
    return pass ? 0 : 1;
}
