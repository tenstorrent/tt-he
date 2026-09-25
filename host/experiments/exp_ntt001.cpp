// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_ntt001 — pure NTT-32768 latency: 3-factor (32,32,32) vs 2-factor (128,256), at equal effort.
// See docs/matmul/exp_ntt001.md.
//
// SCOPE OF "OPTIMISED": the ONLY kernel from kernels/OPTIMISED.md used here is the modmul, on the
// twiddle stages -- and it turns out to be 8.9% of the 3-factor's critical path, too small to
// resolve (docs/matmul/exp_ntt001.md 3).  THE DATAFLOW IS STOCK IN BOTH VARIANTS, and that is where
// both variants actually spend their time:
// BOTH have since been fixed, each behind a flag so the stock arm stays measurable:
//   * NTT1_DEEP=1   -> dataflow/reader_matmul_{mt,mtk}_deep.cpp, the DRAM depth rule.  The stock
//     readers barrier per tile (2 tiles in flight); these hold a whole output plane (8 / 64 tiles).
//     Matmul stages 1.53x (3-factor) and 1.42x (2-factor).  CONFIRMS that they were feed-bound.
//   * NTT1_A2A=bothnoc -> dataflow/all_to_all_corner_{rd,wr}.cpp, the corner turn's reads split
//     across NCRISC/NOC_1 and BRISC/NOC_0 instead of one RISC doing everything.  1.31x on that
//     stage; what is left is L1 assembly, not fabric.
// Ladder, critical-path busy, all bit-exact: 3-factor 121 -> 110.0 -> 101.6, 2-factor 146 -> 114.0.
// The default arms are STOCK so the comparison stays like-for-like; set both flags for best-effort.
//
// WHY THIS EXISTS.  docs/matmul/ntt_32768_perf.md reports 3-factor 319.5 us against 2-factor 334.7 us
// at C=32 -- a 4.8% gap, measured as HOST WALL-CLOCK over a chain of non-blocking enqueues.  Two
// problems with using that to rank the factorizations:
//
//   1. Host wall contains the dispatch cost, and THE TWO VARIANTS DO NOT ISSUE THE SAME NUMBER OF
//      DISPATCHES -- 12 for the 3-factor, 7 for the 2-factor.  Whatever a dispatch costs, the
//      3-factor pays it 1.71x more often.  A metric that includes it is biased against the variant
//      that is (per that same doc) ahead on compute, so the true compute gap is NOT 4.8%.
//   2. docs/guide/profiling.md, from exp_modmul002: a few-percent kernel comparison on this
//      hardware is not resolvable by host wall-clock at any number of repetitions.
//
// So this bench reports BOTH, and never conflates them:
//
//   host wall / NTT      what the old figure measured; kept so the numbers are comparable
//   dispatches / NTT     printed, because it is the confound above and not a free parameter
//   stage-sum / NTT      each stage run ALONE, ITERS times back to back, then drained; the per-stage
//                        costs summed.  This is an ATTRIBUTION, not a latency: it is NOT the
//                        "host device busy" of docs/guide/profiling.md 1, it does not obey
//                        that taxonomy's `wall >= host device busy` ordering (measured here it comes
//                        out LARGER than wall, because in the real chain consecutive DIFFERENT
//                        programs overlap and isolated re-runs of one program do not), and it must
//                        not be quoted as an NTT latency.  Its value is that it localises cost per
//                        stage and exposes the per-dispatch floor.
//   critical-path busy   the pure figure: trace-derived, `Sum over dispatches (max over cores)`.
//                        Needs the profiler; NTT1_PROF=1 arms it and prints the busy_metrics.py
//                        command with the right --expect.  This is the only one of the four that
//                        is pure kernel time, and it is the one to quote for "NTT latency".
//
// WHAT IS OPTIMISED HERE.  Both variants' twiddle stages ran `compute/modmul_sfpi_hybrid.cpp`.
// `compute/modmul_sfpi_1pass.cpp` (kernels/OPTIMISED.md) is a DROP-IN -- identical runtime-arg
// interface -- and measures 4.03x standalone (12.430 -> 3.086 us/tile, docs/he/exp_modmul001.md).
// NTT1_MODMUL=hybrid|1pass selects it, so the swap is measured here rather than assumed to carry.
// The 3-factor has TWO twiddle stages to the 2-factor's one, so this is also a test of whether an
// optimisation that helps the busier pipeline more can move a ranking.
//
// Everything else is lifted unchanged from host/tests/ntt/ntt32768_dev.cpp and
// host/tests/ntt/ntt32768_2factor_dev.cpp so that this is a like-for-like re-measurement and not a
// rewrite: same goldens, same folds, same stage decomposition, same kernels otherwise.
//
//   ./build/exp_ntt001 [q] [seed]
//   env: NTT1_CORES=32  NTT1_ITERS=20  NTT1_MODMUL=1pass|hybrid  NTT1_PROF=0|1  NTT1_CSV=<path>
//
// Both variants are checked against their own host golden on every run, and the goldens are
// themselves spot-checked against the direct DFT: a variant that is fast because it computes
// something else cannot be reported as a win.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <map>
#include <string>
#include <vector>

#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t T = 32, NB = 4, ELEMS = 1024, NTILE = 32;
constexpr uint32_t I8 = ELEMS * 1, I32 = ELEMS * 4;
constexpr uint32_t N = 32768, N1 = 128, N2 = 256;
const uint64_t POW[4] = {1, 128, 16384, 2097152};

using hrc = std::chrono::high_resolution_clock;
double us_since(hrc::time_point t) {
    return std::chrono::duration<double, std::micro>(hrc::now() - t).count(); }

uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) { uint64_t r = 1; b %= m;
    while (e) { if (e & 1) r = (__uint128_t)r * b % m; b = (__uint128_t)b * b % m; e >>= 1; } return r; }
uint64_t prim_root(uint64_t q) { uint64_t phi = q - 1, n = phi; std::vector<uint64_t> f;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) { f.push_back(d); while (n % d == 0) n /= d; }
    if (n > 1) f.push_back(n);
    for (uint64_t g = 2; g < q; ++g) { bool ok = true; for (uint64_t p : f) if (modpow(g, phi / p, q) == 1) ok = false;
        if (ok) return g; } throw std::runtime_error("no root"); }

template <typename E> std::vector<E> tilize(const std::vector<E>& in) {
    std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = in[r * T + c];
    return o; }
template <typename E> std::vector<E> untilize(const std::vector<E>& in) {
    std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[r * T + c] = in[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)];
    return o; }
int8_t limb8(uint32_t v, uint32_t p) { return (int8_t)((v >> (7 * p)) & 127); }

const char* env_s(const char* k, const char* dflt) { const char* v = std::getenv(k); return v ? v : dflt; }
uint32_t env_u(const char* k, uint32_t dflt) { const char* v = std::getenv(k); return v ? (uint32_t)std::stoul(v) : dflt; }

// One variant's measurement, so both are reported through the same struct and the same rules.
struct Result {
    const char* name;
    uint32_t dispatches;      // per NTT
    uint32_t mism;            // vs its own host golden
    double wall_us;           // host wall / NTT, non-blocking chain
    double stagesum_us;       // sum of isolated per-stage costs (attribution, NOT a latency)
    std::vector<std::pair<const char*, double>> stages;   // per-stage blocking wall, us
};

void report(const Result& r) {
    fmt::print("\n  {:<22} dispatches/NTT {:2d}   mism {}/{}\n", r.name, r.dispatches, r.mism, N);
    fmt::print("    host wall      {:8.1f} us/NTT\n", r.wall_us);
    fmt::print("    stage-sum      {:8.1f} us/NTT   ({:.1f} us/dispatch avg)  ATTRIBUTION, not a latency\n",
               r.stagesum_us, r.stagesum_us / r.dispatches);
    for (auto& [nm, us] : r.stages) fmt::print("      {:<12} {:8.1f} us\n", nm, us);
    std::fflush(stdout);
}
}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const uint64_t q = (argc > 1) ? std::stoull(argv[1]) : 167772161ULL;
        const uint32_t seed = (argc > 2) ? std::stoul(argv[2]) : 0;
        const uint32_t C = env_u("NTT1_CORES", 32);
        const int iters = (int)env_u("NTT1_ITERS", 20);
        const std::string mm_kern = env_s("NTT1_MODMUL", "1pass");
        const bool prof = env_u("NTT1_PROF", 0) != 0;
        // NTT1_ONLY restricts the run to ONE variant. Required for a profiler capture: busy_metrics.py
        // gates on `--expect N` matching the host's own dispatch count, and a capture containing both
        // variants plus the timing loops has 228 dispatches, not 12. Under NTT1_PROF the timing loops
        // are skipped as well, so the CSV holds exactly one NTT and --expect is exactly its stage count.
        const bool deep = env_u("NTT1_DEEP", 0) != 0;   // depth-rule matmul readers (exp_ntt001.md 2a)
        const std::string a2a_mode = env_s("NTT1_A2A", "1noc");
        const bool bothnoc = (a2a_mode == "bothnoc" || a2a_mode == "probe");
        // NTT1_A2A=probe swaps the assemble half for all_to_all_corner_probe.cpp, which SKIPS the
        // row-copy assembly and therefore computes a WRONG answer on purpose: (real - probe) is the
        // assembly cost. A probe run MUST fail its correctness gate; that is the proof it was used.
        const bool a2a_probe = (a2a_mode == "probe");
        if (a2a_mode == "exchange" && C != NTILE) {
            fmt::print("NTT1_A2A=exchange needs NTT1_CORES={} (every core must send, or peers hang)\n", NTILE);
            return 2; }
        // NTT1_FUSE=1: dec+mm+rec collapsed into ONE compute kernel per triple
        // (compute/fused_matmul_{a,c}_1pass.cpp -- limbs and planes stay in on-core CBs, no DRAM
        // round trip). 12 dispatches -> 6. Those kernels process exactly ONE tile and have no
        // n_tiles loop, so this path REQUIRES npc==1, i.e. C == NTILE.
        const bool fuse = env_u("NTT1_FUSE", 0) != 0;
        const bool mega = env_u("NTT1_MEGA", 0) != 0;   // whole NTT in ONE dispatch
        // NTT1_ZONES=1 compiles inner per-phase profiler zones into the fused kernels, so
        // decompose / matmul / recombine / twiddle can be attributed inside a fused dispatch.
        const std::map<std::string, std::string> zdef =
            env_u("NTT1_ZONES", 0)
                ? std::map<std::string, std::string>{{"NTT_PHASE_ZONES", "1"},
                                                     {"TB001_PHASE_ZONES", "1"}}   // fused_matmul_*_1pass
                : std::map<std::string, std::string>{};   // whole NTT in ONE dispatch
        if (mega && (NTILE / C != 1)) {
            fmt::print("NTT1_MEGA needs NTT1_CORES={}\n", NTILE); return 2; }
        if (fuse && NTILE / C != 1) {
            fmt::print("NTT1_FUSE needs NTT1_CORES={} (fused kernels do one tile each)\n", NTILE);
            return 2; }   // depth-rule matmul readers (exp_ntt001.md 2a)
        const std::string only = env_s("NTT1_ONLY", prof ? "3f" : "both");
        const bool do3 = (only != "2f"), do2 = (only != "3f");
        if (mm_kern != "1pass" && mm_kern != "hybrid") { fmt::print("NTT1_MODMUL must be 1pass|hybrid\n"); return 2; }
        const std::string modmul_path = (mm_kern == "1pass")
            ? OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_1pass.cpp"
            : OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_hybrid.cpp";

        mont::MontgomeryContext ctx((uint32_t)q);
        const uint32_t QINV = ctx.q_inv_32(), QLO = q & 0xFFFF, QHI = q >> 16;
        const uint32_t K14 = ctx.to_montgomery(16384u);

        fmt::print("exp_ntt001 — NTT-32768 pure latency, 3-factor vs 2-factor\n");
        fmt::print("  q={}  seed={}  cores={}  iters={}  modmul={}  reader={}  a2a={}  fuse={}  profiler={}\n",
                   q, seed, C, iters, mm_kern, deep ? "deep" : "stock", a2a_mode, mega ? "MEGA" : (fuse ? "on" : "off"), prof ? "on" : "off");
        std::fflush(stdout);

        uint64_t g = prim_root(q), wN = modpow(g, (q - 1) / N, q);
        std::mt19937 rng(seed); std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)q - 1);
        std::vector<uint32_t> x(N); for (auto& v : x) v = dist(rng);

        // =========================================================================================
        // Shared device setup
        // =========================================================================================
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        auto mkbuf = [&](uint32_t ntiles, uint32_t page) {
            distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = ntiles * page}, d, mesh.get()); };
        auto crange = [](uint32_t c) { uint32_t w = c < 8 ? c : 8, h = (c + 7) / 8;
            return CoreRange(CoreCoord{0, 0}, CoreCoord{w - 1, h - 1}); };
        auto cat = [](uint32_t c) { return CoreCoord{c % 8, c / 8}; };
        auto mkcbr = [&](Program& p, CoreRange cr, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
            CircularBufferConfig cc(depth * page, {{idx, fmt}}); cc.set_page_size(idx, page);
            CreateCircularBuffer(p, cr, cc); };
        const uint32_t npc = NTILE / C;

        // ---- stage builders, shared by both variants (identical to the two originals) ----
        auto build_decompose = [&](auto in, auto out) {
            Program p = CreateProgram(); CoreRange cr = crange(C);
            mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32);
            mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::Int8, 4, I8);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            std::vector<uint32_t> wc; TensorAccessorArgs(*out).append_to(wc);
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i8_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = wc});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/decompose_limbs.cpp", cr,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true});
            for (uint32_t c = 0; c < C; ++c) { auto co = cat(c);
                SetRuntimeArgs(p, rd, co, {(uint32_t)in->address(), npc, c, C, 1u});
                SetRuntimeArgs(p, wr, co, {(uint32_t)out->address(), npc, c, C, 4u});
                SetRuntimeArgs(p, cp, co, {npc}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // 3-factor matmul: reader_matmul_mt (single K tile)
        auto build_matmul3 = [&](auto w, auto limbs, auto out, uint32_t mode) {
            Program p = CreateProgram(); CoreRange cr = crange(C);
            const uint32_t dep3 = deep ? 2 * NB : 2;          // deep reader pushes NB pairs at once
            mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::Int8, dep3, I8);
            mkcbr(p, cr, CBIndex::c_1, tt::DataFormat::Int8, dep3, I8);
            mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::Int32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*w).append_to(rc); TensorAccessorArgs(*limbs).append_to(rc);
            auto rd = CreateKernel(p, deep ? OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul_mt_deep.cpp"
                                           : OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul_mt.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            std::vector<uint32_t> wc; TensorAccessorArgs(*out).append_to(wc);
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_matmul_mt.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = wc});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/matmul_compute_mt.cpp", cr,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
            for (uint32_t c = 0; c < C; ++c) { auto co = cat(c);
                SetRuntimeArgs(p, rd, co, {(uint32_t)w->address(), (uint32_t)limbs->address(), npc, c, C, mode});
                SetRuntimeArgs(p, wr, co, {(uint32_t)out->address(), npc, c, C});
                SetRuntimeArgs(p, cp, co, {npc}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // 2-factor matmul: reader_matmul_mtk (K-looped)
        // 2-factor fused matmul+bigK-recombine: planes stay in DST, no 16 KB/core plane round trip.
        // The K-looped analogue of build_fused; the decompose stays separate because the 2-factor's
        // matmuls are NOT tile-local (M1 needs 8 input tiles, M2 needs 4 from other cores), so folding
        // it in would have 8 cores each decompose the same 8 tiles.
        auto build_fused_k = [&](auto wconst, auto dyn, auto out, uint32_t mode,
                                 uint32_t Mt, uint32_t Kt, uint32_t Nt) {
            Program p = CreateProgram(); CoreRange cr = crange(C);
            const uint32_t dep2 = deep ? 2 * NB * Kt : 2;
            mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::Int8, dep2, I8);
            mkcbr(p, cr, CBIndex::c_1, tt::DataFormat::Int8, dep2, I8);
            mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*wconst).append_to(rc); TensorAccessorArgs(*dyn).append_to(rc);
            auto rd = CreateKernel(p, deep ? OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul_mtk_deep.cpp"
                                           : OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul_mtk.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_k_1pass.cpp", cr,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                              .dst_full_sync_en = true, .defines = zdef});
            for (uint32_t c = 0; c < C; ++c) { auto co = cat(c);
                SetRuntimeArgs(p, rd, co, {(uint32_t)wconst->address(), (uint32_t)dyn->address(), mode, Mt, Kt, Nt, npc, c, C});
                SetRuntimeArgs(p, wr, co, {(uint32_t)out->address(), npc, c, C, 1u});
                SetRuntimeArgs(p, cp, co, {NB * Kt, (uint32_t)q, QINV, QLO, QHI, K14}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        auto build_matmul2 = [&](auto wconst, auto dyn, auto out, uint32_t mode,
                                 uint32_t Mt, uint32_t Kt, uint32_t Nt) {
            Program p = CreateProgram(); CoreRange cr = crange(C);
            const uint32_t dep2 = deep ? 2 * NB * Kt : 2;     // deep reader pushes K_acc = NB*Kt pairs at once
            mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::Int8, dep2, I8);
            mkcbr(p, cr, CBIndex::c_1, tt::DataFormat::Int8, dep2, I8);
            mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::Int32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*wconst).append_to(rc); TensorAccessorArgs(*dyn).append_to(rc);
            auto rd = CreateKernel(p, deep ? OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul_mtk_deep.cpp"
                                           : OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul_mtk.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            std::vector<uint32_t> wc; TensorAccessorArgs(*out).append_to(wc);
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_matmul_mt.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = wc});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/matmul_compute_mtk.cpp", cr,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
            for (uint32_t c = 0; c < C; ++c) { auto co = cat(c);
                SetRuntimeArgs(p, rd, co, {(uint32_t)wconst->address(), (uint32_t)dyn->address(), mode, Mt, Kt, Nt, npc, c, C});
                SetRuntimeArgs(p, wr, co, {(uint32_t)out->address(), npc, c, C});
                SetRuntimeArgs(p, cp, co, {npc, NB * Kt}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // One dispatch for decompose+matmul+recombine. `modeA` picks fold-LHS (W @ x, stage A/D)
        // or fold-RHS (x @ W, stage C); the two differ only in which operand is the matmul LHS.
        auto build_fused = [&](auto in, auto w, auto out, bool modeA) {
            Program p = CreateProgram(); CoreRange cr = crange(C);
            mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32);      // Xin
            mkcbr(p, cr, CBIndex::c_1, tt::DataFormat::Int8, NB * NB, I8);   // folded W, whole batch
            mkcbr(p, cr, CBIndex::c_2, tt::DataFormat::Int8, NB, I8);        // limbs, stay on core
            mkcbr(p, cr, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);      // planes, stay on core
            mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);     // result
            std::vector<uint32_t> rc; TensorAccessorArgs(*in).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused_batch_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, modeA ? OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a_1pass.cpp"
                                            : OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c_1pass.cpp", cr,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                              .dst_full_sync_en = true, .defines = zdef});
            for (uint32_t c = 0; c < C; ++c) { auto co = cat(c);
                SetRuntimeArgs(p, rd, co, {(uint32_t)in->address(), (uint32_t)w->address(), c});
                SetRuntimeArgs(p, wr, co, {(uint32_t)out->address(), 1u, c, C, 1u});
                SetRuntimeArgs(p, cp, co, {(uint32_t)q, QINV, QLO, QHI, K14}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        auto build_recombine = [&](auto in, auto out, const char* kern) {
            Program p = CreateProgram(); CoreRange cr = crange(C);
            mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::UInt32, 8, I32);
            mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, kern, cr,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true});
            for (uint32_t c = 0; c < C; ++c) { auto co = cat(c);
                SetRuntimeArgs(p, rd, co, {(uint32_t)in->address(), npc, c, C, 4u});
                SetRuntimeArgs(p, wr, co, {(uint32_t)out->address(), npc, c, C, 1u});
                SetRuntimeArgs(p, cp, co, {npc, (uint32_t)q, QINV, QLO, QHI, K14}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // the A/B: modmul_sfpi_1pass vs modmul_sfpi_hybrid, identical runtime-arg interface
        auto build_modmul = [&](auto a, auto b, auto out) {
            Program p = CreateProgram(); CoreRange cr = crange(C);
            mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32);
            mkcbr(p, cr, CBIndex::c_1, tt::DataFormat::UInt32, 2, I32);
            mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul_g.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, modmul_path, cr,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                              .dst_full_sync_en = true, .defines = zdef});
            for (uint32_t c = 0; c < C; ++c) { auto co = cat(c);
                SetRuntimeArgs(p, rd, co, {(uint32_t)a->address(), (uint32_t)b->address(), npc, c, C, 1u});
                SetRuntimeArgs(p, wr, co, {(uint32_t)out->address(), npc, c, C, 1u});
                SetRuntimeArgs(p, cp, co, {npc, (uint32_t)q, QINV, QLO, QHI}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // NTT1_A2A=1pass (default) drives ONE RISC/NOC_1 for both reads and writes; =bothnoc splits
        // the reads across NCRISC/NOC_1 and BRISC/NOC_0 (dram_guide 3.1, 1.79x). Same runtime-arg
        // vector for both kernels, so the host builds one list either way.
        auto build_a2a = [&](auto in, auto out) {
            Program p = CreateProgram(); CoreRange cr = crange(C);
            if (a2a_mode == "exchange") {
                // Core-to-core exchange: each core reads only its own tile and pushes rows to peers.
                // Requires the full grid (a core that never sends hangs every other core's semaphore
                // wait), and needs each peer's PHYSICAL NoC coords, which only the host can resolve.
                mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::UInt32, 1, I32);   // my input tile
                mkcbr(p, cr, CBIndex::c_2, tt::DataFormat::UInt32, 1, I32);   // my output tile
                const uint32_t sem = CreateSemaphore(p, cr, 0);
                auto k = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/all_to_all_corner_x.cpp", cr,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
                std::vector<uint32_t> coords;
                for (uint32_t j = 0; j < NTILE; ++j) {
                    auto ph = mesh->worker_core_from_logical_core(cat(j));
                    coords.push_back((uint32_t)ph.x); coords.push_back((uint32_t)ph.y); }
                for (uint32_t c = 0; c < C; ++c) {
                    std::vector<uint32_t> av = {(uint32_t)in->address(), (uint32_t)out->address(),
                                                NTILE, c, sem};
                    av.insert(av.end(), coords.begin(), coords.end());
                    SetRuntimeArgs(p, k, cat(c), av); }
            } else if (!bothnoc) {
                mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::UInt32, NTILE + 1, I32);
                auto k = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/all_to_all_corner_mc.cpp", cr,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
                for (uint32_t c = 0; c < C; ++c)
                    SetRuntimeArgs(p, k, cat(c), {(uint32_t)in->address(), (uint32_t)out->address(), NTILE, c, C, npc});
            } else {
                mkcbr(p, cr, CBIndex::c_0, tt::DataFormat::UInt32, NTILE / 2, I32);   // NCRISC half
                mkcbr(p, cr, CBIndex::c_1, tt::DataFormat::UInt32, NTILE / 2, I32);   // BRISC half
                mkcbr(p, cr, CBIndex::c_2, tt::DataFormat::UInt32, 1, I32);           // output scratch
                auto krd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/all_to_all_corner_rd.cpp", cr,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
                auto kwr = CreateKernel(p, a2a_probe ? OVERRIDE_KERNEL_PREFIX "dataflow/all_to_all_corner_probe.cpp"
                                                     : OVERRIDE_KERNEL_PREFIX "dataflow/all_to_all_corner_wr.cpp", cr,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
                for (uint32_t c = 0; c < C; ++c) {
                    const std::vector<uint32_t> av = {(uint32_t)in->address(), (uint32_t)out->address(),
                                                      NTILE, c, C, npc};
                    SetRuntimeArgs(p, krd, cat(c), av);
                    SetRuntimeArgs(p, kwr, cat(c), av); }
            }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

        // ---- the measurement, identical for both variants ----
        auto measure = [&](const char* name, std::vector<distributed::MeshWorkload*>& wls,
                           const std::vector<const char*>& nm, uint32_t mism) {
            Result r{name, (uint32_t)wls.size(), mism, 0, 0, {}};
            distributed::Finish(cq);
            // host wall: the whole chain, non-blocking, ITERS times (what the old figure measured)
            auto t0 = hrc::now();
            for (int it = 0; it < iters; ++it) for (auto* wl : wls) distributed::EnqueueMeshWorkload(cq, *wl, false);
            distributed::Finish(cq);
            r.wall_us = us_since(t0) / iters;
            // device-busy: per-stage, each drained, so inter-stage host stalls are excluded
            for (size_t s = 0; s < wls.size(); ++s) {
                distributed::EnqueueMeshWorkload(cq, *wls[s], false); distributed::Finish(cq);
                auto s0 = hrc::now();
                for (int it = 0; it < iters; ++it) distributed::EnqueueMeshWorkload(cq, *wls[s], false);
                distributed::Finish(cq);
                double su = us_since(s0) / iters;
                r.stages.emplace_back(nm[s], su); r.stagesum_us += su; }
            return r; };

        // =========================================================================================
        // Variant A — 3-factor (32,32,32), 12 dispatches
        // =========================================================================================
        uint64_t w1024 = modpow(wN, N / 1024, q), w32 = modpow(wN, N / T, q);
        std::vector<uint32_t> W(ELEMS);
        for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b)
            W[a * T + b] = (uint32_t)modpow(w32, (uint64_t)a * b, q);
        std::vector<int8_t> WfoldA(NB * NB * ELEMS), Wrhs(NB * NB * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
            std::vector<int8_t> la(ELEMS), lc(ELEMS);
            for (uint32_t i = 0; i < ELEMS; ++i) {
                la[i] = limb8((uint32_t)((POW[r] * W[i]) % q), p);
                lc[i] = limb8((uint32_t)((POW[p] * W[i]) % q), r); }
            auto ta = tilize(la); std::copy(ta.begin(), ta.end(), WfoldA.begin() + (p * NB + r) * ELEMS);
            auto tc = tilize(lc); std::copy(tc.begin(), tc.end(), Wrhs.begin() + (p * NB + r) * ELEMS); }
        std::vector<uint32_t> TaTiles(NTILE * ELEMS), TbTiles3(NTILE * ELEMS);
        { std::vector<uint32_t> ta(ELEMS);
          for (uint32_t s = 0; s < T; ++s) for (uint32_t n2 = 0; n2 < T; ++n2)
              ta[s * T + n2] = ctx.to_montgomery((uint32_t)modpow(w1024, (uint64_t)s * n2 % 1024, q));
          auto tt = tilize(ta);
          for (uint32_t n1 = 0; n1 < NTILE; ++n1) std::copy(tt.begin(), tt.end(), TaTiles.begin() + n1 * ELEMS); }
        for (uint32_t n1 = 0; n1 < NTILE; ++n1) { std::vector<uint32_t> tb(ELEMS);
            for (uint32_t s = 0; s < T; ++s) for (uint32_t t = 0; t < T; ++t)
                tb[s * T + t] = ctx.to_montgomery((uint32_t)modpow(wN, (uint64_t)n1 * ((s + T * t) % N) % N, q));
            auto tt = tilize(tb); std::copy(tt.begin(), tt.end(), TbTiles3.begin() + n1 * ELEMS); }
        std::vector<uint32_t> XinTiles(NTILE * ELEMS);
        for (uint32_t n1 = 0; n1 < NTILE; ++n1) { std::vector<uint32_t> slab(ELEMS);
            for (uint32_t n3 = 0; n3 < T; ++n3) for (uint32_t n2 = 0; n2 < T; ++n2)
                slab[n3 * T + n2] = x[n1 + T * n2 + 1024 * n3];
            auto tt = tilize(slab); std::copy(tt.begin(), tt.end(), XinTiles.begin() + n1 * ELEMS); }

        auto mm32 = [&](const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
            std::vector<uint64_t> o(ELEMS, 0);
            for (uint32_t i = 0; i < T; ++i) for (uint32_t k = 0; k < T; ++k) { uint64_t aik = a[i * T + k];
                if (!aik) continue; for (uint32_t j = 0; j < T; ++j)
                    o[i * T + j] = (o[i * T + j] + aik * b[k * T + j]) % q; }
            return o; };
        std::vector<uint64_t> Wm(ELEMS);
        for (uint32_t a = 0; a < T; ++a) for (uint32_t b = 0; b < T; ++b) Wm[a * T + b] = modpow(w32, (uint64_t)a * b, q);
        std::vector<std::vector<uint64_t>> Cbgold(NTILE);
        for (uint32_t n1 = 0; n1 < NTILE; ++n1) {
            std::vector<uint64_t> cube(ELEMS);
            for (uint32_t n3 = 0; n3 < T; ++n3) for (uint32_t n2 = 0; n2 < T; ++n2)
                cube[n3 * T + n2] = x[n1 + T * n2 + 1024 * n3];
            auto B = mm32(Wm, cube);
            for (uint32_t s = 0; s < T; ++s) for (uint32_t n2 = 0; n2 < T; ++n2)
                B[s * T + n2] = B[s * T + n2] * modpow(w1024, (uint64_t)s * n2 % 1024, q) % q;
            auto Cc = mm32(B, Wm);
            std::vector<uint64_t> Cb(ELEMS);
            for (uint32_t s = 0; s < T; ++s) for (uint32_t t = 0; t < T; ++t)
                Cb[s * T + t] = Cc[s * T + t] * modpow(wN, (uint64_t)n1 * ((s + T * t) % N) % N, q) % q;
            Cbgold[n1] = Cb; }
        std::vector<uint32_t> Xref3(N);
        for (uint32_t s = 0; s < T; ++s) { std::vector<uint64_t> D(ELEMS);
            for (uint32_t n1 = 0; n1 < T; ++n1) for (uint32_t t = 0; t < T; ++t) D[n1 * T + t] = Cbgold[n1][s * T + t];
            auto E = mm32(Wm, D);
            for (uint32_t v = 0; v < T; ++v) for (uint32_t t = 0; t < T; ++t)
                Xref3[s + T * t + 1024 * v] = (uint32_t)E[v * T + t]; }

        // =========================================================================================
        // Variant B — 2-factor (128,256), 7 dispatches
        // =========================================================================================
        uint64_t w256 = modpow(wN, N / N2, q), w128 = modpow(wN, N / N1, q);
        std::vector<uint64_t> W256(N2 * N2), W128(N1 * N1);
        for (uint32_t a = 0; a < N2; ++a) for (uint32_t b = 0; b < N2; ++b)
            W256[a * N2 + b] = modpow(w256, (uint64_t)a * b % N2, q);
        for (uint32_t a = 0; a < N1; ++a) for (uint32_t b = 0; b < N1; ++b)
            W128[a * N1 + b] = modpow(w128, (uint64_t)a * b % N1, q);
        const uint32_t M1_Kt = 8, M1_Nt = 8, M2_Mt = 4, M2_Kt = 4;
        std::vector<int8_t> Wrhs256(NB * NB * M1_Kt * M1_Nt * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r)
            for (uint32_t kt = 0; kt < M1_Kt; ++kt) for (uint32_t nt = 0; nt < M1_Nt; ++nt) {
                std::vector<int8_t> blk(ELEMS);
                for (uint32_t i = 0; i < T; ++i) for (uint32_t j = 0; j < T; ++j)
                    blk[i * T + j] = limb8((uint32_t)((POW[p] * W256[(kt * T + i) * N2 + (nt * T + j)]) % q), r);
                auto tt = tilize(blk);
                std::copy(tt.begin(), tt.end(), Wrhs256.begin() + (((p * NB + r) * M1_Kt + kt) * M1_Nt + nt) * ELEMS); }
        std::vector<int8_t> Wfold128(NB * NB * M2_Mt * M2_Kt * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r)
            for (uint32_t mt = 0; mt < M2_Mt; ++mt) for (uint32_t kt = 0; kt < M2_Kt; ++kt) {
                std::vector<int8_t> blk(ELEMS);
                for (uint32_t i = 0; i < T; ++i) for (uint32_t j = 0; j < T; ++j)
                    blk[i * T + j] = limb8((uint32_t)((POW[r] * W128[(mt * T + i) * N1 + (kt * T + j)]) % q), p);
                auto tt = tilize(blk);
                std::copy(tt.begin(), tt.end(), Wfold128.begin() + (((p * NB + r) * M2_Mt + mt) * M2_Kt + kt) * ELEMS); }
        std::vector<uint32_t> TbTiles2(NTILE * ELEMS), ATiles(NTILE * ELEMS);
        for (uint32_t mt = 0; mt < 4; ++mt) for (uint32_t nt = 0; nt < 8; ++nt) {
            std::vector<uint32_t> tb(ELEMS), a(ELEMS);
            for (uint32_t i = 0; i < T; ++i) for (uint32_t j = 0; j < T; ++j) {
                uint32_t n1 = mt * T + i, u = nt * T + j;
                tb[i * T + j] = ctx.to_montgomery((uint32_t)modpow(wN, (uint64_t)n1 * u % N, q));
                a[i * T + j] = x[n1 + N1 * (nt * T + j)]; }
            auto t1 = tilize(tb); std::copy(t1.begin(), t1.end(), TbTiles2.begin() + (mt * 8 + nt) * ELEMS);
            auto t2 = tilize(a);  std::copy(t2.begin(), t2.end(), ATiles.begin()   + (mt * 8 + nt) * ELEMS); }

        auto mmg = [&](const std::vector<uint64_t>& a, uint32_t am, uint32_t ak,
                       const std::vector<uint64_t>& b, uint32_t bn) {
            std::vector<uint64_t> o((size_t)am * bn, 0);
            for (uint32_t i = 0; i < am; ++i) for (uint32_t k = 0; k < ak; ++k) { uint64_t aik = a[(size_t)i * ak + k];
                if (aik) for (uint32_t j = 0; j < bn; ++j)
                    o[(size_t)i * bn + j] = (o[(size_t)i * bn + j] + aik * b[(size_t)k * bn + j]) % q; }
            return o; };
        std::vector<uint64_t> Amat((size_t)N1 * N2);
        for (uint32_t n1 = 0; n1 < N1; ++n1) for (uint32_t n2 = 0; n2 < N2; ++n2)
            Amat[(size_t)n1 * N2 + n2] = x[n1 + N1 * n2];
        auto Sg = mmg(Amat, N1, N2, W256, N2);
        for (uint32_t n1 = 0; n1 < N1; ++n1) for (uint32_t u = 0; u < N2; ++u)
            Sg[(size_t)n1 * N2 + u] = Sg[(size_t)n1 * N2 + u] * modpow(wN, (uint64_t)n1 * u % N, q) % q;
        auto Rg = mmg(W128, N1, N1, Sg, N2);
        std::vector<uint32_t> Xref2(N);
        for (uint32_t v = 0; v < N1; ++v) for (uint32_t u = 0; u < N2; ++u)
            Xref2[u + N2 * v] = (uint32_t)Rg[(size_t)v * N2 + u];

        // one direct-DFT spot check, shared: both goldens must agree with it
        { std::mt19937 r2(123); std::uniform_int_distribution<uint32_t> di(0, N - 1); uint32_t bad = 0;
          for (int it = 0; it < 16; ++it) { uint32_t k = di(r2); unsigned __int128 acc = 0;
              for (uint32_t n = 0; n < N; ++n) acc += (unsigned __int128)x[n] * modpow(wN, (uint64_t)n * k % N, q);
              uint32_t want = (uint32_t)(acc % q);
              if (want != Xref3[k] || want != Xref2[k]) ++bad; }
          if (bad) { fmt::print("GOLDEN SELF-CHECK FAILED ({} bad) — not a device result\n", bad); return 3; } }
        fmt::print("  goldens agree with the direct DFT at 16 random indices\n"); std::fflush(stdout);

        // ---- buffers ----
        auto b_Xin = mkbuf(NTILE, I32), b_WfoldA = mkbuf(NB * NB, I8), b_Wrhs = mkbuf(NB * NB, I8);
        auto b_Ta = mkbuf(NTILE, I32), b_Tb3 = mkbuf(NTILE, I32);
        auto b_xlimbs = mkbuf(NTILE * NB, I8), b_CrA = mkbuf(NTILE * NB, I32), b_G = mkbuf(NTILE, I32);
        auto b_Gp = mkbuf(NTILE, I32), b_aC = mkbuf(NTILE * NB, I8), b_CrC = mkbuf(NTILE * NB, I32), b_H = mkbuf(NTILE, I32);
        auto b_Cb = mkbuf(NTILE, I32), b_D = mkbuf(NTILE, I32);
        auto b_Dlimbs = mkbuf(NTILE * NB, I8), b_CrD = mkbuf(NTILE * NB, I32), b_E = mkbuf(NTILE, I32);
        auto b_A = mkbuf(NTILE, I32), b_Tb2 = mkbuf(NTILE, I32);
        auto b_Wrhs256 = mkbuf(NB * NB * M1_Kt * M1_Nt, I8), b_Wfold128 = mkbuf(NB * NB * M2_Mt * M2_Kt, I8);
        auto b_Alimbs = mkbuf(NTILE * NB, I8), b_Splanes = mkbuf(NTILE * NB, I32), b_S = mkbuf(NTILE, I32);
        auto b_Sp = mkbuf(NTILE, I32), b_Splimbs = mkbuf(NTILE * NB, I8);
        auto b_Rplanes = mkbuf(NTILE * NB, I32), b_R = mkbuf(NTILE, I32);

        distributed::EnqueueWriteMeshBuffer(cq, b_Xin, XinTiles, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_WfoldA, WfoldA, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Wrhs, Wrhs, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Ta, TaTiles, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Tb3, TbTiles3, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_A, ATiles, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Wrhs256, Wrhs256, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Wfold128, Wfold128, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_Tb2, TbTiles2, false);

        // ---- 3-factor pipeline ----
        auto A1 = build_decompose(b_Xin, b_xlimbs);
        auto A2 = build_matmul3(b_WfoldA, b_xlimbs, b_CrA, 0u);
        auto A3 = build_recombine(b_CrA, b_G, OVERRIDE_KERNEL_PREFIX "compute/recombine_compute.cpp");
        auto Bt = build_modmul(b_G, b_Ta, b_Gp);
        auto C1 = build_decompose(b_Gp, b_aC);
        auto C2 = build_matmul3(b_Wrhs, b_aC, b_CrC, 1u);
        auto C3 = build_recombine(b_CrC, b_H, OVERRIDE_KERNEL_PREFIX "compute/recombine_compute.cpp");
        auto TB = build_modmul(b_H, b_Tb3, b_Cb);
        auto A2A = build_a2a(b_Cb, b_D);
        auto D1 = build_decompose(b_D, b_Dlimbs);
        auto D2 = build_matmul3(b_WfoldA, b_Dlimbs, b_CrD, 0u);
        auto D3 = build_recombine(b_CrD, b_E, OVERRIDE_KERNEL_PREFIX "compute/recombine_compute.cpp");
        // NTT1_MEGA=1: the ENTIRE 3-factor NTT as ONE dispatch. Compute holds the whole chain and
        // parks mid-kernel on cb_d while BRISC runs the 32-core corner turn. Needs the full grid.
        auto build_mega = [&]() {
            Program p = CreateProgram(); CoreRange cr = crange(C);
            mkcbr(p, cr, CBIndex::c_0,  tt::DataFormat::UInt32, 1, I32);        // Xin
            mkcbr(p, cr, CBIndex::c_1,  tt::DataFormat::Int8,   NB * NB, I8);   // WfoldA resident
            mkcbr(p, cr, CBIndex::c_2,  tt::DataFormat::Int8,   NB * NB, I8);   // Wrhs   resident
            mkcbr(p, cr, CBIndex::c_3,  tt::DataFormat::UInt32, 1, I32);        // Ta
            mkcbr(p, cr, CBIndex::c_4,  tt::DataFormat::UInt32, 1, I32);        // Tb
            mkcbr(p, cr, CBIndex::c_5,  tt::DataFormat::Int8,   NB, I8);        // limbs
            mkcbr(p, cr, CBIndex::c_6,  tt::DataFormat::UInt32, 1, I32);        // D (from BRISC)
            mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::UInt32, 1, I32);        // Cb (to BRISC)
            mkcbr(p, cr, CBIndex::c_17, tt::DataFormat::UInt32, 1, I32);        // E
            mkcbr(p, cr, CBIndex::c_18, tt::DataFormat::UInt32, 1, I32);        // G
            mkcbr(p, cr, CBIndex::c_19, tt::DataFormat::UInt32, 1, I32);        // G'
            mkcbr(p, cr, CBIndex::c_20, tt::DataFormat::UInt32, 1, I32);        // H
            const uint32_t sem = CreateSemaphore(p, cr, 0);
            std::vector<uint32_t> rc;
            TensorAccessorArgs(*b_Xin).append_to(rc);   TensorAccessorArgs(*b_WfoldA).append_to(rc);
            TensorAccessorArgs(*b_Wrhs).append_to(rc);  TensorAccessorArgs(*b_Ta).append_to(rc);
            TensorAccessorArgs(*b_Tb3).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/ntt32768_3f_fused_rd.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/ntt32768_3f_fused_wr.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/ntt32768_3f_fused.cpp", cr,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                              .dst_full_sync_en = true, .defines = zdef});
            std::vector<uint32_t> coords;
            for (uint32_t j = 0; j < NTILE; ++j) { auto ph = mesh->worker_core_from_logical_core(cat(j));
                coords.push_back((uint32_t)ph.x); coords.push_back((uint32_t)ph.y); }
            for (uint32_t c = 0; c < C; ++c) { auto co = cat(c);
                SetRuntimeArgs(p, rd, co, {(uint32_t)b_Xin->address(), (uint32_t)b_WfoldA->address(),
                                           (uint32_t)b_Wrhs->address(), (uint32_t)b_Ta->address(),
                                           (uint32_t)b_Tb3->address(), c});
                std::vector<uint32_t> wv = {(uint32_t)b_E->address(), NTILE, c, sem};
                wv.insert(wv.end(), coords.begin(), coords.end());
                SetRuntimeArgs(p, wr, co, wv);
                SetRuntimeArgs(p, cp, co, {(uint32_t)q, QINV, QLO, QHI, K14}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };
        auto MEGA = mega ? build_mega() : distributed::MeshWorkload{};

        // NTT1_MEGA=1 for the 2-factor: the whole transform in one kernel, NO DRAM during it.
        // TWO group exchanges, because both its matmuls are non-tile-local (mt group of 8 for M1,
        // nt group of 4 for M2) where the 3-factor needed one corner turn.
        auto build_mega2 = [&]() {
            Program p = CreateProgram(); CoreRange cr = crange(C);
            mkcbr(p, cr, CBIndex::c_0,  tt::DataFormat::UInt32, 1, I32);                 // A tile
            mkcbr(p, cr, CBIndex::c_1,  tt::DataFormat::Int8, NB*NB*M1_Kt, I8);          // W256 slice
            mkcbr(p, cr, CBIndex::c_2,  tt::DataFormat::Int8, NB*NB*M2_Kt, I8);          // W128 slice
            mkcbr(p, cr, CBIndex::c_3,  tt::DataFormat::UInt32, 1, I32);                 // Tb
            mkcbr(p, cr, CBIndex::c_4,  tt::DataFormat::Int8, NB, I8);                   // my A limbs
            mkcbr(p, cr, CBIndex::c_5,  tt::DataFormat::Int8, 8*NB, I8);                 // gathered M1
            mkcbr(p, cr, CBIndex::c_6,  tt::DataFormat::Int8, NB, I8);                   // my S' limbs
            mkcbr(p, cr, CBIndex::c_7,  tt::DataFormat::Int8, 4*NB, I8);                 // gathered M2
            mkcbr(p, cr, CBIndex::c_16, tt::DataFormat::UInt32, 1, I32);                 // R
            mkcbr(p, cr, CBIndex::c_18, tt::DataFormat::UInt32, 1, I32);                 // S
            mkcbr(p, cr, CBIndex::c_19, tt::DataFormat::UInt32, 1, I32);                 // S'
            const uint32_t sem1 = CreateSemaphore(p, cr, 0), sem2 = CreateSemaphore(p, cr, 0);
            std::vector<uint32_t> rc;
            TensorAccessorArgs(*b_A).append_to(rc);        TensorAccessorArgs(*b_Wrhs256).append_to(rc);
            TensorAccessorArgs(*b_Wfold128).append_to(rc); TensorAccessorArgs(*b_Tb2).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/ntt32768_2f_fused_rd.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/ntt32768_2f_fused_xw.cpp", cr,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/ntt32768_2f_fused.cpp", cr,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                              .dst_full_sync_en = true, .defines = zdef});
            std::vector<uint32_t> coords;
            for (uint32_t j = 0; j < NTILE; ++j) { auto ph = mesh->worker_core_from_logical_core(cat(j));
                coords.push_back((uint32_t)ph.x); coords.push_back((uint32_t)ph.y); }
            for (uint32_t c = 0; c < C; ++c) { auto co = cat(c);
                SetRuntimeArgs(p, rd, co, {(uint32_t)b_A->address(), (uint32_t)b_Wrhs256->address(),
                                           (uint32_t)b_Wfold128->address(), (uint32_t)b_Tb2->address(), c});
                std::vector<uint32_t> wv = {(uint32_t)b_R->address(), c, sem1, sem2};
                wv.insert(wv.end(), coords.begin(), coords.end());
                SetRuntimeArgs(p, wr, co, wv);
                SetRuntimeArgs(p, cp, co, {(uint32_t)q, QINV, QLO, QHI, K14}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };
        auto MEGA2 = mega ? build_mega2() : distributed::MeshWorkload{};

        // fused triples: A1A2A3 -> one dispatch, C1C2C3 -> one, D1D2D3 -> one
        auto FA = fuse ? build_fused(b_Xin, b_WfoldA, b_G, true)  : distributed::MeshWorkload{};
        auto FC = fuse ? build_fused(b_Gp,  b_Wrhs,   b_H, false) : distributed::MeshWorkload{};
        auto FD = fuse ? build_fused(b_D,   b_WfoldA, b_E, true)  : distributed::MeshWorkload{};
        std::vector<distributed::MeshWorkload*> wls3;
        std::vector<const char*> nm3;
        if (!fuse) {
            wls3 = {&A1,&A2,&A3,&Bt,&C1,&C2,&C3,&TB,&A2A,&D1,&D2,&D3};
            nm3  = {"A1 dec","A2 mm","A3 rec","B tw","C1 dec","C2 mm",
                    "C3 rec","TB tw","A2A","D1 dec","D2 mm","D3 rec"};
        } else if (mega) {
            wls3 = {&MEGA};
            nm3  = {"NTT fused"};
        } else {
            wls3 = {&FA,&Bt,&FC,&TB,&A2A,&FD};
            nm3  = {"A fused","B tw","C fused","TB tw","A2A","D fused"};
        }

        uint32_t mism3 = 0;
        std::vector<uint32_t> Eraw;
        if (do3) {
        for (auto* wl : wls3) distributed::EnqueueMeshWorkload(cq, *wl, false);
        distributed::EnqueueReadMeshBuffer(cq, Eraw, b_E, true);
        { std::vector<uint32_t> out(N);
          for (uint32_t s = 0; s < T; ++s) {
              std::vector<uint32_t> tile(Eraw.begin() + s * ELEMS, Eraw.begin() + (s + 1) * ELEMS);
              auto Es = untilize(tile);
              for (uint32_t v = 0; v < T; ++v) for (uint32_t t = 0; t < T; ++t)
                  out[s + T * t + 1024 * v] = Es[v * T + t]; }
          for (uint32_t j = 0; j < N; ++j) if (out[j] != Xref3[j]) ++mism3;
          if (std::getenv("NTT1_SUM")) {   // FNV-1a over device buffers, to localise a corruption
              auto fnv = [](const std::vector<uint32_t>& v) { uint64_t h = 1469598103934665603ULL;
                  for (uint32_t x : v) { h ^= x; h *= 1099511628211ULL; } return h; };
              std::vector<uint32_t> Cb, Dd;
              distributed::EnqueueReadMeshBuffer(cq, Cb, b_Cb, true);
              distributed::EnqueueReadMeshBuffer(cq, Dd, b_D, true);
              fmt::print("  fnv1a  Cb(a2a IN) {:016x}   D(a2a OUT) {:016x}   E(final) {:016x}\n",
                         fnv(Cb), fnv(Dd), fnv(Eraw)); } }
        pass &= (mism3 == 0); }

        // ---- 2-factor pipeline ----
        auto S0 = build_decompose(b_A, b_Alimbs);
        auto S1 = build_matmul2(b_Wrhs256, b_Alimbs, b_Splanes, 1u, 4u, M1_Kt, M1_Nt);
        auto S2 = build_recombine(b_Splanes, b_S, OVERRIDE_KERNEL_PREFIX "compute/recombine_bigK.cpp");
        auto S3 = build_modmul(b_S, b_Tb2, b_Sp);
        auto S4 = build_decompose(b_Sp, b_Splimbs);
        auto S5 = build_matmul2(b_Wfold128, b_Splimbs, b_Rplanes, 0u, M2_Mt, M2_Kt, 8u);
        auto S6 = build_recombine(b_Rplanes, b_R, OVERRIDE_KERNEL_PREFIX "compute/recombine_bigK.cpp");
        // fused: S1+S2 -> one dispatch, S5+S6 -> one. 7 dispatches -> 5.
        auto FK1 = fuse ? build_fused_k(b_Wrhs256, b_Alimbs,  b_S, 1u, 4u, M1_Kt, M1_Nt)
                        : distributed::MeshWorkload{};
        auto FK2 = fuse ? build_fused_k(b_Wfold128, b_Splimbs, b_R, 0u, M2_Mt, M2_Kt, 8u)
                        : distributed::MeshWorkload{};
        std::vector<distributed::MeshWorkload*> wls2;
        std::vector<const char*> nm2;
        if (!fuse) {
            wls2 = {&S0,&S1,&S2,&S3,&S4,&S5,&S6};
            nm2  = {"dec A","M1 mm","rec1","twiddle","dec S'","M2 mm","rec2"};
        } else if (mega) {
            wls2 = {&MEGA2};
            nm2  = {"NTT fused"};
        } else {
            wls2 = {&S0,&FK1,&S3,&S4,&FK2};
            nm2  = {"dec A","M1 fused","twiddle","dec S'","M2 fused"};
        }

        uint32_t mism2 = 0;
        std::vector<uint32_t> Rraw;
        if (do2) {
        for (auto* wl : wls2) distributed::EnqueueMeshWorkload(cq, *wl, false);
        distributed::EnqueueReadMeshBuffer(cq, Rraw, b_R, true);
        { std::vector<uint32_t> out(N);
          for (uint32_t mt = 0; mt < 4; ++mt) for (uint32_t nt = 0; nt < 8; ++nt) {
              std::vector<uint32_t> tile(Rraw.begin() + (mt * 8 + nt) * ELEMS,
                                         Rraw.begin() + (mt * 8 + nt + 1) * ELEMS);
              auto blk = untilize(tile);
              for (uint32_t i = 0; i < T; ++i) for (uint32_t j = 0; j < T; ++j)
                  out[(nt * T + j) + N2 * (mt * T + i)] = blk[i * T + j]; }
          for (uint32_t k = 0; k < N; ++k) if (out[k] != Xref2[k]) ++mism2; }
        pass &= (mism2 == 0); }

        // Print "skipped", never "0/N", for a variant that did not run. mism3/mism2 are initialised to
        // 0, so an unconditional print would report a PASS for a check that never executed -- the same
        // failure mode as the PIR demo that once printed "recovered==DB[idx] OK" while checking nothing.
        if (do3) fmt::print("  correctness: 3-factor {}/{}", mism3, N);
        else     fmt::print("  correctness: 3-factor SKIPPED (NTT1_ONLY)");
        if (do2) fmt::print("   2-factor {}/{}\n", mism2, N);
        else     fmt::print("   2-factor SKIPPED (NTT1_ONLY)\n");
        std::fflush(stdout);

        // Under the profiler the capture must hold exactly ONE NTT, so stop here: the correctness
        // pass above issued precisely `wls.size()` dispatches and nothing else has run.
        if (prof) {
            const size_t nd = do3 ? wls3.size() : wls2.size();
            fmt::print("\n  PROFILED CAPTURE — one {} NTT, {} dispatches, nothing else.\n"
                       "  critical-path busy (the PURE figure):\n"
                       "    python3 tools/traces/busy_metrics.py \\\n"
                       "      \"$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv\" \\\n"
                       "      --expect {} --dataflow-ok\n",
                       do3 ? "3-factor" : "2-factor", nd, nd);
            std::fflush(stdout);
            pass &= mesh->close();
            if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
            return pass ? 0 : 1; }

        // ---- measure both, same protocol, warm ----
        auto r3 = measure("3-factor (32,32,32)", wls3, nm3, mism3);
        auto r2 = measure("2-factor (128,256)", wls2, nm2, mism2);
        report(r3); report(r2);

        fmt::print("\n  head to head (cores={}, modmul={}):\n", C, mm_kern);
        fmt::print("    host wall     3f {:7.1f} us   2f {:7.1f} us   3f/2f {:.4f}\n",
                   r3.wall_us, r2.wall_us, r3.wall_us / r2.wall_us);
        fmt::print("    stage-sum     3f {:7.1f} us   2f {:7.1f} us   3f/2f {:.4f}\n",
                   r3.stagesum_us, r2.stagesum_us, r3.stagesum_us / r2.stagesum_us);
        fmt::print("    dispatches    3f {:2d}          2f {:2d}          3f pays {:.2f}x the launches\n",
                   r3.dispatches, r2.dispatches, (double)r3.dispatches / r2.dispatches);

        if (const char* csv = std::getenv("NTT1_CSV")) {
            if (FILE* f = std::fopen(csv, "w")) {
                std::fprintf(f, "variant,cores,modmul,dispatches,mism,wall_us,stagesum_us\n");
                std::fprintf(f, "3factor,%u,%s,%u,%u,%.3f,%.3f\n", C, mm_kern.c_str(),
                             r3.dispatches, r3.mism, r3.wall_us, r3.stagesum_us);
                std::fprintf(f, "2factor,%u,%s,%u,%u,%.3f,%.3f\n", C, mm_kern.c_str(),
                             r2.dispatches, r2.mism, r2.wall_us, r2.stagesum_us);
                for (auto& [nm, us] : r3.stages) std::fprintf(f, "3factor_stage,%u,%s,,,%s,%.3f\n", C, mm_kern.c_str(), nm, us);
                for (auto& [nm, us] : r2.stages) std::fprintf(f, "2factor_stage,%u,%s,,,%s,%.3f\n", C, mm_kern.c_str(), nm, us);
                std::fclose(f); fmt::print("  wrote {}\n", csv); } }

        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
    std::fflush(stdout);
    return pass ? 0 : 1;
}
