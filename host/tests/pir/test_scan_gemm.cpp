// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Modular batched-GEMM first dimension on device, validated bit-exact against a host golden.
// This is the device counterpart of cpu/pir/check_scan_gemm.py: one coefficient slot, one RNS channel,
//     C (m x ncomp) = A (m x N0) @ B (N0 x ncomp)   (mod q)
// lowered to int8 radix-128 limbs, 16 limb-pair matmuls accumulated into 7 int32 planes, recombined
// mod q on host (the SFPU recombine tail is validated separately as recombine_rns).
//
// Deliberately small and single-core first: a hung compute kernel wedges the card until `tt-smi -r`,
// and this is the first new matmul kernel in the chain. Scale/shard only after this is green.
//
// Multi-slot: A is [slot][mt][kt][limb], B is [slot][kt][limb], output tiles enumerate (slot, m-tile).
// Each slot gets INDEPENDENT random operands, so a slot-indexing bug cannot pass by coincidence.
//
// Sharded: output tiles (slot, m-tile) are split across a core range; each core gets its own (o0, n_out).
// The compute kernel and reader are unchanged -- they already take o0/n_out -- so this is purely launch
// geometry, the same pattern as test_scan_mc.
//
//   ./build/test_scan_gemm [m=32] [N0=64] [ncomp=2] [seed] [nslots=1] [cores=1]

#include <cstdint>
#include <chrono>
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

constexpr uint32_t T = 32, ELEMS = T * T, I8 = ELEMS, I32 = ELEMS * 4;
constexpr uint32_t NL = 4, NPLANE = 2 * NL;   // 8: 7 used + 1 zero (recombine takes 4-tile groups)
constexpr uint32_t Q = 268369921;                       // 28-bit NTT-friendly prime

static inline int8_t limb(uint32_t v, uint32_t p) { return (int8_t)((v >> (7 * p)) & 0x7F); }

// 32x32 row-major -> tt tile order (same mapping as the engine's tilize32)
template <typename E>
static std::vector<E> tilize32(const std::vector<E>& in) {
    std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r)
        for (uint32_t c = 0; c < T; ++c)
            o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = in[r * T + c];
    return o;
}
template <typename E>
static std::vector<E> untilize32(const std::vector<E>& in) {
    std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r)
        for (uint32_t c = 0; c < T; ++c)
            o[r * T + c] = in[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)];
    return o;
}

int main(int argc, char** argv) {
    const uint32_t m     = (argc > 1) ? std::stoul(argv[1]) : 32;
    const uint32_t N0    = (argc > 2) ? std::stoul(argv[2]) : 64;
    const uint32_t ncomp = (argc > 3) ? std::stoul(argv[3]) : 2;
    const uint32_t seed  = (argc > 4) ? std::stoul(argv[4]) : 1;
    const uint32_t NS    = (argc > 5) ? std::stoul(argv[5]) : 1;   // coefficient slots (GEMM batch)
    const uint32_t NC    = (argc > 6) ? std::stoul(argv[6]) : 1;   // cores to shard the output over
    if (m % T || N0 % T || ncomp > T) { fmt::print(stderr, "need m%32==0, N0%32==0, ncomp<=32\n"); return 2; }
    const uint32_t MT = m / T, KB = N0 / T, OT = MT * NS;   // one n-tile (ncomp padded into 32 cols)

    bool pass = true;
    try {
        auto mesh = Dp::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        Dp::MeshCoordinateRange dr(mesh->shape());
        CoreCoord core{0, 0};
        const auto grid = mesh->compute_with_storage_grid_size();

        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> di(0, Q - 1);

        // A (m x N0) and B (N0 x ncomp), NON-CONSTANT: the packed-DB regime.
        std::vector<std::vector<std::vector<uint32_t>>> A(NS, std::vector<std::vector<uint32_t>>(m, std::vector<uint32_t>(N0)));
        std::vector<std::vector<std::vector<uint32_t>>> B(NS, std::vector<std::vector<uint32_t>>(N0, std::vector<uint32_t>(ncomp)));
        for (auto& sl : A) for (auto& row : sl) for (auto& v : row) v = di(rng);
        for (auto& sl : B) for (auto& row : sl) for (auto& v : row) v = di(rng);

        // ---- host golden: C = A @ B (mod q) ----
        std::vector<std::vector<std::vector<uint64_t>>> ref(NS, std::vector<std::vector<uint64_t>>(m, std::vector<uint64_t>(ncomp, 0)));
        for (uint32_t sl = 0; sl < NS; ++sl)
            for (uint32_t r = 0; r < m; ++r)
                for (uint32_t c = 0; c < ncomp; ++c) {
                    unsigned __int128 s = 0;
                    for (uint32_t k = 0; k < N0; ++k) s += (unsigned __int128)A[sl][r][k] * B[sl][k][c];
                    ref[sl][r][c] = (uint64_t)(s % Q);
                }

        // ---- limb tiles. A: [mt][kt][limb], B: [kt][limb] (one n-tile) ----
        std::vector<int8_t> Ablob((size_t)NS * MT * KB * NL * ELEMS), Bblob((size_t)NS * KB * NL * ELEMS);
        for (uint32_t sl = 0; sl < NS; ++sl)
        for (uint32_t mt = 0; mt < MT; ++mt)
            for (uint32_t kt = 0; kt < KB; ++kt)
                for (uint32_t l = 0; l < NL; ++l) {
                    std::vector<int8_t> nat(ELEMS, 0);
                    for (uint32_t i = 0; i < T; ++i)
                        for (uint32_t j = 0; j < T; ++j)
                            nat[i * T + j] = limb(A[sl][mt * T + i][kt * T + j], l);
                    auto tl = tilize32(nat);
                    std::copy(tl.begin(), tl.end(),
                              Ablob.begin() + ((((size_t)sl * MT + mt) * KB + kt) * NL + l) * ELEMS);
                }
        for (uint32_t sl = 0; sl < NS; ++sl)
        for (uint32_t kt = 0; kt < KB; ++kt)
            for (uint32_t l = 0; l < NL; ++l) {
                std::vector<int8_t> nat(ELEMS, 0);                 // cols >= ncomp stay zero (padding)
                for (uint32_t i = 0; i < T; ++i)
                    for (uint32_t c = 0; c < ncomp; ++c)
                        nat[i * T + c] = limb(B[sl][kt * T + i][c], l);
                auto tl = tilize32(nat);
                std::copy(tl.begin(), tl.end(), Bblob.begin() + (((size_t)sl * KB + kt) * NL + l) * ELEMS);
            }

        auto mkbuf = [&](uint32_t nt, uint32_t pg) {
            Dp::DeviceLocalBufferConfig d{.page_size = pg, .buffer_type = BufferType::DRAM};
            return Dp::MeshBuffer::create(Dp::ReplicatedBufferConfig{.size = nt * pg}, d, mesh.get()); };
        auto b_A = mkbuf(NS * MT * KB * NL, I8), b_B = mkbuf(NS * KB * NL, I8), b_out = mkbuf(OT * NPLANE, I32);
        Dp::EnqueueWriteMeshBuffer(cq, b_A, Ablob, false);
        Dp::EnqueueWriteMeshBuffer(cq, b_B, Bblob, false);

        if (OT % NC != 0) { fmt::print(stderr, "OT={} must be divisible by cores={}\n", OT, NC); return 2; }
        const uint32_t GX = std::min(NC, (uint32_t)grid.x), GY = (NC + GX - 1) / GX;
        if (GY > (uint32_t)grid.y) { fmt::print(stderr, "cores={} exceeds grid {}x{}\n", NC, grid.x, grid.y); return 2; }
        CoreRange cores(CoreCoord{0, 0}, CoreCoord{GX - 1, GY - 1});
        const uint32_t n_per = OT / NC;

        Program p = CreateProgram();
        auto cb = [&](CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t pg) {
            CircularBufferConfig c(depth * pg, {{idx, fmt}}); c.set_page_size(idx, pg);
            CreateCircularBuffer(p, cores, c); };
        cb(CBIndex::c_0, tt::DataFormat::Int8, 2 * NL, I8);
        cb(CBIndex::c_1, tt::DataFormat::Int8, KB * NL, I8);   // whole slot's B resident in L1
        cb(CBIndex::c_16, tt::DataFormat::Int32, 2 * NPLANE, I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*b_A).append_to(rc); TensorAccessorArgs(*b_B).append_to(rc);
        auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_scan_gemm.cpp", cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
        auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/scan_gemm.cpp", cores,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true});
        for (uint32_t ci = 0; ci < NC; ++ci) {
            const uint32_t gx = ci % GX, gy = ci / GX, o0 = ci * n_per;
            CoreCoord cc{gx, gy};
            SetRuntimeArgs(p, rd, cc, {(uint32_t)b_A->address(), (uint32_t)b_B->address(), o0, n_per, KB, MT});
            SetRuntimeArgs(p, wr, cc, {(uint32_t)b_out->address(), n_per * NPLANE, o0 * NPLANE});
            SetRuntimeArgs(p, cp, cc, {n_per, KB, MT});
        }
        Dp::MeshWorkload wl; wl.add_program(dr, std::move(p));
        auto t0 = std::chrono::high_resolution_clock::now();
        Dp::EnqueueMeshWorkload(cq, wl, false);
        Dp::Finish(cq);
        const double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

        std::vector<uint32_t> raw; Dp::EnqueueReadMeshBuffer(cq, raw, b_out, true);

        // ---- host recombine: C = Σ_d (128^d mod q) · plane_d  (mod q) ----
        std::vector<uint64_t> w(NPLANE);
        { uint64_t acc = 1; for (uint32_t d = 0; d < NPLANE; ++d) { w[d] = acc; acc = acc * 128 % Q; } }
        // plane 7 is zero, so its weight contributes nothing regardless
        uint32_t mism = 0, checked = 0;
        for (uint32_t ot = 0; ot < OT; ++ot) {
            const uint32_t sl = ot / MT, mt = ot % MT;
            std::vector<std::vector<int32_t>> pl(NPLANE, std::vector<int32_t>(ELEMS));
            for (uint32_t d = 0; d < NPLANE; ++d) {
                std::vector<uint32_t> t(raw.begin() + ((size_t)ot * NPLANE + d) * ELEMS,
                                       raw.begin() + ((size_t)ot * NPLANE + d + 1) * ELEMS);
                auto nat = untilize32(t);
                for (uint32_t x = 0; x < ELEMS; ++x) pl[d][x] = (int32_t)nat[x];
            }
            for (uint32_t i = 0; i < T; ++i)
                for (uint32_t c = 0; c < ncomp; ++c) {
                    unsigned __int128 s = 0;
                    for (uint32_t d = 0; d < NPLANE; ++d) s += (unsigned __int128)w[d] * (uint64_t)pl[d][i * T + c];
                    uint64_t got = (uint64_t)(s % Q);
                    if (got != ref[sl][mt * T + i][c]) {
                        if (mism < 6) fmt::print("  MISMATCH slot={} r={} c={} got={} exp={}\n", sl, mt * T + i, c, got, ref[sl][mt * T + i][c]);
                        ++mism; }
                    ++checked; }
        }
        // int8 A-operand bytes streamed (the DB side) and useful modular MACs
        const double a_bytes = (double)NS * MT * KB * NL * ELEMS;
        const double umacs = (double)NS * m * N0 * ncomp;
        fmt::print("\nmodular batched GEMM (m={} N0={} ncomp={} slots={} cores={} q={}): mismatches={}/{}\n",
                   m, N0, ncomp, NS, NC, Q, mism, checked);
        fmt::print("  time={:.3f} ms   A-stream={:.2f} GB/s   useful modmul={:.2f} G/s\n",
                   secs * 1e3, a_bytes / secs / 1e9, umacs / secs / 1e9);
        pass = (mism == 0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
