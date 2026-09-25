// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FULL FIRST-DIM K-TILED scan on the fast path.  Contracts N0 = KB·32 for OT independent output tiles
// by running KB passes: each pass = fast M-batched modular matmul (matmul_modq_mbatch, one k-tile of
// every output tile) + batched running add (add_modq).  Data is laid out K-TILE-MAJOR (b_X[kt·OT+o],
// b_W[(kt·OT+o)·16]) so each pass reads a contiguous OT-tile block → reader_mbatch/reader_binop need no
// striding.  Both the matmul and the add are multi-core sharded.  Per k-tile a core does M output
// tiles; total dispatches = KB·2·ceil(OT/(C·M)).  Validated bit-exact vs an __int128 golden that
// contracts all KB·32 = N0.  All kernels are the validated ones (mbatch + add_modq).
//
//   ./build/test_scan_kt [OT] [KB] [M] [q] [seed]

#include <chrono>
#include <cstdint>
#include <random>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#error "OVERRIDE_KERNEL_PREFIX must be set by CMake"
#endif

namespace {
constexpr uint32_t T = 32, NB = 4, ELEMS = 1024, I8 = ELEMS, I32 = ELEMS * 4;
const uint64_t POW[4] = {1, 128, 16384, 2097152};
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
using Buf = std::shared_ptr<distributed::MeshBuffer>;
}  // namespace

int main(int argc, char** argv) {
    const uint32_t OT = (argc > 1) ? std::stoul(argv[1]) : 2048;   // output tiles (contract N0 each)
    const uint32_t KB = (argc > 2) ? std::stoul(argv[2]) : 16;     // k-tiles → N0 = KB·32
    const uint32_t M  = (argc > 3) ? std::stoul(argv[3]) : 32;     // mbatch tiles/core/program
    const uint64_t q  = (argc > 4) ? std::stoull(argv[4]) : 167772161ULL;
    const uint32_t seed = (argc > 5) ? std::stoul(argv[5]) : 1;
    mont::MontgomeryContext ctx((uint32_t)q);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> di(0, (uint32_t)q - 1);
    auto limb = [](uint32_t v, uint32_t p) -> int8_t { return (int8_t)((v >> (7 * p)) & 127); };

    // OT output tiles, each contracting KB·32 = N0.  Layout K-TILE-MAJOR: [kt][o].
    std::vector<uint32_t> Xin((size_t)KB * OT * ELEMS);
    std::vector<int8_t>   Wfold((size_t)KB * OT * NB * NB * ELEMS);
    std::vector<std::vector<uint32_t>> Cg(OT, std::vector<uint32_t>(ELEMS, 0));
    std::vector<std::vector<unsigned __int128>> accw(OT, std::vector<unsigned __int128>(ELEMS, 0));
    for (uint32_t kt = 0; kt < KB; ++kt) for (uint32_t o = 0; o < OT; ++o) {
        std::vector<uint32_t> A(ELEMS), B(ELEMS);
        for (auto& v : A) v = di(rng); for (auto& v : B) v = di(rng);
        for (uint32_t i = 0; i < T; ++i) for (uint32_t j = 0; j < T; ++j) {
            unsigned __int128 a = 0; for (uint32_t k = 0; k < T; ++k) a += (unsigned __int128)A[i * T + k] * B[k * T + j];
            accw[o][i * T + j] += a; }
        auto Bt = tilize(B); std::copy(Bt.begin(), Bt.end(), Xin.begin() + ((size_t)kt * OT + o) * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t rr = 0; rr < NB; ++rr) {
            std::vector<int8_t> l(ELEMS);
            for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[rr] * A[i]) % q), p);
            auto t = tilize(l);
            std::copy(t.begin(), t.end(), Wfold.begin() + (((size_t)kt * OT + o) * NB * NB + p * NB + rr) * ELEMS); }
    }
    for (uint32_t o = 0; o < OT; ++o) for (uint32_t i = 0; i < ELEMS; ++i) Cg[o][i] = (uint32_t)(accw[o][i] % q);

    bool pass = true;
    try {
        auto md = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = md->mesh_command_queue();
        distributed::MeshCoordinateRange dr(md->shape());
        auto grid = md->compute_with_storage_grid_size();
        const uint32_t GX = grid.x, GY = grid.y;
        fmt::print("grid {}x{} = {} cores; OT={} KB={} (N0={}) M={} q={}\n", GX, GY, GX * GY, OT, KB, KB * T, M, q);
        auto mkbuf = [&](size_t n, uint32_t page) {
            distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = n * page}, d, md.get()); };
        auto b_X = mkbuf((size_t)KB * OT, I32), b_W = mkbuf((size_t)KB * OT * NB * NB, I8);
        auto b_acc0 = mkbuf(OT, I32), b_acc1 = mkbuf(OT, I32), b_par = mkbuf(OT, I32);
        distributed::EnqueueWriteMeshBuffer(cq, b_X, Xin, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_W, Wfold, false);
        std::vector<uint32_t> zero((size_t)OT * ELEMS, 0);
        distributed::EnqueueWriteMeshBuffer(cq, b_acc0, zero, false);

        auto cbf = [&](Program& p, CoreRange cores, CBIndex idx, DataFormat fmt, uint32_t depth, uint32_t pg) {
            CircularBufferConfig c(depth * pg, {{idx, fmt}}); c.set_page_size(idx, pg); CreateCircularBuffer(p, cores, c); };

        // one mbatch pass over OT tiles for k-tile kt: b_par[o] = (Wfold[kt][o] @ Xin[kt][o]) mod q
        auto mbatch_pass = [&](uint32_t gx, uint32_t gy, uint32_t Mb, uint32_t kt) {
            const uint32_t C = gx * gy, per = C * Mb;
            for (uint32_t base = 0; base < OT; base += per) {
                Program p = CreateProgram();
                CoreRange cores(CoreCoord{0, 0}, CoreCoord{gx - 1, gy - 1});
                cbf(p, cores, CBIndex::c_0, DataFormat::UInt32, Mb, I32); cbf(p, cores, CBIndex::c_1, DataFormat::Int8, 32, I8);
                cbf(p, cores, CBIndex::c_2, DataFormat::Int8, Mb * NB, I8); cbf(p, cores, CBIndex::c_3, DataFormat::Int32, Mb * NB, I32);
                cbf(p, cores, CBIndex::c_16, DataFormat::UInt32, Mb, I32);
                std::vector<uint32_t> rc; TensorAccessorArgs(*b_X).append_to(rc); TensorAccessorArgs(*b_W).append_to(rc);
                auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_mbatch.cpp", cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
                auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
                auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/matmul_modq_mbatch.cpp", cores,
                    ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
                for (uint32_t gyi = 0; gyi < gy; ++gyi) for (uint32_t gxi = 0; gxi < gx; ++gxi) {
                    uint32_t idx = gyi * gx + gxi, o_base = base + idx * Mb;
                    if (o_base >= OT) continue;
                    uint32_t m_here = std::min(Mb, OT - o_base);
                    uint32_t x_base = kt * OT + o_base;                 // k-tile-major
                    CoreCoord c{gxi, gyi};
                    SetRuntimeArgs(p, rd, c, {(uint32_t)b_X->address(), (uint32_t)b_W->address(), x_base, x_base * NB * NB, m_here});
                    SetRuntimeArgs(p, wr, c, {(uint32_t)b_par->address(), m_here, o_base});
                    SetRuntimeArgs(p, cp, c, {(uint32_t)q, ctx.q_inv_32(), (uint32_t)(q & 0xFFFF), (uint32_t)(q >> 16), ctx.to_montgomery(16384u), m_here});
                }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false);
            }
        };
        // batched add: acc_out[o] = (acc_in[o] + par[o]) mod q, sharded over OT tiles
        auto add_pass = [&](uint32_t gx, uint32_t gy, uint32_t Mb, Buf acc_in, Buf acc_out) {
            const uint32_t C = gx * gy, per = C * Mb;
            for (uint32_t base = 0; base < OT; base += per) {
                Program p = CreateProgram();
                CoreRange cores(CoreCoord{0, 0}, CoreCoord{gx - 1, gy - 1});
                cbf(p, cores, CBIndex::c_0, DataFormat::UInt32, 2, I32); cbf(p, cores, CBIndex::c_1, DataFormat::UInt32, 2, I32);
                cbf(p, cores, CBIndex::c_16, DataFormat::UInt32, 2, I32);
                auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_binop.cpp", cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
                auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
                auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/add_modq.cpp", cores,
                    ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
                for (uint32_t gyi = 0; gyi < gy; ++gyi) for (uint32_t gxi = 0; gxi < gx; ++gxi) {
                    uint32_t idx = gyi * gx + gxi, o_base = base + idx * Mb;
                    if (o_base >= OT) continue;
                    uint32_t m_here = std::min(Mb, OT - o_base);
                    CoreCoord c{gxi, gyi};
                    SetRuntimeArgs(p, rd, c, {(uint32_t)acc_in->address(), (uint32_t)b_par->address(), m_here, o_base});
                    SetRuntimeArgs(p, wr, c, {(uint32_t)acc_out->address(), m_here, o_base});
                    SetRuntimeArgs(p, cp, c, {m_here, (uint32_t)q});
                }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false);
            }
        };

        auto run = [&](uint32_t gx, uint32_t gy, uint32_t Mb) -> double {
            auto t0 = std::chrono::high_resolution_clock::now();
            Buf a = b_acc0, b = b_acc1;
            for (uint32_t kt = 0; kt < KB; ++kt) { mbatch_pass(gx, gy, Mb, kt); add_pass(gx, gy, Mb, a, b); std::swap(a, b); }
            distributed::Finish(cq);
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        run(1, 1, M);  // warm + produces a result in b_acc0 (KB even → ends back in acc0)
        Buf final_acc = (KB % 2 == 0) ? b_acc0 : b_acc1;
        std::vector<uint32_t> out; distributed::EnqueueReadMeshBuffer(cq, out, final_acc, true);
        uint32_t mism = 0;
        for (uint32_t o = 0; o < OT; ++o) { auto Cm = untilize(std::vector<uint32_t>(out.begin() + (size_t)o * ELEMS, out.begin() + (size_t)(o + 1) * ELEMS));
            for (uint32_t i = 0; i < ELEMS; ++i) if (Cm[i] != Cg[o][i]) ++mism; }
        fmt::print("K-tiled scan validation: {}  ({} bad)\n", mism == 0 ? "OK" : "FAILED", mism);
        pass &= (mism == 0);

        // throughput: count in output tiles (each contracts N0 = KB·32) and in k-tile-matmuls.
        fmt::print("\n cores | gx×gy | time(ms) | out-tiles/s | ktile-mm/s | speedup\n"
                   "-------+-------+----------+-------------+------------+--------\n");
        struct Cfg { uint32_t gx, gy; }; std::vector<Cfg> cfgs = {{1,1},{8,1},{8,2},{8,4},{8,8}};
        double t1c = 0;
        for (auto cf : cfgs) { if (cf.gx > GX || cf.gy > GY) continue; uint32_t C = cf.gx * cf.gy;
            run(cf.gx, cf.gy, M); double best = 1e18; for (int it = 0; it < 3; ++it) best = std::min(best, run(cf.gx, cf.gy, M));
            if (C == 1) t1c = best;
            fmt::print(" {:4}  | {:2}×{:<2} | {:8.2f} | {:11.0f} | {:10.0f} | {:5.2f}×\n",
                       C, cf.gx, cf.gy, best, OT / best * 1e3, (double)OT * KB / best * 1e3, t1c > 0 ? t1c / best : 1.0); }
        pass &= md->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
    return pass ? 0 : 1;
}
