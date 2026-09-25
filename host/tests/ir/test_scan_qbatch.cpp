// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// QUERY-BATCHED scan matmul — batching at the DATA level (docs/pir/SCAN_ALGORITHM.md).  B independent
// queries SHARE the database (the LHS), so they pack into the output COLUMNS: per slot j,
//   A_j = DB_jᵀ · [Q_j^0 | Q_j^1 | … | Q_j^{B-1}]  =  [rest,N0]·[N0, 2B]  →  [rest, 2B]  (mod q).
// The c=2 output columns become 2B — filling the 32-wide tile (B≤16 → one tile).  The COMPUTE KERNEL
// IS UNCHANGED (`matmul_modq` + `add_modq`, contract N0=KB·32 via running add); only the Q tile has
// more non-zero columns.  Bit-exact vs a host __int128 golden.  (For 2B>32: split into N-tiles, same
// kernel — not exercised here.)
//
//   ./build/test_scan_qbatch [N0] [rest] [B] [nslots] [q] [seed]     (2B ≤ 32, rest ≤ 32)

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
    const uint32_t N0   = (argc > 1) ? std::stoul(argv[1]) : 512;
    const uint32_t rest = (argc > 2) ? std::stoul(argv[2]) : 32;
    const uint32_t B    = (argc > 3) ? std::stoul(argv[3]) : 16;    // queries packed into columns
    const uint32_t NS   = (argc > 4) ? std::stoul(argv[4]) : 4;
    const uint64_t q    = (argc > 5) ? std::stoull(argv[5]) : 167772161ULL;
    const uint32_t seed = (argc > 6) ? std::stoul(argv[6]) : 1;
    const uint32_t W = 2 * B;                                       // output width = 2·B (all queries)
    if (rest > T) { fmt::print("rest must be ≤ 32\n"); return 2; }
    const uint32_t KB = (N0 + T - 1) / T, NT = (W + T - 1) / T;     // K-tiles, N-tiles (32-col blocks)
    mont::MontgomeryContext ctx((uint32_t)q);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)q - 1);
    auto limb = [](uint32_t v, uint32_t p) -> int8_t { return (int8_t)((v >> (7 * p)) & 127); };

    // DB[s][k][r] shared across queries; Q[s][b][k][comp]; golden Ag[s][r][b·2+comp]
    std::vector<std::vector<std::vector<uint32_t>>> DB(NS, std::vector<std::vector<uint32_t>>(N0, std::vector<uint32_t>(rest)));
    std::vector<std::vector<std::vector<std::vector<uint32_t>>>> Q(NS, std::vector<std::vector<std::vector<uint32_t>>>(B, std::vector<std::vector<uint32_t>>(N0, std::vector<uint32_t>(2))));
    std::vector<std::vector<std::vector<uint32_t>>> Ag(NS, std::vector<std::vector<uint32_t>>(rest, std::vector<uint32_t>(W)));
    for (uint32_t s = 0; s < NS; ++s) {
        for (uint32_t k = 0; k < N0; ++k) for (uint32_t r = 0; r < rest; ++r) DB[s][k][r] = dist(rng);
        for (uint32_t b = 0; b < B; ++b) for (uint32_t k = 0; k < N0; ++k) for (uint32_t c = 0; c < 2; ++c) Q[s][b][k][c] = dist(rng);
        for (uint32_t r = 0; r < rest; ++r) for (uint32_t b = 0; b < B; ++b) for (uint32_t c = 0; c < 2; ++c) {
            unsigned __int128 a = 0; for (uint32_t k = 0; k < N0; ++k) a += (unsigned __int128)DB[s][k][r] * Q[s][b][k][c];
            Ag[s][r][b * 2 + c] = (uint32_t)(a % q); } }

    bool pass = true;
    try {
        auto md = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = md->mesh_command_queue();
        distributed::MeshCoordinateRange dr(md->shape());
        constexpr CoreCoord core{0, 0};
        auto mkbuf = [&](uint32_t n, uint32_t page) {
            distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = n * page}, d, md.get()); };
        auto cb = [&](CBIndex idx, DataFormat fmt, uint32_t depth, uint32_t pg) {
            CircularBufferConfig c(depth * pg, {{idx, fmt}}); c.set_page_size(idx, pg); return c; };

        auto run_matmul = [&](Buf b_X, Buf b_W, Buf b_C) {   // C = (Wfold @ Xin) mod q, one 32×32 tile
            Program p = CreateProgram();
            CreateCircularBuffer(p, core, cb(CBIndex::c_0, DataFormat::UInt32, 2, I32));
            CreateCircularBuffer(p, core, cb(CBIndex::c_1, DataFormat::Int8, 2, I8));
            CreateCircularBuffer(p, core, cb(CBIndex::c_2, DataFormat::Int8, NB, I8));
            CreateCircularBuffer(p, core, cb(CBIndex::c_3, DataFormat::Int32, NB, I32));
            CreateCircularBuffer(p, core, cb(CBIndex::c_16, DataFormat::UInt32, 2, I32));
            std::vector<uint32_t> rc; TensorAccessorArgs(*b_X).append_to(rc); TensorAccessorArgs(*b_W).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/matmul_modq.cpp", core,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)b_X->address(), (uint32_t)b_W->address()});
            SetRuntimeArgs(p, wr, core, {(uint32_t)b_C->address(), 1u, 0u});
            SetRuntimeArgs(p, cp, core, {(uint32_t)q, ctx.q_inv_32(), (uint32_t)(q & 0xFFFF), (uint32_t)(q >> 16), ctx.to_montgomery(16384u)});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false); };
        auto run_add = [&](Buf b_a, Buf b_b, Buf b_out) {    // (a+b) mod q
            Program p = CreateProgram();
            CreateCircularBuffer(p, core, cb(CBIndex::c_0, DataFormat::UInt32, 2, I32));
            CreateCircularBuffer(p, core, cb(CBIndex::c_1, DataFormat::UInt32, 2, I32));
            CreateCircularBuffer(p, core, cb(CBIndex::c_16, DataFormat::UInt32, 2, I32));
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_binop.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/add_modq.cpp", core,
                ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
            SetRuntimeArgs(p, rd, core, {(uint32_t)b_a->address(), (uint32_t)b_b->address(), 1u, 0u});
            SetRuntimeArgs(p, wr, core, {(uint32_t)b_out->address(), 1u, 0u});
            SetRuntimeArgs(p, cp, core, {1u, (uint32_t)q});
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); distributed::EnqueueMeshWorkload(cq, wl, false); };

        uint32_t mism = 0, checked = 0;
        std::vector<uint32_t> zero(ELEMS, 0);
        for (uint32_t s = 0; s < NS; ++s) for (uint32_t nt = 0; nt < NT; ++nt) {   // split 2B into 32-col N-tiles
            Buf b_acc = mkbuf(1, I32); distributed::EnqueueWriteMeshBuffer(cq, b_acc, zero, false);
            for (uint32_t kt = 0; kt < KB; ++kt) {
                // A_kt = DBᵀ block [rest, 32]; Q_kt = this N-tile's 32 columns (global col gc = nt·32 + lc = b·2+c)
                std::vector<uint32_t> A(ELEMS, 0), Bm(ELEMS, 0);
                for (uint32_t i = 0; i < T; ++i) for (uint32_t m = 0; m < T; ++m) {
                    uint32_t k = kt * T + m; if (i < rest && k < N0) A[i * T + m] = DB[s][k][i]; }
                for (uint32_t m = 0; m < T; ++m) { uint32_t k = kt * T + m; if (k >= N0) continue;
                    for (uint32_t lc = 0; lc < T; ++lc) { uint32_t gc = nt * T + lc; if (gc >= W) continue;
                        Bm[m * T + lc] = Q[s][gc / 2][k][gc % 2]; } }
                std::vector<int8_t> Afold(NB * NB * ELEMS);
                for (uint32_t p = 0; p < NB; ++p) for (uint32_t rr = 0; rr < NB; ++rr) {
                    std::vector<int8_t> l(ELEMS);
                    for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[rr] * A[i]) % q), p);
                    auto t = tilize(l); std::copy(t.begin(), t.end(), Afold.begin() + (p * NB + rr) * ELEMS); }
                Buf b_X = mkbuf(1, I32), b_W = mkbuf(NB * NB, I8), b_par = mkbuf(1, I32), b_acc2 = mkbuf(1, I32);
                distributed::EnqueueWriteMeshBuffer(cq, b_X, tilize(Bm), false);
                distributed::EnqueueWriteMeshBuffer(cq, b_W, Afold, false);
                run_matmul(b_X, b_W, b_par);
                run_add(b_acc, b_par, b_acc2);
                b_acc = b_acc2;
            }
            std::vector<uint32_t> Craw; distributed::EnqueueReadMeshBuffer(cq, Craw, b_acc, true);
            auto Cm = untilize(Craw);
            for (uint32_t r = 0; r < rest; ++r) for (uint32_t lc = 0; lc < T; ++lc) { uint32_t w = nt * T + lc; if (w >= W) continue;
                if (Cm[r * T + lc] != Ag[s][r][w]) { if (mism < 8)
                    fmt::print("  slot{} nt{} r{} col{}(q{} c{}): got {} exp {}\n", s, nt, r, w, w / 2, w % 2, Cm[r * T + lc], Ag[s][r][w]); ++mism; }
                ++checked; }
        }
        fmt::print("query-batched scan (device)  N0={} rest={} B={} (width 2B={}) KB={} slots={} q={}  mismatches={}/{}\n",
                   N0, rest, B, W, KB, NS, q, mism, checked);
        pass = (mism == 0);
        pass &= md->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
