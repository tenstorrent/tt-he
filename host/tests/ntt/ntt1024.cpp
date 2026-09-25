// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Size-1024 NTT on a SINGLE Tensix core, factored as N = 32*32 (Cooley-Tukey
// four-step), built from the repo's two basic kernels:
//
//   stage A:  G  = W  @ Xin     int8 modular matmul   (kernels/shared/compute/matmul_compute.cpp)
//   stage B:  G' = G  (.) T      Montgomery modmul     (kernels/shared/compute/modmul_compute.cpp)
//   stage C:  H  = G' @ W        int8 modular matmul
//   out[32*j1 + j2] = H[j2, j1]  (host reindex; 2-factor => transpose is free)
//
// All three compute kernels run on core {0,0}. q = 167772161 (5*2^25+1), the
// NTT-friendly 28-bit prime from algorithm/ntt (fits the matmul's radix-128 /
// 4-int8-limb scheme and the Montgomery modmul's <2^28 requirement).
//
// Math (n = n1 + 32*n2, j = j2 + 32*j1, w = prim. 1024th root, w32 = w^32):
//   W[a,b]    = w32^{a*b} mod q            (32x32, symmetric; used both matmuls)
//   Xin[n2,n1]= x[n1 + 32*n2]
//   G  = W @ Xin                           G[j2,n1] = sum_n2 w32^{j2*n2} x[..]
//   T[j2,n1]  = w^{j2*n1} mod q            inter-stage twiddle
//   G' = G (.) T
//   H  = G' @ W                            H[j2,j1] = sum_n1 G'[j2,n1] w32^{n1*j1}
//   X[j2 + 32*j1] = H[j2,j1]
//
// Twiddle trick: the modmul kernel returns REDC(a*b) = a*b*R^-1 mod q. Feeding
// G in the normal domain and T pre-converted to Montgomery (T*R) yields
// G*T*R*R^-1 = G*T mod q in the normal domain -- no domain change of live data.
//
// Run:   ./build/ntt1024 [q] [seed]      (defaults: 167772161, 0)

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
using namespace tt::constants;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {

constexpr uint32_t T = 32, NB = 4, BLK = 8, NLIMB = 4;
constexpr uint64_t BASE = 128;
constexpr uint32_t ELEMS = T * T;              // 1024
const uint64_t POW[NLIMB] = {1, 128, 16384, 2097152};

// ---- number theory (matches reference/ntt/ntt_matrix.py) ----
uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r = 1; b %= m;
    while (e) { if (e & 1) r = (__uint128_t)r * b % m; b = (__uint128_t)b * b % m; e >>= 1; }
    return r;
}
uint64_t primitive_root(uint64_t q) {
    uint64_t phi = q - 1, n = phi;
    std::vector<uint64_t> fac;
    for (uint64_t d = 2; d * d <= n; ++d)
        if (n % d == 0) { fac.push_back(d); while (n % d == 0) n /= d; }
    if (n > 1) fac.push_back(n);
    for (uint64_t g = 2; g < q; ++g) {
        bool ok = true;
        for (uint64_t p : fac) if (modpow(g, phi / p, q) == 1) { ok = false; break; }
        if (ok) return g;
    }
    throw std::runtime_error("no primitive root");
}

// ---- TT 32x32 tile (2x2 faces of 16x16) <-> row-major ----
template <typename E>
std::vector<E> tilize(const std::vector<E>& in) {
    std::vector<E> out(ELEMS);
    for (uint32_t r = 0; r < T; ++r)
        for (uint32_t c = 0; c < T; ++c)
            out[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = in[r * T + c];
    return out;
}
template <typename E>
std::vector<E> untilize(const std::vector<E>& in) {
    std::vector<E> out(ELEMS);
    for (uint32_t r = 0; r < T; ++r)
        for (uint32_t c = 0; c < T; ++c)
            out[r * T + c] = in[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)];
    return out;
}

// ---- matmul operand prep (radix-128 limbs), from device/modular_matmul.cpp ----
std::vector<int8_t> build_lhs(const std::vector<uint32_t>& A) {          // NB tiles
    std::vector<int8_t> v(NB * ELEMS, 0);
    for (uint32_t bk = 0; bk < NB; ++bk) {
        std::vector<int8_t> log(ELEMS, 0);
        for (uint32_t i = 0; i < T; ++i)
            for (uint32_t kap = 0; kap < BLK; ++kap) {
                uint32_t a = A[i * T + (bk * BLK + kap)];
                for (uint32_t p = 0; p < NLIMB; ++p)
                    log[i * T + (kap * NLIMB + p)] = int8_t((a / POW[p]) % BASE);
            }
        auto t = tilize(log);
        std::copy(t.begin(), t.end(), v.begin() + bk * ELEMS);
    }
    return v;
}
std::vector<int8_t> build_rhs(const std::vector<uint32_t>& B, uint64_t q) {  // NB*NB tiles, idx bj*NB+bk
    std::vector<int8_t> v(NB * NB * ELEMS, 0);
    for (uint32_t bk = 0; bk < NB; ++bk)
        for (uint32_t bj = 0; bj < NB; ++bj) {
            std::vector<int8_t> log(ELEMS, 0);
            for (uint32_t kap = 0; kap < BLK; ++kap)
                for (uint32_t js = 0; js < BLK; ++js) {
                    uint64_t Bkj = B[(bk * BLK + kap) * T + (bj * BLK + js)];
                    for (uint32_t p = 0; p < NLIMB; ++p) {
                        uint64_t Bp = (POW[p] * Bkj) % q;
                        for (uint32_t r = 0; r < NLIMB; ++r)
                            log[(kap * NLIMB + p) * T + (js * NLIMB + r)] = int8_t((Bp / POW[r]) % BASE);
                    }
                }
            auto t = tilize(log);
            std::copy(t.begin(), t.end(), v.begin() + (bj * NB + bk) * ELEMS);
        }
    return v;
}
std::vector<uint32_t> recombine(const std::vector<int32_t>& result, uint64_t q) {  // -> 32x32 row-major
    std::vector<uint32_t> C(ELEMS, 0);
    for (uint32_t bj = 0; bj < NB; ++bj) {
        std::vector<int32_t> tile(result.begin() + bj * ELEMS, result.begin() + (bj + 1) * ELEMS);
        auto P = untilize(tile);
        for (uint32_t i = 0; i < T; ++i)
            for (uint32_t js = 0; js < BLK; ++js) {
                unsigned __int128 acc = 0;
                for (uint32_t r = 0; r < NLIMB; ++r)
                    acc += (unsigned __int128)(uint32_t)P[i * T + (js * NLIMB + r)] * POW[r];
                C[i * T + (bj * BLK + js)] = uint32_t(acc % q);
            }
    }
    return C;
}

double now_us(std::chrono::high_resolution_clock::time_point a,
              std::chrono::high_resolution_clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count() / 1e3;
}

}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const uint64_t q = (argc > 1) ? std::stoull(argv[1]) : 167772161ULL;  // 5*2^25+1
        const uint32_t seed = (argc > 2) ? std::stoul(argv[2]) : 0;
        const uint32_t N = 1024;
        if (q >= (1ULL << 28)) { fmt::print(stderr, "q must be < 2^28\n"); return 2; }
        if ((q - 1) % N != 0)  { fmt::print(stderr, "N must divide q-1\n"); return 2; }

        // ---- constants: roots, W, T ----
        uint64_t g = primitive_root(q);
        uint64_t w = modpow(g, (q - 1) / N, q);     // primitive 1024th root
        uint64_t w32 = modpow(g, (q - 1) / T, q);   // = w^32, primitive 32nd root
        mont::MontgomeryContext ctx((uint32_t)q);

        std::vector<uint32_t> W(ELEMS), Tw(ELEMS);
        std::vector<uint32_t> Tmont(ELEMS);
        for (uint32_t a = 0; a < T; ++a)
            for (uint32_t b = 0; b < T; ++b)
                W[a * T + b] = (uint32_t)modpow(w32, (uint64_t)a * b, q);
        for (uint32_t j2 = 0; j2 < T; ++j2)
            for (uint32_t n1 = 0; n1 < T; ++n1) {
                uint32_t t = (uint32_t)modpow(w, (uint64_t)j2 * n1, q);
                Tw[j2 * T + n1] = t;
                Tmont[j2 * T + n1] = ctx.to_montgomery(t);
            }

        // ---- input x, reshaped Xin[n2,n1] = x[n1 + 32 n2] ----
        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)q - 1);
        std::vector<uint32_t> x(N);
        for (auto& v : x) v = dist(rng);
        std::vector<uint32_t> Xin(ELEMS);
        for (uint32_t n2 = 0; n2 < T; ++n2)
            for (uint32_t n1 = 0; n1 < T; ++n1)
                Xin[n2 * T + n1] = x[n1 + T * n2];

        // ---- golden: direct NTT  X[j] = sum_n x[n] w^{nj} mod q ----
        std::vector<uint32_t> Xref(N);
        for (uint32_t j = 0; j < N; ++j) {
            unsigned __int128 acc = 0;
            for (uint32_t n = 0; n < N; ++n)
                acc += (unsigned __int128)x[n] * modpow(w, (uint64_t)n * j, q);
            Xref[j] = (uint32_t)(acc % q);
        }

        // ======================= device setup =======================
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto device_range = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};

        constexpr uint32_t i8_tile = ELEMS * sizeof(int8_t);    // 1024
        constexpr uint32_t i32_tile = ELEMS * sizeof(int32_t);  // 4096

        // ---- matmul program (reused for stages A and C) ----
        Program mm = CreateProgram();
        distributed::DeviceLocalBufferConfig i8_dram{.page_size = i8_tile, .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig i32_dram{.page_size = i32_tile, .buffer_type = BufferType::DRAM};
        auto mmA = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = NB * i8_tile}, i8_dram, mesh.get());
        auto mmB = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = NB * NB * i8_tile}, i8_dram, mesh.get());
        auto mmO = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = NB * i32_tile}, i32_dram, mesh.get());

        auto mk_cb = [&](Program& p, CBIndex idx, tt::DataFormat fmt, uint32_t tb) {
            CircularBufferConfig c(2 * tb, {{idx, fmt}}); c.set_page_size(idx, tb);
            CreateCircularBuffer(p, core, c);
        };
        mk_cb(mm, CBIndex::c_0, tt::DataFormat::Int8, i8_tile);
        mk_cb(mm, CBIndex::c_1, tt::DataFormat::Int8, i8_tile);
        mk_cb(mm, CBIndex::c_16, tt::DataFormat::Int32, i32_tile);

        std::vector<uint32_t> mm_rd;
        TensorAccessorArgs(*mmA).append_to(mm_rd);
        TensorAccessorArgs(*mmB).append_to(mm_rd);
        auto mm_reader = CreateKernel(mm, OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = mm_rd});
        std::vector<uint32_t> mm_wr;
        TensorAccessorArgs(*mmO).append_to(mm_wr);
        auto mm_writer = CreateKernel(mm, OVERRIDE_KERNEL_PREFIX "dataflow/writer_matmul.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = mm_wr});
        CreateKernel(mm, OVERRIDE_KERNEL_PREFIX "compute/matmul_compute.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false});
        SetRuntimeArgs(mm, mm_reader, core, {(uint32_t)mmA->address(), (uint32_t)mmB->address()});
        SetRuntimeArgs(mm, mm_writer, core, {(uint32_t)mmO->address()});
        distributed::MeshWorkload mm_wl;
        mm_wl.add_program(device_range, std::move(mm));

        // ---- modmul program (stage B) ----
        Program md = CreateProgram();
        distributed::DeviceLocalBufferConfig u32_dram{.page_size = i32_tile, .buffer_type = BufferType::DRAM};
        auto mdA = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = i32_tile}, u32_dram, mesh.get());
        auto mdB = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = i32_tile}, u32_dram, mesh.get());
        auto mdO = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = i32_tile}, u32_dram, mesh.get());
        mk_cb(md, CBIndex::c_0, tt::DataFormat::UInt32, i32_tile);
        mk_cb(md, CBIndex::c_1, tt::DataFormat::UInt32, i32_tile);
        mk_cb(md, CBIndex::c_16, tt::DataFormat::UInt32, i32_tile);
        auto md_reader = CreateKernel(md, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto md_writer = CreateKernel(md, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto md_compute = CreateKernel(md, OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                          .dst_full_sync_en = true, .math_approx_mode = false});
        SetRuntimeArgs(md, md_reader, core, {(uint32_t)mdA->address(), (uint32_t)mdB->address(), 1u, 0u});
        SetRuntimeArgs(md, md_writer, core, {(uint32_t)mdO->address(), 1u, 0u});
        SetRuntimeArgs(md, md_compute, core,
            {1u, (uint32_t)q, ctx.q_inv_32(), (uint32_t)(q & 0xFFFF), (uint32_t)(q >> 16)});
        distributed::MeshWorkload md_wl;
        md_wl.add_program(device_range, std::move(md));

        // ---- helpers that run one stage on the device ----
        auto run_mm = [&](const std::vector<uint32_t>& A, const std::vector<uint32_t>& B) {
            auto av = build_lhs(A); auto bv = build_rhs(B, q);
            distributed::EnqueueWriteMeshBuffer(cq, mmA, av, false);
            distributed::EnqueueWriteMeshBuffer(cq, mmB, bv, false);
            distributed::EnqueueMeshWorkload(cq, mm_wl, false);
            std::vector<int32_t> res;
            distributed::EnqueueReadMeshBuffer(cq, res, mmO, true);
            return recombine(res, q);
        };
        auto run_modmul = [&](const std::vector<uint32_t>& a_norm, const std::vector<uint32_t>& b_mont) {
            distributed::EnqueueWriteMeshBuffer(cq, mdA, a_norm, false);
            distributed::EnqueueWriteMeshBuffer(cq, mdB, b_mont, false);
            distributed::EnqueueMeshWorkload(cq, md_wl, false);
            std::vector<uint32_t> res;
            distributed::EnqueueReadMeshBuffer(cq, res, mdO, true);
            for (auto& v : res) v = v >= q ? v - (uint32_t)q : v;  // canonical [0,q)
            return res;
        };

        // ======================= run the NTT =======================
        auto G  = run_mm(W, Xin);          // stage A
        auto Gp = run_modmul(G, Tmont);    // stage B  (G normal, T montgomery -> G*T mod q)
        auto H  = run_mm(Gp, W);           // stage C
        std::vector<uint32_t> out(N);
        for (uint32_t j1 = 0; j1 < T; ++j1)
            for (uint32_t j2 = 0; j2 < T; ++j2)
                out[j2 + T * j1] = H[j2 * T + j1];

        uint32_t mism = 0;
        for (uint32_t j = 0; j < N; ++j)
            if (out[j] != Xref[j]) { if (mism < 8) fmt::print("  X[{}] got={} exp={}\n", j, out[j], Xref[j]); ++mism; }
        fmt::print("NTT-1024 (32x32) on core {{0,0}}  q={}  seed={}  mismatches={}/{}\n", q, seed, mism, N);
        pass = (mism == 0);
        if (pass) fmt::print("RESULT: NTT WORKS.\n"); else fmt::print("RESULT: NTT WRONG.\n");

        // ======================= timing =======================
        const int iters = 200;
        auto time_wl = [&](distributed::MeshWorkload& wl) {
            distributed::EnqueueMeshWorkload(cq, wl, false);
            distributed::Finish(cq);  // warmup (JIT + first dispatch)
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) distributed::EnqueueMeshWorkload(cq, wl, false);
            distributed::Finish(cq);
            auto t1 = std::chrono::high_resolution_clock::now();
            return now_us(t0, t1) / iters;
        };
        double t_mm = time_wl(mm_wl);      // device-only, one 32x32 modular matmul
        double t_md = time_wl(md_wl);      // device-only, one 1024-elem modmul tile

        // host recombine cost (per matmul)
        std::vector<int32_t> dummy(NB * ELEMS, 1);
        auto rc0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { volatile auto c = recombine(dummy, q); (void)c; }
        auto rc1 = std::chrono::high_resolution_clock::now();
        double t_rc = now_us(rc0, rc1) / iters;

        // end-to-end single NTT (host prep + 3 device stages + reads + glue)
        const int e2e_iters = 50;
        auto e0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < e2e_iters; ++i) {
            auto g2 = run_mm(W, Xin); auto gp2 = run_modmul(g2, Tmont); auto h2 = run_mm(gp2, W);
            volatile uint32_t s = h2[0]; (void)s;
        }
        auto e1 = std::chrono::high_resolution_clock::now();
        double t_e2e = now_us(e0, e1) / e2e_iters;

        // host operand prep only (radix-128 decompose + tilize), per matmul
        auto p0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { volatile auto a = build_lhs(W); auto b = build_rhs(Xin, q); (void)a; (void)b; }
        auto p1 = std::chrono::high_resolution_clock::now();
        double t_prep = now_us(p0, p1) / iters;

        // full host-orchestrated calls (prep + writes + dispatch + BLOCKING read + recombine)
        run_mm(W, Xin); run_modmul(G, Tmont);  // warm
        auto f0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < e2e_iters; ++i) { volatile auto c = run_mm(W, Xin); (void)c; }
        auto f1 = std::chrono::high_resolution_clock::now();
        double t_mm_full = now_us(f0, f1) / e2e_iters;
        auto m0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < e2e_iters; ++i) { volatile auto c = run_modmul(G, Tmont); (void)c; }
        auto m1 = std::chrono::high_resolution_clock::now();
        double t_md_full = now_us(m0, m1) / e2e_iters;

        fmt::print("\n--- timing (single core {{0,0}}) ---\n");
        fmt::print("matmul  (device only)      : {:.2f} us  x2/NTT\n", t_mm);
        fmt::print("modmul  (device only)      : {:.2f} us  x1/NTT\n", t_md);
        fmt::print("recombine (host)           : {:.2f} us  x2/NTT\n", t_rc);
        fmt::print("operand prep (host)        : {:.2f} us  x2/NTT  (limb decompose + tilize)\n", t_prep);
        fmt::print("matmul  (full host call)   : {:.2f} us  x2/NTT  (prep+writes+dispatch+blocking read+recombine)\n", t_mm_full);
        fmt::print("modmul  (full host call)   : {:.2f} us  x1/NTT\n", t_md_full);
        fmt::print("  -> per-call overhead (full - device): matmul {:.2f}us, modmul {:.2f}us\n",
                   t_mm_full - t_mm, t_md_full - t_md);
        fmt::print("device compute /NTT        : {:.2f} us  (2*matmul + modmul)\n", 2 * t_mm + t_md);
        fmt::print("end-to-end /NTT            : {:.2f} us  (incl host prep+glue+reads)\n", t_e2e);

        pass &= mesh->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Exception:\n{}\n", e.what());
        return 2;
    }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test Finished\n");
    return pass ? 0 : 1;
}
