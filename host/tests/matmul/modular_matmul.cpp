// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// 32x32 modular matmul  C = (A @ B) mod q  on a single Tensix core, using only
// the signed int8 tile matmul (Int8 x Int8 -> Int32).
//
// Design + validation: docs/matmul/history/PLAN.md and the numpy oracle
// reference/matmul/tt_int8_modular_matmul_32x32.py.
//
// Strategy (radix 128 so every limb is a non-negative int8 in [0,127]):
//   A[i,k] = sum_p a[i,k,p] * 128^p                         (p = 0..3)
//   Bp[k,p,j] = (128^p * B[k,j]) mod q                      (< q < 2^28)
//   bp[k,p,j,r] = limb_r(Bp)                                (r = 0..3)
//   C[i,j] = ( sum_r 128^r * sum_{k,p} a[i,k,p]*bp[k,p,j,r] ) mod q
//
// Tiling (NB = 32/8 = 4 blocks along K and J; the I axis fills a tile whole):
//   k = bk*8 + kappa,  j = bj*8 + jsig,  lane = kappa*4 + p,  outcol = jsig*4 + r
//   L_bk[i, kappa*4+p]            = a[i, bk*8+kappa, p]         (NB tiles)
//   R_{bk,bj}[kappa*4+p, jsig*4+r]= bp[bk*8+kappa, p, bj*8+jsig, r]  (NB*NB tiles)
//   P_bj = sum_bk (L_bk @ R_{bk,bj})  -> int32 tile, recombined on host.

#include <cstdint>
#include <random>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {

constexpr uint32_t T = 32;       // tile dim
constexpr uint32_t NB = 4;       // blocks per axis (T / BLK)
constexpr uint32_t BLK = 8;      // logical values per 32-wide tile
constexpr uint32_t NLIMB = 4;    // 128^4 == 2^28
constexpr uint64_t BASE = 128;
constexpr uint32_t ELEMS = T * T;  // 1024

const uint64_t POW[NLIMB] = {1, 128, 16384, 2097152};  // 128^p

// Standard TT 32x32 tile layout: 2x2 faces of 16x16, faces row-major, each face row-major.
template <typename E>
std::vector<E> tilize(const std::vector<E>& in /* 32x32 row-major */) {
    std::vector<E> out(ELEMS);
    for (uint32_t row = 0; row < T; ++row) {
        for (uint32_t col = 0; col < T; ++col) {
            uint32_t face = (row / 16) * 2 + (col / 16);
            uint32_t in_face = (row % 16) * 16 + (col % 16);
            out[face * 256 + in_face] = in[row * T + col];
        }
    }
    return out;
}

template <typename E>
std::vector<E> untilize(const std::vector<E>& in /* tiled */) {
    std::vector<E> out(ELEMS);
    for (uint32_t row = 0; row < T; ++row) {
        for (uint32_t col = 0; col < T; ++col) {
            uint32_t face = (row / 16) * 2 + (col / 16);
            uint32_t in_face = (row % 16) * 16 + (col % 16);
            out[row * T + col] = in[face * 256 + in_face];
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const uint64_t q = (argc > 1) ? std::stoull(argv[1]) : 268435399ULL;  // 2^28 - 57, prime
        const uint32_t seed = (argc > 2) ? std::stoul(argv[2]) : 0;
        if (q >= (1ULL << 28)) {
            fmt::print(stderr, "q must be < 2^28 (got {})\n", q);
            return 2;
        }

        // ---- generate inputs in [0, q) ----
        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(0, q - 1);
        std::vector<uint32_t> A(ELEMS), B(ELEMS);
        for (auto& x : A) x = static_cast<uint32_t>(dist(rng));
        for (auto& x : B) x = static_cast<uint32_t>(dist(rng));

        // ---- reference C = (A @ B) mod q ----
        std::vector<uint32_t> Cref(ELEMS);
        for (uint32_t i = 0; i < T; ++i)
            for (uint32_t j = 0; j < T; ++j) {
                unsigned __int128 acc = 0;
                for (uint32_t k = 0; k < T; ++k)
                    acc += static_cast<uint64_t>(A[i * T + k]) * B[k * T + j];
                Cref[i * T + j] = static_cast<uint32_t>(acc % q);
            }

        // ---- build NB LHS limb-tiles (Int8), concatenated in tile order ----
        std::vector<int8_t> a_vec(NB * ELEMS, 0);
        for (uint32_t bk = 0; bk < NB; ++bk) {
            std::vector<int8_t> log(ELEMS, 0);
            for (uint32_t i = 0; i < T; ++i)
                for (uint32_t kappa = 0; kappa < BLK; ++kappa) {
                    uint32_t Aik = A[i * T + (bk * BLK + kappa)];
                    for (uint32_t p = 0; p < NLIMB; ++p)
                        log[i * T + (kappa * NLIMB + p)] =
                            static_cast<int8_t>((Aik / POW[p]) % BASE);
                }
            auto tiled = tilize(log);
            std::copy(tiled.begin(), tiled.end(), a_vec.begin() + bk * ELEMS);
        }

        // ---- build NB*NB RHS limb-tiles (Int8); index = bj*NB + bk ----
        std::vector<int8_t> b_vec(NB * NB * ELEMS, 0);
        for (uint32_t bk = 0; bk < NB; ++bk)
            for (uint32_t bj = 0; bj < NB; ++bj) {
                std::vector<int8_t> log(ELEMS, 0);
                for (uint32_t kappa = 0; kappa < BLK; ++kappa)
                    for (uint32_t jsig = 0; jsig < BLK; ++jsig) {
                        uint32_t k = bk * BLK + kappa, j = bj * BLK + jsig;
                        uint64_t Bkj = B[k * T + j];
                        for (uint32_t p = 0; p < NLIMB; ++p) {
                            uint64_t Bp = (POW[p] * Bkj) % q;  // 128^p * B mod q
                            for (uint32_t r = 0; r < NLIMB; ++r)
                                log[(kappa * NLIMB + p) * T + (jsig * NLIMB + r)] =
                                    static_cast<int8_t>((Bp / POW[r]) % BASE);
                        }
                    }
                auto tiled = tilize(log);
                std::copy(tiled.begin(), tiled.end(), b_vec.begin() + (bj * NB + bk) * ELEMS);
            }

        // ---- device setup ----
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};

        constexpr uint32_t in_tile_bytes = ELEMS * sizeof(int8_t);    // 1024
        constexpr uint32_t out_tile_bytes = ELEMS * sizeof(int32_t);  // 4096

        distributed::DeviceLocalBufferConfig in_dram{
            .page_size = in_tile_bytes, .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig out_dram_cfg{
            .page_size = out_tile_bytes, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig a_buf{.size = NB * in_tile_bytes};
        distributed::ReplicatedBufferConfig b_buf{.size = NB * NB * in_tile_bytes};
        distributed::ReplicatedBufferConfig out_buf{.size = NB * out_tile_bytes};

        auto a_dram = distributed::MeshBuffer::create(a_buf, in_dram, mesh_device.get());
        auto b_dram = distributed::MeshBuffer::create(b_buf, in_dram, mesh_device.get());
        auto out_dram = distributed::MeshBuffer::create(out_buf, out_dram_cfg, mesh_device.get());

        // ---- circular buffers: c_0/c_1 = Int8 (2-deep), c_16 = Int32 (2-deep) ----
        CircularBufferConfig cb0(2 * in_tile_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Int8}});
        cb0.set_page_size(tt::CBIndex::c_0, in_tile_bytes);
        CreateCircularBuffer(program, core, cb0);

        CircularBufferConfig cb1(2 * in_tile_bytes, {{tt::CBIndex::c_1, tt::DataFormat::Int8}});
        cb1.set_page_size(tt::CBIndex::c_1, in_tile_bytes);
        CreateCircularBuffer(program, core, cb1);

        CircularBufferConfig cb16(2 * out_tile_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Int32}});
        cb16.set_page_size(tt::CBIndex::c_16, out_tile_bytes);
        CreateCircularBuffer(program, core, cb16);

        // ---- kernels ----
        std::vector<uint32_t> reader_ct;
        TensorAccessorArgs(*a_dram).append_to(reader_ct);
        TensorAccessorArgs(*b_dram).append_to(reader_ct);
        KernelHandle reader = CreateKernel(
            program, OVERRIDE_KERNEL_PREFIX "dataflow/reader_matmul.cpp", core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_ct});

        std::vector<uint32_t> writer_ct;
        TensorAccessorArgs(*out_dram).append_to(writer_ct);
        KernelHandle writer = CreateKernel(
            program, OVERRIDE_KERNEL_PREFIX "dataflow/writer_matmul.cpp", core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_ct});

        CreateKernel(
            program, OVERRIDE_KERNEL_PREFIX "compute/matmul_compute.cpp", core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,  // 32-bit dest, required for Int32 accumulation
                .math_approx_mode = false});

        SetRuntimeArgs(program, reader, core, {a_dram->address(), b_dram->address()});
        SetRuntimeArgs(program, writer, core, {out_dram->address()});

        distributed::EnqueueWriteMeshBuffer(cq, a_dram, a_vec, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(cq, b_dram, b_vec, /*blocking=*/false);
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

        std::vector<int32_t> result;  // NB tiles of int32
        distributed::EnqueueReadMeshBuffer(cq, result, out_dram, /*blocking=*/true);

        // ---- recombine + reduce mod q ----
        std::vector<uint32_t> C(ELEMS, 0);
        for (uint32_t bj = 0; bj < NB; ++bj) {
            std::vector<int32_t> tile(result.begin() + bj * ELEMS, result.begin() + (bj + 1) * ELEMS);
            auto P = untilize(tile);  // P[i*T + (jsig*4+r)]
            for (uint32_t i = 0; i < T; ++i)
                for (uint32_t jsig = 0; jsig < BLK; ++jsig) {
                    unsigned __int128 acc = 0;
                    for (uint32_t r = 0; r < NLIMB; ++r)
                        acc += static_cast<unsigned __int128>(
                                   static_cast<uint32_t>(P[i * T + (jsig * NLIMB + r)])) *
                               POW[r];
                    C[i * T + (bj * BLK + jsig)] = static_cast<uint32_t>(acc % q);
                }
        }

        // ---- verify ----
        uint32_t mismatches = 0;
        for (uint32_t idx = 0; idx < ELEMS; ++idx) {
            if (C[idx] != Cref[idx]) {
                if (mismatches < 8)
                    fmt::print("  ({},{}) got={} expected={}\n", idx / T, idx % T, C[idx], Cref[idx]);
                ++mismatches;
            }
        }
        fmt::print("q={}  seed={}  mismatches={}/{}\n", q, seed, mismatches, ELEMS);
        if (mismatches == 0) {
            fmt::print("RESULT: 32x32 modular matmul via int8 WORKS.\n");
        } else {
            pass = false;
            fmt::print("RESULT: modular matmul produced wrong values.\n");
        }

        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Exception:\n{}\n", e.what());
        return 2;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        fmt::print("Test Finished (see RESULT above)\n");
    }
    return pass ? 0 : 1;
}
