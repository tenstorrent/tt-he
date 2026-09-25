// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// fusionA_limb — limb-major Fusion A on hardware (docs/ir FORMALIZATION §1.5).
// A base-conversion / limb reduction uses only `LIMBS` of the 32 tile rows (rows = limbs), so an
// unpacked tile wastes (32-LIMBS)/32 of its compute. Fusion A merges 32/LIMBS independent
// reductions into ONE full tile, with a PER-ROW prime (the multi-Q kernel). Same total work,
// 32/LIMBS fewer tiles. This measures packed vs unpacked and validates the packed per-row results.
//
// Env: FA_GROUPS (independent reductions, default 2048), FA_LIMBS (rows/reduction, default 4),
//      FA_ITERS (default 100).   ./build/fusionA_limb

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#error "OVERRIDE_KERNEL_PREFIX must be set by CMake"
#endif

namespace {
uint32_t envu(const char* n, uint32_t d) { const char* v = std::getenv(n); return v ? (uint32_t)std::stoul(v) : d; }
constexpr uint32_t T = 32, ELEMS = 1024, I32 = ELEMS * 4;
std::vector<uint32_t> tilize(const std::vector<uint32_t>& v) {
    std::vector<uint32_t> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = v[r * T + c];
    return o; }
uint32_t untidx(uint32_t r, uint32_t c) { return ((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16); }
bool is_prime(uint64_t n) { if (n < 2) return false; for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) return false; return true; }
using Buf = std::shared_ptr<distributed::MeshBuffer>;
}  // namespace

int main() {
    const uint32_t G = envu("FA_GROUPS", 2048), LIMBS = envu("FA_LIMBS", 4), iters = envu("FA_ITERS", 100);
    const uint32_t per_tile = T / LIMBS;                    // reductions packed per full tile (32/4 = 8)
    const uint32_t PT = (G + per_tile - 1) / per_tile;      // packed tiles
    const uint32_t UT = G;                                  // unpacked tiles (one reduction each, 4/32 rows used)

    // 32 distinct ~28-bit primes, one per tile row.
    std::vector<uint32_t> P; for (uint64_t c = (1ull << 28) - 1; P.size() < T; c -= 2) if (is_prime(c)) P.push_back((uint32_t)c);
    std::vector<mont::MontgomeryContext> ctx; for (auto p : P) ctx.emplace_back(p);

    // one packed tile's worth of per-row data (reused for every tile — enough to validate + time)
    std::vector<uint32_t> a(ELEMS), b(ELEMS), q(ELEMS), qi(ELEMS), ql(ELEMS), qh(ELEMS), exp(ELEMS);
    for (uint32_t r = 0; r < T; ++r) { uint32_t p = P[r];
        for (uint32_t c = 0; c < T; ++c) { uint32_t i = r * T + c, x = (i * 2654435761u) % p, y = (i * 40503u + 7u) % p;
            a[i] = ctx[r].to_montgomery(x); b[i] = ctx[r].to_montgomery(y); exp[i] = ctx[r].mulmod(a[i], b[i]);
            q[i] = p; qi[i] = ctx[r].q_inv_32(); ql[i] = p & 0xFFFFu; qh[i] = p >> 16; } }
    auto ta = tilize(a), tb = tilize(b), tq = tilize(q), tqi = tilize(qi), tql = tilize(ql), tqh = tilize(qh);

    auto md = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange dr(md->shape());
    constexpr CoreCoord core{0, 0};
    distributed::DeviceLocalBufferConfig dram{.page_size = I32, .buffer_type = BufferType::DRAM};
    auto mkN = [&](uint32_t n) { return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = (uint64_t)n * I32}, dram, md.get()); };
    auto fill = [&](Buf buf, const std::vector<uint32_t>& t, uint32_t n) { std::vector<uint32_t> big(n * ELEMS);
        for (uint32_t k = 0; k < n; ++k) std::copy(t.begin(), t.end(), big.begin() + k * ELEMS); EnqueueWriteMeshBuffer(cq, buf, big, false); };

    auto run = [&](uint32_t ntiles, bool validate) -> double {
        auto ba = mkN(ntiles), bb = mkN(ntiles), bq = mkN(1), bqi = mkN(1), bql = mkN(1), bqh = mkN(1), bo = mkN(ntiles);
        fill(ba, ta, ntiles); fill(bb, tb, ntiles);
        EnqueueWriteMeshBuffer(cq, bq, tq, false); EnqueueWriteMeshBuffer(cq, bqi, tqi, false);
        EnqueueWriteMeshBuffer(cq, bql, tql, false); EnqueueWriteMeshBuffer(cq, bqh, tqh, false);
        Program p = CreateProgram();
        auto cb = [&](CBIndex i) { CircularBufferConfig c(2 * I32, {{i, DataFormat::UInt32}}); c.set_page_size(i, I32); CreateCircularBuffer(p, core, c); };
        for (auto i : {CBIndex::c_0, CBIndex::c_1, CBIndex::c_2, CBIndex::c_3, CBIndex::c_4, CBIndex::c_5, CBIndex::c_16}) cb(i);
        auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_multiq.cpp", core, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto w = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_out.cpp", core, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/mul_modq_multiq.cpp", core, ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
        SetRuntimeArgs(p, rd, core, {(uint32_t)ba->address(), (uint32_t)bb->address(), (uint32_t)bq->address(), (uint32_t)bqi->address(), (uint32_t)bql->address(), (uint32_t)bqh->address(), ntiles});
        SetRuntimeArgs(p, w, core, {(uint32_t)bo->address(), ntiles, 0u});
        SetRuntimeArgs(p, cp, core, {ntiles});
        distributed::MeshWorkload wl; wl.add_program(dr, std::move(p));
        distributed::EnqueueMeshWorkload(cq, wl, false); distributed::Finish(cq);   // warm
        if (validate) { std::vector<uint32_t> out; EnqueueReadMeshBuffer(cq, out, bo, true);
            uint32_t bad = 0; for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
                if (!(out[untidx(r, c)] < 2u * P[r] && ctx[r].finalize(out[untidx(r, c)]) == exp[r * T + c])) ++bad;
            fmt::print("  packed correctness (per-row multi-Q): {} ({}/{})\n", bad ? "FAIL" : "OK", bad, ELEMS); }
        auto t0 = std::chrono::high_resolution_clock::now();
        for (uint32_t it = 0; it < iters; ++it) distributed::EnqueueMeshWorkload(cq, wl, false);
        distributed::Finish(cq);
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - t0).count() / 1e3 / iters;
    };

    fmt::print("limb-major Fusion A: {} reductions, {} limbs each ({} rows used / 32)\n", G, LIMBS, LIMBS);
    double packed = run(PT, true);
    double unpacked = run(UT, false);
    md->close();
    fmt::print("  UNPACKED ({} tiles, {}/32 rows used): {:.1f} us/iter\n", UT, LIMBS, unpacked);
    fmt::print("  PACKED   ({} tiles, 32/32 rows used):  {:.1f} us/iter\n", PT, packed);
    fmt::print("  Fusion A speedup: {:.2f}x  (ideal {}x)\n", unpacked / packed, per_tile);
    return 0;
}
