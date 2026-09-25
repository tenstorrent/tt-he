// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// test_ir_multiq — validate the MULTI-Q modmul on device: out[i]=(a[i]*b[i]) mod q[i] with a
// PER-ROW prime (each of the 32 tile rows uses a different prime) — the limb-major reduction
// pattern that limb packing / base conversion needs. Golden = per-row Montgomery mulmod.
//   ./build/test_ir_multiq

#include <cstdint>
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
constexpr uint32_t T = 32, ELEMS = 1024, I32 = ELEMS * 4;
std::vector<uint32_t> tilize(const std::vector<uint32_t>& v) {
    std::vector<uint32_t> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = v[r * T + c];
    return o; }
uint32_t untidx(uint32_t r, uint32_t c) { return ((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16); }
bool is_prime(uint64_t n) { if (n < 2) return false; for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) return false; return true; }
}  // namespace

int main() {
    // 32 distinct ~28-bit primes, one per tile row.
    std::vector<uint32_t> P; for (uint64_t c = (1ull << 28) - 1; P.size() < T; c -= 2) if (is_prime(c)) P.push_back((uint32_t)c);
    std::vector<mont::MontgomeryContext> ctx; for (auto p : P) ctx.emplace_back(p);

    // per-element (row-major) inputs + params + golden
    std::vector<uint32_t> a(ELEMS), b(ELEMS), q(ELEMS), qi(ELEMS), ql(ELEMS), qh(ELEMS), exp(ELEMS);
    for (uint32_t r = 0; r < T; ++r) {
        uint32_t p = P[r];
        for (uint32_t c = 0; c < T; ++c) {
            uint32_t i = r * T + c, x = (i * 2654435761u) % p, y = (i * 40503u + 7u) % p;
            a[i] = ctx[r].to_montgomery(x); b[i] = ctx[r].to_montgomery(y);
            exp[i] = ctx[r].mulmod(a[i], b[i]);                    // (x*y) mod p[r], canonical
            q[i] = p; qi[i] = ctx[r].q_inv_32(); ql[i] = p & 0xFFFFu; qh[i] = p >> 16;
        }
    }

    auto md = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange dr(md->shape());
    constexpr CoreCoord core{0, 0};
    distributed::DeviceLocalBufferConfig dram{.page_size = I32, .buffer_type = BufferType::DRAM};
    auto mk = [&] { return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = I32}, dram, md.get()); };
    auto ba = mk(), bb = mk(), bq = mk(), bqi = mk(), bql = mk(), bqh = mk(), bc = mk();
    auto wr = [&](auto buf, const std::vector<uint32_t>& v) { EnqueueWriteMeshBuffer(cq, buf, tilize(v), false); };
    wr(ba, a); wr(bb, b); wr(bq, q); wr(bqi, qi); wr(bql, ql); wr(bqh, qh);

    Program p = CreateProgram();
    auto cb = [&](CBIndex i) { CircularBufferConfig c(2 * I32, {{i, DataFormat::UInt32}}); c.set_page_size(i, I32); CreateCircularBuffer(p, core, c); };
    for (auto i : {CBIndex::c_0, CBIndex::c_1, CBIndex::c_2, CBIndex::c_3, CBIndex::c_4, CBIndex::c_5, CBIndex::c_16}) cb(i);
    auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_multiq.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto w = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_out.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/mul_modq_multiq.cpp", core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .dst_full_sync_en = true, .math_approx_mode = false});
    SetRuntimeArgs(p, rd, core, {(uint32_t)ba->address(), (uint32_t)bb->address(), (uint32_t)bq->address(),
                                 (uint32_t)bqi->address(), (uint32_t)bql->address(), (uint32_t)bqh->address(), 1u});
    SetRuntimeArgs(p, w, core, {(uint32_t)bc->address(), 1u, 0u});
    SetRuntimeArgs(p, cp, core, {1u});
    distributed::MeshWorkload wl; wl.add_program(dr, std::move(p));
    distributed::EnqueueMeshWorkload(cq, wl, false);
    std::vector<uint32_t> out; EnqueueReadMeshBuffer(cq, out, bc, true);

    uint32_t bad = 0;
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c) {
        uint32_t got = out[untidx(r, c)];
        if (!(got < 2u * P[r] && ctx[r].finalize(got) == exp[r * T + c])) ++bad;
    }
    md->close();
    fmt::print("multi-Q modmul (32 distinct primes, one per row): {}  ({} / {} bad)\n", bad ? "FAIL" : "OK", bad, ELEMS);
    if (!bad) fmt::print("ALL OK — per-row multi-Q reduction works (enables limb-major packing)\n");
    return bad ? 1 : 0;
}
