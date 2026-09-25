// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_Tg018s — what SFPMUL24 actually computes, and whether modmul should be built on it.
// docs/he/exp_tg018s.md
//
// `modmul_sfpi_full` is 20.357 ms of exp_Tg018m's 64.1 ms multiply (32.2%), and its header explains
// its shape: "Wormhole SFPI has no integer multiply (and no sfpmul24), so each product is the fp32
// byte-split trick". Four fp32 multiplies plus twelve int<->float conversions per 16x16 product,
// and a 32x32 product is four of those.
//
// Blackhole HAS the instruction. `sfpi_lib.h` exposes it behind `#if __riscv_xtttensixbh` as
//     vMag fractional_mul(a, b, FractionalHalf::Low | High)
// and tt-metal's own `mul_int_tile` does not use its width: the Blackhole LLK
// (tt_llk_blackhole/.../ckernel_sfpu_mul_int.h) loads `LO16`, issues `SFPMUL24_MOD1_LOWER` and
// stores `LO16`, so `mul_int_tile` is a 16x16 -> low-16 multiply. That is why `modmul_compute`,
// which routes its products through it, measured 11% SLOWER than the fp32 byte-split in exp_Tg018k
// part A -- it was never the native 32-bit multiply the name suggests.
//
// Before building a modmul on the raw instruction, its semantics have to be pinned down rather than
// inferred: the name is `fractional_mul`, the return type is a sign-MAGNITUDE `vMag`, and the one
// in-tree wide multiply built on it (ckernel_sfpu_div_int32_floor.h) reassembles with a shift of
// **23**, not 24:
//     tmp = Low(c,b) + (High(c,b) + Low(c, b>>23)) << 23
//
// This bench emits the raw Low and High halves for structured operands and fits the model on the
// host, reporting for each candidate the input range over which it holds. Nothing is assumed.
//
//   ./build/exp_tg018s

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;
namespace Dp = tt::tt_metal::distributed;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t ELEMS = 1024, I32 = ELEMS * 4;
using u64 = uint64_t;

// The probe set: every case that distinguishes a plausible model.
struct P { uint32_t a, b; const char* why; };
std::vector<P> probes() {
    std::vector<P> v;
    for (uint32_t k = 0; k < 32; ++k) v.push_back({1u << k, 1u, "a = 2^k, b = 1"});
    for (uint32_t k = 0; k < 32; ++k) v.push_back({1u, 1u << k, "a = 1, b = 2^k"});
    for (uint32_t k = 0; k < 24; ++k) v.push_back({1u << k, 1u << k, "a = b = 2^k"});
    for (uint32_t k = 0; k < 24; ++k) v.push_back({(1u << k) - 1u, (1u << k) - 1u, "a = b = 2^k - 1"});
    std::mt19937 rng(12345);
    auto add = [&](uint32_t bits, const char* why, int n) {
        const uint64_t m = (bits >= 32) ? 0xFFFFFFFFull : ((1ull << bits) - 1);
        for (int i = 0; i < n; ++i) v.push_back({(uint32_t)(rng() & m), (uint32_t)(rng() & m), why}); };
    add(11, "random < 2^11", 60); add(16, "random < 2^16", 60);
    add(22, "random < 2^22", 60); add(23, "random < 2^23", 60);
    add(24, "random < 2^24", 60); add(28, "random < 2^28", 60);
    add(32, "random < 2^32", 60);
    return v;
}

// Candidate models, all as functions of the FULL 64-bit product of the raw operands.
struct Model { const char* name; uint32_t (*lo)(uint32_t, uint32_t); uint32_t (*hi)(uint32_t, uint32_t); };
uint32_t lo23(uint32_t a, uint32_t b) { return (uint32_t)(((u64)a * b) & ((1u << 23) - 1)); }
uint32_t hi23(uint32_t a, uint32_t b) { return (uint32_t)((((u64)a * b) >> 23) & ((1u << 23) - 1)); }
uint32_t lo24(uint32_t a, uint32_t b) { return (uint32_t)(((u64)a * b) & ((1u << 24) - 1)); }
uint32_t hi24(uint32_t a, uint32_t b) { return (uint32_t)((((u64)a * b) >> 24) & ((1u << 24) - 1)); }
uint32_t lo23m(uint32_t a, uint32_t b) { const u64 p = (u64)(a & ((1u<<23)-1)) * (b & ((1u<<23)-1)); return (uint32_t)(p & ((1u<<23)-1)); }
uint32_t hi23m(uint32_t a, uint32_t b) { const u64 p = (u64)(a & ((1u<<23)-1)) * (b & ((1u<<23)-1)); return (uint32_t)((p >> 23) & ((1u<<23)-1)); }
uint32_t lo24m(uint32_t a, uint32_t b) { const u64 p = (u64)(a & ((1u<<24)-1)) * (b & ((1u<<24)-1)); return (uint32_t)(p & ((1u<<24)-1)); }
uint32_t hi24m(uint32_t a, uint32_t b) { const u64 p = (u64)(a & ((1u<<24)-1)) * (b & ((1u<<24)-1)); return (uint32_t)((p >> 24) & ((1u<<24)-1)); }
const std::vector<Model> MODELS = {
    {"low/high 23, full operands",   lo23,  hi23},
    {"low/high 24, full operands",   lo24,  hi24},
    {"low/high 23, operands & 2^23", lo23m, hi23m},
    {"low/high 24, operands & 2^24", lo24m, hi24m},
};
}  // namespace

int main() {
    bool pass = true;
    try {
        auto mesh = Dp::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE);
        auto& cq = mesh->mesh_command_queue();
        auto dr = Dp::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        const CoreRangeSet crs(std::set<CoreRange>{CoreRange(core)});

        const auto PR = probes();
        fmt::print("\nexp_Tg018s — SFPMUL24 semantics probe ({} operand pairs)\n", PR.size());

        auto mkbuf=[&](uint64_t nt){ Dp::DeviceLocalBufferConfig d{.page_size=I32,.buffer_type=BufferType::DRAM};
            return Dp::MeshBuffer::create(Dp::ReplicatedBufferConfig{.size=nt*I32}, d, mesh.get()); };
        auto bA = mkbuf(1), bB = mkbuf(1), bO = mkbuf(2);

        std::vector<uint32_t> hA(ELEMS, 0), hB(ELEMS, 1);
        for (size_t i = 0; i < PR.size() && i < ELEMS; ++i) { hA[i] = PR[i].a; hB[i] = PR[i].b; }
        Dp::EnqueueWriteMeshBuffer(cq, bA, hA, false);
        Dp::EnqueueWriteMeshBuffer(cq, bB, hB, false);
        Dp::Finish(cq);

        Program p = CreateProgram();
        auto mkcb=[&](CBIndex idx, uint32_t depth){ CircularBufferConfig c(depth*I32, {{idx, tt::DataFormat::UInt32}});
            c.set_page_size(idx, I32); CreateCircularBuffer(p, crs, c); };
        mkcb(CBIndex::c_0, 1); mkcb(CBIndex::c_1, 1); mkcb(CBIndex::c_16, 2);
        auto rd=CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp", crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/probe_mul24.cpp", crs,
            ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p, rd, core, {(uint32_t)bA->address(), (uint32_t)bB->address(), 1u, 0u, 0u});
        SetRuntimeArgs(p, wr, core, {(uint32_t)bO->address(), 2u, 0u});
        SetRuntimeArgs(p, cp, core, {1u});
        Dp::MeshWorkload wl; wl.add_program(dr, std::move(p));
        Dp::EnqueueMeshWorkload(cq, wl, false); Dp::Finish(cq);

        std::vector<uint32_t> out;
        Dp::EnqueueReadMeshBuffer(cq, out, bO, true);
        const uint32_t* LO = out.data(); const uint32_t* HI = out.data() + ELEMS;

        // ---- fit ----
        const size_t n = std::min<size_t>(PR.size(), ELEMS);
        fmt::print("\n  {:<34}{:>9}{:>9}{:>9}\n", "model", "lo ok", "hi ok", "both");
        int best = -1; size_t bestb = 0;
        for (size_t m = 0; m < MODELS.size(); ++m) {
            size_t okl=0, okh=0, okb=0;
            for (size_t i = 0; i < n; ++i) {
                const bool l = LO[i]==MODELS[m].lo(PR[i].a, PR[i].b);
                const bool h = HI[i]==MODELS[m].hi(PR[i].a, PR[i].b);
                okl+=l; okh+=h; okb+=(l&&h); }
            fmt::print("  {:<34}{:>9}{:>9}{:>9}\n", MODELS[m].name, okl, okh, okb);
            if (okb > bestb) { bestb = okb; best = (int)m; }
        }
        if (best < 0) { fmt::print("\n  no candidate model fits ANY probe.\n"); }
        else {
            fmt::print("\n  best: {}  ({}/{} pairs)\n", MODELS[best].name, bestb, n);
            // where does it hold / break?
            fmt::print("\n  by probe class:\n");
            std::string cur; size_t ok=0, tot=0;
            auto flush=[&]{ if (tot) fmt::print("    {:<24} {:>4}/{:<4}{}\n", cur, ok, tot,
                                                ok==tot ? "" : "   <-- BREAKS"); };
            for (size_t i = 0; i < n; ++i) {
                if (PR[i].why != cur) { flush(); cur = PR[i].why; ok = tot = 0; }
                ++tot; ok += (LO[i]==MODELS[best].lo(PR[i].a,PR[i].b) && HI[i]==MODELS[best].hi(PR[i].a,PR[i].b)); }
            flush();
        }
        fmt::print("\n  raw samples (a, b, LO, HI, and the true 64-bit product):\n");
        for (size_t i : {0u, 1u, 10u, 22u, 23u, 24u, 31u, 64u, 96u, 120u}) {
            if (i >= n) continue;
            fmt::print("    a={:>10} b={:>10}  LO={:>10} HI={:>10}   a*b={:>20}   [{}]\n",
                       PR[i].a, PR[i].b, LO[i], HI[i], (u64)PR[i].a*PR[i].b, PR[i].why); }

        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    fmt::print("\n{}\n", pass ? "Test Passed" : "Test FAILED");
    return pass ? 0 : 1;
}
