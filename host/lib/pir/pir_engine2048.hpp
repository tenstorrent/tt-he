// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PirEngine2048 — the K-general (K=2 ~56-bit / K=3 ~84-bit) OnionPIR device op library at the
// library ring n=2048.  Encapsulates the per-channel NTT/CRT/gadget constants and the validated
// device-kernel wrappers shared by test_extprod2048 / test_bv_keyswitch2048 / test_query_expand2048
// (previously duplicated byte-for-byte).  Every method is bit-exact-validated by those tests.
//
// WHICH OF lib/pir's THREE HEADERS THIS IS: the real one. 17 tests and the 65,536-row demo
// (examples/pir_demo65536.cpp) build on it, and it runs the optimised kernel set
// (matmul_modq_mbatch_1pass, reader_mbatch_deep, add_modq_1pass, modmul_sfpi_1pass). Header-only.
// pir_demo_api.hpp is a small façade for the 4-entry teaching demo only; pir_permute.hpp is the
// shared coefficient-permutation helper that both use.
//
// Data conventions (all natural coefficient order unless noted):
//   * a "poly" is a 2048-entry std::vector<uint32_t> (mod q_ch), tilized on the fly.
//   * fwd2048/inv2048 are the four-step (32×64) NTT / iNTT per RNS channel.
//   * crt_and_gadget(residues[K]) -> ell signed base-B digit polys (via Garner CRT + gadget).
//   * digit_rns(dig) -> K residue polys.

#pragma once
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "common/montgomery.hpp"
#include "pir_permute.hpp"

namespace pir2048 {
using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
#ifndef OVERRIDE_KERNEL_PREFIX_IR          // the phase-major mbatch kernels live in kernels/ir/
#define OVERRIDE_KERNEL_PREFIX_IR ""
#endif
using MeshBuf = std::shared_ptr<distributed::MeshBuffer>;
using u128 = unsigned __int128;

// ---- host-phase profiling (idle-cause attribution, same split as device/he/he_mul_cross.cpp) ----
// The answer pipeline is host-orchestrated, so wall-clock is host overhead, not device compute. This
// attributes it: BUILD = CreateProgram + CreateKernel (paid per op call because the engine rebuilds
// programs every invocation), ENQ = EnqueueMeshWorkload, WR/RD = buffer traffic (RD is blocking).
// Which term dominates decides the fix: BUILD -> cache workloads; ENQ -> trace replay; RD -> buffer residency.
namespace prof {
inline double g_build = 0, g_enq = 0, g_wr = 0, g_rd = 0;
inline long n_build = 0, n_enq = 0, n_wr = 0, n_rd = 0;
using hrc = std::chrono::high_resolution_clock;
inline double msd(hrc::time_point t) { return std::chrono::duration<double, std::milli>(hrc::now() - t).count(); }
inline void reset() { g_build = g_enq = g_wr = g_rd = 0; n_build = n_enq = n_wr = n_rd = 0; }
inline double total() { return g_build + g_enq + g_wr + g_rd; }
inline void report(const char* tag) {
    double t = total();
    auto row = [&](const char* nm, double g, long n) {
        fmt::print("    {:<22} {:9.1f} ms  {:5.1f}%  {:7} calls  ({:.3f} ms/call)\n",
                   nm, g, t > 0 ? 100.0 * g / t : 0.0, n, n ? g / n : 0.0); };
    fmt::print("  host-phase split [{}]  total {:.1f} ms\n", tag, t);
    row("program BUILD", g_build, n_build);
    row("workload ENQ", g_enq, n_enq);
    row("buffer WRITE", g_wr, n_wr);
    row("buffer READ (block)", g_rd, n_rd);
}
}  // namespace prof

constexpr uint32_t T = 32, NB = 4, N1 = 32, N2 = 64, N = 2048, ELEMS = 1024, I8 = ELEMS, I32 = ELEMS * 4;
static const uint64_t POW[4] = {1, 128, 16384, 2097152};

inline uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) { uint64_t r = 1; b %= m;
    while (e) { if (e & 1) r = (u128)r * b % m; b = (u128)b * b % m; e >>= 1; } return r; }
inline uint64_t prim_root(uint64_t q) { uint64_t phi = q - 1, n = phi; std::vector<uint64_t> f;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) { f.push_back(d); while (n % d == 0) n /= d; }
    if (n > 1) f.push_back(n);
    for (uint64_t g = 2; g < q; ++g) { bool ok = true; for (uint64_t p : f) if (modpow(g, phi / p, q) == 1) ok = false;
        if (ok) return g; } throw std::runtime_error("no root"); }
inline int8_t limb(uint32_t v, uint32_t p) { return (int8_t)((v >> (7 * p)) & 127); }
template <typename E> std::vector<E> tilize32(const std::vector<E>& in) { std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c) o[((r/16)*2+c/16)*256+(r%16)*16+(c%16)] = in[r*T+c]; return o; }
template <typename E> std::vector<E> untilize32(const std::vector<E>& in) { std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c) o[r*T+c] = in[((r/16)*2+c/16)*256+(r%16)*16+(c%16)]; return o; }
inline std::vector<uint32_t> til1(const std::vector<uint32_t>& v) { return pir::tilize_poly(v, 1); }
inline std::vector<uint32_t> unt1(const std::vector<uint32_t>& v) { return pir::untilize_poly(v, 1); }
inline std::vector<uint32_t> til2(const std::vector<uint32_t>& v) { return pir::tilize_poly(v, 2); }
inline std::vector<uint32_t> unt2(const std::vector<uint32_t>& v) { return pir::untilize_poly(v, 2); }
inline std::vector<uint32_t> half(const std::vector<uint32_t>& v, uint32_t t) {
    return std::vector<uint32_t>(v.begin()+t*ELEMS, v.begin()+(t+1)*ELEMS); }

struct Chan { uint32_t q, qinv, qlo, qhi, k14;
    std::vector<MeshBuf> WA, TwF, Winv64, TwI; MeshBuf WCt, Winv32; };

struct Engine {
    distributed::MeshDevice* mesh; distributed::MeshCommandQueue* cq;
    distributed::MeshCoordinateRange dr; CoreCoord core{0, 0};
    uint32_t K, ell, B, LOG2B, twoL;
    std::vector<uint32_t> primes;
    std::vector<mont::MontgomeryContext> ctxp;
    std::vector<Chan> CH;
    std::vector<MeshBuf> bIn, bAcc, bSum, bB, bOut;   // NTT scratch (2 each)
    MeshBuf bOut2;                                    // packed 2-tile transform output (1 readback, not 2)
    // ---- state required by the ported 65 536 application layer ----
    MeshBuf bIn2;                                  // 2-tile contiguous input for fusedA2
    std::vector<MeshBuf> bInv;                     // baked const polys of invm(a,b), index a*4+b
    MeshBuf tFwdIn, tFwdOut, tInvIn, tInvOut;      // 2-tile uint32 gather tables (face-tiled offsets)
    MeshBuf bIn2b, bOut2r;                         // scratch for wh's buffer-resident transform path
                                                   // (bOut2 above is this tree's packed-output scratch)
    static constexpr uint32_t MMAX = 32;           // max polys per batched transform
    std::vector<std::vector<MeshBuf>> WArep;       // [ch][bk2*2+blk] foldLHS replicated MMAX x
    std::vector<MeshBuf> Winv32rep;                // [ch] inverse stage-1 fold, same replication
    std::vector<MeshBuf> bPar; MeshBuf bZero2;         // CRT params (per prime) + zero

    // ---- workload cache + fixed-address buffer pool (docs/he/history/OPTIMIZATION_REPORT.md T2) ----
    // Every dispatch here uses a freshly built Program, so EnqueueMeshWorkload re-serializes binaries,
    // CB configs and runtime args from scratch -- that serialization, not dispatch count, is ~84% of the
    // answer pipeline's host time (measured 0.57 ms/dispatch; re-enqueueing the SAME workload is 0.031,
    // 18.4x, see test_dispatch_micro).  So: record the pipeline once, then replay it per query.
    //
    // Replay requires STABLE BUFFER ADDRESSES, because cached runtime args bake them in.  Ops allocate
    // scratch via mkbuf(), so with the cache on those allocations come from POOL in deterministic order
    // and begin_pass() rewinds it -- pass 2 hands back the same buffers at the same addresses.  Engine
    // constants are allocated in the constructor with the cache OFF, so they never enter the pool.
    // Traffic counters (from the wh engine): the quantities device-residency is trying to drive to zero.
    // Kept alongside prof:: -- prof attributes HOST time by phase, these count DEVICE traffic.
    long n_rdb = 0, n_wrb = 0, n_prog = 0, n_buf = 0; size_t by_rdb = 0, by_wrb = 0;
    bool dry_run = false;                 // count programs without executing (wh: cost model probes)
    std::vector<MeshBuf> zeroCache;       // reusable zero-filled buffers
    struct Shard { uint32_t per, C; };    // per-core split used by the multicore scan
    uint32_t GX = 1, GY = 1;              // compute-grid extents, cached at construction
    static constexpr uint32_t SMB = 32;   // output tiles per core per mbatch pass
    void io_reset() { n_rdb = n_wrb = 0; by_rdb = by_wrb = 0; }
    bool CACHE_ON = false, CACHE_REC = false, pooling = false;
    std::vector<distributed::MeshWorkload> WC; size_t WI = 0;      // recorded workloads, replay cursor
    prof::hrc::time_point t_dispatch;                              // set when cached() declines; BUILD start
    struct PoolEnt { uint32_t nt, pg; MeshBuf b; };
    std::vector<PoolEnt> POOL; size_t pool_cur = 0;                      // scratch pool, allocation cursor

    // ---- Metal Trace on the cached path (docs/he/history/OPTIMIZATION_REPORT.md T2.5) ----
    // Program reuse got enqueue to 0.019 ms/call, but at ~5.9k dispatches that is still ~77% of the
    // per-query cost. Trace makes it O(1) in dispatch count: capture the dispatch sequence once, then
    // issue ONE replay command per segment. A "segment" is a maximal run of emit() with no host boundary,
    // so seg_break() (called from wrb/rdb) closes the open capture trace / arms the next replay segment.
    // Phases: 0 = eager (record here), 1 = capture, 2 = replay.
    int TR_PHASE = 0;
    std::vector<distributed::MeshTraceId> SEG; bool seg_open = false, seg_started = false; size_t seg_idx = 0;

    // Drain the queue. REQUIRED for honest timing once a stage has no blocking readback left: without it the
    // timed region measures host enqueue only (observed: expansion "1.3 ms" with 3976% spread while the real
    // work surfaced in a later fetch at 4.6 ms/readback).
    void finish() { distributed::Finish(*cq); }
    // This tree's per-query entry point: trace needs stable addresses, so pooling is implied here.
    // wh's paths never call begin_pass(); they opt into pooling explicitly via pool_enable().
    void begin_pass() { pooling = true; WI = 0; pool_cur = 0; seg_idx = 0; seg_started = false; }
    // wh's control surface, mapped onto the same state so its ~80 methods port unchanged.
    void pool_reset() { pool_cur = 0; }
    void pool_enable(bool on) { pooling = on; if (on) pool_cur = 0; }
    void wl_reset() { WI = 0; }
    void wl_enable(bool on) { CACHE_ON = on; CACHE_REC = false; if (on) WI = 0; }
    void caches_clear() { WC.clear(); WI = 0; POOL.clear(); pool_cur = 0; zeroCache.clear(); }
    void reset_io() { io_reset(); n_prog = n_buf = 0; }   // rewind cursors (per query)
    size_t n_cached() const { return WC.size(); }
    size_t n_segments() const { return SEG.size(); }
    void seg_break() {
        if (!CACHE_ON) return;
        if (TR_PHASE == 1) { if (seg_open) { mesh->end_mesh_trace(0, SEG.back()); seg_open = false; } }
        else if (TR_PHASE == 2) { seg_started = false; } }
    void close_capture() { if (seg_open) { mesh->end_mesh_trace(0, SEG.back()); seg_open = false; } }

    MeshBuf mkraw(uint32_t nt, uint32_t pg) {
        distributed::DeviceLocalBufferConfig d{.page_size = pg, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * pg}, d, mesh); }

    // ---- low-level device helpers ----
    MeshBuf mkbuf(uint32_t nt, uint32_t pg) {
        if (!pooling) { ++n_buf; return mkraw(nt, pg); }
        if (pool_cur < POOL.size()) {
            const auto& e = POOL[pool_cur];
            // Divergent allocation order would silently alias buffers between passes -> fail loudly.
            if (e.nt != nt || e.pg != pg)
                throw std::runtime_error("pir2048::Engine pool order diverged at slot " + std::to_string(pool_cur)
                    + ": recorded (nt=" + std::to_string(e.nt) + ",pg=" + std::to_string(e.pg)
                    + ") but asked for (nt=" + std::to_string(nt) + ",pg=" + std::to_string(pg) + ")");
            ++pool_cur; return e.b; }
        auto b = mkraw(nt, pg); POOL.push_back({nt, pg, b}); ++pool_cur; return b; }

    // Build-or-replay one dispatch. On replay the lambda is never invoked, so program construction is
    // skipped entirely (that is the point -- it is not enough to cache the Program and rebuild around it).
    // ---- ONE dispatch mechanism, TWO idioms ----------------------------------------------------
    // The `wh` engine independently solved program reuse with a guard-then-build idiom:
    //     if (!cached()) { Program p = ...; run(p); }
    // this tree solved it with a lambda:
    //     emit([&]{ Program p = ...; return p; });
    // They are the same thing: `cached()` IS emit's replay branch, `run()` IS its build branch. Expressing
    // both on the shared state means wh's ~80 methods port unchanged AND inherit Metal Trace, which the wh
    // design never had (workload reuse only -- worth ~5x; reuse + trace measured 23x on expansion).
    //
    // Returns true if this dispatch was served from cache/trace (caller must then skip building).
    bool cached() {
        if (!(CACHE_ON && !CACHE_REC)) { t_dispatch = prof::hrc::now(); return false; }
        // Auto-extend: a dispatch with nothing recorded falls through and is recorded by run(). This is
        // wh's cache semantics; harmless here because the tree's record pass fills WC completely first.
        if (TR_PHASE != 2 && WI >= WC.size()) { t_dispatch = prof::hrc::now(); return false; }
        if (TR_PHASE == 2) {                           // replay: ONE command per segment, not per dispatch
            if (!seg_started) {
                if (seg_idx < SEG.size()) { auto tr = prof::hrc::now();
                    mesh->replay_mesh_trace(0, SEG[seg_idx], false);
                    prof::g_enq += prof::msd(tr); ++prof::n_enq; }
                ++seg_idx; seg_started = true; }
            ++WI; return true; }
        if (TR_PHASE == 1 && !seg_open) { SEG.push_back(mesh->begin_mesh_trace(0)); seg_open = true; }
        auto te = prof::hrc::now(); distributed::EnqueueMeshWorkload(*cq, WC[WI++], false);
        prof::g_enq += prof::msd(te); ++prof::n_enq; return true; }

    // Enqueue a freshly built program, recording it when the cache is arming.
    void run(Program&& p) {
        ++n_prog;                    // freshly BUILT program (the replay paths in cached() do not count)
        // Build time: everything between cached() declining and here. Timing it in emit() alone missed
        // the ~40 `if (!cached()){ ...; run(p); }` sites, so BUILD read 0 calls and its cost silently
        // landed nowhere. Attribute it from the dispatch marker instead, which covers both shapes.
        prof::g_build += prof::msd(t_dispatch); ++prof::n_build;
        distributed::MeshWorkload wl; wl.add_program(dr, std::move(p));
        auto te = prof::hrc::now();
        if (CACHE_ON) { WC.push_back(std::move(wl)); distributed::EnqueueMeshWorkload(*cq, WC[WI++], false); }
        else          { distributed::EnqueueMeshWorkload(*cq, wl, false); }
        prof::g_enq += prof::msd(te); ++prof::n_enq; }
    void run(Program& p) { run(std::move(p)); }        // wh call sites pass an lvalue

    template <class F>
    void emit(F&& build_fn) {
        if (cached()) return;
        Program p = build_fn();          // timed by run() via t_dispatch; do not double-count here
        run(std::move(p)); }
    void mkcb(Program& p, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t pg) {
        CircularBufferConfig c(depth * pg, {{idx, fmt}}); c.set_page_size(idx, pg); CreateCircularBuffer(p, core, c); }
    // Both are host boundaries, so they must break the trace segment first (event-sync is forbidden
    // inside a capture, and a blocking read would otherwise land mid-trace).
    void wrb(MeshBuf b, const std::vector<uint32_t>& v) { seg_break(); auto t = prof::hrc::now();
        distributed::EnqueueWriteMeshBuffer(*cq, b, v, false); prof::g_wr += prof::msd(t); ++prof::n_wr;
        ++n_wrb; by_wrb += 4 * v.size(); }
    std::vector<uint32_t> rdb(MeshBuf b) { seg_break(); std::vector<uint32_t> v; auto t = prof::hrc::now();
        distributed::EnqueueReadMeshBuffer(*cq, v, b, true); prof::g_rd += prof::msd(t); ++prof::n_rd;
        ++n_rdb; by_rdb += 4 * v.size(); return v; }

    // matmul stage A / C for the four-step NTT (int8 folds), 1 out-tile at offset os
    void fusedA(Chan& c, MeshBuf xin, MeshBuf w, MeshBuf out, uint32_t os) { emit([&]{ Program p = CreateProgram();
        mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::Int8,2,I8);
        mkcb(p,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcb(p,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*xin).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)xin->address(),(uint32_t)w->address()}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,os});
        SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14}); return p; }); }
    void fusedC(Chan& c, MeshBuf gp, MeshBuf w, MeshBuf out, uint32_t os) { emit([&]{ Program p = CreateProgram();
        mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::Int8,NB*NB,I8);
        mkcb(p,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcb(p,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*gp).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)gp->address(),(uint32_t)w->address()}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,os});
        SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14}); return p; }); }
    // os = output TILE offset, so several madd2 results can be packed into one multi-tile buffer
    // (lets a transform's two blocks share a buffer and be read back in ONE call instead of two).
    void madd2(Chan& c, MeshBuf in, MeshBuf out, uint32_t os = 0) { emit([&]{ Program p = CreateProgram();
        mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)in->address(),2u,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,os}); SetRuntimeArgs(p,cp,core,{2u,c.q});
        return p; }); }
    // in_start: wh's superset signature. Defaulted, so every existing caller is unaffected; wh's batched
    // scan paths pass a nonzero input tile offset.
    void mmul(Chan& c, MeshBuf a, MeshBuf b, MeshBuf out, uint32_t nt, uint32_t out_start=0, uint32_t in_start=0) { emit([&]{ Program p = CreateProgram();
        mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_1pass.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)a->address(),(uint32_t)b->address(),nt,in_start}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),nt,out_start}); SetRuntimeArgs(p,cp,core,{nt,c.q,c.qinv,c.qlo,c.qhi});
        return p; }); }

    Chan make_chan(uint32_t q) {
        mont::MontgomeryContext ctx(q); Chan c; c.q=q; c.qinv=ctx.q_inv_32(); c.qlo=q&0xFFFF; c.qhi=q>>16; c.k14=ctx.to_montgomery(16384u);
        uint64_t g=prim_root(q);
        uint64_t wF=modpow(g,(q-1)/N,q), psiF=modpow(g,(q-1)/(2*N),q);
        uint64_t wI=mont::modular_inverse(wF,q), psiI=mont::modular_inverse(psiF,q), Ninv=mont::modular_inverse(N,q);
        uint64_t w64F=modpow(wF,N1,q), w32F=modpow(wF,N2,q);
        uint64_t w64I=mont::modular_inverse(w64F,q), w32I=mont::modular_inverse(w32F,q);
        auto foldLHS=[&](const std::vector<uint32_t>& M){ std::vector<int8_t> o(NB*NB*ELEMS);
            for(uint32_t p=0;p<NB;++p)for(uint32_t r=0;r<NB;++r){std::vector<int8_t> l(ELEMS);for(uint32_t i=0;i<ELEMS;++i)l[i]=limb((uint32_t)((POW[r]*M[i])%q),p);auto t=tilize32(l);for(uint32_t i=0;i<ELEMS;++i)o[(p*NB+r)*ELEMS+i]=t[i];}return o; };
        auto foldRHS=[&](const std::vector<uint32_t>& M){ std::vector<int8_t> o(NB*NB*ELEMS);
            for(uint32_t p=0;p<NB;++p)for(uint32_t r=0;r<NB;++r){std::vector<int8_t> l(ELEMS);for(uint32_t i=0;i<ELEMS;++i)l[i]=limb((uint32_t)((POW[p]*M[i])%q),r);auto t=tilize32(l);for(uint32_t i=0;i<ELEMS;++i)o[(p*NB+r)*ELEMS+i]=t[i];}return o; };
        auto i8buf=[&](const std::vector<int8_t>& v){ auto b=mkbuf(NB*NB,I8); distributed::EnqueueWriteMeshBuffer(*cq, b, v, false); return b; };
        c.WA.resize(4); c.TwF.resize(2); c.Winv64.resize(4); c.TwI.resize(2);
        std::vector<uint32_t> WA(N2*N2), WCt(N1*N1);
        for(uint32_t k2=0;k2<N2;++k2)for(uint32_t n2=0;n2<N2;++n2) WA[k2*N2+n2]=(uint32_t)((u128)modpow(w64F,(uint64_t)k2*n2,q)*modpow(psiF,32ull*n2,q)%q);
        for(uint32_t n1=0;n1<N1;++n1)for(uint32_t k1=0;k1<N1;++k1) WCt[n1*N1+k1]=(uint32_t)modpow(w32F,(uint64_t)k1*n1,q);
        for(uint32_t bk2=0;bk2<2;++bk2)for(uint32_t bn2=0;bn2<2;++bn2){ std::vector<uint32_t> blk(ELEMS);
            for(uint32_t a=0;a<T;++a)for(uint32_t b=0;b<T;++b)blk[a*T+b]=WA[(bk2*T+a)*N2+(bn2*T+b)]; c.WA[bk2*2+bn2]=i8buf(foldLHS(blk)); }
        c.WCt=i8buf(foldRHS(WCt));
        std::vector<uint32_t> Winv32(N1*N1), Winv64(N2*N2);
        for(uint32_t n1=0;n1<N1;++n1)for(uint32_t k1=0;k1<N1;++k1) Winv32[n1*N1+k1]=(uint32_t)((u128)modpow(w32I,(uint64_t)n1*k1,q)*Ninv%q);
        for(uint32_t k2=0;k2<N2;++k2)for(uint32_t n2=0;n2<N2;++n2) Winv64[k2*N2+n2]=(uint32_t)((u128)modpow(w64I,(uint64_t)k2*n2,q)*modpow(psiI,32ull*n2,q)%q);
        c.Winv32=i8buf(foldLHS(Winv32));
        for(uint32_t bk2=0;bk2<2;++bk2)for(uint32_t bn2=0;bn2<2;++bn2){ std::vector<uint32_t> blk(ELEMS);
            for(uint32_t a=0;a<T;++a)for(uint32_t b=0;b<T;++b)blk[a*T+b]=Winv64[(bk2*T+a)*N2+(bn2*T+b)]; c.Winv64[bk2*2+bn2]=i8buf(foldRHS(blk)); }
        for(uint32_t bk2=0;bk2<2;++bk2){ std::vector<uint32_t> bf(ELEMS), bi(ELEMS);
            for(uint32_t a=0;a<T;++a)for(uint32_t n1=0;n1<N1;++n1){ uint64_t tf=(u128)modpow(psiF,n1,q)*modpow(wF,(uint64_t)n1*(bk2*T+a),q)%q; bf[a*T+n1]=ctx.to_montgomery((uint32_t)tf); }
            for(uint32_t n1=0;n1<N1;++n1)for(uint32_t k2loc=0;k2loc<T;++k2loc){ uint64_t ti=(u128)modpow(wI,(uint64_t)n1*(bk2*T+k2loc),q)*modpow(psiI,n1,q)%q; bi[n1*T+k2loc]=ctx.to_montgomery((uint32_t)ti); }
            c.TwF[bk2]=mkbuf(1,I32); wrb(c.TwF[bk2],tilize32(bf)); c.TwI[bk2]=mkbuf(1,I32); wrb(c.TwI[bk2],tilize32(bi)); }
        return c; }

    std::vector<uint32_t> fwd2048(Chan& c, const std::vector<uint32_t>& x){
        for(uint32_t bn2=0;bn2<2;++bn2){ std::vector<uint32_t> blk(ELEMS);
            for(uint32_t b=0;b<T;++b)for(uint32_t n1=0;n1<N1;++n1)blk[b*T+n1]=x[n1+32*(bn2*T+b)]; wrb(bIn[bn2],tilize32(blk)); }
        for(uint32_t bk2=0;bk2<2;++bk2){ fusedA(c,bIn[0],c.WA[bk2*2+0],bAcc[bk2],0); fusedA(c,bIn[1],c.WA[bk2*2+1],bAcc[bk2],1);
            madd2(c,bAcc[bk2],bSum[bk2]); mmul(c,bSum[bk2],c.TwF[bk2],bB[bk2],1); fusedC(c,bB[bk2],c.WCt,bOut2,bk2); }
        std::vector<uint32_t> X(N); auto raw=rdb(bOut2);        // both blocks in ONE readback
        for(uint32_t bk2=0;bk2<2;++bk2){ auto blk=untilize32(half(raw,bk2));
            for(uint32_t k2loc=0;k2loc<T;++k2loc)for(uint32_t k1=0;k1<N1;++k1)X[(bk2*T+k2loc)+N2*k1]=blk[k2loc*T+k1]; }
        return X; }
    std::vector<uint32_t> inv2048(Chan& c, const std::vector<uint32_t>& X){
        for(uint32_t bk2=0;bk2<2;++bk2){ std::vector<uint32_t> blk(ELEMS);
            for(uint32_t k1=0;k1<N1;++k1)for(uint32_t k2loc=0;k2loc<T;++k2loc)blk[k1*T+k2loc]=X[(bk2*T+k2loc)+N2*k1]; wrb(bIn[bk2],tilize32(blk)); }
        for(uint32_t bk2=0;bk2<2;++bk2){ fusedA(c,bIn[bk2],c.Winv32,bSum[bk2],0); mmul(c,bSum[bk2],c.TwI[bk2],bB[bk2],1); }
        for(uint32_t bn2=0;bn2<2;++bn2){ fusedC(c,bB[0],c.Winv64[0*2+bn2],bAcc[bn2],0); fusedC(c,bB[1],c.Winv64[1*2+bn2],bAcc[bn2],1); madd2(c,bAcc[bn2],bOut2,bn2); }
        std::vector<uint32_t> x(N); auto raw=rdb(bOut2);         // both blocks in ONE readback
        for(uint32_t bn2=0;bn2<2;++bn2){ auto blk=untilize32(half(raw,bn2));
            for(uint32_t n1=0;n1<N1;++n1)for(uint32_t n2loc=0;n2loc<T;++n2loc)x[n1+32*(bn2*T+n2loc)]=blk[n1*T+n2loc]; }
        return x; }
    std::vector<uint32_t> to_mont(uint32_t ch, std::vector<uint32_t> F){ for(auto& v:F) v=ctxp[ch].to_montgomery(v); return F; }

    // ---- buffer-resident NTT: same dispatch chains as fwd2048/inv2048, but DRAM-in/DRAM-out ----
    // fwd2048/inv2048 each end in a BLOCKING readback plus a host reindex, and after the workload cache
    // landed those readbacks are ~43% of replay time (and, being host boundaries, they are what would
    // fragment a Metal Trace into ~1668 segments). These variants drop both: no readback, and the reindex
    // becomes an on-device 32x32 tile transpose.
    //
    // The transpose is EXACT, not an approximation of the host reindex. Host fwd writes
    //   X[(bk2*T+k2loc) + N2*k1] = blkOut[bk2][k2loc*T + k1]
    // and host inv reads it back as
    //   blkIn[bk2][k1*T + k2loc] = X[(bk2*T+k2loc) + N2*k1]
    // so blkIn[bk2] is exactly the within-tile transpose of blkOut[bk2] -- the same identity
    // he_mul_cross.cpp relies on ("inv's input tile = transpose of fwd's output tile").
    // Ops BETWEEN a fwd and an inv are elementwise (twiddle, MAC), so they are layout-agnostic and need
    // no transpose; only the fwd->inv hand-off does.
    // in_os/out_os are TILE indices, so a poly's two blocks can be transposed in place within packed
    // 2-tile buffers (reader_i32/writer_i32 both already take a start-tile arg).
    void transpose_dev(MeshBuf in, MeshBuf out, uint32_t in_os = 0, uint32_t out_os = 0) {
        emit([&]{ Program p = CreateProgram();
            mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/transpose_tile.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)in->address(),1u,in_os}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,out_os}); SetRuntimeArgs(p,cp,core,{});
            return p; }); }

    // Stage-A/C with an input TILE offset, via reader_fused_off -- lets a stage read block bk2 out of a
    // PACKED 2-tile buffer instead of needing one buffer per block. Same kernels/CBs as fusedA/fusedC.
    void fusedA_off(Chan& c, MeshBuf xin, uint32_t xoff, MeshBuf w, MeshBuf out, uint32_t os) {
        emit([&]{ Program p = CreateProgram();
        mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::Int8,2,I8);
        mkcb(p,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcb(p,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*xin).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused_off.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)xin->address(),(uint32_t)w->address(),xoff}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,os});
        SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14}); return p; }); }
    void fusedC_off(Chan& c, MeshBuf gp, uint32_t goff, MeshBuf w, MeshBuf out, uint32_t os) {
        emit([&]{ Program p = CreateProgram();
        mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::Int8,NB*NB,I8);
        mkcb(p,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcb(p,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*gp).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused_off.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)gp->address(),(uint32_t)w->address(),goff}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,os});
        SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14}); return p; }); }

    // PACKED forms: one 2-tile buffer per poly (tile bk2 = block bk2), which is what mac_ntt/mmul consume.
    // in2 must hold the same reshaped input fwd2048_buf expects, blocks at tiles 0 and 1.
    void fwd2048_buf2(Chan& c, MeshBuf in2, MeshBuf out2) {
        for (uint32_t bk2=0; bk2<2; ++bk2) {
            fusedA_off(c, in2, 0, c.WA[bk2*2+0], bAcc[bk2], 0);
            fusedA_off(c, in2, 1, c.WA[bk2*2+1], bAcc[bk2], 1);
            madd2(c, bAcc[bk2], bSum[bk2]);
            mmul(c, bSum[bk2], c.TwF[bk2], bB[bk2], 1);
            fusedC_off(c, bB[bk2], 0, c.WCt, out2, bk2); } }
    void mkcbr(Program& p, const CoreRange& crs, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t pg) {
        CircularBufferConfig cc(depth * pg, {{idx, fmt}}); cc.set_page_size(idx, pg); CreateCircularBuffer(p, crs, cc); }

    // STAGE-WISE CO-SCHEDULED forward transforms: n INDEPENDENT fwd2048_buf2's in 10 programs instead of
    // 10n. Each of the four stages (fusedA ×2, madd, twiddle-mmul, fusedC) becomes ONE program spread over
    // n cores, per bk2. This is the lever the measurements point at: after cache+trace+residency, expansion's
    // residual is device compute SERIALIZED on one core (readback 96% at 0.506 ms/call), and a key-switch has
    // ell*K = 24 independent transforms. Compile args are taken from instance 0 -- valid because every buffer
    // here comes from mkbuf() with identical DRAM-interleaved config, so the accessor layout is the same.
    // Falls back to the sequential form when n exceeds the grid.
    void fwd2048_buf2_batch(Chan& c, const std::vector<MeshBuf>& in2s, const std::vector<MeshBuf>& out2s) {
        std::vector<Chan*> ch(in2s.size(), &c); fwd2048_buf2_batch(ch, in2s, out2s); }

    // Per-instance channels. The fold/twiddle buffer ADDRESSES and the modulus params (q, qinv, qlo, qhi,
    // k14) are all RUNTIME args, so a single program can mix RNS channels -- each core just receives its
    // own. That widens a key-switch's transform batch from ell to ell*K instances.
    void fwd2048_buf2_batch(const std::vector<Chan*>& chans, const std::vector<MeshBuf>& in2s, const std::vector<MeshBuf>& out2s) {
        const uint32_t n = (uint32_t)in2s.size();
        if (n == 0) return;
        const auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX = std::min(n, (uint32_t)grid.x), GY = (n + GX - 1) / GX;
        static const bool no_batch = (std::getenv("PIR_NOBATCH") != nullptr);   // A/B for the variance probe
        if (n == 1 || no_batch || GY > (uint32_t)grid.y) {
            for (uint32_t i = 0; i < n; ++i) fwd2048_buf2(*chans[i], in2s[i], out2s[i]); return; }
        CoreRange crs(CoreCoord{0, 0}, CoreCoord{GX - 1, GY - 1});
        auto CC = [&](uint32_t i) { return CoreCoord{i % GX, i / GX}; };

        std::vector<MeshBuf> acc(n), sum(n), bb(n);
        for (uint32_t i = 0; i < n; ++i) { acc[i] = mkbuf(2, I32); sum[i] = mkbuf(1, I32); bb[i] = mkbuf(1, I32); }

        for (uint32_t bk2 = 0; bk2 < 2; ++bk2) {
            // stage A, both input blocks -> acc tiles 0 and 1
            for (uint32_t half_i = 0; half_i < 2; ++half_i) {
                emit([&]{ Program p = CreateProgram();
                    mkcbr(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcbr(p,crs,CBIndex::c_1,tt::DataFormat::Int8,2,I8);
                    mkcbr(p,crs,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcbr(p,crs,CBIndex::c_3,tt::DataFormat::Int32,NB,I32);
                    mkcbr(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                    std::vector<uint32_t> rc; TensorAccessorArgs(*in2s[0]).append_to(rc); TensorAccessorArgs(*chans[0]->WA[0]).append_to(rc);
                    auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused_off.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
                    auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                    auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                    for (uint32_t i = 0; i < n; ++i) { auto cc = CC(i);
                        SetRuntimeArgs(p,rd,cc,{(uint32_t)in2s[i]->address(),(uint32_t)chans[i]->WA[bk2*2+half_i]->address(),half_i});
                        SetRuntimeArgs(p,wr,cc,{(uint32_t)acc[i]->address(),1u,half_i});
                        SetRuntimeArgs(p,cp,cc,{chans[i]->q,chans[i]->qinv,chans[i]->qlo,chans[i]->qhi,chans[i]->k14}); }
                    return p; }); }
            // modular add of the two partials
            emit([&]{ Program p = CreateProgram();
                mkcbr(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcbr(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t i = 0; i < n; ++i) { auto cc = CC(i);
                    SetRuntimeArgs(p,rd,cc,{(uint32_t)acc[i]->address(),2u,0u});
                    SetRuntimeArgs(p,wr,cc,{(uint32_t)sum[i]->address(),1u,0u});
                    SetRuntimeArgs(p,cp,cc,{2u,chans[i]->q}); }
                return p; });
            // twiddle multiply
            emit([&]{ Program p = CreateProgram();
                mkcbr(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcbr(p,crs,CBIndex::c_1,tt::DataFormat::UInt32,2,I32);
                mkcbr(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_1pass.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t i = 0; i < n; ++i) { auto cc = CC(i);
                    SetRuntimeArgs(p,rd,cc,{(uint32_t)sum[i]->address(),(uint32_t)chans[i]->TwF[bk2]->address(),1u,0u});
                    SetRuntimeArgs(p,wr,cc,{(uint32_t)bb[i]->address(),1u,0u});
                    SetRuntimeArgs(p,cp,cc,{1u,chans[i]->q,chans[i]->qinv,chans[i]->qlo,chans[i]->qhi}); }
                return p; });
            // stage C -> out2 tile bk2
            emit([&]{ Program p = CreateProgram();
                mkcbr(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcbr(p,crs,CBIndex::c_1,tt::DataFormat::Int8,NB*NB,I8);
                mkcbr(p,crs,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcbr(p,crs,CBIndex::c_3,tt::DataFormat::Int32,NB,I32);
                mkcbr(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                std::vector<uint32_t> rc; TensorAccessorArgs(*bb[0]).append_to(rc); TensorAccessorArgs(*chans[0]->WCt).append_to(rc);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused_off.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t i = 0; i < n; ++i) { auto cc = CC(i);
                    SetRuntimeArgs(p,rd,cc,{(uint32_t)bb[i]->address(),(uint32_t)chans[i]->WCt->address(),0u});
                    SetRuntimeArgs(p,wr,cc,{(uint32_t)out2s[i]->address(),1u,bk2});
                    SetRuntimeArgs(p,cp,cc,{chans[i]->q,chans[i]->qinv,chans[i]->qlo,chans[i]->qhi,chans[i]->k14}); }
                return p; });
        } }

    void inv2048_buf2(Chan& c, MeshBuf in2, MeshBuf out2) {
        for (uint32_t bk2=0; bk2<2; ++bk2) {
            fusedA_off(c, in2, bk2, c.Winv32, bSum[bk2], 0);
            mmul(c, bSum[bk2], c.TwI[bk2], bB[bk2], 1); }
        for (uint32_t bn2=0; bn2<2; ++bn2) {
            fusedC_off(c, bB[0], 0, c.Winv64[0*2+bn2], bAcc[bn2], 0);
            fusedC_off(c, bB[1], 0, c.Winv64[1*2+bn2], bAcc[bn2], 1);
            madd2(c, bAcc[bn2], out2, bn2); } }

    // inb[2]/outb[2] are single-tile buffers, one per bk2/bn2 block. No readback, no Finish.
    void fwd2048_buf(Chan& c, std::vector<MeshBuf>& inb, std::vector<MeshBuf>& outb) {
        for (uint32_t bk2=0; bk2<2; ++bk2) {
            fusedA(c, inb[0], c.WA[bk2*2+0], bAcc[bk2], 0);
            fusedA(c, inb[1], c.WA[bk2*2+1], bAcc[bk2], 1);
            madd2(c, bAcc[bk2], bSum[bk2]);
            mmul(c, bSum[bk2], c.TwF[bk2], bB[bk2], 1);
            fusedC(c, bB[bk2], c.WCt, outb[bk2], 0); } }
    void inv2048_buf(Chan& c, std::vector<MeshBuf>& inb, std::vector<MeshBuf>& outb) {
        for (uint32_t bk2=0; bk2<2; ++bk2) {
            fusedA(c, inb[bk2], c.Winv32, bSum[bk2], 0);
            mmul(c, bSum[bk2], c.TwI[bk2], bB[bk2], 1); }
        for (uint32_t bn2=0; bn2<2; ++bn2) {
            fusedC(c, bB[0], c.Winv64[0*2+bn2], bAcc[bn2], 0);
            fusedC(c, bB[1], c.Winv64[1*2+bn2], bAcc[bn2], 1);
            madd2(c, bAcc[bn2], outb[bn2]); } }

    // ---- signed permutation on device: Galois automorphism (#14) + monomial multiply (#15) ----
    // These were the last host ops in the Subs tree and forced a readback per key-switch output.
    // The index table is FIXED per (j or k), so one table serves every node at a Subs level.

    // natural coefficient index -> flat TILIZED position, matching pir::tilize_poly(v, nt)
    static uint32_t tpos(uint32_t x) {
        const uint32_t tile = x / ELEMS, r = (x % ELEMS) / T, c = (x % ELEMS) % T;
        return tile * ELEMS + ((r/16)*2 + c/16)*256 + (r%16)*16 + (c%16); }

    // table[out_tilized] = src_tilized | (sign<<31) for r[(i*j) mod 2N] = ±a[i]  (automorph)
    std::vector<uint32_t> perm_table_automorph(uint64_t j) const {
        std::vector<uint32_t> tab(N);
        for (uint32_t i = 0; i < N; ++i) {
            const uint64_t t = (i * j) % (2ull * N);
            const uint32_t o = (uint32_t)(t < N ? t : t - N), sgn = (t < N) ? 0u : 1u;
            tab[tpos(o)] = tpos(i) | (sgn << 31); }
        return tab; }

    // table for monomial multiply by x^k (k may be negative): r[(i+k) mod 2N] = ±a[i]
    std::vector<uint32_t> perm_table_monomul(int64_t k) const {
        std::vector<uint32_t> tab(N);
        const int64_t M = 2ll * N; int64_t kk = ((k % M) + M) % M;
        for (uint32_t i = 0; i < N; ++i) {
            const uint64_t t = (uint64_t)(((int64_t)i + kk) % M);
            const uint32_t o = (uint32_t)(t < N ? t : t - N), sgn = (t < N) ? 0u : 1u;
            tab[tpos(o)] = tpos(i) | (sgn << 31); }
        return tab; }

    // flat position of natural index x in the INV-OUTPUT block layout (inverse of read_inv_out2)
    static uint32_t invpos(uint32_t x) {
        const uint32_t n1 = x % 32, rest = x / 32, bn2 = rest / T, n2loc = rest % T;
        const uint32_t r = n1, c = n2loc;
        return bn2 * ELEMS + ((r/16)*2 + c/16)*256 + (r%16)*16 + (c%16); }

    // flat position of natural index x in the FWD-INPUT layout (what write_fwd_in2 builds):
    //   blk[b*T+n1] = x[n1 + 32*(bn2*T+b)]
    static uint32_t fwdinpos(uint32_t x) {
        const uint32_t n1 = x % 32, rest = x / 32, bn2 = rest / T, b = rest % T;
        const uint32_t r = b, c = n1;
        return bn2 * ELEMS + ((r/16)*2 + c/16)*256 + (r%16)*16 + (c%16); }

    // natural tilized -> fwd-input layout, as a permutation (identity map, layout only)
    std::vector<uint32_t> perm_table_natural_to_fwdin() const {
        std::vector<uint32_t> tab(N);
        for (uint32_t i = 0; i < N; ++i) tab[fwdinpos(i)] = tpos(i);
        return tab; }

    // Layout conversion as a PERMUTATION: read an inv-layout buffer, write natural tilized order. The
    // reindex is a fixed permutation, so it costs one permute program and composes with any other
    // permutation for free (that is why keyswitch can return buffers instead of reading back).
    std::vector<uint32_t> perm_table_inv_to_natural() const {
        std::vector<uint32_t> tab(N);
        for (uint32_t i = 0; i < N; ++i) tab[tpos(i)] = invpos(i);   // identity permutation, layout only
        return tab; }

    MeshBuf perm_table_buf(const std::vector<uint32_t>& tab) { auto b = mkbuf(2, I32); wrb(b, tab); return b; }

    // in2/out2 are packed 2-tile polys; tables come from perm_table_*(). No readback.
    void permute_dev(MeshBuf in2, MeshBuf tab, MeshBuf out2, uint32_t q,
                     uint32_t in_off = 0, uint32_t in_str = 1, uint32_t nt_out = 2, uint32_t nt_in = 0) {
        emit([&]{ Program p = CreateProgram();
            const uint32_t din = nt_in ? nt_in : nt_out;
            mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,din,I32);
            mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,nt_out,I32);
            mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,nt_out,I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*in2).append_to(rc);
            TensorAccessorArgs(*tab).append_to(rc); TensorAccessorArgs(*out2).append_to(rc);
            auto k = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/permute_signed.cpp", core,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1, .noc=NOC::RISCV_1_default, .compile_args=rc});
            SetRuntimeArgs(p, k, core, {(uint32_t)in2->address(), (uint32_t)tab->address(),
                                        (uint32_t)out2->address(), nt_out, q, in_off, in_str, nt_in});
            return p; }); }

    // CO-SCHEDULED permutes: n independent signed permutations in ONE program over n cores. The kernel is a
    // scalar 2048-iteration L1 loop on a single RISC, so running them one at a time is a net LOSS versus the
    // host (measured: A5 made expansion 64.4 -> 78.3 ms until these were batched). Per-instance q, so a batch
    // may mix RNS channels.
    void permute_batch(const std::vector<MeshBuf>& ins, MeshBuf tab, const std::vector<MeshBuf>& outs,
                       const std::vector<uint32_t>& qs,
                       const std::vector<uint32_t>& offs = {}, const std::vector<uint32_t>& strs = {}) {
        const uint32_t n = (uint32_t)ins.size(); if (n == 0) return;
        const auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX = std::min(n, (uint32_t)grid.x), GY = (n + GX - 1) / GX;
        if (n == 1 || GY > (uint32_t)grid.y) {
            for (uint32_t i = 0; i < n; ++i) permute_dev(ins[i], tab, outs[i], qs[i],
                offs.empty()?0u:offs[i], strs.empty()?1u:strs[i]); return; }
        CoreRange crs(CoreCoord{0,0}, CoreCoord{GX-1, GY-1});
        emit([&]{ Program p = CreateProgram();
            mkcbr(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32);
            mkcbr(p,crs,CBIndex::c_1,tt::DataFormat::UInt32,2,I32);
            mkcbr(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*ins[0]).append_to(rc);
            TensorAccessorArgs(*tab).append_to(rc); TensorAccessorArgs(*outs[0]).append_to(rc);
            auto k = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/permute_signed.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1, .noc=NOC::RISCV_1_default, .compile_args=rc});
            for (uint32_t i = 0; i < n; ++i) { CoreCoord cc{i % GX, i / GX};
                SetRuntimeArgs(p, k, cc, {(uint32_t)ins[i]->address(), (uint32_t)tab->address(),
                                          (uint32_t)outs[i]->address(), 2u, qs[i],
                                          offs.empty()?0u:offs[i], strs.empty()?1u:strs[i], 0u}); }
            return p; }); }

    // convenience: host poly -> device permute -> host poly (for validation / incremental adoption)
    std::vector<uint32_t> permute_host(const std::vector<uint32_t>& a, MeshBuf tab, uint32_t q) {
        auto bi = mkbuf(2, I32), bo = mkbuf(2, I32);
        wrb(bi, til2(a)); permute_dev(bi, tab, bo, q); return unt2(rdb(bo)); }

    // host poly -> ONE packed 2-tile input buffer for fwd2048_buf2 (blocks at tiles 0 and 1)
    void write_fwd_in2(MeshBuf in2, const std::vector<uint32_t>& x) {
        std::vector<uint32_t> packed(2*ELEMS);
        for (uint32_t bn2=0; bn2<2; ++bn2) { std::vector<uint32_t> blk(ELEMS);
            for (uint32_t b=0;b<T;++b) for (uint32_t n1=0;n1<N1;++n1) blk[b*T+n1]=x[n1+32*(bn2*T+b)];
            auto t=tilize32(blk); std::copy(t.begin(),t.end(),packed.begin()+bn2*ELEMS); }
        wrb(in2, packed); }
    // host poly -> the INV-OUTPUT block layout (the inverse of read_inv_out2's reindex). Needed whenever a
    // natural-order poly must be combined elementwise with an inv2048_buf2 result while both stay on device:
    // o2 lives in block layout blk[n1*T+n2loc] = x[n1 + 32*(bn2*T+n2loc)], NOT natural tilized order, so
    // til2() operands would be added at mismatched positions (observed as selectors 97896/196608).
    void write_inv_layout(MeshBuf out2, const std::vector<uint32_t>& v) {
        std::vector<uint32_t> packed(2*ELEMS);
        for (uint32_t bn2=0; bn2<2; ++bn2) { std::vector<uint32_t> blk(ELEMS);
            for (uint32_t n1=0;n1<N1;++n1) for (uint32_t n2loc=0;n2loc<T;++n2loc)
                blk[n1*T+n2loc] = v[n1 + 32*(bn2*T+n2loc)];
            auto t=tilize32(blk); std::copy(t.begin(), t.end(), packed.begin()+bn2*ELEMS); }
        wrb(out2, packed); }

    // packed 2-tile output of inv2048_buf2 -> host coefficient poly
    std::vector<uint32_t> read_inv_out2(MeshBuf out2) {
        auto raw=rdb(out2); std::vector<uint32_t> x(N);
        for (uint32_t bn2=0; bn2<2; ++bn2) { auto blk=untilize32(half(raw,bn2));
            for (uint32_t n1=0;n1<N1;++n1) for (uint32_t n2loc=0;n2loc<T;++n2loc) x[n1+32*(bn2*T+n2loc)]=blk[n1*T+n2loc]; }
        return x; }

    // host poly -> the 2 input tiles fwd2048_buf expects (same reshape as fwd2048's input stage)
    void write_fwd_in(std::vector<MeshBuf>& inb, const std::vector<uint32_t>& x) {
        for (uint32_t bn2=0; bn2<2; ++bn2) { std::vector<uint32_t> blk(ELEMS);
            for (uint32_t b=0;b<T;++b) for (uint32_t n1=0;n1<N1;++n1) blk[b*T+n1]=x[n1+32*(bn2*T+b)];
            wrb(inb[bn2], tilize32(blk)); } }
    // the 2 output tiles of inv2048_buf -> host poly (same reindex as inv2048's output stage)
    std::vector<uint32_t> read_inv_out(std::vector<MeshBuf>& outb) {
        std::vector<uint32_t> x(N);
        for (uint32_t bn2=0; bn2<2; ++bn2) { auto blk=untilize32(rdb(outb[bn2]));
            for (uint32_t n1=0;n1<N1;++n1) for (uint32_t n2loc=0;n2loc<T;++n2loc) x[n1+32*(bn2*T+n2loc)]=blk[n1*T+n2loc]; }
        return x; }
    // the 2 output tiles of fwd2048_buf -> host freq poly (same reindex as fwd2048's output stage);
    // for validation only -- the resident path never calls this.
    std::vector<uint32_t> read_fwd_out(std::vector<MeshBuf>& outb) {
        std::vector<uint32_t> X(N);
        for (uint32_t bk2=0; bk2<2; ++bk2) { auto blk=untilize32(rdb(outb[bk2]));
            for (uint32_t k2loc=0;k2loc<T;++k2loc) for (uint32_t k1=0;k1<N1;++k1) X[(bk2*T+k2loc)+N2*k1]=blk[k2loc*T+k1]; }
        return X; }

    // ---- Garner CRT-combine (device) ----
    uint32_t invm(uint32_t a, uint32_t b){ return ctxp[b].to_montgomery((uint32_t)mont::modular_inverse(primes[a], primes[b])); }
    void stage_ab(const char* kernel, MeshBuf bA, MeshBuf bB, MeshBuf bO, const std::vector<uint32_t>& cargs,
                  uint32_t a_start, uint32_t b_start, uint32_t nt=2){
        auto sh=split_stream(nt); auto cr=shard_set(sh.C);
        if (!cached()) { Program p=CreateProgram(); mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX + std::string(kernel),cr,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,nt-o);
            auto ca=cargs; ca[1]=n;                       // garner_diff takes {q, n_tiles}
            SetRuntimeArgs(p,rd,cc,{(uint32_t)bA->address(),(uint32_t)bB->address(),n,a_start+o,b_start+o});
            SetRuntimeArgs(p,wr,cc,{(uint32_t)bO->address(),n,o}); SetRuntimeArgs(p,cp,cc,ca); }
        run(p); } }
    void stage_ab(const char* kernel, MeshBuf bA, MeshBuf bB, MeshBuf bO, const std::vector<uint32_t>& cargs){
        emit([&]{ Program p = CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX + std::string(kernel),core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)bA->address(),(uint32_t)bB->address(),2u,0u,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)bO->address(),2u,0u}); SetRuntimeArgs(p,cp,core,cargs);
        return p; }); }
    void stage_crt(MeshBuf bT, uint32_t stride, uint32_t off, MeshBuf bR1, MeshBuf bParQ, MeshBuf bO, uint32_t r1_off, uint32_t nt=2){
        auto sh=split_stream(nt); auto cr=shard_set(sh.C);
        if (!cached()) { Program p=CreateProgram(); mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_2,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,4,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_crt_stride.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/crt_mul_split.cpp",cr,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,nt-o);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)bT->address(),stride,off+o*stride,(uint32_t)bR1->address(),(uint32_t)bParQ->address(),n,r1_off+o});
            SetRuntimeArgs(p,wr,cc,{(uint32_t)bO->address(),2u*n,2u*o}); SetRuntimeArgs(p,cp,cc,{n}); }
        run(p); } }
    void stage_crt(MeshBuf bT, uint32_t stride, uint32_t off, MeshBuf bR1, MeshBuf bParQ, MeshBuf bO){
        emit([&]{ Program p = CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_2,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,4,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_crt_stride.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/crt_mul_split.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)bT->address(),stride,off,(uint32_t)bR1->address(),(uint32_t)bParQ->address(),2u,0u});   // trailing 0u = r1_off
        SetRuntimeArgs(p,wr,core,{(uint32_t)bO->address(),4u,0u}); SetRuntimeArgs(p,cp,core,{2u});
        return p; }); }

    // residues[K] (natural) -> K radix-2^28 limb polys (natural)
    // RESIDENT Garner CRT: residue BUFFERS in -> packed limb buffer out, no readback.
    // Output layout is tile-major / limb-minor (tile index = t*K + limb), which is exactly the order
    // reader_gadget pushes -- so gadgetK_buf can consume it with a plain sequential reader. (The
    // "interleave gotcha" in ONION_MATCH_PLAN §10b concerns the single-bX + reader_i32 path, not this one.)
    MeshBuf crtK_buf(const std::vector<MeshBuf>& bR){
        auto cbuf=[&](uint32_t val){ auto b=mkbuf(2,I32); wrb(b, til2(std::vector<uint32_t>(N,val))); return b; };
        if (K==2){
            auto bD=mkbuf(2,I32), bV1=mkbuf(2,I32), bX=mkbuf(4,I32);
            stage_ab("compute/garner_diff.cpp", bR[0], bR[1], bD, {primes[1], 2u});
            stage_ab("compute/modmul_sfpi_1pass.cpp", bD, cbuf(invm(0,1)), bV1, {2u, primes[1], ctxp[1].q_inv_32(), primes[1]&0xFFFF, primes[1]>>16});
            stage_crt(bV1, 1, 0, bR[0], bPar[0], bX);
            return bX;
        } else {
            auto q1=primes[1], q2=primes[2];
            auto bD01=mkbuf(2,I32), bV1=mkbuf(2,I32), bA=mkbuf(2,I32), bBt=mkbuf(2,I32), bCC=mkbuf(2,I32), bV2=mkbuf(2,I32);
            stage_ab("compute/garner_diff.cpp", bR[0], bR[1], bD01, {q1, 2u});
            stage_ab("compute/modmul_sfpi_1pass.cpp", bD01, cbuf(invm(0,1)), bV1, {2u, q1, ctxp[1].q_inv_32(), q1&0xFFFF, q1>>16});
            stage_ab("compute/garner_diff.cpp", bR[0], bR[2], bA, {q2, 2u});
            stage_ab("compute/modmul_sfpi_1pass.cpp", bA, cbuf(invm(0,2)), bBt, {2u, q2, ctxp[2].q_inv_32(), q2&0xFFFF, q2>>16});
            stage_ab("compute/garner_diff.cpp", bV1, bBt, bCC, {q2, 2u});
            stage_ab("compute/modmul_sfpi_1pass.cpp", bCC, cbuf(invm(1,2)), bV2, {2u, q2, ctxp[2].q_inv_32(), q2&0xFFFF, q2>>16});
            auto bW=mkbuf(4,I32), bP=mkbuf(4,I32), bQ=mkbuf(4,I32), bX=mkbuf(6,I32);
            stage_crt(bV2, 1, 0, bV1, bPar[1], bW);
            stage_crt(bW, 2, 0, bR[0], bPar[0], bP);
            stage_crt(bW, 2, 1, bZero2, bPar[0], bQ);
            { emit([&]{ Program p = CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,4,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,4,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,6,I32);
              auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_pq.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/crt_limb3.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              SetRuntimeArgs(p,rd,core,{(uint32_t)bP->address(),(uint32_t)bQ->address(),2u}); SetRuntimeArgs(p,wr,core,{(uint32_t)bX->address(),6u,0u}); SetRuntimeArgs(p,cp,core,{2u});
              return p; }); }
            return bX;
        } }

    // host-facing wrapper: unchanged semantics, one readback at the end
    std::vector<std::vector<uint32_t>> crtK(const std::vector<std::vector<uint32_t>>& rj){
        std::vector<MeshBuf> bR(K); for (uint32_t j=0;j<K;++j){ bR[j]=mkbuf(2,I32); wrb(bR[j], til2(rj[j])); }
        auto raw = rdb(crtK_buf(bR));
        std::vector<std::vector<uint32_t>> limbs(K);
        for (uint32_t lb=0;lb<K;++lb){ std::vector<uint32_t> t(2*ELEMS);
            std::copy(raw.begin()+(0*K+lb)*ELEMS, raw.begin()+(0*K+lb+1)*ELEMS, t.begin());
            std::copy(raw.begin()+(1*K+lb)*ELEMS, raw.begin()+(1*K+lb+1)*ELEMS, t.begin()+ELEMS); limbs[lb]=unt2(t); }
        return limbs; }

    // RESIDENT gadget: packed limb buffer in -> digit buffer out, no readback. Reads bX SEQUENTIALLY
    // (n = 2K tiles) because its tile order already matches reader_gadget's (t*K + limb) push order.
    // bdig layout: digit k occupies tiles k and ell+k (stride ell, offset k).
    MeshBuf gadgetK_buf(MeshBuf bX){
        auto bdig=mkbuf(2*ell,I32);
        emit([&]{ Program p = CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2*K,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2*ell,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/gadget_decompose_signed.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false,
              .defines={{"GADGET_B",std::to_string(B)},{"GADGET_LOG2B",std::to_string(LOG2B)},{"GADGET_ELL",std::to_string(ell)},{"GADGET_KLIMBS",std::to_string(K)}}});
        SetRuntimeArgs(p,rd,core,{(uint32_t)bX->address(), 2u*K, 0u});
        SetRuntimeArgs(p,wr,core,{(uint32_t)bdig->address(), 2u*ell, 0u});
        SetRuntimeArgs(p,cp,core,{2u});
        return p; });
        return bdig; }

    MeshBuf crt_and_gadget_buf(const std::vector<MeshBuf>& bR){ return gadgetK_buf(crtK_buf(bR)); }

    // A4: out = (a + b) mod q for two SEPARATE buffers (modadd_reduce only sums within one buffer).
    // Needed so ct_add and c0+mac can operate on buffers, which is what lets the Subs tree stay resident.
    void padd_ab(Chan& c, MeshBuf bA, MeshBuf bB, MeshBuf bOut, uint32_t nt = 2){
        emit([&]{ Program p = CreateProgram();
            mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32);
            mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_ab.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)bA->address(),(uint32_t)bB->address(),nt,0u,0u});
            SetRuntimeArgs(p,wr,core,{(uint32_t)bOut->address(),nt,0u});
            SetRuntimeArgs(p,cp,core,{c.q,nt});
            return p; }); }

    // Sharded batched-GEMM scan, lifted verbatim from the validated inline version in test_scan_gemm.cpp
    // (bit-exact 0/2097152 at m=128/N0=512, 512 slots, 32 cores). bA = DB limb tiles [slot][mt][kt][limb],
    // bB = query limb tiles [slot][kt][limb], bOut = n_out groups of 8 int32 planes -> feed scan_recombine.
    // n_out = slots*MT; MTS = MT (m-tiles per slot, so the reader advances B once per slot).
    void scan_gemm(MeshBuf bA, MeshBuf bB, MeshBuf bOut, uint32_t n_out, uint32_t KB, uint32_t MTS,
                   uint32_t ncores = 32) {
        const auto grid = mesh->compute_with_storage_grid_size();
        uint32_t NC = std::min(ncores, (uint32_t)(grid.x * grid.y));
        while (NC > 1 && n_out % NC) --NC;                       // keep the split exact
        const uint32_t GX = std::min(NC, (uint32_t)grid.x), GY = (NC + GX - 1) / GX;
        const uint32_t n_per = n_out / NC;
        emit([&]{ Program p = CreateProgram();
            CoreRange crs(CoreCoord{0,0}, CoreCoord{GX-1, GY-1});
            mkcbr(p,crs,CBIndex::c_0,tt::DataFormat::Int8,2*4,I8);
            mkcbr(p,crs,CBIndex::c_1,tt::DataFormat::Int8,KB*4,I8);      // whole slot's B resident in L1
            mkcbr(p,crs,CBIndex::c_16,tt::DataFormat::Int32,2*8,I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*bA).append_to(rc); TensorAccessorArgs(*bB).append_to(rc);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_scan_gemm.cpp",crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/scan_gemm.cpp",crs,
                ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true});
            for (uint32_t ci=0; ci<NC; ++ci) { CoreCoord cc{ci%GX, ci/GX}; const uint32_t o0=ci*n_per;
                SetRuntimeArgs(p,rd,cc,{(uint32_t)bA->address(),(uint32_t)bB->address(),o0,n_per,KB,MTS});
                SetRuntimeArgs(p,wr,cc,{(uint32_t)bOut->address(), n_per*8u, o0*8u});
                SetRuntimeArgs(p,cp,cc,{n_per,KB,MTS}); }
            return p; }); }

    // C1: mod-q recombine of scan_gemm's 8 packed planes, using EXISTING kernels only.
    //   R = Σ_{d=0..3} 128^d·plane_d  +  128^4 · Σ_{e=0..2} 128^e·plane_{e+4}
    // Each sum is a 4-consecutive-tile group -> recombine_bigK verbatim; then one modmul by mont(128^4 mod q)
    // and one mod-add. bplanes holds n_out groups of 8 int32 tiles; returns n_out canonical tiles.
    MeshBuf scan_recombine(Chan& c, MeshBuf bplanes, uint32_t n_out) {
        auto grp = [&](uint32_t base){ auto bo = mkbuf(n_out, I32);
            emit([&]{ Program p = CreateProgram();
                mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,8,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32_groups.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/recombine_bigK.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                SetRuntimeArgs(p,rd,core,{(uint32_t)bplanes->address(), n_out, 4u, 8u, base});
                SetRuntimeArgs(p,wr,core,{(uint32_t)bo->address(), n_out, 0u});
                SetRuntimeArgs(p,cp,core,{n_out, c.q, c.qinv, c.qlo, c.qhi, c.k14});
                return p; });
            return bo; };
        auto bLo = grp(0), bHi = grp(4);
        // scale the high group by 128^4 mod q (Montgomery constant operand), then add
        uint64_t w4 = 1; for (int i=0;i<4;++i) w4 = w4 * 128 % c.q;
        auto bW = mkbuf(n_out, I32);
        { std::vector<uint32_t> v((size_t)n_out*ELEMS, ctxp[0].to_montgomery((uint32_t)w4));
          for (uint32_t j=0;j<K;++j) if (primes[j]==c.q) v.assign((size_t)n_out*ELEMS, ctxp[j].to_montgomery((uint32_t)w4));
          wrb(bW, v); }
        auto bHiS = mkbuf(n_out, I32); mmul(c, bHi, bW, bHiS, n_out);
        auto bR = mkbuf(n_out, I32); padd_ab(c, bLo, bHiS, bR, n_out);
        return bR; }

    // digit k out of bdig (tiles k, ell+k) -> K residue polys, co-scheduled across the ell digits.
    // RESIDENT digit->RNS: launches the co-scheduled split and returns the packed output BUFFER.
    // Layout: residue (k, ch) occupies tiles k*2K + ch and k*2K + K + ch (stride K within the k block),
    // which is what permute_dev's in_off/in_str args are for -- so no readback is needed.
    MeshBuf digit_rns_launch(MeshBuf bdig){
        const auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX = std::min(ell, (uint32_t)grid.x), GY = (ell + GX - 1) / GX;
        auto bo = mkbuf(ell * 2 * K, I32);
        if (GY <= (uint32_t)grid.y) {
            CoreRange crs(CoreCoord{0,0}, CoreCoord{GX-1, GY-1});
            emit([&]{ Program p = CreateProgram();
                mkcbr(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcbr(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2*K,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32_stride.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/digit_to_rns.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false,.defines={{"DR_K",std::to_string(K)}}});
                std::vector<uint32_t> cargs={2u}; for (uint32_t j=0;j<K;++j) cargs.push_back(primes[j]);
                for (uint32_t k=0;k<ell;++k){ CoreCoord cc{k%GX, k/GX};
                    SetRuntimeArgs(p,rd,cc,{(uint32_t)bdig->address(), 2u, ell, k});   // tiles k, ell+k
                    SetRuntimeArgs(p,wr,cc,{(uint32_t)bo->address(), 2u*K, k*2u*K});
                    SetRuntimeArgs(p,cp,cc,cargs); }
                return p; });
        } else { for (uint32_t k=0;k<ell;++k) { /* fallback: sequential */ } }
        return bo; }

    // host-facing wrapper (unchanged semantics): launch + one readback
    std::vector<std::vector<std::vector<uint32_t>>> digit_rns_from_bdig(MeshBuf bdig){
        std::vector<std::vector<std::vector<uint32_t>>> out(ell);
        auto bo = digit_rns_launch(bdig);
        auto raw = rdb(bo);
        for (uint32_t k=0;k<ell;++k){ out[k].resize(K); const size_t base=(size_t)k*2*K*ELEMS;
            for (uint32_t j=0;j<K;++j){ std::vector<uint32_t> t(2*ELEMS);
                std::copy(raw.begin()+base+(size_t)j*ELEMS,     raw.begin()+base+(size_t)(j+1)*ELEMS,   t.begin());
                std::copy(raw.begin()+base+(size_t)(K+j)*ELEMS, raw.begin()+base+(size_t)(K+j+1)*ELEMS, t.begin()+ELEMS);
                out[k][j]=unt2(t); } }
        return out; }

    // K limb polys -> ell signed digit polys (gadget, GADGET_KLIMBS=K)
    std::vector<std::vector<uint32_t>> gadgetK(const std::vector<std::vector<uint32_t>>& limbs){
        std::vector<MeshBuf> bL(K); for (uint32_t j=0;j<K;++j){ bL[j]=mkbuf(2,I32); wrb(bL[j], til2(limbs[j])); }
        auto bdig=mkbuf(2*ell,I32);
        emit([&]{ Program p = CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2*K,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2*ell,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_gadget.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.defines={{"GADGET_KLIMBS",std::to_string(K)}}});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/gadget_decompose_signed.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false,
              .defines={{"GADGET_B",std::to_string(B)},{"GADGET_LOG2B",std::to_string(LOG2B)},{"GADGET_ELL",std::to_string(ell)},{"GADGET_KLIMBS",std::to_string(K)}}});
        std::vector<uint32_t> rargs; for (uint32_t j=0;j<K;++j) rargs.push_back((uint32_t)bL[j]->address()); rargs.push_back(2u);
        SetRuntimeArgs(p,rd,core,rargs); SetRuntimeArgs(p,wr,core,{(uint32_t)bdig->address(),2u*ell,0u}); SetRuntimeArgs(p,cp,core,{2u});
        return p; });
        auto raw=rdb(bdig); std::vector<std::vector<uint32_t>> digs(ell);
        for (uint32_t k=0;k<ell;++k){ std::vector<uint32_t> tt(2*ELEMS);
            std::copy(raw.begin()+k*ELEMS, raw.begin()+(k+1)*ELEMS, tt.begin());
            std::copy(raw.begin()+(ell+k)*ELEMS, raw.begin()+(ell+k+1)*ELEMS, tt.begin()+ELEMS); digs[k]=unt2(tt); }
        return digs; }
    std::vector<std::vector<uint32_t>> crt_and_gadget(const std::vector<std::vector<uint32_t>>& rj){ return gadgetK(crtK(rj)); }

    // digit (natural) -> K residue polys (natural)
    std::vector<std::vector<uint32_t>> digit_rns(const std::vector<uint32_t>& dig){
        auto bi=mkbuf(2,I32), bo=mkbuf(2*K,I32); wrb(bi, til2(dig));
        emit([&]{ Program p = CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2*K,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/digit_to_rns.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false,.defines={{"DR_K",std::to_string(K)}}});
        std::vector<uint32_t> cargs={2u}; for (uint32_t j=0;j<K;++j) cargs.push_back(primes[j]);
        SetRuntimeArgs(p,rd,core,{(uint32_t)bi->address(),2u,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)bo->address(),2u*K,0u}); SetRuntimeArgs(p,cp,core,cargs);
        return p; });
        auto raw=rdb(bo); std::vector<std::vector<uint32_t>> res(K);
        for (uint32_t j=0;j<K;++j){ std::vector<uint32_t> t(2*ELEMS);
            std::copy(raw.begin()+j*ELEMS, raw.begin()+(j+1)*ELEMS, t.begin());
            std::copy(raw.begin()+(K+j)*ELEMS, raw.begin()+(K+j+1)*ELEMS, t.begin()+ELEMS); res[j]=unt2(t); }
        return res; }
    // CO-SCHEDULED digit->RNS split: one program over nd cores, digit d on core d, its K residue polys
    // written to bo at tile offset d*2*K. Independent per digit, so this collapses nd sequential dispatches
    // into one (see mmul_batch for the measured 12x-over-trace rationale). Called ell times per key-switch
    // and 2*ell times per external product, so it is squarely on expansion's hot path.
    // Returns res[d][ch] = residue poly, matching digit_rns()'s per-call output.
    std::vector<std::vector<std::vector<uint32_t>>> digit_rns_batch(const std::vector<std::vector<uint32_t>>& digs) {
        const uint32_t nd = (uint32_t)digs.size();
        std::vector<std::vector<std::vector<uint32_t>>> out(nd);
        if (nd == 0) return out;
        const auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX = std::min(nd, (uint32_t)grid.x), GY = (nd + GX - 1) / GX;
        if (GY > (uint32_t)grid.y) { for (uint32_t d = 0; d < nd; ++d) out[d] = digit_rns(digs[d]); return out; }

        std::vector<MeshBuf> bi(nd);
        for (uint32_t d = 0; d < nd; ++d) { bi[d] = mkbuf(2, I32); wrb(bi[d], til2(digs[d])); }
        auto bo = mkbuf(nd * 2 * K, I32);
        emit([&]{ Program p = CreateProgram();
            CoreRange crs(CoreCoord{0, 0}, CoreCoord{GX - 1, GY - 1});
            { CircularBufferConfig c0(2*I32, {{CBIndex::c_0, tt::DataFormat::UInt32}}); c0.set_page_size(CBIndex::c_0, I32);
              CreateCircularBuffer(p, crs, c0);
              CircularBufferConfig c16(2*K*I32, {{CBIndex::c_16, tt::DataFormat::UInt32}}); c16.set_page_size(CBIndex::c_16, I32);
              CreateCircularBuffer(p, crs, c16); }
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1, .noc=NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0, .noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/digit_to_rns.cpp", crs,
                ComputeConfig{.fp32_dest_acc_en=true, .dst_full_sync_en=true, .math_approx_mode=false,
                              .defines={{"DR_K", std::to_string(K)}}});
            std::vector<uint32_t> cargs={2u}; for (uint32_t j=0;j<K;++j) cargs.push_back(primes[j]);
            for (uint32_t d = 0; d < nd; ++d) { CoreCoord cc{d % GX, d / GX};
                SetRuntimeArgs(p, rd, cc, {(uint32_t)bi[d]->address(), 2u, 0u});
                SetRuntimeArgs(p, wr, cc, {(uint32_t)bo->address(), 2u*K, d*2u*K});
                SetRuntimeArgs(p, cp, cc, cargs); }
            return p; });
        auto raw = rdb(bo);
        for (uint32_t d = 0; d < nd; ++d) { out[d].resize(K);
            const size_t base = (size_t)d * 2 * K * ELEMS;
            for (uint32_t j = 0; j < K; ++j) { std::vector<uint32_t> t(2*ELEMS);
                std::copy(raw.begin()+base+(size_t)j*ELEMS,     raw.begin()+base+(size_t)(j+1)*ELEMS,   t.begin());
                std::copy(raw.begin()+base+(size_t)(K+j)*ELEMS, raw.begin()+base+(size_t)(K+j+1)*ELEMS, t.begin()+ELEMS);
                out[d][j] = unt2(t); } }
        return out; }

    std::vector<uint32_t> pmul(Chan& c, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b){
        auto ba=mkbuf(2,I32), bb=mkbuf(2,I32), bo=mkbuf(2,I32); wrb(ba,til2(a)); wrb(bb,til2(b)); mmul(c,ba,bb,bo,2); return unt2(rdb(bo)); }
    std::vector<uint32_t> padd(Chan& c, const std::vector<std::vector<uint32_t>>& xs){
        uint32_t nin=(uint32_t)xs.size(); std::vector<uint32_t> out(N);
        for (uint32_t t=0;t<2;++t){ auto bi=mkbuf(nin,I32), bo=mkbuf(1,I32);
            std::vector<uint32_t> buf(nin*ELEMS); for (uint32_t i=0;i<nin;++i){ auto tl=til1(half(xs[i],t)); std::copy(tl.begin(),tl.end(),buf.begin()+i*ELEMS); } wrb(bi,buf);
            emit([&]{ Program p = CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,nin,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)bi->address(),nin,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)bo->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{nin,c.q});
            return p; });
            auto r=unt1(rdb(bo)); std::copy(r.begin(),r.end(),out.begin()+t*ELEMS); }
        return out; }

    // Buffer-resident MAC: sum_k a[k] ⊙ b[k] (NTT-domain, one operand Montgomery) -> summed NTT poly.
    // Products are written into ONE DRAM buffer (position-major via mmul offset) with NO per-product
    // readback; a strided modadd_reduce sums each tile-position; a single readback returns the sum.
    // (Device-residency win: removes the ~n per-product blocking readbacks of the naive pmul+padd MAC.)
    // CO-SCHEDULED batch of independent modmuls: ONE program spread over nk cores, each computing
    // out[2k, 2k+1] = a[k] ⊙ b[k]. Separate programs SERIALIZE on the single command queue no matter which
    // core they target, and trace only removes per-dispatch HOST cost — it still issues nk dispatches. One
    // program with disjoint core ranges actually runs them at once: measured 0.0025 vs 0.0313 ms/op, i.e.
    // 12x better than trace (test_dispatch_micro regime D, bit-exact 0/65536).
    // Falls back to sequential if nk exceeds the grid, so callers need no special case.
    void mmul_batch(Chan& c, const std::vector<MeshBuf>& a, const std::vector<MeshBuf>& b, MeshBuf out) {
        const uint32_t nk = (uint32_t)a.size();
        const auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX = std::min(nk, (uint32_t)grid.x), GY = (nk + GX - 1) / GX;
        if (nk == 0) return;
        if (GY > (uint32_t)grid.y) { for (uint32_t k = 0; k < nk; ++k) mmul(c, a[k], b[k], out, 2, 2*k); return; }
        emit([&]{ Program p = CreateProgram();
            CoreRange crs(CoreCoord{0, 0}, CoreCoord{GX - 1, GY - 1});
            auto cbr = [&](CBIndex idx) { CircularBufferConfig cc(2*I32, {{idx, tt::DataFormat::UInt32}});
                cc.set_page_size(idx, I32); CreateCircularBuffer(p, crs, cc); };
            cbr(CBIndex::c_0); cbr(CBIndex::c_1); cbr(CBIndex::c_16);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1, .noc=NOC::RISCV_1_default});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0, .noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_1pass.cpp", crs,
                ComputeConfig{.fp32_dest_acc_en=true, .dst_full_sync_en=true, .math_approx_mode=false});
            for (uint32_t k = 0; k < nk; ++k) { CoreCoord cc{k % GX, k / GX};
                SetRuntimeArgs(p, rd, cc, {(uint32_t)a[k]->address(), (uint32_t)b[k]->address(), 2u, 0u});
                SetRuntimeArgs(p, wr, cc, {(uint32_t)out->address(), 2u, 2*k});
                SetRuntimeArgs(p, cp, cc, {2u, c.q, c.qinv, c.qlo, c.qhi}); }
            return p; }); }

    // Buffer-returning form: the sum stays in DRAM so the inverse transform can consume it directly
    // (no readback + re-upload between MAC and inv). mac_ntt() below is this plus one readback.
    MeshBuf mac_ntt_buf(Chan& c, const std::vector<MeshBuf>& a, const std::vector<MeshBuf>& b) {
        uint32_t nk = (uint32_t)a.size();
        auto bProd = mkbuf(2*nk, I32);
        mmul_batch(c, a, b, bProd);          // nk independent products in ONE co-scheduled program
        auto bS = mkbuf(2, I32);
        for (uint32_t t=0;t<2;++t){
            emit([&]{ Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,nk,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32_stride.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)bProd->address(), nk, 2u, t});
            SetRuntimeArgs(p,wr,core,{(uint32_t)bS->address(), 1u, t}); SetRuntimeArgs(p,cp,core,{nk, c.q});
            return p; }); }
        return bS; }

    std::vector<uint32_t> mac_ntt(Chan& c, const std::vector<MeshBuf>& a, const std::vector<MeshBuf>& b) {
        uint32_t nk = (uint32_t)a.size();
        auto bProd = mkbuf(2*nk, I32);
        for (uint32_t k=0;k<nk;++k) mmul(c, a[k], b[k], bProd, 2, 2*k);   // product k -> tiles 2k,2k+1 (no rdb)
        auto bSum = mkbuf(2, I32);
        for (uint32_t t=0;t<2;++t){
            emit([&]{ Program p = CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,nk,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32_stride.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)bProd->address(), nk, 2u, t});     // gather position t: tiles t,2+t,4+t,...
            SetRuntimeArgs(p,wr,core,{(uint32_t)bSum->address(), 1u, t}); SetRuntimeArgs(p,cp,core,{nk, c.q});
            return p; });
        }
        return unt2(rdb(bSum));
    }
    MeshBuf tilebuf2(const std::vector<uint32_t>& v){ auto b=mkbuf(2,I32); wrb(b, til2(v)); return b; }

    // construct: bake per-channel NTT constants + CRT params
    Engine(distributed::MeshDevice* m, distributed::MeshCommandQueue* q_, std::vector<uint32_t> pr, uint32_t ell_, uint32_t B_)
        : mesh(m), cq(q_), dr(m->shape()), K((uint32_t)pr.size()), ell(ell_), B(B_), LOG2B(__builtin_ctz(B_)), twoL(2*ell_), primes(std::move(pr)) {
        for (uint32_t j=0;j<K;++j) ctxp.emplace_back(primes[j]);
        for (uint32_t j=0;j<K;++j) CH.push_back(make_chan(primes[j]));
        bIn.resize(2); bAcc.resize(2); bSum.resize(2); bB.resize(2); bOut.resize(2);
        for (uint32_t i=0;i<2;++i){ bIn[i]=mkbuf(1,I32); bAcc[i]=mkbuf(2,I32); bSum[i]=mkbuf(1,I32); bB[i]=mkbuf(1,I32); bOut[i]=mkbuf(1,I32); }
        bOut2 = mkbuf(2,I32);
        // ---- setup required by the ported 65 536 application layer ----
        { auto g = mesh->compute_with_storage_grid_size(); GX = g.x; GY = g.y; }
        bIn2 = mkbuf(2,I32); bIn2b = mkbuf(2,I32); bOut2r = mkbuf(2,I32);
        bake_reindex();                              // tFwdIn/tFwdOut/tInvIn/tInvOut gather tables
        bInv.assign(16, nullptr);                    // Garner-chain const operands, uploaded ONCE
        for (uint32_t a=0;a<K;++a) for (uint32_t b=a+1;b<K;++b) {
            auto bf = mkbuf(2,I32); wrb(bf, til2(std::vector<uint32_t>(N, invm(a,b)))); bInv[a*4+b] = bf; }
        bake_batch_folds();                          // MMAX-replicated folds for the batched transforms
        for (uint32_t j=0;j<K;++j){ auto b=mkbuf(2,I32); std::vector<uint32_t> par(2*ELEMS);
            auto tL=til1(std::vector<uint32_t>(ELEMS,primes[j]&0xFFFF)), tH=til1(std::vector<uint32_t>(ELEMS,primes[j]>>16));
            std::copy(tL.begin(),tL.end(),par.begin()); std::copy(tH.begin(),tH.end(),par.begin()+ELEMS); wrb(b,par); bPar.push_back(b); }
        bZero2 = mkbuf(2,I32); wrb(bZero2, std::vector<uint32_t>(2*ELEMS,0));
    }

    // ==== 65 536-row application layer, ported from the `wh` branch ================================
    // Brought over unchanged: wh's `if (!cached()) { ... run(p); }` idiom now resolves to this
    // engine's unified dispatch, so these also gain Metal Trace, which the wh engine never had.
    static constexpr uint32_t CT_TILES(uint32_t k){ return 4*k; }
    void addp_b(Chan& c, MeshBuf a, uint32_t at, MeshBuf b, uint32_t bt, MeshBuf o, uint32_t ot, uint32_t nt=2){
        auto sh=split_stream(nt); auto cr=shard_set(sh.C);
        if (!cached()) { Program p=CreateProgram(); mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_map.cpp",cr,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t oo=i*sh.per, n=std::min(sh.per,nt-oo);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)a->address(),(uint32_t)b->address(),n,at+oo,bt+oo});
            SetRuntimeArgs(p,wr,cc,{(uint32_t)o->address(),n,ot+oo}); SetRuntimeArgs(p,cp,cc,{n,c.q}); }
        run(p); } }
    std::vector<std::vector<uint32_t>> automorph_ct(const std::vector<std::vector<uint32_t>>& ct, int64_t j){
        return permute_polys(ct, inv_mod_2n(j), 0u); }
    void automorph_ct_b(MeshBuf in, MeshBuf out, int64_t j){ permute_ct_b(in,out,inv_mod_2n(j),0u); }
    void automorph_ctb(MeshBuf in, MeshBuf out, int64_t j, uint32_t B){ permute_ctb(in,out,inv_mod_2n(j),0u,B); }
    // Memoized zero operand. It MUST NOT come from the scratch pool: it allocates only on a cache MISS,
    // so a pooled allocation here is consumed on the cold pass and skipped on every warm one, shifting
    // every later slot and tripping "pool order diverged". Zero is a read-only constant -- allocate it
    // outside the pool and keep it for the Engine's lifetime.
    MeshBuf bZeroN(uint32_t nt){ if (nt==2) return bZero2;
        for (auto& z : zeroCache) if (z->size()==(size_t)nt*I32) return z;
        bool was=pooling; pooling=false; auto b=mkbuf(nt,I32); pooling=was;
        wrb(b, std::vector<uint32_t>((size_t)nt*ELEMS,0)); zeroCache.push_back(b); return b; }
    void bake_batch_folds(){
        WArep.resize(K); Winv32rep.resize(K);
        for (uint32_t ch=0; ch<K; ++ch){
            WArep[ch].resize(4);
            for (uint32_t i=0;i<4;++i){                            // replicate the 16-tile fold MMAX×
                auto src = rdb_i8(CH[ch].WA[i], NB*NB);
                std::vector<int8_t> rep((size_t)MMAX*NB*NB*ELEMS);
                for (uint32_t m=0;m<MMAX;++m) std::copy(src.begin(), src.end(), rep.begin()+(size_t)m*NB*NB*ELEMS);
                auto b=mkbuf(MMAX*NB*NB,I8); distributed::EnqueueWriteMeshBuffer(*cq,b,rep,false); ++n_wrb;
                WArep[ch][i]=b; }
            { auto src = rdb_i8(CH[ch].Winv32, NB*NB);
              std::vector<int8_t> rep((size_t)MMAX*NB*NB*ELEMS);
              for (uint32_t m=0;m<MMAX;++m) std::copy(src.begin(), src.end(), rep.begin()+(size_t)m*NB*NB*ELEMS);
              auto b=mkbuf(MMAX*NB*NB,I8); distributed::EnqueueWriteMeshBuffer(*cq,b,rep,false); ++n_wrb;
              Winv32rep[ch]=b; }
        } }
    void bake_reindex(){
        std::vector<std::pair<uint32_t,uint32_t>> fi, fo, ii, io;
        for (uint32_t h=0;h<2;++h) for (uint32_t b=0;b<T;++b) for (uint32_t n1=0;n1<N1;++n1){
            fi.push_back({h*ELEMS + b*T + n1, n1 + 32*(h*T + b)});          // fwd in:  x -> blk_h
            io.push_back({n1 + 32*(h*T + b), h*ELEMS + n1*T + b}); }        // inv out: blk_bn2 -> x
        for (uint32_t k=0;k<2;++k) for (uint32_t k2=0;k2<T;++k2) for (uint32_t k1=0;k1<N1;++k1){
            fo.push_back({(k*T + k2) + N2*k1, k*ELEMS + k2*T + k1});        // fwd out: blk_bk2 -> X
            ii.push_back({k*ELEMS + k1*T + k2, (k*T + k2) + N2*k1}); }      // inv in:  X -> blk_bk2
        tFwdIn=mktable(fi); tFwdOut=mktable(fo); tInvIn=mktable(ii); tInvOut=mktable(io); }
    uint32_t c_idx(Chan& c){ for (uint32_t i=0;i<K;++i) if (CH[i].q==c.q) return i; return 0; }
    void copy_polys(MeshBuf in, MeshBuf out, uint32_t npoly, uint32_t q,
                    uint32_t ib, uint32_t ips, uint32_t its, uint32_t ob, uint32_t ops, uint32_t ots){
        auto sh=split_stream(npoly); auto cr=shard_set(sh.C);
        if (!cached()) { Program p=CreateProgram(); mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,6,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/permute_poly.cpp",cr,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,npoly-o);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)in->address(),(uint32_t)out->address(),2u,1u,0u,n,1u,1u,
                                    ib+o*ips,ips,its, ob+o*ops,ops,ots, 0u, q}); }
        run(p); } }
    MeshBuf crtK_b(MeshBuf src, uint32_t src0, uint32_t nt=2){
        auto cbuf=[&](uint32_t a, uint32_t b){ return bInv[a*4+b]; };         // baked once in the ctor
        auto R=[&](uint32_t ch){ return src0 + ch*nt; };                      // tile offset of residue ch
        if (K==2){
            auto bD=mkbuf(nt,I32), bV1=mkbuf(nt,I32), bX=mkbuf(2*nt,I32);
            stage_ab("compute/garner_diff.cpp", src, src, bD, {primes[1], nt}, R(0), R(1), nt);
            stage_cmul(bD, cbuf(0,1), bV1, nt, {nt, primes[1], ctxp[1].q_inv_32(), primes[1]&0xFFFF, primes[1]>>16});
            stage_crt(bV1, 1, 0, src, bPar[0], bX, R(0), nt);
            return bX;
        }
        auto q1=primes[1], q2=primes[2];
        auto bD01=mkbuf(nt,I32), bV1=mkbuf(nt,I32), bA=mkbuf(nt,I32), bBt=mkbuf(nt,I32), bCC=mkbuf(nt,I32), bV2=mkbuf(nt,I32);
        stage_ab("compute/garner_diff.cpp", src, src, bD01, {q1, nt}, R(0), R(1), nt);
        stage_cmul(bD01, cbuf(0,1), bV1, nt, {nt, q1, ctxp[1].q_inv_32(), q1&0xFFFF, q1>>16});
        stage_ab("compute/garner_diff.cpp", src, src, bA, {q2, nt}, R(0), R(2), nt);
        stage_cmul(bA, cbuf(0,2), bBt, nt, {nt, q2, ctxp[2].q_inv_32(), q2&0xFFFF, q2>>16});
        stage_ab("compute/garner_diff.cpp", bV1, bBt, bCC, {q2, nt}, 0u, 0u, nt);
        stage_cmul(bCC, cbuf(1,2), bV2, nt, {nt, q2, ctxp[2].q_inv_32(), q2&0xFFFF, q2>>16});
        auto bW=mkbuf(2*nt,I32), bP=mkbuf(2*nt,I32), bQ=mkbuf(2*nt,I32), bX=mkbuf(3*nt,I32);
        stage_crt(bV2, 1, 0, bV1, bPar[1], bW, 0u, nt);
        stage_crt(bW, 2, 0, src, bPar[0], bP, R(0), nt);
        stage_crt(bW, 2, 1, bZeroN(nt), bPar[0], bQ, 0u, nt);
        { if (!cached()) { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,4,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,4,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,6,I32);
          auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_pq.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
          auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
          auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/crt_limb3.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
          SetRuntimeArgs(p,rd,core,{(uint32_t)bP->address(),(uint32_t)bQ->address(),nt}); SetRuntimeArgs(p,wr,core,{(uint32_t)bX->address(),3u*nt,0u}); SetRuntimeArgs(p,cp,core,{nt});
          run(p); } }
        return bX; }
    MeshBuf crt_and_gadget_b(MeshBuf src, uint32_t src0, uint32_t nt=2){ return gadget_b(crtK_b(src,src0,nt), nt); }
    void ctadd_b(MeshBuf a, MeshBuf b, MeshBuf o){
        for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<K;++ch){ uint32_t p=comp*K+ch;
            addp_b(CH[ch], a, 2*p, b, 2*p, o, 2*p); } }
    void ctadd_ctb(MeshBuf a, MeshBuf b, MeshBuf o, uint32_t B){       // 2K programs, any B
        for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<K;++ch){
            uint32_t g=ctb_grp(comp,ch,B); addp_b(CH[ch], a, g, b, g, o, g, 2*B); } }
    void ctadd_ctb_into(MeshBuf a, MeshBuf b, uint32_t Ba, MeshBuf o, uint32_t Bo, uint32_t node0){
        for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<K;++ch) for (uint32_t h=0;h<2;++h){
            uint32_t src=(comp*K+ch)*2*Ba + h*Ba, dst=(comp*K+ch)*2*Bo + h*Bo + node0;
            addp_b(CH[ch], a, src, b, src, o, dst, Ba); } }
    void ctadd_lo(MeshBuf ep, MeshBuf acc, uint32_t Bold, MeshBuf out, uint32_t half){
        for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<K;++ch) for (uint32_t h=0;h<2;++h){
            uint32_t e=(comp*K+ch)*2*half + h*half, a=(comp*K+ch)*2*Bold + h*Bold;
            addp_b(CH[ch], ep, e, acc, a, out, e, half); } }
    uint32_t ctb_grp(uint32_t comp, uint32_t ch, uint32_t B) const { return (comp*K+ch)*2*B; }
    uint32_t ctb_tiles(uint32_t B) const { return 4*K*B; }
    void ctsub_pairs(MeshBuf acc, uint32_t Bold, MeshBuf out, uint32_t half){
        for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<K;++ch) for (uint32_t h=0;h<2;++h){
            uint32_t src=(comp*K+ch)*2*Bold + h*Bold, dst=(comp*K+ch)*2*half + h*half;
            subp_b(CH[ch], acc, src+half, acc, src, out, dst, half); } }
    MeshBuf digit_rns_all_b(MeshBuf bdig, uint32_t nt=2){
        const uint32_t nd=nt*ell; auto bo=mkbuf(nd*K,I32);
        auto sh=split_stream(nd); auto cr=shard_set(sh.C);
        if (!cached()) { Program p=CreateProgram(); mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,2*K,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/digit_to_rns.cpp",cr,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false,.defines={{"DR_K",std::to_string(K)}}});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,nd-o);
            std::vector<uint32_t> cargs={n}; for (uint32_t j=0;j<K;++j) cargs.push_back(primes[j]);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)bdig->address(),n,o}); SetRuntimeArgs(p,wr,cc,{(uint32_t)bo->address(),n*K,o*K}); SetRuntimeArgs(p,cp,cc,cargs); }
        run(p); } return bo; }
    MeshBuf digit_rns_b(MeshBuf bdig, uint32_t k){
        auto bo=mkbuf(2*K,I32);
        if (!cached()) { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2*K,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32_stride.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/digit_to_rns.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false,.defines={{"DR_K",std::to_string(K)}}});
        std::vector<uint32_t> cargs={2u}; for (uint32_t j=0;j<K;++j) cargs.push_back(primes[j]);
        SetRuntimeArgs(p,rd,core,{(uint32_t)bdig->address(),2u,ell,k}); SetRuntimeArgs(p,wr,core,{(uint32_t)bo->address(),2u*K,0u}); SetRuntimeArgs(p,cp,core,cargs);
        run(p); }
        return bo; }
    std::vector<std::vector<uint32_t>> download_ct(MeshBuf b){
        auto raw=rdb(b); std::vector<std::vector<uint32_t>> ct(2*K);
        for (uint32_t p=0;p<2*K;++p) ct[p]=unt2(std::vector<uint32_t>(raw.begin()+(size_t)2*p*ELEMS,
                                                                      raw.begin()+(size_t)(2*p+2)*ELEMS));
        return ct; }
    std::vector<std::vector<std::vector<uint32_t>>> download_ctb(MeshBuf b, uint32_t B){
        auto raw=rdb(b); std::vector<std::vector<std::vector<uint32_t>>> cts(B, std::vector<std::vector<uint32_t>>(2*K));
        for (uint32_t nd=0;nd<B;++nd) for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<K;++ch){
            std::vector<uint32_t> t(2*ELEMS);
            for (uint32_t h=0;h<2;++h){ size_t o=(size_t)(ctb_grp(comp,ch,B)+h*B+nd)*ELEMS;
                std::copy(raw.begin()+o, raw.begin()+o+ELEMS, t.begin()+(size_t)h*ELEMS); }
            cts[nd][comp*K+ch]=unt2(t); }
        return cts; }
    void extprod_levelB(MeshBuf bIn, uint32_t B, const std::vector<std::vector<MeshBuf>>& Cn, MeshBuf out){
        const uint32_t nt=2*B, W=2*B;
        std::vector<MeshBuf> bres(2);
        for (uint32_t comp=0;comp<2;++comp)
            bres[comp] = digit_rns_all_b(crt_and_gadget_b(bIn, ctb_grp(comp,0,B), nt), nt);
        std::vector<MeshBuf> dN(K);
        for (uint32_t ch=0;ch<K;++ch){ dN[ch]=mkbuf(4*B*ell,I32);
            for (uint32_t comp=0;comp<2;++comp)
                fwd2048_batchM(CH[ch], bres[comp], /*in_base=*/ch, /*in_pstride=*/K, /*in_tstride=*/B*ell*K,
                               B*ell, dN[ch], /*out_base=*/2*comp, /*out_pstride=*/4); }
        for (uint32_t ch=0;ch<K;++ch) for (uint32_t col=0;col<2;++col){
            auto bProd=mkbuf(4*B*ell,I32);
            mmul_bcast(CH[ch], dN[ch], 0, Cn[ch][col], 0, 4*ell, bProd, 0, 4*B*ell);
            auto bMac=mkbuf(W,I32);
            reduce_groups(CH[ch], bProd, bMac, /*G=*/W, /*n_in=*/2*ell, /*gdiv=*/2, /*gstride=*/4*ell, /*istride=*/2);
            inv2048_batchM(CH[ch], bMac, 0, 2, 1, B, out, ctb_grp(col,ch,B), 1, B); } }
    MeshBuf fold_levelB(MeshBuf acc, uint32_t Bold, const std::vector<std::vector<MeshBuf>>& Cn){
        const uint32_t half=Bold/2;
        auto d=mkctb(half); ctsub_pairs(acc, Bold, d, half);
        auto ep=mkctb(half); extprod_levelB(d, half, Cn, ep);
        auto nacc=mkctb(half); ctadd_lo(ep, acc, Bold, nacc, half);
        return nacc; }
    void fusedA2(Chan& c, MeshBuf xin2, MeshBuf w0, MeshBuf w1, MeshBuf out, uint32_t os) { if (!cached()) { Program p = CreateProgram();
        mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::Int8,4,I8);
        mkcb(p,CBIndex::c_2,tt::DataFormat::Int8,8,I8); mkcb(p,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*xin2).append_to(rc); TensorAccessorArgs(*w0).append_to(rc); TensorAccessorArgs(*w1).append_to(rc);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused2.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a2.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)xin2->address(),(uint32_t)w0->address(),(uint32_t)w1->address()}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,os});
        SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14}); run(p); } }
    void fusedC2(Chan& c, MeshBuf xin2, MeshBuf w0, MeshBuf w1, MeshBuf out, uint32_t os) { if (!cached()) { Program p = CreateProgram();
        mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::Int8,NB*NB*2,I8);
        mkcb(p,CBIndex::c_2,tt::DataFormat::Int8,8,I8); mkcb(p,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*xin2).append_to(rc); TensorAccessorArgs(*w0).append_to(rc); TensorAccessorArgs(*w1).append_to(rc);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused_c2.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c2.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)xin2->address(),(uint32_t)w0->address(),(uint32_t)w1->address()}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,os});
        SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14}); run(p); } }
    void fwd2048_b(Chan& c, MeshBuf in, MeshBuf out, uint32_t in_base=0, uint32_t in_stride=1,
                   uint32_t out_base=0, uint32_t out_stride=1){
        gather_tbl(in, bIn2b, tFwdIn, in_base, in_stride, 0, 1);
        for (uint32_t bk2=0;bk2<2;++bk2){ fusedA2(c,bIn2b,c.WA[bk2*2+0],c.WA[bk2*2+1],bSum[bk2],0);
            mmul(c,bSum[bk2],c.TwF[bk2],bB[bk2],1); fusedC(c,bB[bk2],c.WCt,bOut2r,bk2); }
        gather_tbl(bOut2r, out, tFwdOut, 0, 1, out_base, out_stride); }
    void fwd2048_batch1(Chan& c, MeshBuf in, uint32_t in_base, uint32_t in_pstride, uint32_t in_tstride,
                        uint32_t M, MeshBuf out, uint32_t out_base, uint32_t out_pstride, uint32_t out_tstride=1){
        auto bXb=mkbuf(2*M,I32), bA0=mkbuf(M,I32), bA1=mkbuf(M,I32), bS=mkbuf(M,I32), bTw=mkbuf(M,I32), bCo=mkbuf(2*M,I32);
        gather_mc(in, bXb, tFwdIn, in_base, in_tstride, 0, M, M, in_pstride, 1);    // blk-major: blk*M+m
        for (uint32_t bk2=0;bk2<2;++bk2){
            mbatchA_k2_mc(c, bXb, 0*M, 1*M, WArep[c_idx(c)][bk2*2+0], WArep[c_idx(c)][bk2*2+1],
                          bS, 0, M);                       // both blocks reduced before one recombine
            mmulM_mc(c, bS, c.TwF[bk2], bTw, M);                                    // twiddle (broadcast)
            mbatchC_mc(c, bTw, 0, c.WCt, bCo, bk2*M, M); }
        gather_mc(bCo, out, tFwdOut, 0, M, out_base, out_tstride, M, 1, out_pstride); }
    void fwd2048_batchM(Chan& c, MeshBuf in, uint32_t in_base, uint32_t in_pstride, uint32_t in_tstride,
                        uint32_t M, MeshBuf out, uint32_t out_base, uint32_t out_pstride, uint32_t out_tstride=1){
        const uint32_t cap = MMAX*grid_cores();
        if (M > cap){ for (uint32_t o=0;o<M;o+=cap){ uint32_t n=std::min(cap,M-o);
                fwd2048_batchM(c, in, in_base+o*in_pstride, in_pstride, in_tstride, n,
                               out, out_base+o*out_pstride, out_pstride, out_tstride); } return; }
        fwd2048_batch1(c, in, in_base, in_pstride, in_tstride, M, out, out_base, out_pstride, out_tstride); }
    MeshBuf gadget_b(MeshBuf bX, uint32_t nt=2){
        auto bdig=mkbuf(nt*ell,I32);
        auto sh=split_stream(nt); auto cr=shard_set(sh.C);
        if (!cached()) { Program p=CreateProgram(); mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,2*K,I32); mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,2*ell,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_gadget_il.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.defines={{"GADGET_KLIMBS",std::to_string(K)}}});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/gadget_decompose_signed.cpp",cr,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false,
              .defines={{"GADGET_B",std::to_string(B)},{"GADGET_LOG2B",std::to_string(LOG2B)},{"GADGET_ELL",std::to_string(ell)},{"GADGET_KLIMBS",std::to_string(K)}}});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,nt-o);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)bX->address(),n,o*K});
            SetRuntimeArgs(p,wr,cc,{(uint32_t)bdig->address(),n*ell,o*ell}); SetRuntimeArgs(p,cp,cc,{n}); }
        run(p); }
        return bdig; }
    void gather_mc(MeshBuf in, MeshBuf out, MeshBuf tbl, uint32_t in_base, uint32_t in_stride,
                   uint32_t out_base, uint32_t out_stride, uint32_t npoly, uint32_t in_pstride, uint32_t out_pstride){
        auto sh=split_stream(npoly); auto cr=shard_set(sh.C); if (!cached()) { Program p=CreateProgram();
        mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,6,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/gather_table.cpp",cr,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,npoly-o);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)in->address(),(uint32_t)out->address(),(uint32_t)tbl->address(),2u,
                                    in_base+o*in_pstride,in_stride,out_base+o*out_pstride,out_stride,n,in_pstride,out_pstride}); }
        run(p); } }
    void gather_tbl(MeshBuf in, MeshBuf out, MeshBuf tbl, uint32_t in_base=0, uint32_t in_stride=1,
                    uint32_t out_base=0, uint32_t out_stride=1, uint32_t npoly=1,
                    uint32_t in_pstride=0, uint32_t out_pstride=0){
        if (!cached()) { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,6,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/gather_table.cpp",core,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        SetRuntimeArgs(p,rd,core,{(uint32_t)in->address(),(uint32_t)out->address(),(uint32_t)tbl->address(),
                                  2u,in_base,in_stride,out_base,out_stride,npoly,in_pstride,out_pstride});
        run(p); } }
    uint32_t grid_cores() const { return GX*GY; }
    void inv2048_b(Chan& c, MeshBuf in, MeshBuf out, uint32_t in_base=0, uint32_t in_stride=1,
                   uint32_t out_base=0, uint32_t out_stride=1){
        gather_tbl(in, bIn2b, tInvIn, in_base, in_stride, 0, 1);
        for (uint32_t bk2=0;bk2<2;++bk2){ fusedA_off(c,bIn2b,bk2,c.Winv32,bSum[bk2],0);
            mmul(c,bSum[bk2],c.TwI[bk2],bIn2,1,bk2); }
        for (uint32_t bn2=0;bn2<2;++bn2) fusedC2(c,bIn2,c.Winv64[0*2+bn2],c.Winv64[1*2+bn2],bOut2r,bn2);
        gather_tbl(bOut2r, out, tInvOut, 0, 1, out_base, out_stride); }
    void inv2048_batchM(Chan& c, MeshBuf in, uint32_t in_base, uint32_t in_pstride, uint32_t in_tstride,
                        uint32_t M, MeshBuf out, uint32_t out_base, uint32_t out_pstride, uint32_t out_tstride=1){
        const uint32_t cap = MMAX*grid_cores();
        if (M > cap){ for (uint32_t o=0;o<M;o+=cap){ uint32_t n=std::min(cap,M-o);
                inv2048_batchM(c, in, in_base+o*in_pstride, in_pstride, in_tstride, n,
                               out, out_base+o*out_pstride, out_pstride, out_tstride); } return; }
        const uint32_t ci=c_idx(c);
        auto bXb=mkbuf(2*M,I32), bA=mkbuf(M,I32), bTw=mkbuf(2*M,I32),
             bC0=mkbuf(M,I32), bC1=mkbuf(M,I32), bCo=mkbuf(2*M,I32);
        gather_mc(in, bXb, tInvIn, in_base, in_tstride, 0, M, M, in_pstride, 1);   // blk-major bk2*M+m
        for (uint32_t bk2=0;bk2<2;++bk2){                                          // stage 1 + twiddle
            mbatchA_mc(c, bXb, bk2*M, Winv32rep[ci], bA, 0, M);
            mmulM_mc_off(c, bA, 0, c.TwI[bk2], bTw, bk2*M, M); }
        for (uint32_t bn2=0;bn2<2;++bn2){                                          // stage 2, K-accum
            mbatchC_k2_mc(c, bTw, 0*M, 1*M, c.Winv64[0*2+bn2], c.Winv64[1*2+bn2],
                          bCo, bn2*M, M); }        // both blocks reduced before one recombine
        gather_mc(bCo, out, tInvOut, 0, M, out_base, out_tstride, M, 1, out_pstride); }
    static uint32_t inv_mod_2n(int64_t j){ uint32_t a=(uint32_t)(((j%(2*(int64_t)N))+2*(int64_t)N)%(2*(int64_t)N)), x=a;
        for (int i=0;i<5;++i) x*= 2u-a*x; return x & (2*N-1); }
    void keyswitch_b(MeshBuf bCt, const std::vector<std::vector<MeshBuf>>& KnB, MeshBuf out){
        auto bdig = crt_and_gadget_b(bCt, 2*K);                     // component 1 starts at poly K
        auto bres = digit_rns_all_b(bdig);
        std::vector<MeshBuf> dN(K);
        for (uint32_t ch=0;ch<K;++ch){ dN[ch]=mkbuf(2*ell,I32);     // poly k at tiles 2k, 2k+1
            fwd2048_batchM(CH[ch], bres, /*in_base=*/ch, /*in_pstride=*/K, /*in_tstride=*/ell*K,
                           ell, dN[ch], /*out_base=*/0, /*out_pstride=*/2); }
        for (uint32_t ch=0;ch<K;++ch){
            auto bmac=mkbuf(4,I32);                                     // col 0 at tiles 0-1, col 1 at 2-3
            for (uint32_t col=0;col<2;++col) mac_all_into(CH[ch], dN[ch], KnB[ch][col], ell, bmac, 2*col);
            auto btmp=mkbuf(4,I32);                                     // both columns in ONE batched inverse
            inv2048_batchM(CH[ch], bmac, 0, 2, 1, 2, btmp, 0, 2);
            addp_b(CH[ch], bCt, 2*ch, btmp, 0, out, 2*ch);              // col 0: + c0
            addp_b(CH[ch], btmp, 2, bZero2, 0, out, 2*(K+ch)); } }
    void keyswitch_levelB(MeshBuf bIn, uint32_t B, const std::vector<std::vector<MeshBuf>>& KnB, MeshBuf out){
        const uint32_t nt=2*B, W=2*B;                       // nt = tiles per residue; W = tiles per digit block
        auto bdig = crt_and_gadget_b(bIn, ctb_grp(1,0,B), nt);
        auto bres = digit_rns_all_b(bdig, nt);
        // ONE forward batch of B*ell polys per channel.  dN comes out in [node][digit] order
        // (poly m=b*ell+k at tiles 2m,2m+1) — which the grouped reduce below consumes directly, so
        // the ell-separate-batches workaround is no longer needed: 12 programs per channel, not ell*12.
        std::vector<MeshBuf> dN(K);
        for (uint32_t ch=0;ch<K;++ch){ dN[ch]=mkbuf(2*B*ell,I32);
            fwd2048_batchM(CH[ch], bres, /*in_base=*/ch, /*in_pstride=*/K, /*in_tstride=*/B*ell*K,
                           B*ell, dN[ch], /*out_base=*/0, /*out_pstride=*/2); }
        for (uint32_t ch=0;ch<K;++ch) for (uint32_t col=0;col<2;++col){
            // every product in ONE multiply: dN tile i = b*2ell + 2k + t needs KSK tile 2k+t = i%(2ell)
            auto bProd=mkbuf(2*B*ell,I32);
            mmul_bcast(CH[ch], dN[ch], 0, KnB[ch][col], 0, 2*ell, bProd, 0, 2*B*ell);
            // reduce the digit axis: one group per (node, half), g = 2b+t -> output tile 2b+t
            auto bMac=mkbuf(W,I32);
            reduce_groups(CH[ch], bProd, bMac, /*G=*/W, /*n_in=*/ell, /*gdiv=*/2, /*gstride=*/2*ell, /*istride=*/2);
            const uint32_t g=ctb_grp(col,ch,B);
            if (col==0){ auto btmp=mkbuf(W,I32);             // inverse into ct layout, then + c0
                inv2048_batchM(CH[ch], bMac, 0, 2, 1, B, btmp, 0, 1, B);
                addp_b(CH[ch], bIn, g, btmp, 0, out, g, W); }
            else       { inv2048_batchM(CH[ch], bMac, 0, 2, 1, B, out, g, 1, B); } } }
    MeshBuf mac_all_b(Chan& c, MeshBuf a, MeshBuf b, uint32_t nk){
        auto o=mkbuf(2,I32); mac_all_into(c,a,b,nk,o,0); return o; }
    void mac_all_into(Chan& c, MeshBuf a, MeshBuf b, uint32_t nk, MeshBuf bSumO, uint32_t o_base){
        auto bProd=mkbuf(2*nk,I32);
        mmul(c, a, b, bProd, 2*nk, 0, 0);
        for (uint32_t t=0;t<2;++t){
            if (!cached()) { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,nk,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32_stride.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)bProd->address(), nk, 2u, t}); SetRuntimeArgs(p,wr,core,{(uint32_t)bSumO->address(),1u,o_base+t}); SetRuntimeArgs(p,cp,core,{nk,c.q});
            run(p); } }
    }
    void maddM(Chan& c, MeshBuf a, uint32_t at, MeshBuf b, uint32_t bt, MeshBuf o, uint32_t ot, uint32_t M){
        if (!cached()) { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_map.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)a->address(),(uint32_t)b->address(),M,at,bt}); SetRuntimeArgs(p,wr,core,{(uint32_t)o->address(),M,ot}); SetRuntimeArgs(p,cp,core,{M,c.q});
        run(p); } }
    void maddM_mc(Chan& c, MeshBuf a, uint32_t at, MeshBuf b, uint32_t bt, MeshBuf o, uint32_t ot, uint32_t M){
        auto sh=split_stream(M); auto cr=shard_set(sh.C); if (!cached()) { Program p=CreateProgram();
        mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_map.cpp",cr,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t of=i*sh.per, n=std::min(sh.per,M-of);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)a->address(),(uint32_t)b->address(),n,at+of,bt+of});
            SetRuntimeArgs(p,wr,cc,{(uint32_t)o->address(),n,ot+of}); SetRuntimeArgs(p,cp,cc,{n,c.q}); }
        run(p); } }
    void mbatchA(Chan& c, MeshBuf in, uint32_t x_base, MeshBuf w, MeshBuf out, uint32_t o_base, uint32_t M){
        if (!cached()) { Program p=CreateProgram();
        mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,M,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::Int8,32,I8);
        mkcb(p,CBIndex::c_2,tt::DataFormat::Int8,M*NB,I8); mkcb(p,CBIndex::c_3,tt::DataFormat::Int32,M*NB,I32);
        mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,M,I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*in).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/reader_mbatch.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "compute/matmul_modq_mbatch.cpp",core,ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)in->address(),(uint32_t)w->address(),x_base,0u,M});
        SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),M,o_base});
        SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14,M}); run(p); } }
    void mbatchA_k2_mc(Chan& c, MeshBuf in, uint32_t xb0, uint32_t xb1, MeshBuf w0, MeshBuf w1,
                       MeshBuf out, uint32_t o_base, uint32_t M){
        auto sh=split(M); auto cr=shard_set(sh.C);
        if (!cached()) { Program p=CreateProgram();
        mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,2*sh.per,I32); mkcb_cr(p,cr,CBIndex::c_1,tt::DataFormat::Int8,32,I8);
        mkcb_cr(p,cr,CBIndex::c_2,tt::DataFormat::Int8,2*sh.per*NB,I8); mkcb_cr(p,cr,CBIndex::c_3,tt::DataFormat::Int32,sh.per*NB,I32);
        mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,sh.per,I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*in).append_to(rc); TensorAccessorArgs(*w0).append_to(rc); TensorAccessorArgs(*w1).append_to(rc);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/reader_mbatch_k2.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/writer_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "compute/matmul_modq_mbatch_k2.cpp",cr,ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,M-o);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)in->address(),(uint32_t)w0->address(),(uint32_t)w1->address(),xb0+o,xb1+o,n});
            SetRuntimeArgs(p,wr,cc,{(uint32_t)out->address(),n,o_base+o});
            SetRuntimeArgs(p,cp,cc,{c.q,c.qinv,c.qlo,c.qhi,c.k14,n}); }
        run(p); } }
    void mbatchA_mc(Chan& c, MeshBuf in, uint32_t x_base, MeshBuf w, MeshBuf out, uint32_t o_base, uint32_t M){
        auto sh=split(M); auto cr=shard_set(sh.C); if (!cached()) { Program p=CreateProgram();
        mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,sh.per,I32); mkcb_cr(p,cr,CBIndex::c_1,tt::DataFormat::Int8,32,I8);
        mkcb_cr(p,cr,CBIndex::c_2,tt::DataFormat::Int8,sh.per*NB,I8); mkcb_cr(p,cr,CBIndex::c_3,tt::DataFormat::Int32,sh.per*NB,I32);
        mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,sh.per,I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*in).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/reader_mbatch.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/writer_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "compute/matmul_modq_mbatch.cpp",cr,ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,M-o);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)in->address(),(uint32_t)w->address(),x_base+o,0u,n});
            SetRuntimeArgs(p,wr,cc,{(uint32_t)out->address(),n,o_base+o});
            SetRuntimeArgs(p,cp,cc,{c.q,c.qinv,c.qlo,c.qhi,c.k14,n}); }
        run(p); } }
    void mbatchC(Chan& c, MeshBuf in, uint32_t x_base, MeshBuf w, MeshBuf out, uint32_t o_base, uint32_t M){
        if (!cached()) { Program p=CreateProgram();
        mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,M,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::Int8,NB*NB,I8);
        mkcb(p,CBIndex::c_2,tt::DataFormat::Int8,M*NB,I8); mkcb(p,CBIndex::c_3,tt::DataFormat::Int32,M*NB,I32);
        mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,M,I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*in).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/reader_mbatch_c.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "compute/matmul_modq_mbatch_c.cpp",core,ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)in->address(),(uint32_t)w->address(),x_base,0u,M});
        SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),M,o_base});
        SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14,M}); run(p); } }
    void mbatchC_k2_mc(Chan& c, MeshBuf in, uint32_t xb0, uint32_t xb1, MeshBuf w0, MeshBuf w1,
                       MeshBuf out, uint32_t o_base, uint32_t M){
        auto sh=split(M); auto cr=shard_set(sh.C);
        if (!cached()) { Program p=CreateProgram();
        mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,2*sh.per,I32); mkcb_cr(p,cr,CBIndex::c_1,tt::DataFormat::Int8,2*NB*NB,I8);
        mkcb_cr(p,cr,CBIndex::c_2,tt::DataFormat::Int8,2*sh.per*NB,I8); mkcb_cr(p,cr,CBIndex::c_3,tt::DataFormat::Int32,sh.per*NB,I32);
        mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,sh.per,I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*in).append_to(rc); TensorAccessorArgs(*w0).append_to(rc); TensorAccessorArgs(*w1).append_to(rc);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/reader_mbatch_c_k2.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/writer_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "compute/matmul_modq_mbatch_c_k2.cpp",cr,ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,M-o);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)in->address(),(uint32_t)w0->address(),(uint32_t)w1->address(),xb0+o,xb1+o,n});
            SetRuntimeArgs(p,wr,cc,{(uint32_t)out->address(),n,o_base+o});
            SetRuntimeArgs(p,cp,cc,{c.q,c.qinv,c.qlo,c.qhi,c.k14,n}); }
        run(p); } }
    void mbatchC_mc(Chan& c, MeshBuf in, uint32_t x_base, MeshBuf w, MeshBuf out, uint32_t o_base, uint32_t M){
        auto sh=split(M); auto cr=shard_set(sh.C); if (!cached()) { Program p=CreateProgram();
        mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,sh.per,I32); mkcb_cr(p,cr,CBIndex::c_1,tt::DataFormat::Int8,NB*NB,I8);
        mkcb_cr(p,cr,CBIndex::c_2,tt::DataFormat::Int8,sh.per*NB,I8); mkcb_cr(p,cr,CBIndex::c_3,tt::DataFormat::Int32,sh.per*NB,I32);
        mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,sh.per,I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*in).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/reader_mbatch_c.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/writer_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "compute/matmul_modq_mbatch_c.cpp",cr,ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,M-o);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)in->address(),(uint32_t)w->address(),x_base+o,0u,n});
            SetRuntimeArgs(p,wr,cc,{(uint32_t)out->address(),n,o_base+o});
            SetRuntimeArgs(p,cp,cc,{c.q,c.qinv,c.qlo,c.qhi,c.k14,n}); }
        run(p); } }
    void mkcb_cr(Program& p, const CoreRangeSet& cr, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t pg){
        CircularBufferConfig c(depth*pg, {{idx,fmt}}); c.set_page_size(idx,pg); CreateCircularBuffer(p,cr,c); }
    MeshBuf mkct(){ return mkbuf(CT_TILES(K), I32); }
    MeshBuf mkctb(uint32_t B){ return mkbuf(ctb_tiles(B), I32); }
    MeshBuf mktable(const std::vector<std::pair<uint32_t,uint32_t>>& ds){   // (dest_lin, src_lin) pairs
        std::vector<uint32_t> t(2*ELEMS, 0);
        for (auto& p : ds) t[physoff(p.first)] = physoff(p.second);
        auto b = mkbuf(2, I32); wrb(b, t); return b; }
    void mmulM_mc(Chan& c, MeshBuf a, MeshBuf b, MeshBuf out, uint32_t M){ mmulM_mc_off(c,a,0,b,out,0,M); }
    void mmulM_mc_off(Chan& c, MeshBuf a, uint32_t a_base, MeshBuf b, MeshBuf out, uint32_t o_base, uint32_t M){
        auto sh=split_stream(M); auto cr=shard_set(sh.C); if (!cached()) { Program p=CreateProgram();
        mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul_bcast.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_1pass.cpp",cr,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,M-o);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)a->address(),(uint32_t)b->address(),n,a_base+o,0u,1u});  // b = one tile
            SetRuntimeArgs(p,wr,cc,{(uint32_t)out->address(),n,o_base+o}); SetRuntimeArgs(p,cp,cc,{n,c.q,c.qinv,c.qlo,c.qhi}); }
        run(p); } }
    void mmul_bcast(Chan& c, MeshBuf a, uint32_t a_start, MeshBuf b, uint32_t b_base, uint32_t cycle,
                    MeshBuf out, uint32_t out_start, uint32_t nt){
        auto sh=split_stream(nt); auto cr=shard_set(sh.C); if (!cached()) { Program p=CreateProgram();
        mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul_bcast.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_1pass.cpp",cr,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,nt-o);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)a->address(),(uint32_t)b->address(),n,a_start+o,b_base,cycle});
            SetRuntimeArgs(p,wr,cc,{(uint32_t)out->address(),n,out_start+o});
            SetRuntimeArgs(p,cp,cc,{n,c.q,c.qinv,c.qlo,c.qhi}); }
        run(p); } }
    std::vector<std::vector<uint32_t>> monomul_ct(const std::vector<std::vector<uint32_t>>& ct, int64_t k){
        return permute_polys(ct, 1u, (uint32_t)((((-k)%(2*(int64_t)N))+2*(int64_t)N)%(2*(int64_t)N))); }
    void monomul_ct_b(MeshBuf in, MeshBuf out, int64_t k){
        permute_ct_b(in,out,1u,(uint32_t)((((-k)%(2*(int64_t)N))+2*(int64_t)N)%(2*(int64_t)N))); }
    void monomul_ctb(MeshBuf in, MeshBuf out, int64_t k, uint32_t B){
        permute_ctb(in,out,1u,(uint32_t)((((-k)%(2*(int64_t)N))+2*(int64_t)N)%(2*(int64_t)N)),B); }
    void monomul_ctb_into(MeshBuf in, uint32_t Bin, MeshBuf out, uint32_t Bout, uint32_t node0, int64_t k){
        permute_ctb_into(in, Bin, out, Bout, node0, 1u,
                         (uint32_t)((((-k)%(2*(int64_t)N))+2*(int64_t)N)%(2*(int64_t)N))); }
    void permute_ct_b(MeshBuf in, MeshBuf out, uint32_t A, uint32_t Bc){
        if (!cached()) { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,6,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/permute_poly.cpp",core,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        std::vector<uint32_t> ra={(uint32_t)in->address(),(uint32_t)out->address(),2u,A,Bc,2*K,K,1u,
                                  0u,2u,1u, 0u,2u,1u, 0u};
        for (uint32_t j=0;j<K;++j) ra.push_back(primes[j]);
        SetRuntimeArgs(p,rd,core,ra);
        run(p); } }
    void permute_ctb(MeshBuf in, MeshBuf out, uint32_t A, uint32_t Bc, uint32_t B){
        for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<K;++ch){
            uint32_t g=ctb_grp(comp,ch,B);
            auto sh=split_stream(B); auto cr=shard_set(sh.C); if (!cached()) { Program p=CreateProgram();
            mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,6,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/permute_poly.cpp",cr,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,B-o);
                SetRuntimeArgs(p,rd,cc,{(uint32_t)in->address(),(uint32_t)out->address(),2u,A,Bc,n,1u,1u,
                                        g+o,1u,B, g+o,1u,B, 0u, primes[ch]}); }
            run(p); } } }
    void permute_ctb_into(MeshBuf in, uint32_t Bin, MeshBuf out, uint32_t Bout, uint32_t node0,
                          uint32_t A, uint32_t Bc){
        for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<K;++ch){
            uint32_t gi=(comp*K+ch)*2*Bin, go=(comp*K+ch)*2*Bout + node0;
            auto sh=split_stream(Bin); auto cr=shard_set(sh.C);
            if (!cached()) { Program p=CreateProgram();
            mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,6,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/permute_poly.cpp",cr,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,Bin-o);
                SetRuntimeArgs(p,rd,cc,{(uint32_t)in->address(),(uint32_t)out->address(),2u,A,Bc,n,1u,1u,
                                        gi+o,1u,Bin, go+o,1u,Bout, 0u, primes[ch]}); }
            run(p); } } }
    std::vector<std::vector<uint32_t>> permute_polys(const std::vector<std::vector<uint32_t>>& polys,
                                                     uint32_t A, uint32_t Bc){
        const uint32_t NP=(uint32_t)polys.size(), NT=2;                  // n=2048 -> 2 tiles/poly
        auto bi=mkbuf(NP*NT,I32), bo=mkbuf(NP*NT,I32);
        std::vector<uint32_t> buf(NP*NT*ELEMS);
        for (uint32_t i=0;i<NP;++i){ auto t=til2(polys[i]); std::copy(t.begin(),t.end(),buf.begin()+(size_t)i*NT*ELEMS); }
        wrb(bi,buf);
        auto grid=mesh->compute_with_storage_grid_size();
        const uint32_t GX=grid.x, GY=grid.y, C=GX*GY, per=(NP+C-1)/C;
        if (!cached()) { Program p=CreateProgram(); CoreRange cores(CoreCoord{0,0},CoreCoord{GX-1,GY-1});
        CircularBufferConfig c(3*NT*I32,{{CBIndex::c_0,tt::DataFormat::UInt32}}); c.set_page_size(CBIndex::c_0,I32);
        CreateCircularBuffer(p,cores,c);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/permute_poly.cpp",cores,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        for (uint32_t gy=0;gy<GY;++gy) for (uint32_t gx=0;gx<GX;++gx){
            uint32_t idx=gy*GX+gx, ps=idx*per, np=(ps>=NP)?0u:std::min(per,NP-ps);
            std::vector<uint32_t> ra={(uint32_t)bi->address(),(uint32_t)bo->address(),NT,A,Bc,np,K,1u,
                                      ps*NT,NT,1u, ps*NT,NT,1u, ps};  // per-ct layout: poly p at p*NT
            for (uint32_t j=0;j<K;++j) ra.push_back(primes[j]);
            SetRuntimeArgs(p,rd,CoreCoord{gx,gy},ra); }
        run(p); }
        auto raw=rdb(bo); std::vector<std::vector<uint32_t>> out(NP);
        for (uint32_t i=0;i<NP;++i) out[i]=unt2(std::vector<uint32_t>(raw.begin()+(size_t)i*NT*ELEMS,
                                                                      raw.begin()+(size_t)(i+1)*NT*ELEMS));
        return out; }
    static uint32_t physoff(uint32_t lin){ const uint32_t t=lin>>10, i=lin&1023, r=i>>5, c=i&31;
        return t*ELEMS + ((r>>4)*2+(c>>4))*256 + (r&15)*16 + (c&15); }
    void prewarm_batch(uint32_t maxB){ for (uint32_t b=1;b<=maxB;b*=2){ bZeroN(2*b); bZeroN(4*b); } }
    std::vector<int8_t> rdb_i8(MeshBuf b, uint32_t nt){ std::vector<int8_t> v;
        distributed::EnqueueReadMeshBuffer(*cq, v, b, true); v.resize((size_t)nt*ELEMS); return v; }
    void reduce_groups(Chan& c, MeshBuf in, MeshBuf out, uint32_t G, uint32_t n_in,
                       uint32_t gdiv, uint32_t gstride, uint32_t istride){
        auto sh=split_stream(G); auto cr=shard_set(sh.C);
        if (!cached()) { Program p=CreateProgram(); mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,2*n_in,I32); mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32_groups_2l.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce_groups.cpp",cr,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t o=i*sh.per, n=std::min(sh.per,G-o);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)in->address(),n,n_in,gdiv,gstride,istride,o});
            SetRuntimeArgs(p,wr,cc,{(uint32_t)out->address(),n,o}); SetRuntimeArgs(p,cp,cc,{n,n_in,c.q}); }
        run(p); } }
    std::vector<std::vector<MeshBuf>> rgsw_operand(const std::vector<std::vector<std::vector<uint32_t>>>& Cn){
        std::vector<std::vector<MeshBuf>> r(K, std::vector<MeshBuf>(2));
        for (uint32_t ch=0;ch<K;++ch) for (uint32_t col=0;col<2;++col) r[ch][col]=mkbuf(4*ell,I32);
        rgsw_operand_into(Cn, r); return r; }
    void rgsw_operand_into(const std::vector<std::vector<std::vector<uint32_t>>>& Cn,
                           std::vector<std::vector<MeshBuf>>& dst){        // same, into fixed buffers
        for (uint32_t ch=0;ch<K;++ch) for (uint32_t col=0;col<2;++col){
            std::vector<uint32_t> buf((size_t)4*ell*ELEMS);
            for (uint32_t k=0;k<ell;++k) for (uint32_t comp=0;comp<2;++comp){
                auto t=til2(Cn[ch][(comp*ell+k)*2+col]);
                std::copy(t.begin(),t.end(),buf.begin()+(size_t)(4*k+2*comp)*ELEMS); }
            wrb(dst[ch][col],buf); } }
    void scan_add_pass(Chan& c, MeshBuf a, MeshBuf sPar, MeshBuf b, uint32_t sOT){
        const uint32_t per = grid_cores()*SMB;
        for (uint32_t base=0; base<sOT; base+=per){ auto cr=shard_set(grid_cores());
            if (!cached()) { Program p=CreateProgram();
            // The scan's accumulate: ~4.6 GB/query. Depth on BOTH sides (reader_binop_deep /
            // writer_i32_deep need c_0,c_1,c_16 at least CHUNK=8 deep) plus the one-pass add.
            mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,16,I32); mkcb_cr(p,cr,CBIndex::c_1,tt::DataFormat::UInt32,16,I32); mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,16,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/reader_binop_deep.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/writer_i32_deep.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "compute/add_modq_1pass.cpp",cr,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            for (uint32_t i=0;i<grid_cores();++i){ uint32_t o=base+i*SMB; if (o>=sOT) continue;
                uint32_t mh=std::min(SMB,sOT-o); auto cc=shard_at(i);
                SetRuntimeArgs(p,rd,cc,{(uint32_t)a->address(),(uint32_t)sPar->address(),mh,o});
                SetRuntimeArgs(p,wr,cc,{(uint32_t)b->address(),mh,o}); SetRuntimeArgs(p,cp,cc,{mh,c.q}); }
            run(p); } } }
    void scan_gather(MeshBuf in, MeshBuf out, uint32_t ROWS, uint32_t COLS, uint32_t SR, uint32_t SC,
                     uint32_t out_base, uint32_t h, uint32_t in_base=0){
        const uint32_t per=(ELEMS+grid_cores()-1)/grid_cores(); auto cr=shard_set(grid_cores());
        if (!cached()) { Program p=CreateProgram(); mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,per+1,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/reshuffle_gather_mc.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        for (uint32_t i=0;i<grid_cores();++i){ uint32_t ss=i*per, no=(ss>=ELEMS)?0:std::min(per,ELEMS-ss);
            SetRuntimeArgs(p,rd,shard_at(i),{(uint32_t)in->address(),(uint32_t)out->address(),ROWS,COLS,SR,SC,ss,no,out_base+h*ELEMS,2u,in_base+h}); }
        run(p); } }
    void scan_levelB(const std::vector<MeshBuf>& bDB, const std::vector<MeshBuf>& bS,
                     uint32_t N0, uint32_t rest, MeshBuf out){
        for (uint32_t ch=0;ch<K;++ch){
            auto bProd=mkbuf(N0*2*rest*2, I32);
            for (uint32_t k=0;k<N0;++k) for (uint32_t comp=0;comp<2;++comp)
                mmul_bcast(CH[ch], bDB[ch], 2*(k*rest), bS[ch], 2*(k*2+comp), 2,
                           bProd, (k*2+comp)*rest*2, 2*rest);
            auto bAcc=mkbuf(4*rest, I32);
            reduce_groups(CH[ch], bProd, bAcc, /*G=*/4*rest, /*n_in=*/N0,
                          /*gdiv=*/4*rest, /*gstride=*/0, /*istride=*/4*rest);
            for (uint32_t comp=0;comp<2;++comp)
                inv2048_batchM(CH[ch], bAcc, comp*2*rest, 2, 1, rest, out, ctb_grp(comp,ch,rest), 1, rest); } }
    void scan_mbatch_pass(Chan& c, MeshBuf sX, MeshBuf sW, MeshBuf sPar, uint32_t kt, uint32_t sOT){
        const uint32_t per = grid_cores()*SMB;
        for (uint32_t base=0; base<sOT; base+=per){ auto cr=shard_set(grid_cores());
            if (!cached()) { Program p=CreateProgram();
            mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,SMB,I32); mkcb_cr(p,cr,CBIndex::c_1,tt::DataFormat::Int8,32,I8);
            mkcb_cr(p,cr,CBIndex::c_2,tt::DataFormat::Int8,SMB*NB,I8); mkcb_cr(p,cr,CBIndex::c_3,tt::DataFormat::Int32,SMB*NB,I32);
            mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,SMB,I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*sX).append_to(rc); TensorAccessorArgs(*sW).append_to(rc);
            // reader_mbatch_deep: the DRAM read-depth rule (docs/he/dram_guide.md). This feed moves the
            // WHOLE folded database on every query -- 6.0 GB -- so its depth sets the scan's bandwidth.
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/reader_mbatch_deep.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/writer_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "compute/matmul_modq_mbatch_1pass.cpp",cr,ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            for (uint32_t i=0;i<grid_cores();++i){ uint32_t o=base+i*SMB; if (o>=sOT) continue;
                uint32_t mh=std::min(SMB,sOT-o), xb=kt*sOT+o; auto cc=shard_at(i);
                SetRuntimeArgs(p,rd,cc,{(uint32_t)sX->address(),(uint32_t)sW->address(),xb,xb*NB*NB,mh});
                SetRuntimeArgs(p,wr,cc,{(uint32_t)sPar->address(),mh,o});
                SetRuntimeArgs(p,cp,cc,{c.q,c.qinv,c.qlo,c.qhi,c.k14,mh}); }
            run(p); } } }
    void scan_mtile(Chan& c, MeshBuf sX, MeshBuf sW, MeshBuf sPar, MeshBuf sAcc0, MeshBuf sAcc1,
                    MeshBuf out, uint32_t KB, uint32_t sOT, uint32_t rest, uint32_t cols=2){
        MeshBuf a=sAcc0, b=sAcc1;
        for (uint32_t kt=0; kt<KB; ++kt){ scan_mbatch_pass(c, sX, sW, sPar, kt, sOT);
                                          scan_add_pass(c, a, sPar, b, sOT); std::swap(a,b); }
        MeshBuf fin = (KB % 2 == 0) ? sAcc0 : sAcc1;
        for (uint32_t h=0; h<2; ++h) scan_scatter(fin, out, cols, rest*cols, h); }
    void scan_scatter(MeshBuf in, MeshBuf out, uint32_t COLS, uint32_t n_poly, uint32_t h){
        const uint32_t per=(n_poly+grid_cores()-1)/grid_cores(); auto cr=shard_set(grid_cores());
        if (!cached()) { Program p=CreateProgram(); mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,per+1,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX_IR "dataflow/reshuffle_scatter_mc.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        for (uint32_t i=0;i<grid_cores();++i){ uint32_t os=i*per, no=(os>=n_poly)?0:std::min(per,n_poly-os);
            SetRuntimeArgs(p,rd,shard_at(i),{(uint32_t)in->address(),(uint32_t)out->address(),ELEMS,h*ELEMS,COLS,os,no,2u,h}); }
        run(p); } }
    CoreCoord shard_at(uint32_t i) const { return CoreCoord{i%GX, i/GX}; }
    CoreRangeSet shard_set(uint32_t C) const {                    // first C cores, row-major
        std::vector<CoreRange> v; uint32_t full=C/GX, rem=C%GX;
        if (full) v.push_back(CoreRange(CoreCoord{0,0}, CoreCoord{GX-1, full-1}));
        if (rem)  v.push_back(CoreRange(CoreCoord{0,full}, CoreCoord{rem-1, full}));
        return CoreRangeSet(v); }
    Shard split(uint32_t M) const {
        uint32_t per=(M+grid_cores()-1)/grid_cores(); if (per>MMAX) per=MMAX;
        if (per==0) per=1; return {per, (M+per-1)/per}; }
    Shard split_stream(uint32_t M) const {
        uint32_t per=(M+grid_cores()-1)/grid_cores(); if (per==0) per=1;
        uint32_t C=(M+per-1)/per; if (C>grid_cores()) C=grid_cores(); return {per, C}; }
    void stage_cmul(MeshBuf a, MeshBuf bc, MeshBuf o, uint32_t nt, const std::vector<uint32_t>& cargs){
        auto sh=split_stream(nt); auto cr=shard_set(sh.C);
        if (!cached()) { Program p=CreateProgram(); mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul_bcast.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_1pass.cpp",cr,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t oo=i*sh.per, n=std::min(sh.per,nt-oo);
            auto ca=cargs; ca[0]=n;                       // modmul_compute takes {n_tiles, q, ...}
            SetRuntimeArgs(p,rd,cc,{(uint32_t)a->address(),(uint32_t)bc->address(),n,oo,0u,1u});  // b = one tile
            SetRuntimeArgs(p,wr,cc,{(uint32_t)o->address(),n,oo}); SetRuntimeArgs(p,cp,cc,ca); }
        run(p); } }
    void subp_b(Chan& c, MeshBuf a, uint32_t at, MeshBuf b, uint32_t bt, MeshBuf o, uint32_t ot, uint32_t nt=2){
        auto sh=split_stream(nt); auto cr=shard_set(sh.C);
        if (!cached()) { Program p=CreateProgram(); mkcb_cr(p,cr,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb_cr(p,cr,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",cr,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modsub_map.cpp",cr,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<sh.C;++i){ auto cc=shard_at(i); uint32_t oo=i*sh.per, n=std::min(sh.per,nt-oo);
            SetRuntimeArgs(p,rd,cc,{(uint32_t)a->address(),(uint32_t)b->address(),n,at+oo,bt+oo});
            SetRuntimeArgs(p,wr,cc,{(uint32_t)o->address(),n,ot+oo}); SetRuntimeArgs(p,cp,cc,{n,c.q}); }
        run(p); } }
    void subs_b(MeshBuf bCt, int64_t j, const std::vector<std::vector<MeshBuf>>& Kn, MeshBuf out, MeshBuf scratch){
        automorph_ct_b(bCt, scratch, j); keyswitch_b(scratch, Kn, out); }
    void subs_levelB(MeshBuf bIn, uint32_t B, int64_t j, const std::vector<std::vector<MeshBuf>>& KnB,
                     MeshBuf out, MeshBuf scratch){
        automorph_ctb(bIn, scratch, j, B); keyswitch_levelB(scratch, B, KnB, out); }
    MeshBuf upload_ct(const std::vector<std::vector<uint32_t>>& ct){
        auto b=mkct(); std::vector<uint32_t> buf(CT_TILES(K)*ELEMS);
        for (uint32_t p=0;p<2*K;++p){ auto t=til2(ct[p]); std::copy(t.begin(),t.end(),buf.begin()+(size_t)2*p*ELEMS); }
        wrb(b,buf); return b; }
    MeshBuf upload_ctb(const std::vector<std::vector<std::vector<uint32_t>>>& cts){
        auto b=mkctb((uint32_t)cts.size()); upload_ctb_into(b, cts); return b; }
    void upload_ctb_into(MeshBuf b, const std::vector<std::vector<std::vector<uint32_t>>>& cts){
        const uint32_t B=(uint32_t)cts.size();
        std::vector<uint32_t> buf((size_t)ctb_tiles(B)*ELEMS);
        for (uint32_t nd=0;nd<B;++nd) for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<K;++ch){
            auto t=til2(cts[nd][comp*K+ch]);
            for (uint32_t h=0;h<2;++h) std::copy(t.begin()+(size_t)h*ELEMS, t.begin()+(size_t)(h+1)*ELEMS,
                buf.begin()+(size_t)(ctb_grp(comp,ch,B)+h*B+nd)*ELEMS); }
        wrb(b,buf); }
    void wr_in2(const std::vector<uint32_t>& blk0, const std::vector<uint32_t>& blk1){
        std::vector<uint32_t> buf(2*ELEMS); auto t0=tilize32(blk0), t1=tilize32(blk1);
        std::copy(t0.begin(),t0.end(),buf.begin()); std::copy(t1.begin(),t1.end(),buf.begin()+ELEMS); wrb(bIn2,buf); }
};
}  // namespace pir2048
