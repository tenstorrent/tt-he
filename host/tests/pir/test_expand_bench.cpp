// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Traced expand-tree BENCHMARK at arbitrary depth — measures what was previously projected.
//
// Correctness of the tree is established elsewhere against real goldens (test_query_expand2048:
// m_exp=2 → 0/49152, m_exp=5 → 0/393216).  Timing and program count do not depend on the ciphertext
// VALUES, only on the shape, so depth can be swept with synthetic key material — which is what makes
// the library depth (m_exp=9, N0=512) measurable without a golden the Python reference cannot produce
// in reasonable time.
//
//   ./build/test_expand_bench [m_exp] [K] [ell]        # default 9 3 8 = the library shape
//
// Reports captured program count and traced ms/query, so the per-level cost can be checked for the
// flatness the projection assumed.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include "pir_engine2048.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace pir2048;

int main(int argc, char** argv) {
    const uint32_t m_exp = (argc > 1) ? (uint32_t)std::atoi(argv[1]) : 9;
    const uint32_t K     = (argc > 2) ? (uint32_t)std::atoi(argv[2]) : 3;
    const uint32_t ell   = (argc > 3) ? (uint32_t)std::atoi(argv[3]) : 8;
    std::vector<uint32_t> pr = {167772161u, 104857601u, 268271617u};
    pr.resize(K);

    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 256u << 20);
        auto& cq = mesh->mesh_command_queue();
        Engine eng(mesh.get(), &cq, pr, ell, 2048);
        std::mt19937 rng(1);

        // synthetic key material: shape-correct, values irrelevant to timing
        std::vector<std::vector<std::vector<MeshBuf>>> KnB(m_exp, std::vector<std::vector<MeshBuf>>(K, std::vector<MeshBuf>(2)));
        for (uint32_t a=0;a<m_exp;++a) for (uint32_t ch=0;ch<K;++ch) for (uint32_t col=0;col<2;++col){
            std::vector<uint32_t> buf((size_t)2*ell*ELEMS);
            for (auto& v : buf) v = rng() % pr[ch];
            auto b = eng.mkbuf(2*ell, I32); eng.wrb(b, buf); KnB[a][ch][col] = b; }
        std::vector<std::vector<uint32_t>> q(2*K);
        for (uint32_t i=0;i<2*K;++i){ q[i].resize(N); for (auto& v : q[i]) v = rng() % pr[i%K]; }

        const uint32_t N0 = 1u << m_exp;
        eng.prewarm_batch(N0);
        MeshBuf bquery = eng.upload_ctb({q});            // before pooling: never recycled
        eng.pool_enable(true); eng.wl_enable(true);

        auto run_tree = [&]{ eng.pool_reset(); eng.wl_reset();
            uint32_t B=1; MeshBuf cts=bquery;
            for (uint32_t a=0;a<m_exp;++a){ int64_t j=(int64_t)(N>>a)+1;
                // A level's two Subs operands — c and its monomial shift — are INDEPENDENT and share
                // the same key-switching key, so they go through ONE 2B-wide batch instead of two
                // B-wide ones.  A forward-NTT batch costs about the same at M=64 as at M=512, so
                // halving the batch COUNT is worth far more than halving each batch's size.
                auto comb=eng.mkctb(2*B);
                eng.permute_ctb_into(cts, B, comb, 2*B, 0,   1u, 0u);                 // identity copy
                eng.monomul_ctb_into(cts, B, comb, 2*B, B, -(int64_t)(1u<<a));    // shifted copy
                auto scr=eng.mkctb(2*B), sb=eng.mkctb(2*B);
                eng.subs_levelB(comb, 2*B, j, KnB[a], sb, scr);
                auto nxt=eng.mkctb(2*B); eng.ctadd_ctb(comb, sb, nxt, 2*B);
                cts=nxt; B*=2; } };

        auto t_b0 = std::chrono::steady_clock::now();
        run_tree();                                       // pass 1: build programs + fill the pool
        distributed::Finish(cq);
        double build_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_b0).count();
        const uint64_t nprog = eng.n_prog, nbuf = eng.n_buf;

        auto tid = mesh->begin_mesh_trace(0);
        run_tree();                                       // capture: every dispatch from the cache
        mesh->end_mesh_trace(0, tid);
        distributed::Finish(cq);

        const uint32_t R = 5;
        mesh->replay_mesh_trace(0, tid, true);            // warm
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t r=0;r<R;++r) mesh->replay_mesh_trace(0, tid, false);
        distributed::Finish(cq);
        double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/R;

        fmt::print("m_exp={} (N0={}) K={} ell={}: {} programs, {} buffers\n", m_exp, N0, K, ell, nprog, nbuf);
        fmt::print("  untraced build+run : {:8.1f} ms\n", build_ms);
        fmt::print("  TRACED replay      : {:8.2f} ms/query   ({:.3f} ms/program, {:.1f} programs/level)\n",
                   ms, ms/nprog, (double)nprog/m_exp);
        fmt::print("  speedup            : {:8.1f}x\n", build_ms/ms);
        mesh->release_mesh_trace(tid);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
