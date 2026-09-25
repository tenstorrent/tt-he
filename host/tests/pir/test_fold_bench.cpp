// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Traced hypercube-fold BENCHMARK at arbitrary cell count — the fold half of the library-shape
// latency, measured rather than projected.
//
// Correctness is established elsewhere against real goldens (test_pir_answer2048: 2 folds → 0/12288,
// 4 folds → 0/12288).  Program count per fold is flat, but the WORK is not: fold 1 processes `cells/2`
// ciphertexts, fold 2 half that, and so on — so the total is dominated by the first fold, the mirror
// image of the expand tree being dominated by its last level.  That is exactly what a flat
// per-program projection gets wrong, hence measuring it.
//
//   ./build/test_fold_bench [cells] [K] [ell]     # default 128 3 8 = the library hypercube tail

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
    const uint32_t cells = (argc > 1) ? (uint32_t)std::atoi(argv[1]) : 128;
    const uint32_t K     = (argc > 2) ? (uint32_t)std::atoi(argv[2]) : 3;
    const uint32_t ell   = (argc > 3) ? (uint32_t)std::atoi(argv[3]) : 8;
    std::vector<uint32_t> pr = {167772161u, 104857601u, 268271617u};
    pr.resize(K);
    uint32_t nfold = 0; for (uint32_t c = cells; c > 1; c >>= 1) ++nfold;

    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 256u << 20);
        auto& cq = mesh->mesh_command_queue();
        Engine eng(mesh.get(), &cq, pr, ell, 2048);
        std::mt19937 rng(1);

        // synthetic fold RGSWs, one per fold; shape-correct, values irrelevant to timing
        std::vector<std::vector<std::vector<MeshBuf>>> Cb(nfold);
        for (uint32_t f = 0; f < nfold; ++f) {
            std::vector<std::vector<std::vector<uint32_t>>> Cn(K, std::vector<std::vector<uint32_t>>(2*ell*2));
            for (uint32_t ch = 0; ch < K; ++ch) for (uint32_t i = 0; i < 2*ell*2; ++i) {
                Cn[ch][i].resize(N); for (auto& v : Cn[ch][i]) v = rng() % pr[ch]; }
            Cb[f] = eng.rgsw_operand(Cn); }

        std::vector<std::vector<std::vector<uint32_t>>> acc(cells, std::vector<std::vector<uint32_t>>(2*K));
        for (auto& ct : acc) for (uint32_t i = 0; i < 2*K; ++i) {
            ct[i].resize(N); for (auto& v : ct[i]) v = rng() % pr[i%K]; }

        eng.prewarm_batch(cells);
        MeshBuf bacc0 = eng.upload_ctb(acc);            // before pooling: never recycled
        eng.pool_enable(true); eng.wl_enable(true);

        auto run_folds = [&]{ eng.pool_reset(); eng.wl_reset();
            uint32_t B = cells; MeshBuf a = bacc0;
            for (uint32_t f = 0; f < nfold; ++f) { a = eng.fold_levelB(a, B, Cb[f]); B /= 2; } };

        auto t_b0 = std::chrono::steady_clock::now();
        run_folds();                                    // pass 1: build programs + fill the pool
        distributed::Finish(cq);
        double build_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_b0).count();
        const uint64_t nprog = eng.n_prog;

        auto tid = mesh->begin_mesh_trace(0);
        run_folds();                                    // capture: every dispatch from the cache
        mesh->end_mesh_trace(0, tid);
        distributed::Finish(cq);

        const uint32_t R = 5;
        mesh->replay_mesh_trace(0, tid, true);          // warm
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t r = 0; r < R; ++r) mesh->replay_mesh_trace(0, tid, false);
        distributed::Finish(cq);
        double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/R;

        fmt::print("cells={} ({} folds) K={} ell={}: {} programs ({:.0f}/fold)\n",
                   cells, nfold, K, ell, nprog, (double)nprog/nfold);
        fmt::print("  untraced build+run : {:8.1f} ms\n", build_ms);
        fmt::print("  TRACED replay      : {:8.2f} ms   ({:.3f} ms/program)\n", ms, ms/nprog);
        mesh->release_mesh_trace(tid);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
