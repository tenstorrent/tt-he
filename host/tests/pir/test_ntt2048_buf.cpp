// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BUFFER-RESIDENT four-step NTT at n=2048: `Engine::fwd2048_b`/`inv2048_b` take and return DRAM
// buffers with ZERO host transfers, because the four-step reindex — a fixed, data-independent
// coefficient permutation the host used to do on both ends of every transform — is now a baked
// device gather (`dataflow/gather_table.cpp`).
//
// Checks, per RNS channel:
//   (a) fwd2048_b == fwd2048   bit-exact (the validated host-reindex path is the reference)
//   (b) inv2048_b == inv2048   bit-exact
//   (c) inv2048_b(fwd2048_b(x)) == x   round-trip
//   (d) (base,stride) addressing: transform a poly living at tiles {ch, K+ch} of an interleaved
//       buffer (the digit_to_rns output layout) — this is what lets stages chain with no repacking
//   (e) host<->device transfer count for a chained fwd→inv, which must be ZERO
//
//   ./build/test_ntt2048_buf [seed]

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
    const uint32_t seed = (argc > 1) ? (uint32_t)std::atoi(argv[1]) : 0;
    std::vector<uint32_t> pr = {167772161u, 104857601u, 268271617u};
    const uint32_t K = (uint32_t)pr.size();

    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 256u << 20);
        auto& cq = mesh->mesh_command_queue();
        Engine eng(mesh.get(), &cq, pr, 8, 2048);

        std::mt19937 rng(seed);
        auto cmp = [&](const char* what, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
            uint32_t mism = 0;
            for (uint32_t x = 0; x < N; ++x) if (a[x] != b[x]) {
                if (mism < 4) fmt::print("  {} [{}] got={} exp={}\n", what, x, a[x], b[x]);
                ++mism; }
            fmt::print("  {:<34} mismatches={}/{} {}\n", what, mism, N, mism ? "FAIL" : "ok");
            pass &= (mism == 0); };

        for (uint32_t ch = 0; ch < K; ++ch) {
            std::vector<uint32_t> x(N);
            for (uint32_t i = 0; i < N; ++i) x[i] = rng() % pr[ch];
            auto bx = eng.tilebuf2(x), bf = eng.mkbuf(2, I32), bi = eng.mkbuf(2, I32);

            // (a) forward: buffer path vs validated host-reindex path
            eng.fwd2048_b(eng.CH[ch], bx, bf);
            cmp(fmt::format("ch{} fwd2048_b == fwd2048", ch).c_str(), unt2(eng.rdb(bf)), eng.fwd2048(eng.CH[ch], x));

            // (b) inverse: feed both paths the same eval-domain poly
            auto X = eng.fwd2048(eng.CH[ch], x);
            auto bX = eng.tilebuf2(X);
            eng.inv2048_b(eng.CH[ch], bX, bi);
            cmp(fmt::format("ch{} inv2048_b == inv2048", ch).c_str(), unt2(eng.rdb(bi)), eng.inv2048(eng.CH[ch], X));

            // (c) buffer-chained round-trip: fwd -> inv with no host touch in between
            eng.reset_io();
            eng.fwd2048_b(eng.CH[ch], bx, bf);
            eng.inv2048_b(eng.CH[ch], bf, bi);
            auto rt = unt2(eng.rdb(bi));
            cmp(fmt::format("ch{} inv_b(fwd_b(x)) == x", ch).c_str(), rt, x);
            fmt::print("  {:<34} host writes={} reads={} (1 read = the final check)\n",
                       fmt::format("ch{} chained fwd->inv I/O", ch), eng.n_wrb, eng.n_rdb);
            pass &= (eng.n_wrb == 0 && eng.n_rdb == 1);
        }

        // (d) (base,stride): poly at tiles {ch, K+ch} of an interleaved 2K-tile buffer
        { const uint32_t ch = 1;
          std::vector<uint32_t> x(N); for (uint32_t i = 0; i < N; ++i) x[i] = rng() % pr[ch];
          auto bmix = eng.mkbuf(2 * K, I32), bf = eng.mkbuf(2, I32);
          std::vector<uint32_t> raw(2 * K * ELEMS, 0);                      // junk in the other slots
          for (auto& v : raw) v = rng() % pr[ch];
          auto t = til2(x);
          std::copy(t.begin(), t.begin() + ELEMS, raw.begin() + (size_t)ch * ELEMS);
          std::copy(t.begin() + ELEMS, t.end(), raw.begin() + (size_t)(K + ch) * ELEMS);
          eng.wrb(bmix, raw);
          eng.fwd2048_b(eng.CH[ch], bmix, bf, /*in_base=*/ch, /*in_stride=*/K);
          cmp("interleaved in (base=ch, stride=K)", unt2(eng.rdb(bf)), eng.fwd2048(eng.CH[ch], x)); }

        // (f) BATCHED forward NTT: M transforms in ~12 programs (phase-major mbatch) must equal the
        //     per-poly buffer path exactly.  Input laid out as the key-switch sees it — poly m at
        //     tile m of an interleaved buffer, halves ell*K apart.
        for (uint32_t M : {12u, 64u, 130u}) { const uint32_t ch=0; const uint32_t S=M;   // S = poly stride between halves
          // M=12 -> 1 tile/core on 12 cores; 64 -> 1 on 64; 130 -> 3 on 44 (crosses the per-core L1 cap)
          std::vector<std::vector<uint32_t>> xs(M);
          auto bin=eng.mkbuf(2*M, I32); std::vector<uint32_t> raw(2*M*ELEMS);
          for (uint32_t m=0;m<M;++m){ xs[m].resize(N);
              for (uint32_t i=0;i<N;++i) xs[m][i]=rng()%pr[ch];
              auto t=til2(xs[m]);
              std::copy(t.begin(), t.begin()+ELEMS, raw.begin()+(size_t)m*ELEMS);          // half 0 at m
              std::copy(t.begin()+ELEMS, t.end(),   raw.begin()+(size_t)(S+m)*ELEMS); }    // half 1 at S+m
          eng.wrb(bin, raw);
          auto bout=eng.mkbuf(2*M, I32);
          eng.reset_io();
          eng.fwd2048_batchM(eng.CH[ch], bin, /*in_base=*/0, /*in_pstride=*/1, /*in_tstride=*/S,
                             M, bout, /*out_base=*/0, /*out_pstride=*/2);
          const uint64_t nprog=eng.n_prog, nw=eng.n_wrb, nr=eng.n_rdb;
          auto got=eng.rdb(bout);
          uint32_t mism=0;
          for (uint32_t m=0;m<M;++m){ auto exp=eng.fwd2048(eng.CH[ch], xs[m]);
              auto g=unt2(std::vector<uint32_t>(got.begin()+(size_t)2*m*ELEMS, got.begin()+(size_t)(2*m+2)*ELEMS));
              for (uint32_t x=0;x<N;++x) if (g[x]!=exp[x]) { if(mism<4) fmt::print("  batch m={} [{}] got={} exp={}\n",m,x,g[x],exp[x]); ++mism; } }
          fmt::print("  {:<34} mismatches={}/{}  in {} programs ({:.1f}/NTT vs 8), host I/O {}w/{}r\n",
                     fmt::format("batched fwd x{}", M), mism, M*N, nprog, (double)nprog/M, nw, nr);
          pass &= (mism==0 && nw==0 && nr==0); }

        // (h) BATCHED INVERSE NTT: same fixed 12 programs; validate against the per-poly inverse and
        //     as a batched round-trip inv(fwd(x)) == x.
        for (uint32_t M : {12u, 64u, 130u}) { const uint32_t ch=2;
          std::vector<std::vector<uint32_t>> xs(M);
          auto bin=eng.mkbuf(2*M,I32); std::vector<uint32_t> raw(2*M*ELEMS);
          for (uint32_t m=0;m<M;++m){ xs[m].resize(N);
              for (uint32_t i=0;i<N;++i) xs[m][i]=rng()%pr[ch];
              auto t=til2(xs[m]); std::copy(t.begin(),t.end(),raw.begin()+(size_t)2*m*ELEMS); }
          eng.wrb(bin,raw);
          auto bf=eng.mkbuf(2*M,I32), bi=eng.mkbuf(2*M,I32);
          eng.reset_io();
          eng.fwd2048_batchM(eng.CH[ch], bin,0,2,1, M, bf,0,2);
          eng.inv2048_batchM(eng.CH[ch], bf, 0,2,1, M, bi,0,2);
          const uint64_t np=eng.n_prog;
          auto gi=eng.rdb(bi), gf=eng.rdb(bf);
          uint32_t rt=0, fw=0;
          for (uint32_t m=0;m<M;++m){
              auto expF=eng.fwd2048(eng.CH[ch], xs[m]);
              auto gF=unt2(std::vector<uint32_t>(gf.begin()+(size_t)2*m*ELEMS, gf.begin()+(size_t)(2*m+2)*ELEMS));
              auto gI=unt2(std::vector<uint32_t>(gi.begin()+(size_t)2*m*ELEMS, gi.begin()+(size_t)(2*m+2)*ELEMS));
              for (uint32_t x=0;x<N;++x){ fw += (gF[x]!=expF[x]); rt += (gI[x]!=xs[m][x]); } }
          fmt::print("  {:<34} fwd={}/{}  inv(fwd(x))==x {}/{}  in {} programs\n",
                     fmt::format("batched inv x{}", M), fw, M*N, rt, M*N, np);
          pass &= (fw==0 && rt==0); }

        // (g) SCALING PROBE — is a batched program dispatch-bound or compute-bound?
        //     fwd2048_batchM costs a FIXED 12 programs regardless of M.  If wall time is flat in M we
        //     are paying dispatch and the lever is bigger batches; if it grows with M the programs are
        //     compute-bound and the lever is sharding those M tiles across the core grid.
        { const uint32_t ch=0; fmt::print("\n  batched-fwd scaling (12 programs at every M):\n");
          for (uint32_t M : {1u,8u,32u,64u,128u,256u,512u}) {
              auto bin=eng.mkbuf(2*M,I32), bout=eng.mkbuf(2*M,I32);
              std::vector<uint32_t> raw(2*M*ELEMS); for (auto& v:raw) v=rng()%pr[ch];
              eng.wrb(bin, raw);
              const uint32_t R=8;
              eng.fwd2048_batchM(eng.CH[ch], bin,0,1,M,M,bout,0,2); distributed::Finish(cq);  // warm
              auto s=std::chrono::steady_clock::now();
              for (uint32_t r=0;r<R;++r) eng.fwd2048_batchM(eng.CH[ch], bin,0,1,M,M,bout,0,2);
              distributed::Finish(cq);
              double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-s).count()/R;
              fmt::print("    M={:<3} {:7.2f} ms/batch  {:6.3f} ms/NTT  {:6.3f} ms/program\n",
                         M, ms, ms/M, ms/12.0); } }

        fmt::print("\n  per-program cost breakdown (12 programs per batch):\n");
        // (i) WHERE DOES THE ~0.7 ms/program GO?  Every timing in this project is host wall-clock,
        //     so split it: build-only (dry_run) vs build+enqueue+execute.
        for (uint32_t M : {1u, 64u, 512u}) { const uint32_t ch=0, R=16;
          auto bin=eng.mkbuf(2*M,I32), bout=eng.mkbuf(2*M,I32);
          std::vector<uint32_t> raw(2*M*ELEMS); for (auto& v:raw) v=rng()%pr[ch];
          eng.wrb(bin,raw);
          eng.fwd2048_batchM(eng.CH[ch], bin,0,1,M,M,bout,0,2); distributed::Finish(cq);   // warm
          auto t0=std::chrono::steady_clock::now();
          for (uint32_t r=0;r<R;++r) eng.fwd2048_batchM(eng.CH[ch], bin,0,1,M,M,bout,0,2);
          distributed::Finish(cq);
          double full=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/(R*12);
          eng.dry_run=true;
          auto t1=std::chrono::steady_clock::now();
          for (uint32_t r=0;r<R;++r) eng.fwd2048_batchM(eng.CH[ch], bin,0,1,M,M,bout,0,2);
          double build=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t1).count()/(R*12);
          eng.dry_run=false;
          fmt::print("    M={:<4} full {:.3f} ms/prog = host-build {:.3f} ({:.0f}%) + enqueue&device {:.3f} ({:.0f}%)\n",
                     M, full, build, 100*build/full, full-build, 100*(full-build)/full); }

        // (j) TRACED NTT throughput — the ceiling the expand stage is running against.  Untraced
        //     numbers are host-bound and say nothing about how well the grid is used; traced they do.
        { const uint32_t ch=0; fmt::print("\n  TRACED forward-NTT throughput (12 programs/batch):\n");
          eng.pool_enable(true); eng.wl_enable(true);
          for (uint32_t M : {64u, 512u, 2048u}) {
              auto bin=eng.mkbuf(2*M,I32), bout=eng.mkbuf(2*M,I32);
              std::vector<uint32_t> raw(2*M*ELEMS); for (auto& v:raw) v=rng()%pr[ch];
              eng.wrb(bin,raw);
              eng.caches_clear();          // shape changes between M: stale programs would replay
              auto go=[&]{ eng.pool_reset(); eng.wl_reset();
                  eng.fwd2048_batchM(eng.CH[ch], bin,0,1,M,M,bout,0,2); };
              go(); distributed::Finish(cq);                       // build
              auto tid=mesh->begin_mesh_trace(0); go(); mesh->end_mesh_trace(0,tid);
              distributed::Finish(cq);
              const uint32_t R=5; mesh->replay_mesh_trace(0,tid,true);
              auto t0=std::chrono::steady_clock::now();
              for (uint32_t r=0;r<R;++r) mesh->replay_mesh_trace(0,tid,false);
              distributed::Finish(cq);
              double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/R;
              fmt::print("    M={:<5} {:7.2f} ms  {:8.4f} ms/NTT  {:8.0f} NTT/s\n", M, ms, ms/M, 1000.0*M/ms);
              mesh->release_mesh_trace(tid); }
          eng.pool_enable(false); eng.wl_enable(false); }

        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
