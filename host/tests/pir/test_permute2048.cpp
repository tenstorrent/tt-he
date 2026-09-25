// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PIR #14/#15 ON DEVICE at n=2048: Galois automorphism σ_j and monomial multiply x^k, run by
// `kernels/shared/dataflow/permute_poly.cpp` via `pir2048::Engine::automorph_ct / monomul_ct`.
// Previously these were the host reindex of `pir_permute.hpp` (the last host arithmetic in the
// Subs tree of query expansion).
//
// (a) vs the CPU-reference golden `reference/pir/permute_case.bin` (MAGIC 'PMUT'): every σ_j and x^k in
//     the case, single modulus.
// (b) multi-poly / multi-prime batch (K=3, 12 polys = 2 cts): exercises the per-poly modulus
//     selection q[i % K] and the multi-core poly shard, vs the validated host `pir::automorph` /
//     `pir::monomial`.
//
//   ./build/test_permute2048 [reference/pir/permute_case.bin]

#include <cstdint>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <random>
#include <vector>
#include "pir_engine2048.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace pir2048;

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "reference/pir/permute_case.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
    auto ru = [&](uint32_t& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    uint32_t magic, n, q, nj, nk;
    ru(magic); ru(n); ru(q); ru(nj); ru(nk);
    if (magic != 0x504D5554u || n != N) { fmt::print(stderr, "bad case (need PMUT n=2048)\n"); return 2; }
    auto rv = [&]() { std::vector<uint32_t> v(N); f.read(reinterpret_cast<char*>(v.data()), 4 * N); return v; };
    auto inp = rv();
    std::vector<uint32_t> js(nj); std::vector<std::vector<uint32_t>> gj(nj);
    for (uint32_t i = 0; i < nj; ++i) { ru(js[i]); gj[i] = rv(); }
    std::vector<int32_t> ks(nk); std::vector<std::vector<uint32_t>> gk(nk);
    for (uint32_t i = 0; i < nk; ++i) { f.read(reinterpret_cast<char*>(&ks[i]), 4); gk[i] = rv(); }
    fmt::print("case {}: n={} q={} nj={} nk={}\n", path, n, q, nj, nk);

    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();

        auto cmp = [&](const char* what, const std::vector<uint32_t>& got, const std::vector<uint32_t>& exp) {
            uint32_t mism = 0;
            for (uint32_t x = 0; x < N; ++x) if (got[x] != exp[x]) {
                if (mism < 4) fmt::print("  {} [{}] got={} exp={}\n", what, x, got[x], exp[x]);
                ++mism; }
            fmt::print("  {:<24} mismatches={}/{} {}\n", what, mism, N, mism ? "FAIL" : "ok");
            pass &= (mism == 0); };

        // (a) golden gate — single modulus, one poly per call
        { Engine eng(mesh.get(), &cq, {q}, 1, 2);                       // ell/B unused by permute
          for (uint32_t i = 0; i < nj; ++i)
              cmp(fmt::format("automorph j={}", js[i]).c_str(), eng.automorph_ct({inp}, js[i])[0], gj[i]);
          for (uint32_t i = 0; i < nk; ++i)
              cmp(fmt::format("monomial k={}", ks[i]).c_str(), eng.monomul_ct({inp}, ks[i])[0], gk[i]); }

        // (b) K=3 batch — 12 polys (2 cts), poly i mod primes[i % 3], sharded across the grid
        { std::vector<uint32_t> pr = {167772161u, 104857601u, 268271617u};
          Engine eng(mesh.get(), &cq, pr, 1, 2);
          const uint32_t NP = 12; const int64_t j = N / 4 + 1, kk = -(int64_t)(N / 8) - 3;
          std::mt19937 rng(7); std::vector<std::vector<uint32_t>> ps(NP);
          for (uint32_t i = 0; i < NP; ++i) { ps[i].resize(N);
              for (uint32_t x = 0; x < N; ++x) ps[i][x] = rng() % pr[i % 3]; }
          auto ga = eng.automorph_ct(ps, j), gm = eng.monomul_ct(ps, kk);
          uint32_t amis = 0, mmis = 0;
          for (uint32_t i = 0; i < NP; ++i) {
              auto ea = pir::automorph(ps[i], (uint32_t)j, pr[i % 3]);
              auto em = pir::monomial(ps[i], (int32_t)kk, pr[i % 3]);
              for (uint32_t x = 0; x < N; ++x) { amis += (ga[i][x] != ea[x]); mmis += (gm[i][x] != em[x]); } }
          fmt::print("  K=3 batch automorph j={}   mismatches={}/{}\n", j, amis, NP * N);
          fmt::print("  K=3 batch monomial k={}  mismatches={}/{}\n", kk, mmis, NP * N);
          pass &= (amis == 0 && mmis == 0);

          // cost of one ct-permute (2K=6 polys, one program) as the Subs tree issues it
          std::vector<std::vector<uint32_t>> ct(ps.begin(), ps.begin() + 6);
          const uint32_t REP = 64;
          auto t0 = std::chrono::steady_clock::now();
          for (uint32_t r = 0; r < REP; ++r) (void)eng.automorph_ct(ct, j);
          auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
          fmt::print("  automorph of one K=3 ct (6 polys, 1 program): {:.2f} ms/ct over {} reps\n", ms / REP, REP); }

        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
