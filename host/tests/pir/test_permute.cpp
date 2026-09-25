// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Validate pir_permute.hpp (Galois automorphism #14 + monomial multiply #15) bit-exact vs the
// CPU golden from cpu/pir/gen_permute_case.py (itself part of the end-to-end-validated
// onionpir_ref). Exercises the tile-layout path: golden natural input -> tilize (as the device
// would hold it) -> automorph_tiled/monomial_tiled -> untilize -> compare to golden.
//
//   ./build/test_permute [reference/pir/permute_case.bin]

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>
#include <fmt/core.h>
#include "pir_permute.hpp"

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "reference/pir/permute_case.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
    auto ru = [&](uint32_t& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    uint32_t magic, n, q, nj, nk;
    ru(magic); ru(n); ru(q); ru(nj); ru(nk);
    if (magic != 0x504D5554u) { fmt::print(stderr, "bad magic\n"); return 2; }
    const uint32_t ntiles = n / pir::TILE_ELEMS;
    std::vector<uint32_t> in(n);
    f.read(reinterpret_cast<char*>(in.data()), 4 * n);
    fmt::print("case {}: n={} q={} nj={} nk={} (ntiles={})\n", path, n, q, nj, nk, ntiles);

    auto in_tiles = pir::tilize_poly(in, ntiles);   // as the device holds the poly
    bool pass = true;

    for (uint32_t t = 0; t < nj; ++t) {
        uint32_t j; ru(j);
        std::vector<uint32_t> gold(n); f.read(reinterpret_cast<char*>(gold.data()), 4 * n);
        auto got = pir::untilize_poly(pir::automorph_tiled(in_tiles, ntiles, j, q), ntiles);
        uint32_t mism = 0; for (uint32_t i = 0; i < n; ++i) if (got[i] != gold[i]) ++mism;
        fmt::print("  automorph j={:5}: mismatches={}/{} {}\n", j, mism, n, mism ? "FAIL" : "ok");
        pass &= (mism == 0);
    }
    for (uint32_t t = 0; t < nk; ++t) {
        int32_t k; f.read(reinterpret_cast<char*>(&k), 4);
        std::vector<uint32_t> gold(n); f.read(reinterpret_cast<char*>(gold.data()), 4 * n);
        auto got = pir::untilize_poly(pir::monomial_tiled(in_tiles, ntiles, k, q), ntiles);
        uint32_t mism = 0; for (uint32_t i = 0; i < n; ++i) if (got[i] != gold[i]) ++mism;
        fmt::print("  monomial  k={:5}: mismatches={}/{} {}\n", k, mism, n, mism ? "FAIL" : "ok");
        pass &= (mism == 0);
    }

    // extra: j=n+1 is the "negate odd coefficients" special case (no permutation)
    {
        auto got = pir::automorph(in, n + 1, q);
        uint32_t mism = 0;
        for (uint32_t i = 0; i < n; ++i) { uint32_t e = (i & 1) ? (q - in[i]) % q : in[i]; if (got[i] != e) ++mism; }
        fmt::print("  automorph j={} == negate-odd: mismatches={}/{} {}\n", n + 1, mism, n, mism ? "FAIL" : "ok");
        pass &= (mism == 0);
    }

    fmt::print("{}\n", pass ? "Test Passed" : "Test FAILED");
    return pass ? 0 : 1;
}
