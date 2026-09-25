// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Cross-check the offline encoder against the DEVICE.
//
// `encode_db --verify` only proves its fast NTT agrees with its own slow NTT. What actually matters is
// that the file matches what `Engine::to_mont(fwd2048(...))` produces, because that is what the scan
// will consume. A convention mismatch — a different primitive root, a different Montgomery R, a
// different tile layout — would pass the encoder's self-check and silently corrupt every retrieval.
//
// So: take rows from the text file, push them through the device transform, and compare bit-for-bit
// against the encoded file.
//
//   ./build/test_db_encode_check [db.txt] [db.pir] [n_rows_to_check]

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "pir_engine2048.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace pir2048;

int main(int argc, char** argv) {
    const char* txt = (argc > 1) ? argv[1] : "reference/pir/db65536.txt";
    const char* bin = (argc > 2) ? argv[2] : "reference/pir/db65536.pir";
    const uint32_t n_check = (argc > 3) ? (uint32_t)std::atoi(argv[3]) : 6;

    std::ifstream ft(txt);
    if (!ft) { fmt::print(stderr, "cannot open {}\n", txt); return 2; }
    std::vector<std::string> rows; std::string line;
    while (std::getline(ft, line)) rows.push_back(line);

    std::ifstream fb(bin, std::ios::binary);
    if (!fb) { fmt::print(stderr, "cannot open {}\n", bin); return 2; }
    uint32_t hdr[8]; fb.read((char*)hdr, sizeof(hdr));
    if (hdr[0] != 0x50444231u) { fmt::print(stderr, "bad magic in {}\n", bin); return 2; }
    const uint32_t n = hdr[1], K = hdr[2], t_plain = hdr[3], n_rows = hdr[4];
    std::vector<uint32_t> primes = {hdr[5], hdr[6], hdr[7]};
    primes.resize(K);
    fmt::print("{}: n={} K={} t={} rows={} primes[0]={}\n", bin, n, K, t_plain, n_rows, primes[0]);
    if (n != N || rows.size() != n_rows) { fmt::print(stderr, "shape mismatch vs {}\n", txt); return 2; }

    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        Engine eng(mesh.get(), &cq, primes, 8, 2048);

        const size_t per_ch = (size_t)n_rows * N;
        std::mt19937 rng(7);
        uint32_t mism = 0, checked = 0;
        for (uint32_t trial = 0; trial < n_check; ++trial) {
            const uint32_t r = (trial == 0) ? 0
                             : (trial == 1) ? n_rows - 1
                             : rng() % n_rows;                      // always cover first and last row
            // the device transform, from the same ASCII source
            std::vector<uint32_t> coeff(N, 0);
            const std::string& s = rows[r];
            for (size_t i = 0; i < s.size() && i < N; ++i) coeff[i] = (uint32_t)(unsigned char)s[i] % t_plain;

            for (uint32_t chx = 0; chx < K; ++chx) {
                auto want = til2(eng.to_mont(chx, eng.fwd2048(eng.CH[chx], coeff)));
                std::vector<uint32_t> got(N);
                fb.seekg((std::streamoff)(sizeof(hdr) + ((size_t)chx * per_ch + (size_t)r * N) * sizeof(uint32_t)));
                fb.read((char*)got.data(), (std::streamsize)(N * sizeof(uint32_t)));
                for (uint32_t i = 0; i < N; ++i) {
                    if (got[i] != want[i]) {
                        if (mism < 6) fmt::print("  row {} ch {} [{}] file={} device={}\n", r, chx, i, got[i], want[i]);
                        ++mism;
                    }
                    ++checked;
                }
            }
            if (trial < 2) fmt::print("  row {:5} ok: \"{}\"\n", r, s.substr(0, 60));
        }
        fmt::print("encoder vs device: mismatches={}/{} over {} rows x {} channels\n",
                   mism, checked, n_check, K);
        pass &= (mism == 0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
