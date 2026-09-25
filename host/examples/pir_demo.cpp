// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// OnionPIR demo on Tenstorrent — private information retrieval.
//   client encrypts a query for index i  ->  SERVER runs on device (never learns i)  ->  client
//   decrypts  ->  DB[i].  All device/TT-Metalium control is hidden behind pir::* (see pir_demo_api.hpp).
//
//   ./build/pir_demo [index] [db_entries]      (default: index 3, 4 entries)
//
// The client query for `index` is produced by the CPU reference (cpu/pir/gen_pir_demo_case.py); the
// server pipeline runs on the device. Each DB slot holds one integer (printed below). Run from device/.
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "pir_demo_api.hpp"

int main(int argc, char** argv) {
    // Quiet the TT-Metalium device logs so the demo output is clean (re-exec once so the logger,
    // which initializes at library load, picks up the level). Set PIR_VERBOSE=1 to keep the logs.
    if (!std::getenv("PIR_VERBOSE") && !std::getenv("PIR_QUIET_")) {
        setenv("TT_METAL_LOGGER_LEVEL", "FATAL", 1);
        setenv("PIR_QUIET_", "1", 1);
        execv("/proc/self/exe", argv);
    }
    int index   = (argc > 1) ? std::atoi(argv[1]) : 3;
    int entries = (argc > 2) ? std::atoi(argv[2]) : 4;      // hypercube = (entries/2) x 2
    if (entries < 2 || (entries & 1)) { printf("db_entries must be even and >= 2\n"); return 2; }
    if (index < 0 || index >= entries) { printf("index must be in [0, %d)\n", entries); return 2; }

    // Client builds the encrypted query for this index (CPU reference); DB is fixed across indices.
    const char* gen = std::getenv("PIR_GEN") ? std::getenv("PIR_GEN") : "reference/pir/gen_pir_demo_case.py";
    const char* casef = "/tmp/pir_demo_case.bin";
    char cmd[1024];
    std::snprintf(cmd, sizeof cmd,
                  "python3 %s --idx %d --N0 %d --out %s > /tmp/pir_gen.log 2>&1",
                  gen, index, entries / 2, casef);
    if (std::system(cmd) != 0) {
        printf("error: could not build the query (need python3 + %s; see /tmp/pir_gen.log)\n", gen);
        return 2;
    }

    printf("\n===== OnionPIR — private information retrieval on Tenstorrent =====\n\n");
    pir::Case c;
    try { c = pir::load_case(casef); }
    catch (const std::exception& e) { printf("error: %s\n", e.what()); return 2; }

    printf("scheme:   OnionPIR, N=%u, composite q = q1*q2 (28-bit RNS), plaintext modulus t=%u\n", c.n, c.t);
    printf("database: %u entries (hypercube %u x 2), each slot holds one integer:\n\n", c.entries(), c.N0);
    for (uint32_t e = 0; e < c.entries(); ++e)
        printf("            DB[%u] = %u%s\n", e, c.DB[0][e][0], e == (uint32_t)index ? "   <- requested" : "");
    printf("\nclient wants index %d  (kept secret from the server)\n\n", index);

    printf("[1] client -> server:  ONE packed query ct + one-time keys (Galois keys, RGSW(s))\n");
    printf("[2] SERVER on device:  expand query (Subs tree) -> unpack fold bit (RLWE->RGSW)\n");
    printf("                       ->  initial-dimension dot product  ->  hypercube fold (external product)\n");
    printf("                       ->  mod-switch the response\n");
    pir::Response resp;
    try { resp = pir::server_answer(c); }
    catch (const std::exception& e) { printf("device error: %s\n", e.what()); return 2; }

    printf("[3] server -> client:  one encrypted response ciphertext\n");
    printf("[4] client decrypts.\n\n");
    std::vector<uint32_t> rec = pir::client_decrypt(c, resp);

    uint32_t mism = 0;
    for (size_t i = 0; i < rec.size(); ++i) if (rec[i] != (uint32_t)c.expected[i]) ++mism;
    bool ok = (mism == 0);

    printf("-----------------------------------------------------------------\n");
    printf("  retrieved DB[%d] = %u    (expected %u)   %s\n",
           index, rec[0], (uint32_t)c.expected[0], ok ? "OK" : "MISMATCH");
    printf("  the server touched all %u entries but never learned index %d.\n", c.entries(), index);
    printf("-----------------------------------------------------------------\n\n");
    printf("%s\n\n", ok ? "Private retrieval succeeded." : "FAILED.");
    return ok ? 0 : 1;
}
