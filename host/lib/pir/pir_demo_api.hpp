// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Minimal OnionPIR API — hides all TT-Metalium device control behind a few calls, so the demo reads
// like the protocol.  Client crypto (keygen / encrypt query / RGSW / decrypt) is the CPU reference
// (cpu/pir/gen_pir_demo_case.py); this API is the SERVER (device) + the client's decrypt of the
// response.  See pir_demo.md.
// WHICH OF lib/pir's THREE HEADERS THIS IS: the small one, and examples/pir_demo.cpp is its ONLY
// consumer -- a 4-entry teaching demo whose whole point is that the source reads like the protocol.
// For anything else use pir_engine2048.hpp: that is the K-general engine at n=2048 which the
// 65,536-row demo and 17 tests run on. Named pir_demo_api (not pir_api) so the scope is visible from
// the filename -- it is not the library's main PIR surface.
//
#pragma once
#include <cstdint>
#include <vector>

namespace pir {

// A retrieval instance: the public database + the client's encrypted query + the client secret
// (kept here only so the demo can decrypt) + the expected entry (for the OK/MISMATCH check).
// All polynomials are per RNS channel (mod q1, mod q2), coefficient domain.
// The client uploads ONE packed query + one-time key material; the server EXPANDS it into the N0
// initial-dimension selectors (Subs tree) and UNPACKS the fold bit into an RGSW (RLWE->RGSW via
// RGSW(s)), all on device — full OnionPIR, nothing pre-expanded.
struct Case {
    uint32_t n, ell, B, N0, m_exp, q1, q2, qp, idx, t;
    std::vector<std::vector<std::vector<uint32_t>>> DB;     // [ch][entry]            (N0*2 plaintext entries)
    std::vector<std::vector<uint32_t>> query;              // [comp*2+ch]            (packed init-dim query: c0q1,c0q2,c1q1,c1q2)
    std::vector<std::vector<uint32_t>> qfold;              // [comp*2+ch]            (packed FOLD-ladder query, expanded on device)
    std::vector<std::vector<std::vector<uint32_t>>> gkeys;  // [level*2+ch][k*2+col]  (Galois KSK, max(m_exp,m_lad) levels)
    std::vector<std::vector<std::vector<uint32_t>>> gsw_s;  // [ch][row*2+col]        (RGSW(s), 2ℓ×2 — unpack key)
    std::vector<int32_t> skey;                             // client secret (ternary)
    std::vector<int32_t> expected;                         // DB[idx] (for the check)
    uint32_t entries() const { return N0 * 2; }
};

Case load_case(const char* path);                         // parse the client's request + DB

// The server's encrypted response, mod-switched to q' — a BFV ciphertext (c0', c1').
struct Response { std::vector<std::vector<uint32_t>> c; };

// SERVER, on device:  initial-dimension dot product -> hypercube fold (external product) -> mod-switch.
Response server_answer(const Case& c);

// CLIENT, on host: decrypt the mod-switched response -> the recovered plaintext entry.
std::vector<uint32_t> client_decrypt(const Case& c, const Response& r);

}  // namespace pir
