// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PIR data-movement ops (#14 Galois automorphism, #15 monomial multiply) as host reindex over the
// device's tile/RNS layout. Both are coefficient permutations with a negacyclic sign flip (x^n=-1);
// on TT they are realized host-side during the inter-stage transfers (like the NTT output reindex),
// no SFPU kernel. Each operand is one RNS channel of one polynomial; apply per channel per poly.
//
// WHICH OF lib/pir's THREE HEADERS THIS IS: the shared helper, and it stays its own 80 lines on
// purpose. It has 13 consumers and 10 of them never include pir_engine2048.hpp, so folding it into
// that header would make ten test programs compile 1711 lines of header-only engine to get these
// two permutations.
//
// Poly is `ntiles` 32x32 tiles (n = 1024*ntiles), values canonical in [0,q). A permutation crosses
// tile boundaries, so we untilize the whole poly to natural order, permute+sign, and re-tilize.

#pragma once
#include <cstdint>
#include <vector>

namespace pir {

constexpr uint32_t TT = 32, TILE_ELEMS = 1024;

// 32x32 face-tiled <-> row-major (matches every device kernel's tilize in this repo)
inline void tilize_tile(const uint32_t* nat, uint32_t* out) {
    for (uint32_t r = 0; r < TT; ++r)
        for (uint32_t c = 0; c < TT; ++c)
            out[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = nat[r * TT + c];
}
inline void untilize_tile(const uint32_t* til, uint32_t* out) {
    for (uint32_t r = 0; r < TT; ++r)
        for (uint32_t c = 0; c < TT; ++c)
            out[r * TT + c] = til[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)];
}

// whole poly (ntiles tiles) tilize/untilize to/from natural coefficient order
inline std::vector<uint32_t> untilize_poly(const std::vector<uint32_t>& tiles, uint32_t ntiles) {
    std::vector<uint32_t> nat(ntiles * TILE_ELEMS);
    for (uint32_t t = 0; t < ntiles; ++t)
        untilize_tile(tiles.data() + t * TILE_ELEMS, nat.data() + t * TILE_ELEMS);
    return nat;
}
inline std::vector<uint32_t> tilize_poly(const std::vector<uint32_t>& nat, uint32_t ntiles) {
    std::vector<uint32_t> tiles(ntiles * TILE_ELEMS);
    for (uint32_t t = 0; t < ntiles; ++t)
        tilize_tile(nat.data() + t * TILE_ELEMS, tiles.data() + t * TILE_ELEMS);
    return tiles;
}

// --- natural-order signed permutations (mod q, ring x^n + 1) ---

// Galois automorphism σ_j: out[·] = ± in[i], where exponent i·j wraps negacyclically mod 2n.
// j must be odd (coprime to 2n) so this is a bijection.
inline std::vector<uint32_t> automorph(const std::vector<uint32_t>& in, uint32_t j, uint32_t q) {
    const uint32_t n = (uint32_t)in.size(), twon = 2 * n;
    std::vector<uint32_t> out(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t p = (uint32_t)((uint64_t)i * j % twon);
        if (p < n) out[p] = in[i];
        else       out[p - n] = (q - in[i]) % q;      // (q - v) % q maps 0 -> 0
    }
    return out;
}

// Monomial multiply x^k (k may be negative): negacyclic coefficient shift with sign.
inline std::vector<uint32_t> monomial(const std::vector<uint32_t>& in, int32_t k, uint32_t q) {
    const int32_t n = (int32_t)in.size(), twon = 2 * n;
    std::vector<uint32_t> out(n);
    for (int32_t i = 0; i < n; ++i) {
        int32_t p = (i + k) % twon; if (p < 0) p += twon;
        if (p < n) out[p] = in[i];
        else       out[p - n] = (q - in[i]) % q;
    }
    return out;
}

// tile-layout wrappers: tiles in -> tiles out (untilize, permute+sign, re-tilize)
inline std::vector<uint32_t> automorph_tiled(const std::vector<uint32_t>& tiles, uint32_t ntiles, uint32_t j, uint32_t q) {
    return tilize_poly(automorph(untilize_poly(tiles, ntiles), j, q), ntiles);
}
inline std::vector<uint32_t> monomial_tiled(const std::vector<uint32_t>& tiles, uint32_t ntiles, int32_t k, uint32_t q) {
    return tilize_poly(monomial(untilize_poly(tiles, ntiles), k, q), ntiles);
}

}  // namespace pir
