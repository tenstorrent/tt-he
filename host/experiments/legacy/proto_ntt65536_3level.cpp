// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host prototype of the 3-level (32, 32, 64) NTT-65536, with the 64-dim done MATRIX-SIDE.
// Validates the index algebra and the twiddle formulas against a reference negacyclic NTT before
// any of it is written as device code.
//
//   n  = n1 + 32*n2 + 1024*n3      n1,n2 in [0,32), n3 in [0,64) = n3_lo + 32*n3_hi
//   k  = kv_lo + 32*kv_hi + 64*ku + 2048*ka
//
//   tile index = (n1, n3_hi)   32 x 2 = 64 tiles ; in-tile = [n3_lo][n2]
//
//   P1  contract n3 (64)  blocked over n3_hi, in-tile n3_lo, W64 as 2x2 blocks  -> kv
//   Ta  twiddle w_2048^{n2 kv}
//   P2  contract n2 (32)  in-tile cols, W32                                     -> ku
//   Tb  twiddle w_N^{n1 (kv + 64 ku)}
//   A2A corner turn n1 <-> kv_lo, within each kv_hi group of 32 tiles
//   P3  contract n1 (32)  in-tile rows, W32                                     -> ka
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
using u128 = unsigned __int128;
using u64 = uint64_t;

static u64 mpow(u64 b, u64 e, u64 m) { u128 r = 1, x = b % m; while (e) { if (e & 1) r = r * x % m; x = x * x % m; e >>= 1; } return (u64)r; }
static bool isp(u64 n) { if (n < 2) return false; for (u64 d = 2; d * d <= n; ++d) if (n % d == 0) return false; return true; }
static u64 proot(u64 q) { u64 phi = q - 1, n = phi; std::vector<u64> f;
    for (u64 d = 2; d * d <= n; ++d) if (n % d == 0) { f.push_back(d); while (n % d == 0) n /= d; }
    if (n > 1) f.push_back(n);
    for (u64 g = 2; g < q; ++g) { bool ok = true; for (u64 p : f) if (mpow(g, phi / p, q) == 1) ok = false; if (ok) return g; } return 0; }

static const uint32_t N = 65536, T = 32, NT = 64;   // NT tiles of T x T

int main() {
    u64 q = 0;
    for (u64 k = ((1ull << 28) - 1) / (2 * N); k > 0; --k) { u64 p = k * 2 * N + 1; if (p < (1ull << 28) && isp(p)) { q = p; break; } }
    u64 g = proot(q), psi = mpow(g, (q - 1) / (2 * N), q), w = (u128)psi * psi % q;
    std::printf("q=%llu  psi=%llu\n", (unsigned long long)q, (unsigned long long)psi);

    std::mt19937_64 rng(1);
    std::vector<uint32_t> x(N);
    for (auto& v : x) v = (uint32_t)(rng() % q);

    // ---- reference: negacyclic forward NTT, O(N log N) ----
    std::vector<u64> ref(N);
    {
        std::vector<u64> a(N);
        for (uint32_t i = 0; i < N; ++i) a[i] = (u128)x[i] * mpow(psi, i, q) % q;
        for (uint32_t i = 1, j = 0; i < N; ++i) { uint32_t b = N >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b; if (i < j) std::swap(a[i], a[j]); }
        for (uint32_t len = 2; len <= N; len <<= 1) { uint32_t st = N / len;
            for (uint32_t i = 0; i < N; i += len) for (uint32_t kk = 0; kk < len / 2; ++kk) {
                u64 wk = mpow(w, (u64)kk * st, q);
                u64 u = a[i + kk], v = (u128)a[i + kk + len / 2] * wk % q;
                a[i + kk] = (u + v) % q; a[i + kk + len / 2] = (u + q - v) % q; } }
        ref = a;
    }

    // ---- 3-level ----
    // twist folded in: psi^n = psi^{n1} * psi^{32 n2} * psi^{1024 n3}
    //   psi^{1024 n3} -> into W64 (P1), psi^{32 n2} -> into Ta, psi^{n1} -> into Tb
    const u64 w64 = mpow(w, N / 64, q);       // primitive 64th root
    const u64 w32 = mpow(w, N / 32, q);       // primitive 32nd root
    const u64 w2048 = mpow(w, N / 2048, q);   // primitive 2048th root

    // tile[t] with t = n1*2 + n3_hi, holding [n3_lo][n2]
    std::vector<std::vector<u64>> D(NT, std::vector<u64>(T * T));
    for (uint32_t n1 = 0; n1 < 32; ++n1)
        for (uint32_t n3 = 0; n3 < 64; ++n3)
            for (uint32_t n2 = 0; n2 < 32; ++n2)
                D[n1 * 2 + (n3 >> 5)][(n3 & 31) * T + n2] = x[n1 + 32 * n2 + 1024 * n3];

    // ---- P1: contract n3 (64), blocked. out[kv] = sum_{n3} W64[kv][n3] * in[n3],
    //      W64[kv][n3] = w64^{kv*n3} * psi^{1024*n3}   (the twist folded in)
    {
        std::vector<std::vector<u64>> O(NT, std::vector<u64>(T * T, 0));
        for (uint32_t n1 = 0; n1 < 32; ++n1)
            for (uint32_t kv = 0; kv < 64; ++kv) {
                auto& out = O[n1 * 2 + (kv >> 5)];
                for (uint32_t n3 = 0; n3 < 64; ++n3) {
                    u64 c = (u128)mpow(w64, (u64)kv * n3 % 64, q) * mpow(psi, 1024ull * n3 % (2 * N), q) % q;
                    const auto& in = D[n1 * 2 + (n3 >> 5)];
                    for (uint32_t n2 = 0; n2 < 32; ++n2)
                        out[(kv & 31) * T + n2] = (out[(kv & 31) * T + n2] + (u128)c * in[(n3 & 31) * T + n2]) % q;
                }
            }
        D = O;
    }
    // ---- Ta: twiddle w_2048^{n2 * kv} ; kv = kv_lo + 32*kv_hi
    for (uint32_t t = 0; t < NT; ++t) { uint32_t kv_hi = t & 1;
        for (uint32_t kv_lo = 0; kv_lo < 32; ++kv_lo) for (uint32_t n2 = 0; n2 < 32; ++n2) {
            uint32_t kv = kv_lo + 32 * kv_hi;
            D[t][kv_lo * T + n2] = (u128)D[t][kv_lo * T + n2] * mpow(w2048, (u64)n2 * kv % 2048, q) % q; } }
    // ---- P2: contract n2 (in-tile cols) with W32[n2][ku] = w32^{n2*ku} * psi^{32*n2}
    {
        std::vector<std::vector<u64>> O(NT, std::vector<u64>(T * T, 0));
        for (uint32_t t = 0; t < NT; ++t)
            for (uint32_t r = 0; r < 32; ++r) for (uint32_t ku = 0; ku < 32; ++ku) {
                u128 acc = 0;
                for (uint32_t n2 = 0; n2 < 32; ++n2) {
                    u64 c = (u128)mpow(w32, (u64)n2 * ku % 32, q) * mpow(psi, 32ull * n2 % (2 * N), q) % q;
                    acc += (u128)D[t][r * T + n2] * c % q;
                }
                O[t][r * T + ku] = (u64)(acc % q);
            }
        D = O;
    }
    // ---- Tb: twiddle w_N^{n1 * (kv + 64*ku)} * psi^{n1}
    for (uint32_t t = 0; t < NT; ++t) { uint32_t n1 = t >> 1, kv_hi = t & 1;
        for (uint32_t kv_lo = 0; kv_lo < 32; ++kv_lo) for (uint32_t ku = 0; ku < 32; ++ku) {
            uint32_t kb = (kv_lo + 32 * kv_hi) + 64 * ku;
            u64 c = (u128)mpow(w, (u64)n1 * kb % N, q) * mpow(psi, n1, q) % q;
            D[t][kv_lo * T + ku] = (u128)D[t][kv_lo * T + ku] * c % q; } }
    // ---- A2A: corner turn n1 <-> kv_lo, within each kv_hi group. new tile = (kv_lo, kv_hi)
    {
        std::vector<std::vector<u64>> O(NT, std::vector<u64>(T * T));
        for (uint32_t n1 = 0; n1 < 32; ++n1) for (uint32_t kv_hi = 0; kv_hi < 2; ++kv_hi)
            for (uint32_t kv_lo = 0; kv_lo < 32; ++kv_lo) for (uint32_t ku = 0; ku < 32; ++ku)
                O[kv_lo * 2 + kv_hi][n1 * T + ku] = D[n1 * 2 + kv_hi][kv_lo * T + ku];
        D = O;
    }
    // ---- P3: contract n1 (in-tile rows) with W32[ka][n1] = w32^{n1*ka}
    {
        std::vector<std::vector<u64>> O(NT, std::vector<u64>(T * T, 0));
        for (uint32_t t = 0; t < NT; ++t)
            for (uint32_t ka = 0; ka < 32; ++ka) for (uint32_t ku = 0; ku < 32; ++ku) {
                u128 acc = 0;
                for (uint32_t n1 = 0; n1 < 32; ++n1)
                    acc += (u128)D[t][n1 * T + ku] * mpow(w32, (u64)n1 * ka % 32, q) % q;
                O[t][ka * T + ku] = (u64)(acc % q);
            }
        D = O;
    }

    // ---- compare: tile (kv_lo, kv_hi) element [ka][ku]  <->  k = kv_lo + 32 kv_hi + 64 ku + 2048 ka
    uint64_t bad = 0;
    for (uint32_t kv_lo = 0; kv_lo < 32; ++kv_lo) for (uint32_t kv_hi = 0; kv_hi < 2; ++kv_hi)
        for (uint32_t ka = 0; ka < 32; ++ka) for (uint32_t ku = 0; ku < 32; ++ku) {
            uint32_t k = kv_lo + 32 * kv_hi + 64 * ku + 2048 * ka;
            if (D[kv_lo * 2 + kv_hi][ka * T + ku] != ref[k]) ++bad;
        }
    std::printf("3-level FWD vs reference: %llu / %u mismatches  [%s]\n",
                (unsigned long long)bad, N, bad ? "FAIL" : "PASS");

    // ================= INVERSE: the same passes in reverse, with inverse roots =================
    // x[n] = N^-1 psi^-n sum_k X[k] w^-nk.  Undo P3, A2A, Tb, P2, Ta, P1 in that order.
    const u64 iw = mpow(w, q - 2, q), ipsi = mpow(psi, q - 2, q);
    const u64 iw64 = mpow(w64, q - 2, q), iw32 = mpow(w32, q - 2, q), iw2048 = mpow(w2048, q - 2, q);
    const u64 Ninv = mpow(N, q - 2, q);
    // input = D (the forward output, k-layout)
    // ---- P3': contract ka (in-tile rows) with iW32[n1][ka] = iw32^{n1*ka} -> rows become n1
    {
        std::vector<std::vector<u64>> O(NT, std::vector<u64>(T * T, 0));
        for (uint32_t t = 0; t < NT; ++t)
            for (uint32_t n1 = 0; n1 < 32; ++n1) for (uint32_t ku = 0; ku < 32; ++ku) {
                u128 acc = 0;
                for (uint32_t ka = 0; ka < 32; ++ka)
                    acc += (u128)D[t][ka * T + ku] * mpow(iw32, (u64)n1 * ka % 32, q) % q;
                O[t][n1 * T + ku] = (u64)(acc % q);
            }
        D = O;
    }
    // ---- A2A back: tile (kv_lo,kv_hi)[n1][ku] -> tile (n1,kv_hi)[kv_lo][ku]
    {
        std::vector<std::vector<u64>> O(NT, std::vector<u64>(T * T));
        for (uint32_t kv_lo = 0; kv_lo < 32; ++kv_lo) for (uint32_t kv_hi = 0; kv_hi < 2; ++kv_hi)
            for (uint32_t n1 = 0; n1 < 32; ++n1) for (uint32_t ku = 0; ku < 32; ++ku)
                O[n1 * 2 + kv_hi][kv_lo * T + ku] = D[kv_lo * 2 + kv_hi][n1 * T + ku];
        D = O;
    }
    // ---- Tb': / (w_N^{n1 kb} psi^{n1})
    for (uint32_t t = 0; t < NT; ++t) { uint32_t n1 = t >> 1, kv_hi = t & 1;
        for (uint32_t kv_lo = 0; kv_lo < 32; ++kv_lo) for (uint32_t ku = 0; ku < 32; ++ku) {
            uint32_t kb = (kv_lo + 32 * kv_hi) + 64 * ku;
            u64 c = (u128)mpow(iw, (u64)n1 * kb % N, q) * mpow(ipsi, n1, q) % q;
            D[t][kv_lo * T + ku] = (u128)D[t][kv_lo * T + ku] * c % q; } }
    // ---- P2': contract ku (in-tile cols) with iW32[ku][n2] = iw32^{n2*ku} * ipsi^{32*n2}
    {
        std::vector<std::vector<u64>> O(NT, std::vector<u64>(T * T, 0));
        for (uint32_t t = 0; t < NT; ++t)
            for (uint32_t r = 0; r < 32; ++r) for (uint32_t n2 = 0; n2 < 32; ++n2) {
                u128 acc = 0;
                for (uint32_t ku = 0; ku < 32; ++ku) {
                    u64 c = (u128)mpow(iw32, (u64)n2 * ku % 32, q) * mpow(ipsi, 32ull * n2 % (2 * N), q) % q;
                    acc += (u128)D[t][r * T + ku] * c % q;
                }
                O[t][r * T + n2] = (u64)(acc % q);
            }
        D = O;
    }
    // ---- Ta': / w_2048^{n2 kv}
    for (uint32_t t = 0; t < NT; ++t) { uint32_t kv_hi = t & 1;
        for (uint32_t kv_lo = 0; kv_lo < 32; ++kv_lo) for (uint32_t n2 = 0; n2 < 32; ++n2) {
            uint32_t kv = kv_lo + 32 * kv_hi;
            D[t][kv_lo * T + n2] = (u128)D[t][kv_lo * T + n2] * mpow(iw2048, (u64)n2 * kv % 2048, q) % q; } }
    // ---- P1': contract kv (blocked) with iW64[n3][kv] = iw64^{kv*n3} * ipsi^{1024*n3} * Ninv
    {
        std::vector<std::vector<u64>> O(NT, std::vector<u64>(T * T, 0));
        for (uint32_t n1 = 0; n1 < 32; ++n1)
            for (uint32_t n3 = 0; n3 < 64; ++n3) {
                auto& out = O[n1 * 2 + (n3 >> 5)];
                for (uint32_t kv = 0; kv < 64; ++kv) {
                    u64 c = (u128)mpow(iw64, (u64)kv * n3 % 64, q) * mpow(ipsi, 1024ull * n3 % (2 * N), q) % q;
                    c = (u128)c * Ninv % q;
                    const auto& in = D[n1 * 2 + (kv >> 5)];
                    for (uint32_t n2 = 0; n2 < 32; ++n2)
                        out[(n3 & 31) * T + n2] = (out[(n3 & 31) * T + n2] + (u128)c * in[(kv & 31) * T + n2]) % q;
                }
            }
        D = O;
    }
    uint64_t badi = 0;
    for (uint32_t n1 = 0; n1 < 32; ++n1) for (uint32_t n3 = 0; n3 < 64; ++n3) for (uint32_t n2 = 0; n2 < 32; ++n2)
        if (D[n1 * 2 + (n3 >> 5)][(n3 & 31) * T + n2] != x[n1 + 32 * n2 + 1024 * n3]) ++badi;
    std::printf("3-level INV round trip     : %llu / %u mismatches  [%s]\n",
                (unsigned long long)badi, N, badi ? "FAIL" : "PASS");
    return (bad || badi) ? 1 : 0;
}
