// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-side OnionPIR CLIENT — everything a querier does, fast enough to be interactive.
//
// `gen_query65536.py` needs ~73 s per index, almost all of it in the 7 fold RGSWs: the reference does
// negacyclic multiplication over an 84-bit composite modulus in Python bigints. That is a property of
// the reference, not of the client's actual work. Doing the same arithmetic in RNS with the host NTT
// takes ~0.1 s, which is what makes "type an index, get the row" possible.
//
// What is index-INDEPENDENT and therefore loaded once (from the key bundle the Python generator
// already writes): the secret key s and the Galois keys. What this file produces per index:
//   * the packed initial-dimension query ciphertext (one-hot at c0, Delta-scaled, pre-divided by 2^m)
//   * the 7 fold RGSWs (the binary hypercube coordinates of the requested row)
//
// Conventions are the reference's, deliberately and exactly (onionpir_ref.py):
//   ct = (c0, c1) with c0 + c1*s = Delta*m + e   (mod q = prod primes),  Delta = q/t
//   RGSW(m): 2*ell rows x 2 cols; rows 0..ell-1 carry m*B^k in col 0, rows ell..2ell-1 in col 1
//   gadget digits are BALANCED in [-B/2, B/2), which is why B^ell >= 2q is required
//   external product: decompose BOTH components, dot the 2*ell digits with the 2*ell rows
//
// The NTT is copied from encode_db.cpp, whose output is verified bit-exact against the device
// (test_db_encode_check, 0/36 864) — so the same routine can also prepare key material for upload
// without a device round-trip per polynomial.

#pragma once
#include <array>
#include <cstdint>
#include <random>
#include <thread>
#include <stdexcept>
#include <vector>
#include "common/montgomery.hpp"

namespace pirclient {

using u32 = uint32_t; using u64 = uint64_t;
using u128 = unsigned __int128; using i128 = __int128;
using Poly = std::vector<u128>;                    // n coefficients in [0, q)

inline u64 modpow(u64 b, u64 e, u64 m) { u64 r = 1; b %= m;
    while (e) { if (e & 1) r = (u128)r * b % m; b = (u128)b * b % m; e >>= 1; } return r; }
inline u64 prim_root(u64 q) { u64 phi = q - 1, x = phi; std::vector<u64> f;
    for (u64 d = 2; d * d <= x; ++d) if (x % d == 0) { f.push_back(d); while (x % d == 0) x /= d; }
    if (x > 1) f.push_back(x);
    for (u64 g = 2; g < q; ++g) { bool ok = true; for (u64 p : f) if (modpow(g, phi / p, q) == 1) ok = false;
        if (ok) return g; } throw std::runtime_error("no primitive root"); }

// ---- per-prime negacyclic NTT.  Forward is encode_db.cpp's, i.e. device-identical. ----------------
struct Chan {
    u32 q = 0, n = 0; u64 ninv = 0;
    std::vector<u32> psi_pow, ipsi_pow, w_pow, iw_pow;
    // Shoup precomputations for ntt_shoup() below: for each constant multiplier c,
    // c_shoup = floor(c * 2^32 / q).  Filled by make_chan; unused by ntt()/intt().
    std::vector<u32> psi_shoup, w_shoup; bool shoup_ok = false;
};

inline Chan make_chan(u32 q, u32 n) {
    Chan c; c.q = q; c.n = n;
    const u64 g = prim_root(q);
    const u64 psi = modpow(g, (q - 1) / (2 * n), q), ipsi = modpow(psi, q - 2, q);
    const u64 w = (u128)psi * psi % q, iw = modpow(w, q - 2, q);
    c.ninv = modpow(n, q - 2, q);
    c.psi_pow.resize(n); c.ipsi_pow.resize(n); c.w_pow.resize(n / 2); c.iw_pow.resize(n / 2);
    u64 p = 1, ip = 1; for (u32 i = 0; i < n; ++i) { c.psi_pow[i] = (u32)p; c.ipsi_pow[i] = (u32)ip;
                                                    p = (u128)p * psi % q; ip = (u128)ip * ipsi % q; }
    u64 x = 1, ix = 1; for (u32 i = 0; i < n / 2; ++i) { c.w_pow[i] = (u32)x; c.iw_pow[i] = (u32)ix;
                                                        x = (u128)x * w % q; ix = (u128)ix * iw % q; }
    // Shoup tables for ntt_shoup().  Valid only while q < 2^31, so that w*x - hi*q stays below 2^32
    // and the wrapping 32-bit arithmetic in mul_shoup is exact; otherwise leave shoup_ok false and
    // callers fall back to ntt().
    c.shoup_ok = (q < (1u << 31));
    if (c.shoup_ok) {
        auto sh = [&](u32 v) { return (u32)((((u128)v) << 32) / q); };
        c.psi_shoup.resize(n); for (u32 i = 0; i < n; ++i) c.psi_shoup[i] = sh(c.psi_pow[i]);
        c.w_shoup.resize(n / 2); for (u32 i = 0; i < n / 2; ++i) c.w_shoup[i] = sh(c.w_pow[i]);
    }
    return c;
}

inline void ntt(const Chan& c, const u32* x, u32* out);   // defined below; ntt_shoup's fallback

// Multiply by a CONSTANT whose Shoup companion cs = floor(c*2^32/q) is precomputed: replaces the
// hardware 64-bit division in the butterfly with a multiply-high and a conditional subtract.
// Requires q < 2^31 and x < q.  Standard Shoup/Harvey modmul.
inline u32 mul_shoup(u32 x, u32 c, u32 cs, u32 q) {
    const u32 hi = (u32)(((u64)cs * x) >> 32);
    const u32 r = c * x - hi * q;                 // wrapping 32-bit; exact, lands in [0, 2q)
    return r >= q ? r - q : r;
}

// Bit-identical to ntt() above -- same pre-twist, same bit-reverse, same DIT order, same output --
// with every `% q` replaced by mul_shoup.  The demo checks it against the device NTT at startup.
inline void ntt_shoup(const Chan& c, const u32* x, u32* out) {
    if (!c.shoup_ok) { ntt(c, x, out); return; }
    const u32 q = c.q, n = c.n;
    for (u32 i = 0, j = 0; i < n; ++i) {
        out[j] = mul_shoup(x[i] % q, c.psi_pow[i], c.psi_shoup[i], q);
        u32 bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; }
    for (u32 len = 1; len < n; len <<= 1) { const u32 step = (n / 2) / len;
        for (u32 i = 0; i < n; i += (len << 1)) for (u32 k = 0; k < len; ++k) {
            const u32 ti = k * step;
            const u32 u = out[i + k], v = mul_shoup(out[i + k + len], c.w_pow[ti], c.w_shoup[ti], q);
            out[i + k] = u + v >= q ? u + v - q : u + v;
            out[i + k + len] = u >= v ? u - v : u + q - v; } }
}

// X[k] = sum_i x[i] psi^i w^(ik) mod q — pre-twist, bit-reverse, DIT; natural order out
inline void ntt(const Chan& c, const u32* x, u32* out) {
    const u32 q = c.q, n = c.n;
    for (u32 i = 0, j = 0; i < n; ++i) {
        out[j] = (u32)((u64)x[i] * c.psi_pow[i] % q);
        u32 bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; }
    for (u32 len = 1; len < n; len <<= 1) { const u32 step = (n / 2) / len;
        for (u32 i = 0; i < n; i += (len << 1)) for (u32 k = 0; k < len; ++k) {
            const u64 tw = c.w_pow[(u64)k * step];
            const u32 u = out[i + k], v = (u32)((u64)out[i + k + len] * tw % q);
            out[i + k] = u + v >= q ? u + v - q : u + v;
            out[i + k + len] = u >= v ? u - v : u + q - v; } }
}
// exact inverse of ntt(): bit-reverse, DIT with w^-1, then scale by n^-1 and post-twist by psi^-i
inline void intt(const Chan& c, const u32* X, u32* out) {
    const u32 q = c.q, n = c.n;
    for (u32 i = 0, j = 0; i < n; ++i) { out[j] = X[i];
        u32 bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; }
    for (u32 len = 1; len < n; len <<= 1) { const u32 step = (n / 2) / len;
        for (u32 i = 0; i < n; i += (len << 1)) for (u32 k = 0; k < len; ++k) {
            const u64 tw = c.iw_pow[(u64)k * step];
            const u32 u = out[i + k], v = (u32)((u64)out[i + k + len] * tw % q);
            out[i + k] = u + v >= q ? u + v - q : u + v;
            out[i + k + len] = u >= v ? u - v : u + q - v; } }
    for (u32 i = 0; i < n; ++i)
        out[i] = (u32)((u128)out[i] * c.ninv % q * c.ipsi_pow[i] % q);
}

using RnsNtt = std::vector<std::vector<u32>>;      // [channel][slot], evaluation domain

// ---- the client ---------------------------------------------------------------------------------
struct Client {
    u32 n = 2048, t = 257, ell = 8, K = 3, m_exp = 9, N0 = 512, nfold = 7, rest = 128;
    u64 Bgad = 2048;
    std::vector<u32> primes; std::vector<Chan> ch;
    u128 q = 1, delta = 0;
    std::vector<u64> garner_inv;                   // Garner chain constants
    std::vector<int32_t> s;                        // the secret key (loaded; fixes the Galois keys)
    RnsNtt sN;                                     // ...pre-transformed (every encryption needs a*s)
    std::vector<u128> gvec;                        // B^k mod q
    std::vector<std::array<Poly, 2>> gsw_key;      // RGSW(s), the bridge for rlwe_to_rgsw
    std::vector<std::array<RnsNtt, 2>> gsw_ntt;    // ...pre-transformed, since every row is reused
    std::mt19937_64 rng{0xC1E27ULL};

    struct Query {
        std::array<Poly, 2> ct;                                   // packed initial-dimension query
        std::vector<std::vector<std::array<Poly, 2>>> folds;      // [fold][row 0..2ell-1][col]
        u32 c0 = 0; std::vector<u32> bits;
    };

    Client(const std::vector<u32>& pr, const std::vector<int32_t>& sec, u32 t_, u32 ell_, u64 B_,
           u32 N0_, u32 nfold_, u32 n_ = 2048)
        : n(n_), t(t_), ell(ell_), K((u32)pr.size()), N0(N0_), nfold(nfold_), Bgad(B_),
          primes(pr), s(sec) {
        if (K != 3) throw std::runtime_error("client assumes K=3 (Garner chain is unrolled)");
        rest = 1u << nfold; m_exp = 0; while ((1u << m_exp) < N0) ++m_exp;
        for (auto p : primes) { ch.push_back(make_chan(p, n)); q *= p; }
        delta = q / t;
        garner_inv = {mont::modular_inverse(primes[0] % primes[1], primes[1]),
                      mont::modular_inverse((u64)((u128)primes[0] * primes[1] % primes[2]), primes[2])};
        gvec.resize(ell); { u128 g = 1; for (u32 k = 0; k < ell; ++k) { gvec[k] = g; g = g * Bgad % q; } }
        sN = fwd_small(s);
        make_gsw_key();
    }

    // ---- composite-modulus helpers ----
    u128 lift(u32 r0, u32 r1, u32 r2) const {
        const u64 p0 = primes[0], p1 = primes[1], p2 = primes[2];
        const u64 t1 = (u64)((u128)((r1 + p1 - r0 % p1) % p1) * garner_inv[0] % p1);
        u128 x = (u128)r0 + (u128)p0 * t1;
        const u64 t2 = (u64)((u128)((r2 + p2 - (u64)(x % p2)) % p2) * garner_inv[1] % p2);
        return x + (u128)p0 * p1 * t2; }
    static u128 addm(u128 a, u128 b, u128 q) { a += b; return a >= q ? a - q : a; }
    static u128 subm(u128 a, u128 b, u128 q) { return a >= b ? a - b : a + q - b; }
    Poly padd(const Poly& a, const Poly& b) const { Poly o(n);
        for (u32 i = 0; i < n; ++i) o[i] = addm(a[i], b[i], q); return o; }
    Poly psub(const Poly& a, const Poly& b) const { Poly o(n);
        for (u32 i = 0; i < n; ++i) o[i] = subm(a[i], b[i], q); return o; }

    // ---- transforms ----
    RnsNtt fwd(const Poly& a) const { RnsNtt o(K, std::vector<u32>(n)); std::vector<u32> r(n);
        for (u32 c = 0; c < K; ++c) { for (u32 i = 0; i < n; ++i) r[i] = (u32)(a[i] % primes[c]);
                                      ntt(ch[c], r.data(), o[c].data()); } return o; }
    RnsNtt fwd_small(const std::vector<int32_t>& a) const { RnsNtt o(K, std::vector<u32>(n)); std::vector<u32> r(n);
        for (u32 c = 0; c < K; ++c) { const u32 p = primes[c];
            for (u32 i = 0; i < n; ++i) r[i] = a[i] >= 0 ? (u32)a[i] : p - (u32)(-a[i]);
            ntt(ch[c], r.data(), o[c].data()); } return o; }
    // per-prime evaluation-domain accumulator -> coefficient-domain Poly mod q
    Poly inv_lift(const std::vector<std::vector<u64>>& acc) const {
        std::vector<std::vector<u32>> co(K, std::vector<u32>(n)); std::vector<u32> in(n);
        for (u32 c = 0; c < K; ++c) { for (u32 i = 0; i < n; ++i) in[i] = (u32)(acc[c][i] % primes[c]);
                                      intt(ch[c], in.data(), co[c].data()); }
        Poly o(n); for (u32 i = 0; i < n; ++i) o[i] = lift(co[0][i], co[1][i], co[2][i]); return o; }
    Poly mul(const RnsNtt& A, const RnsNtt& Bv) const {
        std::vector<std::vector<u64>> acc(K, std::vector<u64>(n));
        for (u32 c = 0; c < K; ++c) { const u64 p = primes[c];
            for (u32 i = 0; i < n; ++i) acc[c][i] = (u64)A[c][i] * Bv[c][i] % p; }
        return inv_lift(acc); }
    Poly mul(const Poly& a, const Poly& b) const { return mul(fwd(a), fwd(b)); }

    // ---- sampling.  Every sampler takes its generator explicitly so the 7 fold RGSWs — which are
    // independent given RGSW(s) — can be built on separate threads without sharing RNG state. ----
    u128 urand(std::mt19937_64& r) { const u128 v = ((u128)r() << 64) | r(); return v % q; }
    u128 urand() { return urand(rng); }
    Poly rand_poly(std::mt19937_64& r) { Poly o(n); for (auto& v : o) v = urand(r); return o; }
    Poly rand_poly() { return rand_poly(rng); }
    std::vector<int32_t> ternary() { std::vector<int32_t> o(n);
        for (auto& v : o) v = (int32_t)(rng() % 3) - 1; return o; }
    Poly err_poly(std::mt19937_64& r) { Poly o(n); for (auto& v : o) { const int e = (int)(r() % 3) - 1;
        v = e >= 0 ? (u128)e : q - (u128)(-e); } return o; }
    Poly err_poly() { return err_poly(rng); }
    Poly s_poly() const { Poly o(n);
        for (u32 i = 0; i < n; ++i) o[i] = s[i] >= 0 ? (u128)s[i] : q - (u128)(-s[i]); return o; }

    // ---- BFV ----
    // c0 = -a*s + e (+ Delta*m or + m), c1 = a
    std::array<Poly, 2> enc(const Poly& m, bool scaled) { return enc(m, scaled, rng); }
    std::array<Poly, 2> enc(const Poly& m, bool scaled, std::mt19937_64& r) {
        auto a = rand_poly(r);
        auto as = mul(fwd(a), sN);
        Poly c0(n);
        for (u32 i = 0; i < n; ++i) c0[i] = subm(0, as[i], q);
        c0 = padd(c0, err_poly(r));
        if (scaled) { for (u32 i = 0; i < n; ++i) c0[i] = addm(c0[i], delta * (m[i] % t) % q, q); }
        else        { c0 = padd(c0, m); }
        return {c0, a}; }
    std::vector<u32> dec(const std::array<Poly, 2>& ct) const {          // for the self-tests
        auto ph = padd(ct[0], mul(ct[1], s_poly()));
        std::vector<u32> m(n); const u128 h = q / 2;
        for (u32 i = 0; i < n; ++i) { i128 v = (i128)ph[i]; if ((u128)v > h) v -= (i128)q;
            const bool neg = v < 0; const i128 av = neg ? -v : v;
            int64_t r = (int64_t)(((u128)av + delta / 2) / delta); if (neg) r = -r;
            m[i] = (u32)(((r % (int64_t)t) + t) % t); }
        return m; }
    Poly phase(const std::array<Poly, 2>& ct) const { return padd(ct[0], mul(ct[1], s_poly())); }

    // ---- gadget ----
    // balanced base-B digits: Sum_k d_k B^k = v (mod q), each d_k in [-B/2, B/2)
    std::vector<std::vector<int32_t>> decompose(const Poly& a) const {
        const u64 half = Bgad / 2, mask = Bgad - 1; u32 sh = 0; while ((1ull << sh) != Bgad) ++sh;
        std::vector<std::vector<int32_t>> d(ell, std::vector<int32_t>(n));
        for (u32 i = 0; i < n; ++i) { u128 x = a[i];
            for (u32 k = 0; k < ell; ++k) { int64_t dk = (int64_t)((u64)x & mask);
                if ((u64)dk >= half) dk -= (int64_t)Bgad;
                d[k][i] = (int32_t)dk;
                x = (dk >= 0) ? (x - (u128)dk) : (x + (u128)(-dk));
                x >>= sh; } }
        return d; }

    // RGSW(m) [X] RLWE(mu) -> RLWE(m*mu); C is pre-transformed (2ell rows x 2 cols)
    std::array<Poly, 2> extprod(const std::vector<std::array<RnsNtt, 2>>& C,
                                const std::array<Poly, 2>& d) const {
        auto u = decompose(d[0]); auto w = decompose(d[1]);
        std::vector<std::vector<u64>> acc0(K, std::vector<u64>(n, 0)), acc1(K, std::vector<u64>(n, 0));
        std::vector<u32> r(n), R(n);
        for (u32 row = 0; row < 2 * ell; ++row) {
            const auto& dig = (row < ell) ? u[row] : w[row - ell];
            for (u32 c = 0; c < K; ++c) { const u32 p = primes[c];
                for (u32 i = 0; i < n; ++i) r[i] = dig[i] >= 0 ? (u32)dig[i] : p - (u32)(-dig[i]);
                ntt(ch[c], r.data(), R.data());
                for (u32 i = 0; i < n; ++i) {
                    acc0[c][i] = (acc0[c][i] + (u64)R[i] * C[row][0][c][i]) % p;
                    acc1[c][i] = (acc1[c][i] + (u64)R[i] * C[row][1][c][i]) % p; } } }
        return {inv_lift(acc0), inv_lift(acc1)}; }

    // ---- key material ----
    void make_gsw_key() {                                   // RGSW(s), once per session
        gsw_key = rgsw_encrypt_small(s);
        gsw_ntt.resize(gsw_key.size());
        for (size_t r = 0; r < gsw_key.size(); ++r)
            gsw_ntt[r] = {fwd(gsw_key[r][0]), fwd(gsw_key[r][1])}; }
    std::vector<std::array<Poly, 2>> rgsw_encrypt_small(const std::vector<int32_t>& m) {
        std::vector<std::array<Poly, 2>> rows;
        for (u32 col = 0; col < 2; ++col) for (u32 k = 0; k < ell; ++k) {
            auto ct = enc(Poly(n, 0), false);                // RLWE(0)
            Poly mg(n); for (u32 i = 0; i < n; ++i) {
                const u128 g = gvec[k];
                mg[i] = m[i] >= 0 ? (u128)m[i] * g % q : q - (u128)(-m[i]) * g % q; if (mg[i] == q) mg[i] = 0; }
            if (col == 0) rows.push_back({padd(ct[0], mg), ct[1]});
            else          rows.push_back({ct[0], padd(ct[1], mg)}); }
        return rows; }

    // ---- the per-index query -------------------------------------------------------------------
    Query make_query(u32 idx) {
        Query Q; Q.c0 = idx / rest;
        { u32 r = idx % rest; for (u32 j = 0; j < nfold; ++j) Q.bits.push_back((r >> (nfold - 1 - j)) & 1u); }
        // packed one-hot at c0, pre-divided by 2^m_exp so the expansion tree's x2^m lands on 1
        Poly m(n, 0); m[Q.c0] = (u128)mont::modular_inverse((1ull << m_exp) % t, t);
        Q.ct = enc(m, true);
        // one RGSW per fold bit: ell scaleless message rows, then the s-scaled rows via RGSW(s).
        // The folds are independent, and the external products dominate, so run them in parallel.
        Q.folds.resize(nfold);
        std::vector<uint64_t> seeds(nfold); for (auto& sd : seeds) sd = rng();
        std::vector<std::thread> th;
        for (u32 f = 0; f < nfold; ++f) th.emplace_back([&, f]{
            std::mt19937_64 r(seeds[f]);
            std::vector<std::array<Poly, 2>> msg;
            for (u32 k = 0; k < ell; ++k) { Poly mm(n, 0); mm[0] = Q.bits[f] ? gvec[k] : 0;
                                            msg.push_back(enc(mm, false, r)); }
            std::vector<std::array<Poly, 2>> rows = msg;                       // top ell: as-is
            for (u32 k = 0; k < ell; ++k) rows.push_back(extprod(gsw_ntt, msg[k]));   // bottom ell: s*
            Q.folds[f] = std::move(rows); });
        for (auto& t : th) t.join();
        return Q; }

    // residues of one polynomial for channel c, as the device wants them
    std::vector<u32> res(const Poly& a, u32 c) const { std::vector<u32> o(n);
        for (u32 i = 0; i < n; ++i) o[i] = (u32)(a[i] % primes[c]); return o; }
};

}  // namespace pirclient
