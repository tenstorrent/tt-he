// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_Tb001b — a CORRECT BFV ct_mul at the exp_Tb001 ring, verified by DECRYPTION.
//
// WHY THIS EXISTS. exp_Tb001 (host/experiments/exp_tb001.cpp) is a faithful *cost model* of the CROSS Set-B
// multiply: right ring, right limb counts, right kernel sequence, bit-exact against a CPU oracle that
// computes the same arithmetic. What it is NOT is a semantically correct BFV multiply --
//
//   * its rescale is a CKKS-shaped drop-prime, not BFV's  ⌊t·e/Q⌉;
//   * its evaluation key is random, so relinearisation moves noise but decrypts to nothing;
//   * "device == oracle" says the device computes what the host meant, never that what the host
//     meant is a ciphertext of the product.
//
// exp_Tb001b closes that gap. Same ring (N=8192), same kernel tree (kernels/experiments/exp_tb001/), same
// measurement stack -- but a real secret key, real encryption, BFV's own rescale, and a correctness
// test that is END-TO-END SEMANTIC: encrypt two random plaintext polynomials, multiply, decrypt, and
// compare against the negacyclic product in R_t. Nothing but a correct scheme passes that.
//
// The reference implementation it follows is host/lib/he/he_api.cpp, which is validated at N=1024, LQ=3 by
// host/examples/he_demo (decrypt(6 (x) 7) == 42). This is the same scheme at the CROSS ring, with two
// changes forced by the scale:
//
//   * FULL-POLYNOMIAL messages. he_api encrypts one integer into coefficient 0 and decrypts only that
//     coefficient. A constant polynomial is a weak test: it cannot see a negacyclic wrap error, which
//     is exactly the class of bug a ring implementation has. Here m is a uniform element of R_t and
//     all N coefficients are checked.
//   * EXACT BIG INTEGERS. Decryption CRT-reconstructs modulo Q, and at LQ=8 that is a 224-bit value
//     whose intermediates exceed 128 bits. he_api's u128 arithmetic is correct only because LQ=3 puts
//     Q at ~84 bits. u512 below is the minimum that makes the reconstruction exact rather than lucky.
//
// STATUS: this file currently implements and verifies the host side -- parameters, keygen, encrypt,
// decrypt, and HE add. The multiply is the next step; see docs/he/exp_tb001b.md for the plan and for
// why BFV's multiply costs strictly more on device than exp_Tb001's CKKS-shaped one.
//
//   ./build/exp_tb001b [seed]
//
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "common/montgomery.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {

using u128 = unsigned __int128;

// ---- device-side geometry, identical to exp_Tb001 (the kernels are shared, so these must match) ----
constexpr uint32_t T = 32, NB = 4, N1 = 32, N2 = 256, ELEMS = 1024;
constexpr uint32_t NBLK = N2 / T;                 // 8 blocks of the four-step NTT
constexpr uint32_t I8 = ELEMS, I32 = ELEMS * 4;
const uint64_t POW[4] = {1, 128, 16384, 2097152};

template <typename E> std::vector<E> tilize32(const std::vector<E>& in) { std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = in[r * T + c]; return o; }
template <typename E> std::vector<E> untilize32(const std::vector<E>& in) { std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[r * T + c] = in[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)]; return o; }
int8_t limb(uint32_t v, uint32_t p) { return (int8_t)((v >> (7 * p)) & 127); }

// ---- the minimal environment the exp_Tb001 device blocks below expect ----
// They are lifted VERBATIM so they cannot drift from the validated versions; the names they reference
// for profiling and trace capture are provided here as stubs. Performance instrumentation comes later
// -- this stage is about correctness, and a stub keeps the copied code byte-identical to its source.
namespace prof {
  double g_build=0, g_wr=0, g_rd=0, g_enq=0, g_fin=0;
  long n_build=0, n_wr=0, n_rd=0, n_enq=0, n_fin=0;
  using hrc = std::chrono::high_resolution_clock;
  inline double msd(hrc::time_point t){ return std::chrono::duration<double,std::milli>(hrc::now()-t).count(); }
}
namespace Dp = tt::tt_metal::distributed;
template<class...A> void t_wr (A&&...a){ auto t=prof::hrc::now(); Dp::EnqueueWriteMeshBuffer(std::forward<A>(a)...); prof::g_wr+=prof::msd(t); ++prof::n_wr; }
template<class...A> void t_rd (A&&...a){ auto t=prof::hrc::now(); Dp::EnqueueReadMeshBuffer (std::forward<A>(a)...); prof::g_rd+=prof::msd(t); ++prof::n_rd; }
template<class...A> void t_enq(A&&...a){ auto t=prof::hrc::now(); Dp::EnqueueMeshWorkload   (std::forward<A>(a)...); prof::g_enq+=prof::msd(t); ++prof::n_enq; }
template<class Q>   void t_fin(Q&& cq) { auto t=prof::hrc::now(); Dp::Finish(cq); prof::g_fin+=prof::msd(t); ++prof::n_fin; }

constexpr uint32_t N = 8192;      // ring degree -- the CROSS Set-B ring, same as exp_Tb001
constexpr uint32_t LQ = 8;        // ciphertext base Q: 8 x 28-bit  ~ 224-bit
constexpr uint32_t LP = 4;        // keyswitch base P: 4 x 28-bit   (dnum=2, as exp_Tb001)
constexpr uint32_t LS = 9;        // BEHZ auxiliary base S: must exceed Q for the un-reduced tensor
constexpr uint32_t T_PLAIN = 12289;   // plaintext modulus t (same as host/lib/he)

// ---------------------------------------------------------------- small utilities
uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) {
    u128 r = 1, x = b % m;
    while (e) { if (e & 1) r = r * x % m; x = x * x % m; e >>= 1; }
    return (uint64_t)r;
}
uint64_t modinv(uint64_t a, uint64_t m) { return modpow(a % m, m - 2, m); }
bool is_prime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) return false;
    return true;
}
// cnt distinct 28-bit primes p = k*2N+1 (so a primitive 2N-th root exists -> negacyclic NTT)
std::vector<uint64_t> ntt_primes(uint32_t cnt) {
    std::vector<uint64_t> out;
    for (uint64_t k = ((1ull << 28) - 1) / (2 * N); out.size() < cnt && k > 0; --k) {
        uint64_t p = k * 2 * N + 1;
        if (p < (1ull << 28) && is_prime(p)) out.push_back(p);
    }
    return out;
}
uint64_t prim_root(uint64_t q) {
    uint64_t phi = q - 1, n = phi; std::vector<uint64_t> f;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) { f.push_back(d); while (n % d == 0) n /= d; }
    if (n > 1) f.push_back(n);
    for (uint64_t g = 2; g < q; ++g) {
        bool ok = true;
        for (uint64_t p : f) if (modpow(g, phi / p, q) == 1) ok = false;
        if (ok) return g;
    }
    std::abort();
}

// ---------------------------------------------------------------- u512: exact big unsigned
// Fixed 512-bit little-endian. Only what the scheme needs: the CRT reconstruction of a 224-bit Q,
// t*v products (~238-bit), and division by Q. Deliberately not a general bignum -- every operation
// here is one the correctness proof depends on, so each is small enough to read and check.
struct u512 {
    uint64_t w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    u512() = default;
    explicit u512(uint64_t v) { w[0] = v; }
    static u512 from_u128(u128 v) { u512 r; r.w[0] = (uint64_t)v; r.w[1] = (uint64_t)(v >> 64); return r; }
    bool is_zero() const { for (int i = 0; i < 8; ++i) if (w[i]) return false; return true; }
};
int cmp(const u512& a, const u512& b) {
    for (int i = 7; i >= 0; --i) { if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1; }
    return 0;
}
u512 add(const u512& a, const u512& b) {
    u512 r; u128 c = 0;
    for (int i = 0; i < 8; ++i) { c += (u128)a.w[i] + b.w[i]; r.w[i] = (uint64_t)c; c >>= 64; }
    return r;
}
u512 sub(const u512& a, const u512& b) {          // a >= b assumed
    u512 r; __int128 br = 0;
    for (int i = 0; i < 8; ++i) {
        __int128 d = (__int128)a.w[i] - b.w[i] - br;
        if (d < 0) { d += ((__int128)1 << 64); br = 1; } else br = 0;
        r.w[i] = (uint64_t)d;
    }
    return r;
}
u512 mul_small(const u512& a, uint64_t m) {        // a * m, no overflow past 512 bits by construction
    u512 r; u128 c = 0;
    for (int i = 0; i < 8; ++i) { c += (u128)a.w[i] * m; r.w[i] = (uint64_t)c; c >>= 64; }
    return r;
}
u512 shl1(const u512& a) {
    u512 r; uint64_t carry = 0;
    for (int i = 0; i < 8; ++i) { r.w[i] = (a.w[i] << 1) | carry; carry = a.w[i] >> 63; }
    return r;
}
int bitlen(const u512& a) {
    for (int i = 7; i >= 0; --i) if (a.w[i]) { int b = 63; while (b >= 0 && !((a.w[i] >> b) & 1)) --b; return i * 64 + b + 1; }
    return 0;
}
// floor(a / d) and a mod d, by shift-and-subtract. Cost is O(bitlen(a) - bitlen(d)), which is why the
// callers below keep the quotient small where they can.
void divmod(const u512& a, const u512& d, u512& q, u512& rem) {
    q = u512(); rem = u512();
    for (int i = bitlen(a) - 1; i >= 0; --i) {
        rem = shl1(rem);
        rem.w[0] |= (a.w[i / 64] >> (i % 64)) & 1ull;
        if (cmp(rem, d) >= 0) { rem = sub(rem, d); q.w[i / 64] |= 1ull << (i % 64); }
    }
}
uint64_t to_u64(const u512& a) { return a.w[0]; }
u512 pow2(uint32_t k) { u512 r; r.w[k / 64] = 1ull << (k % 64); return r; }
u512 shr(const u512& a, uint32_t k) {                 // logical right shift, for the s>>u slice
    u512 r; uint32_t wo = k / 64, bo = k % 64;
    for (uint32_t i = 0; i + wo < 8; ++i) {
        uint64_t lo = a.w[i + wo] >> bo;
        uint64_t hi = (bo && i + wo + 1 < 8) ? (a.w[i + wo + 1] << (64 - bo)) : 0;
        r.w[i] = lo | hi;
    }
    return r;
}
// a mod d for a SMALL d (< 2^32). Word-wise, exact, O(8) -- the general divmod above is O(bitlen)
// and would dominate everything if used per coefficient.
uint64_t mod_small(const u512& a, uint64_t d) {
    u128 r = 0;
    for (int i = 7; i >= 0; --i) r = ((r << 64) | a.w[i]) % d;
    return (uint64_t)r;
}

// ---------------------------------------------------------------- negacyclic NTT, per prime
// R_q = Z_q[x]/(x^N + 1). psi is a primitive 2N-th root, so twisting by psi^i turns the negacyclic
// product into a cyclic one and the transform is the ordinary length-N NTT.
struct NTTCtx {
    uint64_t q, psi, psi_inv, n_inv;
    std::vector<uint64_t> pw, ipw;                 // psi^i and psi^-i
    std::vector<uint64_t> w, iw;                   // bit-reversed roots
    explicit NTTCtx(uint64_t q_) : q(q_) {
        uint64_t g = prim_root(q);
        psi = modpow(g, (q - 1) / (2 * N), q);
        psi_inv = modinv(psi, q);
        n_inv = modinv(N, q);
        pw.resize(N); ipw.resize(N); pw[0] = ipw[0] = 1;
        for (uint32_t i = 1; i < N; ++i) { pw[i] = (u128)pw[i - 1] * psi % q; ipw[i] = (u128)ipw[i - 1] * psi_inv % q; }
        uint64_t wn = (u128)psi * psi % q, wni = modinv(wn, q);
        w.resize(N); iw.resize(N);
        for (uint32_t i = 0; i < N; ++i) { w[i] = modpow(wn, i, q); iw[i] = modpow(wni, i, q); }
    }
    void transform(std::vector<uint64_t>& a, bool inverse) const {
        for (uint32_t i = 1, j = 0; i < N; ++i) {          // bit-reversal permutation
            uint32_t bit = N >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(a[i], a[j]);
        }
        for (uint32_t len = 2; len <= N; len <<= 1) {
            uint32_t step = N / len;
            for (uint32_t i = 0; i < N; i += len) {
                for (uint32_t k = 0; k < len / 2; ++k) {
                    uint64_t wk = inverse ? iw[k * step] : w[k * step];
                    uint64_t u = a[i + k], v = (u128)a[i + k + len / 2] * wk % q;
                    a[i + k] = (u + v) % q;
                    a[i + k + len / 2] = (u + q - v) % q;
                }
            }
        }
        if (inverse) for (uint32_t i = 0; i < N; ++i) a[i] = (u128)a[i] * n_inv % q;
    }
    // coefficient -> evaluation domain (natural order). The psi twist is what makes the length-N
    // transform compute a NEGACYCLIC product; eval form here means "twisted, then transformed".
    std::vector<uint32_t> fwd(const std::vector<uint32_t>& A) const {
        std::vector<uint64_t> a(N);
        for (uint32_t i = 0; i < N; ++i) a[i] = (u128)A[i] * pw[i] % q;
        transform(a, false);
        std::vector<uint32_t> o(N);
        for (uint32_t i = 0; i < N; ++i) o[i] = (uint32_t)a[i];
        return o;
    }
    std::vector<uint32_t> inv(const std::vector<uint32_t>& E) const {
        std::vector<uint64_t> a(N);
        for (uint32_t i = 0; i < N; ++i) a[i] = E[i];
        transform(a, true);
        std::vector<uint32_t> o(N);
        for (uint32_t i = 0; i < N; ++i) o[i] = (uint32_t)((u128)a[i] * ipw[i] % q);
        return o;
    }
    // negacyclic product in Z_q[x]/(x^N+1)
    std::vector<uint32_t> negmul(const std::vector<uint32_t>& A, const std::vector<uint32_t>& B) const {
        std::vector<uint64_t> a(N), b(N);
        for (uint32_t i = 0; i < N; ++i) { a[i] = (u128)A[i] * pw[i] % q; b[i] = (u128)B[i] * pw[i] % q; }
        transform(a, false); transform(b, false);
        for (uint32_t i = 0; i < N; ++i) a[i] = (u128)a[i] * b[i] % q;
        transform(a, true);
        std::vector<uint32_t> o(N);
        for (uint32_t i = 0; i < N; ++i) o[i] = (uint32_t)((u128)a[i] * ipw[i] % q);
        return o;
    }
};

// ---------------------------------------------------------------- an RNS base with exact CRT
// Every step of a BFV multiply is "take residues to an exact integer, do something integral to it,
// put it back". Doing that exactly -- rather than with the approximate base conversions a fast RNS
// implementation uses -- is the whole point of this reference: it is the specification the device
// path will be checked against, so it must have no error terms of its own.
struct CRTBase {
    std::vector<uint64_t> p;
    std::vector<NTTCtx> nt;
    u512 prod, half;
    std::vector<u512> qhat;
    std::vector<uint64_t> qhat_inv;

    void build(const std::vector<uint64_t>& primes, bool with_ntt) {
        p = primes;
        prod = u512(1);
        for (auto q : p) prod = mul_small(prod, q);
        { u512 q2, r2; divmod(prod, u512(2), q2, r2); half = q2; }
        qhat.resize(p.size()); qhat_inv.resize(p.size());
        for (size_t i = 0; i < p.size(); ++i) {
            u512 qh, r; divmod(prod, u512(p[i]), qh, r);
            qhat[i] = qh;
            qhat_inv[i] = modinv(mod_small(qh, p[i]), p[i]);
        }
        if (with_ntt) for (auto q : p) nt.emplace_back(q);
    }
    // residues -> the unique integer in [0, prod)
    u512 reconstruct(const std::vector<uint32_t>& res) const {
        u512 acc;
        for (size_t i = 0; i < p.size(); ++i) {
            uint64_t c = (uint64_t)((u128)res[i] * qhat_inv[i] % p[i]);
            acc = add(acc, mul_small(qhat[i], c));
        }
        while (cmp(acc, prod) >= 0) acc = sub(acc, prod);   // acc < |p|*prod, so subtract
        return acc;
    }
    // ...and its CENTERED value, as (magnitude, sign). BFV bounds hold for centered
    // representatives; using [0, prod) instead makes the tensor overflow its base.
    void reconstruct_centered(const std::vector<uint32_t>& res, u512& mag, bool& neg) const {
        u512 v = reconstruct(res);
        neg = cmp(v, half) > 0;
        mag = neg ? sub(prod, v) : v;
    }
};

// round(t * x / D) for a centered x, returned as residues modulo each prime of `out`.
// This is BFV's rescale when D = Q, and the keyswitch mod-down when t = 1 and D = P.
void scale_round(const u512& mag, bool neg, uint64_t tmul, const u512& D, const u512& Dhalf,
                 const std::vector<uint64_t>& out, std::vector<uint32_t>& dst) {
    u512 num = tmul == 1 ? mag : mul_small(mag, tmul);
    num = add(num, Dhalf);                       // round to nearest
    u512 quo, rem; divmod(num, D, quo, rem);
    for (size_t i = 0; i < out.size(); ++i) {
        uint64_t r = mod_small(quo, out[i]);
        dst[i] = (uint32_t)(neg ? (out[i] - r) % out[i] : r);
    }
}

// ---------------------------------------------------------------- the scheme
struct Ciphertext { std::vector<std::vector<uint32_t>> c0, c1; };   // [limb][coeff], base Q

struct BFV {
    std::vector<uint64_t> Q, P, S;
    uint64_t Msk = 0;
    std::vector<NTTCtx> ntQ;
    u512 Qprod;                                    // exact product of the Q limbs
    std::vector<int64_t> s;                        // secret key, ternary
    std::mt19937_64 rng;
    CRTBase Qb, Tb, QPb;                           // Q ; Q|S (tensor) ; Q|P (keyswitch)
    std::vector<std::vector<uint32_t>> evk0, evk1; // relin key over QP, coefficient domain

    explicit BFV(uint64_t seed) : rng(seed) {
        auto pr = ntt_primes(LQ + LP + LS + 1);
        Q.assign(pr.begin(), pr.begin() + LQ);
        P.assign(pr.begin() + LQ, pr.begin() + LQ + LP);
        S.assign(pr.begin() + LQ + LP, pr.begin() + LQ + LP + LS);
        Msk = pr[LQ + LP + LS];
        for (auto q : Q) ntQ.emplace_back(q);
        Qprod = u512(1);
        for (auto q : Q) Qprod = mul_small(Qprod, q);
        s.resize(N);
        for (auto& x : s) x = (int64_t)(rng() % 3) - 1;

        Qb.build(Q, false);                                    // ntQ already covers Q
        std::vector<uint64_t> Tp = Q; Tp.insert(Tp.end(), S.begin(), S.end());
        Tb.build(Tp, true);                                    // Q|S: holds the UN-REDUCED tensor
        std::vector<uint64_t> QPp = Q; QPp.insert(QPp.end(), P.begin(), P.end());
        QPb.build(QPp, true);                                  // Q|P: the keyswitch base
        keygen_evk();
    }

    // relin key: evk0 = -(a*s) + e + P*s^2, evk1 = a, over QP
    void keygen_evk() {
        std::vector<int64_t> e_int(N);
        for (auto& x : e_int) x = (int64_t)(rng() % 5) - 2;
        std::vector<int64_t> s2(N, 0);                         // s^2 in Z[x]/(x^N+1), exact
        for (uint32_t i = 0; i < N; ++i) {
            if (!s[i]) continue;
            for (uint32_t j = 0; j < N; ++j) {
                uint32_t k = i + j;
                if (k < N) s2[k] += s[i] * s[j]; else s2[k - N] -= s[i] * s[j];
            }
        }
        const size_t LQP = QPb.p.size();
        evk0.assign(LQP, std::vector<uint32_t>(N));
        evk1.assign(LQP, std::vector<uint32_t>(N));
        u512 Pprod(1); for (auto q : P) Pprod = mul_small(Pprod, q);
        for (size_t li = 0; li < LQP; ++li) {
            uint64_t m = QPb.p[li];
            uint64_t Pm = mod_small(Pprod, m);
            std::vector<uint32_t> a(N), sl(N), s2l(N);
            for (uint32_t n = 0; n < N; ++n) {
                a[n] = (uint32_t)(rng() % m);
                sl[n] = (uint32_t)(((s[n] % (int64_t)m) + (int64_t)m) % (int64_t)m);
                s2l[n] = (uint32_t)(((s2[n] % (int64_t)m) + (int64_t)m) % (int64_t)m);
            }
            auto as_ = QPb.nt[li].negmul(a, sl);
            for (uint32_t n = 0; n < N; ++n) {
                uint32_t el = (uint32_t)(((e_int[n] % (int64_t)m) + (int64_t)m) % (int64_t)m);
                evk0[li][n] = (uint32_t)(((u128)m + el + (u128)Pm * s2l[n] % m - as_[n]) % m);
                evk1[li][n] = a[n];
            }
        }
    }

    std::vector<uint32_t> limb_of(const std::vector<int64_t>& poly, uint64_t q) const {
        std::vector<uint32_t> r(N);
        for (uint32_t n = 0; n < N; ++n) r[n] = (uint32_t)(((poly[n] % (int64_t)q) + (int64_t)q) % (int64_t)q);
        return r;
    }

    // ---- encrypt a FULL plaintext polynomial m in R_t --------------------------------
    //   c0 = Delta*m + e - a*s,  c1 = a,  Delta = floor(Q/t)
    // Delta is reduced per limb; the message is scaled by the *integer* floor(Q/t), so decryption's
    // round(t*v/Q) recovers m as long as the noise stays below Q/(2t).
    Ciphertext encrypt(const std::vector<int64_t>& m) {
        std::vector<int64_t> e_int(N);
        for (auto& x : e_int) x = (int64_t)(rng() % 5) - 2;
        // The mask a is uniform in [0, Q). Sampling its residues independently per limb IS that, by
        // CRT -- the residue vector determines one integer and the map is a bijection -- so there is no
        // need to draw a big integer and reduce it. What would be wrong is sampling a fresh a for c1
        // and a different one for the a*s term; the same residues feed both below.
        std::vector<std::vector<uint32_t>> aQ(LQ, std::vector<uint32_t>(N));
        for (uint32_t i = 0; i < LQ; ++i)
            for (uint32_t n = 0; n < N; ++n) aQ[i][n] = (uint32_t)(rng() % Q[i]);
        u512 dq, dr; divmod(Qprod, u512(T_PLAIN), dq, dr);   // Delta = floor(Q/t)
        Ciphertext ct;
        ct.c0.assign(LQ, std::vector<uint32_t>(N));
        ct.c1.assign(LQ, std::vector<uint32_t>(N));
        for (uint32_t i = 0; i < LQ; ++i) {
            uint64_t q = Q[i];
            auto sl = limb_of(s, q);
            auto as_ = ntQ[i].negmul(aQ[i], sl);
            uint64_t Delta_i = mod_small(dq, q);             // Delta mod q_i
            for (uint32_t n = 0; n < N; ++n) {
                uint32_t el = (uint32_t)(((e_int[n] % (int64_t)q) + (int64_t)q) % (int64_t)q);
                uint64_t mm = (uint64_t)(((m[n] % (int64_t)T_PLAIN) + T_PLAIN) % T_PLAIN);
                uint64_t base = (uint64_t)((u128)Delta_i * mm % q);
                ct.c0[i][n] = (uint32_t)(((u128)q + base + el - as_[n]) % q);
                ct.c1[i][n] = aQ[i][n];
            }
        }
        return ct;
    }

    // ---- decrypt to a FULL plaintext polynomial ---------------------------------------
    //   m = round(t/Q * [c0 + c1*s]_Q) mod t,  every coefficient
    std::vector<int64_t> decrypt(const Ciphertext& ct) const {
        // v = c0 + c1*s, per limb (negacyclic, via the same NTT the encryption used)
        std::vector<std::vector<uint32_t>> v(LQ);
        for (uint32_t i = 0; i < LQ; ++i) {
            uint64_t q = Q[i];
            auto sl = limb_of(s, q);
            auto cs = ntQ[i].negmul(ct.c1[i], sl);
            v[i].resize(N);
            for (uint32_t n = 0; n < N; ++n) v[i][n] = (uint32_t)(((u128)ct.c0[i][n] + cs[n]) % q);
        }
        // CRT-reconstruct each coefficient to an exact integer in [0, Q), then round(t*v/Q).
        // Q is 224-bit here, so the reconstruction genuinely needs the big integers: qhat alone is
        // ~196-bit and the per-term product overflows u128.
        std::vector<u512> qhat(LQ);
        std::vector<uint64_t> qhat_inv(LQ);
        for (uint32_t i = 0; i < LQ; ++i) {
            u512 qh, r; divmod(Qprod, u512(Q[i]), qh, r);
            qhat[i] = qh;
            qhat_inv[i] = modinv(mod_small(qh, Q[i]), Q[i]);  // qhat mod q_i, then invert
        }
        u512 half; { u512 q2, r2; divmod(Qprod, u512(2), q2, r2); half = q2; }
        std::vector<int64_t> out(N);
        for (uint32_t n = 0; n < N; ++n) {
            u512 acc;
            for (uint32_t i = 0; i < LQ; ++i) {
                uint64_t c = (uint64_t)((u128)v[i][n] * qhat_inv[i] % Q[i]);
                acc = add(acc, mul_small(qhat[i], c));
            }
            u512 val = acc;                                   // acc < LQ*Q, so subtract rather than divide
            while (cmp(val, Qprod) >= 0) val = sub(val, Qprod);
            // centre: values above Q/2 represent negatives
            bool neg = cmp(val, half) > 0;
            u512 mag = neg ? sub(Qprod, val) : val;
            u512 num = mul_small(mag, T_PLAIN);
            num = add(num, half);                             // round-to-nearest
            u512 quo, rem2; divmod(num, Qprod, quo, rem2);
            int64_t res = (int64_t)(to_u64(quo) % T_PLAIN);
            out[n] = neg ? (int64_t)((T_PLAIN - res) % T_PLAIN) : res;
        }
        return out;
    }

    // ---- the lift, Q -> Q|S on the centered representative (exact, big-integer) --------
    // Exposed as a method rather than a lambda inside multiply() so the device stages can be checked
    // against exactly the values the reference multiply uses.
    std::vector<std::vector<uint32_t>> lift_to_T(const std::vector<std::vector<uint32_t>>& in) const {
        const size_t LT = Tb.p.size();
        std::vector<std::vector<uint32_t>> out(LT, std::vector<uint32_t>(N));
        std::vector<uint32_t> res(LQ);
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t i = 0; i < LQ; ++i) res[i] = in[i][n];
            u512 mag; bool neg; Qb.reconstruct_centered(res, mag, neg);
            for (size_t j = 0; j < LT; ++j) {
                uint64_t r = mod_small(mag, Tb.p[j]);
                out[j][n] = (uint32_t)(neg ? (Tb.p[j] - r) % Tb.p[j] : r);
            }
        }
        return out;
    }

    // the reference tensor in base Q|S: (e0, e1, e2) = (a0*b0, a0*b1 + a1*b0, a1*b1)
    void tensor_ref(const Ciphertext& A, const Ciphertext& B,
                    std::vector<std::vector<uint32_t>>& e0,
                    std::vector<std::vector<uint32_t>>& e1,
                    std::vector<std::vector<uint32_t>>& e2) const {
        const size_t LT = Tb.p.size();
        auto a0 = lift_to_T(A.c0), a1 = lift_to_T(A.c1), b0 = lift_to_T(B.c0), b1 = lift_to_T(B.c1);
        e0.assign(LT, {}); e1.assign(LT, {}); e2.assign(LT, {});
        for (size_t j = 0; j < LT; ++j) {
            uint64_t m = Tb.p[j];
            e0[j] = Tb.nt[j].negmul(a0[j], b0[j]);
            e2[j] = Tb.nt[j].negmul(a1[j], b1[j]);
            auto x = Tb.nt[j].negmul(a0[j], b1[j]);
            auto y = Tb.nt[j].negmul(a1[j], b0[j]);
            e1[j].resize(N);
            for (uint32_t n = 0; n < N; ++n) e1[j][n] = (uint32_t)(((u128)x[n] + y[n]) % m);
        }
    }

    // ---- the BFV rescale, as an RNS implementation actually computes it ---------------
    // BEHZ derives the scaled value from an APPROXIMATE base conversion of [t*e]_Q, so what it
    // produces is floor(t*e/Q) - eps, where eps is that conversion's CRT overflow. Measured on these
    // parameters (TB001B_BEHZ=1): eps in [1,6], bounded by L_Q = 8.
    //
    // That eps is irrelevant to correctness -- the noise budget is ~2^209 -- but it is NOT irrelevant
    // to testing. If the reference computed the exact value, the device could only ever be checked by
    // decryption, which says "the scheme works" but not "stage 4 is right". Modelling the eps here
    // keeps the device checkable BIT-EXACTLY, so a fault localises to a stage instead of surfacing as
    // a decrypt failure three stages later. The exact form is still available as scale_round(), and
    // the decrypt test remains the independent semantic check.
    std::vector<std::vector<uint32_t>> rescale_behz(const std::vector<std::vector<uint32_t>>& e) const {
        const size_t LT = Tb.p.size();
        std::vector<u512> qhat(LQ); std::vector<uint64_t> qhinv(LQ);
        for (uint32_t i = 0; i < LQ; ++i) {
            u512 qh, r; divmod(Qprod, u512(Q[i]), qh, r);
            qhat[i] = qh; qhinv[i] = modinv(mod_small(qh, Q[i]), Q[i]);
        }
        std::vector<std::vector<uint32_t>> out(LQ, std::vector<uint32_t>(N));
        std::vector<uint32_t> res(LT);
        for (uint32_t n = 0; n < N; ++n) {
            for (size_t j = 0; j < LT; ++j) res[j] = e[j][n];
            u512 mag; bool neg; Tb.reconstruct_centered(res, mag, neg);
            // eps: the CRT overflow of the approximate conversion of [t*e]_Q from Q to S
            u512 acc;
            for (uint32_t i = 0; i < LQ; ++i) {
                uint64_t ei = mod_small(mag, Q[i]);
                if (neg) ei = (Q[i] - ei) % Q[i];
                uint64_t te = (uint64_t)((u128)ei * (T_PLAIN % Q[i]) % Q[i]);
                uint64_t y  = (uint64_t)((u128)te * qhinv[i] % Q[i]);
                acc = add(acc, mul_small(qhat[i], y));
            }
            uint32_t eps = 0;
            while (cmp(acc, Qprod) >= 0) { acc = sub(acc, Qprod); ++eps; }
            // floor(t*e/Q) on the SIGNED value, then subtract eps
            u512 num = mul_small(mag, T_PLAIN);
            u512 quo, rem; divmod(num, Qprod, quo, rem);
            u512 vmag; bool vneg;
            if (!neg) {                                  // e >= 0: floor = quo
                if (cmp(quo, u512(eps)) >= 0) { vmag = sub(quo, u512(eps)); vneg = false; }
                else { vmag = sub(u512(eps), quo); vneg = true; }
            } else {                                     // e < 0: floor = -(quo + [rem != 0])
                u512 f = rem.is_zero() ? quo : add(quo, u512(1));
                vmag = add(f, u512(eps)); vneg = true;    // -(f) - eps
            }
            for (uint32_t i = 0; i < LQ; ++i) {
                uint64_t r = mod_small(vmag, Q[i]);
                out[i][n] = (uint32_t)(vneg ? (Q[i] - r) % Q[i] : r);
            }
        }
        return out;
    }

    // the degree-2 ciphertext after tensor + rescale, i.e. stage 5's input
    void rescaled_ref(const Ciphertext& A, const Ciphertext& B,
                      std::vector<std::vector<uint32_t>>& d0,
                      std::vector<std::vector<uint32_t>>& d1,
                      std::vector<std::vector<uint32_t>>& d2) const {
        std::vector<std::vector<uint32_t>> e0, e1, e2;
        tensor_ref(A, B, e0, e1, e2);
        d0 = rescale_behz(e0); d1 = rescale_behz(e1); d2 = rescale_behz(e2);
    }

    // ---- the keyswitch mod-down, as an RNS implementation computes it -----------------
    // Same shape as rescale_behz: k = floor(v/P) is derived from an APPROXIMATE conversion of [v]_P
    // from P to Q, so what comes out is floor(v/P) - eps with eps the overflow of that conversion,
    // bounded by L_P = 4. Modelled here for the same reason -- it keeps the device bit-checkable.
    std::vector<std::vector<uint32_t>> moddown_behz(const std::vector<std::vector<uint32_t>>& v) const {
        const size_t LQP = QPb.p.size();
        u512 PP(1); for (auto q : P) PP = mul_small(PP, q);
        std::vector<u512> phat(LP); std::vector<uint64_t> phinv(LP);
        for (uint32_t j = 0; j < LP; ++j) {
            u512 ph, r; divmod(PP, u512(P[j]), ph, r);
            phat[j] = ph; phinv[j] = modinv(mod_small(ph, P[j]), P[j]);
        }
        std::vector<std::vector<uint32_t>> out(LQ, std::vector<uint32_t>(N));
        std::vector<uint32_t> res(LQP);
        for (uint32_t n = 0; n < N; ++n) {
            for (size_t j = 0; j < LQP; ++j) res[j] = v[j][n];
            u512 mag; bool neg; QPb.reconstruct_centered(res, mag, neg);
            u512 acc;
            for (uint32_t j = 0; j < LP; ++j) {
                uint64_t vj = mod_small(mag, P[j]);
                if (neg) vj = (P[j] - vj) % P[j];
                uint64_t y = (uint64_t)((u128)vj * phinv[j] % P[j]);
                acc = add(acc, mul_small(phat[j], y));
            }
            uint32_t eps = 0;
            while (cmp(acc, PP) >= 0) { acc = sub(acc, PP); ++eps; }
            u512 quo, rem; divmod(mag, PP, quo, rem);
            u512 vmag; bool vneg;
            if (!neg) {
                if (cmp(quo, u512(eps)) >= 0) { vmag = sub(quo, u512(eps)); vneg = false; }
                else { vmag = sub(u512(eps), quo); vneg = true; }
            } else {
                u512 f = rem.is_zero() ? quo : add(quo, u512(1));
                vmag = add(f, u512(eps)); vneg = true;
            }
            for (uint32_t i = 0; i < LQ; ++i) {
                uint64_t r = mod_small(vmag, Q[i]);
                out[i][n] = (uint32_t)(vneg ? (Q[i] - r) % Q[i] : r);
            }
        }
        return out;
    }

    // ---- HE multiply: tensor -> round(t*e/Q) -> relinearize ---------------------------
    // This is BFV's multiply, and the three steps are where it differs from the CKKS-shaped one
    // exp_Tb001 measures:
    //
    //   1. TENSOR OVER Z, not mod Q. The product of two ciphertexts has coefficients up to N*Q^2,
    //      so it is computed in the extended base Q|S -- 17 limbs, ~476 bits, against a bound of
    //      ~2^461. exp_Tb001 has no analogue of this step at all: its tensor is pointwise mod Q.
    //   2. RESCALE round(t*e/Q), not a drop-prime. It divides by the WHOLE modulus and rounds, which
    //      is why the un-reduced tensor was needed; a drop-prime divides by one limb and needs none.
    //   3. RELINEARIZE, the same hybrid keyswitch exp_Tb001 does -- the one step the two share.
    Ciphertext multiply(const Ciphertext& A, const Ciphertext& B) {
        const size_t LT = Tb.p.size(), LQP = QPb.p.size();

        // --- 1. lift both ciphertexts from Q into Q|S, exactly, on centered representatives ---
        auto a0 = lift_to_T(A.c0), a1 = lift_to_T(A.c1), b0 = lift_to_T(B.c0), b1 = lift_to_T(B.c1);

        // --- 2. tensor in Q|S: (e0,e1,e2) = (a0*b0, a0*b1 + a1*b0, a1*b1) ---
        std::vector<std::vector<uint32_t>> e0(LT), e1(LT), e2(LT);
        for (size_t j = 0; j < LT; ++j) {
            uint64_t m = Tb.p[j];
            e0[j] = Tb.nt[j].negmul(a0[j], b0[j]);
            e2[j] = Tb.nt[j].negmul(a1[j], b1[j]);
            auto x = Tb.nt[j].negmul(a0[j], b1[j]);
            auto y = Tb.nt[j].negmul(a1[j], b0[j]);
            e1[j].resize(N);
            for (uint32_t n = 0; n < N; ++n) e1[j][n] = (uint32_t)(((u128)x[n] + y[n]) % m);
        }

        // --- 3. rescale each of the three by the BFV scaling, back into base Q ---
        auto d0 = rescale_behz(e0), d1 = rescale_behz(e1), d2 = rescale_behz(e2);

        // --- 4. relinearize d2 with the evk, in base QP, then mod down by P ---
        // mod-up of d2 from Q into QP, by the APPROXIMATE base conversion -- which is what standard
        // hybrid keyswitching does, and what the device does.
        //
        // It must not be the exact centered lift. That version does not COMPOSE: a computed ciphertext
        // cannot produce its own centered sentinel without a magnitude comparison, which RNS cannot do
        // (see docs/he/exp_tb001b.md). The approximate conversion needs no representative from anyone
        // -- it is a pure matmul on the residues -- so it composes, and its eps*Q error contributes
        // eps*Q*s^2 to the output phase, which is 0 mod Q and therefore invisible to decryption.
        std::vector<std::vector<uint32_t>> d2qp(LQP, std::vector<uint32_t>(N));
        {
            std::vector<u512> qhat(LQ); std::vector<uint64_t> qhinv(LQ);
            std::vector<std::vector<uint64_t>> qh_p(LQ, std::vector<uint64_t>(LP));
            for (uint32_t i = 0; i < LQ; ++i) {
                u512 qh, r; divmod(Qprod, u512(Q[i]), qh, r);
                qhat[i] = qh; qhinv[i] = modinv(mod_small(qh, Q[i]), Q[i]);
                for (uint32_t j = 0; j < LP; ++j) qh_p[i][j] = mod_small(qh, P[j]);
            }
            for (uint32_t i = 0; i < LQ; ++i) d2qp[i] = d2[i];          // Q limbs pass through
            std::vector<uint64_t> y(LQ);
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t i = 0; i < LQ; ++i)
                    y[i] = (uint64_t)((u128)d2[i][n] * qhinv[i] % Q[i]);
                for (uint32_t j = 0; j < LP; ++j) {
                    u128 acc = 0;
                    for (uint32_t i = 0; i < LQ; ++i) acc += (u128)y[i] * qh_p[i][j] % P[j];
                    d2qp[LQ + j][n] = (uint32_t)(acc % P[j]);
                }
            }
        }
        std::vector<std::vector<uint32_t>> v0(LQP), v1(LQP);
        for (size_t j = 0; j < LQP; ++j) {
            v0[j] = QPb.nt[j].negmul(d2qp[j], evk0[j]);
            v1[j] = QPb.nt[j].negmul(d2qp[j], evk1[j]);
        }
        auto k0 = moddown_behz(v0), k1 = moddown_behz(v1);

        // --- 5. combine ---
        Ciphertext o;
        o.c0.assign(LQ, std::vector<uint32_t>(N));
        o.c1.assign(LQ, std::vector<uint32_t>(N));
        for (uint32_t i = 0; i < LQ; ++i) {
            uint64_t q = Q[i];
            for (uint32_t n = 0; n < N; ++n) {
                o.c0[i][n] = (uint32_t)(((u128)d0[i][n] + k0[i][n]) % q);
                o.c1[i][n] = (uint32_t)(((u128)d1[i][n] + k1[i][n]) % q);
            }
        }
        return o;
    }

    // ---- ciphertext I/O domain -------------------------------------------------------
    // COEFFICIENT form is the default I/O. That is the opposite of exp_Tb001, and the reason is
    // measured rather than stylistic: for BFV, eval-form I/O costs 16 MORE transforms per multiply
    // (171 vs 155), not 32 fewer.
    //
    // Worth being precise about what it buys HERE, because it is not what it buys in exp_Tb001.
    // There, eval input removes 32 forward NTTs outright: CKKS's drop-prime rescale keeps the whole
    // multiply in the evaluation domain, so a coefficient input is converted and never converted back.
    // BFV cannot do that. Its base extension Q -> Q|S and its rescale round(t*e/Q) are both
    // COEFFICIENT-WISE -- they are integer operations on residues, not pointwise ring operations --
    // so the multiply round-trips through coefficients whatever the I/O domain is. Eval I/O then
    // costs 32 INTTs at the front and 16 NTTs at the back, and saves 32 by letting the Q-limb
    // evaluation forms be reused for the tensor: 84 transforms at the ends against coeff I/O's 68.
    // So coefficient form is the contract: the fixture stores coefficient-domain ciphertexts and the
    // device will consume and produce them. TB001B_EVAL_IN=1 runs the eval path instead, and the two
    // must agree bit-for-bit on the coefficient-domain result -- that equality is the A/B.
    Ciphertext to_eval(const Ciphertext& ct) const {
        Ciphertext o; o.c0.resize(LQ); o.c1.resize(LQ);
        for (uint32_t i = 0; i < LQ; ++i) { o.c0[i] = ntQ[i].fwd(ct.c0[i]); o.c1[i] = ntQ[i].fwd(ct.c1[i]); }
        return o;
    }
    Ciphertext from_eval(const Ciphertext& ct) const {
        Ciphertext o; o.c0.resize(LQ); o.c1.resize(LQ);
        for (uint32_t i = 0; i < LQ; ++i) { o.c0[i] = ntQ[i].inv(ct.c0[i]); o.c1[i] = ntQ[i].inv(ct.c1[i]); }
        return o;
    }
    // the multiply as the DEVICE will see it: eval in, eval out
    Ciphertext multiply_eval(const Ciphertext& Ae, const Ciphertext& Be) {
        return to_eval(multiply(from_eval(Ae), from_eval(Be)));
    }

    // ---- HE add: per-limb modular addition (this is the whole operation) --------------
    Ciphertext add_ct(const Ciphertext& A, const Ciphertext& B) const {
        Ciphertext o;
        o.c0.assign(LQ, std::vector<uint32_t>(N));
        o.c1.assign(LQ, std::vector<uint32_t>(N));
        for (uint32_t i = 0; i < LQ; ++i) {
            uint64_t q = Q[i];
            for (uint32_t n = 0; n < N; ++n) {
                o.c0[i][n] = (uint32_t)(((u128)A.c0[i][n] + B.c0[i][n]) % q);
                o.c1[i][n] = (uint32_t)(((u128)A.c1[i][n] + B.c1[i][n]) % q);
            }
        }
        return o;
    }
};

// The plaintext-space product a*b in R_t = Z_t[x]/(x^N+1) -- the value a correct ct_mul must
// decrypt to. Schoolbook with a SIGNED accumulator: coefficients are < t = 12289, so each product is
// < 2^27 and N of them stay far inside int64; the negacyclic wrap is a subtraction, not a biased add.
std::vector<int64_t> plain_mul(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    std::vector<__int128> acc(N, 0);
    for (uint32_t i = 0; i < N; ++i) {
        if (!a[i]) continue;
        for (uint32_t j = 0; j < N; ++j) {
            __int128 p = (__int128)a[i] * b[j];
            uint32_t k = i + j;
            if (k < N) acc[k] += p; else acc[k - N] -= p;
        }
    }
    std::vector<int64_t> r(N);
    for (uint32_t n = 0; n < N; ++n) r[n] = (int64_t)(((acc[n] % T_PLAIN) + T_PLAIN) % T_PLAIN);
    return r;
}

// ---------------------------------------------------------------- fixed reference case
// The correctness check reads a FIXED case file rather than generating random data at run time.
// Two reasons this matters once the device is in the loop:
//   * a device failure has to be reproducible from the same bytes on any machine, and a run-time RNG
//     makes "it passed on my card" and "it failed on yours" impossible to compare;
//   * the expected OUTPUT is stored, so the device is checked against a specific ciphertext rather
//     than against whatever the host happens to compute in the same process -- a host-side bug can no
//     longer cancel a device-side one.
// The file carries eval-form inputs, the keys, the expected eval-form output, and the plaintexts, so
// one file supports both the bit-exact check and the end-to-end decrypt check.
constexpr uint32_t CASE_MAGIC = 0x42313042;   // "B01B"
constexpr uint32_t CASE_VERSION = 1;

struct Case {
    uint32_t N_, LQ_, LP_, LS_, t_;
    std::vector<uint64_t> Q, P, S;
    std::vector<int64_t> sk, m1, m2, expect;
    std::vector<std::vector<uint32_t>> evk0, evk1;
    Ciphertext Ae, Be, Re;                    // eval form, base Q
};

void wr32(std::FILE* f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
void wr64(std::FILE* f, uint64_t v) { std::fwrite(&v, 8, 1, f); }
void wrvec32(std::FILE* f, const std::vector<uint32_t>& v) { std::fwrite(v.data(), 4, v.size(), f); }
void wrvec64(std::FILE* f, const std::vector<int64_t>& v) { std::fwrite(v.data(), 8, v.size(), f); }
uint32_t rd32(std::FILE* f) { uint32_t v = 0; if (std::fread(&v, 4, 1, f) != 1) std::abort(); return v; }
uint64_t rd64(std::FILE* f) { uint64_t v = 0; if (std::fread(&v, 8, 1, f) != 1) std::abort(); return v; }
void rdvec32(std::FILE* f, std::vector<uint32_t>& v, size_t n) { v.resize(n); if (std::fread(v.data(), 4, n, f) != n) std::abort(); }
void rdvec64(std::FILE* f, std::vector<int64_t>& v, size_t n) { v.resize(n); if (std::fread(v.data(), 8, n, f) != n) std::abort(); }

void write_ct(std::FILE* f, const Ciphertext& c) {
    for (auto& r : c.c0) wrvec32(f, r);
    for (auto& r : c.c1) wrvec32(f, r);
}
void read_ct(std::FILE* f, Ciphertext& c) {
    c.c0.resize(LQ); c.c1.resize(LQ);
    for (auto& r : c.c0) rdvec32(f, r, N);
    for (auto& r : c.c1) rdvec32(f, r, N);
}

bool write_case(const char* path, BFV& bfv, const Case& c) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    wr32(f, CASE_MAGIC); wr32(f, CASE_VERSION);
    wr32(f, N); wr32(f, LQ); wr32(f, LP); wr32(f, LS); wr32(f, T_PLAIN);
    for (auto q : bfv.Q) wr64(f, q);
    for (auto q : bfv.P) wr64(f, q);
    for (auto q : bfv.S) wr64(f, q);
    wrvec64(f, c.sk);
    for (auto& r : c.evk0) wrvec32(f, r);
    for (auto& r : c.evk1) wrvec32(f, r);
    write_ct(f, c.Ae); write_ct(f, c.Be); write_ct(f, c.Re);
    wrvec64(f, c.m1); wrvec64(f, c.m2); wrvec64(f, c.expect);
    std::fclose(f);
    return true;
}

bool read_case(const char* path, Case& c) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    if (rd32(f) != CASE_MAGIC || rd32(f) != CASE_VERSION) { std::fclose(f); return false; }
    c.N_ = rd32(f); c.LQ_ = rd32(f); c.LP_ = rd32(f); c.LS_ = rd32(f); c.t_ = rd32(f);
    if (c.N_ != N || c.LQ_ != LQ || c.LP_ != LP || c.LS_ != LS || c.t_ != T_PLAIN) { std::fclose(f); return false; }
    c.Q.resize(LQ); for (auto& q : c.Q) q = rd64(f);
    c.P.resize(LP); for (auto& q : c.P) q = rd64(f);
    c.S.resize(LS); for (auto& q : c.S) q = rd64(f);
    rdvec64(f, c.sk, N);
    const size_t LQP = LQ + LP;
    c.evk0.resize(LQP); c.evk1.resize(LQP);
    for (auto& r : c.evk0) rdvec32(f, r, N);
    for (auto& r : c.evk1) rdvec32(f, r, N);
    read_ct(f, c.Ae); read_ct(f, c.Be); read_ct(f, c.Re);
    rdvec64(f, c.m1, N); rdvec64(f, c.m2, N); rdvec64(f, c.expect, N);
    std::fclose(f);
    return true;
}


// ======================================================================================
// DEVICE STAGE 1 — the N=8192 four-step NTT/INTT on G=8 cores, checked against the host
// reference. Everything downstream (lift, tensor, rescale, relin) rides on this transform, so it is
// verified on its own first: if the eval->coeff conversion is wrong, every later stage is wrong in a
// way that is far harder to localise.
//
// The four numbered blocks below are lifted VERBATIM from host/experiments/exp_tb001.cpp. That is deliberate --
// they are the validated, optimised versions (fused 8-partial contraction, SFPI twiddle, width-correct
// FBC) and retyping them is how they drift. What differs is only the environment they run in: 22
// primes instead of 12, and the profiling/trace names stubbed out above.
// ======================================================================================
bool device_stage(const Case& kase, BFV& bfv) {
    // all 22 primes: Q(8) | P(4) | S(9) | m_sk(1). The transform layer is prime-agnostic; every base
    // the BFV multiply touches needs its own twiddle set.
    std::vector<uint64_t> QP;
    for (auto q : bfv.Q) QP.push_back(q);
    for (auto q : bfv.P) QP.push_back(q);
    for (auto q : bfv.S) QP.push_back(q);
    QP.push_back(bfv.Msk);
    std::vector<mont::MontgomeryContext> ctx;
    for (auto m : QP) ctx.emplace_back((uint32_t)m);

    // Stage 1 only transforms the Q limbs, so only their constant sets are generated. Paramgen is
    // O(N2^2) per (prime,direction) and generating all 22 up front costs minutes for no benefit yet.
    // Stage 3 needs transform constants for the tensor base Q|S|m_sk too. Paramgen is O(N2^2) per
    // (prime, direction), so it is only paid when that stage is asked for.
    uint32_t LQP = std::getenv("TB001B_STAGE3") ? (LQ + LP + LS + 1) : LQ;

    auto mesh = distributed::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE);
    auto& cq = mesh->mesh_command_queue();
    auto dr = distributed::MeshCoordinateRange(mesh->shape());
    auto grid = mesh->compute_with_storage_grid_size();
    const uint32_t GX = grid.x, GY = grid.y;
    uint32_t want = GX * GY; if (const char* e = std::getenv("NTT_CORES")) { int v = std::atoi(e); if (v > 0) want = (uint32_t)v; }
    const uint32_t NCORES = std::min<uint32_t>(want, GX * GY);
    const uint32_t EWC = std::min<uint32_t>(NBLK, NCORES);
    auto CC = [&](uint32_t i){ return CoreCoord{i % GX, i / GX}; };
    fmt::print("exp_Tb001b  device stage 1 (NTT/INTT)  grid {}x{} -> G={} cores, {} primes generated\n",
               GX, GY, NCORES, LQP);

    auto mkbuf = [&](uint32_t nt, uint32_t page) {
        distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * page}, d, mesh.get()); };
    auto mkcb = [&](Program& p, const CoreRangeSet& crs, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
        CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, crs, c); };
    auto crs_of = [&](uint32_t cnt){ std::set<CoreRange> s; for (uint32_t i=0;i<cnt;++i) s.insert(CoreRange(CC(i))); return CoreRangeSet(s); };

    // scratch pool, same shapes as exp_Tb001
    std::vector<std::shared_ptr<distributed::MeshBuffer>> in8(NBLK), tw8(NBLK), mid8(NBLK), midB8(NBLK), out8(NBLK), part8(NBLK);
    for (uint32_t i = 0; i < NBLK; ++i) { in8[i]=mkbuf(1,I32); tw8[i]=mkbuf(1,I32); mid8[i]=mkbuf(1,I32);
        midB8[i]=mkbuf(1,I32); out8[i]=mkbuf(1,I32); part8[i]=mkbuf(NBLK,I32); }

    std::string KLBL;                       // dispatch label: unused until the trace layer is ported
    bool CACHE_ON = false; (void)CACHE_ON;
    auto emit = [&](auto&& build){ auto wl = build(); t_enq(cq, wl, false); };

//@@BLOCK mm
        struct MM { std::shared_ptr<distributed::MeshBuffer> in, w, out; uint32_t out_tile; };
        auto build_mm = [&](const std::vector<MM>& jobs, uint32_t base, uint32_t cnt, uint32_t cdep,
                            const char* kern, const std::shared_ptr<distributed::MeshBuffer>& inproto,
                            const std::shared_ptr<distributed::MeshBuffer>& wproto, uint32_t q, uint32_t li) {
            auto __tb=prof::hrc::now(); Program p = CreateProgram(); auto crs = crs_of(cnt);
            mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32); mkcb(p, crs, CBIndex::c_1, tt::DataFormat::Int8, cdep, I8);
            mkcb(p, crs, CBIndex::c_2, tt::DataFormat::Int8, NB, I8); mkcb(p, crs, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);
            mkcb(p, crs, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*inproto).append_to(rc); TensorAccessorArgs(*wproto).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused_batch.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + std::string(kern), crs,
                ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            for (uint32_t i=0;i<cnt;++i){ auto& j=jobs[base+i]; auto c=CC(i);
                SetRuntimeArgs(p, rd, c, {(uint32_t)j.in->address(), (uint32_t)j.w->address()});
                SetRuntimeArgs(p, wr, c, {(uint32_t)j.out->address(), 1u, j.out_tile});
                SetRuntimeArgs(p, cp, c, {q, ctx[li].q_inv_32(), (uint32_t)(QP[li]&0xFFFF), (uint32_t)(QP[li]>>16), ctx[li].to_montgomery(16384u)}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); prof::g_build+=prof::msd(__tb); ++prof::n_build; return wl; };

        // ---- ACCUMULATED block contraction: ACC partials, ONE recombine ----
        // The 8-block contraction ran as 8 dispatches per output block, each ending in a full
        // recombine, then a modadd tree summed the 8 residues. The recombine is Z-linear in the int32
        // planes, so the planes can accumulate in DEST across ACC partials and be recombined ONCE --
        // exact, not approximate. kernels/experiments/exp_tb001/compute/fused_matmul_a_acc.cpp has the argument
        // and the bound. ACC=NBLK folds the whole contraction into one dispatch per output block and
        // leaves the reducer with a single partial; TB001_ACC=1 restores one partial per dispatch.
        uint32_t ACC = NBLK;
        if (const char* e = std::getenv("TB001_ACC")) { int v = std::atoi(e); if (v > 0) ACC = (uint32_t)v; }
        if (ACC > NBLK || NBLK % ACC) ACC = 1;                    // must tile the contraction evenly
        // nsub = the final reduction's conditional-subtract chain length. low = plane0 + 128*plane1
        // scales with ACC, so the chain must too. The chain is valid over the whole unsigned range,
        // so the ceiling is 2^32 (r itself must fit in a tile element). Returns 0 when it does not.
        auto nsub_of = [&](uint32_t nacc, uint64_t q) -> uint32_t {
            const uint64_t plane_max = (uint64_t)NB * 32ull * 127ull * 127ull;  // 4 limbs x 32-dot x 127^2
            const uint64_t hi_max = (uint64_t)nacc * plane_max * 129ull;        // = low_max; both 128-weighted
            const uint64_t r_max = hi_max + q;                                  // low + hi14
            return (r_max < (1ull << 32)) ? (uint32_t)(r_max / q) + 1u : 0u; };
        if (ACC > 1) { uint32_t bad = 0;
            for (uint32_t li = 0; li < LQP; ++li) if (!nsub_of(ACC, QP[li])) ++bad;
            if (bad) { fmt::print("exp_Tb001  ACC={} exceeds the int32 recombine bound on {} of {} primes"
                                  " -> falling back to ACC=1\n", ACC, bad, LQP); ACC = 1; } }
        struct MMA { std::vector<std::shared_ptr<distributed::MeshBuffer>> in, w;
                     std::shared_ptr<distributed::MeshBuffer> out; uint32_t out_tile; };
        auto build_mm_acc = [&](const std::vector<MMA>& jobs, uint32_t base, uint32_t cnt, const char* kern,
                                const std::shared_ptr<distributed::MeshBuffer>& inproto,
                                const std::shared_ptr<distributed::MeshBuffer>& wproto, uint32_t q, uint32_t li) {
            auto __tb=prof::hrc::now(); Program p = CreateProgram(); auto crs = crs_of(cnt);
            mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, ACC, I32);
            mkcb(p, crs, CBIndex::c_1, tt::DataFormat::Int8, ACC*NB*NB, I8);
            mkcb(p, crs, CBIndex::c_2, tt::DataFormat::Int8, ACC*NB, I8);
            mkcb(p, crs, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);
            mkcb(p, crs, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*inproto).append_to(rc); TensorAccessorArgs(*wproto).append_to(rc);
            auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused_acc.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + std::string(kern), crs,
                ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            for (uint32_t i=0;i<cnt;++i){ auto& j=jobs[base+i]; auto c=CC(i);
                uint32_t na=(uint32_t)j.in.size();
                std::vector<uint32_t> ra; ra.push_back(na);
                for (auto& b : j.in) ra.push_back((uint32_t)b->address());
                for (auto& b : j.w)  ra.push_back((uint32_t)b->address());
                SetRuntimeArgs(p, rd, c, ra);
                SetRuntimeArgs(p, wr, c, {(uint32_t)j.out->address(), 1u, j.out_tile});
                SetRuntimeArgs(p, cp, c, {q, ctx[li].q_inv_32(), (uint32_t)(QP[li]&0xFFFF), (uint32_t)(QP[li]>>16),
                                          ctx[li].to_montgomery(16384u), na, nsub_of(na, QP[li])}); }
            distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); prof::g_build+=prof::msd(__tb); ++prof::n_build; return wl; };


//@@BLOCK gen (paramgen + constant residency)
        // ===== paramgen HOISTED: precompute every (prime,dir) folded twiddle/matrix set ONCE =====
        // (was regenerated on every transform call — the dominant ~50 ms host gap between kernels).
        struct CS { std::vector<std::vector<int8_t>> big{std::vector<std::vector<int8_t>>(NBLK*NBLK)};
                    std::vector<int8_t> small; std::vector<std::vector<uint32_t>> tw{std::vector<std::vector<uint32_t>>(NBLK)}; };
        std::map<int,CS> KC;                                    // key = li*2 + fwd
        auto gen_consts = [&](uint32_t li, int fwd) {
            uint64_t q=QP[li]; auto& C=ctx[li]; CS cs; uint64_t g=prim_root(q);
            if (fwd) {
                uint64_t w=modpow(g,(q-1)/N,q), psi=modpow(g,(q-1)/(2*N),q), w256=modpow(w,N1,q), w32=modpow(w,N2,q);
                std::vector<uint32_t> WA(N2*N2),Tw(N2*N1),WCt(N1*N1);
                for (uint32_t k2=0;k2<N2;++k2) for (uint32_t n2=0;n2<N2;++n2)
                    WA[k2*N2+n2]=(uint32_t)((u128)modpow(w256,(uint64_t)k2*n2,q)*modpow(psi,32ull*n2,q)%q);
                for (uint32_t k2=0;k2<N2;++k2) for (uint32_t n1=0;n1<N1;++n1)
                    Tw[k2*N1+n1]=(uint32_t)((u128)modpow(psi,n1,q)*modpow(w,(uint64_t)n1*k2,q)%q);
                for (uint32_t n1=0;n1<N1;++n1) for (uint32_t k1=0;k1<N1;++k1)
                    WCt[n1*N1+k1]=(uint32_t)modpow(w32,(uint64_t)k1*n1,q);
                for (uint32_t bk2=0;bk2<NBLK;++bk2) for (uint32_t bn2=0;bn2<NBLK;++bn2){
                    std::vector<uint32_t> blk(ELEMS);
                    for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b) blk[a*T+b]=WA[(bk2*T+a)*N2+(bn2*T+b)];
                    std::vector<int8_t> fold(NB*NB*ELEMS);
                    for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<int8_t> l(ELEMS);
                        for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[r]*blk[i])%q),p);
                        auto tl=tilize32(l); std::copy(tl.begin(),tl.end(),fold.begin()+(p*NB+r)*ELEMS); }
                    cs.big[bk2*NBLK+bn2]=std::move(fold); }
                { std::vector<int8_t> fold(NB*NB*ELEMS);
                  for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<int8_t> l(ELEMS);
                      for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[p]*WCt[i])%q),r);
                      auto tl=tilize32(l); std::copy(tl.begin(),tl.end(),fold.begin()+(p*NB+r)*ELEMS); }
                  cs.small=std::move(fold); }
                for (uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> blk(ELEMS);
                    for (uint32_t a=0;a<T;++a) for (uint32_t n1=0;n1<N1;++n1) blk[a*T+n1]=C.to_montgomery(Tw[(bk2*T+a)*N1+n1]);
                    cs.tw[bk2]=tilize32(blk); }
            } else {
                uint64_t wf=modpow(g,(q-1)/N,q);
                uint64_t w=mont::modular_inverse(wf,q), w256=mont::modular_inverse(modpow(wf,N1,q),q), w32=mont::modular_inverse(modpow(wf,N2,q),q);
                uint64_t psi=modpow(g,(q-1)/(2*N),q), psinv=mont::modular_inverse(psi,q), Ninv=mont::modular_inverse(N,q);
                std::vector<uint32_t> Winv32(N1*N1),Tw(N1*N2),Winv256(N2*N2);
                for (uint32_t n1=0;n1<N1;++n1) for (uint32_t k1=0;k1<N1;++k1)
                    Winv32[n1*N1+k1]=(uint32_t)((u128)modpow(w32,(uint64_t)n1*k1,q)*Ninv%q);
                for (uint32_t n1=0;n1<N1;++n1) for (uint32_t k2=0;k2<N2;++k2)
                    Tw[n1*N2+k2]=(uint32_t)((u128)modpow(w,(uint64_t)n1*k2,q)*modpow(psinv,n1,q)%q);
                for (uint32_t k2=0;k2<N2;++k2) for (uint32_t n2=0;n2<N2;++n2)
                    Winv256[k2*N2+n2]=(uint32_t)((u128)modpow(w256,(uint64_t)k2*n2,q)*modpow(psinv,32ull*n2,q)%q);
                { std::vector<int8_t> fold(NB*NB*ELEMS);
                  for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<int8_t> l(ELEMS);
                      for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[r]*Winv32[i])%q),p);
                      auto tl=tilize32(l); std::copy(tl.begin(),tl.end(),fold.begin()+(p*NB+r)*ELEMS); }
                  cs.small=std::move(fold); }
                for (uint32_t bk2=0;bk2<NBLK;++bk2) for (uint32_t bn2=0;bn2<NBLK;++bn2){
                    std::vector<uint32_t> blk(ELEMS);
                    for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b) blk[a*T+b]=Winv256[(bk2*T+a)*N2+(bn2*T+b)];
                    std::vector<int8_t> fold(NB*NB*ELEMS);
                    for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<int8_t> l(ELEMS);
                        for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[p]*blk[i])%q),r);
                        auto tl=tilize32(l); std::copy(tl.begin(),tl.end(),fold.begin()+(p*NB+r)*ELEMS); }
                    cs.big[bk2*NBLK+bn2]=std::move(fold); }
                for (uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> blk(ELEMS);
                    for (uint32_t n1=0;n1<N1;++n1) for (uint32_t k2l=0;k2l<T;++k2l) blk[n1*T+k2l]=C.to_montgomery(Tw[n1*N2+(bk2*T+k2l)]);
                    cs.tw[bk2]=tilize32(blk); }
            }
            return cs; };
        for (uint32_t li=0; li<LQP; ++li) { KC[li*2+1]=gen_consts(li,1); KC[li*2+0]=gen_consts(li,0); }

        // ===== constant RESIDENCY: upload every (prime,dir) folded set to its own DRAM buffer ONCE =====
        // (was re-uploaded on every transform call). Transforms then reference these — no per-call const upload.
        std::map<int,std::vector<std::shared_ptr<distributed::MeshBuffer>>> bigWres, twres;
        std::map<int,std::shared_ptr<distributed::MeshBuffer>> smallWres;
        for (auto& kv : KC) { int key=kv.first; const CS& cs=kv.second;
            bigWres[key].resize(NBLK*NBLK); twres[key].resize(NBLK);
            for (uint32_t i=0;i<NBLK*NBLK;++i){ bigWres[key][i]=mkbuf(NB*NB,I8); t_wr(cq, bigWres[key][i], cs.big[i], false); }
            smallWres[key]=mkbuf(NB*NB,I8); t_wr(cq, smallWres[key], cs.small, false);
            for (uint32_t bk2=0;bk2<NBLK;++bk2){ twres[key][bk2]=mkbuf(1,I32); t_wr(cq, twres[key][bk2], cs.tw[bk2], false); } }
        t_fin(cq);


//@@BLOCK stg (transpose, reduce+twiddle, twiddle)
        auto transpose_dev = [&](std::shared_ptr<distributed::MeshBuffer> in, std::shared_ptr<distributed::MeshBuffer> out){
            KLBL="transpose_tile"; emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(1);
            mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto w=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto c=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/transpose_tile.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,r,CC(0),{(uint32_t)in->address(),1u,0u}); SetRuntimeArgs(p,w,CC(0),{(uint32_t)out->address(),1u,0u}); SetRuntimeArgs(p,c,CC(0),{});
            prof::g_build+=prof::msd(__tb); ++prof::n_build; distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); };
        // resident chunked mod-add-reduce + twiddle (EWC cores, 2 tile-blocks/core when EWC<8)
        // npart = how many partials there are to reduce. ACC>1 makes the contraction emit NBLK/ACC
        // of them instead of NBLK, so this is no longer always NBLK.
        auto rmodadd=[&](std::vector<std::shared_ptr<distributed::MeshBuffer>>& ins, std::vector<std::shared_ptr<distributed::MeshBuffer>>& outs, uint32_t q, uint32_t npart=NBLK){
          for(uint32_t base=0;base<NBLK;base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,NBLK-base);
KLBL="modadd_reduce";             emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(cnt);
              mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,NBLK,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
              auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              for(uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                SetRuntimeArgs(p,rd,c,{(uint32_t)ins[t]->address(),npart,0u}); SetRuntimeArgs(p,wr,c,{(uint32_t)outs[t]->address(),1u,0u}); SetRuntimeArgs(p,cp,c,{npart,q}); }
              prof::g_build+=prof::msd(__tb); ++prof::n_build; distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); } };
        // FUSED reduce+twiddle: the rmodadd->rtwiddle pair as ONE program. The 8:1 reduction result
        // stays in DEST instead of round-tripping through DRAM to feed the twiddle multiply, so this
        // drops one dispatch and one 8-tile write + 8-tile read per forward NTT.
        auto rmodadd_tw=[&](std::vector<std::shared_ptr<distributed::MeshBuffer>>& ins, std::vector<std::shared_ptr<distributed::MeshBuffer>>& tw, std::vector<std::shared_ptr<distributed::MeshBuffer>>& outs, uint32_t q, uint32_t li, uint32_t npart=NBLK){
          for(uint32_t base=0;base<NBLK;base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,NBLK-base);
KLBL="modadd_twiddle";             emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(cnt);
              mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,NBLK,I32); mkcb(p,crs,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
              auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_reduce_tw.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_twiddle.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              for(uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                SetRuntimeArgs(p,rd,c,{(uint32_t)ins[t]->address(),npart,(uint32_t)tw[t]->address()});
                SetRuntimeArgs(p,wr,c,{(uint32_t)outs[t]->address(),1u,0u});
                SetRuntimeArgs(p,cp,c,{npart,q,ctx[li].q_inv_32(),(uint32_t)(QP[li]&0xFFFF),(uint32_t)(QP[li]>>16)}); }
              prof::g_build+=prof::msd(__tb); ++prof::n_build; distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); } };
        auto rtwiddle=[&](std::vector<std::shared_ptr<distributed::MeshBuffer>>& a, std::vector<std::shared_ptr<distributed::MeshBuffer>>& b, std::vector<std::shared_ptr<distributed::MeshBuffer>>& o, uint32_t q, uint32_t li){
          for(uint32_t base=0;base<NBLK;base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,NBLK-base);
KLBL="modmul_sfpi_full";             emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(cnt);
              mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
              auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              // modmul_sfpi_full, NOT modmul_compute: identical arithmetic and identical runtime args,
              // measured 14.7 vs 16.5 us/dispatch. The LLK kernel re-inits on every helper call and has
              // to materialise each intermediate into a DST tile -- 17 DUPs exist only for that -- while
              // the SFPI face functions keep intermediates in vector registers. ~72 tile passes vs ~21.
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_full.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              for(uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                SetRuntimeArgs(p,rd,c,{(uint32_t)a[t]->address(),(uint32_t)b[t]->address(),1u,0u}); SetRuntimeArgs(p,wr,c,{(uint32_t)o[t]->address(),1u,0u}); SetRuntimeArgs(p,cp,c,{1u,q,ctx[li].q_inv_32(),(uint32_t)(QP[li]&0xFFFF),(uint32_t)(QP[li]>>16)}); }
              prof::g_build+=prof::msd(__tb); ++prof::n_build; distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); } };

//@@BLOCK chain (fwd_buf / inv_buf)
        // fwd chain, input from DRAM inb[8], raw output to DRAM outb[8]; NO readback
        auto fwd_buf = [&](std::vector<std::shared_ptr<distributed::MeshBuffer>>& inb,
                           std::vector<std::shared_ptr<distributed::MeshBuffer>>& outb, uint32_t li){
            uint64_t q=QP[li]; int key=li*2+1; auto& big=bigWres[key]; auto sm=smallWres[key]; auto& tw=twres[key];
            std::vector<MM> s1; std::vector<MMA> s1a;
            if(ACC>1){ for(uint32_t bk2=0;bk2<NBLK;++bk2) for(uint32_t g=0;g<NBLK/ACC;++g){ MMA j;
                    for(uint32_t a=0;a<ACC;++a){ uint32_t bn2=g*ACC+a; j.in.push_back(inb[bn2]); j.w.push_back(big[bk2*NBLK+bn2]); }
                    j.out=part8[bk2]; j.out_tile=g; s1a.push_back(j); }
                for(uint32_t base=0;base<s1a.size();base+=NCORES){ uint32_t cnt=std::min<uint32_t>(NCORES,s1a.size()-base); KLBL="fused_matmul_a_acc"; emit([&]{ return build_mm_acc(s1a,base,cnt,"compute/fused_matmul_a_acc.cpp",inb[0],big[0],(uint32_t)q,li); }); }
            } else {
                for(uint32_t bk2=0;bk2<NBLK;++bk2) for(uint32_t bn2=0;bn2<NBLK;++bn2) s1.push_back({inb[bn2],big[bk2*NBLK+bn2],part8[bk2],bn2});
                for(uint32_t base=0;base<s1.size();base+=NCORES){ uint32_t cnt=std::min<uint32_t>(NCORES,s1.size()-base); KLBL="fused_matmul_a"; emit([&]{ return build_mm(s1,base,cnt,NB*NB,"compute/fused_matmul_a.cpp",inb[0],big[0],(uint32_t)q,li); }); } }
            // With ACC=NBLK the contraction emits ONE partial, so the fused reduce+twiddle has nothing
            // to reduce -- it is a twiddle multiply. pw_buf is exactly that multiply on the same operand
            // shape, and it runs compute/modmul_sfpi_full.cpp instead of the LLK modadd_twiddle:
            // measured 14.7 vs 16.3 us/dispatch for the same arithmetic. LLK re-inits on every helper
            // call and materialises every intermediate into a DST tile (modmul_compute needs 17 DUPs for
            // that alone); the SFPI face functions keep intermediates in vector registers.
            if(NBLK/ACC==1) rtwiddle(part8,tw,midB8,(uint32_t)q,li);
            else            rmodadd_tw(part8,tw,midB8,(uint32_t)q,li,NBLK/ACC);
            std::vector<MM> s2; for(uint32_t bk2=0;bk2<NBLK;++bk2) s2.push_back({midB8[bk2],sm,outb[bk2],0});
            for(uint32_t base=0;base<s2.size();base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,s2.size()-base); KLBL="fused_matmul_c"; emit([&]{ return build_mm(s2,base,cnt,NB*NB,"compute/fused_matmul_c.cpp",midB8[0],sm,(uint32_t)q,li); }); } };
        // inv chain, input from DRAM inb[8] (already transposed), output to DRAM outb[8]; NO readback
        auto inv_buf = [&](std::vector<std::shared_ptr<distributed::MeshBuffer>>& inb,
                           std::vector<std::shared_ptr<distributed::MeshBuffer>>& outb, uint32_t li){
            uint64_t q=QP[li]; int key=li*2+0; auto& big=bigWres[key]; auto sm=smallWres[key]; auto& tw=twres[key];
            std::vector<MM> s1; for(uint32_t bk2=0;bk2<NBLK;++bk2) s1.push_back({inb[bk2],sm,mid8[bk2],0});
            for(uint32_t base=0;base<s1.size();base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,s1.size()-base); KLBL="fused_matmul_a"; emit([&]{ return build_mm(s1,base,cnt,NB*NB,"compute/fused_matmul_a.cpp",inb[0],sm,(uint32_t)q,li); }); }
            rtwiddle(mid8,tw,midB8,(uint32_t)q,li);
            std::vector<MM> s2; std::vector<MMA> s2a;
            if(ACC>1){ for(uint32_t bn2=0;bn2<NBLK;++bn2) for(uint32_t g=0;g<NBLK/ACC;++g){ MMA j;
                    for(uint32_t a=0;a<ACC;++a){ uint32_t bk2=g*ACC+a; j.in.push_back(midB8[bk2]); j.w.push_back(big[bk2*NBLK+bn2]); }
                    j.out=part8[bn2]; j.out_tile=g; s2a.push_back(j); }
                for(uint32_t base=0;base<s2a.size();base+=NCORES){ uint32_t cnt=std::min<uint32_t>(NCORES,s2a.size()-base); KLBL="fused_matmul_c_acc"; emit([&]{ return build_mm_acc(s2a,base,cnt,"compute/fused_matmul_c_acc.cpp",midB8[0],big[0],(uint32_t)q,li); }); }
            } else {
                for(uint32_t bn2=0;bn2<NBLK;++bn2) for(uint32_t bk2=0;bk2<NBLK;++bk2) s2.push_back({midB8[bk2],big[bk2*NBLK+bn2],part8[bn2],bk2});
                for(uint32_t base=0;base<s2.size();base+=NCORES){ uint32_t cnt=std::min<uint32_t>(NCORES,s2.size()-base); KLBL="fused_matmul_c"; emit([&]{ return build_mm(s2,base,cnt,NB*NB,"compute/fused_matmul_c.cpp",midB8[0],big[0],(uint32_t)q,li); }); } }
            rmodadd(part8,outb,(uint32_t)q,NBLK/ACC); };


    // ---- STAGE 2: the SK lift on device, Q -> S, exact and centered ----
    // Constants exactly as the host validation (TB001B_SKLIFT) derived them; the sentinel column is
    // index LD of C2/P/PINV, and x_sk (the CENTERED value's residue) is input tile LS.
    // prime lookup over the 22-entry QP: 0..7 Q, 8..11 P, 12..20 S, 21 m_sk
    auto pm = [&](uint32_t idx){ return (uint64_t)((idx < LQ) ? bfv.Q[idx]
                              : (idx < LQ + LP) ? bfv.P[idx - LQ]
                              : (idx < LQ + LP + LS) ? bfv.S[idx - LQ - LP] : bfv.Msk); };
    const uint32_t MSKI = LQ + LP + LS;
    auto prodOf = [&](const std::vector<uint32_t>& idx){ u512 r(1); for (auto i : idx) r = mul_small(r, pm(i)); return r; };

    // ---- APPROXIMATE FBC (drops eps), with an optional scalar folded into phase 1 ----
    // This is the conversion BEHZ's rescale uses: folding t into C1 makes phase 2 produce
    // ([t*e]_Q + eps*Q) mod p_j in one pass, which the epilogue then turns into the scaled value.
    auto build_fbc = [&](const std::vector<uint32_t>& sidx, const std::vector<uint32_t>& didx,
                         uint64_t fold,
                         const std::vector<std::vector<std::shared_ptr<distributed::MeshBuffer>>>& src,
                         const std::vector<std::vector<std::shared_ptr<distributed::MeshBuffer>>>& dst,
                         uint32_t cnt_cores) {
        const uint32_t Ls = (uint32_t)sidx.size(), Ld = (uint32_t)didx.size();
        const u512 SP = prodOf(sidx);
        std::vector<u512> shat(Ls);
        for (uint32_t i = 0; i < Ls; ++i) { u512 r; divmod(SP, u512(pm(sidx[i])), shat[i], r); }
        std::vector<uint32_t> ca; ca.push_back(Ls); ca.push_back(Ld);
        for (uint32_t i = 0; i < Ls; ++i) {
            uint64_t si = pm(sidx[i]);
            uint64_t v = (uint64_t)((u128)(fold % si) * modinv(mod_small(shat[i], si), si) % si);
            ca.push_back(ctx[sidx[i]].to_montgomery((uint32_t)v));
        }
        for (uint32_t i = 0; i < Ls; ++i)
            for (uint32_t j = 0; j < Ld; ++j)
                ca.push_back(ctx[didx[j]].to_montgomery((uint32_t)mod_small(shat[i], pm(didx[j]))));
        for (uint32_t i = 0; i < Ls; ++i) ca.push_back((uint32_t)pm(sidx[i]));
        for (uint32_t i = 0; i < Ls; ++i) ca.push_back(ctx[sidx[i]].q_inv_32());
        for (uint32_t j = 0; j < Ld; ++j) ca.push_back((uint32_t)pm(didx[j]));
        for (uint32_t j = 0; j < Ld; ++j) ca.push_back(ctx[didx[j]].q_inv_32());
        for (uint32_t base = 0; base < NBLK; base += cnt_cores) {
            uint32_t cnt = std::min<uint32_t>(cnt_cores, NBLK - base);
            emit([&]{ Program p = CreateProgram(); auto crs = crs_of(cnt);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,Ls,I32);
                mkcb(p,crs,CBIndex::c_2,tt::DataFormat::UInt32,Ls,I32);
                mkcb(p,crs,CBIndex::c_3,tt::DataFormat::UInt32,Ls,I32);
                mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,Ld,I32);
                auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_gather.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_scatter.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/baseconv_fbc.cpp", crs,
                    ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t k = 0; k < cnt; ++k) { uint32_t t = base + k; auto c = CC(k);
                    std::vector<uint32_t> ra; ra.push_back(Ls);
                    for (uint32_t i = 0; i < Ls; ++i) ra.push_back((uint32_t)src[i][t]->address());
                    SetRuntimeArgs(p, rd, c, ra);
                    std::vector<uint32_t> wa; wa.push_back(Ld); wa.push_back(Ld);
                    for (uint32_t j = 0; j < Ld; ++j) wa.push_back((uint32_t)dst[j][t]->address());
                    SetRuntimeArgs(p, wr, c, wa);
                    SetRuntimeArgs(p, cp, c, ca); }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; });
        }
    };

    // ---- EXACT SK conversion, any source/destination prime lists ----
    // Generalised so the lift (Q -> S) and the rescale's second phase (S -> Q) are the same builder
    // with different lists -- the algorithm does not care which bases they are, only that the sentinel
    // reports the value we want represented.
    auto build_skc = [&](const std::vector<uint32_t>& sidx, const std::vector<uint32_t>& didx,
                         const std::vector<std::vector<std::shared_ptr<distributed::MeshBuffer>>>& src,
                         const std::vector<std::shared_ptr<distributed::MeshBuffer>>& xsk,
                         const std::vector<std::vector<std::shared_ptr<distributed::MeshBuffer>>>& dst,
                         uint32_t cnt_cores) {
        const uint32_t Ls = (uint32_t)sidx.size(), Ld = (uint32_t)didx.size(), Ldt = Ld + 1;
        const u512 SP = prodOf(sidx);
        std::vector<uint32_t> ca; ca.push_back(Ls); ca.push_back(Ld);
        std::vector<u512> shat(Ls);
        for (uint32_t i = 0; i < Ls; ++i) { u512 r; divmod(SP, u512(pm(sidx[i])), shat[i], r); }
        for (uint32_t i = 0; i < Ls; ++i)                         // C1[i] = mont(shat_i^-1 mod s_i)
            ca.push_back(ctx[sidx[i]].to_montgomery(modinv(mod_small(shat[i], pm(sidx[i])), pm(sidx[i]))));
        for (uint32_t i = 0; i < Ls; ++i)                         // C2[i][j] = mont(shat_i mod p_j)
            for (uint32_t j = 0; j < Ldt; ++j) {
                uint32_t dj = (j < Ld) ? didx[j] : MSKI;
                ca.push_back(ctx[dj].to_montgomery((uint32_t)mod_small(shat[i], pm(dj))));
            }
        for (uint32_t i = 0; i < Ls; ++i) ca.push_back((uint32_t)pm(sidx[i]));
        for (uint32_t i = 0; i < Ls; ++i) ca.push_back(ctx[sidx[i]].q_inv_32());
        for (uint32_t j = 0; j < Ldt; ++j) ca.push_back((uint32_t)pm((j < Ld) ? didx[j] : MSKI));
        for (uint32_t j = 0; j < Ldt; ++j) ca.push_back(ctx[(j < Ld) ? didx[j] : MSKI].q_inv_32());
        for (uint32_t j = 0; j < Ld;  ++j)                        // G[j] = mont(S mod p_j)
            ca.push_back(ctx[didx[j]].to_montgomery((uint32_t)mod_small(SP, pm(didx[j]))));
        ca.push_back(ctx[MSKI].to_montgomery(modinv(mod_small(SP, bfv.Msk), bfv.Msk)));

        for (uint32_t base = 0; base < NBLK; base += cnt_cores) {
            uint32_t cnt = std::min<uint32_t>(cnt_cores, NBLK - base);
            emit([&]{
                Program p = CreateProgram(); auto crs = crs_of(cnt);
                // depths are the ACTUAL widths -- c_3 is reserved once per destination inside phase 2,
                // so a depth that is not a multiple of Ls straddles the buffer end (exp_Tb001 §4)
                mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, Ls + 1, I32);
                mkcb(p, crs, CBIndex::c_2, tt::DataFormat::UInt32, Ls, I32);
                mkcb(p, crs, CBIndex::c_3, tt::DataFormat::UInt32, Ls, I32);
                mkcb(p, crs, CBIndex::c_4, tt::DataFormat::UInt32, Ldt, I32);
                mkcb(p, crs, CBIndex::c_5, tt::DataFormat::UInt32, 1, I32);
                mkcb(p, crs, CBIndex::c_16, tt::DataFormat::UInt32, Ld, I32);
                auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_gather.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_scatter.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/baseconv_sk.cpp", crs,
                    ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t k = 0; k < cnt; ++k) {
                    uint32_t t = base + k; auto c = CC(k);
                    std::vector<uint32_t> ra; ra.push_back(Ls + 1);
                    for (uint32_t i = 0; i < Ls; ++i) ra.push_back((uint32_t)src[i][t]->address());
                    ra.push_back((uint32_t)xsk[t]->address());
                    SetRuntimeArgs(p, rd, c, ra);
                    std::vector<uint32_t> wa; wa.push_back(Ld); wa.push_back(Ld);
                    for (uint32_t j = 0; j < Ld; ++j) wa.push_back((uint32_t)dst[j][t]->address());
                    SetRuntimeArgs(p, wr, c, wa);
                    SetRuntimeArgs(p, cp, c, ca);
                }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; });
        }
    };

    // ---- the check: device INTT of the fixture's eval-form input == host from_eval ----
    using Buf = std::shared_ptr<distributed::MeshBuffer>;
    using Buf8 = std::vector<Buf>;
    auto mkBuf8 = [&]() -> Buf8 { Buf8 b(NBLK); for(uint32_t i=0;i<NBLK;++i) b[i]=mkbuf(1,I32); return b; };
    auto up_natE = [&](const std::vector<uint32_t>& x){ Buf8 b=mkBuf8();      // natural eval -> E layout
        for(uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> blk(ELEMS);
            for(uint32_t k2l=0;k2l<T;++k2l) for(uint32_t k1=0;k1<N1;++k1) blk[k2l*T+k1]=x[(bk2*T+k2l)+N2*k1];
            t_wr(cq,b[bk2],tilize32(blk),false);} return b; };
    auto rd_natC = [&](Buf8& b){ std::vector<uint32_t> x(N);                  // C'' layout -> natural coeff
        for(uint32_t bn2=0;bn2<NBLK;++bn2){ std::vector<uint32_t> raw; t_rd(cq,raw,b[bn2],true);
            auto blk=untilize32(raw); for(uint32_t n1=0;n1<N1;++n1) for(uint32_t n2l=0;n2l<T;++n2l) x[n1+32*(bn2*T+n2l)]=blk[n1*T+n2l]; } return x; };
    auto transpose8_dev = [&](Buf8& in, Buf8& out){ for(uint32_t i=0;i<NBLK;++i) transpose_dev(in[i],out[i]); };
    Buf8 ttmp = mkBuf8();
    auto INTT_res = [&](Buf8& inE, Buf8& outC, uint32_t li){ transpose8_dev(inE,ttmp); inv_buf(ttmp,outC,li); };

    // pointwise helpers (8 tiles over EWC cores), mirroring exp_Tb001's ew()
    auto ew2 = [&](const char* rdk, const char* cpk, Buf8& x, Buf8& y, Buf8& o,
                   std::vector<uint32_t> cargs, bool modmul) {
        for (uint32_t base = 0; base < NBLK; base += EWC) {
            uint32_t cnt = std::min<uint32_t>(EWC, NBLK - base);
            emit([&]{ Program p = CreateProgram(); auto crs = crs_of(cnt);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32);
                mkcb(p,crs,CBIndex::c_1,tt::DataFormat::UInt32,2,I32);
                mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto r = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + std::string(rdk), crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto w = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto c = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + std::string(cpk), crs,
                    ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t k = 0; k < cnt; ++k) { uint32_t t = base + k; auto cc = CC(k);
                    if (modmul) SetRuntimeArgs(p,r,cc,{(uint32_t)x[t]->address(),(uint32_t)y[t]->address(),1u,0u});
                    else        SetRuntimeArgs(p,r,cc,{(uint32_t)x[t]->address(),(uint32_t)y[t]->address(),1u,0u,0u});
                    SetRuntimeArgs(p,w,cc,{(uint32_t)o[t]->address(),1u,0u});
                    SetRuntimeArgs(p,c,cc,cargs); }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; });
        }
    };


    auto up_natC = [&](const std::vector<uint32_t>& x){ Buf8 b=mkBuf8();      // natural coeff -> C'' layout
        for(uint32_t bn2=0;bn2<NBLK;++bn2){ std::vector<uint32_t> blk(ELEMS);
            for(uint32_t n1=0;n1<N1;++n1) for(uint32_t n2l=0;n2l<T;++n2l) blk[n1*T+n2l]=x[n1+32*(bn2*T+n2l)];
            t_wr(cq,b[bn2],tilize32(blk),false);} return b; };
    auto rd_natE = [&](Buf8& b){ std::vector<uint32_t> x(N);                  // E layout -> natural eval
        for(uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> raw; t_rd(cq,raw,b[bk2],true);
            auto blk=untilize32(raw); for(uint32_t k2l=0;k2l<T;++k2l) for(uint32_t k1=0;k1<N1;++k1) x[(bk2*T+k2l)+N2*k1]=blk[k2l*T+k1]; } return x; };
    auto NTT_res = [&](Buf8& inC, Buf8& outE, uint32_t li){ transpose8_dev(inC,ttmp); fwd_buf(ttmp,outE,li); };

    // BOTH directions are checked. The coefficient-form contract makes the FORWARD transform the one
    // the multiply actually uses (coeff in -> eval for the tensor), but the rescale comes back through
    // the inverse, so a fault in either would surface downstream.
    Ciphertext ev = bfv.to_eval(kase.Ae);        // host reference for the forward direction
    uint32_t bad = 0, checked = 0, badi = 0;
    for (uint32_t poly = 0; poly < 2; ++poly) {
        const auto& src_c = poly == 0 ? kase.Ae.c0 : kase.Ae.c1;      // coefficient domain now
        const auto& want_e = poly == 0 ? ev.c0 : ev.c1;
        for (uint32_t li = 0; li < LQ; ++li) {
            Buf8 cbuf = up_natC(src_c[li]);
            Buf8 ebuf = mkBuf8();
            NTT_res(cbuf, ebuf, li);                                   // forward: coeff -> eval
            t_fin(cq);
            auto gote = rd_natE(ebuf);
            for (uint32_t n = 0; n < N; ++n) { ++checked; if (gote[n] != want_e[li][n]) ++bad; }

            Buf8 e2 = up_natE(want_e[li]);
            Buf8 c2 = mkBuf8();
            INTT_res(e2, c2, li);                                      // inverse: eval -> coeff
            t_fin(cq);
            auto gotc = rd_natC(c2);
            for (uint32_t n = 0; n < N; ++n) if (gotc[n] != src_c[li][n]) ++badi;
        }
    }
    fmt::print("[D1] device NTT (coeff->eval)  vs host   mismatches = {}/{}  [{}]\n",
               bad, checked, bad ? "FAIL" : "PASS");
    fmt::print("[D2] device INTT(eval->coeff)  vs host   mismatches = {}/{}  [{}]\n",
               badi, checked, badi ? "FAIL" : "PASS");
    // ---- [D3] STAGE 2: the SK lift on device vs the exact reference ----
    uint32_t bad3 = 0, tot3 = 0;
    {
        const auto& srcc = kase.Ae.c0;                 // one polynomial suffices to validate
        std::vector<std::vector<uint32_t>> want(LS, std::vector<uint32_t>(N));
        std::vector<uint32_t> xsk_h(N);
        std::vector<uint32_t> res(LQ);
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t i = 0; i < LQ; ++i) res[i] = srcc[i][n];
            u512 mag; bool neg; bfv.Qb.reconstruct_centered(res, mag, neg);
            uint64_t r = mod_small(mag, bfv.Msk);
            xsk_h[n] = (uint32_t)(neg ? (bfv.Msk - r) % bfv.Msk : r);       // CENTERED value's residue
            for (uint32_t j = 0; j < LS; ++j) {
                uint64_t rr = mod_small(mag, bfv.S[j]);
                want[j][n] = (uint32_t)(neg ? (bfv.S[j] - rr) % bfv.S[j] : rr);
            }
        }
        std::vector<Buf8> src(LQ);
        for (uint32_t i = 0; i < LQ; ++i) src[i] = up_natC(srcc[i]);
        Buf8 xsk = up_natC(xsk_h);
        std::vector<Buf8> dst(LS);
        for (uint32_t j = 0; j < LS; ++j) dst[j] = mkBuf8();
        std::vector<uint32_t> sidx(LQ), didx(LS);
        for (uint32_t i = 0; i < LQ; ++i) sidx[i] = i;
        for (uint32_t j = 0; j < LS; ++j) didx[j] = LQ + LP + j;
        build_skc(sidx, didx, src, xsk, dst, EWC);
        t_fin(cq);
        for (uint32_t j = 0; j < LS; ++j) {
            auto got = rd_natC(dst[j]);
            for (uint32_t n = 0; n < N; ++n) { ++tot3; if (got[n] != want[j][n]) ++bad3; }
        }
        fmt::print("[D3] device SK lift Q->S (exact, centered) vs reference   mismatches = {}/{}  [{}]\n",
                   bad3, tot3, bad3 ? "FAIL" : "PASS");
    }

    // ---- [D4] STAGE 3: the tensor in Q|S|m_sk, on device ----
    // Fed with the HOST's lifted values so this isolates the tensor from stage 2, which is verified
    // on its own above. modmul_sfpi_full is a MONTGOMERY multiply (a*b*R^-1), so the B side is
    // montified before upload; the NTT is Z_q-linear, so mont() rides through it unchanged.
    uint32_t bad4 = 0, tot4 = 0;
    if (std::getenv("TB001B_STAGE3")) {
        const uint32_t LT = LQ + LS, LTM = LT + 1;      // Q|S, then m_sk
        auto qpi = [&](uint32_t j){ return j < LQ ? j : (j < LT ? LQ + LP + (j - LQ) : LQ + LP + LS); };
        auto pmod = [&](uint32_t j){ return j < LQ ? bfv.Q[j] : (j < LT ? bfv.S[j - LQ] : bfv.Msk); };

        std::vector<std::vector<uint32_t>> re0, re1, re2;
        bfv.tensor_ref(kase.Ae, kase.Be, re0, re1, re2);

        // the 18-limb lifted inputs: Q|S from the reference lift, m_sk computed the same way
        auto lift18 = [&](const std::vector<std::vector<uint32_t>>& in) {
            auto t = bfv.lift_to_T(in);
            t.resize(LTM, std::vector<uint32_t>(N));
            std::vector<uint32_t> res(LQ);
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t i = 0; i < LQ; ++i) res[i] = in[i][n];
                u512 mag; bool neg; bfv.Qb.reconstruct_centered(res, mag, neg);
                uint64_t r = mod_small(mag, bfv.Msk);
                t[LT][n] = (uint32_t)(neg ? (bfv.Msk - r) % bfv.Msk : r);
            }
            return t;
        };
        // Expected m_sk residues of the tensor. The reference base is Q|S, so the sentinel limb has
        // no counterpart there -- reconstruct e exactly from the 17 T limbs, centre it, and reduce.
        // This limb is what stage 4's SK reads, so leaving it unchecked would defer a fault into the
        // rescale where it is far harder to localise.
        std::vector<std::vector<uint32_t>> wsk(3, std::vector<uint32_t>(N));
        {
            std::vector<uint32_t> rt(LT);
            const std::vector<std::vector<std::vector<uint32_t>>*> es{&re0, &re1, &re2};
            for (uint32_t k = 0; k < 3; ++k)
                for (uint32_t n = 0; n < N; ++n) {
                    for (uint32_t j = 0; j < LT; ++j) rt[j] = (*es[k])[j][n];
                    u512 mag; bool neg; bfv.Tb.reconstruct_centered(rt, mag, neg);
                    uint64_t r = mod_small(mag, bfv.Msk);
                    wsk[k][n] = (uint32_t)(neg ? (bfv.Msk - r) % bfv.Msk : r);
                }
        }
        auto A0 = lift18(kase.Ae.c0), A1 = lift18(kase.Ae.c1);
        auto B0 = lift18(kase.Be.c0), B1 = lift18(kase.Be.c1);

        for (uint32_t j = 0; j < LTM; ++j) {
            const uint32_t li = qpi(j); const uint64_t m = pmod(j);
            std::vector<uint32_t> pw{1u, (uint32_t)m, ctx[li].q_inv_32(),
                                     (uint32_t)(m & 0xFFFF), (uint32_t)(m >> 16)};
            std::vector<uint32_t> ad{1u, (uint32_t)m};
            auto up = [&](const std::vector<uint32_t>& v, bool mont) {
                std::vector<uint32_t> t(N);
                for (uint32_t n = 0; n < N; ++n) t[n] = mont ? ctx[li].to_montgomery(v[n]) : v[n];
                return up_natC(t); };
            Buf8 a0c = up(A0[j], false), a1c = up(A1[j], false);
            Buf8 b0c = up(B0[j], true),  b1c = up(B1[j], true);     // B side montified
            Buf8 a0e = mkBuf8(), a1e = mkBuf8(), b0e = mkBuf8(), b1e = mkBuf8();
            NTT_res(a0c, a0e, li); NTT_res(a1c, a1e, li);
            NTT_res(b0c, b0e, li); NTT_res(b1c, b1e, li);
            Buf8 E0 = mkBuf8(), E2 = mkBuf8(), t01 = mkBuf8(), t10 = mkBuf8(), E1 = mkBuf8();
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", a0e, b0e, E0, pw, true);
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", a1e, b1e, E2, pw, true);
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", a0e, b1e, t01, pw, true);
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", a1e, b0e, t10, pw, true);
            ew2("dataflow/reader_ab.cpp","compute/modadd_map.cpp", t01, t10, E1, ad, false);
            Buf8 c0 = mkBuf8(), c1 = mkBuf8(), c2 = mkBuf8();
            INTT_res(E0, c0, li); INTT_res(E1, c1, li); INTT_res(E2, c2, li);
            t_fin(cq);
            auto g0 = rd_natC(c0), g1 = rd_natC(c1), g2 = rd_natC(c2);
            for (uint32_t n = 0; n < N; ++n) {
                tot4 += 3;
                if (j < LT) {                              // Q|S limbs: against the reference tensor
                    if (g0[n] != re0[j][n]) ++bad4;
                    if (g1[n] != re1[j][n]) ++bad4;
                    if (g2[n] != re2[j][n]) ++bad4;
                } else {                                   // m_sk limb: against the exact e mod m_sk
                    if (g0[n] != wsk[0][n]) ++bad4;
                    if (g1[n] != wsk[1][n]) ++bad4;
                    if (g2[n] != wsk[2][n]) ++bad4;
                }
            }
        }
        fmt::print("[D4] device tensor in Q|S|m_sk vs reference   mismatches = {}/{}  [{}]\n",
                   bad4, tot4, bad4 ? "FAIL" : "PASS");
    }

    // ---- [D5] STAGE 4: the BEHZ rescale on device ----
    // Phase A: FBC Q -> S|m_sk with t folded into C1, then r_j = (t*e_j - A_j)*Q^-1 mod p_j.
    // Phase B: the SAME baseconv_sk as the lift, S -> Q, with phase A's m_sk channel as the sentinel.
    // That is why m_sk is carried through the tensor at all: phase A computes it so phase B can be exact.
    uint32_t bad5 = 0, tot5 = 0;
    if (std::getenv("TB001B_STAGE3")) {
        const uint32_t LT = LQ + LS;
        std::vector<uint32_t> qidx(LQ), sidx(LS), tsrc(LQ);
        for (uint32_t i = 0; i < LQ; ++i) { qidx[i] = i; tsrc[i] = i; }
        for (uint32_t j = 0; j < LS; ++j) sidx[j] = LQ + LP + j;
        std::vector<uint32_t> dSm = sidx; dSm.push_back(MSKI);          // S | m_sk, phase A's targets

        auto fillc = [&](uint32_t li, uint32_t v) {
            Buf8 b = mkBuf8(); std::vector<uint32_t> t(ELEMS, v); auto tl = tilize32(t);
            for (uint32_t i = 0; i < NBLK; ++i) t_wr(cq, b[i], tl, false); (void)li; return b; };

        std::vector<std::vector<uint32_t>> re0, re1, re2;
        bfv.tensor_ref(kase.Ae, kase.Be, re0, re1, re2);
        auto want = bfv.rescale_behz(re0);                              // the target, exactly

        // e in 18 limbs: Q|S from the reference tensor, m_sk reconstructed exactly
        std::vector<Buf8> eQ(LQ), eS(LS); Buf8 eM;
        {
            std::vector<uint32_t> rt(LT), msk(N);
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t j = 0; j < LT; ++j) rt[j] = re0[j][n];
                u512 mag; bool neg; bfv.Tb.reconstruct_centered(rt, mag, neg);
                uint64_t r = mod_small(mag, bfv.Msk);
                msk[n] = (uint32_t)(neg ? (bfv.Msk - r) % bfv.Msk : r);
            }
            for (uint32_t i = 0; i < LQ; ++i) eQ[i] = up_natC(re0[i]);
            for (uint32_t j = 0; j < LS; ++j) eS[j] = up_natC(re0[LQ + j]);
            eM = up_natC(msk);
        }

        // phase A
        std::vector<Buf8> A(LS + 1);
        for (uint32_t j = 0; j <= LS; ++j) A[j] = mkBuf8();
        build_fbc(tsrc, dSm, T_PLAIN, eQ, A, EWC);
        std::vector<Buf8> R(LS); Buf8 Rm;
        for (uint32_t j = 0; j <= LS; ++j) {
            uint32_t dj = dSm[j]; uint64_t m = pm(dj);
            std::vector<uint32_t> pw{1u,(uint32_t)m,ctx[dj].q_inv_32(),(uint32_t)(m&0xFFFF),(uint32_t)(m>>16)};
            Buf8& ej = (j < LS) ? eS[j] : eM;
            Buf8 te = mkBuf8(), df = mkBuf8(), rj = mkBuf8();
            Buf8 kt = fillc(dj, ctx[dj].to_montgomery((uint32_t)(T_PLAIN % m)));
            Buf8 kq = fillc(dj, ctx[dj].to_montgomery((uint32_t)modinv(mod_small(bfv.Qprod, m), m)));
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", ej, kt, te, pw, true);
            ew2("dataflow/reader_ab.cpp","compute/modsub_reduce.cpp", te, A[j], df, {(uint32_t)m}, false);
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", df, kq, rj, pw, true);
            if (j < LS) R[j] = rj; else Rm = rj;
        }
        // phase B: exact SK, S -> Q, sentinel = phase A's m_sk channel
        std::vector<Buf8> D(LQ);
        for (uint32_t i = 0; i < LQ; ++i) D[i] = mkBuf8();
        build_skc(sidx, qidx, R, Rm, D, EWC);
        t_fin(cq);
        for (uint32_t i = 0; i < LQ; ++i) {
            auto got = rd_natC(D[i]);
            for (uint32_t n = 0; n < N; ++n) { ++tot5; if (got[n] != want[i][n]) ++bad5; }
        }
        fmt::print("[D5] device BEHZ rescale floor(t*e/Q)-eps vs reference   mismatches = {}/{}  [{}]\n",
                   bad5, tot5, bad5 ? "FAIL" : "PASS");
    }

    // ---- [D6] STAGE 5: relinearise + combine, and the whole multiply end to end ----
    uint32_t bad6 = 0, tot6 = 0;
    if (std::getenv("TB001B_STAGE3")) {
        std::vector<uint32_t> qidx(LQ), pidx(LP);
        for (uint32_t i = 0; i < LQ; ++i) qidx[i] = i;
        for (uint32_t j = 0; j < LP; ++j) pidx[j] = LQ + j;
        const uint32_t LQP2 = LQ + LP;
        auto fillc = [&](uint32_t v) {
            Buf8 b = mkBuf8(); std::vector<uint32_t> t(ELEMS, v); auto tl = tilize32(t);
            for (uint32_t i = 0; i < NBLK; ++i) t_wr(cq, b[i], tl, false); return b; };

        std::vector<std::vector<uint32_t>> d0, d1, d2;
        bfv.rescaled_ref(kase.Ae, kase.Be, d0, d1, d2);

        // d2's sentinel residue, from its centered value in base Q
        std::vector<uint32_t> d2m(N), res(LQ);
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t i = 0; i < LQ; ++i) res[i] = d2[i][n];
            u512 mag; bool neg; bfv.Qb.reconstruct_centered(res, mag, neg);
            uint64_t r = mod_small(mag, bfv.Msk);
            d2m[n] = (uint32_t)(neg ? (bfv.Msk - r) % bfv.Msk : r);
        }
        std::vector<Buf8> D2q(LQ); for (uint32_t i = 0; i < LQ; ++i) D2q[i] = up_natC(d2[i]);
        Buf8 D2m = up_natC(d2m);

        // 1. mod-up d2 from Q into P, approximate (see the note in multiply(): the exact centered
        //    lift does not compose, and this error is 0 mod Q in the output phase)
        std::vector<Buf8> D2p(LP); for (uint32_t j = 0; j < LP; ++j) D2p[j] = mkBuf8();
        build_fbc(qidx, pidx, 1, D2q, D2p, EWC);
        (void)D2m;

        // 2. transform d2 and the evk into the eval domain over QP; 3. MAC; 4. back to coefficients
        std::vector<Buf8> V0(LQP2), V1(LQP2);
        for (uint32_t j = 0; j < LQP2; ++j) {
            const uint32_t li = j; const uint64_t m = pm(li);
            std::vector<uint32_t> pw{1u,(uint32_t)m,ctx[li].q_inv_32(),(uint32_t)(m&0xFFFF),(uint32_t)(m>>16)};
            Buf8 dc = (j < LQ) ? D2q[j] : D2p[j - LQ];
            auto upm = [&](const std::vector<uint32_t>& v) {          // montified: pw is Montgomery
                std::vector<uint32_t> t(N);
                for (uint32_t n = 0; n < N; ++n) t[n] = ctx[li].to_montgomery(v[n]);
                return up_natC(t); };
            Buf8 e0c = upm(kase.evk0[j]), e1c = upm(kase.evk1[j]);
            Buf8 de = mkBuf8(), E0 = mkBuf8(), E1 = mkBuf8();
            NTT_res(dc, de, li); NTT_res(e0c, E0, li); NTT_res(e1c, E1, li);
            Buf8 p0 = mkBuf8(), p1 = mkBuf8();
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", de, E0, p0, pw, true);
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", de, E1, p1, pw, true);
            V0[j] = mkBuf8(); V1[j] = mkBuf8();
            INTT_res(p0, V0[j], li); INTT_res(p1, V1[j], li);
        }

        // 5. mod down by P: FBC P -> Q, then k_i = (v_i - A_i) * P^-1 mod q_i
        u512 PP(1); for (auto q : bfv.P) PP = mul_small(PP, q);
        auto moddown_dev = [&](std::vector<Buf8>& V) {
            std::vector<Buf8> vp(LP), A(LQ), K(LQ);
            for (uint32_t j = 0; j < LP; ++j) vp[j] = V[LQ + j];
            for (uint32_t i = 0; i < LQ; ++i) A[i] = mkBuf8();
            build_fbc(pidx, qidx, 1, vp, A, EWC);
            for (uint32_t i = 0; i < LQ; ++i) {
                uint64_t m = bfv.Q[i];
                std::vector<uint32_t> pw{1u,(uint32_t)m,ctx[i].q_inv_32(),(uint32_t)(m&0xFFFF),(uint32_t)(m>>16)};
                Buf8 df = mkBuf8(); K[i] = mkBuf8();
                Buf8 kp = fillc(ctx[i].to_montgomery((uint32_t)modinv(mod_small(PP, m), m)));
                ew2("dataflow/reader_ab.cpp","compute/modsub_reduce.cpp", V[i], A[i], df, {(uint32_t)m}, false);
                ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", df, kp, K[i], pw, true);
            }
            return K; };
        auto K0 = moddown_dev(V0), K1 = moddown_dev(V1);

        // 6. combine
        std::vector<Buf8> R0(LQ), R1(LQ);
        for (uint32_t i = 0; i < LQ; ++i) {
            uint64_t m = bfv.Q[i];
            Buf8 a0 = up_natC(d0[i]), a1 = up_natC(d1[i]);
            R0[i] = mkBuf8(); R1[i] = mkBuf8();
            ew2("dataflow/reader_ab.cpp","compute/modadd_map.cpp", a0, K0[i], R0[i], {1u,(uint32_t)m}, false);
            ew2("dataflow/reader_ab.cpp","compute/modadd_map.cpp", a1, K1[i], R1[i], {1u,(uint32_t)m}, false);
        }
        t_fin(cq);
        for (uint32_t i = 0; i < LQ; ++i) {
            auto g0 = rd_natC(R0[i]), g1 = rd_natC(R1[i]);
            for (uint32_t n = 0; n < N; ++n) {
                tot6 += 2;
                if (g0[n] != kase.Re.c0[i][n]) ++bad6;
                if (g1[n] != kase.Re.c1[i][n]) ++bad6;
            }
        }
        fmt::print("[D6] device relin+combine vs the FIXED case output   mismatches = {}/{}  [{}]\n",
                   bad6, tot6, bad6 ? "FAIL" : "PASS");
    }

    // ---- [D7] THE WHOLE MULTIPLY ON DEVICE, from its own intermediates ----
    // Every stage above is fed reference inputs, which is what makes a fault localise. This runs the
    // chain: lift -> tensor -> rescale -> relinearise -> combine, with nothing coming back to the host
    // in between, and compares against the FIXED case output.
    uint32_t bad7 = 0, tot7 = 0; bool ran7 = false;
    if (std::getenv("TB001B_CHAIN")) {
        ran7 = true;
        const uint32_t LT = LQ + LS, LTM = LT + 1;
        std::vector<uint32_t> qidx(LQ), sidx(LS), pidx(LP), dSm;
        for (uint32_t i = 0; i < LQ; ++i) qidx[i] = i;
        for (uint32_t j = 0; j < LS; ++j) sidx[j] = LQ + LP + j;
        for (uint32_t j = 0; j < LP; ++j) pidx[j] = LQ + j;
        dSm = sidx; dSm.push_back(MSKI);
        auto fillc = [&](uint32_t v) {
            Buf8 b = mkBuf8(); std::vector<uint32_t> t(ELEMS, v); auto tl = tilize32(t);
            for (uint32_t i = 0; i < NBLK; ++i) t_wr(cq, b[i], tl, false); return b; };
        auto PWA = [&](uint32_t li){ uint64_t m = pm(li);
            return std::vector<uint32_t>{1u,(uint32_t)m,ctx[li].q_inv_32(),
                                         (uint32_t)(m&0xFFFF),(uint32_t)(m>>16)}; };

        // ---- 1. lift a base-Q polynomial into the 18-limb tensor base ----
        // The Q limbs are the input unchanged (x and x_c are the same class mod Q) and the m_sk limb
        // IS the carried sentinel, so only the 9 S limbs are computed.
        auto lift18 = [&](const std::vector<std::vector<uint32_t>>& c) {
            std::vector<uint32_t> xsk(N), res(LQ);
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t i = 0; i < LQ; ++i) res[i] = c[i][n];
                u512 mag; bool neg; bfv.Qb.reconstruct_centered(res, mag, neg);
                uint64_t r = mod_small(mag, bfv.Msk);
                xsk[n] = (uint32_t)(neg ? (bfv.Msk - r) % bfv.Msk : r);
            }
            std::vector<Buf8> src(LQ); for (uint32_t i = 0; i < LQ; ++i) src[i] = up_natC(c[i]);
            Buf8 xs = up_natC(xsk);
            std::vector<Buf8> S9(LS); for (uint32_t j = 0; j < LS; ++j) S9[j] = mkBuf8();
            build_skc(qidx, sidx, src, xs, S9, EWC);
            std::vector<Buf8> out(LTM);
            for (uint32_t i = 0; i < LQ; ++i) out[i] = src[i];
            for (uint32_t j = 0; j < LS; ++j) out[LQ + j] = S9[j];
            out[LT] = xs;
            return out; };
        auto tli = [&](uint32_t j){ return j < LQ ? j : (j < LT ? LQ + LP + (j - LQ) : MSKI); };

        auto A0 = lift18(kase.Ae.c0), A1 = lift18(kase.Ae.c1);
        auto B0 = lift18(kase.Be.c0), B1 = lift18(kase.Be.c1);

        // ---- 2-4. transform, tensor, transform back ----
        std::vector<std::vector<Buf8>> E(3, std::vector<Buf8>(LTM));
        for (uint32_t j = 0; j < LTM; ++j) {
            const uint32_t li = tli(j); auto pw = PWA(li);
            Buf8 R2 = fillc(ctx[li].to_montgomery(ctx[li].to_montgomery(1)));   // pw(x,R^2) = mont(x)
            Buf8 b0m = mkBuf8(), b1m = mkBuf8();
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", B0[j], R2, b0m, pw, true);
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", B1[j], R2, b1m, pw, true);
            Buf8 ae0 = mkBuf8(), ae1 = mkBuf8(), be0 = mkBuf8(), be1 = mkBuf8();
            NTT_res(A0[j], ae0, li); NTT_res(A1[j], ae1, li);
            NTT_res(b0m, be0, li);   NTT_res(b1m, be1, li);
            Buf8 e0 = mkBuf8(), e2 = mkBuf8(), t01 = mkBuf8(), t10 = mkBuf8(), e1 = mkBuf8();
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", ae0, be0, e0, pw, true);
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", ae1, be1, e2, pw, true);
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", ae0, be1, t01, pw, true);
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", ae1, be0, t10, pw, true);
            ew2("dataflow/reader_ab.cpp","compute/modadd_map.cpp", t01, t10, e1, {1u,(uint32_t)pm(li)}, false);
            E[0][j] = mkBuf8(); E[1][j] = mkBuf8(); E[2][j] = mkBuf8();
            INTT_res(e0, E[0][j], li); INTT_res(e1, E[1][j], li); INTT_res(e2, E[2][j], li);
        }

        // ---- 5. rescale: returns d in base Q, plus d's m_sk residue (free from phase A) ----
        auto rescale_dev = [&](std::vector<Buf8>& e, Buf8& dmsk) {
            std::vector<Buf8> eQ(LQ), A(LS + 1), R(LS);
            for (uint32_t i = 0; i < LQ; ++i) eQ[i] = e[i];
            for (uint32_t j = 0; j <= LS; ++j) A[j] = mkBuf8();
            build_fbc(qidx, dSm, T_PLAIN, eQ, A, EWC);
            Buf8 Rm;
            for (uint32_t j = 0; j <= LS; ++j) {
                uint32_t dj = dSm[j]; uint64_t m = pm(dj); auto pw = PWA(dj);
                Buf8& ej = (j < LS) ? e[LQ + j] : e[LT];
                Buf8 te = mkBuf8(), df = mkBuf8(), rj = mkBuf8();
                Buf8 kt = fillc(ctx[dj].to_montgomery((uint32_t)(T_PLAIN % m)));
                Buf8 kq = fillc(ctx[dj].to_montgomery((uint32_t)modinv(mod_small(bfv.Qprod, m), m)));
                ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", ej, kt, te, pw, true);
                ew2("dataflow/reader_ab.cpp","compute/modsub_reduce.cpp", te, A[j], df, {(uint32_t)m}, false);
                ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", df, kq, rj, pw, true);
                if (j < LS) R[j] = rj; else Rm = rj;
            }
            std::vector<Buf8> D(LQ); for (uint32_t i = 0; i < LQ; ++i) D[i] = mkBuf8();
            build_skc(sidx, qidx, R, Rm, D, EWC);
            dmsk = Rm; return D; };
        Buf8 m0, m1, m2;
        auto D0 = rescale_dev(E[0], m0), D1 = rescale_dev(E[1], m1), D2 = rescale_dev(E[2], m2);

        // ---- 6. relinearise d2 ----
        std::vector<Buf8> D2p(LP); for (uint32_t j = 0; j < LP; ++j) D2p[j] = mkBuf8();
        build_fbc(qidx, pidx, 1, D2, D2p, EWC);           // APPROXIMATE mod-up -- it composes; see above
        (void)m2;
        const uint32_t LQP2 = LQ + LP;
        std::vector<Buf8> V0(LQP2), V1(LQP2);
        for (uint32_t j = 0; j < LQP2; ++j) {
            const uint32_t li = j; auto pw = PWA(li);
            Buf8 dc = (j < LQ) ? D2[j] : D2p[j - LQ];
            auto upm = [&](const std::vector<uint32_t>& v) {
                std::vector<uint32_t> t(N);
                for (uint32_t n = 0; n < N; ++n) t[n] = ctx[li].to_montgomery(v[n]);
                return up_natC(t); };
            Buf8 e0c = upm(kase.evk0[j]), e1c = upm(kase.evk1[j]);
            Buf8 de = mkBuf8(), K0e = mkBuf8(), K1e = mkBuf8();
            NTT_res(dc, de, li); NTT_res(e0c, K0e, li); NTT_res(e1c, K1e, li);
            Buf8 p0 = mkBuf8(), p1 = mkBuf8();
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", de, K0e, p0, pw, true);
            ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", de, K1e, p1, pw, true);
            V0[j] = mkBuf8(); V1[j] = mkBuf8();
            INTT_res(p0, V0[j], li); INTT_res(p1, V1[j], li);
        }
        u512 PP(1); for (auto q : bfv.P) PP = mul_small(PP, q);
        auto moddown_dev = [&](std::vector<Buf8>& V) {
            std::vector<Buf8> vp(LP), A(LQ), K(LQ);
            for (uint32_t j = 0; j < LP; ++j) vp[j] = V[LQ + j];
            for (uint32_t i = 0; i < LQ; ++i) A[i] = mkBuf8();
            build_fbc(pidx, qidx, 1, vp, A, EWC);
            for (uint32_t i = 0; i < LQ; ++i) {
                uint64_t m = bfv.Q[i]; auto pw = PWA(i);
                Buf8 df = mkBuf8(); K[i] = mkBuf8();
                Buf8 kp = fillc(ctx[i].to_montgomery((uint32_t)modinv(mod_small(PP, m), m)));
                ew2("dataflow/reader_ab.cpp","compute/modsub_reduce.cpp", V[i], A[i], df, {(uint32_t)m}, false);
                ew2("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp", df, kp, K[i], pw, true);
            }
            return K; };
        auto K0 = moddown_dev(V0), K1 = moddown_dev(V1);

        // ---- 7. combine ----
        std::vector<Buf8> R0(LQ), R1(LQ);
        for (uint32_t i = 0; i < LQ; ++i) {
            uint64_t m = bfv.Q[i];
            R0[i] = mkBuf8(); R1[i] = mkBuf8();
            ew2("dataflow/reader_ab.cpp","compute/modadd_map.cpp", D0[i], K0[i], R0[i], {1u,(uint32_t)m}, false);
            ew2("dataflow/reader_ab.cpp","compute/modadd_map.cpp", D1[i], K1[i], R1[i], {1u,(uint32_t)m}, false);
        }
        t_fin(cq);
        Ciphertext got; got.c0.resize(LQ); got.c1.resize(LQ);
        for (uint32_t i = 0; i < LQ; ++i) {
            got.c0[i] = rd_natC(R0[i]); got.c1[i] = rd_natC(R1[i]);
            for (uint32_t n = 0; n < N; ++n) {
                tot7 += 2;
                if (got.c0[i][n] != kase.Re.c0[i][n]) ++bad7;
                if (got.c1[i][n] != kase.Re.c1[i][n]) ++bad7;
            }
        }
        fmt::print("[D7] WHOLE MULTIPLY on device vs the FIXED case   mismatches = {}/{}  [{}]\n",
                   bad7, tot7, bad7 ? "FAIL" : "PASS");
        auto dec = bfv.decrypt(got);
        uint32_t badd = 0;
        for (uint32_t n = 0; n < N; ++n) if (dec[n] != kase.expect[n]) ++badd;
        fmt::print("[D8] decrypt(device output) == m1*m2             mismatches = {}/{}  [{}]\n",
                   badd, N, badd ? "FAIL" : "PASS");
        bad7 += badd;
    }

    bool ok = (bad == 0 && badi == 0 && bad3 == 0 && bad4 == 0 && bad5 == 0 && bad6 == 0 && bad7 == 0);
    (void)ran7;
    ok &= mesh->close();
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const uint64_t seed = (argc > 1) ? std::stoull(argv[1]) : 0;

    BFV bfv(seed);
    std::printf("exp_Tb001b  CORRECT BFV, decrypt-verified   N=%u  LQ=%u (%d-bit Q)  LP=%u  LS=%u  t=%u\n",
                N, LQ, bitlen(bfv.Qprod), LP, LS, T_PLAIN);
    std::printf("  Q limbs:"); for (auto q : bfv.Q) std::printf(" %llu", (unsigned long long)q); std::printf("\n");
    std::printf("  seed=%llu   secret key ternary, error uniform in [-2,2]\n", (unsigned long long)seed);

    std::mt19937_64 mr(seed + 12345);
    std::vector<int64_t> m1(N), m2(N);
    for (auto& x : m1) x = (int64_t)(mr() % T_PLAIN);
    for (auto& x : m2) x = (int64_t)(mr() % T_PLAIN);

    bool pass = true;

    // ---- 1. encrypt / decrypt round trip, all N coefficients ----
    {
        auto ct = bfv.encrypt(m1);
        auto got = bfv.decrypt(ct);
        uint32_t bad = 0;
        for (uint32_t n = 0; n < N; ++n) if (got[n] != m1[n]) ++bad;
        std::printf("[1] decrypt(encrypt(m))          mismatches = %u/%u  [%s]\n",
                    bad, N, bad ? "FAIL" : "PASS");
        pass &= (bad == 0);
    }

    // ---- 2. HE add: decrypt(enc(m1) + enc(m2)) == m1 + m2 mod t ----
    {
        auto a = bfv.encrypt(m1), b = bfv.encrypt(m2);
        auto got = bfv.decrypt(bfv.add_ct(a, b));
        uint32_t bad = 0;
        for (uint32_t n = 0; n < N; ++n) {
            int64_t want = (m1[n] + m2[n]) % T_PLAIN;
            if (got[n] != want) ++bad;
        }
        std::printf("[2] decrypt(enc(m1) + enc(m2))   mismatches = %u/%u  [%s]\n",
                    bad, N, bad ? "FAIL" : "PASS");
        pass &= (bad == 0);
    }

    // ---- 3a. write / verify the FIXED reference case (eval-form I/O) ----
    // TB001B_GEN=<path> regenerates it; otherwise the file is READ and everything below is checked
    // against its stored bytes, so the check does not depend on this process's RNG at all.
    {
        const char* gen = std::getenv("TB001B_GEN");
        const char* path = gen ? gen : (argc > 2 ? argv[2] : "reference/he/tb001b_case.bin");
        if (gen) {
            Case c;
            c.sk = bfv.s; c.m1 = m1; c.m2 = m2; c.expect = plain_mul(m1, m2);
            c.evk0 = bfv.evk0; c.evk1 = bfv.evk1;
            c.Ae = bfv.encrypt(m1);                      // COEFFICIENT domain -- the device contract
            c.Be = bfv.encrypt(m2);
            c.Re = bfv.multiply(c.Ae, c.Be);
            if (!write_case(path, bfv, c)) { std::printf("could not write %s\n", path); return 1; }
            std::printf("[3a] wrote reference case -> %s\n", path);
        }
        Case c;
        if (!read_case(path, c)) {
            std::printf("[3a] reference case %s missing or stale — regenerate with "
                        "TB001B_GEN=%s ./build/exp_tb001b %llu\n", path, path, (unsigned long long)seed);
            return 1;
        }
        // Adopt the case's keys, so the multiply below is checked against the FILE and not against
        // whatever this process's RNG produced. If the parameters or keygen ever drift, this is where
        // it shows up -- as a mismatch against bytes on disk rather than as a silent pass.
        bfv.s = c.sk; bfv.evk0 = c.evk0; bfv.evk1 = c.evk1;
        bool same_primes = (c.Q == bfv.Q && c.P == bfv.P && c.S == bfv.S);
        // Both I/O paths must land on the same coefficient-domain ciphertext. Running the eval path
        // through to_eval/from_eval and comparing against the coefficient path is the A/B for the
        // 16-transform difference: same answer, different transform count.
        const bool EVAL_IO = std::getenv("TB001B_EVAL_IN") != nullptr;
        auto got = EVAL_IO ? bfv.from_eval(bfv.multiply_eval(bfv.to_eval(c.Ae), bfv.to_eval(c.Be)))
                           : bfv.multiply(c.Ae, c.Be);
        uint32_t bad = 0;
        for (uint32_t i = 0; i < LQ; ++i)
            for (uint32_t n = 0; n < N; ++n) {
                if (got.c0[i][n] != c.Re.c0[i][n]) ++bad;
                if (got.c1[i][n] != c.Re.c1[i][n]) ++bad;
            }
        std::printf("[3a] host vs FIXED case (%s I/O)  primes %s  ciphertext mismatches = %u/%u  [%s]\n",
                    EVAL_IO ? "eval" : "coeff", same_primes ? "match" : "DIFFER",
                    bad, 2 * LQ * N, (bad || !same_primes) ? "FAIL" : "PASS");
        pass &= (bad == 0 && same_primes);

        // and the same case decrypts to the stored plaintext product
        auto dec = bfv.decrypt(c.Re);            // c.Re is already coefficient domain
        uint32_t badd = 0;
        for (uint32_t n = 0; n < N; ++n) if (dec[n] != c.expect[n]) ++badd;
        std::printf("[3b] decrypt(case output) == m1*m2   mismatches = %u/%u  [%s]\n",
                    badd, N, badd ? "FAIL" : "PASS");
        pass &= (badd == 0);
    }

    // ---- 3. THE TEST THAT MATTERS: decrypt(enc(m1) (x) enc(m2)) == m1*m2 in R_t ----
    // Nothing but a semantically correct BFV multiply passes this. It is a different KIND of check
    // from exp_Tb001's "device == CPU oracle": that one says the device computed what the host meant,
    // this one says what the host meant is a ciphertext of the product.
    {
        auto a = bfv.encrypt(m1), b = bfv.encrypt(m2);
        auto want = plain_mul(m1, m2);
        auto got = bfv.decrypt(bfv.multiply(a, b));
        uint32_t bad = 0;
        for (uint32_t n = 0; n < N; ++n) if (got[n] != want[n]) ++bad;
        std::printf("[3] decrypt(enc(m1) (x) enc(m2))  mismatches = %u/%u  [%s]\n",
                    bad, N, bad ? "FAIL" : "PASS");
        // Non-vacuity: show that the compared values are real. `want` comes from a schoolbook integer
        // convolution of the plaintexts and `got` from the RNS+bignum HE pipeline -- entirely
        // disjoint code paths, so agreement on uniform inputs is not something a bug can fake.
        std::printf("      sample  m1[0]=%lld m2[0]=%lld -> want[0]=%lld got[0]=%lld"
                    "   want[1]=%lld got[1]=%lld   want[8191]=%lld got[8191]=%lld\n",
                    (long long)m1[0], (long long)m2[0], (long long)want[0], (long long)got[0],
                    (long long)want[1], (long long)got[1],
                    (long long)want[N-1], (long long)got[N-1]);
        if (bad) for (uint32_t n = 0, shown = 0; n < N && shown < 5; ++n)
            if (got[n] != want[n]) { std::printf("      n=%u got=%lld want=%lld\n",
                                                 n, (long long)got[n], (long long)want[n]); ++shown; }
        pass &= (bad == 0);
    }

    // ---- TB001B_BEHZ: how big is BEHZ's rescale error, and does the scheme survive it? ----
    // The validated N=1024 path (host/lib/he/he_api.cpp) derives the scaled result from an APPROXIMATE
    // base conversion, so it computes floor(t*e/Q) - eps rather than the exact value. eps is the CRT
    // overflow of that conversion. Modelled here with exact big integers: compute eps exactly, apply
    // it, and see (a) how large it gets and (b) whether the multiply still decrypts.
    if (std::getenv("TB001B_BEHZ")) {
        Case c;
        if (!read_case("reference/he/tb001b_case.bin", c)) {
            std::printf("no case file at reference/he/tb001b_case.bin -- it is not committed "
                        "(5.8 MB, regenerable). Build it with:\n"
                        "  TB001B_GEN=reference/he/tb001b_case.bin ./build/exp_tb001b 0\n");
            return 1; }
        bfv.s = c.sk; bfv.evk0 = c.evk0; bfv.evk1 = c.evk1;
        std::vector<std::vector<uint32_t>> e0, e1, e2;
        bfv.tensor_ref(c.Ae, c.Be, e0, e1, e2);
        const size_t LT = bfv.Tb.p.size();
        // eps for one coefficient of one polynomial: the overflow of the FBC of [t*e]_Q from Q to S
        std::vector<u512> qhat(LQ); std::vector<uint64_t> qhinv(LQ);
        for (uint32_t i = 0; i < LQ; ++i) {
            u512 qh, r; divmod(bfv.Qprod, u512(bfv.Q[i]), qh, r);
            qhat[i] = qh; qhinv[i] = modinv(mod_small(qh, bfv.Q[i]), bfv.Q[i]);
        }
        uint64_t hist[16] = {0}; uint32_t maxeps = 0; uint64_t tot = 0;
        std::vector<uint32_t> rt(LT);
        const std::vector<std::vector<std::vector<uint32_t>>*> es{&e0, &e1, &e2};
        for (uint32_t k = 0; k < 3; ++k)
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t j = 0; j < LT; ++j) rt[j] = (*es[k])[j][n];
                u512 mag; bool neg; bfv.Tb.reconstruct_centered(rt, mag, neg);
                // y_i = (t*e * qhat_i^-1) mod q_i, on the value as it appears mod Q
                u512 acc;
                for (uint32_t i = 0; i < LQ; ++i) {
                    uint64_t ei = mod_small(mag, bfv.Q[i]);
                    if (neg) ei = (bfv.Q[i] - ei) % bfv.Q[i];
                    uint64_t te = (uint64_t)((u128)ei * (T_PLAIN % bfv.Q[i]) % bfv.Q[i]);
                    uint64_t y  = (uint64_t)((u128)te * qhinv[i] % bfv.Q[i]);
                    acc = add(acc, mul_small(qhat[i], y));
                }
                uint32_t eps = 0;
                while (cmp(acc, bfv.Qprod) >= 0) { acc = sub(acc, bfv.Qprod); ++eps; }
                if (eps > maxeps) maxeps = eps;
                if (eps < 16) ++hist[eps];
                ++tot;
            }
        std::printf("\nBEHZ rescale overflow eps (exact), over %llu coefficients:\n", (unsigned long long)tot);
        for (uint32_t v = 0; v < 16; ++v) if (hist[v])
            std::printf("   eps=%2u : %8llu  (%5.2f%%)\n", v, (unsigned long long)hist[v],
                        100.0 * (double)hist[v] / (double)tot);
        std::printf("   max eps = %u   (bounded by L_Q = %u)\n", maxeps, LQ);
        std::printf("   -> the device rescale lands at floor(t*e/Q) - eps, so it CANNOT match the exact\n"
                    "      reference bit-for-bit; the noise budget is ~2^209 and eps <= %u, so the scheme\n"
                    "      is unaffected. Bit-exact checking must target a BEHZ-modelling reference.\n\n",
                    maxeps);
    }

    // ---- TB001B_SKLIFT: the SK lift algorithm, on the host, vs the exact reference ----
    // Settles the centering question before any kernel work. CLAIM: if the sentinel channel carries
    // the CENTERED value's residue while the q-residues carry the same class, SK's eps-recovery
    // absorbs the centering wrap and the correction lands on x_c mod p_j directly -- no offset trick,
    // no sign detection, no magnitude comparison.
    //
    // Why it should work: A_sk = (x + eps*Q) mod m_sk carries the CRT overflow, while the sentinel
    // reports x_c = x - w*Q. Their difference is (eps + w)*Q, so the recovered correction is eps+w,
    // and subtracting (eps+w)*Q from the combine gives x - w*Q = x_c. The wrap and the overflow are
    // the same kind of quantity, so one subtraction removes both.
    if (std::getenv("TB001B_SKLIFT")) {
        Case c;
        if (!read_case("reference/he/tb001b_case.bin", c)) {
            std::printf("no case file at reference/he/tb001b_case.bin -- it is not committed "
                        "(5.8 MB, regenerable). Build it with:\n"
                        "  TB001B_GEN=reference/he/tb001b_case.bin ./build/exp_tb001b 0\n");
            return 1; }
        const uint64_t msk = bfv.Msk;
        std::vector<u512> qhat(LQ); std::vector<uint64_t> qhinv(LQ), qhat_sk(LQ);
        for (uint32_t i = 0; i < LQ; ++i) {
            u512 qh, r; divmod(bfv.Qprod, u512(bfv.Q[i]), qh, r);
            qhat[i] = qh;
            qhinv[i] = modinv(mod_small(qh, bfv.Q[i]), bfv.Q[i]);
            qhat_sk[i] = mod_small(qh, msk);
        }
        const uint64_t Qinv_sk = modinv(mod_small(bfv.Qprod, msk), msk);
        // destination base = S
        std::vector<std::vector<uint64_t>> qhat_p(LQ, std::vector<uint64_t>(LS));
        std::vector<uint64_t> Q_p(LS);
        for (uint32_t j = 0; j < LS; ++j) {
            Q_p[j] = mod_small(bfv.Qprod, bfv.S[j]);
            for (uint32_t i = 0; i < LQ; ++i) qhat_p[i][j] = mod_small(qhat[i], bfv.S[j]);
        }
        uint64_t bad = 0, tot = 0;
        std::vector<uint32_t> res(LQ);
        for (auto* src : std::vector<const std::vector<std::vector<uint32_t>>*>{&c.Ae.c0, &c.Ae.c1}) {
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t i = 0; i < LQ; ++i) res[i] = (*src)[i][n];
                // the reference: exact centered reconstruction
                u512 mag; bool neg; bfv.Qb.reconstruct_centered(res, mag, neg);
                // the sentinel channel: the CENTERED value mod m_sk
                uint64_t r_sk = mod_small(mag, msk);
                uint64_t x_sk = neg ? (msk - r_sk) % msk : r_sk;
                // --- SK ---
                std::vector<uint64_t> y(LQ);
                u128 asum = 0;
                for (uint32_t i = 0; i < LQ; ++i) {
                    y[i] = (uint64_t)((u128)res[i] * qhinv[i] % bfv.Q[i]);
                    asum += (u128)y[i] * qhat_sk[i] % msk;
                }
                uint64_t A_sk = (uint64_t)(asum % msk);
                uint64_t eps = (uint64_t)((u128)((A_sk + msk - x_sk) % msk) * Qinv_sk % msk);
                for (uint32_t j = 0; j < LS; ++j) {
                    u128 acc = 0;
                    for (uint32_t i = 0; i < LQ; ++i) acc += (u128)y[i] * qhat_p[i][j] % bfv.S[j];
                    uint64_t A_j = (uint64_t)(acc % bfv.S[j]);
                    uint64_t out = (uint64_t)((A_j + (u128)bfv.S[j] * 16 - (u128)eps * Q_p[j] % bfv.S[j]) % bfv.S[j]);
                    // the reference value for this destination limb
                    uint64_t rr = mod_small(mag, bfv.S[j]);
                    uint64_t want = neg ? (bfv.S[j] - rr) % bfv.S[j] : rr;
                    ++tot; if (out != want) ++bad;
                }
            }
        }
        std::printf("\n[SK] host SK lift (centered sentinel) vs exact reference: %llu/%llu wrong  [%s]\n",
                    (unsigned long long)bad, (unsigned long long)tot, bad ? "FAIL" : "PASS");
        std::printf("     eps range check: m_sk=%llu must exceed L+1=%u  -> %s\n\n",
                    (unsigned long long)msk, LQ + 1, msk > LQ + 1 ? "ok" : "TOO SMALL");
    }

    // ---- TB001B_USWEEP: how exact is the CRNS fixed-point eps, as a function of u? ----
    // Pure numerics, no device. The lift's inputs are ciphertext coefficients spanning all of [0,Q),
    // which is the case in question. Ground truth is the exact CRT reconstruction, so this measures
    // the METHOD, not an implementation of it.
    if (std::getenv("TB001B_USWEEP")) {
        Case c;
        if (!read_case("reference/he/tb001b_case.bin", c)) {
            std::printf("no case file at reference/he/tb001b_case.bin -- it is not committed "
                        "(5.8 MB, regenerable). Build it with:\n"
                        "  TB001B_GEN=reference/he/tb001b_case.bin ./build/exp_tb001b 0\n");
            return 1; }
        std::vector<u512> qhat(LQ); std::vector<uint64_t> qhinv(LQ);
        for (uint32_t i = 0; i < LQ; ++i) {
            u512 qh, r; divmod(bfv.Qprod, u512(bfv.Q[i]), qh, r);
            qhat[i] = qh; qhinv[i] = modinv(mod_small(qh, bfv.Q[i]), bfv.Q[i]);
        }
        const std::vector<const std::vector<std::vector<uint32_t>>*> srcs =
            {&c.Ae.c0, &c.Ae.c1, &c.Be.c0, &c.Be.c1};
        std::printf("\nCRNS fixed-point eps on the LIFT (input uniform on [0,Q), Q=%d bits, L=%u)\n",
                    bitlen(bfv.Qprod), LQ);
        std::printf("  %5s  %12s  %12s  %s\n", "u", "wrong eps", "rate", "predicted rate ~ L*2^(28-u)");
        for (uint32_t u : {24u, 30u, 32u, 40u, 44u, 48u, 56u, 64u, 96u, 160u, 224u, 232u, 254u}) {
            std::vector<u512> f(LQ);
            u512 P2 = pow2(u);
            for (uint32_t i = 0; i < LQ; ++i) { u512 rem; divmod(P2, u512(bfv.Q[i]), f[i], rem); }
            uint64_t bad = 0, tot = 0;
            for (auto* src : srcs)
                for (uint32_t n = 0; n < N; ++n) {
                    u512 acc, sfx;
                    for (uint32_t i = 0; i < LQ; ++i) {
                        uint64_t y = (uint64_t)((u128)(*src)[i][n] * qhinv[i] % bfv.Q[i]);
                        acc = add(acc, mul_small(qhat[i], y));
                        sfx = add(sfx, mul_small(f[i], y));
                    }
                    uint32_t eps_exact = 0;                       // acc = x + eps*Q, eps < L
                    while (cmp(acc, bfv.Qprod) >= 0) { acc = sub(acc, bfv.Qprod); ++eps_exact; }
                    uint32_t eps_hat = (uint32_t)to_u64(shr(sfx, u));
                    ++tot; if (eps_hat != eps_exact) ++bad;
                }
            double pred = (double)LQ * std::pow(2.0, 28.0 - (double)u);
            std::printf("  %5u  %12llu  %12.3e  %.3e\n", u, (unsigned long long)bad,
                        (double)bad / (double)tot, pred > 1 ? 1.0 : pred);
        }
        std::printf("  (%u samples per row = 4 polynomials x %u coefficients)\n\n", 4 * N, N);
    }

    // ---- device stage 1 (opt-in until the whole multiply is ported) ----
    if (std::getenv("TB001B_DEV")) {
        Case c;
        const char* path = "reference/he/tb001b_case.bin";
        if (!read_case(path, c)) { std::printf("no reference case at %s\n", path); return 1; }
        bfv.s = c.sk; bfv.evk0 = c.evk0; bfv.evk1 = c.evk1;
        pass &= device_stage(c, bfv);
    }

    std::printf("\n%s\n", pass ? "Test Passed" : "Test FAILED");
    return pass ? 0 : 1;
}
