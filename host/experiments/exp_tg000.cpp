// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_Tg000 — a CORRECT CKKS ct_mul at the exp_Tb001 ring, verified end to end by DECRYPT + DECODE.
//
// WHY THIS EXISTS. The repo has two experiments at this ring already, and neither is a CKKS scheme:
//
//   * exp_Tb001 (host/experiments/exp_tb001.cpp) is a faithful COST MODEL of the CROSS Set-B multiply. Its op
//     graph is CKKS-shaped -- pointwise tensor, drop-prime rescale, dnum digit key-switch -- but its
//     inputs are random residues, its evaluation key is random, and "correct" means "bit-identical to
//     a CPU oracle running the same arithmetic". Nothing in it encrypts or decrypts.
//   * exp_Tb001b (host/experiments/exp_tb001b.cpp) is a semantically correct BFV multiply, decrypt-verified.
//     BFV is a different scheme: its tensor happens over Z in an extended base and its rescale is
//     ⌊t·e/Q⌉. Half of that machinery (the SK conversion, the auxiliary base S, BEHZ) exists only
//     because BFV needs it.
//
// exp_Tg000 is the missing corner: the CKKS scheme, semantically correct, on device. Same ring
// (N=8192), same kernel tree (kernels/experiments/exp_tb001/), same four-step NTT, same approximate FBC, same
// dnum=2 hybrid key-switch. What is new is a real CKKS encoder, a real key pair, and a test that
// closes the loop in the plaintext space the scheme actually targets:
//
//     decode(decrypt(Enc(z1) (x) Enc(z2)))  ==  z1 (.) z2      (slot-wise complex product)
//
// CKKS is APPROXIMATE, so that "==" is a tolerance, not a bit-comparison -- and the tolerance is the
// interesting number, not a formality. It is reported as a measured max/RMS slot error and as
// effective precision in bits, so a regression in noise growth shows up as a number moving rather
// than as a pass that quietly means less than it did.
//
// WHAT MAKES CKKS CHEAPER THAN BFV HERE. The whole of exp_Tb001b's base-extension apparatus is gone:
//
//   * the tensor is POINTWISE MOD Q. CKKS never lifts the ciphertext out of its own base -- the
//     product is taken mod Q and the growth is absorbed by the rescale afterwards. There is no
//     auxiliary base S, no Shenoy-Kumaresan conversion, no centred sentinel.
//   * the rescale is a DROP-PRIME: divide by the top modulus q_{L-1} exactly, in RNS, with a
//     single-source base conversion. That is `bconv_one`, the kernel exp_Tb001 already has.
//   * only the key-switch needs a general base conversion, and it is the same approximate FBC.
//
// So the device path is exp_Tb001's op graph with real data in it, and NO new kernel was written for
// this experiment. See docs/he/exp_tg000.md.
//
//   ./build/exp_tg000 [seed]
//
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <set>
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
using cdouble = std::complex<double>;

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

// ---- the minimal environment the exp_Tb001 device blocks expect ----
// Those blocks are lifted VERBATIM so they cannot drift from the validated versions; the profiling
// names they reference are stubbed here. This experiment is about correctness -- the measurement
// stack (HE_PROF_DRAIN, the 1-1 dispatch map, dispatch_csv.py) is exp_Tb001's job.
// Identical to exp_Tb001's, so the four latency metrics mean exactly the same thing here and the two
// experiments' numbers can be put in one table. `seg_hook` fires at every host boundary (t_wr/t_rd/
// t_fin) to break Metal Trace segments; `drain_fn` empties the device profiler buffer before it
// overflows and starts dropping markers, which is what makes a 1-1 dispatch map possible.
namespace prof {
  double g_build=0, g_wr=0, g_rd=0, g_enq=0, g_fin=0, g_replay=0;
  long n_build=0, n_wr=0, n_rd=0, n_enq=0, n_fin=0, n_replay=0;
  using hrc = std::chrono::high_resolution_clock;
  inline double msd(hrc::time_point t){ return std::chrono::duration<double,std::milli>(hrc::now()-t).count(); }
  inline void reset(){ g_build=g_wr=g_rd=g_enq=g_fin=g_replay=0; n_build=n_wr=n_rd=n_enq=n_fin=n_replay=0; }
  std::function<void()> seg_hook;
  std::function<void()> drain_fn;
  long drain_every=0, drain_ctr=0;
}
namespace Dp = tt::tt_metal::distributed;
template<class...A> void t_wr (A&&...a){ if(prof::seg_hook) prof::seg_hook(); auto t=prof::hrc::now(); Dp::EnqueueWriteMeshBuffer(std::forward<A>(a)...); prof::g_wr+=prof::msd(t); ++prof::n_wr; }
template<class...A> void t_rd (A&&...a){ if(prof::seg_hook) prof::seg_hook(); auto t=prof::hrc::now(); Dp::EnqueueReadMeshBuffer (std::forward<A>(a)...); prof::g_rd+=prof::msd(t); ++prof::n_rd; }
template<class...A> void t_enq(A&&...a){ auto t=prof::hrc::now(); Dp::EnqueueMeshWorkload   (std::forward<A>(a)...); prof::g_enq+=prof::msd(t); ++prof::n_enq;
    if(prof::drain_every && (++prof::drain_ctr % prof::drain_every)==0 && prof::drain_fn) prof::drain_fn(); }
template<class Q>   void t_fin(Q&& cq) { if(prof::seg_hook) prof::seg_hook();
                                         auto t=prof::hrc::now(); Dp::Finish(cq); prof::g_fin+=prof::msd(t); ++prof::n_fin; }

// ---------------------------------------------------------------- CKKS parameters
constexpr uint32_t N     = 8192;        // ring degree -- the CROSS Set-B ring, same as exp_Tb001
constexpr uint32_t NSLOT = N / 2;       // CKKS packs N/2 COMPLEX slots into a real polynomial
constexpr uint32_t LQ    = 8;           // ciphertext base Q: 8 x 28-bit  ~ 224-bit
constexpr uint32_t LP    = 4;           // key-switch base P: 4 x 28-bit  (dnum=2, alpha=4)
constexpr uint32_t LQP   = LQ + LP;     // 12
constexpr uint32_t DNUM  = 2;           // hybrid key-switch digits, as exp_Tb001
// ---- the two precision levers, runtime so both configurations live in ONE binary ----
// NDROP: how many primes the scale is made of, and therefore how many the rescale drops. The
// rescale residual is ~sqrt(||s||^2/12) units AT THE OUTPUT SCALE whatever NDROP is, so widening
// Delta from one prime to two moves the same absolute error 28 bits further down.
// CENTRE: use the centred remainder [c]_{q_L} in (-q_L/2, q_L/2] instead of [0, q_L).
// Defaults are the GOOD configuration; TG000_NDROP=1 / TG000_NOCENTRE=1 reproduce the baseline.
uint32_t g_ndrop = 2;
bool     g_centre = true;
inline uint32_t LOUT() { return LQ - g_ndrop; }      // Q limbs left after the rescale
inline uint32_t DROPI(uint32_t k) { return LQ - 1 - k; }   // the k-th prime dropped, top down

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
// Same minimal bignum as exp_Tb001b. Needed in exactly two places: the CRT reconstruction of a
// 224-bit Q at decryption, and the CRT idempotents w_j that define the key-switch digits. Everything
// else in CKKS stays inside u128 -- which is the point: no base extension means no wide arithmetic.
struct u512 {
    uint64_t w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    u512() = default;
    explicit u512(uint64_t v) { w[0] = v; }
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
u512 mul_small(const u512& a, uint64_t m) {
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
void divmod(const u512& a, const u512& d, u512& q, u512& rem) {
    q = u512(); rem = u512();
    for (int i = bitlen(a) - 1; i >= 0; --i) {
        rem = shl1(rem);
        rem.w[0] |= (a.w[i / 64] >> (i % 64)) & 1ull;
        if (cmp(rem, d) >= 0) { rem = sub(rem, d); q.w[i / 64] |= 1ull << (i % 64); }
    }
}
uint64_t mod_small(const u512& a, uint64_t d) {          // word-wise, O(8), for per-coefficient use
    u128 r = 0;
    for (int i = 7; i >= 0; --i) r = ((r << 64) | a.w[i]) % d;
    return (uint64_t)r;
}
double to_double(const u512& a) {
    double r = 0;
    for (int i = 7; i >= 0; --i) r = r * 18446744073709551616.0 + (double)a.w[i];
    return r;
}

// ---------------------------------------------------------------- negacyclic NTT, per prime
// Identical to exp_Tb001b's. psi is a primitive 2N-th root, so twisting by psi^i turns the negacyclic
// product into a cyclic one and the transform is the ordinary length-N NTT. This is the HOST
// reference the device four-step transform is checked against, limb for limb.
struct NTTCtx {
    uint64_t q, psi, psi_inv, n_inv;
    std::vector<uint64_t> pw, ipw, w, iw;
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
        for (uint32_t i = 1, j = 0; i < N; ++i) {
            uint32_t bit = N >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(a[i], a[j]);
        }
        for (uint32_t len = 2; len <= N; len <<= 1) {
            uint32_t step = N / len;
            for (uint32_t i = 0; i < N; i += len)
                for (uint32_t k = 0; k < len / 2; ++k) {
                    uint64_t wk = inverse ? iw[k * step] : w[k * step];
                    uint64_t u = a[i + k], v = (u128)a[i + k + len / 2] * wk % q;
                    a[i + k] = (u + v) % q;
                    a[i + k + len / 2] = (u + q - v) % q;
                }
        }
        if (inverse) for (uint32_t i = 0; i < N; ++i) a[i] = (u128)a[i] * n_inv % q;
    }
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

// ---------------------------------------------------------------- CKKS encoding: the canonical embedding
// This is the ONE part of CKKS with no analogue anywhere else in the repo, so it is written from the
// definition rather than as a fast transform.
//
// R = Z[x]/(x^N+1) embeds into C^{N/2} by evaluating at half of the primitive 2N-th roots of unity:
// the group (Z/2N)* is generated by 5 and -1, so {zeta^(5^j)} for j < N/2 picks exactly one root from
// each conjugate pair, and a REAL polynomial is determined by its values there. That map is a ring
// homomorphism -- which is the whole reason CKKS multiplies slot-wise: a polynomial product IS the
// slot-wise product of the embeddings.
//
// The inverse follows from V*V = N*I for V[g][k] = zeta^(gk) with g over the N units of Z/2N:
//     a_k = (2/N) * sum_j Re( z_j * zeta^(-5^j k) )
//
// Both directions are the O(N^2/2) direct sums, not the O(N log N) "special FFT". At N=8192 that is
// 33.5M terms and a fraction of a second, paid a handful of times per run -- and it is the definition,
// so it cannot be subtly wrong in the way a hand-rolled bit-reversed radix-2 with a 5^j ordering can.
// This experiment is the correctness reference; the fast encoder belongs in a performance one.
struct Embed {
    std::vector<double> cr, ci;      // cos/sin of pi*k/N, k = 0 .. 2N-1  (zeta^k)
    std::vector<uint32_t> rot;       // rot[j] = 5^j mod 2N
    Embed() {
        cr.resize(2 * N); ci.resize(2 * N);
        for (uint32_t k = 0; k < 2 * N; ++k) {
            double th = M_PI * (double)k / (double)N;
            cr[k] = std::cos(th); ci[k] = std::sin(th);
        }
        rot.resize(NSLOT);
        uint64_t g = 1;
        for (uint32_t j = 0; j < NSLOT; ++j) { rot[j] = (uint32_t)g; g = g * 5 % (2 * N); }
    }
    // slots -> real coefficients
    std::vector<double> inverse(const std::vector<cdouble>& z) const {
        std::vector<double> a(N, 0.0);
        const double f = 2.0 / (double)N;
        for (uint32_t j = 0; j < NSLOT; ++j) {
            const double zr = z[j].real() * f, zi = z[j].imag() * f;
            for (uint32_t k = 0; k < N; ++k) {
                uint32_t e = (uint32_t)((2 * N - (uint32_t)(((uint64_t)rot[j] * k) & (2 * N - 1))) & (2 * N - 1));
                a[k] += zr * cr[e] - zi * ci[e];          // Re( z_j * zeta^(-5^j k) )
            }
        }
        return a;
    }
    // real coefficients -> slots
    std::vector<cdouble> forward(const std::vector<double>& a) const {
        std::vector<cdouble> z(NSLOT);
        for (uint32_t j = 0; j < NSLOT; ++j) {
            double re = 0, im = 0;
            for (uint32_t k = 0; k < N; ++k) {
                uint32_t e = (uint32_t)(((uint64_t)rot[j] * k) & (2 * N - 1));
                re += a[k] * cr[e]; im += a[k] * ci[e];
            }
            z[j] = cdouble(re, im);
        }
        return z;
    }
};

// ---------------------------------------------------------------- an RNS base with exact CRT
struct CRTBase {
    std::vector<uint64_t> p;
    u512 prod, half;
    std::vector<u512> qhat;
    std::vector<uint64_t> qhat_inv;
    void build(const std::vector<uint64_t>& primes) {
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
    }
    u512 reconstruct(const std::vector<uint32_t>& res) const {
        u512 acc;
        for (size_t i = 0; i < p.size(); ++i) {
            uint64_t c = (uint64_t)((u128)res[i] * qhat_inv[i] % p[i]);
            acc = add(acc, mul_small(qhat[i], c));
        }
        while (cmp(acc, prod) >= 0) acc = sub(acc, prod);
        return acc;
    }
    // ...and its value as a SIGNED real. CKKS phases are tiny compared to Q (that is what the noise
    // budget means), so the centred value is small and a double holds it exactly in practice.
    double reconstruct_signed(const std::vector<uint32_t>& res) const {
        u512 v = reconstruct(res);
        bool neg = cmp(v, half) > 0;
        return neg ? -to_double(sub(prod, v)) : to_double(v);
    }
};

// ---------------------------------------------------------------- the scheme
// A ciphertext is (c0, c1) over the first `c0.size()` primes of Q, in the COEFFICIENT domain, at a
// known scale. Level and scale travel with the ciphertext because CKKS arithmetic is only defined
// between operands that agree on both.
struct Ciphertext {
    std::vector<std::vector<uint32_t>> c0, c1;   // [limb][coeff]
    long double scale = 0;
    uint32_t L() const { return (uint32_t)c0.size(); }
};

struct CKKS {
    std::vector<uint64_t> QP;                    // 12 primes: Q(8) | P(4)
    std::vector<NTTCtx> nt;                      // one per QP prime
    std::vector<mont::MontgomeryContext> mctx;
    CRTBase Qb8, Qb7;                            // exact CRT at the input level and the output level
    u512 Qprod, Pprod;
    long double Delta = 0;                       // the scaling factor == the product of the dropped primes
    uint64_t DeltaI = 0;                         // ...and the same value as an exact integer
    Embed emb;

    std::vector<int64_t> s, s2;                  // secret key (ternary) and s^2 in Z[x]/(x^N+1)
    std::mt19937_64 rng;
    std::vector<std::vector<uint32_t>> group, comp;                    // dnum digits and complements
    std::vector<std::vector<std::vector<uint32_t>>> evk0, evk1;        // [digit][limb][coeff], over QP

    explicit CKKS(uint64_t seed) : rng(seed) {
        QP = ntt_primes(LQP);
        for (auto q : QP) { nt.emplace_back(q); mctx.emplace_back((uint32_t)q); }
        std::vector<uint64_t> Q(QP.begin(), QP.begin() + LQ);
        Qb8.build(Q);
        Qb7.build(std::vector<uint64_t>(QP.begin(), QP.begin() + LOUT()));
        Qprod = Qb8.prod;
        Pprod = u512(1); for (uint32_t j = 0; j < LP; ++j) Pprod = mul_small(Pprod, QP[LQ + j]);
        // THE SCALE IS THE PRODUCT OF THE PRIMES THE RESCALE DROPS. Then the product's scale Delta^2
        // divided by those same primes lands back on Delta EXACTLY -- no scale drift, no rescaling of
        // the scale itself. With NDROP=2 that is a ~2^56 scale built from two 28-bit primes, which is
        // how a 28-bit kernel reaches a 56-bit scale without touching the kernel.
        // long double (64-bit mantissa on x86-64) because a 56-bit integer is NOT exact in a double.
        DeltaI = 1; for (uint32_t k = 0; k < g_ndrop; ++k) DeltaI *= QP[DROPI(k)];
        Delta = (long double)DeltaI;

        s.resize(N);
        for (auto& x : s) x = (int64_t)(rng() % 3) - 1;
        s2.assign(N, 0);
        for (uint32_t i = 0; i < N; ++i) {
            if (!s[i]) continue;
            for (uint32_t j = 0; j < N; ++j) {
                uint32_t k = i + j;
                if (k < N) s2[k] += s[i] * s[j]; else s2[k - N] -= s[i] * s[j];
            }
        }
        // dnum=2 digits: Q split into two 4-limb blocks, P is 4 limbs -- so each digit and P are both
        // ~112 bits. That balance is what keeps the key-switch noise small; see the doc.
        group = {{0, 1, 2, 3}, {4, 5, 6, 7}};
        comp.assign(DNUM, {});
        for (uint32_t j = 0; j < DNUM; ++j)
            for (uint32_t li = 0; li < LQP; ++li) {
                bool own = false;
                for (auto g : group[j]) if (g == li) own = true;
                if (!own) comp[j].push_back(li);
            }
        keygen_evk();
    }

    uint64_t q(uint32_t li) const { return QP[li]; }

    std::vector<uint32_t> limb_of(const std::vector<int64_t>& poly, uint64_t m) const {
        std::vector<uint32_t> r(N);
        for (uint32_t n = 0; n < N; ++n) r[n] = (uint32_t)(((poly[n] % (int64_t)m) + (int64_t)m) % (int64_t)m);
        return r;
    }
    std::vector<uint32_t> smallnoise() const { return {}; }

    // ---- relinearisation key, hybrid / digit-decomposed --------------------------------
    // evk_j = ( -a_j*s + e_j + P * w_j * s^2 ,  a_j )  over QP, where w_j is the CRT IDEMPOTENT of
    // digit j: w_j = 1 mod every prime of group[j], 0 mod every other prime of Q.
    //
    // Textbooks write w_j as Qhat_j * [Qhat_j^-1]_{D_j}, which needs a 112-bit modular inverse. The
    // idempotent characterisation is the same integer and needs none: its Q-residues are 0/1 by
    // definition, and one exact CRT reconstruction gives the integer whose residues mod the P primes
    // the key also needs.
    void keygen_evk() {
        std::vector<u512> w(DNUM);
        for (uint32_t j = 0; j < DNUM; ++j) {
            std::vector<uint32_t> res(LQ, 0);
            for (auto g : group[j]) res[g] = 1;
            w[j] = Qb8.reconstruct(res);
        }
        evk0.assign(DNUM, std::vector<std::vector<uint32_t>>(LQP));
        evk1.assign(DNUM, std::vector<std::vector<uint32_t>>(LQP));
        for (uint32_t j = 0; j < DNUM; ++j) {
            std::vector<int64_t> e_int(N);
            for (auto& x : e_int) x = (int64_t)(rng() % 5) - 2;
            for (uint32_t li = 0; li < LQP; ++li) {
                uint64_t m = QP[li];
                uint64_t Pm = mod_small(Pprod, m);
                uint64_t wm = mod_small(w[j], m);
                uint64_t Pw = (uint64_t)((u128)Pm * wm % m);
                std::vector<uint32_t> a(N), sl = limb_of(s, m), s2l = limb_of(s2, m);
                for (uint32_t n = 0; n < N; ++n) a[n] = (uint32_t)(rng() % m);
                auto as_ = nt[li].negmul(a, sl);
                evk0[j][li].resize(N); evk1[j][li] = a;
                for (uint32_t n = 0; n < N; ++n) {
                    uint32_t el = (uint32_t)(((e_int[n] % (int64_t)m) + (int64_t)m) % (int64_t)m);
                    evk0[j][li][n] = (uint32_t)(((u128)m + el + (u128)Pw * s2l[n] % m - as_[n]) % m);
                }
            }
        }
    }

    // ---- encode / decode ---------------------------------------------------------------
    // encode: round(Delta * canonical^-1(z)) as an integer polynomial. decode undoes it at whatever
    // scale the ciphertext claims -- which after a rescale is Delta again, by construction.
    std::vector<int64_t> encode(const std::vector<cdouble>& z, long double scale) const {
        auto a = emb.inverse(z);
        std::vector<int64_t> m(N);
        for (uint32_t k = 0; k < N; ++k) m[k] = (int64_t)llroundl(scale * (long double)a[k]);
        return m;
    }
    std::vector<cdouble> decode(const std::vector<double>& coeff, long double scale) const {
        std::vector<double> a(N);
        for (uint32_t k = 0; k < N; ++k) a[k] = (double)((long double)coeff[k] / scale);
        return emb.forward(a);
    }

    // ---- encrypt / decrypt -------------------------------------------------------------
    //   c0 = -a*s + e + m,  c1 = a          (the message is already scaled: Delta lives inside m)
    //   phase(c) = c0 + c1*s = m + e,       decode(phase / scale)
    Ciphertext encrypt(const std::vector<int64_t>& m, long double scale) {
        Ciphertext ct;
        ct.scale = scale;
        ct.c0.assign(LQ, std::vector<uint32_t>(N));
        ct.c1.assign(LQ, std::vector<uint32_t>(N));
        std::vector<int64_t> e_int(N);
        for (auto& x : e_int) x = (int64_t)(rng() % 5) - 2;
        // TG000_ZEROERR: drop the LWE error, which makes the ciphertext insecure and the scheme's OWN
        // approximation error the only thing left. Purely a diagnostic for where precision goes.
        if (std::getenv("TG000_ZEROERR")) for (auto& x : e_int) x = 0;
        std::vector<std::vector<uint32_t>> aQ(LQ, std::vector<uint32_t>(N));
        for (uint32_t i = 0; i < LQ; ++i)
            for (uint32_t n = 0; n < N; ++n) aQ[i][n] = (uint32_t)(rng() % QP[i]);
        for (uint32_t i = 0; i < LQ; ++i) {
            uint64_t mq = QP[i];
            auto sl = limb_of(s, mq);
            auto as_ = nt[i].negmul(aQ[i], sl);
            for (uint32_t n = 0; n < N; ++n) {
                uint32_t el = (uint32_t)(((e_int[n] % (int64_t)mq) + (int64_t)mq) % (int64_t)mq);
                uint32_t ml = (uint32_t)(((m[n] % (int64_t)mq) + (int64_t)mq) % (int64_t)mq);
                ct.c0[i][n] = (uint32_t)(((u128)mq + ml + el - as_[n]) % mq);
                ct.c1[i][n] = aQ[i][n];
            }
        }
        return ct;
    }
    // the phase as a signed real per coefficient (NOT reduced mod anything -- CKKS has no plaintext
    // modulus; the "plaintext" is the small centred integer the noise sits on top of)
    std::vector<double> phase(const Ciphertext& ct) const {
        const uint32_t L = ct.L();
        const CRTBase& B = (L == LQ) ? Qb8 : Qb7;      // the only two levels a single multiply visits
        std::vector<std::vector<uint32_t>> v(L);
        for (uint32_t i = 0; i < L; ++i) {
            uint64_t mq = QP[i];
            auto sl = limb_of(s, mq);
            auto cs = nt[i].negmul(ct.c1[i], sl);
            v[i].resize(N);
            for (uint32_t n = 0; n < N; ++n) v[i][n] = (uint32_t)(((u128)ct.c0[i][n] + cs[n]) % mq);
        }
        std::vector<double> out(N);
        std::vector<uint32_t> res(L);
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t i = 0; i < L; ++i) res[i] = v[i][n];
            out[n] = B.reconstruct_signed(res);
        }
        return out;
    }
    std::vector<cdouble> decrypt(const Ciphertext& ct) const { return decode(phase(ct), ct.scale); }

    // ---- the approximate base conversion, exactly as the device kernel computes it ------
    // out_j = (sum_i y_i * Shat_i) mod p_j  with  y_i = x_i * Shat_i^-1 mod s_i.
    // What it represents is x + eps*S for an eps < Ls that is never materialised. Both the key-switch
    // mod-up and the mod-down use it, and BOTH tolerate the eps for a structural reason, not a
    // numerical one -- see the doc. The host models it rather than the exact conversion so device and
    // host can be compared BIT-EXACTLY.
    std::vector<std::vector<uint32_t>> fbc(const std::vector<std::vector<uint32_t>>& x,
                                           const std::vector<uint32_t>& sI,
                                           const std::vector<uint32_t>& dI) const {
        const uint32_t Ls = (uint32_t)sI.size(), Ld = (uint32_t)dI.size();
        u128 S = 1; for (uint32_t i = 0; i < Ls; ++i) S *= QP[sI[i]];
        std::vector<std::vector<uint32_t>> out(Ld, std::vector<uint32_t>(N));
        std::vector<uint64_t> y(Ls), c1(Ls);                       // c1[i] = Shat_i^-1 mod s_i
        std::vector<std::vector<uint64_t>> c2(Ls, std::vector<uint64_t>(Ld));   // c2[i][j] = Shat_i mod p_j
        for (uint32_t i = 0; i < Ls; ++i) {
            uint64_t si = QP[sI[i]];
            c1[i] = modinv((uint64_t)((S / si) % si), si);
            for (uint32_t j = 0; j < Ld; ++j) c2[i][j] = (uint64_t)((S / si) % QP[dI[j]]);
        }
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t i = 0; i < Ls; ++i) y[i] = (uint64_t)((u128)x[i][n] * c1[i] % QP[sI[i]]);
            for (uint32_t j = 0; j < Ld; ++j) {
                u128 acc = 0;
                for (uint32_t i = 0; i < Ls; ++i) acc += (u128)y[i] * c2[i][j];
                out[j][n] = (uint32_t)(acc % QP[dI[j]]);
            }
        }
        return out;
    }

    // ---- the drop-prime rescale --------------------------------------------------------
    //   c'_i = (c_i - [c]_{q_L}) * q_L^-1  mod q_i,   i < L
    // Exact integer division in RNS: c - [c]_{q_L} is divisible by q_L by construction, so this is a
    // division and not a rounding. The only approximation is that [c]_{q_L} is taken in [0, q_L)
    // rather than centred, which moves each coefficient by at most one unit at the OUTPUT scale --
    // i.e. by at most 1/Delta in the decoded value.
    //
    // This is the CKKS operation exp_Tb001 models, and on device it is `bconv_one` + modsub + modmul.
    std::vector<std::vector<uint32_t>> rescale(const std::vector<std::vector<uint32_t>>& in) const {
        std::vector<std::vector<uint32_t>> c = in;
        for (uint32_t k = 0; k < g_ndrop; ++k) {
            const uint32_t d = DROPI(k);                  // the prime being dropped this step
            const uint64_t qL = QP[d], qHalf = qL / 2;
            std::vector<std::vector<uint32_t>> out(d, std::vector<uint32_t>(N));
            for (uint32_t i = 0; i < d; ++i) {
                const uint64_t m = QP[i], qLinv = modinv(qL % m, m);
                const uint64_t h = qHalf % m;
                for (uint32_t n = 0; n < N; ++n) {
                    // CENTRING, done by SHIFTING rather than by a signed remainder. Adding qHalf to
                    // the value before a plain [0,q_L) rescale is *identically* the centred rescale:
                    //     [c + qHalf]_{q_L} = [c]_{q_L} + qHalf - q_L * [ [c]_{q_L} > qHalf ]
                    // so the shift cancels and what is subtracted is the CENTRED remainder. That is
                    // why this needs no new kernel -- see docs/he/exp_tg000.md.
                    const uint64_t ci = g_centre ? ((uint64_t)c[i][n] + h) % m : c[i][n];
                    const uint64_t xd = g_centre ? ((uint64_t)c[d][n] + qHalf) % qL : c[d][n];
                    const uint64_t corr = xd % m;
                    out[i][n] = (uint32_t)(((u128)ci + m - corr) % m * qLinv % m);
                }
            }
            c = std::move(out);
        }
        return c;
    }

    // ---- the tensor: POINTWISE MOD Q, no lift ------------------------------------------
    void tensor(const Ciphertext& A, const Ciphertext& B,
                std::vector<std::vector<uint32_t>>& e0,
                std::vector<std::vector<uint32_t>>& e1,
                std::vector<std::vector<uint32_t>>& e2) const {
        e0.assign(LQ, {}); e1.assign(LQ, {}); e2.assign(LQ, {});
        for (uint32_t i = 0; i < LQ; ++i) {
            uint64_t m = QP[i];
            e0[i] = nt[i].negmul(A.c0[i], B.c0[i]);
            e2[i] = nt[i].negmul(A.c1[i], B.c1[i]);
            auto x = nt[i].negmul(A.c0[i], B.c1[i]);
            auto y = nt[i].negmul(A.c1[i], B.c0[i]);
            e1[i].resize(N);
            for (uint32_t n = 0; n < N; ++n) e1[i][n] = (uint32_t)(((u128)x[n] + y[n]) % m);
        }
    }

    // ---- relinearisation: hybrid key-switch of e2 --------------------------------------
    // For each digit j: take e2's residues on group[j], base-extend them to every other QP prime with
    // the approximate FBC, multiply by evk_j, accumulate. Then divide the accumulator by P.
    void relin(const std::vector<std::vector<uint32_t>>& e2,
               std::vector<std::vector<uint32_t>>& k0,
               std::vector<std::vector<uint32_t>>& k1) const {
        std::vector<std::vector<uint32_t>> acc0(LQP, std::vector<uint32_t>(N, 0)),
                                           acc1(LQP, std::vector<uint32_t>(N, 0));
        for (uint32_t j = 0; j < DNUM; ++j) {
            std::vector<std::vector<uint32_t>> xs;
            for (auto li : group[j]) xs.push_back(e2[li]);
            auto o = fbc(xs, group[j], comp[j]);
            std::vector<std::vector<uint32_t>> vj(LQP);
            for (auto li : group[j]) vj[li] = e2[li];
            for (uint32_t k = 0; k < comp[j].size(); ++k) vj[comp[j][k]] = o[k];
            for (uint32_t li = 0; li < LQP; ++li) {
                uint64_t m = QP[li];
                auto p0 = nt[li].negmul(vj[li], evk0[j][li]);
                auto p1 = nt[li].negmul(vj[li], evk1[j][li]);
                for (uint32_t n = 0; n < N; ++n) {
                    acc0[li][n] = (uint32_t)(((u128)acc0[li][n] + p0[n]) % m);
                    acc1[li][n] = (uint32_t)(((u128)acc1[li][n] + p1[n]) % m);
                }
            }
        }
        std::vector<uint32_t> qidx(LQ), pidx(LP);
        for (uint32_t i = 0; i < LQ; ++i) qidx[i] = i;
        for (uint32_t j = 0; j < LP; ++j) pidx[j] = LQ + j;
        auto moddown = [&](std::vector<std::vector<uint32_t>>& acc) {
            std::vector<std::vector<uint32_t>> aP;
            for (auto pp : pidx) aP.push_back(acc[pp]);
            auto f = fbc(aP, pidx, qidx);
            std::vector<std::vector<uint32_t>> out(LQ, std::vector<uint32_t>(N));
            for (uint32_t i = 0; i < LQ; ++i) {
                uint64_t m = QP[i], Pinv = modinv(mod_small(Pprod, m), m);
                for (uint32_t n = 0; n < N; ++n)
                    out[i][n] = (uint32_t)((((u128)acc[i][n] + m - f[i][n]) % m) * Pinv % m);
            }
            return out; };
        k0 = moddown(acc0); k1 = moddown(acc1);
    }

    // ---- THE MULTIPLY: tensor -> relinearise -> combine -> rescale ---------------------
    // Rescale LAST, which is the canonical CKKS ordering and differs from exp_Tb001's cost model
    // (that one rescales its four inputs first, to model a chain where the operands arrive at
    // scale Delta^2). Rescale-last is what makes the output scale equal the input scale, which is
    // what "end to end" needs -- and it uses the FULL 12-limb QP for the key-switch rather than 11.
    Ciphertext multiply(const Ciphertext& A, const Ciphertext& B) const {
        std::vector<std::vector<uint32_t>> e0, e1, e2, k0, k1;
        tensor(A, B, e0, e1, e2);
        relin(e2, k0, k1);
        std::vector<std::vector<uint32_t>> r0(LQ, std::vector<uint32_t>(N)), r1(LQ, std::vector<uint32_t>(N));
        for (uint32_t i = 0; i < LQ; ++i) {
            uint64_t m = QP[i];
            for (uint32_t n = 0; n < N; ++n) {
                r0[i][n] = (uint32_t)(((u128)e0[i][n] + k0[i][n]) % m);
                r1[i][n] = (uint32_t)(((u128)e1[i][n] + k1[i][n]) % m);
            }
        }
        Ciphertext o;
        o.c0 = rescale(r0);
        o.c1 = rescale(r1);
        o.scale = A.scale * B.scale / Delta;      // == Delta, by construction
        return o;
    }

    // ---- ct_add: per-limb modular addition, and that is the whole operation -------------
    Ciphertext add_ct(const Ciphertext& A, const Ciphertext& B) const {
        Ciphertext o; o.scale = A.scale;
        const uint32_t L = A.L();
        o.c0.assign(L, std::vector<uint32_t>(N));
        o.c1.assign(L, std::vector<uint32_t>(N));
        for (uint32_t i = 0; i < L; ++i) {
            uint64_t m = QP[i];
            for (uint32_t n = 0; n < N; ++n) {
                o.c0[i][n] = (uint32_t)(((u128)A.c0[i][n] + B.c0[i][n]) % m);
                o.c1[i][n] = (uint32_t)(((u128)A.c1[i][n] + B.c1[i][n]) % m);
            }
        }
        return o;
    }
};

// ---------------------------------------------------------------- slot-space comparison
struct SlotErr { double maxe = 0, rms = 0; double bits() const { return maxe > 0 ? -std::log2(maxe) : 99.0; } };
SlotErr slot_err(const std::vector<cdouble>& got, const std::vector<cdouble>& want) {
    SlotErr e; double acc = 0;
    for (uint32_t j = 0; j < NSLOT; ++j) {
        double d = std::abs(got[j] - want[j]);
        e.maxe = std::max(e.maxe, d); acc += d * d;
    }
    e.rms = std::sqrt(acc / (double)NSLOT);
    return e;
}

// ---------------------------------------------------------------- fixed reference case
// Same discipline as exp_Tb001b: the device is checked against BYTES ON DISK, not against whatever
// this process's RNG produced, so a failure is reproducible on another card and a host-side bug
// cannot cancel a device-side one.
constexpr uint32_t CASE_MAGIC = 0x47303030;   // "G000"
constexpr uint32_t CASE_VERSION = 2;

struct Case {
    std::vector<uint64_t> QP;
    std::vector<int64_t> sk;
    std::vector<std::vector<std::vector<uint32_t>>> evk0, evk1;
    Ciphertext A, B, R;                     // coefficient domain; A,B at LQ limbs, R at LOUT()
    std::vector<cdouble> z1, z2;            // the slot vectors, and their product is the target
    uint64_t scale = 0;                     // exact: a 56-bit scale is not representable in a double
    uint32_t ndrop = 0; uint32_t centre = 0;
};

void wr32(std::FILE* f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
void wr64(std::FILE* f, uint64_t v) { std::fwrite(&v, 8, 1, f); }
void wrd (std::FILE* f, double v)   { std::fwrite(&v, 8, 1, f); }
uint32_t rd32(std::FILE* f) { uint32_t v = 0; if (std::fread(&v, 4, 1, f) != 1) std::abort(); return v; }
uint64_t rd64(std::FILE* f) { uint64_t v = 0; if (std::fread(&v, 8, 1, f) != 1) std::abort(); return v; }
double   rdd (std::FILE* f) { double   v = 0; if (std::fread(&v, 8, 1, f) != 1) std::abort(); return v; }
void wrv32(std::FILE* f, const std::vector<uint32_t>& v) { std::fwrite(v.data(), 4, v.size(), f); }
void wrv64(std::FILE* f, const std::vector<int64_t>& v)  { std::fwrite(v.data(), 8, v.size(), f); }
void rdv32(std::FILE* f, std::vector<uint32_t>& v, size_t n) { v.resize(n); if (std::fread(v.data(), 4, n, f) != n) std::abort(); }
void rdv64(std::FILE* f, std::vector<int64_t>& v, size_t n)  { v.resize(n); if (std::fread(v.data(), 8, n, f) != n) std::abort(); }

void write_ct(std::FILE* f, const Ciphertext& c) {
    wr32(f, c.L()); wr64(f, (uint64_t)c.scale);
    for (auto& r : c.c0) wrv32(f, r);
    for (auto& r : c.c1) wrv32(f, r);
}
void read_ct(std::FILE* f, Ciphertext& c) {
    uint32_t L = rd32(f); c.scale = (long double)rd64(f);
    c.c0.resize(L); c.c1.resize(L);
    for (auto& r : c.c0) rdv32(f, r, N);
    for (auto& r : c.c1) rdv32(f, r, N);
}

bool write_case(const char* path, const CKKS& ck, const Case& c) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    wr32(f, CASE_MAGIC); wr32(f, CASE_VERSION);
    wr32(f, N); wr32(f, LQ); wr32(f, LP); wr32(f, DNUM);
    wr32(f, g_ndrop); wr32(f, g_centre ? 1u : 0u);
    for (auto q : ck.QP) wr64(f, q);
    wr64(f, c.scale);
    wrv64(f, c.sk);
    for (uint32_t j = 0; j < DNUM; ++j) { for (auto& r : c.evk0[j]) wrv32(f, r); for (auto& r : c.evk1[j]) wrv32(f, r); }
    write_ct(f, c.A); write_ct(f, c.B); write_ct(f, c.R);
    for (auto& z : c.z1) { wrd(f, z.real()); wrd(f, z.imag()); }
    for (auto& z : c.z2) { wrd(f, z.real()); wrd(f, z.imag()); }
    std::fclose(f);
    return true;
}

bool read_case(const char* path, Case& c) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    if (rd32(f) != CASE_MAGIC || rd32(f) != CASE_VERSION) { std::fclose(f); return false; }
    if (rd32(f) != N || rd32(f) != LQ || rd32(f) != LP || rd32(f) != DNUM) { std::fclose(f); return false; }
    c.ndrop = rd32(f); c.centre = rd32(f);
    // A fixture built for a different rescale is a DIFFERENT ciphertext, not a stale one -- reject it
    // loudly rather than let the device be compared against bytes that were never its target.
    if (c.ndrop != g_ndrop || c.centre != (g_centre ? 1u : 0u)) { std::fclose(f); return false; }
    c.QP.resize(LQP); for (auto& q : c.QP) q = rd64(f);
    c.scale = rd64(f);
    rdv64(f, c.sk, N);
    c.evk0.assign(DNUM, std::vector<std::vector<uint32_t>>(LQP));
    c.evk1.assign(DNUM, std::vector<std::vector<uint32_t>>(LQP));
    for (uint32_t j = 0; j < DNUM; ++j) { for (auto& r : c.evk0[j]) rdv32(f, r, N); for (auto& r : c.evk1[j]) rdv32(f, r, N); }
    read_ct(f, c.A); read_ct(f, c.B); read_ct(f, c.R);
    c.z1.resize(NSLOT); c.z2.resize(NSLOT);
    for (auto& z : c.z1) { double re = rdd(f), im = rdd(f); z = cdouble(re, im); }
    for (auto& z : c.z2) { double re = rdd(f), im = rdd(f); z = cdouble(re, im); }
    std::fclose(f);
    return true;
}

// ======================================================================================
// THE DEVICE PATH
//
// Every numbered block below is lifted VERBATIM from host/experiments/exp_tb001.cpp / exp_tb001b.cpp -- the
// optimised, validated versions (fused 8-partial contraction, SFPI twiddle, width-correct FBC,
// single-source bconv_one). Retyping them is how they drift. What differs is the environment: 12
// primes, CKKS's op graph, and no profiling scaffolding.
//
// The ladder is deliberate. Each stage is fed REFERENCE inputs so a fault localises to that stage;
// only the last one runs the whole chain from its own intermediates.
//
//   [D1] forward NTT           [D2] inverse NTT        [D3] tensor (pointwise mod Q)
//   [D4] relinearise           [D5] rescale            [D6] whole multiply vs the FIXED case
//   [D7] decode(decrypt(device output)) vs z1 (.) z2
// ======================================================================================
bool device_stage(const Case& kase, CKKS& ck) {
    const auto& QP = ck.QP;
    auto& ctx = ck.mctx;

    auto mesh = distributed::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE);
    auto& cq = mesh->mesh_command_queue();
    auto dr = distributed::MeshCoordinateRange(mesh->shape());
    auto grid = mesh->compute_with_storage_grid_size();
    const uint32_t GX = grid.x, GY = grid.y;
    uint32_t want = GX * GY; if (const char* e = std::getenv("NTT_CORES")) { int v = std::atoi(e); if (v > 0) want = (uint32_t)v; }
    const uint32_t NCORES = std::min<uint32_t>(want, GX * GY);
    const uint32_t EWC = std::min<uint32_t>(NBLK, NCORES);
    auto CC = [&](uint32_t i){ return CoreCoord{i % GX, i / GX}; };
    fmt::print("exp_Tg000  device  grid {}x{} -> G={} cores, {} primes\n", GX, GY, NCORES, LQP);

    auto mkbuf = [&](uint32_t nt_, uint32_t page) {
        distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt_ * page}, d, mesh.get()); };
    auto mkcb = [&](Program& p, const CoreRangeSet& crs, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
        CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, crs, c); };
    auto crs_of = [&](uint32_t cnt){ std::set<CoreRange> s; for (uint32_t i=0;i<cnt;++i) s.insert(CoreRange(CC(i))); return CoreRangeSet(s); };

    std::vector<std::shared_ptr<distributed::MeshBuffer>> mid8(NBLK), midB8(NBLK), part8(NBLK);
    for (uint32_t i = 0; i < NBLK; ++i) { mid8[i]=mkbuf(1,I32); midB8[i]=mkbuf(1,I32); part8[i]=mkbuf(NBLK,I32); }

    // ---- the measurement stack, ported verbatim in shape from exp_Tb001 ----
    // emit(): CACHE off -> build + enqueue. CACHE on: REC -> build + cache; TR_PHASE 1 -> capture a
    // Metal Trace; TR_PHASE 2 -> replay the segment instead of enqueueing. A "segment" is a maximal
    // run of dispatches with no host boundary between them, and seg_break() is what ends one.
    std::vector<distributed::MeshWorkload> hold; hold.reserve(4096);
    bool CACHE_ON=false, CACHE_REC=false; std::vector<distributed::MeshWorkload> WC; size_t WI=0;
    int TR_PHASE=0; std::vector<distributed::MeshTraceId> SEG; bool seg_open=false, seg_started=false;
    size_t seg_idx=0; bool MEASURE_DEV=false;
    std::string KLBL = "?";                       // which KERNEL a dispatch ran
    std::string HOP="-", HPH="-", OPD="-";        // which HE POLYNOMIAL OP it belongs to
    auto L=[&](const char* op, const char* ph, std::string opd){ HOP=op; HPH=ph; OPD=std::move(opd); };
    struct DL { std::string k, op, ph, opd; };
    std::vector<DL> DLBL;
    const char* LBLPATH = std::getenv("TG000_LABELS");
    auto dump_labels=[&](){ if(!LBLPATH) return; std::FILE* f=std::fopen(LBLPATH,"w"); if(!f) return;
        std::fprintf(f,"seq,host_kernel,he_op,he_phase,operand\n");
        for(size_t i=0;i<DLBL.size();++i)
            std::fprintf(f,"%zu,%s,%s,%s,%s\n",i,DLBL[i].k.c_str(),DLBL[i].op.c_str(),
                         DLBL[i].ph.c_str(),DLBL[i].opd.c_str());
        std::fclose(f);
        fmt::print("wrote {} : {} dispatch labels (join on seq with tools/traces/dispatch_csv.py)\n",
                   LBLPATH, DLBL.size()); };
    auto emit = [&](auto&& build){
        if(LBLPATH){ if(DLBL.size()<=WI) DLBL.resize(WI+1); DLBL[WI]={KLBL,HOP,HPH,OPD}; }
        if(!CACHE_ON){ auto wl=build(); t_enq(cq,(hold.emplace_back(std::move(wl)),hold.back()),false); return; }
        if(TR_PHASE==2){
            if(!seg_started){ if(seg_idx<SEG.size()){ auto rt=prof::hrc::now();
                    mesh->replay_mesh_trace(0, SEG[seg_idx], MEASURE_DEV);
                    if(MEASURE_DEV){ prof::g_replay+=prof::msd(rt); ++prof::n_replay; } }
                ++seg_idx; seg_started=true; }
            ++WI; return; }
        if(CACHE_REC) WC.emplace_back(build());
        if(TR_PHASE==1 && !seg_open){ SEG.push_back(mesh->begin_mesh_trace(0)); seg_open=true; }
        t_enq(cq, WC[WI++], false); };
    auto seg_break=[&](){
        if(!CACHE_ON) return;
        if(TR_PHASE==1){ if(seg_open){ mesh->end_mesh_trace(0, SEG.back()); seg_open=false; } }
        else if(TR_PHASE==2){ seg_started=false; } };
    prof::seg_hook = seg_break;

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
        // The recombine is Z-linear in the int32 planes, so the planes accumulate in DEST across ACC
        // partials and are recombined ONCE -- exact, not approximate. ACC=NBLK folds the whole
        // contraction into one dispatch per output block. TG000_ACC=1 restores one partial per dispatch.
        uint32_t ACC = NBLK;
        if (const char* e = std::getenv("TG000_ACC")) { int v = std::atoi(e); if (v > 0) ACC = (uint32_t)v; }
        if (ACC > NBLK || NBLK % ACC) ACC = 1;
        auto nsub_of = [&](uint32_t nacc, uint64_t q) -> uint32_t {
            const uint64_t plane_max = (uint64_t)NB * 32ull * 127ull * 127ull;
            const uint64_t hi_max = (uint64_t)nacc * plane_max * 129ull;
            const uint64_t r_max = hi_max + q;
            return (r_max < (1ull << 32)) ? (uint32_t)(r_max / q) + 1u : 0u; };
        if (ACC > 1) { uint32_t bad = 0;
            for (uint32_t li = 0; li < LQP; ++li) if (!nsub_of(ACC, QP[li])) ++bad;
            if (bad) { fmt::print("exp_Tg000  ACC={} exceeds the int32 recombine bound on {} of {} primes"
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

        std::map<int,std::vector<std::shared_ptr<distributed::MeshBuffer>>> bigWres, twres;
        std::map<int,std::shared_ptr<distributed::MeshBuffer>> smallWres;
        for (auto& kv : KC) { int key=kv.first; const CS& cs=kv.second;
            bigWres[key].resize(NBLK*NBLK); twres[key].resize(NBLK);
            for (uint32_t i=0;i<NBLK*NBLK;++i){ bigWres[key][i]=mkbuf(NB*NB,I8); t_wr(cq, bigWres[key][i], cs.big[i], false); }
            smallWres[key]=mkbuf(NB*NB,I8); t_wr(cq, smallWres[key], cs.small, false);
            for (uint32_t bk2=0;bk2<NBLK;++bk2){ twres[key][bk2]=mkbuf(1,I32); t_wr(cq, twres[key][bk2], cs.tw[bk2], false); } }
        t_fin(cq);
        KC.clear();

//@@BLOCK stg (reduce+twiddle, twiddle)
        auto rmodadd=[&](std::vector<std::shared_ptr<distributed::MeshBuffer>>& ins, std::vector<std::shared_ptr<distributed::MeshBuffer>>& outs, uint32_t q, uint32_t npart=NBLK){
          for(uint32_t base=0;base<NBLK;base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,NBLK-base);
            KLBL="modadd_reduce"; emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(cnt);
              mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,NBLK,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
              auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              for(uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                SetRuntimeArgs(p,rd,c,{(uint32_t)ins[t]->address(),npart,0u}); SetRuntimeArgs(p,wr,c,{(uint32_t)outs[t]->address(),1u,0u}); SetRuntimeArgs(p,cp,c,{npart,q}); }
              prof::g_build+=prof::msd(__tb); ++prof::n_build; distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); } };
        auto rmodadd_tw=[&](std::vector<std::shared_ptr<distributed::MeshBuffer>>& ins, std::vector<std::shared_ptr<distributed::MeshBuffer>>& tw, std::vector<std::shared_ptr<distributed::MeshBuffer>>& outs, uint32_t q, uint32_t li, uint32_t npart=NBLK){
          for(uint32_t base=0;base<NBLK;base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,NBLK-base);
            KLBL="modadd_twiddle"; emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(cnt);
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
            KLBL="modmul_sfpi_full"; emit([&]{ auto __tb=prof::hrc::now(); Program p=CreateProgram(); auto crs=crs_of(cnt);
              mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
              auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_full.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              for(uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                SetRuntimeArgs(p,rd,c,{(uint32_t)a[t]->address(),(uint32_t)b[t]->address(),1u,0u}); SetRuntimeArgs(p,wr,c,{(uint32_t)o[t]->address(),1u,0u}); SetRuntimeArgs(p,cp,c,{1u,q,ctx[li].q_inv_32(),(uint32_t)(QP[li]&0xFFFF),(uint32_t)(QP[li]>>16)}); }
              prof::g_build+=prof::msd(__tb); ++prof::n_build; distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); } };

//@@BLOCK chain (fwd_buf / inv_buf)
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
            if(NBLK/ACC==1) rtwiddle(part8,tw,midB8,(uint32_t)q,li);
            else            rmodadd_tw(part8,tw,midB8,(uint32_t)q,li,NBLK/ACC);
            std::vector<MM> s2; for(uint32_t bk2=0;bk2<NBLK;++bk2) s2.push_back({midB8[bk2],sm,outb[bk2],0});
            for(uint32_t base=0;base<s2.size();base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,s2.size()-base); KLBL="fused_matmul_c"; emit([&]{ return build_mm(s2,base,cnt,NB*NB,"compute/fused_matmul_c.cpp",midB8[0],sm,(uint32_t)q,li); }); } };
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

    // ---------------- buffer plumbing and the resident primitives ----------------
    using Buf = std::shared_ptr<distributed::MeshBuffer>;
    using Buf8 = std::vector<Buf>;
    // Buffer POOL: with the workload cache on, hand out fixed buffers in deterministic order (PI is
    // reset per multiply) so addresses are STABLE across calls -- the cached workloads bake their
    // operand addresses in, and a fresh allocation per call would invalidate every one of them.
    std::vector<Buf8> POOL; size_t PI = 0;
    auto mkBuf8 = [&]() -> Buf8 {
        if(!CACHE_ON){ Buf8 b(NBLK); for(uint32_t i=0;i<NBLK;++i) b[i]=mkbuf(1,I32); return b; }
        if(PI>=POOL.size()){ Buf8 b(NBLK); for(uint32_t i=0;i<NBLK;++i) b[i]=mkbuf(1,I32); POOL.push_back(b); }
        return POOL[PI++]; };
    auto up_natC = [&](const std::vector<uint32_t>& x){ Buf8 b=mkBuf8();      // natural coeff -> C'' layout
        for(uint32_t bn2=0;bn2<NBLK;++bn2){ std::vector<uint32_t> blk(ELEMS);
            for(uint32_t n1=0;n1<N1;++n1) for(uint32_t n2l=0;n2l<T;++n2l) blk[n1*T+n2l]=x[n1+32*(bn2*T+n2l)];
            t_wr(cq,b[bn2],tilize32(blk),false);} return b; };
    auto up_natE = [&](const std::vector<uint32_t>& x, bool mont, uint32_t li){ Buf8 b=mkBuf8();  // natural eval -> E
        for(uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> blk(ELEMS);
            for(uint32_t k2l=0;k2l<T;++k2l) for(uint32_t k1=0;k1<N1;++k1){ uint32_t v=x[(bk2*T+k2l)+N2*k1]; blk[k2l*T+k1]=mont?ctx[li].to_montgomery(v):v; }
            t_wr(cq,b[bk2],tilize32(blk),false);} return b; };
    auto rd_natC = [&](Buf8& b){ std::vector<uint32_t> x(N);                  // C'' -> natural coeff
        for(uint32_t bn2=0;bn2<NBLK;++bn2){ std::vector<uint32_t> raw; t_rd(cq,raw,b[bn2],true);
            auto blk=untilize32(raw); for(uint32_t n1=0;n1<N1;++n1) for(uint32_t n2l=0;n2l<T;++n2l) x[n1+32*(bn2*T+n2l)]=blk[n1*T+n2l]; } return x; };
    auto rd_natE = [&](Buf8& b){ std::vector<uint32_t> x(N);                  // E -> natural eval
        for(uint32_t bk2=0;bk2<NBLK;++bk2){ std::vector<uint32_t> raw; t_rd(cq,raw,b[bk2],true);
            auto blk=untilize32(raw); for(uint32_t k2l=0;k2l<T;++k2l) for(uint32_t k1=0;k1<N1;++k1) x[(bk2*T+k2l)+N2*k1]=blk[k2l*T+k1]; } return x; };
    auto fill8 = [&](uint32_t v){ Buf8 b=mkBuf8(); std::vector<uint32_t> t(ELEMS,v); auto tl=tilize32(t);
        for(uint32_t i=0;i<NBLK;++i) t_wr(cq,b[i],tl,false); return b; };

    auto transpose8_dev = [&](Buf8& in, Buf8& out){
        for(uint32_t base=0;base<NBLK;base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,NBLK-base);
            KLBL="transpose_tile"; emit([&]{ Program p=CreateProgram(); auto crs=crs_of(cnt);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto w=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto c=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/transpose_tile.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for(uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto cc=CC(k);
                    SetRuntimeArgs(p,r,cc,{(uint32_t)in[t]->address(),1u,0u}); SetRuntimeArgs(p,w,cc,{(uint32_t)out[t]->address(),1u,0u}); SetRuntimeArgs(p,c,cc,{}); }
                distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); } };
    Buf8 ttmp = mkBuf8();
    long n_xform = 0;                       // transforms issued, so the chain can report its own cost
    auto NTT_res  = [&](Buf8& inC, Buf8& outE, uint32_t li){ ++n_xform; transpose8_dev(inC,ttmp); fwd_buf(ttmp,outE,li); };
    auto INTT_res = [&](Buf8& inE, Buf8& outC, uint32_t li){ ++n_xform; transpose8_dev(inE,ttmp); inv_buf(ttmp,outC,li); };

    auto ew = [&](const char* rdk, const char* cpk, Buf8& x, Buf8& y, Buf8& o,
                  std::vector<uint32_t> cargs, bool modmul) {
        KLBL = std::string(cpk).substr(8, std::string(cpk).size()-12);
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
        } };
    auto PWA = [&](uint32_t li){ uint64_t m = QP[li];
        return std::vector<uint32_t>{1u,(uint32_t)m,ctx[li].q_inv_32(),(uint32_t)(m&0xFFFF),(uint32_t)(m>>16)}; };
    auto pw_buf   = [&](Buf8& a, Buf8& b, Buf8& o, uint32_t li){ ew("dataflow/reader_modmul.cpp","compute/modmul_sfpi_full.cpp",a,b,o,PWA(li),true); };
    auto madd_buf = [&](Buf8& a, Buf8& b, Buf8& o, uint32_t li){ ew("dataflow/reader_ab.cpp","compute/modadd_map.cpp",a,b,o,{1u,(uint32_t)QP[li]},false); };
    auto msub_buf = [&](Buf8& a, Buf8& b, Buf8& o, uint32_t li){ ew("dataflow/reader_ab.cpp","compute/modsub_reduce.cpp",a,b,o,{(uint32_t)QP[li]},false); };

    // general (Ls -> Ld) approximate FBC, and the single-source special case the rescale needs
    auto fbc_res = [&](const std::vector<Buf8>& src, const std::vector<uint32_t>& srcIdx,
                       const std::vector<uint32_t>& dstIdx) -> std::vector<Buf8> {
        uint32_t Ls=(uint32_t)srcIdx.size(), Ld=(uint32_t)dstIdx.size();
        u128 S=1; for(uint32_t i=0;i<Ls;++i) S*=QP[srcIdx[i]];
        std::vector<uint32_t> ca; ca.push_back(Ls); ca.push_back(Ld);
        for(uint32_t i=0;i<Ls;++i){ uint32_t si=srcIdx[i]; uint64_t qi=QP[si];
            ca.push_back(ctx[si].to_montgomery((uint32_t)modinv((uint64_t)((S/qi)%qi), qi))); }
        for(uint32_t i=0;i<Ls;++i) for(uint32_t j=0;j<Ld;++j){ uint64_t pj=QP[dstIdx[j]];
            ca.push_back(ctx[dstIdx[j]].to_montgomery((uint32_t)((S/QP[srcIdx[i]])%pj))); }
        for(uint32_t i=0;i<Ls;++i) ca.push_back((uint32_t)QP[srcIdx[i]]);
        for(uint32_t i=0;i<Ls;++i) ca.push_back(ctx[srcIdx[i]].q_inv_32());
        for(uint32_t j=0;j<Ld;++j) ca.push_back((uint32_t)QP[dstIdx[j]]);
        for(uint32_t j=0;j<Ld;++j) ca.push_back(ctx[dstIdx[j]].q_inv_32());
        std::vector<Buf8> dst(Ld); for(uint32_t j=0;j<Ld;++j) dst[j]=mkBuf8();
        for(uint32_t base=0;base<NBLK;base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,NBLK-base);
            KLBL="baseconv_fbc"; emit([&]{ Program p=CreateProgram(); auto crs=crs_of(cnt);
                // depths = the ACTUAL widths; c_3 is reserved once per destination inside phase 2, so a
                // depth that is not a multiple of Ls straddles the buffer end (exp_Tb001 doc, §4)
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,Ls,I32); mkcb(p,crs,CBIndex::c_2,tt::DataFormat::UInt32,Ls,I32);
                mkcb(p,crs,CBIndex::c_3,tt::DataFormat::UInt32,Ls,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,Ld,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_gather.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_scatter.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/baseconv_fbc.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for(uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                    std::vector<uint32_t> ra; ra.push_back(Ls);
                    for(uint32_t i=0;i<Ls;++i) ra.push_back((uint32_t)(src[i][t]->address()));
                    SetRuntimeArgs(p,rd,c,ra);
                    std::vector<uint32_t> wa; wa.push_back(Ld); wa.push_back(Ld);
                    for(uint32_t j=0;j<Ld;++j) wa.push_back((uint32_t)(dst[j][t]->address()));
                    SetRuntimeArgs(p,wr,c,wa);
                    SetRuntimeArgs(p,cp,c,ca); }
                distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); }
        return dst; };
    auto bconv1_res = [&](Buf8 src, const std::vector<uint32_t>& dstIdx) -> std::vector<Buf8> {
        uint32_t Ld = (uint32_t)dstIdx.size();
        std::vector<uint32_t> ca; ca.push_back(Ld);
        for (auto dj : dstIdx) ca.push_back((uint32_t)QP[dj]);
        std::vector<Buf8> dst(Ld); for (uint32_t j=0;j<Ld;++j) dst[j]=mkBuf8();
        for (uint32_t base=0;base<NBLK;base+=EWC){ uint32_t cnt=std::min<uint32_t>(EWC,NBLK-base);
            KLBL="bconv_one"; emit([&]{ Program p=CreateProgram(); auto crs=crs_of(cnt);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32);
                mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,8,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_gather.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_scatter.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/bconv_one.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for(uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                    SetRuntimeArgs(p,rd,c,{1u,(uint32_t)src[t]->address()});
                    std::vector<uint32_t> wa{Ld,Ld};
                    for(uint32_t j=0;j<Ld;++j) wa.push_back((uint32_t)dst[j][t]->address());
                    SetRuntimeArgs(p,wr,c,wa);
                    SetRuntimeArgs(p,cp,c,ca); }
                distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); return wl; }); }
        return dst; };

    // index sets, once
    std::vector<uint32_t> qidx(LQ), pidx(LP);
    for (uint32_t i = 0; i < LQ; ++i) qidx[i] = i;
    for (uint32_t j = 0; j < LP; ++j) pidx[j] = LQ + j;

    uint32_t bad1=0, bad2=0, tot12=0, bad3=0, tot3=0, bad4=0, tot4=0, bad5=0, tot5=0, bad6=0, tot6=0;

    // A measurement run SKIPS the correctness ladder. Two reasons: its dispatches would contaminate a
    // drained capture (exp_Tb001 does the same for the same reason), and its per-stage buffers would
    // be allocated before the POOL and shift every cached address.
    const bool MEAS = std::getenv("TG000_TRACE") || std::getenv("TG000_DRAIN");

    // ---- [D1]/[D2] the four-step transform, both directions, on every Q limb ----
    if (!MEAS) {
        const auto& src = kase.A.c0;
        for (uint32_t li = 0; li < LQ; ++li) {
            auto want_e = ck.nt[li].fwd(src[li]);
            Buf8 cbuf = up_natC(src[li]), ebuf = mkBuf8();
            NTT_res(cbuf, ebuf, li);
            Buf8 e2 = up_natE(want_e, false, li), c2 = mkBuf8();
            INTT_res(e2, c2, li);
            t_fin(cq);
            auto gote = rd_natE(ebuf), gotc = rd_natC(c2);
            for (uint32_t n = 0; n < N; ++n) { ++tot12;
                if (gote[n] != want_e[n]) ++bad1;
                if (gotc[n] != src[li][n]) ++bad2; }
        }
        fmt::print("[D1] device NTT (coeff->eval)  vs host   mismatches = {}/{}  [{}]\n", bad1, tot12, bad1?"FAIL":"PASS");
        fmt::print("[D2] device INTT(eval->coeff)  vs host   mismatches = {}/{}  [{}]\n", bad2, tot12, bad2?"FAIL":"PASS");
    }

    // ---- [D3] the tensor: pointwise mod Q, no base extension ----
    // This is the step where CKKS and BFV part company. BFV needs 18 limbs and an exact lift here
    // (exp_Tb001b [D3]/[D4]); CKKS needs 8 limbs and four modmuls.
    if (!MEAS) {
        std::vector<std::vector<uint32_t>> re0, re1, re2;
        ck.tensor(kase.A, kase.B, re0, re1, re2);
        for (uint32_t li = 0; li < LQ; ++li) {
            std::vector<uint32_t> b0m(N), b1m(N);
            for (uint32_t n = 0; n < N; ++n) { b0m[n]=ctx[li].to_montgomery(kase.B.c0[li][n]); b1m[n]=ctx[li].to_montgomery(kase.B.c1[li][n]); }
            Buf8 a0e=mkBuf8(), a1e=mkBuf8(), b0e=mkBuf8(), b1e=mkBuf8();
            Buf8 a0c=up_natC(kase.A.c0[li]), a1c=up_natC(kase.A.c1[li]), b0c=up_natC(b0m), b1c=up_natC(b1m);
            NTT_res(a0c,a0e,li); NTT_res(a1c,a1e,li); NTT_res(b0c,b0e,li); NTT_res(b1c,b1e,li);
            Buf8 E0=mkBuf8(), E2=mkBuf8(), t01=mkBuf8(), t10=mkBuf8(), E1=mkBuf8();
            pw_buf(a0e,b0e,E0,li); pw_buf(a1e,b1e,E2,li); pw_buf(a0e,b1e,t01,li); pw_buf(a1e,b0e,t10,li);
            madd_buf(t01,t10,E1,li);
            Buf8 c0=mkBuf8(), c1=mkBuf8(), c2=mkBuf8();
            INTT_res(E0,c0,li); INTT_res(E1,c1,li); INTT_res(E2,c2,li);
            t_fin(cq);
            auto g0=rd_natC(c0), g1=rd_natC(c1), g2=rd_natC(c2);
            for (uint32_t n = 0; n < N; ++n) { tot4 += 3;
                if (g0[n]!=re0[li][n]) ++bad4;
                if (g1[n]!=re1[li][n]) ++bad4;
                if (g2[n]!=re2[li][n]) ++bad4; }
        }
        std::swap(bad3, bad4); std::swap(tot3, tot4);      // report as D3
        fmt::print("[D3] device tensor (pointwise mod Q) vs reference   mismatches = {}/{}  [{}]\n",
                   bad3, tot3, bad3?"FAIL":"PASS");
    }

    // ---- evk in the eval domain, Montgomery, resident. Key material, not multiply cost. ----
    std::map<int,Buf8> evk0m, evk1m;
    for (uint32_t j = 0; j < DNUM; ++j)
        for (uint32_t li = 0; li < LQP; ++li) {
            evk0m[j*100+li] = up_natE(ck.nt[li].fwd(kase.evk0[j][li]), true, li);
            evk1m[j*100+li] = up_natE(ck.nt[li].fwd(kase.evk1[j][li]), true, li);
        }
    std::map<uint32_t,Buf8> PinvC;
    for (uint32_t i = 0; i < LQ; ++i)
        PinvC[i] = fill8(ctx[i].to_montgomery((uint32_t)modinv(mod_small(ck.Pprod, QP[i]), QP[i])));
    // per rescale STEP k: q_L^-1 for each surviving limb, and qHalf for the centring shift
    std::vector<std::map<uint32_t,Buf8>> qLinvC(g_ndrop), HalfC(g_ndrop), HalfE(g_ndrop);
    for (uint32_t k = 0; k < g_ndrop; ++k) {
        const uint32_t d = DROPI(k); const uint64_t qL = QP[d], qHalf = qL / 2;
        for (uint32_t i = 0; i <= d; ++i) {
            if (i < d) qLinvC[k][i] = fill8(ctx[i].to_montgomery((uint32_t)modinv(qL % QP[i], QP[i])));
            HalfC[k][i] = fill8((uint32_t)(qHalf % QP[i]));   // natural, modadd_map is not Montgomery
            // ...and the SAME shift in the eval domain. Adding qHalf to every COEFFICIENT is adding
            // the polynomial qHalf*(1 + x + ... + x^(N-1)), whose transform is not a constant -- so it
            // is computed once on the host and uploaded, not filled.
            HalfE[k][i] = up_natE(ck.nt[i].fwd(std::vector<uint32_t>(N, (uint32_t)(qHalf % QP[i]))), false, i);
        }
    }

    // ---- the key-switch, as a reusable device routine (used by [D4] and by the chain) ----
    // e2c: e2 in COEFFICIENT form per Q limb; e2e: the same in EVAL form (the digit's own limbs need
    // no transform, they are already there).
    auto relin_dev = [&](std::map<int,Buf8>& e2c, std::map<int,Buf8>& e2e,
                         std::map<int,Buf8>& k0, std::map<int,Buf8>& k1) {
        std::map<int,Buf8> acc0, acc1;
        // Digit 0 SEEDS the accumulator instead of being added to a zero buffer. Adding zero is a
        // no-op arithmetically, but the zero buffers cost 24 host uploads INSIDE the multiply -- host
        // boundaries, which break the Metal Trace into extra segments -- plus 24 pointless modadds.
        for (uint32_t j = 0; j < DNUM; ++j) {
            L("keyswitch","ks.modup_fbc","j"+std::to_string(j));
            std::vector<Buf8> xs; for (auto li : ck.group[j]) xs.push_back(e2c[li]);
            auto o = fbc_res(xs, ck.group[j], ck.comp[j]);
            std::map<int,Buf8> vje;
            for (auto li : ck.group[j]) vje[li] = e2e[li];
            L("keyswitch","ks.modup_ntt","j"+std::to_string(j));
            for (uint32_t k = 0; k < ck.comp[j].size(); ++k) { uint32_t li = ck.comp[j][k];
                Buf8 ve = mkBuf8(); NTT_res(o[k], ve, li); vje[li] = ve; }
            L("keyswitch","ks.evk_mac","j"+std::to_string(j));
            for (uint32_t li = 0; li < LQP; ++li) {
                Buf8 p0=mkBuf8(), p1=mkBuf8();
                pw_buf(vje[li], evk0m[j*100+li], p0, li);
                pw_buf(vje[li], evk1m[j*100+li], p1, li);
                if (j == 0) { acc0[li]=p0; acc1[li]=p1; }
                else { Buf8 n0=mkBuf8(), n1=mkBuf8();
                       madd_buf(acc0[li],p0,n0,li); madd_buf(acc1[li],p1,n1,li);
                       acc0[li]=n0; acc1[li]=n1; } }
        }
        // mod-down. Only the P limbs leave the eval domain -- they are the FBC's sources. P^-1 is a
        // scalar and the NTT is Z_q-linear, so the subtract and the scale happen in EVAL against the
        // accumulator the MAC already holds: NTT((a-f)*Pinv) = Pinv*(a_eval - NTT(f)).
        auto moddown = [&](std::map<int,Buf8>& acc, std::map<int,Buf8>& out, const char* md) {
            L("keyswitch","ks.moddown_intt",md);
            std::vector<Buf8> aP;
            for (auto pp : pidx) { Buf8 c = mkBuf8(); INTT_res(acc[pp], c, pp); aP.push_back(c); }
            L("keyswitch","ks.moddown_fbc",md);
            auto f = fbc_res(aP, pidx, qidx);
            L("keyswitch","ks.moddown_tail",md);
            for (uint32_t i = 0; i < LQ; ++i) {
                Buf8 fe=mkBuf8(), diff=mkBuf8(), ke=mkBuf8();
                NTT_res(f[i], fe, i);
                msub_buf(acc[i], fe, diff, i);
                pw_buf(diff, PinvC[i], ke, i);
                out[i] = ke; } };
        moddown(acc0, k0, "acc0"); moddown(acc1, k1, "acc1");
    };

    // ---- [D4] relinearisation, fed the reference e2 ----
    if (!MEAS) {
        std::vector<std::vector<uint32_t>> re0, re1, re2, rk0, rk1;
        ck.tensor(kase.A, kase.B, re0, re1, re2);
        ck.relin(re2, rk0, rk1);
        std::map<int,Buf8> e2c, e2e, k0, k1;
        for (uint32_t li = 0; li < LQ; ++li) {
            e2c[li] = up_natC(re2[li]);
            e2e[li] = up_natE(ck.nt[li].fwd(re2[li]), false, li);
        }
        relin_dev(e2c, e2e, k0, k1);
        std::vector<Buf8> kc(LQ);
        for (uint32_t i = 0; i < LQ; ++i) { kc[i] = mkBuf8(); INTT_res(k0[i], kc[i], i); }
        t_fin(cq);
        for (uint32_t i = 0; i < LQ; ++i) {
            auto got = rd_natC(kc[i]);
            for (uint32_t n = 0; n < N; ++n) { ++tot5; if (got[n] != rk0[i][n]) ++bad5; }
        }
        std::swap(bad4,bad5); std::swap(tot4,tot5);        // report as D4
        fmt::print("[D4] device relinearise (dnum={} hybrid key-switch) vs reference   mismatches = {}/{}  [{}]\n",
                   DNUM, bad4, tot4, bad4?"FAIL":"PASS");
    }

    // ---- the rescale, as a device routine: drop g_ndrop primes, in COEFFICIENT form ----
    // exp_Tb001 does this in the eval domain because its I/O contract is eval; here the output must be
    // coefficients, so the same three kernels run one domain earlier and 14 transforms disappear.
    //
    // CENTRING adds one modadd per surviving limb per step and NOTHING else -- no new kernel. The
    // identity is that a plain [0,q_L) rescale of (c + qHalf) equals the CENTRED rescale of c, because
    // [c + qHalf]_{q_L} = [c]_{q_L} + qHalf - q_L*[[c]_{q_L} > qHalf], so the shift cancels and what
    // gets subtracted is the centred remainder. Verified exhaustively on random cases including the
    // real 28-bit primes; see docs/he/exp_tg000.md.
    // The EVAL-domain rescale, for the eval I/O contract. Same three kernels, one domain later:
    // only the dropped limb leaves the eval domain (it is bconv_one's source), and the corrections
    // come back. P^-1 and q_L^-1 are scalars and the NTT is Z_q-linear, so the subtract and the scale
    // still happen in eval:  NTT((c - corr)*qLinv) = qLinv*(c_eval - NTT(corr)).
    //
    // This is where the coefficient contract wins and the eval contract pays: in coefficients a drop
    // step costs ZERO transforms, in eval it costs 1 INTT + (d) NTTs -- and with a 2-prime scale there
    // are two steps. See docs/he/exp_tg000.md for the accounting.
    auto rescale_dev_eval = [&](std::vector<Buf8>& cin, std::vector<Buf8>& out) {
        std::vector<Buf8> c = cin;
        for (uint32_t k = 0; k < g_ndrop; ++k) {
            const uint32_t d = DROPI(k);
            std::vector<uint32_t> dst(d); for (uint32_t i = 0; i < d; ++i) dst[i] = i;
            if (g_centre) {                       // shift by qHalf -- in eval, that is + NTT(qHalf*sum x^n)
                std::vector<Buf8> cs(d + 1);
                for (uint32_t i = 0; i <= d; ++i) { cs[i] = mkBuf8(); madd_buf(c[i], HalfE[k][i], cs[i], i); }
                c = cs;
            }
            Buf8 lastc = mkBuf8(); INTT_res(c[d], lastc, d);          // the dropped limb -> coefficients
            auto corr = bconv1_res(lastc, dst);
            std::vector<Buf8> o(d);
            for (uint32_t i = 0; i < d; ++i) {
                Buf8 ce = mkBuf8(), diff = mkBuf8(); o[i] = mkBuf8();
                NTT_res(corr[i], ce, i);
                msub_buf(c[i], ce, diff, i);
                pw_buf(diff, qLinvC[k][i], o[i], i);
            }
            c = o;
        }
        for (uint32_t i = 0; i < LOUT(); ++i) out[i] = c[i]; };

    auto rescale_dev = [&](std::vector<Buf8>& cin, std::vector<Buf8>& out) {
        std::vector<Buf8> c = cin;
        for (uint32_t k = 0; k < g_ndrop; ++k) {
            const uint32_t d = DROPI(k);
            std::vector<uint32_t> dst(d); for (uint32_t i = 0; i < d; ++i) dst[i] = i;
            if (g_centre) {                                  // c <- c + qHalf, every limb incl. the dropped one
                std::vector<Buf8> cs(d + 1);
                for (uint32_t i = 0; i <= d; ++i) { cs[i] = mkBuf8(); madd_buf(c[i], HalfC[k][i], cs[i], i); }
                c = cs;
            }
            auto corr = bconv1_res(c[d], dst);
            std::vector<Buf8> o(d);
            for (uint32_t i = 0; i < d; ++i) {
                Buf8 diff = mkBuf8(); o[i] = mkBuf8();
                msub_buf(c[i], corr[i], diff, i);
                pw_buf(diff, qLinvC[k][i], o[i], i);
            }
            c = o;
        }
        for (uint32_t i = 0; i < LOUT(); ++i) out[i] = c[i]; };

    // ---- [D5] rescale, fed the reference degree-1 ciphertext ----
    if (!MEAS) {
        std::vector<std::vector<uint32_t>> re0, re1, re2, rk0, rk1;
        ck.tensor(kase.A, kase.B, re0, re1, re2);
        ck.relin(re2, rk0, rk1);
        std::vector<std::vector<uint32_t>> r0(LQ, std::vector<uint32_t>(N));
        for (uint32_t i = 0; i < LQ; ++i)
            for (uint32_t n = 0; n < N; ++n) r0[i][n] = (uint32_t)(((u128)re0[i][n] + rk0[i][n]) % QP[i]);
        auto want = ck.rescale(r0);
        std::vector<Buf8> c(LQ), o(LOUT());
        for (uint32_t i = 0; i < LQ; ++i) c[i] = up_natC(r0[i]);
        rescale_dev(c, o);
        t_fin(cq);
        for (uint32_t i = 0; i < LOUT(); ++i) {
            auto got = rd_natC(o[i]);
            for (uint32_t n = 0; n < N; ++n) { ++tot6; if (got[n] != want[i][n]) ++bad6; }
        }
        std::swap(bad5,bad6); std::swap(tot5,tot6);        // report as D5
        fmt::print("[D5] device rescale (drop {} prime(s), centred={}) vs reference   mismatches = {}/{}  [{}]\n",
                   g_ndrop, g_centre?1:0, bad5, tot5, bad5?"FAIL":"PASS");
    }

    // ======================================================================================
    // THE RESIDENT CKKS MULTIPLY — the measured unit
    //
    // Inputs are uploaded ONCE, outside; the routine itself contains no t_wr/t_rd, so it is one
    // uninterrupted run of dispatches and captures as a SINGLE Metal Trace segment. That property is
    // what makes the latency metrics meaningful: a host boundary in the middle would split the trace
    // and put host time inside the "device busy" window.
    // ======================================================================================
    // ---- inputs, and the SETUP transform that is deliberately not part of the measured op ----
    // EVAL I/O (the default, and exp_Tb001's contract): the ciphertext the multiply consumes is
    // already in the evaluation domain, so the coeff->eval conversion happens ONCE here and never
    // again. It still runs on device -- the same four-step NTT, the same kernels -- it is simply
    // encryption-side work, not multiply work, so it sits outside the trace capture and outside every
    // latency number. TG000_COEFF_IO=1 puts it back inside for an A/B in the same binary.
    const bool COEFF_IO = std::getenv("TG000_COEFF_IO") != nullptr;
    std::map<int,Buf8> A0c, A1c, B0c, B1c;                  // coefficient-domain inputs
    std::map<int,Buf8> A0e, A1e, B0e, B1e;                  // ...and their eval form
    for (uint32_t li = 0; li < LQ; ++li) {
        std::vector<uint32_t> b0m(N), b1m(N);
        for (uint32_t n = 0; n < N; ++n) { b0m[n]=ctx[li].to_montgomery(kase.B.c0[li][n]); b1m[n]=ctx[li].to_montgomery(kase.B.c1[li][n]); }
        A0c[li]=up_natC(kase.A.c0[li]); A1c[li]=up_natC(kase.A.c1[li]);
        B0c[li]=up_natC(b0m);           B1c[li]=up_natC(b1m);
    }
    if (!COEFF_IO) {                                        // ON DEVICE, but at setup
        for (uint32_t li = 0; li < LQ; ++li) {
            A0e[li]=mkBuf8(); NTT_res(A0c[li],A0e[li],li);
            A1e[li]=mkBuf8(); NTT_res(A1c[li],A1e[li],li);
            B0e[li]=mkBuf8(); NTT_res(B0c[li],B0e[li],li);
            B1e[li]=mkBuf8(); NTT_res(B1c[li],B1e[li],li);
        }
    }
    t_fin(cq);
    fmt::print("exp_Tg000  ciphertext I/O domain: {}   input NTTs inside the multiply: {}\n",
               COEFF_IO ? "COEFFICIENT (TG000_COEFF_IO=1)" : "EVALUATION (default)", COEFF_IO ? 4*LQ : 0);

    auto dev_mul = [&](std::vector<Buf8>& O0, std::vector<Buf8>& O1) {
        PI = 0; WI = 0; seg_idx = 0; seg_started = false; hold.clear();
        // 1. inputs. Eval I/O: nothing to do, they are already in the E layout. Coeff I/O: 4 x LQ
        //    forward transforms. B is Montgomery-encoded host-side, which rides through the NTT
        //    unchanged (mont is a scalar; the transform is Z_q-linear).
        std::map<int,Buf8> a0e, a1e, b0e, b1e;
        if (!COEFF_IO) { a0e=A0e; a1e=A1e; b0e=B0e; b1e=B1e; }
        else {
            L("input","input.ntt","-");
            for (uint32_t li = 0; li < LQ; ++li) {
                a0e[li]=mkBuf8(); NTT_res(A0c[li],a0e[li],li);
                a1e[li]=mkBuf8(); NTT_res(A1c[li],a1e[li],li);
                b0e[li]=mkBuf8(); NTT_res(B0c[li],b0e[li],li);
                b1e[li]=mkBuf8(); NTT_res(B1c[li],b1e[li],li);
            }
        }
        // 2. tensor, pointwise mod Q
        std::map<int,Buf8> E0, E1, E2;
        for (uint32_t li = 0; li < LQ; ++li) {
            L("tensor","tensor.pointwise","q"+std::to_string(li));
            Buf8 e0=mkBuf8(), e2=mkBuf8(), t01=mkBuf8(), t10=mkBuf8(), e1=mkBuf8();
            pw_buf(a0e[li],b0e[li],e0,li); pw_buf(a1e[li],b1e[li],e2,li);
            pw_buf(a0e[li],b1e[li],t01,li); pw_buf(a1e[li],b0e[li],t10,li);
            madd_buf(t01,t10,e1,li);
            E0[li]=e0; E1[li]=e1; E2[li]=e2;
        }
        // 3. relinearise e2 (the FBC's sources are coefficients, so e2 comes back once)
        std::map<int,Buf8> e2c, k0, k1;
        L("keyswitch","ks.to_coeff","e2");
        for (uint32_t li = 0; li < LQ; ++li) { Buf8 c = mkBuf8(); INTT_res(E2[li], c, li); e2c[li] = c; }
        relin_dev(e2c, E2, k0, k1);
        // 4. combine, in eval
        std::vector<Buf8> R0(LQ), R1(LQ);
        L("combine","combine.add","-");
        for (uint32_t i = 0; i < LQ; ++i) {
            R0[i]=mkBuf8(); R1[i]=mkBuf8();
            madd_buf(E0[i], k0[i], R0[i], i);
            madd_buf(E1[i], k1[i], R1[i], i);
        }
        // 5. rescale. Eval I/O keeps everything in the eval domain and the output IS eval; coeff I/O
        //    converts all 8 limbs back first and then the rescale costs no transform at all.
        if (!COEFF_IO) {
            L("rescale","rescale.drop","r0"); rescale_dev_eval(R0, O0);
            L("rescale","rescale.drop","r1"); rescale_dev_eval(R1, O1);
        } else {
            std::vector<Buf8> C0(LQ), C1(LQ);
            L("output","output.intt","-");
            for (uint32_t i = 0; i < LQ; ++i) { C0[i]=mkBuf8(); C1[i]=mkBuf8(); INTT_res(R0[i],C0[i],i); INTT_res(R1[i],C1[i],i); }
            L("rescale","rescale.drop","r0"); rescale_dev(C0, O0);
            L("rescale","rescale.drop","r1"); rescale_dev(C1, O1);
        }
    };

    // ---- [D6]/[D7] the whole multiply from its own intermediates ----
    uint32_t bad7 = 0, tot7 = 0;
    auto check_output = [&](std::vector<Buf8>& O0, std::vector<Buf8>& O1, bool report) {
        Ciphertext got; got.scale = kase.R.scale;
        got.c0.resize(LOUT()); got.c1.resize(LOUT());
        uint32_t b = 0, t = 0;
        for (uint32_t i = 0; i < LOUT(); ++i) {
            // eval I/O: the device output IS the eval form, so it comes back through the host inverse
            // transform before it can be compared with the fixture (which is coefficient domain).
            got.c0[i] = COEFF_IO ? rd_natC(O0[i]) : ck.nt[i].inv(rd_natE(O0[i]));
            got.c1[i] = COEFF_IO ? rd_natC(O1[i]) : ck.nt[i].inv(rd_natE(O1[i]));
            for (uint32_t n = 0; n < N; ++n) { t += 2;
                if (got.c0[i][n] != kase.R.c0[i][n]) ++b;
                if (got.c1[i][n] != kase.R.c1[i][n]) ++b; }
        }
        if (report) {
            bad7 = b; tot7 = t;
            fmt::print("[D6] WHOLE MULTIPLY on device vs the FIXED case   mismatches = {}/{}  [{}]\n",
                       b, t, b ? "FAIL" : "PASS");
            std::vector<cdouble> want(NSLOT);
            for (uint32_t j = 0; j < NSLOT; ++j) want[j] = kase.z1[j] * kase.z2[j];
            auto dec = ck.decrypt(got);
            auto e = slot_err(dec, want);
            const double TOL = 1e-2;
            fmt::print("[D7] decode(decrypt(device out)) vs z1(.)z2   max={:.3e}  rms={:.3e}  ~{:.1f} bits  [{}]\n",
                       e.maxe, e.rms, e.bits(), e.maxe < TOL ? "PASS" : "FAIL");
            fmt::print("     slot 0: got ({:+.6f},{:+.6f})   want ({:+.6f},{:+.6f})\n",
                       dec[0].real(), dec[0].imag(), want[0].real(), want[0].imag());
            if (e.maxe >= TOL) bad7 += 1;
        }
        return b; };

    if (std::getenv("TG000_CHAIN")) {
        const long enq0 = prof::n_enq, xf0 = n_xform;
        std::vector<Buf8> O0(LOUT()), O1(LOUT());
        dev_mul(O0, O1);
        t_fin(cq);
        fmt::print("     chain cost: {} program dispatches, {} transforms ({} I/O, G={})\n",
                   prof::n_enq - enq0, n_xform - xf0, COEFF_IO ? "coefficient" : "evaluation", NCORES);
        check_output(O0, O1, true);
    }

    // ======================================================================================
    // TG000_TRACE — the four latency metrics, exactly as exp_Tb001 defines them.
    //   REC   build + cache every workload   (addresses are pinned by the POOL)
    //   CAP   re-run, wrapping each segment in a Metal Trace
    //   REP   replay -> validate, then time
    // TG000_PROF additionally profiles ONE traced replay and prints the busy_metrics.py command with
    // the dispatch count and host numbers of THAT SAME replay filled in -- all four metrics have to
    // come from one execution or the guaranteed orderings can appear violated.
    // ======================================================================================
    if (std::getenv("TG000_TRACE")) {
        CACHE_ON = true;
        std::vector<Buf8> O0(LOUT()), O1(LOUT());
        CACHE_REC = true;  TR_PHASE = 0;
        fmt::print("REC: warm+cache workloads ({} cores)...\n", NCORES);
        dev_mul(O0, O1); t_fin(cq); CACHE_REC = false;
        TR_PHASE = 1; fmt::print("CAP: recording traces...\n");
        dev_mul(O0, O1);
        if (seg_open) { mesh->end_mesh_trace(0, SEG.back()); seg_open = false; }
        t_fin(cq);
        fmt::print("captured {} trace segment(s) from {} cached workloads\n", SEG.size(), WC.size());
        TR_PHASE = 2;
        dev_mul(O0, O1); t_fin(cq);
        uint32_t b = check_output(O0, O1, true);
        fmt::print("TRACED RESIDENT device-vs-fixed-case = {}  [{}]\n", b, b ? "FAIL" : "PASS");

        int it = 5; if (const char* e = std::getenv("TG000_ITERS")) { int v = std::atoi(e); if (v > 0) it = v; }
        // (1) host wall, non-blocking replays
        dev_mul(O0, O1); t_fin(cq); prof::reset();
        auto t0 = prof::hrc::now();
        for (int i = 0; i < it; ++i) { dev_mul(O0, O1); }
        t_fin(cq);
        double wall = prof::msd(t0) / it;
        // (2) host device busy: blocking segment replays, sum of the host's stopwatch around each
        MEASURE_DEV = true; prof::g_replay = 0; prof::n_replay = 0;
        for (int i = 0; i < it; ++i) dev_mul(O0, O1);
        t_fin(cq); MEASURE_DEV = false;
        const double devbusy = prof::g_replay / it;
        fmt::print("TIMING: TRACED RESIDENT CKKS ct_mul\n");
        fmt::print("  host wall            : {:.3f} ms/op   ({} cores, {} iters)\n", wall, NCORES, it);
        fmt::print("  host device busy     : {:.3f} ms/op   ({:.1f} segment replays)\n", devbusy, (double)prof::n_replay/it);

        if (std::getenv("TG000_PROF")) {
            t_fin(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);
            std::string csv = std::string(std::getenv("TT_METAL_HOME")) + "/generated/profiler/.logs/profile_log_device.csv";
            std::remove(csv.c_str());
            const auto tw0 = prof::hrc::now();
            MEASURE_DEV = true; prof::g_replay = 0; prof::n_replay = 0;
            dev_mul(O0, O1);
            MEASURE_DEV = false;
            t_fin(cq);
            const double prof_wall = prof::msd(tw0);
            const size_t prof_disp = WI;
            tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);
            fmt::print("TG000_PROF: profiled ONE traced replay ({} cores) -> {}\n", NCORES, csv);
            fmt::print("TG000_PROF: this replay  wall time = {:.3f} ms   dispatches = {}\n", prof_wall, prof_disp);
            fmt::print("TG000_PROF: this replay  host-measured device busy = {:.3f} ms over {} segment replays\n",
                       prof::g_replay, prof::n_replay);
            fmt::print("TG000_PROF: 1-1 check -> tools/traces/busy_metrics.py <csv> --expect {} --wall-ms {:.3f} --host-busy-ms {:.3f}\n",
                       prof_disp, prof_wall, prof::g_replay);
            dump_labels();
        }
        bad7 += b;
    }

    // ======================================================================================
    // TG000_DRAIN — a FULL device trace of ONE multiply with no dropped markers, by emptying the
    // profiler buffer every K dispatches. This is the capture the per-kernel breakdown needs; the
    // traced-replay capture above drops markers once the DRAM buffer fills.
    // ======================================================================================
    if (std::getenv("TG000_DRAIN")) {
        CACHE_ON = true; CACHE_REC = true;
        std::vector<Buf8> O0(LOUT()), O1(LOUT());
        fmt::print("REC: warm+cache workloads ({} cores)...\n", NCORES);
        dev_mul(O0, O1); t_fin(cq); CACHE_REC = false;
        std::string csv = std::string(std::getenv("TT_METAL_HOME")) + "/generated/profiler/.logs/profile_log_device.csv";
        distributed::Finish(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh); std::remove(csv.c_str());
        int K = 100; if (const char* e = std::getenv("TG000_DRAIN_K")) { int v = std::atoi(e); if (v > 0) K = v; }
        prof::drain_ctr = 0; prof::drain_every = K;
        prof::drain_fn = [&]{ distributed::Finish(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh); };
        fmt::print("DRAINED capture of ONE cached ct_mul (drain every {} dispatches)...\n", K);
        const long enq0 = prof::n_enq;
        dev_mul(O0, O1);
        const long host_dispatches = prof::n_enq - enq0;
        prof::drain_every = 0; distributed::Finish(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);
        fmt::print("TG000_DRAIN: host dispatches in this capture = {}  (trace must show exactly this many)\n",
                   host_dispatches);
        dump_labels();
        fmt::print("TG000_DRAIN: FULL trace (no dropped markers) -> {}\n", csv);
        bad7 += check_output(O0, O1, true);
    }

    bool ok = (bad1==0 && bad2==0 && bad3==0 && bad4==0 && bad5==0 && bad6==0 && bad7==0);
    (void)rd_natE; (void)fill8;
    prof::seg_hook = nullptr; prof::drain_fn = nullptr;
    ok &= mesh->close();
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const uint64_t seed = (argc > 1) ? std::stoull(argv[1]) : 0;

    // The two precision levers, read BEFORE the scheme is constructed -- Delta and the level chain
    // both depend on them.
    if (const char* e = std::getenv("TG000_NDROP")) { int v = std::atoi(e); if (v >= 1 && v <= 3) g_ndrop = (uint32_t)v; }
    if (std::getenv("TG000_NOCENTRE")) g_centre = false;

    CKKS ck(seed);
    std::printf("exp_Tg000  CKKS ct_mul, decrypt+decode verified   N=%u  slots=%u  LQ=%u (%d-bit Q)  LP=%u  dnum=%u\n",
                N, NSLOT, LQ, bitlen(ck.Qprod), LP, DNUM);
    std::printf("  Q limbs:"); for (uint32_t i = 0; i < LQ; ++i) std::printf(" %llu", (unsigned long long)ck.QP[i]); std::printf("\n");
    std::printf("  scale Delta = %llu = 2^%.2f  (product of the %u prime(s) the rescale drops, so the\n"
                "               output scale is Delta again)   rescale remainder: %s   level %u -> %u\n",
                (unsigned long long)ck.DeltaI, std::log2((double)ck.DeltaI), g_ndrop,
                g_centre ? "CENTRED" : "non-centred [0,q_L)", LQ, LOUT());
    std::printf("  seed=%llu   secret key ternary, error uniform in [-2,2]\n", (unsigned long long)seed);

    bool pass = true;
    std::mt19937_64 mr(seed + 12345);
    auto rand_slots = [&]() {
        std::vector<cdouble> z(NSLOT);
        for (auto& v : z) {
            double re = (double)(mr() % 2000001) / 1000000.0 - 1.0;
            double im = (double)(mr() % 2000001) / 1000000.0 - 1.0;
            v = cdouble(re, im);
        }
        return z; };
    auto z1 = rand_slots(), z2 = rand_slots();

    // ---- 0. the encoder is its own inverse-pair. Everything downstream assumes it. ----
    {
        auto a = ck.emb.inverse(z1);
        auto back = ck.emb.forward(a);
        auto e = slot_err(back, z1);
        std::printf("[0] canonical embedding round trip      max err = %.3e  [%s]\n",
                    e.maxe, e.maxe < 1e-9 ? "PASS" : "FAIL");
        pass &= (e.maxe < 1e-9);
    }

    // ---- 1. encrypt / decrypt round trip, all N/2 slots ----
    {
        auto ct = ck.encrypt(ck.encode(z1, ck.Delta), ck.Delta);
        auto got = ck.decrypt(ct);
        auto e = slot_err(got, z1);
        std::printf("[1] decrypt(encrypt(z))                 max = %.3e  rms = %.3e  ~%.1f bits  [%s]\n",
                    e.maxe, e.rms, e.bits(), e.maxe < 1e-5 ? "PASS" : "FAIL");
        pass &= (e.maxe < 1e-5);
    }

    // ---- 2. HE add ----
    {
        auto a = ck.encrypt(ck.encode(z1, ck.Delta), ck.Delta);
        auto b = ck.encrypt(ck.encode(z2, ck.Delta), ck.Delta);
        auto got = ck.decrypt(ck.add_ct(a, b));
        std::vector<cdouble> want(NSLOT);
        for (uint32_t j = 0; j < NSLOT; ++j) want[j] = z1[j] + z2[j];
        auto e = slot_err(got, want);
        std::printf("[2] decrypt(enc(z1) + enc(z2))          max = %.3e  rms = %.3e  ~%.1f bits  [%s]\n",
                    e.maxe, e.rms, e.bits(), e.maxe < 1e-5 ? "PASS" : "FAIL");
        pass &= (e.maxe < 1e-5);
    }

    // ---- 3. THE TEST THAT MATTERS: decrypt(enc(z1) (x) enc(z2)) == z1 (.) z2 ----
    // Nothing but a semantically correct CKKS multiply passes this: it needs the encoder to be a ring
    // homomorphism, the tensor to be right, the relinearisation to actually cancel the s^2 term, and
    // the rescale to land the output back on scale Delta.
    {
        auto a = ck.encrypt(ck.encode(z1, ck.Delta), ck.Delta);
        auto b = ck.encrypt(ck.encode(z2, ck.Delta), ck.Delta);
        auto out = ck.multiply(a, b);
        auto got = ck.decrypt(out);
        std::vector<cdouble> want(NSLOT);
        for (uint32_t j = 0; j < NSLOT; ++j) want[j] = z1[j] * z2[j];
        auto e = slot_err(got, want);
        std::printf("[3] decrypt(enc(z1) (x) enc(z2))        max = %.3e  rms = %.3e  ~%.1f bits  [%s]\n",
                    e.maxe, e.rms, e.bits(), e.maxe < 1e-2 ? "PASS" : "FAIL");
        std::printf("      level %u -> %u limbs, scale %.0f -> %.0f (unchanged, by construction)\n",
                    a.L(), out.L(), (double)a.scale, (double)out.scale);
        std::printf("      slot 0: z1=(%+.4f,%+.4f) z2=(%+.4f,%+.4f) -> want (%+.6f,%+.6f) got (%+.6f,%+.6f)\n",
                    z1[0].real(), z1[0].imag(), z2[0].real(), z2[0].imag(),
                    want[0].real(), want[0].imag(), got[0].real(), got[0].imag());
        pass &= (e.maxe < 1e-2);

        // TG000_NOISE: where the precision actually goes. Reported because "max slot error" alone
        // hides its own shape -- if max/rms is ~4 the error is the Gaussian one the noise analysis
        // predicts, and if it is much larger something COHERENT is leaking into a few slots.
        if (std::getenv("TG000_NOISE")) {
            std::vector<double> d(NSLOT);
            for (uint32_t j = 0; j < NSLOT; ++j) d[j] = std::abs(got[j] - want[j]);
            std::vector<double> srt = d;
            std::sort(srt.begin(), srt.end());
            std::printf("      slot error distribution: p50=%.3e p90=%.3e p99=%.3e max=%.3e  max/rms=%.1f\n",
                        srt[NSLOT/2], srt[NSLOT*9/10], srt[NSLOT*99/100], srt[NSLOT-1], e.maxe/e.rms);
            for (uint32_t k = 0; k < 4; ++k) {
                uint32_t worst = (uint32_t)(std::find(d.begin(), d.end(), srt[NSLOT-1-k]) - d.begin());
                std::printf("        #%u worst slot j=%-5u  5^j mod 2N = %-6u  err=%.3e\n",
                            k, worst, ck.emb.rot[worst], d[worst]);
            }
            // The slot error IS the canonical embedding of the coefficient-domain error polynomial,
            // so inverting it recovers that polynomial exactly -- no convolution needed. A nonzero
            // MEAN there is a systematic offset, and it is what the small-|5^j| slots amplify:
            // sum_k zeta^(gk) = -2/(zeta^g - 1), whose magnitude is ~2N/(pi*g).
            std::vector<cdouble> dz(NSLOT);
            for (uint32_t j = 0; j < NSLOT; ++j) dz[j] = got[j] - want[j];
            auto ec = ck.emb.inverse(dz);
            double mu = 0, sd = 0;
            for (uint32_t k = 0; k < N; ++k) mu += ec[k];
            mu /= (double)N;
            for (uint32_t k = 0; k < N; ++k) sd += (ec[k]-mu)*(ec[k]-mu);
            sd = std::sqrt(sd/(double)N);
            std::printf("      coefficient-domain error: mean=%.3e (%.2f units of 1/Delta)  sd=%.3e (%.2f units)\n",
                        mu, mu*(double)ck.DeltaI, sd, sd*(double)ck.DeltaI);
            std::printf("      predicted slot error at 5^j=1 from the mean alone: %.3e\n",
                        std::fabs(mu) * 2.0 * (double)N / M_PI);
            auto m1e = ck.encode(z1, ck.Delta);
            double ms = 0; for (auto v : m1e) ms += (double)v * (double)v;
            ms = std::sqrt(ms / (double)N);
            double s2n = 0; for (auto v : ck.s) s2n += (double)(v * v);
            // The two candidate sources, side by side. The second wins by more than an order of
            // magnitude, which is the whole point of printing them.
            std::printf("      sd(encoded m) = %.3e\n", ms);
            std::printf("      predicted (m1*e2 + m2*e1)/Delta          = %6.2f units   (the LWE cross terms)\n",
                        std::sqrt(2.0 * (double)N * ms * ms * 2.0) / (double)ck.DeltaI);
            std::printf("      predicted rescale residual * s           = %6.2f units   (sqrt(||s||^2/12), non-centred adds a mean)\n",
                        std::sqrt(s2n / 12.0));
        }
    }

    // ---- 4. the FIXED reference case ----
    // TG000_GEN=<path> regenerates it; otherwise it is READ and the device is checked against the
    // bytes on disk, so a failure reproduces on any card and a host bug cannot cancel a device bug.
    Case kase;
    {
        const char* gen = std::getenv("TG000_GEN");
        const char* path = gen ? gen : (argc > 2 ? argv[2] : "reference/he/tg000_case.bin");
        if (gen) {
            Case c;
            c.scale = ck.Delta; c.sk = ck.s; c.evk0 = ck.evk0; c.evk1 = ck.evk1;
            c.z1 = z1; c.z2 = z2;
            c.A = ck.encrypt(ck.encode(z1, ck.Delta), ck.Delta);
            c.B = ck.encrypt(ck.encode(z2, ck.Delta), ck.Delta);
            c.R = ck.multiply(c.A, c.B);
            if (!write_case(path, ck, c)) { std::printf("could not write %s\n", path); return 1; }
            std::printf("[4a] wrote reference case -> %s\n", path);
        }
        if (!read_case(path, kase)) {
            std::printf("[4a] reference case %s missing or stale — regenerate with "
                        "TG000_GEN=%s ./build/exp_tg000 %llu\n", path, path, (unsigned long long)seed);
            return 1;
        }
        // Adopt the case's key material so everything below is checked against the FILE.
        ck.s = kase.sk; ck.evk0 = kase.evk0; ck.evk1 = kase.evk1;
        ck.s2.assign(N, 0);
        for (uint32_t i = 0; i < N; ++i) { if (!ck.s[i]) continue;
            for (uint32_t j = 0; j < N; ++j) { uint32_t k = i + j;
                if (k < N) ck.s2[k] += ck.s[i]*ck.s[j]; else ck.s2[k-N] -= ck.s[i]*ck.s[j]; } }
        bool same_primes = (kase.QP == ck.QP);
        auto got = ck.multiply(kase.A, kase.B);
        uint32_t bad = 0;
        for (uint32_t i = 0; i < LOUT(); ++i)
            for (uint32_t n = 0; n < N; ++n) {
                if (got.c0[i][n] != kase.R.c0[i][n]) ++bad;
                if (got.c1[i][n] != kase.R.c1[i][n]) ++bad; }
        std::printf("[4a] host vs FIXED case   primes %s   ciphertext mismatches = %u/%u  [%s]\n",
                    same_primes ? "match" : "DIFFER", bad, 2*LOUT()*N, (bad||!same_primes) ? "FAIL" : "PASS");
        pass &= (bad == 0 && same_primes);

        std::vector<cdouble> want(NSLOT);
        for (uint32_t j = 0; j < NSLOT; ++j) want[j] = kase.z1[j] * kase.z2[j];
        auto e = slot_err(ck.decrypt(kase.R), want);
        std::printf("[4b] decode(decrypt(case output)) == z1(.)z2   max = %.3e  rms = %.3e  ~%.1f bits  [%s]\n",
                    e.maxe, e.rms, e.bits(), e.maxe < 1e-2 ? "PASS" : "FAIL");
        pass &= (e.maxe < 1e-2);
    }

    // ---- device (opt-in) ----
    if (std::getenv("TG000_DEV")) pass &= device_stage(kase, ck);

    std::printf("\n%s\n", pass ? "Test Passed" : "Test FAILED");
    return pass ? 0 : 1;
}
