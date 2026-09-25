// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// exp_Tg018ntt — exp_Tg018m with the COMMUNICATION findings of docs/matmul/exp_ntt001.md applied
// to its NTT.
//
// exp_ntt001 took NTT-32768 from 121 to 34.0 us of critical-path busy (3.56x) with five rungs, and
// found that fixing communication moved it further than the choice of factorization did. This bench
// asks which of those rungs exp_Tg018m has NOT already had, and applies them. The audit:
//
//   rung (exp_ntt001)                          | exp_Tg018m already? | why
//   -------------------------------------------|---------------------|-------------------------------
//   deep matmul readers (NTT1_DEEP)            | YES                 | reader_fused_batch_pg.cpp puts
//                                              |                     | all 17 reads in flight already
//   fused dec+mm+rec triples (NTT1_FUSE)       | YES                 | fused_matmul_{a,c}.cpp in
//                                              |                     | kernels/experiments/exp_tb001 ARE the fused
//                                              |                     | 1pass kernels; that fusion was
//                                              |                     | exported FROM here
//   both-NoC corner turn (A2A=bothnoc)         | YES                 | a2a_corner_{rd,wr}.cpp
//   core-to-core corner turn (A2A=exchange)    | **NO**              | <-- what this bench adds
//   whole NTT in one kernel (NTT1_MEGA)        | no                  | not attempted; see the doc
//
// So there is exactly ONE untransferred rung, and it happens to be aimed at exp_Tg018m's largest
// non-arithmetic term: the verified 50.920 ms census puts a2a_corner at 9.103 ms, 17.9%, second only
// to the NTT matmuls (25.4%) and modmul (25.2%).
//
// WHAT THE EXCHANGE DOES. a2a_corner_{rd,wr} have every core read ALL 32 input tiles (128 KB) to take
// K=2 rows out of each; exp_ntt001 measured that shape as 64% read-bound. But those rows are already
// on-chip in other cores' L1, so the DRAM traffic is redundant. dataflow/a2a_corner_x.cpp has each
// core read only the K tiles it owns and push each row straight into the peer that needs it, in the
// receiver's final face layout, so the receiver assembles nothing either. See that kernel's header
// for the protocol and for the two things that differ from all_to_all_corner_x.cpp (exp_Tg018's
// page stride of 2, and K>1 tiles per core).
//
// The two arms are in ONE binary and differ only in this stage, so the comparison is controlled:
//
//   TG18NTT_A2A=bothnoc    (default) exp_Tg018m's corner turn, unchanged -- the control
//   TG18NTT_A2A=exchange   the core-to-core exchange
//
// Everything else -- scheme, parameters, kernels, the 0/6029312 verification -- is exp_Tg018m
// unchanged, so this is also directly comparable to docs/he/exp_tg018m.md.
//
//   ./build/exp_tg018ntt [seed]                        # host reference + ladder
//   TG18NTT_DEV=1 TG18NTT_CHAIN=1 ./build/exp_tg018ntt # the whole multiply on device
//
// Original exp_Tg018m header follows.
//
// exp_Tg018m — exp_Tg018 with the DRAM findings of docs/he/dram_guide.md applied.
//
// THE BUG THIS FIXES.  exp_Tg018 allocates every polynomial tile as its OWN single-page buffer
// (`mkbuf(1, I32)`, 64 of them per B64). tt-metal maps interleaved page p to DRAM bank
// p % NUM_DRAM_BANKS -- so a one-page buffer's only page is page 0, and page 0 is **always bank 0**.
// The entire working set of the multiply therefore lives in ONE of the card's eight DRAM channels.
//
// It shows up exactly where the guide predicts. The corner turn is 43.8% of the multiply (103.6 ms,
// 170.4 us per dispatch) and moves 4 MB per dispatch, which is **24.6 GB/s** -- against a measured
// 53.9 GB/s for a single saturated channel and 433 GB/s for the chip. It was never a corner-turn
// problem; it was a bank-allocation problem, and every other stage pays it too.
//
// WHAT CHANGES HERE
//   1. B64 becomes ONE buffer of NTILE pages instead of NTILE buffers of one page, so tile t lands on
//      bank t % 8 and a 64-tile read set is spread over all eight channels. Same for the per-tile
//      twiddle sets. This is the whole point of the experiment.
//   2. Diagonal core placement, because row-major packs the first 13 cores into one grid row and a
//      row carries 86.4 GB/s no matter how many cores are in it (dram_guide.md §3.2).
// Everything else -- the scheme, the parameters, the kernels' arithmetic, the verification -- is
// exp_Tg018 unchanged, so the two are directly comparable.
//
// Original header follows.
//
// exp_Tg018 — CKKS ct_mul at L_Q = 48, dnum = 3, N = 65536.  docs/he/exp_tg018.md
//
// The scaled-up sibling of exp_Tg000 (N=8192, L_Q=8, dnum=2). Same scheme, same kernel tree, same
// verification discipline — decrypt + decode against the slot-wise product — at a parameter set 8×
// the ring and 6× the limbs. What changes, and why each change is forced:
//
//   * N = 65536 because log2(QP) ~ 1779 bits needs a 2^16 ring at 128-bit security. That in turn
//     breaks the two-factor four-step: N2 = N/32 = 2048 makes stage A cost N2^2*N1 = 136M MACs,
//     130x N log N. The transform has to decompose further -- see the doc.
//   * 24-word (1536-bit) big integers. exp_Tg000's u512 cannot hold a 1337-bit Q, let alone the CRT
//     reconstruction that decryption needs.
//   * A THREADED canonical embedding. The O(N^2/2) direct sum is 2.15G terms per direction here
//     against 0.03G at N=8192; it is still the definition rather than a hand-rolled special FFT,
//     because this file is the correctness reference, but it is no longer free.
//   * dnum = 3, alpha = 16, P = 16. Each digit is ~2^448 and so is P, so the key-switch noise ratio
//     stays ~1 -- the same balance exp_Tg000 gets from dnum = 2.
//
// STATUS: milestone 1 (host scheme + fixture). The device path is the next milestone; see the doc.
//
//   ./build/exp_tg018 [seed]
//
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <functional>
#include <thread>
#include <vector>

#include <fmt/base.h>
#include <set>
#include <memory>
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

// ---- device-side geometry for the 3-level NTT-65536 (docs/he/exp_tg018.md) ----
//   n = n1 + 32*n2 + 1024*n3,  n3 = n3_lo + 32*n3_hi
//   tile index = n1*2 + n3_hi   (64 tiles),  in-tile = [n3_lo][n2]
constexpr uint32_t T = 32, NB = 4, ELEMS = 1024, NTILE = 64, NHALF = 32;
constexpr uint32_t I8 = ELEMS, I32 = ELEMS * 4;
const uint64_t POW[4] = {1, 128, 16384, 2097152};

template <typename E> std::vector<E> tilize32(const std::vector<E>& in) { std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = in[r * T + c]; return o; }
template <typename E> std::vector<E> untilize32(const std::vector<E>& in) { std::vector<E> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[r * T + c] = in[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)]; return o; }
int8_t limb(uint32_t v, uint32_t p) { return (int8_t)((v >> (7 * p)) & 127); }

namespace prof {
  double g_wr=0, g_rd=0, g_enq=0, g_fin=0, g_replay=0;
  long n_wr=0, n_rd=0, n_enq=0, n_fin=0, n_replay=0;
  using hrc = std::chrono::high_resolution_clock;
  inline double msd(hrc::time_point t){ return std::chrono::duration<double,std::milli>(hrc::now()-t).count(); }
  inline void reset(){ g_wr=g_rd=g_enq=g_fin=g_replay=0; n_wr=n_rd=n_enq=n_fin=n_replay=0; }
  std::function<void()> seg_hook;   // fires at every host boundary, to break Metal Trace segments
  std::function<void()> drain_fn;   // periodic profiler drain, so no markers are dropped
  long drain_every=0, drain_ctr=0;
}
namespace Dp = tt::tt_metal::distributed;
template<class...A> void t_wr (A&&...a){ if(prof::seg_hook) prof::seg_hook(); auto t=prof::hrc::now(); Dp::EnqueueWriteMeshBuffer(std::forward<A>(a)...); prof::g_wr+=prof::msd(t); ++prof::n_wr; }
template<class...A> void t_rd (A&&...a){ if(prof::seg_hook) prof::seg_hook(); auto t=prof::hrc::now(); Dp::EnqueueReadMeshBuffer (std::forward<A>(a)...); prof::g_rd+=prof::msd(t); ++prof::n_rd; }
template<class...A> void t_enq(A&&...a){ auto t=prof::hrc::now(); Dp::EnqueueMeshWorkload   (std::forward<A>(a)...); prof::g_enq+=prof::msd(t); ++prof::n_enq;
    if(prof::drain_every && (++prof::drain_ctr % prof::drain_every)==0 && prof::drain_fn) prof::drain_fn(); }
template<class Q>   void t_fin(Q&& cq) { if(prof::seg_hook) prof::seg_hook(); auto t=prof::hrc::now(); Dp::Finish(cq); prof::g_fin+=prof::msd(t); ++prof::n_fin; }

using u128 = unsigned __int128;
using cdouble = std::complex<double>;

// ---------------------------------------------------------------- CKKS parameters
constexpr uint32_t N     = 65536;
constexpr uint32_t NSLOT = N / 2;       // 32768 complex slots
constexpr uint32_t LQ    = 48;          // ciphertext base Q: 48 x 28-bit ~ 1337-bit
constexpr uint32_t LP    = 16;          // key-switch base P: 16 x 28-bit  (= alpha)
constexpr uint32_t LQP   = LQ + LP;     // 64
constexpr uint32_t DNUM  = 3;           // hybrid key-switch digits; alpha = LQ/DNUM = 16
constexpr uint32_t ALPHA = LQ / DNUM;   // 16

uint32_t g_ndrop = 2;                   // primes in the scale == primes the rescale drops
bool     g_centre = true;               // centred rescale remainder (the +qHalf shift identity)
inline uint32_t LOUT() { return LQ - g_ndrop; }
inline uint32_t DROPI(uint32_t k) { return LQ - 1 - k; }

uint32_t nthreads() {
    unsigned h = std::thread::hardware_concurrency();
    if (const char* e = std::getenv("TG18NTT_THREADS")) { int v = std::atoi(e); if (v > 0) h = (unsigned)v; }
    return h ? h : 1;
}
// parallel_for over [0, n): the O(N^2) embedding is the only thing here that needs it.
template <class F> void par_for(uint32_t n, F&& f) {
    const uint32_t NTH = std::min<uint32_t>(nthreads(), n);   // NOT T -- that is the tile width
    if (NTH <= 1) { for (uint32_t i = 0; i < n; ++i) f(i); return; }
    std::vector<std::thread> th;
    std::atomic<uint32_t> next{0};
    for (uint32_t t = 0; t < NTH; ++t)
        th.emplace_back([&]{ for (;;) { uint32_t i = next.fetch_add(1); if (i >= n) break; f(i); } });
    for (auto& x : th) x.join();
}

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

// ---------------------------------------------------------------- BIG: 1536-bit unsigned
// Q is 1337 bits and the CRT reconstruction accumulates to ~1350, so 24 x 64 = 1536 is the smallest
// safe width. Same minimal operation set as exp_Tg000's u512, widened.
constexpr int BW = 24;
struct BIG {
    uint64_t w[BW] = {0};
    BIG() = default;
    explicit BIG(uint64_t v) { w[0] = v; }
};
int cmp(const BIG& a, const BIG& b) {
    for (int i = BW - 1; i >= 0; --i) if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1;
    return 0;
}
BIG add(const BIG& a, const BIG& b) {
    BIG r; u128 c = 0;
    for (int i = 0; i < BW; ++i) { c += (u128)a.w[i] + b.w[i]; r.w[i] = (uint64_t)c; c >>= 64; }
    return r;
}
BIG sub(const BIG& a, const BIG& b) {
    BIG r; __int128 br = 0;
    for (int i = 0; i < BW; ++i) {
        __int128 d = (__int128)a.w[i] - b.w[i] - br;
        if (d < 0) { d += ((__int128)1 << 64); br = 1; } else br = 0;
        r.w[i] = (uint64_t)d;
    }
    return r;
}
BIG mul_small(const BIG& a, uint64_t m) {
    BIG r; u128 c = 0;
    for (int i = 0; i < BW; ++i) { c += (u128)a.w[i] * m; r.w[i] = (uint64_t)c; c >>= 64; }
    return r;
}
BIG shl1(const BIG& a) {
    BIG r; uint64_t carry = 0;
    for (int i = 0; i < BW; ++i) { r.w[i] = (a.w[i] << 1) | carry; carry = a.w[i] >> 63; }
    return r;
}
int bitlen(const BIG& a) {
    for (int i = BW - 1; i >= 0; --i) if (a.w[i]) { int b = 63; while (b >= 0 && !((a.w[i] >> b) & 1)) --b; return i * 64 + b + 1; }
    return 0;
}
void divmod(const BIG& a, const BIG& d, BIG& q, BIG& rem) {
    q = BIG(); rem = BIG();
    for (int i = bitlen(a) - 1; i >= 0; --i) {
        rem = shl1(rem);
        rem.w[0] |= (a.w[i / 64] >> (i % 64)) & 1ull;
        if (cmp(rem, d) >= 0) { rem = sub(rem, d); q.w[i / 64] |= 1ull << (i % 64); }
    }
}
uint64_t mod_small(const BIG& a, uint64_t d) {
    u128 r = 0;
    for (int i = BW - 1; i >= 0; --i) r = ((r << 64) | a.w[i]) % d;
    return (uint64_t)r;
}
double to_double(const BIG& a) {
    double r = 0;
    for (int i = BW - 1; i >= 0; --i) r = r * 18446744073709551616.0 + (double)a.w[i];
    return r;
}

// ---------------------------------------------------------------- negacyclic NTT, per prime
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
        w.resize(N); iw.resize(N); w[0] = iw[0] = 1;
        for (uint32_t i = 1; i < N; ++i) { w[i] = (u128)w[i - 1] * wn % q; iw[i] = (u128)iw[i - 1] * wni % q; }
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
        std::vector<uint64_t> a(E.begin(), E.end());
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

// ---------------------------------------------------------------- the canonical embedding
// Unchanged in form from exp_Tg000 -- written from the definition, not as a special FFT -- but
// threaded, because 2.15G terms per direction is no longer a rounding error.
struct Embed {
    std::vector<double> cr, ci;
    std::vector<uint32_t> rot;
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
    // slots -> real coefficients.  a_k = (2/N) sum_j Re( z_j * zeta^(-5^j k) )
    // Parallelised over k (not j) so each thread owns disjoint outputs and no reduction is needed.
    std::vector<double> inverse(const std::vector<cdouble>& z) const {
        std::vector<double> a(N, 0.0);
        const double f = 2.0 / (double)N;
        const uint32_t CH = 256;                       // chunk of k per task
        par_for((N + CH - 1) / CH, [&](uint32_t c) {
            for (uint32_t k = c * CH; k < std::min(N, (c + 1) * CH); ++k) {
                double acc = 0;
                for (uint32_t j = 0; j < NSLOT; ++j) {
                    uint32_t e = (uint32_t)((2 * N - (uint32_t)(((uint64_t)rot[j] * k) & (2 * N - 1))) & (2 * N - 1));
                    acc += z[j].real() * cr[e] - z[j].imag() * ci[e];
                }
                a[k] = acc * f;
            }
        });
        return a;
    }
    // real coefficients -> slots
    std::vector<cdouble> forward(const std::vector<double>& a) const {
        std::vector<cdouble> z(NSLOT);
        const uint32_t CH = 256;
        par_for((NSLOT + CH - 1) / CH, [&](uint32_t c) {
            for (uint32_t j = c * CH; j < std::min(NSLOT, (c + 1) * CH); ++j) {
                double re = 0, im = 0;
                for (uint32_t k = 0; k < N; ++k) {
                    uint32_t e = (uint32_t)(((uint64_t)rot[j] * k) & (2 * N - 1));
                    re += a[k] * cr[e]; im += a[k] * ci[e];
                }
                z[j] = cdouble(re, im);
            }
        });
        return z;
    }
};

// ---------------------------------------------------------------- an RNS base with exact CRT
struct CRTBase {
    std::vector<uint64_t> p;
    BIG prod, half;
    std::vector<BIG> qhat;
    std::vector<uint64_t> qhat_inv;
    void build(const std::vector<uint64_t>& primes) {
        p = primes;
        prod = BIG(1);
        for (auto q : p) prod = mul_small(prod, q);
        { BIG q2, r2; divmod(prod, BIG(2), q2, r2); half = q2; }
        qhat.resize(p.size()); qhat_inv.resize(p.size());
        for (size_t i = 0; i < p.size(); ++i) {
            BIG qh, r; divmod(prod, BIG(p[i]), qh, r);
            qhat[i] = qh;
            qhat_inv[i] = modinv(mod_small(qh, p[i]), p[i]);
        }
    }
    BIG reconstruct(const std::vector<uint32_t>& res) const {
        BIG acc;
        for (size_t i = 0; i < p.size(); ++i) {
            uint64_t c = (uint64_t)((u128)res[i] * qhat_inv[i] % p[i]);
            acc = add(acc, mul_small(qhat[i], c));
        }
        while (cmp(acc, prod) >= 0) acc = sub(acc, prod);
        return acc;
    }
    double reconstruct_signed(const std::vector<uint32_t>& res) const {
        BIG v = reconstruct(res);
        bool neg = cmp(v, half) > 0;
        return neg ? -to_double(sub(prod, v)) : to_double(v);
    }
};

// ---------------------------------------------------------------- the scheme
struct Ciphertext {
    std::vector<std::vector<uint32_t>> c0, c1;
    long double scale = 0;
    uint32_t L() const { return (uint32_t)c0.size(); }
};

struct CKKS {
    std::vector<uint64_t> QP;
    std::vector<NTTCtx> nt;
    CRTBase QbIn, QbOut;
    BIG Qprod, Pprod;
    long double Delta = 0;
    uint64_t DeltaI = 0;
    Embed emb;

    std::vector<int64_t> s, s2;
    std::mt19937_64 rng;
    std::vector<std::vector<uint32_t>> group, comp;
    std::vector<std::vector<std::vector<uint32_t>>> evk0, evk1;

    explicit CKKS(uint64_t seed) : rng(seed) {
        QP = ntt_primes(LQP);
        if (QP.size() < LQP) { std::printf("only %zu NTT primes available, need %u\n", QP.size(), LQP); std::abort(); }
        std::printf("  building %u NTT contexts at N=%u ...\n", LQP, N); std::fflush(stdout);
        nt.reserve(LQP);
        for (auto q : QP) nt.emplace_back(q);
        QbIn.build(std::vector<uint64_t>(QP.begin(), QP.begin() + LQ));
        QbOut.build(std::vector<uint64_t>(QP.begin(), QP.begin() + LOUT()));
        Qprod = QbIn.prod;
        Pprod = BIG(1); for (uint32_t j = 0; j < LP; ++j) Pprod = mul_small(Pprod, QP[LQ + j]);
        DeltaI = 1; for (uint32_t k = 0; k < g_ndrop; ++k) DeltaI *= QP[DROPI(k)];
        Delta = (long double)DeltaI;

        s.resize(N);
        for (auto& x : s) x = (int64_t)(rng() % 3) - 1;
        // s^2 in Z[x]/(x^N+1), exact. O(N^2) with a sparse outer loop over the ternary s.
        s2.assign(N, 0);
        std::vector<uint32_t> nz;
        for (uint32_t i = 0; i < N; ++i) if (s[i]) nz.push_back(i);
        for (uint32_t i : nz)
            for (uint32_t j = 0; j < N; ++j) {
                if (!s[j]) continue;
                uint32_t k = i + j;
                if (k < N) s2[k] += s[i] * s[j]; else s2[k - N] -= s[i] * s[j];
            }
        // dnum=3: three 16-limb digits; comp[j] = every QP prime NOT in digit j
        group.assign(DNUM, {});
        for (uint32_t j = 0; j < DNUM; ++j)
            for (uint32_t i = 0; i < ALPHA; ++i) group[j].push_back(j * ALPHA + i);
        comp.assign(DNUM, {});
        for (uint32_t j = 0; j < DNUM; ++j)
            for (uint32_t li = 0; li < LQP; ++li) {
                bool own = false;
                for (auto g : group[j]) if (g == li) own = true;
                if (!own) comp[j].push_back(li);
            }
        keygen_evk();
    }

    std::vector<uint32_t> limb_of(const std::vector<int64_t>& poly, uint64_t m) const {
        std::vector<uint32_t> r(N);
        for (uint32_t n = 0; n < N; ++n) r[n] = (uint32_t)(((poly[n] % (int64_t)m) + (int64_t)m) % (int64_t)m);
        return r;
    }

    // evk_j = ( -a_j*s + e_j + P*w_j*s^2 , a_j ) over QP, with w_j the CRT IDEMPOTENT of digit j
    void keygen_evk() {
        std::printf("  keygen: %u digits x %u limbs ...\n", DNUM, LQP); std::fflush(stdout);
        std::vector<BIG> w(DNUM);
        for (uint32_t j = 0; j < DNUM; ++j) {
            std::vector<uint32_t> res(LQ, 0);
            for (auto g : group[j]) res[g] = 1;
            w[j] = QbIn.reconstruct(res);
        }
        evk0.assign(DNUM, std::vector<std::vector<uint32_t>>(LQP));
        evk1.assign(DNUM, std::vector<std::vector<uint32_t>>(LQP));
        for (uint32_t j = 0; j < DNUM; ++j) {
            std::vector<int64_t> e_int(N);
            for (auto& x : e_int) x = (int64_t)(rng() % 5) - 2;
            std::vector<std::vector<uint32_t>> a(LQP, std::vector<uint32_t>(N));
            for (uint32_t li = 0; li < LQP; ++li)
                for (uint32_t n = 0; n < N; ++n) a[li][n] = (uint32_t)(rng() % QP[li]);
            par_for(LQP, [&](uint32_t li) {
                uint64_t m = QP[li];
                uint64_t Pw = (uint64_t)((u128)mod_small(Pprod, m) * mod_small(w[j], m) % m);
                auto sl = limb_of(s, m), s2l = limb_of(s2, m);
                auto as_ = nt[li].negmul(a[li], sl);
                evk0[j][li].resize(N); evk1[j][li] = a[li];
                for (uint32_t n = 0; n < N; ++n) {
                    uint32_t el = (uint32_t)(((e_int[n] % (int64_t)m) + (int64_t)m) % (int64_t)m);
                    evk0[j][li][n] = (uint32_t)(((u128)m + el + (u128)Pw * s2l[n] % m - as_[n]) % m);
                }
            });
        }
    }

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

    Ciphertext encrypt(const std::vector<int64_t>& m, long double scale) {
        Ciphertext ct; ct.scale = scale;
        ct.c0.assign(LQ, std::vector<uint32_t>(N));
        ct.c1.assign(LQ, std::vector<uint32_t>(N));
        std::vector<int64_t> e_int(N);
        for (auto& x : e_int) x = (int64_t)(rng() % 5) - 2;
        std::vector<std::vector<uint32_t>> aQ(LQ, std::vector<uint32_t>(N));
        for (uint32_t i = 0; i < LQ; ++i)
            for (uint32_t n = 0; n < N; ++n) aQ[i][n] = (uint32_t)(rng() % QP[i]);
        par_for(LQ, [&](uint32_t i) {
            uint64_t mq = QP[i];
            auto sl = limb_of(s, mq);
            auto as_ = nt[i].negmul(aQ[i], sl);
            for (uint32_t n = 0; n < N; ++n) {
                uint32_t el = (uint32_t)(((e_int[n] % (int64_t)mq) + (int64_t)mq) % (int64_t)mq);
                uint32_t ml = (uint32_t)(((m[n] % (int64_t)mq) + (int64_t)mq) % (int64_t)mq);
                ct.c0[i][n] = (uint32_t)(((u128)mq + ml + el - as_[n]) % mq);
                ct.c1[i][n] = aQ[i][n];
            }
        });
        return ct;
    }

    std::vector<double> phase(const Ciphertext& ct) const {
        const uint32_t L = ct.L();
        const CRTBase& B = (L == LQ) ? QbIn : QbOut;
        std::vector<std::vector<uint32_t>> v(L, std::vector<uint32_t>(N));
        par_for(L, [&](uint32_t i) {
            uint64_t mq = QP[i];
            auto sl = limb_of(s, mq);
            auto cs = nt[i].negmul(ct.c1[i], sl);
            for (uint32_t n = 0; n < N; ++n) v[i][n] = (uint32_t)(((u128)ct.c0[i][n] + cs[n]) % mq);
        });
        std::vector<double> out(N);
        const uint32_t CH = 1024;
        par_for((N + CH - 1) / CH, [&](uint32_t c) {
            std::vector<uint32_t> res(L);
            for (uint32_t n = c * CH; n < std::min(N, (c + 1) * CH); ++n) {
                for (uint32_t i = 0; i < L; ++i) res[i] = v[i][n];
                out[n] = B.reconstruct_signed(res);
            }
        });
        return out;
    }
    std::vector<cdouble> decrypt(const Ciphertext& ct) const { return decode(phase(ct), ct.scale); }

    // approximate base conversion, exactly as the device kernel computes it
    std::vector<std::vector<uint32_t>> fbc(const std::vector<std::vector<uint32_t>>& x,
                                           const std::vector<uint32_t>& sI,
                                           const std::vector<uint32_t>& dI) const {
        const uint32_t Ls = (uint32_t)sI.size(), Ld = (uint32_t)dI.size();
        // Ls = 16 primes of 28 bits = 448 bits: the source product no longer fits u128, so Shat_i
        // and its inverse go through BIG. exp_Tg000 could stay in u128 because Ls was 4.
        BIG S(1); for (uint32_t i = 0; i < Ls; ++i) S = mul_small(S, QP[sI[i]]);
        std::vector<uint64_t> c1(Ls);
        std::vector<std::vector<uint64_t>> c2(Ls, std::vector<uint64_t>(Ld));
        for (uint32_t i = 0; i < Ls; ++i) {
            uint64_t si = QP[sI[i]];
            BIG sh, r; divmod(S, BIG(si), sh, r);
            c1[i] = modinv(mod_small(sh, si), si);
            for (uint32_t j = 0; j < Ld; ++j) c2[i][j] = mod_small(sh, QP[dI[j]]);
        }
        std::vector<std::vector<uint32_t>> out(Ld, std::vector<uint32_t>(N));
        const uint32_t CH = 1024;
        par_for((N + CH - 1) / CH, [&](uint32_t ch) {
            std::vector<uint64_t> y(Ls);
            for (uint32_t n = ch * CH; n < std::min(N, (ch + 1) * CH); ++n) {
                for (uint32_t i = 0; i < Ls; ++i) y[i] = (uint64_t)((u128)x[i][n] * c1[i] % QP[sI[i]]);
                for (uint32_t j = 0; j < Ld; ++j) {
                    u128 acc = 0;
                    // 16 terms of (28-bit x 28-bit) = 56 bits each; 16 of them is 60 bits, so the
                    // u128 accumulator is still exact without a per-term reduction.
                    for (uint32_t i = 0; i < Ls; ++i) acc += (u128)y[i] * c2[i][j];
                    out[j][n] = (uint32_t)(acc % QP[dI[j]]);
                }
            }
        });
        return out;
    }

    // the drop-prime rescale, g_ndrop steps, centred by the +qHalf shift
    std::vector<std::vector<uint32_t>> rescale(const std::vector<std::vector<uint32_t>>& in) const {
        std::vector<std::vector<uint32_t>> c = in;
        for (uint32_t k = 0; k < g_ndrop; ++k) {
            const uint32_t d = DROPI(k);
            const uint64_t qL = QP[d], qHalf = qL / 2;
            std::vector<std::vector<uint32_t>> out(d, std::vector<uint32_t>(N));
            par_for(d, [&](uint32_t i) {
                const uint64_t m = QP[i], qLinv = modinv(qL % m, m), h = qHalf % m;
                for (uint32_t n = 0; n < N; ++n) {
                    const uint64_t ci = g_centre ? ((uint64_t)c[i][n] + h) % m : c[i][n];
                    const uint64_t xd = g_centre ? ((uint64_t)c[d][n] + qHalf) % qL : c[d][n];
                    out[i][n] = (uint32_t)(((u128)ci + m - xd % m) % m * qLinv % m);
                }
            });
            c = std::move(out);
        }
        return c;
    }

    void tensor(const Ciphertext& A, const Ciphertext& B,
                std::vector<std::vector<uint32_t>>& e0,
                std::vector<std::vector<uint32_t>>& e1,
                std::vector<std::vector<uint32_t>>& e2) const {
        e0.assign(LQ, {}); e1.assign(LQ, {}); e2.assign(LQ, {});
        par_for(LQ, [&](uint32_t i) {
            uint64_t m = QP[i];
            e0[i] = nt[i].negmul(A.c0[i], B.c0[i]);
            e2[i] = nt[i].negmul(A.c1[i], B.c1[i]);
            auto x = nt[i].negmul(A.c0[i], B.c1[i]);
            auto y = nt[i].negmul(A.c1[i], B.c0[i]);
            e1[i].resize(N);
            for (uint32_t n = 0; n < N; ++n) e1[i][n] = (uint32_t)(((u128)x[n] + y[n]) % m);
        });
    }

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
            par_for(LQP, [&](uint32_t li) {
                uint64_t m = QP[li];
                auto p0 = nt[li].negmul(vj[li], evk0[j][li]);
                auto p1 = nt[li].negmul(vj[li], evk1[j][li]);
                for (uint32_t n = 0; n < N; ++n) {
                    acc0[li][n] = (uint32_t)(((u128)acc0[li][n] + p0[n]) % m);
                    acc1[li][n] = (uint32_t)(((u128)acc1[li][n] + p1[n]) % m);
                }
            });
        }
        std::vector<uint32_t> qidx(LQ), pidx(LP);
        for (uint32_t i = 0; i < LQ; ++i) qidx[i] = i;
        for (uint32_t j = 0; j < LP; ++j) pidx[j] = LQ + j;
        auto moddown = [&](std::vector<std::vector<uint32_t>>& acc) {
            std::vector<std::vector<uint32_t>> aP;
            for (auto pp : pidx) aP.push_back(acc[pp]);
            auto f = fbc(aP, pidx, qidx);
            std::vector<std::vector<uint32_t>> out(LQ, std::vector<uint32_t>(N));
            par_for(LQ, [&](uint32_t i) {
                uint64_t m = QP[i], Pinv = modinv(mod_small(Pprod, m), m);
                for (uint32_t n = 0; n < N; ++n)
                    out[i][n] = (uint32_t)((((u128)acc[i][n] + m - f[i][n]) % m) * Pinv % m);
            });
            return out; };
        k0 = moddown(acc0); k1 = moddown(acc1);
    }

    Ciphertext multiply(const Ciphertext& A, const Ciphertext& B) const {
        std::vector<std::vector<uint32_t>> e0, e1, e2, k0, k1;
        tensor(A, B, e0, e1, e2);
        relin(e2, k0, k1);
        std::vector<std::vector<uint32_t>> r0(LQ, std::vector<uint32_t>(N)), r1(LQ, std::vector<uint32_t>(N));
        par_for(LQ, [&](uint32_t i) {
            uint64_t m = QP[i];
            for (uint32_t n = 0; n < N; ++n) {
                r0[i][n] = (uint32_t)(((u128)e0[i][n] + k0[i][n]) % m);
                r1[i][n] = (uint32_t)(((u128)e1[i][n] + k1[i][n]) % m);
            }
        });
        Ciphertext o;
        o.c0 = rescale(r0);
        o.c1 = rescale(r1);
        o.scale = A.scale * B.scale / Delta;
        return o;
    }

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


// ======================================================================================
// THE DEVICE 3-LEVEL NTT-65536.  docs/he/exp_tg018.md; validated on the host first in
// host/experiments/proto_ntt65536_3level.cpp (0/65536 both directions).
//
//   tile index = n1*2 + n3_hi  (64 tiles),  in-tile = [n3_lo][n2]
//   P1 contract n3 (blocked over n3_hi, W64 as 2x2)   fused_matmul_a_acc, ACC=2
//   Ta  x w_2048^{n2 kv}                              modmul_sfpi_full
//   P2 contract n2 (in-tile cols, W32, RHS)           fused_matmul_c
//   Tb  x w_N^{n1 kb} psi^{n1}                        modmul_sfpi_full
//   A2A corner turn n1 <-> kv_lo, per kv_hi group     a2a_corner_bufs
//   P3 contract n1 (in-tile rows, W32, LHS)           fused_matmul_a
// ======================================================================================
struct DevNTT {
    // folded constant sets for one (prime, direction)
    struct CS {
        std::vector<std::vector<int8_t>> w64{std::vector<std::vector<int8_t>>(4)};  // [kv_hi*2 + n3_hi]
        std::vector<int8_t> w32a, w32b;                                             // P2 (RHS), P3 (LHS)
        std::vector<std::vector<uint32_t>> ta{std::vector<std::vector<uint32_t>>(2)};
        std::vector<std::vector<uint32_t>> tb{std::vector<std::vector<uint32_t>>(NTILE)};
    };

    // LHS fold (W multiplies from the left):  l[i] = limb((POW[r]*W[i]) mod q, p) -> slot p*NB+r
    static std::vector<int8_t> foldL(const std::vector<uint32_t>& W, uint64_t q) {
        std::vector<int8_t> f(NB * NB * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
            std::vector<int8_t> l(ELEMS);
            for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[r] * W[i]) % q), p);
            auto tl = tilize32(l); std::copy(tl.begin(), tl.end(), f.begin() + (p * NB + r) * ELEMS);
        }
        return f; }
    // RHS fold (W multiplies from the right): l[i] = limb((POW[p]*W[i]) mod q, r)
    static std::vector<int8_t> foldR(const std::vector<uint32_t>& W, uint64_t q) {
        std::vector<int8_t> f(NB * NB * ELEMS);
        for (uint32_t p = 0; p < NB; ++p) for (uint32_t r = 0; r < NB; ++r) {
            std::vector<int8_t> l(ELEMS);
            for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[p] * W[i]) % q), r);
            auto tl = tilize32(l); std::copy(tl.begin(), tl.end(), f.begin() + (p * NB + r) * ELEMS);
        }
        return f; }

    static CS gen(uint64_t q, const mont::MontgomeryContext& C, bool fwd) {
        CS cs;
        const uint64_t g = prim_root(q);
        const uint64_t psi_f = modpow(g, (q - 1) / (2 * N), q), w_f = (u128)psi_f * psi_f % q;
        const uint64_t psi = fwd ? psi_f : modinv(psi_f, q);
        const uint64_t w   = fwd ? w_f   : modinv(w_f, q);
        const uint64_t w64 = modpow(w, N / 64, q), w32 = modpow(w, N / 32, q), w2048 = modpow(w, N / 2048, q);
        const uint64_t Ninv = fwd ? 1 : modinv(N, q);

        // P1: the 64-point matrix as 2x2 blocks.  Forward contracts n3 and outputs kv; the INVERSE
        // contracts kv and outputs n3, so the matrix TRANSPOSES -- and that matters because the
        // psi^{1024 n3} factor rides on n3, which is the column one way and the row the other. Getting
        // this wrong is invisible in the forward (where it happens to sit on the column both times)
        // and fails totally in the inverse.
        for (uint32_t bo = 0; bo < 2; ++bo) for (uint32_t bi = 0; bi < 2; ++bi) {
            std::vector<uint32_t> blk(ELEMS);
            for (uint32_t ro = 0; ro < T; ++ro) for (uint32_t ci = 0; ci < T; ++ci) {
                uint32_t kv, n3;
                if (fwd) { kv = ro + 32 * bo; n3 = ci + 32 * bi; }   // out = kv, in = n3
                else     { n3 = ro + 32 * bo; kv = ci + 32 * bi; }   // out = n3, in = kv
                u128 c = (u128)modpow(w64, (uint64_t)kv * n3 % 64, q) * modpow(psi, 1024ull * n3 % (2 * N), q) % q;
                if (!fwd) c = c * Ninv % q;                          // N^-1 folded into the last pass
                blk[ro * T + ci] = (uint32_t)c;
            }
            cs.w64[bo * 2 + bi] = foldL(blk, q);
        }
        // P2: same transposition. Forward W[n2][ku]; inverse W[ku][n2], with psi^{32 n2} on n2.
        { std::vector<uint32_t> W(ELEMS);
          for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c) {
              uint32_t n2 = fwd ? r : c, ku = fwd ? c : r;
              W[r * T + c] = (uint32_t)((u128)modpow(w32, (uint64_t)n2 * ku % 32, q) * modpow(psi, 32ull * n2 % (2 * N), q) % q); }
          cs.w32a = foldR(W, q); }
        // P3: W32b[ka][n1] = w32^{n1 ka} -- symmetric in (ka,n1) and carries no psi, so it needs no
        // transposition; the inverse differs only in w32 being the inverse root.
        { std::vector<uint32_t> W(ELEMS);
          for (uint32_t ka = 0; ka < T; ++ka) for (uint32_t n1 = 0; n1 < T; ++n1)
              W[ka * T + n1] = (uint32_t)modpow(w32, (uint64_t)n1 * ka % 32, q);
          cs.w32b = foldL(W, q); }
        // Ta[kv_hi][kv_lo][n2] = w2048^{n2 kv}  (Montgomery: it feeds modmul_sfpi_full)
        for (uint32_t kh = 0; kh < 2; ++kh) {
            std::vector<uint32_t> t(ELEMS);
            for (uint32_t kl = 0; kl < T; ++kl) for (uint32_t n2 = 0; n2 < T; ++n2)
                t[kl * T + n2] = C.to_montgomery((uint32_t)modpow(w2048, (uint64_t)n2 * (kl + 32 * kh) % 2048, q));
            cs.ta[kh] = tilize32(t); }
        // Tb[t=(n1,kv_hi)][kv_lo][ku] = w_N^{n1 kb} psi^{n1},  kb = kv_lo + 32 kv_hi + 64 ku
        for (uint32_t n1 = 0; n1 < 32; ++n1) for (uint32_t kh = 0; kh < 2; ++kh) {
            std::vector<uint32_t> t(ELEMS);
            for (uint32_t kl = 0; kl < T; ++kl) for (uint32_t ku = 0; ku < T; ++ku) {
                uint32_t kb = (kl + 32 * kh) + 64 * ku;
                u128 c = (u128)modpow(w, (uint64_t)n1 * kb % N, q) * modpow(psi, n1, q) % q;
                t[kl * T + ku] = C.to_montgomery((uint32_t)c);
            }
            cs.tb[n1 * 2 + kh] = tilize32(t); }
        return cs; }
};

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

double now_s() {
    using c = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(c::now().time_since_epoch()).count();
}

// ======================================================================================
// THE DEVICE STAGE — the 3-level transform above, plus the CKKS graph at 64 limbs.
// Shape-identical to exp_Tg000's graph; what differs is 64 primes instead of 12, dnum=3
// instead of 2, and a transform that is six passes instead of three.
// ======================================================================================
bool device_stage(CKKS& ck) {
    auto mesh = distributed::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE);
    auto& cq = mesh->mesh_command_queue();
    auto dr = distributed::MeshCoordinateRange(mesh->shape());
    auto grid = mesh->compute_with_storage_grid_size();
    const uint32_t GX = grid.x, GY = grid.y;
    // G -- the core budget, and the only knob for it. exp_Tg018 inherited an env var called
    // NTT_CORES from host/experiments/legacy/nntt8192_mc.cpp by way of he_mul_cross and exp_Tb001; it never meant
    // "cores for the NTT", it was always just G. The workload here is described by G, B and S and
    // nothing else, so the name is gone. Default is the whole grid.
    uint32_t want = GX * GY;
    if (const char* e = std::getenv("TG18NTT_G")) { int v = std::atoi(e); if (v > 0) want = (uint32_t)v; }
    const uint32_t G = std::min<uint32_t>(want, GX * GY);
    const uint32_t NC = G;          // NC is G; the alias keeps the dispatch loops below readable
    // PLACEMENT: row-major, deliberately -- and this is the one guide rule that does NOT transfer.
    //
    // dram_guide.md §3.2 measures 2.42x from spreading cores over grid rows, because a row carries
    // only 86.4 GB/s however many cores sit in it. That was measured on a kernel with 32-64 tiles per
    // core, deep in the bandwidth-bound regime. exp_Tg018 dispatches ONE tile per core, which puts it
    // entirely inside the transient M(G) -- latency-bound, not bandwidth-bound. Spreading cores then
    // buys nothing and costs NoC distance on every transfer.
    //
    // Measured here, whole multiply at G=64:  row-major 133.6 ms, diagonal 333.5 ms -- diagonal is
    // 2.5x WORSE. Spread for bandwidth, pack for latency.
    const bool DIAG = std::getenv("TG18NTT_DIAG") != nullptr;
    auto CC = [&](uint32_t i){ return DIAG ? CoreCoord{i % GX, i % GY}
                                           : CoreCoord{i % GX, i / GX}; };
    auto crs_of = [&](uint32_t c){ std::set<CoreRange> ss; for (uint32_t i=0;i<c;++i) ss.insert(CoreRange(CC(i))); return CoreRangeSet(ss); };
    auto mkbuf = [&](uint32_t nt, uint32_t page) {
        distributed::DeviceLocalBufferConfig d{.page_size = page, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * page}, d, mesh.get()); };
    // NTT CONSTANTS LIVE IN L1, NOT DRAM. The folded W matrices are a pure function of the prime and
    // the ring -- they never change, and every HE operation re-reads them. Today each 16 KB set is
    // read from DRAM ~304 times per multiply (64 cores x ~4.75 dispatches) and again on every op
    // after that. All 128 sets are 12.0 MB, which as an L1-interleaved buffer is 95 KB per core --
    // 6.6% of L1. The 16 tiles of a set land on 16 different L1 banks, so 64 readers spread over 16
    // sources instead of piling on one. No kernel change: TensorAccessor takes the buffer type from
    // its compile args, so the readers follow the allocation.
    const bool W_DRAM = std::getenv("TG18NTT_W_DRAM") != nullptr;   // A/B switch
    auto mkbufW = [&](uint32_t nt, uint32_t page) {
        distributed::DeviceLocalBufferConfig d{.page_size = page,
            .buffer_type = W_DRAM ? BufferType::DRAM : BufferType::L1};
        return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * page}, d, mesh.get()); };
    auto mkcb = [&](Program& p, const CoreRangeSet& crs, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t page) {
        CircularBufferConfig c(depth * page, {{idx, fmt}}); c.set_page_size(idx, page); CreateCircularBuffer(p, crs, c); };
    // ---- measurement stack, same shape as exp_Tb001 / exp_Tg000 so the metrics mean the same ----
    std::vector<distributed::MeshWorkload> hold; hold.reserve(8192);
    bool CACHE_ON=false, CACHE_REC=false; std::vector<distributed::MeshWorkload> WC; size_t WI=0;
    int TR_PHASE=0; std::vector<distributed::MeshTraceId> SEG; bool seg_open=false, seg_started=false;
    size_t seg_idx=0; bool MEASURE_DEV=false;
    std::string KLBL="?", HOP="-", HPH="-", OPD="-";
    auto L=[&](const char* op, const char* ph, std::string opd){ HOP=op; HPH=ph; OPD=std::move(opd); };
    struct DL { std::string k, op, ph, opd; };
    std::vector<DL> DLBL;
    const char* LBLPATH = std::getenv("TG18NTT_LABELS");
    auto dump_labels=[&](){ if(!LBLPATH) return; std::FILE* f=std::fopen(LBLPATH,"w"); if(!f) return;
        std::fprintf(f,"seq,host_kernel,he_op,he_phase,operand\n");
        for(size_t i=0;i<DLBL.size();++i)
            std::fprintf(f,"%zu,%s,%s,%s,%s\n",i,DLBL[i].k.c_str(),DLBL[i].op.c_str(),DLBL[i].ph.c_str(),DLBL[i].opd.c_str());
        std::fclose(f);
        fmt::print("wrote {} : {} dispatch labels\n", LBLPATH, DLBL.size()); };
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
    auto seg_break=[&](){ if(!CACHE_ON) return;
        if(TR_PHASE==1){ if(seg_open){ mesh->end_mesh_trace(0, SEG.back()); seg_open=false; } }
        else if(TR_PHASE==2){ seg_started=false; } };
    prof::seg_hook = seg_break;
    fmt::print("exp_Tg018m device  grid {}x{} -> G={} of {}   {} primes, dnum={}, alpha={}\n",
               GX, GY, G, GX * GY, LQP, DNUM, ALPHA);

    using Buf = std::shared_ptr<distributed::MeshBuffer>;
    // ONE buffer of NTILE pages, not NTILE buffers of one page. Page t -> DRAM bank t % 8, so a
    // 64-tile working set is spread over all eight channels instead of piled on bank 0.
    using B64 = Buf;
    // Buffer POOL: with the cache on, hand out fixed buffers in deterministic order (PI reset per
    // multiply) so the addresses baked into cached workloads stay valid.
    std::vector<B64> POOL; size_t PI = 0;
    auto mkB64 = [&]() -> B64 {
        if(!CACHE_ON) return mkbuf(NTILE, I32);
        if(PI>=POOL.size()) POOL.push_back(mkbuf(NTILE, I32));
        return POOL[PI++]; };
    auto A = [](const B64& b){ return (uint32_t)b->address(); };   // base address; page == tile index


    std::vector<mont::MontgomeryContext> ctx;
    for (auto q : ck.QP) ctx.emplace_back((uint32_t)q);

    // ---- resident constants: every (prime, direction) set, generated once ----
    struct RC { std::vector<Buf> w64{std::vector<Buf>(4)}; Buf w32a, w32b; Buf ta; Buf tb; };
    std::map<int, RC> KC;
    { double t0 = now_s();
      std::vector<DevNTT::CS> cs(2 * LQP);
      par_for(2 * LQP, [&](uint32_t k){ cs[k] = DevNTT::gen(ck.QP[k >> 1], ctx[k >> 1], (k & 1) != 0); });
      for (uint32_t li = 0; li < LQP; ++li) for (uint32_t d = 0; d < 2; ++d) {
          const auto& c = cs[li * 2 + d]; RC r;
          for (uint32_t i = 0; i < 4; ++i) { r.w64[i] = mkbufW(NB*NB, I8); t_wr(cq, r.w64[i], c.w64[i], false); }
          r.w32a = mkbufW(NB*NB, I8); t_wr(cq, r.w32a, c.w32a, false);
          r.w32b = mkbufW(NB*NB, I8); t_wr(cq, r.w32b, c.w32b, false);
          { r.ta = mkbuf(2, I32); std::vector<uint32_t> two(2 * ELEMS);
            for (uint32_t i = 0; i < 2; ++i)
                std::copy(c.ta[i].begin(), c.ta[i].end(), two.begin() + (size_t)i * ELEMS);
            t_wr(cq, r.ta, two, false); }
          { r.tb = mkbuf(NTILE, I32); std::vector<uint32_t> all(NTILE * ELEMS);
            for (uint32_t i = 0; i < NTILE; ++i)
                std::copy(c.tb[i].begin(), c.tb[i].end(), all.begin() + (size_t)i * ELEMS);
            t_wr(cq, r.tb, all, false); }
          KC[li * 2 + d] = std::move(r);   // key = li*2 + fwd; cs[li*2+d] was generated with fwd = (d!=0)
      }
      t_fin(cq);
      fmt::print("  transform constants resident: {} sets, {:.1f} s\n", 2 * LQP, now_s() - t0); }

    // ---- layout helpers ----
    auto up_C = [&](const std::vector<uint32_t>& x){ B64 b = mkB64();
        std::vector<uint32_t> all(NTILE * ELEMS);
        for (uint32_t n1 = 0; n1 < 32; ++n1) for (uint32_t nh = 0; nh < 2; ++nh) {
            std::vector<uint32_t> t(ELEMS);
            for (uint32_t nl = 0; nl < T; ++nl) for (uint32_t n2 = 0; n2 < T; ++n2)
                t[nl * T + n2] = x[n1 + 32 * n2 + 1024 * (nl + 32 * nh)];
            auto tl = tilize32(t);
            std::copy(tl.begin(), tl.end(), all.begin() + (size_t)(n1 * 2 + nh) * ELEMS); }
        t_wr(cq, b, all, false);
        return b; };
    auto up_Cm = [&](const std::vector<uint32_t>& x, uint32_t li){ std::vector<uint32_t> m(N);
        for (uint32_t n = 0; n < N; ++n) m[n] = ctx[li].to_montgomery(x[n]); return up_C(m); };
    auto rd_C = [&](B64& b){ std::vector<uint32_t> x(N), raw;
        t_rd(cq, raw, b, true);
        for (uint32_t n1 = 0; n1 < 32; ++n1) for (uint32_t nh = 0; nh < 2; ++nh) {
            std::vector<uint32_t> tile(raw.begin() + (size_t)(n1 * 2 + nh) * ELEMS,
                                       raw.begin() + (size_t)(n1 * 2 + nh + 1) * ELEMS);
            auto t = untilize32(tile);
            for (uint32_t nl = 0; nl < T; ++nl) for (uint32_t n2 = 0; n2 < T; ++n2)
                x[n1 + 32 * n2 + 1024 * (nl + 32 * nh)] = t[nl * T + n2]; }
        return x; };
    auto up_E = [&](const std::vector<uint32_t>& X, bool mont, uint32_t li){ B64 b = mkB64();
        std::vector<uint32_t> all(NTILE * ELEMS);
        for (uint32_t kl = 0; kl < 32; ++kl) for (uint32_t kh = 0; kh < 2; ++kh) {
            std::vector<uint32_t> t(ELEMS);
            for (uint32_t ka = 0; ka < T; ++ka) for (uint32_t ku = 0; ku < T; ++ku) {
                uint32_t v = X[kl + 32 * kh + 64 * ku + 2048 * ka];
                t[ka * T + ku] = mont ? ctx[li].to_montgomery(v) : v; }
            auto tl = tilize32(t);
            std::copy(tl.begin(), tl.end(), all.begin() + (size_t)(kl * 2 + kh) * ELEMS); }
        t_wr(cq, b, all, false);
        return b; };
    auto fill64 = [&](uint32_t v){ B64 b = mkB64(); std::vector<uint32_t> t(ELEMS, v); auto tl = tilize32(t);
        std::vector<uint32_t> all(NTILE * ELEMS);
        for (uint32_t i = 0; i < NTILE; ++i) std::copy(tl.begin(), tl.end(), all.begin() + (size_t)i * ELEMS);
        t_wr(cq, b, all, false); return b; };

    // ---- matmul builders (as in the transform check above) ----
    struct MM { Buf in, w, out; uint32_t pg, li; };  // pg = tile index = the page in its 64-page buffer
    auto build_mm = [&](const std::vector<MM>& j, uint32_t base, uint32_t cnt, const char* kern,
                        const Buf& ip, const Buf& wp) {
        Program p = CreateProgram(); auto crs = crs_of(cnt);
        mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, 2, I32);
        mkcb(p, crs, CBIndex::c_1, tt::DataFormat::Int8, NB*NB, I8);
        mkcb(p, crs, CBIndex::c_2, tt::DataFormat::Int8, NB, I8);
        mkcb(p, crs, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);
        mkcb(p, crs, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*ip).append_to(rc); TensorAccessorArgs(*wp).append_to(rc);
        auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused_batch_pg.cpp", crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX + std::string(kern), crs,
            ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<cnt;++i){ auto& b=j[base+i]; auto c=CC(i);
            SetRuntimeArgs(p, rd, c, {(uint32_t)b.in->address(), (uint32_t)b.w->address(), b.pg});
            SetRuntimeArgs(p, wr, c, {(uint32_t)b.out->address(), 1u, b.pg});
            const uint64_t q = ck.QP[b.li]; const auto& C = ctx[b.li];
            SetRuntimeArgs(p, cp, c, {(uint32_t)q, C.q_inv_32(), (uint32_t)(q&0xFFFF), (uint32_t)(q>>16), C.to_montgomery(16384u)}); }
        distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };
    struct MMA { std::vector<Buf> in, w; Buf out; uint32_t pg, opg, li; };  // pg = first input page
    auto nsub_of = [&](uint32_t nacc, uint64_t q) -> uint32_t {
        const uint64_t plane_max = (uint64_t)NB * 32ull * 127ull * 127ull;
        const uint64_t r_max = (uint64_t)nacc * plane_max * 129ull + q;
        return (r_max < (1ull << 32)) ? (uint32_t)(r_max / q) + 1u : 0u; };
    auto build_mma = [&](const std::vector<MMA>& j, uint32_t base, uint32_t cnt,
                         const Buf& ip, const Buf& wp) {
        Program p = CreateProgram(); auto crs = crs_of(cnt); const uint32_t ACC = 2;
        mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, ACC, I32);
        mkcb(p, crs, CBIndex::c_1, tt::DataFormat::Int8, ACC*NB*NB, I8);
        mkcb(p, crs, CBIndex::c_2, tt::DataFormat::Int8, ACC*NB, I8);
        mkcb(p, crs, CBIndex::c_3, tt::DataFormat::Int32, NB, I32);
        mkcb(p, crs, CBIndex::c_16, tt::DataFormat::UInt32, 2, I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*ip).append_to(rc); TensorAccessorArgs(*wp).append_to(rc);
        auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused_acc_pg.cpp", crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp", crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a_acc.cpp", crs,
            ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        for (uint32_t i=0;i<cnt;++i){ auto& b=j[base+i]; auto c=CC(i);
            std::vector<uint32_t> ra{ACC, b.pg};
            for (auto& x : b.in) ra.push_back((uint32_t)x->address());
            for (auto& x : b.w)  ra.push_back((uint32_t)x->address());
            SetRuntimeArgs(p, rd, c, ra);
            SetRuntimeArgs(p, wr, c, {(uint32_t)b.out->address(), 1u, b.opg});
            const uint64_t q = ck.QP[b.li]; const auto& C = ctx[b.li];
            SetRuntimeArgs(p, cp, c, {(uint32_t)q, C.q_inv_32(), (uint32_t)(q&0xFFFF), (uint32_t)(q>>16),
                                      C.to_montgomery(16384u), ACC, nsub_of(ACC, q)}); }
        distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; };

    // ---- elementwise over the 64 tiles ----
    // ---- elementwise: each job carries its own limb, so one dispatch spans limbs ----
    struct EW { Buf x, y, o; uint32_t xpg, ypg, li; };  // x and o share a page; y may not
    auto ew_jobs = [&](const char* rdk, const char* cpk, const std::vector<EW>& j,
                       const std::function<std::vector<uint32_t>(uint32_t)>& args, bool modmul) {
        KLBL = std::string(cpk).substr(8, std::string(cpk).size()-12);
        for (uint32_t base = 0; base < j.size(); base += NC) {
            uint32_t cnt = std::min<uint32_t>(NC, (uint32_t)j.size() - base);
            emit([&]{ Program p = CreateProgram(); auto crs = crs_of(cnt);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32);
                mkcb(p,crs,CBIndex::c_1,tt::DataFormat::UInt32,2,I32);
                mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX+std::string(rdk),crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX+std::string(cpk),crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t k=0;k<cnt;++k){ auto& b=j[base+k]; auto c=CC(k);
                    (void)modmul;   // reader_ab is a superset of reader_modmul: independent a/b starts
                    SetRuntimeArgs(p,rd,c,{(uint32_t)b.x->address(),(uint32_t)b.y->address(),1u,b.xpg,b.ypg});
                    SetRuntimeArgs(p,wr,c,{(uint32_t)b.o->address(),1u,b.xpg});
                    SetRuntimeArgs(p,cp,c,args(b.li)); }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; }); } };
    auto ew = [&](const char* rdk, const char* cpk, B64& x, B64& y, B64& o, std::vector<uint32_t> ca, bool modmul) {
        KLBL = std::string(cpk).substr(8, std::string(cpk).size()-12);
        for (uint32_t base = 0; base < NTILE; base += NC) { uint32_t cnt = std::min<uint32_t>(NC, NTILE - base);
            emit([&]{ Program p = CreateProgram(); auto crs = crs_of(cnt);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32);
                mkcb(p,crs,CBIndex::c_1,tt::DataFormat::UInt32,2,I32);
                mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX+std::string(rdk),crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX+std::string(cpk),crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                    (void)modmul; SetRuntimeArgs(p,rd,c,{A(x),A(y),1u,t,t});
                    SetRuntimeArgs(p,wr,c,{A(o),1u,t});
                    SetRuntimeArgs(p,cp,c,ca); }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; }); } };
    // SCALAR-OPERAND ops: one DRAM operand instead of two. exp_Tg018 built the second operand with
    // fill64() -- a 64-tile, 256 KB DRAM buffer holding one repeated 32-bit number -- and read 4 KB of
    // it per tile. The compute kernels already materialise q / q_inv / q_low / q_high in DST with
    // fill_tile_int; a scalar is no different, so it travels as a runtime arg and cb_in1 disappears.
    auto ews1 = [&](const char* cpk, B64& x, B64& o, std::vector<uint32_t> ca) {
        KLBL = std::string(cpk).substr(8, std::string(cpk).size()-12);
        for (uint32_t base = 0; base < NTILE; base += NC) { uint32_t cnt = std::min<uint32_t>(NC, NTILE - base);
            emit([&]{ Program p = CreateProgram(); auto crs = crs_of(cnt);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32);
                mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX+std::string(cpk),crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                    SetRuntimeArgs(p,rd,c,{A(x),1u,t});
                    SetRuntimeArgs(p,wr,c,{A(o),1u,t});
                    SetRuntimeArgs(p,cp,c,ca); }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; }); } };
    // BATCHED scalar ops. The single-buffer form below dispatches 64 tiles = 64 cores, whatever G is;
    // with 46-48 limbs to do, that is 754 dispatches each leaving 66 of 130 cores idle. Collecting
    // (limb, tile) into one job list fills the grid, exactly as ew_jobs already does for the twiddles.
    // Every argument is per-core, so one dispatch spans limbs with different scalars and primes.
    struct EWS { Buf x, o; uint32_t pg, sc, li; };
    auto ews1_jobs = [&](const char* cpk, const std::vector<EWS>& j,
                         const std::function<std::vector<uint32_t>(const EWS&)>& args) {
        KLBL = std::string(cpk).substr(8, std::string(cpk).size()-12);
        for (size_t base = 0; base < j.size(); base += NC) {
            uint32_t cnt = std::min<uint32_t>(NC, (uint32_t)(j.size() - base));
            emit([&]{ Program p = CreateProgram(); auto crs = crs_of(cnt);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32);
                mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX+std::string(cpk),crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t k=0;k<cnt;++k){ const EWS& b=j[base+k]; auto c=CC(k);
                    SetRuntimeArgs(p,rd,c,{(uint32_t)b.x->address(),1u,b.pg});
                    SetRuntimeArgs(p,wr,c,{(uint32_t)b.o->address(),1u,b.pg});
                    SetRuntimeArgs(p,cp,c,args(b)); }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; }); } };
    auto pws_jobs = [&](const std::vector<EWS>& j){
        ews1_jobs("compute/modmul_scalar.cpp", j, [&](const EWS& b){ uint64_t m = ck.QP[b.li];
            return std::vector<uint32_t>{1u,(uint32_t)m,ctx[b.li].q_inv_32(),(uint32_t)(m&0xFFFF),(uint32_t)(m>>16),b.sc}; }); };
    auto madds_jobs = [&](const std::vector<EWS>& j){
        ews1_jobs("compute/modadd_scalar.cpp", j, [&](const EWS& b){
            return std::vector<uint32_t>{1u,(uint32_t)ck.QP[b.li],b.sc}; }); };
    auto msub_jobs = [&](const std::vector<EW>& j){
        ew_jobs("dataflow/reader_ab.cpp","compute/modsub_reduce.cpp", j,
                [&](uint32_t li){ return std::vector<uint32_t>{(uint32_t)ck.QP[li]}; }, false); };

    // two-operand batched forms, for the tensor and the combine -- same reason as pws_jobs above
    auto pw_jobs = [&](const std::vector<EW>& j){
        ew_jobs("dataflow/reader_ab.cpp","compute/modmul_fused3.cpp", j,
                [&](uint32_t li){ uint64_t m = ck.QP[li];
                    return std::vector<uint32_t>{1u,(uint32_t)m,ctx[li].q_inv_32(),
                                                 (uint32_t)(m&0xFFFF),(uint32_t)(m>>16)}; }, true); };
    auto madd_jobs = [&](const std::vector<EW>& j){
        ew_jobs("dataflow/reader_ab.cpp","compute/modadd_map.cpp", j,
                [&](uint32_t li){ return std::vector<uint32_t>{1u,(uint32_t)ck.QP[li]}; }, false); };

    auto pws = [&](B64& x, uint32_t sc, B64& o, uint32_t li){ uint64_t m = ck.QP[li];
        ews1("compute/modmul_scalar.cpp", x, o,
             {1u,(uint32_t)m,ctx[li].q_inv_32(),(uint32_t)(m&0xFFFF),(uint32_t)(m>>16),sc}); };
    auto madds = [&](B64& x, uint32_t sc, B64& o, uint32_t li){
        ews1("compute/modadd_scalar.cpp", x, o, {1u,(uint32_t)ck.QP[li],sc}); };

    auto PWA = [&](uint32_t li){ uint64_t m = ck.QP[li];
        return std::vector<uint32_t>{1u,(uint32_t)m,ctx[li].q_inv_32(),(uint32_t)(m&0xFFFF),(uint32_t)(m>>16)}; };
    auto pw   = [&](B64& a, B64& b, B64& o, uint32_t li){ ew("dataflow/reader_ab.cpp","compute/modmul_fused3.cpp",a,b,o,PWA(li),true); };
    auto madd = [&](B64& a, B64& b, B64& o, uint32_t li){ ew("dataflow/reader_ab.cpp","compute/modadd_map.cpp",a,b,o,{1u,(uint32_t)ck.QP[li]},false); };
    auto msub = [&](B64& a, B64& b, B64& o, uint32_t li){ ew("dataflow/reader_ab.cpp","compute/modsub_reduce.cpp",a,b,o,{(uint32_t)ck.QP[li]},false); };

    // ---- the six-pass transform, on shared scratch (calls are sequential) ----
    B64 S1 = mkB64(), S2 = mkB64(), S3 = mkB64(), S4 = mkB64(), S5 = mkB64();
    auto a2a = [&](B64& in, B64& o) {
        KLBL = "a2a_corner";
        for (uint32_t kh = 0; kh < 2; ++kh) {
            const uint32_t cnt = std::min<uint32_t>(NC, NHALF);
            emit([&]{ Program p = CreateProgram(); auto crs = crs_of(cnt);
                // BOTH NoCs: NCRISC/NOC_1 fetches tiles 0..15, BRISC/NOC_0 fetches 16..31 and then
                // assembles and writes. The corner turn is read-dominated (128 KB in, 4 KB out per
                // core), so splitting the reads is what pays -- docs/he/dram_guide.md §3.1.
                mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, NHALF / 2, I32);
                mkcb(p, crs, CBIndex::c_1, tt::DataFormat::UInt32, NHALF / 2, I32);
                mkcb(p, crs, CBIndex::c_2, tt::DataFormat::UInt32, 1, I32);
                auto k = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/a2a_corner_rd.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto k2 = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/a2a_corner_wr.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                for (uint32_t c = 0; c < cnt; ++c) {
                    uint32_t n_out = 0; for (uint32_t s = c; s < NHALF; s += cnt) ++n_out;
                    const std::vector<uint32_t> ra{NHALF, n_out, c, cnt, kh, A(in), A(o)};
                    SetRuntimeArgs(p, k, CC(c), ra); SetRuntimeArgs(p, k2, CC(c), ra); }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; }); } };
    auto twid = [&](B64& x, Buf tw, B64& o, uint32_t li) {
        KLBL = "modmul_fused3";
        for (uint32_t base = 0; base < NTILE; base += NC) { uint32_t cnt = std::min<uint32_t>(NC, NTILE - base);
            emit([&]{ Program p = CreateProgram(); auto crs = crs_of(cnt);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32);
                mkcb(p,crs,CBIndex::c_1,tt::DataFormat::UInt32,2,I32);
                mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_fused3.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                uint64_t q = ck.QP[li];
                for (uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                    SetRuntimeArgs(p,rd,c,{A(x),A(tw),1u,t,t});
                    SetRuntimeArgs(p,wr,c,{A(o),1u,t});
                    SetRuntimeArgs(p,cp,c,{1u,(uint32_t)q,ctx[li].q_inv_32(),(uint32_t)(q&0xFFFF),(uint32_t)(q>>16)}); }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; }); } };
    // ---- one dispatch spans LPD limbs, so a pass fills LPD*64 cores ----
    // A single transform has exactly 64 tiles of work, so at G=64 every core does one tile and 66 of
    // the 130 sit idle. There are hundreds of INDEPENDENT transforms in a multiply, so batching limbs
    // is what fills the grid -- each core carries its own prime through per-core runtime args.
    // limbs per dispatch: a SCHEDULING choice, DERIVED from G -- not a configuration axis.
    //
    // The workload is described by G, B, S and nothing else (docs/he/exp_tb001.md). G is a budget the
    // program "must be built to fit and USE"; one N=65536 transform is only 64 tiles, so a program
    // handed G=128 has to pack two limbs into each launch or half the budget idles. That packing is
    // how the program uses its G, so it follows from G rather than being set beside it.
    //
    // It is emphatically not B: B replicates the whole ct_mul across disjoint core groups
    // (ct_muls = B*S), while this still completes exactly ONE ct_mul and only widens each dispatch
    // inside it. Both identities hold: ct_muls = B*S = 1, cores occupied = G*B.
    //
    // TG18NTT_LPD overrides it, for the A/B that shows G alone does nothing (G=128 with LPD=1 measures
    // identically to G=64). That is a diagnostic, not a knob anyone should need to set.
    // LPD -- limbs batched into one dispatch -- DERIVED from G and from the tightest stage.
    //
    // The old rule was NC/NTILE: "how many limbs fill the matmul pass". That is the wrong stage. Each
    // limb offers a different number of independent jobs, and the grid is only full when the SMALLEST
    // of those counts covers G:
    //     transform matmul   NTILE           = 64 jobs per limb
    //     corner turn        2 * NHALF / K   = 32 jobs per limb at K=2   <- binding
    // so LPD = ceil(G / 32) = 5 at G=130, not 2. Under the old rule the corner turn ran 64 of 130
    // cores while the matmul ran all of them -- the same shape of bug the FBC had.
    //
    // K appears in the formula because raising it shortens the corner turn's job list, which is the
    // other half of why K=4 regressed: it needs LPD >= 9 to keep the grid full.
    //
    // ...AND THAT FORMULA MEASURES WORSE, so it is not what runs. ceil(130/32) = 5 gives 78.091 ms
    // against 69.374 at LPD=4. Filling the tightest stage is only half the problem: LPD also sets how
    // the OTHER stages chunk, and 5 limbs give the matmul 320 jobs, which at G=130 is 130+130+60 --
    // a third of a dispatch wasted on the tail. LPD=4's 256 jobs chunk as 130+126. No LPD divides
    // every stage evenly, so the balance is empirical:
    //
    //     LPD=2  72.678 ms      LPD=4  69.374      LPD=5  78.091      LPD=8  69.615
    //
    // 4 it is. Recorded as a measured constant with a known reason, not a derived one -- the
    // derivation above is the right way to think about occupancy and the wrong way to pick LPD.
    const uint32_t A2A_K = [&]{ uint32_t k = 2;
        if (const char* e = std::getenv("TG18NTT_A2A_OUT")) { int v = std::atoi(e);
            if (v > 0 && NHALF % (uint32_t)v == 0) k = (uint32_t)v; } return k; }();
    // TG18NTT_A2A: `bothnoc` (default, = exp_Tg018m's stage, the control) or `exchange`
    // (docs/matmul/exp_ntt001.md §2b-sexies's core-to-core corner turn, 5.40x there).
    const std::string A2A_MODE = [&]{ const char* e = std::getenv("TG18NTT_A2A");
                                      return std::string(e ? e : "bothnoc"); }();
    // `xprobe` is the exchange with both DRAM accesses deleted -- it prices what a fully DRAM-free
    // corner turn could be worth. It computes a WRONG answer on purpose; the correctness gate MUST
    // fail for its timing to mean anything (exp_ntt001 2b-quinquies).
    // TG18NTT_FUSE=1: the whole NTT as ONE dispatch with the corner turn on-chip (ntt64_fused.cpp).
    // It subsumes TG18NTT_A2A entirely -- there is no separate corner-turn stage left to select.
    bool NTT_FUSE_RT = std::getenv("TG18NTT_FUSE") != nullptr;
    const bool& NTT_FUSE = NTT_FUSE_RT;
    const bool A2A_X = (A2A_MODE == "exchange" || A2A_MODE == "xprobe");
    const bool A2A_PROBE = (A2A_MODE == "xprobe");
    if (A2A_MODE != "bothnoc" && A2A_MODE != "exchange" && A2A_MODE != "xprobe") {
        fmt::print(stderr, "TG18NTT_A2A must be bothnoc, exchange or xprobe, got '{}'\n", A2A_MODE); return false; }
    if (A2A_PROBE) fmt::print("  *** xprobe: corner turn computes a WRONG answer by design; "
                              "the correctness gate MUST fail or this timing is invalid ***\n");
    fmt::print("  corner turn: {}  (K={} outputs/core, {} cores per block)\n",
               A2A_MODE, A2A_K, NHALF / A2A_K);
    uint32_t LPD = 4;
    if (const char* e = std::getenv("TG18NTT_LPD")) { int v = std::atoi(e); if (v > 0) LPD = (uint32_t)v; }
    fmt::print("  scheduling: {} limb(s) per dispatch ({} of {} cores used per pass)\n",
               LPD, std::min(NC, LPD * NTILE), NC);
    struct NJob { B64 in, out; uint32_t li; };
    // Scratch allocated EAGERLY, and returned BY VALUE. Two reasons, both bugs I hit:
    //   * growing SC lazily reallocates the vector and invalidates a B64& taken earlier in the same
    //     pass (two scratch() calls per pass), which segfaults on the stale shared_ptr;
    //   * allocating it inside dev_mul would consume POOL slots on the first call and not on later
    //     ones, shifting every cached workload's baked addresses.
    std::vector<B64> SC;
    for (uint32_t i = 0; i < LPD * 5; ++i) SC.push_back(mkB64());
    auto scratch = [&](uint32_t slot, uint32_t which) -> B64 { return SC[slot * 5 + which]; };
    auto a2a_jobs = [&](const std::vector<std::pair<B64,B64>>& g) {   // each pair = one 64-tile buffer set
        KLBL = "a2a_corner";
        struct AJ { const B64* in; const B64* out; uint32_t kh; };
        std::vector<AJ> aj;
        for (auto& pr : g) for (uint32_t kh = 0; kh < 2; ++kh) aj.push_back({&pr.first, &pr.second, kh});
        // OUTPUTS PER CORE. Every core reads ALL NHALF input tiles -- that is what an all-to-all is --
        // but produced only ONE output tile, so those 32 reads were amortised over one write. They are
        // pure redundancy: give a core K outputs and the same 32 reads serve all K, cutting both the
        // dispatch count and the traffic per output by K. The kernel already takes (out_start,
        // out_stride, n_out), so this is a host-side change only.
        //
        // K has an OPTIMUM, and it is not large. Two things push back. The K face shuffles are real
        // RISC work that does not shrink. And raising K shrinks `per`, so a dispatch needs B >= K
        // transforms in its batch to still fill the grid -- below that, cores go idle and the
        // utilisation loss outweighs the traffic saved. Measured, whole multiply at G=64:
        //     K=1  122.427 ms      K=2  120.618 ms      K=4  129.082 ms
        // K=2 it is.
        const uint32_t K = A2A_K;
        const uint32_t per = NHALF / K, tot = (uint32_t)aj.size() * per;

        // ---- EXCHANGE (TG18NTT_A2A=exchange): each core reads only the K tiles it OWNS and pushes
        // rows straight into the peers that need them.  docs/matmul/exp_ntt001.md §2b-sexies.
        //
        // The row assignment is IDENTICAL to the DRAM arm above -- core sIdx of a block owns output
        // rows {sIdx, sIdx+per, ...}, which is exactly a2a_corner_wr's (out_start=sIdx,
        // out_stride=per) -- so this is a like-for-like swap of the same stage, not a reshape.
        //
        // ONE THING MUST CHANGE ON THE HOST: chunking.  The DRAM arm strides `base += NC`, which can
        // put part of a block in one dispatch and the rest in the next; that is harmless when cores
        // are independent and DEADLOCKS an exchange, because a core waits for all `per` peers of its
        // block. So chunk in WHOLE BLOCKS: bpd = NC/per blocks per dispatch, bpd*per cores. At NC=130,
        // per=16 that is 8 blocks = 128 cores, so two cores go idle -- the price of the barrier.
        if (A2A_X) {
            const uint32_t bpd = std::max<uint32_t>(1u, NC / per);      // whole blocks per dispatch
            const uint32_t njob = (uint32_t)aj.size();
            for (uint32_t jb = 0; jb < njob; jb += bpd) {
                const uint32_t nb = std::min(bpd, njob - jb);           // blocks in this dispatch
                const uint32_t cnt = nb * per;
                emit([&]{ Program p = CreateProgram(); auto crs = crs_of(cnt);
                    mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, K, I32);   // the K tiles I own
                    mkcb(p, crs, CBIndex::c_2, tt::DataFormat::UInt32, K, I32);   // the K I assemble
                    const uint32_t sem = CreateSemaphore(p, crs, 0);
                    auto k = CreateKernel(p, A2A_PROBE ? OVERRIDE_KERNEL_PREFIX "dataflow/a2a_corner_x_probe.cpp"
                                                      : OVERRIDE_KERNEL_PREFIX "dataflow/a2a_corner_x.cpp", crs,
                        DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                    for (uint32_t b = 0; b < nb; ++b) {
                        const AJ& j = aj[jb + b];
                        // physical coords of THIS block's per peers, in peer-index order
                        std::vector<uint32_t> co;
                        for (uint32_t q = 0; q < per; ++q) {
                            auto ph = mesh->worker_core_from_logical_core(CC(b * per + q));
                            co.push_back((uint32_t)ph.x); co.push_back((uint32_t)ph.y); }
                        for (uint32_t q = 0; q < per; ++q) {
                            std::vector<uint32_t> ra{NHALF, per, q, j.kh, A(*j.in), A(*j.out), sem};
                            ra.insert(ra.end(), co.begin(), co.end());
                            SetRuntimeArgs(p, k, CC(b * per + q), ra); } }
                    distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; }); }
            return; }

        for (uint32_t base = 0; base < tot; base += NC) {
            uint32_t cnt = std::min<uint32_t>(NC, tot - base);
            emit([&]{ Program p = CreateProgram(); auto crs = crs_of(cnt);
                // BOTH NoCs: NCRISC/NOC_1 fetches tiles 0..15, BRISC/NOC_0 fetches 16..31 and then
                // assembles and writes. The corner turn is read-dominated (128 KB in, 4 KB out per
                // core), so splitting the reads is what pays -- docs/he/dram_guide.md §3.1.
                mkcb(p, crs, CBIndex::c_0, tt::DataFormat::UInt32, NHALF / 2, I32);
                mkcb(p, crs, CBIndex::c_1, tt::DataFormat::UInt32, NHALF / 2, I32);
                mkcb(p, crs, CBIndex::c_2, tt::DataFormat::UInt32, 1, I32);
                auto k = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/a2a_corner_rd.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto k2 = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/a2a_corner_wr.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                for (uint32_t c = 0; c < cnt; ++c) {
                    uint32_t g = (base + c) / per, sIdx = (base + c) % per;
                    const B64& in = *aj[g].in; const B64& out = *aj[g].out; uint32_t kh = aj[g].kh;
                    const std::vector<uint32_t> ra{NHALF, K, sIdx, per, kh, A(in), A(out)};
                    SetRuntimeArgs(p, k, CC(c), ra); SetRuntimeArgs(p, k2, CC(c), ra); }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; }); } };
    // ---- THE WHOLE NTT AS ONE DISPATCH (TG18NTT_FUSE=1) -----------------------------------------
    // Six passes and their five DRAM round trips collapse into one program. A block of P=16 cores
    // carries one transform: core `me` owns n1 in {me, me+16}, i.e. tiles (pages) {2me, 2me+1,
    // 2me+32, 2me+33}. Everything but the corner turn is tile-local, and the corner turn maps input
    // tile n1 -> output row n1 within a half, so that ownership is a fixed point of the whole chain.
    // BRISC runs the exchange core-to-core while the compute kernel is parked on cb_d; NOTHING in
    // between touches DRAM. See kernels/experiments/exp_tb001/compute/ntt64_fused.cpp.
    const uint32_t FP = 16;                                       // cores per block == per transform
    auto ntt_batch_fused = [&](const std::vector<NJob>& jobs, bool fwd) {
        if (jobs.empty()) return;
        KLBL = "ntt64_fused";
        const uint32_t bpd = std::max<uint32_t>(1u, NC / FP);      // whole blocks per dispatch
        const uint32_t nj = (uint32_t)jobs.size();
        for (uint32_t base = 0; base < nj; base += bpd) {
            const uint32_t nb = std::min(bpd, nj - base), cnt = nb * FP;
            emit([&]{ Program p = CreateProgram(); auto crs = crs_of(cnt);
                mkcb(p, crs, CBIndex::c_0,  tt::DataFormat::UInt32, 4, I32);   // in
                mkcb(p, crs, CBIndex::c_1,  tt::DataFormat::Int8,  64, I8);    // w64 x4
                mkcb(p, crs, CBIndex::c_2,  tt::DataFormat::Int8,  16, I8);    // w32a
                mkcb(p, crs, CBIndex::c_3,  tt::DataFormat::Int8,  16, I8);    // w32b
                mkcb(p, crs, CBIndex::c_4,  tt::DataFormat::UInt32, 2, I32);   // ta
                mkcb(p, crs, CBIndex::c_5,  tt::DataFormat::UInt32, 4, I32);   // tb
                mkcb(p, crs, CBIndex::c_6,  tt::DataFormat::Int8,   8, I8);    // limbs
                mkcb(p, crs, CBIndex::c_7,  tt::DataFormat::UInt32, 4, I32);   // working p
                mkcb(p, crs, CBIndex::c_8,  tt::DataFormat::UInt32, 4, I32);   // working q
                mkcb(p, crs, CBIndex::c_9,  tt::DataFormat::UInt32, 4, I32);   // a2a source
                mkcb(p, crs, CBIndex::c_10, tt::DataFormat::UInt32, 4, I32);   // a2a result
                mkcb(p, crs, CBIndex::c_16, tt::DataFormat::UInt32, 4, I32);   // out
                const uint32_t sem = CreateSemaphore(p, crs, 0);
                std::vector<uint32_t> rc; TensorAccessorArgs(*KC[jobs[base].li*2+1].w32a).append_to(rc);
                auto rd = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/ntt64_fused_rd.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
                auto wr = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "dataflow/ntt64_fused_wr.cpp", crs,
                    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                // TG18NTT_ZONES=1 splits this kernel into its SFPU and matrix-engine phases, for the
                // "what fraction of ct_mul is SFPU work" question. Needs TG18NTT_DRAIN_ONLY.
                std::map<std::string, std::string> zdef;
                if (std::getenv("TG18NTT_ZONES")) zdef["NTT64_PHASE_ZONES"] = "1";
                auto cp = CreateKernel(p, OVERRIDE_KERNEL_PREFIX "compute/ntt64_fused.cpp", crs,
                    ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false,
                                  .defines=zdef});
                for (uint32_t b = 0; b < nb; ++b) {
                    const NJob& J = jobs[base + b];
                    RC& r = KC[J.li * 2 + (fwd ? 1 : 0)];
                    const uint64_t q = ck.QP[J.li]; const auto& C = ctx[J.li];
                    std::vector<uint32_t> co;                       // this block's P peers, in order
                    for (uint32_t t = 0; t < FP; ++t) {
                        auto ph = mesh->worker_core_from_logical_core(CC(b * FP + t));
                        co.push_back((uint32_t)ph.x); co.push_back((uint32_t)ph.y); }
                    for (uint32_t me = 0; me < FP; ++me) {
                        auto c = CC(b * FP + me);
                        SetRuntimeArgs(p, rd, c, {me, (uint32_t)J.in->address(),
                            (uint32_t)r.w64[0]->address(), (uint32_t)r.w64[1]->address(),
                            (uint32_t)r.w64[2]->address(), (uint32_t)r.w64[3]->address(),
                            (uint32_t)r.w32a->address(), (uint32_t)r.w32b->address(),
                            (uint32_t)r.ta->address(), (uint32_t)r.tb->address()});
                        SetRuntimeArgs(p, cp, c, {(uint32_t)q, C.q_inv_32(), (uint32_t)(q&0xFFFF),
                                                  (uint32_t)(q>>16), C.to_montgomery(16384u), fwd?1u:0u,
                                                  (uint32_t)(std::getenv("TG18NTT_FUSE_DBG") ? std::atoi(std::getenv("TG18NTT_FUSE_DBG")) : 0)});
                        std::vector<uint32_t> wa{me, (uint32_t)J.out->address(), sem};
                        wa.insert(wa.end(), co.begin(), co.end());
                        SetRuntimeArgs(p, wr, c, wa); } }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; }); } };

    // the six passes, run across a batch of same-direction transforms
    auto ntt_batch = [&](std::vector<NJob> jobs, bool fwd) {
        if (jobs.empty()) return;
        if (NTT_FUSE) { ntt_batch_fused(jobs, fwd); return; }
        const uint32_t B = (uint32_t)jobs.size();
        auto W32 = [&](uint32_t slot, bool a){ RC& r = KC[jobs[slot].li * 2 + (fwd ? 1 : 0)]; return a ? r.w32a : r.w32b; };
        auto mmpass = [&](uint32_t src, uint32_t dst, bool a, const char* kern){
            std::vector<MM> j;
            for (uint32_t b = 0; b < B; ++b) { B64 S = (src == 9) ? jobs[b].in : scratch(b, src);
                B64 D = (dst == 9) ? jobs[b].out : scratch(b, dst);
                for (uint32_t t = 0; t < NTILE; ++t) j.push_back({S, W32(b, a), D, t, jobs[b].li}); }
            KLBL = std::string(kern).substr(8, std::string(kern).size()-12);
            for (uint32_t base = 0; base < j.size(); base += NC) { uint32_t cnt = std::min<uint32_t>(NC, (uint32_t)j.size()-base);
                emit([&]{ return build_mm(j, base, cnt, kern, j[0].in, j[0].w); }); } };
        auto p1pass = [&](uint32_t src, uint32_t dst){
            std::vector<MMA> j;
            for (uint32_t b = 0; b < B; ++b) { RC& r = KC[jobs[b].li * 2 + (fwd ? 1 : 0)];
                B64 S = (src == 9) ? jobs[b].in : scratch(b, src);
                B64 D = (dst == 9) ? jobs[b].out : scratch(b, dst);
                for (uint32_t n1 = 0; n1 < 32; ++n1) for (uint32_t bo = 0; bo < 2; ++bo)
                    j.push_back({{S, S}, {r.w64[bo*2+0], r.w64[bo*2+1]}, D, n1*2, n1*2+bo, jobs[b].li}); }
            KLBL = "fused_matmul_a_acc";
            for (uint32_t base = 0; base < j.size(); base += NC) { uint32_t cnt = std::min<uint32_t>(NC, (uint32_t)j.size()-base);
                emit([&]{ return build_mma(j, base, cnt, j[0].in[0], j[0].w[0]); }); } };
        auto twpass = [&](uint32_t src, uint32_t dst, bool useTa){
            std::vector<EW> j;
            for (uint32_t b = 0; b < B; ++b) { RC& r = KC[jobs[b].li * 2 + (fwd ? 1 : 0)];
                B64 S = (src == 9) ? jobs[b].in : scratch(b, src);
                B64 D = (dst == 9) ? jobs[b].out : scratch(b, dst);
                for (uint32_t t = 0; t < NTILE; ++t)
                    j.push_back({S, useTa ? r.ta : r.tb, D, t, useTa ? (t & 1) : t, jobs[b].li}); }
            ew_jobs("dataflow/reader_ab.cpp","compute/modmul_fused3.cpp", j,
                    [&](uint32_t li){ uint64_t m = ck.QP[li];
                        return std::vector<uint32_t>{1u,(uint32_t)m,ctx[li].q_inv_32(),(uint32_t)(m&0xFFFF),(uint32_t)(m>>16)}; }, true); };
        auto a2apass = [&](uint32_t src, uint32_t dst){
            std::vector<std::pair<B64,B64>> g;
            for (uint32_t b = 0; b < B; ++b) g.push_back({(src==9)?jobs[b].in:scratch(b,src), (dst==9)?jobs[b].out:scratch(b,dst)});
            a2a_jobs(g); };
        if (fwd) { p1pass(9,0); twpass(0,1,true); mmpass(1,2,true,"compute/fused_matmul_c.cpp");
                   twpass(2,3,false); a2apass(3,4); mmpass(4,9,false,"compute/fused_matmul_a.cpp"); }
        else     { mmpass(9,0,false,"compute/fused_matmul_a.cpp"); a2apass(0,1); twpass(1,2,false);
                   mmpass(2,3,true,"compute/fused_matmul_c.cpp"); twpass(3,4,true); p1pass(4,9); } };
    // single-transform convenience wrapper, used by the correctness ladder
    auto ntt = [&](B64& in, B64& o, uint32_t li, bool fwd) { ntt_batch({{in, o, li}}, fwd); };
    // run a list of transforms LPD at a time
    auto ntt_all = [&](std::vector<NJob> js, bool fwd) {
        // Fused: a dispatch carries NC/FP transforms (8 at G=130), not LPD limbs -- the shape is set
        // by the block size, and ntt_batch_fused chunks internally, so hand it the whole list.
        if (NTT_FUSE) { ntt_batch(js, fwd); return; }
        for (size_t i = 0; i < js.size(); i += LPD)
            ntt_batch(std::vector<NJob>(js.begin() + i, js.begin() + std::min(js.size(), i + LPD)), fwd); };

    // ---- MATRIX-ENGINE FBC (TG18NTT_FBC=fpu) --------------------------------------------------
    // docs/he/exp_tg018r.md measured the matrix engine beating the SFPU on the base conversion in
    // EVERY cell of (Ls,Ld) in {8,16,32}^2 -- 1.19x at 8->8 to 10.3x at 32->32 -- and the kernel has
    // been sitting in this same tree, used only by that bench. Two things kept it out of the multiply:
    //
    //   * LAYOUT. reader_fbc_tr.cpp does `y.get_noc_addr(ot*L + i)`: one interleaved buffer holding
    //     every limb. This pipeline keeps L separate 64-page B64s. reader/writer_fbc_tr_pg.cpp take
    //     the L base addresses as runtime args instead -- the only change, same transpose.
    //   * THE CONSTANTS ARE A BUFFER, not runtime args. The int8 folded C2 is 16 tiles, so it cannot
    //     ride in the arg list the way the SFPU route's C2 does. It is therefore built ONCE per
    //     destination prime set and cached: the multiply has only 5 fbc call sites (3 ModUp digits +
    //     2 ModDown) and their prime sets are fixed, so the cache fills on the record pass and no
    //     DRAM write ever happens inside a traced replay.
    //
    // Ld <= 32 is a HARD limit here, not a tuning choice: the matrix engine contracts within a tile,
    // and the contracted axis is the source limb, so one tile holds at most 32 destinations. The
    // SFPU route chunks at 8 because its cost is Ls x Ld; this one chunks at 32 because that is the
    // tile. At Ls=16 that is 2 chunks instead of 6.
    const bool FBC_FPU = [&]{ const char* e = std::getenv("TG18NTT_FBC");
                              return e && std::string(e) == "fpu"; }();
    constexpr uint32_t NRF = 4, NPF = 4;                    // radix planes, radix limbs
    // Ld cap. 32 is the tile and therefore the ALGORITHMIC limit, but 32 is also the one width that
    // comes back wrong through reader/writer_fbc_tr_pg -- see docs/he/exp_tg018ntt.md §4e. 24 is the
    // widest VERIFIED setting and already splits Ldt=48 into two even chunks, which is all the width
    // was wanted for, so it is the default and 32 is refused rather than left as a silent trap.
    const uint32_t CHF = [&]{ const char* e = std::getenv("TG18NTT_FBC_CH");
                              return e ? (uint32_t)std::atoi(e) : 24u; }();
    const uint64_t POWF[NRF] = {1, 128, 16384, 2097152};
    std::map<std::string, std::pair<Buf, Buf>> fbcK;        // dest-prime-set -> (Bt, par)
    if (FBC_FPU) {
        if (CHF < 1 || CHF > 24) { fmt::print("TG18NTT_FBC_CH must be in 1..24 (see §4e); got {}\n", CHF);
                                   return false; }
        fmt::print("  FBC route: MATRIX ENGINE (fbc_fpu_tr), Ld chunked to {}\n", CHF); }

    // ---- approximate FBC. LS reaches 16 (alpha), so the destination list is CHUNKED to 8:
    //      LS=16, LD=12 would be 266 runtime args, past the 256 budget.
    auto fbc = [&](const std::vector<B64>& src, const std::vector<uint32_t>& sI,
                   const std::vector<uint32_t>& dI) -> std::vector<B64> {
        const uint32_t Ls = (uint32_t)sI.size(), Ldt = (uint32_t)dI.size();
        BIG S(1); for (uint32_t i = 0; i < Ls; ++i) S = mul_small(S, ck.QP[sI[i]]);
        std::vector<BIG> shat(Ls);
        for (uint32_t i = 0; i < Ls; ++i) { BIG rr; divmod(S, BIG(ck.QP[sI[i]]), shat[i], rr); }
        std::vector<B64> dst(Ldt); for (uint32_t j = 0; j < Ldt; ++j) dst[j] = mkB64();
        // LD is no longer capped at 8: C2 lives in a CB now, so the only limits are L1 (c_16 holds
        // LD tiles) and the kernel's P/PINV arrays. 48 destinations in one pass instead of six of 8,
        // which stops phase 1 -- the LS-source y scaling -- from being recomputed six times.
        const uint32_t CH = 8;
        // FILL THE GRID. The tile axis is only NTILE=64 long, so chunking it by NC left 64 of 130
        // cores idle on every FBC dispatch -- through 29% of the multiply. The destination-chunk loop
        // (c0) is independent of the tile loop, so the two are flattened into ONE job list and
        // chunked by NC together: 6 chunks x 64 tiles = 384 jobs fills 130 cores instead of 64.
        // Every argument here is already per-core (SetRuntimeArgs takes a CoreCoord), so a dispatch
        // can carry jobs from different c0 chunks with different constants.
        //
        // Jobs are grouped by Ld first. Ld is 8 except for a ragged last chunk, and the c_16 CB depth
        // must equal the writer's group size B or a group straddles the wrap -- so mixed Ld in one
        // program would be unsafe.
        // PHASE 1 HOISTED. y_i = x_i * C1[i] mod q_i depends only on the sources, so it is computed
        // ONCE into Y and every destination chunk reads Y instead of redoing it. That is ~21% of the
        // FBC: measured 6 chunks x 113.3 us against 28.8 + 6 x 84.5. It costs Ls tile writes plus Ls
        // reads per chunk that were reads of `src` before, so the extra traffic is one write pass.
        std::vector<B64> Y(Ls); for (uint32_t i = 0; i < Ls; ++i) Y[i] = mkB64();
        { std::vector<uint32_t> sa{Ls, Ls};
          for (uint32_t i = 0; i < Ls; ++i)
              sa.push_back(ctx[sI[i]].to_montgomery((uint32_t)modinv(mod_small(shat[i], ck.QP[sI[i]]), ck.QP[sI[i]])));
          for (uint32_t i = 0; i < Ls; ++i) sa.push_back((uint32_t)ck.QP[sI[i]]);
          for (uint32_t i = 0; i < Ls; ++i) sa.push_back(ctx[sI[i]].q_inv_32());
          KLBL = "fbc_scale";
          for (uint32_t base = 0; base < NTILE; base += NC) {
            uint32_t cnt = std::min<uint32_t>(NC, NTILE - base);
            emit([&]{ Program p = CreateProgram(); auto crs = crs_of(cnt);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,Ls,I32);
                mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,Ls,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_gather_batch_pg.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_scatter_batch_pg.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fbc_scale.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                    std::vector<uint32_t> ra{Ls, Ls, t};
                    for (uint32_t i=0;i<Ls;++i) ra.push_back(A(src[i]));
                    SetRuntimeArgs(p,rd,c,ra);
                    std::vector<uint32_t> wa{Ls, Ls, Ls, t};
                    for (uint32_t i=0;i<Ls;++i) wa.push_back(A(Y[i]));
                    SetRuntimeArgs(p,wr,c,wa);
                    SetRuntimeArgs(p,cp,c,sa); }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; }); } }

        if (FBC_FPU) {
            struct FJF { uint32_t c0, Ld, t; Buf bt, par; };
            std::vector<FJF> JF;
            for (uint32_t c0 = 0; c0 < Ldt; c0 += CHF) {
                const uint32_t Ld = std::min(CHF, Ldt - c0);
                // Bt depends on shat, which is derived from the SOURCE set -- so sI belongs in the
                // key as much as the destination chunk does. Two call sites can share a destination
                // chunk and not a source set (ModUp digit j converts group[j] -> comp[j]).
                std::string key;
                for (uint32_t i = 0; i < Ls; ++i) key += std::to_string(sI[i]) + ",";
                key += "|";
                for (uint32_t j = 0; j < Ld; ++j) key += std::to_string(dI[c0 + j]) + ",";
                if (!fbcK.count(key)) {
                    // Bt[rr*NPF+p][j][i] = limb_rr( (128^p * (shat_i mod p_j)) mod p_j ).
                    // NO Montgomery fold -- fbc_fpu_tr's recombine returns the plain
                    // sum_rr 128^rr * plane_rr mod p_j, so this matches the SFPU route bit for bit.
                    std::vector<int8_t> Bt((size_t)NRF * NPF * ELEMS, 0);
                    for (uint32_t rr = 0; rr < NRF; ++rr) for (uint32_t pp = 0; pp < NPF; ++pp) {
                        std::vector<int8_t> row(ELEMS, 0);
                        for (uint32_t j = 0; j < Ld; ++j) for (uint32_t i = 0; i < Ls; ++i) {
                            const uint64_t pj = ck.QP[dI[c0 + j]];
                            const uint64_t v  = mod_small(shat[i], pj);
                            const uint64_t Bp = (uint64_t)(((unsigned __int128)POWF[pp] * v) % pj);
                            row[j * T + i] = limb((uint32_t)Bp, rr); }
                        auto til = tilize32(row);
                        std::copy(til.begin(), til.end(), Bt.begin() + (size_t)(rr * NPF + pp) * ELEMS); }
                    // recombine params, broadcast along the row that owns destination j
                    std::vector<uint32_t> par(5 * ELEMS, 0);
                    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c) {
                        const uint32_t j = (r < Ld) ? r : 0, at = r * T + c;
                        const uint64_t pj = ck.QP[dI[c0 + j]]; const auto& C = ctx[dI[c0 + j]];
                        par[0 * ELEMS + at] = C.to_montgomery(16384u);
                        par[1 * ELEMS + at] = C.q_inv_32();
                        par[2 * ELEMS + at] = (uint32_t)(pj >> 16);
                        par[3 * ELEMS + at] = (uint32_t)(pj & 0xFFFFu);
                        par[4 * ELEMS + at] = (uint32_t)pj; }
                    for (uint32_t t = 0; t < 5; ++t) {
                        std::vector<uint32_t> rm(par.begin() + t * ELEMS, par.begin() + (t + 1) * ELEMS);
                        auto til = tilize32(rm);
                        std::copy(til.begin(), til.end(), par.begin() + t * ELEMS); }
                    auto bBt = mkbuf(NRF * NPF, I8); auto bPar = mkbuf(5, I32);
                    t_wr(cq, bBt, Bt, false); t_wr(cq, bPar, par, false); t_fin(cq);
                    fbcK[key] = {bBt, bPar}; }
                auto [bBt, bPar] = fbcK[key];
                for (uint32_t t = 0; t < NTILE; ++t) JF.push_back({c0, Ld, t, bBt, bPar}); }
            // FILL THE GRID, exactly as the SFPU route does. The tile axis is only NTILE=64 long, so
            // one dispatch per destination chunk leaves 66 of 130 cores idle. The chunk loop and the
            // tile loop are independent, so they flatten into ONE job list: 3 x 64 = 192 jobs is two
            // full dispatches instead of three two-thirds-empty ones. Every argument here is already
            // per-core, and unlike the SFPU route this program's CB depths do not depend on Ld (the
            // writer's group size is 1 and c_16 is depth 2 regardless), so jobs with different Ld may
            // share a dispatch and no grouping-by-Ld is needed.
            {   const uint32_t NGRP = ELEMS / T;            // 32 coefficient groups per tile
                KLBL = "fbc_fpu_tr";
                for (uint32_t base = 0; base < JF.size(); base += NC) {
                    uint32_t cnt = std::min<uint32_t>(NC, (uint32_t)JF.size() - base);
                    emit([&]{ Program p = CreateProgram(); auto crs = crs_of(cnt);
                        mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32);
                        mkcb(p,crs,CBIndex::c_1,tt::DataFormat::Int8,NRF*NPF,I8);
                        mkcb(p,crs,CBIndex::c_2,tt::DataFormat::Int8,NPF,I8);
                        mkcb(p,crs,CBIndex::c_3,tt::DataFormat::UInt32,NRF,I32);
                        mkcb(p,crs,CBIndex::c_4,tt::DataFormat::UInt32,5,I32);
                        mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                        // every bBt is mkbuf(NRF*NPF, I8), so all jobs share one accessor spec
                        std::vector<uint32_t> rc; TensorAccessorArgs(*JF[base].bt).append_to(rc);
                        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fbc_tr_pg.cpp",crs,
                            DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
                        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_fbc_tr_pg.cpp",crs,
                            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fbc_fpu_tr.cpp",crs,
                            ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,
                                          .dst_full_sync_en=true,.math_approx_mode=false});
                        for (uint32_t k=0;k<cnt;++k){ const auto& j = JF[base+k]; auto c=CC(k);
                            std::vector<uint32_t> ra{(uint32_t)j.bt->address(), (uint32_t)j.par->address(),
                                                     Ls, NGRP, j.t};
                            for (uint32_t i=0;i<Ls;++i) ra.push_back(A(Y[i]));
                            SetRuntimeArgs(p,rd,c,ra);
                            std::vector<uint32_t> wa{j.Ld, NGRP, j.t};
                            for (uint32_t q=0;q<j.Ld;++q) wa.push_back(A(dst[j.c0+q]));
                            SetRuntimeArgs(p,wr,c,wa);
                            SetRuntimeArgs(p,cp,c,{NGRP}); }
                        distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; }); } }
            return dst; }

        struct FJ { uint32_t c0, Ld, t; };
        std::map<uint32_t, std::vector<FJ>> byLd;
        std::map<uint32_t, std::vector<uint32_t>> caOf;      // c0 -> compute args
        for (uint32_t c0 = 0; c0 < Ldt; c0 += CH) {
            const uint32_t Ld = std::min(CH, Ldt - c0);
            std::vector<uint32_t> ca{Ls, Ld};
            for (uint32_t i = 0; i < Ls; ++i) for (uint32_t j = 0; j < Ld; ++j)
                ca.push_back(ctx[dI[c0+j]].to_montgomery((uint32_t)mod_small(shat[i], ck.QP[dI[c0+j]])));
            for (uint32_t j = 0; j < Ld; ++j) ca.push_back((uint32_t)ck.QP[dI[c0+j]]);
            for (uint32_t j = 0; j < Ld; ++j) ca.push_back(ctx[dI[c0+j]].q_inv_32());
            caOf[c0] = std::move(ca);

            for (uint32_t t = 0; t < NTILE; ++t) byLd[Ld].push_back({c0, Ld, t});
        }
        KLBL = "fbc_combine";
        for (auto& [Ld, J] : byLd) {
          for (uint32_t base = 0; base < J.size(); base += NC) {
            uint32_t cnt = std::min<uint32_t>(NC, (uint32_t)J.size() - base);
            emit([&]{ Program p = CreateProgram(); auto crs = crs_of(cnt);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,Ls,I32); mkcb(p,crs,CBIndex::c_2,tt::DataFormat::UInt32,Ls,I32);
                mkcb(p,crs,CBIndex::c_3,tt::DataFormat::UInt32,Ls,I32); mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,Ld,I32);
                // BATCHED feed/drain: B == the CB depth, so all Ls reads and all Ld writes are in
                // flight behind ONE barrier each instead of one round trip per tile.
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_gather_batch_pg.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_scatter_batch_pg.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fbc_combine.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t k=0;k<cnt;++k){ const FJ& j = J[base+k]; auto c=CC(k);
                    std::vector<uint32_t> ra{Ls, Ls, j.t};            // n, B, page ; c_0 depth == Ls
                    for (uint32_t i=0;i<Ls;++i) ra.push_back(A(Y[i]));
                    SetRuntimeArgs(p,rd,c,ra);
                    std::vector<uint32_t> wa{j.Ld, j.Ld, j.Ld, j.t};  // n_out, n_total, B, page
                    for (uint32_t q=0;q<j.Ld;++q) wa.push_back(A(dst[j.c0+q]));
                    SetRuntimeArgs(p,wr,c,wa);
                    SetRuntimeArgs(p,cp,c,caOf[j.c0]); }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; }); } }
        return dst; };
    // single-source conversion for the rescale
    auto bconv1 = [&](B64 src, const std::vector<uint32_t>& dI) -> std::vector<B64> {
        const uint32_t Ld = (uint32_t)dI.size();
        std::vector<uint32_t> ca{Ld}; for (auto d : dI) ca.push_back((uint32_t)ck.QP[d]);
        std::vector<B64> dst(Ld); for (uint32_t j=0;j<Ld;++j) dst[j]=mkB64();
        KLBL = "bconv_one";
        for (uint32_t base = 0; base < NTILE; base += NC) { uint32_t cnt = std::min<uint32_t>(NC, NTILE - base);
            emit([&]{ Program p=CreateProgram(); auto crs=crs_of(cnt);
                mkcb(p,crs,CBIndex::c_0,tt::DataFormat::UInt32,2,I32);
                mkcb(p,crs,CBIndex::c_16,tt::DataFormat::UInt32,8,I32);
                auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_gather_batch_pg.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_scatter_batch_pg.cpp",crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/bconv_one.cpp",crs,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t k=0;k<cnt;++k){ uint32_t t=base+k; auto c=CC(k);
                    SetRuntimeArgs(p,rd,c,{1u,1u,t,A(src)});                      // n=1, B=1, page=t
                    // Ld reaches 46 while c_16 is 8 deep, so writes go 8 at a time; every group starts
                    // buffer-aligned because B == depth, which is what makes it wrap-safe.
                    std::vector<uint32_t> wa{Ld, Ld, 8u, t};
                    for (uint32_t j=0;j<Ld;++j) wa.push_back(A(dst[j]));
                    SetRuntimeArgs(p,wr,c,wa);
                    SetRuntimeArgs(p,cp,c,ca); }
                distributed::MeshWorkload wl; wl.add_program(dr, std::move(p)); return wl; }); }
        return dst; };

    std::vector<uint32_t> qidx(LQ), pidx(LP);
    for (uint32_t i = 0; i < LQ; ++i) qidx[i] = i;
    for (uint32_t j = 0; j < LP; ++j) pidx[j] = LQ + j;

    // key-switch and rescale constants
    // 160 scalars, held as 160 uint32_t. exp_Tg018 held them as 160 x 64-tile DRAM buffers -- 40 MB
    // of DRAM, and a 4 KB read per tile at every use, to carry 160 distinct 32-bit numbers.
    std::map<uint32_t, uint32_t> PinvS;
    for (uint32_t i = 0; i < LQ; ++i)
        PinvS[i] = ctx[i].to_montgomery((uint32_t)modinv(mod_small(ck.Pprod, ck.QP[i]), ck.QP[i]));
    std::vector<std::map<uint32_t,uint32_t>> qLinvS(g_ndrop), HalfS(g_ndrop);
    for (uint32_t k = 0; k < g_ndrop; ++k) {
        const uint32_t d = DROPI(k); const uint64_t qL = ck.QP[d], qHalf = qL / 2;
        for (uint32_t i = 0; i <= d; ++i) {
            if (i < d) qLinvS[k][i] = ctx[i].to_montgomery((uint32_t)modinv(qL % ck.QP[i], ck.QP[i]));
            HalfS[k][i] = (uint32_t)(qHalf % ck.QP[i]); } }
    (void)fill64;
    auto rescale_dev = [&](std::vector<B64>& cin, std::vector<B64>& out) {
        std::vector<B64> c = cin;
        for (uint32_t k = 0; k < g_ndrop; ++k) {
            const uint32_t d = DROPI(k);
            std::vector<uint32_t> dst(d); for (uint32_t i = 0; i < d; ++i) dst[i] = i;
            if (g_centre) { std::vector<B64> cs(d + 1); std::vector<EWS> jm;
                for (uint32_t i = 0; i <= d; ++i) { cs[i] = mkB64();
                    for (uint32_t t = 0; t < NTILE; ++t) jm.push_back({c[i], cs[i], t, HalfS[k][i], i}); }
                madds_jobs(jm);
                c = cs; }
            auto corr = bconv1(c[d], dst);
            std::vector<B64> o(d), df(d);
            { std::vector<EW> js; std::vector<EWS> jp;
              for (uint32_t i = 0; i < d; ++i) { df[i] = mkB64(); o[i] = mkB64();
                  for (uint32_t t = 0; t < NTILE; ++t) js.push_back({c[i], corr[i], df[i], t, t, i}); }
              msub_jobs(js);
              for (uint32_t i = 0; i < d; ++i)
                  for (uint32_t t = 0; t < NTILE; ++t) jp.push_back({df[i], o[i], t, qLinvS[k][i], i});
              pws_jobs(jp); }
            c = o; }
        for (uint32_t i = 0; i < LOUT(); ++i) out[i] = c[i]; };

    // ---- evk in the EVAL domain, Montgomery, resident. Key material, not multiply cost. ----
    std::map<int, B64> evk0m, evk1m;
    { double t0 = now_s();
      std::vector<std::vector<uint32_t>> f0(DNUM * LQP), f1(DNUM * LQP);
      par_for(DNUM * LQP, [&](uint32_t k){ uint32_t j = k / LQP, li = k % LQP;
          f0[k] = ck.nt[li].fwd(ck.evk0[j][li]); f1[k] = ck.nt[li].fwd(ck.evk1[j][li]); });
      for (uint32_t j = 0; j < DNUM; ++j) for (uint32_t li = 0; li < LQP; ++li) {
          evk0m[j * 100 + li] = up_E(f0[j * LQP + li], true, li);
          evk1m[j * 100 + li] = up_E(f1[j * LQP + li], true, li); }
      t_fin(cq);
      fmt::print("  evk resident: {} digits x {} limbs, {:.1f} s\n", DNUM, LQP, now_s() - t0); }

    // ---- relinearise: dnum=3 hybrid key-switch ----
    auto relin_dev = [&](std::vector<B64>& e2c, std::vector<B64>& e2e,
                         std::vector<B64>& k0, std::vector<B64>& k1) {
        std::map<int, B64> acc0, acc1;
        for (uint32_t j = 0; j < DNUM; ++j) {
            std::vector<B64> xs; for (auto li : ck.group[j]) xs.push_back(e2c[li]);
            auto o = fbc(xs, ck.group[j], ck.comp[j]);
            std::map<int, B64> vje;
            for (auto li : ck.group[j]) vje[li] = e2e[li];
            { std::vector<NJob> nj;
              for (uint32_t k = 0; k < ck.comp[j].size(); ++k) { uint32_t li = ck.comp[j][k];
                  B64 ve = mkB64(); nj.push_back({o[k], ve, li}); vje[li] = ve; }
              ntt_all(nj, true); }
            for (uint32_t li = 0; li < LQP; ++li) {
                B64 p0 = mkB64(), p1 = mkB64();
                pw(vje[li], evk0m[j * 100 + li], p0, li);
                pw(vje[li], evk1m[j * 100 + li], p1, li);
                if (j == 0) { acc0[li] = p0; acc1[li] = p1; }
                else { B64 n0 = mkB64(), n1 = mkB64();
                       madd(acc0[li], p0, n0, li); madd(acc1[li], p1, n1, li);
                       acc0[li] = n0; acc1[li] = n1; } } }
        // mod-down: only the P limbs leave the eval domain; P^-1 is a scalar and the transform is
        // Z_q-linear, so the subtract and the scale stay in eval against the accumulator.
        auto moddown = [&](std::map<int, B64>& acc, std::vector<B64>& out) {
            std::vector<B64> aP; std::vector<NJob> nj;
            for (auto pp : pidx) { B64 c = mkB64(); nj.push_back({acc[pp], c, pp}); aP.push_back(c); }
            ntt_all(nj, false);
            auto f = fbc(aP, pidx, qidx);
            std::vector<B64> fe(LQ); nj.clear();
            for (uint32_t i = 0; i < LQ; ++i) { fe[i] = mkB64(); nj.push_back({f[i], fe[i], i}); }
            ntt_all(nj, true);
            std::vector<B64> dfv(LQ); std::vector<EW> js;
            for (uint32_t i = 0; i < LQ; ++i) {
                dfv[i] = mkB64(); out[i] = mkB64();
                for (uint32_t t = 0; t < NTILE; ++t) js.push_back({acc[i], fe[i], dfv[i], t, t, i}); }
            msub_jobs(js);
            { std::vector<EWS> jp;
              for (uint32_t i = 0; i < LQ; ++i)
                  for (uint32_t t = 0; t < NTILE; ++t) jp.push_back({dfv[i], out[i], t, PinvS[i], i});
              pws_jobs(jp); } };
        moddown(acc0, k0); moddown(acc1, k1); };

    uint32_t bad3=0, tot3=0, bad4=0, tot4=0, bad5=0, tot5=0;
    const uint32_t NCHK = std::getenv("TG18NTT_FULL") ? LQ : 4;   // limbs checked in the ladder

    // ---- [D0] fused NTT vs the unfused chain, on device, same input (TG18NTT_NTTCHK=1) ----
    // Bisecting a wrong fused NTT against the host reference is slow and tells you only "wrong".
    // Both chains are on the device and must agree TILE FOR TILE, so this localises a bug to a page.
    if (std::getenv("TG18NTT_NTTCHK")) {
        const uint32_t li = 0;
        B64 in = mkB64(), oU = mkB64(), oF = mkB64();
        std::vector<uint32_t> raw((size_t)NTILE * ELEMS);
        { std::mt19937 rg(4242); const uint64_t q = ck.QP[li];
          for (auto& v : raw) v = (uint32_t)(rg() % q); }
        t_wr(cq, in, raw, false); t_fin(cq);
        for (uint32_t dir = 0; dir < 2; ++dir) {
            const bool fwd = (dir == 0);
            const bool sv = NTT_FUSE_RT; NTT_FUSE_RT = false;
            // With FUSE_DBG the fused kernel does ONLY the corner turn, so the reference is the DRAM
            // corner turn -- not the whole NTT. That separates plumbing+exchange from arithmetic.
            const int dbg = std::getenv("TG18NTT_FUSE_DBG") ? std::atoi(std::getenv("TG18NTT_FUSE_DBG")) : 0;
            if (dbg == 1) a2a_jobs({{in, oU}});
            else          ntt_batch({{in, oU, li}}, fwd);
            t_fin(cq);
            NTT_FUSE_RT = true;
            ntt_batch({{in, oF, li}}, fwd); t_fin(cq);
            NTT_FUSE_RT = sv;
            std::vector<uint32_t> u, f;
            if (dbg >= 2 && dbg <= 5) { B64 sref = scratch(0, (uint32_t)(dbg - 2)); t_rd(cq, u, sref, true); }
            else t_rd(cq, u, oU, true);
            t_rd(cq, f, oF, true);
            uint32_t bad = 0, firstpg = 0xFFFFFFFFu, badpg = 0;
            for (uint32_t pg = 0; pg < NTILE; ++pg) { uint32_t d = 0;
                for (uint32_t i = 0; i < ELEMS; ++i) if (u[(size_t)pg*ELEMS+i] != f[(size_t)pg*ELEMS+i]) ++d;
                if (d) { ++badpg; if (firstpg == 0xFFFFFFFFu) firstpg = pg; } bad += d; }
            fmt::print("[D0] fused vs unfused NTT ({}) : {}/{} words differ, {}/{} pages, first bad page {}  [{}]\n",
                       fwd ? "fwd" : "inv", bad, (uint64_t)NTILE*ELEMS, badpg, NTILE,
                       firstpg == 0xFFFFFFFFu ? -1 : (int)firstpg, bad ? "FAIL" : "PASS");
            bad3 += bad; }
        if (std::getenv("TG18NTT_NTTCHK_ONLY")) return bad3 == 0;
    }

    // ---- [D3] tensor: pointwise mod Q ----
    {
        auto A = ck.encrypt(ck.encode(std::vector<cdouble>(NSLOT, cdouble(0.5, -0.25)), ck.Delta), ck.Delta);
        auto B = ck.encrypt(ck.encode(std::vector<cdouble>(NSLOT, cdouble(-0.125, 0.75)), ck.Delta), ck.Delta);
        std::vector<std::vector<uint32_t>> re0, re1, re2; ck.tensor(A, B, re0, re1, re2);
        for (uint32_t li = 0; li < NCHK; ++li) {
            B64 a0c=up_C(A.c0[li]), a1c=up_C(A.c1[li]), b0c=up_Cm(B.c0[li],li), b1c=up_Cm(B.c1[li],li);
            B64 a0=mkB64(), a1=mkB64(), b0=mkB64(), b1=mkB64();
            ntt(a0c,a0,li,true); ntt(a1c,a1,li,true); ntt(b0c,b0,li,true); ntt(b1c,b1,li,true);
            B64 E0=mkB64(), E2=mkB64(), t01=mkB64(), t10=mkB64(), E1=mkB64();
            pw(a0,b0,E0,li); pw(a1,b1,E2,li); pw(a0,b1,t01,li); pw(a1,b0,t10,li); madd(t01,t10,E1,li);
            B64 c0=mkB64(), c1=mkB64(), c2=mkB64();
            ntt(E0,c0,li,false); ntt(E1,c1,li,false); ntt(E2,c2,li,false);
            t_fin(cq);
            auto g0=rd_C(c0), g1=rd_C(c1), g2=rd_C(c2);
            for (uint32_t n=0;n<N;++n){ tot3+=3;
                if (g0[n]!=re0[li][n]) ++bad3; if (g1[n]!=re1[li][n]) ++bad3; if (g2[n]!=re2[li][n]) ++bad3; }
        }
        fmt::print("[D3] device tensor (pointwise mod Q) vs reference   mismatches = {}/{}  [{}]\n",
                   bad3, tot3, bad3?"FAIL":"PASS");
    }
    // ---- [D4] relinearise, fed the reference e2 ----
    {
        auto A = ck.encrypt(ck.encode(std::vector<cdouble>(NSLOT, cdouble(0.3, 0.1)), ck.Delta), ck.Delta);
        auto Bc = ck.encrypt(ck.encode(std::vector<cdouble>(NSLOT, cdouble(-0.2, 0.4)), ck.Delta), ck.Delta);
        std::vector<std::vector<uint32_t>> re0, re1, re2, rk0, rk1;
        ck.tensor(A, Bc, re0, re1, re2);
        ck.relin(re2, rk0, rk1);
        std::vector<B64> e2c(LQ), e2e(LQ), k0(LQ), k1(LQ);
        std::vector<std::vector<uint32_t>> fe(LQ);
        par_for(LQ, [&](uint32_t li){ fe[li] = ck.nt[li].fwd(re2[li]); });
        for (uint32_t li = 0; li < LQ; ++li) { e2c[li] = up_C(re2[li]); e2e[li] = up_E(fe[li], false, li); }
        relin_dev(e2c, e2e, k0, k1);
        std::vector<B64> kc(NCHK);
        for (uint32_t i = 0; i < NCHK; ++i) { kc[i] = mkB64(); ntt(k0[i], kc[i], i, false); }
        t_fin(cq);
        for (uint32_t i = 0; i < NCHK; ++i) { auto got = rd_C(kc[i]);
            for (uint32_t n = 0; n < N; ++n) { ++tot4; if (got[n] != rk0[i][n]) ++bad4; } }
        fmt::print("[D4] device relinearise (dnum={} hybrid key-switch) vs reference   mismatches = {}/{}  [{}]\n",
                   DNUM, bad4, tot4, bad4?"FAIL":"PASS");
    }

    // ---- [D5] rescale: drop 2 primes, centred ----
    {
        auto A = ck.encrypt(ck.encode(std::vector<cdouble>(NSLOT, cdouble(0.6, -0.3)), ck.Delta), ck.Delta);
        auto want = ck.rescale(A.c0);
        std::vector<B64> c(LQ), o(LOUT());
        for (uint32_t i = 0; i < LQ; ++i) c[i] = up_C(A.c0[i]);
        rescale_dev(c, o);
        t_fin(cq);
        for (uint32_t i = 0; i < NCHK; ++i) { auto got = rd_C(o[i]);
            for (uint32_t n = 0; n < N; ++n) { ++tot5; if (got[n] != want[i][n]) ++bad5; } }
        fmt::print("[D5] device rescale (drop {} prime(s), centred={}) vs reference   mismatches = {}/{}  [{}]\n",
                   g_ndrop, g_centre?1:0, bad5, tot5, bad5?"FAIL":"PASS");
    }

    // ======================================================================================
    // THE RESIDENT MULTIPLY — the measured unit.
    // Inputs are uploaded once, outside; the routine contains no t_wr/t_rd, so it captures as a
    // single Metal Trace segment. Coefficient I/O for now: it is what the host reference produces,
    // so [D6] compares directly.
    // ======================================================================================
    uint32_t bad6 = 0, tot6 = 0;
    if (std::getenv("TG18NTT_CHAIN") || std::getenv("TG18NTT_TRACE")) {
        double t0 = now_s();
        auto z1 = std::vector<cdouble>(NSLOT), z2 = std::vector<cdouble>(NSLOT);
        std::mt19937_64 mr(777);
        for (auto& v : z1) v = cdouble((double)(mr()%2000001)/1e6-1.0, (double)(mr()%2000001)/1e6-1.0);
        for (auto& v : z2) v = cdouble((double)(mr()%2000001)/1e6-1.0, (double)(mr()%2000001)/1e6-1.0);
        auto A = ck.encrypt(ck.encode(z1, ck.Delta), ck.Delta);
        auto Bc = ck.encrypt(ck.encode(z2, ck.Delta), ck.Delta);
        auto want = ck.multiply(A, Bc);
        fmt::print("  host reference multiply: {:.1f} s\n", now_s() - t0);

        std::map<int,B64> A0c, A1c, B0c, B1c;
        for (uint32_t li = 0; li < LQ; ++li) {
            A0c[li]=up_C(A.c0[li]); A1c[li]=up_C(A.c1[li]);
            B0c[li]=up_Cm(Bc.c0[li],li); B1c[li]=up_Cm(Bc.c1[li],li); }
        t_fin(cq);

        auto dev_mul = [&](std::vector<B64>& O0, std::vector<B64>& O1) {
            PI = 0; WI = 0; seg_idx = 0; seg_started = false; hold.clear();
            std::map<int,B64> a0e, a1e, b0e, b1e;
            L("input","input.ntt","-");
            { std::vector<NJob> nj;
              for (uint32_t li = 0; li < LQ; ++li) {
                  a0e[li]=mkB64(); a1e[li]=mkB64(); b0e[li]=mkB64(); b1e[li]=mkB64();
                  nj.push_back({A0c[li], a0e[li], li}); nj.push_back({A1c[li], a1e[li], li});
                  nj.push_back({B0c[li], b0e[li], li}); nj.push_back({B1c[li], b1e[li], li}); }
              ntt_all(nj, true); }
            std::vector<B64> E0(LQ), E1(LQ), E2(LQ), T01(LQ), T10(LQ);
            L("tensor","tensor.pointwise","-");
            { // one job list per product, each spanning every limb, so the grid fills. The five ops
              // were 5 x LQ = 240 dispatches of 64 cores; they are now 5 lists of LQ*NTILE jobs.
              std::vector<EW> p0, p2, q01, q10;
              for (uint32_t li = 0; li < LQ; ++li) {
                  E0[li]=mkB64(); E2[li]=mkB64(); T01[li]=mkB64(); T10[li]=mkB64(); E1[li]=mkB64();
                  for (uint32_t t = 0; t < NTILE; ++t) {
                      p0.push_back ({a0e[li], b0e[li], E0[li],  t, t, li});
                      p2.push_back ({a1e[li], b1e[li], E2[li],  t, t, li});
                      q01.push_back({a0e[li], b1e[li], T01[li], t, t, li});
                      q10.push_back({a1e[li], b0e[li], T10[li], t, t, li}); } }
              pw_jobs(p0); pw_jobs(p2); pw_jobs(q01); pw_jobs(q10);
              std::vector<EW> a;
              for (uint32_t li = 0; li < LQ; ++li)
                  for (uint32_t t = 0; t < NTILE; ++t) a.push_back({T01[li], T10[li], E1[li], t, t, li});
              madd_jobs(a); }
            std::vector<B64> e2c(LQ), k0(LQ), k1(LQ);
            L("keyswitch","ks.to_coeff","e2");
            { std::vector<NJob> nj;
              for (uint32_t li = 0; li < LQ; ++li) { e2c[li] = mkB64(); nj.push_back({E2[li], e2c[li], li}); }
              ntt_all(nj, false); }
            relin_dev(e2c, E2, k0, k1);
            std::vector<B64> C0(LQ), C1(LQ);
            L("combine","combine.add","-");
            std::vector<B64> R0(LQ), R1(LQ);
            { std::vector<EW> a, b;
              for (uint32_t i = 0; i < LQ; ++i) { R0[i]=mkB64(); R1[i]=mkB64();
                  for (uint32_t t = 0; t < NTILE; ++t) {
                      a.push_back({E0[i], k0[i], R0[i], t, t, i});
                      b.push_back({E1[i], k1[i], R1[i], t, t, i}); } }
              madd_jobs(a); madd_jobs(b); }
            L("output","output.intt","-");
            { std::vector<NJob> nj;
              for (uint32_t i = 0; i < LQ; ++i) { C0[i]=mkB64(); C1[i]=mkB64();
                  nj.push_back({R0[i], C0[i], i}); nj.push_back({R1[i], C1[i], i}); }
              ntt_all(nj, false); }
            L("rescale","rescale.drop","r0"); rescale_dev(C0, O0);
            L("rescale","rescale.drop","r1"); rescale_dev(C1, O1); };

        auto check = [&](std::vector<B64>& O0, std::vector<B64>& O1) {
            Ciphertext got; got.scale = want.scale;
            got.c0.resize(LOUT()); got.c1.resize(LOUT());
            uint32_t b = 0;
            for (uint32_t i = 0; i < LOUT(); ++i) {
                got.c0[i] = rd_C(O0[i]); got.c1[i] = rd_C(O1[i]);
                for (uint32_t n = 0; n < N; ++n) { tot6 += 2;
                    if (got.c0[i][n] != want.c0[i][n]) ++b;
                    if (got.c1[i][n] != want.c1[i][n]) ++b; } }
            bad6 = b;
            fmt::print("[D6] WHOLE MULTIPLY on device vs the host reference   mismatches = {}/{}  [{}]\n",
                       b, tot6, b ? "FAIL" : "PASS");
            std::vector<cdouble> tgt(NSLOT);
            for (uint32_t j2 = 0; j2 < NSLOT; ++j2) tgt[j2] = z1[j2] * z2[j2];
            auto e = slot_err(ck.decrypt(got), tgt);
            fmt::print("[D7] decode(decrypt(device out)) vs z1(.)z2   max={:.3e}  rms={:.3e}  ~{:.1f} bits  [{}]\n",
                       e.maxe, e.rms, e.bits(), e.maxe < 1e-2 ? "PASS" : "FAIL");
            if (e.maxe >= 1e-2) bad6 += 1; };

        if (std::getenv("TG18NTT_TRACE")) {
            // TG18NTT_DRAIN_ONLY: REC, then the drained capture IMMEDIATELY -- no CAP, no replay, no
            // timing loops. Those phases each leave markers in the device buffer, and the read+remove
            // before the drain does not reliably clear them, so their residue shows up as extra
            // dispatch groups and breaks the 1-1 map. This mode makes the CSV hold exactly one multiply.
            const bool DRAIN_ONLY = std::getenv("TG18NTT_DRAIN_ONLY") != nullptr;
            CACHE_ON = true;
            std::vector<B64> O0(LOUT()), O1(LOUT());
            CACHE_REC = true; TR_PHASE = 0;
            fmt::print("REC: warm+cache workloads (G={})...\n", NC);
            t0 = now_s(); dev_mul(O0, O1); t_fin(cq); CACHE_REC = false;
            fmt::print("  REC took {:.1f} s, {} workloads cached\n", now_s() - t0, WC.size());
            if (DRAIN_ONLY) {
                TR_PHASE = 0; WI = 0; PI = 0;
                t_fin(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);
                std::string csv = std::string(std::getenv("TT_METAL_HOME")) + "/generated/profiler/.logs/profile_log_device.csv";
                std::remove(csv.c_str());
                int K = 40; if (const char* e = std::getenv("TG18NTT_DRAIN_K")) { int v = std::atoi(e); if (v > 0) K = v; }
                prof::drain_ctr = 0; prof::drain_every = K;
                prof::drain_fn = [&]{ distributed::Finish(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh); };
                fmt::print("DRAIN-ONLY capture of ONE cached ct_mul (drain every {} dispatches)...\n", K);
                const long e0 = prof::n_enq;
                dev_mul(O0, O1);
                const long ndisp = prof::n_enq - e0;
                prof::drain_every = 0; prof::drain_fn = nullptr;
                distributed::Finish(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);
                fmt::print("TG18NTT_DRAIN_ONLY: dispatches in this capture = {}\n", ndisp);
                fmt::print("TG18NTT_DRAIN_ONLY: -> tools/traces/busy_metrics.py <csv> --expect {} --dataflow-ok\n", ndisp);
                dump_labels();
                check(O0, O1);
                prof::seg_hook = nullptr;
                bool okd = (bad3==0 && bad4==0 && bad5==0 && bad6==0);
                okd &= mesh->close();
                return okd;
            }
            TR_PHASE = 1; fmt::print("CAP: recording traces...\n");
            dev_mul(O0, O1);
            if (seg_open) { mesh->end_mesh_trace(0, SEG.back()); seg_open = false; }
            t_fin(cq);
            fmt::print("captured {} trace segment(s)\n", SEG.size());
            TR_PHASE = 2;
            dev_mul(O0, O1); t_fin(cq);
            check(O0, O1);
            int it = 3; if (const char* e = std::getenv("TG18NTT_ITERS")) { int v = std::atoi(e); if (v > 0) it = v; }
            dev_mul(O0, O1); t_fin(cq); prof::reset();
            auto tw = prof::hrc::now();
            for (int i = 0; i < it; ++i) dev_mul(O0, O1);
            t_fin(cq);
            double wall = prof::msd(tw) / it;
            MEASURE_DEV = true; prof::g_replay = 0; prof::n_replay = 0;
            for (int i = 0; i < it; ++i) dev_mul(O0, O1);
            t_fin(cq); MEASURE_DEV = false;
            fmt::print("TIMING: TRACED RESIDENT CKKS ct_mul  (L_Q={}, dnum={}, N={})\n", LQ, DNUM, N);
            fmt::print("  host wall            : {:.3f} ms/op   (G={}, {} iters)\n", wall, NC, it);
            fmt::print("  host device busy     : {:.3f} ms/op   ({:.1f} segment replays)\n",
                       prof::g_replay / it, (double)prof::n_replay / it);
            if (std::getenv("TG18NTT_DRAIN")) {
                // A traced replay is ONE atomic device operation, so the profiler buffer cannot be
                // emptied part-way through it -- at 6020 dispatches x 5 RISCs it overflows and
                // tt-metal drops markers, which breaks the 1-1 map. The complete marker set therefore
                // has to come from the CACHED (untraced) path, drained every K dispatches. Its wall
                // time is inflated by the drains and is not used; only the trace-derived pair is.
                TR_PHASE = 0; WI = 0; PI = 0;
                t_fin(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);
                std::string csv = std::string(std::getenv("TT_METAL_HOME")) + "/generated/profiler/.logs/profile_log_device.csv";
                std::remove(csv.c_str());
                int K = 200; if (const char* e = std::getenv("TG18NTT_DRAIN_K")) { int v = std::atoi(e); if (v > 0) K = v; }
                prof::drain_ctr = 0; prof::drain_every = K;
                prof::drain_fn = [&]{ distributed::Finish(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh); };
                fmt::print("DRAINED capture of ONE cached ct_mul (drain every {} dispatches)...\n", K);
                const long e0 = prof::n_enq;
                dev_mul(O0, O1);
                const long ndisp = prof::n_enq - e0;
                prof::drain_every = 0; distributed::Finish(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);
                fmt::print("TG18NTT_DRAIN: host dispatches in this capture = {}\n", ndisp);
                fmt::print("TG18NTT_DRAIN: 1-1 check -> tools/traces/busy_metrics.py <csv> --expect {} --wall-ms {:.3f} --host-busy-ms {:.3f}\n",
                           ndisp, wall, prof::g_replay / it);
                dump_labels();
            } else if (std::getenv("TG18NTT_PROF")) {
                t_fin(cq); tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);
                std::string csv = std::string(std::getenv("TT_METAL_HOME")) + "/generated/profiler/.logs/profile_log_device.csv";
                std::remove(csv.c_str());
                const auto tw0 = prof::hrc::now();
                MEASURE_DEV = true; prof::g_replay = 0; prof::n_replay = 0;
                dev_mul(O0, O1);
                MEASURE_DEV = false; t_fin(cq);
                const double pwall = prof::msd(tw0); const size_t pdisp = WI;
                tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);
                fmt::print("TG18NTT_PROF: this replay  wall = {:.3f} ms   dispatches = {}\n", pwall, pdisp);
                fmt::print("TG18NTT_PROF: 1-1 check -> tools/traces/busy_metrics.py <csv> --expect {} --wall-ms {:.3f} --host-busy-ms {:.3f}\n",
                           pdisp, pwall, prof::g_replay);
                dump_labels();
            }
        } else {
            const long enq0 = prof::n_enq; t0 = now_s();
            std::vector<B64> O0(LOUT()), O1(LOUT());
            dev_mul(O0, O1);
            t_fin(cq);
            fmt::print("  chain: {} program dispatches, {:.1f} s wall (G={}, uncached)\n",
                       prof::n_enq - enq0, now_s() - t0, NC);
            check(O0, O1);
        }
    }

    bool ok = (bad3 == 0 && bad4 == 0 && bad5 == 0 && bad6 == 0);
    prof::seg_hook = nullptr;
    ok &= mesh->close();
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const uint64_t seed = (argc > 1) ? std::stoull(argv[1]) : 0;
    if (const char* e = std::getenv("TG18NTT_NDROP")) { int v = std::atoi(e); if (v >= 1 && v <= 3) g_ndrop = (uint32_t)v; }
    if (std::getenv("TG18NTT_NOCENTRE")) g_centre = false;

    std::printf("exp_Tg018  CKKS ct_mul   N=%u  slots=%u  L_Q=%u  P=%u  QP=%u  dnum=%u (alpha=%u)  threads=%u\n",
                N, NSLOT, LQ, LP, LQP, DNUM, ALPHA, nthreads());
    double t0 = now_s();
    CKKS ck(seed);
    std::printf("  setup %.1f s   log2(Q)=%d  log2(QP)~%d   Delta=%llu=2^%.2f   level %u -> %u   rescale %s\n",
                now_s() - t0, bitlen(ck.Qprod), bitlen(ck.Qprod) + bitlen(ck.Pprod),
                (unsigned long long)ck.DeltaI, std::log2((double)ck.DeltaI), LQ, LOUT(),
                g_centre ? "CENTRED" : "non-centred");

    bool pass = true;
    std::mt19937_64 mr(seed + 12345);
    auto rand_slots = [&]() {
        std::vector<cdouble> z(NSLOT);
        for (auto& v : z) v = cdouble((double)(mr() % 2000001) / 1e6 - 1.0, (double)(mr() % 2000001) / 1e6 - 1.0);
        return z; };
    auto z1 = rand_slots(), z2 = rand_slots();

    { double t = now_s();
      auto a = ck.emb.inverse(z1);
      auto back = ck.emb.forward(a);
      auto e = slot_err(back, z1);
      std::printf("[0] canonical embedding round trip      max = %.3e  [%s]   (%.1f s)\n",
                  e.maxe, e.maxe < 1e-9 ? "PASS" : "FAIL", now_s() - t);
      pass &= (e.maxe < 1e-9); }

    { double t = now_s();
      auto ct = ck.encrypt(ck.encode(z1, ck.Delta), ck.Delta);
      auto e = slot_err(ck.decrypt(ct), z1);
      std::printf("[1] decrypt(encrypt(z))                 max = %.3e  rms = %.3e  ~%.1f bits  [%s]   (%.1f s)\n",
                  e.maxe, e.rms, e.bits(), e.maxe < 1e-5 ? "PASS" : "FAIL", now_s() - t);
      pass &= (e.maxe < 1e-5); }

    { double t = now_s();
      auto a = ck.encrypt(ck.encode(z1, ck.Delta), ck.Delta);
      auto b = ck.encrypt(ck.encode(z2, ck.Delta), ck.Delta);
      std::vector<cdouble> want(NSLOT);
      for (uint32_t j = 0; j < NSLOT; ++j) want[j] = z1[j] + z2[j];
      auto e = slot_err(ck.decrypt(ck.add_ct(a, b)), want);
      std::printf("[2] decrypt(enc(z1) + enc(z2))          max = %.3e  rms = %.3e  ~%.1f bits  [%s]   (%.1f s)\n",
                  e.maxe, e.rms, e.bits(), e.maxe < 1e-5 ? "PASS" : "FAIL", now_s() - t);
      pass &= (e.maxe < 1e-5); }

    // THE TEST THAT MATTERS
    { double t = now_s();
      auto a = ck.encrypt(ck.encode(z1, ck.Delta), ck.Delta);
      auto b = ck.encrypt(ck.encode(z2, ck.Delta), ck.Delta);
      auto out = ck.multiply(a, b);
      auto got = ck.decrypt(out);
      std::vector<cdouble> want(NSLOT);
      for (uint32_t j = 0; j < NSLOT; ++j) want[j] = z1[j] * z2[j];
      auto e = slot_err(got, want);
      std::printf("[3] decrypt(enc(z1) (x) enc(z2))        max = %.3e  rms = %.3e  ~%.1f bits  [%s]   (%.1f s)\n",
                  e.maxe, e.rms, e.bits(), e.maxe < 1e-2 ? "PASS" : "FAIL", now_s() - t);
      std::printf("      level %u -> %u limbs, scale %.0f -> %.0f\n",
                  a.L(), out.L(), (double)a.scale, (double)out.scale);
      std::printf("      slot 0: want (%+.6f,%+.6f)  got (%+.6f,%+.6f)\n",
                  want[0].real(), want[0].imag(), got[0].real(), got[0].imag());
      pass &= (e.maxe < 1e-2); }

    if (std::getenv("TG18NTT_DEV")) pass &= device_stage(ck);

    std::printf("\n%s\n", pass ? "Test Passed" : "Test FAILED");
    return pass ? 0 : 1;
}
