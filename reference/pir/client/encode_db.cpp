// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Offline encoder: text database -> device-ready PIR database file.
//
// The point of this program is that a demo run should do NO cryptographic preprocessing: it opens the
// file and uploads tiles.  So everything expensive and deterministic happens here, once, in parallel:
//
//     ASCII line -> 2048 plaintext coefficients (mod t)   [one character per coefficient]
//                -> forward negacyclic NTT, per RNS channel
//                -> Montgomery domain
//                -> 32x32 face tilization (2 tiles per polynomial)
//                -> file
//
// which is exactly what `Engine::scan_levelB` expects to find in `bDB[ch]` (cell c at tiles 2c).
//
// Conventions are pinned to the device so the output is bit-identical to
// `to_mont(ch, fwd2048(CH[ch], cell))`; `--verify` checks a sample against a slow independent path.
//   * primitive root: smallest g with g^((q-1)/p) != 1 for every prime p | q-1  (matches prim_root)
//   * psi = g^((q-1)/2n), w = psi^2;  X[k] = sum_n x[n] * psi^n * w^(n k) mod q   (negacyclic)
//   * Montgomery R = 2^32  (cpu/montgomery.hpp)
//
// Rows are independent, so this is embarrassingly parallel over rows — the reason it is C++ and not
// Python.
//
// --fold additionally emits the int8-folded SLOT-MAJOR database that the benchmarked matmul scan
// reads (`test_scan_ntt2048`/`test_scan_kt` — the path behind the OnionPIR comparison).  Two things
// differ from the .pir output and both are silent if got wrong:
//   * the matmul scan consumes PLAIN NTT values — NOT Montgomery.  (`pmul` needs one operand in
//     Montgomery form; `matmul_modq` is a plain modular matmul and does not.)
//   * the layout is [m-tile][channel][k-tile][slot][16 int8 tiles], i.e. grouped by m-tile, because
//     the scan produces 32 output rows at a time.  Grouping this way lets the demo either hold all
//     6.0 GB resident or stream one 1.5 GB m-tile at a time, with no kernel change.
//
//   g++ -O3 -march=native -pthread -o encode_db encode_db.cpp
//   ./encode_db db65536.txt db65536.pir [--threads N] [--verify] [--fold db65536.fold --rest 128]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <random>
#include <thread>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;

namespace {

constexpr u32 N = 2048;            // ring degree = characters per row
constexpr u32 T_PLAIN = 257;       // plaintext modulus: one ASCII char per coefficient
constexpr u32 ELEMS = 1024;        // coefficients per 32x32 tile
constexpr u32 NTILE = N / ELEMS;   // 2 tiles per polynomial
// The RNS primes.  These MUST be onionpir_ref.ntt_primes(3, 2048) — the three 28-bit NTT-friendly
// primes just below 2^28, descending — because the client's keys, query and secret are generated
// against exactly that modulus.  An encoder built on a different (equally valid) triple produces a
// database that passes every self-check and is still unusable: only the channels whose prime happens
// to coincide decrypt, and the rest come back as noise.  That is how this file was wrong once.
const u32 PRIMES[3] = {268369921u, 268361729u, 268271617u};
constexpr u32 K = 3;
constexpr u32 TT_ = 32, NB = 4;                     // tile side, radix-128 limbs
const u64 POW[4] = {1, 128, 16384, 2097152};        // 128^r, the fold's row scaling
constexpr u32 FOLD_MAGIC = 0x50464C44u;             // 'PFLD'

constexpr u32 MAGIC = 0x50444231u;   // 'PDB1'

u64 modpow(u64 b, u64 e, u64 m) { u64 r = 1; b %= m;
    while (e) { if (e & 1) r = (u128)r * b % m; b = (u128)b * b % m; e >>= 1; } return r; }

// smallest primitive root — same search order as pir2048::prim_root, so the roots match the device
u64 prim_root(u64 q) {
    u64 phi = q - 1, n = phi; std::vector<u64> f;
    for (u64 d = 2; d * d <= n; ++d) if (n % d == 0) { f.push_back(d); while (n % d == 0) n /= d; }
    if (n > 1) f.push_back(n);
    for (u64 g = 2; g < q; ++g) {
        bool ok = true;
        for (u64 p : f) if (modpow(g, phi / p, q) == 1) { ok = false; break; }
        if (ok) return g;
    }
    return 0;
}

// per-channel precomputation: bit-reversed twiddles for an iterative negacyclic NTT
struct Chan {
    u32 q;
    u64 rmod;                     // R mod q; the Montgomery lift is x -> x*R mod q
    std::vector<u32> psi_pow;     // psi^i, i < N   (the negacyclic pre-twist)
    std::vector<u32> w_pow;       // w^i,   i < N/2 (butterfly twiddles, natural order)
};

Chan make_chan(u32 q) {
    Chan c; c.q = q;
    // montgomery.hpp: to_montgomery(a) = (a << 32) % q, i.e. a*R with R = 2^32 — NOT a*R^2.
    c.rmod = ((u64)1 << 32) % q;
    const u64 g = prim_root(q);
    const u64 psi = modpow(g, (q - 1) / (2 * N), q);
    const u64 w = (u128)psi * psi % q;
    c.psi_pow.resize(N); c.w_pow.resize(N / 2);
    u64 p = 1; for (u32 i = 0; i < N; ++i) { c.psi_pow[i] = (u32)p; p = (u128)p * psi % q; }
    u64 x = 1; for (u32 i = 0; i < N / 2; ++i) { c.w_pow[i] = (u32)x; x = (u128)x * w % q; }
    return c;
}

// X[k] = sum_n x[n] * psi^n * w^(n k) mod q, computed as an O(n log n) DIT NTT.
// Pre-twist by psi^n, then a standard cyclic NTT; output ends up in natural order.
void ntt_negacyclic(const Chan& c, const u32* x, u32* out) {
    const u32 q = c.q;
    // pre-twist + bit-reverse permutation
    for (u32 i = 0, j = 0; i < N; ++i) {
        out[j] = (u32)((u64)x[i] * c.psi_pow[i] % q);
        u32 bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
    }
    for (u32 len = 1; len < N; len <<= 1) {
        const u32 step = (N / 2) / len;             // twiddle stride for this stage
        for (u32 i = 0; i < N; i += (len << 1)) {
            for (u32 k = 0; k < len; ++k) {
                const u64 tw = c.w_pow[(u64)k * step];
                const u32 u = out[i + k];
                const u32 v = (u32)((u64)out[i + k + len] * tw % q);
                out[i + k]       = u + v >= q ? u + v - q : u + v;
                out[i + k + len] = u >= v ? u - v : u + q - v;
            }
        }
    }
}

// 32x32 face layout, matching pir::tilize_poly / the kernels' faceidx
inline u32 faceidx(u32 r, u32 col) { return ((r >> 4) * 2 + (col >> 4)) * 256 + (r & 15) * 16 + (col & 15); }

inline int8_t limb(u32 v, u32 p) { return (int8_t)((v >> (7 * p)) & 127); }

// int8 tilize of one 1024-element tile (same face layout as tilize2)
void tilize1_i8(const int8_t* nat, int8_t* out) {
    for (u32 r = 0; r < TT_; ++r)
        for (u32 c = 0; c < TT_; ++c) out[faceidx(r, c)] = nat[r * TT_ + c];
}

void tilize2(const u32* nat, u32* out) {
    for (u32 t = 0; t < NTILE; ++t)
        for (u32 r = 0; r < 32; ++r)
            for (u32 col = 0; col < 32; ++col)
                out[t * ELEMS + faceidx(r, col)] = nat[t * ELEMS + r * 32 + col];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <db.txt> <db.pir> [--threads N] [--verify]\n", argv[0]); return 2; }
    const char* in_path = argv[1]; const char* out_path = argv[2];
    unsigned nthreads = std::thread::hardware_concurrency(); if (!nthreads) nthreads = 4;
    bool verify = false; const char* fold_path = nullptr; u32 rest = 128;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) nthreads = (unsigned)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--verify")) verify = true;
        else if (!std::strcmp(argv[i], "--fold") && i + 1 < argc) fold_path = argv[++i];
        else if (!std::strcmp(argv[i], "--rest") && i + 1 < argc) rest = (u32)std::atoi(argv[++i]);
    }

    auto t_start = std::chrono::steady_clock::now();

    // ---- read the text database ----
    std::vector<std::string> rows;
    { std::ifstream f(in_path);
      if (!f) { std::fprintf(stderr, "cannot open %s\n", in_path); return 2; }
      std::string line; while (std::getline(f, line)) rows.push_back(line); }
    const u32 n_rows = (u32)rows.size();
    if (!n_rows) { std::fprintf(stderr, "empty database\n"); return 2; }
    auto t_read = std::chrono::steady_clock::now();

    std::vector<Chan> ch(K);
    for (u32 j = 0; j < K; ++j) ch[j] = make_chan(PRIMES[j]);

    // ---- output buffer: [channel][row][2 tiles]  (cell c at tiles 2c, as bDB[ch] expects) ----
    const size_t per_ch = (size_t)n_rows * N;
    std::vector<u32> outbuf((size_t)K * per_ch);
    std::vector<u32> plain;                       // plain NTT, kept only when --fold is requested
    if (fold_path) plain.resize((size_t)K * per_ch);

    std::atomic<u32> next{0};
    const u32 CHUNK = 64;
    auto worker = [&] {
        std::vector<u32> coeff(N), evald(N), tiled(N);
        for (;;) {
            const u32 base = next.fetch_add(CHUNK);
            if (base >= n_rows) break;
            const u32 end = std::min(base + CHUNK, n_rows);
            for (u32 r = base; r < end; ++r) {
                // ASCII -> coefficients mod t; one character per coefficient, zero padded
                const std::string& s = rows[r];
                const size_t len = std::min<size_t>(s.size(), N);
                for (size_t i = 0; i < len; ++i) coeff[i] = (u32)((unsigned char)s[i]) % T_PLAIN;
                std::fill(coeff.begin() + len, coeff.end(), 0u);
                for (u32 j = 0; j < K; ++j) {
                    ntt_negacyclic(ch[j], coeff.data(), evald.data());
                    if (!plain.empty())                                 // PLAIN NTT for the matmul fold
                        std::memcpy(&plain[(size_t)j * per_ch + (size_t)r * N], evald.data(), N * sizeof(u32));
                    std::vector<u32> mont(evald);
                    for (u32 i = 0; i < N; ++i)                         // Montgomery lift: x -> x*R
                        mont[i] = (u32)((u128)mont[i] * ch[j].rmod % ch[j].q);
                    tilize2(mont.data(), tiled.data());
                    std::memcpy(&outbuf[(size_t)j * per_ch + (size_t)r * N], tiled.data(), N * sizeof(u32));
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (unsigned i = 0; i < nthreads; ++i) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    auto t_enc = std::chrono::steady_clock::now();

    // ---- write: header then [channel][row][2 tiles] ----
    { std::ofstream f(out_path, std::ios::binary);
      if (!f) { std::fprintf(stderr, "cannot write %s\n", out_path); return 2; }
      u32 hdr[8] = {MAGIC, N, K, T_PLAIN, n_rows, PRIMES[0], PRIMES[1], PRIMES[2]};
      f.write((const char*)hdr, sizeof(hdr));
      f.write((const char*)outbuf.data(), (std::streamsize)(outbuf.size() * sizeof(u32))); }
    auto t_write = std::chrono::steady_clock::now();

    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const double MB = outbuf.size() * sizeof(u32) / 1048576.0;
    std::printf("encoded %u rows -> %s\n", n_rows, out_path);
    std::printf("  read  %7.1f ms | encode %7.1f ms (%u threads) | write %7.1f ms | total %7.1f ms\n",
                ms(t_start, t_read), ms(t_read, t_enc), nthreads, ms(t_enc, t_write), ms(t_start, t_write));
    std::printf("  %.1f MB  (%u rows x %u channels x %u coeffs x 4 B), %.0f rows/s encode\n",
                MB, n_rows, K, N, n_rows / (ms(t_read, t_enc) / 1000.0));

    // ---- optional: int8-folded slot-major DB for the benchmarked matmul scan ----
    if (fold_path) {
        const u32 N0 = n_rows / rest;                 // first hypercube dimension
        if (N0 * rest != n_rows) { std::fprintf(stderr, "rows must equal N0*rest\n"); return 2; }
        const u32 KB = (N0 + TT_ - 1) / TT_, MB = (rest + TT_ - 1) / TT_;
        const size_t per_mt = (size_t)K * KB * N * NB * NB * ELEMS;      // int8 per m-tile block
        std::printf("fold: N0=%u rest=%u -> KB=%u k-tiles, MB=%u m-tiles, %.2f GB total (%.2f GB/m-tile)\n",
                    N0, rest, KB, MB, (double)MB * per_mt / (1ull << 30), (double)per_mt / (1ull << 30));

        std::ofstream ff(fold_path, std::ios::binary);
        if (!ff) { std::fprintf(stderr, "cannot write %s\n", fold_path); return 2; }
        u32 fh[8] = {FOLD_MAGIC, N, K, n_rows, N0, rest, KB, MB};
        ff.write((const char*)fh, sizeof(fh));

        auto t_f0 = std::chrono::steady_clock::now();
        std::vector<int8_t> blk(per_mt);
        for (u32 mt = 0; mt < MB; ++mt) {
            std::atomic<u32> nxt{0};
            auto fworker = [&] {
                std::vector<u32> A(ELEMS); std::vector<int8_t> l(ELEMS), ti(ELEMS);
                for (;;) {
                    const u32 idx = nxt.fetch_add(1);
                    if (idx >= K * KB) break;
                    const u32 c = idx / KB, kt = idx % KB;
                    const u32 q = ch[c].q;
                    for (u32 j = 0; j < N; ++j) {
                        // A[r_local][lk] = DB_ntt[(kt*32+lk)*rest + (mt*32+r_local)][slot j]
                        std::fill(A.begin(), A.end(), 0u);
                        for (u32 lk = 0; lk < TT_; ++lk) {
                            const u32 kk = kt * TT_ + lk; if (kk >= N0) continue;
                            for (u32 rl = 0; rl < TT_; ++rl) {
                                const u32 r = mt * TT_ + rl; if (r >= rest) continue;
                                A[rl * TT_ + lk] = plain[(size_t)c * per_ch + (size_t)(kk * rest + r) * N + j];
                            }
                        }
                        for (u32 pp = 0; pp < NB; ++pp) for (u32 rr = 0; rr < NB; ++rr) {
                            for (u32 i = 0; i < ELEMS; ++i) l[i] = limb((u32)((POW[rr] * A[i]) % q), pp);
                            tilize1_i8(l.data(), ti.data());
                            std::memcpy(&blk[(((size_t)c * KB + kt) * N + j) * NB * NB * ELEMS
                                             + ((size_t)pp * NB + rr) * ELEMS], ti.data(), ELEMS);
                        }
                    }
                }
            };
            std::vector<std::thread> fp;
            for (unsigned i = 0; i < nthreads; ++i) fp.emplace_back(fworker);
            for (auto& th : fp) th.join();
            ff.write((const char*)blk.data(), (std::streamsize)per_mt);
            std::printf("  m-tile %u/%u written\n", mt + 1, MB);
        }
        std::printf("  fold total %.1f s\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_f0).count());
    }

    if (verify) {
        // independent O(n^2) evaluation of X[k] = sum_n x[n] psi^n w^(nk) on a few rows/points
        std::printf("  verify: slow direct NTT on sampled coefficients... ");
        std::mt19937 rng(1); bool ok = true;
        for (u32 trial = 0; trial < 8 && ok; ++trial) {
            const u32 r = rng() % n_rows, j = rng() % K, k = rng() % N;
            const std::string& s = rows[r];
            const u32 q = ch[j].q;
            const u64 psi = modpow(prim_root(q), (q - 1) / (2 * N), q);
            const u64 w = (u128)psi * psi % q;
            u64 acc = 0;
            for (u32 i = 0; i < N; ++i) {
                const u64 xi = i < s.size() ? (u64)((unsigned char)s[i]) % T_PLAIN : 0;
                if (!xi) continue;
                acc = (acc + (u128)xi * modpow(psi, i, q) % q * modpow(w, (u64)i * k % (q - 1), q)) % q;
            }
            const u64 want = (u128)acc * ch[j].rmod % q;                  // Montgomery lift
            const u32 got = outbuf[(size_t)j * per_ch + (size_t)r * N + faceidx((k % ELEMS) / 32, k % 32)
                                   + (k / ELEMS) * ELEMS];
            if (got != want) { std::printf("MISMATCH row=%u ch=%u k=%u got=%u want=%llu\n",
                                           r, j, k, got, (unsigned long long)want); ok = false; }
        }
        if (ok) std::printf("ok\n");
        else return 1;
    }
    return 0;
}
