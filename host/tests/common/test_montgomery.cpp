// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// test_montgomery.cpp
//
// Tests for mont::MontgomeryContext (kernel is 32-bit-only; q < 2^28).
//   1. mul_hi_lo builds the 32x32->64 product correctly
//   2. REDC(z) ≡ z * R^-1 (mod q)         (vs a 64-bit host oracle)
//   3. result lies in [0, 2q)
//   4. Montgomery multiply matches schoolbook (a*b) mod q
//   5. to/from Montgomery roundtrip, and boundary values
//
// The kernel under test never uses a 64-bit variable. 64-bit appears only in
// the host ORACLE below, whose sole job is to be an independent ground truth.
//
// Build:  g++ -O2 -std=c++17 -o test_montgomery test_montgomery.cpp && ./test_montgomery

#include "montgomery.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::printf("  FAIL: %s\n", what);
        ++g_failures;
    }
}

// ---- host oracle (64-bit allowed: ground truth only, not the kernel) ----
uint64_t redc_oracle(const mont::MontgomeryContext& ctx, uint32_t z_high, uint32_t z_low) {
    uint64_t z = (static_cast<uint64_t>(z_high) << 32) | z_low;
    uint64_t r_inv = mont::modular_inverse(1ull << 32, ctx.prime());
    return ((z % ctx.prime()) * (r_inv % ctx.prime())) % ctx.prime();
}

void run_for_prime(uint32_t prime, std::mt19937_64& rng) {
    std::printf("prime = %u  (%.1f bits)\n", prime, /*log2*/ 32.0 - __builtin_clz(prime));
    mont::MontgomeryContext ctx(prime);

    std::uniform_int_distribution<uint32_t> dist_high(0, prime - 1);   // z_high in [0, q)
    std::uniform_int_distribution<uint32_t> dist_u32(0, 0xFFFFFFFFu);  // z_low full range
    std::uniform_int_distribution<uint32_t> dist_a(0, prime - 1);

    const int N = 200000;

    // 1: mul_hi_lo against a 64-bit oracle product.
    for (int i = 0; i < N; ++i) {
        uint32_t a = dist_a(rng), b = dist_a(rng);
        uint32_t hi, lo;
        mont::mul_hi_lo(a, b, hi, lo);
        uint64_t want = static_cast<uint64_t>(a) * b;
        check(((static_cast<uint64_t>(hi) << 32) | lo) == want, "mul_hi_lo wrong");
    }

    // 2 + 3: reduction value correct mod q, and in [0, 2q).
    for (int i = 0; i < N; ++i) {
        uint32_t z_high = dist_high(rng);
        uint32_t z_low  = dist_u32(rng);
        uint32_t out = ctx.reduce_optimized(z_high, z_low);
        check(out < 2 * prime, "result not in [0, 2q)");
        check(ctx.finalize(out) == redc_oracle(ctx, z_high, z_low), "REDC value wrong mod q");
    }

    // 4 + 5: modular multiply (lazy operands up to 2q) and roundtrip.
    std::uniform_int_distribution<uint32_t> coin(0, 1);
    for (int i = 0; i < N; ++i) {
        uint32_t a = dist_a(rng), b = dist_a(rng);
        uint32_t am = ctx.to_montgomery(a);          // in [0, q)
        uint32_t bm = ctx.to_montgomery(b);

        check(ctx.from_montgomery(am) == a, "roundtrip a failed");

        // Make the operands lazy: am, am+q both ≡ a*R (mod q), and lie in [0, 2q).
        uint32_t al = am + (coin(rng) ? prime : 0);
        uint32_t bl = bm + (coin(rng) ? prime : 0);

        uint32_t prod = ctx.from_montgomery(ctx.mulmod(al, bl));
        uint32_t want = static_cast<uint32_t>((static_cast<uint64_t>(a) * b) % prime);
        check(prod == want, "modular multiply wrong (lazy operands)");
    }

    // boundaries.
    std::vector<std::pair<uint32_t, uint32_t>> edges = {
        {0, 0}, {0, 1}, {0, 0xFFFFFFFFu}, {prime - 1, 0xFFFFFFFFu}, {prime - 1, 0}};
    for (auto [zh, zl] : edges) {
        uint32_t out = ctx.reduce_optimized(zh, zl);
        check(out < 2 * prime, "edge result not in [0, 2q)");
        check(ctx.finalize(out) == redc_oracle(ctx, zh, zl), "edge REDC wrong");
    }

    std::printf("  ran %d reductions + %d multiplies + %d mul_hi_lo\n", N, N, N);
}

}  // namespace

int main() {
    std::mt19937_64 rng(0xC0FFEE);

    // Odd primes below 2^28 (268435456), including NTT-friendly ones.
    std::vector<uint32_t> primes = {
        3, 17, 257, 65537,
        167772161u,   // 5 * 2^25 + 1
        201326593u,   // 3 * 2^26 + 1
        251658241u,   // 15 * 2^24 + 1  (largest of these, ~27.9 bits)
    };
    for (uint32_t p : primes) run_for_prime(p, rng);

    if (g_failures == 0) {
        std::printf("\nALL TESTS PASSED\n");
        return 0;
    }
    std::printf("\n%d FAILURE(S)\n", g_failures);
    return 1;
}
