// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// montgomery.hpp
//
// C++ port of `montgomery_reduce_u64_to_u32_optimized_v1` / JaxMontgomeryContext.
// Computes Montgomery reduction
//
//     REDC(z) = z * R^-1 (mod q),    R = 2^32,    q (prime) < 2^28
//
// The DEVICE computation uses only 32-bit variables: the reduction never forms
// a 64-bit value, and the 32x32->64 product feeding it is built from 16x16->32
// limbs (the q < 2^28 cap keeps every partial-sum carry inside 32 bits).
//
// 64-bit math appears ONLY in host-side setup (modular_inverse / format
// conversion), the analogue of Python's arbitrary-precision context. These run
// once to build the constants; they are not the per-element device kernel.
//
// Output is "lazy": in [0, 2q), i.e. the final conditional subtraction is NOT
// performed (matching the JAX kernel, which returns z_high + q - t_final).

#pragma once

#include <cstdint>
#include <stdexcept>

namespace mont {

// q < 2^PRIME_BITS. The 28-bit cap guarantees the limb-wise 32x32->64 multiply
// below never overflows a 32-bit accumulator.
constexpr uint32_t PRIME_BITS = 28;

// ---------------------------------------------------------------------------
// Host-side number theory (used only to build the context constants).
// ---------------------------------------------------------------------------

inline int64_t egcd(int64_t a, int64_t b, int64_t& x, int64_t& y) {
    if (b == 0) { x = 1; y = 0; return a; }
    int64_t x1, y1;
    int64_t g = egcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return g;
}

inline uint64_t modular_inverse(uint64_t a, uint64_t m) {
    int64_t x, y;
    int64_t g = egcd(static_cast<int64_t>(a % m), static_cast<int64_t>(m), x, y);
    if (g != 1) throw std::invalid_argument("modular_inverse: a not invertible mod m");
    int64_t r = x % static_cast<int64_t>(m);
    if (r < 0) r += static_cast<int64_t>(m);
    return static_cast<uint64_t>(r);
}

// ---------------------------------------------------------------------------
// 32-bit-only 32x32 -> 64 multiply, result returned as (hi, lo) 32-bit halves.
//
// Valid for a, b < 2^30 (so it covers lazy operands up to 2q < 2^29):
//   a = a_h*2^16 + a_l with a_h < 2^14, likewise b.
//   cross = a_l*b_h + a_h*b_l  < 2^30 + 2^30 = 2^31   (fits u32)
//   mid   = (ll >> 16) + (cross & 0xFFFF) < 2^16 + 2^16 = 2^17  (fits u32)
// so no accumulator exceeds 32 bits.
// ---------------------------------------------------------------------------
inline void mul_hi_lo(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
    const uint32_t MASK16 = 0xFFFFu;
    const uint32_t a_l = a & MASK16, a_h = a >> 16;
    const uint32_t b_l = b & MASK16, b_h = b >> 16;

    const uint32_t ll    = a_l * b_l;            // < 2^32
    const uint32_t cross = a_l * b_h + a_h * b_l; // < 2^29  (needs a,b < 2^30)
    const uint32_t hh    = a_h * b_h;            // < 2^24

    const uint32_t mid = (ll >> 16) + (cross & MASK16);  // < 2^17
    lo = (ll & MASK16) | (mid << 16);
    hi = hh + (cross >> 16) + (mid >> 16);
}

// ---------------------------------------------------------------------------
// Montgomery context
// ---------------------------------------------------------------------------

class MontgomeryContext {
public:
    explicit MontgomeryContext(uint32_t prime) : prime_(prime) {
        if (prime_ >= (1u << PRIME_BITS)) throw std::invalid_argument("prime must be < 2^28");
        if ((prime_ & 1u) == 0)           throw std::invalid_argument("prime must be odd");

        q_inv_32_ = static_cast<uint32_t>(modular_inverse(prime_, 1ull << 32)); // q^-1 mod 2^32
        q_low16_  = prime_ & 0xFFFFu;
        q_high16_ = prime_ >> 16;
        w_inv_    = static_cast<uint32_t>(modular_inverse(1ull << 32, prime_));  // R^-1 mod q
    }

    uint32_t prime()    const { return prime_; }
    uint32_t q_inv_32() const { return q_inv_32_; }

    // -----------------------------------------------------------------------
    // The optimized kernel: 32-bit variables only. Input is the u64 value z
    // already split into halves (z = z_high * 2^32 + z_low). Returns a lazy
    // result in [0, 2q). Faithful port of the JAX code.
    // -----------------------------------------------------------------------
    uint32_t reduce_optimized(uint32_t z_high, uint32_t z_low) const {
        const uint32_t MASK16 = 0xFFFFu;
        const uint32_t SHIFT16 = 16;

        const uint32_t q        = prime_;
        const uint32_t q_low    = q_low16_;
        const uint32_t q_high   = q_high16_;
        const uint32_t q_inv_32 = q_inv_32_;

        const uint32_t t      = z_low * q_inv_32;          // wraps mod 2^32
        const uint32_t t_low  = t & MASK16;
        const uint32_t t_high = (t >> SHIFT16) & MASK16;

        const uint32_t prod_high     = t_high * q_high;    // 16x16 -> <2^32
        const uint32_t prod_mid_high = t_high * q_low;
        const uint32_t prod_mid_low  = t_low  * q_high;
        const uint32_t prod_low      = t_low  * q_low;

        const uint32_t mid_low  = (prod_mid_high & MASK16)
                                + (prod_mid_low  & MASK16)
                                + (prod_low >> SHIFT16);
        const uint32_t mid_high = (prod_mid_high >> SHIFT16)
                                + (prod_mid_low  >> SHIFT16)
                                + (mid_low >> SHIFT16);

        const uint32_t t_final = prod_high + mid_high;     // = floor(t*q / 2^32)
        const uint32_t b = z_high + q - t_final;           // in [0, 2q)
        return b;
    }

    // Fully reduce a lazy [0, 2q) value into [0, q).
    uint32_t finalize(uint32_t v) const { return v >= prime_ ? v - prime_ : v; }

    // -----------------------------------------------------------------------
    // Modular multiply (32-bit only). Operands a, b are Montgomery-domain
    // residues in lazy/redundant form, i.e. anything in [0, 2q] -- the same
    // range reduce_optimized outputs, so results chain back in directly.
    //
    // Returns (a * b) mod q (the modular product, in Montgomery form),
    // canonical in [0, q).
    //
    // Bounds with q < 2^28:  a,b <= 2q < 2^29  -> mul_hi_lo safe (< 2^30);
    // a*b <= 4q^2 < q*2^32 = q*R  -> reduce_optimized output in [0, 2q).
    // -----------------------------------------------------------------------
    uint32_t mulmod(uint32_t a, uint32_t b) const {
        uint32_t hi, lo;
        mul_hi_lo(a, b, hi, lo);                    // a*b as (hi, lo), 32-bit only
        return finalize(reduce_optimized(hi, lo));  // (a*b) mod q in [0, q)
    }

    // ---- host-side format conversion (bignum-style, like the Python context) ----
    uint32_t to_montgomery(uint32_t a) const {          // a -> a*R mod q
        return static_cast<uint32_t>((static_cast<uint64_t>(a) << 32) % prime_);
    }
    uint32_t from_montgomery(uint32_t a) const {        // a -> a*R^-1 mod q
        return static_cast<uint32_t>((static_cast<uint64_t>(a) * w_inv_) % prime_);
    }

private:
    uint32_t prime_;
    uint32_t q_inv_32_;
    uint32_t q_low16_;
    uint32_t q_high16_;
    uint32_t w_inv_;
};

}  // namespace mont
