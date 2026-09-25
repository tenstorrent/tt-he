# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Barrett vs Montgomery modular reduction — the two candidates, one operation per line.

Why this file exists
--------------------
`docs/vector/modular_multiplication.md` §3 argues that Montgomery, not Barrett, is the right
reduction for the Tensix SFPU, and the argument is made entirely in **I/O bit-widths**: the SFPU
gives a 32x32 -> *low* 32 multiply natively and nothing else, so every place an algorithm needs the
HIGH half of a wide product is a place it has to pay for hand-built 16x16 limb assembly.

    Barrett     needs a high half (z*m, 85b) AND a correction step.
    Montgomery  needs one high half (t*q) and, with lazy reduction, NO correction step
                -- because R = 2**32 makes "mod R" and "/ R" free word selects.

The two functions below are that argument in executable form. Both are full modular MULTIPLIES --
the product `a*b` is the first line, not an input -- written one operation per line with the input
and output bit-width of each annotated, so the cost of each step is legible without running
anything. This is the form used in the doc and on the slide.

**Three multiplies each, marked `MUL 1/2/3`.** That the counts are equal is the central and most
counter-intuitive result of `docs/he/exp_modmul002.md` §2: Montgomery's widest product is 60 bits
and Barrett's is 58, and on 16-bit limbs both decompose to 11 `mul16` = 22 `sfpmul24`. Barrett does
not carry a higher-precision multiply and does not issue more of them. The measured 2.3% gap
(3.0866 vs 3.1578 µs/tile) is Barrett's extra *shifts* and its correction, not its multiplies.

What these are NOT
------------------
Not the device algorithm. The kernel never forms a 64-bit (let alone 85-bit) number -- it builds
`a*b` from four 16x16->32 limb products and recovers `floor(t*q / 2**32)` by hand. That version is
in §5 of the same doc, and its bit-exact C++ golden is `tests/common/montgomery.hpp`. These two are
the *specifications* the limb code implements: `montgomery()` here and the 9-multiply device
sequence agree on every input, which is what makes the annotations below the honest cost model.

Bit-widths
----------
Every width annotated in the function bodies is a worst-case bound under the project's load-bearing
constraint `q < 2**28` (see doc §4). That constraint is what makes the first line safe: `a, b < q`
forces `z = a*b < 2**56`, the contract the reduction that follows is written to. `main()` checks the
two claims that are operationally load-bearing rather than merely descriptive: that one conditional
subtract suffices (the pre-correction residue is in [0, 2q)), and that the output is canonical.

Run
---
    python reference/vector/modmul_reference.py

Companion docs: `docs/vector/modular_multiplication.md` (§3 is this file, §5 is the device form).
"""

import random

# ------------------------------------------------------------------------------------------------
# Parameters.  q is a 28-bit prime; the sweep in main() runs all of the project's primes, including
# the non-prime boundary value 2**28-1 that the device tests use as a worst case.
# ------------------------------------------------------------------------------------------------
q = 268435399                       # 28b modulus

# precompute (host), Barrett:
L = 56                              # >= bitlength(z)
m = (1 << L) // q                   # reciprocal of q                 out: ~29b

# precompute (host), Montgomery:
R = 1 << 32
qinv = pow(q, -1, R)                # q^{-1} mod R                    out: 32b


def set_modulus(new_q):
    """Re-derive both algorithms' precomputed constants for a new modulus."""
    global q, m, qinv
    q, m, qinv = new_q, (1 << L) // new_q, pow(new_q, -1, R)


# --------------------------------------------------------------------
# Barrett  --  works on ordinary integers, no domain change.
#
# NOTE ON THE DEVICE KERNEL.  The one-pass kernel
# kernels/experiments/exp_modmul002/compute/modmul_barrett_1pass.cpp does NOT
# follow this form literally.  On 16-bit limbs a product costs
# limbs(x)*limbs(y), so the 56b x 29b below is 4 x 2 = EIGHT partial
# products; pre-shifting z to 29 bits first makes it 2 x 2 = FOUR.
# The kernel therefore uses HAC 14.42 (q1 = z >> 27,
# q3 = (q1*mu) >> 29), paying one extra conditional subtract (r < 3q,
# not 2q) to save four multiplies.  Same algorithm, same results;
# this file is the specification, the kernel is the limb-efficient
# realisation.
# --------------------------------------------------------------------
def barrett(a, b):            # a, b < q < 2**28          28b
    z  = a * b                # MUL 1  28b*28b -> 56b
    t  = z * m                # MUL 2  56b*29b -> 85b  <-- high half
    k  = t >> L               #        85b>>56 -> 29b  k ~= floor(z/q)
    kq = k * q                # MUL 3  29b*28b -> 57b
    r  = z - kq               #        56b-57b -> 30b  r in [0, 2q)
    if r >= q: r = r - q      #        30b     -> 28b  1-2 subs total
    return r                  #                   28b  = a*b mod q


# --------------------------------------------------------------------
# Montgomery  --  REDC(z) = z * R^-1 mod q, with R = 2**32.  Called
# redc() in the doc.  Input is a product already in Montgomery form,
# z = a~ * b~ < q**2 < 2**56.
# --------------------------------------------------------------------
def montgomery(a, b):         # a~, b~ < q, Mont. form    28b
    z  = a * b                # MUL 1  28b*28b -> 56b
    z_low  = z & (R - 1)      #        56b mask-> 32b  = z mod R, FREE
    z_high = z >> 32          #        56b>>32 -> 24b  = z / R,   FREE
    t  = z_low * qinv         # MUL 2  32b*32b -> 64b  keep LOW 32
    t  = t & (R - 1)          #        64b mask-> 32b  = t mod R
    tq = t * q                # MUL 3  32b*28b -> 60b
    tf = tq >> 32             #        60b>>32 -> 28b  <-- high half
    r  = z_high + q - tf      #        24b+28b -> 29b  r in [0, 2q)
    if r >= q: r = r - q      #        29b     -> 28b  lazy skips this
    return r                  #                28b/29b  = a~*b~*R^-1


# Both take `b = 1` to act as the bare reduction of an arbitrary z < 2**56 --
# `montgomery(z, 1)` is exactly REDC(z), the form docs/vector/modular_multiplication.md §3 calls
# redc(). The self-check uses that to test the reductions over the full z range, not just products.

# ------------------------------------------------------------------------------------------------
# Domain conversion.  This is the whole cost of choosing Montgomery, and it is why the choice is
# free in an NTT: convert once in, once out, and pay the cheap inner loop millions of times between.
# ------------------------------------------------------------------------------------------------
def to_mont(a):     return (a * R) % q          # a  -> a~ = a*R mod q
def from_mont(am):  return montgomery(am, 1)    # a~ -> a   = a~ * R^-1 mod q


def mulmod_montgomery(a, b):
    """(a*b) mod q via the Montgomery domain -- the in/out conversions made explicit."""
    return from_mont(montgomery(to_mont(a), to_mont(b)))


# ------------------------------------------------------------------------------------------------
# Self-check.  Both algorithms against Python's own `%`, over every prime the device tests sweep,
# on boundary operands and random ones -- plus the pre-correction range claim the annotations make.
# ------------------------------------------------------------------------------------------------
PRIMES = [167772161, 201326593, 251658241, 268435399,
          268435455]                            # 2**28-1: not prime, the device tests' worst case

# Pre-correction residues, extracted so the range claim "r in [0, 2q)" is checkable. These duplicate
# the last two lines of each function above and must be kept in step with them.
def _barrett_pre(z):     return z - ((z * m) >> L) * q
def _montgomery_pre(z):  return (z >> 32) + q - ((((z & (R - 1)) * qinv) & (R - 1)) * q >> 32)


def main(trials=20000, seed=0):
    rng = random.Random(seed)
    for p in PRIMES:
        set_modulus(p)
        assert q * q < (1 << L), f"q={q} violates z = a*b < 2**{L}"

        # (1) the reductions alone, over the full z < 2**56 range (b=1 makes MUL 1 the identity)
        edges = [0, 1, q - 1, q, q + 1, 2 * q - 1, (q - 1) * (q - 1), (1 << L) - 1]
        for z in edges + [rng.randrange(1 << L) for _ in range(trials)]:
            assert barrett(z, 1)    == z % q,              f"barrett  q={q} z={z}"
            assert montgomery(z, 1) == z * pow(R, -1, q) % q, f"redc  q={q} z={z}"
            assert 0 <= _barrett_pre(z)    < 2 * q,  f"barrett  pre-correction out of [0,2q): q={q} z={z}"
            assert 0 <= _montgomery_pre(z) < 2 * q,  f"redc     pre-correction out of [0,2q): q={q} z={z}"

        # (2) as a modular MULTIPLY -- the form both functions now take. Barrett works on plain
        # integers; Montgomery needs the domain, and its conversions are made explicit here.
        ops = [0, 1, 2, q - 1] + [rng.randrange(q) for _ in range(200)]
        for a in ops:
            for b in ops[:16] + [rng.randrange(q) for _ in range(8)]:
                want = a * b % q
                assert barrett(a, b)           == want, f"barrett  q={q} a={a} b={b}"
                assert mulmod_montgomery(a, b) == want, f"mulmod redc     q={q} a={a} b={b}"
                # a*b never leaves the 56b contract, because a, b < q < 2**28
                assert a * b < (1 << L)

        # (3) the LAZY regime -- operands left in [0, 2q) instead of [0, q), as the device keeps them
        # (doc §3, "two engineering refinements"). Then z = a~*b~ < 4q**2, which OVERFLOWS barrett's
        # z < 2**56 contract at 28 bits but stays inside montgomery's z < q*R. This asymmetry is why
        # lazy reduction is available to one and not the other, so it is asserted rather than argued.
        assert 4 * q * q > (1 << L),  f"q={q}: lazy z no longer exceeds barrett's contract -- claim stale"
        assert 4 * q * q < q * R,     f"q={q}: lazy z exceeds montgomery's z < q*R -- claim stale"
        for _ in range(trials // 4):
            za, zb = rng.randrange(2 * q), rng.randrange(2 * q)      # lazy operands, both in [0, 2q)
            r = _montgomery_pre(za * zb)                             # NO conditional subtract
            assert 0 <= r < 2 * q,                        f"lazy redc out of [0,2q): q={q} {za}*{zb}"
            assert r % q == za * zb * pow(R, -1, q) % q,  f"lazy redc wrong value: q={q} {za}*{zb}"

        print(f"q = {q:>9}  ({q.bit_length()}b)   barrett + montgomery: OK   lazy redc: OK   "
              f"[m={m} ({m.bit_length()}b), qinv={qinv} ({qinv.bit_length()}b), "
              f"2**{L} mod q = {(1 << L) % q}]")

    print(f"\nall {len(PRIMES)} moduli agree with Python `%` on every input; one conditional subtract\n"
          f"suffices for both. Montgomery additionally holds in the device's LAZY regime -- operands\n"
          f"in [0, 2q), no correction at all -- which barrett's z < 2**{L} contract does not reach.")


if __name__ == "__main__":
    main()
