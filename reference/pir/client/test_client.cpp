// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host self-tests for pir_client.hpp, run before any of it goes near the device.
//
// Each check compares against something INDEPENDENT of the thing under test — the NTT against
// schoolbook convolution, the gadget against its own recomposition identity, the RGSW against what a
// mux is supposed to select. That is the discipline this project learned the hard way: a routine that
// only agrees with itself can be entirely wrong and still pass.
//
//   g++ -O2 -o test_client test_client.cpp && ./test_client

#include <cstdio>
#include <random>
#include <vector>
#include "pir_client.hpp"

using namespace pirclient;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-52s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++fails;
}

int main() {
    const std::vector<u32> primes = {268369921u, 268361729u, 268271617u};   // onionpir_ref.ntt_primes(3,2048)
    const u32 n = 2048, t = 257, ell = 8; const u64 B = 2048;
    std::mt19937_64 rg(5);
    std::vector<int32_t> s(n); for (auto& v : s) v = (int32_t)(rg() % 3) - 1;

    std::printf("pir_client host self-test (n=%u, K=3, t=%u, ell=%u, B=%llu)\n", n, t, ell, (unsigned long long)B);
    Client C(primes, s, t, ell, B, /*N0=*/512, /*nfold=*/7, n);

    // 1. the inverse transform really inverts the (device-identical) forward one
    { std::vector<u32> x(n), X(n), y(n); bool ok = true;
      for (u32 c = 0; c < 3; ++c) {
          for (auto& v : x) v = (u32)(rg() % primes[c]);
          ntt(C.ch[c], x.data(), X.data()); intt(C.ch[c], X.data(), y.data());
          for (u32 i = 0; i < n; ++i) ok &= (x[i] == y[i]); }
      check(ok, "intt(ntt(x)) == x, all three channels"); }

    // 2. NTT multiplication vs schoolbook negacyclic convolution mod q
    { Poly a(n), b(n);
      for (u32 i = 0; i < n; ++i) { a[i] = C.urand(); b[i] = C.urand(); }
      auto got = C.mul(a, b);
      Poly want(n, 0);                                   // O(n^2) reference, mod the composite q
      for (u32 i = 0; i < n; ++i) { if (!a[i]) continue;
          for (u32 j = 0; j < n; ++j) { if (!b[j]) continue;
              const u128 p = (u128)((__int128)0) + (u128)(( (unsigned __int128)a[i] % C.q) * 1);
              (void)p;
              // a[i]*b[j] mod q needs 168 bits, so reduce via the RNS-free route: use __int128 halves
              u128 prod = 0, x = a[i] % C.q, y = b[j] % C.q;     // russian-peasant mulmod
              while (y) { if (y & 1) prod = Client::addm(prod, x, C.q);
                          x = Client::addm(x, x, C.q); y >>= 1; }
              const u32 k = i + j;
              if (k < n) want[k] = Client::addm(want[k], prod, C.q);
              else       want[k - n] = Client::subm(want[k - n], prod, C.q); } }
      bool ok = true; for (u32 i = 0; i < n; ++i) ok &= (got[i] == want[i]);
      check(ok, "RNS-NTT mul == schoolbook negacyclic mul mod q"); }

    // 3. encrypt/decrypt round trip
    { Poly m(n); for (u32 i = 0; i < n; ++i) m[i] = rg() % t;
      auto ct = C.enc(m, true); auto d = C.dec(ct);
      bool ok = true; for (u32 i = 0; i < n; ++i) ok &= (d[i] == (u32)m[i]);
      check(ok, "dec(enc(m)) == m"); }

    // 4. balanced gadget digits recompose to the original
    { Poly a(n); for (auto& v : a) v = C.urand();
      auto d = C.decompose(a); bool ok = true;
      for (u32 i = 0; i < n && ok; ++i) { u128 acc = 0;
          for (u32 k = 0; k < ell; ++k) { const int32_t dk = d[k][i];
              const u128 term = dk >= 0 ? (u128)dk * C.gvec[k] % C.q
                                        : C.q - (u128)(-dk) * C.gvec[k] % C.q;
              acc = Client::addm(acc, term % C.q, C.q);
              ok &= (dk >= -(int32_t)(B / 2) && dk < (int32_t)(B / 2)); }
          ok &= (acc == a[i]); }
      check(ok, "sum d_k B^k == a (mod q), digits balanced"); }

    // 5. RGSW(s) [X] RLWE(mu) == RLWE(s*mu) — the bridge rlwe_to_rgsw depends on
    { Poly mu(n, 0); for (u32 i = 0; i < 8; ++i) mu[i] = C.urand();
      auto ct = C.enc(mu, false);                        // scaleless: phase is mu + e
      auto out = C.extprod(C.gsw_ntt, ct);
      auto got = C.phase(out);
      auto want = C.mul(mu, C.s_poly());                 // s*mu, exact
      u32 bad = 0; const u128 tol = C.q >> 20;           // phase carries the accumulated noise
      for (u32 i = 0; i < n; ++i) { const u128 diff = Client::subm(got[i], want[i], C.q);
          if (diff > tol && (C.q - diff) > tol) ++bad; }
      check(bad == 0, "extprod(RGSW(s), RLWE(mu)) phase == s*mu + small"); }

    // 6. what the SERVER actually does with a fold RGSW: mux picks x or y by the bit
    { auto Q = C.make_query(/*idx=*/40000);
      std::printf("    query idx=40000 -> c0=%u bits=", Q.c0);
      for (auto b : Q.bits) std::printf("%u", b); std::printf("\n");
      Poly mx(n), my(n); for (u32 i = 0; i < n; ++i) { mx[i] = rg() % t; my[i] = rg() % t; }
      auto x = C.enc(mx, true), y = C.enc(my, true);
      bool ok = true;
      for (u32 f = 0; f < 7; ++f) {
          std::vector<std::array<RnsNtt, 2>> Cn(2 * ell);
          for (u32 r = 0; r < 2 * ell; ++r) Cn[r] = {C.fwd(Q.folds[f][r][0]), C.fwd(Q.folds[f][r][1])};
          std::array<Poly, 2> d = {C.psub(y[0], x[0]), C.psub(y[1], x[1])};
          auto ep = C.extprod(Cn, d);
          std::array<Poly, 2> mux = {C.padd(ep[0], x[0]), C.padd(ep[1], x[1])};
          auto dm = C.dec(mux); const Poly& want = Q.bits[f] ? my : mx;
          for (u32 i = 0; i < n; ++i) ok &= (dm[i] == (u32)want[i]); }
      check(ok, "mux_1of2 with each of the 7 fold RGSWs selects correctly"); }

    // 7. the packed query decrypts to the one-hot the expansion tree expects
    { auto Q = C.make_query(/*idx=*/65535);
      auto d = C.dec(Q.ct);
      const u32 inv = (u32)mont::modular_inverse((1ull << 9) % t, t);
      bool ok = (d[Q.c0] == inv);
      for (u32 i = 0; i < n; ++i) if (i != Q.c0) ok &= (d[i] == 0);
      check(ok, "packed query = one-hot at c0, scaled by (2^m)^-1 mod t"); }

    std::printf(fails ? "\ntest_client: %d FAILED\n" : "\ntest_client: all passed\n", fails);
    return fails ? 1 : 0;
}
