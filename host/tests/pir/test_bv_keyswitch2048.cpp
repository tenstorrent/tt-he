// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BV-style key-switch (#10) on device at n=2048, K RNS channels (K=2 ~56-bit / K=3 ~84-bit).
// (c0,c1) under s' -> (d0,d1) under s: (d0,d1)=(c0,0)+Σ_k G^{-1}(c1)_k ⊛ KSK[k]  (ℓ×2 KSK).
// A "half external product": decompose c1, MAC vs KSK, add c0.  Thin consumer of pir2048::Engine.
// Consumes ks_case OR subs_case (subs pre-automorphed host-side).  K read from the V2 case header.
// Bit-exact vs cpu/pir/gen_ks_case.py / gen_subs_case.py (--K 2 or --K 3).
//
//   ./build/test_bv_keyswitch2048 [reference/pir/ks_n2048.bin]        # K from the file

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>
#include "pir_engine2048.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace pir2048;

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "reference/pir/ks_n2048.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
    auto ru = [&](uint32_t& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    uint32_t magic, ver, n, ell, B, K;
    ru(magic); ru(ver); ru(n); ru(ell); ru(B); ru(K);
    if (magic != 0x4B535754u || ver != 2u || n != N) { fmt::print(stderr, "bad case (need KSWT V2 n=2048)\n"); return 2; }
    std::vector<uint32_t> primes(K); f.read(reinterpret_cast<char*>(primes.data()), 4 * K);
    fmt::print("case {}: n={} ell={} B={} K={} primes[0]={}\n", path, n, ell, B, K, primes[0]);
    auto rdvec = [&](uint32_t m) { std::vector<uint32_t> v(m); f.read(reinterpret_cast<char*>(v.data()), 4*m); return v; };
    auto rddig = [&]() { std::vector<int32_t> v(N); f.read(reinterpret_cast<char*>(v.data()), 4*N); return v; };
    std::vector<std::vector<uint32_t>> c0r(K), c1r(K);
    for (uint32_t j = 0; j < K; ++j) c0r[j] = rdvec(N);
    for (uint32_t j = 0; j < K; ++j) c1r[j] = rdvec(N);
    std::vector<std::vector<std::vector<uint32_t>>> Kc(K, std::vector<std::vector<uint32_t>>(ell*2));
    for (uint32_t j = 0; j < K; ++j) for (uint32_t k = 0; k < ell; ++k) for (uint32_t col = 0; col < 2; ++col) Kc[j][k*2+col] = rdvec(N);
    std::vector<std::vector<uint32_t>> outG(2*K);
    for (uint32_t d = 0; d < 2; ++d) for (uint32_t j = 0; j < K; ++j) outG[d*K+j] = rdvec(N);
    std::vector<std::vector<int32_t>> digitsG(ell); for (uint32_t k = 0; k < ell; ++k) digitsG[k] = rddig();

    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        Engine eng(mesh.get(), &cq, primes, ell, B);

        std::vector<std::vector<std::vector<uint32_t>>> Kntt(K, std::vector<std::vector<uint32_t>>(ell*2));
        for (uint32_t ch=0; ch<K; ++ch) for (uint32_t i=0;i<ell*2;++i) Kntt[ch][i]=eng.to_mont(ch, eng.fwd2048(eng.CH[ch], Kc[ch][i]));

        auto digC1=eng.crt_and_gadget(c1r);
        { uint32_t dm=0; for (uint32_t k=0;k<ell;++k) for (uint32_t i=0;i<N;++i) if (digC1[k][i]!=(uint32_t)digitsG[k][i]) ++dm;
          fmt::print("  front-half (CRT+gadget) digit mismatches={}\n", dm); pass &= (dm==0); }

        std::vector<std::vector<std::vector<uint32_t>>> digNtt(K, std::vector<std::vector<uint32_t>>(ell));
        for (uint32_t k=0;k<ell;++k){ auto res=eng.digit_rns(digC1[k]); for (uint32_t ch=0;ch<K;++ch) digNtt[ch][k]=eng.fwd2048(eng.CH[ch], res[ch]); }
        uint32_t mism=0, checked=0;
        for (uint32_t col=0; col<2; ++col) for (uint32_t ch=0; ch<K; ++ch){
            std::vector<std::vector<uint32_t>> prods(ell);
            for (uint32_t k=0;k<ell;++k) prods[k]=eng.pmul(eng.CH[ch], digNtt[ch][k], Kntt[ch][k*2+col]);
            auto mac=eng.inv2048(eng.CH[ch], eng.padd(eng.CH[ch], prods));
            std::vector<uint32_t> d = (col==0) ? eng.padd(eng.CH[ch], {c0r[ch], mac}) : mac;
            auto& gold=outG[col*K+ch];
            for (uint32_t i=0;i<N;++i){ if (d[i]!=gold[i]){ if (mism<8) fmt::print("  col{} ch{} [{}] got={} exp={}\n",col,ch,i,d[i],gold[i]); ++mism; } ++checked; } }
        fmt::print("BV key-switch 2048 (device, K={})  mismatches={}/{}\n", K, mism, checked);
        pass &= (mism==0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
