// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FULL external product (#16) on device at n=2048, K RNS channels (K=2 ~56-bit / K=3 ~84-bit) — the
// OnionPIR fold op at the library ring size.  RGSW C (2ℓ×2) ⊡ RLWE d=(d0,d1) -> RLWE (out0,out1).
// Thin consumer of pir2048::Engine (all device ops validated there).  K read from the V2 case header.
// Bit-exact vs cpu/pir/gen_extprod_case.py (--K 2 or --K 3).
//
//   ./build/test_extprod2048 [reference/pir/extprod_n2048.bin]        # K from the file

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>
#include "pir_engine2048.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace pir2048;

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "reference/pir/extprod_n2048.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
    auto ru = [&](uint32_t& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    uint32_t magic, ver, n, twoL, B, K;
    ru(magic); ru(ver); ru(n); ru(twoL); ru(B); ru(K);
    if (magic != 0x45585052u || ver != 2u || n != N) { fmt::print(stderr, "bad case (need V2 n=2048)\n"); return 2; }
    std::vector<uint32_t> primes(K); f.read(reinterpret_cast<char*>(primes.data()), 4 * K);
    const uint32_t ell = twoL / 2;
    fmt::print("case {}: n={} ell={} B={} K={} primes[0]={}\n", path, n, ell, B, K, primes[0]);
    auto rdvec = [&](uint32_t m) { std::vector<uint32_t> v(m); f.read(reinterpret_cast<char*>(v.data()), 4*m); return v; };
    auto rddig = [&]() { std::vector<int32_t> v(N); f.read(reinterpret_cast<char*>(v.data()), 4*N); return v; };
    std::vector<std::vector<uint32_t>> d0r(K), d1r(K);
    for (uint32_t j = 0; j < K; ++j) d0r[j] = rdvec(N);
    for (uint32_t j = 0; j < K; ++j) d1r[j] = rdvec(N);
    std::vector<std::vector<std::vector<uint32_t>>> Cc(K, std::vector<std::vector<uint32_t>>(twoL*2));
    for (uint32_t j = 0; j < K; ++j) for (uint32_t r = 0; r < twoL; ++r) for (uint32_t col = 0; col < 2; ++col) Cc[j][r*2+col] = rdvec(N);
    std::vector<std::vector<uint32_t>> outG(2*K);
    for (uint32_t col = 0; col < 2; ++col) for (uint32_t j = 0; j < K; ++j) outG[col*K+j] = rdvec(N);
    std::vector<std::vector<int32_t>> digitsG(twoL); for (uint32_t r = 0; r < twoL; ++r) digitsG[r] = rddig();

    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        Engine eng(mesh.get(), &cq, primes, ell, B);

        // RGSW C in device-NTT Montgomery
        std::vector<std::vector<std::vector<uint32_t>>> Cntt(K, std::vector<std::vector<uint32_t>>(twoL*2));
        for (uint32_t ch=0; ch<K; ++ch) for (uint32_t i=0;i<twoL*2;++i) Cntt[ch][i]=eng.to_mont(ch, eng.fwd2048(eng.CH[ch], Cc[ch][i]));

        // front half: CRT + gadget on d0,d1
        auto digD0=eng.crt_and_gadget(d0r), digD1=eng.crt_and_gadget(d1r);
        std::vector<std::vector<uint32_t>> digU(twoL);
        for (uint32_t k=0;k<ell;++k){ digU[k]=digD0[k]; digU[ell+k]=digD1[k]; }
        { uint32_t dm=0; for (uint32_t r=0;r<twoL;++r) for (uint32_t i=0;i<N;++i) if (digU[r][i]!=(uint32_t)digitsG[r][i]) ++dm;
          fmt::print("  front-half (CRT+gadget) digit mismatches={}\n", dm); pass &= (dm==0); }

        // digit_rns + fwd2048, then MAC + inv2048
        std::vector<std::vector<std::vector<uint32_t>>> digNtt(K, std::vector<std::vector<uint32_t>>(twoL));
        for (uint32_t r=0;r<twoL;++r){ auto res=eng.digit_rns(digU[r]); for (uint32_t ch=0;ch<K;++ch) digNtt[ch][r]=eng.fwd2048(eng.CH[ch], res[ch]); }
        uint32_t mism=0, checked=0;
        for (uint32_t col=0; col<2; ++col) for (uint32_t ch=0; ch<K; ++ch){
            std::vector<std::vector<uint32_t>> prods(twoL);
            for (uint32_t r=0;r<twoL;++r) prods[r]=eng.pmul(eng.CH[ch], digNtt[ch][r], Cntt[ch][r*2+col]);
            auto out=eng.inv2048(eng.CH[ch], eng.padd(eng.CH[ch], prods));
            auto& gold=outG[col*K+ch];
            for (uint32_t i=0;i<N;++i){ if (out[i]!=gold[i]){ if (mism<8) fmt::print("  col{} ch{} [{}] got={} exp={}\n",col,ch,i,out[i],gold[i]); ++mism; } ++checked; } }
        fmt::print("external product 2048 (device, K={})  mismatches={}/{}\n", K, mism, checked);
        pass &= (mism==0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
