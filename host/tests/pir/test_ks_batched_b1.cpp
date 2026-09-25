// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Regression test for the BATCHED (CTB) key-switch, at B=1.
//
// Why this test exists: four helpers on the batched path — crt_and_gadget_b, digit_rns_all_b,
// mmul_bcast, reduce_groups — had NO value-checking anywhere in the suite. They were ported from the
// `wh` branch together with the engine, but two of the KERNELS they depend on were not, and TT-Metal
// silently ignores surplus runtime args, so a stale kernel reads the wrong tiles and yields
// plausible-but-wrong values rather than an error. That cost a long bisect (docs/pir/pir.md §7.3).
//
// At B=1 the CTB layout is DEFINITIONALLY the per-ciphertext layout, so every batched helper can be
// diffed against its validated single-ciphertext counterpart in this tree, and the Garner CRT can be
// diffed against host big-integer arithmetic — an independent ground truth that does not assume either
// device path is correct.
//
// Deliberately cheap: only the ONE Galois key level the probes use is uploaded, not all m_exp levels
// (that prelude is ~240 transform programs and dominates test_query_file_check's runtime).
//
//   ./build/test_ks_batched_b1 [reference/pir/query_small.bin]

#include <cstdint>
#include <fstream>
#include <vector>
#include <cstdio>
#include "pir_engine2048.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace pir2048;

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "reference/pir/query_small.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }

    uint32_t h[10]; f.read((char*)h, sizeof(h));
    if (h[0] != 0x51363553u) { fmt::print(stderr, "bad magic\n"); return 2; }
    const uint32_t n = h[1], K = h[2], ell = h[3], B = h[4], m_exp = h[6];
    std::vector<uint32_t> primes(K); f.read((char*)primes.data(), 4 * K);
    if (n != N) { fmt::print(stderr, "need n=2048\n"); return 2; }
    if (K != 3) { fmt::print(stderr, "this probe assumes K=3\n"); return 2; }

    auto rv = [&]{ std::vector<uint32_t> v(N); f.read((char*)v.data(), 4*N); return v; };
    { std::vector<int32_t> s(N); f.read((char*)s.data(), 4*N); }               // secret: client-only
    std::vector<std::vector<uint32_t>> query(2*K);
    for (uint32_t comp=0; comp<2; ++comp) for (uint32_t ch=0; ch<K; ++ch) query[comp*K+ch] = rv();
    // only level 0 is read; the rest of the gkey block is skipped
    std::vector<std::vector<uint32_t>> K0(K*ell*2);
    for (uint32_t ch=0;ch<K;++ch) for (uint32_t k=0;k<ell;++k) for (uint32_t col=0;col<2;++col)
        K0[ch*(ell*2) + k*2 + col] = rv();
    fmt::print("{}: n={} K={} ell={} B={} m_exp={} (level-0 keys only)\n", path, n, K, ell, B, m_exp);

    uint32_t fails = 0;
    auto report = [&](const char* nm, uint32_t m, uint32_t tot){
        fmt::print("  {:<46} {}/{}{}\n", nm, m, tot, m ? "   <== FAIL" : "");
        if (m) ++fails; };

    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 64u << 20);
        auto& cq = mesh->mesh_command_queue();
        Engine eng(mesh.get(), &cq, primes, ell, B);

        auto slice=[&](uint32_t comp){ std::vector<std::vector<uint32_t>> r(K);
            for(uint32_t ch=0;ch<K;++ch) r[ch]=query[comp*K+ch]; return r; };

        // ---------- host ground truth: Garner CRT of component 1, as K radix-2^28 limbs ----------
        auto minv=[](uint64_t a, uint64_t m){ int64_t t=0,nt=1,r=(int64_t)m,nr=(int64_t)(a%m);
            while(nr){ int64_t q=r/nr,tmp=t-q*nt; t=nt; nt=tmp; tmp=r-q*nr; r=nr; nr=tmp; }
            if(t<0) t+=(int64_t)m; return (uint64_t)t; };
        const uint64_t q0=primes[0], q1=primes[1], q2=primes[2];
        const uint64_t i01=minv(q0,q1), i02=minv(q0,q2), i12=minv(q1,q2);
        std::vector<std::vector<uint32_t>> hostL(K, std::vector<uint32_t>(N));
        for (uint32_t x=0;x<N;++x){
            uint64_t r0=query[K+0][x]%q0, r1=query[K+1][x]%q1, r2=query[K+2][x]%q2;
            uint64_t v1=((r1 + q1 - r0%q1)%q1)*i01%q1;
            uint64_t w =((r2 + q2 - r0%q2)%q2)*i02%q2;
            uint64_t v2=((w  + q2 - v1%q2)%q2)*i12%q2;
            __uint128_t X = (__uint128_t)r0 + (__uint128_t)v1*q0 + (__uint128_t)v2*q0*q1;
            for (uint32_t lb=0;lb<K;++lb) hostL[lb][x] = (uint32_t)((X >> (28*lb)) & 0xFFFFFFFull); }

        // bX layout is tile-major / limb-minor: limb lb of coeff-tile t at tile t*K+lb.
        auto limb_of=[&](const std::vector<uint32_t>& raw, uint32_t lb){
            std::vector<uint32_t> t(2*ELEMS);
            std::copy(raw.begin()+(size_t)(0*K+lb)*ELEMS, raw.begin()+(size_t)(0*K+lb+1)*ELEMS, t.begin());
            std::copy(raw.begin()+(size_t)(1*K+lb)*ELEMS, raw.begin()+(size_t)(1*K+lb+1)*ELEMS, t.begin()+ELEMS);
            return unt2(t); };

        // ---------- 1. CTB round trip must be an identity ----------
        auto bq = eng.upload_ctb({query});
        { auto rt = eng.download_ctb(bq, 1); uint32_t m=0;
          for (uint32_t i=0;i<2*K;++i) for (uint32_t x=0;x<N;++x) if (rt[0][i][x]!=query[i][x]) ++m;
          report("ctb upload->download identity", m, 2*K*N); }

        // ---------- 2. Garner CRT, both paths, against HOST ----------
        std::vector<MeshBuf> bR(K);
        for (uint32_t ch=0;ch<K;++ch){ bR[ch]=eng.mkbuf(2,I32); eng.wrb(bR[ch], til2(query[K+ch])); }
        auto bXs = eng.crtK_buf(bR);                                  // single-ct (single-core)
        // NOTE: keep a Finish between the two CRT calls. Enqueuing both before one Finish makes a
        // device-side fault unattributable to either -- that cost real time during the 7.3 bisect.
        distributed::Finish(cq);
        auto bXb = eng.crtK_b(bq, eng.ctb_grp(1,0,1), 2);             // batched   (multicore, r1_off)
        distributed::Finish(cq);
        { auto raw=eng.rdb(bXs); uint32_t m=0;
          for (uint32_t lb=0;lb<K;++lb){ auto g=limb_of(raw,lb);
              for (uint32_t x=0;x<N;++x) if (g[x]!=hostL[lb][x]) ++m; }
          report("crtK_buf vs host Garner", m, K*N); }
        { auto raw=eng.rdb(bXb); uint32_t m=0;
          for (uint32_t lb=0;lb<K;++lb){ auto g=limb_of(raw,lb);
              for (uint32_t x=0;x<N;++x) if (g[x]!=hostL[lb][x]) ++m; }
          report("crtK_b vs host Garner  [reader_crt_stride r1_off]", m, K*N); }

        // ---------- 3. crt_and_gadget_b vs the single-ct gadget ----------
        // bdig layout: digit k of coeff-tile t at tile t*ell+k, so digit k = tiles [k, ell+k] at nt=2.
        auto digS = eng.crt_and_gadget(slice(1));
        auto bdig = eng.crt_and_gadget_b(bq, eng.ctb_grp(1,0,1), 2);
        distributed::Finish(cq);
        { auto raw=eng.rdb(bdig); uint32_t m=0;
          for (uint32_t k=0;k<ell;++k){ std::vector<uint32_t> t(2*ELEMS);
              std::copy(raw.begin()+(size_t)k*ELEMS,       raw.begin()+(size_t)(k+1)*ELEMS,     t.begin());
              std::copy(raw.begin()+(size_t)(ell+k)*ELEMS, raw.begin()+(size_t)(ell+k+1)*ELEMS, t.begin()+ELEMS);
              auto g=unt2(t);
              for (uint32_t x=0;x<N;++x) if (g[x]!=digS[k][x]) ++m; }
          report("crt_and_gadget_b vs crt_and_gadget", m, ell*N); }

        // ---------- 4. the whole batched key-switch at B=1 vs the single-ct one ----------
        // Exercises digit_rns_all_b, fwd2048_batchM, mmul_bcast and reduce_groups (the last needs
        // reader_i32_groups_2l, NOT this tree's same-named consecutive-groups reader).
        std::vector<std::vector<MeshBuf>> KnB(K, std::vector<MeshBuf>(2));
        for (uint32_t ch=0;ch<K;++ch) for (uint32_t col=0;col<2;++col){
            std::vector<uint32_t> buf((size_t)2*ell*ELEMS);
            for (uint32_t k=0;k<ell;++k){
                auto tl = til2(eng.to_mont(ch, eng.fwd2048(eng.CH[ch], K0[ch*(ell*2)+k*2+col])));
                std::copy(tl.begin(), tl.end(), buf.begin()+(size_t)2*k*ELEMS); }
            auto b = eng.mkbuf(2*ell, I32); eng.wrb(b, buf); KnB[ch][col] = b; }

        std::vector<std::vector<std::vector<uint32_t>>> Kn(K, std::vector<std::vector<uint32_t>>(2*ell));
        for (uint32_t ch=0;ch<K;++ch) for (uint32_t i=0;i<2*ell;++i)
            Kn[ch][i] = eng.to_mont(ch, eng.fwd2048(eng.CH[ch], K0[ch*(ell*2)+i]));
        std::vector<std::vector<std::vector<uint32_t>>> dN(K, std::vector<std::vector<uint32_t>>(ell));
        for (uint32_t k=0;k<ell;++k){ auto res=eng.digit_rns(digS[k]);
            for (uint32_t ch=0;ch<K;++ch) dN[ch][k]=eng.fwd2048(eng.CH[ch],res[ch]); }
        std::vector<std::vector<uint32_t>> ref(2*K);
        for (uint32_t col=0;col<2;++col) for (uint32_t ch=0;ch<K;++ch){
            std::vector<std::vector<uint32_t>> prods(ell);
            for (uint32_t k=0;k<ell;++k) prods[k]=eng.pmul(eng.CH[ch], dN[ch][k], Kn[ch][k*2+col]);
            auto mac=eng.inv2048(eng.CH[ch], eng.padd(eng.CH[ch], prods));
            ref[col*K+ch] = (col==0) ? eng.padd(eng.CH[ch], {query[ch], mac}) : mac; }

        auto o1 = eng.mkctb(1);
        eng.keyswitch_levelB(bq, 1, KnB, o1);
        distributed::Finish(cq);
        auto got = eng.download_ctb(o1, 1);
        { uint32_t m=0; for (uint32_t i=0;i<2*K;++i) for (uint32_t x=0;x<N;++x){
              if (got[0][i][x]!=ref[i][x]) { if (m<4) fmt::print("    comp{} ch{} [{}] batched={} single={}\n",
                                                                i/K, i%K, x, got[0][i][x], ref[i][x]); ++m; } }
          report("keyswitch_levelB(B=1) vs single-ct keyswitch", m, 2*K*N); }

        if (!mesh->close()) ++fails;
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }

    if (!fails) fmt::print("Test Passed\n"); else fmt::print("Test FAILED ({} probe(s))\n", fails);
    return fails ? 1 : 0;
}
