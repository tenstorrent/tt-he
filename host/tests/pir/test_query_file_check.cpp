// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Verify the client query file's SERIALIZATION against the device.
//
// `gen_query65536.py` writes the packed query, the Galois keys and the fold RGSWs in a specific byte
// order. Getting that order wrong does not crash anything — it silently produces garbage at the far
// end of the pipeline. This is exactly the failure mode the DB encoder already hit once (it multiplied
// by R^2 instead of R and its own self-check passed), so the query file gets the same treatment: read
// it, run the REAL device expansion with those keys, and compare against reference selectors carried
// in the file.
//
// Run it on a small case (`--N0 32 --golden`) where the Python reference can afford to produce the
// selectors. The config (n, K, ell, B, t, primes) is identical to the 65 536-row case, so a pass here
// says the *layout* is right; only N0 differs.
//
//   ./build/test_query_file_check [query_small.bin]

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>
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
    const uint32_t n = h[1], K = h[2], ell = h[3], B = h[4], N0 = h[5], m_exp = h[6],
                   nfold = h[7], idx = h[8], t_plain = h[9];
    std::vector<uint32_t> primes(K); f.read((char*)primes.data(), 4 * K);
    fmt::print("{}: n={} K={} ell={} B={} N0={} m_exp={} nfold={} idx={} t={}\n",
               path, n, K, ell, B, N0, m_exp, nfold, idx, t_plain);
    if (n != N) { fmt::print(stderr, "need n=2048\n"); return 2; }

    auto rv = [&]{ std::vector<uint32_t> v(N); f.read((char*)v.data(), 4*N); return v; };
    { std::vector<int32_t> s(N); f.read((char*)s.data(), 4*N); }              // secret: client-only

    std::vector<std::vector<uint32_t>> query(2*K);
    for (uint32_t comp=0; comp<2; ++comp) for (uint32_t ch=0; ch<K; ++ch) query[comp*K+ch] = rv();
    // gkeys[level][ch][k*2+col]
    std::vector<std::vector<std::vector<std::vector<uint32_t>>>> Klvl(
        m_exp, std::vector<std::vector<std::vector<uint32_t>>>(K, std::vector<std::vector<uint32_t>>(ell*2)));
    for (uint32_t a=0;a<m_exp;++a) for (uint32_t ch=0;ch<K;++ch)
        for (uint32_t k=0;k<ell;++k) for (uint32_t col=0;col<2;++col) Klvl[a][ch][k*2+col] = rv();
    uint32_t has_gold = 0; f.read((char*)&has_gold, 4);
    if (!has_gold) { fmt::print(stderr, "file has no golden selectors — regenerate with --golden\n"); return 2; }
    std::vector<std::vector<std::vector<uint32_t>>> selG(N0, std::vector<std::vector<uint32_t>>(2*K));
    for (uint32_t k=0;k<N0;++k) for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<K;++ch)
        selG[k][comp*K+ch] = rv();

    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 64u << 20);
        auto& cq = mesh->mesh_command_queue();
        Engine eng(mesh.get(), &cq, primes, ell, B);

        // key material -> NTT-Montgomery on device, exactly as the pipeline does
        std::vector<std::vector<std::vector<MeshBuf>>> KnB(m_exp, std::vector<std::vector<MeshBuf>>(K, std::vector<MeshBuf>(2)));
        for (uint32_t a=0;a<m_exp;++a) for (uint32_t ch=0;ch<K;++ch) for (uint32_t col=0;col<2;++col){
            std::vector<uint32_t> buf((size_t)2*ell*ELEMS);
            for (uint32_t k=0;k<ell;++k){
                auto tl = til2(eng.to_mont(ch, eng.fwd2048(eng.CH[ch], Klvl[a][ch][k*2+col])));
                std::copy(tl.begin(), tl.end(), buf.begin()+(size_t)2*k*ELEMS); }
            auto b = eng.mkbuf(2*ell, I32); eng.wrb(b, buf); KnB[a][ch][col] = b; }

        // the real level-batched Subs tree, same code the benchmark runs
        MeshBuf bquery = eng.upload_ctb({query});
        // PROBE: CTB upload->download must be an identity. Separates the layout helpers from the compute.
        { auto rt = eng.download_ctb(bquery, 1); uint32_t rm=0;
          for (uint32_t i=0;i<2*K;++i) for (uint32_t x=0;x<N;++x) if (rt[0][i][x]!=query[i][x]) ++rm;
          fmt::print("  PROBE ctb upload->download identity: mismatches={}/{}\n", rm, 2*K*N); }
        // PROBE 2: at B=1 the CTB layout IS the per-ciphertext layout (§3.4), so wh's batched key-switch
        // must agree with this tree's validated single-ct key-switch on the same ct and keys. The four
        // helpers inside keyswitch_levelB (crt_and_gadget_b, digit_rns_all_b, mmul_bcast, reduce_groups)
        // are checked by no other test; this is the missing regression test.
        {
            const uint32_t a0 = 0;
            auto slice=[&](const std::vector<std::vector<uint32_t>>& ct, uint32_t comp){
                std::vector<std::vector<uint32_t>> r(K); for(uint32_t ch=0;ch<K;++ch) r[ch]=ct[comp*K+ch]; return r; };
            // --- reference: this tree's single-ct key-switch ---
            std::vector<std::vector<std::vector<uint32_t>>> Kn(K, std::vector<std::vector<uint32_t>>(2*ell));
            for(uint32_t ch=0;ch<K;++ch) for(uint32_t i=0;i<2*ell;++i)
                Kn[ch][i]=eng.to_mont(ch, eng.fwd2048(eng.CH[ch], Klvl[a0][ch][i]));
            auto digC1 = eng.crt_and_gadget(slice(query,1));
            std::vector<std::vector<std::vector<uint32_t>>> dN(K, std::vector<std::vector<uint32_t>>(ell));
            for(uint32_t k=0;k<ell;++k){ auto res=eng.digit_rns(digC1[k]);
                for(uint32_t ch=0;ch<K;++ch) dN[ch][k]=eng.fwd2048(eng.CH[ch],res[ch]); }
            std::vector<std::vector<uint32_t>> ref(2*K);
            for(uint32_t col=0;col<2;++col) for(uint32_t ch=0;ch<K;++ch){
                std::vector<std::vector<uint32_t>> prods(ell);
                for(uint32_t k=0;k<ell;++k) prods[k]=eng.pmul(eng.CH[ch], dN[ch][k], Kn[ch][k*2+col]);
                auto mac=eng.inv2048(eng.CH[ch], eng.padd(eng.CH[ch], prods));
                ref[col*K+ch] = (col==0) ? eng.padd(eng.CH[ch], {query[ch], mac}) : mac; }
            // PROBE 2f: walk crtK_b's Garner chain stage by stage against host values.
            // garner_diff(a,b) = (b-a) mod q (verified by 2c).
            {
                auto minv=[](uint64_t a, uint64_t m){ int64_t t=0,nt2=1; int64_t r=(int64_t)m,nr=(int64_t)(a%m);
                    while(nr){ int64_t q=r/nr; int64_t tmp=t-q*nt2; t=nt2; nt2=tmp; tmp=r-q*nr; r=nr; nr=tmp; }
                    if(t<0) t+=(int64_t)m; return (uint64_t)t; };
                const uint64_t q0=primes[0], q1=primes[1], q2=primes[2];
                const uint64_t i01=minv(q0,q1), i02=minv(q0,q2), i12=minv(q1,q2);
                std::vector<std::vector<uint32_t>> H(6, std::vector<uint32_t>(N));   // D01,v1,A,Bt,CC,v2
                for (uint32_t x=0;x<N;++x){
                    uint64_t r0=query[3][x]%q0, r1=query[4][x]%q1, r2=query[5][x]%q2;
                    uint64_t D01=(r1 + q1 - r0%q1)%q1,  v1=D01*i01%q1;
                    uint64_t A  =(r2 + q2 - r0%q2)%q2,  Bt=A*i02%q2;
                    uint64_t CC =(Bt + q2 - v1%q2)%q2,  v2=CC*i12%q2;
                    H[0][x]=(uint32_t)D01; H[1][x]=(uint32_t)v1; H[2][x]=(uint32_t)A;
                    H[3][x]=(uint32_t)Bt;  H[4][x]=(uint32_t)CC; H[5][x]=(uint32_t)v2; }

                auto bq0f = eng.upload_ctb({query});
                const uint32_t s0 = eng.ctb_grp(1,0,1);
                auto bD01=eng.mkbuf(2,I32), bV1=eng.mkbuf(2,I32), bA=eng.mkbuf(2,I32),
                     bBt=eng.mkbuf(2,I32), bCC=eng.mkbuf(2,I32), bV2=eng.mkbuf(2,I32);
                eng.stage_ab("compute/garner_diff.cpp", bq0f, bq0f, bD01, {(uint32_t)q1, 2u}, s0+0, s0+2, 2u);
                eng.stage_cmul(bD01, eng.bInv[0*4+1], bV1, 2u, {2u,(uint32_t)q1,eng.ctxp[1].q_inv_32(),(uint32_t)(q1&0xFFFF),(uint32_t)(q1>>16)});
                eng.stage_ab("compute/garner_diff.cpp", bq0f, bq0f, bA, {(uint32_t)q2, 2u}, s0+0, s0+4, 2u);
                eng.stage_cmul(bA, eng.bInv[0*4+2], bBt, 2u, {2u,(uint32_t)q2,eng.ctxp[2].q_inv_32(),(uint32_t)(q2&0xFFFF),(uint32_t)(q2>>16)});
                eng.stage_ab("compute/garner_diff.cpp", bV1, bBt, bCC, {(uint32_t)q2, 2u}, 0u, 0u, 2u);
                eng.stage_cmul(bCC, eng.bInv[1*4+2], bV2, 2u, {2u,(uint32_t)q2,eng.ctxp[2].q_inv_32(),(uint32_t)(q2&0xFFFF),(uint32_t)(q2>>16)});
                distributed::Finish(cq);
                const char* nms[6]={"D01","v1 ","A  ","Bt ","CC ","v2 "};
                MeshBuf bs[6]={bD01,bV1,bA,bBt,bCC,bV2};
                for (uint32_t i=0;i<6;++i){ auto g=unt2(eng.rdb(bs[i])); uint32_t m=0;
                    for (uint32_t x=0;x<N;++x) if (g[x]!=H[i][x]) { if(m<2) fmt::print("    [{}] [{}] dev={} host={}\n",nms[i],x,g[x],H[i][x]); ++m; }
                    fmt::print("  PROBE chain {} : mismatches={}/{}\n", nms[i], m, N); }
            }

            // PROBE 2e: HOST ground truth for the Garner CRT, so neither device path is assumed golden.
            // X = CRT(r0,r1,r2) as a big integer, split into K radix-2^28 limbs.
            {
                auto minv=[](uint64_t a, uint64_t m){ int64_t t=0,nt=1; int64_t r=(int64_t)m,nr=(int64_t)(a%m);
                    while(nr){ int64_t q=r/nr; int64_t tmp=t-q*nt; t=nt; nt=tmp; tmp=r-q*nr; r=nr; nr=tmp; }
                    if(t<0) t+=(int64_t)m; return (uint64_t)t; };
                const uint64_t q0=primes[0], q1=primes[1], q2=primes[2];
                const uint64_t i01=minv(q0,q1), i02=minv(q0,q2), i12=minv(q1,q2);
                std::vector<std::vector<uint32_t>> hostL(K, std::vector<uint32_t>(N));
                for (uint32_t x=0;x<N;++x){
                    uint64_t r0=query[3][x]%q0, r1=query[4][x]%q1, r2=query[5][x]%q2;
                    uint64_t v0=r0;
                    uint64_t v1=( (r1 + q1 - v0%q1) % q1 ) * i01 % q1;
                    uint64_t w =( (r2 + q2 - v0%q2) % q2 ) * i02 % q2;
                    uint64_t v2=( (w  + q2 - v1%q2) % q2 ) * i12 % q2;
                    __uint128_t X = (__uint128_t)v0 + (__uint128_t)v1*q0 + (__uint128_t)v2*q0*q1;
                    for (uint32_t lb=0;lb<K;++lb) hostL[lb][x] = (uint32_t)((X >> (28*lb)) & 0xFFFFFFFull); }
                auto cmp=[&](const char* nm, const std::vector<uint32_t>& raw){ uint32_t m=0;
                    for (uint32_t lb=0;lb<K;++lb){ std::vector<uint32_t> t(2*ELEMS);
                        std::copy(raw.begin()+(size_t)(0*K+lb)*ELEMS, raw.begin()+(size_t)(0*K+lb+1)*ELEMS, t.begin());
                        std::copy(raw.begin()+(size_t)(1*K+lb)*ELEMS, raw.begin()+(size_t)(1*K+lb+1)*ELEMS, t.begin()+ELEMS);
                        auto g=unt2(t);
                        for (uint32_t x=0;x<N;++x) if (g[x]!=hostL[lb][x]) { if(m<3) fmt::print("  [{}] limb{} [{}] dev={} host={}\n",nm,lb,x,g[x],hostL[lb][x]); ++m; } }
                    fmt::print("  PROBE {} vs HOST Garner: mismatches={}/{}\n", nm, m, K*N); };
                std::vector<MeshBuf> bR(K);
                for (uint32_t ch=0;ch<K;++ch){ bR[ch]=eng.mkbuf(2,I32); eng.wrb(bR[ch], til2(query[K+ch])); }
                auto bXs = eng.crtK_buf(bR);
                auto bq0e = eng.upload_ctb({query});
                auto bXb = eng.crtK_b(bq0e, eng.ctb_grp(1,0,1), 2);
                distributed::Finish(cq);
                cmp("crtK_buf", eng.rdb(bXs));
                cmp("crtK_b  ", eng.rdb(bXb));
            }

            // PROBE 2d: same crtK_b chain, but with the K residues repacked to src0=0 (tiles 0,2,4).
            // If this matches crtK while src0=6 does not, the bug is in the src0/R() offset handling;
            // if it also fails, the bug is in the multicore stages themselves.
            { auto bpack = eng.mkbuf(6, I32); std::vector<uint32_t> pk((size_t)6*ELEMS);
              for (uint32_t ch=0; ch<K; ++ch){ auto tl = til2(query[K+ch]);
                  std::copy(tl.begin(), tl.end(), pk.begin()+(size_t)2*ch*ELEMS); }
              eng.wrb(bpack, pk);
              auto bX0 = eng.crtK_b(bpack, 0u, 2u);
              distributed::Finish(cq);
              auto raw = eng.rdb(bX0); auto limbsS = eng.crtK(slice(query,1)); uint32_t pm=0;
              for (uint32_t lb=0;lb<K;++lb){ std::vector<uint32_t> t(2*ELEMS);
                  std::copy(raw.begin()+(size_t)(0*K+lb)*ELEMS, raw.begin()+(size_t)(0*K+lb+1)*ELEMS, t.begin());
                  std::copy(raw.begin()+(size_t)(1*K+lb)*ELEMS, raw.begin()+(size_t)(1*K+lb+1)*ELEMS, t.begin()+ELEMS);
                  auto got = unt2(t);
                  for (uint32_t x=0;x<N;++x) if (got[x]!=limbsS[lb][x]) ++pm; }
              fmt::print("  PROBE crtK_b(src0=0, repacked) vs crtK: mismatches={}/{}\n", pm, K*N); }

            // PROBE 2c: very first stage of crtK_b's Garner chain, against host truth.
            // R(0)=ctb_grp(1,0,1)+0=6, R(1)=8 -> garner_diff gives (q[4]-q[3]) mod primes[1].
            { auto bq0c = eng.upload_ctb({query});
              auto bD = eng.mkbuf(2, I32);
              eng.stage_ab("compute/garner_diff.cpp", bq0c, bq0c, bD, {primes[1], 2u}, 6u, 8u, 2u);
              distributed::Finish(cq);
              auto got = unt2(eng.rdb(bD)); uint32_t dm=0;
              for (uint32_t x=0;x<N;++x){
                  uint64_t q1=primes[1];
                  uint32_t exp=(uint32_t)((query[4][x]%q1 + q1 - query[3][x]%q1)%q1);  // (b-a): Garner wants r1-r0
                  if (got[x]!=exp){ if(dm<4) fmt::print("  [gd] [{}] dev={} host={}  (a={} b={})\n",x,got[x],exp,query[3][x],query[4][x]); ++dm; } }
              fmt::print("  PROBE stage_ab garner_diff vs host: mismatches={}/{}\n", dm, N); }

            // PROBE 2b: split crt_and_gadget_b = gadget_b(crtK_b(..)). Both bX layouts are tile-major
            // limb-minor (t*K+lb), so crtK_b is directly comparable to this tree's crtK.
            { auto bq0b = eng.upload_ctb({query});
              auto bX = eng.crtK_b(bq0b, eng.ctb_grp(1,0,1), 2);
              distributed::Finish(cq);
              auto raw = eng.rdb(bX);
              auto limbsS = eng.crtK(slice(query,1));
              uint32_t xm=0;
              for (uint32_t lb=0;lb<K;++lb){ std::vector<uint32_t> t(2*ELEMS);
                  std::copy(raw.begin()+(size_t)(0*K+lb)*ELEMS, raw.begin()+(size_t)(0*K+lb+1)*ELEMS, t.begin());
                  std::copy(raw.begin()+(size_t)(1*K+lb)*ELEMS, raw.begin()+(size_t)(1*K+lb+1)*ELEMS, t.begin()+ELEMS);
                  auto got = unt2(t);
                  for (uint32_t x=0;x<N;++x) if (got[x]!=limbsS[lb][x]) { if(xm<4) fmt::print("  [crt-b] limb{} [{}] batched={} single={}\n",lb,x,got[x],limbsS[lb][x]); ++xm; } }
              fmt::print("  PROBE crtK_b vs crtK: mismatches={}/{}\n", xm, K*N); }

            // PROBE 2a: first helper in the chain -- crt_and_gadget_b vs this tree's crt_and_gadget.
            // bdig layout: digit k of coeff-tile t at tile t*ell+k, so digit k = tiles [k, ell+k] at nt=2.
            { auto bq0 = eng.upload_ctb({query});
              auto bdig = eng.crt_and_gadget_b(bq0, eng.ctb_grp(1,0,1), 2);
              distributed::Finish(cq);
              auto raw = eng.rdb(bdig); uint32_t gm=0;
              for (uint32_t k=0;k<ell;++k){ std::vector<uint32_t> t(2*ELEMS);
                  std::copy(raw.begin()+(size_t)k*ELEMS,       raw.begin()+(size_t)(k+1)*ELEMS,       t.begin());
                  std::copy(raw.begin()+(size_t)(ell+k)*ELEMS, raw.begin()+(size_t)(ell+k+1)*ELEMS,   t.begin()+ELEMS);
                  auto got = unt2(t);
                  for (uint32_t x=0;x<N;++x) if (got[x]!=digC1[k][x]) { if(gm<4) fmt::print("  [cg-b] digit{} [{}] batched={} single={}\n",k,x,got[x],digC1[k][x]); ++gm; } }
              fmt::print("  PROBE crt_and_gadget_b vs crt_and_gadget: mismatches={}/{}\n", gm, ell*N); }

            // --- under test: wh's batched key-switch at B=1 ---
            auto bq1 = eng.upload_ctb({query});
            auto o1 = eng.mkctb(1);
            eng.keyswitch_levelB(bq1, 1, KnB[a0], o1);
            distributed::Finish(cq);
            auto got1 = eng.download_ctb(o1, 1);
            uint32_t km=0; for(uint32_t i=0;i<2*K;++i) for(uint32_t x=0;x<N;++x)
                if (got1[0][i][x]!=ref[i][x]) { if(km<4) fmt::print("  [ks-b1] comp{} ch{} [{}] batched={} single={}\n",i/K,i%K,x,got1[0][i][x],ref[i][x]); ++km; }
            fmt::print("  PROBE keyswitch_levelB(B=1) vs single-ct keyswitch: mismatches={}/{}\n", km, 2*K*N);
        }
        eng.prewarm_batch(N0);
        eng.pool_enable(true); eng.wl_enable(true);
        eng.pool_reset(); eng.wl_reset();
        uint32_t Bl = 1; MeshBuf cts = bquery;
        for (uint32_t a=0;a<m_exp;++a){ int64_t j=(int64_t)(N>>a)+1;
            auto comb=eng.mkctb(2*Bl);
            eng.permute_ctb_into(cts, Bl, comb, 2*Bl, 0,  1u, 0u);
            eng.monomul_ctb_into(cts, Bl, comb, 2*Bl, Bl, -(int64_t)(1u<<a));
            auto scr=eng.mkctb(2*Bl), sb=eng.mkctb(2*Bl);
            eng.subs_levelB(comb, 2*Bl, j, KnB[a], sb, scr);
            auto nxt=eng.mkctb(2*Bl); eng.ctadd_ctb(comb, sb, nxt, 2*Bl);
            cts=nxt; Bl*=2; }
        distributed::Finish(cq);
        auto got = eng.download_ctb(cts, Bl);

        uint32_t mism = 0, checked = 0;
        for (uint32_t k=0;k<N0;++k) for (uint32_t i=0;i<2*K;++i) for (uint32_t x=0;x<N;++x){
            if (got[k][i][x] != selG[k][i][x]) {
                if (mism < 6) fmt::print("  sel{} comp{} ch{} [{}] device={} client-golden={}\n",
                                         k, i/K, i%K, x, got[k][i][x], selG[k][i][x]);
                ++mism; }
            ++checked; }
        fmt::print("query file -> device expansion: mismatches={}/{} ({} selectors)\n", mism, checked, N0);
        pass &= (mism == 0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
