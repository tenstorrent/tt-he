// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Query expansion (#18) + RLWE->RGSW unpack (#18b) on device at n=2048, K RNS channels.
// Subs tree (host automorph/shift/add + device BV key-switch) -> N0 one-hot selectors; unpack
// (extprod vs RGSW(s)) -> RGSW rows.  Thin orchestration over pir2048::Engine.  K from the V2 header.
// Bit-exact vs cpu/pir/gen_expand_case.py (--K 2 or --K 3).
//
//   ./build/test_query_expand2048 [reference/pir/expand_n2048.bin]    # K from the file

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>
#include "pir_engine2048.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace pir2048;

namespace {
using CtV = std::vector<std::vector<uint32_t>>;   // RLWE ct: 2*K polys, indexed [comp*K + ch]
std::vector<uint32_t> automorph_poly(const std::vector<uint32_t>& poly, int64_t j, uint32_t q) {
    const int64_t twon = 2*(int64_t)N; std::vector<uint32_t> r(N, 0);
    for (uint32_t i = 0; i < N; ++i) { int64_t p = ((int64_t)i*j) % twon; if (p < 0) p += twon; uint32_t v = poly[i];
        if (p < (int64_t)N) r[p] = (uint32_t)(((uint64_t)r[p]+v)%q); else r[p-N] = (uint32_t)(((uint64_t)r[p-N]+q-v)%q); } return r; }
std::vector<uint32_t> mono_mul_poly(const std::vector<uint32_t>& poly, int64_t k, uint32_t q) {
    const int64_t twon = 2*(int64_t)N; std::vector<uint32_t> r(N, 0);
    for (uint32_t i = 0; i < N; ++i) { int64_t p = ((int64_t)i+k) % twon; if (p < 0) p += twon; uint32_t v = poly[i];
        if (p < (int64_t)N) r[p] = (uint32_t)(((uint64_t)r[p]+v)%q); else r[p-N] = (uint32_t)(((uint64_t)r[p-N]+q-v)%q); } return r; }
}  // namespace

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "reference/pir/expand_n2048.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
    auto ru = [&](uint32_t& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    uint32_t magic, ver, n, ell, B, N0, m_exp, K, t;
    ru(magic); ru(ver); ru(n); ru(ell); ru(B); ru(N0); ru(m_exp); ru(K); ru(t);
    if (magic != 0x45585044u || ver != 2u || n != N) { fmt::print(stderr, "bad case (need EXPD V2 n=2048)\n"); return 2; }
    std::vector<uint32_t> primes(K); f.read(reinterpret_cast<char*>(primes.data()), 4 * K);
    const uint32_t twoL = 2*ell;
    fmt::print("case {}: n={} ell={} B={} N0={} m_exp={} K={} primes[0]={}\n", path, n, ell, B, N0, m_exp, K, primes[0]);
    auto rv = [&]() { std::vector<uint32_t> v(N); f.read(reinterpret_cast<char*>(v.data()), 4*N); return v; };
    { std::vector<int32_t> s(N); f.read(reinterpret_cast<char*>(s.data()), 4*N); }        // skip secret
    CtV query(2*K); for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<K;++ch) query[comp*K+ch]=rv();
    std::vector<std::vector<std::vector<std::vector<uint32_t>>>> Klvl(m_exp, std::vector<std::vector<std::vector<uint32_t>>>(K, std::vector<std::vector<uint32_t>>(ell*2)));
    for (uint32_t a=0;a<m_exp;++a) for (uint32_t ch=0;ch<K;++ch) for (uint32_t k=0;k<ell;++k) for (uint32_t col=0;col<2;++col) Klvl[a][ch][k*2+col]=rv();
    std::vector<CtV> selG(N0); for (uint32_t k=0;k<N0;++k){ selG[k].resize(2*K); for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<K;++ch) selG[k][comp*K+ch]=rv(); }
    uint32_t bit; ru(bit); (void)bit;
    std::vector<std::vector<std::vector<uint32_t>>> Gc(K, std::vector<std::vector<uint32_t>>(twoL*2));  // RGSW(s) [ch][row*2+col]
    for (uint32_t ch=0;ch<K;++ch) for (uint32_t row=0;row<twoL;++row) for (uint32_t col=0;col<2;++col) Gc[ch][row*2+col]=rv();
    std::vector<CtV> msg(ell); for (uint32_t k=0;k<ell;++k){ msg[k].resize(2*K); for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<K;++ch) msg[k][comp*K+ch]=rv(); }
    std::vector<CtV> botG(ell); for (uint32_t k=0;k<ell;++k){ botG[k].resize(2*K); for (uint32_t col=0;col<2;++col) for (uint32_t ch=0;ch<K;++ch) botG[k][col*K+ch]=rv(); }

    bool pass = true;
    try {
        // 256 MB trace region reserved at init (required before any trace capture).
        auto mesh = distributed::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 256u << 20);
        auto& cq = mesh->mesh_command_queue();
        Engine eng(mesh.get(), &cq, primes, ell, B);
        // inv-layout -> natural tilized, as a permutation table (used by every keyswitch output)
        auto bI2N = eng.perm_table_buf(eng.perm_table_inv_to_natural());
        auto bN2F = eng.perm_table_buf(eng.perm_table_natural_to_fwdin());

        auto slice=[&](const CtV& ct, uint32_t comp){ std::vector<std::vector<uint32_t>> r(K); for(uint32_t ch=0;ch<K;++ch) r[ch]=ct[comp*K+ch]; return r; };
        // BV key-switch: decompose c1 -> MAC vs KSK(NTT-mont) -> +c0 to col0
        // RESIDENT key-switch: identical structure to the answer pipeline's resident extproduct, so this is
        // reuse rather than new machinery. Readback was 91% of expansion's replay time; this removes the
        // per-transform round trip (fwd read back only to be re-uploaded for the MAC) and keeps
        // MAC -> transpose -> inverse entirely in DRAM, with ONE readback per (col, channel).
        using BufCt = std::vector<MeshBuf>;   // 2K polys, NATURAL tilized layout
        auto keyswitch=[&](const BufCt& ct, const std::vector<std::vector<MeshBuf>>& Knb)->BufCt{
            // RESIDENT crt -> gadget -> digit_rns: residues go in as buffers and the limbs/digits never
            // return to host, so the two readbacks that crt_and_gadget() + digit_rns() used to cost per
            // key-switch collapse to one (the digit_rns output, still needed for the fwd input reshape).
            std::vector<MeshBuf> bR(K);
            for(uint32_t ch=0;ch<K;++ch) bR[ch]=ct[1*K+ch];        // c1 already resident
            auto bdig = eng.crt_and_gadget_buf(bR);
            std::vector<std::vector<MeshBuf>> dNb(K, std::vector<MeshBuf>(ell));
            // digit_rns_batch: all ell splits in ONE co-scheduled program. Measured ~3% WORSE in the
            // answer pipeline (only 48 calls there); testing it here where key-switch calls it ell=8 times
            // per node -- the "measure per stage" lesson, applied rather than assumed.
            auto bres = eng.digit_rns_launch(bdig);   // resident: no readback
            // All ell digit transforms of a channel are INDEPENDENT -> batch them stage-wise, so each
            // channel costs 10 programs instead of 10*ell. (Batching across channels too would need one
            // program per (stage, channel) anyway, since the twiddle/fold constants are per-channel.)
            // ALL ell*K digit transforms in one batch. Mixing channels is fine: the fold/twiddle buffer
            // addresses and modulus params are runtime args, so each core gets its own channel.
            // The fwd-input reshape is ALSO a fixed permutation, so it runs on device straight out of the
            // resident digit buffer (residue (k,ch) at tiles k*2K+ch, k*2K+K+ch -> stride K). This removes
            // the last host boundary inside a key-switch: the digit readback + host write_fwd_in2.
            { std::vector<Chan*> chans; std::vector<MeshBuf> ins, outs;
              std::vector<uint32_t> qs, offs, strs;
              const uint32_t twoK = 2*K;
              for(uint32_t ch=0;ch<K;++ch) for(uint32_t k=0;k<ell;++k){
                  ins.push_back(bres); offs.push_back(k*twoK + ch); strs.push_back(K);
                  qs.push_back(primes[ch]);
                  outs.push_back(eng.mkbuf(2,I32)); chans.push_back(&eng.CH[ch]); }
              std::vector<MeshBuf> fin(ins.size());
              for(size_t i=0;i<ins.size();++i) fin[i]=eng.mkbuf(2,I32);
              eng.permute_batch(ins, bN2F, fin, qs, offs, strs);      // -> fwd-input layout, on device
              eng.fwd2048_buf2_batch(chans, fin, outs);
              for(uint32_t ch=0;ch<K;++ch) for(uint32_t k=0;k<ell;++k) dNb[ch][k]=outs[ch*ell+k]; }
            BufCt d(2*K);
            for(uint32_t col=0;col<2;++col) for(uint32_t ch=0;ch<K;++ch){
                std::vector<MeshBuf> bcol(ell); for(uint32_t k=0;k<ell;++k) bcol[k]=Knb[ch][k*2+col];
                auto s2=eng.mac_ntt_buf(eng.CH[ch], dNb[ch], bcol);
                auto s2t=eng.mkbuf(2,I32); for(uint32_t bk2=0;bk2<2;++bk2) eng.transpose_dev(s2,s2t,bk2,bk2);
                auto o2=eng.mkbuf(2,I32); eng.inv2048_buf2(eng.CH[ch], s2t, o2);
                // inv-layout -> NATURAL on device: the reindex is just a permutation, so it costs one
                // permute program and lets the result stay a buffer (replaces read_inv_out2's readback).
                auto nat=eng.mkbuf(2,I32); eng.permute_dev(o2, bI2N, nat, primes[ch]);
                if (col==0) { auto bsum=eng.mkbuf(2,I32); eng.padd_ab(eng.CH[ch], nat, ct[0*K+ch], bsum);
                              d[col*K+ch]=bsum; }
                else          d[col*K+ch]=nat; }
            return d; };
        // external product: RGSW(2ℓ×2, NTT-mont) ⊡ RLWE d -> RLWE
        auto extprod=[&](const CtV& d, const std::vector<std::vector<std::vector<uint32_t>>>& Cn)->CtV{
            auto dg0=eng.crt_and_gadget(slice(d,0)), dg1=eng.crt_and_gadget(slice(d,1));
            std::vector<std::vector<uint32_t>> digU(twoL); for(uint32_t k=0;k<ell;++k){ digU[k]=dg0[k]; digU[ell+k]=dg1[k]; }
            std::vector<std::vector<std::vector<uint32_t>>> dN(K, std::vector<std::vector<uint32_t>>(twoL));
            for(uint32_t r=0;r<twoL;++r){ auto res=eng.digit_rns(digU[r]); for(uint32_t ch=0;ch<K;++ch) dN[ch][r]=eng.fwd2048(eng.CH[ch],res[ch]); }
            CtV ep(2*K);
            for(uint32_t col=0;col<2;++col) for(uint32_t ch=0;ch<K;++ch){ std::vector<std::vector<uint32_t>> prods(twoL);
                for(uint32_t r=0;r<twoL;++r) prods[r]=eng.pmul(eng.CH[ch], dN[ch][r], Cn[ch][r*2+col]); ep[col*K+ch]=eng.inv2048(eng.CH[ch], eng.padd(eng.CH[ch], prods)); }
            return ep; };
        // A5: the whole Subs tree runs on buffers. automorph/monomul are permute_dev (validated
        // 0/2048 x 17 in test_permute_dev); ctadd is padd_ab. Tables are per level and reused by every node.
        auto permute_ct=[&](const std::vector<MeshBuf>& ct, MeshBuf tab){ std::vector<MeshBuf> r(2*K);
            std::vector<uint32_t> qs(2*K);
            for(uint32_t c=0;c<2;++c) for(uint32_t ch=0;ch<K;++ch){ r[c*K+ch]=eng.mkbuf(2,I32); qs[c*K+ch]=primes[ch]; }
            eng.permute_batch(ct, tab, r, qs);   // all 2K permutes in ONE co-scheduled program
            return r; };
        auto ctadd    =[&](const std::vector<MeshBuf>& a, const std::vector<MeshBuf>& b){ std::vector<MeshBuf> r(2*K);
            for(uint32_t c=0;c<2;++c) for(uint32_t ch=0;ch<K;++ch){ r[c*K+ch]=eng.mkbuf(2,I32);
                eng.padd_ab(eng.CH[ch], a[c*K+ch], b[c*K+ch], r[c*K+ch]); } return r; };

        // Key-switch keys: transformed ONCE into RESIDENT packed buffers, in the same layout the resident
        // MAC consumes. to_mont moves to the coefficient-domain input (see test_ntt2048_res check [5]).
        std::vector<std::vector<std::vector<MeshBuf>>> KnbL(m_exp, std::vector<std::vector<MeshBuf>>(K, std::vector<MeshBuf>(ell*2)));
        for(uint32_t a=0;a<m_exp;++a) for(uint32_t ch=0;ch<K;++ch) for(uint32_t i=0;i<ell*2;++i){
            auto in2=eng.mkbuf(2,I32); eng.write_fwd_in2(in2, eng.to_mont(ch, Klvl[a][ch][i]));
            KnbL[a][ch][i]=eng.mkbuf(2,I32); eng.fwd2048_buf2(eng.CH[ch], in2, KnbL[a][ch][i]); }
        std::vector<std::vector<std::vector<uint32_t>>> GSn(K, std::vector<std::vector<uint32_t>>(twoL*2));
        for(uint32_t ch=0;ch<K;++ch) for(uint32_t i=0;i<twoL*2;++i) GSn[ch][i]=eng.to_mont(ch, eng.fwd2048(eng.CH[ch], Gc[ch][i]));
        std::vector<MeshBuf> tabJ(m_exp), tabK(m_exp);
        for(uint32_t a=0;a<m_exp;++a){ tabJ[a]=eng.perm_table_buf(eng.perm_table_automorph((uint64_t)(N>>a)+1));
                                       tabK[a]=eng.perm_table_buf(eng.perm_table_monomul(-(int64_t)(1u<<a))); }
        auto subs=[&](const std::vector<MeshBuf>& ct, MeshBuf tabJ, uint32_t a){ return keyswitch(permute_ct(ct,tabJ), KnbL[a]); };

        // (a) EXPAND: packed one-hot query -> N0 selector cts (Subs tree)
        // Instrumented because expansion is ~52% of the CPU reference's server time (vs 34% for the first dimension),
        // so it -- not the scan -- sets the end-to-end figure. Same host-phase split as the answer test.
        using eclk = std::chrono::high_resolution_clock;
        // Expansion as ONE callable so the record/capture/replay ladder of docs/pir/latency_optimization.md
        // §3/§4b applies unchanged. The measured split (82% enqueue = fresh-program serialization) is the
        // same signature the answer pipeline had, so caching + trace should give ~5x here too.
        auto run_expand = [&]() -> std::vector<std::vector<MeshBuf>> {
            eng.begin_pass();
            std::vector<MeshBuf> q0(2*K);
            for(uint32_t i=0;i<2*K;++i){ q0[i]=eng.mkbuf(2,I32); eng.wrb(q0[i], til2(query[i])); }
            std::vector<std::vector<MeshBuf>> cts={q0};
            // PHASE B1: the 2^a nodes of a level are INDEPENDENT, so their permutes batch across nodes,
            // not just across the 2K polys of one ciphertext. Three batched permutes per level (automorph of
            // c, monomial shift of c, automorph of c_sh) replace 3*2^a per-node calls -- 2^a x the width and
            // 2^a x fewer programs. The key-switches themselves remain per node (batching those is B1's
            // larger half: every key-switch stage would have to merge across nodes).
            for(uint32_t a=0;a<m_exp;++a){
                const uint32_t nn=(uint32_t)cts.size(), W=2*K;
                std::vector<uint32_t> qs(nn*W);
                for(uint32_t b=0;b<nn;++b) for(uint32_t i=0;i<W;++i) qs[b*W+i]=primes[i%K];
                auto gather=[&](void)->std::vector<MeshBuf>{ std::vector<MeshBuf> v(nn*W);
                    for(uint32_t x=0;x<nn*W;++x) v[x]=eng.mkbuf(2,I32); return v; };
                std::vector<MeshBuf> flat(nn*W);
                for(uint32_t b=0;b<nn;++b) for(uint32_t i=0;i<W;++i) flat[b*W+i]=cts[b][i];

                auto aut  = gather(); eng.permute_batch(flat, tabJ[a], aut,  qs);   // automorph(c)
                auto shf  = gather(); eng.permute_batch(flat, tabK[a], shf,  qs);   // monomul(c)
                auto autS = gather(); eng.permute_batch(shf,  tabJ[a], autS, qs);   // automorph(c_sh)

                std::vector<std::vector<MeshBuf>> nw(2*nn);
                for(uint32_t b=0;b<nn;++b){
                    std::vector<MeshBuf> pa(W), ps(W), pas(W);
                    for(uint32_t i=0;i<W;++i){ pa[i]=aut[b*W+i]; ps[i]=shf[b*W+i]; pas[i]=autS[b*W+i]; }
                    nw[b]      = ctadd(cts[b], keyswitch(pa,  KnbL[a]));
                    nw[b+nn]   = ctadd(ps,     keyswitch(pas, KnbL[a])); }
                cts=std::move(nw); }
            return cts; };                       // buffers: the tree never returns to host

        // Reading the selectors back is VALIDATION SCAFFOLDING, not part of expansion: in the real pipeline
        // they feed unpack/answer on device. It is 96 of the 126 readbacks, so leaving it inside the timed
        // region would attribute test overhead to the algorithm. Timed separately.
        auto fetch=[&](const std::vector<std::vector<MeshBuf>>& cts){
            std::vector<CtV> out(cts.size());
            for(uint32_t k=0;k<cts.size();++k){ out[k].resize(2*K);
                for(uint32_t i=0;i<2*K;++i) out[k][i]=unt2(eng.rdb(cts[k][i])); }
            return out; };

        eng.CACHE_ON = true; eng.CACHE_REC = true;
        pir2048::prof::reset();
        auto t_rec0 = eclk::now();
        auto ctsRecB = run_expand(); eng.finish(); auto ctsRec = fetch(ctsRecB);
        const double rec_ms = std::chrono::duration<double, std::milli>(eclk::now() - t_rec0).count();
        eng.CACHE_REC = false;
        fmt::print("  EXPAND RECORD (m_exp={} levels -> {} selectors): {:.1f} ms  ({} workloads, {} pool bufs)\n",
                   m_exp, N0, rec_ms, eng.n_cached(), eng.POOL.size());
        pir2048::prof::report("expand record");

        eng.TR_PHASE = 1;                       // capture RECORDS, does not EXECUTE -> discard its output
        auto ctsCap = run_expand(); (void)ctsCap;
        eng.close_capture();
        eng.TR_PHASE = 2;
        fmt::print("  EXPAND CAPTURE: {} traces from {} workloads\n", eng.n_segments(), eng.n_cached());

        int eiters = 3; if (const char* e = std::getenv("PIR_ITERS")) { int v = std::atoi(e); if (v > 0) eiters = v; }
        std::vector<double> eres; std::vector<CtV> cts;
        for (int i = 0; i < eiters; ++i) {
            if (i == eiters - 1) pir2048::prof::reset();
            auto t = eclk::now(); auto ctsB = run_expand(); eng.finish();   // drain: tree has no blocking op
            eres.push_back(std::chrono::duration<double, std::milli>(eclk::now() - t).count());
            if (i == eiters - 1) cts = fetch(ctsB); }
        double emin = eres[0], emax = eres[0], esum = 0;
        for (double v : eres) { emin = std::min(emin, v); emax = std::max(emax, v); esum += v; }
        std::vector<double> esort = eres; std::sort(esort.begin(), esort.end());
        const double emed = esort[esort.size()/2];
        // min alone hides a long tail; for a serving workload the median/tail is what users see.
        fmt::print("  EXPAND REPLAY x{}: min={:.1f}  median={:.1f}  mean={:.1f}  max={:.1f} ms  (spread {:.1f}%)   {:.2f}x vs record\n",
                   eiters, emin, emed, esum/eres.size(), emax, 100.0*(emax-emin)/emin, rec_ms/emin);
        pir2048::prof::report("expand replay");
        { uint32_t rr=0; for(uint32_t k=0;k<N0;++k) for(uint32_t i=0;i<2*K;++i) for(uint32_t x=0;x<N;++x)
              if(cts[k][i][x]!=ctsRec[k][i][x]) ++rr;
          fmt::print("  expand replay-vs-record: {}/{} mismatches\n", rr, N0*2*K*N); pass &= (rr==0); }
        // MUST leave replay mode before the unpack section below: with CACHE_ON/TR_PHASE still set, the
        // unpack's ops would replay the EXPAND's cached workloads instead of building their own (observed
        // as unpack 98304/98304). The cache is keyed by dispatch ORDER, not by op identity.
        eng.CACHE_ON = false; eng.CACHE_REC = false; eng.TR_PHASE = 0;
        uint32_t smis=0, schk=0;
        for(uint32_t k=0;k<N0;++k) for(uint32_t i=0;i<2*K;++i) for(uint32_t x=0;x<N;++x){ if(cts[k][i][x]!=selG[k][i][x]){ if(smis<8) fmt::print("  sel{} comp{} ch{} [{}] got={} exp={}\n",k,i/K,i%K,x,cts[k][i][x],selG[k][i][x]); ++smis; } ++schk; }
        fmt::print("query_expand 2048 (device, K={})  selector mismatches={}/{}\n", K, smis, schk); pass &= (smis==0);

        // (b) UNPACK: RGSW(s) ⊡ msg row -> golden bottom RGSW rows
        uint32_t umis=0, uchk=0;
        for(uint32_t k=0;k<ell;++k){ CtV ep=extprod(msg[k], GSn);
            for(uint32_t i=0;i<2*K;++i) for(uint32_t x=0;x<N;++x){ if(ep[i][x]!=botG[k][i][x]){ if(umis<8) fmt::print("  unpack{} col{} ch{} [{}] got={} exp={}\n",k,i/K,i%K,x,ep[i][x],botG[k][i][x]); ++umis; } ++uchk; } }
        fmt::print("query_unpack 2048 (device, K={})  bottom-row mismatches={}/{}\n", K, umis, uchk); pass &= (umis==0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
