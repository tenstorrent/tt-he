// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PIR ANSWER pipeline (#19) on device at n=2048, K RNS channels — the initial-dimension scan +
// binary hypercube folds that turn the expanded selectors into the response ct recovering DB[idx].
// Thin orchestration over pir2048::Engine (expand/unpack validated separately).  K from the header.
// Bit-exact vs cpu/pir/gen_pir_answer_case.py (onionpir_ref.pir_answer).
//
//   ./build/test_pir_answer2048 [reference/pir/pir_answer_n2048.bin]   # K from the file

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
using CtV = std::vector<std::vector<uint32_t>>;   // 2*K polys [comp*K+ch]

// One PANS case. Split out so a SECOND case (same structure, different query index) can be loaded and
// replayed on the pipeline recorded for the first — the check that the cached workloads depend only on
// DB *structure*, not on the query or DB contents.
struct Case {
    uint32_t n{}, ell{}, B{}, K{}, t{}, ndims{}, N0{}, nfold{}, idx{}, Ncells{1}, rest{};
    std::vector<uint32_t> dims, primes;
    std::vector<std::vector<std::vector<uint32_t>>> db;      // [cell][ch]
    std::vector<CtV> sel0;                                   // [k][comp*K+ch]
    std::vector<std::vector<std::vector<std::vector<uint32_t>>>> sb;   // [f][row*2+col][ch]
    CtV ansG;
    std::vector<int32_t> sk;                                 // secret key (for the decrypt-check)
    std::vector<int32_t> dbidx;                              // plaintext DB[idx]
    std::string dimstr() const { std::string s; for (uint32_t i=0;i<ndims;++i) s += (i?",":"") + std::to_string(dims[i]); return s; }
    bool golden_is_zero() const {                            // true when generated with --no-golden
        for (const auto& p : ansG) for (uint32_t v : p) if (v) return false; return true; }
};

// Client decrypt, mirroring onionpir_ref.decrypt: phase = c0 + c1*s per RNS channel, Garner-CRT the K
// residues up to the ~84-bit q, then round(v / delta) % t on the BALANCED representative (delta = q//t).
// This is what makes 128 MB-class validation possible: with each slot storing its own index, a correct
// retrieval decrypts to the index requested, so no Python golden answer is needed at all.
// (Rounding here is half-away-from-zero vs Python's banker's rounding; ties are measure-zero under noise.)
std::vector<uint32_t> pir_decrypt(const Case& c, const CtV& ct) {
    using u128 = unsigned __int128; using i128 = __int128;
    const uint32_t K = c.K, n = c.n;
    std::vector<std::vector<uint64_t>> ph(K, std::vector<uint64_t>(n, 0));
    for (uint32_t ch = 0; ch < K; ++ch) {
        const uint64_t q = c.primes[ch];
        std::vector<uint64_t> acc(n, 0);
        for (uint32_t i = 0; i < n; ++i) {                     // negacyclic c1 * s
            const uint64_t a = ct[1*K+ch][i] % q; if (!a) continue;
            for (uint32_t j = 0; j < n; ++j) {
                const int64_t sj = c.sk[j]; if (!sj) continue;
                const uint64_t mag = (uint64_t)(sj < 0 ? -sj : sj);
                const uint64_t prod = (a * mag) % q;
                uint32_t k = i + j; bool neg = (sj < 0);
                if (k >= n) { k -= n; neg = !neg; }            // x^n = -1
                acc[k] = neg ? (acc[k] + q - prod) % q : (acc[k] + prod) % q; } }
        for (uint32_t i = 0; i < n; ++i) ph[ch][i] = (acc[i] + ct[0*K+ch][i]) % q;
    }
    u128 q = 1; for (uint32_t p : c.primes) q *= p;
    const u128 delta = q / c.t, half = q >> 1;
    std::vector<uint32_t> out(n);
    for (uint32_t i = 0; i < n; ++i) {
        u128 x = ph[0][i], radix = c.primes[0];                // Garner CRT
        for (uint32_t j = 1; j < K; ++j) {
            const uint64_t p = c.primes[j];
            const uint64_t xm = (uint64_t)(x % p);
            const uint64_t diff = (ph[j][i] + p - xm) % p;
            const uint64_t inv = mont::modular_inverse((uint32_t)(radix % p), (uint32_t)p);
            const uint64_t kk = (uint64_t)((u128)diff * inv % p);
            x += radix * kk; radix *= p; }
        const i128 v = (x > half) ? (i128)x - (i128)q : (i128)x;
        i128 m = (v >= 0) ? (i128)(((u128)v + delta/2) / delta)
                          : -(i128)(((u128)(-v) + delta/2) / delta);
        m %= (i128)c.t; if (m < 0) m += c.t;
        out[i] = (uint32_t)m; }
    return out;
}

bool load_case(const char* path, Case& c) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return false; }
    auto ru = [&](uint32_t& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    uint32_t magic, ver;
    ru(magic); ru(ver); ru(c.n); ru(c.ell); ru(c.B); ru(c.K); ru(c.t); ru(c.ndims); ru(c.N0); ru(c.nfold); ru(c.idx);
    if (magic != 0x50414E53u || ver != 1u || c.n != N) { fmt::print(stderr, "bad case (need PANS n=2048)\n"); return false; }
    c.dims.resize(c.ndims); f.read(reinterpret_cast<char*>(c.dims.data()), 4 * c.ndims);
    c.primes.resize(c.K);   f.read(reinterpret_cast<char*>(c.primes.data()), 4 * c.K);
    c.Ncells = 1; for (auto d : c.dims) c.Ncells *= d;
    c.rest = c.Ncells / c.N0;
    auto rv = [&]() { std::vector<uint32_t> v(N); f.read(reinterpret_cast<char*>(v.data()), 4*N); return v; };
    c.sk.resize(N); f.read(reinterpret_cast<char*>(c.sk.data()), 4*N);                    // secret key
    c.db.assign(c.Ncells, std::vector<std::vector<uint32_t>>(c.K));
    for (uint32_t x = 0; x < c.Ncells; ++x) for (uint32_t ch = 0; ch < c.K; ++ch) c.db[x][ch] = rv();
    c.sel0.assign(c.N0, CtV{});
    for (uint32_t k = 0; k < c.N0; ++k) { c.sel0[k].resize(2*c.K);
        for (uint32_t comp = 0; comp < 2; ++comp) for (uint32_t ch = 0; ch < c.K; ++ch) c.sel0[k][comp*c.K+ch] = rv(); }
    const uint32_t twoL = 2*c.ell;
    c.sb.assign(c.nfold, std::vector<std::vector<std::vector<uint32_t>>>(twoL*2, std::vector<std::vector<uint32_t>>(c.K)));
    for (uint32_t fdx = 0; fdx < c.nfold; ++fdx) for (uint32_t row = 0; row < twoL; ++row) for (uint32_t col = 0; col < 2; ++col)
        for (uint32_t ch = 0; ch < c.K; ++ch) c.sb[fdx][row*2+col][ch] = rv();
    c.ansG.assign(2*c.K, std::vector<uint32_t>{});
    for (uint32_t comp = 0; comp < 2; ++comp) for (uint32_t ch = 0; ch < c.K; ++ch) c.ansG[comp*c.K+ch] = rv();
    c.dbidx.resize(N); f.read(reinterpret_cast<char*>(c.dbidx.data()), 4*N);              // plaintext DB[idx]
    fmt::print("case {}: n={} ell={} B={} K={} dims=[{}] Ncells={} N0={} rest={} idx={}\n",
               path, c.n, c.ell, c.B, c.K, c.dimstr(), c.Ncells, c.N0, c.rest, c.idx);
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "reference/pir/pir_answer_n2048.bin";
    Case A; if (!load_case(path, A)) return 2;
    // Optional 2nd case: SAME structure, different query. Replayed on the pipeline recorded for case A.
    Case Bq; bool have2 = false;
    if (argc > 2) {
        if (!load_case(argv[2], Bq)) return 2;
        if (Bq.n != A.n || Bq.ell != A.ell || Bq.K != A.K || Bq.ndims != A.ndims || Bq.N0 != A.N0 || Bq.dims != A.dims) {
            fmt::print(stderr, "2nd case must share STRUCTURE (n, ell, K, dims) with the 1st\n"); return 2; }
        have2 = true;
    }
    const uint32_t ell = A.ell, B = A.B, K = A.K, ndims = A.ndims, N0 = A.N0, nfold = A.nfold;
    const uint32_t Ncells = A.Ncells, rest = A.rest, twoL = 2*ell;
    const std::string dimstr = A.dimstr();
    auto& primes = A.primes;
    // run_answer() reads these by reference, so swapping in case B's contents changes the QUERY without
    // rebuilding anything -- exactly the deployment model (recompile only when the DB structure changes).
    auto& db = A.db; auto& sel0 = A.sel0; auto& sb = A.sb; auto& ansG = A.ansG;
    (void)B; (void)nfold;

    bool pass = true;
    try {
        // 256 MB trace region reserved at init -- required before any trace capture (HE T1).
        auto mesh = distributed::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 256u << 20);
        auto& cq = mesh->mesh_command_queue();
        Engine eng(mesh.get(), &cq, primes, ell, B);

        // CtV coeff-domain add/sub per channel
        auto ctsub=[&](const CtV& a, const CtV& b){ CtV r(2*K); for(uint32_t c=0;c<2;++c) for(uint32_t ch=0;ch<K;++ch){ uint32_t qc=primes[ch]; auto& ra=r[c*K+ch]; ra.resize(N);
            for(uint32_t x=0;x<N;++x) ra[x]=(a[c*K+ch][x]+qc-b[c*K+ch][x])%qc; } return r; };
        auto ctadd=[&](const CtV& a, const CtV& b){ CtV r(2*K); for(uint32_t c=0;c<2;++c) for(uint32_t ch=0;ch<K;++ch){ uint32_t qc=primes[ch]; auto& ra=r[c*K+ch]; ra.resize(N);
            for(uint32_t x=0;x<N;++x) ra[x]=(a[c*K+ch][x]+b[c*K+ch][x])%qc; } return r; };
        auto slice=[&](const CtV& ct, uint32_t comp){ std::vector<std::vector<uint32_t>> r(K); for(uint32_t ch=0;ch<K;++ch) r[ch]=ct[comp*K+ch]; return r; };
        // external product RGSW(Cn, NTT-mont) ⊡ RLWE d -> RLWE
        auto extprod=[&](const CtV& d, const std::vector<std::vector<MeshBuf>>& Cnb)->CtV{
            auto dg0=eng.crt_and_gadget(slice(d,0)), dg1=eng.crt_and_gadget(slice(d,1));
            std::vector<std::vector<uint32_t>> digU(twoL); for(uint32_t k=0;k<ell;++k){ digU[k]=dg0[k]; digU[ell+k]=dg1[k]; }
            // Digit transforms stay RESIDENT: write the reshaped digit into a packed 2-tile buffer and
            // transform in place, so no freq-domain poly is read back to host only to be re-uploaded for
            // the MAC (that round trip was 2*twoL*K readbacks + the same in writes, per extprod).
            std::vector<std::vector<MeshBuf>> dNb(K, std::vector<MeshBuf>(twoL));
            // NOTE: eng.digit_rns_batch() co-schedules all twoL splits into one program. Measured ~3% SLOWER
            // here (only 48 of 5619 dispatches are digit_rns, and the consolidated readback lands on the
            // critical path now that device time is exposed), so the per-digit form is kept. The batch form
            // is retained in the engine for EXPANSION, which has ~4088 such calls (8 per key-switch x 511)
            // -- re-measure there rather than assuming it transfers.
            for(uint32_t r=0;r<twoL;++r){ auto res=eng.digit_rns(digU[r]);
                for(uint32_t ch=0;ch<K;++ch){ auto in2=eng.mkbuf(2,I32); eng.write_fwd_in2(in2, res[ch]);
                    dNb[ch][r]=eng.mkbuf(2,I32); eng.fwd2048_buf2(eng.CH[ch], in2, dNb[ch][r]); } }
            // Resident MAC. The naive form did pmul() per row (a blocking readback each) then padd()
            // (two more), i.e. 2*twoL*K + 4*K readbacks per extprod. mac_ntt writes every product into ONE
            // DRAM buffer at a tile offset and sums it with a strided modadd, so each (col,ch) costs a
            // single readback. Operands must already be in Montgomery/NTT form, which dN and Cnb are.
            CtV ep(2*K);
            for(uint32_t col=0;col<2;++col) for(uint32_t ch=0;ch<K;++ch){ std::vector<MeshBuf> bcol(twoL);
                for(uint32_t r=0;r<twoL;++r) bcol[r]=Cnb[ch][r*2+col];
                // MAC -> transpose -> inverse transform, all in DRAM; ONE readback at the end.
                auto s2 = eng.mac_ntt_buf(eng.CH[ch], dNb[ch], bcol);
                auto s2t = eng.mkbuf(2,I32);
                for(uint32_t bk2=0;bk2<2;++bk2) eng.transpose_dev(s2, s2t, bk2, bk2);
                auto o2 = eng.mkbuf(2,I32);
                eng.inv2048_buf2(eng.CH[ch], s2t, o2);
                ep[col*K+ch]=eng.read_inv_out2(o2); }
            return ep; };

        using clk = std::chrono::high_resolution_clock;
        double scan_ms = 0, fold_ms = 0;
        // The whole answer as ONE callable, so it can be RECORDED once and then REPLAYED per query
        // (T2 of docs/he/history/OPTIMIZATION_REPORT.md): pass 1 builds and caches every dispatch; later passes
        // re-enqueue the cached workloads and skip program construction entirely. Scratch comes from the
        // engine's fixed-address pool (begin_pass rewinds it), so replayed runtime args still address the
        // right buffers. The replay cost is what a PIR server actually serves queries at.
        auto run_answer = [&]() -> CtV {
        eng.begin_pass();
        auto t_scan0 = clk::now();
        // ---- (b) initial-dimension scan: acc[r][comp] = Σ_k DB[k*rest+r] ⊛ sel0[k][comp] ----
        // NTT the DB (Montgomery) + selectors (plain) once, MAC pointwise, iNTT back.
        // C2: the first dimension as a batched modular GEMM on the int8 matrix engine when the shape allows
        // (m=rest and k=N0 must be multiples of 32). Falls back to the pointwise MAC otherwise, so the
        // small bit-exact case still exercises the original path as the primary correctness gate.
        const bool USE_GEMM = (rest % T == 0) && (N0 % T == 0);
        // NOTE: the GEMM computes a PLAIN modular product, whereas pmul is a Montgomery modmul -- so the
        // GEMM path must NOT pre-Montgomery the DB, or every product is off by a factor of R.
        std::vector<std::vector<std::vector<uint32_t>>> DBntt(K, std::vector<std::vector<uint32_t>>(Ncells));
        for (uint32_t ch=0; ch<K; ++ch) for (uint32_t c=0;c<Ncells;++c){ auto f=eng.fwd2048(eng.CH[ch], db[c][ch]);
            DBntt[ch][c] = USE_GEMM ? f : eng.to_mont(ch, f); }
        std::vector<std::vector<std::vector<uint32_t>>> Sntt(K, std::vector<std::vector<uint32_t>>(N0*2));
        for (uint32_t ch=0; ch<K; ++ch) for (uint32_t k=0;k<N0;++k) for (uint32_t comp=0;comp<2;++comp) Sntt[ch][k*2+comp]=eng.fwd2048(eng.CH[ch], sel0[k][comp*K+ch]);
        std::vector<CtV> acc(rest);
        for (uint32_t r=0;r<rest;++r) acc[r].resize(2*K);

        if (USE_GEMM) {
            const uint32_t MT = rest/T, KB = N0/T, NOUT = N*MT;
            auto tl8=[&](const std::vector<int8_t>& in){ std::vector<int8_t> o(ELEMS);
                for(uint32_t r=0;r<T;++r) for(uint32_t c=0;c<T;++c)
                    o[((r/16)*2+c/16)*256+(r%16)*16+(c%16)]=in[r*T+c]; return o; };
            auto lm=[&](uint32_t v,uint32_t l){ return (int8_t)((v>>(7*l))&0x7F); };
            for (uint32_t ch=0; ch<K; ++ch) {
                // A[slot][mt][kt][limb] : A_tile[i][j] = DB[(kt*T+j)*rest + (mt*T+i)][slot]
                std::vector<int8_t> Ab((size_t)NOUT*KB*NB*ELEMS, 0), Bb((size_t)N*KB*NB*ELEMS, 0);
                for (uint32_t sl=0; sl<N; ++sl) for (uint32_t mt=0; mt<MT; ++mt) for (uint32_t kt=0; kt<KB; ++kt)
                    for (uint32_t l=0;l<NB;++l){ std::vector<int8_t> nat(ELEMS,0);
                        for(uint32_t i=0;i<T;++i) for(uint32_t j=0;j<T;++j)
                            nat[i*T+j]=lm(DBntt[ch][(kt*T+j)*rest + (mt*T+i)][sl], l);
                        auto t=tl8(nat);
                        std::copy(t.begin(),t.end(), Ab.begin()+((((size_t)sl*MT+mt)*KB+kt)*NB+l)*ELEMS); }
                // B[slot][kt][limb] : B_tile[i][c] = SEL[(kt*T+i)*2 + c][slot], cols >= 2 zero
                for (uint32_t sl=0; sl<N; ++sl) for (uint32_t kt=0; kt<KB; ++kt)
                    for (uint32_t l=0;l<NB;++l){ std::vector<int8_t> nat(ELEMS,0);
                        for(uint32_t i=0;i<T;++i) for(uint32_t c=0;c<2;++c)
                            nat[i*T+c]=lm(Sntt[ch][(kt*T+i)*2+c][sl], l);
                        auto t=tl8(nat);
                        std::copy(t.begin(),t.end(), Bb.begin()+(((size_t)sl*KB+kt)*NB+l)*ELEMS); }
                auto bA=eng.mkbuf((uint32_t)(NOUT*KB*NB), I8), bB=eng.mkbuf((uint32_t)(N*KB*NB), I8);
                distributed::EnqueueWriteMeshBuffer(cq,bA,Ab,false); distributed::EnqueueWriteMeshBuffer(cq,bB,Bb,false);
                auto bP=eng.mkbuf(NOUT*8, I32);
                eng.scan_gemm(bA,bB,bP,NOUT,KB,MT);
                auto bR=eng.scan_recombine(eng.CH[ch], bP, NOUT);
                auto raw=eng.rdb(bR);
                // gather: tile (slot,mt) element [i][c] is coefficient `slot` of eval poly (r=mt*T+i, comp=c)
                auto ut=[&](const uint32_t* in){ std::vector<uint32_t> o(ELEMS);
                    for(uint32_t r=0;r<T;++r) for(uint32_t c=0;c<T;++c)
                        o[r*T+c]=in[((r/16)*2+c/16)*256+(r%16)*16+(c%16)]; return o; };
                std::vector<std::vector<std::vector<uint32_t>>> ev(rest, std::vector<std::vector<uint32_t>>(2, std::vector<uint32_t>(N,0)));
                for (uint32_t sl=0; sl<N; ++sl) for (uint32_t mt=0; mt<MT; ++mt){
                    auto nat=ut(&raw[((size_t)sl*MT+mt)*ELEMS]);
                    for(uint32_t i=0;i<T;++i) for(uint32_t c=0;c<2;++c) ev[mt*T+i][c][sl]=nat[i*T+c]; }
                for (uint32_t r=0;r<rest;++r) for (uint32_t comp=0;comp<2;++comp)
                    acc[r][comp*K+ch]=eng.inv2048(eng.CH[ch], ev[r][comp]);
            }
        } else {
            for (uint32_t r=0;r<rest;++r)
                for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<K;++ch){ std::vector<std::vector<uint32_t>> prods(N0);
                    for (uint32_t k=0;k<N0;++k) prods[k]=eng.pmul(eng.CH[ch], Sntt[ch][k*2+comp], DBntt[ch][k*rest+r]);
                    acc[r][comp*K+ch]=eng.inv2048(eng.CH[ch], eng.padd(eng.CH[ch], prods)); }
        }
        auto t_scan1 = clk::now();

        // ---- (c) fold the higher dims: acc[r] = mux(C_bit, acc[r], acc[half+r]) ----
        for (uint32_t dim_idx=1; dim_idx<ndims; ++dim_idx){
            const auto& fb = sb[dim_idx-1];                          // fb[row*2+col][ch] = poly
            // RGSW rows: resident, transformed ONCE per fold, and in the SAME packed layout as the digit
            // transforms -- required, because the MAC is elementwise and both operands must share a layout.
            // to_mont moves to the coefficient-domain INPUT: valid because to_mont is scaling by R and the
            // NTT is linear mod q, verified 0/2048 as check [5] of test_ntt2048_res.
            std::vector<std::vector<MeshBuf>> Cnb(K, std::vector<MeshBuf>(twoL*2));
            for (uint32_t ch=0;ch<K;++ch) for (uint32_t rc=0;rc<twoL*2;++rc){
                auto in2=eng.mkbuf(2,I32); eng.write_fwd_in2(in2, eng.to_mont(ch, fb[rc][ch]));
                Cnb[ch][rc]=eng.mkbuf(2,I32); eng.fwd2048_buf2(eng.CH[ch], in2, Cnb[ch][rc]); }
            uint32_t half=(uint32_t)acc.size()/2; std::vector<CtV> nacc(half);
            for (uint32_t r=0;r<half;++r) nacc[r]=ctadd(extprod(ctsub(acc[half+r], acc[r]), Cnb), acc[r]);
            acc=std::move(nacc);
        }
        auto t_fold1 = clk::now();
        scan_ms = std::chrono::duration<double, std::milli>(t_scan1 - t_scan0).count();
        fold_ms = std::chrono::duration<double, std::milli>(t_fold1 - t_scan1).count();
        return acc[0];
        };   // end run_answer

        // ---- pass 1: RECORD (build + cache every dispatch) ----
        eng.CACHE_ON = true; eng.CACHE_REC = true;
        pir2048::prof::reset();
        auto ansRec = run_answer();
        eng.CACHE_REC = false;
        const double rec_ms = scan_ms + fold_ms;
        fmt::print("  RECORD pass: scan={:.1f} ms  folds={:.1f} ms  total={:.1f} ms   ({} workloads cached, {} pool buffers)\n",
                   scan_ms, fold_ms, rec_ms, eng.n_cached(), eng.POOL.size());
        pir2048::prof::report("record");

        // ---- pass 1b: CAPTURE the cached dispatch sequence as Metal Traces ----
        // Must come AFTER the record pass: the first enqueue of a fresh program writes its binary, and
        // trace forbids writes during capture. One trace per segment (maximal run with no host boundary).
        eng.TR_PHASE = 1;
        auto ansCap = run_answer();
        (void)ansCap;   // capture RECORDS commands, it does not EXECUTE them -> its output is meaningless
        eng.close_capture();
        fmt::print("  CAPTURE pass: {} traces from {} cached workloads (output intentionally discarded)\n",
                   eng.n_segments(), eng.n_cached());
        eng.TR_PHASE = 2;

        // ---- pass 2: REPLAY, timed over several iterations ----
        // A single replay pass is NOT a usable measuring stick: the enqueue term alone swings ~12%
        // run-to-run, which hides changes of ~25 ms and would let us "optimize" against noise. So report
        // min/mean/max over PIR_ITERS iterations (min = least host-interference estimate of the real cost)
        // and take the phase split from a single clean final iteration.
        int iters = 5; if (const char* e = std::getenv("PIR_ITERS")) { int v = std::atoi(e); if (v > 0) iters = v; }
        std::vector<double> reps; reps.reserve(iters);
        CtV acc0;
        for (int i = 0; i < iters; ++i) {
            if (i == iters - 1) pir2048::prof::reset();       // profile only the last iteration
            acc0 = run_answer();
            reps.push_back(scan_ms + fold_ms);
        }
        double rmin = reps[0], rmax = reps[0], rsum = 0;
        for (double v : reps) { rmin = std::min(rmin, v); rmax = std::max(rmax, v); rsum += v; }
        const double rep_ms = rmin, rmean = rsum / reps.size();
        fmt::print("  REPLAY x{}: min={:.1f} ms  mean={:.1f} ms  max={:.1f} ms  (spread {:.1f}%)   min is {:.2f}x faster than record\n",
                   iters, rmin, rmean, rmax, rmin > 0 ? 100.0 * (rmax - rmin) / rmin : 0.0,
                   rep_ms > 0 ? rec_ms / rep_ms : 0.0);
        pir2048::prof::report("replay (last iter)");

        // replay must be bit-identical to the recorded pass (same inputs, same fixed buffers)
        uint32_t rrm = 0;
        for (uint32_t i=0;i<2*K;++i) for (uint32_t x=0;x<N;++x) if (acc0[i][x]!=ansRec[i][x]) ++rrm;
        fmt::print("  replay-vs-record: {}/{} mismatches\n", rrm, 2*K*N);
        pass &= (rrm==0);

        // ---- compare final answer bit-exact vs golden ----
        uint32_t mism=0, checked=0;
        for (uint32_t i=0;i<2*K;++i) for (uint32_t x=0;x<N;++x){ if (acc0[i][x]!=ansG[i][x]){ if (mism<8) fmt::print("  comp{} ch{} [{}] got={} exp={}\n",i/K,i%K,x,acc0[i][x],ansG[i][x]); ++mism; } ++checked; }
        if (A.golden_is_zero()) {
            fmt::print("pir_answer 2048 (device, K={}, dims=[{}])  golden ABSENT (--no-golden) -- "
                       "skipping bit-exact compare, relying on the decrypt-check below\n", K, dimstr);
        } else {
            fmt::print("pir_answer 2048 (device, K={}, dims=[{}])  mismatches={}/{}\n", K, dimstr, mism, checked);
            pass &= (mism==0);
        }

        // ---- DECRYPT-CHECK: the validation that scales to 128 MB (no golden answer needed) ----
        // Decrypt the device's own response and compare against the plaintext DB[idx] carried in the case.
        // Gated here against a case that DOES have a golden, so the harness itself is proven before being
        // relied on where no golden exists.
        {
            auto dec = pir_decrypt(A, acc0);
            uint32_t dm = 0, dn = 0;
            for (uint32_t i = 0; i < N; ++i) { uint32_t want = (uint32_t)((A.dbidx[i] % (int32_t)A.t + (int32_t)A.t) % (int32_t)A.t);
                if (dec[i] != want) { if (dm < 6) fmt::print("  [decrypt] coeff{} got={} exp={}\n", i, dec[i], want); ++dm; } ++dn; }
            fmt::print("  DECRYPT-CHECK  decrypt(device answer) vs plaintext DB[{}]: mismatches={}/{}\n",
                       A.idx, dm, dn);
            fmt::print("               (coeff0: got={} expected={})\n", dec[0],
                       (uint32_t)((A.dbidx[0] % (int32_t)A.t + (int32_t)A.t) % (int32_t)A.t));
            pass &= (dm==0);
        }

        // ---- pass 3: NEW QUERY on the pipeline recorded for case A (nothing rebuilt) ----
        // The claim being tested: cached workloads bake in DB *structure* only. Selectors and DB contents
        // arrive as untraced buffer writes, so a different query must replay correctly with zero rebuild.
        // If this passes, the record pass is a one-time compile cost amortized over every query served.
        if (have2) {
            db = Bq.db; sel0 = Bq.sel0; sb = Bq.sb;          // swap the QUERY (and DB contents), keep structure
            pir2048::prof::reset();
            auto accB = run_answer();
            fmt::print("  NEW-QUERY replay (idx {} -> idx {}): total={:.1f} ms  ({} workloads, 0 rebuilt)\n",
                       A.idx, Bq.idx, scan_ms + fold_ms, eng.n_cached());
            uint32_t bm = 0, bc = 0;
            for (uint32_t i=0;i<2*K;++i) for (uint32_t x=0;x<N;++x){
                if (accB[i][x]!=Bq.ansG[i][x]){ if (bm<8) fmt::print("  [newq] comp{} ch{} [{}] got={} exp={}\n",i/K,i%K,x,accB[i][x],Bq.ansG[i][x]); ++bm; } ++bc; }
            fmt::print("  NEW-QUERY vs its own golden: mismatches={}/{}\n", bm, bc);
            pass &= (bm==0);
            // sanity: the two queries must give DIFFERENT answers, else we'd be comparing a stale result
            uint32_t diff = 0; for (uint32_t i=0;i<2*K;++i) for (uint32_t x=0;x<N;++x) if (accB[i][x]!=acc0[i][x]) ++diff;
            fmt::print("  NEW-QUERY differs from first answer in {}/{} coeffs (must be > 0)\n", diff, bc);
            pass &= (diff > 0);
        } else {
            fmt::print("  (pass a 2nd case file as argv[2] to test a NEW QUERY on the recorded pipeline)\n");
        }
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
