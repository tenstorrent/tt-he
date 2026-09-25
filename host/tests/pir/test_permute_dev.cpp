// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// On-device Galois automorphism (#14) + monomial multiply (#15), bit-exact vs the host references
// (automorph_poly / mono_mul_poly in pir_permute.hpp).
//
// These were the last host operations inside the Subs tree, forcing a readback per key-switch output —
// ~79% of expansion's remaining latency. Both are the same primitive: a fixed signed permutation of the N
// coefficients. Implemented as a dataflow kernel doing an indexed L1 copy (dataflow kernels are plain C++
// on a RISC-V core with direct L1 pointer access), with the permutation precomputed on host in TILIZED
// coordinates so one table serves every node at a Subs level.
//
// Covers the automorphism indices the Subs tree actually uses, j = (N >> a) + 1 for each level a, plus the
// monomial shifts k = -(1 << a), plus edge cases (j=1 identity, k=0, k=±N which is pure negation).
//
//   ./build/test_permute_dev [q] [seed]

#include <cstdint>
#include <random>
#include <vector>
#include "pir_engine2048.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace pir2048;
static constexpr uint32_t ELEMS_T = 1024;

namespace {
// Host references (same as test_query_expand2048's). NOTE both are bijections only when the index map is
// injective on 0..N-1: for automorph that needs j ODD (coprime to 2N) -- which is exactly what the Subs
// tree uses, j = (N>>a)+1. The device gather table assumes that bijection (every output written once).
std::vector<uint32_t> automorph_ref(const std::vector<uint32_t>& poly, int64_t j, uint32_t q) {
    const int64_t twon = 2*(int64_t)N; std::vector<uint32_t> r(N, 0);
    for (uint32_t i = 0; i < N; ++i) { int64_t p = ((int64_t)i*j) % twon; if (p < 0) p += twon; uint32_t v = poly[i];
        if (p < (int64_t)N) r[p] = (uint32_t)(((uint64_t)r[p]+v)%q); else r[p-N] = (uint32_t)(((uint64_t)r[p-N]+q-v)%q); } return r; }
std::vector<uint32_t> monomul_ref(const std::vector<uint32_t>& poly, int64_t k, uint32_t q) {
    const int64_t twon = 2*(int64_t)N; std::vector<uint32_t> r(N, 0);
    for (uint32_t i = 0; i < N; ++i) { int64_t p = ((int64_t)i+k) % twon; if (p < 0) p += twon; uint32_t v = poly[i];
        if (p < (int64_t)N) r[p] = (uint32_t)(((uint64_t)r[p]+v)%q); else r[p-N] = (uint32_t)(((uint64_t)r[p-N]+q-v)%q); } return r; }
}  // namespace

int main(int argc, char** argv) {
    const uint32_t q    = (argc > 1) ? std::stoul(argv[1]) : 268369921u;
    const uint32_t seed = (argc > 2) ? std::stoul(argv[2]) : 3u;
    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        Engine eng(mesh.get(), &cq, {q}, 3, 1024);

        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> di(0, q - 1);
        std::vector<uint32_t> a(N);
        for (auto& v : a) v = di(rng);
        a[0] = 0; a[1] = q - 1; a[N-1] = 1;              // exercise the v==0 and near-q paths

        uint32_t total_mis = 0;

        // ---- automorphism: j = 1 (identity), then the Subs tree's j = (N >> lvl) + 1 ----
        std::vector<uint64_t> js{1};
        for (uint32_t lvl = 0; lvl < 5; ++lvl) js.push_back((uint64_t)(N >> lvl) + 1);
        for (uint64_t j : js) {
            auto tab = eng.perm_table_buf(eng.perm_table_automorph(j));
            auto got = eng.permute_host(a, tab, q);
            auto exp = automorph_ref(a, (int64_t)j, q);
            uint32_t mis = 0; for (uint32_t x = 0; x < N; ++x) if (got[x] != exp[x]) ++mis;
            fmt::print("  automorph j={:5}: mismatches={}/{}\n", j, mis, N);
            total_mis += mis;
        }

        // ---- monomial multiply: k = 0, ±(1<<lvl), and ±N (pure negation) ----
        std::vector<int64_t> ks{0, (int64_t)N, -(int64_t)N};
        for (uint32_t lvl = 0; lvl < 4; ++lvl) { ks.push_back(1ll << lvl); ks.push_back(-(1ll << lvl)); }
        for (int64_t k : ks) {
            auto tab = eng.perm_table_buf(eng.perm_table_monomul(k));
            auto got = eng.permute_host(a, tab, q);
            auto exp = monomul_ref(a, k, q);
            uint32_t mis = 0; for (uint32_t x = 0; x < N; ++x) if (got[x] != exp[x]) ++mis;
            fmt::print("  monomul   k={:5}: mismatches={}/{}\n", k, mis, N);
            total_mis += mis;
        }

        // ---- two-buffer modular add (A4): out = (a+b) mod q ----
        { std::vector<uint32_t> b(N); for (auto& v : b) v = di(rng);
          b[0] = q - 1; b[1] = 1; b[2] = 0;                       // force the conditional-subtract path
          auto bA = eng.mkbuf(2, I32), bB = eng.mkbuf(2, I32), bO = eng.mkbuf(2, I32);
          eng.wrb(bA, til2(a)); eng.wrb(bB, til2(b));
          eng.padd_ab(eng.CH[0], bA, bB, bO);
          auto got = unt2(eng.rdb(bO));
          uint32_t mis = 0;
          for (uint32_t x = 0; x < N; ++x) { uint32_t exp = (uint32_t)(((uint64_t)a[x] + b[x]) % q);
              if (got[x] != exp) { if (mis < 4) fmt::print("  [modadd] {}: got={} exp={}\n", x, got[x], exp); ++mis; } }
          fmt::print("  modadd_ab (a+b) mod q      : mismatches={}/{}\n", mis, N);
          total_mis += mis; }

        // ---- C1: on-device mod-q recombine of scan_gemm's 8 packed planes ----
        // R = Σ_{d=0..3} 128^d·p_d + 128^4·Σ_{e=0..2} 128^e·p_{e+4}, via 2x recombine_bigK + modmul + add.
        { const uint32_t NOUT = 4, NPL = 8;
          std::uniform_int_distribution<uint32_t> dp(0, (1u<<20) - 1);   // matmul-plane magnitude
          std::vector<std::vector<uint32_t>> pl(NOUT*NPL, std::vector<uint32_t>(ELEMS_T, 0));
          for (uint32_t o=0;o<NOUT;++o) for (uint32_t d=0;d<7;++d)      // plane 7 stays zero
              for (uint32_t x=0;x<ELEMS_T;++x) pl[o*NPL+d][x] = dp(rng);
          auto bp = eng.mkbuf(NOUT*NPL, I32);
          { std::vector<uint32_t> blob((size_t)NOUT*NPL*ELEMS_T);
            for (uint32_t t=0;t<NOUT*NPL;++t) std::copy(pl[t].begin(), pl[t].end(), blob.begin()+(size_t)t*ELEMS_T);
            eng.wrb(bp, blob); }
          auto bR = eng.scan_recombine(eng.CH[0], bp, NOUT);
          auto raw = eng.rdb(bR);
          uint64_t w[8]; { uint64_t acc=1; for (int d=0;d<8;++d){ w[d]=acc; acc=acc*128%q; } }
          uint32_t mis=0;
          for (uint32_t o=0;o<NOUT;++o) for (uint32_t x=0;x<ELEMS_T;++x) {
              unsigned __int128 sum=0; for (uint32_t d=0;d<NPL;++d) sum += (unsigned __int128)w[d]*pl[o*NPL+d][x];
              uint32_t exp=(uint32_t)(sum % q), got=raw[(size_t)o*ELEMS_T+x];
              if (got!=exp) { if (mis<4) fmt::print("  [recomb] o{} [{}]: got={} exp={}\n",o,x,got,exp); ++mis; } }
          fmt::print("  scan_recombine (8 planes)  : mismatches={}/{}\n", mis, NOUT*ELEMS_T);
          total_mis += mis; }

        // ---- C2 tail: Engine::scan_gemm + scan_recombine vs a host GEMM (the full scan arithmetic) ----
        { const uint32_t T32=32, MT=1, KB=1, NS=4, NCOMP=2, NOUT=NS*MT;
          std::uniform_int_distribution<uint32_t> dq(0, q-1);
          std::vector<std::vector<std::vector<uint32_t>>> A(NS, std::vector<std::vector<uint32_t>>(T32, std::vector<uint32_t>(T32)));
          std::vector<std::vector<std::vector<uint32_t>>> Bm(NS, std::vector<std::vector<uint32_t>>(T32, std::vector<uint32_t>(NCOMP)));
          for(auto&sl:A) for(auto&r:sl) for(auto&v:r) v=dq(rng);
          for(auto&sl:Bm) for(auto&r:sl) for(auto&v:r) v=dq(rng);
          auto lm=[&](uint32_t v,uint32_t l){ return (int8_t)((v>>(7*l))&0x7F); };
          auto tl8=[&](const std::vector<int8_t>& in){ std::vector<int8_t> o(ELEMS_T);
              for(uint32_t r=0;r<T32;++r) for(uint32_t c=0;c<T32;++c)
                  o[((r/16)*2+c/16)*256+(r%16)*16+(c%16)]=in[r*T32+c]; return o; };
          std::vector<int8_t> Ab((size_t)NS*4*ELEMS_T,0), Bb((size_t)NS*4*ELEMS_T,0);
          for(uint32_t sl=0;sl<NS;++sl) for(uint32_t l=0;l<4;++l){
              std::vector<int8_t> na(ELEMS_T,0), nb(ELEMS_T,0);
              for(uint32_t i=0;i<T32;++i) for(uint32_t j=0;j<T32;++j) na[i*T32+j]=lm(A[sl][i][j],l);
              for(uint32_t i=0;i<T32;++i) for(uint32_t c=0;c<NCOMP;++c) nb[i*T32+c]=lm(Bm[sl][i][c],l);
              auto ta=tl8(na), tb=tl8(nb);
              std::copy(ta.begin(),ta.end(),Ab.begin()+((size_t)sl*4+l)*ELEMS_T);
              std::copy(tb.begin(),tb.end(),Bb.begin()+((size_t)sl*4+l)*ELEMS_T); }
          auto bA=eng.mkbuf(NS*4,1024), bB=eng.mkbuf(NS*4,1024), bP=eng.mkbuf(NOUT*8,I32);
          distributed::EnqueueWriteMeshBuffer(cq,bA,Ab,false); distributed::EnqueueWriteMeshBuffer(cq,bB,Bb,false);
          eng.scan_gemm(bA,bB,bP,NOUT,KB,MT,1);
          auto bR=eng.scan_recombine(eng.CH[0], bP, NOUT);
          auto raw=eng.rdb(bR);
          auto ut=[&](const std::vector<uint32_t>& in){ std::vector<uint32_t> o(ELEMS_T);
              for(uint32_t r=0;r<T32;++r) for(uint32_t c=0;c<T32;++c)
                  o[r*T32+c]=in[((r/16)*2+c/16)*256+(r%16)*16+(c%16)]; return o; };
          uint32_t mis=0;
          for(uint32_t sl=0;sl<NS;++sl){ std::vector<uint32_t> t(raw.begin()+(size_t)sl*ELEMS_T, raw.begin()+(size_t)(sl+1)*ELEMS_T);
              auto nat=ut(t);
              for(uint32_t i=0;i<T32;++i) for(uint32_t c=0;c<NCOMP;++c){ unsigned __int128 acc=0;
                  for(uint32_t k=0;k<T32;++k) acc += (unsigned __int128)A[sl][i][k]*Bm[sl][k][c];
                  uint32_t exp=(uint32_t)(acc%q);
                  if(nat[i*T32+c]!=exp){ if(mis<4) fmt::print("  [gemm] sl{} r{} c{}: got={} exp={}\n",sl,i,c,nat[i*T32+c],exp); ++mis; } } }
          fmt::print("  scan_gemm+recombine (engine): mismatches={}/{}\n", mis, NS*T32*NCOMP);
          total_mis += mis; }

        fmt::print("\non-device permutation + modadd_ab + scan_recombine + engine GEMM, q={}: total mismatches={}\n",
                   q, total_mis);
        pass = (total_mis == 0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
