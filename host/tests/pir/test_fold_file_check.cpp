// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Verify db65536.fold — the encoder's int8 slot-major database — by RUNNING THE REAL SCAN on it.
//
// This is `test_scan_ntt2048` with exactly two substitutions:
//   * DB_ntt comes from db65536.pir (device-verified by test_db_encode_check) instead of random data,
//     un-Montgomery'd back to the plain values the matmul scan wants;
//   * the int8 fold is READ FROM db65536.fold instead of being recomputed inline.
// Everything else — gather, K-tiled mbatch matmul, scatter, the eval-domain golden — is the validated
// path. So a pass means the encoder's fold formula AND its byte layout are what the scan expects.
//
// Runs one m-tile (rest=32), which is what the scan produces at a time and how the file is grouped.
//
//   ./build/test_fold_file_check [mtile] [../../cpu/pir/db65536.pir] [../../cpu/pir/db65536.fold]
#include <chrono>
#include <cstdint>
#include <fstream>
#include <random>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "common/montgomery.hpp"
#include "pir_engine2048.hpp"

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
#ifndef OVERRIDE_KERNEL_PREFIX_IR
#define OVERRIDE_KERNEL_PREFIX_IR ""
#endif
#define KIR OVERRIDE_KERNEL_PREFIX_IR

namespace {
constexpr uint32_t T = 32, NB = 4, N = 2048, ELEMS = 1024, I8 = ELEMS, I32 = ELEMS * 4, HALVES = 2;
const uint64_t POW[4] = {1, 128, 16384, 2097152};
using u128 = unsigned __int128;
using Buf = std::shared_ptr<distributed::MeshBuffer>;
// til2/unt2: 2-tile (n=2048) poly ↔ tile pair; slot s in tile s/1024 at faceidx((s%1024)/32,(s%1024)%32)
inline uint32_t faceidx(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }
std::vector<uint32_t> til1(const std::vector<uint32_t>& v) { std::vector<uint32_t> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c) o[faceidx(r, c)] = v[r * T + c]; return o; }
std::vector<int8_t> til1_8(const std::vector<int8_t>& v) { std::vector<int8_t> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c) o[faceidx(r, c)] = v[r * T + c]; return o; }
std::vector<uint32_t> unt1(const std::vector<uint32_t>& v) { std::vector<uint32_t> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c) o[r * T + c] = v[faceidx(r, c)]; return o; }
}  // namespace

int main(int argc, char** argv) {
    const uint32_t mtile = (argc > 1) ? std::stoul(argv[1]) : 0;
    const char* pir_path  = (argc > 2) ? argv[2] : "reference/pir/db65536.pir";
    const char* fold_path = (argc > 3) ? argv[3] : "reference/pir/db65536.fold";
    uint32_t fh[8];
    { std::ifstream ff(fold_path, std::ios::binary);
      if (!ff) { fmt::print(stderr, "cannot open {}\n", fold_path); return 2; }
      ff.read((char*)fh, sizeof(fh));
      if (fh[0] != 0x50464C44u) { fmt::print(stderr, "bad fold magic\n"); return 2; } }
    const uint32_t N0 = fh[4], DBrest = fh[5], KBf = fh[6], MBf = fh[7];
    const uint32_t rest = T, seed = 1;                       // one m-tile per run
    fmt::print("fold {}: N0={} rest={} KB={} MB={}, checking m-tile {}\n", fold_path, N0, DBrest, KBf, MBf, mtile);
    if (mtile >= MBf) { fmt::print(stderr, "mtile out of range\n"); return 2; }
    const uint32_t KB = (N0 + T - 1) / T, K = 3;
    // the primes come from db65536.pir's header, NOT from a local constant: the encoder and the client
    // must agree on the modulus, and hardcoding a second (equally valid) triple here would hide exactly
    // the mismatch this file exists to catch.
    std::vector<uint32_t> P3(K);
    { std::ifstream fpp(pir_path, std::ios::binary); uint32_t hh[8];
      if (!fpp) { fmt::print(stderr, "cannot open {}\n", pir_path); return 2; }
      fpp.read((char*)hh, sizeof(hh)); P3 = {hh[5], hh[6], hh[7]}; }
    std::vector<mont::MontgomeryContext> ctx; for (auto q : P3) ctx.emplace_back(q);
    std::mt19937 rng(seed);

    // coeff-domain DB[ch][k*rest+r], query[ch][k*2+c]
    std::vector<std::vector<std::vector<uint32_t>>> DBc(K, std::vector<std::vector<uint32_t>>(N0 * rest)),
        Qc(K, std::vector<std::vector<uint32_t>>(N0 * 2));
    for (uint32_t ch = 0; ch < K; ++ch) { std::uniform_int_distribution<uint32_t> di(0, P3[ch] - 1);
        for (auto& p : Qc[ch])  { p.resize(N); for (auto& v : p) v = di(rng); } }   // query stays synthetic

    bool pass = true;
    try {
        auto md = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = md->mesh_command_queue();
        distributed::MeshCoordinateRange dr(md->shape());
        auto grid = md->compute_with_storage_grid_size();
        const uint32_t GX = grid.x, GY = grid.y;
        fmt::print("grid {}x{}={} cores; REAL config n={} K={} N0={} rest={} KB={}\n", GX, GY, GX*GY, N, K, N0, rest, KB);
        pir2048::Engine eng(md.get(), &cq, P3, 5, 2048);   // ell/B unused by the scan; valid values for the ctor
        auto mkbuf = [&](size_t n, uint32_t pg){ distributed::DeviceLocalBufferConfig d{.page_size=pg,.buffer_type=BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size=n*pg}, d, md.get()); };
        auto mkcb = [&](Program& p, CoreRange cr, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t pg){
            CircularBufferConfig c(depth*pg, {{idx, fmt}}); c.set_page_size(idx, pg); CreateCircularBuffer(p, cr, c); };
        auto wrb = [&](Buf b, const std::vector<uint32_t>& v){ distributed::EnqueueWriteMeshBuffer(cq, b, v, false); };
        auto rdb = [&](Buf b){ std::vector<uint32_t> v; distributed::EnqueueReadMeshBuffer(cq, v, b, true); return v; };

        // ---- DB_ntt straight from db65536.pir (verified) ----
        // the file holds mont(NTT) for the MAC path; the matmul scan wants PLAIN, so undo the lift.
        std::ifstream fp(pir_path, std::ios::binary);
        if (!fp) { fmt::print(stderr, "cannot open {}\n", pir_path); return 2; }
        uint32_t ph[8]; fp.read((char*)ph, sizeof(ph));
        if (ph[0] != 0x50444231u) { fmt::print(stderr, "bad pir magic\n"); return 2; }
        const uint32_t n_rows_f = ph[4]; const size_t per_ch_f = (size_t)n_rows_f * N;
        // ---- step 1: DB_ntt (offline) + Q_ntt via real fwd2048; PLAIN (mbatch is a plain modular matmul) ----
        std::vector<std::vector<std::vector<uint32_t>>> DBn(K, std::vector<std::vector<uint32_t>>(N0*rest)),
            Qn(K, std::vector<std::vector<uint32_t>>(N0*2));
        for (uint32_t ch = 0; ch < K; ++ch) {
            const uint32_t q = P3[ch];
            const uint32_t rinv = (uint32_t)mont::modular_inverse(((uint64_t)1 << 32) % q, q);
            for (uint32_t k = 0; k < N0; ++k) for (uint32_t rl = 0; rl < rest; ++rl) {
                const size_t cell = (size_t)k * DBrest + (mtile * T + rl);     // this m-tile's rows
                std::vector<uint32_t> tl(N);
                fp.seekg((std::streamoff)(sizeof(ph) + ((size_t)ch * per_ch_f + cell * N) * sizeof(uint32_t)));
                fp.read((char*)tl.data(), (std::streamsize)(N * sizeof(uint32_t)));
                auto nat = pir2048::unt2(tl);
                for (auto& v : nat) v = (uint32_t)((uint64_t)v * rinv % q);    // mont -> plain
                DBn[ch][(size_t)k * rest + rl] = nat; }
            for (uint32_t i = 0; i < N0*2;    ++i) Qn[ch][i]  = eng.fwd2048(eng.CH[ch], Qc[ch][i]);
        }
        // host eval-domain golden: A_ntt[ch][r*2+c][j] = Σ_k DBn[k*rest+r][j]·Qn[k*2+c][j] mod q
        std::vector<std::vector<std::vector<uint32_t>>> Ag(K, std::vector<std::vector<uint32_t>>(rest*2, std::vector<uint32_t>(N, 0)));
        for (uint32_t ch = 0; ch < K; ++ch) for (uint32_t r = 0; r < rest; ++r) for (uint32_t c = 0; c < 2; ++c)
            for (uint32_t j = 0; j < N; ++j) { u128 a = 0;
                for (uint32_t k = 0; k < N0; ++k) a += (u128)DBn[ch][k*rest+r][j] * Qn[ch][k*2+c][j];
                Ag[ch][r*2+c][j] = (uint32_t)(a % P3[ch]); }

        // ---- device scan buffers (per channel) ----
        const uint32_t sOT = N, Mb = 32;
        auto limb = [](uint32_t v, uint32_t p) -> int8_t { return (int8_t)((v >> (7*p)) & 127); };
        auto b_Qk  = mkbuf((size_t)T*2*HALVES, I32);                 // one k-tile's query polys (til2)
        auto sX    = mkbuf((size_t)KB*sOT, I32), sW = mkbuf((size_t)KB*sOT*NB*NB, I8);
        auto sPar  = mkbuf(sOT, I32), sAcc0 = mkbuf(sOT, I32), sAcc1 = mkbuf(sOT, I32);
        auto b_Antt= mkbuf((size_t)rest*2*HALVES, I32);   // rest*2 polys × HALVES tiles/poly

        // the wiring now lives in the Engine (scan_mbatch_pass / scan_add_pass / scan_gather /
        // scan_scatter), lifted verbatim.  These delegate, so this test — which already proved the
        // inline version correct on this database — now proves the LIFTED version correct too.
        auto mbatch_pass = [&](uint32_t kt, pir2048::Chan& c){ eng.scan_mbatch_pass(c, sX, sW, sPar, kt, sOT); };
        auto add_pass    = [&](Buf a, Buf b, pir2048::Chan& c){ eng.scan_add_pass(c, a, sPar, b, sOT); };
        auto gather      = [&](Buf in, Buf out, uint32_t ROWS, uint32_t COLS, uint32_t SR, uint32_t SC,
                               uint32_t out_base, uint32_t h){ eng.scan_gather(in,out,ROWS,COLS,SR,SC,out_base,h); };
        auto scatter     = [&](Buf in, Buf out, uint32_t COLS, uint32_t n_poly, uint32_t h){
                               eng.scan_scatter(in,out,COLS,n_poly,h); };

        auto t0 = std::chrono::high_resolution_clock::now();
        uint32_t mism = 0, checked = 0;
        for (uint32_t ch = 0; ch < K; ++ch) {
            pir2048::Chan& c = eng.CH[ch]; const uint32_t q = P3[ch];
            // DB fold slot-major (offline/host): A_j[r][lk] = DBn[(kt*32+lk)*rest+r][j]
            // ---- THE THING UNDER TEST: read the fold from db65536.fold instead of recomputing it.
            // file layout is [m-tile][channel][k-tile][slot][16 int8 tiles]; each m-tile block holds
            // K*KB*N*16 tiles, so this channel's slice starts at (mtile block) + ch's offset.
            std::vector<int8_t> Wb((size_t)KB*sOT*NB*NB*ELEMS, 0);
            { std::ifstream ff(fold_path, std::ios::binary);
              const size_t per_mt = (size_t)K * KBf * N * NB * NB * ELEMS;
              const size_t off = sizeof(fh) + (size_t)mtile * per_mt
                               + (size_t)ch * KBf * N * NB * NB * ELEMS;
              ff.seekg((std::streamoff)off);
              ff.read((char*)Wb.data(), (std::streamsize)((size_t)KB*sOT*NB*NB*ELEMS)); }
            distributed::EnqueueWriteMeshBuffer(cq, sW, Wb, false);
            std::vector<uint32_t> zero((size_t)sOT*ELEMS, 0); wrb(sAcc0, zero);

            // QUERY reshuffle ON DEVICE (per k-tile, n=2048 → 2 halves): til2(Q_ntt) → sX
            for (uint32_t kt = 0; kt < KB; ++kt) {
                uint32_t ROWS = std::min(T, N0 - kt*T);
                std::vector<uint32_t> Qk((size_t)T*2*HALVES*ELEMS, 0);   // full b_Qk size; unused rows stay 0
                for (uint32_t lk = 0; lk < ROWS; ++lk) for (uint32_t cc = 0; cc < 2; ++cc) {
                    auto& poly = Qn[ch][(kt*T+lk)*2+cc];
                    for (uint32_t hh = 0; hh < HALVES; ++hh) { std::vector<uint32_t> half(poly.begin()+hh*ELEMS, poly.begin()+(hh+1)*ELEMS);
                        auto t = til1(half); std::copy(t.begin(), t.end(), Qk.begin()+(size_t)((lk*2+cc)*HALVES+hh)*ELEMS); }
                }
                wrb(b_Qk, Qk);
                for (uint32_t hh = 0; hh < HALVES; ++hh) gather(b_Qk, sX, ROWS, 2u, 2u, 1u, kt*sOT, hh);
            }
            // matmul K-tiled
            Buf a = sAcc0, b = sAcc1;
            for (uint32_t kt = 0; kt < KB; ++kt) { mbatch_pass(kt, c); add_pass(a, b, c); std::swap(a, b); }
            Buf fin = (KB % 2 == 0) ? sAcc0 : sAcc1;
            // OUTPUT reshuffle ON DEVICE (slot→poly, 2 halves): fin → b_Antt (rest*2 polys × 2 tiles)
            for (uint32_t hh = 0; hh < HALVES; ++hh) scatter(fin, b_Antt, 2u, rest*2, hh);
            auto Araw = rdb(b_Antt);
            // compare A_ntt (poly-major, 2 tiles/poly) vs eval golden
            for (uint32_t pp = 0; pp < rest*2; ++pp) for (uint32_t hh = 0; hh < HALVES; ++hh){
                auto pt = unt1(std::vector<uint32_t>(Araw.begin()+((size_t)pp*HALVES+hh)*ELEMS, Araw.begin()+((size_t)(pp*HALVES+hh)+1)*ELEMS));
                for (uint32_t jl = 0; jl < ELEMS; ++jl){ uint32_t j = hh*ELEMS+jl;
                    if (pt[jl] != Ag[ch][pp][j]) { if (mism<8) fmt::print("  ch{} p'{} j{} got={} exp={}\n",ch,pp,j,pt[jl],Ag[ch][pp][j]); ++mism; } ++checked; } }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        fmt::print("real-config eval-domain scan (device)  mismatches={}/{}  scan-wall={:.1f} ms\n",
                   mism, checked, std::chrono::duration<double,std::milli>(t1-t0).count());
        pass = (mism == 0); pass &= md->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
