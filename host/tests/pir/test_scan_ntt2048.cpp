// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// REAL-CONFIG scan on device: n=2048, K=3 (3×28-bit RNS), first-dim N0 (=512), rest, all 2048 slots.
// Validates the eval-domain scan core — real `fwd2048` (engine) → **n=2048 poly→slot gather** → K-tiled
// **mbatch matmul** → **slot→poly scatter** — bit-exact vs a cheap host eval-domain golden
//   A_ntt[ch][r][c][j] = Σ_k DB_ntt[ch][k*rest+r][j] · Q_ntt[ch][k*2+c][j]  mod q_ch.
// (The coeff `inv2048` tail is validated separately by the round-trip; a negacyclic coeff golden at
// N0=512/n=2048 would be ~26 B host ops, so we compare in the eval domain.)  DB NTT is offline (here
// computed on-device via the engine, one-time).  Reuses the mbatch + reshuffle kernels (kernels_ir).
//
//   ./build/test_scan_ntt2048 [N0] [rest] [seed]

#include <chrono>
#include <cstdint>
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
    const uint32_t N0   = (argc > 1) ? std::stoul(argv[1]) : 512;
    const uint32_t rest = (argc > 2) ? std::stoul(argv[2]) : 2;
    const uint32_t seed = (argc > 3) ? std::stoul(argv[3]) : 1;
    if (rest > T) { fmt::print("rest must be ≤ 32\n"); return 2; }
    const uint32_t KB = (N0 + T - 1) / T, K = 3;
    // three NTT-friendly primes < 2^28 with 4096 | (q-1) (n=2048 needs a 2N=4096-th root):
    const std::vector<uint32_t> P3 = {167772161u, 104857601u, 268271617u};
    std::vector<mont::MontgomeryContext> ctx; for (auto q : P3) ctx.emplace_back(q);
    std::mt19937 rng(seed);

    // coeff-domain DB[ch][k*rest+r], query[ch][k*2+c]
    std::vector<std::vector<std::vector<uint32_t>>> DBc(K, std::vector<std::vector<uint32_t>>(N0 * rest)),
        Qc(K, std::vector<std::vector<uint32_t>>(N0 * 2));
    for (uint32_t ch = 0; ch < K; ++ch) { std::uniform_int_distribution<uint32_t> di(0, P3[ch] - 1);
        for (auto& p : DBc[ch]) { p.resize(N); for (auto& v : p) v = di(rng); }
        for (auto& p : Qc[ch])  { p.resize(N); for (auto& v : p) v = di(rng); } }

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

        // ---- step 1: DB_ntt (offline) + Q_ntt via real fwd2048; PLAIN (mbatch is a plain modular matmul) ----
        std::vector<std::vector<std::vector<uint32_t>>> DBn(K, std::vector<std::vector<uint32_t>>(N0*rest)),
            Qn(K, std::vector<std::vector<uint32_t>>(N0*2));
        for (uint32_t ch = 0; ch < K; ++ch) {
            for (uint32_t i = 0; i < N0*rest; ++i) DBn[ch][i] = eng.fwd2048(eng.CH[ch], DBc[ch][i]);
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

        // reused builder lambdas (mbatch matmul / add / gather / scatter), sharded across the grid
        auto mbatch_pass = [&](uint32_t kt, pir2048::Chan& c){ const uint32_t C=GX*GY, per=C*Mb;
            for (uint32_t base=0; base<sOT; base+=per){ Program p=CreateProgram(); CoreRange cores(CoreCoord{0,0},CoreCoord{GX-1,GY-1});
                mkcb(p,cores,CBIndex::c_0,tt::DataFormat::UInt32,Mb,I32); mkcb(p,cores,CBIndex::c_1,tt::DataFormat::Int8,32,I8); mkcb(p,cores,CBIndex::c_2,tt::DataFormat::Int8,Mb*NB,I8); mkcb(p,cores,CBIndex::c_3,tt::DataFormat::Int32,Mb*NB,I32); mkcb(p,cores,CBIndex::c_16,tt::DataFormat::UInt32,Mb,I32);
                std::vector<uint32_t> rc; TensorAccessorArgs(*sX).append_to(rc); TensorAccessorArgs(*sW).append_to(rc);
                auto rd=CreateKernel(p,KIR "dataflow/reader_mbatch.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
                auto wr=CreateKernel(p,KIR "dataflow/writer_i32.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,KIR "compute/matmul_modq_mbatch.cpp",cores,ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t gy=0;gy<GY;++gy) for (uint32_t gx=0;gx<GX;++gx){ uint32_t idx=gy*GX+gx, o=base+idx*Mb; if (o>=sOT) continue; uint32_t mh=std::min(Mb,sOT-o), xb=kt*sOT+o; CoreCoord cc{gx,gy};
                    SetRuntimeArgs(p,rd,cc,{(uint32_t)sX->address(),(uint32_t)sW->address(),xb,xb*NB*NB,mh});
                    SetRuntimeArgs(p,wr,cc,{(uint32_t)sPar->address(),mh,o}); SetRuntimeArgs(p,cp,cc,{c.q,c.qinv,c.qlo,c.qhi,c.k14,mh}); }
                distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); } };
        auto add_pass = [&](Buf a, Buf b, pir2048::Chan& c){ const uint32_t C=GX*GY, per=C*Mb;
            for (uint32_t base=0; base<sOT; base+=per){ Program p=CreateProgram(); CoreRange cores(CoreCoord{0,0},CoreCoord{GX-1,GY-1});
                mkcb(p,cores,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,cores,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,cores,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto rd=CreateKernel(p,KIR "dataflow/reader_binop.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,KIR "dataflow/writer_i32.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,KIR "compute/add_modq.cpp",cores,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t gy=0;gy<GY;++gy) for (uint32_t gx=0;gx<GX;++gx){ uint32_t idx=gy*GX+gx, o=base+idx*Mb; if (o>=sOT) continue; uint32_t mh=std::min(Mb,sOT-o); CoreCoord cc{gx,gy};
                    SetRuntimeArgs(p,rd,cc,{(uint32_t)a->address(),(uint32_t)sPar->address(),mh,o}); SetRuntimeArgs(p,wr,cc,{(uint32_t)b->address(),mh,o}); SetRuntimeArgs(p,cp,cc,{mh,c.q}); }
                distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); } };
        auto gather = [&](Buf in, Buf out, uint32_t ROWS, uint32_t COLS, uint32_t SR, uint32_t SC, uint32_t out_base, uint32_t h){
            const uint32_t C=GX*GY, per=(ELEMS+C-1)/C;   // 1024 slots per half
            Program p=CreateProgram(); CoreRange cores(CoreCoord{0,0},CoreCoord{GX-1,GY-1});
            mkcb(p,cores,CBIndex::c_0,tt::DataFormat::UInt32,per+1,I32);
            auto rd=CreateKernel(p,KIR "dataflow/reshuffle_gather_mc.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            for (uint32_t gy=0;gy<GY;++gy) for (uint32_t gx=0;gx<GX;++gx){ uint32_t idx=gy*GX+gx, ss=idx*per; uint32_t no=(ss>=ELEMS)?0:std::min(per,ELEMS-ss);
                SetRuntimeArgs(p,rd,CoreCoord{gx,gy},{(uint32_t)in->address(),(uint32_t)out->address(),ROWS,COLS,SR,SC,ss,no,out_base+h*ELEMS,HALVES,h}); }
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };
        auto scatter = [&](Buf in, Buf out, uint32_t COLS, uint32_t n_poly, uint32_t h){   // out poly p' half h
            const uint32_t C=GX*GY, per=(n_poly+C-1)/C;
            Program p=CreateProgram(); CoreRange cores(CoreCoord{0,0},CoreCoord{GX-1,GY-1});
            mkcb(p,cores,CBIndex::c_0,tt::DataFormat::UInt32,per+1,I32);
            auto rd=CreateKernel(p,KIR "dataflow/reshuffle_scatter_mc.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            for (uint32_t gy=0;gy<GY;++gy) for (uint32_t gx=0;gx<GX;++gx){ uint32_t idx=gy*GX+gx, os=idx*per; uint32_t no=(os>=n_poly)?0:std::min(per,n_poly-os);
                // half h: read input slot-tiles [h*1024 + j]; write poly tile p'*HALVES + h; read pos from poly p'
                SetRuntimeArgs(p,rd,CoreCoord{gx,gy},{(uint32_t)in->address(),(uint32_t)out->address(),ELEMS,h*ELEMS,COLS,os,no,HALVES,h}); }
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };

        auto t0 = std::chrono::high_resolution_clock::now();
        uint32_t mism = 0, checked = 0;
        for (uint32_t ch = 0; ch < K; ++ch) {
            pir2048::Chan& c = eng.CH[ch]; const uint32_t q = P3[ch];
            // DB fold slot-major (offline/host): A_j[r][lk] = DBn[(kt*32+lk)*rest+r][j]
            std::vector<int8_t> Wb((size_t)KB*sOT*NB*NB*ELEMS, 0);
            for (uint32_t kt = 0; kt < KB; ++kt) for (uint32_t j = 0; j < N; ++j) {
                std::vector<uint32_t> A(ELEMS, 0);
                for (uint32_t lk = 0; lk < T; ++lk){ uint32_t kk=kt*T+lk; if (kk>=N0) continue;
                    for (uint32_t r = 0; r < rest; ++r) A[r*T+lk] = DBn[ch][kk*rest+r][j]; }
                for (uint32_t pp = 0; pp < NB; ++pp) for (uint32_t rr = 0; rr < NB; ++rr){ std::vector<int8_t> l(ELEMS);
                    for (uint32_t i = 0; i < ELEMS; ++i) l[i]=limb((uint32_t)((POW[rr]*A[i])%q),pp); auto ti=til1_8(l);
                    std::copy(ti.begin(),ti.end(),Wb.begin()+(((size_t)kt*sOT+j)*NB*NB+pp*NB+rr)*ELEMS); }
            }
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
