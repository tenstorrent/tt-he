// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// END-TO-END coeff-in / coeff-out first-dimension scan on device — the NTT wired around the fast
// mbatch matmul.  For each RNS channel:
//   acc[r][c] = Σ_{k<N0} DB[k*rest+r] ⊛ query[k*2+c]   (negacyclic poly product, mod q).
// Pipeline (all COMPUTE on device):
//   step 1  Q_ntt = fwd2048(query)                                  [device NTT]
//   reshuffle  poly-major (each poly = N slots) → SLOT-MAJOR matmul operands  [host index reorg]
//   step 2  A_ntt[j] = DB_ntt[j]ᵀ · Q_ntt[j]  batched over slots j  [device mbatch K-tiled matmul]
//   reshuffle  slot-major A_ntt → poly-major                        [host index reorg]
//   step 3  acc = inv2048(A_ntt)                                    [device iNTT]
// DB_ntt slot-major + int8 fold is OFFLINE (DB preprocessing); only the query and output reshuffles
// are online (small, currently host-side — the on-device NoC reshuffle is a later perf step).
// Validated bit-exact vs a host negacyclic-convolution golden.  n=1024 (1 tile/poly) for clarity.
//
//   ./build/test_scan_ntt [N0] [rest] [q1] [q2] [seed]

#include <cstdint>
#include <random>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "common/montgomery.hpp"
#include "pir_permute.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""          // NTT kernels live in device/kernels/
#endif
#ifndef OVERRIDE_KERNEL_PREFIX_IR
#define OVERRIDE_KERNEL_PREFIX_IR ""       // mbatch/add kernels live in device/kernels_ir/
#endif
#define KIR OVERRIDE_KERNEL_PREFIX_IR

namespace {
constexpr uint32_t T = 32, NB = 4, N = 1024, ELEMS = 1024, I8 = ELEMS, I32 = ELEMS * 4;
const uint64_t POW[4] = {1, 128, 16384, 2097152};
using u128 = unsigned __int128;
using Buf = std::shared_ptr<distributed::MeshBuffer>;
uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) { uint64_t r = 1; b %= m;
    while (e) { if (e & 1) r = (u128)r * b % m; b = (u128)b * b % m; e >>= 1; } return r; }
uint64_t prim_root(uint64_t q) { uint64_t phi = q - 1, n = phi; std::vector<uint64_t> f;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) { f.push_back(d); while (n % d == 0) n /= d; }
    if (n > 1) f.push_back(n);
    for (uint64_t g = 2; g < q; ++g) { bool ok = true; for (uint64_t p : f) if (modpow(g, phi / p, q) == 1) ok = false;
        if (ok) return g; } throw std::runtime_error("no root"); }
int8_t limb8(uint32_t v, uint32_t p) { return (int8_t)((v >> (7 * p)) & 127); }
std::vector<uint32_t> til(const std::vector<uint32_t>& v) { return pir::tilize_poly(v, 1); }
std::vector<uint32_t> unt(const std::vector<uint32_t>& v) { return pir::untilize_poly(v, 1); }
std::vector<int8_t> til8(const std::vector<int8_t>& in) { std::vector<int8_t> o(ELEMS);
    for (uint32_t r = 0; r < T; ++r) for (uint32_t c = 0; c < T; ++c)
        o[((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16)] = in[r * T + c]; return o; }
}  // namespace

int main(int argc, char** argv) {
    const uint32_t N0   = (argc > 1) ? std::stoul(argv[1]) : 64;
    const uint32_t rest = (argc > 2) ? std::stoul(argv[2]) : 32;
    const uint64_t q1   = (argc > 3) ? std::stoull(argv[3]) : 167772161ULL;
    const uint64_t q2   = (argc > 4) ? std::stoull(argv[4]) : 104857601ULL;  // 25·2^22+1, <2^28, 2048|q-1
    const uint32_t seed = (argc > 5) ? std::stoul(argv[5]) : 1;
    if (rest > T) { fmt::print("rest must be ≤ 32\n"); return 2; }
    const uint32_t KB = (N0 + T - 1) / T;                 // k-tiles contracting N0
    const uint64_t QS[2] = {q1, q2};
    std::mt19937 rng(seed);

    // coeff-domain DB[ch][k*rest+r] and query[ch][k*2+c], N=1024 each; golden = negacyclic Σ_k DB⊛query
    std::vector<std::vector<std::vector<uint32_t>>> DBc(2, std::vector<std::vector<uint32_t>>(N0 * rest, std::vector<uint32_t>(N)));
    std::vector<std::vector<std::vector<uint32_t>>> Qc(2, std::vector<std::vector<uint32_t>>(N0 * 2, std::vector<uint32_t>(N)));
    std::vector<std::vector<std::vector<uint32_t>>> outG(2, std::vector<std::vector<uint32_t>>(rest * 2, std::vector<uint32_t>(N, 0)));
    for (uint32_t ch = 0; ch < 2; ++ch) {
        std::uniform_int_distribution<uint32_t> di(0, (uint32_t)QS[ch] - 1);
        for (auto& p : DBc[ch]) for (auto& v : p) v = di(rng);
        for (auto& p : Qc[ch]) for (auto& v : p) v = di(rng);
        for (uint32_t r = 0; r < rest; ++r) for (uint32_t c = 0; c < 2; ++c) {
            std::vector<__int128> acc(N, 0);   // negacyclic (mod x^N+1): + if a+b<N, − if a+b≥N
            for (uint32_t k = 0; k < N0; ++k) { auto& A = DBc[ch][k * rest + r]; auto& B = Qc[ch][k * 2 + c];
                for (uint32_t a = 0; a < N; ++a) for (uint32_t b = 0; b < N; ++b) {
                    uint32_t s = a + b; __int128 prod = (__int128)A[a] * B[b];
                    if (s < N) acc[s] += prod; else acc[s - N] -= prod; } }
            for (uint32_t i = 0; i < N; ++i) { __int128 v = acc[i] % (__int128)QS[ch]; if (v < 0) v += QS[ch];
                outG[ch][r * 2 + c][i] = (uint32_t)v; }
        }
    }

    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t GX = grid.x, GY = grid.y;
        constexpr CoreCoord core = {0, 0};
        auto mkbuf = [&](size_t nt, uint32_t pg) { distributed::DeviceLocalBufferConfig d{.page_size = pg, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * pg}, d, mesh.get()); };
        auto mkcb = [&](Program& p, CoreRange cr, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t pg) {
            CircularBufferConfig c(depth * pg, {{idx, fmt}}); c.set_page_size(idx, pg); CreateCircularBuffer(p, cr, c); };
        auto wrbuf = [&](auto buf, const std::vector<uint32_t>& v) { distributed::EnqueueWriteMeshBuffer(cq, buf, v, false); };
        auto rdbuf = [&](auto buf) { std::vector<uint32_t> v; distributed::EnqueueReadMeshBuffer(cq, v, buf, true); return v; };

        // ---------- device negacyclic NTT (from test_initdim, n=1024) ----------
        struct Chan { uint32_t q, qinv, qlo, qhi, k14; Buf WfA, Wr, Tm, WfAi, Wri, Tmi; };
        auto foldA = [&](const std::vector<uint32_t>& Wm, uint32_t q) { std::vector<int8_t> o(NB*NB*ELEMS);
            for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<uint32_t> l(ELEMS);
                for (uint32_t i=0;i<ELEMS;++i) l[i]=limb8((uint32_t)((POW[r]*Wm[i])%q),p); auto t=til(l);
                for (uint32_t i=0;i<ELEMS;++i) o[(p*NB+r)*ELEMS+i]=(int8_t)t[i]; } return o; };
        auto foldC = [&](const std::vector<uint32_t>& Wm, uint32_t q) { std::vector<int8_t> o(NB*NB*ELEMS);
            for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<uint32_t> l(ELEMS);
                for (uint32_t i=0;i<ELEMS;++i) l[i]=limb8((uint32_t)((POW[p]*Wm[i])%q),r); auto t=til(l);
                for (uint32_t i=0;i<ELEMS;++i) o[(p*NB+r)*ELEMS+i]=(int8_t)t[i]; } return o; };
        auto make_chan = [&](uint32_t q) { mont::MontgomeryContext ctx(q); Chan c; c.q=q; c.qinv=ctx.q_inv_32(); c.qlo=q&0xFFFF; c.qhi=q>>16; c.k14=ctx.to_montgomery(16384u);
            uint64_t g=prim_root(q), w=modpow(g,(q-1)/N,q), w32=modpow(g,(q-1)/T,q), psi=modpow(g,(q-1)/(2*N),q);
            std::vector<uint32_t> Wc(ELEMS),Wa(ELEMS),Tw(ELEMS);
            for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b){ uint32_t wv=(uint32_t)modpow(w32,(uint64_t)a*b,q); Wc[a*T+b]=wv; Wa[a*T+b]=(uint32_t)((u128)wv*modpow(psi,32ull*b,q)%q); }
            for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b) Tw[a*T+b]=(uint32_t)((u128)modpow(w,(uint64_t)a*b,q)*modpow(psi,b,q)%q);
            uint64_t wi=mont::modular_inverse(w,q),w32i=mont::modular_inverse(w32,q),psii=mont::modular_inverse(psi,q),Ninv=mont::modular_inverse(N,q);
            std::vector<uint32_t> Wci(ELEMS),Wai(ELEMS),Twi(ELEMS);
            for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b){ uint32_t wv=(uint32_t)modpow(w32i,(uint64_t)a*b,q); Wci[a*T+b]=(uint32_t)((u128)wv*modpow(psii,32ull*b,q)%q); Wai[a*T+b]=(uint32_t)((u128)wv*Ninv%q); }
            for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b) Twi[a*T+b]=(uint32_t)((u128)modpow(wi,(uint64_t)a*b,q)*modpow(psii,a,q)%q);
            auto mont_til=[&](const std::vector<uint32_t>& v){ std::vector<uint32_t> m(ELEMS); for(uint32_t i=0;i<ELEMS;++i) m[i]=ctx.to_montgomery(v[i]); return til(m); };
            c.WfA=mkbuf(NB*NB,I8); c.Wr=mkbuf(NB*NB,I8); c.Tm=mkbuf(1,I32); c.WfAi=mkbuf(NB*NB,I8); c.Wri=mkbuf(NB*NB,I8); c.Tmi=mkbuf(1,I32);
            distributed::EnqueueWriteMeshBuffer(cq,c.WfA,foldA(Wa,q),false); distributed::EnqueueWriteMeshBuffer(cq,c.Wr,foldC(Wc,q),false); wrbuf(c.Tm,mont_til(Tw));
            distributed::EnqueueWriteMeshBuffer(cq,c.WfAi,foldA(Wai,q),false); distributed::EnqueueWriteMeshBuffer(cq,c.Wri,foldC(Wci,q),false); wrbuf(c.Tmi,mont_til(Twi));
            return c; };
        auto b_in=mkbuf(1,I32), b_g=mkbuf(1,I32), b_gp=mkbuf(1,I32), b_h=mkbuf(1,I32);
        auto fusedA=[&](auto xin,auto w,auto out,const Chan& c){ Program p=CreateProgram();
            mkcb(p,core,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,core,CBIndex::c_1,tt::DataFormat::Int8,2,I8); mkcb(p,core,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcb(p,core,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,core,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*xin).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)xin->address(),(uint32_t)w->address()}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14});
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };
        auto fusedC=[&](auto gp,auto w,auto out,const Chan& c){ Program p=CreateProgram();
            mkcb(p,core,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,core,CBIndex::c_1,tt::DataFormat::Int8,NB*NB,I8); mkcb(p,core,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcb(p,core,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,core,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*gp).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)gp->address(),(uint32_t)w->address()}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14});
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };
        auto modmulP=[&](auto a,auto b,auto out,const Chan& c){ Program p=CreateProgram();
            mkcb(p,core,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,core,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,core,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)a->address(),(uint32_t)b->address(),1u,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{1u,c.q,c.qinv,c.qlo,c.qhi});
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };
        auto reshape=[&](const std::vector<uint32_t>& x){ std::vector<uint32_t> X(ELEMS); for (uint32_t n2=0;n2<T;++n2) for (uint32_t n1=0;n1<T;++n1) X[n2*T+n1]=x[n1+T*n2]; return til(X); };
        auto dereindex=[&](const std::vector<uint32_t>& raw){ auto H=unt(raw); std::vector<uint32_t> X(N); for (uint32_t j1=0;j1<T;++j1) for (uint32_t j2=0;j2<T;++j2) X[j2+T*j1]=H[j2*T+j1]; return X; };
        auto fwd=[&](const Chan& c,const std::vector<uint32_t>& x){ wrbuf(b_in,reshape(x)); fusedA(b_in,c.WfA,b_g,c); modmulP(b_g,c.Tm,b_gp,c); fusedC(b_gp,c.Wr,b_h,c); return dereindex(rdbuf(b_h)); };
        auto inv=[&](const Chan& c,const std::vector<uint32_t>& X){ wrbuf(b_in,reshape(X)); fusedA(b_in,c.WfAi,b_g,c); modmulP(b_g,c.Tmi,b_gp,c); fusedC(b_gp,c.Wri,b_h,c); return dereindex(rdbuf(b_h)); };
        Chan CH[2] = { make_chan((uint32_t)q1), make_chan((uint32_t)q2) };

        // ---------- fast mbatch modular matmul (from test_scan_kt), multi-core ----------
        Buf sX, sW, sPar, sAcc0, sAcc1; uint32_t sOT = 0;   // scan buffers, set per channel
        auto mbatch_pass = [&](uint32_t gx, uint32_t gy, uint32_t Mb, uint32_t kt, const Chan& c) {
            const uint32_t C = gx * gy, per = C * Mb;
            for (uint32_t base = 0; base < sOT; base += per) {
                Program p = CreateProgram(); CoreRange cores(CoreCoord{0,0}, CoreCoord{gx-1, gy-1});
                mkcb(p,cores,CBIndex::c_0,tt::DataFormat::UInt32,Mb,I32); mkcb(p,cores,CBIndex::c_1,tt::DataFormat::Int8,32,I8);
                mkcb(p,cores,CBIndex::c_2,tt::DataFormat::Int8,Mb*NB,I8); mkcb(p,cores,CBIndex::c_3,tt::DataFormat::Int32,Mb*NB,I32); mkcb(p,cores,CBIndex::c_16,tt::DataFormat::UInt32,Mb,I32);
                std::vector<uint32_t> rc; TensorAccessorArgs(*sX).append_to(rc); TensorAccessorArgs(*sW).append_to(rc);
                auto rd=CreateKernel(p,KIR "dataflow/reader_mbatch.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
                auto wr=CreateKernel(p,KIR "dataflow/writer_i32.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,KIR "compute/matmul_modq_mbatch.cpp",cores,ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t gyi=0;gyi<gy;++gyi) for (uint32_t gxi=0;gxi<gx;++gxi){ uint32_t idx=gyi*gx+gxi, o_base=base+idx*Mb; if (o_base>=sOT) continue;
                    uint32_t m_here=std::min(Mb, sOT-o_base), x_base=kt*sOT+o_base; CoreCoord cc{gxi,gyi};
                    SetRuntimeArgs(p,rd,cc,{(uint32_t)sX->address(),(uint32_t)sW->address(),x_base,x_base*NB*NB,m_here});
                    SetRuntimeArgs(p,wr,cc,{(uint32_t)sPar->address(),m_here,o_base});
                    SetRuntimeArgs(p,cp,cc,{c.q,c.qinv,c.qlo,c.qhi,c.k14,m_here}); }
                distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); } };
        auto add_pass = [&](uint32_t gx, uint32_t gy, uint32_t Mb, Buf ain, Buf aout, const Chan& c) {
            const uint32_t C = gx * gy, per = C * Mb;
            for (uint32_t base = 0; base < sOT; base += per) {
                Program p = CreateProgram(); CoreRange cores(CoreCoord{0,0}, CoreCoord{gx-1, gy-1});
                mkcb(p,cores,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,cores,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,cores,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
                auto rd=CreateKernel(p,KIR "dataflow/reader_binop.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
                auto wr=CreateKernel(p,KIR "dataflow/writer_i32.cpp",cores,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
                auto cp=CreateKernel(p,KIR "compute/add_modq.cpp",cores,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
                for (uint32_t gyi=0;gyi<gy;++gyi) for (uint32_t gxi=0;gxi<gx;++gxi){ uint32_t idx=gyi*gx+gxi, o_base=base+idx*Mb; if (o_base>=sOT) continue;
                    uint32_t m_here=std::min(Mb, sOT-o_base); CoreCoord cc{gxi,gyi};
                    SetRuntimeArgs(p,rd,cc,{(uint32_t)ain->address(),(uint32_t)sPar->address(),m_here,o_base});
                    SetRuntimeArgs(p,wr,cc,{(uint32_t)aout->address(),m_here,o_base});
                    SetRuntimeArgs(p,cp,cc,{m_here,c.q}); }
                distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); } };

        // ---------- on-device reshuffles (poly↔slot all-to-all), sharded across the grid ----------
        auto gather_pass = [&](Buf in, Buf out, uint32_t ROWS, uint32_t COLS, uint32_t SR, uint32_t SC, uint32_t out_base, uint32_t n_total){
            const uint32_t Cc = GX*GY, per = (n_total + Cc - 1)/Cc;
            Program p = CreateProgram(); CoreRange cores(CoreCoord{0,0}, CoreCoord{GX-1,GY-1});
            CircularBufferConfig c((per+1)*I32, {{CBIndex::c_0, tt::DataFormat::UInt32}}); c.set_page_size(CBIndex::c_0, I32); CreateCircularBuffer(p, cores, c);
            auto rd = CreateKernel(p, KIR "dataflow/reshuffle_gather_mc.cpp", cores, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            for (uint32_t gy=0; gy<GY; ++gy) for (uint32_t gx=0; gx<GX; ++gx){ uint32_t idx=gy*GX+gx, os=idx*per; uint32_t n_out=(os>=n_total)?0:std::min(per,n_total-os);
                SetRuntimeArgs(p, rd, CoreCoord{gx,gy}, {(uint32_t)in->address(),(uint32_t)out->address(),ROWS,COLS,SR,SC,os,n_out,out_base,1u,0u}); }
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };
        auto scatter_pass = [&](Buf in, Buf out, uint32_t S_in, uint32_t COLS, uint32_t n_total){
            const uint32_t Cc = GX*GY, per = (n_total + Cc - 1)/Cc;
            Program p = CreateProgram(); CoreRange cores(CoreCoord{0,0}, CoreCoord{GX-1,GY-1});
            CircularBufferConfig c((per+1)*I32, {{CBIndex::c_0, tt::DataFormat::UInt32}}); c.set_page_size(CBIndex::c_0, I32); CreateCircularBuffer(p, cores, c);
            auto rd = CreateKernel(p, KIR "dataflow/reshuffle_scatter_mc.cpp", cores, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            for (uint32_t gy=0; gy<GY; ++gy) for (uint32_t gx=0; gx<GX; ++gx){ uint32_t idx=gy*GX+gx, os=idx*per; uint32_t n_out=(os>=n_total)?0:std::min(per,n_total-os);
                SetRuntimeArgs(p, rd, CoreCoord{gx,gy}, {(uint32_t)in->address(),(uint32_t)out->address(),S_in,0u,COLS,os,n_out,1u,0u}); }
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };

        const uint32_t GXY_x = GX, GXY_y = GY, Mb = 32;
        uint32_t mism = 0, checked = 0;
        for (uint32_t ch = 0; ch < 2; ++ch) {
            const uint32_t q = (uint32_t)QS[ch];
            // step 1: Q_ntt = fwd(query); DB_ntt = fwd(DB) (DB NTT is offline in the real pipeline)
            std::vector<std::vector<uint32_t>> DBntt(N0 * rest), Qntt(N0 * 2);
            for (uint32_t idx = 0; idx < N0 * rest; ++idx) DBntt[idx] = fwd(CH[ch], DBc[ch][idx]);
            for (uint32_t idx = 0; idx < N0 * 2; ++idx) Qntt[idx] = fwd(CH[ch], Qc[ch][idx]);

            // reshuffle to SLOT-MAJOR + fold.  OT = N slots (rest≤32, c=2 → 1 tile/slot).  layout k-tile-major.
            sOT = N;
            sX = mkbuf((size_t)KB * sOT, I32); sW = mkbuf((size_t)KB * sOT * NB * NB, I8);
            sPar = mkbuf(sOT, I32); sAcc0 = mkbuf(sOT, I32); sAcc1 = mkbuf(sOT, I32);
            // DB reshuffle+fold is OFFLINE → keep on host (Wb).  A_j[r][lk] = DBntt[(kt*32+lk)*rest+r][j].
            std::vector<int8_t> Wb((size_t)KB * sOT * NB * NB * ELEMS, 0);
            auto limb = [](uint32_t v, uint32_t p) -> int8_t { return (int8_t)((v >> (7 * p)) & 127); };
            for (uint32_t kt = 0; kt < KB; ++kt) for (uint32_t j = 0; j < N; ++j) {
                std::vector<uint32_t> A(ELEMS, 0);
                for (uint32_t lk = 0; lk < T; ++lk) { uint32_t kk = kt * T + lk; if (kk >= N0) continue;
                    for (uint32_t r = 0; r < rest; ++r) A[r * T + lk] = DBntt[kk * rest + r][j]; }
                for (uint32_t p = 0; p < NB; ++p) for (uint32_t rr = 0; rr < NB; ++rr) {
                    std::vector<int8_t> l(ELEMS);
                    for (uint32_t i = 0; i < ELEMS; ++i) l[i] = limb((uint32_t)((POW[rr] * A[i]) % q), p);
                    auto ti = til8(l);
                    std::copy(ti.begin(), ti.end(), Wb.begin() + (((size_t)kt * sOT + j) * NB * NB + p * NB + rr) * ELEMS); }
            }
            distributed::EnqueueWriteMeshBuffer(cq, sW, Wb, false);
            std::vector<uint32_t> zero((size_t)sOT * ELEMS, 0); wrbuf(sAcc0, zero);

            // QUERY reshuffle ON DEVICE (poly→slot gather): per k-tile, til(Q_ntt) → sX[kt*sOT + j].
            //   B_j[lk][c] = Q_ntt[(kt*32+lk)*2+c][j]  → ROWS=validlk, COLS=2, SR=2, SC=1.
            Buf b_Qk = mkbuf((size_t)T * 2, I32);
            for (uint32_t kt = 0; kt < KB; ++kt) {
                uint32_t ROWS = std::min(T, N0 - kt * T);
                std::vector<uint32_t> Qk((size_t)ROWS * 2 * ELEMS, 0);
                for (uint32_t lk = 0; lk < ROWS; ++lk) for (uint32_t c = 0; c < 2; ++c) {
                    auto t = til(Qntt[(kt * T + lk) * 2 + c]); std::copy(t.begin(), t.end(), Qk.begin() + (size_t)(lk * 2 + c) * ELEMS); }
                wrbuf(b_Qk, Qk);
                gather_pass(b_Qk, sX, ROWS, 2u, 2u, 1u, kt * sOT, sOT);
            }

            // step 2: K-tiled mbatch matmul, accumulate
            Buf a = sAcc0, b = sAcc1;
            for (uint32_t kt = 0; kt < KB; ++kt) { mbatch_pass(GXY_x, GXY_y, Mb, kt, CH[ch]); add_pass(GXY_x, GXY_y, Mb, a, b, CH[ch]); std::swap(a, b); }
            Buf finalAcc = (KB % 2 == 0) ? sAcc0 : sAcc1;

            // OUTPUT reshuffle ON DEVICE (slot→poly scatter): finalAcc → A_poly[r*2+c].
            Buf b_Antt = mkbuf((size_t)rest * 2, I32);
            scatter_pass(finalAcc, b_Antt, sOT, 2u, rest * 2);
            auto Araw = rdbuf(b_Antt);
            std::vector<std::vector<uint32_t>> Antt(rest * 2, std::vector<uint32_t>(N, 0));
            for (uint32_t pp = 0; pp < rest * 2; ++pp) { auto pt = unt(std::vector<uint32_t>(Araw.begin() + (size_t)pp * ELEMS, Araw.begin() + (size_t)(pp + 1) * ELEMS));
                for (uint32_t j = 0; j < N; ++j) Antt[pp][j] = pt[j]; }

            // step 3: acc = inv(A_ntt); validate vs golden
            for (uint32_t r = 0; r < rest; ++r) for (uint32_t c = 0; c < 2; ++c) {
                auto acc = inv(CH[ch], Antt[r * 2 + c]); auto& gold = outG[ch][r * 2 + c];
                for (uint32_t i = 0; i < N; ++i) { if (acc[i] != gold[i]) { if (mism < 8) fmt::print("  ch{} r{} c{} [{}] got={} exp={}\n", ch, r, c, i, acc[i], gold[i]); ++mism; } ++checked; } }
        }
        fmt::print("scan-via-NTT (device, coeff-in/coeff-out)  N0={} rest={} KB={}  mismatches={}/{}\n", N0, rest, KB, mism, checked);
        pass = (mism == 0); pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
