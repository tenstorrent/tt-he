// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BV-style key-switch (#10) on device, end-to-end, two RNS channels.
//   (c0,c1) under s' -> (d0,d1) under s:  (d0,d1) = (c0,0) + Σ_k G^{-1}(c1)_k ⊛ KSK[k]  (ℓ×2 KSK).
// The external-product machinery restricted to ONE decomposed component (c1) plus adding c0.
// Chains validated pieces (no new kernels):  CRT-combine(c1) -> gadget -> ℓ digits -> digit_to_rns
//   -> nNTT/channel -> MAC vs KSK (ℓ rows, 2 cols) -> niNTT -> add c0 to column 0.
// KSK rows are nNTT'd ON-DEVICE. Validated bit-exact vs cpu/pir/gen_ks_case.py (onionpir_ref.keyswitch).
//
//   ./build/test_bv_keyswitch [reference/pir/ks_case.bin]

#include <cstdint>
#include <cstdio>
#include <fstream>
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
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {
constexpr uint32_t T = 32, NB = 4, N = 1024, ELEMS = 1024, I8 = ELEMS, I32 = ELEMS * 4;
const uint64_t POW[4] = {1, 128, 16384, 2097152};
using u128 = unsigned __int128;
uint64_t modpow(uint64_t b, uint64_t e, uint64_t m) { uint64_t r = 1; b %= m;
    while (e) { if (e & 1) r = (u128)r * b % m; b = (u128)b * b % m; e >>= 1; } return r; }
uint64_t prim_root(uint64_t q) { uint64_t phi = q - 1, n = phi; std::vector<uint64_t> f;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) { f.push_back(d); while (n % d == 0) n /= d; }
    if (n > 1) f.push_back(n);
    for (uint64_t g = 2; g < q; ++g) { bool ok = true; for (uint64_t p : f) if (modpow(g, phi / p, q) == 1) ok = false;
        if (ok) return g; } throw std::runtime_error("no root"); }
int8_t limb(uint32_t v, uint32_t p) { return (int8_t)((v >> (7 * p)) & 127); }
std::vector<uint32_t> til(const std::vector<uint32_t>& v) { return pir::tilize_poly(v, 1); }
std::vector<uint32_t> unt(const std::vector<uint32_t>& v) { return pir::untilize_poly(v, 1); }
}  // namespace

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "reference/pir/ks_case.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
    auto ru = [&](uint32_t& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    uint32_t magic, n, ell, B, q1, q2;
    // V2 header (KSWT): MAGIC, VERSION, n, ell, B, K, then K primes. V1 had q1/q2 inline and no
    // VERSION -- this reader was left behind when the generator was bumped, so it read VERSION as n.
    uint32_t ver, K;
    ru(magic); ru(ver); ru(n); ru(ell); ru(B); ru(K);
    if (magic != 0x4B535754u || ver != 2u || n != N) {
        fmt::print(stderr, "bad case (need KSWT V2 n={})\n", N); return 2;
    }
    std::vector<uint32_t> primes(K); f.read(reinterpret_cast<char*>(primes.data()), 4 * K);
    if (K < 2) { fmt::print(stderr, "bad case (need K>=2, got {})\n", K); return 2; }
    q1 = primes[0]; q2 = primes[1];
    const uint32_t LOG2B = __builtin_ctz(B);   // B is a power of two
    fmt::print("case {}: n={} ell={} B={} q1={} q2={}\n", path, n, ell, B, q1, q2);

    auto rdvec = [&](uint32_t m) { std::vector<uint32_t> v(m); f.read(reinterpret_cast<char*>(v.data()), 4 * m); return v; };
    auto rddig = [&]() { std::vector<int32_t> v(N); f.read(reinterpret_cast<char*>(v.data()), 4 * N); return v; };
    // layout: c0q1,c0q2,c1q1,c1q2, KSK[ch][k*2+col], out(d0q1,d0q2,d1q1,d1q2), digits[ell]
    std::vector<std::vector<uint32_t>> cch(4);            // c0q1,c0q2,c1q1,c1q2
    for (uint32_t i = 0; i < 4; ++i) cch[i] = rdvec(N);
    std::vector<std::vector<std::vector<uint32_t>>> Kc(2, std::vector<std::vector<uint32_t>>(ell * 2));  // [ch][k*2+col]
    for (uint32_t ch = 0; ch < 2; ++ch) for (uint32_t i = 0; i < ell * 2; ++i) Kc[ch][i] = rdvec(N);
    std::vector<std::vector<uint32_t>> outG(4);
    for (uint32_t i = 0; i < 4; ++i) outG[i] = rdvec(N);   // d0q1,d0q2,d1q1,d1q2
    std::vector<std::vector<int32_t>> digitsG(ell);
    for (uint32_t r = 0; r < ell; ++r) digitsG[r] = rddig();

    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto dr = distributed::MeshCoordinateRange(mesh->shape());
        constexpr CoreCoord core = {0, 0};
        auto mkbuf = [&](uint32_t nt, uint32_t pg) { distributed::DeviceLocalBufferConfig d{.page_size = pg, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = nt * pg}, d, mesh.get()); };
        auto mkcb = [&](Program& p, CBIndex idx, tt::DataFormat fmt, uint32_t depth, uint32_t pg) {
            CircularBufferConfig c(depth * pg, {{idx, fmt}}); c.set_page_size(idx, pg); CreateCircularBuffer(p, core, c); };
        auto wrbuf = [&](auto buf, const std::vector<uint32_t>& v) { distributed::EnqueueWriteMeshBuffer(cq, buf, v, false); };
        auto rdbuf = [&](auto buf, uint32_t ntiles) { std::vector<uint32_t> v; (void)ntiles; distributed::EnqueueReadMeshBuffer(cq, v, buf, true); return v; };

        // ---- per-channel NTT constants (nntt + nintt), as in test_negconv ----
        struct Chan {
            uint32_t q, qinv, qlo, qhi, k14;
            std::shared_ptr<distributed::MeshBuffer> WfA, Wr, Tm, WfAi, Wri, Tmi;
        };
        auto foldA = [&](const std::vector<uint32_t>& Wm, uint32_t q) { std::vector<int8_t> o(NB*NB*ELEMS);
            for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<uint32_t> l(ELEMS);
                for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[r]*Wm[i])%q),p); auto t=til(l);
                for (uint32_t i=0;i<ELEMS;++i) o[(p*NB+r)*ELEMS+i]=(int8_t)t[i]; } return o; };
        auto foldC = [&](const std::vector<uint32_t>& Wm, uint32_t q) { std::vector<int8_t> o(NB*NB*ELEMS);
            for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<uint32_t> l(ELEMS);
                for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[p]*Wm[i])%q),r); auto t=til(l);
                for (uint32_t i=0;i<ELEMS;++i) o[(p*NB+r)*ELEMS+i]=(int8_t)t[i]; } return o; };

        auto make_chan = [&](uint32_t q) {
            mont::MontgomeryContext ctx(q); Chan c; c.q=q; c.qinv=ctx.q_inv_32(); c.qlo=q&0xFFFF; c.qhi=q>>16; c.k14=ctx.to_montgomery(16384u);
            uint64_t g=prim_root(q), w=modpow(g,(q-1)/N,q), w32=modpow(g,(q-1)/T,q), psi=modpow(g,(q-1)/(2*N),q);
            std::vector<uint32_t> Wc(ELEMS),Wa(ELEMS),Tw(ELEMS);
            for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b){ uint32_t wv=(uint32_t)modpow(w32,(uint64_t)a*b,q);
                Wc[a*T+b]=wv; Wa[a*T+b]=(uint32_t)((u128)wv*modpow(psi,32ull*b,q)%q); }
            for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b) Tw[a*T+b]=(uint32_t)((u128)modpow(w,(uint64_t)a*b,q)*modpow(psi,b,q)%q);
            uint64_t wi=mont::modular_inverse(w,q),w32i=mont::modular_inverse(w32,q),psii=mont::modular_inverse(psi,q),Ninv=mont::modular_inverse(N,q);
            std::vector<uint32_t> Wci(ELEMS),Wai(ELEMS),Twi(ELEMS);
            for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b){ uint32_t wv=(uint32_t)modpow(w32i,(uint64_t)a*b,q);
                Wci[a*T+b]=(uint32_t)((u128)wv*modpow(psii,32ull*b,q)%q); Wai[a*T+b]=(uint32_t)((u128)wv*Ninv%q); }
            for (uint32_t a=0;a<T;++a) for (uint32_t b=0;b<T;++b) Twi[a*T+b]=(uint32_t)((u128)modpow(wi,(uint64_t)a*b,q)*modpow(psii,a,q)%q);
            auto mont_til=[&](const std::vector<uint32_t>& v){ std::vector<uint32_t> m(ELEMS); for(uint32_t i=0;i<ELEMS;++i) m[i]=ctx.to_montgomery(v[i]); return til(m); };
            c.WfA=mkbuf(NB*NB,I8); c.Wr=mkbuf(NB*NB,I8); c.Tm=mkbuf(1,I32); c.WfAi=mkbuf(NB*NB,I8); c.Wri=mkbuf(NB*NB,I8); c.Tmi=mkbuf(1,I32);
            { auto b=foldA(Wa,q); std::vector<uint32_t> u(b.begin(),b.end()); }
            auto i8vec=[&](const std::vector<int8_t>& b){ return std::vector<int8_t>(b); };
            distributed::EnqueueWriteMeshBuffer(cq,c.WfA,foldA(Wa,q),false); distributed::EnqueueWriteMeshBuffer(cq,c.Wr,foldC(Wc,q),false);
            wrbuf(c.Tm,mont_til(Tw)); distributed::EnqueueWriteMeshBuffer(cq,c.WfAi,foldA(Wai,q),false);
            distributed::EnqueueWriteMeshBuffer(cq,c.Wri,foldC(Wci,q),false); wrbuf(c.Tmi,mont_til(Twi)); (void)i8vec;
            return c; };
        Chan CH[2] = { make_chan(q1), make_chan(q2) };

        // scratch buffers reused across NTT calls
        auto b_in=mkbuf(1,I32), b_g=mkbuf(1,I32), b_gp=mkbuf(1,I32), b_h=mkbuf(1,I32);
        auto fusedA=[&](auto xin,auto w,auto out,const Chan& c){ Program p=CreateProgram();
            mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::Int8,2,I8);
            mkcb(p,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcb(p,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*xin).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)xin->address(),(uint32_t)w->address()}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,0u});
            SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14}); distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };
        auto fusedC=[&](auto gp,auto w,auto out,const Chan& c){ Program p=CreateProgram();
            mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::Int8,NB*NB,I8);
            mkcb(p,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcb(p,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*gp).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)gp->address(),(uint32_t)w->address()}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,0u});
            SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14}); distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };
        // modmul(a,b) mod c.q  (a,b tiles) -> out tile
        auto modmulP=[&](auto a,auto b,auto out,const Chan& c){ Program p=CreateProgram();
            mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)a->address(),(uint32_t)b->address(),1u,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,0u});
            SetRuntimeArgs(p,cp,core,{1u,c.q,c.qinv,c.qlo,c.qhi}); distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };

        // host-vector NTT wrappers (natural order in/out), like test_negconv
        auto reshape=[&](const std::vector<uint32_t>& x){ std::vector<uint32_t> X(ELEMS);
            for (uint32_t n2=0;n2<T;++n2) for (uint32_t n1=0;n1<T;++n1) X[n2*T+n1]=x[n1+T*n2]; return til(X); };
        auto dereindex=[&](const std::vector<uint32_t>& raw){ auto H=unt(raw); std::vector<uint32_t> X(N);
            for (uint32_t j1=0;j1<T;++j1) for (uint32_t j2=0;j2<T;++j2) X[j2+T*j1]=H[j2*T+j1]; return X; };
        auto fwd=[&](const Chan& c,const std::vector<uint32_t>& x){ wrbuf(b_in,reshape(x)); fusedA(b_in,c.WfA,b_g,c); modmulP(b_g,c.Tm,b_gp,c); fusedC(b_gp,c.Wr,b_h,c);
            return dereindex(rdbuf(b_h,1)); };
        auto inv=[&](const Chan& c,const std::vector<uint32_t>& X){ wrbuf(b_in,reshape(X)); fusedA(b_in,c.WfAi,b_g,c); modmulP(b_g,c.Tmi,b_gp,c); fusedC(b_gp,c.Wri,b_h,c);
            return dereindex(rdbuf(b_h,1)); };
        // pointwise modmul on natural vectors (a plain, b montgomery) -> plain
        auto pmul=[&](const Chan& c,const std::vector<uint32_t>& a,const std::vector<uint32_t>& b){
            auto ba=mkbuf(1,I32), bb=mkbuf(1,I32), bo=mkbuf(1,I32); wrbuf(ba,til(a)); wrbuf(bb,til(b)); modmulP(ba,bb,bo,c); return unt(rdbuf(bo,1)); };

        // ---- CRT-combine + gadget: d (per channel) -> ℓ signed digits (natural) ----
        mont::MontgomeryContext ctx2(q2);
        const uint32_t qinv2=ctx2.q_inv_32(), qlo2=q2&0xFFFF, qhi2=q2>>16;
        const uint32_t inv_mont=ctx2.to_montgomery((uint32_t)mont::modular_inverse(q1,q2)), q1L=q1&0xFFFF, q1H=q1>>16;
        auto b_parQ=mkbuf(2,I32); { std::vector<uint32_t> par(2*ELEMS); auto tL=til(std::vector<uint32_t>(ELEMS,q1L)); auto tH=til(std::vector<uint32_t>(ELEMS,q1H));
            std::copy(tL.begin(),tL.end(),par.begin()); std::copy(tH.begin(),tH.end(),par.begin()+ELEMS); wrbuf(b_parQ,par); }

        auto crt_and_gadget=[&](const std::vector<uint32_t>& dq1,const std::vector<uint32_t>& dq2){
            // garner_diff: diff = (dq2 - dq1) mod q2  (reader_ab: A=dq1->c_0, B=dq2->c_1)
            auto ba=mkbuf(1,I32), bb=mkbuf(1,I32), bdiff=mkbuf(1,I32), bt=mkbuf(1,I32), bX=mkbuf(2,I32);
            wrbuf(ba,til(dq1)); wrbuf(bb,til(dq2));
            { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
              auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/garner_diff.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              SetRuntimeArgs(p,rd,core,{(uint32_t)ba->address(),(uint32_t)bb->address(),1u,0u,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)bdiff->address(),1u,0u});
              SetRuntimeArgs(p,cp,core,{q2,1u}); distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); }
            // t = diff * inv_mont mod q2
            { auto bimont=mkbuf(1,I32); wrbuf(bimont,til(std::vector<uint32_t>(N,inv_mont)));
              Chan c2; c2.q=q2; c2.qinv=qinv2; c2.qlo=qlo2; c2.qhi=qhi2; modmulP(bdiff,bimont,bt,c2); }
            // crt_mul_split: (t, dq1, par) -> X0,X1
            { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_2,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,4,I32);
              auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_crt.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/crt_mul_split.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
              SetRuntimeArgs(p,rd,core,{(uint32_t)bt->address(),(uint32_t)ba->address(),(uint32_t)b_parQ->address(),1u}); SetRuntimeArgs(p,wr,core,{(uint32_t)bX->address(),2u,0u});
              SetRuntimeArgs(p,cp,core,{1u}); distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); }
            // gadget_decompose(X0,X1) -> ell digit tiles. bX holds [X0, X1] as consecutive tiles;
            // reader_i32 streams both into c_0 (the kernel consumes 2 tiles per coefficient-tile).
            auto bdig=mkbuf(ell,I32);
            { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,4,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2*ell,I32);
              auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
              auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
              auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/gadget_decompose_signed.cpp",core,
                  ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false,
                                .defines={{"GADGET_B",std::to_string(B)},{"GADGET_LOG2B",std::to_string(LOG2B)},{"GADGET_ELL",std::to_string(ell)}}});
              SetRuntimeArgs(p,rd,core,{(uint32_t)bX->address(),2u,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)bdig->address(),ell,0u});
              SetRuntimeArgs(p,cp,core,{1u}); distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); }
            auto raw=rdbuf(bdig,ell); std::vector<std::vector<uint32_t>> digs(ell);
            for (uint32_t k=0;k<ell;++k){ std::vector<uint32_t> t(raw.begin()+k*ELEMS,raw.begin()+(k+1)*ELEMS); digs[k]=unt(t); }
            return digs; };  // ell natural-order signed-digit vectors (as uint32 bit patterns)

        // digit_to_rns: signed digit (natural, uint32 bits) -> (mod q1, mod q2) natural
        auto digit_rns=[&](const std::vector<uint32_t>& dig){
            auto bi=mkbuf(1,I32), bo=mkbuf(2,I32); wrbuf(bi,til(dig));
            Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,4,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/digit_to_rns.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)bi->address(),1u,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)bo->address(),2u,0u}); SetRuntimeArgs(p,cp,core,{1u,q1,q2});
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false);
            auto raw=rdbuf(bo,2); std::vector<uint32_t> a(raw.begin(),raw.begin()+ELEMS), b(raw.begin()+ELEMS,raw.end());
            return std::make_pair(unt(a),unt(b)); };

        // modadd of a list (natural) mod c.q via modadd_reduce (n_in tiles)
        auto padd=[&](const Chan& c,const std::vector<std::vector<uint32_t>>& xs){
            uint32_t nin=(uint32_t)xs.size(); auto bi=mkbuf(nin,I32), bo=mkbuf(1,I32);
            std::vector<uint32_t> buf(nin*ELEMS); for (uint32_t i=0;i<nin;++i){ auto t=til(xs[i]); std::copy(t.begin(),t.end(),buf.begin()+i*ELEMS); } wrbuf(bi,buf);
            Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,nin,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)bi->address(),nin,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)bo->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{nin,c.q});
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); return unt(rdbuf(bo,1)); };

        // ---- setup: KSK in device-nNTT Montgomery per channel ----  Kntt[ch][k*2+col]
        mont::MontgomeryContext ctxc[2] = { mont::MontgomeryContext(q1), mont::MontgomeryContext(q2) };
        std::vector<std::vector<std::vector<uint32_t>>> Kntt(2, std::vector<std::vector<uint32_t>>(ell*2));
        for (uint32_t ch=0; ch<2; ++ch) for (uint32_t k=0;k<ell;++k) for (uint32_t col=0;col<2;++col){
            auto F=fwd(CH[ch], Kc[ch][k*2+col]);
            for (auto& v:F) v=ctxc[ch].to_montgomery(v);
            Kntt[ch][k*2+col]=F; }

        // ---- front-half: CRT-combine + gadget on c1 -> ell digits ----
        auto digC1=crt_and_gadget(cch[2], cch[3]);                  // c1 channels q1,q2
        { uint32_t dm=0; for (uint32_t k=0;k<ell;++k) for (uint32_t i=0;i<N;++i) if (digC1[k][i]!=(uint32_t)digitsG[k][i]) ++dm;
          fmt::print("  front-half (CRT+gadget) digit mismatches={}\n", dm); pass &= (dm==0); }

        // digit_to_rns + nNTT per channel
        std::vector<std::vector<std::vector<uint32_t>>> digNtt(2, std::vector<std::vector<uint32_t>>(ell));
        for (uint32_t k=0;k<ell;++k){ auto pr=digit_rns(digC1[k]);
            digNtt[0][k]=fwd(CH[0],pr.first); digNtt[1][k]=fwd(CH[1],pr.second); }

        // MAC vs KSK (ell rows) -> niNTT -> add c0 to column 0
        uint32_t mism=0, checked=0;
        for (uint32_t col=0; col<2; ++col) for (uint32_t ch=0; ch<2; ++ch){
            std::vector<std::vector<uint32_t>> prods(ell);
            for (uint32_t k=0;k<ell;++k) prods[k]=pmul(CH[ch], digNtt[ch][k], Kntt[ch][k*2+col]);
            auto mac_coeff=inv(CH[ch], padd(CH[ch], prods));
            std::vector<uint32_t> d = (col==0) ? padd(CH[ch], {cch[ch], mac_coeff}) : mac_coeff;  // d0=c0+mac0, d1=mac1
            auto& gold=outG[col*2+ch];
            for (uint32_t i=0;i<N;++i){ if (d[i]!=gold[i]){ if (mism<8) fmt::print("  col{} ch{} [{}] got={} exp={}\n",col,ch,i,d[i],gold[i]); ++mism; } ++checked; }
        }
        fmt::print("BV key-switch (device)  mismatches={}/{}\n", mism, checked);
        pass=(mism==0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
