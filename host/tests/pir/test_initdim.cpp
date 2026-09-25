// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Initial-dimension dot product (#12) on device — the throughput-critical PIR op, 2 RNS channels.
//   response[r][comp] = Σ_{k<N0} DB[k*rest+r] ⊛ query[k][comp]   (plaintext × ct, negacyclic, mod q).
// NTT-domain multiply-accumulate: DB (public plaintext) pre-nNTT'd + Montgomery on device; per (r,comp,ch)
//   acc_ntt = Σ_k query_ntt[k][comp] ⊙ DB_ntt[k*rest+r]  (modmul + modadd), then niNTT.
// Reuses fused_matmul_a/_c (NTT), modmul_compute, modadd_reduce — no new kernels. Validated bit-exact
// vs cpu/pir/gen_initdim_case.py (onionpir_ref pt_ct_mul + accumulate).
//
//   ./build/test_initdim [reference/pir/initdim_case.bin]

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
    const char* path = (argc > 1) ? argv[1] : "reference/pir/initdim_case.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fmt::print(stderr, "cannot open {}\n", path); return 2; }
    auto ru = [&](uint32_t& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    uint32_t magic, n, N0, rest, q1, q2;
    ru(magic); ru(n); ru(N0); ru(rest); ru(q1); ru(q2);
    if (magic != 0x494E4954u || n != N) { fmt::print(stderr, "bad case\n"); return 2; }
    fmt::print("case {}: n={} N0={} rest={} q1={} q2={}\n", path, n, N0, rest, q1, q2);
    auto rdvec = [&](uint32_t m) { std::vector<uint32_t> v(m); f.read(reinterpret_cast<char*>(v.data()), 4 * m); return v; };
    // query[ch][k*2+comp], DB[ch][idx], out[ch][r*2+comp]
    std::vector<std::vector<std::vector<uint32_t>>> Qc(2, std::vector<std::vector<uint32_t>>(N0 * 2));
    for (uint32_t ch = 0; ch < 2; ++ch) for (uint32_t k = 0; k < N0; ++k) for (uint32_t comp = 0; comp < 2; ++comp) Qc[ch][k * 2 + comp] = rdvec(N);
    std::vector<std::vector<std::vector<uint32_t>>> DBc(2, std::vector<std::vector<uint32_t>>(N0 * rest));
    for (uint32_t ch = 0; ch < 2; ++ch) for (uint32_t idx = 0; idx < N0 * rest; ++idx) DBc[ch][idx] = rdvec(N);
    std::vector<std::vector<std::vector<uint32_t>>> outG(2, std::vector<std::vector<uint32_t>>(rest * 2));
    for (uint32_t ch = 0; ch < 2; ++ch) for (uint32_t r = 0; r < rest; ++r) for (uint32_t comp = 0; comp < 2; ++comp) outG[ch][r * 2 + comp] = rdvec(N);

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
        auto rdbuf = [&](auto buf) { std::vector<uint32_t> v; distributed::EnqueueReadMeshBuffer(cq, v, buf, true); return v; };

        struct Chan { uint32_t q, qinv, qlo, qhi, k14; std::shared_ptr<distributed::MeshBuffer> WfA, Wr, Tm, WfAi, Wri, Tmi; };
        auto foldA = [&](const std::vector<uint32_t>& Wm, uint32_t q) { std::vector<int8_t> o(NB*NB*ELEMS);
            for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<uint32_t> l(ELEMS);
                for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[r]*Wm[i])%q),p); auto t=til(l);
                for (uint32_t i=0;i<ELEMS;++i) o[(p*NB+r)*ELEMS+i]=(int8_t)t[i]; } return o; };
        auto foldC = [&](const std::vector<uint32_t>& Wm, uint32_t q) { std::vector<int8_t> o(NB*NB*ELEMS);
            for (uint32_t p=0;p<NB;++p) for (uint32_t r=0;r<NB;++r){ std::vector<uint32_t> l(ELEMS);
                for (uint32_t i=0;i<ELEMS;++i) l[i]=limb((uint32_t)((POW[p]*Wm[i])%q),r); auto t=til(l);
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
        Chan CH[2] = { make_chan(q1), make_chan(q2) };

        auto b_in=mkbuf(1,I32), b_g=mkbuf(1,I32), b_gp=mkbuf(1,I32), b_h=mkbuf(1,I32);
        auto fusedA=[&](auto xin,auto w,auto out,const Chan& c){ Program p=CreateProgram();
            mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::Int8,2,I8); mkcb(p,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcb(p,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*xin).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)xin->address(),(uint32_t)w->address()}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14});
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };
        auto fusedC=[&](auto gp,auto w,auto out,const Chan& c){ Program p=CreateProgram();
            mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::Int8,NB*NB,I8); mkcb(p,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcb(p,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            std::vector<uint32_t> rc; TensorAccessorArgs(*gp).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)gp->address(),(uint32_t)w->address()}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14});
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };
        auto modmulP=[&](auto a,auto b,auto out,const Chan& c){ Program p=CreateProgram();
            mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_compute.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)a->address(),(uint32_t)b->address(),1u,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{1u,c.q,c.qinv,c.qlo,c.qhi});
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };

        auto reshape=[&](const std::vector<uint32_t>& x){ std::vector<uint32_t> X(ELEMS); for (uint32_t n2=0;n2<T;++n2) for (uint32_t n1=0;n1<T;++n1) X[n2*T+n1]=x[n1+T*n2]; return til(X); };
        auto dereindex=[&](const std::vector<uint32_t>& raw){ auto H=unt(raw); std::vector<uint32_t> X(N); for (uint32_t j1=0;j1<T;++j1) for (uint32_t j2=0;j2<T;++j2) X[j2+T*j1]=H[j2*T+j1]; return X; };
        auto fwd=[&](const Chan& c,const std::vector<uint32_t>& x){ wrbuf(b_in,reshape(x)); fusedA(b_in,c.WfA,b_g,c); modmulP(b_g,c.Tm,b_gp,c); fusedC(b_gp,c.Wr,b_h,c); return dereindex(rdbuf(b_h)); };
        auto inv=[&](const Chan& c,const std::vector<uint32_t>& X){ wrbuf(b_in,reshape(X)); fusedA(b_in,c.WfAi,b_g,c); modmulP(b_g,c.Tmi,b_gp,c); fusedC(b_gp,c.Wri,b_h,c); return dereindex(rdbuf(b_h)); };
        auto pmul=[&](const Chan& c,const std::vector<uint32_t>& a,const std::vector<uint32_t>& b){ auto ba=mkbuf(1,I32),bb=mkbuf(1,I32),bo=mkbuf(1,I32); wrbuf(ba,til(a)); wrbuf(bb,til(b)); modmulP(ba,bb,bo,c); return unt(rdbuf(bo)); };
        auto padd=[&](const Chan& c,const std::vector<std::vector<uint32_t>>& xs){ uint32_t nin=(uint32_t)xs.size(); auto bi=mkbuf(nin,I32),bo=mkbuf(1,I32);
            std::vector<uint32_t> buf(nin*ELEMS); for (uint32_t i=0;i<nin;++i){ auto t=til(xs[i]); std::copy(t.begin(),t.end(),buf.begin()+i*ELEMS); } wrbuf(bi,buf);
            Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,nin,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
            auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
            auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modadd_reduce.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
            SetRuntimeArgs(p,rd,core,{(uint32_t)bi->address(),nin,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)bo->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{nin,c.q});
            distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); return unt(rdbuf(bo)); };

        // setup: DB in device-nNTT Montgomery per channel; query nNTT plain per channel
        mont::MontgomeryContext ctxc[2] = { mont::MontgomeryContext(q1), mont::MontgomeryContext(q2) };
        std::vector<std::vector<std::vector<uint32_t>>> DBntt(2, std::vector<std::vector<uint32_t>>(N0*rest));
        for (uint32_t ch=0; ch<2; ++ch) for (uint32_t idx=0; idx<N0*rest; ++idx){ auto F=fwd(CH[ch], DBc[ch][idx]); for (auto& v:F) v=ctxc[ch].to_montgomery(v); DBntt[ch][idx]=F; }
        std::vector<std::vector<std::vector<uint32_t>>> Qntt(2, std::vector<std::vector<uint32_t>>(N0*2));
        for (uint32_t ch=0; ch<2; ++ch) for (uint32_t k=0;k<N0;++k) for (uint32_t comp=0;comp<2;++comp) Qntt[ch][k*2+comp]=fwd(CH[ch], Qc[ch][k*2+comp]);

        // initial dimension: acc[r][comp] = Σ_k query[k][comp] ⊙ DB[k*rest+r]
        uint32_t mism=0, checked=0;
        for (uint32_t r=0;r<rest;++r) for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<2;++ch){
            std::vector<std::vector<uint32_t>> prods(N0);
            for (uint32_t k=0;k<N0;++k) prods[k]=pmul(CH[ch], Qntt[ch][k*2+comp], DBntt[ch][k*rest+r]);
            auto acc=inv(CH[ch], padd(CH[ch], prods));
            auto& gold=outG[ch][r*2+comp];
            for (uint32_t i=0;i<N;++i){ if (acc[i]!=gold[i]){ if (mism<8) fmt::print("  r{} comp{} ch{} [{}] got={} exp={}\n",r,comp,ch,i,acc[i],gold[i]); ++mism; } ++checked; }
        }
        fmt::print("initial-dimension (device)  mismatches={}/{}\n", mism, checked);
        pass=(mism==0); pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("Test Passed\n"); else fmt::print("Test FAILED\n");
    return pass ? 0 : 1;
}
