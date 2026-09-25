// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// OnionPIR API implementation — the SERVER device pipeline + client decrypt.  All TT-Metalium
// control lives here; pir_demo.cpp just narrates.  Composes the validated PIR blocks (initial
// dimension #12, external product #16, mod-switch #11).  Params n=1024, composite q=q1·q2 (28-bit),
// gadget B=2^20/ℓ from the case.
#include "pir_demo_api.hpp"

#include <cstdint>
#include <cmath>
#include <fstream>
#include <stdexcept>
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
        if (ok) return g; } throw std::runtime_error("no primitive root"); }
int8_t limb(uint32_t v, uint32_t p) { return (int8_t)((v >> (7 * p)) & 127); }
std::vector<uint32_t> til(const std::vector<uint32_t>& v) { return pir::tilize_poly(v, 1); }
std::vector<uint32_t> unt(const std::vector<uint32_t>& v) { return pir::untilize_poly(v, 1); }
// client-side negacyclic multiply mod m (for BFV decrypt of the mod-switched response)
std::vector<uint32_t> negmul(const std::vector<uint32_t>& a, const std::vector<int32_t>& b, uint64_t m) {
    uint32_t nn = a.size(); std::vector<int64_t> c(nn, 0);
    for (uint32_t i = 0; i < nn; ++i) for (uint32_t j = 0; j < nn; ++j) {
        int64_t v = (int64_t)((u128)a[i] * ((b[j] % (int64_t)m + m) % m) % m); uint32_t k = i + j;
        if (k < nn) c[k] = (c[k] + v) % m; else c[k - nn] = (c[k - nn] - v % m + m) % m; }
    std::vector<uint32_t> o(nn); for (uint32_t i = 0; i < nn; ++i) o[i] = (uint32_t)((c[i] % (int64_t)m + m) % m); return o; }
// Galois automorphism x->x^j (host reindex, signed) and monomial x^k, on a coeff poly mod q.
std::vector<uint32_t> automorph_poly(const std::vector<uint32_t>& poly, int64_t j, uint32_t q) {
    const int64_t twon = 2 * (int64_t)N; std::vector<uint32_t> r(N, 0);
    for (uint32_t i = 0; i < N; ++i) { int64_t p = ((int64_t)i * j) % twon; if (p < 0) p += twon; uint32_t v = poly[i];
        if (p < (int64_t)N) r[p] = (uint32_t)(((uint64_t)r[p] + v) % q); else r[p - N] = (uint32_t)(((uint64_t)r[p - N] + q - v) % q); }
    return r; }
std::vector<uint32_t> mono_mul_poly(const std::vector<uint32_t>& poly, int64_t k, uint32_t q) {
    const int64_t twon = 2 * (int64_t)N; std::vector<uint32_t> r(N, 0);
    for (uint32_t i = 0; i < N; ++i) { int64_t p = ((int64_t)i + k) % twon; if (p < 0) p += twon; uint32_t v = poly[i];
        if (p < (int64_t)N) r[p] = (uint32_t)(((uint64_t)r[p] + v) % q); else r[p - N] = (uint32_t)(((uint64_t)r[p - N] + q - v) % q); }
    return r; }
}  // namespace

namespace pir {

Case load_case(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot open case file ") + path);
    auto ru = [&](uint32_t& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    uint32_t magic; Case c;
    ru(magic); ru(c.n); ru(c.ell); ru(c.B); ru(c.N0); ru(c.m_exp); ru(c.q1); ru(c.q2); ru(c.qp); ru(c.idx); ru(c.t);
    if (magic != 0x50495245u || c.n != N) throw std::runtime_error("bad PIR case file (need 'PIRE' full-OnionPIR format)");
    const uint32_t twoL = 2 * c.ell, NE = c.N0 * 2;
    uint32_t m_lad = 0; while ((1u << m_lad) < c.ell) ++m_lad;             // fold-ladder expand depth
    const uint32_t mk = c.m_exp > m_lad ? c.m_exp : m_lad;                 // Galois keys cover max levels
    auto rv = [&]() { std::vector<uint32_t> v(N); f.read(reinterpret_cast<char*>(v.data()), 4 * N); return v; };
    c.DB.assign(2, std::vector<std::vector<uint32_t>>(NE));
    for (uint32_t ch = 0; ch < 2; ++ch) for (uint32_t e = 0; e < NE; ++e) c.DB[ch][e] = rv();
    c.query.assign(4, std::vector<uint32_t>());                 // packed init-dim query: c0q1,c0q2,c1q1,c1q2
    for (uint32_t comp = 0; comp < 2; ++comp) for (uint32_t ch = 0; ch < 2; ++ch) c.query[comp * 2 + ch] = rv();
    c.qfold.assign(4, std::vector<uint32_t>());                 // packed fold-ladder query (scaleless)
    for (uint32_t comp = 0; comp < 2; ++comp) for (uint32_t ch = 0; ch < 2; ++ch) c.qfold[comp * 2 + ch] = rv();
    c.gkeys.assign(mk * 2, std::vector<std::vector<uint32_t>>(c.ell * 2)); // per level: per ch: ell*2
    for (uint32_t a = 0; a < mk; ++a) for (uint32_t ch = 0; ch < 2; ++ch) for (uint32_t k = 0; k < c.ell; ++k) for (uint32_t col = 0; col < 2; ++col) c.gkeys[a * 2 + ch][k * 2 + col] = rv();
    c.gsw_s.assign(2, std::vector<std::vector<uint32_t>>(twoL * 2));        // RGSW(s): per ch: 2ell*2
    for (uint32_t ch = 0; ch < 2; ++ch) for (uint32_t i = 0; i < twoL * 2; ++i) c.gsw_s[ch][i] = rv();
    c.skey.resize(N); f.read(reinterpret_cast<char*>(c.skey.data()), 4 * N);
    c.expected.resize(N); f.read(reinterpret_cast<char*>(c.expected.data()), 4 * N);
    return c;
}

Response server_answer(const Case& cs) {
    const uint32_t ell = cs.ell, B = cs.B, N0 = cs.N0, q1 = cs.q1, q2 = cs.q2, qp = cs.qp;
    const uint32_t twoL = 2 * ell, NE = N0 * 2;
    uint32_t LOG2B = 0; while ((1u << LOG2B) < B) ++LOG2B;
    const uint64_t q = (uint64_t)q1 * q2;
    const uint32_t m_exp = cs.m_exp;
    const auto& DBc = cs.DB;
    // Sc (N0 init-dim selectors) and Cc (fold RGSW, 2ℓ×2) are produced ON DEVICE below by
    // expand (Subs tree) + unpack (RLWE->RGSW), not taken pre-expanded from the case.
    std::vector<std::vector<std::vector<uint32_t>>> Sc(2, std::vector<std::vector<uint32_t>>(N0 * 2));
    std::vector<std::vector<std::vector<uint32_t>>> Cc(2, std::vector<std::vector<uint32_t>>(twoL * 2));

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
    const uint32_t QCH[2] = {q1, q2};

    auto b_in=mkbuf(1,I32), b_g=mkbuf(1,I32), b_gp=mkbuf(1,I32), b_h=mkbuf(1,I32);
    auto fusedA=[&](auto xin,auto w,auto out,const Chan& c){ Program p=CreateProgram();
        mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::Int8,2,I8); mkcb(p,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcb(p,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*xin).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_a_1pass.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)xin->address(),(uint32_t)w->address()}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14});
        distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };
    auto fusedC=[&](auto gp,auto w,auto out,const Chan& c){ Program p=CreateProgram();
        mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::Int8,NB*NB,I8); mkcb(p,CBIndex::c_2,tt::DataFormat::Int8,NB,I8); mkcb(p,CBIndex::c_3,tt::DataFormat::Int32,NB,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        std::vector<uint32_t> rc; TensorAccessorArgs(*gp).append_to(rc); TensorAccessorArgs(*w).append_to(rc);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_fused.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=rc});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/fused_matmul_c_1pass.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)gp->address(),(uint32_t)w->address()}); SetRuntimeArgs(p,wr,core,{(uint32_t)out->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{c.q,c.qinv,c.qlo,c.qhi,c.k14});
        distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); };
    auto modmulP=[&](auto a,auto b,auto out,const Chan& c){ Program p=CreateProgram();
        mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_1pass.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
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

    // CRT-combine + gadget: d (per channel) -> ℓ signed digits (natural)
    mont::MontgomeryContext ctx2(q2);
    const uint32_t qinv2=ctx2.q_inv_32(), qlo2=q2&0xFFFF, qhi2=q2>>16;
    const uint32_t inv_mont=ctx2.to_montgomery((uint32_t)mont::modular_inverse(q1,q2)), q1L=q1&0xFFFF, q1H=q1>>16;
    auto b_parQ=mkbuf(2,I32); { std::vector<uint32_t> par(2*ELEMS); auto tL=til(std::vector<uint32_t>(ELEMS,q1L)); auto tH=til(std::vector<uint32_t>(ELEMS,q1H));
        std::copy(tL.begin(),tL.end(),par.begin()); std::copy(tH.begin(),tH.end(),par.begin()+ELEMS); wrbuf(b_parQ,par); }
    auto crt_and_gadget=[&](const std::vector<uint32_t>& dq1,const std::vector<uint32_t>& dq2){
        auto ba=mkbuf(1,I32), bb=mkbuf(1,I32), bdiff=mkbuf(1,I32), bt=mkbuf(1,I32), bX=mkbuf(2,I32);
        wrbuf(ba,til(dq1)); wrbuf(bb,til(dq2));
        { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
          auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
          auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
          auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/garner_diff.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
          SetRuntimeArgs(p,rd,core,{(uint32_t)ba->address(),(uint32_t)bb->address(),1u,0u,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)bdiff->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{q2,1u});
          distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); }
        { auto bimont=mkbuf(1,I32); wrbuf(bimont,til(std::vector<uint32_t>(N,inv_mont)));
          Chan c2; c2.q=q2; c2.qinv=qinv2; c2.qlo=qlo2; c2.qhi=qhi2; modmulP(bdiff,bimont,bt,c2); }
        { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_2,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,4,I32);
          auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_crt.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
          auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
          auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/crt_mul_split.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
          SetRuntimeArgs(p,rd,core,{(uint32_t)bt->address(),(uint32_t)ba->address(),(uint32_t)b_parQ->address(),1u}); SetRuntimeArgs(p,wr,core,{(uint32_t)bX->address(),2u,0u}); SetRuntimeArgs(p,cp,core,{1u});
          distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); }
        auto bdig=mkbuf(ell,I32);
        { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,4,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2*ell,I32);
          auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
          auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
          auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/gadget_decompose_signed.cpp",core,
              ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false,
                            .defines={{"GADGET_B",std::to_string(B)},{"GADGET_LOG2B",std::to_string(LOG2B)},{"GADGET_ELL",std::to_string(ell)}}});
          SetRuntimeArgs(p,rd,core,{(uint32_t)bX->address(),2u,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)bdig->address(),ell,0u}); SetRuntimeArgs(p,cp,core,{1u});
          distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); }
        auto raw=rdbuf(bdig); std::vector<std::vector<uint32_t>> digs(ell);
        for (uint32_t k=0;k<ell;++k){ std::vector<uint32_t> t(raw.begin()+k*ELEMS,raw.begin()+(k+1)*ELEMS); digs[k]=unt(t); }
        return digs; };
    auto digit_rns=[&](const std::vector<uint32_t>& dig){
        auto bi=mkbuf(1,I32), bo=mkbuf(2,I32); wrbuf(bi,til(dig));
        Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,4,I32);
        auto rd=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto wr=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/digit_to_rns.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
        SetRuntimeArgs(p,rd,core,{(uint32_t)bi->address(),1u,0u}); SetRuntimeArgs(p,wr,core,{(uint32_t)bo->address(),2u,0u}); SetRuntimeArgs(p,cp,core,{1u,q1,q2});
        distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false);
        auto raw=rdbuf(bo); std::vector<uint32_t> a(raw.begin(),raw.begin()+ELEMS), b(raw.begin()+ELEMS,raw.end());
        return std::make_pair(unt(a),unt(b)); };

    // external product: ep = C ⊡ d  (d per component/channel; returns ep[col*2+ch] coeff)
    auto extprod=[&](const std::vector<uint32_t>& d0q1,const std::vector<uint32_t>& d0q2,
                     const std::vector<uint32_t>& d1q1,const std::vector<uint32_t>& d1q2,
                     const std::vector<std::vector<std::vector<uint32_t>>>& Cn){
        auto dg0=crt_and_gadget(d0q1,d0q2), dg1=crt_and_gadget(d1q1,d1q2);
        std::vector<std::vector<uint32_t>> digU(twoL);
        for (uint32_t k=0;k<ell;++k){ digU[k]=dg0[k]; digU[ell+k]=dg1[k]; }
        std::vector<std::vector<std::vector<uint32_t>>> dN(2, std::vector<std::vector<uint32_t>>(twoL));
        for (uint32_t r=0;r<twoL;++r){ auto pr=digit_rns(digU[r]); dN[0][r]=fwd(CH[0],pr.first); dN[1][r]=fwd(CH[1],pr.second); }
        std::vector<std::vector<uint32_t>> ep(4);
        for (uint32_t col=0;col<2;++col) for (uint32_t ch=0;ch<2;++ch){
            std::vector<std::vector<uint32_t>> prods(twoL);
            for (uint32_t r=0;r<twoL;++r) prods[r]=pmul(CH[ch], dN[ch][r], Cn[ch][r*2+col]);
            ep[col*2+ch]=inv(CH[ch], padd(CH[ch], prods)); }
        return ep; };

    // =================== EXPAND (Subs tree) + UNPACK (RLWE->RGSW), on device ===================
    using CtV = std::vector<std::vector<uint32_t>>;                 // 4 polys [comp*2+ch]: c0q1,c0q2,c1q1,c1q2
    mont::MontgomeryContext ctxE[2] = { mont::MontgomeryContext(q1), mont::MontgomeryContext(q2) };
    // pre-nNTT (Montgomery) the per-level Galois keys and RGSW(s)
    uint32_t m_lad = 0; while ((1u << m_lad) < ell) ++m_lad;             // fold-ladder expand depth
    const uint32_t mk = m_exp > m_lad ? m_exp : m_lad;                   // Galois keys cover max levels
    std::vector<std::vector<std::vector<std::vector<uint32_t>>>> KnttL(mk,
        std::vector<std::vector<std::vector<uint32_t>>>(2, std::vector<std::vector<uint32_t>>(ell * 2)));
    for (uint32_t a=0;a<mk;++a) for (uint32_t ch=0;ch<2;++ch) for (uint32_t i=0;i<ell*2;++i){
        auto F=fwd(CH[ch], cs.gkeys[a*2+ch][i]); for (auto& v:F) v=ctxE[ch].to_montgomery(v); KnttL[a][ch][i]=F; }
    std::vector<std::vector<std::vector<uint32_t>>> GSn(2, std::vector<std::vector<uint32_t>>(twoL*2));
    for (uint32_t ch=0;ch<2;++ch) for (uint32_t i=0;i<twoL*2;++i){
        auto F=fwd(CH[ch], cs.gsw_s[ch][i]); for (auto& v:F) v=ctxE[ch].to_montgomery(v); GSn[ch][i]=F; }

    // device BV key-switch: automorphed (c0,c1 per ch) + KSK(NTT-Mont) -> (d0,d1 per ch)
    auto keyswitch=[&](const CtV& ct, const std::vector<std::vector<uint32_t>>& K0, const std::vector<std::vector<uint32_t>>& K1)->CtV{
        auto digC1=crt_and_gadget(ct[2], ct[3]);
        std::vector<std::vector<std::vector<uint32_t>>> digNtt(2, std::vector<std::vector<uint32_t>>(ell));
        for (uint32_t k=0;k<ell;++k){ auto pr=digit_rns(digC1[k]); digNtt[0][k]=fwd(CH[0],pr.first); digNtt[1][k]=fwd(CH[1],pr.second); }
        const std::vector<std::vector<uint32_t>>* K[2]={&K0,&K1}; CtV d(4);
        for (uint32_t col=0;col<2;++col) for (uint32_t ch=0;ch<2;++ch){
            std::vector<std::vector<uint32_t>> prods(ell);
            for (uint32_t k=0;k<ell;++k) prods[k]=pmul(CH[ch], digNtt[ch][k], (*K[ch])[k*2+col]);
            auto mac=inv(CH[ch], padd(CH[ch], prods));
            d[col*2+ch] = (col==0) ? padd(CH[ch], {ct[ch], mac}) : mac; }
        return d; };
    auto automorph=[&](const CtV& ct, int64_t j)->CtV{ CtV r(4); for (uint32_t i=0;i<4;++i) r[i]=automorph_poly(ct[i], j, QCH[i&1]); return r; };
    auto monomul  =[&](const CtV& ct, int64_t k)->CtV{ CtV r(4); for (uint32_t i=0;i<4;++i) r[i]=mono_mul_poly(ct[i], k, QCH[i&1]); return r; };
    auto ctadd    =[&](const CtV& a, const CtV& b)->CtV{ CtV r(4); for (uint32_t i=0;i<4;++i){ uint32_t qc=QCH[i&1]; r[i].resize(N);
        for (uint32_t x=0;x<N;++x) r[i][x]=(a[i][x]+b[i][x])%qc; } return r; };
    auto subs=[&](const CtV& ct, int64_t j, uint32_t a)->CtV{ return keyswitch(automorph(ct, j), KnttL[a][0], KnttL[a][1]); };
    // Subs tree: one packed ct -> 2^levels expanded cts (same routine for init-dim one-hot AND fold ladder)
    auto expand=[&](const CtV& q0, uint32_t levels)->std::vector<CtV>{
        std::vector<CtV> cts = { q0 };
        for (uint32_t a=0;a<levels;++a){ int64_t j=(int64_t)(N>>a)+1; std::vector<CtV> nw(2*cts.size());
            for (uint32_t b=0;b<cts.size();++b){ const CtV& c=cts[b];
                nw[b]=ctadd(c, subs(c,j,a));                             // even half
                CtV c_sh=monomul(c, -(int64_t)(1u<<a));                  // x^{-2^a}
                nw[b+cts.size()]=ctadd(c_sh, subs(c_sh,j,a)); }          // odd half
            cts=std::move(nw); }
        return cts; };

    // (0a) EXPAND init-dim one-hot -> N0 selector cts
    { auto sc=expand(cs.query, m_exp);
      for (uint32_t k=0;k<N0;++k) for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<2;++ch)
        Sc[ch][k*2+comp]=sc[k][comp*2+ch]; }

    // (0b) EXPAND fold ladder -> ℓ message rows, then UNPACK: RGSW(s) ⊡ row -> RGSW(bit) (top=rows, bottom=extprod)
    { auto mr=expand(cs.qfold, m_lad);
      for (uint32_t k=0;k<ell;++k){
        for (uint32_t ch=0;ch<2;++ch){ Cc[ch][k*2+0]=mr[k][0*2+ch]; Cc[ch][k*2+1]=mr[k][1*2+ch]; }
        auto ep=extprod(mr[k][0], mr[k][1], mr[k][2], mr[k][3], GSn);
        for (uint32_t col=0;col<2;++col) for (uint32_t ch=0;ch<2;++ch) Cc[ch][(ell+k)*2+col]=ep[col*2+ch]; } }

    // ---- (a) pre-nNTT the public DB (Montgomery), the query selectors, and the RGSW fold key ----
    mont::MontgomeryContext ctxc[2] = { mont::MontgomeryContext(q1), mont::MontgomeryContext(q2) };
    std::vector<std::vector<std::vector<uint32_t>>> DBn(2, std::vector<std::vector<uint32_t>>(NE));
    for (uint32_t ch=0;ch<2;++ch) for (uint32_t e=0;e<NE;++e){ auto F=fwd(CH[ch], DBc[ch][e]); for (auto& v:F) v=ctxc[ch].to_montgomery(v); DBn[ch][e]=F; }
    std::vector<std::vector<std::vector<uint32_t>>> Sn(2, std::vector<std::vector<uint32_t>>(N0*2));
    for (uint32_t ch=0;ch<2;++ch) for (uint32_t k=0;k<N0;++k) for (uint32_t comp=0;comp<2;++comp) Sn[ch][k*2+comp]=fwd(CH[ch], Sc[ch][k*2+comp]);
    std::vector<std::vector<std::vector<uint32_t>>> C1n(2, std::vector<std::vector<uint32_t>>(twoL*2));
    for (uint32_t ch=0;ch<2;++ch) for (uint32_t i=0;i<twoL*2;++i){ auto F=fwd(CH[ch], Cc[ch][i]); for (auto& v:F) v=ctxc[ch].to_montgomery(v); C1n[ch][i]=F; }

    // ---- (b) initial dimension: acc[r][comp][ch] = Σ_k sel0[k][comp] ⊙ DB[k*2+r] ----
    std::vector<std::vector<uint32_t>> acc0(4), acc1(4);
    for (uint32_t r=0;r<2;++r) for (uint32_t comp=0;comp<2;++comp) for (uint32_t ch=0;ch<2;++ch){
        std::vector<std::vector<uint32_t>> prods(N0);
        for (uint32_t k=0;k<N0;++k) prods[k]=pmul(CH[ch], Sn[ch][k*2+comp], DBn[ch][k*2+r]);
        (r==0?acc0:acc1)[comp*2+ch]=inv(CH[ch], padd(CH[ch], prods)); }

    // ---- (c) hypercube fold: ans = mux(C1, acc0, acc1) = extprod(C1, acc1-acc0) + acc0 ----
    std::vector<std::vector<uint32_t>> d(4);
    for (uint32_t i=0;i<4;++i){ uint32_t qc=QCH[i&1]; d[i].resize(N); for (uint32_t k=0;k<N;++k) d[i][k]=(acc1[i][k]+qc-acc0[i][k])%qc; }
    auto ep=extprod(d[0],d[1],d[2],d[3], C1n);
    std::vector<std::vector<uint32_t>> ans(4);
    for (uint32_t i=0;i<4;++i){ uint32_t qc=QCH[i&1]; ans[i].resize(N); for (uint32_t k=0;k<N;++k) ans[i][k]=(ep[i][k]+acc0[i][k])%qc; }

    // ---- (d) mod-switch each component: CRT-combine (device) then scale-round (host) -> mod q' ----
    auto crt_combine=[&](const std::vector<uint32_t>& cq1,const std::vector<uint32_t>& cq2){
        auto ba=mkbuf(1,I32),bb=mkbuf(1,I32),bdiff=mkbuf(1,I32),bt=mkbuf(1,I32),bX=mkbuf(2,I32);
        wrbuf(ba,til(cq1)); wrbuf(bb,til(cq2));
        { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
          auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_ab.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
          auto w=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
          auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/garner_diff.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
          SetRuntimeArgs(p,r,core,{(uint32_t)ba->address(),(uint32_t)bb->address(),1u,0u,0u}); SetRuntimeArgs(p,w,core,{(uint32_t)bdiff->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{q2,1u});
          distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); }
        { auto bimont=mkbuf(1,I32); wrbuf(bimont,til(std::vector<uint32_t>(N,inv_mont)));
          Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,2,I32);
          auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
          auto w=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_modmul.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
          auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/modmul_sfpi_1pass.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
          SetRuntimeArgs(p,r,core,{(uint32_t)bdiff->address(),(uint32_t)bimont->address(),1u,0u}); SetRuntimeArgs(p,w,core,{(uint32_t)bt->address(),1u,0u}); SetRuntimeArgs(p,cp,core,{1u,q2,qinv2,qlo2,qhi2});
          distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); }
        { Program p=CreateProgram(); mkcb(p,CBIndex::c_0,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_1,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_2,tt::DataFormat::UInt32,2,I32); mkcb(p,CBIndex::c_16,tt::DataFormat::UInt32,4,I32);
          auto r=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/reader_crt.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
          auto w=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "dataflow/writer_i32.cpp",core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
          auto cp=CreateKernel(p,OVERRIDE_KERNEL_PREFIX "compute/crt_mul_split.cpp",core,ComputeConfig{.fp32_dest_acc_en=true,.dst_full_sync_en=true,.math_approx_mode=false});
          SetRuntimeArgs(p,r,core,{(uint32_t)bt->address(),(uint32_t)ba->address(),(uint32_t)b_parQ->address(),1u}); SetRuntimeArgs(p,w,core,{(uint32_t)bX->address(),2u,0u}); SetRuntimeArgs(p,cp,core,{1u});
          distributed::MeshWorkload wl; wl.add_program(dr,std::move(p)); distributed::EnqueueMeshWorkload(cq,wl,false); }
        auto raw=rdbuf(bX); auto X0=unt(std::vector<uint32_t>(raw.begin(),raw.begin()+ELEMS)), X1=unt(std::vector<uint32_t>(raw.begin()+ELEMS,raw.end()));
        std::vector<uint64_t> C(N); for (uint32_t i=0;i<N;++i) C[i]=(uint64_t)X0[i]|((uint64_t)X1[i]<<28); return C; };

    Response resp; resp.c.assign(2, std::vector<uint32_t>(N));
    for (uint32_t comp=0;comp<2;++comp){ auto C=crt_combine(ans[comp*2+0], ans[comp*2+1]);
        for (uint32_t i=0;i<N;++i) resp.c[comp][i]=(uint32_t)(((u128)qp*C[i]+q/2)/q % qp); }
    mesh->close();
    return resp;
}

std::vector<uint32_t> client_decrypt(const Case& cs, const Response& r) {
    const uint32_t qp = cs.qp, t = cs.t;
    auto c1s = negmul(r.c[1], cs.skey, qp);
    std::vector<uint32_t> rec(N);
    for (uint32_t i = 0; i < N; ++i) {
        int64_t ph = ((int64_t)r.c[0][i] + c1s[i]) % qp;
        int64_t cen = (ph > (int64_t)qp / 2) ? ph - (int64_t)qp : ph;
        int64_t v = llround((double)cen * t / qp);
        rec[i] = (uint32_t)((v % (int64_t)t + t) % t);
    }
    return rec;
}

}  // namespace pir
