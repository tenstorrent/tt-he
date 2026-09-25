// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Buffer-resident NTT round-trip at the PIR ring (n=2048), via pir2048::Engine.
// The PIR equivalent of he_mul_cross.cpp's "Brick 3.1": prove that
//     fwd2048_buf -> transpose_dev -> inv2048_buf
// is bit-exact with NO host round-trip and ONE Finish, so the answer pipeline's per-op blocking
// readbacks (~43% of replay time, and the host boundaries that would shatter a Metal Trace into
// ~1668 segments) can be removed.
//
// Three checks, in increasing strength:
//   1. resident fwd output == host fwd2048 output          (fwd_buf computes the same values)
//   2. resident fwd -> transpose -> inv == original x      (the transpose IS the host reindex)
//   3. per-op readback count: host path vs resident path   (what the residency actually buys)
//
//   ./build/test_ntt2048_res [q] [seed]

#include <cstdint>
#include <random>
#include <vector>
#include "pir_engine2048.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace pir2048;

int main(int argc, char** argv) {
    const uint32_t q = (argc > 1) ? std::stoul(argv[1]) : 268369921u;   // 28-bit NTT-friendly
    const uint32_t seed = (argc > 2) ? std::stoul(argv[2]) : 7u;
    bool pass = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        // ell/B are irrelevant here (no gadget); K=1 channel is enough for the NTT layer.
        Engine eng(mesh.get(), &cq, {q}, 3, 1024);
        auto& c = eng.CH[0];

        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> di(0, q - 1);
        std::vector<uint32_t> x(N); for (auto& v : x) v = di(rng);

        // ---- reference: the validated host path (readback + host reindex per transform) ----
        prof::reset();
        auto Xhost = eng.fwd2048(c, x);
        auto xhost = eng.inv2048(c, Xhost);
        const long rd_host = prof::n_rd;
        uint32_t mm_rt_host = 0; for (uint32_t i = 0; i < N; ++i) if (xhost[i] != x[i]) ++mm_rt_host;

        // ---- resident: DRAM in -> fwd -> on-device transpose -> inv -> DRAM out, ONE readback ----
        std::vector<MeshBuf> inb(2), fo(2), tin(2), io(2);
        for (uint32_t i = 0; i < 2; ++i) { inb[i]=eng.mkbuf(1,I32); fo[i]=eng.mkbuf(1,I32);
                                          tin[i]=eng.mkbuf(1,I32); io[i]=eng.mkbuf(1,I32); }
        prof::reset();
        eng.write_fwd_in(inb, x);
        eng.fwd2048_buf(c, inb, fo);
        for (uint32_t bk2 = 0; bk2 < 2; ++bk2) eng.transpose_dev(fo[bk2], tin[bk2]);   // on-device reindex
        eng.inv2048_buf(c, tin, io);
        auto xres = eng.read_inv_out(io);                                              // the ONLY readback
        const long rd_res = prof::n_rd;
        uint32_t mm_rt_res = 0; for (uint32_t i = 0; i < N; ++i) if (xres[i] != x[i]) ++mm_rt_res;

        // check 1: resident fwd values identical to host fwd (read the fwd tiles separately)
        eng.write_fwd_in(inb, x);
        eng.fwd2048_buf(c, inb, fo);
        auto Xres = eng.read_fwd_out(fo);
        uint32_t mm_fwd = 0; for (uint32_t i = 0; i < N; ++i) if (Xres[i] != Xhost[i]) ++mm_fwd;

        // ---- PACKED path: one 2-tile buffer per poly (what mac_ntt/mmul consume) ----
        // Same chain, but every stage reads its block out of a packed buffer via reader_fused_off, and
        // the transpose moves tile bk2 in place. If this matches, the resident extproduct can keep a poly
        // as ONE buffer end to end and the MAC needs no re-upload.
        auto p_in = eng.mkbuf(2, I32), p_fwd = eng.mkbuf(2, I32), p_tr = eng.mkbuf(2, I32), p_out = eng.mkbuf(2, I32);
        {
            std::vector<MeshBuf> tmp{p_in, p_in};        // write_fwd_in wants 2 slots; both blocks -> p_in
            std::vector<uint32_t> blk0(ELEMS), blk1(ELEMS);
            for (uint32_t b=0;b<T;++b) for (uint32_t n1=0;n1<N1;++n1) { blk0[b*T+n1]=x[n1+32*(0*T+b)];
                                                                        blk1[b*T+n1]=x[n1+32*(1*T+b)]; }
            std::vector<uint32_t> packed(2*ELEMS);
            auto t0=tilize32(blk0), t1=tilize32(blk1);
            std::copy(t0.begin(),t0.end(),packed.begin()); std::copy(t1.begin(),t1.end(),packed.begin()+ELEMS);
            eng.wrb(p_in, packed);
        }
        eng.fwd2048_buf2(c, p_in, p_fwd);
        for (uint32_t bk2 = 0; bk2 < 2; ++bk2) eng.transpose_dev(p_fwd, p_tr, bk2, bk2);
        eng.inv2048_buf2(c, p_tr, p_out);
        auto praw = eng.rdb(p_out);
        std::vector<uint32_t> xpk(N);
        for (uint32_t bn2=0; bn2<2; ++bn2) { auto blk=untilize32(half(praw,bn2));
            for (uint32_t n1=0;n1<N1;++n1) for (uint32_t n2loc=0;n2loc<T;++n2loc) xpk[n1+32*(bn2*T+n2loc)]=blk[n1*T+n2loc]; }
        uint32_t mm_pk = 0; for (uint32_t i = 0; i < N; ++i) if (xpk[i] != x[i]) ++mm_pk;

        // ---- Montgomery/transform commutation ----
        // The pipeline does to_mont(fwd2048(x)): transform on device, Montgomery-convert ON HOST. A fully
        // resident transform has nowhere to put that host step. to_mont is multiplication by R mod q and
        // the NTT is linear mod q, so fwd(to_mont(x)) SHOULD equal to_mont(fwd(x)) -- letting us
        // Montgomerize the coefficient-domain input instead. Verify rather than assume: an off-by-a-factor
        // here would corrupt every RGSW row in the MAC while still "looking like" valid ciphertext.
        auto mont_after  = eng.to_mont(0, Xhost);                    // current pipeline order
        auto mont_before = eng.fwd2048(c, eng.to_mont(0, x));        // resident-friendly order
        uint32_t mm_mont = 0; for (uint32_t i = 0; i < N; ++i) if (mont_after[i] != mont_before[i]) ++mm_mont;

        fmt::print("\nbuffer-resident NTT-2048  q={} seed={}\n", q, seed);
        fmt::print("  [5] to_mont(fwd(x)) == fwd(to_mont(x))   : mismatches={}/{}\n", mm_mont, N);
        fmt::print("  [4] PACKED fwd2->transpose->inv2  vs x  : mismatches={}/{}\n", mm_pk, N);
        fmt::print("  [1] resident fwd  vs host fwd2048        : mismatches={}/{}\n", mm_fwd, N);
        fmt::print("  [2] resident fwd->transpose->inv  vs x   : mismatches={}/{}\n", mm_rt_res, N);
        fmt::print("      (host fwd2048->inv2048 vs x          : mismatches={}/{})\n", mm_rt_host, N);
        fmt::print("  [3] blocking readbacks per round-trip    : host={}  resident={}\n", rd_host, rd_res);
        pass = (mm_fwd == 0 && mm_rt_res == 0 && mm_rt_host == 0 && mm_pk == 0);
        pass &= mesh->close();
    } catch (const std::exception& e) { fmt::print(stderr, "Exception:\n{}\n", e.what()); return 2; }
    if (pass) fmt::print("\nTest Passed\n"); else fmt::print("\nTest FAILED\n");
    return pass ? 0 : 1;
}
