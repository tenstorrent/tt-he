#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""1-of-2 multiplexer (the hypercube-fold atom, #19) test case, n=1024.
    mux(C_bit, x, y) = ExternalProduct(C_bit, y - x) + x   ->  y if bit=1 else x.
Wires the validated external product with modsub/modadd. Client crypto (encrypt/rgsw/decrypt) is the
CPU reference; the server op (extprod on y-x, +x) runs on device. Validated vs onionpir_ref.mux_1of2,
and the golden also decrypts to the selected message.

Binary (LE): u32 MAGIC=0x4d555830('MUX0'), n, ell, B, q1, q2, bit
    x  : x0q1,x0q2,x1q1,x1q2   (n u32 each)
    y  : y0q1,y0q2,y1q1,y1q2
    C  : per ch(q1,q2): per r(2ℓ): per col(2): n u32   (RGSW(bit) rows mod q_ch)
    out: g0q1,g0q2,g1q1,g1q2   (golden mux, per channel)

    python3 gen_mux_case.py [--bit 0|1]
"""
import argparse, random, struct
import onionpir_ref as ref
MAGIC = 0x4D555830

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024); ap.add_argument("--ell", type=int, default=3)
    ap.add_argument("--B", type=int, default=1 << 20); ap.add_argument("--bit", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="mux_case.bin")
    a = ap.parse_args()
    rng = random.Random(a.seed); pp = ref.Params(n=a.n, B=a.B, ell=a.ell); q, q1, q2 = pp.q, pp.q1, pp.q2
    s = ref.keygen(pp, rng)
    mx = [rng.randrange(pp.t) for _ in range(a.n)]; my = [rng.randrange(pp.t) for _ in range(a.n)]
    x = ref.encrypt(pp, s, mx, rng); y = ref.encrypt(pp, s, my, rng)
    C = ref.rgsw_encrypt(pp, s, [a.bit] + [0] * (a.n - 1), rng)   # 2ℓ rows [col0,col1]
    g = ref.mux_1of2(pp, C, x, y)                                 # golden mux
    sel = ref.decrypt(pp, s, g); want = (my if a.bit else mx)
    assert sel == want, "golden mux did not select the right message"
    twoL = 2 * a.ell
    with open(a.out, "wb") as f:
        f.write(struct.pack("<7I", MAGIC, a.n, a.ell, a.B, q1, q2, a.bit))
        for ct in (x, y):
            for comp in (0, 1):
                for qi in (q1, q2): f.write(struct.pack(f"<{a.n}I", *[v % qi for v in ct[comp]]))
        for qi in (q1, q2):
            for r in range(twoL):
                for col in range(2): f.write(struct.pack(f"<{a.n}I", *[v % qi for v in C[r][col]]))
        for comp in (0, 1):
            for qi in (q1, q2): f.write(struct.pack(f"<{a.n}I", *[v % qi for v in g[comp]]))
    print(f"wrote {a.out}: n={a.n} ell={a.ell} bit={a.bit}  (golden mux selects {'y' if a.bit else 'x'}, decrypt ok)")

if __name__ == "__main__":
    main()
