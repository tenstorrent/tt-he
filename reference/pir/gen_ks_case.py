#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""BV-style key-switch (#10) test case for the device.

BV key-switch: (c0,c1) under key s' -> (d0,d1) under key s, same message.
    (d0, d1) = (c0, 0) + Σ_k G^{-1}(c1)_k ⊛ KSK[k]        KSK[k] = (·, ·)  (ℓ rows × 2 cols)
It is the external-product machinery restricted to ONE decomposed component (c1) plus adding c0.
Validated bit-exact vs onionpir_ref.keyswitch (used by the end-to-end-validated Subs/expansion).

K-parameterized (K = RNS limb count): K=2 (~56-bit q), K=3 (~84-bit q).

Binary (VERSION 2, little-endian):
    u32 MAGIC=0x4b535754('KSWT'), u32 VERSION=2, u32 n, u32 ell, u32 B, u32 K
    K * u32   primes
    c0res : K*n u32 ;  c1res : K*n u32
    KSK   : per ch j: per k(ℓ): per col(2): n u32   (KSK[k][col] mod q_j)
    out   : d0res(K*n) , d1res(K*n)
    digits: ell i32   (G^{-1}(c1), host sanity check)

    python3 gen_ks_case.py --n 2048 --ell 6 --B 1024                 # K=2 default
    python3 gen_ks_case.py --K 3 --n 2048 --ell 8 --B 2048 --out ks_k3.bin
"""
import argparse, random, struct
import onionpir_ref as ref

MAGIC = 0x4B535754
VERSION = 2

def gen(n, ell, B, K, seed):
    rng = random.Random(seed)
    pp = ref.Params(n=n, B=B, ell=ell, K=K)
    q, primes = pp.q, pp.primes
    c0 = [rng.randrange(q) for _ in range(n)]
    c1 = [rng.randrange(q) for _ in range(n)]
    ksk = [[[rng.randrange(q) for _ in range(n)] for _ in range(2)] for _ in range(ell)]  # [k][col]
    d0, d1 = ref.keyswitch(pp, c0, c1, ksk)
    digits = ref.gadget_decompose_signed(c1, B, ell, q)
    return {"n": n, "ell": ell, "B": B, "K": K, "primes": list(primes),
            "c0": c0, "c1": c1, "ksk": ksk, "d0": d0, "d1": d1, "digits": digits}

def write(path, c):
    n, ell, K, primes = c["n"], c["ell"], c["K"], c["primes"]
    with open(path, "wb") as f:
        f.write(struct.pack("<6I", MAGIC, VERSION, n, ell, c["B"], K))
        f.write(struct.pack(f"<{K}I", *primes))
        for d in (c["c0"], c["c1"]):
            for qi in primes:
                f.write(struct.pack(f"<{n}I", *[x % qi for x in d]))
        for qi in primes:
            for k in range(ell):
                for col in range(2):
                    f.write(struct.pack(f"<{n}I", *[x % qi for x in c["ksk"][k][col]]))
        for d in (c["d0"], c["d1"]):
            for qi in primes:
                f.write(struct.pack(f"<{n}I", *[x % qi for x in d]))
        for k in range(ell):
            f.write(struct.pack(f"<{n}i", *c["digits"][k]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--ell", type=int, default=6)
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ks_case.bin")
    a = ap.parse_args()
    c = gen(a.n, a.ell, a.B, a.K, a.seed)
    write(a.out, c)
    print(f"wrote {a.out}: n={c['n']} ell={c['ell']} B={c['B']} K={c['K']} primes={c['primes']}")

if __name__ == "__main__":
    main()
