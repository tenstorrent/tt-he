# RNS-BFV Kernel Hierarchy

This note describes a practical kernel hierarchy for supporting BFV homomorphic
evaluation with RNS representation.

Assumption: encryption and decryption are out of scope. We only focus on
homomorphic evaluation operations, including addition, multiplication,
modulus switching, base conversion, and relinearization.

The lowest available hardware/software kernels are assumed to be:

```python
vec_modadd(a, b, q)
vec_modsub(a, b, q)
vec_modmul(a, b, q)
vec_mul_const(a, const, q)
ntt(a, q, ntt_params)
intt(a, q, ntt_params)
```

Everything above these kernels is composition and scheduling logic.

## 1. Overall Hierarchy

```text
L1: RNS polynomial kernels
    - RNS add/sub
    - RNS pointwise multiplication
    - RNS NTT / INTT
    - RNS polynomial multiplication

L2: BFV evaluator arithmetic
    - ciphertext add/sub
    - plaintext add/mul
    - raw ciphertext multiplication

L3: Fast RNS base conversion
    - Q -> P
    - P -> Q
    - group base extension

L4: Modulus management
    - mod-up Q -> QP
    - mod-down QP -> Q
    - modulus switching / drop last modulus

L5: RNS relinearization / key switching
    - RNS group decomposition
    - mod-up each group
    - multiply by evaluation key
    - accumulate in QP
    - mod-down to Q
```

Important point:

```text
Relinearization does not need to reconstruct big integers or decompose into
normal base-w digits. In efficient RNS-BFV, it can work directly over RNS
modulus groups.
```

## 2. Data Structures

### RNS Base

```python
class RNSBase:
    def __init__(self, moduli):
        # Example: [q0, q1, q2, q3]
        self.moduli = moduli

    @property
    def size(self):
        return len(self.moduli)
```

### RNS Polynomial

An `RNSPoly` stores one polynomial under several RNS moduli.

```python
class RNSPoly:
    def __init__(self, limbs, base: RNSBase, is_ntt=False):
        # limbs[i] is a coefficient vector modulo base.moduli[i].
        #
        # Example:
        #   limbs[0] = poly mod q0
        #   limbs[1] = poly mod q1
        #   limbs[2] = poly mod q2
        self.limbs = limbs
        self.base = base
        self.is_ntt = is_ntt

    @property
    def num_limbs(self):
        return len(self.limbs)
```

### Ciphertext

A BFV ciphertext is a list of RNS polynomials.

```python
class Ciphertext:
    def __init__(self, polys):
        # Normal ciphertext after encryption:
        #   polys = [c0, c1]
        #
        # Raw result after multiplication:
        #   polys = [c0, c1, c2]
        self.polys = polys

    @property
    def size(self):
        return len(self.polys)
```

## 3. L1: RNS Polynomial Kernels

L1 is the main bridge between low-level vector kernels and polynomial-level
BFV arithmetic.

### RNS Addition

```python
def rns_add(a: RNSPoly, b: RNSPoly) -> RNSPoly:
    assert a.base.moduli == b.base.moduli
    assert a.is_ntt == b.is_ntt

    out = []

    for ai, bi, qi in zip(a.limbs, b.limbs, a.base.moduli):
        out.append(vec_modadd(ai, bi, qi))

    return RNSPoly(out, a.base, is_ntt=a.is_ntt)
```

### RNS Subtraction

```python
def rns_sub(a: RNSPoly, b: RNSPoly) -> RNSPoly:
    assert a.base.moduli == b.base.moduli
    assert a.is_ntt == b.is_ntt

    out = []

    for ai, bi, qi in zip(a.limbs, b.limbs, a.base.moduli):
        out.append(vec_modsub(ai, bi, qi))

    return RNSPoly(out, a.base, is_ntt=a.is_ntt)
```

### RNS Pointwise Multiplication

This is just vector modular multiplication per RNS limb.

If inputs are in NTT form, this corresponds to polynomial multiplication in the
NTT domain.

```python
def rns_mul_pointwise(a: RNSPoly, b: RNSPoly) -> RNSPoly:
    assert a.base.moduli == b.base.moduli
    assert a.is_ntt == b.is_ntt

    out = []

    for ai, bi, qi in zip(a.limbs, b.limbs, a.base.moduli):
        out.append(vec_modmul(ai, bi, qi))

    return RNSPoly(out, a.base, is_ntt=a.is_ntt)
```

### RNS NTT

```python
def rns_ntt(a: RNSPoly, ntt_params) -> RNSPoly:
    if a.is_ntt:
        return a

    out = []

    for ai, qi, params_i in zip(a.limbs, a.base.moduli, ntt_params):
        out.append(ntt(ai, qi, params_i))

    return RNSPoly(out, a.base, is_ntt=True)
```

### RNS INTT

```python
def rns_intt(a: RNSPoly, ntt_params) -> RNSPoly:
    if not a.is_ntt:
        return a

    out = []

    for ai, qi, params_i in zip(a.limbs, a.base.moduli, ntt_params):
        out.append(intt(ai, qi, params_i))

    return RNSPoly(out, a.base, is_ntt=False)
```

### RNS Polynomial Multiplication

```python
def rns_poly_mul(a: RNSPoly, b: RNSPoly, ntt_params) -> RNSPoly:
    A = rns_ntt(a, ntt_params)
    B = rns_ntt(b, ntt_params)

    C = rns_mul_pointwise(A, B)

    return rns_intt(C, ntt_params)
```

### RNS Polynomial Multiplication in NTT Form

If inputs are already in NTT form, polynomial multiplication becomes only
pointwise vector multiplication.

```python
def rns_poly_mul_ntt(a_ntt: RNSPoly, b_ntt: RNSPoly) -> RNSPoly:
    assert a_ntt.is_ntt
    assert b_ntt.is_ntt

    return rns_mul_pointwise(a_ntt, b_ntt)
```

## 4. L2: BFV Evaluator Arithmetic

L2 builds ciphertext operations from L1 RNS polynomial kernels.

### Ciphertext Addition

```python
def ct_add(a: Ciphertext, b: Ciphertext) -> Ciphertext:
    max_size = max(a.size, b.size)
    out = []

    for i in range(max_size):
        if i < a.size and i < b.size:
            out.append(rns_add(a.polys[i], b.polys[i]))
        elif i < a.size:
            out.append(a.polys[i])
        else:
            out.append(b.polys[i])

    return Ciphertext(out)
```

### Ciphertext Subtraction

```python
def rns_zero_like(x: RNSPoly) -> RNSPoly:
    out = []

    for xi in x.limbs:
        out.append(zero_vec_like(xi))

    return RNSPoly(out, x.base, is_ntt=x.is_ntt)
```

```python
def ct_sub(a: Ciphertext, b: Ciphertext) -> Ciphertext:
    max_size = max(a.size, b.size)
    out = []

    for i in range(max_size):
        if i < a.size and i < b.size:
            out.append(rns_sub(a.polys[i], b.polys[i]))
        elif i < a.size:
            out.append(a.polys[i])
        else:
            zero = rns_zero_like(b.polys[i])
            out.append(rns_sub(zero, b.polys[i]))

    return Ciphertext(out)
```

### Ciphertext-Plaintext Addition

For BFV, plaintext addition normally modifies only `c0`.

```python
def ct_plain_add(ct: Ciphertext, pt: RNSPoly) -> Ciphertext:
    out = list(ct.polys)
    out[0] = rns_add(out[0], pt)
    return Ciphertext(out)
```

The plaintext should already be encoded into the same RNS base as the
ciphertext. In BFV, this usually involves scaling by something like `Q / t`
depending on the exact variant and encoding convention.

### Ciphertext-Plaintext Multiplication

```python
def ct_plain_mul(ct: Ciphertext, pt: RNSPoly, ntt_params) -> Ciphertext:
    out = []

    for ci in ct.polys:
        out.append(rns_poly_mul(ci, pt, ntt_params))

    return Ciphertext(out)
```

Fast path if plaintext is already in NTT form:

```python
def ct_plain_mul_ntt(ct_ntt: Ciphertext, pt_ntt: RNSPoly) -> Ciphertext:
    out = []

    for ci_ntt in ct_ntt.polys:
        out.append(rns_poly_mul_ntt(ci_ntt, pt_ntt))

    return Ciphertext(out)
```

### Raw Ciphertext Multiplication

For two size-2 ciphertexts:

```text
ct_a = (c0, c1)
ct_b = (d0, d1)
```

raw multiplication gives:

```text
e0 = c0 * d0
e1 = c0 * d1 + c1 * d0
e2 = c1 * d1
```

```python
def ct_mul_raw(a: Ciphertext, b: Ciphertext, ntt_params) -> Ciphertext:
    assert a.size == 2
    assert b.size == 2

    c0, c1 = a.polys
    d0, d1 = b.polys

    e0 = rns_poly_mul(c0, d0, ntt_params)

    t0 = rns_poly_mul(c0, d1, ntt_params)
    t1 = rns_poly_mul(c1, d0, ntt_params)
    e1 = rns_add(t0, t1)

    e2 = rns_poly_mul(c1, d1, ntt_params)

    return Ciphertext([e0, e1, e2])
```

Fast path if all ciphertext polynomials are already in NTT form:

```python
def ct_mul_raw_ntt(a: Ciphertext, b: Ciphertext) -> Ciphertext:
    assert a.size == 2
    assert b.size == 2

    c0, c1 = a.polys
    d0, d1 = b.polys

    e0 = rns_poly_mul_ntt(c0, d0)

    t0 = rns_poly_mul_ntt(c0, d1)
    t1 = rns_poly_mul_ntt(c1, d0)
    e1 = rns_add(t0, t1)

    e2 = rns_poly_mul_ntt(c1, d1)

    return Ciphertext([e0, e1, e2])
```

## 5. L3: Fast RNS Base Conversion

Efficient RNS-BFV needs base conversion. The goal is to convert residues from
one RNS base to another without reconstructing large integers.

Example:

```text
Q = q0 * q1 * q2 * ...
P = p0 * p1 * ...

x mod Q  ->  x mod P
```

### Precomputed Parameters

```python
class FastBConvParams:
    def __init__(
        self,
        src_base: RNSBase,
        dst_base: RNSBase,
        src_hat_inv_mod_src,
        src_hat_mod_dst,
    ):
        self.src_base = src_base
        self.dst_base = dst_base

        # src_hat_inv_mod_src[i] =
        #     (SRC / src_i)^(-1) mod src_i
        self.src_hat_inv_mod_src = src_hat_inv_mod_src

        # src_hat_mod_dst[i][j] =
        #     (SRC / src_i) mod dst_j
        self.src_hat_mod_dst = src_hat_mod_dst
```

### Fast Base Conversion

This is a CRT-style linear combination:

```python
def fast_base_convert(x: RNSPoly, params: FastBConvParams) -> RNSPoly:
    assert x.base.moduli == params.src_base.moduli
    assert not x.is_ntt

    dst_limbs = []

    for j, pj in enumerate(params.dst_base.moduli):
        acc = zero_vec_like_mod(x.limbs[0], pj)

        for i, qi in enumerate(params.src_base.moduli):
            # alpha_i = x_i * (SRC / qi)^(-1) mod qi
            alpha_i = vec_mul_const(
                x.limbs[i],
                params.src_hat_inv_mod_src[i],
                qi,
            )

            # contribution = alpha_i * ((SRC / qi) mod pj) mod pj
            contrib = vec_mul_const(
                alpha_i,
                params.src_hat_mod_dst[i][j],
                pj,
            )

            acc = vec_modadd(acc, contrib, pj)

        dst_limbs.append(acc)

    return RNSPoly(dst_limbs, params.dst_base, is_ntt=False)
```

Note:

```text
This is still an approximate/fast base conversion in practical RNS-BFV.
Real implementations often add correction terms, Shenoy-Kumaresan correction,
or small Montgomery reduction depending on the BFV variant.
```

## 6. L4: Mod-Up, Mod-Down, and Modulus Switching

### Mod-Up: Q -> QP

For key switching, we often extend a polynomial from base `Q` into a larger
base:

```text
QP = Q union P
```

```python
def mod_up_Q_to_QP(x_q: RNSPoly, q_to_p_params, base_qp: RNSBase) -> RNSPoly:
    assert not x_q.is_ntt

    x_p = fast_base_convert(x_q, q_to_p_params)

    return RNSPoly(
        limbs=x_q.limbs + x_p.limbs,
        base=base_qp,
        is_ntt=False,
    )
```

### Mod-Down: QP -> Q

After key-switch accumulation in `QP`, convert back to `Q`.

Common form:

```text
y_i = (x_i - [x_P]_i) * P^{-1} mod q_i
```

```python
class ModDownParams:
    def __init__(
        self,
        base_q: RNSBase,
        base_p: RNSBase,
        base_qp: RNSBase,
        p_to_q_params: FastBConvParams,
        P_inv_mod_q,
    ):
        self.base_q = base_q
        self.base_p = base_p
        self.base_qp = base_qp
        self.p_to_q_params = p_to_q_params

        # P_inv_mod_q[i] = P^(-1) mod q_i
        self.P_inv_mod_q = P_inv_mod_q
```

```python
def mod_down_QP_to_Q(x_qp: RNSPoly, params: ModDownParams) -> RNSPoly:
    assert x_qp.base.moduli == params.base_qp.moduli
    assert not x_qp.is_ntt

    Lq = params.base_q.size

    x_q = RNSPoly(
        x_qp.limbs[:Lq],
        params.base_q,
        is_ntt=False,
    )

    x_p = RNSPoly(
        x_qp.limbs[Lq:],
        params.base_p,
        is_ntt=False,
    )

    x_p_to_q = fast_base_convert(x_p, params.p_to_q_params)

    out = []

    for xi_q, xi_p_to_q, qi, pinv_qi in zip(
        x_q.limbs,
        x_p_to_q.limbs,
        params.base_q.moduli,
        params.P_inv_mod_q,
    ):
        tmp = vec_modsub(xi_q, xi_p_to_q, qi)
        tmp = vec_mul_const(tmp, pinv_qi, qi)
        out.append(tmp)

    return RNSPoly(out, params.base_q, is_ntt=False)
```

### Modulus Switching: Drop Last Modulus

For level management, we may drop the last modulus:

```text
Q  = q0 * q1 * ... * qL
Q' = q0 * q1 * ... * q_{L-1}
```

Common RNS formula:

```text
y_i = (x_i - x_L) * q_L^{-1} mod q_i
```

```python
class DropLastModulusParams:
    def __init__(self, base_q: RNSBase, base_q_without_last: RNSBase, q_last_inv_mod_qi):
        self.base_q = base_q
        self.base_q_without_last = base_q_without_last

        # q_last_inv_mod_qi[i] = q_last^(-1) mod q_i
        self.q_last_inv_mod_qi = q_last_inv_mod_qi
```

```python
def mod_switch_drop_last(x_q: RNSPoly, params: DropLastModulusParams) -> RNSPoly:
    assert x_q.base.moduli == params.base_q.moduli
    assert not x_q.is_ntt

    x_last = x_q.limbs[-1]
    out = []

    for xi, qi, qlast_inv_qi in zip(
        x_q.limbs[:-1],
        x_q.base.moduli[:-1],
        params.q_last_inv_mod_qi,
    ):
        correction = reduce_vec_to_mod(x_last, qi)
        tmp = vec_modsub(xi, correction, qi)
        tmp = vec_mul_const(tmp, qlast_inv_qi, qi)
        out.append(tmp)

    return RNSPoly(out, params.base_q_without_last, is_ntt=False)
```

Apply to a ciphertext:

```python
def ct_mod_switch_drop_last(ct: Ciphertext, params: DropLastModulusParams) -> Ciphertext:
    out = []

    for ci in ct.polys:
        out.append(mod_switch_drop_last(ci, params))

    return Ciphertext(out)
```

## 7. L5: RNS Relinearization / Key Switching

After raw multiplication:

```text
ct_raw = (c0, c1, c2)
```

The `c2` term corresponds to an `s^2` term. Relinearization converts it back
into two ciphertext components under `s`:

```text
(c0, c1, c2) -> (c0', c1')
```

Efficient RNS-BFV does this directly in RNS. It does not need to reconstruct
large integers.

### RNS Decomposition Groups

Example:

```text
Q = [q0, q1, q2, q3]

groups = [
    [q0, q1],
    [q2, q3],
]
```

```python
class RNSDecompGroup:
    def __init__(self, indices):
        # Example: [0, 1] or [2, 3]
        self.indices = indices
```

```python
def extract_rns_group(x_q: RNSPoly, group: RNSDecompGroup) -> RNSPoly:
    limbs = [x_q.limbs[i] for i in group.indices]
    moduli = [x_q.base.moduli[i] for i in group.indices]

    return RNSPoly(
        limbs,
        RNSBase(moduli),
        is_ntt=x_q.is_ntt,
    )
```

### Relinearization Key

Each group has one pair of evaluation-key polynomials.

```python
class RelinKey:
    def __init__(self, key_pairs):
        # key_pairs[g] = (evk0_g, evk1_g)
        #
        # evk0_g and evk1_g are RNSPoly objects in QP base.
        # In an efficient implementation, they are often stored in NTT form.
        self.key_pairs = key_pairs
```

### Group Mod-Up

For each RNS group, we extend it to the full `QP` base.

```python
def mod_up_group_to_QP(c2_g: RNSPoly, group: RNSDecompGroup, params) -> RNSPoly:
    """
    Convert one RNS decomposition group into full QP base.

    This is a group-specific base extension:
        group_base -> QP

    In real implementations, each group has its own precomputed conversion
    parameters.
    """

    group_to_qp_params = params.group_to_qp_params[group]
    return fast_base_convert(c2_g, group_to_qp_params)
```

### Key-Switch Accumulation

```python
class KeySwitchParams:
    def __init__(
        self,
        base_q: RNSBase,
        base_p: RNSBase,
        base_qp: RNSBase,
        decomp_groups,
        group_to_qp_params,
        mod_down_params,
        ntt_params_qp,
    ):
        self.base_q = base_q
        self.base_p = base_p
        self.base_qp = base_qp
        self.decomp_groups = decomp_groups
        self.group_to_qp_params = group_to_qp_params
        self.mod_down_params = mod_down_params
        self.ntt_params_qp = ntt_params_qp
```

```python
def rns_zero(base: RNSBase, degree, is_ntt=False) -> RNSPoly:
    limbs = []

    for qi in base.moduli:
        limbs.append(zero_vec(degree, modulus=qi))

    return RNSPoly(limbs, base, is_ntt=is_ntt)
```

```python
def key_switch_rns(c2: RNSPoly, relin_key: RelinKey, params: KeySwitchParams):
    """
    Switch c2 * s^2 into two components under s.

    Returns:
        k0, k1 in base Q
    """

    assert c2.base.moduli == params.base_q.moduli
    assert not c2.is_ntt

    degree = len(c2.limbs[0])

    acc0_qp = rns_zero(params.base_qp, degree, is_ntt=True)
    acc1_qp = rns_zero(params.base_qp, degree, is_ntt=True)

    for group, (evk0_g, evk1_g) in zip(params.decomp_groups, relin_key.key_pairs):
        # 1. Extract one RNS group from c2.
        c2_g = extract_rns_group(c2, group)

        # 2. Mod-up this group to QP.
        c2_g_qp = mod_up_group_to_QP(c2_g, group, params)

        # 3. Convert to NTT form for pointwise multiplication.
        c2_g_qp_ntt = rns_ntt(c2_g_qp, params.ntt_params_qp)

        # 4. Multiply by evaluation-key pair and accumulate.
        #
        # evk0_g and evk1_g are assumed to already be in NTT form.
        prod0 = rns_mul_pointwise(c2_g_qp_ntt, evk0_g)
        prod1 = rns_mul_pointwise(c2_g_qp_ntt, evk1_g)

        acc0_qp = rns_add(acc0_qp, prod0)
        acc1_qp = rns_add(acc1_qp, prod1)

    # 5. Convert accumulated results back to coefficient domain.
    acc0_qp = rns_intt(acc0_qp, params.ntt_params_qp)
    acc1_qp = rns_intt(acc1_qp, params.ntt_params_qp)

    # 6. Mod-down from QP to Q.
    k0_q = mod_down_QP_to_Q(acc0_qp, params.mod_down_params)
    k1_q = mod_down_QP_to_Q(acc1_qp, params.mod_down_params)

    return k0_q, k1_q
```

### Relinearization

```python
def relinearize_rns(ct3: Ciphertext, relin_key: RelinKey, params: KeySwitchParams) -> Ciphertext:
    assert ct3.size == 3

    c0, c1, c2 = ct3.polys

    k0, k1 = key_switch_rns(c2, relin_key, params)

    out0 = rns_add(c0, k0)
    out1 = rns_add(c1, k1)

    return Ciphertext([out0, out1])
```

### Full BFV Multiplication

```python
def ct_mul(a: Ciphertext, b: Ciphertext, relin_key: RelinKey, params) -> Ciphertext:
    raw = ct_mul_raw(a, b, params.ntt_params_q)
    return relinearize_rns(raw, relin_key, params.key_switch_params)
```

Fast path if inputs are already in NTT form:

```python
def ct_mul_ntt(a_ntt: Ciphertext, b_ntt: Ciphertext, relin_key: RelinKey, params) -> Ciphertext:
    raw_ntt = ct_mul_raw_ntt(a_ntt, b_ntt)

    # Relinearization usually expects c2 in coefficient form for base conversion.
    raw = Ciphertext([
        rns_intt(poly, params.ntt_params_q)
        for poly in raw_ntt.polys
    ])

    return relinearize_rns(raw, relin_key, params.key_switch_params)
```

## 8. Kernel Mapping Summary

The real primitive kernels are still only:

| Primitive kernel | Used by |
|---|---|
| `vec_modadd` | RNS add, ciphertext add, accumulation, base conversion |
| `vec_modsub` | RNS sub, mod-down, modulus switching |
| `vec_modmul` | NTT-domain polynomial multiplication |
| `vec_mul_const` | base conversion, mod-down, modulus switching |
| `ntt` | polynomial multiplication, key-switch multiplication |
| `intt` | polynomial multiplication, key-switch output |

Higher-level operations are schedules built from these kernels:

| BFV operation | Main schedule |
|---|---|
| `ct_add` | RNS vector add per ciphertext component |
| `ct_plain_add` | RNS vector add into `c0` |
| `ct_plain_mul` | RNS NTT, pointwise multiply, INTT |
| `ct_mul_raw` | 3 polynomial products plus one add |
| `base_convert` | CRT-style vector multiply-constant and add |
| `mod_switch_drop_last` | subtract dropped limb, multiply by inverse |
| `relinearize_rns` | RNS group extraction, mod-up, NTT multiply with eval key, accumulate, mod-down |

## 9. Suggested Benchmark Units

For simulator or accelerator evaluation, use three granularities.

### Kernel Level

```text
vec_modadd
vec_modsub
vec_modmul
vec_mul_const
ntt
intt
```

### RNS Polynomial Level

```text
rns_add
rns_sub
rns_poly_mul
fast_base_convert
mod_down_QP_to_Q
mod_switch_drop_last
```

### BFV Operation Level

```text
ct_add
ct_plain_mul
ct_mul_raw
relinearize_rns
ct_mul = ct_mul_raw + relinearize_rns
ct_mod_switch_drop_last
```

## 10. Notes on Efficient Implementations

Efficient libraries usually optimize the following:

```text
1. Store frequently used polynomials in NTT form.
2. Store evaluation keys in NTT form.
3. Fuse multiply-add accumulation in the NTT domain.
4. Use lazy modular reduction when safe.
5. Precompute all CRT constants for base conversion.
6. Use RNS group decomposition instead of integer digit decomposition.
7. Use auxiliary base P for key switching and mod-down.
```

Implementation references worth checking:

- [CROSS](https://github.com/EfficientPPML/CROSS)
- [Lattigo](https://github.com/tuneinsight/lattigo)
- Microsoft SEAL-style RNS-BFV implementations
- HPS / BEHZ RNS-BFV papers and implementation notes

## 11. Mental Model

The clean mental model is:

```text
Vector kernels
    -> RNS polynomial kernels
        -> BFV arithmetic
            -> base conversion / modulus management
                -> relinearization / key switching
```

But computationally, everything reduces back to:

```text
vector add/sub
vector multiply
multiply by constants
NTT / INTT
```

So for hardware or simulator design, BFV support can be modeled as a hierarchy
of schedules over a small set of reusable kernels.

