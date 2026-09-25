# The PIR scan — algorithm, then hardware mapping (kept separate)

Two abstraction levels, deliberately not mixed:
- **Part 1 — the algorithm** (hardware-agnostic): what the scan computes, as vector/tensor ops with
  explicit shapes. This is fixed; there is one good scan and it's the NTT one.
- **Part 2 — the hardware mapping** (implementation): which existing kernels / engine units run each
  step. This is a choice made *under* the algorithm and never alters it.

---

# Part 1 — The algorithm (vector level, shapes explicit)

## Objects and shapes

A polynomial = a length-`N` vector of integers mod `q` (one coefficient = one element).

```
N     = ring dimension           (2048)     → a poly is  int[N]
q     = modulus                              → every element ∈ Z_q
N0    = first-dimension size      (512)
rest  = product of other dims     (128)      → Ncells = N0·rest
K     = RNS channels              (3)        → carried as an outer batch; per channel it's the same
```

```
DB : int[N0, rest, N]     # database: N0·rest plaintext polys
Q  : int[N0, 2, N]        # query: N0 one-hot selector ciphertexts; a ct = 2 polys → the "2"
R  : int[rest, 2, N]      # output: `rest` ciphertexts
```

## Primitive: NTT

```
NTT  : int[N] → int[N]        iNTT(NTT(x)) = x
NTT(a ⊛ b) = NTT(a) ⊙ NTT(b)      # ⊛ = poly-mul,  ⊙ = elementwise product mod q
```
The one property that matters: NTT turns polynomial multiply (`⊛`) into elementwise multiply (`⊙`).

## The four steps

```
# Step 0  OFFLINE (once; DB is query-independent)
DB_ntt : int[N0, rest, N]
DB_ntt[k,r,:] = NTT( DB[k,r,:] )

# Step 1  transform the query (small: N0 polys)
Q_ntt : int[N0, 2, N]
Q_ntt[k,t,:] = NTT( Q[k,t,:] )

# Step 2  THE SCAN — contract k, elementwise over j
A_ntt : int[rest, 2, N]
A_ntt[r,t,j] = ( Σ_{k=0}^{N0-1} DB_ntt[k,r,j] · Q_ntt[k,t,j] ) mod q

# Step 3  inverse-transform the outputs (small: rest·2 polys)
R : int[rest, 2, N]
R[r,t,:] = iNTT( A_ntt[r,t,:] )
```

## Step 2, precisely: it is a BATCHED MATMUL

Einsum (`c` = ciphertext component, renamed from `t` to avoid clashing with the plaintext modulus):
`A[r,c,j] = Σ_k DB[k,r,j] · Q[k,c,j]`  =  `"krj, kcj -> rcj"`  (mod q).

| axis | meaning | role | real value (OnionPIR `CONFIG_N2048`) |
|---|---|---|---|
| `k` | first-dim size `N0` | **contracted** (the sum) | **512** |
| `r` | remaining dims `rest`=2⁷ | free (from DB) | **128** |
| `c` | ciphertext components (c0,c1) | free (from Q) | **2** |
| `j` | NTT slots = ring dim `N` | **batch** (elementwise) | **2048** |
| `K` | RNS channels | outer batch | **3** |

Tensors, per channel:  `DB:[512,128,2048]  ·  Q:[512,2,2048]  →  A:[128,2,2048]`.

- `k` — **contracted** (the first-dim selection)
- `r`, `c` — free output axes
- `j` (and `K`) — **batch** axes (shared on both inputs and output → not summed)

Fix a slot `j`. Let `D = DB_ntt[:,:,j] : [N0,rest]` and `W = Q_ntt[:,:,j] : [N0,2]`. Then

```
A_ntt[:,:,j]  =  Dᵀ · W        # [rest, N0] · [N0, 2] → [rest, 2]   (contract k)
```

a genuine matmul. **Step 2 is `N` such matmuls, batched over `j`** (and over the `K` channels):

```
bmm :  DB_ntt : [N,  N0, rest]        # per-slot LHS (transposed), batch = N
       Q_ntt  : [N,  N0, 2   ]        # per-slot RHS,              batch = N
       A_ntt  : [N,  rest, 2 ]        # A_ntt[j] = DB_ntt[j]ᵀ · Q_ntt[j]
```

## Work / optimality

```
work(Step 2) = N0 · rest · 2 · N · K  multiply-adds  =  2·K·N · Ncells  ≈  O(database size)
```
One multiply-add per database element. The expensive `⊛→⊙` transform is paid **once, offline** on the
DB (Step 0); Steps 1 and 3 transform only small things (query = `N0` polys; outputs = `rest·2` polys),
independent of database size. You cannot beat "one op per element," so this is the optimal scan, and it
is what OnionPIR (and every lattice PIR) uses. The only algorithm-level knob is the hypercube shape
(`N0` vs `rest`), which trades first-dim work against fold count/noise — not a different algorithm.

---

# Part 2 — Hardware mapping (implementation, separate level)

The algorithm above does not change. We only pick which kernel/engine runs each step.

## Steps 0, 1, 3 — the NTTs → reuse the NTT kernels, batched across cores

These are independent length-`N` NTTs (Step 0: `N0·rest·K`; Step 1: `N0·2·K`; Step 3: `rest·2·K`).
- **Reuse** the existing four-step NTT (`fwd2048`/`inv2048`, tile-by-tile — 2 tiles per 2048-poly).
- The many NTTs are independent → **batch dimension across cores** (shard the batch over the ≤32
  Tensix cores), and batch tiles within a program (this is the parked "batched NTT" work — the win is
  amortizing program dispatch across the batch).
- Step 0 is **offline** (one-time DB preprocessing), so it never sits on the online query path.

## Step 2 — the batched matmul → matrix unit (FPU)

Step 2 is a batched matmul (above), so it maps to the **int8 matmul kernel on the matrix engine** — the
highest-throughput unit — using the radix-128 limb lowering for the 28-bit modular MAC (`matmul_fbc` /
`scan_mm` style: int8 32×32 → int32 accumulate, then SFPU recombine mod q). Per slot `j` (per channel):

```
[rest, N0] · [N0, 2] → [rest, 2]      # contract N0=512 (=16 tiles), rows rest=128 (=4 tiles)
```

Per slot `j` (per channel) the matmul is `[r=128, k=512] · [k=512, c=2] → [r=128, c=2]`, tiled onto
the 32×32 engine as:

| matmul axis | size | tiles of 32 |
|---|---|---|
| M = `r` | 128 | 4, full |
| K = `k` (contract) | 512 | 16, full |
| **N = `c`** | **2** | **1 tile, only 2/32 columns used** |

**Utilization (single query):** `k` and `r` fill the engine; only the output width `c=2` is skinny:
```
useful MACs/tile = 32·2·32  =  2,048       engine MACs/tile = 32·32·32 = 32,768
→ 6.25% utilization  →  ~16× wasted on the column axis
```
Cause: the matrix engine is built for matrix×matrix (wide N); one query's `Q_j` has only `c=2` columns.

**Decision — batch queries for throughput.** The `c=2` waste is filled by batching **B independent
queries**, because different queries **share the same DB** (`DB_j` is the shared LHS), so they occupy
the output columns of one matmul:
```
single query : [128,512]·[512, 2]   → N=2   (2/32 filled, ~16× waste)  ← correctness / one-query latency
batch B=16   : [128,512]·[512,32]   → N=32  (full tile, no waste)      ← throughput / QPS
```
Same kernel, just more columns in `Q` — no structural change (the "batch = more tiles" point on the
column axis). So: **single query for correctness + the CPU one-query latency comparison; batched query
(`B≈16`) for the throughput/QPS headline.** Shard the batch (`j`, `K`, queries) across ≤32 cores.

> **✅ VALIDATED on n150** (`kernels_ir/test_scan_qbatch.cpp`): query-batching is a pure **data-level**
> change — B queries (sharing the DB = LHS) pack into the output columns `[rest, 2B]`, the compute
> kernel (`matmul_modq` + `add_modq`, K-tiled) is **unchanged**. `N0=512, rest=32`: B=16 (2B=32, one
> N-tile) 0/2048; B=64 (2B=128, four N-tiles) 0/8192. For `2B>32` the batch splits into 32-column
> **N-tiles** — still the same kernel. The dispatch-free form of Step 2 is `matmul_modq_mbatch`
> (phase-major M-batch; see `SCAN_STATUS.md`), which applies query-batching at the data level (wider
> tiles) with no kernel change.

`test_scan_mc` already measures this engine doing a 512-wide contraction over DB-scale data (~28 GB/s) —
so it is a legitimate *throughput reference for Step 2*, provided it is fed the real `DB_ntt`/`Q_ntt`
and the recombine is wired in and validated bit-exact (that wiring is the remaining work).

## Summary table

| step | op (algorithm) | shape | kernel / engine (implementation) | on-line? |
|---|---|---|---|---|
| 0 | `NTT(DB)` | `[N0,rest,N]` | `fwd2048`, batched across cores | offline |
| 1 | `NTT(Q)` | `[N0,2,N]` | `fwd2048`, batched | online (small) |
| 2 | `Σ_k DB⊙Q` = bmm | `[N,N0,rest]·[N,N0,2]` | int8 matmul, matrix engine, sharded | online (dominant) |
| 3 | `iNTT(A)` | `[rest,2,N]` | `inv2048`, batched | online (small) |

---

## Correction note

An earlier draft (`SCAN_UNIFY_DESIGN.md`) argued Step 2 "is not a matmul" and floated a
coefficient-domain reformulation. That was **wrong on two counts**: (1) it mixed the algorithm level
with the hardware level — proposing to *change the algorithm* (to O(N²)) so a hardware unit would fit,
which is backwards; (2) Step 2 *is* a matmul — a **batched** matmul over the slot axis `j` — so no
reformulation is needed. The algorithm stays the NTT scan of Part 1; the matrix unit runs Step 2's
batched matmul directly. That file is superseded by this one.
