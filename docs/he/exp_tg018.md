# exp_Tg018 — CKKS ct_mul at L_Q = 48, dnum = 3, N = 65536

**Status: CLOSED — the plan and running record for work that concluded elsewhere.** 15 sections are
**[DONE]**; the 1 **[WIP]** and 2 **[TODO]** were overtaken rather than finished. The measured
outcome of this lineage is [`exp_tg018m.md`](exp_tg018m.md) (50.920 ms, the baseline) and
[`exp_tg018ntt.md`](exp_tg018ntt.md) (**23.991 ms**, which is now the repo's benchmark —
`host/benchmarks/ckks_ctmul.cpp`). Read those for numbers; read this for why the approach was
chosen. Nothing here is a measurement unless it says "measured".

`host/experiments/exp_tg018.cpp` · `kernels/experiments/exp_tb001/` · `docs/he/exp_tg000.md` is the predecessor

## Goal

A CKKS `ct_mul` at a **48-limb ciphertext modulus with `dnum = 3`**, on the existing optimised kernel
tree, at high per-core utilisation, with a performance report using the same four latency metrics as
exp_Tb001 and exp_Tg000.

## Parameters

Two were specified, the rest are chosen here.

| | value | why |
|---|---|---|
| `L_Q` | **48** × 28-bit ≈ **1344-bit** `Q` | specified |
| `dnum` | **3** | specified → `alpha = 16` limbs per digit |
| `P` | **16** × 28-bit | must be ≥ one digit (`alpha = 16`) or the key-switch noise blows up |
| `QP` | **64 limbs** | 48 + 16; this is what the multiply actually touches |
| `N` | **65536** | forced by security: `log₂(QP) ≈ 1792` bits needs `N = 2¹⁶` at 128-bit |
| `Δ` | 2 primes ≈ 2⁵⁶ | carried over from exp_Tg000 — the rescale residual is a fixed number of units *at the output scale*, so a wide `Δ` is what buys precision |
| rescale | centred, drop 2 | the `+qHalf` shift identity from exp_Tg000; no kernel change |
| `B`, `S` | 1, 1 | specified |
| `G` | to be chosen | target: fill the 13×10 grid — see *Utilisation* |

`dnum = 3` with `alpha = |P| = 16` is the balanced hybrid key-switch: each digit is ≈ 2⁴⁴⁸ and so is
`P`, so the noise ratio is ≈ 1. This is the same reasoning as exp_Tg000's `dnum = 2`, one step up.

## Why N = 65536 needs a new NTT — and it is not the memory

**[DONE — analysis]** The existing four-step is `N = N1·N2` with `N1 = 32` (the tile dimension) and
`N2 = N/32`. Extending it naively to `N = 65536` gives `N2 = 2048`, `NBLK = 64`, and a stage-1
twiddle matrix of `NBLK × NBLK` blocks — **64 MB per (prime, direction), 8 GB across 64 primes.**

8 GB is *not* the blocker. The p150b has 32 GB of GDDR6 and would hold it, and the constants are
uploaded once at setup, so there is no per-multiply traffic. The first version of this plan called
8 GB the problem; that was wrong.

**The real problem is that the same `N2` drives the arithmetic, and it drives it quadratically.**
Stage A contracts `N2` and costs `N2²·N1` MACs, so a decomposition whose W is 64× larger is also
doing 4× the MACs of a balanced one and 21× the MACs of a 3-level one:

| decomposition | MACs | vs `N log N` | stage-1 W / set | × 128 sets |
|---|---|---|---|---|
| 2-factor `32 × 2048` (naive extension) | 136.3M | **130×** | 64.00 MB | 8.00 GB |
| 2-factor `256 × 256` (balanced) | 33.6M | 32× | 1.00 MB | 0.125 GB |
| **3-level `32 × 32 × (32×2)`** | **6.4M** | **6×** | **0.02 MB** | **0.002 GB** |

The 8 GB is a *symptom* of a bad decomposition, not an independent constraint — memory and MAC count
are the same `N2²` term. That is the argument for decomposing further, and it stands on the middle
column, not the right-hand one. **Directed: 3-level `(32, 32, 64)`.**

### What the 3-level decomposition has to look like

The repo already has a 3-factor NTT at `N = 32768` — `tests/ntt/ntt32768_dev.cpp`, designed in
`docs/matmul/ntt_32768.md`. Its structure is the model:

```
32768 = 32 × 32 × 32,  n = n1 + 32·n2 + 1024·n3,  32 tiles, tile n1 holds Xin[n1][n3][n2]

Block 1 (per tile, = the verified on-tile NTT-1024):
    A  G  = W @ Xin          contract n3   (left-multiply, in-tile)
    B  G' = G ⊙ Ta           twiddle a
    C  H  = G' @ W           contract n2   (right-multiply, in-tile)
TB     Cb = H ⊙ Tb           twiddle b, couples the tile axis n1
A2A    D[s][n1][t] = Cb[n1][s][t]          all-to-all corner turn
Block 2:
    D  E  = W @ D            contract n1, now in-tile
```

**The constraint that shapes everything: the matrix engine contracts only *within* a tile.** A tile
is 32×32, so the in-tile slots hold exactly two axes of 32. With three axes (32, 32, 64) and 64 tiles
of 1024, *one* axis is always left as a pure tile axis and needs a corner turn to be contracted.

**The 64-axis is handled matrix-side, not NTT-side.** The 64-point DFT is not further factored into
butterflies; it is just a 64×64 matrix, used as a **blocked matmul with 2×2 blocks of 32×32** — the
same blocked contraction `fused_matmul_a_acc` already does, at `ACC = 2`. For that to work the
64-axis must have an in-tile component, which fixes the layout:

```
n = n1 + 32·n2 + 1024·n3,   n1, n2 ∈ [0,32),  n3 ∈ [0,64) = n3_lo + 32·n3_hi

tile index  = (n1, n3_hi)      32 × 2 = 64 tiles
in-tile     = [n3_lo][n2]      32 × 32

Pass A   contract n2      in-tile, right-multiply by W32       (fused_matmul_c)      64 tiles
twiddle a
Pass B   contract n3      BLOCKED over n3_hi, in-tile n3_lo,
                          W64 as 2×2 blocks of 32×32           (fused_matmul_a_acc, ACC=2)
twiddle b
A2A      corner turn on n1 (32 ↔ in-tile), twice — once per n3_hi group of 32 tiles
Pass C   contract n1      in-tile, left-multiply by W32        (fused_matmul_a)      64 tiles
```

Doing the 64-dim matrix-side rather than as a radix-2 NTT step is what keeps this clean: no extra
`ω_64` twiddle stage and no butterfly, just one more blocked contraction on a kernel that already
exists. MACs are `N·(32 + 64 + 32) = 8.4M` — Pass A 2.1M, Pass B 4.2M, Pass C 2.1M.

Constants per (prime, direction): `W32` fwd and inv (16 KB each), `W64` as 4 blocks (64 KB), and two
twiddle sets of 64 tiles (256 KB each). Well under a megabyte, against 64 MB for the naive extension.

### The alternative, and the repo's own evidence against assuming

`65536 = 256 × 256` is the balanced two-factor: `NBLK = 8` for *both* stages, 125 MB of constants
total, **no corner turn**, and it reuses the existing `fused_matmul_a_acc` blocked contraction with
only one change — `N1` becomes blocked too, instead of a single in-tile 32×32 W. It costs 5.3× the
MACs of the 3-level.

That sounds decisive until you read `docs/matmul/ntt_32768_perf.md`, where both variants were built
and measured at `N = 32768`:

| variant | measured |
|---|---|
| 3-factor (32×32×32) | **319.5 µs** |
| 2-factor (128×256) | 334.7 µs |

**4.8% apart.** The 3-factor's MAC advantage was very nearly cancelled by its extra reduction, extra
twiddle and the corner turn. At `N = 65536` the corner turn moves 4× the data of the 32768 one while
the MAC gap widens, so which wins is genuinely open — MAC counts alone have already been shown, in
this repo, on this hardware, not to predict the answer.

**Decision: build the 3-level as directed.** It has the best MAC count and the smallest constants, and
it is the one that keeps paramgen cheap at 64 primes. The 256×256 two-factor stays documented as the
fallback if the corner turn measures as dominant — that is a measurement, not a guess, and the plan
is to take it if it comes to that.

## Utilisation

**[TODO]** exp_Tg000 runs at `G = 8` and that is a hard ceiling there: `NBLK = 8`, so every
elementwise, FBC and transform stage has exactly 8 tiles of work and 122 of the 130 cores idle.

`L_Q = 48` changes the picture completely — there are 64 limbs of *independent* work at every stage.
The plan is to **fan out across limbs**: one program dispatch covers `(limb, tile)` pairs, with each
core carrying its own prime.

Concretely this means moving the per-prime constants from *per-program* runtime args to *per-core*
runtime args:

| builder | today | needed |
|---|---|---|
| `build_mm_acc` | `q`, `q_inv_32`, `q_low/high` fixed per call | per-core, from the job's own limb |
| `ew` (`pw_buf`/`madd_buf`/`msub_buf`) | one `cargs` for every core | per-core `cargs` |
| `fbc_res`, `bconv1_res` | one constant vector per call | per-core, or batch same-shape conversions |

With 64 tiles per polynomial at `N = 65536` a *single* limb already fills 64 cores, so the fan-out
matters most for the elementwise stages (1 tile-group × 64 limbs) rather than the transforms.
Target `G = 128` (13×10 grid = 130 usable, 128 = 2 limbs × 64 tiles or 64 limbs × 2 tiles).

## Milestones

| # | milestone | status |
|---|---|---|
| 0 | Parameter and decomposition analysis (this document) | **[DONE]** |
| 1 | Host CKKS at `N=65536`, `L_Q=48`, `P=16`, `dnum=3` — encode/encrypt/multiply/decrypt/decode, verified in slot space | **[DONE]** |
| 2a | NTT-65536 3-level **algebra** validated on the host against the reference transform | **[DONE]** |
| 2b | ...the same pipeline on device, checked limb-by-limb | **[DONE]** |
| 2a+ | inverse transform validated on the host too | **[DONE]** |
| 3 | Device ladder: tensor → relinearise (dnum=3) → rescale, each against a reference | **[DONE]** |
| 4 | Whole multiply on device, bit-exact against the host reference, decrypt+decode verified | **[DONE]** |
| 5 | Limb fan-out for utilisation; pick `G` | **[DONE]** |
| 6 | Performance report — **host-measured metrics only**, see below | **[DONE]** |

Milestone 1 is the natural first step because it is device-free and it produces the fixture
everything else is checked against.

## Milestone 1 — host scheme **[DONE]**

`host/experiments/exp_tg018.cpp`, host only, 8 threads, ~36 s end to end:

```
exp_Tg018  CKKS ct_mul   N=65536  slots=32768  L_Q=48  P=16  QP=64  dnum=3 (alpha=16)
  setup 14.3 s   log2(Q)=1337  log2(QP)~1780   Delta=48806359526146049=2^55.44   level 48 -> 46
[0] canonical embedding round trip      max = 3.900e-14  [PASS]   (2.3 s)
[1] decrypt(encrypt(z))                 max = 4.170e-14  rms = 1.138e-14  ~44.4 bits  [PASS]
[2] decrypt(enc(z1) + enc(z2))          max = 6.271e-14  rms = 1.616e-14  ~43.9 bits  [PASS]
[3] decrypt(enc(z1) (x) enc(z2))        max = 1.905e-12  rms = 3.172e-13  ~38.9 bits  [PASS]  (10.4 s)
```

64 NTT-friendly 28-bit primes exist for `2N = 131072` (80 do), giving `log₂ Q = 1337` and
`log₂ QP = 1780`. Three things had to change from exp_Tg000 and each was forced:

* **24-word (1536-bit) `BIG`** replaces `u512` — a 1337-bit `Q` does not fit, and the CRT
  reconstruction accumulates to ~1350 bits.
* **`BIG` inside the FBC.** `alpha = 16` primes is a 448-bit source product, so `Ŝ_i` and its
  inverse no longer fit `u128` the way they did at `alpha = 4`. The per-coefficient accumulator
  still does: 16 terms of 56 bits is 60 bits, so no per-term reduction is needed.
* **Threading.** The `O(N²/2)` canonical embedding is 2.15G terms per direction here against 0.03G
  at `N = 8192`. Still the definition, not a special FFT — this file is the correctness reference —
  but parallelised over disjoint output chunks.

Precision is **38.9 bits** after one multiply (44.4 fresh), consistent with exp_Tg000's model: the
floor is the rescale residual `√(‖s‖²/12)` at the output scale, and `‖s‖² ≈ 2N/3` is 8× larger here
while `Δ = 2^55.44` is essentially unchanged — so ~1.5 bits worse than exp_Tg000's 42.4, as expected.

## Milestone 2a — the decomposition validated **[DONE]**

`host/experiments/proto_ntt65536_3level.cpp` is a host-only prototype of exactly the pipeline above, compared
against a reference negacyclic NTT-65536:

```
q=268042241  psi=252917277
3-level FWD vs reference: 0 / 65536 mismatches  [PASS]
3-level INV round trip   : 0 / 65536 mismatches  [PASS]
```

**Both directions.** The inverse is the same six passes in reverse order with inverse roots —
`P3' → A2A → Tb' → P2' → Ta' → P1'` — with `N⁻¹` and `ψ⁻ⁿ` folded into the same three places the
forward folds `ψⁿ`. So the device needs one pipeline and two constant sets, not two pipelines.

Index algebra, in full, so the device port is mechanical:

```
n = n1 + 32·n2 + 1024·n3          n1,n2 ∈ [0,32),  n3 = n3_lo + 32·n3_hi ∈ [0,64)
k = kv_lo + 32·kv_hi + 64·ku + 2048·ka

P1   out[kv] = Σ_{n3} W64[kv][n3]·in[n3],   W64[kv][n3] = ω₆₄^{kv·n3} · ψ^{1024·n3}
Ta   × ω₂₀₄₈^{n2·kv}
P2   out[ku] = Σ_{n2} in[n2]·W32[n2][ku],   W32[n2][ku] = ω₃₂^{n2·ku} · ψ^{32·n2}
Tb   × ω_N^{n1·(kv + 64·ku)} · ψ^{n1}
A2A  corner turn n1 ↔ kv_lo within each kv_hi group
P3   out[ka] = Σ_{n1} W32[ka][n1]·in[n1],   W32[ka][n1] = ω₃₂^{n1·ka}
```

The negacyclic twist `ψⁿ = ψ^{n1}·ψ^{32·n2}·ψ^{1024·n3}` is folded into P1's `W64`, `Ta` and `Tb`
respectively, so there is no separate pre-twist pass — the same trick exp_Tb001 uses at N=8192.

Kernel mapping, all of them already in the tree:

| pass | kernel | shape |
|---|---|---|
| P1 | `fused_matmul_a_acc` | blocked, `ACC = 2`, W on the left |
| Ta, Tb | `modmul_sfpi_full` | per-element twiddle |
| P2 | `fused_matmul_c` | in-tile, W on the right |
| A2A | `all_to_all_corner_mc` | needs copying into `kernels/experiments/exp_tb001/` — a new divergence |
| P3 | `fused_matmul_a` | in-tile, W on the left |

## Milestone 2b — the transform on device **[DONE]**

```
exp_Tg018  device NTT-65536 (3-level)  grid 13x10 -> G=64  checking 2 prime(s)
[D1] device NTT-65536 (coeff->eval) vs host   mismatches = 0/131072  [PASS]
[D2] device INTT-65536 round trip             mismatches = 0/131072  [PASS]
```

The forward passed on the first device run. The inverse failed `65536/65536` — total, which by the
signature this repo has learned to read means **structural, not numerical**.

**The bug is worth recording because it is invisible in the forward.** P1 and P2 are not symmetric
matrices: the forward contracts `n3` and outputs `kv`, the inverse contracts `kv` and outputs `n3`,
so the matrix transposes — and the `ψ^{1024·n3}` factor rides on `n3`, which is the *column* one way
and the *row* the other. Building the inverse by taking the forward's index layout and substituting
inverse roots gives a matrix that is right everywhere except where `ψ` sits. `W32b` (P3) has no `ψ`
and is symmetric in its indices, so it needs no transposition at all — which is exactly why the
mistake is easy to make on two of three passes and miss.

One new kernel: `kernels/experiments/exp_tb001/dataflow/a2a_corner_bufs.cpp`, the corner turn over per-tile
buffers. The shared `all_to_all_corner_mc.cpp` addresses its inputs as pages of one buffer, and a
`TensorAccessor` cannot be pointed at "page t of a larger buffer" by address arithmetic because DRAM
interleaving puts page `i` on bank `i mod nbanks`. Teaching the corner turn per-tile addresses is a
smaller change than teaching every matmul reader a page offset.

## Milestones 3 and 4 — the whole multiply on device **[DONE]**

```
exp_Tg018  device  grid 13x10 -> G=64   64 primes, dnum=3, alpha=16
  transform constants resident: 128 sets, 0.7 s
  evk resident: 3 digits x 64 limbs, 2.0 s
[D3] device tensor (pointwise mod Q) vs reference          0/786432  [PASS]
[D4] device relinearise (dnum=3 hybrid key-switch)         0/262144  [PASS]
[D5] device rescale (drop 2 prime(s), centred=1)           0/262144  [PASS]
  host reference multiply: 28.4 s
  chain: 6020 program dispatches, 8.9 s wall (G=64)
[D6] WHOLE MULTIPLY on device vs the host reference       0/6029312  [PASS]
[D7] decode(decrypt(device out)) vs z1(.)z2   max=1.827e-12  rms=3.163e-13  ~39.0 bits  [PASS]
```

**That 8.9 s is not a measurement.** It is raw wall time on the un-cached, un-traced path, where every
one of the 6020 dispatches is built and enqueued individually. It is recorded only to show the chain
completes; the four latency metrics come in milestone 6, after the workload cache and Metal Trace are
wired in as they are for exp_Tb001 and exp_Tg000.

Two things had to change to get here, both recorded as divergences:

* **`baseconv_fbc.cpp` array bounds → `C1[16], C2[16][12], Q[16], QINV[16]`.** `dnum = 3` makes a
  digit `alpha = 16` primes wide, and the arrays were sized for `LS ≤ 10`. Overflowing them corrupts
  the compute RISC's stack silently. exp_Tg000 was re-run after the change and is unaffected.
* **The destination list is CHUNKED to 8.** `LS = 16, LD = 12` is 266 runtime args, past the 256
  budget — so the binding limit on the FBC is *arguments*, not stack. A 48-wide conversion runs as
  6 dispatches of 8.

The bug worth recording from this stage: `[D3]` first failed `786432/786432` — total, so structural
again — because the forward and inverse **constant sets were swapped in the cache key**. The
generator indexed `cs[li*2+d]` with `fwd = (d != 0)` while the cache stored `KC[li*2 + (d ? 0 : 1)]`.
Every transform ran with the other direction's twiddles. The transform itself had already been proven
correct in isolation, which is exactly why the ladder is built stage by stage: the failure could only
be in the new plumbing.

## Milestones 5 and 6 — utilisation and the performance report **[DONE]**

### The measurement

**Host-measured metrics only.** The two trace-derived metrics (busy span, critical path busy) are
*not* reported: see *Why there is no trace-derived pair* below.

| | value |
|---|---|
| parameters | `N=65536`, `L_Q=48`, `P=16`, `QP=64`, `dnum=3`, `alpha=16`, `Δ=2^55.44` |
| workload | `G=128`, `B=1`, `S=1`, coefficient I/O |
| dispatches | **3588** |
| **host wall** | **240.543 ms/op** (5 iters) |
| **host device busy** | **222.066 ms/op** (1 segment replay) |
| correctness | `0/6029312` vs the host reference; decode+decrypt **39.0 bits** |

One trace segment, so the multiply is fully resident — no host boundary inside it.

### Utilisation: why `G=128`

A single transform has exactly 64 tiles of work, so at `G=64` every core does one tile and 66 of the
130 sit idle. There are hundreds of *independent* transforms in a multiply, so the fix is to batch
limbs: one dispatch spans several transforms with **each core carrying its own prime**, through
per-core runtime args rather than per-program ones.

**The workload is still `G`, `B`, `S` and nothing else.** How many limbs share a launch is a
*scheduling* consequence of `G`, not a fourth axis — `exp_tb001.md` defines `G` as a budget the
program "must be built to fit and *use*", and a program handed `G=128` has to pack two limbs per
launch or half of it idles. So it is **derived**: `limbs per dispatch = G / 64`. Both identities hold
as specified: `ct_muls = B·S = 1`, `cores occupied = G·B = 128 ≤ 130`.

It is emphatically not `B`, which replicates the whole multiply across disjoint core groups and would
make `ct_muls = B·S = 2`. This completes exactly **one** ct_mul and only widens each dispatch inside
it. `TG018_LPD` can override the derivation, but only as the diagnostic below — not as a setting.

| `G` | limbs/dispatch (derived) | cores used | dispatches | host wall | host device busy |
|---|---|---|---|---|---|
| 32 | 1 | 32 | 10824 | 317.7 ms | 317.7 ms |
| 64 | 1 | 64 | 5412 | 240.7 ms | 239.2 ms |
| **128** | **2** | **128** | **3588** | **240.5 ms** | **222.1 ms** |

The derivation is what makes `G` mean something. Forcing `TG018_LPD=1` at `G=128` measures
**239.210 ms with the same 5412 dispatches** — identical to `G=64`, because a pass has only 64 jobs to
hand out and the extra 64 cores get none. Raising the budget without the scheduling change buys
exactly nothing, which is the point of deriving it rather than exposing it.

`G=64 → 128` cuts dispatches by 34% but device busy by only 7%. **That is the finding: at this
parameter set the workload is not dispatch-bound the way exp_Tb001's is.** exp_Tb001 at `N=8192`
found 1219 dispatches at ~16 µs of wall each accounting for its entire span, so removing dispatches
was the whole game. Here each dispatch carries 8× the ring and the transform compute dominates, so
halving the dispatch count buys 7%, not 34%. `G=256` is not reachable — the grid is 130 cores.

`G=32 → 64` is worth 24%, so the fan-out is still the right call; it just saturates near one tile
per core.

### Device busy — the real trace, 1-1 verified

The first attempt at this failed its own 1-1 check, and the reason was **not** dropped markers. Two
separate defects, now fixed in the tooling:

1. **`busy_metrics.py` excluded the corner turn structurally.** It filtered dispatches with
   `any(RISC startswith TRISC)`, i.e. it required a compute kernel. `a2a_corner_bufs.cpp` is
   dataflow-only, so its 608 dispatches emitted no TRISC envelope and were dropped — while `--expect`
   still counted them. The map could never close. `--dataflow-ok` keeps them and checks them for
   `≥1` RISC instead of `==5`.
2. **The capture was contaminated.** The drain ran after REC, CAP, two replays and the timing loops,
   and the read-then-remove before it does not reliably clear the device buffer — that residue showed
   up as ~1000 spurious dispatch groups. `TG018_DRAIN_ONLY=1` goes REC → drained capture directly, so
   the CSV holds exactly one multiply.

With both fixed the map closes exactly: **5412 host dispatches, 5412 in the trace, RISCs complete,
join verified against the host labels.**

```
critical path busy    236.622 ms   trace   (1-1 verified, --drain --dataflow-ok)
host device busy      238.439 ms   host
                    -----------
gap                    +1.817 ms   (0.8%: intra-segment idle + ~52 us/call blocking overhead)
```

`busy span` is **not reportable** from a drained capture and is suppressed rather than fudged: the
drain interleaves `Finish` + profiler reads, so the window spans 7,792 ms of mostly pauses.
`critical path busy` is a *sum* of per-dispatch maxima and is unaffected by what happens between
dispatches, which is the only reason a drained capture is worth taking.

Per kernel (`docs/he/data/tg018_dispatch.csv`), `G=64`, `B=1`, `S=1`:

| kernel | N | total ms | % |
|---|--:|--:|--:|
| **`a2a_corner`** | 608 | **103.623** | **43.8%** |
| `modmul_sfpi_full` | 2074 | 48.433 | 20.5% |
| `baseconv_fbc` | 30 | 25.572 | 10.8% |
| `fused_matmul_a_acc` | 608 | 18.998 | 8.0% |
| `fused_matmul_a` | 608 | 12.470 | 5.3% |
| `fused_matmul_c` | 608 | 12.286 | 5.2% |
| `modadd_map` | 590 | 9.747 | 4.1% |
| `modsub_reduce` | 282 | 4.644 | 2.0% |
| `bconv_one` | 4 | 0.848 | 0.4% |
| **total** | **5412** | **236.622** | |

Per HE operation:

| HE op | dispatches | ms | % |
|---|--:|--:|--:|
| keyswitch | 2782 | 135.157 | 57.1% |
| input (coeff→eval) | 1152 | 55.800 | 23.6% |
| output (eval→coeff) | 576 | 27.739 | 11.7% |
| rescale | 566 | 11.263 | 4.8% |
| tensor | 240 | 5.077 | 2.1% |
| combine | 96 | 1.587 | 0.7% |

**`a2a_corner` is 43.8% — more than every matmul combined (18.6%).** The corner turn the 3-level
decomposition forces, not the arithmetic it saves, is what this transform costs. MACs predicted 6×
`N log N`; the data reshuffle needed to get there dominates. That makes the 256×256 two-factor — no
corner turn at 5.3× the MACs — the obvious next A/B rather than a documented fallback.

### The census method, and how it held up

Before the trace was fixed, the same number was estimated by census: take `N_k` from the host label
file and `mean_busy_k` from whatever sample the trace captured, then
`critical path busy ≈ Σ N_k × mean_k`. `tools/traces/kernel_census.py` does this and reports the
per-kernel spread so the uniformity assumption is checked rather than assumed — every compute kernel
came in at **CV ≤ 5.5%**, several under 1%.

Against the real measurement it was **236.61 vs 236.622 ms — 0.005%**, and on the corner turn (the one
row it had to bound rather than measure) 103.36 vs 103.623 ms, **0.25%**.

Two things justify the composition, both measured rather than argued:

| | measured | |
|---|---|---|
| cores *within* a dispatch | sum-over-cores ÷ max-over-cores = **50.6** (median 53.4) of 64 | concurrent, so `mean` must take the **max**; summing would be ~50× high |
| dispatches *between* each other | **0 overlapping pairs of 6412** | serialized on the queue, so composing across types is a **sum** |

So `N` counts *program dispatches* and the parallelism lives entirely inside `mean`. A corollary worth
keeping: **do not just add the trace's group latencies.** The contaminated capture held 6412 groups
against 5412 real dispatches and summed to 289.2 ms — 21% high. Taking `N` from the host and only
`mean` from the trace was robust to the spurious markers as well as to the dropped ones.

The census is still worth having: it is the only route when a workload genuinely cannot be captured
1-1, and it now has a calibration point showing it good to ~0.01% on a workload this size.

### On `busy span`

Three of the four metrics are reported above. `busy span` is not, and cannot be from a drained
capture — it is a *window*, so the drain pauses are inside it. Getting it needs an undrained single
pass, which at 5412 dispatches × 64 cores overflows the profiler's per-RISC buffer (the traced replay
captured ~1500 of 6020). That is the one genuine tooling limit left here, and it is the metric the
other three constrain most tightly anyway: `wall ≥ busy span ≥ critical path busy` is already pinned
between 239.6 and 236.6 ms.

## Optimisation leads

Nothing here is done. This is the record of what the measurement exposed, so the next pass starts
from evidence rather than from a fresh guess.

### 1. The corner turn — 43.8% of device busy

`a2a_corner` is **103.6 ms of 236.6 ms**, more than every matmul combined (18.6%). The 3-level
decomposition was chosen because it does 6.4M MACs against the naive extension's 136M, and that part
is real — but the data reshuffle it forces to get there costs more than the arithmetic it saves.

Three things to try, cheapest first:

* **Parallelise it.** Each core currently produces exactly one output tile and stages all 32 input
  tiles in L1 (128 KB) to do it, so the same 32 tiles are read 32 times over. The read volume is
  `NHALF²` per group where the useful data is `NHALF`. A two-stage turn, or staging fewer inputs per
  core and doing more passes, trades that quadratic read for more dispatches.
* **Fold it into the neighbouring pass.** The turn sits between `Tb` and `P3`. If `P3`'s reader could
  gather its rows directly from the 32 source tiles, the turn disappears as a separate dispatch —
  this is the same move that produced `modadd_twiddle` in exp_Tb001.
* **Do the reshuffle over the NoC instead of through DRAM.** `NOC_BENCH`'s stated purpose in
  `host/experiments/exp_tb001.cpp` is *"DRAM round-trip vs NoC core-to-core all-to-all, for a G²-tile
  reshuffle"* — which is exactly what this kernel is. It currently goes through DRAM (32 tile reads +
  1 write per core, 50.8 GB/s effective, **18% of the 287 GB/s DRAM cap**). At `G=64` the measured
  alternatives are DRAM 287 GB/s counting write+read against NoC **406 GB/s one-way**, so per useful
  byte the NoC path is ~2.8× better *and* skips the round trip entirely. This is the
  strongest-supported lead in this document: the benchmark that justifies it already exists, and it
  targets the kernel that is 43.8% of the multiply.
* **Take the 256×256 two-factor instead.** No corner turn at all, at 5.3× the MACs
  (33.6M vs 6.4M). Given the matmuls currently total 43.8 ms against the turn's 103.6, 5.3× the
  matmul work would be ~232 ms — worse on its face, *but* it also removes `Ta`, `Tb` and two of the
  three passes. This needs building and measuring, not estimating; the repo's own N=32768 data
  (3-factor and 2-factor within 4.8% despite a 4× MAC gap) is exactly the warning against predicting
  it.

### 2. Why a core is slower at G=64 than at G=8 — and what it is NOT

Per-core work per dispatch is **identical** between exp_Tb001/exp_Tg000 and this experiment — one
32×32 tile either way; `N` changes only how many dispatches there are, not what each one does. So the
elementwise and single-block-matmul rows are like-for-like and the only variable is `G`.

**Per core** (the dispatch latency *is* one core's time, since every core in a dispatch does the same
thing). `wk/core` is the kernel's runtime loop trip count; full data in
`docs/he/data/tb001_vs_tg018_percore.csv`:

| kernel | tb001 `G=8` wk/core | µs/disp | µs/wk | tg018 `G=64` wk/core | µs/disp | µs/wk | per-core |
|---|--:|--:|--:|--:|--:|--:|--:|
| `modsub_reduce` | 1 | 2.6 | 2.60 | 1 | 16.5 | 16.50 | **6.35×** |
| `modadd_map` | 1 | 2.9 | 2.90 | 1 | 16.5 | 16.50 | **5.69×** |
| `bconv_one` | 7 | 7.6 | 1.09 | 46 | 212.0 | 4.61 | 4.24× |
| `fused_matmul_a_acc` | 8 | 35.0 | 4.38 | 2 | 31.2 | 15.60 | 3.57× |
| `fused_matmul_a` | 1 | 10.8 | 10.80 | 1 | 20.5 | 20.50 | 1.90× |
| `fused_matmul_c` | 1 | 10.9 | 10.90 | 1 | 20.2 | 20.20 | 1.85× |
| `modmul_sfpi_full` | 1 | 14.7 | 14.70 | 1 | 23.4 | 23.40 | 1.59× |
| `baseconv_fbc` | 28 | 386.0 | 13.79 | 128 | 852.4 | 6.66 | **0.48×** |
| `a2a_corner` | — | — | — | 32 | 170.4 | 5.33 | — |

So **aggregate** throughput improves 1.26–5.04× (§2b) while **per core** everything except
`baseconv_fbc` gets 1.6–6.4× slower. Both are true; the 8× core count is the bridge. The per-core
column is the actionable one because it isolates the inefficiency instead of hiding it behind `G`.

#### It is NOT memory bandwidth. Measured, not argued.

`NOC_BENCH` (`bench_dram_rt.cpp` / `bench_noc_a2a.cpp`, tight loop, R=200–400) on this card, two
independent runs per point:

| `G` | DRAM round-trip | NoC L1→L1 |
|--:|---|---|
| 8 | 151.4 | 95.7 |
| 32 | 273.0 | 217.6 |
| 64 | **287.1 / 287.1** | 364.2 / 406.4 |
| 96 | **300.4 / 300.7** | 547.0 / 545.7 |
| 128 | **304.6 / 305.0** | 654.2 / 659.1 |

**Only the DRAM column is a ceiling.** It reproduces to 0.1% and plateaus (273 → 287 → 300 → 305), so
~305 GB/s is a sound reference for this access pattern. Caveats worth keeping attached: it is a
*practical* cap and not the silicon one (~60% of the part's peak); it counts **write + read** over one
wall time, so a read-only kernel should not be held to it exactly; and it is measured at 4 KB pages
with `G` cores each streaming `G` tiles.

**The NoC column is not a cap.** 364 → 547 → 659 GB/s, still climbing at `G=128` with no plateau and
11% run-to-run spread at `G=64`. Core-to-core traffic recruits more independent links as more cores
participate, so aggregate NoC bandwidth is a *function of `G` and the pattern*, not a single number.
There is no "the NoC bandwidth" to quote.

The DRAM figure is used below only for a **negative** inference — the kernels are far enough beneath it
that bandwidth cannot be the binding constraint — which is robust even if the true cap were half of it.
It must **not** be used to predict how fast a kernel could run if optimised; that needs a
microbenchmark matching the kernel's own access pattern. Against that, the multiply's kernels achieve **48–92 GB/s at
`G=64`, i.e. 17–32% of what the same core count demonstrably delivers.** And the transfer itself is a
*minority* of every dispatch:

| kernel | DRAM µs @ `G=64` | of µs/disp | % |
|---|--:|--:|--:|
| `modadd_map` | 2.75 | 16.5 | **17%** |
| `modsub_reduce` | 2.75 | 16.5 | **17%** |
| `modmul_sfpi_full` | 2.75 | 23.4 | 12% |
| `fused_matmul_a` | 5.51 | 20.5 | 27% |
| `fused_matmul_a_acc` | 10.10 | 31.2 | 32% |
| `baseconv_fbc` | 22.03 | 852.4 | 3% |
| `bconv_one` | 43.14 | 212.0 | 20% |

`modadd_map` spends 2.75 µs moving data and **13.75 µs not moving data**. Bandwidth is not the wall.

#### The mechanism — CONFIRMED (was an inference; see the per-RISC measurement below)

Exposed DRAM **round-trip latency under queuing**. `reader_ab` and `reader_modmul` issue their two
reads and then `noc_async_read_barrier()`, so the core idles until both land. With 64 cores issuing
concurrently rather than 8 there are 8× as many requests queued at the controller, so each
*completion* takes longer even though aggregate throughput is far from the limit. That predicts the
observed ordering exactly:

| kernel | reads in flight | compute to hide behind | per-core |
|---|--:|---|--:|
| `modadd`/`modsub` | 2 | essentially none | 5.7–6.4× |
| `modmul_sfpi_full` | 2 | ~14 µs of SFPI | 1.59× |
| `fused_matmul_a/c` | **17** (`reader_fused_batch`) | moderate | 1.85–1.90× |
| `baseconv_fbc` | 16 | 128 modmuls | **improved** |

`NOC_BENCH` reaching 285 GB/s is consistent: 200 iterations with reads continuously in flight is
throughput-shaped, not latency-exposed.

**The clean test, not yet run:** raise `n_tiles` on `modadd_map` behind a *prefetching* reader and see
whether per-dispatch time grows sub-linearly in the tile count. If it does, latency; if it grows
linearly, something else.

#### Two earlier attributions in this document were wrong

Recorded because the reasoning error is reusable, not just the conclusion:

1. **"A fixed per-dispatch launch cost."** Inferred from the absolute deltas clustering at +9…+14 µs.
   Killed by the `fused_matmul_a_acc` vs `fused_matmul_a` split: the *marginal* cost of one extra
   block-matmul on one core went 3.53 → 10.70 µs (**3.03×**) while fixed+recombine went 6.77 → 9.80 µs
   (1.45×). The marginal, data-touching part is what slowed, so it is not launch overhead.
2. **"A shared bandwidth ceiling of ~48–92 GB/s."** Inferred from bytes ÷ dispatch-time, which is an
   *effective rate* over a window containing compute — not a bandwidth measurement, and not comparable
   to a hardware figure. Killed by `NOC_BENCH`: the card gives 285.6 GB/s at the same `G=64`.

**The lesson: bytes ÷ dispatch-time is not bandwidth.** Comparing two *different* kernels' effective
rates conflates their arithmetic intensity; only the same kernel at two trip counts, or a dedicated
microbenchmark, separates movement from compute.

#### Where the non-transfer time goes — measured per RISC

For the `modadd`/`modsub` class (881 dispatches in the 14–19 µs band), the per-RISC `-KERNEL` spans:

| RISC | role | mean µs |
|---|---|--:|
| BRISC | writer | 16.66 |
| NCRISC | reader | 15.52 |
| TRISC_0 | unpack | 15.85 |
| TRISC_1 | math | 16.22 |
| **TRISC_2** | **pack** | **0.57** |

Four of five RISCs span essentially the whole dispatch — they are **blocked on each other**, not
working. The only one reflecting real work is the packer at **0.57 µs**. So a 16.5 µs dispatch is
`read round-trip + ~1 µs of compute + write round-trip`, fully serialised, and backing the latency out
gives **~1 µs per round trip at `G=8` against ~7.5 µs at `G=64`** — a 7.5× growth that matches the
5.7× per-core degradation.

This confirms the latency hypothesis and kills the fixed-launch-cost one for good: a launch cost would
not put the packer at 0.57 µs while the reader sits at 15.5.

#### Tier-1 batching: implemented, correct, and a NULL RESULT

`reader_gather_batch.cpp` / `writer_scatter_batch.cpp` (new, this tree) hoist the barrier out of the
per-tile loop so `Ls` reads and `Ld` writes are in flight behind one barrier instead of one round trip
each. Wired into `baseconv_fbc` and `bconv_one`.

Correctness held (`0/6029312`, 39.0 bits). **Runtime: 238.675 ms against 238.439 — no change.**

The reason is instructive and it was a prediction error on my part, not a bug: `baseconv_fbc` does
`cb_wait_front(cb_x, LS)`, so its 16 reads genuinely were exposed — but its dispatch is **852 µs**, of
which the reads are ~2%. I had estimated ~5 ms of headroom by applying `modadd`'s *contended* 11 µs
latency to a kernel whose 1-in-flight pattern keeps the queue shallow and its round trips short. The
kernels with exposed latency and the kernels with a barrier-per-tile reader are **disjoint sets**.

The change is kept: it is the right pattern, costs nothing, and raises in-flight depth for any future
caller at a wider shape.

#### Why the bandwidth is unusable: outstanding requests, not capacity

> **SCOPE NARROWED — see `docs/he/history/exp_tg018dram.md`.** Everything in this section is correct **for
> the kernels it was drawn on** — `modadd_map`, `modsub_reduce`, the corner turn — which have almost
> no compute per tile. A direct sweep of the two *dominant* kernels then showed it does **not**
> generalise to them:
>
> * `modmul_sfpi_full` is **compute-bound at every core count** (12.45 µs of SFPU work per tile
>   against 12 KB of traffic). At `G=130` it delivers 123.5 GB/s and its compute asks for 125 — it
>   is already getting everything it wants.
> * The "16% of achievable" figure below is **bytes ÷ whole-dispatch time**, which on a
>   compute-bound kernel measures arithmetic intensity, not the memory system. Same configuration,
>   reader envelope only: **164 GB/s**, not 47.6.
> * Deeper requests made **both** dominant kernels 1.17–1.69× *slower* — with 8–12 µs of compute per
>   tile, a one-tile-at-a-time reader already overlaps perfectly, and batching just converts a smooth
>   request stream into bursts that queue against each other.
> * The real ceiling is **~220 GB/s**, measured three independent ways, not the 305 GB/s used below.
>
> The mechanism described here is real and the `G=8`→`G=64` queueing explanation stands. It is the
> *generalisation* to the whole multiply that was too broad.

This is the answer to "the card has 512 GB/s, why can we only use 48?"

**Bandwidth is a rate you only realise by keeping many requests outstanding. It is not a resource you
receive by asking once.** A 500-lane highway carries nothing extra if you send two cars and wait for
them to return before sending two more: throughput is `2 cars ÷ round-trip time`, and widening the
road does not change it. Only putting more cars on it does. That is exactly
`read a; read b; noc_async_read_barrier()` — two requests, then block.

The controlled evidence. **Same `G=64`, same DRAM, same interleaved buffers — only the in-flight count
differs:**

| in flight per core | bytes/core | throughput | of the 305 GB/s cap |
|---|--:|--:|--:|
| 2 tiles (`modadd_map`) | 8 KB | 47.6 GB/s | 16% |
| 17 tiles (`fused_matmul_a`) | 20 KB | 76.7 GB/s | 25% |
| 64 tiles (`bench_dram_rt`) | 256 KB | **287 GB/s** | **94%** |

A **6× throughput swing at fixed core count**, from nothing but request depth. The bandwidth was
always there; the kernels never asked for it.

And the `G=8` → `G=64` comparison, which is the question that started this:

| | requests outstanding | round trip | throughput | per-core time |
|---|--:|--:|--:|--:|
| `G=8` | 16 | ~1.9 µs | 33.9 GB/s | 2.9 µs |
| `G=64` | 128 | ~10.7 µs | 47.6 GB/s | 16.5 µs |

8× the requests bought 1.4× the throughput because the round trip stretched 5.6×. That is **queueing**,
and it is the correct behaviour of a more heavily used shared path. **Per core it looks 5.7× worse, but
nothing got worse** — each core waits in a longer line because more work is being served, and the
aggregate went up. exp_Tb001's cores are not faster; they queue behind 7 others instead of 63.

#### The bottleneck, ranked — all four candidates tested

| candidate | verdict | evidence |
|---|---|---|
| DRAM bandwidth | **no** | 16% of a measured 305 GB/s cap; transfer is 17% of the dispatch |
| Compute | **no** | packer does 0.57 µs of a 16.5 µs dispatch |
| Launch / dispatch overhead | **no** | per-RISC data — a launch cost cannot leave pack at 0.57 µs while the reader sits at 15.5 µs |
| **Outstanding-request depth** | **YES** | 6× throughput swing at fixed `G` from in-flight count alone |

#### Why the kernels cannot ask for more

Three structural reasons, all in the elementwise path:

* **`n_tiles = 1`** — one tile per core, so exactly **2 read requests**, full stop.
* **0.57 µs of compute** — nothing to overlap the wait against. (`baseconv_fbc` has 852 µs, which is
  precisely why its exposed reads cost it nothing and why batching them was a null result.)
* **an immediate barrier** — the core blocks rather than issuing further work.

There are only two ways to put more requests in flight: **more tiles per core**, or **larger requests**
(16 KB packets rather than 4 KB tiles). No third option exists, which is why Tier-1 batching could not
have helped — it deepened kernels that already had enough in flight.

**One line:** we cannot use the bandwidth because we only ever have **8 KB per core outstanding against
a machine that needs ~256 KB per core to saturate**, and with 0.57 µs of compute there is nothing to
hide the wait behind.

#### What follows for optimisation

`modadd_map` + `modsub_reduce` are 872 dispatches, **14.4 ms of the 236.6**, at 17% DRAM occupancy.

* **Fusion is still the strongest move** — the intermediate never reaches DRAM, so the round trip
  disappears entirely. `modadd_twiddle` in exp_Tb001 is the precedent.
* **A prefetching reader alone is NOT enough** — measured, see the null result above. It only helps a
  kernel whose exposed round trips are a meaningful fraction of its dispatch, and for the two kernels
  that qualify (`modadd_map`, `modsub_reduce`) `n_tiles = 1`, so there is nothing to batch without
  also giving each core more tiles.
* **Bank-aware access is the remaining DRAM-side lever**, and the only one that helps the latency-bound
  kernels without changing the schedule or fusing: it attacks the round-trip latency itself (NoC route
  overlap, multiple readers per bank) rather than needing more concurrency to hide it.
  `get_optimal_dram_bank_to_logical_worker_assignment(NOC_0)` is available in the installed tt-metalium.
  Note it assigns one core per bank (8 on Blackhole), so the full recipe implies sharded buffers, which
  crosses into scheduling territory.
* **A narrower `G` for the cheap stages** remains plausible but is now the weakest of the three: it
  reduces queue depth without removing the round trip.

**Reordered by `exp_Tg018dram`.** The sweep of the two dominant kernels puts a different move at the
top of this list, and it is not a DRAM lever at all — it is *not re-reading the same weights*:

* **`w_share` on the fused matmul family (43.8 ms) — up to 1.80×.** A work tile is 4 KB of data and
  16 KB of folded W, and every tile of a dispatch already runs against one prime, hence one W. Read
  it once per core instead of once per tile and the kernel goes from memory-bound back to
  compute-bound at `G=130` (15.08 → 8.38 µs/tile against a 7.99 floor). It needs no new arithmetic —
  only more than one tile per dispatch, so there is something to share across.
* **More tiles per core, on its own — −22%.** `modmul` at `G=64` goes from 16.16 µs/tile at `T=1` to
  12.64 at `T=64`, because the first tile's round trip stops being paid once per tile. This is the
  deferred "Tier 2", now quantified.
* **`modmul_sfpi_full` (48.4 ms) will not respond to any memory optimisation** — it runs at 96% of
  its single-core floor with all 130 cores loaded. The only lever there is the SFPI Montgomery
  multiply itself. This also explains why Tier 1 *had* to be a null result, and closes it.
* **Do not batch the readers.** Tier 3's 16 KB-packet direction inherits the risk: it deepens
  requests on kernels that do not need depth. Measured: 1.17–1.69× *slower* on both kernels.
* **FIRST: every tile in this experiment is on DRAM bank 0.** Tiles are allocated as separate
  single-page buffers, and tt-metal puts page 0 of every buffer on bank 0 — so the whole working set
  uses one of eight channels. `exp_Tg018m` fixes only that and the multiply goes **238.7 → 133.6 ms
  (1.79×)** at bit-identical output, with the corner turn 4.1× and the elementwise maps 3.7–3.9×.
  See `docs/he/exp_tg018m.md`. This also supersedes the depth explanation for `modadd_map` /
  `modsub_reduce` above: they were latency-bound as recorded, but from single-banking rather than
  request depth, which is why deepening them was a null result.
* **Placement and the second NoC come first, and neither is a kernel change.** `docs/he/dram_guide.md`
  measures the machine directly: a grid row carries 86.4 GB/s and one core already saturates it, so
  the tree's row-major placement pins every `G <= 13` dispatch at ~86 GB/s; and a kernel that reads
  only in its reader uses one of the two NoCs. Fixing both took the same workload from 245 to
  **433 GB/s**, which is 99.3% of what the eight DRAM channels deliver. Ordering by measured value:
  placement, second NoC, more tiles per core, `w_share` — and not deeper batching.

### 2b. Framing it as scaling efficiency, not as "tg018 is slow"

exp_Tb001 finishes in 7.494 ms and this in 236.622 ms, which invites the wrong conclusion. It is a
**78× larger problem**, and per unit of work this experiment is the *more* efficient of the two:

| | tb001 | tg018 | ratio |
|---|--:|--:|--:|
| tile-work (dispatches × `G`) | 4,416 | 346,368 | **78.4×** |
| critical path busy | 7.494 ms | 236.622 ms | **31.6×** |
| **µs per tile processed** | **1.697** | **0.683** | **0.40×** |

78× the work in 31.6× the time — **2.48× more efficient per tile**, entirely because it fills 64 cores
instead of 8.

The useful target falls out of the same-kernel per-tile view. Widening 8 → 64 cores *should* give 8×
per tile; what is actually realised:

| kernel | tb001 `G=8` µs/tile | tg018 `G=64` µs/tile | gain | % of ideal 8× |
|---|--:|--:|--:|--:|
| `modmul_sfpi_full` | 1.837 | 0.366 | 5.03× | **63%** |
| `fused_matmul_c` | 1.363 | 0.316 | 4.32× | 54% |
| `fused_matmul_a` | 1.350 | 0.320 | 4.21× | 53% |
| `modadd_map` | 0.362 | 0.258 | 1.41× | 18% |
| `modsub_reduce` | 0.325 | 0.258 | 1.26× | **16%** |

So the target is not "make the kernels faster" — it is **the best kernel leaves 37% of the widening on
the table and the cheapest leave 82%.** Same conclusion as lead 2, now with a ceiling attached.

### 2c. Per-core work: identical for most kernels, NOT for four

The comparison above is only legitimate where a core does the same thing in both experiments. Per
dispatch, per core, both process exactly **one 32×32 tile** — same kernel, same 1024 elements, same
format; `N` changes the dispatch *count*, not the per-core work. Four kernels break that:

| kernel | tb001 per core | tg018 per core | |
|---|---|---|---|
| `fused_matmul_a_acc` | `ACC=8` partials | `ACC=2` | tb001 does **4× more** |
| `baseconv_fbc` | `Ls=4` → 36 modmuls | `Ls=16` → 144 | tg018 does **4× more** |
| `bconv_one` | `Ld=7` | `Ld=46` | tg018 does **6.6× more** |
| `a2a_corner` | — | reads **32 tiles**, writes 1 | tg018 only |

That last row is the structural reason lead 1 exists: a2a is the only kernel whose per-core *read*
volume is 32:1 against its output.

### 3. Rows that are NOT comparable, and why

Recorded so nobody reads a speedup or regression into them later:

| kernel | tg000 `G=8` | tg018 `G=64` | why not comparable |
|---|--:|--:|---|
| `fused_matmul_a_acc` | 35.0 | 31.2 | `ACC=8` there, **`ACC=2`** here — the 64-dim is 2 blocks. Looks faster; it is doing a quarter of the accumulation. |
| `baseconv_fbc` | 194.9 | 852.4 | `Ls` 4 → **16** (`alpha=16` digits). More sources, more work. |
| `bconv_one` | 7.6 | 212.0 | `Ld` 7 → **46**. It loops over destinations. |
| `a2a_corner` | — | 170.4 | no counterpart at either earlier ring |

### 4. Eval-form I/O

`input` (coeff→eval) is 55.8 ms and `output` (eval→coeff) 27.7 ms — **35.3% of device busy** spent on
domain conversion. Eval-form I/O removes the input transforms outright but forces an eval-domain
rescale, which at `N=8192` was roughly a wash. Here the balance differs and it has not been measured.

### 5. Key-switch is 57.1%

`keyswitch` is 135.2 ms of 236.6, and it has had no attention at all. exp_Tb001's outstanding leads
apply directly: merging runs of `modmul_sfpi_full`, a fused evk multiply-accumulate, and folding the
forward twiddle into the following read.

## Known risks

* ~~Host bignum~~ — **resolved**, 24-word `BIG`.
* ~~Host reference cost~~ — **resolved**, threaded; 36 s end to end, no special FFT needed.
* **The corner turn is a single-core all-to-all** in the 32768 code. At `N = 65536` the tile axis is
  64, so the 3-level needs *two* 32-tile corner turns (after a radix-2 butterfly splits the axis).
  If the corner turn does not parallelise it could dominate — this is the main open risk in the
  directed decomposition, and the reason the 256×256 fallback is documented rather than discarded.
* **Transform count.** The coefficient-I/O chain measured at **6020 dispatches**: 192 input NTTs +
  48 to-coeff + 144 mod-up + 128 mod-down + 96 output INTTs = 608 transforms at ~7 dispatches each,
  plus the pointwise and FBC stages. Eval-form I/O would remove the 192 input transforms outright —
  that is the first thing to try before measuring.
* **The corner turn is now costed (43.7%) but only by census + bound, not directly.** Measuring it
  cleanly needs either a profiler zone in a dataflow kernel (nothing in this repo does that yet) or an
  A/B that removes the dispatches and differences the wall time.
* **Eval-form I/O is not implemented.** It would remove the 192 input transforms and add ~190 for an
  eval-domain rescale — roughly a wash here, unlike at `N=8192`. Worth measuring, not assuming.

## Progress log

* **2026-08-07** — Parameters settled (`L_Q=48`, `P=16`, `QP=64`, `dnum=3`, `alpha=16`, `N=65536`).
  Worked out that the existing four-step extended naively to `N=65536` needs 8 GB of stage-1
  twiddles. **Initially recorded that 8 GB as the blocker, which was wrong** — the p150b has 32 GB
  and the upload is once, at setup. Redid the analysis on arithmetic instead: the same `N2 = 2048`
  costs `N2²·N1 = 136M` MACs, 130× `N log N` and 21× the 3-level, so memory and compute are the same
  quadratic term and the middle column of the table is the argument. Identified that the tile axis is
  64 while the corner turn requires 32, so "(32,32,64)" is really `32×32×32×2` on this hardware.
  Surveyed `tests/ntt/ntt32768_dev.cpp` (3-factor) and `ntt32768_2factor_dev.cpp` as the reusable
  scheduling models, and found the repo's own measurement putting them only 4.8% apart at `N=32768` —
  recorded as the reason the 256×256 two-factor stays a live fallback rather than a rejected option.
