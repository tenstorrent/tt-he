# BFV on Tenstorrent — future plan

**Where we are (wrapped up).** HE **addition** and HE **multiplication** (tensor → rescale →
relinearize → add) both run correctly on device — *all arithmetic on device*, bit-exact to the exact
big-int golden, exposed through the `HEContext` demo. `ct_mul` is ~12 ms device-active (NTT- and
dispatch-bound). See `README.md` / `PLAN.md` / `ct_mul_perf.md` / `kernels.md`.

This doc is the roadmap from here, prioritized. Three tracks: **performance**, **capability**,
**applications**. Nothing below is a correctness gap in what exists — these are the next things to
build.

---

## Track A — performance (make `ct_mul` fast)

`ct_mul` is **NTT-bound (55%)** and **dispatch-bound (501 mostly single-tile programs)**. Levers,
highest-value first:

1. **Fuse the NTT twiddle into the matmul stage.** The twiddle modmul is a standalone single-tile
   Montgomery modmul between the two int8 matmuls — the NTT-1024 report flags it as fusible into
   `fused_matmul_a` / `fused_matmul_c`. (`ct_mul_perf.md` §5, `kernels.md` §3.)

   **Note on scale.** This item called it "the single largest kernel, 19%, biggest single win". The
   modmul *kernel itself* has since been optimised 4.03× and is CLOSED (`docs/he/exp_modmul001.md`):
   12.432 → 3.086 µs/tile, one SFPI pass, arithmetic-bound at ~2 cycles/op. So what is left here is
   the **dispatch and data-movement** saving from fusing it into the matmul, not the arithmetic —
   the arithmetic is already at its floor. Re-size the prize before building.
2. **Batch limbs / reduce dispatch count.** 501 dispatches × ~1.6 µs launch = ~0.8 ms pure launch
   tax, and each is single-tile (32×32) — the device is under-fed. Packing multiple RNS limbs per tile
   (or per program) cuts both the dispatch count and per-op setup. This is also what makes the
   *wall-clock* (not just device-active) drop.
3. **Cheaper FBC recombine.** 3 base conversions cost ~1.05 ms; ~85% is the per-destination-modulus
   Montgomery recombine (the `bconv` doc's standing target).
4. **FPU-path base conversion. — DONE AND MEASURED, `docs/he/history/exp_tg018r.md`.** The FBC combine runs
   on the int8 matrix engine and **wins every shape measured** at N=2^16 on 64 cores: 1.19× at
   `8→8`, **10.3× at `32→32`**, 1.98× at the `Ls=16, Ld=8` shape exp_Tg018 runs, all bit-identical
   to the SFPU route.

   **The all-to-all transpose this item called for turned out not to be needed.** Transposing the
   *operands* rather than the data — `plane'[j][c] = Σ Bᵀ[j][lane]·Aᵀ[lane][c]` instead of
   `plane[c][j] = Σ A[c][lane]·B[lane][j]` — makes the matmul want the limb axis **inside** the tile,
   which is exactly what a strided read of the limb-major tiles we already store produces, because a
   tilized row is two contiguous 16-element runs. The transpose collapses to 2 NoC copies of 64 bytes
   per (coefficient-group, limb) in `reader_fbc_tr.cpp` and the same in `writer_fbc_tr.cpp`, inside
   the one dispatch, with the radix planes never leaving L1. It costs ~10 µs of 277.

   Remaining: port it into exp_Tg018m, then lift the `Ld` cap (which was `baseconv_fbc`'s stack
   arrays, not the dispatch budget) to run 32-wide column groups.

Target: a fused, batched `ct_mul` well under the current 12 ms, and a wall-clock dominated by device
work rather than JIT + readbacks.

---

## Track B — zero-host & capability

1. **Zero-host end-to-end (data movement only, no arithmetic left).** Remaining host touches are the
   ~48 stage-*handoff* readbacks — `rescale` / `fbc_fused` / `moddown` still take host arrays. Give
   them **buffer-in/buffer-out** signatures so residues stay resident in DRAM across the whole chain;
   then one **build-once, single-program** `ct_mul`. The `reindex` transpose is already on-device and,
   since `ct_mul` is order-agnostic (pointwise / per-coefficient), can be **dropped entirely** by
   carrying a consistent permuted order + pre-permuted twiddles/evk. (`PLAN.md` §4.)
2. **Proper fix for the fused-FBC `n≠4` bug.** Today the FBC is padded to width 4 (inert limb). A real
   fix — per-tile `cb_push_back` in `baseconv_fbc.cpp`, or CB depth = `n` — removes the pad and lets any
   modulus count run natively. (`PLAN.md` §3, `tt_fused_fbc_n4_bug` memory.)
3. **Multiplicative depth ≥ 2.** `Q` = 3 (84-bit) is a *one-multiply* budget. Deeper circuits need
   either a larger `Q` (→ the `L_src→L_dst` FBC generalization, or the proper FBC fix above so 5-limb
   bases work) or a noise re-check in the golden. Add a **modulus-switch / level** mechanism.
4. **More HE operations** (reuse the built primitives):
   - **plaintext × ciphertext** and **plaintext add** — trivial from `modmul`/`modadd`; needed by PIR.
   - **rotations / Galois automorphisms** — a key-switch (relin machinery) under a Galois key; unlocks
     SIMD-slot algorithms.
5. **Larger `N` / multi-core.** Scale past single-Tensix N=1024 — the all-to-all transpose (Track A.4)
   is the shared NTT/bconv primitive at large N, and the int8 matmul already has a multi-core variant
   (`ntt1024_dev_mc`).

---

## Track C — applications

- **OnionPIR-style PIR demo** (`reference/pir/`). It needs only the *easy* ops already built —
  ciphertext add, plaintext-multiply, and key-switch (relin) — so it is the natural first end-to-end
  application. The hard `ct×ct` multiply (already done) is more than PIR requires.
- **A benchmark suite** — HE add / plain-mul / ct-mul / (later) rotate, reported as `kernels.md` /
  `ct_mul_perf.md` are today, to track the Track-A speedups.

---

## Suggested order

1. **A.1 (fuse twiddle)** + **A.2 (batch limbs)** — the biggest, most self-contained speed wins.
2. **B.1 (zero-host single program)** — closes the last host touches; also unlocks a clean perf loop.
3. **C.1 (PIR demo)** — a compelling application on top of the finished add/mul/key-switch.
4. **B.3 (depth ≥ 2)** + **B.4 (rotations)** — capability growth for real circuits.

Each item is independent of the others and does not regress the completed add/mul evaluator.
