# reference/he — the BFV/CKKS CPU goldens and the device test cases

Everything here runs on the **host**, in NumPy/Python big-int arithmetic. It is the ground truth the
device is checked against: the device suite needs none of it at runtime, because the `.bin` case
files are committed (`tools/gen_fixtures.sh` regenerates them).

Design docs live in [`docs/he/`](../../docs/he/) — start at
[`docs/he/README.md`](../../docs/he/README.md) for the evaluator, and
[`docs/he/bfv_build_plan.md`](../../docs/he/bfv_build_plan.md) for the primitive-by-primitive
derivation the kernels cite.

## The goldens

| | |
|---|---|
| **`bfv_mul_ref.py`** | **the ground truth**: full BFV multiply in exact big-int arithmetic — encrypt → tensor → rescale → relinearize → decrypt. When a device result is called "bit-exact", this is what it matched. |
| `ckks_mul_ref.py` | the CKKS `ct_mul` oracle, coefficient domain — the counterpart for `exp_tg000` and the `ckks_ctmul` benchmark |
| `bfv_relin_ref.py` | relinearization / key-switch, RNS |
| `bfv_relin_dnum_ref.py`, `bfv_relin_dnum.py` | the same with `dnum > 1` digit decomposition (hybrid key-switch) |
| `base_convert.py` | RNS fast base conversion — the primitive under mod-up / mod-down |
| `bfv_bconv_crns.py` | the CRNS (Cox–Rower) exact base conversion, an `m_sk`-free alternative. Runnable: it prints `RESULT: ALL VERIFIED` after a u-sweep and an end-to-end cross-check against the `m_sk` path. |

## The case files

`gen_ctmul_case.py` and `gen_rescale_case.py` write a `.bin` holding inputs **and** the expected
output, so `host/tests/he/bfv_ct_mul.cpp` and `bfv_rescale.cpp` are comparisons rather than
reimplementations. `_1`, `_7`, `_42` are other seeds; the unsuffixed file is seed 0.

`tb001b_case.bin` and `tg000_case.bin` are **not committed** — 5.8 MB between them, and the only
things that read them are `exp_tb001b` and `exp_tg000`, which are labelled `study` (so neither the
correctness gate nor CI ever opens them) and which write their own case file on request:

```sh
TB001B_GEN=reference/he/tb001b_case.bin ./build/exp_tb001b 0
TG000_GEN=reference/he/tg000_case.bin   ./build/exp_tg000   0
```

Both targets therefore also carry the `fixture` label, which means exactly this: needs a generated
artifact the repo does not ship. That is why they have no `gen_*.py` here — the experiment binary is
the generator.

## Regenerating

```sh
sh tools/gen_fixtures.sh          # every fixture; needs numpy
```

The generators write straight into this directory, so there is no move step and a clean `git status`
afterwards is the check.
