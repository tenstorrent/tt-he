# reference/pir — the OnionPIR golden model, its fixture generators, and the C++ client

Everything here runs on the **host**. Nothing in this directory is required to run the device
suite: the `.bin` fixtures are committed, so the tests need no Python at all (see `.gitignore` here
and `tools/gen_fixtures.sh`). The generators are a maintainer tool for regenerating them.

The design docs that used to sit here now live in [`docs/pir/`](../../docs/pir/) —
`pir_primer_and_sizing.md`, `pir_special_kernels.md`, `pir_external_product_deepdive.md`, and
`history/onionpir_build_plan.md`. Start at [`docs/pir/pir_demo.md`](../../docs/pir/pir_demo.md) for
the protocol end to end.

## What is what

| | |
|---|---|
| **`onionpir_ref.py`** | **the golden model.** The whole OnionPIR v2 protocol in NumPy — keygen, query encryption, the Subs expansion tree, external product, fold, mod-switch, decrypt. Every `gen_*.py` imports it, and the docs cite it ~80 times. This file is the reason the directory exists. |
| `ntt2048_ref.py` | the n=2048 NTT/INTT reference the library ring is built on |
| `gen_*.py` (13) | one per device test: writes a `.bin` case file holding inputs **and** the expected output, so the device test is a comparison rather than a reimplementation. Driven by `tools/gen_fixtures.sh`. |
| `*.bin` (15) | those case files, **committed on purpose** |
| `client/` | the C++ client: `pir_client.hpp` (query build + response decrypt), `test_client.cpp` (host self-tests, a `nohw` ctest test), `encode_db.cpp` (builds the 65 536-row database) |
| `analyze_pir.py`, `check_scan_gemm.py` | analysis, not part of any test — `check_scan_gemm.py` is the host-side proof that the batched-GEMM scan equals the per-slot dot product |

## Why the generators are flat and not in a `gen/` subfolder

Twelve of the thirteen do `import onionpir_ref as ref` — a plain sibling import. Moving them into a
subfolder would mean either dragging the golden model in with them (misfiling the model as a
generator, and rewriting ~80 doc references) or adding `sys.path` juggling to twelve files. The flat
layout is what makes the sibling import work, so it stays.

## Regenerating

```sh
sh tools/gen_fixtures.sh          # every fixture; needs numpy
```

Deterministic: a clean `git status` afterwards is the check. Five artifacts are **not** committed
because of their size — `db65536.fold` (6.1 GB), `db65536.pir` (1.6 GB), `db65536.txt`,
`query65536.bin` and `query_small.bin`. `.gitignore` covers them, `encode_db` and `gen_query65536.py` build them, and
the four tests that need them carry the ctest label `fixture` so the default suite skips them.
