# Contributing to tt-he

Thanks for your interest. This project builds homomorphic encryption (BFV) and
PIR primitives on Tenstorrent Tensix cores, with every kernel validated
bit-exact against a CPU/Python golden. That validation rule is the thing this
project cares about most, and it shapes most of what follows.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs

Open a **GitHub Issue**. The most useful reports include:

- what you ran (the exact command — `./build/vector_modmul`, `sh tests/run_all.sh`, …),
- your hardware (Blackhole p150, Wormhole n150) and your tt-metalium version,
- the output, including the mismatch count if a validation check failed,
- what you expected instead.

Run `sh tools/doctor.sh` first and include its output — it checks every
prerequisite and most reports turn out to be environment problems it can name.

If a run hangs, the device is often wedged from an earlier crashed run. Try
`tt-smi -r` and retry before filing.

**Security vulnerabilities do not go in Issues** — see [SECURITY.md](SECURITY.md).

## Proposing changes

Bug fixes and new functionality are submitted as **Pull Requests** against
`main`. For anything large — a new subsystem, a change to the kernel/host
contract, a different algorithm — open an Issue first so the approach can be
discussed before you spend time on it.

**Review cadence is weekly.** Maintainers review open PRs at least once a week.
If a PR sits longer than that without a response, please ping it.

## Requirements for a pull request

### 1. Correctness is proved against a golden, not asserted

Every kernel here is validated bit-exact against a CPU or Python reference. A
change to arithmetic, a kernel, or the compiler needs the same:

- a golden to compare against — either an existing model in `reference/` or a
  new one you add there,
- a test that reports a mismatch count, in the subsystem directory it belongs to
  under `host/tests/` (`vector/ matmul/ ntt/ baseconv/ he/ pir/ ir/ common/`),
- the mismatch count in the PR description, in the repo's usual form: `0/8192`.

A change that improves performance must show the validation still passes. "It's
faster" is not a result on its own; "it's faster and still `0/114688`" is.

### 2. Tests pass

```bash
sh tests/run_all.sh                         # device suite
ctest --test-dir build -L '^nohw$'          # host-only, no hardware needed
```

If you have no Tenstorrent hardware, run the host-only tests and say so in the
PR — a maintainer will run the device suite. CI runs the hardware-free checks
(build, host tests, shell script and markdown link checks, SPDX headers) on
every push; the device suite is a self-hosted opt-in job.

For a performance change, `sh tools/bench/baseline_sweep.sh out.txt` (~1 hour, needs
the card free) diffed against `.baselines/baseline_all_green.txt` is the
full-repo check.

### 3. Every source file carries an SPDX header

New `.cpp`, `.hpp`, `.py`, `.sh` and `.cmake` files start with:

```
// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
```

(`#` instead of `//` for Python, shell and CMake.) CI fails without it. By
contributing you agree your contribution is licensed under Apache 2.0, and
documentation under CC BY 4.0 — see [LICENSE](LICENSE) and
[LICENSE-DOCS](LICENSE-DOCS).

**Do not add third-party source to this repository** without raising it in an
Issue first. Cite papers and reference implementations in prose instead; if code
genuinely must be vendored, it needs its own license header and an entry in
[NOTICE](NOTICE).

### 4. Shell scripts are POSIX `sh`

Scripts are invoked as `sh tools/...` in the docs, and `/bin/sh` is dash on
Debian and Ubuntu — so a bashism breaks the documented command for everyone. CI
checks this with `dash -n`. It has caught real breakage (`set -o pipefail`).

### 5. Documentation keeps up

- Docs live in `docs/`, one directory per subsystem, indexed by
  [`docs/README.md`](docs/README.md). Read it for the conventions: content files
  are `lower_snake_case.md`, each carries a `Status:` header, and superseded
  material moves to the topic's `history/` rather than being deleted.
- **One owner per number.** A measurement lives in exactly one doc; everything
  else links to it. Two copies drift apart, and here they already have.
- **State the hardware and the code state** next to any performance figure.
  Wormhole n150 numbers and Blackhole p150 numbers are not comparable.
- The three repo checks must pass: `ctest -L nohw` runs `tools/check/links.sh`
  (markdown links resolve), `tools/check/paths.sh` (no source names a path that does not
  exist, `host/` holds only C++, every ctest label matches its folder) and
  `tools/check/divergence.sh` (no undocumented drift between an experiment's private kernels
  and `kernels/shared/`).
- There is no `CHANGELOG.md`. The write-up belongs in the subsystem doc that owns
  the number — `docs/he/`, `docs/matmul/`, and so on — and the commit message carries
  the rest. **Record the null results and the retractions too**; in this repo those
  have repeatedly been the most useful part.

## Commit messages

A short subject naming the area, then a body explaining *why*. Where a commit
carries a result, put it in the subject — the history reads as a log of what was
learned, and it is more useful that way:

```
exp_Tg018m: every tile was on DRAM bank 0 -- fixing it is 1.79x on the whole multiply
docs(dram_guide): promote the single-page-buffer trap to section 0
```

## Setting up

See the [README](README.md) for prerequisites and the quick start. The short
version:

```bash
sh tools/setup.sh        # finds your tt-metalium install, writes config.local.env
. env.sh                 # exports the required env vars
cmake -B build -G Ninja && cmake --build build -j
```

`config.local.env` is machine-specific and gitignored. Never commit absolute
paths, and never commit anything from `build/` or `generated/`.
