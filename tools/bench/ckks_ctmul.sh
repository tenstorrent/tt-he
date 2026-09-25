#!/bin/sh
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# bench_ckks_ctmul.sh — build + run THE benchmark: the best CKKS ct_mul this repo has.
#
#   sh tools/bench/ckks_ctmul.sh
#
# 23.991 ms/op, bit-exact 0/6029312, at N=65536, L_Q=48, dnum=3, G=130 on a Blackhole p150.
# Owned by docs/he/exp_tg018ntt.md 4e; the ladder that got there is 50.863 -> 43.932 -> 32.448 ->
# 23.991 (corner turn on-chip, then the whole NTT in ONE dispatch, then base conversion moved off
# the SFPU onto the matrix engine). 2884 dispatches -> 1278, host wall 155.1 -> 42.0 ms.
#
# WHY THE ENV IS PINNED HERE. host/benchmarks/ckks_ctmul.cpp is a controlled multi-arm experiment --
# every rung of that ladder is the same binary with a different switch, which is what makes the
# comparison trustworthy. So the arm is not baked into the source; the benchmark is the statement of
# WHICH arm is the published one, and it lives here and in the ckks_ctmul ctest ENVIRONMENT. Run the
# other rungs through the exp_tg018ntt target instead (same binary, docs/he/exp_tg018ntt.md 6).
#
# NOTE: needs the device FREE -- one device program at a time, and a concurrent job wedges the card
# (then `tt-smi -r`). This one is heavy: N=65536 at L_Q=48 JITs a lot of kernels on a cold cache.
#
# POSIX sh, like every other script here: the docs say `sh tools/...`, /bin/sh is dash on
# Debian/Ubuntu, and `set -o pipefail` is a bashism that would abort it at line 1.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"                         # tools/bench/
REPO="$(cd "$HERE/../.." && pwd)"
BUILD="$REPO/build"

. "$REPO/env.sh" || exit 1

# The published arm (docs/he/exp_tg018ntt.md 4e). Overridable so this script can also serve a
# comparison run, but the DEFAULT is the number the README quotes.
export TG18NTT_DEV="${TG18NTT_DEV:-1}"
export TG18NTT_TRACE="${TG18NTT_TRACE:-1}"
export TG18NTT_FUSE="${TG18NTT_FUSE:-1}"
export TG18NTT_FBC="${TG18NTT_FBC:-fpu}"

echo "== building ckks_ctmul =="
cmake --build "$BUILD" --target ckks_ctmul || exit 1

echo "== TG18NTT_FUSE=$TG18NTT_FUSE TG18NTT_FBC=$TG18NTT_FBC (expect ~23.99 ms/op, 0/6029312) =="
"$BUILD/ckks_ctmul" "$@"
