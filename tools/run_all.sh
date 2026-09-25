#!/bin/sh
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# run_all.sh — run the device suite and print one line per test.
#
#   sh tools/run_all.sh              # the correctness gate: host/tests/, skipping absent fixtures
#   sh tools/run_all.sh --all        # include the tests needing a .bin case file the repo omits
#   sh tools/run_all.sh --perf       # THE benchmark: ckks_ctmul (host/benchmarks/)
#   sh tools/run_all.sh --study      # the exp_* studies + legacy/ programs -- slow
#   sh tools/run_all.sh --everything # hw + perf + study
#   sh tools/run_all.sh --list       # just list what the gate would run
#
# Wraps ctest so the labels do the selection (see cmake/HeDeviceTarget.cmake). One label per
# host/ subfolder, so the folder a program lives in decides when it runs:
#   hw       correctness tests      host/tests/       must always pass -- this is the gate
#   perf     the benchmark          host/benchmarks/  ckks_ctmul, the number the README leads with
#   study    finished experiments   host/experiments/ conclusions in docs/; legacy/ = pre-exp_ programs
#   nohw     host-only tests (no device needed)
#   fixture  additionally needs a GENERATED artifact the repo does not ship (db65536.pir,
#            db65536.fold, query_small.bin) -- four PIR tests. Ordinary case files are tracked.
# NOTE: ctest -L takes a REGEX, so the selectors are anchored (^hw$) -- a bare "hw" also matches "nohw".
#
# WHY THIS RESETS THE DEVICE: a hung program leaves the card wedged, and then the NEXT program hangs
# too even though nothing is wrong with it -- one bad test would otherwise report ~40 false failures.
# `tt-smi -r` returns while the card is still re-initializing, so we let it settle before continuing;
# without that pause the reset itself manufactures spurious exit-2 / SIGABRT results.
set -u   # no `pipefail`: must run under POSIX sh (dash)

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$REPO/build"
[ -f "$BUILD/CTestTestfile.cmake" ] || {
    echo "no configured build at $BUILD"
    echo "  . env.sh && cmake -B build -G Ninja && cmake --build build -j"
    exit 1
}

SEL="-L ^hw$ -LE ^fixture$"
case "${1:-}" in
    --all)        SEL="-L ^hw$" ;;
    --perf)       SEL="-L ^perf$" ;;
    --study)      SEL="-L ^study$" ;;
    --everything) SEL="-L ^(hw|perf|study)$" ;;
    --list) ( cd "$BUILD" && ctest -N -L "^hw$" ); exit 0 ;;
    "")     ;;
    *)      sed -n '2,18p' "$0"; exit 2 ;;
esac

# shellcheck disable=SC1091
. "$REPO/env.sh" || exit 1

reset_device() {
    command -v tt-smi >/dev/null 2>&1 || return 0
    tt-smi -r >/dev/null 2>&1
    sleep 3          # the reset returns before the card is ready; see the note above
}

# ctest runs each test itself; we reset before the run and after any failure, then re-run nothing --
# a failure that was caused by a wedged card shows up as a second failure, which is the signal we want.
reset_device

# shellcheck disable=SC2086
( cd "$BUILD" && ctest $SEL --output-on-failure --timeout 300 )
rc=$?

if [ "$rc" != "0" ]; then
    echo
    echo "Some tests failed. Before believing the results, note:"
    echo "  * a wedged card makes every subsequent test fail -- reset and re-run: tt-smi -r"
    echo "  * the 19 'fixture' tests need .bin case files the repo does not ship (run without --all)"
    reset_device
fi
exit $rc
