#!/bin/sh
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# baseline_sweep.sh — build everything, run every binary, record one line per target.
#
#   sh tools/bench/baseline_sweep.sh before.txt      # snapshot now
#   ...make changes...
#   sh tools/bench/baseline_sweep.sh after.txt       # snapshot again
#   diff <(sed -E 's/exit=[0-9]+ +//' before.txt | sort) \
#        <(sed -E 's/exit=[0-9]+ +//' after.txt  | sort)
#
# WHY: `cmake --build` proves nothing about correctness. The failure mode that matters in this repo is
# a binary that still builds and still exits 0 but silently stops validating. Diffing the recorded
# verdict line per target catches that; a green build does not. This harness caught three real bugs
# during the 2026-07 restructure (a kernel deadlock in kernels/ir, a hardcoded spec path, and 19
# ctest labels that were silently dropped).
#
# DEVICE DISCIPLINE -- read this before running:
#   * ONE device program at a time. Do not run anything else on the card while this is going. Doing so
#     produces false failures (observed: SIGBUS, exit 2) that are indistinguishable from real ones
#     until you re-run each binary alone.
#   * A hung run wedges the card, and then the NEXT binary hangs too. This resets after any non-zero
#     exit for that reason.
#   * `tt-smi -r` returns before the card is ready, so we pause after resetting.
#
# Takes roughly an hour for the full set. The per-binary timeout is 400s because he_mul_setb
# legitimately needs more than 120s while still passing.
#
# Expected result: all binaries pass. Test fixtures are committed, so no Python is needed to run this.
set -u   # no `pipefail`: must run under POSIX sh (dash)

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1
# shellcheck disable=SC1091
. ./env.sh >/dev/null 2>&1 || { echo "env.sh failed -- run: sh tools/doctor.sh"; exit 1; }

OUT="${1:-baseline.txt}"
: > "$OUT"

echo "== building all targets ==" | tee -a "$OUT"
cmake --build build -j"$(nproc)" > /tmp/he_baseline_build.log 2>&1
echo "build exit=$? ; targets built=$(grep -c 'Built target' /tmp/he_baseline_build.log)" | tee -a "$OUT"
echo | tee -a "$OUT"

echo "== per-binary verdicts ==" | tee -a "$OUT"
reset_device() { command -v tt-smi >/dev/null 2>&1 && { tt-smi -r >/dev/null 2>&1; sleep 3; }; }
reset_device
prev_rc=0
for exe in $(find build -maxdepth 1 -type f -executable | sort); do
    name=$(basename "$exe")
    [ "$prev_rc" -ne 0 ] && reset_device      # a wedged card would fail the next binary too
    # Some targets share a source and differ only by compile-time defines, so they cannot all use the
    # same default fixture; CMakeLists names the right one via ARGS. Read it from there rather than
    # duplicating the mapping here.
    targs=$(awk -v t="he_device_target($name" 'index($0,t){f=1} f&&/ARGS /{sub(/.*ARGS /,"");sub(/ .*/,"");print;exit} f&&/\)/{exit}' CMakeLists.txt)
    # shellcheck disable=SC2086
    out=$(timeout 400 "./$exe" $targs 2>&1); rc=$?   # run from the repo root: HE_RUN_DIR is the root
    prev_rc=$rc
    verdict=$(printf '%s' "$out" | grep -oE "mismatches[^,]{0,18}= ?[0-9]+/[0-9]+|mismatches=[0-9]+/[0-9]+|device-vs-oracle = [0-9]+/[0-9]+|OK: all [0-9]+ [a-z]+ match|RESULT:[^\\n]{0,60}|Test Passed|Test Failed|PASS|FAIL" | head -2 | tr '\n' ' ')
    printf '%-28s exit=%-4s %s\n' "$name" "$rc" "${verdict:-<no verdict line>}" | tee -a "$OUT"
done
echo | tee -a "$OUT"
echo "== done ==" | tee -a "$OUT"
