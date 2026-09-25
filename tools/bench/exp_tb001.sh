#!/bin/sh
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# bench_exp_tb001.sh — build + run exp_Tb001 (host/experiments/exp_tb001.cpp) in its fastest TRACED-RESIDENT
# mode and report the metrics per multiply, plus the device-trace breakdown.
#
# exp_Tb001 = the CROSS-aligned BFV Set-B multiply (N=8192, L=8, dnum=2) with EVALUATION form as the
# default ciphertext I/O. Same algorithm and parameters as host/experiments/legacy/he_mul_cross.cpp; the difference is
# that the multiply consumes eval-form ciphertexts rather than converting coeff->eval on device on
# every call, which removes 32 of the 122 transforms. See docs/he/exp_tb001.md.
#
# Usage:   sh tools/bench/exp_tb001.sh [seed]      (env: HE_ITERS=N, default 20)
#
#   TB001_COEFF_IN=1   fall back to coeff-form input (he_mul_cross's behaviour) for an A/B in the
#                      same binary, same card, everything else identical
#   TB001_KLAT=1       host-side kernel latency per kernel (perturbing: a Finish per dispatch)
#   TB001_LABELS=PATH  write per-dispatch HE-poly-op labels, to join with tools/traces/dispatch_csv.py
#   TB001_OP=add       measure a ct_add instead of the ct_mul, same config and same metrics
#                      (16 dispatches, 0.047 ms critical path busy -- docs/he/exp_tb001.md)
#   TB001_ACC=1|2|4    partials per contraction dispatch (default 8 = fused); 1 = one per dispatch
#
# NOTE: needs the device FREE (one device program at a time; a concurrent job wedges it -> `tt-smi -r`).
#
# POSIX sh, like every other script here -- /bin/sh is dash on Debian/Ubuntu and `set -o pipefail`
# is a bashism that would abort this on line 1. CI's `dash -n` is a syntax check and cannot catch it.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"                         # tools/bench/
REPO="$(cd "$HERE/../.." && pwd)"                                # repo root: CMakeLists + build/
BUILD="$REPO/build"
TRACES="$REPO/tools/traces"                                   # the analyzers (tracked)
OUT="$REPO/generated/traces"                                  # rendered output (gitignored)
mkdir -p "$OUT"

# shellcheck disable=SC1091
. "$REPO/env.sh" || exit 1

SEED="${1:-0}"
export HE_ITERS="${HE_ITERS:-20}"

# THE WORKLOAD. G is the program core number and it is FIXED at 8 for this experiment
# (docs/he/exp_tb001.md "Workload"). It is not a tuning knob: changing it changes the workload, so a
# number produced with a different G is not comparable with anything else in that doc.
export NTT_CORES="${NTT_CORES:-8}"

CSV="$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
LOG="$(mktemp)"

echo "== [1/3] build exp_tb001 =="
if [ ! -d "$BUILD" ]; then
    cmake -S "$REPO" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release >/dev/null
fi
cmake --build "$BUILD" --target exp_tb001 -j || { echo "BUILD FAILED"; exit 1; }

echo "== [2/3] run TRACED RESIDENT (profiler on, seed=$SEED, iters=$HE_ITERS, G=$NTT_CORES) =="
TT_METAL_DEVICE_PROFILER=1 HE_TRACE2=1 HE_PROF2=1 \
    "$BUILD/exp_tb001" "$SEED" 2>/dev/null | tee "$LOG"

echo
echo "== [3/3] device-trace breakdown (single traced replay) =="
if [ -f "$CSV" ]; then
    python3 "$TRACES/trace_hemul.py" "$CSV" "$OUT/exp_tb001_best" 2>/dev/null \
        | grep -iE "span|device|total|NTT|pointwise|FBC|%" | head -20 || true
    echo "  (full trace: $OUT/exp_tb001_best.html)"
else
    echo "  (no profiler CSV at $CSV — skipping breakdown)"
fi

echo
echo "===== exp_Tb001 — CROSS Set-B, eval-form I/O — G=$NTT_CORES, B=1, S=1 (N=8192, L=8, dnum=2) ====="
grep -E "ciphertext I/O domain" "$LOG"      || echo "  (input-domain line not found)"
grep -E "device-vs-oracle" "$LOG" | tail -1 || echo "  (correctness line not found)"
grep -E "host wall|DEVICE-BUSY span" "$LOG" || echo "  (timing lines not found — did the run reach TRACED RESIDENT?)"
grep -E "HE_PROF2: 1-1 check" "$LOG"        || true
echo "==========================================================================================="
rm -f "$LOG"
