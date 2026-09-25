#!/bin/sh
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# bench_hemul_cross.sh — build + run the CROSS Set-B HE-multiply (host/experiments/legacy/he_mul_cross.cpp) in its
# fastest TRACED-RESIDENT mode and report the current best HOST WALL and DEVICE-BUSY span per multiply.
#
#   TRACED RESIDENT = buffer-resident intermediates + Metal Trace capture/replay (REC->CAP->REPLAY).
#   he_mul_cross prints exactly:
#       host wall        : ms/mul   (end-to-end host, non-blocking replays)
#       DEVICE-BUSY span : ms/mul   (sum of blocking segment-replay walls = end-to-end device latency)
#   HE_PROF2 additionally dumps a clean single-replay device trace, which trace_hemul.py breaks down.
#
# Usage:   ./bench_hemul_cross.sh [seed]     (env: HE_ITERS=N  overrides replay-iters, default 20)
# NOTE: needs the device FREE (one device program at a time; a concurrent job wedges it -> `tt-smi -r`).
#
# Other modes of he_mul_cross (docs/he/cross_comparison.md §9 owns the full flag list):
#   NTT_CORES=4|8|16|64            cores per ct_mul
#   HE_PROF_DRAIN=1 HE_DRAIN_K=100 one FULL trace with no dropped markers (periodic profiler drain);
#                                  HE_PROF2 below drops markers when the device buffer overflows
#
# POSIX sh, like every other script here: the docs say `sh tools/bench/hemul_cross.sh`, /bin/sh is dash
# on Debian/Ubuntu, and `set -o pipefail` is a bashism that aborts it at line 1. CI's `dash -n` is a
# SYNTAX check, so it cannot catch that -- it has to stay POSIX by construction.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"                         # tools/bench/
REPO="$(cd "$HERE/../.." && pwd)"                                # repo root: CMakeLists + build/
BUILD="$REPO/build"
TRACES="$REPO/tools/traces"                                   # the analyzers (tracked)
OUT="$REPO/generated/traces"                                  # rendered output (gitignored)
mkdir -p "$OUT"

# --- tt-metalium env (resolved once, repo-wide: see <repo>/env.sh) ---
. "$REPO/env.sh" || exit 1

SEED="${1:-0}"
export HE_ITERS="${HE_ITERS:-20}"
# Tensix cores per ONE ct_mul (this program computes a single multiply; it does not batch several).
# The matmul waves are chunked by this, so more cores = fewer dispatches, and the workload is
# dispatch-bound. Measured on p150, optimized build, device-busy / dispatches (HE_TRACE2, HE_ITERS=10):
#    4 -> 41.62 ms / 3236      8 -> 22.90 ms / 1618     16 -> 17.68 ms / 1130
#   32 -> 15.16 ms /  886     64 -> 14.10 ms /  764
# 64 is the optimum. NOTE: it was 32 until the fused reduce+twiddle landed (64 measured 18.06 ms
# before it) -- the fusion removed a dispatch whose cost scaled badly with core count, which moved
# the sweet spot. Re-measure this sweep after any change to the dispatch structure.
export NTT_CORES="${NTT_CORES:-64}"
CSV="$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
LOG="$(mktemp)"

echo "== [1/3] build he_mul_cross =="
if [ ! -d "$BUILD" ]; then
    cmake -S "$REPO" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release >/dev/null
fi
cmake --build "$BUILD" --target he_mul_cross -j || { echo "BUILD FAILED"; exit 1; }

echo "== [2/3] run TRACED RESIDENT (profiler on, seed=$SEED, iters=$HE_ITERS) =="
TT_METAL_DEVICE_PROFILER=1 HE_TRACE2=1 HE_PROF2=1 \
    "$BUILD/he_mul_cross" "$SEED" 2>/dev/null | tee "$LOG"

echo
echo "== [3/3] device-trace breakdown (single traced replay) =="
if [ -f "$CSV" ]; then
    python3 "$TRACES/trace_hemul.py" "$CSV" "$OUT/hemul_cross_best" 2>/dev/null \
        | grep -iE "span|device|total|NTT|pointwise|FBC|%" | head -20 || true
    echo "  (full trace: $OUT/hemul_cross_best.html)"
else
    echo "  (no profiler CSV at $CSV — skipping breakdown)"
fi

echo
echo "================= BEST ct_mul — CROSS Set-B (N=8192, L=8, dnum=2, ${NTT_CORES} cores) ================="
grep -E "device-vs-oracle" "$LOG" | tail -1 || echo "  (correctness line not found)"
grep -E "host wall|DEVICE-BUSY span" "$LOG"   || echo "  (timing lines not found — did the run reach TRACED RESIDENT?)"
echo "============================================================================================"
rm -f "$LOG"
