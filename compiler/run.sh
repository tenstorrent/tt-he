#!/bin/sh
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# run.sh — one-shot HE compiler run. Edit the WORKLOAD constants below, then:  sh run.sh
# It sets the tt-metalium env, then compiles -> schedules -> generates code -> runs on the n150
# -> prints FINAL PERFORMANCE.  (One TT device job at a time; if it wedges: tt-smi -r.)

# ============================ EDIT THESE ============================
GROUPS=64            # workload: independent pointwise groups (s = w*x + y*z)
CORES=32            # hardware: Tensix cores to use (64 = full grid)
PRIME=251658241      # 28-bit modulus
N=1024              # elements per tile (leave 1024)
# ===================================================================

# --- run (from this script's own directory, so it works no matter where you call it) ---
DIR=$(cd "$(dirname "$0")" && pwd)

# --- tt-metalium environment (resolved once, repo-wide: see <repo>/env.sh) ---
. "$DIR/../env.sh" || exit 1
python3 "$DIR/run_he.py" --groups "$GROUPS" --cores "$CORES" --prime "$PRIME" --N "$N" 2>/dev/null
