#!/bin/sh
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# exp_noc001 sweep -- three a2a routes over two scaling axes, plus the barrier probe.
# Every cell is gated: the three real routes MUST report mism 0, the probe MUST report mism != 0.
# The script aborts on a missing profiler CSV rather than letting an absent capture sum to 0.00,
# which has silently produced empty tables three times in this repo (exp_ntt001.md 5).
#   sh tools/bench/exp_noc001.sh [out.csv]
set -e
cd "$(dirname "$0")/../.."
. ./env.sh >/dev/null 2>&1
CSV="$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
OUT="${1:-docs/noc/data/noc001.csv}"
LOG=$(mktemp)
echo "G,P,block_B,route,mism,a2a_us,a2a_put_us,a2a_sync_us,a2a_ld_us,a2a_wb_us,kernel_us" > "$OUT"
for cell in "4 1" "8 1" "16 1" "32 1" "64 1" "32 2" "32 4"; do
  set -- $cell; G=$1; P=$2
  for R in bulk strided noc syncprobe; do
    rm -f "$CSV"
    env TT_METAL_DEVICE_PROFILER=1 NOC1_PROF=1 NOC1_G=$G NOC1_P=$P NOC1_ROUTE=$R \
        ./build/exp_noc001 > "$LOG" 2>&1 || true   # the probe exits 1 BY DESIGN: gate below decides
    M=$(grep -oE "mism +[0-9]+" "$LOG" | head -1 | awk '{print $2}')
    B=$(grep -oE "block [0-9]+ B" "$LOG" | head -1 | awk '{print $2}')
    [ -s "$CSV" ] || { echo "G=$G P=$P $R: no profiler CSV"; exit 1; }
    [ -n "$M" ]   || { echo "G=$G P=$P $R: no mism line"; tail -20 "$LOG"; exit 1; }
    if [ "$R" = syncprobe ]; then
      [ "$M" != 0 ] || { echo "G=$G P=$P: probe PASSED its gate -- it is not proving itself"; exit 1; }
    else
      [ "$M" = 0 ]  || { echo "G=$G P=$P $R: mism $M"; exit 1; }
    fi
    Z=$(python3 tools/traces/zone_phases.py "$CSV")
    g() { echo "$Z" | awk -v k="$1" '$1==k{print $3; f=1} END{if(!f) print 0}'; }
    echo "$G,$P,$B,$R,$M,$(g a2a),$(g a2a_put),$(g a2a_sync),$(g a2a_ld),$(g a2a_wb),$(g NCRISC-KERNEL)" >> "$OUT"
  done
done
rm -f "$LOG"
column -s, -t "$OUT"
