#!/bin/sh
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# check_divergence.sh — fail if a private experiment kernel has drifted from its shared original
# without that being written down.
#
#   sh tools/check/divergence.sh
#
# Needs no hardware and no SDK. kernels/experiments/exp_tb001/ is a private copy of kernels/shared/
# so kernel experiments cannot regress the 76 binaries that share the validated tree. Its README
# states the invariant and the obligation:
#
#     "They were copied verbatim, so exp_Tb001's first run reproduces he_mul_cross's arithmetic
#      exactly. ... When you *do* diverge one, say so at the top of that file and in
#      docs/he/exp_tb001.md -- otherwise the next person cannot tell an intentional experiment
#      from a stale copy."
#
# That obligation was manual, so this makes it mechanical: any file whose CODE differs from the
# shared copy must be named in the experiment's README. Undocumented drift is the failure.
#
# COMMENTS ARE IGNORED, and that is the point rather than a shortcut. When an optimised kernel is
# promoted into the shared tree it gains a "promoted ... RETAINED and unchanged" header, so four
# files differ by that header alone while compiling to identical code. Comparing raw bytes reports
# them as divergences and trains everyone to ignore this check; comparing code does not.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

EXP="kernels/experiments/exp_tb001"
SHARED="kernels/shared"
DOC="$EXP/README.md"

[ -d "$EXP" ] && [ -d "$SHARED" ] && [ -f "$DOC" ] || {
    echo "check_divergence: expected $EXP, $SHARED and $DOC to exist"
    exit 1
}

# Strip // comments and blank lines: what is left is the code.
code() { sed 's|//.*||' "$1" | grep -v '^[[:space:]]*$'; }

same=0; newonly=0; commentonly=0; documented=0; bad=0
for rel in $(cd "$EXP" && find . -name '*.cpp' | sed 's|^\./||' | sort); do
    exp="$EXP/$rel"; shared="$SHARED/$rel"
    # No shared counterpart: a new kernel written for this experiment, not a divergence.
    [ -f "$shared" ] || { newonly=$((newonly + 1)); continue; }
    if cmp -s "$exp" "$shared"; then same=$((same + 1)); continue; fi
    a="${TMPDIR:-/tmp}/cd_a.$$"; b="${TMPDIR:-/tmp}/cd_b.$$"
    code "$exp" >"$a"; code "$shared" >"$b"
    if cmp -s "$a" "$b"; then
        commentonly=$((commentonly + 1))
    elif grep -q "$(basename "$rel")" "$DOC"; then
        documented=$((documented + 1))
    else
        echo "UNDOCUMENTED divergence: $rel"
        echo "    code differs from $shared but $(basename "$rel") is not named in $DOC"
        bad=$((bad + 1))
    fi
    rm -f "$a" "$b"
done

echo "check_divergence: $same identical, $commentonly comment-only, $newonly new-here-only, $documented documented divergences"
if [ "$bad" -gt 0 ]; then
    echo "check_divergence: $bad undocumented divergence(s) -- record each in $DOC (what changed, and why)"
    exit 1
fi
echo "check_divergence: OK"
exit 0
