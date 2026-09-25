#!/bin/sh
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# doctor.sh — check every prerequisite and, for each failure, print the command that fixes it.
#
#   sh tools/doctor.sh            # report
#   sh tools/doctor.sh --fix      # also perform the safe fixes (currently: device reset)
#
# Exit 0 = ready to build and run. Exit 1 = at least one BUILD-blocking problem.
# Notes and warnings never fail the run: Python packages and SDK-version drift are reported, not gated.
set -u   # NB: no `pipefail` -- these scripts must run under POSIX sh (dash)

REPO="$(cd "$(dirname "$0")/.." && pwd)"
FIX=0; [ "${1:-}" = "--fix" ] && FIX=1
fail=0; warn=0

say()  { printf '%s\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=$((fail+1)); }
note() { printf '  \033[33mnote\033[0m  %s\n' "$*"; warn=$((warn+1)); }
fixhint() { printf '        -> %s\n' "$*"; }

say "== 1. tt-metalium install (build + run) =="
if PREFIX_OUT=$(TTM_ENV_QUIET=1 . "$REPO/env.sh" 2>&1 && echo "$TT_METALIUM_PREFIX_RESOLVED"); then
    PREFIX=$(printf '%s' "$PREFIX_OUT" | tail -1)
    ok "prefix: $PREFIX"
    [ -f "$PREFIX/lib/cmake/tt-metalium/tt-metalium-config.cmake" ] \
        && ok "CMake package files present" \
        || { bad "no host/lib/cmake/tt-metalium/tt-metalium-config.cmake"; fixhint "sh tools/install_ttmetalium.sh -s /path/to/tt-metal"; }
    [ -d "$PREFIX/libexec/tt-metalium" ] \
        && ok "runtime tree present (kernels, sfpi, firmware)" \
        || { bad "no libexec/tt-metalium -- kernels cannot JIT-compile at runtime"; fixhint "sh tools/install_ttmetalium.sh -s /path/to/tt-metal"; }
else
    bad "no usable tt-metalium install found"
    printf '%s\n' "$PREFIX_OUT" | sed 's/^/        /'
    # Distinguish the most common mistake: pointing at a build tree instead of an install.
    for cand in "${TT_METALIUM_PREFIX:-}" "$HOME/work/tt-metal/build_Release" "$REPO/../tt-metal/build_Release"; do
        if [ -n "$cand" ] && [ -f "$cand/CMakeCache.txt" ]; then
            fixhint "$cand is a tt-metal BUILD TREE, not an install -- it can never satisfy find_package."
            fixhint "sh tools/install_ttmetalium.sh -s $(dirname "$cand")"
            break
        fi
    done
fi

say "== 2. tt-metalium version =="
EXPECTED=$(grep -s '^TT_METALIUM_VERSION_EXPECTED=' "$REPO/config.local.env" "$REPO/config.defaults.env" 2>/dev/null | head -1 | cut -d= -f2)
VERFILE="${PREFIX:-}/lib/cmake/tt-metalium/tt-metalium-config-version.cmake"
if [ -n "${PREFIX:-}" ] && [ -f "$VERFILE" ]; then
    INSTALLED=$(grep -oE 'set\(PACKAGE_VERSION "[^"]+"' "$VERFILE" | grep -oE '[0-9.]+')
    case "$INSTALLED" in
        "$EXPECTED"*) ok "installed $INSTALLED, expected $EXPECTED" ;;
        *) note "installed $INSTALLED, this repo was validated on $EXPECTED"
           fixhint "usually fine. If kernels fail to compile (moved headers / renamed APIs), this is why." ;;
    esac
else
    note "cannot read installed version"
fi

say "== 3. toolchain (build) =="
if command -v cmake >/dev/null 2>&1; then
    CMV=$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+'| head -1)
    awk -v v="$CMV" 'BEGIN{split(v,a,".");exit !(a[1]>3||(a[1]==3&&a[2]>=22))}' \
        && ok "cmake $CMV (>= 3.22)" || { bad "cmake $CMV is older than 3.22"; fixhint "install a newer cmake"; }
else
    bad "cmake not found"; fixhint "apt install cmake  (>= 3.22)"
fi
if echo 'int main(){return 0;}' | c++ -std=c++20 -x c++ - -o /tmp/.he_cxx_probe 2>/dev/null; then
    ok "c++ compiler accepts -std=c++20"; rm -f /tmp/.he_cxx_probe
else
    bad "c++ compiler does not support C++20"; fixhint "install g++ >= 10 or clang >= 12"
fi

say "== 4. Python goldens (not gated) =="
PY=$(grep -s '^PYTHON=' "$REPO/config.local.env" "$REPO/config.defaults.env" 2>/dev/null | head -1 | cut -d= -f2)
PY=${PY:-python3}
if command -v "$PY" >/dev/null 2>&1; then
    ok "$PY $("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
    "$PY" -c "import numpy" 2>/dev/null && ok "numpy importable" \
        || { note "numpy missing -- reference models/goldens will not run (device work is unaffected)"; fixhint "$PY -m pip install -r requirements.txt"; }
else
    note "$PY not found"; fixhint "install Python 3.10+, then $PY -m pip install -r requirements.txt"
fi

say "== 5. device (run only) =="
if [ -e /dev/tenstorrent/0 ]; then
    ok "/dev/tenstorrent/0 present"
    HOLDERS=$(command -v lsof >/dev/null 2>&1 && sudo -n lsof /dev/tenstorrent/0 2>/dev/null | awk 'NR>1{print $1"("$2")"}' | sort -u | tr '\n' ' ')
    if [ -n "${HOLDERS:-}" ]; then
        note "device is held by: $HOLDERS -- only one program may use it at a time"
        fixhint "close it, then: tt-smi -r"
    else
        ok "device is free"
    fi
    if [ "$FIX" = "1" ]; then
        # tt-smi -r returns while the card is still re-initializing ("Re-initializing boards after
        # reset..."), so a program started immediately after can fail spuriously (exit 2 / SIGABRT).
        # Observed twice on 2026-07-31. Give it a moment to settle.
        command -v tt-smi >/dev/null 2>&1 && { tt-smi -r >/dev/null 2>&1; sleep 3; ok "device reset (--fix)"; }
    fi
else
    note "no /dev/tenstorrent/0 -- builds fine, but nothing can run on device"
    fixhint "check the driver is loaded (tt-smi)"
fi

say ""
if [ "$fail" -gt 0 ]; then
    say "NOT READY: $fail blocking problem(s), $warn note(s). Fix the FAIL lines above."
    exit 1
fi
say "READY to build and run. ($warn note(s) -- see above; none block the build.)"
say "If a device run hangs, it is usually a wedged card from an earlier crash: tt-smi -r"
exit 0
