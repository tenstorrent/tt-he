#!/bin/sh
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# install_ttmetalium.sh — turn a tt-metal SOURCE BUILD into an install prefix this repo can consume.
#
#   sh tools/install_ttmetalium.sh -s ~/work/tt-metal                       # -> $HOME/work/ttm-install
#   sh tools/install_ttmetalium.sh -s ~/work/tt-metal -p /usr/local -S      # -S = sudo (system-wide)
#
# WHY THIS EXISTS.  `cmake --install` on a tt-metal build tree does not just work, for two
# non-obvious reasons found while porting this repo to a new VM (2026-07-28):
#
#   1. Several CPM deps and tools are EXCLUDE_FROM_ALL but still carry install() rules, so a
#      normal `ninja` never builds them and the install dies partway with
#      "file INSTALL cannot find .../libfmt.so.11.1.4".  Beware: `ninja -n` reports only the
#      CMake-regen step here, so it looks like the build is complete when it is not.
#      We build them explicitly: see EXTRA_TARGETS.
#   2. A full-tree install also pulls test/benchmark deps (gtest, benchmark) that are likewise
#      unbuilt.  We install only the metalium COMPONENTS instead, which skips them.
#
# The result satisfies env.sh: host/lib/cmake/tt-metalium/tt-metalium-config.cmake + libexec/tt-metalium.
set -u   # NB: no `pipefail` -- these scripts must run under POSIX sh (dash)

SRC=""; BUILD=""; PREFIX="$HOME/work/ttm-install"; SUDO=""
while getopts "s:b:p:Sh" o; do case "$o" in
    s) SRC="$OPTARG" ;;
    b) BUILD="$OPTARG" ;;
    p) PREFIX="$OPTARG" ;;
    S) SUDO="sudo" ;;
    h) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "see -h" >&2; exit 2 ;;
esac; done

[ -n "$SRC" ] || { echo "need -s <tt-metal source dir>  (see -h)" >&2; exit 2; }
SRC="$(cd "$SRC" && pwd)" || exit 2
[ -n "$BUILD" ] || BUILD="$SRC/build_Release"
[ -f "$BUILD/CMakeCache.txt" ] || { echo "no CMake build at $BUILD — configure+build tt-metal first" >&2; exit 1; }

# Targets with install() rules that are excluded from the default 'all' target.
#   fmt                    -> libfmt.so.N (metalium links it)
#   generate_rank_bindings -> tools/scaleout binary
#   precompile_fw          -> the tool, and
#   precompile-fw          -> the custom command that GENERATES tt_metal/pre-compiled/
EXTRA_TARGETS="fmt generate_rank_bindings precompile_fw precompile-fw"

# Only the metalium side; deliberately excludes ttnn-* and test/benchmark components.
COMPONENTS="fmt-core spdlog-dev tt-logger-dev json-dev enchantum umd-runtime umd-dev jit-build metalium-runtime metalium-dev"

echo "== [1/3] build tt-metal (default target) =="
cmake --build "$BUILD" -j"$(nproc)" || { echo "BUILD FAILED"; exit 1; }

echo "== [2/3] build install-only targets ($EXTRA_TARGETS) =="
# shellcheck disable=SC2086
cmake --build "$BUILD" --target $EXTRA_TARGETS -j"$(nproc)" || { echo "BUILD FAILED (extra targets)"; exit 1; }

echo "== [3/3] install components -> $PREFIX =="
rc=0
for c in $COMPONENTS; do
    if out=$($SUDO cmake --install "$BUILD" --prefix "$PREFIX" --component "$c" 2>&1); then
        printf '  ok    %s\n' "$c"
    else
        printf '  FAIL  %s\n' "$c"
        printf '%s\n' "$out" | grep -iE "error|cannot find" | head -3 | sed 's/^/        /'
        rc=1
    fi
done

echo
if [ -f "$PREFIX/lib/cmake/tt-metalium/tt-metalium-config.cmake" ] && [ -d "$PREFIX/libexec/tt-metalium" ]; then
    echo "OK: usable prefix at $PREFIX"
    case "$PREFIX" in
        /usr/local|"$HOME/work/ttm-install") echo "   env.sh finds this automatically." ;;
        *) echo "   export TT_METALIUM_PREFIX=$PREFIX   (so env.sh finds it)" ;;
    esac
else
    echo "INCOMPLETE: $PREFIX lacks host/lib/cmake/tt-metalium/tt-metalium-config.cmake and/or libexec/tt-metalium"
    rc=1
fi
exit $rc
