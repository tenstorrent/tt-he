#!/bin/sh
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# check_paths.sh — fail if a source, script or CMake line names a repo path that does not exist.
#
#   sh tools/check/paths.sh
#
# Needs no hardware and no SDK. It exists because this class of bug is invisible to the compiler:
# fixture paths and generated-output directories are RUNTIME strings, so a stale one builds cleanly
# and then fails only when someone runs the program. The 2026-08 restructure produced eight of them
# across four files -- strings still naming tests/testdata/ after the fixtures moved to
# reference/he/. Checked here, that is a one-line diff instead of a failed demo.
#
# TWO RULES, because inputs and outputs are not the same claim:
#   - a SOURCE path (.cpp .hpp .py .sh .he) is always read, so the FILE must exist.
#   - a DATA path (.bin .pir .fold .csv .html .json .txt) may be written by the program, or may be a
#     fixture the repo deliberately does not ship, so only its PARENT DIRECTORY must exist. That is
#     exactly the difference between a stale path and an ungenerated one: tests/testdata/x.bin has
#     no directory, reference/pir/query_small.bin does.
#     One exception, and it is the whole reason this check once failed on a clean checkout: a
#     GITIGNORED directory is program output. It exists only after a run, so it is absent from every
#     fresh clone -- including CI's -- and demanding it turns `git clone && ctest` red for a path
#     that is perfectly correct (compiler/generated/he_spec.bin was exactly this). Ignored means
#     ungenerated, so it is accepted without the directory.
# `#include` lines are skipped entirely -- kernels include SDK headers by paths like
# tools/profiler/kernel_profiler.hpp that resolve inside the tt-metalium install, not here.
# A candidate must be anchored at a repo top-level directory, because that is what makes it
# cwd-relative. Kernel names ("compute/modmul_compute.cpp") resolve through
# OVERRIDE_KERNEL_PREFIX and headers ("cpu/montgomery.hpp") through HE_INCLUDE_DIRS, so
# neither is a path from here and neither is checked.
# The anchor list deliberately includes RETIRED names -- tests/ bench/ examples/ lib/ device/
# algorithm/ -- because a stale path names an old directory (tests/testdata/ rather than
# reference/he/). Listing only current dirs would miss exactly the bug this checks for.
#
# NOT caught: paths assembled piecewise, e.g. os.path.join(ROOT, "host", "tests", ...). Those are
# invisible to any grep; only running the program finds them.
#
# It also enforces the two layout invariants the restructure established, because a folder only
# means something if something breaks when it stops being true:
#   1. host/ holds host C++ and nothing else -- scripts in tools/, data in reference/.
#   2. every ctest label matches the folder its program lives in (hw/perf/study).
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

TMP="${TMPDIR:-/tmp}/check_paths.$$"
: >"$TMP"
trap 'rm -f "$TMP"' EXIT INT HUP TERM

# ---- 1. quoted repo-relative paths that do not resolve -----------------------------------------
for f in $(git ls-files '*.cpp' '*.hpp' '*.py' '*.sh' 'CMakeLists.txt' 'cmake/*.cmake'); do
    grep -v '^[[:space:]]*#include' "$f" 2>/dev/null |
        grep -oh '"\(host\|kernels\|reference\|compiler\|tools\|docs\|cmake\|tests\|bench\|examples\|lib\|device\|algorithm\)/[A-Za-z0-9_./-]*"' |
        tr -d '"' | sort -u |
        while read -r p; do
            case "$p" in
                *.cpp|*.hpp|*.py|*.sh|*.he)
                    [ -e "$p" ] || echo "$f: reads a source path that does not exist: $p" >>"$TMP" ;;
                *.bin|*.pir|*.fold|*.csv|*.html|*.json|*.txt)
                    d=$(dirname "$p")
                    [ -d "$d" ] && continue
                    # gitignored -> generated at run time, legitimately absent from a clean clone
                    git check-ignore -q "$d/" 2>/dev/null && continue
                    echo "$f: names $p, but the directory $d/ does not exist" >>"$TMP" ;;
            esac
        done
done

# ---- 2. CMake ARGS fixtures (cmake validates SOURCES and KERNELS itself, but not ARGS) ---------
# A target labelled `fixture` may legitimately name an artifact the repo does not ship; anything
# else must resolve.
grep -n 'ARGS [A-Za-z0-9_./-]*' CMakeLists.txt | while IFS= read -r hit; do
    p=$(printf '%s' "$hit" | sed 's/.*ARGS \([A-Za-z0-9_./-]*\).*/\1/')
    case "$p" in *.bin|*.pir|*.fold) ;; *) continue ;; esac
    [ -e "$p" ] && continue
    printf '%s' "$hit" | grep -q fixture && continue
    echo "CMakeLists.txt: ARGS names a missing artifact on a non-fixture target: $p" >>"$TMP"
done

# ---- 2b. every host/<x>/ path names a real host/ subfolder ------------------------------------
# Unquoted too, and that matters: the path rewrite that created host/ collided with a later rule and
# produced `host/tools/run_all.sh` in a comment. Quoted-string scanning missed it because comments
# are not quoted, and the compiler cannot see a comment at all. The set of host/ subfolders is
# closed, so anything else under it is wrong by construction -- no false positives possible.
grep -rno 'host/[a-z_0-9][a-z_0-9]*/' --include='*.cpp' --include='*.hpp' --include='*.py' \
     --include='*.sh' --include='*.md' --include='*.cmake' --include='CMakeLists.txt' . 2>/dev/null |
    while IFS= read -r hit; do
        # the checkers themselves quote counterexamples on purpose
        case "$hit" in ./tools/check/*.sh:*) continue ;; esac
        seg=$(printf '%s' "$hit" | sed 's|.*host/\([a-z_0-9]*\)/.*|\1|')
        case "$seg" in
            lib|tests|examples|benchmarks|experiments) ;;
            *) echo "$hit: host/$seg/ is not a host/ subfolder" >>"$TMP" ;;
        esac
    done

# ---- 3. host/ holds host C++ and nothing else --------------------------------------------------
for f in $(git ls-files 'host/*'); do
    case "$f" in
        *.cpp|*.hpp) ;;
        *) echo "host/ must hold only C++: $f  (scripts -> tools/, data -> reference/)" >>"$TMP" ;;
    esac
done

# ---- 4. every ctest label matches the folder the program lives in ------------------------------
# The mapping is not 1:1 (one source under host/benchmarks/ backs both the pinned benchmark and the
# multi-arm study), so what is checked is the property that still holds and still matters: anything
# under host/benchmarks/ or host/experiments/ MEASURES, so it must not be labelled `hw` -- that
# would drop a benchmark or a finished study into the correctness gate, where a timing run has no
# business. Conversely host/tests/ and host/examples/ must be `hw`.
#
# No awk: an earlier version used match()/RSTART, which works in gawk and silently matches nothing
# in mawk -- so on a mawk machine (Debian/Ubuntu default) the check passed without checking. A
# check that cannot fail is worse than no check. This flattens each he_device_target(...) onto one
# line with tr+sed and then uses only grep and sed.
tr '\n' ' ' < CMakeLists.txt |
    sed 's/he_device_target(/\n&/g' |
    grep '^he_device_target(' |
    while IFS= read -r t; do
        name=$(printf '%s' "$t" | sed 's/^he_device_target([ \t]*\([A-Za-z_0-9]*\).*/\1/')
        src=$(printf '%s' "$t" | sed -n 's/.*SOURCES[ \t]*\([^ \t]*\).*/\1/p')
        [ -n "$src" ] || continue
        case "$t" in *"TEST "*) ;; *) continue ;; esac
        lbl=$(printf '%s' "$t" | sed -n 's/.*TEST[ \t]*\([a-z][a-z ]*\).*/\1/p' | sed 's/[ \t]*$//')
        case " $lbl " in
            *" perf "*)  have=perf ;;
            *" study "*) have=study ;;
            *" hw "*)    have=hw ;;
            *) continue ;;
        esac
        case "$src" in
            # host/experiments/ (incl. legacy/) measures; it must never sit in the correctness
            # gate, but perf and study are both legitimate there -- the three programs whose numbers
            # the README quotes carry perf so CI keeps running them, the rest carry study.
            host/benchmarks/*|host/experiments/*) want=notgate ;;
            host/tests/*|host/examples/*) want=hw ;;
            *) continue ;;
        esac
        if [ "$want" = notgate ]; then
            [ "$have" = hw ] &&
                echo "label mismatch: $name is 'hw' but $src measures -- use perf or study, or it joins the gate" >>"$TMP"
            continue
        fi
        [ "$have" = "$want" ] ||
            echo "label mismatch: $name is '$have' but $src implies '$want'" >>"$TMP"
    done

n=$(wc -l <"$TMP")
if [ "$n" -gt 0 ]; then
    echo "check_paths: $n problem(s)"
    cat "$TMP"
    exit 1
fi
echo "check_paths: OK (paths resolve, host/ is C++ only, labels match folders)"
exit 0
