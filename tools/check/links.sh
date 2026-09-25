#!/bin/sh
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# check_links.sh — fail if any markdown link points at a file that does not exist.
#
#   sh tools/check/links.sh
#
# Cheap, needs no hardware and no SDK, and catches the most common rot in this repo: docs move during
# consolidation and their inbound links are missed. Every doc reorganization here has produced at
# least one dangling link that this check found.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

python3 - <<'PY'
import pathlib, re, subprocess, sys
files = subprocess.run(['git','ls-files','*.md'], capture_output=True, text=True).stdout.split()
bad = []
for f in files:
    p = pathlib.Path(f)
    if not p.exists():
        # A tracked file absent from the working tree used to crash this script with a Python
        # traceback. It is a real problem (and it hid a stray file from a repo-wide grep once),
        # so report it instead of dying.
        bad.append((f, '<file is tracked but absent from the working tree>')); continue
    for m in re.finditer(r'\]\(([^)#][^)]*\.md)\)', p.read_text(encoding='utf-8', errors='ignore')):
        target = m.group(1).split('#')[0]
        if target.startswith(('http://','https://')):
            continue
        if not (p.parent / target).resolve().exists():
            bad.append((f, m.group(1)))
print(f"checked {len(files)} markdown files")
for f, t in bad:
    print(f"  DANGLING  {f} -> {t}")
print(f"dangling links: {len(bad)}")
sys.exit(1 if bad else 0)
PY
