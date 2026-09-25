# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""_paths.py — resolve the repository layout for every module in this package.

WHY THIS FILE EXISTS. Each module used to carry its own
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "he"))
which was correct when this package lived at `reference/ir/`. The repository restructure moved the
package to `compiler/` and the HE references to `reference/he/`, so all ten of those lines
pointed at a directory that no longer exists and seven of the eight self-tests died on import. One
resolver, imported for effect, is the fix: there is now a single place to change when the layout moves
again.

    import _paths            # noqa: F401  (imported for effect)
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                  # repository root
HE = os.path.join(ROOT, "reference", "algorithm", "he")        # bfv_* / ckks_* references
DOCS = os.path.join(ROOT, "docs")
KERNELS = os.path.join(ROOT, "kernels")
BUILD = os.path.join(ROOT, "build")
GEN = os.path.join(HERE, "generated")                          # emitted host + spec land here

if not os.path.isdir(HE):                                      # fail loudly, not on a confusing ImportError
    raise RuntimeError(f"_paths: HE reference dir not found: {HE}\n"
                       f"  (repo root resolved to {ROOT}; is this package still at <root>/compiler?)")

for _p in (HE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
