#!/bin/sh
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# gen_fixtures.sh — regenerate every committed test fixture.
#
#   sh tools/gen_fixtures.sh
#
# The .bin fixtures are COMMITTED, so running the device tests needs no Python at all. This script
# exists so they stay reproducible rather than becoming magic blobs: it records the exact generator
# and arguments behind each one. Regenerate and `git diff` to detect drift after touching a generator.
#
# Why fixtures at all: each file carries inputs AND the expected output, computed independently in
# exact big-integer Python. A device test that computed its own expectation with the same
# radix-128/Montgomery logic as the kernels would pass even when that logic is wrong.
#
# Needs numpy: see environment.yml (conda) or requirements.txt (pip), and set PYTHON= in
# config.local.env. Everything else -- building and running the tests -- does not.
set -u   # no `pipefail`: must run under POSIX sh (dash)

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY=$(sed -n 's/^[ \t]*PYTHON=//p' "$REPO/config.local.env" "$REPO/config.defaults.env" 2>/dev/null | head -1)
PY=${PY:-python3}
"$PY" -c "import numpy" 2>/dev/null || {
    echo "numpy not available via PYTHON=$PY"
    echo "  conda env create -f environment.yml && echo \"PYTHON=\$(which python)\" >> config.local.env"
    exit 1
}
echo "using $PY ($("$PY" -c 'import numpy;print("numpy",numpy.__version__)'))"

gen() {  # gen <dir> <label> <script> [args...]
    dir=$1; label=$2; shift 2
    printf '  %-22s ' "$label"
    if ( cd "$REPO/$dir" && "$PY" "$@" >/dev/null 2>&1 ); then echo ok; else echo FAILED; fi
}

echo "== PIR fixtures (reference/pir) =="
# n defaults to 2048 in these generators; the n=1024 tests need it passed explicitly.
gen reference/pir gadget_k2_l8     gen_gadget_case.py    --out gadget_k2_l8.bin              # defaults: B=256 ell=8 K=2
gen reference/pir gadget_k3_l8     gen_gadget_case.py    --K 3 --B 2048 --ell 8 --out gadget_k3_l8.bin
gen reference/pir gadget_ep        gen_gadget_case.py    --B 1024 --ell 6 --out gadget_ep.bin
gen reference/pir ks_case          gen_ks_case.py        --n 1024 --out ks_case.bin
gen reference/pir ks_n2048         gen_ks_case.py        --out ks_n2048.bin
gen reference/pir extprod_case     gen_extprod_case.py   --n 1024 --out extprod_case.bin
gen reference/pir extprod_n2048    gen_extprod_case.py   --out extprod_n2048.bin
gen reference/pir expand_case      gen_expand_case.py    --out expand_case.bin
gen reference/pir expand_n2048     gen_expand_case.py    --out expand_n2048.bin
gen reference/pir initdim_case     gen_initdim_case.py   --out initdim_case.bin
gen reference/pir modswitch_case   gen_modswitch_case.py --out modswitch_case.bin
gen reference/pir mux_case         gen_mux_case.py       --out mux_case.bin
gen reference/pir permute_case     gen_permute_case.py   --out permute_case.bin
gen reference/pir pir_answer_n2048 gen_pir_answer_case.py --out pir_answer_n2048.bin

# No move step: the generators write into reference/he, which is where the fixtures live. The old
# layout generated here and moved them to reference/testdata, and that window is what left two
# stale copies behind in this directory (see the commit that removed them).
echo "== HE fixtures (reference/he) =="
gen reference/he ctmul_case   gen_ctmul_case.py   0      # seed is positional
gen reference/he rescale_case gen_rescale_case.py 0

echo
echo "done. 'git status' should be clean unless a generator changed."
