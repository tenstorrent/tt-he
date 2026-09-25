# Verification snapshots

Output of `tools/bench/baseline_sweep.sh` — one line per binary: exit code + the verdict line it printed.
Diff two snapshots to prove a change altered no behaviour. The `.txt` files are gitignored (machine- and
time-specific); this README is not.

Snapshots taken across the 2026-07 restructure, in order:

| file | state |
|---|---|
| `baseline_pre.txt` | before any restructure work (found `test_ir_matmul` hanging — a third copy of the kernel deadlock) |
| `baseline_post.txt` | after Phase 1 (build system + ctest); differs from pre by exactly that one fix |
| `baseline_move.txt` | after the kernels hoist + CMakeLists moved to the root — identical to post |
| `baseline_split.txt` | after the host/lib/examples/bench/tests split — identical to move |
| `baseline_all_green.txt` | **77/77 pass, 0 fail (2026-08-03)** — the reference snapshot. First fully green run: every device binary verified, including the 19 that had never run before the fixtures were committed. Diff against this one. |
| `baseline_phase2_final.txt` | **Phase 2 complete** (reference/ + compiler/, fixtures relocated, `device/` gone). Differs from split by exactly one *added* line: `test_montgomery PASS`, the new host-only test. 77 binaries, 58 pass, 19 fixture-blocked. Compare against this one. |

Expected result on this hardware: **77 pass, 0 fail**. Fixtures are committed, so no Python is needed
to reproduce this — `sh tools/bench/baseline_sweep.sh out.txt`.

Run one program at a time on the device. Concurrent runs produce false failures.
