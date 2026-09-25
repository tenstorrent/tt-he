# Browser UI for the 65 536-row PIR demo — plan

Goal: type a row index in a browser, watch the device answer it, see that the server never learned
which row. Two processes, as requested: a **backend that owns the Tenstorrent device** and a
**frontend that owns the UI**.

This plan is written against measured behaviour of `examples/pir_demo65536.cpp` (see `pir.md` §7.3 and
`pir65536_demo.md` §10), not against how the demo *ought* to behave. The measurements drive the design.

---

## 1. The four facts that decide the architecture

| # | measured fact | consequence for the UI |
|---|---|---|
| 1 | **Cold pass is per *process*, not per machine.** Q=1 ≈ 4.3 s, Q=3 = 18 s, **Q=5 = 579 s**. A second Q=5 run with the on-disk kernel cache warm cost the same 579 s. `WC` lives in memory and dies with the process. | The device process must be **long-lived** and must **survive frontend restarts**. This single fact rules out "spawn the binary per request" and strongly favours a detached daemon over a Python-spawned child. |
| 2 | **Changing batch width calls `caches_clear()`** and pays a full cold pass for the new width. | The UI must not let a user drift across widths casually. Q=1 interactive is cheap; Q=5 is a **ten-minute** stall. Width changes need an explicit, guarded action with progress. |
| 3 | **The device is exclusive** — one process holds it. Crashes/timeouts leave it wedged; recovery is `tt-smi -r`. | Exactly one backend, enforced by a lock. A health endpoint plus a documented recovery path. Running `ctest` while the UI is up will fail. |
| 4 | **Steady state is fast and flat:** 247–305 ms/query; replay enqueue is ~0.02 ms/call regardless of Q. | Once warm, the UI feels instant. All the engineering risk is in warm-up and width management, not in serving. |

Startup budget for a Q=1-warm backend: DB residency ~1.5 s + key prep ~0.2 s + Q=1 cold pass ~4.3 s
≈ **6 s**. That is acceptable to pay at boot; it is not acceptable to pay per request.

---

## 2. Architecture

    ┌────────────────────────┐        JSON Lines over a          ┌───────────────────────┐
    │  Frontend  (Python)    │  ── Unix domain socket ─────────▶ │  Backend  (C++)       │
    │  stdlib http.server    │        /tmp/pir_demo.sock          │  pir_demo65536 --serve│
    │  serves HTML/JS + SSE  │ ◀────────────────────────────────  │  OWNS THE DEVICE      │
    │  restartable, cheap    │                                    │  DB resident, warm    │
    └────────────────────────┘                                    └───────────────────────┘
            │ :8071                                                        │
        browser  ◀── ssh -L 8071:localhost:8071                        p150b + 6 GB DB

**Why a socket daemon rather than Python spawning the binary as a child:** with a child process, every
frontend restart kills the backend and repays the cold pass — 6 s at Q=1, but **ten minutes** if the
session had reached Q=5. Decoupling means UI iteration is free. This is the main design decision and
fact 1 is the reason for it.

**Why not pybind11 / in-process Python:** would put the TT runtime in the same process as the web
server, and rebuild the engine as an extension module. Large lift, and a UI bug then takes the device
down with it. Rejected.

**Why not an HTTP server in C++:** no HTTP library in this tree, and we want Python for the UI anyway.
A line-delimited socket protocol is ~60 lines of POSIX and needs no dependency.

**Why stdlib `http.server` and vanilla JS:** nothing is installed — no Flask, no FastAPI, not even
Dash/Plotly (so `tools/traces/trace_app.py` cannot currently run either). A zero-dependency frontend
works on this VM today and keeps the "forward a port, open a browser" workflow the repo already uses.

---

## 3. Wire protocol — JSON Lines, one request per line

Requests (frontend → backend):

    {"op":"status"}
    {"op":"query","idx":[40000]}
    {"op":"query","idx":[3,777,40000]}      // one shared scan; changes width -> may be SLOW
    {"op":"warm","q":1}                     // pay a width's cold pass deliberately

Events (backend → frontend), streamed, terminated by `result` or `error`:

    {"ev":"accepted","idx":[40000],"q":1,"width_change":false}
    {"ev":"cold","q":5,"est_s":579,"why":"width change clears the workload cache"}
    {"ev":"stage","name":"expand","ms":280.4}
    {"ev":"result","rows":[{"idx":40000,"text":"40000 Visa - ...","verified":true}],
     "timings":{"fold_rgsw":224,"expand":280,"convert":79,"scan":190,"folds":93,"decrypt":72},
     "per_query_ms":289,"scan_per_query_ms":63.3,
     "programs":{"expand":0,"convert":0,"scan":0,"folds":0,"total":0},
     "client_ms":363,"rows_touched":65536}
    {"ev":"error","msg":"index out of range [0,65536)"}

`programs.total == 0` is the honest signal that the query was fully replayed — worth surfacing in the
UI, because it is what makes the demo's "compile once, replay per query" claim visible. (That counter
was dead until recently; see `pir.md`. Do not trust it in older builds.)

**Serialize everything.** One device, one query at a time. The backend processes requests strictly in
order and the UI shows a queue depth. Do not add concurrency here — it cannot help.

---

## 4. Backend: `--serve` mode in `pir_demo65536.cpp`

The existing `--interactive` REPL already does exactly the right thing: parse an index list, call
`run_indices(idxs, twice=false)`. `--serve` is that loop with a socket instead of `std::cin` and JSON
instead of prose. Concretely:

1. **Socket setup** (~60 lines): `socket/bind/listen/accept` on `AF_UNIX`, line-buffered read loop.
   Accept one client at a time; a second connection gets a busy reply.
2. **Single-instance lock**: `flock` on `/tmp/pir_demo.lock`. Refuse to start if held — otherwise the
   second process fights for the device and both wedge (fact 3).
3. **Emit JSON alongside the existing prints.** `answer()` already computes every number in the
   `result` event; add a `json_out` flag and a small emitter. Keep the human prints — they are the
   fallback when the UI misbehaves.
4. **Warm at startup**: after DB residency, run one throwaway Q=1 query so the first real request is
   warm. Emit `{"ev":"ready"}`.
5. **Graceful shutdown** on SIGINT/SIGTERM: close the mesh cleanly. An abrupt kill is what wedges the
   device, and a wedged device then makes the *next* run look broken (this cost real time during the
   §7.3 work — see the operational note in `pir.md`).

Non-goal for v1: moving the client (query generation, ~360 ms) out of the C++ process. It stays where
it is; the UI just reports it under a "client" label so the client/server split stays visible.

---

## 5. Frontend: Python, standard library only

`tools/pir_ui/serve.py`, ~200 lines:

* `ThreadingHTTPServer` on `:8071`.
* `GET /` → a single self-contained HTML page (no CDN; the VM has no outbound access to rely on).
* `POST /api/query` → forwards to the socket, streams events back as **Server-Sent Events** so the
  browser can render stage-by-stage progress and, crucially, show something during a cold pass instead
  of appearing hung.
* `GET /api/status` → device state, DB residency, current width, warm/cold, queue depth.
* If the socket is absent or refuses: render a clear "backend not running" panel with the exact command
  to start it, plus the `tt-smi -r` recovery hint. Never fail silently.

---

## 6. What the UI shows

The demo's point is not "a database lookup is possible" — it is **the server answered without learning
the index**. The UI should earn that claim, not assert it.

1. **Query panel** — index box `[0, 65536)`, submit. Optional: search `db65536.txt` client-side to find
   an index by company name (host-side convenience, clearly labelled as *not* part of the protocol).
2. **Stage timeline** — a bar per stage (fold RGSWs / expand / convert / scan / folds / decrypt) that
   fills live from `stage` events. This is the most informative single widget: it makes the scan's
   dominance and the folds' behaviour legible.
3. **Result** — the retrieved row text, plus a green/red **verified** badge from the `db65536.txt`
   comparison the demo already performs.
4. **Privacy panel** — "server touched **all 65 536 rows**", "server saw only ciphertext", and the
   client-side ms spent generating the query. State plainly that the *index* never leaves the client;
   do not overclaim beyond that (the row *length* and the DB shape are public).
5. **Batch panel** — up to 16 indices in one shared scan, showing scan ms/query dropping as width grows
   (63.3 → 39.8 ms/query from Q=3 → Q=5). This is the honest illustration of amortisation.
   **Guarded**: a width change shows the estimated cold-pass cost and requires a second click.
6. **Backend status** — warm/cold, current width, programs built last query, device health.

---

## 7. Phasing

| phase | deliverable | why this order |
|---|---|---|
| 0 | `--serve` + socket + `status`; lock file; graceful shutdown | proves the daemon can hold the device and stay up — the riskiest assumption |
| 1 | `query` for Q=1 with full timings + verification | end-to-end value with the cheap width only |
| 2 | Python server + single-query page (no streaming) | first thing anyone can actually click |
| 3 | SSE streaming, stage timeline, startup warm-up, cold-pass indicator | makes it feel live and stops the "is it hung?" problem |
| 4 | Batch mode with guarded width changes | the expensive, hazardous feature, added once the basics are solid |
| 5 | Privacy panel, name search, polish | presentation |

Phases 0–2 are the minimum useful demo. Phase 3 is what makes it pleasant.

---

## 8. Risks

| risk | mitigation |
|---|---|
| **Width thrash** — a curious user alternating 1 and 5 indices pays 579 s repeatedly | guard width changes behind explicit confirmation with an estimate; default the UI to Q=1; consider offering only Q ∈ {1, 2, 3} |
| Backend wedges the device | lock file, graceful shutdown, health endpoint, documented `tt-smi -r`; the frontend must surface this rather than hang |
| Long-lived process stability is unproven | the longest observed run is ~10 min (a Q=5 cold pass). Soak-test the daemon before demoing |
| Device contention with `ctest` | document that the suite and the UI cannot run together; the lock file makes the failure obvious |
| A UI bug appears to be a device bug | keep the human-readable prints in `--serve`; always compare against the CLI before blaming the device |

---

## 9. The engine change that would most improve this UI

Width changes are only catastrophic because `resize_for()` calls `caches_clear()`, throwing away every
recorded workload. **Keying the workload cache by batch width** — keep a `WC` per width instead of one
global — would make width switching free after each width has been paid once, turning the worst UX
hazard into a one-time cost per width per process.

That is an engine change, not a UI change, and it interacts with the scratch pool (whose arena marks
also assume a single active width, and whose order check will fire loudly if that assumption breaks —
see the `bZeroN` bug in `pir.md`). Worth doing, but deliberately **out of scope for v1**: the UI should
be built and demoed against the engine as it is, and this optimisation evaluated separately.

Related and larger: the cold pass itself is O(N²) in recorded workloads (folds is 96 % of it). If that
is ever fixed, every number in §1 improves and the width hazard mostly evaporates.

---

## 10. STATUS — phases 0–3 built and working (2026-08-17)

    # 1. backend (owns the device; leave it running)
    ./build/pir_demo65536 --serve /tmp/pir_demo.sock
    # 2. frontend (restart freely; the backend stays warm)
    python3 tools/pir_ui/serve.py 8071
    ssh -L 8071:localhost:8071 <vm>     # then open http://localhost:8071

Measured end to end through HTTP → SSE → socket → device:

| case | result |
|---|---|
| `status` | `rows=65536 max_batch=16 width=1 ready=true` |
| Q=1 warm (`40000`) | verified, **387 ms**, 0 programs built, 687 ms browser round trip |
| Q=2 cold (`3,777`) | verified, 8.3 s, 5382 programs — estimate said 10 s |
| Q=2 warm (`12345,65535`) | verified, **311 ms/query**, 0 programs |
| malformed (`abc`, `1,abc`, `-5`, `70000`) | rejected with a clear message |
| SIGTERM | clean shutdown; device immediately usable **without** `tt-smi -r` |

### Two bugs found by building the UI

1. **`parse_indices` validated a dangling pointer** (pre-existing, also affects `--interactive`).
   `strtoll` sets `e` into `tok`'s buffer, then `tok.clear()` nulls the byte `e` addresses, so the
   `*e` check afterwards always saw `'\0'`. **Every non-numeric token silently became index 0** —
   typing `abc` returned row 0 with a green "verified" badge, and `1,abc` fetched rows 1 and 0 while
   paying an 8.4 s cold pass for a bogus width. Fixed by validating before clearing.
2. **`std::signal` made graceful shutdown impossible.** glibc's `signal()` implies `SA_RESTART`, so the
   blocking `accept()` auto-restarted and SIGTERM did nothing; the daemon could only be `kill -9`'d,
   which is exactly what wedges the device for the next run. Fixed with `sigaction` and `sa_flags = 0`.

### Deviations from the plan above, and why

* **Requests accept either JSON (`{"op":"query","idx":[40000]}`) or a plain line (`query 40000`).**
  Responses are always JSON Lines as specified. The plain form makes the backend testable with `nc`
  and costs about six lines of tolerant parsing.
* **No mid-run `stage` events yet.** `answer()` is synchronous, so emitting per-stage progress would
  mean refactoring it; the stage timeline renders from the `result` event instead. The "is it hung?"
  problem is covered by the `accepted`/`cold` events plus a live elapsed timer, which is what actually
  mattered. Mid-run events remain available if the timeline needs to animate.

### Not done

* Phase 4 (batch panel) and phase 5 (privacy panel polish, name search) — the privacy stats are in the
  page, but there is no batch UI beyond typing a comma-separated list.
* **Only Q=1 and Q=2 have been exercised through the UI.** Q≥3 works in the CLI but has not been
  clicked through, and Q=5 costs ~579 s of cold pass — deliberately avoided during development.

### Phase 4/5 increment (same day)

* **Three distinct backend states.** The `--serve` branch runs *after* DB residency and key prep, so for
  the first ~10 s no socket exists and the page said "backend not reachable" — indistinguishable from
  "not running", and the first thing that actually confused a user. The backend now publishes
  `<sock>.state` (`starting` → `ready`, removed on exit) before that slow setup, so the UI reports
  **not running / starting / ready** correctly and reconnects on its own. Verified across a full
  restart cycle.
* **Width guard (phase 4).** The page tracks the width the backend is built for and, the moment the
  typed index count differs, warns inline with a cold-pass estimate and relabels the button
  `Rebuild for N (~X s)`. Width changes are now a deliberate act rather than an accidental ten-minute
  stall, which was risk #1 in §8.
* **Name lookup (phase 5).** `GET /api/find?q=` greps the local `db65536.txt` and returns clickable
  hits. Labelled in the UI as a local lookup that is *not* part of the private query — it reads a
  plaintext file on the host and never touches the device. Being explicit matters: a name search that
  looked like part of the protocol would undercut the very claim the demo exists to make.

Verified after these changes: Q=1 385 ms / 0 programs; Q=2 width change warned at 10 s, took 9.2 s,
5382 programs, both rows verified; malformed input still rejected; `--idx` CLI unaffected.

Still open: no batch *panel* beyond typing a comma-separated list, and Q≥3 remains unexercised through
the UI by choice (Q=5 costs ~579 s of cold pass).

---

## 11. Three follow-up questions, answered by measurement (2026-08-17)

### Q1. Can we build once, store a binary, and never rebuild — even after reopening?

**No, and the reason matters.** The compiled kernels are *already* cached to disk — `~/.cache/tt-metal-cache`
is 1.9 GB and is reused across runs. The cold pass is **not compilation**: a second Q=5 run with that
cache fully warm cost the same 579 s. The split is 100 % *workload ENQUEUE*, i.e. per-process device-side
program setup (first `EnqueueMeshWorkload`). That state lives on the device and dies with the process, so
there is nothing host-side to serialise. This installation also exposes no Light Metal capture/replay API
to persist a trace.

What actually holds:
* **Reopening the browser costs nothing** — the daemon keeps the device warm; that is why the backend is
  a separate long-lived process. Only restarting the *backend* repays anything.
* `--warm N` pays one width's cold pass at boot (~4.5 s at Q=1) so the first click is fast.
* **`--warm` deliberately takes ONE width.** Warming several is wasted work: `resize_for()` calls
  `caches_clear()`, wiping the entire cache instead of keeping one per width. Measured: `--warm 1,2` cost
  15 s and still left one width warm — switching 1→2 afterwards rebuilt all 5382 programs (8.1 s), and
  switching back to 1 rebuilt 2943 (3.7 s).
* The real fix is the engine change in §9 (cache keyed by width) or cutting the program count that drives
  the O(N²) enqueue cost. Both remain out of scope here.

### Q2. Show the database, searchable by id and name — done

`GET /api/rows?center=&n=` returns a window of `db65536.txt`; the page shows it in a scrollable panel with
the current row highlighted, and clicking a row loads it into the query box. The search box accepts either
a bare number (jumps the viewer to that index) or a substring (`Visa` → clickable hits). Both are labelled
as *local* lookups over the plaintext file: they never touch the device, and saying so plainly matters —
a search that looked like part of the protocol would undercut the claim the demo exists to make.

### Q3. Make the crypto visible — done

The result now carries a `crypto` block, rendered as five stages:

| stage | what the UI shows |
|---|---|
| 1 plaintext query | one-hot over 2048 coefficients, all zero except position `c0` |
| 2 encrypted query | first 10 ciphertext coefficients, e.g. `80293235, 204891129, 33935907, …` |
| 3 answer ciphertext | first 10 coefficients returned by the server |
| 4 decrypted coefficients | first 16, each < t = 257 — **and the same values as characters** |
| 5 decoded row | the text |

Stage 4 is the point: `[52, 48, 48, 48, 48, 32, 86, 105, 115, 97, …]` renders as `"40000 Visa - Vis"`.
Ciphertext coefficients sit near 2²⁸ and look random; plaintext coefficients are small because they are
the row's ASCII bytes. The contrast is visible without needing to trust a label.

### Also fixed while doing this

* **Daemon stdout was block-buffered**, so a redirected log sat empty for minutes and the backend looked
  hung. Line-buffered in `--serve`: the log is the operator's only window into the device.
* `--warm` warms in a single pass and leaves that width resident (an earlier multi-width version left the
  *last* width warm, so the common Q=1 case was cold — see Q1).

---

## 12. Layout and live-progress revision (2026-08-17)

* **Split panes.** Left = the local plaintext database; right = the private query and its results. The
  two halves of the demo are now visually separate, which is also the honest framing: one side is a
  local file, the other is the device.
* **Search moved into the DB pane, and unwired from the query box.** Browsing or searching no longer
  fills the index field — clicking around the plaintext has no effect on what gets asked for. Worth
  keeping that way: a viewer that drove the query blurred exactly the boundary the demo is about.
  The search accepts an id (jumps the viewer) or a name (jumps to the first match).
* **Live stage bars (real progress, not a spinner).** `answer()` now calls a `serve::on_stage` hook as
  each stage's timer closes, and the socket loop forwards `{"ev":"stage","name":…,"ms":…}`. The browser
  fills each bar as the device reports it, highlights the running stage, and shows elapsed seconds.
  The scan emits ~12 updates (one per m-tile), so it fills progressively rather than jumping.
  These are measured device timings, not an animation.
* **Result moved below the timings**, with the crypto walk-through and privacy stats under it, so the
  right-hand column reads top-to-bottom: ask → watch it run → see the row → see the ciphertext.

---

## 13. One-command launcher — `examples/pir_ui.py`

    python3 host/examples/pir_ui.py              # start everything, print the URL
    python3 host/examples/pir_ui.py --stop       # release the device
    python3 host/examples/pir_ui.py --reset      # tt-smi -r first, if the device is wedged
    python3 host/examples/pir_ui.py --restart    # force a fresh backend
    python3 host/examples/pir_ui.py --port 9000 --warm 2

It supervises the same two processes; it does not merge them. Behaviour worth knowing:

* **Ctrl-C stops the UI and leaves the backend warm.** Restarting the frontend is then instant. That is
  the whole point of the split (§2) and the launcher must not undo it — use `--stop` to release the
  device.
* **Reuses a warm backend** instead of starting a second one. The device is exclusive; two backends
  would fight over it. Measured: a second launch reports "already running and warm" and is instant,
  against ~9 s for a cold start.
* **Picks the newest binary and names it**, warning about older ones. This tree has more than one build
  directory, and running a stale binary is a genuinely confusing failure — the code looks fixed and the
  behaviour is not. (That exact mistake produced a 47-failure ctest run during the §7.3 work.)
* **Preflights the 4 data files (7.5 GB)** and fails with the command to regenerate them, rather than
  letting the backend die 30 s later with a file-open error.
* **Sources `env.sh` in a shell** — it defines shell functions, so it cannot be parsed into `os.environ`.
* On startup failure it prints the last 15 lines of the backend log and suggests `--reset`.

---

## 14. Batch representation in the crypto panel

The panel used to peek at query 0 only, which misrepresented a batch. It now carries one entry per
query, because that is what the scheme actually does:

* **There is no batched encoding.** Each query is its own plaintext (`slot = idx ÷ 128`, value
  `(2^m_exp)^-1 mod t`) and its own ciphertext, encrypted independently. Nothing is packed together.
* **The queries meet only at the scan.** Query `qi`'s expanded selectors are written into *its own two
  columns* of a shared operand (`out_base = (qi*2+comp)*2`, `out_pstride = 2*COLS`, `COLS = 2Q`), and one
  GEMM over all 65 536 rows answers the whole batch. 2 columns per query out of 32 → the 16-query cap.
* The panel therefore shows: Q plaintexts → Q ciphertexts → **one shared scan** (with columns used) →
  Q answer ciphertexts → Q decrypted coefficient sets → Q rows. Q=1 collapses to the plain single-query
  view with no per-query labels.

Measured, warm, showing what is shared and what is not:

| stage | Q=1 | Q=2 | ratio | |
|---|---|---|---|---|
| expand | 83.1 | 170.7 | 2.05x | per query |
| convert | 32.8 | 56.6 | 1.73x | per query |
| **scan** | **128.0** | **140.3** | **1.10x** | **shared** |
| folds | 26.4 | 55.9 | 2.12x | per query |
| decrypt | 24.1 | 50.9 | 2.11x | per query |

Everything per-query doubles; the scan moves 10 %. That is the entire case for batching: the scan is
fixed by the database, and a single query leaves 14 of 16 column pairs idle.

Leakage note for the panel's honesty: batching reveals *how many* queries were sent (2Q of 32 columns),
never which rows — the scan still touches every row.
