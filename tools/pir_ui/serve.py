#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Browser UI for the 65 536-row PIR demo — frontend process.

Two processes, by design (docs/pir/pir_ui_plan.md):

  1. backend   ./build/pir_demo65536 --serve /tmp/pir_demo.sock     <- owns the Tenstorrent device
  2. frontend  python3 tools/pir_ui/serve.py [port]                 <- this file, owns the UI

They are decoupled on purpose. The PIR cold pass is per *process* (Q=1 ~4.3 s, Q=5 ~579 s), so the
backend must outlive the frontend: restart this script as often as you like, the device stays warm.

  python3 tools/pir_ui/serve.py 8071
  ssh -L 8071:localhost:8071 <vm>     then open  http://localhost:8071

Standard library only — this VM has no Flask/FastAPI/Dash installed.
"""
import http.server, socketserver, socket, json, os, sys, threading, time, urllib.parse

SOCK = "/tmp/pir_demo.sock"
DBTXT = "reference/pir/db65536.txt"      # local plaintext, used ONLY for the name->index lookup
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8071

# The device answers one query at a time. Serialising here keeps the backend's socket loop simple and
# gives the UI an honest "busy" instead of two requests interleaving on one device.
_lock = threading.Lock()


def backend(lines, timeout=900):
    """Send request lines, yield decoded JSON events until a terminal one."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(SOCK)
    try:
        for l in lines:
            s.sendall((l + "\n").encode())
        buf = b""
        while True:
            d = s.recv(65536)
            if not d:
                return
            buf += d
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                ev = json.loads(line)
                yield ev
                if ev.get("ev") in ("result", "error", "status", "bye"):
                    return
    finally:
        s.close()


def starting():
    """True if a backend is mid-startup: it publishes a state file before its ~10 s of DB load and key
    prep, so the UI can say "starting" instead of the misleading "not running"."""
    try:
        with open(SOCK + ".state") as f:
            return f.read().strip() == "starting"
    except OSError:
        return False


def backend_down(e):
    if starting():
        return {"ev": "warming",
                "msg": "backend is starting: 6 GB DB residency + key prep + Q=1 warm-up (~40 s on a "
                       "cold page cache). This page will connect by itself."}
    return {"ev": "error", "kind": "backend",
            "msg": f"backend not reachable on {SOCK} ({e.__class__.__name__}). Start it with:  "
                   f"./build/pir_demo65536 --serve {SOCK}   — and if the device is wedged after a "
                   f"crash or timeout, run  tt-smi -r  first."}


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                                            # quiet; the backend log is the interesting one

    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        # The page carries its own JS, so a cached copy silently keeps stale behaviour after an edit --
        # a changed stage order looked "unfixed" purely because the browser reused the old script.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/status":
            try:
                evs = list(backend(["status"], timeout=4))
                return self._send(200, json.dumps(evs[-1] if evs else {"ev": "error", "msg": "no reply"}))
            except socket.timeout:
                # Socket exists and accepted us, but no reply: the backend binds before it warms up,
                # so this is startup (DB residency + Q=1 warm-up, ~40 s), not a dead backend.
                return self._send(200, json.dumps(
                    {"ev": "warming", "msg": "backend is starting: 6 GB DB residency + Q=1 warm-up "
                                             "(~40 s). This page will connect as soon as it is ready."}))
            except Exception as e:
                return self._send(200, json.dumps(backend_down(e)))
        if u.path == "/api/rows":
            qs = urllib.parse.parse_qs(u.query)
            try:
                c = int(qs.get("center", ["0"])[0]); n = min(int(qs.get("n", ["15"])[0]), 60)
            except ValueError:
                c, n = 0, 15
            return self._send(200, json.dumps(self._rows(c, n)))
        if u.path == "/api/find":
            return self._send(200, json.dumps(self._find(
                urllib.parse.parse_qs(u.query).get("q", [""])[0])))
        if u.path == "/api/query":
            q = urllib.parse.parse_qs(u.query).get("idx", [""])[0].strip()
            return self._stream(q)
        return self._send(404, json.dumps({"ev": "error", "msg": "not found"}))

    def _rows(self, center, n):
        """A window of the LOCAL plaintext DB around `center`, for the viewer panel.

        Same caveat as _find: this reads db65536.txt on the host. It is here so you can see what the
        database looks like and pick a row; the private retrieval is the separate device query."""
        lo = max(0, center - n // 2)
        out = []
        try:
            with open(DBTXT, "r", errors="replace") as f:
                for i, line in enumerate(f):
                    if i < lo:
                        continue
                    if i >= lo + n:
                        break
                    out.append({"idx": i, "text": line.rstrip()[:110]})
        except OSError as e:
            return {"ev": "rows", "rows": [], "msg": f"cannot read {DBTXT}: {e}"}
        return {"ev": "rows", "rows": out, "center": center}

    def _find(self, needle):
        """Look up row indices by substring, from the LOCAL plaintext copy.

        This is a demo convenience for finding an index to ask for, and is deliberately not part of the
        PIR protocol: it reads db65536.txt on this host and touches the device not at all. In a real
        deployment the client already knows the index it wants; the private part is the retrieval."""
        needle = needle.strip().lower()
        if len(needle) < 2:
            return {"ev": "find", "hits": [], "msg": "type at least 2 characters"}
        hits = []
        try:
            with open(DBTXT, "r", errors="replace") as f:
                for i, line in enumerate(f):
                    if needle in line.lower():
                        hits.append({"idx": i, "text": line.rstrip()[:96]})
                        if len(hits) >= 12:
                            break
        except OSError as e:
            return {"ev": "find", "hits": [], "msg": f"cannot read {DBTXT}: {e}"}
        return {"ev": "find", "hits": hits}

    def _stream(self, idx):
        """Server-Sent Events: forward backend JSON lines as they arrive.

        This is what makes a cold pass survivable in a browser — the page shows `accepted` and `cold`
        (with an estimate) immediately, then sits on a live timer, instead of a blank spinner for what
        can be ten minutes at Q=5."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(obj):
            try:
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise

        if not _lock.acquire(blocking=False):
            emit({"ev": "error", "msg": "the device is busy with another query — one at a time"})
            return
        try:
            t0 = time.time()
            for ev in backend([f"query {idx}"]):
                ev["elapsed_ms"] = round(1000 * (time.time() - t0))
                emit(ev)
        except Exception as e:
            try:
                emit(backend_down(e))
            except Exception:
                pass
        finally:
            _lock.release()


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>PIR — 65 536 rows on Tenstorrent</title>
<style>
:root{--bg:#0e1116;--card:#171b22;--line:#262c36;--fg:#e6edf3;--dim:#8b949e;--ok:#3fb950;--bad:#f85149;
      --acc:#58a6ff;--warn:#d29922}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:26px 20px 60px}
h1{font-size:20px;margin:0 0 4px} .sub{color:var(--dim);margin:0 0 20px}
.cols{display:grid;grid-template-columns:minmax(300px,1fr) minmax(400px,1.2fr);gap:16px;align-items:start}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
.hd{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
input{background:#0d1117;border:1px solid var(--line);color:var(--fg);border-radius:7px;padding:9px 11px;
      font:inherit;width:100%}
button{background:var(--acc);border:0;color:#06121f;font-weight:600;border-radius:7px;padding:9px 16px;
       font:inherit;cursor:pointer;white-space:nowrap}
button:disabled{opacity:.5;cursor:not-allowed}
.muted{color:var(--dim)} .sm{font-size:12.5px}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600}
.b-ok{background:rgba(63,185,80,.15);color:var(--ok)} .b-bad{background:rgba(248,81,73,.15);color:var(--bad)}
.b-warn{background:rgba(210,153,34,.15);color:var(--warn)}
table{width:100%;border-collapse:collapse} td{padding:3px 0;vertical-align:middle}
td.k{color:var(--dim);width:118px} td.v{text-align:right;width:82px;font-family:ui-monospace,monospace}
.bar{height:9px;background:#0d1117;border-radius:5px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--acc);transition:width .18s linear}
tr.run td.k{color:var(--acc)} tr.run .bar>i{background:var(--warn)}
tr.idle td.k,tr.idle td.v{opacity:.45}
.rowtext{background:#0d1117;border:1px solid var(--line);border-radius:7px;padding:10px 12px;
         font-family:ui-monospace,monospace;font-size:13px;word-break:break-word;margin-top:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.stat{background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:10px}
.stat b{display:block;font-size:19px;font-weight:600} .stat span{color:var(--dim);font-size:12px}
.err{border-color:#5b2526;background:rgba(248,81,73,.08)}
.warnbox{border-color:#5a4413;background:rgba(210,153,34,.08)}
.dbview{margin-top:10px;max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:8px;
        background:#0d1117}
.dbrow{display:flex;gap:9px;padding:3px 10px;font-family:ui-monospace,monospace;font-size:12.5px;
       border-bottom:1px solid #11161d}
.dbrow.hit{background:rgba(88,166,255,.13)}
.dbrow i{color:var(--dim);font-style:normal;min-width:54px}
.cs{margin-bottom:11px} .cs .lab{font-size:12.5px;color:var(--dim);margin-bottom:3px}
.cs .val{background:#0d1117;border:1px solid var(--line);border-radius:6px;padding:7px 9px;
         font-family:ui-monospace,monospace;font-size:12px;word-break:break-all}
.arrow{color:var(--dim);text-align:center;font-size:11px;margin:-6px 0 5px}
</style></head><body><div class="wrap">

<h1>Private Information Retrieval — 65 536 rows</h1>
<p class="sub">The server fetches your row <em>without learning which row you asked for</em>.
Tenstorrent Blackhole p150b.</p>

<div class="cols">

  <!-- ================= LEFT: the local plaintext database ================= -->
  <div>
    <div class="card">
      <div class="hd"><b>Local database</b><span class="muted sm">plaintext, on this host</span></div>
      <div style="margin-top:10px"><input id="find" placeholder="search by id or name — e.g. 40000 or Visa"></div>
      <div id="hits" class="sm" style="margin-top:6px"></div>
      <div id="dbview"></div>
      <div class="muted sm" style="margin-top:9px">
        This panel just reads <span class="mono">db65536.txt</span> locally so you can see what is in the
        database. It never touches the device and is <b>not</b> part of the private query — browsing here
        tells the server nothing.</div>
    </div>
  </div>

  <!-- ================= RIGHT: the real, private query ================= -->
  <div>
    <div class="card">
      <div class="hd"><b>Private query</b><span id="status" class="muted sm">connecting…</span></div>
      <div class="row" style="margin-top:10px">
        <input id="idx" placeholder="row index 0–65535, or 3,777" value="40000"
               autocomplete="off" style="flex:1 1 190px;width:auto">
        <button id="go">Retrieve</button>
      </div>
      <div id="note" class="muted sm" style="margin-top:9px"></div>
      <div id="widthwarn" class="sm" style="display:none;margin-top:9px"></div>
    </div>

    <div id="coldbox" class="card warnbox" style="display:none">
      <b>Rebuilding for this batch width.</b>
      <div class="muted sm" style="margin-top:5px" id="coldmsg"></div>
    </div>

    <div id="err" class="card err" style="display:none"></div>

    <div id="timecard" class="card" style="display:none">
      <div class="hd"><b>Where the time went</b><span id="elapsed" class="muted sm"></span></div>
      <div class="muted sm" style="margin:4px 0 12px">Filling live as the device reports each stage.</div>
      <table id="stages"></table>
    </div>

    <div id="resultcard" class="card" style="display:none">
      <div class="hd"><b>Result</b><span id="verdict"></span></div>
      <div id="rows"></div>
    </div>

    <div id="cryptocard" class="card" style="display:none">
      <b>The query, end to end</b>
      <div class="muted sm" style="margin:4px 0 12px">Same data at five points. Ciphertext coefficients are
        pseudorandom residues near 2<sup>28</sup>; the decrypted ones are small because they are the row's
        ASCII bytes.</div>
      <div id="crypto"></div>
    </div>

    <div id="privcard" class="card" style="display:none">
      <b>What the server learned</b>
      <div class="grid" style="margin-top:11px">
        <div class="stat"><b id="touched">—</b><span>rows touched</span></div>
        <div class="stat"><b>0</b><span>rows it could skip</span></div>
        <div class="stat"><b id="clientms">—</b><span>client-side query gen</span></div>
        <div class="stat"><b id="progs">—</b><span>programs compiled</span></div>
      </div>
      <div class="muted sm" style="margin-top:11px" id="privnote"></div>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
let busy = false, width = null, maxBatch = 16;

// EXECUTION order, confirmed by the live stage events: expand -> convert -> scan -> fold RGSWs ->
// folds -> decrypt. "fold RGSWs" prepares this query's fold operands and runs just before the folds,
// not first; the old CLI label numbered it [0] and that ordering was wrong.
const STAGES = [["expand","expand"],["convert","convert"],["scan","scan (all rows)"],
                ["fold_rgsw","fold RGSWs"],["folds","folds"],["decrypt","decrypt"]];
let live = {}, running = null;

function fmt(ms){ return ms >= 1000 ? (ms/1000).toFixed(2)+" s" : Math.round(ms)+" ms"; }
function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }

function drawStages(total){
  const max = Math.max(...STAGES.map(([k]) => live[k]||0), 1);
  $("#stages").innerHTML = STAGES.map(([k,label]) => {
    const v = live[k]||0, cls = running===k ? "run" : (v>0 ? "" : "idle");
    return `<tr class="${cls}"><td class="k">${label}</td>`
      + `<td><div class="bar"><i style="width:${(100*v/max).toFixed(1)}%"></i></div></td>`
      + `<td class="v">${v>0?fmt(v):"—"}</td></tr>`; }).join("")
    + (total!=null ? `<tr><td class="k"><b>device total</b></td><td></td>`
        + `<td class="v"><b>${fmt(total)}</b></td></tr>` : "");
}

function showErr(m){ const e=$("#err"); e.style.display="block"; e.textContent=m; }

function poll(){
  fetch("/api/status").then(r=>r.json()).then(s=>{
    if (s.ev === "status") {
      width = s.width; maxBatch = s.max_batch;
      $("#err").style.display="none";
      $("#status").innerHTML = '<span class="badge b-ok">backend ready</span>';
      $("#note").textContent = `${s.rows.toLocaleString()} rows · up to ${s.max_batch} indices share one `
        + `scan · built for ${s.width} ${s.width===1?"query":"queries"}`;
      guard();
    } else if (s.ev === "warming") {
      $("#status").innerHTML = '<span class="badge b-warn">warming…</span>';
      $("#note").textContent = s.msg; setTimeout(poll, 5000);
    } else {
      showErr(s.msg || "backend error");
      $("#status").innerHTML = '<span class="badge b-bad">no backend</span>'; setTimeout(poll, 5000);
    }
  }).catch(e => { showErr(String(e)); setTimeout(poll, 5000); });
}
poll();

function count(){ return $("#idx").value.split(/[,\s]+/).filter(x=>x.length).length; }
function estCold(q){ return q<=1?5:q<=2?10:q<=3?20:q<=4?120:600; }
function guard(){
  const n = count(), w = $("#widthwarn"), b = $("#go");
  if (width === null || n === 0 || n === width) { w.style.display="none"; b.textContent="Retrieve"; return; }
  if (n > maxBatch) { w.style.display="block";
    w.innerHTML = `<span class="badge b-bad">too many</span> at most ${maxBatch} indices share one scan.`;
    b.textContent = "Retrieve"; return; }
  w.style.display = "block";
  w.innerHTML = `<span class="badge b-warn">width change</span> backend is built for <b>${width}</b>, `
    + `you typed <b>${n}</b> — rebuilding costs roughly <b>${estCold(n)} s</b>, once.`;
  b.textContent = `Rebuild for ${n} (~${estCold(n)} s)`;
}
$("#idx").addEventListener("input", guard);
$("#idx").addEventListener("keydown", e => { if (e.key === "Enter") $("#go").click(); });

// ---- left pane: browse only. Deliberately does NOT touch the query box. ----
function loadDB(center){
  fetch(`/api/rows?center=${center}&n=25`).then(r=>r.json()).then(d=>{
    $("#dbview").innerHTML = (d.rows||[]).map(r =>
      `<div class="dbrow${r.idx===center?" hit":""}"><i>#${r.idx}</i>`
      + `<span>${escapeHtml(r.text)}</span></div>`).join("");
    const hit = document.querySelector("#dbview .hit");
    if (hit) hit.scrollIntoView({block:"center"});
  });
}
loadDB(40000);

let findTimer = null;
$("#find").addEventListener("input", () => {
  clearTimeout(findTimer);
  findTimer = setTimeout(() => {
    const q = $("#find").value.trim();
    if (/^\d+$/.test(q)) { const i = Math.min(+q, 65535);
      $("#hits").innerHTML = `<span class="muted">showing index ${i}</span>`; loadDB(i); return; }
    if (q.length < 2) { $("#hits").innerHTML = ""; return; }
    fetch("/api/find?q=" + encodeURIComponent(q)).then(r=>r.json()).then(d => {
      const hits = d.hits || [];
      $("#hits").innerHTML = hits.length
        ? `<span class="muted">${hits.length} match${hits.length>1?"es":""} — showing #${hits[0].idx}</span>`
        : `<span class="muted">${escapeHtml(d.msg||"no match")}</span>`;
      if (hits.length) loadDB(hits[0].idx);
    });
  }, 200);
});

// ---- right pane: the actual private query ----
$("#go").onclick = () => {
  if (busy) return;
  const idx = $("#idx").value.trim(); if (!idx) return;
  busy = true; $("#go").disabled = true;
  live = {}; running = null;
  $("#err").style.display="none"; $("#coldbox").style.display="none";
  $("#resultcard").style.display="none"; $("#cryptocard").style.display="none";
  $("#privcard").style.display="none";
  $("#timecard").style.display="block"; drawStages(null);
  $("#status").innerHTML = '<span class="badge b-warn">working…</span>';

  const t0 = Date.now();
  const tick = setInterval(() => { if (busy)
    $("#elapsed").textContent = ((Date.now()-t0)/1000).toFixed(1) + " s elapsed"; }, 100);

  const es = new EventSource("/api/query?idx=" + encodeURIComponent(idx));
  const done = () => { busy=false; clearInterval(tick); running=null; $("#go").disabled=false; es.close(); };

  es.onmessage = m => {
    const ev = JSON.parse(m.data);
    if (ev.ev === "stage") {                       // live from the device, as each stage completes
      live[ev.name] = ev.ms; running = ev.name; drawStages(null);
    } else if (ev.ev === "cold") {
      $("#coldbox").style.display = "block";
      $("#coldmsg").textContent = `Batch width ${ev.q} is not built in this backend process yet, so this `
        + `pass compiles and records it — roughly ${ev.est_s} s. Repeat queries at this width are fast.`;
    } else if (ev.ev === "result") {
      render(ev); done(); width = ev.q; guard();
      $("#status").innerHTML = '<span class="badge b-ok">backend ready</span>';
      $("#note").textContent = `65,536 rows · up to ${maxBatch} indices share one scan · built for `
        + `${width} ${width===1?"query":"queries"}`;
    } else if (ev.ev === "error") {
      showErr(ev.msg); done(); $("#timecard").style.display="none";
      $("#status").innerHTML = '<span class="badge b-bad">error</span>';
    }
  };
  es.onerror = () => { if (busy) { showErr("lost the event stream"); done(); } };
};

function render(ev){
  const T = ev.timings;
  live = {fold_rgsw:T.fold_rgsw, expand:T.expand, convert:T.convert,
          scan:T.scan, folds:T.folds, decrypt:T.decrypt};
  running = null; drawStages(ev.dev_ms);
  $("#elapsed").textContent = ev.q>1
    ? `${fmt(ev.wall_ms)} wall · ${fmt(ev.per_query_ms)}/query`
    : `${fmt(ev.wall_ms)} wall`;

  $("#resultcard").style.display="block";
  $("#verdict").innerHTML = ev.verified
    ? '<span class="badge b-ok">verified against db65536.txt</span>'
    : '<span class="badge b-bad">MISMATCH</span>';
  $("#rows").innerHTML = ev.rows.map(r =>
    `<div class="rowtext">${r.ok ? "" : "⚠ "}<span class="muted">#${r.idx}</span>  `
    + `${escapeHtml(r.text)}</div>`).join("");

  const c = ev.crypto || {};
  const qs = c.queries || [];
  const nums = a => (a||[]).join(", ");
  const ascii = a => (a||[]).map(v => (v>=32 && v<127) ? String.fromCharCode(v) : "·").join("");
  const many = qs.length > 1;
  // Per-query block. A batch is Q independent plaintexts/ciphertexts, so show each one; anything else
  // would misrepresent the scheme (nothing is packed together — only the scan is shared).
  const perq = fn => qs.map((q,i) =>
      `<div class="val" style="margin-top:${i?6:0}px">`
    + (many ? `<span class="muted">query ${i} · row #${q.idx} → columns ${q.col},${q.col+1}</span><br>` : "")
    + fn(q) + `</div>`).join("");

  $("#cryptocard").style.display="block";
  $("#crypto").innerHTML =
      `<div class="cs"><div class="lab">1 · plaintext quer${many?"ies":"y"} (client)`
    + `${many?` — ${qs.length} independent plaintexts, nothing packed together`:""}</div>`
    + perq(q => `2048 coefficients, all zero except <b>slot ${q.c0}</b> = <b>${c.slot_val}</b>`
        + `<br><span class="muted">slot = idx ÷ ${(65536/(c.n0||512))|0}; value = `
        + `(2<sup>${c.m_exp}</sup>)<sup>−1</sup> mod ${c.t}, cancelling the ×2<sup>${c.m_exp}</sup> the `
        + `expansion tree adds so each selector decrypts to 1</span>`)
    + `</div>`
    + `<div class="cs"><div class="lab">1b · remaining address bits — <i>not</i> in those polynomials`
    + `</div>` + perq(q => `${q.fold_bits || "—"} <span class="muted">· idx mod `
        + `${(65536/(c.n0||512))|0}, sent as separate RGSW ciphertexts, consumed by the folds</span>`)
    + `</div>`
    + `<div class="arrow">▼ encrypt (BFV, client-side, independently per query)</div>`
    + `<div class="cs"><div class="lab">2 · encrypted quer${many?"ies":"y"} sent to the server — first 10 `
    + `coefficients</div>` + perq(q => nums(q.ct_query)) + `</div>`
    + `<div class="arrow">▼ expand each into ${c.n0} selectors, then <b>ONE shared scan</b></div>`
    + `<div class="cs"><div class="lab">3 · the shared step</div><div class="val">`
    + `a single pass over all 65,536 rows answers ${many?`all ${qs.length} queries`:"the query"} — `
    + `<b>${c.cols_used} of ${c.cols_total}</b> matmul columns used (2 per query)`
    + (many ? `<br><span class="muted">this is why batching pays: the scan cost is fixed by the `
              + `database, so its per-query share falls as the batch grows</span>`
            : `<br><span class="muted">14 of 16 column pairs sit idle at batch size 1 — batching fills `
              + `them</span>`) + `</div></div>`
    + `<div class="arrow">▼ folds, then decrypt (per query again)</div>`
    + `<div class="cs"><div class="lab">4 · answer ciphertext${many?"s":""} — first 10 coefficients</div>`
    + perq(q => nums(q.ct_answer)) + `</div>`
    + `<div class="arrow">▼ decrypt (client-side, secret key)</div>`
    + `<div class="cs"><div class="lab">5 · decrypted coefficients — first 16, each &lt; t = ${c.t}</div>`
    + perq(q => `${nums(q.pt_out)}<br><span class="muted">as characters:</span> `
        + `"${escapeHtml(ascii(q.pt_out))}"`) + `</div>`
    + `<div class="arrow">▼ decode</div>`
    + `<div class="cs"><div class="lab">6 · the row${many?"s":""}</div>`
    + (ev.rows||[]).map((r,i) => `<div class="val" style="margin-top:${i?6:0}px">`
        + `<span class="muted">#${r.idx}</span> ${escapeHtml(r.text)}</div>`).join("") + `</div>`;

  $("#privcard").style.display="block";
  $("#touched").textContent = ev.rows_touched.toLocaleString();
  $("#clientms").textContent = fmt(ev.client_ms);
  $("#progs").textContent = ev.programs.total;
  $("#privnote").innerHTML = ev.programs.total === 0
    ? "The scan touched every row, so the access pattern reveals nothing. <b>0 programs compiled</b> — "
      + "this query replayed work recorded earlier, which is why it is fast."
    : `The scan touched every row. <b>${ev.programs.total} programs</b> were compiled and recorded on this `
      + "pass; later queries at this width replay them.";
}
</script>
</div></body></html>
"""


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    print(f"PIR UI on http://localhost:{PORT}   (backend socket {SOCK})")
    print(f"  forward it:  ssh -L {PORT}:localhost:{PORT} <this-vm>")
    Server(("127.0.0.1", PORT), H).serve_forever()
