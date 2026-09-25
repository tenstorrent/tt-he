#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Start the whole PIR browser demo with one command.

    python3 host/examples/pir_ui.py                 # start everything, print the URL
    python3 host/examples/pir_ui.py --port 9000     # different port
    python3 host/examples/pir_ui.py --warm 2        # pre-pay a batch width other than 1
    python3 host/examples/pir_ui.py --reset         # tt-smi -r first (use if the device is wedged)
    python3 host/examples/pir_ui.py --stop          # stop backend and frontend
    python3 host/examples/pir_ui.py --restart       # force a fresh backend (repays the warm-up)

It supervises the two processes described in docs/pir/pir_ui_plan.md:

  backend   pir_demo65536 --serve <sock>    owns the Tenstorrent device, stays warm
  frontend  tools/pir_ui/serve.py <port>    serves the browser UI

Ctrl-C stops the FRONTEND but deliberately leaves the BACKEND running, because the backend's warm-up
is per process (~5 s at width 1, and minutes at larger batch widths). Restarting the UI should be free.
Use --stop when you actually want the device released.
"""
import argparse, os, signal, socket, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN_CANDIDATES = ["build/pir_demo65536", "../build/pir_demo65536", "/tmp/cfgreal/pir_demo65536"]
DATA = ["reference/pir/db65536.pir", "reference/pir/db65536.fold",
        "reference/pir/db65536.txt", "reference/pir/query65536.bin"]
FRONTEND = "tools/pir_ui/serve.py"


def say(msg, kind=""):
    tag = {"ok": "  ok  ", "err": " FAIL ", "warn": " note ", "": "      "}[kind]
    print(f"[{tag}] {msg}", flush=True)


def newest_binary():
    """Pick the most recently built binary and say which.

    There is usually more than one build tree here, and running a stale one is a genuinely confusing
    failure: the code looks fixed and the behaviour is not."""
    found = [(os.path.getmtime(p), p) for p in
             (os.path.join(ROOT, c) if not c.startswith("/") else c for c in BIN_CANDIDATES)
             if os.path.exists(p)]
    if not found:
        return None, []
    found.sort(reverse=True)
    return found[0][1], [p for _, p in found]


def backend_state(sock):
    """('ready'|'starting'|'down', detail) without disturbing a running backend."""
    if not os.path.exists(sock):
        try:
            with open(sock + ".state") as f:
                if f.read().strip() == "starting":
                    return "starting", "socket not up yet"
        except OSError:
            pass
        return "down", "no socket"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(3)
    try:
        s.connect(sock)
        s.sendall(b"status\n")
        data = s.recv(4096)
        return ("ready", data.decode(errors="replace").strip()) if data else ("starting", "no reply")
    except socket.timeout:
        return "starting", "accepted but silent (warming)"
    except OSError as e:
        return "down", str(e)
    finally:
        s.close()


def pids(pattern):
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def stop_all(sock, port):
    for pat, name in ((f"pir_demo65536 --serve {sock}", "backend"), (f"{FRONTEND} {port}", "frontend")):
        got = pids(pat)
        for p in got:
            try:
                os.kill(p, signal.SIGTERM)          # backend traps this and closes the device cleanly
            except ProcessLookupError:
                pass
        if got:
            say(f"stopped {name} (pid {', '.join(map(str, got))})", "ok")
        else:
            say(f"{name} was not running")
    time.sleep(3)
    for leftover in (sock, sock + ".state", sock + ".lock"):
        try:
            os.unlink(leftover)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description="start the PIR browser demo")
    ap.add_argument("--port", type=int, default=8071)
    ap.add_argument("--sock", default="/tmp/pir_demo.sock")
    ap.add_argument("--warm", default="1", help="batch width to pre-pay at boot (one width only)")
    ap.add_argument("--reset", action="store_true", help="tt-smi -r before starting")
    ap.add_argument("--stop", action="store_true", help="stop both processes and exit")
    ap.add_argument("--restart", action="store_true", help="kill any backend and start a fresh one")
    ap.add_argument("--log", default="/tmp/pir_serve.log")
    a = ap.parse_args()
    os.chdir(ROOT)

    if a.stop:
        stop_all(a.sock, a.port)
        return 0

    # ---- preflight: fail with something actionable, not a stack trace -----------------------------
    binary, all_bins = newest_binary()
    if not binary:
        say("no pir_demo65536 binary found. Build it first:", "err")
        say("    cmake --build build --target pir_demo65536 -j8")
        return 2
    say(f"binary   {binary}", "ok")
    if len(all_bins) > 1:
        others = ", ".join(b for b in all_bins if b != binary)
        say(f"also present (older, ignored): {others}", "warn")

    missing = [d for d in DATA if not os.path.exists(os.path.join(ROOT, d))]
    if missing:
        say(f"missing data files: {', '.join(missing)}", "err")
        say("    generate with reference/pir/gen_query65536.py and encode_db")
        return 2
    gb = sum(os.path.getsize(os.path.join(ROOT, d)) for d in DATA) / 2**30
    say(f"data     {len(DATA)} files, {gb:.1f} GB", "ok")

    if a.reset:
        say("tt-smi -r ...")
        subprocess.run(["tt-smi", "-r"], capture_output=True)
        time.sleep(15)

    if a.restart:
        for p in pids(f"pir_demo65536 --serve {a.sock}"):
            os.kill(p, signal.SIGTERM)
        time.sleep(5)

    # ---- backend: reuse a warm one if it is there. The device is exclusive; never start a second. --
    state, detail = backend_state(a.sock)
    if state == "ready":
        say("backend already running and warm — reusing it (no warm-up to repay)", "ok")
    else:
        if pids(f"pir_demo65536 --serve {a.sock}") and state == "down":
            say("a backend process exists but its socket is gone; use --restart", "err")
            return 2
        if state == "down":
            log = open(a.log, "w")
            # env.sh defines shell functions, so it has to be sourced by a shell.
            cmd = (f'source "{ROOT}/env.sh" >/dev/null 2>&1; '
                   f'exec "{binary}" --serve "{a.sock}" --warm "{a.warm}"')
            subprocess.Popen(["bash", "-c", cmd], stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)          # survives this launcher exiting
            say(f"backend starting (log: {a.log})")
        say("waiting for the device: 6 GB DB residency + key prep + warm-up ...")
        t0 = time.time()
        while time.time() - t0 < 900:
            state, detail = backend_state(a.sock)
            if state == "ready":
                break
            if state == "down" and time.time() - t0 > 60:
                say("backend exited during startup. Last lines of its log:", "err")
                try:
                    print("".join(open(a.log).readlines()[-15:]))
                except OSError:
                    pass
                say("if the device is wedged, retry with --reset")
                return 2
            print(f"\r        {time.time()-t0:5.0f}s  {state} ...", end="", flush=True)
            time.sleep(2)
        print()
        if state != "ready":
            say("backend did not become ready in 15 minutes", "err")
            return 2
        say(f"backend ready after {time.time()-t0:.0f}s", "ok")

    # ---- frontend ---------------------------------------------------------------------------------
    for p in pids(f"{FRONTEND} {a.port}"):
        os.kill(p, signal.SIGTERM)                            # cheap to restart; always take the new code
        time.sleep(1)
    fe = subprocess.Popen([sys.executable, os.path.join(ROOT, FRONTEND), str(a.port)])
    time.sleep(1.5)
    if fe.poll() is not None:
        say(f"frontend exited immediately (port {a.port} already in use?)", "err")
        return 2

    say(f"UI on http://localhost:{a.port}", "ok")
    print()
    print(f"    forward it:  ssh -L {a.port}:localhost:{a.port} {os.uname().nodename}")
    print(f"    backend log: {a.log}")
    print()
    print("    Ctrl-C stops the UI and LEAVES the backend warm, so restarting is instant.")
    print(f"    Release the device with:  python3 host/examples/pir_ui.py --stop")
    print()

    try:
        fe.wait()
    except KeyboardInterrupt:
        say("stopping the UI; backend stays warm", "ok")
        fe.terminate()
        try:
            fe.wait(timeout=5)
        except subprocess.TimeoutExpired:
            fe.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
