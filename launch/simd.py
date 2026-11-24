"""Simulation service for an isolated agent sandbox.

Why: inside the agent sandbox Qwen could read the simulator's own files (LIBERO-Pro source, BDDL task files,
robot and scene assets) because the simulator ran in the same sandbox. With isolation on, launch/sandbox.sh
hides third_party/, .libero and the provider venvs from the agent, mounts launch/cap-harness-shim over
.venv-libero/bin/cap-harness, and starts this service OUTSIDE the sandbox:

    python -I simd.py --spool <host spool dir> --sbx-tmp <SBX_TMP> --parent <pid of the sandbox> \
        --sandbox <launch/sandbox.sh> --cap-harness <repo>/.venv-libero/bin/cap-harness

The shim drops a request (argv, cwd, a few environment variables, its pid) into <spool>/req/. The service runs the
real cap-harness in a second sandbox that has the same view plus those files (sandbox.sh with STUDY_SIM_MODE=1),
so every path means the same thing on both sides, and streams stdout/stderr/exit code back through <spool>/run/<id>/.

Rules:
- only the cap-harness subcommands run, run-batch and doctor (and --help);
- no argument may name, or resolve to, a hidden tree (third_party, .libero, the provider venvs);
- only CUDA_VISIBLE_DEVICES, MUJOCO_EGL_DEVICE_ID and a few harmless variables pass from the agent; the rest of
  the environment is the job's, as before;
- the run is killed (whole process group) when the shim that asked for it exits or writes <run>/kill, and every
  run is killed when the sandbox (--parent) is gone.
The generated program itself was already confined by the harness (only `import math`, AST check, safe builtins).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ALLOWED_SUBCOMMANDS = {"run", "run-batch", "doctor"}
ALLOWED_ENV = ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID", "CAP_HARNESS_LOG_LEVEL", "PYTHONUNBUFFERED",
               "OMP_NUM_THREADS", "MKL_NUM_THREADS")
HIDDEN_MARKERS = ("third_party", "/.libero", "bddl", ".venv-curobo", ".venv-sam3", ".venv-graspnet", ".venv-pyroki",
                  "site-packages", "/proc/", "\x00")  # "/.libero", not ".libero": .venv-libero is fine


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}", flush=True)


class Service:
    def __init__(self, a: argparse.Namespace) -> None:
        self.spool = Path(a.spool)
        self.sbx_tmp = os.path.realpath(a.sbx_tmp)
        self.parent = a.parent
        self.sandbox = a.sandbox
        self.cap_harness = a.cap_harness
        self.repo = os.path.realpath(a.repo)
        self.runs_host = os.path.realpath(a.runs_host)
        self.runs_link = a.runs_link
        home = os.path.realpath(os.path.expanduser("~"))
        self.hidden_host = [os.path.join(self.repo, d) for d in
                            ("third_party", ".libero", ".venv-curobo", ".venv-sam3", ".venv-graspnet", ".venv-pyroki")]
        self.hidden_host.append(os.path.join(home, ".libero"))
        self.procs: dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        self.stop = False

    # ---- validation -------------------------------------------------------------------------------------------
    def to_host(self, p: str) -> str:
        """Map a path as the sandbox sees it to the host path (only /tmp and the runs link differ)."""
        if p == "/tmp" or p.startswith("/tmp/"):
            return self.sbx_tmp + p[4:]
        if self.runs_link and (p == self.runs_link or p.startswith(self.runs_link + "/")):
            return self.runs_host + p[len(self.runs_link):]
        return p

    def check(self, req: dict) -> str | None:
        argv = req.get("argv")
        if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
            return "malformed request"
        if not argv or argv[0] in ("-h", "--help"):
            return None
        if argv[0] not in ALLOWED_SUBCOMMANDS:
            return f"cap-harness {argv[0]!r} is not available in this environment (run, run-batch, doctor only)"
        cwd = req.get("cwd") or "/"
        if self.hidden(cwd) or any(m in cwd.lower() for m in HIDDEN_MARKERS):
            return "the working directory is not available in this environment"
        for raw in argv[1:]:
            vals = [raw] + ([raw.split("=", 1)[1]] if raw.startswith("-") and "=" in raw else [])
            for v in vals:
                low = v.lower()
                if any(m in low for m in HIDDEN_MARKERS) or low.startswith(".libero") or (v and self.hidden(v, cwd)):
                    return f"argument {raw!r} points into files that are not available in this environment"
        return None

    def hidden(self, p: str, cwd: str = "/") -> bool:
        """True if p (as the sandbox sees it, relative to cwd) resolves into a hidden tree on the host."""
        host = os.path.realpath(self.to_host(os.path.normpath(os.path.join(cwd, p))))
        return any(host == h or host.startswith(h + "/") for h in self.hidden_host)

    # ---- running ----------------------------------------------------------------------------------------------
    def start(self, rid: str, req: dict, rundir: Path) -> None:
        reason = self.check(req)
        if reason:
            (rundir / "err").write_text(f"cap-harness (isolated sandbox): {reason}\n")
            (rundir / "out").write_text("")
            (rundir / "rc").write_text("2")
            log(f"REJECT {rid} argv={req.get('argv')} cwd={req.get('cwd')}: {reason}")
            return
        env = dict(os.environ)
        for k in ALLOWED_ENV:
            env.pop(k, None)
        for k, v in (req.get("env") or {}).items():
            if k in ALLOWED_ENV and isinstance(v, str):
                env[k] = v
        env["STUDY_SIM_MODE"] = "1"
        env["STUDY_SIM_CWD"] = req.get("cwd") or "/"
        cmd = ["bash", self.sandbox, self.sbx_tmp, "--", self.cap_harness, *req["argv"]]
        out = open(rundir / "out", "wb")
        err = open(rundir / "err", "wb")
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env, start_new_session=True)
        out.close(); err.close()
        with self.lock:
            self.procs[rid] = p
        log(f"START {rid} pid={p.pid} shim={req.get('pid')} argv={req['argv']} cwd={req.get('cwd')} "
            f"env={ {k: v for k, v in (req.get('env') or {}).items() if k in ALLOWED_ENV} }")
        threading.Thread(target=self.watch, args=(rid, req, rundir, p), daemon=True).start()

    def kill(self, p: subprocess.Popen) -> None:
        for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                return
            try:
                p.wait(wait)
                return
            except subprocess.TimeoutExpired:
                continue

    def watch(self, rid: str, req: dict, rundir: Path, p: subprocess.Popen) -> None:
        shim = int(req.get("pid") or 0)
        t0 = time.time()
        why = None
        while p.poll() is None:
            if (rundir / "kill").exists():
                why = "kill requested"
            elif shim and not alive(shim):
                why = f"shim {shim} gone"
            elif self.stop:
                why = "service stopping"
            if why:
                self.kill(p)
                break
            time.sleep(1)
        rc = p.wait()
        rc = rc if rc >= 0 else 128 - rc  # killed by a signal: report it the way a shell does
        (rundir / "rc").write_text(str(rc))
        with self.lock:
            self.procs.pop(rid, None)
        log(f"END {rid} rc={rc} {time.time() - t0:.0f}s{' (' + why + ')' if why else ''}")

    def loop(self) -> None:
        req_dir = self.spool / "req"; run_dir = self.spool / "run"
        req_dir.mkdir(parents=True, exist_ok=True); run_dir.mkdir(parents=True, exist_ok=True)
        log(f"up: spool={self.spool} sbx_tmp={self.sbx_tmp} parent={self.parent} runs={self.runs_link} -> {self.runs_host}")
        last_beat = 0.0
        while not self.stop:
            if self.parent and not alive(self.parent):
                log(f"sandbox {self.parent} gone; stopping")
                break
            now = time.time()
            if now - last_beat >= 2:
                tmp = self.spool / ".alive.tmp"; tmp.write_text(f"{now:.0f}\n"); tmp.replace(self.spool / "alive")
                last_beat = now
            for f in sorted(req_dir.glob("*.json")):
                rid = f.stem
                rundir = run_dir / rid
                try:
                    req = json.loads(f.read_text())
                    rundir.mkdir(parents=True, exist_ok=True)
                    f.replace(rundir / "req.json")
                except Exception as e:  # noqa: BLE001
                    log(f"bad request {f.name}: {e}")
                    f.unlink(missing_ok=True)
                    continue
                try:
                    self.start(rid, req, rundir)
                except Exception as e:  # noqa: BLE001
                    (rundir / "err").write_text(f"cap-harness (isolated sandbox): service error: {e}\n")
                    (rundir / "rc").write_text("3")
                    log(f"ERROR {rid}: {e}")
            time.sleep(0.25)
        self.stop = True
        with self.lock:
            procs = list(self.procs.values())
        for p in procs:
            self.kill(p)
        log("down")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie still answers kill(0): treat it as gone
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spool", required=True)
    ap.add_argument("--sbx-tmp", required=True)
    ap.add_argument("--parent", type=int, required=True)
    ap.add_argument("--sandbox", required=True)
    ap.add_argument("--cap-harness", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--runs-host", required=True)
    ap.add_argument("--runs-link", required=True)
    a = ap.parse_args()
    try:  # die with the shell that started us (it becomes bwrap through exec, so the pid stays the same)
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        pass
    svc = Service(a)
    signal.signal(signal.SIGTERM, lambda *_: setattr(svc, "stop", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(svc, "stop", True))
    svc.loop()


if __name__ == "__main__":
    sys.exit(main())
