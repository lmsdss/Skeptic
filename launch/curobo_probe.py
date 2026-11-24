#!/usr/bin/env python3
"""Real cuRobo health probe: one small joint-space plan from the Panda home pose (the trajopt path).

A cuRobo service whose CUDA context has died keeps answering /openapi.json (so the supervisor and a plain health
probe see it as up) but fails every plan with "CUDA error: an illegal memory access". This sends a real plan.
  curobo_probe.py [port] [timeout_s]
Exit 0: planned. 1: the service is broken or unreachable. 2: the service rejected the probe request (4xx), i.e. the
probe itself is wrong, not the service. Prints one line.
"""
import json
import sys
import time
import urllib.error
import urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8118
timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 120
home = [0.0, -0.161, 0.0, -2.4446, 0.0, 2.2268, 0.7854]   # the LIBERO Panda reset pose
goal = list(home)
goal[0] += 0.3
goal[3] += 0.2
body = {
    "base_frame": "robot_base", "model": "panda", "arm": "primary",
    "joint_positions": {"primary": home},
    "joint_names": {"primary": [f"robot0_joint{i}" for i in range(1, 8)]},
    "base_transforms": {"primary": [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]},
    "end_effector_links": {"primary": "panda_hand"},
    "target": {"type": "joints", "joint_positions": goal},
}
req = urllib.request.Request(f"http://127.0.0.1:{port}/plan", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    print(f"ok {time.time() - t0:.1f}s")
    sys.exit(0)
except urllib.error.HTTPError as e:
    msg = " ".join(e.read().decode(errors="replace").split())[:200]
    print(f"http {e.code} after {time.time() - t0:.1f}s: {msg}")
    sys.exit(2 if 400 <= e.code < 500 else 1)
except Exception as e:  # noqa: BLE001 - refused, reset, timeout: all mean the service is not usable
    print(f"{type(e).__name__} after {time.time() - t0:.1f}s: {e}")
    sys.exit(1)
