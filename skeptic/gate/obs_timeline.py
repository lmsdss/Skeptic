"""Observation timeline of a failed run, with an end-state block.

Built only from what the run already recorded and its program could read: the program's calls and their results
(trace/events.jsonl, trace/calls/), the robot's own readings at every step (episode/steps.jsonl), and the agentview
depth images at the keyframes (media/keyframes/; the recorder stores one per motion or gripper call). No
simulator internals, no task files, no goal. Objects are named only by the text prompts the program itself segmented.
The timeline ends with an "end state" block: for each tracked object, whether it is still at its starting
place, where the hand last released it (proprio) and what the depth camera shows there in the last view with the hand
away (vision), and over which other tracked object's starting footprint that point lies. It says nothing about the goal.
Up to 6 tracked objects, since the Long tasks name many objects.

Thresholds: the API's own (gripper tolerance 0.05 of the opening, 0.04 m opening per finger; a motion's status as the
API reports it), the gripper geometry (robosuite panda_gripper.xml, see PAD_*) and the physical floors frozen in a
calibration beforehand (depth change 1 cm, vacated share 0.2). Every fact is tagged [proprio] (robot's own
readings or API results), [vision] (depth camera measurement) or [inferred] (an explanation, not a measurement).

  python3 obs_timeline.py run <run dir>...
      writes obs_timeline.md into each run dir and prints one summary line per run
  python3 obs_timeline.py after-cli --since <epoch s> -- <cap-harness arguments>
      called by the cap-harness wrapper after `run` / `run-batch`: does the same for every failed development
      run the command just wrote (seeds OBS_TIMELINE_SEEDS, default 51-65; never validation or held-out output roots) and
      appends the summary to <output root>/obs_summary.md
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

GRIP_TOL = 0.05            # API gripper tolerance (fraction of the opening)
FINGER_MAX_M = 0.04        # API: gripper position 1.0 = each finger 0.04 m from the centre
DEPTH_STEP_M = 0.01        # floor: a depth change below 1 cm is noise
VAC_FRAC = 0.2             # floor: an object left its place when >= 20 % of its pixels now show a farther surface
FAR_HAND_M = 0.20          # a depth view counts as "before the approach" when the hand was this far from the grasp
LIFT_M = 0.03              # the hand rose at least this much while holding
# Finger pad region around the hand point (robosuite panda_gripper.xml, the gripper LIBERO uses): the fingers close
# along the hand's y axis; each pad is 1.6 cm wide along the hand's x axis and spans 1.16 cm behind to 0.44 cm beyond
# the hand point along the approach axis (hand z); the fingertips end 1.1 cm beyond it.
PAD_SIDE_M, PAD_BACK_M, PAD_FRONT_M, TIP_FRONT_M = 0.008, 0.0116, 0.0044, 0.011
MAX_OBJECTS = 6
MOTION = {"move_to_pose", "move_to_joints", "go_home", "execute_trajectory", "move_synchronized", "step"}
GRIPPER = {"open_gripper", "close_gripper", "set_gripper", "set_grippers"}
END_RADIUS_M = 0.04        # end state: surfaces within this xy distance of a release point
FOOT_MARGIN_M = 0.01       # end state: margin around an object's starting footprint (xy)


# ---------------------------------------------------------------- small helpers
def arr(v):
    if isinstance(v, dict) and "$array_inline" in v:
        return np.asarray(v["$array_inline"], float).reshape(v.get("shape") or -1)
    if isinstance(v, list):
        return np.asarray(v, float)
    return None


def quat_R(q):
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def fmt(v, n=3):
    return "(" + ", ".join(f"{x:.{n}f}" for x in v) + ")"


def gap_mm(w):
    return 2 * FINGER_MAX_M * float(w) * 1000


def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def seeds_from(spec):
    out = set()
    for t in spec.replace(",", " ").split():
        a, _, b = t.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


# ---------------------------------------------------------------- one run
class Run:
    def __init__(self, d):
        self.d = Path(d)
        self.outcome = load_json(self.d / "outcome.json") or {}
        self.meta = load_json(self.d / "run.json") or {}
        steps_n = int(self.outcome.get("steps_executed") or 0)
        sim_s = float(self.outcome.get("simulator_duration_s") or 0)
        self.dt = sim_s / steps_n if steps_n and sim_s else 0.05
        reset = load_json(self.d / "episode/reset.json") or {}
        obs = reset.get("observation") or {}
        self.state = {}  # step -> (position, quaternion, gripper)
        self._put(0, obs.get("robot_state"))
        try:
            with open(self.d / "episode/steps.jsonl") as f:
                for line in f:
                    r = json.loads(line)
                    self._put(int(r.get("step_index", 0)), r.get("robot_state"))
        except (OSError, ValueError):
            pass
        self.last_step = max(self.state) if self.state else 0
        cam = (obs.get("cameras") or {}).get("agentview") or {}
        self.K = np.asarray(cam["intrinsics"], float) if cam.get("intrinsics") else None
        pose = cam.get("camera_pose") or {}
        self.T = None
        if self.K is not None and pose.get("position") and pose.get("quaternion_wxyz"):
            self.T = np.eye(4)
            self.T[:3, :3] = quat_R(arr(pose["quaternion_wxyz"]))
            self.T[:3, 3] = arr(pose["position"])
        self._spans()
        self._keyframes()
        self._dcache = {}
        self.table_z = self._table_z()

    def _put(self, step, rs):
        if not isinstance(rs, dict):
            return
        try:
            ee = rs["end_effector_poses"]["primary"]
            self.state[step] = (arr(ee["position"]), arr(ee["quaternion_wxyz"]), float(rs["gripper_positions"]["primary"]))
        except (KeyError, TypeError, ValueError):
            pass

    def at(self, step):
        """Robot state after `step` (the closest recorded step at or before it)."""
        if step in self.state:
            return self.state[step]
        keys = [k for k in self.state if k <= step]
        return self.state[max(keys)] if keys else (self.state[min(self.state)] if self.state else None)

    def step_of(self, t):
        return int(round(float(t or 0) / self.dt))

    def _spans(self):
        spans, order = {}, []
        try:
            lines = (self.d / "trace/events.jsonl").read_text().splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            sid = e.get("span_id")
            if e.get("event") == "span_start":
                spans[sid] = {"id": sid, "name": e.get("name"), "cat": e.get("category"), "parent": e.get("parent_id"),
                              "s0": self.step_of(e.get("simulator_time_s"))}
                order.append(sid)
            elif e.get("event") == "span_end" and sid in spans:
                spans[sid]["s1"] = self.step_of(e.get("simulator_time_s"))
                spans[sid]["ok"] = e.get("ok")
        self.spans = [spans[s] for s in order]
        self.top = [s for s in self.spans if s["parent"] is None and s["cat"] == "program"]

    def call(self, span, part):
        """input.json / output.json of a span (trace/calls/call-<span number>-<name>)."""
        try:
            n = int(str(span["id"]).rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return None
        hits = list((self.d / "trace/calls").glob(f"call-{n:06d}-*"))
        return load_json(hits[0] / f"{part}.json") if hits else None

    def _keyframes(self):
        self.kf = []
        root = self.d / "media/keyframes"
        for k in sorted(root.iterdir()) if root.is_dir() else []:
            m = load_json(k / "metadata.json") or {}
            a = (m.get("cameras") or {}).get("agentview") or {}
            f = a.get("depth") or a.get("depth_mm")
            if f:
                self.kf.append({"step": int(m.get("step_index", 0)), "label": m.get("label", k.name), "path": k / f})

    def depth(self, kf):
        p = kf["path"]
        if p not in self._dcache:
            if p.suffix == ".npy":
                D = np.load(p).astype(np.float32)
            else:
                import imageio.v2 as imageio
                D = imageio.imread(p).astype(np.float32) / 1000.0
            D[~np.isfinite(D)] = 0
            self._dcache = {p: D}  # one frame at a time keeps memory flat
        return self._dcache[p]

    def depth_at(self, step):
        """Latest depth keyframe at or before `step`."""
        c = [k for k in self.kf if k["step"] <= step]
        return c[-1] if c else None

    def cloud(self, D, rows=None, cols=None):
        if rows is None:
            rows, cols = np.nonzero(D > 0)
        z = D[rows, cols].astype(np.float64)
        ok = (z > 0.05) & (z < 5)
        rows, cols, z = rows[ok], cols[ok], z[ok]
        K = self.K
        P = np.stack([(cols - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z], axis=1)
        return P @ self.T[:3, :3].T + self.T[:3, 3], rows, cols

    def _table_z(self):
        if not self.kf or self.T is None:
            return None
        D = self.depth(self.kf[0])
        rows, cols = np.mgrid[0:D.shape[0]:4, 0:D.shape[1]:4]
        P, _, _ = self.cloud(D, rows.ravel(), cols.ravel())
        P = P[(np.abs(P[:, 2]) < 0.5) & (np.abs(P[:, 1]) < 1.0) & (P[:, 0] > -0.5) & (P[:, 0] < 1.5)]
        if len(P) < 100:
            return None
        h, e = np.histogram(P[:, 2], bins=np.arange(-0.5, 0.5, 0.005))
        i = int(np.argmax(h))
        return float(np.median(P[(P[:, 2] >= e[i]) & (P[:, 2] < e[i + 1]), 2]))


# ---------------------------------------------------------------- tracked objects (the program's own prompts)
class Tracked:
    def __init__(self, label, rows, cols, d0, box, ref_step):
        self.label, self.rows, self.cols, self.d0, self.box, self.ref_step = label, rows, cols, d0, box, ref_step
        self.left_at = None

    def vacated(self, D):
        d1 = D[self.rows, self.cols]
        ok = (d1 > 0) & (self.d0 > 0)
        if ok.sum() < 20:
            return None
        return float(((d1 - self.d0) > DEPTH_STEP_M)[ok].mean())

    def contains(self, r, c):
        x0, y0, x1, y1 = self.box
        return x0 <= c <= x1 and y0 <= r <= y1


def tracked_objects(run):
    out, seen = [], set()
    if run.T is None or run.table_z is None:
        return out
    for s in run.spans:
        if s["name"] != "segment_text" or len(out) >= MAX_OBJECTS:
            continue
        o = run.call(s, "output") or {}
        segs = [g for g in o.get("segmentations") or [] if str(g.get("camera_name", "")).endswith("agentview")]
        if not o.get("ok") or not segs:
            continue
        g = max(segs, key=lambda g: float(g.get("score") or 0))
        label = str((o.get("diagnostics") or {}).get("query") or g.get("label") or "object")
        if label in seen or not g.get("box_xyxy"):
            continue
        kf = run.depth_at(s["s0"])
        if kf is None:
            continue
        D = run.depth(kf)
        x0, y0, x1, y1 = (int(round(v)) for v in g["box_xyxy"])
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, D.shape[1] - 1), min(y1, D.shape[0] - 1)
        rows, cols = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        P, rows, cols = run.cloud(D, rows.ravel(), cols.ravel())
        keep = P[:, 2] > run.table_z + DEPTH_STEP_M
        if keep.sum() < 30:
            continue
        st = run.at(kf["step"])
        if st is not None and np.min(np.linalg.norm(P[keep] - st[0], axis=1)) < 0.08:
            continue  # the hand is in the box in this view; wait for a later segmentation of the same prompt
        rows, cols = rows[keep], cols[keep]
        step = max(1, len(rows) // 4000)
        seen.add(label)
        out.append(Tracked(label, rows[::step], cols[::step], D[rows[::step], cols[::step]].copy(), (x0, y0, x1, y1), kf["step"]))
    return out


# ---------------------------------------------------------------- geometry checks
def view_before(run, point, step):
    """Latest depth keyframe at or before `step` taken while the hand was >= FAR_HAND_M from `point`."""
    for k in reversed([k for k in run.kf if k["step"] <= step]):
        st = run.at(k["step"])
        if st is not None and np.linalg.norm(st[0] - point) >= FAR_HAND_M:
            return k
    return None  # every view has the hand nearby: it would be mistaken for the scene


def grasp_geometry(run, objs, c, q, step):
    """Scene surface around the closing fingers, seen before the approach."""
    kf = view_before(run, c, step)
    if kf is None or run.T is None or run.table_z is None:
        return None
    P, rows, cols = run.cloud(run.depth(kf))
    near = (np.linalg.norm(P - c, axis=1) < 0.15) & (P[:, 2] > run.table_z + 0.005)
    P, rows, cols = P[near], rows[near], cols[near]
    R = quat_R(q)
    res = {"kf": kf, "n_between": 0, "nearest": None}
    if len(P) == 0:
        return res
    rel = P - c
    a, b, h = rel @ R[:, 1], rel @ R[:, 0], rel @ R[:, 2]  # closing axis, pad width, approach
    out = np.stack([np.maximum(np.abs(a) - FINGER_MAX_M, 0), np.maximum(np.abs(b) - PAD_SIDE_M, 0),
                    np.maximum(np.maximum(h - PAD_FRONT_M, -PAD_BACK_M - h), 0)], axis=1)
    dist = np.linalg.norm(out, axis=1)  # distance to the region the open finger pads sweep when they close
    between = dist == 0
    res["n_between"] = int(between.sum())
    if between.any():
        res["span_mm"] = float((a[between].max() - a[between].min()) * 1000)
        labels = [o.label for r, cc in zip(rows[between], cols[between]) for o in objs if o.contains(r, cc)]
        res["obj"] = max(set(labels), key=labels.count) if labels else None
    col = (np.abs(a) <= FINGER_MAX_M) & (np.abs(b) <= PAD_SIDE_M) & (h > PAD_FRONT_M)
    if col.any():  # first surface straight ahead of the pads along the approach
        res["ahead_h"] = float(h[col].min())
    i = int(np.argmin(np.where(between, np.inf, dist))) if (~between).any() else None
    if i is not None:
        res["nearest"] = {"dist": float(dist[i]), "a": float(a[i]), "b": float(b[i]), "h": float(h[i]), "z": float(P[i, 2]),
                          "obj": next((o.label for o in objs if o.contains(rows[i], cols[i])), None)}
    return res


def obstacle(run, reached, q, w, target, step):
    """First surface the hand's outline (finger pads and tips at their opening `w`, swept from the reached point toward
    the target) meets, in a depth view taken while the hand was far away (so the hand itself is not mistaken for it).
    None: no such view (not measured); otherwise {"kf", "hit"} with hit None when the path is clear in that view."""
    kf = view_before(run, reached, step)
    if kf is None or run.T is None:
        return None
    seg = np.asarray(target, float) - reached
    L = float(np.linalg.norm(seg))
    if L < 0.02:
        return {"kf": kf, "hit": None, "short": True}
    u = seg / L
    P, _, _ = run.cloud(run.depth(kf))
    P = P[np.linalg.norm(P - reached, axis=1) < L + 0.12]
    R = quat_R(q)
    jw = FINGER_MAX_M * float(w) + 0.01  # outer faces of the fingers
    best = None
    for a in np.linspace(-jw, jw, 5):
        for b in (-PAD_SIDE_M, 0.0, PAD_SIDE_M):
            for h in (-PAD_BACK_M, 0.0, TIP_FRONT_M):
                rel = P - (reached + a * R[:, 1] + b * R[:, 0] + h * R[:, 2])
                s = rel @ u
                hit = (s > -0.005) & (s < L) & (np.linalg.norm(rel - np.outer(s, u), axis=1) < 0.01)
                if hit.sum() >= 5:
                    k = np.argsort(s[hit])[2]  # third-nearest hit: ignore single stray pixels
                    if best is None or s[hit][k] < best[0]:
                        best = (float(s[hit][k]), float(P[hit][k][2]))
    return {"kf": kf, "hit": None if best is None else {"dist": max(0.0, best[0]), "z": best[1]}}


def moved_since(objs, view_step, step):
    """Tracked objects seen leaving their place after the depth view at `view_step` and by `step`: that view is stale there."""
    return [(o.label, o.left_at) for o in objs if o.left_at is not None and view_step < o.left_at <= step]


def stale_note(objs, view_step, step):
    m = moved_since(objs, view_step, step)
    return ("; that view predates " + ", ".join(f"'{lb}' moving at step {s}" for lb, s in m) +
            ", so the scene there may have changed since") if m else ""


# ---------------------------------------------------------------- timeline
def api_result(o):
    if not isinstance(o, dict):
        return "", None
    err = o.get("error") or {}
    msg = (err.get("message") or "")[:160] if isinstance(err, dict) else str(err)[:160]
    st = o.get("status") or ("ok" if o.get("ok") else "failed")
    return f"API {'ok' if o.get('ok') else 'NOT ok'}, status {st}" + (f", error: {msg}" if msg else ""), o.get("ok")


def episode_over(o):
    """The call failed because the episode no longer accepts actions (its step limit was reached)."""
    err = (o or {}).get("error") if isinstance(o, dict) else None
    msg = (err.get("message") or "") if isinstance(err, dict) else str(err or "")
    return "terminated episode" in msg


def start_cloud(run, o):
    """A tracked object's points at its reference view (its stored pixels and their depth then)."""
    z = o.d0.astype(np.float64)
    ok = (z > 0.05) & (z < 5)
    rows, cols, z = o.rows[ok], o.cols[ok], z[ok]
    K = run.K
    P = np.stack([(cols - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z], axis=1)
    return P @ run.T[:3, :3].T + run.T[:3, 3]


def view_after(run, point, step):
    """Latest depth keyframe at or after `step` taken while the hand was >= FAR_HAND_M (xy) from `point`."""
    for k in reversed(run.kf):
        if k["step"] < step:
            break
        st = run.at(k["step"])
        if st is not None and float(np.linalg.norm(st[0][:2] - point[:2])) >= FAR_HAND_M:
            return k
    return None


def end_state(run, objs, releases, holding, held=()):
    """Where each tracked object finished. Facts only: its starting place, the hand's last release point for it
    and the surface the depth camera shows there afterwards, and which other tracked object's starting footprint that
    point lies over. No goal, no judgement of success."""
    run.end_short = []
    if not objs or run.T is None or run.K is None or not run.kf:
        return []
    last = run.kf[-1]
    out = ["", f"end state (last depth view: step {last['step']}; footprints are each object's starting box in x-y, "
               f"+/-{FOOT_MARGIN_M * 100:.0f} cm):"]
    feet = {}
    for o in objs:
        P = start_cloud(run, o)
        if len(P) < 10:
            continue
        lo, hi = np.percentile(P[:, :2], 5, axis=0), np.percentile(P[:, :2], 95, axis=0)
        feet[o.label] = (lo - FOOT_MARGIN_M, hi + FOOT_MARGIN_M, float(np.percentile(P[:, 2], 98)),
                         np.median(P[:, :2], axis=0))
    D_last = run.depth(last)
    for o in objs:
        f = feet.get(o.label)
        start = (f"started at x-y ({f[3][0]:.3f}, {f[3][1]:.3f}), top z {f[2]:.3f}" if f else "starting place not measured")
        v = o.vacated(D_last)
        rel = [r for r in releases if r["obj"] == o.label]
        if v is not None and v <= VAC_FRAC and o.left_at is None:
            out.append(f"  '{o.label}': {start}; still at its starting place in the last view [vision]")
            run.end_short.append(f"'{o.label}' still at its start")
            continue
        moved = (f"left its starting place at step {o.left_at}" if o.left_at is not None else
                 (f"{v * 100:.0f}% of its starting pixels show a farther surface in the last view" if v is not None else
                  "its starting pixels are not visible in the last view"))
        if not rel:
            held_end = holding is not None and o.label in (holding.get("carried"), holding.get("obj"))
            d1 = D_last[o.rows, o.cols]
            ok = (d1 > 0) & (o.d0 > 0)
            dz = (f"; its starting pixels in the last view: median depth change {float(np.median(d1[ok] - o.d0[ok])) * 100:+.1f} cm "
                  f"(+ = farther from the camera) [vision]" if ok.sum() >= 20 else "")
            if held_end and run.at(run.last_step) is not None:
                tail = "; still held at the end of the episode, hand at " + fmt(run.at(run.last_step)[0]) + " [proprio]"
                short = "still held at the end"
            elif o.label in held:
                tail = "; the jaws closed on it but no open followed while holding it, so where it went is not measured"
                short = "moved, no release recorded"
            else:
                tail = "; no grasp by the hand was attributed to it [inferred]"
                short = "moved, no grasp attributed"
            out.append(f"  '{o.label}': {start}; {moved} [vision]{dz}{tail}")
            run.end_short.append(f"'{o.label}' {short}")
            continue
        r = rel[-1]
        p = r["hand"]
        over = [lab for lab, (lo, hi, top, _) in feet.items()
                if lab != o.label and lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]]
        if over:
            where = "over the starting footprint of " + ", ".join(f"'{lab}' (its top z {feet[lab][2]:.3f})" for lab in over)
        else:
            dists = {lab: float(np.linalg.norm(np.maximum(np.maximum(lo - p[:2], p[:2] - hi), 0)))
                     for lab, (lo, hi, top, _) in feet.items() if lab != o.label}
            near = min(dists, key=dists.get) if dists else None
            where = ("over no other tracked object's starting footprint" +
                     (f" (nearest: '{near}', {dists[near] * 100:.1f} cm outside it)" if near else ""))
        line = (f"  '{o.label}': {start}; {moved} [vision]; last released at step {r['step']} with the hand at {fmt(p)}"
                + (f", {(p[2] - run.table_z) * 100:.1f} cm above the table plane" if run.table_z is not None else "")
                + f", {where} [proprio]")
        kf, below, cond = view_after(run, p, r["step"]), np.inf, ""
        if kf is None:  # no view with the hand away: the last view after the release, counting only surfaces below the hand
            later = [k for k in run.kf if k["step"] >= r["step"]]
            kf, below = (later[-1] if later else None), float(p[2]) - 0.02
            cond = " (the hand is still above that point in this view; only surfaces at least 2 cm below the hand point counted)"
        if kf is None:
            line += "; no depth view after the release: the surface there not measured [vision]"
        else:
            D = run.depth(kf)
            rows, cols = np.mgrid[0:D.shape[0]:2, 0:D.shape[1]:2]
            Q, _, _ = run.cloud(D, rows.ravel(), cols.ravel())
            near = Q[(np.linalg.norm(Q[:, :2] - p[:2], axis=1) <= END_RADIUS_M) & (Q[:, 2] < below) &
                     (Q[:, 2] > (run.table_z if run.table_z is not None else -1e9) + DEPTH_STEP_M)]
            if len(near) < 10:
                line += (f"; depth view at step {kf['step']}{cond}: no surface above the table plane within {END_RADIUS_M * 100:.0f} cm "
                         f"(x-y) of that point [vision]")
            else:
                top = float(np.percentile(near[:, 2], 95))
                line += (f"; depth view at step {kf['step']}{cond}: within {END_RADIUS_M * 100:.0f} cm (x-y) of that point the highest "
                         f"surface is at z {top:.3f}" + (f", {(top - run.table_z) * 100:.1f} cm above the table plane"
                                                        if run.table_z is not None else "") + " [vision]")
        out.append(line)
        run.end_short.append(f"'{o.label}' released " + (("over '" + "', '".join(over) + "'") if over else "over no tracked object"))
    return out


def build(run_dir):
    run = Run(run_dir)
    objs = tracked_objects(run)
    lines, events = [], []
    holding = None  # dict while the jaws hold something
    releases = []   # every open while holding: {"obj", "step", "hand"}
    held = set()    # tracked objects the jaws closed on at least once
    calls = list(run.top)
    max_steps = (run.meta.get("capture") or {}).get("max_steps")
    ended_at, after_end = None, {}  # step at which the episode stopped accepting actions; calls made after that
    i = 0
    while i < len(calls):
        sp = calls[i]
        name = sp["name"]
        if name not in MOTION | GRIPPER:
            i += 1
            continue
        s0 = sp["s0"]
        if name == "step":  # group consecutive step calls
            j = i
            while j + 1 < len(calls) and calls[j + 1]["name"] == "step":
                j += 1
            s1 = calls[j].get("s1", s0)
            n = j - i + 1
            head = f"#{s0}-{s1} step x{n}"
            i = j + 1
        else:
            s1 = sp.get("s1", s0)
            n = 1
            head = f"#{s0}-{s1} {name}"
            i += 1
        if ended_at is not None:  # nothing ran: list these once at the end instead of as failures
            after_end[name] = after_end.get(name, 0) + n
            continue
        st0, st1 = run.at(s0), run.at(s1)
        if st0 is None or st1 is None:
            continue
        facts, evs = [], []

        def ev(cat, tag, text):
            evs.append({"cat": cat, "tag": tag, "text": text, "step": s0, "fn": name})

        inp, out = run.call(sp, "input"), run.call(sp, "output")
        api, ok = api_result(out)
        over = episode_over(out)
        if over:
            ended_at = s1
            ev("episode_ended", "proprio", f"the episode stopped accepting actions at step {s1}"
               + (f" (this run's limit: max_steps {max_steps})" if max_steps else "")
               + f"; {name} " + ("was cut off there" if s1 > s0 else "did not run") + "; later motion and gripper calls did nothing")
        if name in MOTION:
            target = None
            if name == "move_to_pose":
                a0 = (inp or {}).get("args") or []
                pose = a0[0] if a0 and isinstance(a0[0], dict) else ((inp or {}).get("kwargs") or {}).get("pose")
                if isinstance(pose, dict) and pose.get("position"):
                    target = arr(pose["position"])
            moved = float(np.linalg.norm(st1[0] - st0[0]))
            if target is not None:
                off = float(np.linalg.norm(target - st1[0]))
                facts.append(f"target {fmt(target)}, reached {fmt(st1[0])}, {off * 100:.1f} cm from the target [proprio]")
            else:
                facts.append(f"hand {fmt(st0[0])} -> {fmt(st1[0])} ({moved * 100:.1f} cm) [proprio]")
            if api:
                facts.append(api + " [proprio]")
            if ok is False and name != "step" and not over:
                status = (out or {}).get("status") or "failed"
                if s1 > s0:
                    ev("motion_unfinished", "proprio", f"{name} did not complete ({status})" + (
                        f": stopped {off * 100:.1f} cm short of its target" if target is not None else ""))
                else:
                    ev("motion_unfinished", "proprio", f"{name} did not run ({status}): no simulator step was taken, the hand "
                       "did not move" + (f"; it is {off * 100:.1f} cm from the target" if target is not None else ""))
                if target is not None:
                    ob = obstacle(run, st1[0], st1[1], st1[2], target, s0)
                    if ob is None:
                        facts.append(f"path to the target not measured: no depth view with the hand at least "
                                     f"{FAR_HAND_M * 100:.0f} cm away [vision]")
                    elif not ob.get("short"):
                        vs, hit = ob["kf"]["step"], ob["hit"]
                        note = stale_note(objs, vs, s0)
                        if hit:
                            facts.append(f"depth view at step {vs}: the hand's outline, moved on from where it stopped toward the "
                                         f"target, meets a surface {hit['dist'] * 100:.1f} cm ahead, at z {hit['z']:.3f}{note} [vision]")
                            if s1 > s0 and hit["dist"] <= 0.015:
                                ev("blocked", "inferred", f"the hand probably stopped against a surface at z {hit['z']:.3f}"
                                   + (" (seen in a view taken before the scene changed)" if note else ""))
                        else:
                            facts.append(f"depth view at step {vs}: no surface on the way from where the hand stopped to the "
                                         f"target{note} [vision]")
            if holding is not None:
                ws = [run.state[k][2] for k in sorted(run.state) if s0 < k <= s1]
                zs = [run.state[k][0][2] for k in sorted(run.state) if s0 < k <= s1]
                if zs:
                    holding["zmax"] = max(holding["zmax"], max(zs))
                lost = next((k for k in sorted(run.state) if s0 < k <= s1 and run.state[k][2] <= GRIP_TOL), None)
                if lost is not None:
                    ev("slip", "proprio", f"the jaws closed from {gap_mm(holding['w']):.1f} mm to {gap_mm(run.state[lost][2]):.1f} mm "
                                          f"at step {lost} without an open: the held object left the fingers (hand at {fmt(run.state[lost][0])})")
                    holding = None
                elif ws:
                    facts.append(f"still holding, jaws {gap_mm(min(ws)):.1f}-{gap_mm(max(ws)):.1f} mm [proprio]")
        elif over:
            facts.append(api + " [proprio]")
        else:  # gripper
            diag = (out or {}).get("diagnostics") or {}
            w = float(diag.get("final_position", st1[2]))
            target_w = diag.get("target_position")
            closing = name == "close_gripper" or (target_w is not None and float(target_w) < st0[2] - GRIP_TOL)
            if closing:
                comp = diag.get("completion") or ("target_reached" if w <= GRIP_TOL else "?")
                c, q = st0[0], st0[1]
                if w <= GRIP_TOL:
                    facts.append(f"jaws closed to {gap_mm(w):.1f} mm (API completion: {comp}) at hand {fmt(c)} [proprio]")
                    ev("empty_close", "proprio", f"the jaws closed to {gap_mm(w):.1f} mm: nothing wider than that was between the fingers")
                    holding = None
                else:
                    facts.append(f"jaws stopped at {gap_mm(w):.1f} mm (API completion: {comp}) at hand {fmt(c)}: "
                                 f"something about {gap_mm(w):.0f} mm wide is between the fingers [proprio]")
                    holding = {"w": w, "z0": float(c[2]), "zmax": float(c[2]), "step": s1, "obj": None, "left0": None}
                gg = grasp_geometry(run, objs, c, q, s0)
                if gg is None and run.kf:
                    facts.append(f"no depth view with the hand at least {FAR_HAND_M * 100:.0f} cm away before this close: "
                                 f"surfaces around the fingers not measured [vision]")
                if gg is not None:
                    txt = f"depth view before the approach (keyframe at step {gg['kf']['step']}): "
                    nb, nr = gg["n_between"], gg.get("nearest")
                    if nb:
                        txt += (f"{nb} surface points lay in the region the finger pads close over, spanning {gg['span_mm']:.0f} mm "
                                f"along the closing axis" + (f", in the box of '{gg['obj']}'" if gg.get("obj") else ""))
                        if holding is not None:
                            holding["obj"] = gg.get("obj")
                            if gg.get("obj"):
                                held.add(gg.get("obj"))
                    else:
                        txt += "no surface lay in the region the finger pads close over"
                        if gg.get("ahead_h") is not None:
                            txt += (f"; straight ahead of the pads, the first surface is {gg['ahead_h'] * 100:+.1f} cm along the approach "
                                    f"(the pads reach +{PAD_FRONT_M * 100:.1f} cm)")
                            if w <= GRIP_TOL:
                                evs[-1]["text"] += f"; first surface straight ahead of the pads at {gg['ahead_h'] * 100:+.1f} cm along the approach"
                        if nr:
                            where = (f"nearest surface {nr['dist'] * 100:.1f} cm outside that region: {nr['a'] * 100:+.1f} cm along the "
                                     f"closing axis, {nr['b'] * 100:+.1f} cm to the side, {nr['h'] * 100:+.1f} cm along the approach")
                            txt += (f"; {where} (the pads reach +{PAD_FRONT_M * 100:.1f} cm, the fingertips +{TIP_FRONT_M * 100:.1f} cm), "
                                    f"at z {nr['z']:.3f}" + (f", in the box of '{nr['obj']}'" if nr["obj"] else ""))
                            if w <= GRIP_TOL:
                                evs[-1]["text"] += "; " + where + (f" ('{nr['obj']}')" if nr["obj"] else "")
                    facts.append(txt + stale_note(objs, gg["kf"]["step"], s0) + " [vision]")
            else:
                facts.append(f"jaws opened to {gap_mm(w):.1f} mm at hand {fmt(st1[0])}" + (f"; {api}" if ok is False else "") + " [proprio]")
                if holding is not None:
                    releases.append({"obj": holding.get("carried") or holding.get("obj"), "step": s1, "hand": st0[0]})
                    facts.append(f"release: the hand rose {(holding['zmax'] - holding['z0']) * 100:.1f} cm while holding and "
                                 f"let go at z {st0[0][2]:.3f}" + (f", {(st0[0][2] - run.table_z) * 100:.1f} cm above the table plane" if run.table_z is not None else "") + " [proprio]")
                    holding = None
        # tracked objects after this call
        kf = run.depth_at(s1)
        if objs and kf is not None and kf["step"] > s0:
            D = run.depth(kf)
            for o in objs:
                if o.left_at is not None or kf["step"] <= o.ref_step:
                    continue
                v = o.vacated(D)
                if v is not None and v > VAC_FRAC:
                    o.left_at = s1
                    if holding is not None and name in MOTION:
                        facts.append(f"'{o.label}' left its place with the hand ({v * 100:.0f}% of its pixels now show a farther surface) [vision]")
                        if holding.get("carried") is None:   # the first object that left with the hand is the one carried
                            holding["carried"] = o.label
                            held.add(o.label)
                        if holding.get("obj") == o.label:
                            holding["left0"] = s1
                    else:
                        ev("moved_while_not_held", "vision", f"'{o.label}' left its place during this call while the jaws held nothing "
                                                            f"({v * 100:.0f}% of its pixels now show a farther surface)")
        if holding is not None and holding.get("obj") and name in MOTION and holding["zmax"] - holding["z0"] >= LIFT_M:
            o = next((o for o in objs if o.label == holding["obj"]), None)
            if o is not None and o.left_at is None and not holding.get("flagged"):
                holding["flagged"] = True
                ev("not_lifted", "vision", f"the hand rose {(holding['zmax'] - holding['z0']) * 100:.1f} cm with the jaws stopped at "
                                           f"{gap_mm(holding['w']):.0f} mm, but '{o.label}' (nearest the fingers at the close) stayed in place")
        lines.append(head + (" | " + " | ".join(facts) if facts else ""))
        for e in evs:
            lines.append(f"  !! {e['cat']} [{e['tag']}]: {e['text']}")
            events.append(e)
    if after_end:
        lines.append(f"after step {ended_at}: " + ", ".join(f"{k} x{v}" for k, v in after_end.items())
                     + " called after the episode had ended; none of them ran [proprio]")
    oc = run.outcome
    end = [f"end: task_success {oc.get('task_success')}, {oc.get('termination_reason')}, {oc.get('steps_executed')} steps"]
    if oc.get("error"):
        err = oc["error"]
        end.append(f"error: {(err.get('message') if isinstance(err, dict) else str(err))[:200]}")
    for o in objs:
        end.append(f"'{o.label}': " + (f"left its starting place at step {o.left_at} [vision]" if o.left_at is not None else
                                       "still at its starting place at the last depth view [vision]"))
    end += end_state(run, objs, releases, holding, held)
    head = [f"# Observation timeline: seed {run.meta.get('seed')} ({run.meta.get('suite')} task {run.meta.get('task_id')})",
            "Built automatically from this run's own records: the program's calls and results (trace/), the robot's readings "
            "(episode/steps.jsonl) and the agentview depth images (media/keyframes/). No simulator internals, no goal.",
            "Tags: [proprio] robot readings and API results; [vision] depth-camera measurement; [inferred] an explanation, "
            "not a measurement. Positions: robot base frame, metres; 'hand' is the end-effector point the API reports. "
            "Jaw gap: distance between the finger pads. Around a close, offsets are in the hand's own axes: along the "
            "closing axis, to the side (pad width) and along the approach (+ = beyond the hand point, away from the palm). "
            "#a-b: simulator steps of the call. Objects: the prompts this program segmented (agentview). The last block, "
            "'end state', says where each tracked object finished.",
            f"table plane z: {run.table_z:.3f}" if run.table_z is not None else "table plane: unknown",
            f"events: {', '.join(e['cat'] + ' #' + str(e['step']) for e in events) if events else 'none'}",
            f"tracked objects: {', '.join(repr(o.label) for o in objs) if objs else 'none'}", ""]
    return run, head + lines + [""] + end, events


def summary(run, events):
    oc = run.outcome
    s = f"seed {run.meta.get('seed')}: {oc.get('termination_reason')}"
    if events:
        parts = []
        for e in events[:6]:
            t = e["text"]
            parts.append(f"{e['cat']} #{e['step']} ({t[:150]})")
        s += " | " + " | ".join(parts) + (f" | +{len(events) - 6} more" if len(events) > 6 else "")
    else:
        s += " | no physical event detected"
    if getattr(run, "end_short", None):
        s += " | end: " + "; ".join(run.end_short)
    return s


def process(run_dir):
    try:
        run, lines, events = build(run_dir)
        (Path(run_dir) / "obs_timeline.md").write_text("\n".join(lines) + "\n")
        return summary(run, events)
    except Exception as exc:  # noqa: BLE001 - a broken timeline must never break the caller
        return f"{run_dir}: timeline failed ({type(exc).__name__}: {exc})"


# ---------------------------------------------------------------- entry points
def arg_value(args, flag):
    for k, a in enumerate(args):
        if a == flag and k + 1 < len(args):
            return args[k + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def after_cli(since, args):
    if not args or args[0] not in ("run", "run-batch"):
        return
    root = arg_value(args, "--output-root")
    if not root:
        return
    root = Path(root).resolve()
    if {"validation", "holdout"} & set(root.parts):
        return
    seeds = seeds_from(os.environ.get("OBS_TIMELINE_SEEDS", "51-65"))
    runs = []
    for oc in list(root.glob("outcome.json")) + list(root.glob("*/outcome.json")) + list(root.glob("*/*/outcome.json")):
        d = oc.parent
        try:
            if oc.stat().st_mtime < since - 5 or (d / "obs_timeline.md").exists():
                continue
            o, m = json.loads(oc.read_text()), json.loads((d / "run.json").read_text())
        except (OSError, ValueError):
            continue
        if o.get("task_success") is not True and int(m.get("seed", -1)) in seeds:
            runs.append((int(m["seed"]), str(d)))
    if not runs:
        return
    runs.sort()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(4, len(runs))) as ex:
        lines = list(ex.map(process, [d for _, d in runs]))
    rel = [os.path.relpath(d, root) for _, d in runs]
    out = [f"[observation timeline] {len(runs)} failed run(s) in {root} ({time.time() - t0:.0f} s); "
           f"each run's full timeline: <run dir>/obs_timeline.md"]
    out += [f"  {ln}  [{r}/obs_timeline.md]" for ln, r in zip(lines, rel)]
    print("\n".join(out), flush=True)
    with open(root / "obs_summary.md", "a") as f:
        f.write("\n".join(out) + "\n")


def main(argv):
    if len(argv) >= 2 and argv[0] == "run":
        for d in argv[1:]:
            print(process(d))
    elif len(argv) >= 3 and argv[0] == "after-cli" and argv[1] == "--since":
        rest = argv[3:]
        if rest and rest[0] == "--":
            rest = rest[1:]
        after_cli(float(argv[2]), rest)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
