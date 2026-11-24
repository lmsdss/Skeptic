#!/usr/bin/env python3
"""Skill-library admission gate: only knowledge from a successful attempt enters the library.

An attempt (a task's first attempt or a phase-3 rerun) may change the library only if
  (1) its selected fix_code.py passed at least 8 of the 15 development seeds, counted here from the
      attempt's own finished runs of exactly that program (source/program.py in each run directory), and
  (2) its verifier returned `verified` or `refuted` (not `void`) and graded at least one proposal `sound`;
      only the `sound` proposals may be promoted. (Requiring `verified` would let one defective entry sink the
      whole verdict of an otherwise sound attempt.)
Inside the sandbox the library is mounted read-only. This script stages a copy for the coordinator to edit;
a daemon outside the sandbox (started by the batch job from its own copy of this file) re-checks the attempt
on disk, lints the staged copy and only then writes it into the library.

Inside the sandbox (the coordinator, from the repo root):
  python3 campaign/gate/promote.py eligible <TASK_DIR>     # development score + verdict: may it get a verifier?
  python3 campaign/gate/promote.py stage <TASK_DIR>        # a copy of the library to edit, if the attempt qualifies
  python3 campaign/gate/promote.py submit <STAGING_DIR>    # hand the edited copy over and wait for the result
Outside (the batch job):
  python3 promote.py daemon --lib DIR --runs DIR --repo DIR --log FILE
"""
import argparse
import datetime
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEV_PASS_MIN = 8
REPO = Path(__file__).resolve().parents[2]          # <repo>/campaign/gate/promote.py inside the sandbox
SKILLS_REL = Path(".claude/skills/iterative-debugging/skills")
LINT_REL = Path(".claude/skills/iterative-debugging/scripts/lint_skills.py")
RUN_DIR = re.compile(r"^run-\d{8}T\d{6}")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def library_sha(d):
    """Same digest as scripts/campaign.py skill_library_sha (README.md excluded)."""
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob("*.md") if x.name != "README.md"):
        h.update(str(p.relative_to(d)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def sha_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def program_runs(attempt, code_sha):
    """(seed, run name, task_success) for every finished run of exactly this program under attempt/debug/."""
    out = []
    for root, dirs, _ in os.walk(attempt / "debug"):
        for d in [d for d in dirs if RUN_DIR.match(d)]:
            r = Path(root) / d
            prog, outcome, run = r / "source" / "program.py", r / "outcome.json", r / "run.json"
            if not (prog.is_file() and outcome.is_file()) or sha_file(prog) != code_sha:
                continue
            try:
                seed = json.loads(run.read_text()).get("seed") if run.is_file() else int(Path(root).name)
                ok = bool(json.loads(outcome.read_text()).get("task_success"))
            except (ValueError, OSError):
                continue
            if isinstance(seed, int):
                out.append((seed, d, ok))
        dirs[:] = [d for d in dirs if not RUN_DIR.match(d)]   # never descend into a run directory
    return out


def check(run_root, attempt, need_verified):
    """Facts about one attempt and the reasons (if any) it may not change the library."""
    rel = attempt.relative_to(run_root)
    dev_seeds = json.loads((run_root / "campaign.json").read_text())["dev_seeds"]
    fix = attempt / "fix_code.py"
    facts = {"attempt": str(rel), "dev_seeds": len(dev_seeds), "dev_covered": 0, "dev_passes": 0}
    reasons = []
    if not fix.is_file():
        reasons.append("no fix_code.py in the attempt")
    else:
        latest = {}
        for seed, name, ok in program_runs(attempt, sha_file(fix)):
            if seed in dev_seeds and (seed not in latest or name > latest[seed][0]):
                latest[seed] = (name, ok)
        facts["dev_covered"] = len(latest)
        facts["dev_passes"] = sum(ok for _, ok in latest.values())
        missing = sorted(set(dev_seeds) - set(latest))
        if missing:
            reasons.append(f"fix_code.py (this exact file) has no finished run on development seeds {missing}")
        if facts["dev_passes"] < DEV_PASS_MIN:
            reasons.append(f"development {facts['dev_passes']}/{len(dev_seeds)} is below {DEV_PASS_MIN}/{len(dev_seeds)}")
    vdir = run_root / "verification" / "__".join(rel.parts)
    report = vdir / "verification_report.json"
    facts["verify_dir"] = str(vdir.relative_to(run_root))
    facts["verdict"] = None
    if report.is_file():
        try:
            v = json.loads(report.read_text())
            facts["verdict"] = v.get("verdict")
            facts["verify_passes"] = f"{v.get('passes')}/{v.get('trials')}"
            facts["sound_proposals"] = [p.get("title") for p in v.get("proposals", []) if p.get("disposition") == "sound"]
        except ValueError:
            facts["verdict"] = "unreadable"
    if need_verified:
        if facts["verdict"] not in ("verified", "refuted"):
            reasons.append(f"verifier verdict is {facts['verdict']!r} in {facts['verify_dir']}, not 'verified' or 'refuted'")
        elif not facts.get("sound_proposals"):
            reasons.append(f"the verifier graded no proposal 'sound' in {facts['verify_dir']}")
    return facts, reasons


def run_root_inside():
    latest = REPO / "outputs" / "runs" / "LATEST"
    if not latest.exists():
        sys.exit("no outputs/runs/LATEST: initialize the campaign first")
    return latest.resolve()


def attempt_path(run_root, arg):
    p = Path(arg)
    p = (p if p.is_absolute() else (Path.cwd() / p if (Path.cwd() / p).exists() else run_root / p)).resolve()
    try:
        p.relative_to(run_root)
    except ValueError:
        sys.exit(f"{arg} is not a task directory under {run_root}")
    if not p.is_dir():
        sys.exit(f"{arg}: no such task directory")
    return p


def cmd_eligible(a):
    run_root = run_root_inside()
    facts, reasons = check(run_root, attempt_path(run_root, a.task_dir), need_verified=False)
    print(json.dumps(facts, indent=1))
    missing = facts["dev_seeds"] - facts["dev_covered"]
    if reasons and missing and facts["dev_passes"] + missing >= DEV_PASS_MIN:
        # The score is not settled: the selected file itself was not swept on every development seed (typically an
        # edit after the sweep). The attempt can still qualify, from runs of exactly this file and nothing else.
        attempt = attempt_path(run_root, a.task_dir)
        parts = attempt.relative_to(run_root).parts          # <benchmark>/<suite>/task_<id>[/rerun_NN]
        seeds = " ".join(str(s) for s in json.loads((run_root / "campaign.json").read_text())["dev_seeds"])
        out = attempt / "debug" / "attempts" / "fix_code_sweep"
        print("NOT YET ELIGIBLE: " + "; ".join(reasons))
        print("The selected fix_code.py has no finished development run on the seeds above, so its score is unknown.")
        print("Do not edit fix_code.py. Sweep exactly that file on the development seeds, on a free slot <GPU> per the")
        print("ledger (about 15 minutes), then run this eligible command again; only its answer then counts:")
        print(f'  CUDA_VISIBLE_DEVICES=<GPU> MUJOCO_EGL_DEVICE_ID=<GPU> .venv-libero/bin/cap-harness run-batch \\\n'
              f'    --benchmark {parts[0]} --suite {parts[1]} --task-id {parts[2].split("_")[1]} --seeds {seeds} \\\n'
              f'    --program {attempt / "fix_code.py"} --output-root {out} \\\n'
              f'    --flat-run-dir --init-mode seeded --workers 4 --log-dir {out}/logs --results-jsonl {out}/results.jsonl')
        return 1
    if reasons:
        print("NOT ELIGIBLE: " + "; ".join(reasons))
        print("Do not dispatch a verifier for this attempt. Record its proposals in curriculum.md as not eligible.")
        return 1
    print(f"ELIGIBLE: development {facts['dev_passes']}/{facts['dev_seeds']}. A verifier may be dispatched.")
    return 0


def cmd_stage(a):
    run_root = run_root_inside()
    attempt = attempt_path(run_root, a.task_dir)
    facts, reasons = check(run_root, attempt, need_verified=True)
    if reasons:
        print(json.dumps(facts, indent=1))
        print("NOT ELIGIBLE: " + "; ".join(reasons))
        print("Nothing from this attempt may enter the library.")
        return 1
    pdir = run_root / "promotions"
    pdir.mkdir(exist_ok=True)
    n = 1 + max([int(p.name[:3]) for p in pdir.iterdir() if p.name[:3].isdigit()], default=0)
    sdir = pdir / f"{n:03d}-{'__'.join(attempt.relative_to(run_root).parts)}"
    lib = REPO / SKILLS_REL
    shutil.copytree(lib, sdir / "skills")
    for p in (sdir / "skills").rglob("*"):
        p.chmod(0o640 if p.is_file() else 0o750)
    request = {"attempt": facts["attempt"], "base_hash": library_sha(lib), "staged_at": now(), "facts": facts}
    (sdir / "request.json").write_text(json.dumps(request, indent=1) + "\n")
    print(f"STAGED {sdir}")
    print(f"Edit the library copy under {sdir}/skills as step 6c says (only proposals the verifier graded sound),")
    print(f"then run: python3 campaign/gate/promote.py submit {sdir}")
    return 0


def cmd_submit(a):
    sdir = Path(a.staging_dir).resolve()
    if not (sdir / "request.json").is_file():
        sys.exit(f"{sdir}: no request.json; create the staging directory with `promote.py stage`")
    (sdir / "result.json").unlink(missing_ok=True)
    (sdir / "READY").write_text(now() + "\n")
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        if (sdir / "result.json").is_file():
            res = json.loads((sdir / "result.json").read_text())
            print(("APPLIED" if res.get("applied") else "REJECTED") + "\n" + json.dumps(res, indent=1))
            return 0 if res.get("applied") else 1
        time.sleep(5)
    print(f"NO RESULT after {a.timeout // 60} min: the promotion daemon did not answer. Note it in curriculum.md "
          "and continue without promoting; submit again later.")
    return 2


# ---------------- daemon (outside the sandbox) ----------------

def lint_staged(repo, staged, lint):
    """Run lint_skills.py on the staged library: it reads skills/ next to its scripts/ and docs/, src/ at the repo root."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        base = root / ".claude/skills/iterative-debugging"
        (base / "scripts").mkdir(parents=True)
        shutil.copyfile(lint, base / "scripts" / "lint_skills.py")
        shutil.copytree(staged, base / "skills")
        for name in ("docs", "src"):
            (root / name).symlink_to(repo / name)
        r = subprocess.run([sys.executable, str(base / "scripts" / "lint_skills.py")], capture_output=True,
                           text=True, timeout=600, cwd=root)
        return r.returncode, (r.stdout + r.stderr)[-4000:]


def process(sdir, lib, repo, lint, log, diffs):
    run_root = sdir.parent.parent
    res = {"staging": sdir.name, "processed_at": now(), "applied": False}
    try:
        req = json.loads((sdir / "request.json").read_text())
        attempt = (run_root / req["attempt"]).resolve()
        attempt.relative_to(run_root.resolve())
        facts, reasons = check(run_root, attempt, need_verified=True)   # recomputed here, not taken from the request
        res["facts"] = facts
        cur = library_sha(lib)
        if req.get("base_hash") != cur:
            reasons.append(f"the library changed since staging ({req.get('base_hash')} -> {cur}): stage again")
        staged = sdir / "skills"
        changed, added = [], []
        for p in sorted(staged.iterdir()):
            if p.is_symlink() or not p.is_file() or p.suffix != ".md":
                reasons.append(f"{p.name}: only plain .md files may be staged")
            elif not (lib / p.name).exists():
                added.append(p.name)
            elif (lib / p.name).read_bytes() != p.read_bytes():
                changed.append(p.name)
        for t in sorted(lib.glob("*.md")):
            if not (staged / t.name).exists():
                reasons.append(f"{t.name} is missing from the staged copy: library files may not be deleted")
        if not reasons and not (changed or added):
            reasons.append("the staged copy is identical to the library: nothing to apply")
        if not reasons:
            rc, out = lint_staged(repo, staged, lint)
            if rc != 0:
                reasons.append("lint_skills.py failed on the staged copy:\n" + out)
        if reasons:
            res["reasons"] = reasons
        else:
            old = {n: (lib / n).read_text() if (lib / n).exists() else "" for n in changed + added}
            for n in changed + added:
                tmp = lib / f".{n}.gatetmp"
                shutil.copyfile(staged / n, tmp)
                os.replace(tmp, lib / n)
            res.update(applied=True, old_hash=cur, new_hash=library_sha(lib), changed=changed, added=added)
            with open(diffs / f"{sdir.name}.diff", "w") as f:
                for n in changed + added:
                    f.writelines(difflib.unified_diff(old[n].splitlines(True), (lib / n).read_text().splitlines(True),
                                                      f"a/{n}", f"b/{n}"))
    except Exception as exc:   # a malformed request must not stop the daemon
        res["reasons"] = [f"daemon error: {type(exc).__name__}: {exc}"]
    (sdir / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    (sdir / "READY").unlink(missing_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps({"run_root": run_root.name, **res}) + "\n")


def cmd_daemon(a):
    lib, runs, repo = Path(a.lib).resolve(), Path(a.runs).resolve(), Path(a.repo).resolve()
    lint = Path(a.lint).resolve() if a.lint else repo / LINT_REL
    log = Path(a.log).resolve()
    diffs = log.parent / "promotion-diffs"
    diffs.mkdir(parents=True, exist_ok=True)
    print(f"{now()} daemon: lib={lib} hash={library_sha(lib)} runs={runs}", flush=True)
    while True:
        seen = set()
        for ready in sorted(runs.glob("*/promotions/*/READY")):
            sdir = ready.parent.resolve()   # runs/LATEST is a symlink to a run root: each request once, by its real path
            if sdir in seen or not (sdir / "READY").exists():
                continue
            seen.add(sdir)
            process(sdir, lib, repo, lint, log, diffs)
            print(f"{now()} processed {sdir.parent.parent.name}/{sdir.name}: "
                  f"{json.loads((sdir / 'result.json').read_text())['applied']}", flush=True)
        time.sleep(a.poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eligible"); p.add_argument("task_dir"); p.set_defaults(fn=cmd_eligible)
    p = sub.add_parser("stage"); p.add_argument("task_dir"); p.set_defaults(fn=cmd_stage)
    p = sub.add_parser("submit"); p.add_argument("staging_dir"); p.add_argument("--timeout", type=int, default=1800)
    p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("daemon")
    for o in ("--lib", "--runs", "--repo", "--log"):
        p.add_argument(o, required=True)
    p.add_argument("--lint", default=None); p.add_argument("--poll", type=int, default=10)
    p.set_defaults(fn=cmd_daemon)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
