#!/bin/bash
# Agent sandbox: runs a command (the coordinator session, and with it every worker and verifier it dispatches) in a
# bubblewrap mount namespace that shows one arm's view of the agents' workspace (an ASPIRE v1.0 checkout).
#
#   launch/sandbox.sh <private tmp dir> -- <command ...>
#
# Environment (set by launch/campaign.sh from launch/config.sh):
#   ARM                aspire (the released ASPIRE protocol) | skeptic; not passed into the sandbox
#   ASPIRE_V1          the agents' workspace
#   STUDY_LIB            the arm's skill library, mounted over .claude/skills/iterative-debugging/skills:
#                        aspire:  read-write (the released protocol: the coordinator edits the library itself)
#                        skeptic: read-only  (only the promotion daemon, outside the sandbox, writes it)
#   STUDY_RUNS           the arm's runs dir for the current phase, mounted at outputs/runs
#   ASPIRE_CLAUDE_CFG  the arm's Claude Code config dir (transcripts)
#   UV_CACHE_DIR, HF_HOME   mounted read-only
#   STUDY_HIDE_DIRS      space-separated host directories hidden from the agents (e.g. the directory holding other
#                      runs, results and logs); every path above is mounted back explicitly. This project
#                      (prompts, method sources, launcher) is always hidden.
#   skeptic only: STUDY_GATE (promote.py + obs_timeline.py, read-only at campaign/gate), STUDY_OVERLAY (cap-harness,
#                 artifacts.py, subagent-prompt.md, each read-only over the workspace's file), OBS_TIMELINE_SEEDS
#   STUDY_ISOLATE        1 (default): simulator isolation, see below
#
# Hidden from both arms: this project, STUDY_HIDE_DIRS, the workspace's .git, examples/ (worked baseline solutions),
# AGENTS.md (real-robot rules), outputs/ apart from runs, and the user's own agent state in $HOME. The workspace's
# campaign/ shows only the task lists, the Claude settings file and (skeptic) the gate. The simulator venv, src/ and third_party/ are read-only.
# The two arms differ only in the skeptic mounts.
#
# Simulator isolation (STUDY_ISOLATE=1): the sandbox also hides third_party/ (LIBERO-Pro source, BDDL task files,
# robosuite and scene assets, the provider code), .libero, ~/.libero and the provider venvs, so no agent can read the
# simulator's own files. Its .venv-libero/bin/cap-harness is launch/cap-harness-shim, which hands each call to
# launch/simd.py, started here OUTSIDE the sandbox. simd.py runs the real cap-harness through this script with
# STUDY_SIM_MODE=1: the same view without the isolation mounts, the caller's working directory, and --die-with-parent.
set -e
SBX_TMP=$(readlink -m "$1"); shift; [ "$1" = -- ] && shift
H=$(readlink -f "$HOME")
REPO=$(readlink -f "$ASPIRE_V1"); CFG=$(readlink -m "$ASPIRE_CLAUDE_CFG")
LIB=$(readlink -f "$STUDY_LIB"); RUNS=$(readlink -m "$STUDY_RUNS")
UV=$(readlink -m "$UV_CACHE_DIR"); HF=$(readlink -m "$HF_HOME")
SKILLS=$REPO/.claude/skills/iterative-debugging/skills
SELF=$(readlink -f "$0"); HERE=$(dirname "$SELF"); PROJECT=$(dirname "$HERE")
LOGDIR=${STUDY_LOG_DIR:-$REPO/../study-logs}/${STUDY_TAG:-$ARM}/simd
[ -d "$LIB" ] || { echo "sandbox: library $LIB missing" >&2; exit 2; }
mkdir -p "$UV" "$HF"
HIDE=(--tmpfs "$PROJECT"); for d in ${STUDY_HIDE_DIRS:-}; do [ -d "$d" ] && HIDE+=(--tmpfs "$(readlink -f "$d")"); done
case "${ARM:-}" in
  aspire)
    ARM_MOUNTS=(--bind "$LIB" "$SKILLS") ;;
  skeptic)
    GATE=$(readlink -f "$STUDY_GATE"); OV=$(readlink -f "$STUDY_OVERLAY")
    for f in "$GATE/promote.py" "$GATE/obs_timeline.py" "$OV/cap-harness" "$OV/artifacts.py" "$OV/subagent-prompt.md"; do
      [ -f "$f" ] || { echo "sandbox: $f missing" >&2; exit 2; }
    done
    [ -x "$OV/cap-harness" ] || { echo "sandbox: $OV/cap-harness is not executable" >&2; exit 2; }
    ARM_MOUNTS=(--ro-bind "$LIB" "$SKILLS" --ro-bind "$GATE" "$REPO/campaign/gate"
      --ro-bind "$OV/cap-harness" "$REPO/.venv-libero/bin/cap-harness"
      --ro-bind "$OV/artifacts.py" "$REPO/src/cap_harness/artifacts.py"
      --ro-bind "$OV/subagent-prompt.md" "$REPO/.claude/skills/iterative-debugging/subagent-prompt.md"
      --setenv OBS_TIMELINE_SEEDS "${OBS_TIMELINE_SEEDS:-51-65}") ;;
  *) echo "sandbox: ARM must be aspire or skeptic" >&2; exit 2 ;;
esac
ISO_MOUNTS=(); SIM_OPTS=(--chdir "$REPO")   # agent sessions start in the workspace
if [ "${STUDY_SIM_MODE:-0}" = 1 ]; then
  # the simulation side of an isolated sandbox: same view, simulator files visible, started by simd.py
  SIM_OPTS=(--die-with-parent)
  [ -n "${STUDY_SIM_CWD:-}" ] && SIM_OPTS+=(--chdir "$STUDY_SIM_CWD")
elif [ "${STUDY_ISOLATE:-1}" = 1 ]; then
  SHIM=$HERE/cap-harness-shim
  [ -x "$SHIM" ] && [ -f "$HERE/simd.py" ] || { echo "sandbox: isolation files missing in $HERE" >&2; exit 2; }
  ISO_MOUNTS=(--tmpfs "$REPO/third_party" --tmpfs "$REPO/.libero")
  for v in .venv-curobo .venv-sam3 .venv-graspnet .venv-pyroki; do [ -d "$REPO/$v" ] && ISO_MOUNTS+=(--tmpfs "$REPO/$v"); done
  [ -d "$H/.libero" ] && ISO_MOUNTS+=(--tmpfs "$H/.libero")
  ISO_MOUNTS+=(--ro-bind "$SHIM" "$REPO/.venv-libero/bin/cap-harness" --setenv STUDY_SIM_SPOOL "/tmp/.simspool-$$")
  SPOOL=$SBX_TMP/.simspool-$$; rm -rf "$SPOOL"; mkdir -p "$SPOOL/req" "$SPOOL/run" "$LOGDIR"
  "$REPO/.venv-libero/bin/python" -I "$HERE/simd.py" --spool "$SPOOL" --sbx-tmp "$SBX_TMP" --parent $$ \
    --sandbox "$SELF" --cap-harness "$REPO/.venv-libero/bin/cap-harness" --repo "$REPO" \
    --runs-host "$RUNS" --runs-link "$REPO/outputs/runs" < /dev/null >> "$LOGDIR/simd-$(date +%Y%m%d-%H%M%S)-$$.log" 2>&1 &
  for _ in $(seq 150); do [ -s "$SPOOL/alive" ] && break; sleep 0.2; done
  [ -s "$SPOOL/alive" ] || { echo "sandbox: the simulation service did not start (see $LOGDIR)" >&2; exit 2; }
fi
mkdir -p "$CFG" "$SBX_TMP" "$RUNS"
EMPTY=$SBX_TMP.empty; : > "$EMPTY"   # an empty file shown in place of AGENTS.md (a /dev/null bind is unreadable on nodev file systems)
# Launcher variables (the arm, the run name, host paths of this project and the data dir) are not passed into the
# sandbox.
UNSET=(); for v in ARM STUDY_TAG STUDY_GATE STUDY_OVERLAY STUDY_LIB STUDY_RUNS STUDY_RUNS_BASE STUDY_DATA STUDY_HIDE_DIRS STUDY_LOG_DIR SKEPTIC_ROOT OLDPWD; do
  UNSET+=(--unsetenv "$v"); done
HOME_HIDE=()
for d in .claude .codex .cursor .cursor-server .config; do HOME_HIDE+=(--tmpfs "$H/$d"); done
for f in .claude.json .netrc .bash_history; do HOME_HIDE+=(--ro-bind /dev/null "$H/$f"); done
exec bwrap --dev-bind / / \
  "${HIDE[@]}" \
  --bind "$REPO" "$REPO" --ro-bind "$REPO/.venv-libero" "$REPO/.venv-libero" --ro-bind "$REPO/src" "$REPO/src" \
  --ro-bind "$REPO/third_party" "$REPO/third_party" \
  --tmpfs "$REPO/.git" --tmpfs "$REPO/examples" --ro-bind "$EMPTY" "$REPO/AGENTS.md" \
  --tmpfs "$REPO/campaign" --ro-bind "$PROJECT/tasks" "$REPO/campaign/tasks" \
  --ro-bind "$HERE/claude-settings.json" "$REPO/campaign/launch/claude-settings.json" \
  --bind "$CFG" "$CFG" --ro-bind "$HF" "$HF" \
  "${ARM_MOUNTS[@]}" "${ISO_MOUNTS[@]}" \
  --tmpfs "$REPO/outputs" --bind "$RUNS" "$REPO/outputs/runs" \
  --ro-bind "$UV" "$UV" --setenv UV_NO_CACHE 1 \
  "${HOME_HIDE[@]}" \
  --bind "$SBX_TMP" /tmp --setenv TMPDIR /tmp \
  --setenv CLAUDE_CONFIG_DIR "$CFG" --setenv PYTHONDONTWRITEBYTECODE 1 \
  "${UNSET[@]}" "${SIM_OPTS[@]}" \
  -- "$@"
