#!/bin/bash
# Checks one arm's sandbox view (launch/sandbox.sh). Exit 0 only if every check passes.
#   bash launch/sandbox_check.sh <report file>
# Needs the campaign's environment (ARM, ASPIRE_V1, STUDY_LIB, STUDY_RUNS, ASPIRE_CLAUDE_CFG, UV_CACHE_DIR, HF_HOME,
# STUDY_HIDE_DIRS; skeptic also STUDY_GATE, STUDY_OVERLAY, OBS_TIMELINE_SEEDS).
# - The library seen inside must be STUDY_LIB (same content hash on both sides); read-only for skeptic.
# - Instead of naming paths that must be hidden, it lists what the sandbox shows in every STUDY_HIDE_DIRS entry:
#   only the entries that lead to the repository, the Claude config dir, the HF cache and the uv cache.
# - With simulator isolation on (STUDY_ISOLATE=1, the default) it also requires: third_party/, .libero and ~/.libero
#   empty, libero and robosuite not importable, the cap-harness forwarder in place, a call that names a hidden file
#   rejected, and one real LLM-free episode (seed 51, GPU STUDY_CHECK_GPU, default 2) run through the forwarder.
#   For skeptic the forwarded episode must come back with the observation timeline (the wrapper ran on the
#   simulation side).
set -u
REPORT=$1
HERE=$(cd "$(dirname "$0")" && pwd -P); PROJECT=$(dirname "$HERE")
libsha() { (cd "$1" && cat $(ls *.md | sort) | sha256sum | cut -c1-16); }
WANT_LIB=$(libsha "$STUDY_LIB")
REPO=$(readlink -f "$ASPIRE_V1")
ISO=${STUDY_ISOLATE:-1}
CHK=${TMPDIR:-/tmp}/study-sbx-check-$ARM; mkdir -p "$CHK"
# what each hidden root may still show: the first path component below it of every path mounted back
expect_in() {  # expect_in <hidden root>
  local root p rel; root=$(readlink -f "$1")
  for p in "$REPO" "$PROJECT" "$(readlink -m "$ASPIRE_CLAUDE_CFG")" "$(readlink -m "$HF_HOME")" "$(readlink -m "$UV_CACHE_DIR")"; do
    case "$p" in "$root"/*) rel=${p#"$root"/}; echo "${rel%%/*}" ;; esac
  done | sort -u | tr '\n' ' '
}
HIDE_LIST=""; for d in ${STUDY_HIDE_DIRS:-}; do [ -d "$d" ] && HIDE_LIST+="$(readlink -f "$d") "; done
rm -f "$STUDY_RUNS/.sbx_marker"
bash "$HERE/sandbox.sh" "$CHK" -- bash -c '
  ARM=$1 ISO=$2 HIDE_LIST=$3 PROJECT=$4
  SK=.claude/skills/iterative-debugging/skills
  for d in $HIDE_LIST; do echo "HIDDEN $d $(ls -A "$d" | sort | tr "\n" " ")"; done
  echo "HOMECLAUDE $(ls -A "$HOME/.claude" | tr "\n" " ")"
  echo "PROJECT $(ls -A "$PROJECT" | tr "\n" " ")"
  echo "GIT $(ls -A .git | tr "\n" " ")"
  echo "EXAMPLES $(ls -A examples | tr "\n" " ")"
  echo "AGENTSMD $(wc -c < AGENTS.md)"
  grep -q "^# ASPIRE: Agentic" README.md && echo "README_UPSTREAM"
  echo "OUTPUTS $(ls -A outputs | tr "\n" " ")"
  echo "CAMPAIGNDIR $(ls -A campaign | sort | tr "\n" " ")"
  echo "LIBSHA $(cd "$SK" && cat $(ls *.md | sort) | sha256sum | cut -c1-16)"
  if touch "$SK/.write_test" 2>/dev/null; then echo "LIB_WRITABLE"; rm -f "$SK/.write_test"; else echo "LIB_READONLY"; fi
  touch .venv-libero/.write_test 2>/dev/null && { echo "VENV_WRITABLE"; rm -f .venv-libero/.write_test; }
  touch src/.write_test 2>/dev/null && { echo "SRC_WRITABLE"; rm -f src/.write_test; }
  if [ "$ARM" = skeptic ]; then
    echo "GATEDIR $(ls -A campaign/gate | sort | tr "\n" " ")"
    if touch campaign/gate/.write_test 2>/dev/null; then echo "GATE_WRITABLE"; rm -f campaign/gate/.write_test; else echo "GATE_READONLY"; fi
    grep -q "^# cap-harness with the automatic observation timeline" .venv-libero/bin/cap-harness && echo "WRAPPER_OK"
    .venv-libero/bin/python -c "import cap_harness.artifacts as a; print(\"RECORDER\", a.DEPTH_KEYFRAME_CAMERA)"
    grep -q "obs_timeline.md" .claude/skills/iterative-debugging/subagent-prompt.md && echo "PROMPT_TIMELINE"
    grep -q "It ends with an \`end state\` block" .claude/skills/iterative-debugging/subagent-prompt.md && echo "PROMPT_END_STATE"
    echo "OBS_SEEDS ${OBS_TIMELINE_SEEDS:-}"
  else
    grep -q "^# cap-harness with the automatic observation timeline" .venv-libero/bin/cap-harness && echo "WRAPPER_TIMELINE"
    grep -q "obs_timeline.md" .claude/skills/iterative-debugging/subagent-prompt.md && echo "PROMPT_TIMELINE"
    .venv-libero/bin/python -c "import cap_harness.artifacts as a; print(\"RECORDER\", getattr(a, \"DEPTH_KEYFRAME_CAMERA\", \"release\"))"
  fi
  if [ "$ISO" = 1 ]; then
    echo "THIRDPARTY $(ls -A third_party | tr "\n" " ")"
    echo "DOTLIBERO $(ls -A .libero | tr "\n" " ")"
    echo "HOMELIBERO $(ls -A "$HOME/.libero" 2>/dev/null | tr "\n" " ")"
    .venv-libero/bin/python -c "import libero.libero" > /dev/null 2>&1 && echo "LIBERO_IMPORTABLE" || echo "LIBERO_BLOCKED"
    .venv-libero/bin/python -c "import robosuite" > /dev/null 2>&1 && echo "ROBOSUITE_IMPORTABLE" || echo "ROBOSUITE_BLOCKED"
    grep -q "^# Isolated sandbox: cap-harness forwarder" .venv-libero/bin/cap-harness && echo "SHIM_OK"
    CH=$PWD/.venv-libero/bin/cap-harness; C=/tmp/simcheck; rm -rf "$C"; mkdir -p "$C"
    printf "obs = get_observation()\nresult = {\"note\": \"isolated sandbox check\"}\n" > "$C/p.py"
    "$CH" run --benchmark libero-pro --suite libero_goal_swap --task-id 0 --seed 51 --program third_party/x.py \
      --output-root "$C/rej" > "$C/rej.log" 2>&1; echo "SIM_REJECT_RC $?"
    grep -q "not available in this environment" "$C/rej.log" && echo "SIM_REJECT_MSG"
    (cd "$C" && CUDA_VISIBLE_DEVICES=${STUDY_CHECK_GPU:-2} MUJOCO_EGL_DEVICE_ID=${STUDY_CHECK_GPU:-2} timeout 900 "$CH" run-batch \
      --benchmark libero-pro --suite libero_goal_swap --task-id 0 --seeds 51 --init-mode seeded --workers 1 --flat-run-dir \
      --no-videos --program p.py --output-root out --results-jsonl out/results.jsonl > run.log 2>&1); echo "SIM_RUN_RC $?"
    grep -Eq "\"program_ok\": ?true" "$C/out/results.jsonl" 2>/dev/null && echo "SIM_EPISODE_OK"
    [ -f "$C/out/results.jsonl" ] || tail -5 "$C/run.log" | sed "s/^/SIM_LOG /"
    if [ "$ARM" = skeptic ]; then   # the wrapper ran on the simulation side
      grep -q "^\[observation timeline\] 1 failed run" "$C/run.log" && echo "SIM_TIMELINE_LINE"
      ls "$C"/out/0051/run-*/obs_timeline.md > /dev/null 2>&1 && echo "SIM_TIMELINE_FILE"
    fi
    rm -rf "$C"
  fi
  echo "NOBYTECODE ${PYTHONDONTWRITEBYTECODE:-}"
  echo marker > outputs/runs/.sbx_marker && echo "RUNS_WRITE_OK"
  true' _ "$ARM" "$ISO" "$HIDE_LIST" "$PROJECT" > "$REPORT" 2>&1
rc=$?
echo "sandbox rc=$rc expected_lib=$WANT_LIB isolate=$ISO host marker: $(cat "$STUDY_RUNS/.sbx_marker" 2>&1)" >> "$REPORT"
ok=1
need() { grep -qx -- "$1" "$REPORT" || { ok=0; echo "MISSING: $1" >> "$REPORT"; }; }
deny() { grep -qx -- "$1" "$REPORT" && { ok=0; echo "UNEXPECTED: $1" >> "$REPORT"; }; }
[ $rc -eq 0 ] || ok=0
for d in $HIDE_LIST; do need "HIDDEN $d $(expect_in "$d")"; done
need "HOMECLAUDE "; need "PROJECT "; need "GIT "; need "EXAMPLES "; need "AGENTSMD 0"; need "README_UPSTREAM"; need "OUTPUTS runs "
need "LIBSHA $WANT_LIB"; need "NOBYTECODE 1"; need "RUNS_WRITE_OK"; deny "VENV_WRITABLE"; deny "SRC_WRITABLE"
if [ "$ARM" = skeptic ]; then
  need "CAMPAIGNDIR gate launch tasks "; need "LIB_READONLY"; need "GATE_READONLY"; need "GATEDIR obs_timeline.py promote.py "
  if [ "$ISO" = 1 ]; then need "SIM_TIMELINE_LINE"; need "SIM_TIMELINE_FILE"; else need "WRAPPER_OK"; fi
  need "RECORDER agentview"; need "PROMPT_TIMELINE"; need "PROMPT_END_STATE"; need "OBS_SEEDS ${OBS_TIMELINE_SEEDS:-51-65}"
else
  need "CAMPAIGNDIR launch tasks "; need "LIB_WRITABLE"; need "RECORDER release"; deny "WRAPPER_TIMELINE"; deny "PROMPT_TIMELINE"
fi
if [ "$ISO" = 1 ]; then
  need "THIRDPARTY "; need "DOTLIBERO "; need "HOMELIBERO "; need "LIBERO_BLOCKED"; need "ROBOSUITE_BLOCKED"; need "SHIM_OK"
  need "SIM_REJECT_RC 2"; need "SIM_REJECT_MSG"; need "SIM_RUN_RC 0"; need "SIM_EPISODE_OK"
fi
[ "$(cat "$STUDY_RUNS/.sbx_marker" 2>/dev/null)" = marker ] || ok=0
rm -f "$STUDY_RUNS/.sbx_marker"
echo "sandbox check $ARM: $([ $ok = 1 ] && echo PASS || echo FAIL)" >> "$REPORT"
[ $ok = 1 ]
