#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=5-00:00:00
# One study on all 80 LIBERO-Pro tasks for one arm, with an open-weight model (served locally by vLLM) as
# coordinator, workers and verifiers, on one node with 4 x H100. The agents work in the ASPIRE v1.0 workspace
# ($ASPIRE_V1, made by setup.sh) and follow its iterative-debugging protocol.
#
#   bash launch/campaign.sh <arm> [run index]      (from this project's root, inside a 4-GPU allocation, or sbatch)
#
#   arm = skeptic  Skeptic: the observation timeline with its end-state block, the worker template that explains it,
#                  and the skill-library admission gate (promotion daemon outside the sandbox)
#   arm = aspire   the released ASPIRE v1.0 protocol (the coordinator edits the library itself)
#
# Both arms run with the same model, hardware layout, sandbox, simulator isolation, seeds, budgets, order rule and
# rerun threshold, and both start from the v1.0 release skill library (hash 91d636a706623f4e). As in the paper the
# study runs in two phases, each its own campaign (own run root, progress file and phase-3 rerun sweep):
#   A  the 60 Object, Goal and Spatial tasks (position + task perturbation)  tasks/libero_pro_A60.json
#   B  the 20 Long tasks, starting from the library phase A leaves              tasks/libero_pro_B20.json
# (the agents see them at campaign/tasks/ in the workspace)
# Layout: vLLM TP=2 on GPUs 0-1 (70% of each, up to 16 requests at once), SAM3 on GPU 0, Contact-GraspNet + cuRobo on
# GPU 1 (profile h100), PyRoki on CPU; eight simulation slots, two per GPU (0a 0b 1a 1b 2a 2b 3a 3b).
# Everything is written under $STUDY_DATA/<run name>/ (library, runs, logs, Claude config). The run name appears in
# paths the agents see, so give it no meaning (run1, run2, ...); a run name belongs to one arm. Run the same command
# again in a new allocation to resume. Stop gracefully:
# touch $STUDY_DATA/<tag>/STOP (no new coordinator session; the running one is not killed).
# Without SLURM the job end is taken as now + STUDY_HOURS (default 96) hours; the drain times follow from it.
# STUDY_PREFLIGHT_ONLY=1: everything up to the first coordinator session (vLLM, providers, preflight, load test,
# sandbox check, promotion daemon, the kickoff rendered, one headless Claude turn through the sandbox), then stop.
set -uo pipefail

ARM=${1:?usage: campaign.sh skeptic|aspire <run name>}; TAG=${2:?usage: campaign.sh skeptic|aspire <run name>}
case $ARM in skeptic|aspire) ;; *) echo "arm must be skeptic or aspire"; exit 2 ;; esac
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9-]*$ ]] || { echo "run name: letters, digits and '-' only"; exit 2; }
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd -P)
[ -n "${SLURM_JOB_ID:-}" ] && [ -f "${SLURM_SUBMIT_DIR:-}/launch/config.sh" ] && HERE=$SLURM_SUBMIT_DIR/launch   # sbatch runs a copy
source "$HERE/config.sh" || { echo "copy launch/config.example.sh to launch/config.sh and edit it"; exit 2; }
PROJECT=$(dirname "$HERE")   # this project: method sources, prompts, task lists (hidden from the agents)
R=$ASPIRE_V1                 # the agents' workspace
[ -x "$R/.venv-libero/bin/python" ] || { echo "no workspace at $R; run launch/setup.sh first"; exit 2; }
[ "$(git -C "$R" rev-parse HEAD 2>/dev/null)" = fa00f0ef2bc1d91d594b9aa502c22133c9f538f7 ] || echo "WARNING: the workspace is not at the ASPIRE v1.0 commit the paper used (fa00f0e)"
export ARM STUDY_TAG=$TAG STUDY_ISOLATE=${STUDY_ISOLATE:-1}
JOB=${SLURM_JOB_ID:-local-$(date +%m%d%H%M)}
D=$STUDY_DATA/$TAG
OUT=$D/logs/job-$JOB
export STUDY_LOG_DIR=$STUDY_DATA
export STUDY_LIB=$D/lib STUDY_RUNS_BASE=$D/runs ASPIRE_CLAUDE_CFG=$D/claude-config
# The gate holds only if the agents cannot reach the library, the runs of other arms or the daemon's copy of
# promote.py through their host paths: STUDY_DATA must be hidden and outside the repository (a hidden directory inside
# the repository would be mounted back with it).
REPO_REAL=$(readlink -f "$R"); DATA_REAL=$(readlink -m "$STUDY_DATA"); PROJ_REAL=$(readlink -f "$PROJECT"); HIDDEN=0
case "$DATA_REAL/" in "$REPO_REAL"/*) echo "STUDY_DATA must lie outside the workspace"; exit 2 ;; esac
case "$REPO_REAL/" in "$PROJ_REAL"/*) echo "the workspace must lie outside this project (the project is hidden from the agents)"; exit 2 ;; esac
case "$PROJ_REAL/" in "$REPO_REAL"/*) echo "this project must lie outside the workspace (the agents would see it)"; exit 2 ;; esac
for d in ${STUDY_HIDE_DIRS:-}; do
  dd=$(readlink -m "$d")
  case "$dd/" in "$REPO_REAL"/*) echo "STUDY_HIDE_DIRS entry $d is inside the workspace; it would be visible again"; exit 2 ;; esac
  case "$DATA_REAL/" in "$dd"/*) HIDDEN=1 ;; esac
done
[ "$HIDDEN" = 1 ] || { echo "STUDY_DATA ($DATA_REAL) must lie inside one of STUDY_HIDE_DIRS"; exit 2; }
chmod +x "$PROJECT/skeptic/overlay/cap-harness" "$HERE/cap-harness-shim" "$HERE"/*.sh 2>/dev/null   # in case an unpacker dropped the modes
mkdir -p "$OUT"
if [ -f "$D/arm" ] && [ "$(cat "$D/arm")" != "$ARM" ]; then echo "run name $TAG belongs to arm $(cat "$D/arm")"; exit 2; fi
echo "$ARM" > "$D/arm"

# ---------------- knobs (the paper's runs) ----------------
DEV_SEEDS=${DEV_SEEDS:-"51-65"}; UNSEEN_SEEDS=${UNSEEN_SEEDS:-"66-80"}; HELDOUT_COUNT=${HELDOUT_COUNT:-50}
MAX_STEPS=${MAX_STEPS:-1000}; SIM_GPUS=0,1,2,3
RERUN_THRESHOLD=${RERUN_THRESHOLD:-0.8}
MAX_SESSIONS=${MAX_SESSIONS:-60}
INITIAL_LIB_HASH=91d636a706623f4e
PHASES="A B"
GATED=0; [ "$ARM" = skeptic ] && GATED=1
if [ "$GATED" = 1 ]; then
  # Skeptic: the gate (promote.py, obs_timeline.py) as a read-only copy made for this job, and the overlay
  grep -q 'It ends with an `end state` block' "$PROJECT/skeptic/overlay/subagent-prompt.md" || { echo "Skeptic worker template check failed"; exit 2; }
  mkdir -p "$OUT/gate-sandbox"; chmod -R u+w "$OUT/gate-sandbox"
  cp "$PROJECT"/skeptic/gate/{promote.py,obs_timeline.py} "$OUT/gate-sandbox/" && chmod -R a-w "$OUT/gate-sandbox" \
    || { echo "gate copy failed"; exit 2; }
  export STUDY_GATE=$OUT/gate-sandbox STUDY_OVERLAY=$PROJECT/skeptic/overlay OBS_TIMELINE_SEEDS=$DEV_SEEDS
fi
TMPBASE=${TMPDIR:-}
[ -d "$TMPBASE" ] || TMPBASE=$(mktemp -d "/tmp/study-$TAG-XXXX")
EXAMPLES=$R/examples/libero   # preflight programs only; hidden from every agent
echo "arm=$ARM run=$TAG isolate=$STUDY_ISOLATE job=$JOB host=$(hostname -s) out=$OUT tmp=$TMPBASE"

# ---------------- node health: GPU kernel log ----------------
node_bad() {
  echo "node check failed on $(hostname -s): $1"; printf '%s\n' "$2" | tail -5
  echo "resume in another allocation (exclude this node): bash launch/campaign.sh $ARM $TAG"
  exit 9
}
T0=$(cut -d' ' -f1 /proc/uptime)
GPU_BUSES=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | awk '{b = tolower(substr($1, 5)); sub(/\.0$/, "", b); print b}')
NGPU=$(echo "$GPU_BUSES" | grep -c .)
[ "$NGPU" -eq 4 ] || { echo "this job sees $NGPU GPUs, not 4; aborting"; exit 2; }
xids() {  # xids <uptime seconds>: kernel-log Xid lines for this job's GPUs logged after that time
  dmesg 2>/dev/null | awk -v t0="$1" -v buses="$GPU_BUSES" '
    BEGIN { n = split(buses, b, /[ \n]+/); for (i = 1; i <= n; i++) if (b[i] != "") want["(PCI:" b[i] ")"] = 1 }
    /NVRM: Xid/ { ts = $0; sub(/^\[ */, "", ts); sub(/\].*/, "", ts); if (ts + 0 <= t0) next
                  for (k in want) if (index($0, k)) { print; break } }'
}
xid_bad() {  # hardware Xids, graphics-pipeline MMU faults (Xid 31 ENGINE GRAPHICS), or >= 5 Xid 109 on one GPU in 10 min
  awk '{ ts = $0; sub(/^\[ */, "", ts); sub(/\].*/, "", ts); ts += 0
         bus = $0; sub(/.*\(PCI:/, "", bus); sub(/\).*/, "", bus)
         code = -1; if (match($0, /\): [0-9]+,/)) code = substr($0, RSTART + 3, RLENGTH - 4) + 0 }
       code == 48 || code == 79 || code == 95 || code == 119 || code == 120 { print; next }
       code == 31 && /ENGINE GRAPHICS/ { print; next }
       code == 109 { k = ++n[bus]; t[bus, k] = ts; if (k >= 5 && ts - t[bus, k - 4] <= 600) print }'
}
dmesg > /dev/null 2>&1 || echo "WARNING: kernel log unreadable; GPU Xid check skipped"
echo "GPUs $(echo $GPU_BUSES)"
BAD=$(xids 0 | xid_bad)
if [ "${STUDY_XID_SINCE_BOOT:-1}" = 1 ] && [ -n "$BAD" ]; then
  node_bad "this job's GPUs logged a broken-GPU Xid pattern since boot (STUDY_XID_SINCE_BOOT=0 skips this check)" "$BAD"
fi
cd "$R"

# ---------------- model server ----------------
export QWEN_PORT=8200 QWEN_TP=2 QWEN_MEM_UTIL=0.70 QWEN_KV_DTYPE=auto QWEN_MAX_LEN=262144 QWEN_EFFORT=${QWEN_EFFORT:-medium}
export QWEN_MAX_SEQS=16                   # eight slots + coordinator + verifiers
export VLLM_CACHE_ROOT=$TMPBASE/vllm-cache   # this job's own compile cache
source "$HERE/llm.sh"
MODEL=$STUDY_MODEL_NAME EFFORT=$QWEN_EFFORT
for p in 8114 8115 8116 8118 $QWEN_PORT; do
  if (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null; then echo "port $p already in use on $(hostname); aborting"; exit 2; fi
done
SUP_PID= WATCH_PID= DAEMON_PID= XID_PID= CUROBO_WATCH_PID= GPUMEM_PID=
CUROBO_CMD='cap_harness.providers.curobo.service --port 8118'
cleanup() {
  for p in $WATCH_PID $XID_PID $DAEMON_PID $CUROBO_WATCH_PID $GPUMEM_PID; do kill "$p" 2>/dev/null; done
  pkill -u "$USER" -f "promote.py daemon --lib $STUDY_LIB"
  [ -f "$OUT/curobo_supervisor.pid" ] && kill -TERM "$(cat "$OUT/curobo_supervisor.pid")" 2>/dev/null
  [ -n "$SUP_PID" ] && kill -TERM "$SUP_PID" 2>/dev/null
  sleep 3; pkill -u "$USER" -f "$CUROBO_CMD"
  pkill -u "$USER" -f "vllm serve .*--port $QWEN_PORT"
  sleep 10; pkill -9 -u "$USER" -f "vllm serve .*--port $QWEN_PORT"; pkill -9 -u "$USER" -f "^VLLM::"
  wait 2>/dev/null; echo "end=$(date -Is)"
}
trap cleanup EXIT
start_llm 0,1 "$OUT" || node_bad "vLLM did not start" "$(grep -v shm_broadcast "$OUT/vllm.log" | tail -3)"
vllm_up() { curl -sf --max-time 10 "http://127.0.0.1:$QWEN_PORT/health" > /dev/null; }
( down=0
  while sleep 60; do
    if vllm_up; then down=0; else down=$((down + 1)); fi
    if [ $down -ge 3 ]; then
      echo "vllm watchdog: down for 3 min; restarting $(date -Is)"
      pkill -u "$USER" -f "vllm serve .*--port $QWEN_PORT"; sleep 15
      start_llm 0,1 "$OUT"; down=0
    fi
  done ) >> "$OUT/vllm_watchdog.log" 2>&1 &
WATCH_PID=$!

source .venv-libero/bin/activate
export UV_PROJECT_ENVIRONMENT="$R/.venv-libero" UV_NO_SYNC=1 HF_HUB_OFFLINE=1
mkdir -p "$STUDY_RUNS_BASE/A" "$STUDY_RUNS_BASE/B" "$ASPIRE_CLAUDE_CFG"

phase_latest() { readlink -f "$STUDY_RUNS_BASE/$1/LATEST" 2>/dev/null; }
phase_done() { local L; L=$(phase_latest "$1"); [ -n "$L" ] && [ -f "$L/CAMPAIGN_COMPLETE.md" ]; }
all_phases_done() { local p; for p in $PHASES; do phase_done "$p" || return 1; done; }
if all_phases_done; then echo "both phases complete"; exit 0; fi
# the arm's own copy of the skill library, made from the v1.0 release library at the first start
if [ ! -d "$STUDY_LIB" ]; then
  mkdir -p "$STUDY_LIB" && cp -a .claude/skills/iterative-debugging/skills/. "$STUDY_LIB/"
fi
libhash() { PYTHONDONTWRITEBYTECODE=1 python3 -c "import sys; sys.path.insert(0, '$PROJECT/skeptic/gate'); import promote; from pathlib import Path; print(promote.library_sha(Path('$STUDY_LIB')))"; }
LIBHASH=$(libhash)
if [ -z "$(phase_latest A)" ] && [ "$LIBHASH" != "$INITIAL_LIB_HASH" ]; then
  echo "fresh start but library hash is $LIBHASH, not the v1.0 release library $INITIAL_LIB_HASH; aborting"; exit 2
fi

DRAIN_STAGE1_H=${DRAIN_STAGE1_H:-3}; DRAIN_VERIFY_H=${DRAIN_VERIFY_H:-1.5}
END_STR=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null)
if [[ "$END_STR" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T ]]; then JOB_END=$(date -d "$END_STR" +%s); else JOB_END=$(( $(date +%s) + ${STUDY_HOURS:-96}*3600 )); fi
DRAIN_STAGE1=$(python3 -c "print(int($JOB_END - $DRAIN_STAGE1_H*3600))")
DRAIN_VERIFY=$(python3 -c "print(int($JOB_END - $DRAIN_VERIFY_H*3600))")
hm() { date -d "@$1" "+%Y-%m-%d %H:%M %Z"; }
unset CLAUDECODE $(env | grep -o -E '^(CLAUDE_CODE_[A-Z_]*|ANTHROPIC_[A-Z_]*)')
use_llm_for_claude
export CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0 DISABLE_AUTOUPDATER=1

expand() { local s=""; for t in $1; do if [[ $t == *-* ]]; then s+=" $(seq ${t%-*} ${t#*-} | tr '\n' ' ')"; else s+=" $t"; fi; done; echo $s; }
DEV_LIST=$(expand "$DEV_SEEDS"); UNSEEN_LIST=$(expand "$UNSEEN_SEEDS")
{
  echo "job=$JOB arm=$ARM run=$TAG isolate=$STUDY_ISOLATE host=$(hostname) start=$(date -Is) claude=$(claude --version 2>/dev/null)"
  echo "model=$MODEL effort=$EFFORT vllm_tp=$QWEN_TP vllm_mem=$QWEN_MEM_UTIL max_seqs=$QWEN_MAX_SEQS max_len=$QWEN_MAX_LEN slots=0a,0b,1a,1b,2a,2b,3a,3b dev=[$DEV_LIST] unseen=[$UNSEEN_LIST] heldout=1..$HELDOUT_COUNT max_steps=$MAX_STEPS rerun=$RERUN_THRESHOLD"
  echo "library=$STUDY_LIB hash=$LIBHASH runs=$STUDY_RUNS_BASE cfg=$ASPIRE_CLAUDE_CFG${STUDY_GATE:+ gate=$STUDY_GATE overlay=$STUDY_OVERLAY obs_seeds=$OBS_TIMELINE_SEEDS}"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
} | tee "$OUT/header.txt"

# ---------------- provider services (cuRobo under its own supervisor, checked with a real plan) ----------------
CAP_HARNESS_SERVICE_LOG_DIR="$OUT/services" scripts/supervise_services.sh \
  --profile h100 --providers sam3,contact_graspnet,pyroki --ready-timeout 900 > "$OUT/supervisor.log" 2>&1 &
SUP_PID=$!
CUROBO_N=0
start_curobo() {
  CUROBO_N=$((CUROBO_N + 1))
  CAP_HARNESS_SERVICE_LOG_DIR="$OUT/services/curobo-$CUROBO_N" scripts/supervise_services.sh \
    --profile h100 --providers curobo --ready-timeout 900 > "$OUT/supervisor-curobo-$CUROBO_N.log" 2>&1 &
  echo $! > "$OUT/curobo_supervisor.pid"
}
curobo_ready() {
  local pid; pid=$(cat "$OUT/curobo_supervisor.pid")
  for i in $(seq 1 90); do
    grep -q "all provider services ready" "$OUT/supervisor-curobo-$CUROBO_N.log" 2>/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 10
  done
  return 1
}
restart_curobo() {
  local pid r rc; pid=$(cat "$OUT/curobo_supervisor.pid" 2>/dev/null)
  echo "$(date -Is) restarting cuRobo ($1)"
  [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 30); do pgrep -u "$USER" -f "$CUROBO_CMD" > /dev/null || break; sleep 1; done
  pkill -9 -u "$USER" -f "$CUROBO_CMD"; sleep 3
  start_curobo
  curobo_ready || { echo "$(date -Is) cuRobo instance $CUROBO_N did not become ready"; return 1; }
  r=$(python3 "$HERE/curobo_probe.py" 8118 900); rc=$?
  echo "$(date -Is) cuRobo instance $CUROBO_N: probe $r"
  return $rc
}
services_up() { for p in 8114 8115 8116 8118; do curl -sf --max-time 5 "http://127.0.0.1:$p/openapi.json" > /dev/null || return 1; done; }
start_curobo
for i in $(seq 1 120); do
  grep -q "all provider services ready" "$OUT/supervisor.log" && grep -q "all provider services ready" "$OUT/supervisor-curobo-1.log" && break
  kill -0 $SUP_PID 2>/dev/null && kill -0 "$(cat "$OUT/curobo_supervisor.pid")" 2>/dev/null \
    || { echo "supervisor exited:"; tail -20 "$OUT/supervisor.log" "$OUT/supervisor-curobo-1.log"; exit 3; }
  sleep 10
done
grep -q "all provider services ready" "$OUT/supervisor.log" && grep -q "all provider services ready" "$OUT/supervisor-curobo-1.log" \
  || { echo "providers not ready"; tail -20 "$OUT/supervisor.log" "$OUT/supervisor-curobo-1.log"; exit 3; }
echo "providers ready $(date -Is)"

# ---------------- LLM-free preflight on all four GPUs ----------------
PF=$OUT/preflight; mkdir -p "$PF"
.venv-libero/bin/cap-harness doctor --reset --output "$PF/doctor.json" > "$PF/doctor.log" 2>&1
echo "doctor exit=$?"
for g in ${SIM_GPUS//,/ }; do
  CUDA_VISIBLE_DEVICES=$g .venv-libero/bin/python "$HERE/egl_probe.py" > "$PF/egl_gpu$g.txt" 2>&1
  grep -q "egl\[$g\] nvidia .* cuda_ordinal=0\$" "$PF/egl_gpu$g.txt" || { echo "EGL device $g is not CUDA device $g; aborting"; cat "$PF/egl_gpu$g.txt"; exit 5; }
done
echo "EGL mapping ok for GPUs $SIM_GPUS"
for g in ${SIM_GPUS//,/ }; do
  CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 15m .venv-libero/bin/cap-harness run --benchmark libero-pro \
    --suite libero_goal_swap --task-id 0 --seed 51 --init-mode seeded --program "$EXAMPLES"/open_middle_drawer.py \
    --output-root "$PF/runs/gpu$g" > "$PF/example_gpu$g.log" 2>&1
  echo "example gpu$g exit=$?"
done
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 timeout 15m .venv-libero/bin/cap-harness run --benchmark libero-pro \
  --suite libero_object_swap --task-id 0 --seed 1 --init-mode seeded --program "$EXAMPLES"/object_swap_curobo_seed1.py \
  --output-root "$PF/runs" > "$PF/curobo_warmup.log" 2>&1
echo "curobo warm-up exit=$?"
CUROBO_WATCH=1
r=$(python3 "$HERE/curobo_probe.py" 8118 600); rc=$?
echo "curobo probe: $r"
case $rc in
  0) ;;
  2) echo "WARNING: cuRobo rejected the probe request (the probe is wrong, not the service); the cuRobo watchdog is off"; CUROBO_WATCH=0 ;;
  *) restart_curobo "the start-up probe failed: $r" || node_bad "cuRobo fails a real plan even after a restart" "$r" ;;
esac
# Load test at full campaign load: two slots x 4 render workers = 8 episodes on every GPU at once (32 in all), with
# vLLM and the providers up. GPU memory is sampled meanwhile (peak MiB used per GPU).
( while sleep 20; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' '; echo; done ) > "$PF/gpu_mem_load.log" 2>&1 &
GPUMEM_PID=$!
LPIDS=()
for g in ${SIM_GPUS//,/ }; do
  CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 25m .venv-libero/bin/cap-harness run-batch --benchmark libero-pro \
    --suite libero_goal_swap --task-id 0 --seeds 51 52 53 54 55 56 57 58 --init-mode seeded --workers 8 --flat-run-dir \
    --program "$EXAMPLES"/open_middle_drawer.py --output-root "$PF/load/gpu$g" \
    --results-jsonl "$PF/load/gpu$g/results.jsonl" > "$PF/load_gpu$g.log" 2>&1 &
  LPIDS+=($!)
done
wait "${LPIDS[@]}"
kill $GPUMEM_PID 2>/dev/null; GPUMEM_PID=
NLOAD=$(cat "$PF"/load/gpu*/results.jsonl 2>/dev/null | wc -l)
echo "load test: $NLOAD/32 episodes finished $(date -Is); peak GPU memory used (MiB): $(awk '{for (i = 1; i < NF; i += 2) { g = $i + 0; m = $(i + 1) + 0; if (m > p[g]) p[g] = m }} END {for (g in p) printf "gpu%s=%d ", g, p[g]}' "$PF/gpu_mem_load.log")"
[ "$NLOAD" -eq 32 ] || node_bad "the load test finished $NLOAD of 32 episodes" "$(tail -3 "$PF"/load_gpu*.log)"
NEW=$(xids "$T0" | xid_bad)
[ -n "$NEW" ] && node_bad "this job's GPUs logged a broken-GPU Xid pattern during startup, preflight and load test" "$NEW"
echo "GPU kernel log: no broken-GPU pattern since job start ($(xids "$T0" | wc -l) Xid lines)"
( last=0
  while sleep 300; do
    n=$(xids "$T0" | wc -l)
    if [ "$n" -ne "$last" ]; then echo "$(date -Is) Xid lines on this job's GPUs since start: $n"; xids "$T0" | tail -3; last=$n; fi
    BAD=$(xids "$T0" | xid_bad)
    if [ -n "$BAD" ] && [ ! -f "$OUT/gpu_broken" ]; then
      { echo "$(date -Is) broken-GPU pattern during the run on $(hostname -s); ending the coordinator session"
        printf '%s\n' "$BAD" | head -5; } | tee "$OUT/gpu_broken"
      pkill -u "$USER" -f '^claude -p '
    fi
  done ) >> "$OUT/gpu_xid.log" 2>&1 &
XID_PID=$!
note() {  # note <text>: an operator note in the current phase's curriculum.md
  local L; L=$(phase_latest "$PH")
  [ -n "$L" ] && printf '\n### %s — operator note (job %s)\n- %s\n' "$(date '+%Y-%m-%d %H:%M %Z')" "$JOB" "$1" >> "$L/curriculum.md"
}
PH=A
if [ "$CUROBO_WATCH" = 1 ]; then
  ( restarts=0
    while sleep 300; do
      r=$(python3 "$HERE/curobo_probe.py" 8118 120); rc=$?
      [ $rc -eq 0 ] && continue
      [ $rc -eq 2 ] && { echo "$(date -Is) cuRobo rejected the probe request ($r); not acting"; continue; }
      echo "$(date -Is) cuRobo probe failed: $r"; sleep 30
      r2=$(python3 "$HERE/curobo_probe.py" 8118 120) && { echo "$(date -Is) passed on the re-check: $r2"; continue; }
      [ $restarts -ge 20 ] && { echo "$(date -Is) 20 restarts already; leaving cuRobo as it is"; continue; }
      restarts=$((restarts + 1))
      if restart_curobo "failed a real plan twice: $r2"; then
        PH=$(cat "$OUT/phase" 2>/dev/null || echo A)
        note "The cuRobo service (:8118) had stopped planning (every request failed: $r2). The job restarted it; it plans again since $(date '+%H:%M'). cuRobo is available."
      fi
    done ) >> "$OUT/curobo_watch.log" 2>&1 &
  CUROBO_WATCH_PID=$!
  echo "cuRobo watchdog on (real plan every 5 min; restarts cuRobo alone)"
fi
python3 - "$PF" <<'PY' || { echo "preflight failed; not starting Claude"; exit 4; }
import json, sys, glob
pf = sys.argv[1]
outs = sorted(glob.glob(f"{pf}/runs/**/outcome.json", recursive=True))
ok = [json.load(open(o)) for o in outs]
print("preflight episodes:", [(o.split('/runs/')[1].split('/outcome')[0], d.get("program_ok"), d.get("task_success")) for o, d in zip(outs, ok)])
sys.exit(0 if ok and all(d.get("program_ok") for d in ok) else 1)
PY
SBX_TMP=$TMPBASE/study-sbx-$TAG; mkdir -p "$SBX_TMP"
if [ "$GATED" = 1 ]; then
  # the observation timeline inside the sandbox: an episode that closes the gripper on nothing must get a timeline
  # with an empty_close event, and its keyframes must carry depth images
  export STUDY_RUNS=$STUDY_RUNS_BASE/A
  mkdir -p "$SBX_TMP/obs-preflight"
  cat > "$SBX_TMP/obs-preflight/grip_check.py" <<'PROG'
state = get_robot_state()
q = [float(v) for v in state.joint_positions["primary"]]
open_gripper()
q[0] += 0.15
move_to_joints(q)
close_gripper()
open_gripper()
result = {"note": "preflight: close on nothing"}
PROG
  CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 timeout 10m bash "$HERE/sandbox.sh" "$SBX_TMP" -- .venv-libero/bin/cap-harness run-batch \
    --benchmark libero-pro --suite libero_goal_swap --task-id 0 --seeds 51 --init-mode seeded --workers 1 --flat-run-dir \
    --no-videos --program /tmp/obs-preflight/grip_check.py --output-root /tmp/obs-preflight/out \
    --results-jsonl /tmp/obs-preflight/out/results.jsonl > "$PF/obs_timeline.log" 2>&1
  TL=$(ls "$SBX_TMP"/obs-preflight/out/0051/run-*/obs_timeline.md 2>/dev/null | head -1)
  if [ -z "$TL" ] || ! grep -q "empty_close" "$TL" || ! ls "$SBX_TMP"/obs-preflight/out/0051/run-*/media/keyframes/*/agentview-depth-mm.png > /dev/null 2>&1 \
     || ! grep -q "^\[observation timeline\] 1 failed run" "$PF/obs_timeline.log"; then
    echo "observation timeline preflight failed:"; tail -20 "$PF/obs_timeline.log"; exit 7
  fi
  cp "$TL" "$PF/obs_timeline_example.md"; rm -rf "$SBX_TMP/obs-preflight"
  echo "observation timeline ok: $(grep -m1 '^events:' "$PF/obs_timeline_example.md")"
  mkdir -p "$OUT/gate"
  cp "$STUDY_GATE/promote.py" "$OUT/gate/promote.py"
  cp .claude/skills/iterative-debugging/scripts/lint_skills.py "$OUT/gate/lint_skills.py"
fi

# ---------------- coordinator, phase by phase ----------------
start_daemon() {  # Skeptic: the promotion daemon (outside the sandbox) for one phase's runs dir
  [ -n "$DAEMON_PID" ] && kill "$DAEMON_PID" 2>/dev/null
  pkill -u "$USER" -f "promote.py daemon --lib $STUDY_LIB"; sleep 2
  ( while true; do
      python3 "$OUT/gate/promote.py" daemon --lib "$STUDY_LIB" --runs "$1" --repo "$R" \
        --lint "$OUT/gate/lint_skills.py" --log "$D/logs/promotions.jsonl"
      echo "daemon exited $? at $(date -Is); restarting in 30 s"; sleep 30
    done ) >> "$OUT/promote_daemon.log" 2>&1 &
  DAEMON_PID=$!
  sleep 5; grep -q "runs=$(readlink -f "$1")" "$OUT/promote_daemon.log" || { echo "promotion daemon did not start:"; tail -5 "$OUT/promote_daemon.log"; return 1; }
  echo "promotion daemon up for $1: $(grep 'daemon: lib=' "$OUT/promote_daemon.log" | tail -1)"
}
INIT_ARGS="--dev-seeds $DEV_LIST --unseen-seeds $UNSEEN_LIST --heldout-count $HELDOUT_COUNT"
PROJ_KEY=$(pwd -P | sed 's#[^A-Za-z0-9]#-#g')   # Claude Code's project directory name for this path
STOPPED=   # why the session loop ended: complete | drain | stop | error
set_phase_vars() {  # set_phase_vars <A|B>: the phase's task list and notes (into the caller's variables)
  if [ "$1" = A ]; then
    TASK_FILE=campaign/tasks/libero_pro_A60.json NT=60
    TASK_DESC="libero_object_swap, libero_object_task, libero_goal_swap, libero_goal_task, libero_spatial_swap and libero_spatial_task, task_id 0-9 each (60 tasks)"
    PHASE_NOTE="phase 1 of 2 of the user's LIBERO-Pro study: the 60 Object, Goal and Spatial tasks (position and task perturbations). Phase 2, the 20 Long tasks, runs afterwards as its own campaign with the skill library this phase leaves; do not start it."
  else
    TASK_FILE=campaign/tasks/libero_pro_B20.json NT=20
    TASK_DESC="libero_10_swap and libero_10_task, task_id 0-9 each (20 tasks)"
    PHASE_NOTE="phase 2 of 2 of the user's LIBERO-Pro study: the 20 Long tasks (position and task perturbations). Phase 1, the 60 Object, Goal and Spatial tasks, is finished; the skill library holds what it promoted, and this campaign continues from that library."
  fi
  ORDER_RULE="your curriculum decision over all $NT tasks, exactly as main-agent-prompt.md describes. The user adds no ordering constraint."
  BUDGET_RULE="exactly as written in main-agent-prompt.md and subagent-prompt.md. The user adds no rule of their own."
}
render_kickoff() {  # the kickoff for phase $PH with the session NOTE and SID
  sed -e "s#@SESSION_NOTE@#$NOTE#g" -e "s#@ORDER_RULE@#$ORDER_RULE#g" -e "s#@BUDGET_RULE@#$BUDGET_RULE#g" \
    -e "s#@TASK_FILE@#$TASK_FILE#g" -e "s#@TASK_DESC@#$TASK_DESC#g" -e "s#@PHASE_NOTE@#$PHASE_NOTE#g" \
    -e "s#@DEV_SEEDS@#$DEV_LIST#g" -e "s#@UNSEEN_SEEDS@#$UNSEEN_LIST#g" -e "s#@HELDOUT_COUNT@#$HELDOUT_COUNT#g" \
    -e "s#@MAX_STEPS@#$MAX_STEPS#g" -e "s#@RERUN_THRESHOLD@#$RERUN_THRESHOLD#g" \
    -e "s#@RUN_NAME@#libero-pro-$TAG-$PH#g" -e "s#@INIT_ARGS@#$INIT_ARGS#g" \
    -e "s#@JOB_END@#$(hm $JOB_END)#g" -e "s#@DRAIN_STAGE1@#$DRAIN_STAGE1#g" -e "s#@DRAIN_STAGE1_HM@#$(hm $DRAIN_STAGE1)#g" \
    -e "s#@DRAIN_VERIFY@#$DRAIN_VERIFY#g" -e "s#@DRAIN_VERIFY_HM@#$(hm $DRAIN_VERIFY)#g" \
    -e "s#@SUBAGENT_DIR@#$ASPIRE_CLAUDE_CFG/projects/$PROJ_KEY/$SID/subagents#g" "$PROJECT/prompts/coordinator-$ARM.md"
}
claude_session() {  # claude_session <prompt> <session id> <stream file> <stderr file>
  bash "$HERE/sandbox.sh" "$SBX_TMP" -- claude -p "$1" --session-id "$2" --model "$MODEL" --effort "$EFFORT" \
    --settings campaign/launch/claude-settings.json --permission-mode bypassPermissions \
    --output-format stream-json --verbose > "$3" 2> "$4"
}
run_phase() {
  PH=$1; echo "$PH" > "$OUT/phase"
  export STUDY_RUNS=$STUDY_RUNS_BASE/$PH
  local TASK_FILE TASK_DESC NT PHASE_NOTE ORDER_RULE BUDGET_RULE
  set_phase_vars "$PH"
  bash "$HERE/sandbox_check.sh" "$OUT/sandbox_check_$PH.txt" || { echo "sandbox check failed:"; cat "$OUT/sandbox_check_$PH.txt"; STOPPED=error; return 1; }
  echo "sandbox ok for phase $PH ($(tail -1 "$OUT/sandbox_check_$PH.txt"))"
  if [ "$GATED" = 1 ]; then start_daemon "$STUDY_RUNS" || { STOPPED=error; return 1; }; fi
  local QUICK=0 n LATEST T1 SID NOTE PROMPT S B NEWROOT
  for n in $(seq 1 "$MAX_SESSIONS"); do
    LATEST=$(phase_latest "$PH")
    if [ -n "$LATEST" ] && [ -f "$LATEST/CAMPAIGN_COMPLETE.md" ]; then echo "phase $PH complete: $LATEST"; STOPPED=complete; break; fi
    if [ "$(date +%s)" -ge "$DRAIN_STAGE1" ]; then echo "drain time reached"; STOPPED=drain; break; fi
    if [ -f "$D/STOP" ]; then echo "$D/STOP present; no new session"; STOPPED=stop; break; fi
    if [ -f "$OUT/gpu_broken" ]; then node_bad "broken-GPU pattern during the run (gpu_xid.log)" "$(cat "$OUT/gpu_broken")"; fi
    if [ -n "$LATEST" ] && [ -f "$LATEST/BLOCKED_SERVICES.md" ]; then
      B=$OUT/BLOCKED_SERVICES-$PH-$(date +%m%d%H%M).md
      if grep -qi curobo "$LATEST/BLOCKED_SERVICES.md" && python3 "$HERE/curobo_probe.py" 8118 120 > /dev/null && services_up; then
        mv "$LATEST/BLOCKED_SERVICES.md" "$B"
        note "BLOCKED_SERVICES.md (cuRobo) was archived by the operator: cuRobo (:8118) passes a real test plan now and every provider service is up. The job re-tests cuRobo every 5 minutes and restarts it if it fails. Resume the campaign as the kickoff says."
        echo "archived a cuRobo BLOCKED_SERVICES.md ($B)"
      else
        echo "coordinator reported a service problem:"; cat "$LATEST/BLOCKED_SERVICES.md"; STOPPED=error; break
      fi
    fi
    if [ "$QUICK" -ge 3 ]; then echo "3 sessions in a row ended within 10 min; stopping"; STOPPED=error; break; fi
    if [ "$GATED" = 1 ] && ! kill -0 "$DAEMON_PID" 2>/dev/null; then echo "promotion daemon loop is gone; stopping"; STOPPED=error; break; fi
    NEWROOT=$(ls -A "$R" | sort | comm -13 "$OUT/repo_root_at_start.txt" - | tr '\n' ' ')
    [ -n "$NEWROOT" ] && echo "WARNING: new entries in the repository root since this job started: $NEWROOT"
    for i in $(seq 1 40); do vllm_up && break; sleep 30; done
    vllm_up || { echo "vLLM not healthy for 20 min; stopping"; tail -30 "$OUT/vllm.log"; STOPPED=error; break; }
    T1=$(date +%s)
    SID=$(python3 -c 'import uuid; print(uuid.uuid4())')
    if [ -z "$LATEST" ]; then NOTE="fresh kickoff of phase $PH (job $JOB, session $n)."
    else NOTE="RESTART: session $n of job $JOB in phase $PH. A previous coordinator session ended; resume the campaign at outputs/runs/LATEST."; fi
    PROMPT=$(render_kickoff)
    if grep -q "@[A-Z_]*@" <<< "$PROMPT"; then echo "kickoff placeholder left: $(grep -o '@[A-Z_]*@' <<< "$PROMPT" | sort -u | tr '\n' ' ')"; STOPPED=error; break; fi
    S=$OUT/session-$PH-$(printf %02d $n); mkdir -p "$S"; echo "$PROMPT" > "$S/prompt.txt"; echo "$SID" > "$S/session_id"
    echo "phase $PH session $n sid=$SID start=$(date -Is)"
    claude_session "$PROMPT" "$SID" "$S/stream.jsonl" "$S/stderr.log"
    echo "phase $PH session $n exit=$? end=$(date -Is)"
    python3 - "$S/stream.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1], errors="replace"):
    try: d = json.loads(line)
    except json.JSONDecodeError: continue
    if d.get("type") == "result": print("RESULT", d.get("subtype"), "turns", d.get("num_turns"), "is_error", d.get("is_error"))
PY
    if [ $(( $(date +%s) - T1 )) -lt 600 ]; then QUICK=$((QUICK + 1)); else QUICK=0; fi
    echo "next start in 120s"; sleep 120
  done
  [ -z "$STOPPED" ] && STOPPED=error && echo "$MAX_SESSIONS sessions used up in this job"
  if [ "$STOPPED" = complete ]; then cp -r "$STUDY_LIB" "$OUT/library-at-end-$PH"; fi
  [ "$STOPPED" = complete ]
}
echo "job ends $(hm $JOB_END); no new Stage 1 after $(hm $DRAIN_STAGE1); no new verifier after $(hm $DRAIN_VERIFY)"
ls -A "$R" | sort > "$OUT/repo_root_at_start.txt"   # the repository root is writable in the sandbox: log what appears
if [ "${STUDY_PREFLIGHT_ONLY:-0}" = 1 ]; then
  PH=A; export STUDY_RUNS=$STUDY_RUNS_BASE/A
  bash "$HERE/sandbox_check.sh" "$OUT/sandbox_check_A.txt" || { echo "sandbox check failed:"; cat "$OUT/sandbox_check_A.txt"; exit 7; }
  echo "sandbox ok for phase A ($(tail -1 "$OUT/sandbox_check_A.txt"))"
  if [ "$GATED" = 1 ]; then start_daemon "$STUDY_RUNS" || exit 8; fi
  set_phase_vars A; NOTE="fresh kickoff of phase A (job $JOB, session 1)."; SID=$(python3 -c 'import uuid; print(uuid.uuid4())')
  PROMPT=$(render_kickoff); echo "$PROMPT" > "$PF/kickoff_A.txt"
  if grep -q "@[A-Z_]*@" <<< "$PROMPT"; then echo "kickoff placeholder left: $(grep -o '@[A-Z_]*@' <<< "$PROMPT" | sort -u | tr '\n' ' ')"; exit 7; fi
  echo "kickoff for phase A rendered: $(wc -c < "$PF/kickoff_A.txt") bytes, no placeholder left ($PF/kickoff_A.txt)"
  # one headless turn through vLLM and the sandbox, with a throwaway config dir (no transcript in the arm's)
  ASPIRE_CLAUDE_CFG=$OUT/claude-check-config bash "$HERE/sandbox.sh" "$TMPBASE/study-sbx-claude-check-$TAG" -- \
    claude -p "Reply with the single word OK." --model "$MODEL" --effort "$EFFORT" \
    --settings campaign/launch/claude-settings.json --permission-mode bypassPermissions --output-format json \
    > "$PF/claude_check.json" 2> "$PF/claude_check.err"
  echo "claude check exit=$? result: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(repr(d.get('result')), 'is_error', d.get('is_error'))" "$PF/claude_check.json" 2>&1)"
  echo "STUDY_PREFLIGHT_ONLY=1: stopping before the coordinator $(date -Is)"
  exit 0
fi
for P in $PHASES; do
  phase_done "$P" && continue
  STOPPED=
  run_phase "$P" || break
done
if all_phases_done; then
  echo "study complete: $(for P in $PHASES; do phase_latest "$P"; done | tr '\n' ' ')"
elif [ "$STOPPED" = drain ]; then
  echo "drain time reached; resume in a new allocation: bash launch/campaign.sh $ARM $TAG"
else
  echo "stopped ($STOPPED)"
fi
for P in $PHASES; do
  L=$(phase_latest "$P"); [ -n "$L" ] || continue
  STUDY_RUNS=$STUDY_RUNS_BASE/$P bash "$HERE/sandbox.sh" "$SBX_TMP" -- .venv-libero/bin/python .claude/skills/iterative-debugging/scripts/gen_progress.py --print 2>/dev/null | tail -25
done
