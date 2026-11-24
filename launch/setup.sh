#!/usr/bin/env bash
# One-time setup (CPU only, needs network):
#   1. checks out the open-source ASPIRE v1.0 code base (pinned commit, with its pinned third-party submodules) at
#      $ASPIRE_V1, the agents' workspace;
#   2. adds H100 support to it (a cap-harness profile and the sm_90 architecture entry);
#   3. builds its environments with its own, unmodified bootstrap scripts:
#        .venv-libero  (simulator client: LIBERO-Pro + cap-harness)
#        .venv-sam3 .venv-graspnet .venv-pyroki .venv-curobo  (provider services)
#   cp launch/config.example.sh launch/config.sh   # edit it first
#   bash launch/setup.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/config.sh"
UPSTREAM_URL=https://github.com/NVlabs/ASPIRE.git
UPSTREAM_COMMIT=fa00f0ef2bc1d91d594b9aa502c22133c9f538f7   # ASPIRE v1.0 release
case "$(readlink -m "$ASPIRE_V1")/" in "$SKEPTIC_ROOT"/*) echo "ASPIRE_V1 must lie outside this project"; exit 2 ;; esac
[ -d "$ASPIRE_V1/.git" ] || git clone -q "$UPSTREAM_URL" "$ASPIRE_V1"
cd "$ASPIRE_V1"
git checkout -q "$UPSTREAM_COMMIT"
git submodule update --init third_party/LIBERO-PRO third_party/robosuite third_party/contact_graspnet_pytorch \
  third_party/sam3 third_party/curobo
# H100 support: the profile (SAM3 on GPU 0, Contact-GraspNet and cuRobo on GPU 1) and the sm_90 build entry.
cp "$SKEPTIC_ROOT/configs/h100.json" configs/profiles/h100.json
python3 - <<'PY'
import json
p = "configs/dependency-lock.json"
d = json.load(open(p))
arches = d["gpu_runtime"]["arches"]
if "sm_90" not in arches:
    arches["sm_90"] = {"cuda_arch_list": "9.0", "profiles": ["h100"]}
    open(p, "w").write(json.dumps(d, indent=2) + "\n")
PY
mkdir -p campaign outputs "$STUDY_DATA" "$UV_CACHE_DIR"   # campaign/: mount point for what the sandbox shows the agents
# bootstrap_libero.sh uses <workspace>/.uv-cache; point it at the data directory.
[[ -L .uv-cache || ! -e .uv-cache ]] && ln -sfn "$UV_CACHE_DIR" .uv-cache
echo "== $(date -Is) host=$(hostname) python=$($CAP_HARNESS_PYTHON --version) uv=$(uv --version)"
scripts/bootstrap_libero.sh --python "$CAP_HARNESS_PYTHON"
scripts/bootstrap_providers.sh --providers sam3,contact_graspnet,pyroki,curobo --profile "$CAP_HARNESS_PROFILE"
command -v bwrap > /dev/null || echo "WARNING: bubblewrap (bwrap) not found; the agent sandbox needs it"
command -v claude > /dev/null || echo "WARNING: Claude Code CLI (claude) not found on PATH"
[ -x "$VLLM_ENV/bin/vllm" ] || echo "WARNING: no vllm in $VLLM_ENV"
echo "== $(date -Is) done"
