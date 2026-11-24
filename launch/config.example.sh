# Skeptic launcher configuration. Copy to launch/config.sh, edit, and keep it out of version control.
# Sourced by launch/setup.sh and launch/campaign.sh; do not execute it.

SKEPTIC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"   # this project (not exported: hidden from agents)

# The agents' workspace: a checkout of the open-source ASPIRE v1.0 code base (LIBERO-Pro harness, simulator client,
# provider services, iterative-debugging protocol), created by setup.sh. The agents run with this directory as their
# working directory and see its path, so keep it outside this project and outside STUDY_HIDE_DIRS, and give it a name
# that does not name the method.
export ASPIRE_V1=/path/to/aspire-v1

# Everything a campaign writes: skill libraries, runs, logs, agent transcripts. An 80-task study creates ~1M files
# (per-episode keyframes); use a file system with a large inode quota. Outside this project and the workspace.
export STUDY_DATA=/path/to/study-data
# Hidden from every agent (tmpfs in the sandbox). Must contain STUDY_DATA (campaign.sh checks this). Add any directory
# with other results or worked solutions the agents must not see. This project is always hidden.
export STUDY_HIDE_DIRS="$STUDY_DATA"
# Example of a scheduler submission, with its log outside the workspace:
#   sbatch --partition=<h100 partition> --output=$STUDY_DATA/slurm-%j.out launch/campaign.sh skeptic 1

# Model server: a Python venv with vLLM, and the Hugging Face cache holding the model and the gated SAM3 weights.
export VLLM_ENV=$HOME/vllm-env
export HF_HOME=/path/to/hf-cache
# The coordinator prompts (prompts/*.md) name this model, effort medium and 94 GB GPUs, as in the paper's runs;
# edit them too if you change the model, QWEN_EFFORT or the hardware.
export STUDY_MODEL=Qwen/Qwen3.8-27B-FP8              # served as $STUDY_MODEL_NAME to Claude Code
export STUDY_MODEL_NAME=qwen3.8-27b

# Toolchains for the workspace environments (ASPIRE's docs/host-setup.md). Python 3.10-3.12 for cap-harness.
export CAP_HARNESS_PYTHON=$(command -v python3.12)
export CAP_HARNESS_PROFILE=h100                    # configs/h100.json, installed into the workspace by setup.sh
export TORCH_CUDA_ARCH_LIST=9.0                    # H100 (sm_90)
# A CUDA 12.8 toolkit (nvcc) is needed once, by setup.sh, to build Contact-GraspNet's pointnet2 ops. Load it only in
# the shell that runs setup.sh (e.g. `module load CUDA/12.8.0`), not here: campaign.sh sources this file before it
# starts vLLM, which brings its own CUDA build.
export UV_CACHE_DIR=$STUDY_DATA/uv-cache
export CAP_HARNESS_PROVIDER_CACHE="$UV_CACHE_DIR" CAP_HARNESS_CUROBO_CACHE="$UV_CACHE_DIR"
export PATH="$HOME/.local/bin:$PATH"               # uv, claude

# LIBERO / MuJoCo runtime.
export LIBERO_CONFIG_PATH="$ASPIRE_V1/.libero"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device
