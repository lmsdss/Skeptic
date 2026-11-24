# Skeptic

Skeptic grounds an LLM agent's failure diagnosis in measured evidence, and it lets knowledge into the agent's
skill library only after that knowledge has been verified. This repository holds the code for the paper's
LIBERO-Pro study:
- all 80 tasks: Object, Goal, Spatial and Long, each with position (`*_swap`) and task (`*_task`) perturbations;
- one open-weight model, served locally by vLLM, as coordinator, workers and verifiers.

## Components

1. **Observation timeline with an end state**
   (`skeptic/gate/obs_timeline.py`, `skeptic/overlay/cap-harness`, `skeptic/overlay/artifacts.py`).
   - Every failed development episode (seeds 51–65) automatically gets a timeline (`obs_timeline.md`), and the
     batch runner prints one line per failed seed.
   - Every motion and gripper call is listed with what the robot's own readings and the depth camera measured:
     - whether the hand reached its target;
     - how far the jaws closed, and which surface lay between or nearest the finger pads;
     - whether a segmented object left its place, rose with the hand, or slipped.
   - The **end state** closes the timeline. For each segmented object it gives:
     - whether the object is still at its starting place;
     - where the hand last released it;
     - over which other object's starting footprint that was;
     - which surface the depth camera shows there afterwards.
   - Every line is tagged `[proprio]` (robot readings), `[vision]` (depth camera) or `[inferred]` (an explanation,
     not a measurement).
   - The timeline uses observations only. It never reads simulator internals or the task goal. The recorder stores
     an agentview depth image (uint16, mm) with each keyframe for it.
2. **Worker guidance** (`skeptic/overlay/subagent-prompt.md`): tells every worker that the timeline exists and what
   its end state means.
3. **Verified skill admission** (`skeptic/gate/promote.py`, and the `SKILL LIBRARY ADMISSION` section of
   `prompts/coordinator-skeptic.md`).
   - Only a successful attempt may add to the library. An attempt qualifies when its selected program passes at
     least 8 of the 15 development seeds.
   - From a qualifying attempt, only the proposals its verifier graded `sound` may enter the library.
   - The library is read-only for every agent. A promotion daemon outside the agents' sandbox applies a staged edit,
     but only after all of these checks pass:
     - it recounts the development passes from the attempt's own runs;
     - it checks the program's hash, the verdict and the library version;
     - it runs the library's lint.
   - A rerun of a task never sees the earlier attempt. Knowledge reaches a rerun only through the library.

### How it runs

- **The agent loop.** A coordinator dispatches workers that write and debug a program per task. Verifiers check
  the workers' proposals. The loop is the iterative-debugging protocol of the open-source
  [ASPIRE v1.0](https://github.com/NVlabs/ASPIRE) code base, which also provides the LIBERO-Pro harness, the
  simulator client and the perception and planning services.
- **The workspace.** `launch/setup.sh` checks out that code base at a pinned commit, as the agents' workspace.
  Skeptic's components are mounted into it for the `skeptic` arm only.
- **The baseline.** Running the same workspace with ASPIRE's released protocol gives the baseline arm, `aspire`.
- **What both arms share.** The model, the infrastructure, the seeds, the budgets and the starting skill library
  (the ASPIRE v1.0 release library, hash `91d636a706623f4e`).

| arm | what runs |
|---|---|
| `skeptic` | the ASPIRE v1.0 protocol with Skeptic's components |
| `aspire` | the released ASPIRE v1.0 protocol (baseline) |

The arm is never passed into the agents' sandbox. Every run also has a name, and the agents see that name in their
paths, so give it no meaning (`run1`, `run2`, ...).

## Layout

```
skeptic/gate/      promote.py (admission gate + promotion daemon), obs_timeline.py (timeline + end state)
skeptic/overlay/   cap-harness (CLI wrapper), artifacts.py (recorder with depth keyframes), subagent-prompt.md
prompts/           coordinator-skeptic.md, coordinator-aspire.md (@...@ placeholders filled by campaign.sh)
tasks/             libero_pro_A60.json (phase A: 60 tasks), libero_pro_B20.json (phase B: 20 Long tasks)
configs/h100.json  H100 service profile for the workspace
launch/            config.example.sh   paths and model; copy to config.sh
                   setup.sh            workspace checkout (pinned) + environments
                   campaign.sh         one study (phase A, then phase B) for one arm
                   llm.sh              vLLM server; points the agent runtime at it
                   sandbox.sh          bubblewrap sandbox for every agent session
                   sandbox_check.sh    verifies the sandbox view before each phase
                   simd.py, cap-harness-shim   simulator isolation (the simulator runs outside the agent sandbox)
                   curobo_probe.py, egl_probe.py   service and GPU checks
                   claude-settings.json            permission rules for the headless agent sessions
                   qwen_chat_template.jinja        chat template that maps the runtime's effort levels
```

## Requirements

- One Linux node with 4 x H100 94 GB (the coordinator prompts say "94 GB") and at least 64 CPU cores. A study
  takes several days: run it in a 4-GPU allocation, and re-run the same command to resume.
- [bubblewrap](https://github.com/containers/bubblewrap) (we used 0.6.3).
- **Agent runtime.** The agents run in a headless command-line coding-agent harness (the `claude` CLI, versions
  2.1.319–2.1.325), the runtime ASPIRE's skills are packaged for. It is configured to send every request to the local
  vLLM server (see `launch/llm.sh`): all model calls, from the coordinator, workers and verifiers, go to the
  open-weight model, no external model or API is called, and no API key or account is needed.
- vLLM (we used 0.30.0), with the model weights in the Hugging Face cache. The paper used `Qwen/Qwen3.8-27B-FP8`
  at reasoning effort `medium`. The coordinator prompts name this model; edit them if you change it.
- The workspace's host prerequisites: CUDA, `uv`, and the gated SAM3 weights. See `docs/host-setup.md` in the
  workspace.
- Network access for `setup.sh`.

## Run

From this repository's root:

```bash
cp launch/config.example.sh launch/config.sh   # set ASPIRE_V1, STUDY_DATA, STUDY_HIDE_DIRS, VLLM_ENV, HF_HOME, ...
bash launch/setup.sh                            # once, CPU only

# inside a 4 x H100 allocation (or: sbatch --output=$STUDY_DATA/slurm-%j.out launch/campaign.sh skeptic run1)
STUDY_PREFLIGHT_ONLY=1 bash launch/campaign.sh skeptic run0   # smoke test: services, load test, sandbox check
bash launch/campaign.sh skeptic run1    # Skeptic  -> $STUDY_DATA/run1/
bash launch/campaign.sh aspire run2     # baseline -> $STUDY_DATA/run2/ (on another node)
touch $STUDY_DATA/run1/STOP             # graceful stop: no new coordinator session
```

`campaign.sh` does the following:
1. Starts vLLM on GPUs 0–1, the services (SAM3, Contact-GraspNet, cuRobo, PyRoki) and their watchdogs.
2. Runs an LLM-free preflight and a 32-episode load test on all four GPUs.
3. Checks the sandbox view.
4. For `skeptic`, starts the promotion daemon.
5. Runs headless coordinator sessions until phase A writes `CAMPAIGN_COMPLETE.md`, then does the same for phase B.

Under SLURM the job's end time sets when the coordinator stops dispatching. Without SLURM it is now + `STUDY_HOURS`
(default 96). The campaign stops on a `BLOCKED_SERVICES.md` written by the coordinator. A cuRobo problem clears by
itself once cuRobo plans again. Anything else needs an operator: fix the cause, move the file away, and run the
command again.

Outputs go under `$STUDY_DATA/<run name>/`:
- `lib/`: the arm's skill library;
- `runs/{A,B}/`: the campaign runs;
- `claude-config/`: agent transcripts;
- `logs/`: job logs;
- `simd/`: simulator-service logs.

Per-task results are in each run's `progress.md` and `CAMPAIGN_COMPLETE.md`.

### Sandbox and isolation (both arms)

Every coordinator, worker and verifier runs inside `launch/sandbox.sh`, with the workspace as its working
directory. The following are hidden from the agents:
- this repository (method sources, prompts, launcher);
- `STUDY_HIDE_DIRS`, which must contain `STUDY_DATA` (`campaign.sh` checks this): other runs, logs and transcripts;
- the workspace's `.git`, `examples/` (worked solutions) and `AGENTS.md` (real-robot rules);
- the user's own agent state.

Inside the workspace the agents see `campaign/tasks/` and `campaign/launch/claude-settings.json`. For `skeptic` they also
see the gate, at `campaign/gate/`.

The workspace's `.venv-libero`, `src/` and `third_party/` are read-only. With simulator isolation,
`third_party/`, `.libero` and the service venvs are hidden as well. Every `cap-harness` call is forwarded to
`simd.py`, which runs the real simulator outside the agent sandbox, so no agent can read LIBERO's source, BDDL task
files or assets.

The agents see the workspace's path. Choose a name for it that does not name the method.

## License

Apache-2.0 (see `LICENSE`). See `NOTICE` for the files derived from ASPIRE.
