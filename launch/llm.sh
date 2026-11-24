# The open-weight model behind Claude Code: served locally by vLLM, used by the coordinator, every worker and every
# verifier. Source this file; do not execute it.
#   start_llm <gpu[,gpu]> <log dir>   # start vLLM and wait until it answers (the first start compiles kernels)
#   use_llm_for_claude                # point the `claude` CLI in this shell at the local vLLM server
# Knobs (environment): STUDY_MODEL, STUDY_MODEL_NAME, VLLM_ENV, QWEN_PORT, QWEN_TP, QWEN_MAX_LEN, QWEN_EFFORT
# (low | medium | xhigh), QWEN_MTP (1 = speculative decoding), QWEN_KV_DTYPE (fp8 | auto), QWEN_MEM_UTIL, QWEN_MAX_SEQS.
# The values used for the paper's runs are the defaults set in launch/campaign.sh.
STUDY_MODEL=${STUDY_MODEL:-Qwen/Qwen3.8-27B-FP8}
STUDY_MODEL_NAME=${STUDY_MODEL_NAME:-qwen3.8-27b}
VLLM_ENV=${VLLM_ENV:-$HOME/vllm-env}
QWEN_PORT=${QWEN_PORT:-8200}
QWEN_MAX_LEN=${QWEN_MAX_LEN:-262144}
QWEN_EFFORT=${QWEN_EFFORT:-medium}
QWEN_MTP=${QWEN_MTP:-1}
QWEN_TP=${QWEN_TP:-1}
QWEN_KV_DTYPE=${QWEN_KV_DTYPE:-fp8}
QWEN_MEM_UTIL=${QWEN_MEM_UTIL:-0.92}   # lower it when simulation shares the GPUs
QWEN_MAX_SEQS=${QWEN_MAX_SEQS:-8}
# The model's chat template accepts only the effort levels low | medium | xhigh; this copy maps the levels Claude Code
# may send (minimal, high, max) onto those, which otherwise fail with HTTP 400.
QWEN_TEMPLATE=${QWEN_TEMPLATE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/qwen_chat_template.jinja}

start_llm() {
  local gpu=$1 out=$2 spec=() cu13
  [ "$QWEN_MTP" = 1 ] && spec=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}')
  if (exec 3<>/dev/tcp/127.0.0.1/$QWEN_PORT) 2>/dev/null; then echo "port $QWEN_PORT already in use"; return 1; fi
  echo "vllm start gpu=$gpu tp=$QWEN_TP seqs=$QWEN_MAX_SEQS mem=$QWEN_MEM_UTIL kv=$QWEN_KV_DTYPE max_len=$QWEN_MAX_LEN effort=$QWEN_EFFORT mtp=$QWEN_MTP $(date -Is)"
  # torch.compile looks for nvcc on PATH: put the venv's own CUDA toolchain first if it has one. FlashInfer's
  # sampler is JIT-compiled with that nvcc; the native PyTorch sampler avoids a header mismatch.
  cu13=$(ls -d "$VLLM_ENV"/lib/python3*/site-packages/nvidia/cu13/bin 2>/dev/null | head -1)
  PATH=${cu13:+$cu13:}$PATH VLLM_USE_FLASHINFER_SAMPLER=0 \
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$gpu "$VLLM_ENV/bin/vllm" serve "$STUDY_MODEL" \
    --served-model-name "$STUDY_MODEL_NAME" --host 127.0.0.1 --port "$QWEN_PORT" \
    --tensor-parallel-size "$QWEN_TP" --max-num-seqs "$QWEN_MAX_SEQS" \
    --max-model-len "$QWEN_MAX_LEN" --kv-cache-dtype "$QWEN_KV_DTYPE" --gpu-memory-utilization "$QWEN_MEM_UTIL" \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --chat-template "$QWEN_TEMPLATE" \
    --default-chat-template-kwargs "{\"reasoning_effort\":\"$QWEN_EFFORT\"}" \
    --limit-mm-per-prompt '{"video":0}' \
    "${spec[@]}" > "$out/vllm.log" 2>&1 &
  QWEN_PID=$!
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$QWEN_PORT/health" > /dev/null; then echo "vllm up after $((i * 10))s"; return 0; fi
    kill -0 $QWEN_PID 2>/dev/null || { echo "vllm died during startup:"; tail -40 "$out/vllm.log"; return 1; }
    sleep 10
  done
  echo "vllm not up after 30 min"; return 1
}

use_llm_for_claude() {
  unset ANTHROPIC_API_KEY
  export ANTHROPIC_BASE_URL=http://127.0.0.1:$QWEN_PORT
  export ANTHROPIC_AUTH_TOKEN=local-vllm   # vLLM runs without --api-key; any value keeps claude off a subscription login
  export ANTHROPIC_MODEL=$STUDY_MODEL_NAME CLAUDE_CODE_SUBAGENT_MODEL=$STUDY_MODEL_NAME
  export ANTHROPIC_DEFAULT_OPUS_MODEL=$STUDY_MODEL_NAME ANTHROPIC_DEFAULT_SONNET_MODEL=$STUDY_MODEL_NAME
  export ANTHROPIC_DEFAULT_HAIKU_MODEL=$STUDY_MODEL_NAME
  # Claude Code does not know an unknown model's context length. Leave room for the output reserved per request
  # (32000 tokens): a limit equal to vLLM's makes long agent contexts fail with HTTP 400.
  export CLAUDE_CODE_MAX_OUTPUT_TOKENS=32000
  export CLAUDE_CODE_MAX_CONTEXT_TOKENS=$((QWEN_MAX_LEN - 40000))
  export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
  export API_TIMEOUT_MS=1800000   # a local model thinks slowly; do not time out long turns
}
