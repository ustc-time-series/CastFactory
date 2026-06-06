#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  MODEL_DIR=/path/to/hf bash scripts/serve_vllm_12000.sh
  bash scripts/serve_vllm_12000.sh /path/to/hf [extra vllm serve args...]

Environment:
  VLLM_BIN=vllm                       vLLM executable. Default: vllm
  HOST=0.0.0.0                        Bind host. Default: 0.0.0.0
  PORT=12000                          Bind port. Default: 12000
  SERVED_MODEL_NAME=castfactory-eval  OpenAI API model name. Default: castfactory-eval
  TENSOR_PARALLEL_SIZE=1              Add --tensor-parallel-size when set.
  GPU_MEMORY_UTILIZATION=0.9          Add --gpu-memory-utilization when set.
  MAX_MODEL_LEN=4096                  Add --max-model-len when set.
  DTYPE=auto                          Add --dtype when set.
  TRUST_REMOTE_CODE=true              Add --trust-remote-code when true.
  VLLM_EXTRA_ARGS="..."               Extra args split on shell whitespace.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_DIR="${MODEL_DIR:-}"
if [[ -z "$MODEL_DIR" && $# -gt 0 ]]; then
  MODEL_DIR="$1"
  shift
fi
if [[ -z "$MODEL_DIR" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -d "$MODEL_DIR" || ! -f "$MODEL_DIR/config.json" ]]; then
  echo "[serve_vllm_12000] MODEL_DIR must be a HuggingFace checkpoint directory with config.json: $MODEL_DIR" >&2
  exit 1
fi

VLLM_BIN="${VLLM_BIN:-vllm}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-12000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-castfactory-eval}"

cmd=(
  "$VLLM_BIN" serve "$MODEL_DIR"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
)

if [[ -n "${TENSOR_PARALLEL_SIZE:-}" ]]; then
  cmd+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
fi
if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
  cmd+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
  cmd+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "${DTYPE:-}" ]]; then
  cmd+=(--dtype "$DTYPE")
fi
if [[ "${TRUST_REMOTE_CODE:-false}" == "true" ]]; then
  cmd+=(--trust-remote-code)
fi
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<<"$VLLM_EXTRA_ARGS"
  cmd+=("${extra_args[@]}")
fi
cmd+=("$@")

echo "[serve_vllm_12000] model: $MODEL_DIR"
echo "[serve_vllm_12000] url:   http://$HOST:$PORT/v1"
echo "[serve_vllm_12000] name:  $SERVED_MODEL_NAME"
exec "${cmd[@]}"
