#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  VERL_CKPT_DIR=/path/to/verl/global_step_x/actor HF_OUTPUT_DIR=/path/to/hf bash scripts/package_verl_ckpt.sh
  bash scripts/package_verl_ckpt.sh /path/to/verl/global_step_x/actor /path/to/hf [extra verl.model_merger args...]

Environment:
  BACKEND=fsdp|megatron        Checkpoint backend for verl.model_merger. Default: fsdp
  PYTHON=python                Python executable with verl installed. Default: python
  TIE_WORD_EMBEDDING=true      Add --tie-word-embedding.
  IS_VALUE_MODEL=true          Add --is-value-model.
  USE_CPU_INITIALIZATION=true  Add --use_cpu_initialization.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

VERL_CKPT_DIR="${VERL_CKPT_DIR:-}"
HF_OUTPUT_DIR="${HF_OUTPUT_DIR:-}"

if [[ -z "$VERL_CKPT_DIR" && $# -gt 0 ]]; then
  VERL_CKPT_DIR="$1"
  shift
fi
if [[ -z "$HF_OUTPUT_DIR" && $# -gt 0 ]]; then
  HF_OUTPUT_DIR="$1"
  shift
fi

BACKEND="${BACKEND:-fsdp}"
PYTHON="${PYTHON:-python}"

if [[ "$BACKEND" != "fsdp" && "$BACKEND" != "megatron" ]]; then
  echo "[package_verl_ckpt] BACKEND must be 'fsdp' or 'megatron', got: $BACKEND" >&2
  exit 2
fi
if [[ -z "$VERL_CKPT_DIR" || -z "$HF_OUTPUT_DIR" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -d "$VERL_CKPT_DIR" ]]; then
  echo "[package_verl_ckpt] VERL_CKPT_DIR does not exist or is not a directory: $VERL_CKPT_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$HF_OUTPUT_DIR")"

cmd=(
  "$PYTHON" -m verl.model_merger merge
  --backend "$BACKEND"
  --local_dir "$VERL_CKPT_DIR"
  --target_dir "$HF_OUTPUT_DIR"
)

if [[ "${TIE_WORD_EMBEDDING:-false}" == "true" ]]; then
  cmd+=(--tie-word-embedding)
fi
if [[ "${IS_VALUE_MODEL:-false}" == "true" ]]; then
  cmd+=(--is-value-model)
fi
if [[ "${USE_CPU_INITIALIZATION:-false}" == "true" ]]; then
  cmd+=(--use_cpu_initialization)
fi
cmd+=("$@")

echo "[package_verl_ckpt] backend: $BACKEND"
echo "[package_verl_ckpt] source:  $VERL_CKPT_DIR"
echo "[package_verl_ckpt] target:  $HF_OUTPUT_DIR"
"${cmd[@]}"

if [[ ! -f "$HF_OUTPUT_DIR/config.json" ]]; then
  echo "[package_verl_ckpt] merged checkpoint is missing config.json: $HF_OUTPUT_DIR" >&2
  exit 1
fi

echo "[package_verl_ckpt] HuggingFace checkpoint ready: $HF_OUTPUT_DIR"
