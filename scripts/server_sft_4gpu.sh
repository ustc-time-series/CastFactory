#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

RECIPE="$REPO_ROOT/examples/sft/etth1_qwen3_1_7b_sft_4gpu.yaml"

echo "[SFT] GPUs: $CUDA_VISIBLE_DEVICES"
echo "[SFT] Recipe: $RECIPE"

torchrun --standalone --nproc_per_node=4 \
  -m castfactory.cli.run "$RECIPE" --mode fit \
  model.backbone.model_name=/home/zyt/LLM/Qwen3-1.7B \
  training.init_checkpoint=./checkpoints/etth1_qwen3_1_7b_4gpu/etth1_ot_qwen_cpt \
  training.checkpoint_dir=./checkpoints/etth1_qwen3_1_7b_4gpu/etth1_ot_qwen_sft \
  training.args.num_train_epochs=1 \
  training.args.learning_rate=2.0e-4 \
  training.backend.max_prompt_length=2048 \
  training.backend.max_response_length=1024 \
  training.backend.max_seq_length=3072 \
  training.backend.per_device_train_batch_size=1 \
  training.backend.gradient_accumulation_steps=8 \
  training.backend.fsdp_config.transformer_layer_cls_to_wrap=Qwen3DecoderLayer \
  "$@"
