#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

RECIPE="$REPO_ROOT/examples/rlvr/etth1_qwen3_1_7b_grpo_4gpu.yaml"
RLVR_DIR="./checkpoints/etth1_qwen3_1_7b_4gpu/etth1_ot_qwen_grpo"

echo "[RLVR] GPUs: $CUDA_VISIBLE_DEVICES"
echo "[RLVR] Recipe: $RECIPE"
echo "[RLVR] Output dir: $RLVR_DIR"

EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    actor_rollout_ref.*)
      EXTRA_ARGS+=("training.backend.verl.$arg")
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

python -m castfactory.cli.run "$RECIPE" --mode fit \
  model.backbone.model_name=/home/zyt/LLM/Qwen3-1.7B \
  training.init_checkpoint=./checkpoints/etth1_qwen3_1_7b_4gpu/etth1_ot_qwen_sft \
  training.checkpoint_dir=$RLVR_DIR \
  training.epochs=1 \
  training.batch_size=16 \
  training.backend.algorithm=grpo \
  training.backend.inference_engine=vllm \
  training.backend.num_generations=8 \
  training.backend.verl.data.train_batch_size=16 \
  training.backend.verl.data.max_prompt_length=4096 \
  training.backend.verl.data.max_response_length=8192 \
  training.backend.verl.data.filter_overlong_prompts=true \
  training.backend.verl.data.truncation=error \
  training.backend.verl.actor_rollout_ref.actor.optim.lr=1.0e-6 \
  training.backend.verl.actor_rollout_ref.model.use_remove_padding=true \
  training.backend.verl.actor_rollout_ref.model.enable_gradient_checkpointing=true \
  training.backend.verl.actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  training.backend.verl.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  training.backend.verl.actor_rollout_ref.actor.use_kl_loss=true \
  training.backend.verl.actor_rollout_ref.actor.kl_loss_coef=0.001 \
  training.backend.verl.actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  training.backend.verl.actor_rollout_ref.actor.entropy_coeff=0 \
  training.backend.verl.actor_rollout_ref.actor.fsdp_config.param_offload=false \
  training.backend.verl.actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  training.backend.verl.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  training.backend.verl.actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  training.backend.verl.actor_rollout_ref.rollout.name=vllm \
  training.backend.verl.actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  training.backend.verl.actor_rollout_ref.rollout.n=8 \
  training.backend.verl.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  training.backend.verl.actor_rollout_ref.ref.fsdp_config.param_offload=true \
  training.backend.verl.algorithm.use_kl_in_reward=false \
  training.backend.verl.trainer.critic_warmup=0 \
  training.backend.verl.trainer.logger='["console"]' \
  training.backend.verl.trainer.project_name=castfactory_etth1_qwen3_1_7b \
  training.backend.verl.trainer.experiment_name=etth1_ot_qwen3_1_7b_grpo_4gpu \
  training.backend.verl.trainer.n_gpus_per_node=4 \
  training.backend.verl.trainer.nnodes=1 \
  training.backend.verl.trainer.save_freq=20 \
  training.backend.verl.trainer.test_freq=5 \
  training.backend.verl.trainer.total_epochs=1 \
  "${EXTRA_ARGS[@]}"

LAUNCH_FILE="$RLVR_DIR/rlvr/launch_command.txt"
if [[ ! -s "$LAUNCH_FILE" ]]; then
  echo "[RLVR] Missing launch command: $LAUNCH_FILE" >&2
  exit 1
fi

echo "[RLVR] Launch command written to: $LAUNCH_FILE"
echo "[RLVR] Running verl command with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
bash -lc "$(cat "$LAUNCH_FILE")"
