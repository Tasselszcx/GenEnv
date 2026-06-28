#!/bin/bash
# GenEnv 8xH800 smoke test (adapted to conda env `genenv` + verl 0.4.1 + vllm 0.8.5)
# Stack: torch 2.6.0+cu124, vllm 0.8.5.post1, verl 0.4.1, transformers 4.51.1, flash-attn 2.7.4.post1
set -x

ENV=/home/hadoop-efficient-llm/miniconda3/envs/genenv
GENENV=/home/hadoop-efficient-llm/projects/GenEnv

# --- isolation: keep system ~/.local (verl 0.9 editable) and spark PYTHONPATH out ---
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PATH="/usr/bin:/usr/sbin:$PATH"
export PYTHONPATH="$GENENV"

# --- runtime ---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=1
# NOTE: vllm 0.8.5 V1 engine does NOT support XFORMERS backend; leave default (FLASH_ATTN).
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export RAY_TMPDIR=/tmp/ray_genenv
mkdir -p "$RAY_TMPDIR"

MODEL_PATH="/home/hadoop-efficient-llm/dolphinfs_ssd_hadoop-efficient-llm/models/Qwen/Qwen3-8B"
ENV_MODEL_PATH="$MODEL_PATH"
TRAIN_DATA="$GENENV/data/train_math_toy.parquet"
VAL_DATA="$GENENV/data/val_math_toy.parquet"
OUTPUT_DIR="/tmp/genenv_ckpt/smoke"
mkdir -p "$OUTPUT_DIR"

# GENENV_ENABLE: pass "True" for co-training path, "False" for standard GRPO stack validation
GENENV_ENABLE="${GENENV_ENABLE:-False}"

cd "$GENENV"
"$ENV/bin/python" -u -m genenv.train \
    genenv.enable=$GENENV_ENABLE \
    trainer.ray_address=null \
    genenv.filtering_k=0.1 \
    genenv.num_generations_per_prompt=2 \
    env_model_path="$ENV_MODEL_PATH" \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=8 \
    data.val_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.val_temperature=0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='genenv' \
    trainer.experiment_name='genenv_math_toy_smoke' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=1 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.total_epochs=1 \
    "${@}"
