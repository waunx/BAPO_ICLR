#!/usr/bin/env bash
set -xeuo pipefail

project_name='DAPO'
exp_name='DAPO-DeepSeek-1.5b-MATH'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=512
n_resp_per_prompt=16
train_prompt_mini_bsz=32

# Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-8}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}


MODEL_PATH=/mnt/bn/robotics-rl-lf/vlm_models/DeepSeek-R1-Distill-Qwen-1.5B
TRAIN_DATA_PATH=/opt/tiger/e2e_alg/vlm/verl/datas/DeepScaleR-Preview-Dataset/deepscaler_train.parquet
VAL_DATA_PATH=/opt/tiger/e2e_alg/vlm/verl/datas/DeepScaleR-Preview-Dataset/deepscaler_val.parquet


# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=4
use_dynamic_bsz=True

offload=True
gen_tp=4
fsdp_size=32

# remember to set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for this model

python3 -m verl.trainer.main_ppo \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="$VAL_DATA_PATH" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.train_batch_size=256 \
    actor_rollout_ref.rollout.n=8 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0\
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=30720 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=30720 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="off_policy_grpo" \
    trainer.experiment_name='DAPO-DeepSeek-1.5b-v2' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=50 \
    trainer.save_freq=50 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="/mnt/bn/robotics-rl-lf/vlm_ckpts/off_policy_grpo/DAPO-DeepSeek-1.5b-v2" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10
