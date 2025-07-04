set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

export WANDB_MODE=offline
MODEL_PATH=/data/wx_data/models/Qwen2.5-Math-1.5B
TRAIN_DATA_PATH=/data/wx_data/datasets/DeepScaleR-Preview-Dataset/deepscaler_train.parquet
VAL_DATA_PATH=/data/wx_data/datasets/DeepScaleR-Preview-Dataset/math.parquet



python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_adaptive_filter \
    algorithm.enable_off_policy_samples=True \
    algorithm.enable_off_policy_rollout=False \
    algorithm.initial_c_value=0.15 \
    algorithm.adaptive_c_freq=1 \
    algorithm.off_policy_update_freq=20 \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="$VAL_DATA_PATH" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='off_policy_grpo_debug' \
    trainer.experiment_name='off-promote_itself-qwen_math1_5b-deepscaler_2k' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=3 $@