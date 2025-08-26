set -x


export PYTHONWARNINGS="ignore::FutureWarning"


MODEL_PATH=/mnt/bn/robotics-rl-lf/vlm_models/DeepSeek-R1-Distill-Qwen-7B
TRAIN_DATA_PATH=/opt/tiger/e2e_alg/vlm/verl/datas/DeepScaleR-Preview-Dataset/deepscaler_train.parquet
VAL_DATA_PATH=/opt/tiger/e2e_alg/vlm/verl/datas/DeepScaleR-Preview-Dataset/deepscaler_val.parquet


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.enable_off_policy_samples=False \
    algorithm.enable_off_policy_reeval=False \
    algorithm.enable_off_policy_rollout=False \
    algorithm.enable_adaptive_target_range=False \
    algorithm.train_batch_filter=gaussian \
    algorithm.fix_max_size=True \
    algorithm.off_policy_update_freq=5 \
    algorithm.off_policy_reeval_freq=5 \
    algorithm.max_total_uids=256 \
    algorithm.max_reeval_ids=128 \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="$VAL_DATA_PATH" \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.lambda_1=1 \
    actor_rollout_ref.actor.lambda_2=1 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='off_policy_grpo' \
    trainer.experiment_name='on_sample_on_rollout-deepseek7b-train-v1.0' \
    trainer.prompt_tracking.save_dir='/mnt/bn/robotics-rl-lf/vlm_datasets/DeepScaleR-Preview-Dataset/on_sample_on_rollout-deepseek7b-train-v1.0-tracking_results' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=3 $@