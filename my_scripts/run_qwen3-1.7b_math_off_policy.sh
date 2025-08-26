set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

export PYTHONWARNINGS="ignore::FutureWarning"


MODEL_PATH=/opt/tiger/e2e_alg/vlm/vlm_models/Qwen3-1.7B

TRAIN_DATA_PATH=/opt/tiger/e2e_alg/vlm/verl/datas/DeepScaleR-Preview-Dataset/deepscaler_train.parquet

VAL_DATA_PATH_1=/mnt/bn/robotics-rl-lf/vlm_datasets/TotalEval/aime_val.parquet
VAL_DATA_PATH_2=/mnt/bn/robotics-rl-lf/vlm_datasets/TotalEval/math_val.parquet
VAL_DATA_PATH_3=/mnt/bn/robotics-rl-lf/vlm_datasets/TotalEval/aime25_val.parquet


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_adaptive_filter \
    algorithm.enable_off_policy_samples=True \
    algorithm.enable_off_policy_reeval=True \
    algorithm.enable_off_policy_rollout=True \
    algorithm.enable_multi_level_downsampling=False \
    algorithm.enable_adaptive_target_range=True \
    algorithm.train_batch_filter=gaussian \
    algorithm.fix_max_size=True \
    algorithm.off_policy_update_freq=20 \
    algorithm.off_policy_reeval_freq=5 \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="[\"$VAL_DATA_PATH_1\",\"$VAL_DATA_PATH_2\",\"$VAL_DATA_PATH_3\"]" \
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
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='off_policy_grpo_debug' \
    trainer.experiment_name='off_sample_off_rollout-qwen_3_1.7b-deepscaler' \
    trainer.prompt_tracking.save_dir='/mnt/bn/robotics-rl-lf/vlm_datasets/TotalEval/off_sample_off_rollout-qwen_3_1.7b-tracking_results' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 $@
