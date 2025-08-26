set -x
ENGINE=${1:-vllm}
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

# MODEL_PATH=/opt/tiger/e2e_alg/vlm/vlm_models/Qwen2.5-VL-3B-Instruct
# TRAIN_DATA_PATH=/mnt/bn/robotics-rl-lf/vlm_datasets/geometry3k/train.parquet
# VAL_DATA_PATH=/mnt/bn/robotics-rl-lf/vlm_datasets/geometry3k/test_v1.parquet

# MODEL_PATH=/opt/tiger/e2e_alg/vlm/vlm_models/Qwen2.5-VL-3B-Instruct
# TRAIN_DATA_PATH=/mnt/bn/robotics-rl-lf/vlm_datasets/data/spatial_reasoning_select_v6/train_v1.parquet
# VAL_DATA_PATH=/mnt/bn/robotics-rl-lf/vlm_datasets/data/spatial_reasoning_select_v6/test_v1.parquet

MODEL_PATH=/mnt/bn/robotics-rl-lf/vlm_ckpts/qwen2_5vl_3b/full/cor_align_models/checkpoint-186

TRAIN_DATA_PATH=/mnt/bn/robotics-rl-lf/vlm_datasets/data/obstacle_classification_dataset_v2/obstacle_classification_train_v2.parquet
VAL_DATA_PATH=/mnt/bn/robotics-rl-lf/vlm_datasets/data/obstacle_classification_dataset_v2/obstacle_classification_test_v2.parquet

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.enable_off_policy_samples=False \
    algorithm.enable_off_policy_reeval=False \
    algorithm.enable_off_policy_rollout=False \
    algorithm.enable_multi_level_downsampling=False \
    algorithm.enable_adaptive_target_range=False \
    algorithm.train_batch_filter=gaussian \
    algorithm.fix_max_size=True \
    algorithm.off_policy_update_freq=20 \
    algorithm.off_policy_reeval_freq=5 \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$VAL_DATA_PATH \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.lambda_1=1 \
    actor_rollout_ref.actor.lambda_2=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name='off_policy_grpo_debug' \
    trainer.experiment_name='on_sample_on_rollout-qwen2_5_vl_3b_sft-spatial_hard-v3.0' \
    trainer.prompt_tracking.save_dir='/mnt/bn/robotics-rl-lf/vlm_datasets/data/obstacle_classification_dataset_v2/on_sample_on_rollout-qwen2_5_vl_3b_sft-spatial_hard-v3.0-tracking_result' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 $@