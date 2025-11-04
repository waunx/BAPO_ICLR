# export CUDA_VISIBLE_DEVICES=4,5,6,7

clip_ratio_low=0.2
clip_ratio_high=0.28


enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0


enable_filter_groups=True
filter_groups_metric=seq_final_reward
max_num_gen_batches=1

loss_agg_mode="token-mean"


MODEL_PATH=/data/wx_data/models/Qwen3-8B

TRAIN_DATA_PATH=/data/wx_data/datasets/BAPO/DeepScaleR/deepscaler_train.parquet
VAL_DATA_PATH=/data/wx_data/datasets/DeepScaleR/aime.parquet


# Algorithm
temperature=0.99
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# remember to set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for this model
python3 -m recipe.dapo.main_off_dapo \
    algorithm.adv_estimator=grpo \
    algorithm.enable_off_policy_samples=True \
    algorithm.enable_off_policy_rollout=True \
    algorithm.enable_off_policy_reeval=True \
    algorithm.enable_completion_batch=True \
    algorithm.enable_adaptive_target_range=False \
    algorithm.train_batch_filter=range \
    algorithm.fix_max_size=True \
    algorithm.off_policy_update_freq=2 \
    algorithm.off_policy_reeval_freq=2 \
    algorithm.max_total_uids=512 \
    algorithm.max_reeval_ids=256 \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="$VAL_DATA_PATH" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    actor_rollout_ref.rollout.n=8 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0\
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name="BAPO" \
    trainer.experiment_name='off-bapo-freq_2_2-qwen3_8b-val8k_aime' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=50 \
    trainer.total_epochs=3 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10
