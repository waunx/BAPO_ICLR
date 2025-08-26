set -x

data_path=/opt/tiger/e2e_alg/vlm/verl/datas/DeepScaleR-Preview-Dataset/olympiad_bench.parquet
model_path=/mnt/bn/robotics-rl-lf/vlm_ckpts/off_policy_grpo/off_sample_off_rollout-deepseek1.5b-train-v2.0/global_step_450
save_path=$model_path/olympiad_bench_val_8k_deepseek1.5b.parquet

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=8 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.n=1 \
    rollout.prompt_length=2048 \
    rollout.response_length=8192 \
    rollout.max_num_batched_tokens=34816 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8
