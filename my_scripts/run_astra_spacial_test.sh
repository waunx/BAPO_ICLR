set -x

data_path=/mnt/bn/robotics-rl-lf/vlm_datasets/data/spatial_reasoning_select_v6/test_v1.parquet
save_path=/mnt/bn/robotics-rl-lf/vlm_datasets/data/spatial_reasoning_select_v6/test_astra_res.parquet
model_path=/mnt/bn/robotics-rl-lf/astra_models/astra-jt-qwen2_5-v2_expert-1-1

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=8 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=0.6 \
    rollout.top_k=10000 \
    rollout.top_p=1 \
    rollout.prompt_length=4096 \
    rollout.response_length=16384 \
    rollout.max_num_batched_tokens=34816 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8

