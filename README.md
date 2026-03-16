# BAPO-ICLR2026

This repository is the **official implementation** of:

> **Buffer Matters: Unleashing the Power of Off-Policy Reinforcement Learning in Large Language Model Reasoning**  
> Xu Wan, Yansheng Wang, Wenqi Huang, Mingyang Sun  
> *ICLR 2026*

The codebase is developed **on top of [verl](https://github.com/volcengine/verl)** (Volcano Engine Reinforcement Learning for LLMs). We adopt verl’s PPO/GRPO training stack and extend it with BAPO’s off-policy buffer, completion batches, filter groups, and rollout importance sampling.

---

## Citation

If you use this code or find the paper helpful, please cite:

```bibtex
@inproceedings{wan2026buffer,
  title={Buffer Matters: Unleashing the Power of Off-Policy Reinforcement Learning in Large Language Model Reasoning},
  author={Xu Wan and Yansheng Wang and Wenqi Huang and Mingyang Sun},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
}
```

**verl (base framework):**

Please also cite the verl project and its associated paper (e.g. HybridFlow) as appropriate when using this codebase.

---

## BAPO-Related Files

BAPO-specific logic and config live in the following places.

### Recipe and config (BAPO entry and trainer)

| Path | Description |
|------|-------------|
| `recipe/bapo/main_bapo.py` | Entry point: Hydra app, Ray init, worker setup, reward manager, BAPO trainer. |
| `recipe/bapo/bapo_ray_trainer.py` | **RayBAPOTrainer**: off-policy buffer, completion batch, filter groups, rollout IS weight computation and application. |
| `recipe/bapo/config/bapo_trainer.yaml` | BAPO defaults: algorithm (off-policy flags, filter_groups, rollout_is), reward_model (dapo), trainer.project_name. |
| `my_scripts/run_qwen3-8b_dapo_math_off.sh` | Example launch script: sets paths and Hydra overrides for a typical BAPO run. |

### verl extensions (off-policy, IS, rollout state)

| Path | Description |
|------|-------------|
| `verl/trainer/ppo/ray_trainer.py` | Off-policy dataflow: completion batch, off-policy update/reeval, rollout state save/restore and temp policy switch. |
| `verl/trainer/ppo/core_algos.py` | GRPO off-policy advantage with importance sampling: `compute_importance_weights`, `compute_grpo_off_policy_importance_sampling_advantage`. |
| `verl/trainer/ppo/mismatch_helper.py` | Rollout–training mismatch: `compute_rollout_importance_weights` (and rejection/mask logic). |
| `verl/trainer/config/ppo_trainer.yaml` | Defaults for algorithm (e.g. off_policy_*, rollout_is_*, filter_groups) and trainer (e.g. default_local_dir). |
| `verl/workers/fsdp_workers.py` | FSDP workers: rollout state save/restore, temporary current-policy override for generation. |
| `verl/workers/sharding_manager/fsdp_vllm.py` | vLLM sharding manager: state save/restore and param-update control for rollout. |

All other files follow the upstream verl structure (models, data, reward, checkpointing, etc.).

---

## How to Run

### 1. Environment and dependencies

Install dependencies as in [verl](https://github.com/volcengine/verl) (e.g. PyTorch, Ray, vLLM, transformers). Run from the **repository root** so Hydra can resolve `verl/trainer/config`.

### 2. Set paths in the launch script

Edit `my_scripts/run_qwen3-8b_dapo_math_off.sh` and set:

- `MODEL_PATH` – HuggingFace model path (e.g. Qwen3-8B).
- `TRAIN_DATA_PATH` – Training parquet (e.g. DeepScaleR).
- `VAL_DATA_PATH` – Validation parquet (e.g. AIME 2024).

Optional: adjust `CUDA_VISIBLE_DEVICES` and script variables (e.g. `clip_ratio_*`, `temperature`, `enable_filter_groups`) or override any Hydra key on the `python3 -m recipe.bapo.main_bapo \` line.

### 3. Launch

From the repo root (e.g. `BAPO_official/`):

```bash
bash my_scripts/run_qwen3-8b_dapo_math_off.sh
```

