# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
import tempfile
import shutil
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type
import time


import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path
from verl.utils.debug.performance import _timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


class VERLAdaptiveCManager:
    """VERL框架专用的轻量级自适应C管理器 - 支持Reference Policy推断"""
    
    def __init__(self, config):
        self.config = config
        self.current_C = config.algorithm.get("initial_c_value", 0.4)
        self.update_frequency = config.algorithm.get("adaptive_c_frequency", 10)
        self.step_count = 0
        
        # 历史统计维护
        self.history_window = config.algorithm.get("history_window", 1000)
        self.mu_k_history = []
        self.mu_alpha_history = []
        
        # Reference policy相关配置
        self.alpha_policy_wg = None
        self.reward_fn = None
        self.tokenizer = None
        
        # UID级别的alpha统计缓存
        self.uid_alpha_stats = {}  # uid -> {"mu_alpha": float, "samples": list, "timestamp": int}
        self.alpha_update_frequency = config.algorithm.get("alpha_update_frequency", 20)
        self.alpha_sample_size = config.algorithm.get("alpha_sample_size", 8) # config.actor_rollout_ref.rollout.n
        self.alpha_cache_expire_steps = config.algorithm.get("alpha_cache_expire_steps", 50)
        
        # 预定义的C候选值（避免复杂优化）
        self.c_candidates = np.linspace(0.2, 0.6, 9)  # [0.2, 0.25, 0.3, ..., 0.6]
    

    def compute_filtering_conditions(self, batch: DataProto) -> dict:
        """计算过滤条件信息，但不执行实际的当前策略推理"""
        scores_alpha = batch.batch["token_level_scores"].sum(dim=-1)
        uids = batch.non_tensor_batch["uid"]
        
        # 计算每个UID的alpha策略μ值
        uid_to_alpha_mu = {}
        uid_scores_alpha = {}
        for i, uid in enumerate(uids):
            if uid not in uid_scores_alpha:
                uid_scores_alpha[uid] = []
            uid_scores_alpha[uid].append(scores_alpha[i].item())
        
        for uid, score_list in uid_scores_alpha.items():
            uid_to_alpha_mu[uid] = np.mean(score_list)
        
        # 返回过滤条件信息
        filtering_info = {
            "uid_to_alpha_mu": uid_to_alpha_mu,
            "uids_need_current_policy": [],
            "current_C": self.current_C
        }
        
        # 识别需要当前策略推理的UID（μα ≤ C）
        for uid, mu_alpha in uid_to_alpha_mu.items():
            if mu_alpha <= self.current_C:
                filtering_info["uids_need_current_policy"].append(str(uid))
        
        return filtering_info

    def set_alpha_components(self, alpha_policy_wg, reward_fn, tokenizer=None):
        """设置alpha policy组件（这里的ref_policy_wg实际上是alpha policy）"""
        self.alpha_policy_wg = alpha_policy_wg  # 重命名以避免混淆
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        print("✓ Alpha policy components set for adaptive C manager")
        
    def update_and_get_C(self, batch: DataProto) -> float:
        """更新并返回当前最优C值"""
        self.step_count += 1
        
        # 收集当前batch的统计信息
        self._collect_statistics(batch)
        
        # 定期更新reference policy的alpha统计
        if (self.step_count % self.alpha_update_frequency == 0 and 
            self.alpha_policy_wg is not None):
            self._update_alpha_policy_statistics(batch)
        
        # 每N步重新计算最优C
        if self.step_count % self.update_frequency == 0 and len(self.mu_k_history) > 50:
            self._update_optimal_C()
        
        return self.current_C
    
    def _collect_statistics(self, batch: DataProto):
        """收集统计信息（VERL数据格式适配）"""
        scores = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        
        # 按UID分组计算平均分数
        uid_to_scores = {}
        for i, uid in enumerate(uids):
            if uid not in uid_to_scores:
                uid_to_scores[uid] = []
            uid_to_scores[uid].append(scores[i])
        
        # 计算当前策略的μ值
        for uid, score_list in uid_to_scores.items():
            mu_k = np.mean(score_list)
            self.mu_k_history.append(mu_k)
            
            # 获取reference policy的μ值
            mu_alpha = self._get_alpha_mu(uid, mu_k)
            self.mu_alpha_history.append(mu_alpha)
        
        # 保持滑动窗口
        if len(self.mu_k_history) > self.history_window:
            self.mu_k_history = self.mu_k_history[-self.history_window:]
            self.mu_alpha_history = self.mu_alpha_history[-self.history_window:]
    
    def _update_alpha_policy_statistics(self, batch: DataProto):
        """使用reference policy更新alpha统计"""
        if self.alpha_policy_wg is None or self.reward_fn is None:
            print("Warning: Reference policy or reward function not available")
            return
        
        try:
            # 选择需要更新alpha统计的UIDs
            uids_to_update = []
            for uid in batch.non_tensor_batch["uid"]:
                if uid not in self.uid_alpha_stats:
                    uids_to_update.append(uid)
                else:
                    # 检查缓存是否过期
                    age = self.step_count - self.uid_alpha_stats[uid].get("timestamp", 0)
                    if age > self.alpha_cache_expire_steps:
                        uids_to_update.append(uid)
            
            # 去重并限制批次大小
            # uids_to_update = list(set(uids_to_update))[:16]
            uids_to_update = list(set(uids_to_update))
            
            if not uids_to_update:
                return
            
            print(f"[Step {self.step_count}] Updating alpha stats for {len(uids_to_update)} UIDs")
            
            # 构建reference policy推断batch
            alpha_batch = self._prepare_alpha_inference_batch(batch, uids_to_update)
            if alpha_batch is None:
                return
            
            # 使用reference policy生成样本
            alpha_generated = self._generate_with_reference_policy(alpha_batch)
            
            # 计算alpha样本的奖励
            alpha_rewards = self._compute_alpha_rewards(alpha_generated)
            
            # 更新UID的alpha统计
            self._update_uid_alpha_stats(uids_to_update, alpha_rewards)
            
        except Exception as e:
            print(f"Error updating reference policy statistics: {e}")
            import traceback
            traceback.print_exc()
    
    def _prepare_alpha_inference_batch(self, batch: DataProto, uids_to_update: list) -> DataProto:
        """准备用于alpha推断的batch"""
        # 找到需要更新的UID对应的索引
        uid_to_indices = {}
        for i, uid in enumerate(batch.non_tensor_batch["uid"]):
            if uid in uids_to_update:
                if uid not in uid_to_indices:
                    uid_to_indices[uid] = []
                uid_to_indices[uid].append(i)
        
        # 每个UID取一个样本进行推断
        selected_indices = []
        for uid, indices in uid_to_indices.items():
            selected_indices.append(indices[0])  # 取第一个样本
        
        if not selected_indices:
            return None
        
        # 提取子batch
        sub_batch_dict = {}
        sub_non_tensor_dict = {}
        
        for key, tensor in batch.batch.items():
            if isinstance(tensor, torch.Tensor):
                sub_batch_dict[key] = tensor[selected_indices]
            else:
                sub_batch_dict[key] = [tensor[i] for i in selected_indices]
        
        for key, data in batch.non_tensor_batch.items():
            if isinstance(data, np.ndarray):
                sub_non_tensor_dict[key] = data[selected_indices]
            else:
                sub_non_tensor_dict[key] = [data[i] for i in selected_indices]
        
        # 创建子batch
        alpha_batch = DataProto(
            batch=sub_batch_dict,
            non_tensor_batch=sub_non_tensor_dict
        )
        
        return alpha_batch
    
    def _generate_with_reference_policy(self, alpha_batch: DataProto) -> DataProto:
        """使用alpha policy生成样本"""
        # 准备生成batch
        gen_keys_to_remove = ["responses", "token_level_scores", "token_level_rewards", "old_log_probs"]
        gen_batch = alpha_batch.pop(
            batch_keys=[k for k in gen_keys_to_remove if k in alpha_batch.batch],
            non_tensor_batch_keys=[]
        )
        
        # 设置meta_info
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else 2,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else 0,
            "recompute_log_prob": False,
            "do_sample": True,
            "validate": False,
        }
        
        # 使用alpha policy生成
        generated_batch = self.alpha_policy_wg.generate_sequences(gen_batch)
        
        return alpha_batch.union(generated_batch)
    
    def _compute_alpha_rewards(self, alpha_batch: DataProto) -> np.ndarray:
        """计算alpha样本的奖励"""
        try:
            if callable(self.reward_fn):
                result = self.reward_fn(alpha_batch, return_dict=True)
                if isinstance(result, dict) and "reward_tensor" in result:
                    reward_tensor = result["reward_tensor"]
                else:
                    reward_tensor = result
            else:
                # 使用batch中已有的scores
                reward_tensor = alpha_batch.batch.get("token_level_scores", 
                                                     torch.zeros(len(alpha_batch.batch)))
            
            # 计算序列级奖励
            if isinstance(reward_tensor, torch.Tensor):
                rewards = reward_tensor.sum(dim=-1).cpu().numpy()
            else:
                rewards = np.array(reward_tensor)
            
            return rewards
            
        except Exception as e:
            print(f"Error computing alpha rewards: {e}")
            # 返回默认值
            return np.zeros(len(alpha_batch.batch.get("input_ids", [])))
    
    def _update_uid_alpha_stats(self, uids_to_update: list, alpha_rewards: np.ndarray):
        """更新UID的alpha统计"""
        # 假设每个UID生成了alpha_sample_size个样本
        samples_per_uid = len(alpha_rewards) // len(uids_to_update) if uids_to_update else 1
        
        for i, uid in enumerate(uids_to_update):
            start_idx = i * samples_per_uid
            end_idx = start_idx + samples_per_uid
            uid_rewards = alpha_rewards[start_idx:end_idx]
            
            if uid not in self.uid_alpha_stats:
                self.uid_alpha_stats[uid] = {
                    "samples": [],
                    "mu_alpha": 0.0,
                    "timestamp": self.step_count
                }
            
            # 添加新样本
            self.uid_alpha_stats[uid]["samples"].extend(uid_rewards.tolist())
            self.uid_alpha_stats[uid]["timestamp"] = self.step_count
            
            # 保持滑动窗口
            if len(self.uid_alpha_stats[uid]["samples"]) > 1024:
                self.uid_alpha_stats[uid]["samples"] = \
                    self.uid_alpha_stats[uid]["samples"][-1024:]
            
            # 重新计算mu_alpha
            samples = self.uid_alpha_stats[uid]["samples"]
            self.uid_alpha_stats[uid]["mu_alpha"] = np.mean(samples)

    def _get_alpha_mu(self, uid: str, current_mu: float) -> float:
        """获取reference policy的μα值 - 基于真实推断
        
        优先级：
        1. 使用缓存的reference policy真实统计
        2. 使用其他UID的alpha统计全局平均
        3. 降级到保守估计
        """
        
        # 第一优先级：使用该UID的reference policy真实统计
        if uid in self.uid_alpha_stats:
            cache_age = self.step_count - self.uid_alpha_stats[uid].get("timestamp", 0)
            if cache_age <= self.alpha_cache_expire_steps and len(self.uid_alpha_stats[uid].get("samples", [])) > 0:
                return self.uid_alpha_stats[uid]["mu_alpha"]
        
        # 第二优先级：如果有其他UID的alpha统计，使用全局平均
        if self.uid_alpha_stats:
            valid_alpha_values = [
                stats["mu_alpha"] for stats in self.uid_alpha_stats.values()
                if (len(stats.get("samples", [])) >= 3 and 
                    self.step_count - stats.get("timestamp", 0) <= self.alpha_cache_expire_steps)
            ]
            if valid_alpha_values:
                global_alpha_mean = np.mean(valid_alpha_values)
                return global_alpha_mean
        
        # 第三优先级：降级策略 - 基于理论的保守估计
        # 根据off-policy GRPO论文，reference policy通常比当前策略表现略差
        conservative_factor = self.config.algorithm.get("alpha_conservative_factor", 0.85)
        alpha_mu = current_mu * conservative_factor
        
        # 确保在合理范围内 [0, 1]
        alpha_mu = max(0.0, min(1.0, alpha_mu))
        
        return alpha_mu

    def _update_optimal_C(self):
        """更新最优C值（简化版本）"""
        if len(self.mu_k_history) < 50:
            return
        
        mu_k_array = np.array(self.mu_k_history)
        mu_alpha_array = np.array(self.mu_alpha_history)
        
        best_c = self.current_C
        best_efficiency = 0
        
        # 简化的效率计算（避免复杂优化）
        for c_candidate in self.c_candidates:
            efficiency = self._compute_simple_efficiency(mu_k_array, mu_alpha_array, c_candidate)
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_c = c_candidate
        
        # 平滑更新
        alpha = 0.7
        self.current_C = alpha * self.current_C + (1 - alpha) * best_c
        
        print(f"[Adaptive C] Updated C: {self.current_C:.3f}, Efficiency: {best_efficiency:.3f}")
    
    def _compute_simple_efficiency(self, mu_k: np.ndarray, mu_alpha: np.ndarray, C: float) -> float:
        """简化的效率计算"""
        # 条件1: 1 > μ_πk > C
        condition1 = (mu_k > C) & (mu_k < 1.0)
        
        # 条件2: μ_πk <= C and 1 > μ_α > C  
        condition2 = (mu_k <= C) & (mu_alpha > C) & (mu_alpha < 1.0)
        
        # 样本保留率
        retention_rate = np.mean(condition1 | condition2)
        
        if retention_rate < 0.1:  # 太少样本
            return 0.0
        
        # 简化的质量指标
        filtered_mask = condition1 | condition2
        if np.sum(filtered_mask) == 0:
            return 0.0
            
        filtered_mu_k = mu_k[filtered_mask]
        
        # 质量因子（简化版）
        std_quality = max(0, 1 - abs(np.std(filtered_mu_k) - 0.1))
        mean_quality = max(0, 1 - abs(np.mean(filtered_mu_k) - 0.8))
        quality_factor = 0.5 * std_quality + 0.5 * mean_quality
        
        # 简化的常数项（避免复杂计算）
        bound_constant = 1 + (1 - C) * 2  # 简化的惩罚项
        
        efficiency = (retention_rate * quality_factor) / bound_constant
        return efficiency
    
    def get_metrics(self) -> dict:
        """返回监控指标"""
        base_metrics = {
            "adaptive_c/current_C": self.current_C,
            "adaptive_c/step_count": self.step_count,
            "adaptive_c/history_size": len(self.mu_k_history)
        }
        
        # 添加reference policy相关指标
        if self.uid_alpha_stats:
            alpha_values = [stats["mu_alpha"] for stats in self.uid_alpha_stats.values()]
            base_metrics.update({
                "adaptive_c/alpha_stats_count": len(self.uid_alpha_stats),
                "adaptive_c/alpha_mu_mean": np.mean(alpha_values) if alpha_values else 0.0,
                "adaptive_c/alpha_mu_std": np.std(alpha_values) if len(alpha_values) > 1 else 0.0,
            })
        
        return base_metrics


class AdaptiveCGRPOManager:
    def __init__(self, config):
        self.config = config
        self.c_optimizer = VERLAdaptiveCManager(config)
        self.current_C = self.c_optimizer.current_C

    def set_alpha_components(self, alpha_policy_wg, reward_fn, tokenizer=None):
        """设置reference policy组件"""
        self.c_optimizer.set_alpha_components(alpha_policy_wg, reward_fn, tokenizer)
        
    def update_and_get_C(self, batch: DataProto) -> float:
        """统一接口"""
        return self.c_optimizer.update_and_get_C(batch)
    
    def compute_filtering_mask(self, batch: DataProto) -> torch.Tensor:
        """统一接口"""
        return self.c_optimizer.compute_filtering_mask(batch)
    
    def compute_filtering_conditions(self, batch: DataProto) -> dict:
        """统一接口"""
        return self.c_optimizer.compute_filtering_conditions(batch)

    def get_metrics(self) -> dict:
        """返回监控指标"""
        return self.c_optimizer.get_metrics()

        


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None, off_policy_stats=None):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    
    elif adv_estimator == AdvantageEstimator.GRPO_OFF_POLICY:
        # off_policy_stats = data.batch.get("off_policy_stats", {})
        # off_policy_stats = data.non_tensor_batch.get("off_policy_stats", {})
        off_policy_stats = off_policy_stats or {}
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]

        advantages, returns = core_algos.compute_grpo_off_policy_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            off_policy_stats=off_policy_stats,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        
    elif adv_estimator == AdvantageEstimator.GRPO_ADAPTIVE_FILTER:
        filtering_mask = data.batch.get("filtering_mask", torch.ones(data.batch.batch_size[0], dtype=torch.bool))
        # off_policy_stats = data.batch.get("off_policy_stats", None)
        # off_policy_stats = data.non_tensor_batch.get("off_policy_stats", None)
        off_policy_stats = off_policy_stats

        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
            
        advantages, returns = core_algos.compute_grpo_adaptive_filter_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            filtering_mask=filtering_mask,
            off_policy_stats=off_policy_stats,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {"token_level_rewards": data.batch["token_level_rewards"],
                      "response_mask": data.batch["response_mask"],
                      "config": config,
        }
        if "uid" in data.non_tensor_batch: # optional
            adv_kwargs['index'] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:# optional
            adv_kwargs['reward_baselines'] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GRPO_OFF_POLICY,
            AdvantageEstimator.GRPO_ADAPTIVE_FILTER,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        # 添加adaptive C manager
        if self.config.algorithm.get("use_adaptive_filtering", False):
            self.adaptive_c_manager = AdaptiveCGRPOManager(self.config)
            print("✓ Adaptive C GRPO manager initialized")
        else:
            self.adaptive_c_manager = None

        # Off-policy GRPO 配置
        self.enable_off_policy_grpo = config.algorithm.get("enable_off_policy_grpo", True)
        self.off_policy_update_freq = config.algorithm.get("off_policy_update_freq", 10)  # v 参数
        self.off_policy_stats_buffer = {}  # 存储alpha策略的统计量
        self.steps_since_alpha_update = 0  # 追踪alpha策略更新
        self.current_stage = 0

        # Alpha policy checkpoint管理
        self.alpha_policy_checkpoint = None  # alpha策略checkpoint路径
        # self._alpha_temp_dir = tempfile.mkdtemp(prefix="alpha_policy_")
        self._alpha_temp_dir = "/data/wx_data/tmp"
        os.makedirs(self._alpha_temp_dir, exist_ok=True)


        print(f"✓ Off-policy GRPO enabled: {self.enable_off_policy_grpo}")
        if self.enable_off_policy_grpo:
            print(f"✓ Alpha policy update frequency: {self.off_policy_update_freq}")
            print(f"✓ Alpha policy temp dir: {self._alpha_temp_dir}")

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)


    def _create_temp_subdir(self, prefix="temp"):
        """在alpha_temp_dir下创建子目录"""
        import uuid
        import time
        
        subdir_name = f"{prefix}_{int(time.time())}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        temp_path = os.path.join(self._alpha_temp_dir, subdir_name)
        os.makedirs(temp_path, exist_ok=True)
        return temp_path

    def _should_update_alpha_policy(self):
        """判断是否应该更新alpha policy (πθold)"""
        if not self.enable_off_policy_grpo:
            return False
        return self.steps_since_alpha_update >= self.off_policy_update_freq

    def _update_alpha_policy(self):
        """更新alpha policy (πθold) - 保存当前模型为checkpoint"""
        try:
            print(f"[Step {self.global_steps}] Updating alpha policy (πθold) from current actor...")
            
            # 生成新的checkpoint路径
            alpha_ckpt_path = os.path.join(self._alpha_temp_dir, f"alpha_step_{self.global_steps}")
            actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
            # 保存当前actor状态到alpha policy checkpoint
            self.actor_rollout_wg.save_checkpoint(
                local_path=alpha_ckpt_path,
                hdfs_path=actor_remote_path,
                global_step=self.global_steps,
            )
            
            # 清理旧的alpha policy checkpoint
            if self.alpha_policy_checkpoint is not None and os.path.exists(self.alpha_policy_checkpoint):
                try:
                    shutil.rmtree(self.alpha_policy_checkpoint)
                except:
                    pass
            
            # 更新alpha policy checkpoint路径
            self.alpha_policy_checkpoint = alpha_ckpt_path
            self.steps_since_alpha_update = 0
            
            print(f"✓ Alpha policy checkpoint saved to: {alpha_ckpt_path}")
            
        except Exception as e:
            print(f"✗ Error updating alpha policy: {e}")
            import traceback
            traceback.print_exc()
            self.steps_since_alpha_update = 0

    def _should_update_reference_policy(self):
        """判断是否应该更新reference policy (πref)"""
        stage_length = self.config.algorithm.get("stage_length", 100)
        return self.global_steps % stage_length == 0 and self.use_reference_policy

    def _update_reference_policy(self):
        """更新reference policy (πref) - 用于KL正则化"""
        try:
            print(f"[Stage {self.current_stage}] Updating reference policy (πref) from current actor...")
            
            if not self.ref_in_actor:
                # 创建临时checkpoint用于同步
                ref_temp_dir = self._create_temp_subdir("ref_policy")
                ref_ckpt_path = os.path.join(ref_temp_dir, f"ref_step_{self.global_steps}")
                actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
                try:
                    # 保存当前actor状态
                    self.actor_rollout_wg.save_checkpoint(
                        local_path=ref_ckpt_path,
                        hdfs_path=actor_remote_path,
                        global_step=self.global_steps,
                    )
                    
                    # 加载到reference policy
                    self.ref_policy_wg.load_checkpoint(
                        local_path=ref_ckpt_path,
                        del_local_after_load=False
                    )
                    
                    print(f"✓ Reference policy updated via checkpoint sync")
                    
                finally:
                    # 清理临时文件
                    try:
                        shutil.rmtree(ref_temp_dir)
                    except:
                        pass
            
            self.current_stage += 1
            print(f"✓ Reference policy (πref) successfully updated at stage {self.current_stage}")
            
        except Exception as e:
            print(f"✗ Error updating reference policy: {e}")
            import traceback
            traceback.print_exc()


    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")



    def _compute_adaptive_filtering_mask(self, batch: DataProto, filtering_info: dict) -> torch.Tensor:
        """在 trainer 中计算自适应过滤 mask"""
        scores_alpha = batch.batch["token_level_scores"].sum(dim=-1)
        uids = batch.non_tensor_batch["uid"]
        
        uid_to_alpha_mu = filtering_info["uid_to_alpha_mu"]
        uids_need_current_policy = filtering_info["uids_need_current_policy"]
        current_C = self.adaptive_c_manager.current_C
        
        # 对需要的UID进行当前策略推理
        uid_to_current_mu = {}
        if uids_need_current_policy:
            uid_to_current_mu = self._evaluate_current_policy_for_filtering(
                batch, uids_need_current_policy
            )
        
        # 应用筛选条件
        filtering_mask = torch.zeros(len(scores_alpha), dtype=torch.bool, device=scores_alpha.device)
        
        for i, uid in enumerate(uids):
            mu_alpha = uid_to_alpha_mu[uid]
            
            # 条件1: 1 > μα,r(x) > C
            if (mu_alpha > current_C) and (mu_alpha < 1.0):
                filtering_mask[i] = True
            # 条件2: μα,r(x) ≤ C and 1 > μπk,r(x) > C  
            elif mu_alpha <= current_C and str(uid) in uid_to_current_mu:
                mu_k = uid_to_current_mu[str(uid)]
                if (mu_k > current_C) and (mu_k < 1.0):
                    filtering_mask[i] = True
        
        return filtering_mask

    def _evaluate_current_policy_for_filtering(self, alpha_batch: DataProto, uids_need_current_policy: list) -> dict:
        """对指定UID用当前策略推理，用于自适应过滤判断"""
        try:
            print(f"[Filtering] Evaluating current policy for {len(uids_need_current_policy)} UIDs")
            
            if not uids_need_current_policy:
                return {}
            
            # Step 1: 选择需要评估的UID样本
            selected_indices = []
            uid_order = []
            
            for i, uid in enumerate(alpha_batch.non_tensor_batch["uid"]):
                uid_str = str(uid)
                if uid_str in uids_need_current_policy:
                    selected_indices.append(i)
                    uid_order.append(uid_str)
            
            if not selected_indices:
                return {}
            
            print(f"[Filtering] Selected {len(selected_indices)} samples from {len(alpha_batch)} total")
            
            # Step 2: 提取子集
            selected_indices_tensor = torch.tensor(selected_indices)
            test_batch = alpha_batch.select_idxs(selected_indices_tensor)
            
            # Step 3: 先repeat再处理，完全模仿_validate
            rollout_n = self.config.actor_rollout_ref.rollout.n
            # test_batch = test_batch.repeat(repeat_times=rollout_n, interleave=True)
            print(f"[Filtering] Batch size after repeat: {len(test_batch)}")
            
            # Step 4: 移除alpha policy的输出，只保留输入
            alpha_outputs_to_remove = ["responses", "token_level_scores", "token_level_rewards", 
                                    "old_log_probs", "ref_log_prob", "rollout_log_probs", "response_mask"]
            alpha_outputs_to_remove = [k for k in alpha_outputs_to_remove if k in test_batch.batch]
            
            if alpha_outputs_to_remove:
                print(f"[Filtering] Removing alpha outputs: {alpha_outputs_to_remove}")
                _ = test_batch.pop(batch_keys=alpha_outputs_to_remove, non_tensor_batch_keys=[])
            
            print(f"[Filtering] Test batch keys after cleanup: {list(test_batch.batch.keys())}")
            
            # Step 5: pop出生成字段，完全模仿_validate
            batch_keys_to_pop = ["prompts", "input_ids", "attention_mask", "position_ids"]
            non_tensor_keys_to_pop = ["raw_prompt_ids"]
            
            # 过滤存在的字段
            batch_keys_to_pop = [k for k in batch_keys_to_pop if k in test_batch.batch]
            non_tensor_keys_to_pop = [k for k in non_tensor_keys_to_pop if k in test_batch.non_tensor_batch]
            
            if not batch_keys_to_pop:
                print(f"[Filtering] No generation keys found in batch: {list(test_batch.batch.keys())}")
                return {}
            
            # pop返回用于生成的batch，test_batch保留剩余字段
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_keys_to_pop,
            )
            
            print(f"[Filtering] Generation batch keys: {list(test_gen_batch.batch.keys())}")
            print(f"[Filtering] Remaining test batch keys: {list(test_batch.batch.keys())}")
            
            # Step 6: 设置生成参数
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": True,
                "validate": True,
            }
            
            # Step 7: padding和生成
            from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
            
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            print(f"[Filtering] Padded batch size: {len(test_gen_batch_padded)}, pad_size: {pad_size}")
            
            # 用当前策略生成
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_wg.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()
            
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print(f"[Filtering] Generated batch size: {len(test_output_gen_batch)}")
            
            # Step 8: 合并数据，模仿_validate
            test_batch_final = test_batch.union(test_output_gen_batch)
            print(f"[Filtering] Final batch size: {len(test_batch_final)}")
            print(f"[Filtering] Final batch keys: {list(test_batch_final.batch.keys())}")
            
            # Step 9: 计算奖励
            try:
                result = self.reward_fn(test_batch_final, return_dict=True)
                if isinstance(result, dict) and "reward_tensor" in result:
                    reward_tensor = result["reward_tensor"]
                else:
                    reward_tensor = result
                
                rewards = reward_tensor.sum(-1).cpu().numpy()
                print(f"[Filtering] Computed rewards for {len(rewards)} samples")
                
            except Exception as e:
                print(f"[Filtering] Error computing rewards: {e}")
                rewards = np.ones(len(test_batch_final)) * 0.5
            
            # Step 10: 按UID计算μπk
            uid_to_mu_k = {}
            
            # 14. 按UID计算μπk，每个UID有rollout.n个样本
            uid_to_mu_k = {}
            
            # for i, uid_str in enumerate(uid_order):
            for i in range(0, len(uid_order), rollout_n):
                uid_str = uid_order[i]
                uid_rewards = rewards[i:i + rollout_n]  # 取这rollout_n个reward

                if len(uid_rewards) == 0:
                    # print(f"[Filtering] Warning: Empty rewards slice for UID {uid_str} at index {i}")
                    uid_to_mu_k[uid_str] = 0.0
                else:
                    uid_to_mu_k[uid_str] = np.mean(uid_rewards)
                    # print(f"[Filtering] UID {uid_str}: {len(uid_rewards)} samples, mean = {uid_to_mu_k[uid_str]:.3f}")
        
            print(f"[Filtering] Successfully evaluated {len(uid_to_mu_k)} UIDs with current policy")
            return uid_to_mu_k
            
        except Exception as e:
            print(f"[Filtering] Error in current policy evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {}
            
            print(f"[Filtering] Successfully evaluated {len(uid_to_mu_k)} UIDs with current policy")
            return uid_to_mu_k
        


    def _sample_with_alpha_policy(self, gen_batch, should_update_alpha, timing_raw):
        """使用alpha policy进行采样"""
        
        # 更新alpha policy checkpoint（如果需要）
        if should_update_alpha:
            with _timer("update_alpha_policy", timing_raw):
                self._update_alpha_policy()
        
        # 如果没有alpha policy checkpoint，使用当前模型（on-policy）
        if self.alpha_policy_checkpoint is None or not os.path.exists(self.alpha_policy_checkpoint):
            print(f"[Step {self.global_steps}] No alpha policy checkpoint, using current model")
            result = self._sample_on_policy(gen_batch, timing_raw)
            # 标记为当前策略采样（因为没有alpha checkpoint）
            self._mark_batch_source(result, is_alpha_sampling=False)
            return result
        
        # 使用alpha policy checkpoint进行采样
        with _timer("switch_to_alpha", timing_raw):
            # 创建临时目录保存当前状态
            current_temp_dir = self._create_temp_subdir("current_actor")
            current_ckpt_path = os.path.join(current_temp_dir, "current")
            actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

            try:
                print(f"[Step {self.global_steps}] Switching to alpha policy for sampling")
                
                # 1. 保存当前actor状态
                self.actor_rollout_wg.save_checkpoint(
                    local_path=current_ckpt_path,
                    hdfs_path=actor_remote_path,
                    global_step=self.global_steps,
                )
                
                # 2. 加载alpha policy状态
                self.actor_rollout_wg.load_checkpoint(
                    local_path=self.alpha_policy_checkpoint,
                    del_local_after_load=False
                )
                
                # 3. 使用alpha policy进行采样
                if not self.async_rollout_mode:
                    result = self.actor_rollout_wg.generate_sequences(gen_batch)
                else:
                    self.async_rollout_manager.wake_up()
                    result = self.async_rollout_manager.generate_sequences(gen_batch)
                    self.async_rollout_manager.sleep()
                
                # 标记为alpha策略采样
                self._mark_batch_source(result, is_alpha_sampling=True)
                
                # 4. 恢复当前actor状态
                self.actor_rollout_wg.load_checkpoint(
                    local_path=current_ckpt_path,
                    del_local_after_load=False
                )
                
                print(f"[Step {self.global_steps}] Restored current actor state")
                return result
                
            except Exception as e:
                print(f"Error in alpha policy sampling: {e}")
                # 确保恢复状态
                try:
                    self.actor_rollout_wg.load_checkpoint(
                        local_path=current_ckpt_path,
                        del_local_after_load=False
                    )
                    print("Emergency restore of current actor state")
                except:
                    print("Failed to restore current actor state!")
                raise
                
            finally:
                # 清理临时文件
                try:
                    shutil.rmtree(current_temp_dir)
                except:
                    pass

    def _sample_on_policy(self, gen_batch, timing_raw):
        """标准on-policy采样"""
        print(f"[Step {self.global_steps}] Sampling from current actor policy (on-policy)")
        if not self.async_rollout_mode:
            result = self.actor_rollout_wg.generate_sequences(gen_batch)
        else:
            self.async_rollout_manager.wake_up()
            result = self.async_rollout_manager.generate_sequences(gen_batch)
            self.async_rollout_manager.sleep()
        
        # 标记为当前策略采样
        self._mark_batch_source(result, is_alpha_sampling=False)
        return result



    def _collect_off_policy_stats(self, batch: DataProto):
        """收集alpha策略(πθold)和当前策略的统计量"""
        if not self.enable_off_policy_grpo:
            return
        
        scores = batch.batch["token_level_scores"].sum(dim=-1)
        index = batch.non_tensor_batch["uid"]
        
        # 判断当前batch是来自alpha策略还是当前策略
        # 根据采样来源进行区分
        is_alpha_sampling = getattr(batch, '_is_alpha_sampling', True)  # 默认假设是alpha采样
        
        for i, idx in enumerate(index):
            idx_str = str(idx)
            
            if idx_str not in self.off_policy_stats_buffer:
                self.off_policy_stats_buffer[idx_str] = {
                    'alpha_scores': [],      # alpha策略的分数
                    'current_scores': [],    # 当前策略的分数
                    'mu_alpha': 0.0,         # alpha策略期望奖励
                    'sigma_alpha': 1.0,      # alpha策略标准差
                    'mu_current': 0.0,       # 当前策略期望奖励
                    'sigma_current': 1.0,    # 当前策略标准差
                    'last_update_step': self.global_steps,
                    'alpha_sample_count': 0,
                    'current_sample_count': 0
                }
            
            score_value = scores[i].item()
            
            if is_alpha_sampling:
                # 来自alpha策略的采样
                self.off_policy_stats_buffer[idx_str]['alpha_scores'].append(score_value)
                self.off_policy_stats_buffer[idx_str]['alpha_sample_count'] += 1
                
                # 更新alpha策略统计量
                if len(self.off_policy_stats_buffer[idx_str]['alpha_scores']) > 1:
                    alpha_scores_tensor = torch.tensor(self.off_policy_stats_buffer[idx_str]['alpha_scores'])
                    self.off_policy_stats_buffer[idx_str]['mu_alpha'] = torch.mean(alpha_scores_tensor).item()
                    self.off_policy_stats_buffer[idx_str]['sigma_alpha'] = max(torch.std(alpha_scores_tensor).item(), 1e-6)
            else:
                # 来自当前策略的采样（用于比较和验证）
                self.off_policy_stats_buffer[idx_str]['current_scores'].append(score_value)
                self.off_policy_stats_buffer[idx_str]['current_sample_count'] += 1
                
                # 更新当前策略统计量
                if len(self.off_policy_stats_buffer[idx_str]['current_scores']) > 1:
                    current_scores_tensor = torch.tensor(self.off_policy_stats_buffer[idx_str]['current_scores'])
                    self.off_policy_stats_buffer[idx_str]['mu_current'] = torch.mean(current_scores_tensor).item()
                    self.off_policy_stats_buffer[idx_str]['sigma_current'] = max(torch.std(current_scores_tensor).item(), 1e-6)
            
            self.off_policy_stats_buffer[idx_str]['last_update_step'] = self.global_steps
            
            # 保持滑动窗口
            max_samples = 1024
            if len(self.off_policy_stats_buffer[idx_str]['alpha_scores']) > max_samples:
                self.off_policy_stats_buffer[idx_str]['alpha_scores'] = \
                    self.off_policy_stats_buffer[idx_str]['alpha_scores'][-max_samples:]
            
            if len(self.off_policy_stats_buffer[idx_str]['current_scores']) > max_samples:
                self.off_policy_stats_buffer[idx_str]['current_scores'] = \
                    self.off_policy_stats_buffer[idx_str]['current_scores'][-max_samples:]

    def _mark_batch_source(self, batch: DataProto, is_alpha_sampling: bool):
        """标记batch的来源（alpha策略 vs 当前策略）"""
        batch._is_alpha_sampling = is_alpha_sampling
        

    def _cleanup_old_stats(self):
        """清理过期的统计数据"""
        if not self.enable_off_policy_grpo:
            return
        
        # 保留最近N个更新周期的数据
        steps_to_keep = self.off_policy_update_freq * 3
        current_step = self.global_steps
        
        keys_to_remove = []
        for idx_str, stats in self.off_policy_stats_buffer.items():
            if current_step - stats['last_update_step'] > steps_to_keep:
                keys_to_remove.append(idx_str)
        
        for key in keys_to_remove:
            del self.off_policy_stats_buffer[key]
        
        if keys_to_remove:
            print(f"🧹 Cleaned up {len(keys_to_remove)} expired stat entries")


    # 设置adaptive C manager的reference policy访问
    def _setup_adaptive_c_alpha_policy(self):
        """设置自适应C管理器 - 应该使用alpha policy而不是reference policy"""
        if self.adaptive_c_manager is None:
            return
        
        # 确定alpha policy worker group (用于自适应筛选)
        alpha_policy_wg = None
        if self.use_reference_policy and not self.ref_in_actor:
            # 使用独立的reference policy worker作为alpha policy
            alpha_policy_wg = self.ref_policy_wg
            print("✓ Using reference policy worker as alpha policy for adaptive filtering")
        elif self.ref_in_actor:
            # 使用actor作为alpha policy
            alpha_policy_wg = self.actor_rollout_wg
            print("✓ Using actor as alpha policy (LoRA mode)")
        else:
            print("Warning: No alpha policy available for adaptive filtering")
            return
        
        # 注意：这里传递的是alpha policy，不是KL正则化的reference policy
        self.adaptive_c_manager.set_alpha_components(
            alpha_policy_wg=alpha_policy_wg,  # 实际上是alpha policy
            reward_fn=self.reward_fn,
            tokenizer=self.tokenizer
        )
        
        print("✓ Adaptive C manager configured with alpha policy access")


    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()


            self.alpha_policy_wg = self.ref_policy_wg
            if self.alpha_policy_wg:
                print("✓ Alpha policy worker reuses reference policy worker")

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        BaseCheckpointManager.local_mkdir(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)


    def _compute_filtering_statistics(self, batch: DataProto, current_C: float) -> dict:
        """计算过滤统计信息 - 根据论文定义正确区分mu_alpha和mu_k"""
        
        # batch中的token_level_scores来自alpha策略的采样结果
        alpha_scores = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        filtering_mask = batch.batch["filtering_mask"].cpu().numpy()
        
        # 按UID分组计算alpha策略的统计量
        uid_to_alpha_stats = {}
        for i, uid in enumerate(uids):
            uid_str = str(uid)
            if uid_str not in uid_to_alpha_stats:
                uid_to_alpha_stats[uid_str] = {"alpha_scores": [], "filtered": []}
            uid_to_alpha_stats[uid_str]["alpha_scores"].append(alpha_scores[i])
            uid_to_alpha_stats[uid_str]["filtered"].append(filtering_mask[i])
        
        # 计算各种统计指标
        condition1_count = 0  # 满足条件1: 1 > μ_α > C 的样本数
        condition2_count = 0  # 满足条件2: μ_α ≤ C and 1 > μ_πk > C 的样本数
        total_uids = len(uid_to_alpha_stats)
        
        mu_alpha_values = []  # alpha策略的期望奖励
        mu_k_values = []      # 当前策略的期望奖励
        mu_gap_values = []    # μ_πk - μ_α 的差值
        
        valid_condition1_uids = 0
        valid_condition2_uids = 0
        
        for uid_str, stats in uid_to_alpha_stats.items():
            # 计算alpha策略的期望奖励 μ_α,r(x)
            mu_alpha = np.mean(stats["alpha_scores"])
            mu_alpha_values.append(mu_alpha)
            
            # 获取当前策略的期望奖励 μ_πk,r(x)
            # 这里需要查询off_policy_stats_buffer或使用其他方法获取当前策略的统计
            if uid_str in self.off_policy_stats_buffer:
                # 从缓存中获取当前策略的历史统计
                current_policy_scores = self.off_policy_stats_buffer[uid_str].get('current_policy_scores', [])
                if current_policy_scores:
                    mu_k = np.mean(current_policy_scores)
                else:
                    # 如果没有当前策略的历史数据，使用保守估计
                    mu_k = mu_alpha * 1.1  # 假设当前策略比alpha策略略好
            else:
                # 降级策略：基于alpha策略进行估计
                improvement_factor = self.config.algorithm.get("current_policy_improvement_factor", 1.05)
                mu_k = min(1.0, mu_alpha * improvement_factor)
            
            mu_k_values.append(mu_k)
            mu_gap_values.append(mu_k - mu_alpha)
            
            # 根据论文定义统计满足各个条件的样本数量
            sample_count = len(stats["alpha_scores"])
            
            # 条件1: 1 > μ_α,r(x) > C
            if (mu_alpha > current_C) and (mu_alpha < 1.0):
                condition1_count += sample_count
                valid_condition1_uids += 1
            
            # 条件2: μ_α,r(x) ≤ C and 1 > μ_πk,r(x) > C  
            elif (mu_alpha <= current_C) and (mu_k > current_C) and (mu_k < 1.0):
                condition2_count += sample_count
                valid_condition2_uids += 1
        
        total_samples = len(alpha_scores)
        
        # 计算统计指标
        stats_dict = {
            # 基本过滤统计
            "adaptive_filtering/condition1_samples": condition1_count,
            "adaptive_filtering/condition2_samples": condition2_count,
            "adaptive_filtering/condition1_uids": valid_condition1_uids,
            "adaptive_filtering/condition2_uids": valid_condition2_uids,
            
            # Alpha策略统计 (μ_α,r(x))
            "adaptive_filtering/mu_alpha_mean": np.mean(mu_alpha_values) if mu_alpha_values else 0,
            "adaptive_filtering/mu_alpha_std": np.std(mu_alpha_values) if len(mu_alpha_values) > 1 else 0,
            "adaptive_filtering/mu_alpha_min": np.min(mu_alpha_values) if mu_alpha_values else 0,
            "adaptive_filtering/mu_alpha_max": np.max(mu_alpha_values) if mu_alpha_values else 0,
            
            # 当前策略统计 (μ_πk,r(x))
            "adaptive_filtering/mu_k_mean": np.mean(mu_k_values) if mu_k_values else 0,
            "adaptive_filtering/mu_k_std": np.std(mu_k_values) if len(mu_k_values) > 1 else 0,
            "adaptive_filtering/mu_k_min": np.min(mu_k_values) if mu_k_values else 0,
            "adaptive_filtering/mu_k_max": np.max(mu_k_values) if mu_k_values else 0,
            
            # 策略差距统计 (μ_πk - μ_α)
            "adaptive_filtering/mu_gap_mean": np.mean(mu_gap_values) if mu_gap_values else 0,
            "adaptive_filtering/mu_gap_std": np.std(mu_gap_values) if len(mu_gap_values) > 1 else 0,
            "adaptive_filtering/mu_gap_positive_rate": np.mean([gap > 0 for gap in mu_gap_values]) if mu_gap_values else 0,
            
            # 阈值相关统计
            "adaptive_filtering/alpha_above_C_rate": np.mean([mu > current_C for mu in mu_alpha_values]) if mu_alpha_values else 0,
            "adaptive_filtering/k_above_C_rate": np.mean([mu > current_C for mu in mu_k_values]) if mu_k_values else 0,
            
            # 整体统计
            "adaptive_filtering/total_uids": total_uids,
            "adaptive_filtering/effective_retention_rate": (condition1_count + condition2_count) / total_samples if total_samples > 0 else 0,
        }
        
        return stats_dict

    def _get_off_policy_buffer_stats(self) -> dict:
        """获取off-policy buffer的统计信息"""
        if not self.off_policy_stats_buffer:
            return {}
        
        alpha_sample_counts = [stats['alpha_sample_count'] for stats in self.off_policy_stats_buffer.values()]
        current_sample_counts = [stats['current_sample_count'] for stats in self.off_policy_stats_buffer.values()]
        
        alpha_mus = [stats['mu_alpha'] for stats in self.off_policy_stats_buffer.values() if stats['alpha_sample_count'] > 0]
        current_mus = [stats['mu_current'] for stats in self.off_policy_stats_buffer.values() if stats['current_sample_count'] > 0]
        
        return {
            "off_policy_buffer/total_uids": len(self.off_policy_stats_buffer),
            "off_policy_buffer/alpha_samples_total": sum(alpha_sample_counts),
            "off_policy_buffer/current_samples_total": sum(current_sample_counts),
            "off_policy_buffer/alpha_samples_mean": np.mean(alpha_sample_counts) if alpha_sample_counts else 0,
            "off_policy_buffer/current_samples_mean": np.mean(current_sample_counts) if current_sample_counts else 0,
            "off_policy_buffer/alpha_mu_mean": np.mean(alpha_mus) if alpha_mus else 0,
            "off_policy_buffer/current_mu_mean": np.mean(current_mus) if current_mus else 0,
            "off_policy_buffer/mu_improvement": np.mean(current_mus) - np.mean(alpha_mus) if alpha_mus and current_mus else 0,
        }

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # 设置adaptive C manager（在worker初始化之后）
        self._setup_adaptive_c_alpha_policy()

        # 初始化alpha policy checkpoint
        if self.enable_off_policy_grpo:
            print("✓ Initializing alpha policy checkpoint...")
            self._update_alpha_policy()  # 创建初始alpha policy checkpoint

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # === 核心：使用alpha policy进行采样 ===
                    # Step 1: 检查是否需要更新alpha policy (πθold)
                    should_update_alpha = self._should_update_alpha_policy()

                    # generate a batch
                    # Step 2: 使用当前alpha policy进行采样
                    with _timer("gen", timing_raw):
                        
                        if self.enable_off_policy_grpo:
                            gen_batch_output = self._sample_with_alpha_policy(
                                gen_batch, should_update_alpha, timing_raw
                            )
                        else:
                            gen_batch_output = self._sample_on_policy(gen_batch, timing_raw)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor


                        # === 收集off-policy统计量 ===
                        if self.enable_off_policy_grpo:
                            self._collect_off_policy_stats(batch)

                        # 添加adaptive filtering逻辑
                        if self.adaptive_c_manager is not None:
                            # 更新并获取当前最优C
                            current_C = self.adaptive_c_manager.update_and_get_C(batch)
                            
                            # 计算过滤条件信息
                            filtering_info = self.adaptive_c_manager.compute_filtering_conditions(batch)
                            
                            # 在trainer中计算筛选mask（这样可以访问async_rollout_mode等）
                            filtering_mask = self._compute_adaptive_filtering_mask(batch, filtering_info)
                            
                            # 添加到batch中
                            batch.batch["filtering_mask"] = filtering_mask
                            
                            # 记录指标
                            retention_rate = torch.mean(filtering_mask.float()).item()

                            # 计算按条件分类的样本统计
                            filtered_stats = self._compute_filtering_statistics(batch, current_C)

                            # adaptive_metrics = self.adaptive_c_manager.get_metrics()
                            adaptive_metrics = {}
                            adaptive_metrics.update(filtered_stats)
                            adaptive_metrics.update({
                                "adaptive_filtering/retention_rate": retention_rate,
                                "adaptive_filtering/current_C": current_C, 
                                "adaptive_filtering/filtered_samples": torch.sum(filtering_mask).item(),
                                "adaptive_filtering/total_samples": len(filtering_mask)
                            })

                            # 添加off-policy buffer统计
                            if self.off_policy_stats_buffer:
                                buffer_stats = self._get_off_policy_buffer_stats()
                                adaptive_metrics.update(buffer_stats)

                            metrics.update(adaptive_metrics)

                            # === 新增：准备off-policy统计量 ===
                            off_policy_stats = None
                            if self.enable_off_policy_grpo:
                                off_policy_stats = {
                                    idx_str: {
                                        'mu': stats['mu_alpha'],      # 使用alpha策略统计量
                                        'sigma': stats['sigma_alpha']
                                    }
                                    for idx_str, stats in self.off_policy_stats_buffer.items()
                                    if stats['alpha_sample_count'] + stats['current_sample_count'] >= 2
                                }
                                

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                            off_policy_stats=off_policy_stats
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    # Step 5: 更新当前策略πθ
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)


                    # Step 6: 检查是否需要更新reference policy (πref)
                    if self._should_update_reference_policy():
                        with _timer("update_ref_policy", timing_raw):
                            self._update_reference_policy()

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                    "off_policy/steps_since_alpha_update": self.steps_since_alpha_update,
                    "off_policy/alpha_checkpoint_exists": self.alpha_policy_checkpoint is not None,
                })
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.steps_since_alpha_update += 1
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def __del__(self):
        """清理资源"""
        if hasattr(self, '_alpha_temp_dir') and os.path.exists(self._alpha_temp_dir):
            try:
                # 删除目录下的所有内容，但保留目录本身
                for filename in os.listdir(self._alpha_temp_dir):
                    file_path = os.path.join(self._alpha_temp_dir, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                print(f"Cleaned up contents of alpha policy temp dir: {self._alpha_temp_dir}")
            except Exception as e:
                print(f"Failed to clean up alpha policy temp dir: {e}")