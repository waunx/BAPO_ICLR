# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO-style (PPO GRPO BAPO) Trainer with Ray-based single controller.
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
import random
from typing import Union, List


import hashlib
import base64

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from typing import Optional, Type, List, Dict
import scipy.stats as stats


from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
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


class OffPolicyDataSample:
    def __init__(self, uid: str, score: float, is_alpha: bool, sample_data: dict):
        self.uid = uid
        self.score = score
        self.is_alpha = is_alpha 
        self.sample_data = sample_data  
        self.timestamp = time.time()
    
    def to_dict(self):
        return {
            'uid': self.uid,
            'score': self.score,
            'is_alpha': self.is_alpha,
            'sample_data': self.sample_data,
            'timestamp': self.timestamp
        }


class OffPolicyGroupBuffer:    
    def __init__(self, 
                 max_samples_per_uid: int = 32,
                 max_scores_per_uid: int = 8,
                 max_total_uids: int = 4096,
                 max_high_quality_staleness_steps: int = 3):
        self.max_samples_per_uid = max_samples_per_uid
        self.max_scores_per_uid = max_scores_per_uid
        self.max_total_uids = max_total_uids
        # High-quality buffer: only keep UIDs that have been updated in the last N steps (staleness-based eviction)
        self.max_high_quality_staleness_steps = max_high_quality_staleness_steps
        
        self.alpha_scores = defaultdict(list)  # {u_i: [r_i^i, r_i^2, ...]}
        self.current_scores = defaultdict(list)  # {u_i: [r_i^i, r_i^2, ...]}
        self.sample_data = defaultdict(list)  # {u_i: [OffPolicyDataSample, OffPolicyDataSample, ...]}
        self.uid_last_step = {}  # {u_i: step} last training step when this uid was updated (for staleness eviction)
        
        self.uid_order = [] 
    
    def add_score(self, uid_str: str, scores: Union[float, List[float]], is_alpha: bool, target_range=None, buffer_range=None):
        """
        Add scores to the buffer.
        
        Args:
            uid_str: UID string
            scores: Scores, can be a single score or a list of scores
            is_alpha: Whether the scores come from the alpha policy
            target_range: Current target accuracy range from OffPolicyManager
            buffer_range: Current buffer accuracy range from OffPolicyManager
        """
        if isinstance(scores, (int, float)):
            scores = [scores]
        
        if is_alpha:
            self.alpha_scores[uid_str].extend(scores)
            if len(self.alpha_scores[uid_str]) > self.max_scores_per_uid:
                excess = len(self.alpha_scores[uid_str]) - self.max_scores_per_uid
                self.alpha_scores[uid_str] = self.alpha_scores[uid_str][excess:]
        else:
            self.current_scores[uid_str].extend(scores)
            if len(self.current_scores[uid_str]) > self.max_scores_per_uid:
                excess = len(self.current_scores[uid_str]) - self.max_scores_per_uid
                self.current_scores[uid_str] = self.current_scores[uid_str][excess:]
        
    def add_sample(self, uid_str: str, sample: OffPolicyDataSample, target_range=None, buffer_range=None, step: Optional[int] = None):
        """
        Add sample to buffer.
        
        Args:
            uid_str: UID string  
            sample: OffPolicyDataSample instance
            target_range: Current target accuracy range from OffPolicyManager
            buffer_range: Current buffer accuracy range from OffPolicyManager
            step: Current training step (for high-quality staleness-based eviction)
        """
        if uid_str in self.uid_order:
            self.uid_order.remove(uid_str)
        self.uid_order.append(uid_str)
        
        if step is not None:
            self.uid_last_step[uid_str] = step
        
        self.sample_data[uid_str].append(sample)
        
        if len(self.sample_data[uid_str]) > self.max_samples_per_uid:
            self.sample_data[uid_str].pop(0)
        
        self._check_and_evict(target_range=target_range, buffer_range=buffer_range, current_step=step)

    

    def _check_and_evict(self, target_range=None, buffer_range=None, current_step: Optional[int] = None):
        """
        Evict UIDs when buffer exceeds capacity. Ensure each category (difficult buffer & high-quality buffer) <= max_total_uids/2.
        For high-quality (target) UIDs: evict by staleness first (keep at most last max_high_quality_staleness_steps),
        then by oldest in uid_order if still over limit.
        """
        total_uids = len(self.uid_order)
        
        if total_uids <= self.max_total_uids:
            return
        
        # Set default ranges
        if target_range is None:
            target_range = (3/8, 5/8) 
        if buffer_range is None:
            buffer_range = (0/8, 1/8)
        
        max_high = 8
        max_bad = self.max_total_uids - max_high
        
        # Categorize UIDs 
        target_only_uids = []
        buffer_only_uids = []
        overlap_uids = []  # UIDs in both ranges
        other_uids = []
        
        for uid_str in self.uid_order:
            uid_stats = self.get_uid_stats(uid_str)
            if uid_stats is None:
                other_uids.append(uid_str)
                continue
                
            latest_score = uid_stats['mu_current'] if uid_stats['current_count'] > 0 else uid_stats['mu_alpha']
            
            in_target = target_range[0] <= latest_score <= target_range[1]
            in_buffer = buffer_range[0] <= latest_score <= buffer_range[1]
            
            if in_target and in_buffer:
                overlap_uids.append(uid_str)
            elif in_target:
                target_only_uids.append(uid_str)
            elif in_buffer:
                buffer_only_uids.append(uid_str)
            else:
                other_uids.append(uid_str)
        
        target_uids = target_only_uids + overlap_uids
        buffer_uids = buffer_only_uids
        
        uids_to_remove = []
        
        # 1. High-quality samples limit: evict by staleness first (only keep last max_high_quality_staleness_steps)
        if len(target_uids) > max_high:
            excess_target = len(target_uids) - max_high
            # Order target_uids by uid_order (oldest first) for deterministic eviction
            target_ordered = [uid for uid in self.uid_order if uid in target_uids]
            if current_step is not None and self.max_high_quality_staleness_steps >= 0:
                # First evict stale: last_step < current_step - max_high_quality_staleness_steps
                step_threshold = current_step - self.max_high_quality_staleness_steps
                stale_target = [uid for uid in target_ordered if self.uid_last_step.get(uid, -1) < step_threshold]
                # Then take oldest among remaining if we still need more to evict
                target_to_remove = stale_target[:excess_target]
                if len(target_to_remove) < excess_target:
                    remaining_to_evict = excess_target - len(target_to_remove)
                    remaining_target = [u for u in target_ordered if u not in target_to_remove]
                    target_to_remove.extend(remaining_target[:remaining_to_evict])
            else:
                target_to_remove = target_ordered[:excess_target]
            uids_to_remove.extend(target_to_remove)
            target_uids = [uid for uid in target_uids if uid not in target_to_remove]
        
        # 2. Difficult samples limit
        if len(buffer_uids) > max_bad:
            excess_buffer = len(buffer_uids) - max_bad
            # Remove oldest UIDs first
            buffer_to_remove = [uid for uid in self.uid_order if uid in buffer_uids][:excess_buffer]
            uids_to_remove.extend(buffer_to_remove)
            buffer_uids = [uid for uid in buffer_uids if uid not in buffer_to_remove]
        
        remaining_total = len(target_uids) + len(buffer_uids) + len(other_uids)
        if remaining_total > self.max_total_uids:
            excess_count = remaining_total - self.max_total_uids
            
            # Priority: other > buffer > target
            other_remove_count = min(len(other_uids), excess_count)
            if other_remove_count > 0:
                other_to_remove = other_uids[:other_remove_count]
                uids_to_remove.extend(other_to_remove)
                excess_count -= other_remove_count
            
            if excess_count > 0:
                buffer_remove_count = min(len(buffer_uids), excess_count)
                buffer_to_remove = [uid for uid in self.uid_order if uid in buffer_uids][:buffer_remove_count]
                uids_to_remove.extend(buffer_to_remove)
                excess_count -= buffer_remove_count
            
            if excess_count > 0:
                target_remove_count = min(len(target_uids), excess_count)
                target_to_remove = [uid for uid in self.uid_order if uid in target_uids][:target_remove_count]
                uids_to_remove.extend(target_to_remove)
        
        if uids_to_remove:
            for uid_str in uids_to_remove:
                if uid_str in self.uid_order:
                    self.uid_order.remove(uid_str)
                
                self.alpha_scores.pop(uid_str, None)
                self.current_scores.pop(uid_str, None)
                self.sample_data.pop(uid_str, None)
                self.uid_last_step.pop(uid_str, None)
                        
    def get_uid_stats(self, uid_str: str) -> Optional[dict]:
        alpha_scores = self.alpha_scores.get(uid_str, [])
        current_scores = self.current_scores.get(uid_str, [])
            
        stats = {
            'mu_alpha': np.mean(alpha_scores),
            'sigma_alpha': max(np.std(alpha_scores), 1e-6),
            'mu_current': np.mean(current_scores) if current_scores else np.mean(alpha_scores),
            'sigma_current': max(np.std(current_scores), 1e-6) if current_scores else max(np.std(alpha_scores), 1e-6),
            'alpha_count': len(alpha_scores),
            'current_count': len(current_scores),
        }
        return stats
    
    def sample_groups_by_filter(self, filter_func, target_groups: int) -> dict[str, List[OffPolicyDataSample]]:
        valid_groups = {}
        
        for uid_str, samples in self.sample_data.items():
            uid_stats = self.get_uid_stats(uid_str)
            if uid_stats is None:  
                continue
                
            if filter_func(uid_stats):  
                valid_groups[uid_str] = samples
        
        if len(valid_groups) == 0:
            return {}
        
        if target_groups >= len(valid_groups):
            return valid_groups
        else:
            selected_uids = random.sample(list(valid_groups.keys()), target_groups)
            return {uid: valid_groups[uid] for uid in selected_uids}
    
    def get_all_stats(self) -> dict[str, dict]:
        stats = {}
        for uid_str in self.uid_order:
            uid_stats = self.get_uid_stats(uid_str)
            if uid_stats:
                stats[uid_str] = uid_stats
        return stats
    
    def get_metrics(self) -> dict[str, float]:
        metrics = {}
        
        valid_uids = sum(1 for uid_str in self.uid_order if self.get_uid_stats(uid_str) is not None)
        total_samples = sum(len(samples) for samples in self.sample_data.values())
        
        metrics.update({
            'buffer/total_uids': len(self.uid_order),
            'buffer/valid_uids': valid_uids,
            'buffer/total_samples': total_samples,
        })
        
        all_alpha_scores = [score for scores in self.alpha_scores.values() for score in scores]
        all_current_scores = [score for scores in self.current_scores.values() for score in scores]
        
        if all_alpha_scores:
            metrics['buffer/alpha_score_mean'] = np.mean(all_alpha_scores)
        if all_current_scores:
            metrics['buffer/current_score_mean'] = np.mean(all_current_scores)
            
        return metrics
    
    def clear(self):
        self.alpha_scores.clear()
        self.current_scores.clear()
        self.sample_data.clear()
        self.uid_order.clear()
  

class TemporaryCurrentPolicyContext:
    """Temporary switch to current policy context manager"""
    
    def __init__(self, actor_wg):
        self.actor_wg = actor_wg
        
    def __enter__(self):
        print("[Context] Switching to current policy...")
        self.actor_wg.save_rollout_state()
        self.actor_wg.set_temp_current_policy()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("[Context] Restoring original policy...")
        try:
            self.actor_wg.clear_temp_policy_override()
        except Exception as e:
            print(f"Error restoring policy state: {e}")
        
        if exc_type is not None:
            print(f"Exception occurred in context: {exc_type.__name__}: {exc_val}")


class OffPolicyManager:
    def __init__(self, config):
        self.config = config


        self.enable_off_policy_samples = config.algorithm.get("enable_off_policy_samples", True) 
        self.enable_off_policy_rollout = config.algorithm.get("enable_off_policy_rollout", True) 
        self.enable_off_policy_reeval = config.algorithm.get("enable_off_policy_reeval", False) 
        self.enable_multi_level_downsampling = config.algorithm.get("enable_multi_level_downsampling", False)
        self.enable_adaptive_target_range = config.algorithm.get("enable_adaptive_target_range", False)
        self.enable_completion = config.algorithm.get("enable_completion_batch", False)


        self.bad_thr_c1 = config.algorithm.get("bad_thr_c1", 1/8)
        self.high_thr_c2_max = config.algorithm.get("high_thr_c2_max", 4/8)
        self.high_thr_c3_max = config.algorithm.get("high_thr_c3_max", 5/8) 
        self.high_thr_c2_min = config.algorithm.get("high_thr_c2_min", 1/8)  
        self.high_thr_c3_min = config.algorithm.get("high_thr_c3_min", 2/8) 
        
        
        self.train_batch_filter = config.algorithm.get("train_batch_filter", "gaussian")  # gaussian/range/uniform/e2h_gaussian
        self.mu = config.algorithm.get("mu", 4)
        self.sigma = config.algorithm.get("sigma", 2)
        self.total_steps = 0

        self.alpha_update_freq = config.algorithm.get("off_policy_update_freq", 10)
        self.reeval_freq = config.algorithm.get("off_policy_reeval_freq", 1)
        self.max_reeval_ids = config.algorithm.get("max_reeval_ids", 256)  

        self.lambda_1 = config.algorithm.get("lambda_1", 2.0)  # X2 weight
        self.lambda_2 = config.algorithm.get("lambda_2", 1.0)  # X3 weight
        self.max_prompt_length = config.data.get("max_prompt_length", 1024)

        self.score_candidates = [i/8 for i in range(9)]  # [0/8, 1/8, 2/8, ..., 7/8, 8/8]

        self.train_accuracy_range = (1/8, 7/8) # DAPO only
        self.target_accuracy_range = (self.high_thr_c2_min, self.high_thr_c3_min)  # high-quality samples acc thr
        self.buffer_accuracy_range = (0/8, self.bad_thr_c1)  # difficult samples acc thr
        self.promotion_threshold = (self.bad_thr_c1, 1)  # prompoted difficult samples acc thr
        
        self.step_count = 0
        
        self.buffer = OffPolicyGroupBuffer(
            max_samples_per_uid=config.algorithm.get("max_samples_per_uid", 8),
            max_scores_per_uid=config.algorithm.get("max_scores_per_uid", 32),
            max_total_uids=config.algorithm.get("max_total_uids", 1024),
            max_high_quality_staleness_steps=config.algorithm.get("max_high_quality_staleness_steps", 3)
        )

    
        self.alpha_policy_step = 0

        self.promotion_stats = {
            "total_reevaluated": 0,
            "successful_promotions": 0,
            "removed_from_buffer": 0
        }

    def _is_in_target_range(self, score: float) -> bool:
        return self.target_accuracy_range[0] <= score <= self.target_accuracy_range[1]
    
    def _is_in_buffer_range(self, score: float) -> bool:
        return self.buffer_accuracy_range[0] <= score <= self.buffer_accuracy_range[1]
    
    def _is_in_train_range(self, score: float) -> bool:
        return self.train_accuracy_range[0] <= score <= self.train_accuracy_range[1]
    
    def _is_promoted(self, score: float) -> bool:
        return self.promotion_threshold[0] < score < self.promotion_threshold[1]

    def collect_data(self, batch, is_alpha_sampling: bool, step: Optional[int] = None):
        """Collect data to buffer, based on uid group average score.
        step: current training step (for high-quality staleness-based eviction).
        """
        if not self.enable_off_policy_samples:
            return
            
        scores = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        
        # Calculate average score per uid group
        uid_to_scores = defaultdict(list)
        uid_to_indices = defaultdict(list)
        
        for i, uid in enumerate(uids):
            uid_str = str(uid)
            score = scores[i]
            uid_to_scores[uid_str].append(score)
            uid_to_indices[uid_str].append(i)
        
        # Check if each uid group should be added to buffer
        for uid_str, score_list in uid_to_scores.items():
            group_mean_score = np.mean(score_list)
            

            # Only add group to buffer if mean score is in [0, c1] range or in target range [c2, c3]
            if self._is_in_buffer_range(group_mean_score) or self._is_in_target_range(group_mean_score):
                # Add all samples in this group to buffer
                indices = uid_to_indices[uid_str]
                # Update buffer stats for this group 
                self.buffer.add_score(uid_str, score_list, is_alpha_sampling, target_range=self.target_accuracy_range, buffer_range=self.buffer_accuracy_range)
                for idx in indices:
                    sample_data = self._extract_sample_data(batch, idx)
                    
                    sample = OffPolicyDataSample(
                        uid=uid_str,
                        score=scores[idx],  # Save actual response score
                        is_alpha=is_alpha_sampling,
                        sample_data=sample_data
                    )
                    
                    self.buffer.add_sample(uid_str, sample, target_range=self.target_accuracy_range, buffer_range=self.buffer_accuracy_range, step=step)
                    

    def filter_current_batch_samples(self, batch: 'DataProto', max_ratio: float = 0.75) -> 'DataProto':
        if self.train_batch_filter == "gaussian":
            return self.gaussian_sample_from_current_batch(batch, max_ratio)
        elif self.train_batch_filter == "e2h_gaussian": 
            return self.e2h_gaussian_sample_from_batch(batch, current_step=self.step_count, total_steps=self.total_steps)
        elif self.train_batch_filter == "range": # DAPO
            return self.range_sample_from_current_batch(batch, max_ratio)
        elif self.train_batch_filter == "uniform":
            return self.uniform_sample_from_current_batch(batch, max_ratio)
        else:
            raise ValueError(f"Unknown train_batch_filter: {self.train_batch_filter}")
    

    def _get_score_index(self, score: float) -> int:
        closest_idx = 0
        min_diff = abs(score - self.score_candidates[0])
        
        for i, candidate in enumerate(self.score_candidates):
            diff = abs(score - candidate)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        return closest_idx


    def e2h_gaussian_sample_from_batch(
        self, 
        batch: 'DataProto', 
        current_step: int,
        total_steps: int,
        max_ratio: float = 0.75,
        beta: float = 0.75,
        sigma: float = 2
    ) -> 'DataProto':
        scores = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        
        uid_to_indices = defaultdict(list)
        uid_to_scores = defaultdict(list)
        for i, (uid, score) in enumerate(zip(uids, scores)):
            uid_str = str(uid)
            uid_to_indices[uid_str].append(i)
            uid_to_scores[uid_str].append(score)
        
        score_to_indices = {i: [] for i in range(9)}  # 0=hardest, 8=easiest
        for uid_str, score_list in uid_to_scores.items():
            group_mean_score = np.mean(score_list)
            score_idx = self._get_score_index(group_mean_score)
            score_to_indices[score_idx].extend(uid_to_indices[uid_str])
        
        available_score_indices = [i for i in range(9) if len(score_to_indices[i]) > 0]
        if not available_score_indices:
            print("Warning: No valid score indices found, returning original batch")
            return batch
        
        min_difficulty = min(available_score_indices)  # 0=hardest
        max_difficulty = max(available_score_indices)  # 8=easiest
        difficulty_range = max_difficulty - min_difficulty
        
        progress = current_step / total_steps if total_steps != 0 else 0.0
        
        x_t = max_difficulty - (progress ** beta) * difficulty_range
        
        print(f"E2H-G Sampling: step={current_step}/{total_steps}, progress={progress:.2f}")
        print(f"Current batch difficulty range: Level-{min_difficulty} (hardest) to Level-{max_difficulty} (easiest)")
        print(f"Dynamic sampling position x_t: {x_t:.2f}")
        
        raw_weights = {}
        for difficulty_level in available_score_indices:
            weight = np.exp(-(difficulty_level - x_t) **2 / (2 * sigma** 2))
            raw_weights[difficulty_level] = weight
        
        total_weight = sum(raw_weights.values())
        e2h_weights = {lvl: w / total_weight for lvl, w in raw_weights.items()}
        self.gaussian_weights = e2h_weights
        
        print("E2H-G Weights (difficulty level -> weight):")
        for lvl in sorted(available_score_indices):
            print(f"  Level-{lvl}: weight={e2h_weights[lvl]:.3f}, available={len(score_to_indices[lvl])}")
        
        total_samples = len(batch)
        target_total_samples = int(total_samples * max_ratio)
        selected_indices = []
        
        for difficulty_level in available_score_indices:
            weight = e2h_weights[difficulty_level]
            available_indices = score_to_indices[difficulty_level]
            target_count = int(target_total_samples * weight)
            
            if len(available_indices) <= target_count:
                selected_indices.extend(available_indices)
                print(f"Level-{difficulty_level}: selected all {len(available_indices)} samples")
            else:
                sampled = random.sample(available_indices, target_count)
                selected_indices.extend(sampled)
                print(f"Level-{difficulty_level}: sampled {target_count} from {len(available_indices)} samples")
        
        if not selected_indices:
            print("Warning: No samples selected, returning original batch")
            return batch
        
        selected_indices_tensor = torch.tensor(selected_indices)
        divisible_count = (len(selected_indices_tensor) // 8) * 8
        if divisible_count > 0:
            selected_indices_tensor = selected_indices_tensor[:divisible_count]
        else:
            print("Warning: Selected samples <8, returning original batch")
            return batch
        
        sampled_batch = batch.select_idxs(selected_indices_tensor)
        print(f"E2H-G sampling: {len(sampled_batch)}/{total_samples} selected ({len(sampled_batch)/total_samples:.1%})")
        return sampled_batch

    def gaussian_sample_from_current_batch(self, batch: 'DataProto', max_ratio: float = 0.75) -> 'DataProto':
        scores = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        
        # Group by UID and calculate mean scores
        score_to_indices = {i: [] for i in range(9)}
        uid_to_indices = defaultdict(list)
        uid_to_scores = defaultdict(list)
        
        for i, (uid, score) in enumerate(zip(uids, scores)):
            uid_str = str(uid)
            uid_to_indices[uid_str].append(i)
            uid_to_scores[uid_str].append(score)
        
        for uid_str, score_list in uid_to_scores.items():
            group_mean_score = np.mean(score_list)
            score_idx = self._get_score_index(group_mean_score)
            score_to_indices[score_idx].extend(uid_to_indices[uid_str])
        
        
        available_score_indices = [i for i in range(8) if len(score_to_indices[i]) > 0]
        
        if not available_score_indices:
            print("Warning: No valid score indices found, returning original batch")
            return batch
        
        min_score_idx = min(available_score_indices)
        max_score_idx = max(available_score_indices)
        
        # print(f"Current batch score range: {self.score_candidates[min_score_idx]:.3f} to {self.score_candidates[max_score_idx]:.3f}")
        print(f"Current batch score range: Level-{min_score_idx} to Level-{max_score_idx}")
        print(f"Available score indices: {available_score_indices}")
        
        total_samples = len(batch)
        target_total_samples = int(total_samples * max_ratio)
        
        
        raw_weights = {}
        for score_idx in available_score_indices:
            
            range_center = (min_score_idx + max_score_idx) / 2.0
            weight = stats.norm.pdf(score_idx, range_center, self.sigma)
            raw_weights[score_idx] = weight
        
        
        total_weight = sum(raw_weights.values())
        self.gaussian_weights = {score_idx: w/total_weight for score_idx, w in raw_weights.items()}
        
        print("Current Gaussian weights for available score ranges:")
        for score_idx in available_score_indices:
            score_value = self.score_candidates[score_idx]
            weight = self.gaussian_weights[score_idx]
            sample_count = len(score_to_indices[score_idx])
            print(f"  {score_value:.3f}: weight={weight:.3f}, available={sample_count} samples")
        
        selected_indices = []
        
        
        for score_idx in available_score_indices:
            weight = self.gaussian_weights[score_idx]
            available_indices = score_to_indices[score_idx]
            target_count = int(target_total_samples * weight)
            
            if len(available_indices) <= target_count:
                
                selected_indices.extend(available_indices)
                print(f"Score {self.score_candidates[score_idx]:.3f}: selected all {len(available_indices)} samples")
            else:
                
                sampled = random.sample(available_indices, target_count)
                selected_indices.extend(sampled)
                print(f"Score {self.score_candidates[score_idx]:.3f}: sampled {target_count} from {len(available_indices)} samples")
        
        if not selected_indices:
            print("Warning: No samples selected in gaussian sampling, returning original batch")
            return batch
        
        
        selected_indices_tensor = torch.tensor(selected_indices)
        divisible_count = (len(selected_indices_tensor) // 8) * 8
        if divisible_count > 0:
            selected_indices_tensor = selected_indices_tensor[:divisible_count]
        else:
            print("Warning: Selected samples less than 8, returning original batch")
            return batch
        
        sampled_batch = batch.select_idxs(selected_indices_tensor)
        
        print(f"Gaussian sampling: {len(sampled_batch)}/{total_samples} samples selected ({len(sampled_batch)/total_samples:.1%})")
        print(f"✓ Truncated to {len(sampled_batch)} samples (divisible by 8)")
        return sampled_batch

    def range_sample_from_current_batch(self, batch: 'DataProto', max_ratio: float = 0.75) -> 'DataProto':
        scores = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        
        uid_to_indices = defaultdict(list)
        uid_to_scores = defaultdict(list)
        
        for i, (uid, score) in enumerate(zip(uids, scores)):
            uid_str = str(uid)
            uid_to_indices[uid_str].append(i)
            uid_to_scores[uid_str].append(score)
        
        selected_indices = []
        for uid_str, score_list in uid_to_scores.items():
            group_mean_score = np.mean(score_list)
            
            if self._is_in_train_range(group_mean_score):
                selected_indices.extend(uid_to_indices[uid_str])
        
        if not selected_indices:
            print("Warning: No samples in range filter, returning original batch")
            return batch
        
        
        total_samples = len(batch)
        target_total_samples = int(total_samples * max_ratio)
        
        if len(selected_indices) > target_total_samples:
            selected_indices = random.sample(selected_indices, target_total_samples)
        
        selected_indices_tensor = torch.tensor(selected_indices)
        divisible_count = (len(selected_indices_tensor) // 8) * 8
        if divisible_count > 0:
            selected_indices_tensor = selected_indices_tensor[:divisible_count]
            sampled_batch = batch.select_idxs(selected_indices_tensor)
            print(f"Range sampling: {len(sampled_batch)}/{total_samples} samples selected")
            return sampled_batch
        else:
            print("Warning: Selected samples less than 8, returning original batch")
            return batch

    def uniform_sample_from_current_batch(self, batch: 'DataProto', max_ratio: float = 0.75) -> 'DataProto':
        total_samples = len(batch)
        target_total_samples = int(total_samples * max_ratio)
        
        
        all_indices = list(range(total_samples))
        selected_indices = random.sample(all_indices, min(target_total_samples, total_samples))
        
        selected_indices_tensor = torch.tensor(selected_indices)
        divisible_count = (len(selected_indices_tensor) // 8) * 8
        if divisible_count > 0:
            selected_indices_tensor = selected_indices_tensor[:divisible_count]
            sampled_batch = batch.select_idxs(selected_indices_tensor)
            print(f"Uniform sampling: {len(sampled_batch)}/{total_samples} samples selected")
            return sampled_batch
        else:
            print("Warning: Selected samples less than 8, returning original batch")
            return batch

    def sample_filtered_groups(self, target_groups: int) -> Optional['DataProto']:
        """Sample high-quality data from buffer for training"""
        if not self.enable_off_policy_samples:
            return None
        
        def filter_func(stats):
            mu_alpha = stats['mu_alpha']
            mu_current = stats['mu_current']
            
            # Use latest score for evaluation
            latest_score = mu_current if stats['current_count'] > 0 else mu_alpha
            
            # Only select data in target learning range
            return self._is_in_target_range(latest_score)
        
        sampled_groups = self.buffer.sample_groups_by_filter(filter_func, target_groups)
        
        if not sampled_groups:
            return None
        
        all_samples = []
        for uid_str, samples in sampled_groups.items():
            all_samples.extend(samples)
        
        # print(f"Sampled {len(all_samples)} samples from buffer for training")
        return self._construct_dataproto_from_samples(all_samples)
    
    def get_available_groups_count(self) -> int:
        """Get count of available data in buffer for 4/8-7/8 range"""
        if not self.enable_off_policy_samples:
            return 0
        
        def filter_func(stats):
            mu_alpha = stats['mu_alpha']
            mu_current = stats['mu_current']
            
            latest_score = mu_current if stats['current_count'] > 0 else mu_alpha
            return self._is_in_target_range(latest_score)
        
        count = 0
        for uid_str in self.buffer.sample_data.keys():
            uid_stats = self.buffer.get_uid_stats(uid_str)
            if uid_stats and filter_func(uid_stats):
                count += 1
        
        return count
    
    def reevaluate_and_promote_buffer_samples(self, actor_wg, reward_fn, tokenizer) -> Optional['DataProto']:
        """Re-evaluate samples in buffer, if group average score is in 4/8-7/8 range, promote to training set and remove from buffer"""
        if not self.enable_off_policy_samples:
            return None
        
        # Get all samples from buffer, grouped by uid
        uid_to_samples = {}
        uid_sample = {}
        uid_to_original_mean_score = {}

        def filter_func(stats):
            mu_alpha = stats['mu_alpha']

            latest_score = mu_alpha
            return self._is_in_buffer_range(latest_score)
        
        for uid_str, samples in self.buffer.sample_data.items():
            uid_stats = self.buffer.get_uid_stats(uid_str)

            if samples and uid_stats and filter_func(uid_stats):
                uid_to_samples[uid_str] = samples

                uid_sample[uid_str] = samples[0]
                # Compute average score for all samples in this uid group
                sample_scores = [sample.score for sample in samples]
                uid_to_original_mean_score[uid_str] = np.mean(sample_scores)
        
        if not uid_sample:
            return None
        
        print(f"Found {len(uid_sample)} groups in buffer for re-evaluation...")
        
        MAX_REEVALUATE_SAMPLES = self.max_reeval_ids
    
        # Ensure sample count is divisible by 8 (because we want to distribute to 8 GPUs)
        available_uids = list(uid_sample.keys())
        if len(available_uids) > MAX_REEVALUATE_SAMPLES:
            import random
            available_uids = random.sample(available_uids, MAX_REEVALUATE_SAMPLES)
            print(f"Limited re-evaluation from {len(uid_sample)} to {MAX_REEVALUATE_SAMPLES} groups")

        target_count = (len(available_uids) // actor_wg.world_size) * actor_wg.world_size  
        
        if target_count == 0:
            print(f"Not enough samples for re-evaluation (need at least {actor_wg.world_size})")
            return None
        
        # If we need to truncate, only keep the first target_count samples
        if target_count < len(available_uids):
            selected_uids = available_uids[:target_count]
            print(f"Truncated from {len(available_uids)} to {target_count} samples to ensure divisibility by {actor_wg.world_size}")
        else:
            selected_uids = available_uids
        
        # Build samples for re-evaluation - each uid one sample
        selected_samples = [uid_sample[uid] for uid in selected_uids]
        
        self.promotion_stats["total_reevaluated"] += len(selected_uids)
        print(f"Re-evaluating {len(selected_uids)} groups (will generate {len(selected_uids)} * {self.config.actor_rollout_ref.rollout.n} responses)...")
        
        with TemporaryCurrentPolicyContext(actor_wg):
            reevaluated_batch = self._construct_dataproto_from_samples_for_re_eval(selected_samples)


            keys_to_remove = ["prompts", "responses", "token_level_scores", "token_level_rewards", 
                            "old_log_probs", "ref_log_prob", "rollout_log_probs", "response_mask"]
            for key in keys_to_remove:
                if key in reevaluated_batch.batch:
                    reevaluated_batch.batch.pop(key)
                
            gen_keys = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_keys = ["raw_prompt_ids"]
            
            if "multi_modal_data" in reevaluated_batch.non_tensor_batch:
                non_tensor_keys.append("multi_modal_data")
            if "raw_prompt" in reevaluated_batch.non_tensor_batch:
                non_tensor_keys.append("raw_prompt")
            if "tools_kwargs" in reevaluated_batch.non_tensor_batch:
                non_tensor_keys.append("tools_kwargs")

            batch_keys_to_pop = [k for k in gen_keys if k in reevaluated_batch.batch]
            non_tensor_keys_to_pop = [k for k in non_tensor_keys if k in reevaluated_batch.non_tensor_batch]
            
            if not batch_keys_to_pop:
                return None
            
            gen_batch = reevaluated_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_keys_to_pop
            )
            
            assert len(gen_batch) == target_count, f"gen_batch size {len(gen_batch)} != target_count {target_count}"
            assert len(gen_batch) % actor_wg.world_size == 0, f"gen_batch size {len(gen_batch)} is not divisible by actor_wg.world_size {actor_wg.world_size}"
            
            gen_batch.meta_info = {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": True,
                "is_re_rollout": True,  # Set flag, generate rollout.n responses
            }
            
            try:
                # Re-generate using current policy - each prompt will generate rollout.n * 2 responses
                generated = actor_wg.generate_sequences(gen_batch)
                # reevaluated_batch = reevaluated_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n * 2, interleave=True)
                reevaluated_batch = reevaluated_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                reevaluated_full_batch = reevaluated_batch.union(generated)

                rollout_n = self.config.actor_rollout_ref.rollout.n
                # expected_total_samples = target_count * rollout_n * 2
                expected_total_samples = target_count * rollout_n
                actual_total_samples = len(reevaluated_full_batch)
                print(f"Generated {actual_total_samples} responses for {target_count} prompts (expected: {expected_total_samples})")
                
                # Calculate new reward
                if callable(reward_fn):
                    result = reward_fn(reevaluated_full_batch, return_dict=True)
                    if isinstance(result, dict) and "reward_tensor" in result:
                        reward_tensor = result["reward_tensor"]
                    else:
                        reward_tensor = result
                else:
                    reward_tensor = torch.zeros(len(reevaluated_full_batch))
                
                reevaluated_full_batch.batch["token_level_scores"] = reward_tensor

                # Check which groups are promoted - need to calculate mean score for each group and down sample
                all_scores = reward_tensor.sum(-1).cpu().numpy()
                promoted_sample_indices = []
                uids_to_remove = []
                uids_to_promoted = []
                
                
                for i, uid_str in enumerate(selected_uids):
                    
                    start_idx = i * rollout_n
                    end_idx = start_idx + rollout_n
                    group_scores = all_scores[start_idx:end_idx]
                    
                    reevaluated_score = np.mean(group_scores)
            
                    original_score = uid_to_original_mean_score[uid_str]
                    
                    if reevaluated_score > original_score:
                    # if self._is_promoted(reevaluated_score):

                        global_indices = [start_idx + local_idx for local_idx in range(rollout_n)]
                        promoted_sample_indices.extend(global_indices)
                        uids_to_promoted.append(uid_str)

                        self.promotion_stats["successful_promotions"] += 1

                    # if not (self._is_in_buffer_range(group_mean_score) or self._is_in_target_range(group_mean_score)):
                    # if self._is_promoted(reevaluated_score):
                    #     uids_to_remove.append(uid_str)

                    # update buffer score
                    # self.buffer.add_score(uid_str, group_mean_score, is_alpha=False)
                    self.buffer.add_score(uid_str, group_scores.tolist(), is_alpha=False, target_range=self.target_accuracy_range, buffer_range=self.buffer_accuracy_range)
                    
                # remove from buffer promoted groups
                for uid_str in uids_to_remove:
                    if uid_str in self.buffer.sample_data:
                        del self.buffer.sample_data[uid_str]
                    if uid_str in self.buffer.alpha_scores:
                        del self.buffer.alpha_scores[uid_str]
                    if uid_str in self.buffer.current_scores:
                        del self.buffer.current_scores[uid_str]
                    if uid_str in self.buffer.uid_order:
                        self.buffer.uid_order.remove(uid_str)
                    self.promotion_stats["removed_from_buffer"] += 1
                
                # return promoted samples for training
                if promoted_sample_indices:
                    promoted_indices = torch.tensor(promoted_sample_indices)
                    promoted_batch = reevaluated_full_batch.select_idxs(promoted_indices)
                    
                    print(f"🎉 {len(uids_to_promoted)} groups ({len(promoted_sample_indices)} responses) promoted from buffer to training!")
                    return promoted_batch
                else:
                    print("No groups were promoted from buffer")
                    return None
                    
            except Exception as e:
                print(f"Error in buffer re-evaluation: {e}")
                return None

    def _update_adaptive_target_range(self):

        if not self.enable_adaptive_target_range:
            return {}
    
        buffer_metrics = self.buffer.get_metrics()
        alpha_score_mean = buffer_metrics.get('buffer/alpha_score_mean', None)


        if alpha_score_mean is None:
            return {}
        
        total_uids = buffer_metrics.get('buffer/total_uids', 0)
        if total_uids < 5: 
            return {}
        
    
        # When alpha_score_mean is lower, select higher score samples (easy samples, easier to learn)
        c2_high, c3_high = self.high_thr_c2_max, self.high_thr_c3_max  
        # When alpha_score_mean is higher, select lower score samples (hard samples, more challenging)
        c2_low, c3_low = self.high_thr_c2_min, self.high_thr_c3_min
        
        alpha_min, alpha_max = 0.2, 0.8
        
        if alpha_score_mean <= alpha_min:
            ratio = 0.0  #  easy_range
        elif alpha_score_mean >= alpha_max:
            ratio = 1.0  #  hard_range
        else:
            ratio = (alpha_score_mean - alpha_min) / (alpha_max - alpha_min)
        
   
        new_lower = c2_high - (c2_high - c2_low) * ratio
        new_upper = c3_high - (c3_high - c3_low) * ratio
        
        # Update target range
        self.target_accuracy_range = (new_lower, new_upper)
        
        return {
            "adaptive_range/new_lower": new_lower,
            "adaptive_range/new_upper": new_upper,
        }

    def get_metrics(self):

        if not self.enable_off_policy_samples:
            return {}
            
        buffer_metrics = self.buffer.get_metrics()
        

        all_stats = self.buffer.get_all_stats()
        
        buffer_range_count = 0
        target_range_count = 0
        promoted_range_count = 0
        
        for uid_str, stats in all_stats.items():
            latest_score = stats['mu_current'] if stats['current_count'] > 0 else stats['mu_alpha']
            
            if self._is_in_buffer_range(latest_score):
                buffer_range_count += 1
            if self._is_in_target_range(latest_score):
                target_range_count += 1
            # elif self._is_promoted(latest_score):
            #     promoted_range_count += 1
        
        promotion_rate = (self.promotion_stats["successful_promotions"] / 
                         max(self.promotion_stats["total_reevaluated"], 1))
        remove_rate = (self.promotion_stats["removed_from_buffer"] / 
                       max(self.promotion_stats["total_reevaluated"], 1))

        metrics = {
            "off_policy/buffer_range_samples": buffer_range_count,      
            "off_policy/target_range_samples": target_range_count,     
            "off_policy/promotion_rate": promotion_rate,
            "off_policy/remove_rate": remove_rate,
            "off_policy/total_reevaluated": self.promotion_stats["total_reevaluated"],
            "off_policy/buffer_size": buffer_metrics.get('buffer/total_uids', 0),
        }
        
        metrics.update({f"off_policy/{k}": v for k, v in buffer_metrics.items()})
        
        return metrics

    def _extract_sample_data(self, batch, idx: int) -> dict:
        sample_data = {}
    
        for key, tensor in batch.batch.items():
            sample_data[key] = tensor[idx].clone().detach()

        for key, values in batch.non_tensor_batch.items():
            sample_data[key] = values[idx]
                
        return sample_data
    
    def _construct_dataproto_from_samples(self, samples):
        if not samples:
            raise ValueError("No samples provided")
        
        tensor_data = {}
        non_tensor_data = {}
        
        tensor_keys = set()
        non_tensor_keys = set()
        
        for sample in samples:
            for key, value in sample.sample_data.items():
                
                if torch.is_tensor(value):
                    tensor_keys.add(key)
                else:
                    non_tensor_keys.add(key)
        
        for key in tensor_keys:
            values = []
            for sample in samples:
                if key in sample.sample_data:
                    # we should reuse the old_log_probs for buffer samples
                    if key in ['input_ids', 'position_ids', 'attention_mask', 'old_log_probs']:
                        values.append(sample.sample_data[key])
                    else:
                        values.append(sample.sample_data[key])
                else:
                    print(f"Warning: Sample {sample.uid} missing tensor key {key}")
                    
            if values:
                tensor_data[key] = torch.stack(values)
        
        for key in non_tensor_keys:
            values = []
            for sample in samples:
                if key in sample.sample_data:
                    values.append(sample.sample_data[key])
                else:
                    values.append(None)  
            non_tensor_data[key] = np.array(values, dtype=object)
            
        dataproto = DataProto.from_dict(
            tensors=tensor_data,
            non_tensors=non_tensor_data
        )
        
        return dataproto
    
    def _construct_dataproto_from_samples_for_re_eval(self, samples):
        if not samples:
            raise ValueError("No samples provided")
        
        tensor_data = {}
        non_tensor_data = {}
        
        tensor_keys = set()
        non_tensor_keys = set()
        
        first_sample = samples[0]
        gen_keys = list(first_sample.sample_data.keys())

        # cause re-eval use current training policy, so we need re-compute log_prob
        if "gen_multi_modal_data" in gen_keys:
            
            gen_tensor_keys = ["gen_input_ids", "gen_attention_mask", "gen_position_ids"]
            gen_non_tensor_keys = ["gen_raw_prompt_ids", "gen_multi_modal_data", "gen_tools_kwargs", "gen_multi_modal_inputs"]
            
            for gen_key in gen_tensor_keys:
                if gen_key in samples[0].sample_data:  
                    original_key = gen_key[4:] 
                    values = []
                    for sample in samples:
                        if gen_key in sample.sample_data:
                            # print(f"key:{original_key}, sample.sample_data[{gen_key}]: {sample.sample_data[gen_key].shape}")
                            values.append(sample.sample_data[gen_key])
                        else:
                            print(f"Warning: Sample {sample.uid} missing gen key {gen_key}")
                            
                    if values:
                        tensor_data[original_key] = torch.stack(values)
            
            for gen_key in gen_non_tensor_keys:
                if gen_key in samples[0].sample_data: 
                    original_key = gen_key[4:]  
                    values = []
                    for sample in samples:
                        if gen_key in sample.sample_data:
                            # print(f"key:{original_key}, sample.sample_data[{gen_key}]: {sample.sample_data[gen_key]}")
                            values.append(sample.sample_data[gen_key])
                        else:
                            values.append(None)  
                    non_tensor_data[original_key] = np.array(values, dtype=object)

            # print(f"Available keys in sample_data: {list(first_sample.sample_data.keys())}")
            # ['responses', 'position_ids', 'attention_mask', 'prompts', 'rollout_log_probs', 'input_ids', 'response_mask', 'token_level_scores', 
            # 'old_log_probs', 'ref_log_prob', 'token_level_rewards', 'gen_input_ids', 'gen_attention_mask', 'gen_position_ids', 'data_source', 'ability', 'reward_model', 'extra_info', 
            #  'multi_modal_inputs', 'index', 'uid', 'tools_kwargs', 'multi_modal_data', 'gen_raw_prompt_ids', 'gen_tools_kwargs', 'gen_multi_modal_data']

            other_keys = ['uid', 'data_source', 'ability', 'reward_model', 'extra_info', 'index', 'tools_kwargs']
            
            for key in other_keys:
                try:
                    if key in first_sample.sample_data:
                        values = []
                        for sample in samples:
                            if key in sample.sample_data:
                                values.append(sample.sample_data[key])
                            else:
                                values.append(None)
                        non_tensor_data[key] = np.array(values, dtype=object)
                        # print(f"Added other key: {key}")
                    else:
                        print(f"Other key {key} not found in samples")
                except Exception as e:
                    print(f"Error processing key {key}: {e}")
                    continue
        else:
            for sample in samples:
                for key, value in sample.sample_data.items():
                    
                    if torch.is_tensor(value):
                        tensor_keys.add(key)
                    else:
                        non_tensor_keys.add(key)
            
            for key in tensor_keys:
                values = []
                for sample in samples:
                    if key in sample.sample_data:
                        if key in ['input_ids', 'position_ids', 'attention_mask']:
                            values.append(sample.sample_data[key][:self.max_prompt_length])
                        else:
                            values.append(sample.sample_data[key])
                    else:
                        print(f"Warning: Sample {sample.uid} missing tensor key {key}")
                        
                if values:
                    tensor_data[key] = torch.stack(values)
            
            for key in non_tensor_keys:
                values = []
                for sample in samples:
                    if key in sample.sample_data:
                        values.append(sample.sample_data[key])
                    else:
                        values.append(None)  
                non_tensor_data[key] = np.array(values, dtype=object)
        

        dataproto = DataProto.from_dict(
            tensors=tensor_data,
            non_tensors=non_tensor_data
        )
        
        return dataproto
    
    def get_off_policy_stats(self) -> Optional[dict]:
        if not self.enable_off_policy_samples:
            return None
        
        all_stats = self.buffer.get_all_stats()
        
        result = {}
        for uid_str, stats in all_stats.items():
            if stats['current_count'] > 0:
                result[uid_str] = {
                    'mu': stats['mu_current'],
                    'sigma': stats['sigma_current'],
                    'source': 'current',
                }

            else:
                result[uid_str] = {
                    'mu': stats['mu_alpha'],
                    'sigma': stats['sigma_alpha'],
                    'source': 'alpha',
                }
        
        return result
    

    def should_update_vllm_params(self) -> bool:
        """
        whether should update vllm params or not
        """
        if not self.enable_off_policy_rollout:
            return True  
        return self.step_count % self.alpha_update_freq == 0
    
    def prepare_generation(self, actor_wg, global_steps):
        """
        prepare generation params for vllm
        if should_update, then update vllm params to current fsdp params
        """
        should_update = self.should_update_vllm_params()
        try:
            actor_wg.set_param_update_control(
                should_update=should_update,
                current_step=global_steps
            )
            
            policy_type = "current" if should_update else "alpha"
            print(f"[Step {global_steps}] Using {policy_type} policy for generation")
            
        except Exception as e:
            print(f"Warning: Failed to set param update control: {e}")
            should_update = True  
        
        return should_update

    def sample_with_alpha_policy(self, gen_batch, actor_wg, global_steps):
        """
        using delayed vllm policy to sample generations
        """
        if not self.enable_off_policy_rollout:
            return actor_wg.generate_sequences(gen_batch), False
        
        try:
            self.prepare_generation(actor_wg, global_steps)
            result = actor_wg.generate_sequences(gen_batch)
            return result, True  # is_alpha_sampling
            
        except Exception as e:
            print(f"Error in controlled sampling: {e}")
            return actor_wg.generate_sequences(gen_batch), False


class PromptAccuracyTracker:
    """
    Tracks accuracy changes for individual prompts during RL training.
    Records baseline accuracy and monitors improvement over training steps.
    """
    
    def __init__(self, 
                 save_dir: str = "./prompt_tracking",
                 track_top_k_improved: int = 50,
                 track_top_k_degraded: int = 20):
        """
        Args:
            save_dir: Directory to save tracking results
            track_top_k_improved: Number of most improved prompts to track
            track_top_k_degraded: Number of most degraded prompts to track
        """
        self.save_dir = save_dir
        self.track_top_k_improved = track_top_k_improved
        self.track_top_k_degraded = track_top_k_degraded
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Storage for prompt data
        self.prompt_data = {}  # {prompt_hash: {"content": str, "baseline_acc": float, "history": [(step, acc), ...]}}
        self.baseline_recorded = False
        
    def _hash_prompt(self, prompt: str) -> str:
        """Create a consistent hash for prompt identification"""
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()[:16]
    
    def record_baseline(self, prompts: List[str], accuracies: List[float], step: int = 0):
        """
        Record baseline accuracy for each prompt at the beginning of training.
        
        Args:
            prompts: List of prompt texts
            accuracies: List of corresponding accuracies
            step: Training step (usually 0 for baseline)
        """
        if self.baseline_recorded:
            print("Warning: Baseline already recorded. Skipping...")
            return
            
        print(f"Recording baseline accuracy for {len(prompts)} prompts at step {step}")
        
        for prompt, acc in zip(prompts, accuracies):
            prompt_hash = self._hash_prompt(prompt)
            self.prompt_data[prompt_hash] = {
                "content": prompt,
                "baseline_acc": acc,
                "history": [(step, acc)]
            }
        
        self.baseline_recorded = True
        self._save_baseline()
        print(f"✓ Baseline recorded for {len(self.prompt_data)} unique prompts")
    
    def update_accuracies(self, prompts: List[str], accuracies: List[float], step: int):
        """
        Update accuracy for prompts at current training step.
        
        Args:
            prompts: List of prompt texts  
            accuracies: List of corresponding accuracies
            step: Current training step
        """
        if not self.baseline_recorded:
            print("Warning: No baseline recorded yet. Call record_baseline() first.")
            return
            
        updated_count = 0
        new_prompts = 0
        
        for prompt, acc in zip(prompts, accuracies):
            prompt_hash = self._hash_prompt(prompt)
            
            if prompt_hash in self.prompt_data:
                self.prompt_data[prompt_hash]["history"].append((step, acc))
                updated_count += 1
            else:
                # New prompt not in baseline (shouldn't happen with fixed eval set)
                print(f"Warning: New prompt found at step {step}: {prompt[:50]}...")
                new_prompts += 1
        
        print(f"Updated accuracy for {updated_count} prompts at step {step}")
        if new_prompts > 0:
            print(f"Warning: {new_prompts} new prompts not in baseline")
            
        # Save tracking results periodically
        self._save_tracking_results(step)
    
    def get_improvement_stats(self, step: int) -> Dict:
        """
        Calculate improvement statistics for current step.
        
        Args:
            step: Current training step
            
        Returns:
            Dictionary with improvement statistics
        """
        if not self.baseline_recorded:
            return {}
            
        improvements = []
        degradations = []
        
        for prompt_hash, data in self.prompt_data.items():
            if len(data["history"]) < 2:
                continue
                
            baseline_acc = data["baseline_acc"]
            current_acc = data["history"][-1][1]  # Latest accuracy
            improvement = current_acc - baseline_acc
            
            if improvement > 0:
                improvements.append({
                    "prompt_hash": prompt_hash,
                    "content": data["content"][:100] + "..." if len(data["content"]) > 100 else data["content"],
                    "baseline_acc": baseline_acc,
                    "current_acc": current_acc,
                    "improvement": improvement
                })
            elif improvement < 0:
                degradations.append({
                    "prompt_hash": prompt_hash,
                    "content": data["content"][:100] + "..." if len(data["content"]) > 100 else data["content"],
                    "baseline_acc": baseline_acc,
                    "current_acc": current_acc,
                    "degradation": abs(improvement)
                })
        
        # Sort by improvement/degradation magnitude
        improvements.sort(key=lambda x: x["improvement"], reverse=True)
        degradations.sort(key=lambda x: x["degradation"], reverse=True)
        
        total_prompts = len(self.prompt_data)
        improved_prompts = len(improvements)
        degraded_prompts = len(degradations)
        unchanged_prompts = total_prompts - improved_prompts - degraded_prompts
        
        stats = {
            "step": step,
            "total_prompts": total_prompts,
            "improved_prompts": improved_prompts,
            "degraded_prompts": degraded_prompts,
            "unchanged_prompts": unchanged_prompts,
            "improvement_rate": improved_prompts / total_prompts if total_prompts > 0 else 0,
            "degradation_rate": degraded_prompts / total_prompts if total_prompts > 0 else 0,
            "top_improved": improvements[:self.track_top_k_improved],
            "top_degraded": degradations[:self.track_top_k_degraded],
        }
        
        if improvements:
            stats["avg_improvement"] = np.mean([x["improvement"] for x in improvements])
            stats["max_improvement"] = improvements[0]["improvement"]
        
        if degradations:
            stats["avg_degradation"] = np.mean([x["degradation"] for x in degradations])
            stats["max_degradation"] = degradations[0]["degradation"]
            
        return stats
    
    def _save_baseline(self):
        """Save baseline data to file"""
        baseline_file = os.path.join(self.save_dir, "baseline.json")
        baseline_data = {
            "prompts": len(self.prompt_data),
            "data": {k: {"content": v["content"], "baseline_acc": v["baseline_acc"]} 
                    for k, v in self.prompt_data.items()}
        }
        
        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, ensure_ascii=False, indent=2)
        
        print(f"Baseline saved to {baseline_file}")
    
    def _save_tracking_results(self, step: int):
        """Save current tracking results"""
        tracking_file = os.path.join(self.save_dir, f"tracking_step_{step}.json")
        
        # Get improvement stats
        stats = self.get_improvement_stats(step)
        
        # Full history data
        history_data = {
            "step": step,
            "stats": stats,
            "full_history": {k: v["history"] for k, v in self.prompt_data.items()}
        }
        
        with open(tracking_file, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        
        # Also save a summary
        summary_file = os.path.join(self.save_dir, "latest_summary.json")
        summary = {
            "last_updated_step": step,
            "total_prompts": stats.get("total_prompts", 0),
            "improvement_rate": stats.get("improvement_rate", 0),
            "degradation_rate": stats.get("degradation_rate", 0),
            "top_5_improved": stats.get("top_improved", [])[:5],
            "top_5_degraded": stats.get("top_degraded", [])[:5]
        }
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    def get_metrics_for_logging(self, step: int) -> Dict[str, float]:
        """
        Get metrics suitable for logging to wandb/tensorboard.
        
        Args:
            step: Current training step
            
        Returns:
            Dictionary of metrics with proper naming for logging
        """
        stats = self.get_improvement_stats(step)
        
        metrics = {
            "prompt_tracking/total_prompts": stats.get("total_prompts", 0),
            "prompt_tracking/improved_prompts": stats.get("improved_prompts", 0),
            "prompt_tracking/degraded_prompts": stats.get("degraded_prompts", 0),
            "prompt_tracking/improvement_rate": stats.get("improvement_rate", 0),
            "prompt_tracking/degradation_rate": stats.get("degradation_rate", 0),
        }
        
        if "avg_improvement" in stats:
            metrics["prompt_tracking/avg_improvement"] = stats["avg_improvement"]
        if "max_improvement" in stats:
            metrics["prompt_tracking/max_improvement"] = stats["max_improvement"]
        if "avg_degradation" in stats:
            metrics["prompt_tracking/avg_degradation"] = stats["avg_degradation"]
        if "max_degradation" in stats:
            metrics["prompt_tracking/max_degradation"] = stats["max_degradation"]
            
        return metrics
    
    def load_from_checkpoint(self, checkpoint_dir: str) -> bool:
        """
        Load tracking data from checkpoint.
        
        Args:
            checkpoint_dir: Directory containing saved tracking data
            
        Returns:
            True if loaded successfully, False otherwise
        """
        baseline_file = os.path.join(checkpoint_dir, "baseline.json")
        
        if not os.path.exists(baseline_file):
            print(f"No baseline file found at {baseline_file}")
            return False
            
        try:
            with open(baseline_file, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
            
            # Reconstruct prompt_data
            for prompt_hash, data in baseline_data["data"].items():
                self.prompt_data[prompt_hash] = {
                    "content": data["content"],
                    "baseline_acc": data["baseline_acc"],
                    "history": [(0, data["baseline_acc"])]  # Start with baseline
                }
            
            self.baseline_recorded = True
            print(f"✓ Loaded tracking data for {len(self.prompt_data)} prompts")
            return True
            
        except Exception as e:
            print(f"Error loading tracking data: {e}")
            return False


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


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False, off_policy_mode=False):
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

    if off_policy_mode and "rollout_log_probs" in data.batch:
        # Off-policy
        kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["rollout_log_probs"], kl_penalty=kl_penalty)
    else:
        # On-policy  
        kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)

    # kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
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


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, 
                     multi_turn=False, norm_adv_by_std_in_grpo=True, config=None, 
                     off_policy_stats=None):
    
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)

    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
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
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
            
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    
    elif adv_estimator == AdvantageEstimator.GRPO_OFF_POLICY:
        off_policy_stats = off_policy_stats or {}
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]

        advantages, returns = core_algos.compute_grpo_off_policy_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            off_policy_stats=off_policy_stats,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == AdvantageEstimator.GRPO_OFF_POLICY_IS:
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]

        advantages, returns = core_algos.compute_grpo_off_policy_importance_sampling_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            current_log_probs=data.batch["old_log_probs"],
            rollout_log_probs=data.batch["rollout_log_probs"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == AdvantageEstimator.GRPO_ADAPTIVE_FILTER:
        filtering_mask = data.batch.get("filtering_mask", torch.ones(len(data), dtype=torch.bool))
        
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
            
        advantages, returns = core_algos.compute_grpo_adaptive_filter_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            filtering_mask=filtering_mask,
            off_policy_stats=off_policy_stats,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs['index'] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            adv_kwargs['reward_baselines'] = data.batch["reward_baselines"]

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
            AdvantageEstimator.GRPO_OFF_POLICY_IS,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        # [wanxu] add offpolicy manager
        self.off_policy_manager = OffPolicyManager(self.config)
        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self._init_prompt_tracker()
        self.off_policy_manager.total_steps = self.total_training_steps

    def _init_prompt_tracker(self):
        """Initialize prompt accuracy tracker"""
        tracker_config = self.config.trainer.get("prompt_tracking", {})
        
        if not tracker_config.get("enable", False):
            self.prompt_tracker = None
            return
            
        save_dir = tracker_config.get("save_dir", "./prompt_tracking_results")
        track_improved = tracker_config.get("track_top_k_improved", 50)
        track_degraded = tracker_config.get("track_top_k_degraded", 20)
        
        self.prompt_tracker = PromptAccuracyTracker(
            save_dir=save_dir,
            track_top_k_improved=track_improved,
            track_top_k_degraded=track_degraded
        )
        
        print("✓ Prompt accuracy tracker initialized")


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
        """Enhanced validation that includes prompt tracking"""
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        
        # For prompt tracking
        prompt_accuracies = []
        prompt_texts = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch (avg@32)
            test_input_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_input_batch.batch["input_ids"]

            if "multi_modal_data" in test_batch.non_tensor_batch:
                input_texts = []
                multi_modal_datas = test_input_batch.non_tensor_batch["multi_modal_data"]
                for idx, (ids, mm_data) in enumerate(zip(input_ids, multi_modal_datas)):
                    
                    text = self.tokenizer.decode(ids, skip_special_tokens=True)

                    mm_hash = self._get_multi_modal_hash(mm_data) if mm_data is not None else "none"

                    composite_prompt = f"{text}|MM_HASH:{mm_hash}"
                    input_texts.append(composite_prompt)
            else:
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

            test_input_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": True,
                "validate": self.config.actor_rollout_ref.rollout.val_kwargs.n == 1,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # print("self.actor_rollout_wg.world_size", self.actor_rollout_wg.world_size)
            
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

            test_batch = test_input_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            # For prompt tracking: calculate per-prompt accuracy (average over n responses per prompt)
            # Use rollout.n (repeat_times in validation) so baseline_acc is avg@n, not 0/1 from val_kwargs.n=1
            if self.prompt_tracker is not None:
                n_responses = self.config.actor_rollout_ref.rollout.n
                prompt_batch_size = len(input_texts) // n_responses
                print("prompt_batch_size", prompt_batch_size)
                for i in range(prompt_batch_size):
                    start_idx = i * n_responses
                    end_idx = (i + 1) * n_responses
                    prompt_text = input_texts[start_idx]  # All responses for this prompt have same input
                    prompt_scores = scores[start_idx:end_idx]
                    prompt_accuracy = np.mean(prompt_scores)  # Average accuracy across responses
                    
                    prompt_texts.append(prompt_text)
                    prompt_accuracies.append(prompt_accuracy)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        # Update prompt tracker
        if self.prompt_tracker is not None:
            if self.global_steps == 0:  # Record baseline
                self.prompt_tracker.record_baseline(prompt_texts, prompt_accuracies, self.global_steps)
            else:  # Update accuracies
                self.prompt_tracker.update_accuracies(prompt_texts, prompt_accuracies, self.global_steps)

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

        print("data_sources", len(data_sources))
        print("sample_inputs", len(sample_inputs))
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

        # Add prompt tracking metrics if available
        if self.prompt_tracker is not None and self.global_steps > 0:
            tracking_metrics = self.prompt_tracker.get_metrics_for_logging(self.global_steps)
            metric_dict.update(tracking_metrics)
            
            # Print improvement summary
            # stats = self.prompt_tracker.get_improvement_stats(self.global_steps)
            # print(f"\n=== Prompt Tracking Summary (Step {self.global_steps}) ===")
            # print(f"Total prompts: {stats.get('total_prompts', 0)}")
            # print(f"Improved: {stats.get('improved_prompts', 0)} ({stats.get('improvement_rate', 0):.2%})")
            # print(f"Degraded: {stats.get('degraded_prompts', 0)} ({stats.get('degradation_rate', 0):.2%})")
            
            # if stats.get('top_improved'):
            #     print(f"\nTop improved prompts:")
            #     for i, item in enumerate(stats['top_improved'][:3], 1):
            #         print(f"  {i}. +{item['improvement']:.3f}: {item['content'][:60]}...")
                    
            # if stats.get('top_degraded'):
            #     print(f"\nTop degraded prompts:")
            #     for i, item in enumerate(stats['top_degraded'][:3], 1):
            #         print(f"  {i}. -{item['degradation']:.3f}: {item['content'][:60]}...")
            # print("=" * 50)

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

        
        # Save prompt tracking data if available
        if self.prompt_tracker is not None:
            tracking_dir = os.path.join(local_global_step_folder, "prompt_tracking") 
            os.makedirs(tracking_dir, exist_ok=True)
            
            # Copy current tracking files to checkpoint
            import shutil
            src_dir = self.prompt_tracker.save_dir
            if os.path.exists(src_dir):
                for filename in os.listdir(src_dir):
                    if filename.endswith('.json'):
                        src_file = os.path.join(src_dir, filename)
                        dst_file = os.path.join(tracking_dir, filename)
                        shutil.copy2(src_file, dst_file)
                print(f"✓ Saved prompt tracking data to {tracking_dir}")


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

        if self.prompt_tracker is not None and global_step_folder is not None:
            tracking_dir = os.path.join(global_step_folder, "prompt_tracking")
            if os.path.exists(tracking_dir):
                success = self.prompt_tracker.load_from_checkpoint(tracking_dir)
                if success:
                    print(f"✓ Loaded prompt tracking data from {tracking_dir}")
                else:
                    print(f"⚠ Failed to load prompt tracking data from {tracking_dir}")

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

    def _get_multi_modal_hash(self, multi_modal_data) -> str:

        if multi_modal_data is None:
            return "none"
        
        try:
            if isinstance(multi_modal_data, dict):
                hash_parts = []
                
                for key in sorted(multi_modal_data.keys()):
                    value = multi_modal_data[key]
                    
                    if isinstance(value, list):
                        list_hash_parts = []
                        for item in value:
                            if hasattr(item, 'mode') and hasattr(item, 'size'):  # PIL image
                                img_info = f"img_{item.mode}_{item.size[0]}x{item.size[1]}"
                                list_hash_parts.append(img_info)
                            else:
                                list_hash_parts.append(f"item_{type(item).__name__}")
                        
                        hash_parts.append(f"{key}:[{','.join(list_hash_parts)}]")
                    elif hasattr(value, 'mode') and hasattr(value, 'size'):  # single PIL image
                        img_info = f"img_{value.mode}_{value.size[0]}x{value.size[1]}"
                        hash_parts.append(f"{key}:{img_info}")
                    else:
                        hash_parts.append(f"{key}:{type(value).__name__}")
                
                content = "|".join(hash_parts)
            else:
                content = f"{type(multi_modal_data).__name__}"
            
            return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
            
        except Exception as e:
            fallback_content = f"{type(multi_modal_data).__name__}_{id(multi_modal_data) % 1000000}"
            return hashlib.md5(fallback_content.encode('utf-8')).hexdigest()[:16]


    def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Compute IS weights and apply rejection sampling for rollout-training mismatch.

        Computes importance sampling weights to correct for distribution mismatch between
        rollout and training policies. Applies rejection sampling (mask mode/veto) by
        modifying response_mask. Always updates response_mask; conditionally adds IS weights.

        Key behavior:
        - response_mask: ALWAYS updated with rejection (mask mode + veto excluded from training)
        - rollout_is_weights: Added to batch ONLY if config.algorithm.rollout_is=True

        This separation ensures:
        - Rejection works even when IS weights are disabled (rollout_is=False)
        - Metrics can be monitored before enabling IS weight application

        Args:
            batch: DataProto with old_log_probs, rollout_log_probs, response_mask

        Returns:
            Tuple of (updated_batch, metrics):
                updated_batch: Batch with modified response_mask (always) and rollout_is_weights (if rollout_is=True)
                metrics: Dict of IS and mismatch metrics, all with "mismatch/" prefix
        """
        # Compute rollout IS weights if enabled and data is available
        # rollout_is_threshold is the main on/off switch (None = disabled, float = enabled)
        rollout_is_threshold = self.config.algorithm.get("rollout_is_threshold", 3)
        if rollout_is_threshold is not None and rollout_is_threshold > 0 and "rollout_log_probs" in batch.batch:
            # Compute IS weights and get modified response_mask
            rollout_is_weights, modified_response_mask, rollout_is_metrics = compute_rollout_importance_weights(
                old_log_prob=batch.batch["old_log_probs"],
                rollout_log_prob=batch.batch["rollout_log_probs"],
                response_mask=batch.batch["response_mask"],
                rollout_is_level=self.config.algorithm.rollout_is_level,
                rollout_is_mode=self.config.algorithm.rollout_is_mode,
                rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
                rollout_is_threshold_lower=self.config.algorithm.get("rollout_is_threshold_lower", None),
                rollout_is_veto_threshold=self.config.algorithm.get("rollout_is_veto_threshold", None),
            )

            # ALWAYS update response_mask with rejection (even if rollout_is=False)
            # - Mask mode: tokens with outlier IS ratios excluded
            # - Veto: sequences with catastrophic tokens excluded
            # This ensures correct loss normalization (rejected samples not in denominator)
            batch.batch["response_mask"] = modified_response_mask

            # Conditionally add IS weights based on rollout_is config flag
            # - rollout_is=True: Enable IS weight correction in policy loss
            # - rollout_is=False: Metrics-only mode (rejection still applied via mask)
            apply_weights = self.config.algorithm.get("rollout_is", True)

            if apply_weights:
                # Add IS weights (safety-bounded, mode-processed) to enable weight correction
                batch = batch.union(rollout_is_weights)

            return batch, rollout_is_metrics

        # Return unchanged batch and empty metrics if IS is disabled
        return batch, {}


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


                if "multi_modal_data" in batch.non_tensor_batch:
                    # support VLM, we need add several necessary keys
                    gen_tensor_keys = ["input_ids", "attention_mask", "position_ids"]
                    for key in gen_tensor_keys:
                        if key in batch.batch:
                            batch.batch[f"gen_{key}"] = batch.batch[key]

                    gen_non_tensor_keys = ["raw_prompt_ids", "tools_kwargs", "multi_modal_data", "multi_modal_inputs"]
                    for key in gen_non_tensor_keys:
                        if key in batch.non_tensor_batch:
                            batch.non_tensor_batch[f"gen_{key}"] = batch.non_tensor_batch[key]

                # pop() means that batch will remove these keys and the poped keys will save as gen_batch
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps
                

                with _timer("step", timing_raw):

                    # Step 1: Generate batch using current/alpha policy
                    with _timer("gen", timing_raw):
                        if self.off_policy_manager.enable_off_policy_rollout:
                            gen_batch_output, is_alpha_sampling = self.off_policy_manager.sample_with_alpha_policy(
                                gen_batch, 
                                self.actor_rollout_wg,
                                self.global_steps
                            )
                        else:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            is_alpha_sampling = True
                        
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
                    
                    # TODO: This encode mode make uid unique for each training steps, need to modify
                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    
                    # union() means combine gen_batch infos and output infos into batch
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


                    # Step 2: Calculate batch rewards
                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor


                    # Step 3: Save all data's rewards into buffer (different from [prompts, responses] key we only choose important data)
                    # We want to get all historical rewards for tracking the mean score

                    scores = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
                    uids = batch.non_tensor_batch["uid"]

                    # Calculate average score per uid group
                    uid_to_scores = defaultdict(list)
                    for i, uid in enumerate(uids):
                        uid_str = str(uid)
                        score = scores[i]
                        uid_to_scores[uid_str].append(score)

                    # Add group mean score for each uid
                    for uid_str, score_list in uid_to_scores.items():
                        # buffer have two main parts: the one is the stats buffer (rewards), the another one is the detailed data (prompts + responses + other keys)
                        # For each uid (key), the stat is the group scores from alpha/current policy generation.
                        self.off_policy_manager.buffer.add_score(uid_str, score_list, is_alpha_sampling, target_range=self.off_policy_manager.target_accuracy_range, buffer_range=self.off_policy_manager.buffer_accuracy_range)

                    # Step 4: Re-evaluate bad cases using current policy
                    promoted_batch = None
                    if self.off_policy_manager.enable_off_policy_samples and self.off_policy_manager.enable_off_policy_reeval:
                        
                        if self.global_steps % self.off_policy_manager.reeval_freq == 0: 
                            promoted_batch = self.off_policy_manager.reevaluate_and_promote_buffer_samples(
                                self.actor_rollout_wg, self.reward_fn, self.tokenizer
                            )
                            if promoted_batch is not None:
                                promoted_batch.batch["response_mask"] = compute_response_mask(promoted_batch)
                                
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
                            if self.off_policy_manager.enable_off_policy_samples:
                                batch.batch["ref_log_prob"] = batch.batch["old_log_probs"]

                            else:
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

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty, off_policy_mode=self.off_policy_manager.enable_off_policy_samples)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        sequence_reward = batch.batch["token_level_rewards"].sum(-1)
                        metrics.update({"critic/origin_rewards/mean": torch.mean(sequence_reward).detach().item()})
                        
                        # Step 5: Add data into off policy buffer (r \in [0, c1] or r \in [c2, c3])
                        self.off_policy_manager.collect_data(batch, is_alpha_sampling, step=self.global_steps)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        # Step 6: Data filter and merge
                        final_batch = batch
                        
                        if self.off_policy_manager.enable_off_policy_samples:

                            batches_to_combine = []
                            combined_uids = set()  
                            batch_sources = []  

                            initial_batch_size = len(batch)
                            
                            # 6.1. Filter current fresh samples into train batch (gaussian/range/uniform)
                            current_batch_filtered = self.off_policy_manager.filter_current_batch_samples(batch)
                            
                            if len(current_batch_filtered) > 0:
                                batches_to_combine.append(current_batch_filtered)
                                batch_sources.append("current") 
                                for uid in current_batch_filtered.non_tensor_batch["uid"]:
                                    combined_uids.add(str(uid))
                                    
                                print(f"✓ Using {len(current_batch_filtered)} samples from current batch")

                            # 6.2. Filter buffer samples to reach initial_batch_size
                            if self.off_policy_manager.enable_off_policy_reeval:
                                current_sample_count = len(current_batch_filtered) if len(current_batch_filtered) > 0 else 0
                                remaining_needed = initial_batch_size - current_sample_count
                                
                                print(f"Target batch size: {initial_batch_size}, Current: {current_sample_count}, Need: {remaining_needed}")
                                
                                if remaining_needed > 0:
                                    buffer_samples_added = 0
                                    
                                    # 6.2.1. Prioritized sampling from promoted batch
                                    if promoted_batch is not None:
                                        promoted_uids = [str(uid) for uid in promoted_batch.non_tensor_batch["uid"]]
                                        non_duplicate_mask = torch.tensor([uid not in combined_uids for uid in promoted_uids])
                                        
                                        if torch.sum(non_duplicate_mask) > 0:
                                            available_promoted = torch.sum(non_duplicate_mask).item()
                                            # When sampling promoted batch
                                            is_vlm_model = "multi_modal_inputs" in promoted_batch.non_tensor_batch
                                            if is_vlm_model:
                                                def make_divisible(size, divisor):
                                                    return (size // divisor) * divisor
                                                ppo_micro_batch_size = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                                                world_size = self.actor_rollout_wg.world_size
                                                min_batch_unit = ppo_micro_batch_size * world_size

                                                promoted_to_take = min(available_promoted, remaining_needed)
                                                promoted_to_take = make_divisible(promoted_to_take, min_batch_unit)
                      
                                                if promoted_to_take > 0:  # Only proceed if we have a valid divisible size
                                                    if available_promoted > promoted_to_take:
                                                        valid_indices = torch.where(non_duplicate_mask)[0]
                                                        perm = torch.randperm(len(valid_indices))
                                                        selected_indices = valid_indices[perm[:promoted_to_take]]
                                                        promoted_batch_filtered = promoted_batch.select_idxs(selected_indices)
                                                        print(f"✓ Randomly sampled {promoted_to_take} from {available_promoted} promoted samples")
                                                    else:
                                                        # Need to ensure even the "all available" case is divisible
                                                        available_divisible = make_divisible(available_promoted, min_batch_unit)
                                                        if available_divisible > 0:
                                                            valid_indices = torch.where(non_duplicate_mask)[0]
                                                            perm = torch.randperm(len(valid_indices))
                                                            selected_indices = valid_indices[perm[:available_divisible]]
                                                            promoted_batch_filtered = promoted_batch.select_idxs(selected_indices)
                                                            print(f"✓ Using {available_divisible} divisible promoted samples from {available_promoted} available")
                                                        else:
                                                            promoted_batch_filtered = None
                                                            print(f"✓ Skipping promoted samples: {available_promoted} samples not divisible by {min_batch_unit}")
                                                    buffer_samples_added = len(promoted_batch_filtered) 

                                            else:
                                                promoted_to_take = min(available_promoted, remaining_needed)
                                            
                                                if available_promoted > promoted_to_take:
                
                                                    valid_indices = torch.where(non_duplicate_mask)[0]
                                                    perm = torch.randperm(len(valid_indices))
                                                    selected_indices = valid_indices[perm[:promoted_to_take]]
                                                    promoted_batch_filtered = promoted_batch.select_idxs(selected_indices)
                                                    print(f"✓ Randomly sampled {promoted_to_take} from {available_promoted} promoted samples")
                                                else:
                                
                                                    promoted_batch_filtered = promoted_batch.select_idxs(non_duplicate_mask)
                                                    print(f"✓ Using all {available_promoted} promoted samples")
                                            
                                                buffer_samples_added = len(promoted_batch_filtered)
                                            
                                            if buffer_samples_added > 0:
                                                if "old_log_probs" not in promoted_batch_filtered.batch:
                                                    old_log_prob_promoted = self.actor_rollout_wg.compute_log_prob(promoted_batch_filtered)
                                                    old_log_prob_promoted.batch.pop("entropys")
                                                    promoted_batch_filtered = promoted_batch_filtered.union(old_log_prob_promoted)
                                                else:
                                                    print("we reuse the old log probs from the buffer.")
                                                
                                                if self.use_reference_policy and "ref_log_prob" not in promoted_batch_filtered.batch:
                                                    if self.off_policy_manager.enable_off_policy_samples:
                                                        promoted_batch_filtered.batch["ref_log_prob"] = promoted_batch_filtered.batch["old_log_probs"]
                                                    else:
                                                        if not self.ref_in_actor:
                                                            ref_log_prob_promoted = self.ref_policy_wg.compute_ref_log_prob(promoted_batch_filtered)
                                                        else:
                                                            ref_log_prob_promoted = self.actor_rollout_wg.compute_ref_log_prob(promoted_batch_filtered)
                                                        promoted_batch_filtered = promoted_batch_filtered.union(ref_log_prob_promoted)
                                                
                                                if self.use_critic and "values" not in promoted_batch_filtered.batch:
                                                    values_promoted = self.critic_wg.compute_values(promoted_batch_filtered)
                                                    promoted_batch_filtered = promoted_batch_filtered.union(values_promoted)
                                                
                                                # Apply rewards
                                                if self.config.algorithm.use_kl_in_reward:
                                                    promoted_batch_filtered, _ = apply_kl_penalty(
                                                        promoted_batch_filtered, 
                                                        kl_ctrl=self.kl_ctrl_in_reward, 
                                                        kl_penalty=self.config.algorithm.kl_penalty,
                                                        off_policy_mode=self.off_policy_manager.enable_off_policy_samples
                                                    )
                                                else:
                                                    promoted_batch_filtered.batch["token_level_rewards"] = promoted_batch_filtered.batch["token_level_scores"]

                                                if "response_mask" not in promoted_batch_filtered.batch.keys():
                                                    promoted_batch_filtered.batch["response_mask"] = compute_response_mask(promoted_batch_filtered)

                                                self.off_policy_manager.collect_data(promoted_batch_filtered, True, step=self.global_steps)

                                                batches_to_combine.append(promoted_batch_filtered)
                                                batch_sources.append("promoted")  

                                                for uid in promoted_batch_filtered.non_tensor_batch["uid"]:
                                                    combined_uids.add(str(uid))
                                    
                                    # 6.2.2. Sampling high-quality buffer samples to reach initial_batch_size
                                    still_needed = remaining_needed - buffer_samples_added
                                    
                                    if still_needed > 0 and self.off_policy_manager.enable_adaptive_target_range:
                                        adaptive_range_metrics = self.off_policy_manager._update_adaptive_target_range()
                                        metrics.update(adaptive_range_metrics)


                                    if still_needed > 0 and self.off_policy_manager.enable_completion:
                                        print(f"Still need {still_needed} more samples to reach target batch size")
                                        
                                        # buffer target range [c2, c3]
                                        available_buffer_samples = self.off_policy_manager.get_available_groups_count()
                                        if available_buffer_samples > 0:
                                            
                                            buffer_batch = self.off_policy_manager.sample_filtered_groups(
                                                min(still_needed, available_buffer_samples)
                                            )
                                            
                                            if buffer_batch is not None:
                                                buffer_uids = [str(uid) for uid in buffer_batch.non_tensor_batch["uid"]]
                                                non_duplicate_mask = torch.tensor([uid not in combined_uids for uid in buffer_uids])
                                                
                                                if torch.sum(non_duplicate_mask) > 0:
                                                    buffer_batch_non_dup = buffer_batch.select_idxs(non_duplicate_mask)
                                                    available_buffer = len(buffer_batch_non_dup)
                                                    
                                                    
                                                    if available_buffer > still_needed:
                                                        perm = torch.randperm(available_buffer)
                                                        selected_indices = perm[:still_needed]
                                                        buffer_batch_filtered = buffer_batch_non_dup.select_idxs(selected_indices)
                                                        print(f"✓ Randomly sampled {still_needed} from {available_buffer} buffer samples ({self.off_policy_manager.target_accuracy_range[0]}-{self.off_policy_manager.target_accuracy_range[1]})")
                                                    else:
                                                        buffer_batch_filtered = buffer_batch_non_dup
                                                        print(f"✓ Using all {available_buffer} available buffer samples ({self.off_policy_manager.target_accuracy_range[0]}-{self.off_policy_manager.target_accuracy_range[1]})")
                                                    

                                                    if "old_log_probs" not in buffer_batch_filtered.batch:
                                                        old_log_prob_buffer = self.actor_rollout_wg.compute_log_prob(buffer_batch_filtered)
                                                        old_log_prob_buffer.batch.pop("entropys")
                                                        buffer_batch_filtered = buffer_batch_filtered.union(old_log_prob_buffer)
                                                    else:
                                                        print("we reuse the old log probs from the buffer.")

                                                    if self.use_reference_policy and "ref_log_prob" not in buffer_batch_filtered.batch:
                                                        if self.off_policy_manager.enable_off_policy_samples:
                                                            buffer_batch_filtered.batch["ref_log_prob"] = buffer_batch_filtered.batch["old_log_probs"]  
                                                        else:
                                                            if not self.ref_in_actor:
                                                                ref_log_prob_buffer = self.ref_policy_wg.compute_ref_log_prob(buffer_batch_filtered)
                                                            else:
                                                                ref_log_prob_buffer = self.actor_rollout_wg.compute_ref_log_prob(buffer_batch_filtered)
                                                            buffer_batch_filtered = buffer_batch_filtered.union(ref_log_prob_buffer)
                                                    
                                                    if self.use_critic and "values" not in buffer_batch_filtered.batch:
                                                        values_buffer = self.critic_wg.compute_values(buffer_batch_filtered)
                                                        buffer_batch_filtered = buffer_batch_filtered.union(values_buffer)
                                                    
                                                    if self.config.algorithm.use_kl_in_reward:
                                                        buffer_batch_filtered, _ = apply_kl_penalty(
                                                            buffer_batch_filtered, 
                                                            kl_ctrl=self.kl_ctrl_in_reward, 
                                                            kl_penalty=self.config.algorithm.kl_penalty,
                                                            off_policy_mode=self.off_policy_manager.enable_off_policy_samples
                                                        )
                                                    else:
                                                        buffer_batch_filtered.batch["token_level_rewards"] = buffer_batch_filtered.batch["token_level_scores"]

                                                    if "response_mask" not in buffer_batch_filtered.batch.keys():
                                                        buffer_batch_filtered.batch["response_mask"] = compute_response_mask(buffer_batch_filtered)

                                                    batches_to_combine.append(buffer_batch_filtered)
                                                    batch_sources.append("target") 

                                                    print(f"✓ Added {len(buffer_batch_filtered)} buffer samples to reach target batch size")
                                                else:
                                                    print("✓ No non-duplicate buffer samples available")
                                            else:
                                                print("✓ No suitable buffer samples found")
                                        else:
                                            print(f"✓ No buffer samples available to reach target size")
                                            print(f"Warning: Final batch size will be {current_sample_count + buffer_samples_added} instead of {initial_batch_size}")
                                    elif self.off_policy_manager.enable_completion:
                                        print(f"✓ Target batch size reached with promoted samples")
                                    else:
                                        print(f"✓ Current batch size is {current_sample_count}")
                                else:
                                    print(f"✓ Current batch already meets target size")
                               
                            # 6.3. Combine Batch
                            if len(batches_to_combine) == 0:
                                final_batch = batch.select_idxs(torch.tensor([], dtype=torch.long))
                                final_batch.batch["batch_sources"] = torch.tensor([], dtype=torch.long)
                            elif len(batches_to_combine) == 1:
                                final_batch = batches_to_combine[0]
                                
                                final_batch = self._clean_batch_for_training(final_batch)
                                
                                # 0=current, 1=promoted, 2=target
                                source_map = {"current": 0, "promoted": 1, "target": 2}
                                final_batch.batch["batch_sources"] = torch.full((len(final_batch),), source_map[batch_sources[0]], dtype=torch.long)
                            else:
                                source_tensors = []
                                source_map = {"current": 0, "promoted": 1, "target": 2}
                                
                                cleaned_batches = []
                                for i, batch_part in enumerate(batches_to_combine):
                                    cleaned_batch = self._clean_batch_for_training(batch_part)
                                    cleaned_batches.append(cleaned_batch)
                                    
                                    source_id = source_map[batch_sources[i]]
                                    source_tensor = torch.full((len(cleaned_batch),), source_id, dtype=torch.long)
                                    source_tensors.append(source_tensor)
                                
                                batches_to_combine = cleaned_batches

                                final_batch = DataProto.concat(batches_to_combine)
                                final_batch.batch["batch_sources"] = torch.cat(source_tensors, dim=0)
                            
                            final_batch.batch["filtering_mask"] = torch.ones(len(final_batch), dtype=torch.bool)        
                            

                            is_vlm_model = "multi_modal_inputs" in final_batch.non_tensor_batch
                            if is_vlm_model:
                                # ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                                # min_batch_unit = self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * self.actor_rollout_wg.world_size
                                min_batch_unit = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.actor_rollout_wg.world_size
                                current_batch_size = len(final_batch)
                                
    
                                target_batch_size = (current_batch_size // min_batch_unit) * min_batch_unit
                                
                                if target_batch_size != current_batch_size and target_batch_size > 0:
                                    print(f"VLM mode: Adjusting batch size from {current_batch_size} to {target_batch_size} to be divisible by min_batch_unit={min_batch_unit}")
                                    indices = torch.randperm(current_batch_size)[:target_batch_size]
                                    final_batch = final_batch.select_idxs(indices)
                                    
                                elif target_batch_size == 0:
                                    print(f"Warning: VLM mode batch size {current_batch_size} < min_batch_unit {min_batch_unit}, using original batch")
                                else:
                                    print(f"VLM mode: Batch size {current_batch_size} is already divisible by min_batch_unit={min_batch_unit}")
                            
                            final_size = len(final_batch)
         
                            unique_sources, counts = torch.unique(final_batch.batch["batch_sources"], return_counts=True)

                            print(f"Final batch composition:")
                            print(f"  Target size: {initial_batch_size}")
                            print(f"  Actual size: {final_size}")
                            print(f"  Achievement rate: {final_size/initial_batch_size:.2%}")
                            print(f"  Sources combined: {len(batches_to_combine)}")
                        
                        else:
                            final_batch.batch["filtering_mask"] = torch.ones(len(final_batch), dtype=torch.bool)
                            final_batch.batch["batch_sources"] = torch.zeros(len(final_batch), dtype=torch.long)  

                        # Get off policy batch stats from off policy buffer
                        off_policy_stats = self.off_policy_manager.get_off_policy_stats()
                        
                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        final_batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(final_batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)
                        print("is_metrics", is_metrics)
                        final_batch.batch["clip_center"] = torch.ones_like(final_batch.batch["old_log_probs"])
                        batch = compute_advantage(
                            final_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=True,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                            off_policy_stats=off_policy_stats, # with off policy stats and data filter 
                        )

                    if "multi_modal_inputs" in final_batch.non_tensor_batch:
                        # for vlm model, we should make sure the filtered batch size is divisible by ppo_mini_batch_size
                        # if the batch size is smaller than ppo_mini_batch_size, we choose not train
                        if len(final_batch) < self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n:
                            is_train = False
                        else:
                            is_train = True
                    else:
                        is_train = True
                    # update critic
                    if is_train and self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if is_train and self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

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
                })

                # off policy metrics
                off_policy_metrics = self.off_policy_manager.get_metrics()
                metrics.update(off_policy_metrics)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.off_policy_manager.step_count += 1
  
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def _clean_batch_for_training(self, batch: DataProto) -> DataProto:

        required_tensor_keys = {
            "input_ids", "attention_mask", "position_ids", "prompts", "responses", 
            "token_level_scores", "token_level_rewards", "old_log_probs", 
            "response_mask", "rollout_log_probs", "clip_center",
        }
        
        required_non_tensor_keys = {
            "uid", "data_source", "ability", "reward_model", "extra_info",
            "tools_kwargs", "raw_prompt_ids", "raw_prompt", "multi_modal_inputs"
        }
        
        if self.use_reference_policy:
            required_tensor_keys.add("ref_log_prob")
        
        if self.use_critic:
            required_tensor_keys.add("values")
        
        cleaned_tensor_batch = {}
        cleaned_non_tensor_batch = {}
        
        for key in required_tensor_keys:
            if key in batch.batch:
                cleaned_tensor_batch[key] = batch.batch[key]
        
        for key in required_non_tensor_keys:
            if key in batch.non_tensor_batch:
                cleaned_non_tensor_batch[key] = batch.non_tensor_batch[key]
        
        cleaned_batch = DataProto.from_dict(
            tensors=cleaned_tensor_batch,
            non_tensors=cleaned_non_tensor_batch
        )
        
        if hasattr(batch, 'meta_info'):
            cleaned_batch.meta_info = batch.meta_info.copy()
        
        return cleaned_batch

    def __del__(self):
        if hasattr(self, '_alpha_temp_dir') and os.path.exists(self._alpha_temp_dir):
            try:
                for filename in os.listdir(self._alpha_temp_dir):
                    file_path = os.path.join(self._alpha_temp_dir, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                print(f"Cleaned up contents of alpha policy temp dir: {self._alpha_temp_dir}")
            except Exception as e:
                print(f"Failed to clean up alpha policy temp dir: {e}")

