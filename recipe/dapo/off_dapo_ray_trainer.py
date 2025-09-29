# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
DAPO PPO Trainer with Off-Policy Support using Ray-based single controller.
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator, 
    RayPPOTrainer, 
    _timer, 
    apply_kl_penalty, 
    compute_advantage, 
    compute_response_mask,
    OffPolicyManager,
    TemporaryCurrentPolicyContext
)


class RayOffDAPOTrainer(RayPPOTrainer):
    """
    DAPO trainer with off-policy support.
    Combines DAPO's filter_groups logic with off-policy sampling strategies.
    Optimized with immediate old_log_prob computation after generation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize off-policy manager from parent class
        # self.off_policy_manager is already initialized in RayPPOTrainer.__init__

    def _apply_dapo_filter_groups(self, batch: DataProto) -> tuple[DataProto, float]:
        """
        Apply DAPO's filter_groups logic to the batch.
        Returns filtered batch and the ratio of kept data.
        """
        if not self.config.algorithm.filter_groups.enable:
            return batch, 1.0
        
        metric_name = self.config.algorithm.filter_groups.metric
        if metric_name == "seq_final_reward":
            # Turn to numpy for easier filtering
            batch.non_tensor_batch["seq_final_reward"] = batch.batch["token_level_rewards"].sum(dim=-1).numpy()
        elif metric_name == "seq_reward":
            batch.non_tensor_batch["seq_reward"] = batch.batch["token_level_scores"].sum(dim=-1).numpy()

        # Collect the sequence reward for each trajectory
        prompt_uid2metric_vals = defaultdict(list)
        for uid, metric_val in zip(batch.non_tensor_batch["uid"], batch.non_tensor_batch[metric_name]):
            prompt_uid2metric_vals[uid].append(metric_val)

        prompt_uid2metric_std = {}
        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
        
        # Calculate ratio of kept data
        original_prompts = len(prompt_uid2metric_vals)
        kept_prompts = len(kept_prompt_uids)
        keep_ratio = kept_prompts / original_prompts if original_prompts > 0 else 0.0

        kept_traj_idxs = []
        for idx, traj_from_prompt_uid in enumerate(batch.non_tensor_batch["uid"]):
            if traj_from_prompt_uid in kept_prompt_uids:
                kept_traj_idxs.append(idx)

        filtered_batch = batch.select_idxs(torch.tensor(kept_traj_idxs)) if kept_traj_idxs else batch.select_idxs(torch.tensor([], dtype=torch.long))
        
        print(f"DAPO filter: kept {kept_prompts}/{original_prompts} prompts ({keep_ratio:.2%})")
        return filtered_batch, keep_ratio

    def fit(self):
        """
        The training loop combining DAPO and off-policy strategies.
        Optimized with immediate old_log_prob computation after generation.
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

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        
        # Add variables to collect data for buffer addition after training
        collected_batches_for_buffer = []
        collected_is_off_policy_flags = []

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")

                # Support VLM by adding gen_ prefixes
                if "multi_modal_data" in new_batch.non_tensor_batch:
                    gen_tensor_keys = ["input_ids", "attention_mask", "position_ids"]
                    for key in gen_tensor_keys:
                        if key in new_batch.batch:
                            new_batch.batch[f"gen_{key}"] = new_batch.batch[key]

                    gen_non_tensor_keys = ["raw_prompt_ids", "tools_kwargs", "multi_modal_data", "multi_modal_inputs"]
                    for key in gen_non_tensor_keys:
                        if key in new_batch.non_tensor_batch:
                            new_batch.non_tensor_batch[f"gen_{key}"] = new_batch.non_tensor_batch[key]

                gen_batch = new_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # Step 1: Generate batch using current/alpha policy (off-policy logic)
                    with _timer("gen", timing_raw):
                        if self.off_policy_manager.enable_off_policy_rollout:
                            gen_batch_output, is_off_policy_rollout = self.off_policy_manager.sample_with_alpha_policy(
                                gen_batch, 
                                self.actor_rollout_wg,
                                self.global_steps
                            )
                        else:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            is_off_policy_rollout = True

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    # OPTIMIZATION: Immediately compute old_log_prob after generation (before any filtering)
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(new_batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = compute_response_mask(new_batch)
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        new_batch = new_batch.union(old_log_prob)

                    # Step 2: Calculate batch rewards
                    with _timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, 
                                kl_ctrl=self.kl_ctrl_in_reward, 
                                kl_penalty=self.config.algorithm.kl_penalty,
                                off_policy_mode=self.off_policy_manager.enable_off_policy_samples
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                        sequence_reward = new_batch.batch["token_level_rewards"].sum(-1)
                        metrics.update({"critic/origin_rewards/mean": torch.mean(sequence_reward).detach().item()})
                    
                    # Step 3: Collect data for buffer addition (but don't add yet)
                    # new_batch now already has old_log_probs computed
                    collected_batches_for_buffer.append(deepcopy(new_batch))
                    collected_is_off_policy_flags.append(is_off_policy_rollout)

                    # Step 4: Apply DAPO filter_groups logic
                    new_batch_filtered, dapo_keep_ratio = self._apply_dapo_filter_groups(new_batch)

                    # DAPO accumulation logic - continue generating if not enough data
                    if self.config.algorithm.filter_groups.enable:
                        kept_prompt_uids = set()
                        for uid in new_batch_filtered.non_tensor_batch["uid"]:
                            kept_prompt_uids.add(uid)
                        
                        num_prompt_in_batch += len(kept_prompt_uids)
                        batch = new_batch_filtered if batch is None else DataProto.concat([batch, new_batch_filtered])

                        prompt_bsz = self.config.data.train_batch_size

                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                continue
                            else:
                                dapo_keep_ratio = num_prompt_in_batch / prompt_bsz
                        else:
                            # Align the batch
                            traj_bsz = int(self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n)
                            dapo_keep_ratio = 1.0
                            if len(batch) > traj_bsz:
                                batch = batch[:traj_bsz]
                            else:
                                batch = batch
                    else:
                        batch = new_batch_filtered

                    # Step 5: Re-evaluate bad cases using current policy
                    promoted_batch = None
                    if self.off_policy_manager.enable_off_policy_samples and self.off_policy_manager.enable_off_policy_reeval:
                        if self.global_steps % self.off_policy_manager.reeval_freq == 0:
                            promoted_batch = self.off_policy_manager.reevaluate_and_promote_buffer_samples(
                                self.actor_rollout_wg, self.reward_fn, self.tokenizer
                            )
                            if promoted_batch is not None:
                                if self.config.algorithm.use_kl_in_reward:
                                    promoted_batch, kl_metrics = apply_kl_penalty(
                                        promoted_batch, 
                                        kl_ctrl=self.kl_ctrl_in_reward, 
                                        kl_penalty=self.config.algorithm.kl_penalty,
                                        off_policy_mode=self.off_policy_manager.enable_off_policy_samples
                                    )
                                else:
                                    promoted_batch.batch["token_level_rewards"] = promoted_batch.batch["token_level_scores"]
                                promoted_batch.non_tensor_batch["seq_final_reward"] = promoted_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                                # For promoted batch, need to recompute old_log_prob since actions are newly generated
                                if promoted_batch is not None:
                                    print("Recomputing old_log_probs for promoted batch (newly generated actions)...")
                                    with _timer("promoted_old_log_prob", timing_raw):
                                        promoted_old_log_prob = self.actor_rollout_wg.compute_log_prob(promoted_batch)
                                        promoted_old_log_prob.batch.pop("entropys", None)
                                        promoted_batch.batch["old_log_probs"] = promoted_old_log_prob.batch["old_log_probs"]

                    # === Updating Phase ===

                    with _timer("adv", timing_raw):
                        # Step 6: Off-policy data merging and filtering
                        final_batch = batch
                    
                        if self.off_policy_manager.enable_off_policy_samples:
                            batches_to_combine = []
                            combined_uids = set()
                            batch_sources = []
                        
                            # 6.1. Using DAPO filtering data as the initial batch
                            current_batch_filtered = batch
                            print(f"DAPO kept {dapo_keep_ratio:.2%} data in the initial batch")
                            
                            if len(current_batch_filtered) > 0:
                                batches_to_combine.append(current_batch_filtered)
                                batch_sources.append("current")
                                for uid in current_batch_filtered.non_tensor_batch["uid"]:
                                    combined_uids.add(str(uid))
                                print(f"Source-1: Using {len(current_batch_filtered)} samples from current batch")

                            
                            if self.off_policy_manager.enable_off_policy_samples:
                                current_sample_count = len(current_batch_filtered) if len(current_batch_filtered) > 0 else 0
                                remaining_needed = prompt_bsz * self.config.actor_rollout_ref.rollout.n - current_sample_count
                                print(f"Need {remaining_needed} more samples to reach target batch size {prompt_bsz * self.config.actor_rollout_ref.rollout.n}")
                                # 6.2. Add promoted batch
                                if remaining_needed > 0 and promoted_batch is not None:
                                    promoted_uids = [str(uid) for uid in promoted_batch.non_tensor_batch["uid"]]
                                    non_duplicate_mask = torch.tensor([uid not in combined_uids for uid in promoted_uids])
                                    
                                    if torch.sum(non_duplicate_mask) > 0:
                                        available_promoted = torch.sum(non_duplicate_mask).item()
                                        promoted_to_take = min(available_promoted, remaining_needed)
                                        
                                        if available_promoted > promoted_to_take:
                                            valid_indices = torch.where(non_duplicate_mask)[0]
                                            perm = torch.randperm(len(valid_indices))
                                            selected_indices = valid_indices[perm[:promoted_to_take]]
                                            promoted_batch_filtered = promoted_batch.select_idxs(selected_indices)
                                        else:
                                            promoted_batch_filtered = promoted_batch.select_idxs(non_duplicate_mask)
                                        
                                        print(f"Source-2: Using {len(promoted_batch_filtered)} samples from promoted batch")
                                        batches_to_combine.append(promoted_batch_filtered)
                                        batch_sources.append("promoted")
                                        
                                        for uid in promoted_batch_filtered.non_tensor_batch["uid"]:
                                            combined_uids.add(str(uid))

                                
                                still_needed = remaining_needed - len(promoted_batch_filtered) if promoted_batch is not None else remaining_needed
                                # 6.3. Add target buffer samples
                                if still_needed > 0 and self.off_policy_manager.enable_completion:
                                    adaptive_range_metrics = self.off_policy_manager._update_adaptive_target_range()
                                    metrics.update(adaptive_range_metrics)
                                    print(f"Still need {still_needed} more samples to reach target batch size")
         
                                    available_groups = self.off_policy_manager.get_available_groups_count()
                                    if available_groups > 0:
                                        # Convert still_needed (samples) to groups needed
                                        rollout_n = self.config.actor_rollout_ref.rollout.n
                                        groups_needed = still_needed // rollout_n  # Ceiling division
                                        
                                        buffer_batch = self.off_policy_manager.sample_filtered_groups(
                                            min(groups_needed, available_groups)  # Correct: passing number of groups
                                        )
                                        print(f"Available groups: {available_groups}, groups needed: {groups_needed}, groups sampled: {len(buffer_batch)}")
                                        if buffer_batch is not None:
                                            buffer_uids = [str(uid) for uid in buffer_batch.non_tensor_batch["uid"]]
                                            non_duplicate_mask = torch.tensor([uid not in combined_uids for uid in buffer_uids])
                                            
                                            if torch.sum(non_duplicate_mask) > 0:
                                                buffer_batch_filtered = buffer_batch.select_idxs(non_duplicate_mask)
                                                metric_name = self.config.algorithm.filter_groups.metric
                                                if metric_name == "seq_final_reward":
                                                    buffer_batch_filtered.non_tensor_batch["seq_final_reward"] = buffer_batch_filtered.batch["token_level_rewards"].sum(dim=-1).numpy()
                                                elif metric_name == "seq_reward":
                                                    buffer_batch_filtered.non_tensor_batch["seq_reward"] = buffer_batch_filtered.batch["token_level_scores"].sum(dim=-1).numpy()
                                                print(f"Source-3: Using {len(buffer_batch_filtered)} samples from historical batch")
                                                batches_to_combine.append(buffer_batch_filtered)
                                                batch_sources.append("target")

                            # 6.4. Combine batches
                            if len(batches_to_combine) == 0:
                                final_batch = batch.select_idxs(torch.tensor([], dtype=torch.long))
                                final_batch.batch["batch_sources"] = torch.tensor([], dtype=torch.long)
                            elif len(batches_to_combine) == 1:
                                final_batch = batches_to_combine[0]
                                final_batch = self._clean_batch_for_training(final_batch)
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
                                    source_tensor = torch.full((len(batch_part),), source_id, dtype=torch.long)
                                    source_tensors.append(source_tensor)

                                batches_to_combine = cleaned_batches

                                final_batch = DataProto.concat(batches_to_combine)
                                final_batch.batch["batch_sources"] = torch.cat(source_tensors, dim=0)
                            
                            final_batch.batch["filtering_mask"] = torch.ones(len(final_batch), dtype=torch.bool)
                            
                            print(f"Final batch composition: {len(final_batch)} samples from {len(batches_to_combine)} sources")
                        
                        else:
                            final_batch.batch["filtering_mask"] = torch.ones(len(final_batch), dtype=torch.bool)
                            final_batch.batch["batch_sources"] = torch.zeros(len(final_batch), dtype=torch.long)

                        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

                        # Balance the number of valid tokens across DP ranks
                        if self.config.trainer.balance_batch:
                            self._balance_batch(final_batch, metrics=metrics)

                        # compute global_valid tokens
                        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

                        # old_log_probs are already computed for all data, no need to recompute

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer("ref", timing_raw):
                                if self.off_policy_manager.enable_off_policy_samples:
                                    final_batch.batch["ref_log_prob"] = final_batch.batch["old_log_probs"]
                                else:
                                    if not self.ref_in_actor:
                                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(final_batch)
                                    else:
                                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(final_batch)
                                    final_batch = final_batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with _timer("values", timing_raw):
                                values = self.critic_wg.compute_values(final_batch)
                                final_batch = final_batch.union(values)
                                
                        # Get off policy batch stats
                        off_policy_stats = self.off_policy_manager.get_off_policy_stats()
                        
                        final_batch = compute_advantage(
                            final_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=True,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                            off_policy_stats=off_policy_stats,
                        )

                    # Check if we should train (for VLM models)
                    if "multi_modal_inputs" in final_batch.non_tensor_batch:
                        if len(final_batch) < self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n:
                            is_train = False
                        else:
                            is_train = True
                    else:
                        is_train = True

                    # update critic
                    if is_train and self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(final_batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if is_train and self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            final_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(final_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # === POST-TRAINING BUFFER ADDITION ===
                    # Now that training is complete, add all collected batches to buffer
                    if len(collected_batches_for_buffer) > 0:
                        print(f"Adding {len(collected_batches_for_buffer)} collected batches to buffer after training...")
                        
                        # Combine all collected batches into one
                        if len(collected_batches_for_buffer) == 1:
                            combined_buffer_batch = collected_batches_for_buffer[0]
                            combined_is_off_policy = collected_is_off_policy_flags[0]
                        else:
                            combined_buffer_batch = DataProto.concat(collected_batches_for_buffer)
                            # Use the first flag or majority vote for is_off_policy_rollout
                            combined_is_off_policy = collected_is_off_policy_flags[0]  # or any logic you prefer
                        
                        # Add the combined batch to buffer (batch already has old_log_probs computed)
                        self.off_policy_manager.collect_data(combined_buffer_batch, combined_is_off_policy)
                        
                        print(f"Successfully added combined batch with {len(combined_buffer_batch)} samples to buffer")
                    
                    # Clear collected batches for next iteration
                    collected_batches_for_buffer = []
                    collected_is_off_policy_flags = []

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

                # DAPO specific metrics
                metrics.update({
                    "train/num_gen_batches": num_gen_batches,
                    "train/keep_ratio": dapo_keep_ratio,
                })

                # collect metrics
                metrics.update(compute_data_metrics(batch=final_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=final_batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=final_batch, timing_raw=timing_raw, n_gpus=n_gpus))
                
                timing_raw = defaultdict(float)  # clear timing

                # Reset for next iteration
                batch = None
                final_batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.off_policy_manager.step_count += 1
                self.global_steps += 1

