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

# try:
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
# except ImportError:
#     print("To use Math-Verify, please install it first by running `pip install math-verify`.")

import pandas as pd
import numpy as np
import re


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score


def evaluate_parquet_scores(file_path):
    """
    评估Parquet文件中每条数据的分数，支持多个rollout
    
    Args:
        file_path (str): Parquet文件路径
    """
    # 读取Parquet文件
    df = pd.read_parquet(file_path)
    print(f"成功读取文件，共包含 {len(df)} 条数据\n")
    
    # 存储每个题目的平均得分
    question_scores = []
    
    # 遍历每条数据并计算分数
    for idx, row in df.iterrows():
        try:
            # 从数据中提取必要参数
            data_source = row['data_source']
            # 处理responses字段（多个rollout）
            rollouts = row['responses'] if row['responses'] is not None else []
            # 从reward_model中提取ground_truth
            ground_truth = row['reward_model']['ground_truth']
            
            # 计算每个rollout的得分
            rollout_scores = []
            for i, solution_str in enumerate(rollouts):
                
                score = compute_score(
                    model_output=solution_str,
                    ground_truth=ground_truth,
                )
                rollout_scores.append(score)
            
            # 计算该题目的平均得分
            if rollout_scores:  # 检查是否有rollout分数（避免空列表）
                avg_score = np.mean(rollout_scores)
                question_scores.append(avg_score)
                
                # 输出结果
                print(f"第 {idx} 条数据：")
                print(f"数据来源: {data_source}")
                print(f"参考答案: {ground_truth}")
                print(f"rollout数量: {len(rollout_scores)}")
                print(f"各rollout得分: {rollout_scores}")
                print(f"题目平均得分: {avg_score:.4f}")
                print("-" * 80)
            else:
                print(f"第 {idx} 条数据：没有有效的rollout数据")
                print("-" * 80)
                
        except Exception as e:
            print(f"第 {idx} 条数据处理出错: {str(e)}")
            print("-" * 80)
    
    # 计算整体平均得分
    if question_scores:
        overall_avg = np.mean(question_scores)
        print(f"\n===== 评估总结 =====")
        print(f"总题目数量: {len(question_scores)}")
        print(f"整体平均得分: {overall_avg:.4f}")
    else:
        print("\n没有有效的评分数据")


# 执行评估
if __name__ == "__main__":
    # 替换为你的Parquet文件路径
    parquet_path = "/mnt/bn/robotics-rl-lf/vlm_ckpts/off_policy_grpo/off_sample_off_rollout-deepseek1.5b-train-v2.0/global_step_400/amc_val_deepseek1.5b.parquet"
    evaluate_parquet_scores(parquet_path)
