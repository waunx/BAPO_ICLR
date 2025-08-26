import pandas as pd
import numpy as np
import re

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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


def compute_score(solution_str, ground_truth) -> float:
    """计算单个回答的得分"""
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            print(answer, ground_truth)
            if is_equiv(answer, ground_truth):
                retval = 1.0
    except Exception as e:
        print(f"评分时出错: {e}")

    return retval


def is_equiv(str1, str2, verbose=False):
    """判断两个答案是否等价"""
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    """移除答案中的boxed标记"""
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left) :]

    left = "\\boxed{"
    if s[: len(left)] == left and s[-1] == "}":
        return s[len(left) : -1]

    return s  # 如果没有找到boxed标记，返回原始字符串


def last_boxed_only_string(string):
    """提取最后一个boxed中的内容"""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def fix_fracs(string):
    """修复分数格式"""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def fix_a_slash_b(string):
    """将a/b格式转换为\frac{a}{b}"""
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except:  # noqa: E722
        return string


def remove_right_units(string):
    """移除右侧的单位描述"""
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def fix_sqrt(string):
    """修复平方根格式"""
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def remove_leading_zeros(s):
    """移除数字中的前导零，但保留单个零和小数中的前导零"""
    # 处理整数情况 (123, 055)
    if re.fullmatch(r'\d+', s):
        if len(s) > 1 and s[0] == '0':
            return s.lstrip('0') or '0'  # 防止全零情况变成空字符串
        return s
    
    # 处理带小数点的情况 (0.5, 001.23, .34)
    if '.' in s:
        parts = s.split('.')
        # 处理整数部分
        if parts[0]:  # 有整数部分
            parts[0] = parts[0].lstrip('0') or '0'
        # 处理小数部分（不移除末尾零，因为可能有精度要求）
        return '.'.join(parts)
    
    return s


def strip_string(string):
    """标准化字符串，用于比较答案"""
    # 移除换行
    string = string.replace("\n", "")

    # 移除反斜杠空格
    string = string.replace("\\!", "")

    # 替换双反斜杠为单反斜杠
    string = string.replace("\\\\", "\\")

    # 替换tfrac和dfrac为frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # 移除\left和\right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # 移除度符号
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # 移除美元符号
    string = string.replace("\\$", "")

    # 移除右侧单位
    string = remove_right_units(string)

    # 移除百分比符号
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # 处理小数格式
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) > 0 and string[0] == ".":
        string = "0" + string

    # 处理等号开头的情况
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # 修复平方根格式
    string = fix_sqrt(string)

    # 移除空格
    string = string.replace(" ", "")

    # 修复分数格式
    string = fix_fracs(string)

    # 特殊处理0.5
    if string == "0.5":
        string = "\\frac{1}{2}"

    # 修复a/b格式
    string = fix_a_slash_b(string)
    
    # 移除数字前导零（新增处理）
    # 先检查是否为纯数字或数字表达式
    if re.fullmatch(r'[\d.]+', string):
        string = remove_leading_zeros(string)

    return string


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
                    solution_str=solution_str,
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
    parquet_path = "/mnt/bn/robotics-rl-lf/vlm_ckpts/off_policy_grpo/off_sample_off_rollout-deepseek1.5b-train-v2.0/global_step_400/minerva_val_deepseek1.5b.parquet"
    evaluate_parquet_scores(parquet_path)
