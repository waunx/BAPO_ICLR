import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple

class PolicyComparisonAnalyzer:
    """
    对比分析器，专门用于比较on-policy和off-policy RLVR的效果
    """
    
    def __init__(self, on_policy_dir: str, off_policy_dir: str):
        """
        Args:
            on_policy_dir: on-policy tracking结果保存目录
            off_policy_dir: off-policy tracking结果保存目录
        """
        self.on_policy_dir = on_policy_dir
        self.off_policy_dir = off_policy_dir
        self.on_baseline_data = None
        self.off_baseline_data = None
        self.on_tracking_data = {}
        self.off_tracking_data = {}
        
    def load_data(self):
        """加载两种策略的所有tracking数据"""
        # 加载on-policy数据
        print("Loading on-policy data...")
        self._load_single_policy_data(self.on_policy_dir, 'on')
        
        # 加载off-policy数据
        print("Loading off-policy data...")
        self._load_single_policy_data(self.off_policy_dir, 'off')
        
        print(f"On-policy steps: {list(self.on_tracking_data.keys())}")
        print(f"Off-policy steps: {list(self.off_tracking_data.keys())}")
    
    def _load_single_policy_data(self, tracking_dir: str, policy_type: str):
        """加载单个策略的数据"""
        # 加载baseline
        baseline_file = os.path.join(tracking_dir, "baseline.json")
        if not os.path.exists(baseline_file):
            raise FileNotFoundError(f"Baseline file not found: {baseline_file}")
            
        with open(baseline_file, 'r', encoding='utf-8') as f:
            baseline_data = json.load(f)
        
        if policy_type == 'on':
            self.on_baseline_data = baseline_data
        else:
            self.off_baseline_data = baseline_data
        
        # 加载tracking步骤数据
        tracking_files = []
        for filename in os.listdir(tracking_dir):
            if filename.startswith("tracking_step_") and filename.endswith(".json"):
                step = int(filename.replace("tracking_step_", "").replace(".json", ""))
                tracking_files.append((step, filename))
        
        tracking_files.sort()
        
        tracking_data = {}
        for step, filename in tracking_files:
            filepath = os.path.join(tracking_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                tracking_data[step] = json.load(f)
        
        if policy_type == 'on':
            self.on_tracking_data = tracking_data
        else:
            self.off_tracking_data = tracking_data
    
    def categorize_prompts_by_baseline_accuracy(self, baseline_data, num_bins: int = 9):
        """根据baseline准确率将prompts分成9个区间"""
        prompt_hash_to_acc = {}
        for prompt_hash, data in baseline_data["data"].items():
            acc = data["baseline_acc"]
            prompt_hash_to_acc[prompt_hash] = acc
        
        bins = [i/8 for i in range(9)]  # [0, 0.125, 0.25, ..., 1.0]
        categorized_prompts = defaultdict(list)
        
        for prompt_hash, acc in prompt_hash_to_acc.items():
            bin_idx = np.digitize(acc, bins) - 1
            
            if acc == 1.0:
                bin_idx = 8
            elif bin_idx < 0:
                bin_idx = 0
            elif bin_idx >= 9:
                bin_idx = 8
            
            categorized_prompts[bin_idx].append(prompt_hash)
        
        return dict(categorized_prompts), bins
    
    def get_accuracy_distribution_for_step(self, policy_type: str, step, num_bins: int = 9):
        """获取特定策略和步骤的准确率分布"""
        if policy_type == 'on':
            baseline_data = self.on_baseline_data
            tracking_data = self.on_tracking_data
        else:
            baseline_data = self.off_baseline_data
            tracking_data = self.off_tracking_data
        
        categorized_prompts, bins = self.categorize_prompts_by_baseline_accuracy(baseline_data, num_bins)
        
        if step == "baseline":
            step_distribution = defaultdict(lambda: defaultdict(int))
            for original_bin, prompt_hashes in categorized_prompts.items():
                step_distribution[original_bin][original_bin] = len(prompt_hashes)
            return step_distribution
        
        if step not in tracking_data:
            raise ValueError(f"Step {step} not found in {policy_type}-policy tracking data")
        
        step_data = tracking_data[step]
        full_history = step_data["full_history"]
        step_distribution = defaultdict(lambda: defaultdict(int))
        
        for original_bin, prompt_hashes in categorized_prompts.items():
            for prompt_hash in prompt_hashes:
                if prompt_hash in full_history:
                    history = full_history[prompt_hash]
                    current_acc = history[-1][1]
                    
                    current_bin = np.digitize(current_acc, bins) - 1
                    if current_acc == 1.0:
                        current_bin = 8
                    elif current_bin < 0:
                        current_bin = 0
                    elif current_bin >= 9:
                        current_bin = 8
                    
                    step_distribution[current_bin][original_bin] += 1
        
        return step_distribution
    
    def calculate_improvement_rate(self, baseline_dist, final_dist, target_bins, num_bins=9):
        """计算特定bins的改善率"""
        improvements = []
        for original_bin in target_bins:
            if original_bin not in baseline_dist:
                improvements.append(0)
                continue
                
            total_original = sum(baseline_dist[original_bin].values())
            if total_original == 0:
                improvements.append(0)
                continue
            
            # 计算移动到更高bins的数量
            improved_count = 0
            for current_bin in range(original_bin + 1, num_bins):
                improved_count += final_dist[current_bin].get(original_bin, 0)
            
            improvement_rate = improved_count / total_original
            improvements.append(improvement_rate)
        
        return improvements
    
    def plot_policy_comparison(self, num_bins: int = 9, figsize: Tuple[int, int] = (24, 6)):  # 加宽画布以适应3个子图
        if not self.on_tracking_data or not self.off_tracking_data:
            raise ValueError("Both policy data must be loaded")
        
        # 获取最后步骤
        on_last_step = max(self.on_tracking_data.keys())
        off_last_step = max(self.off_tracking_data.keys())
        
        # 获取分布数据（使用on-policy的baseline作为统一基准）
        baseline_distribution = self.get_accuracy_distribution_for_step('on', "baseline", num_bins)
        on_final_distribution = self.get_accuracy_distribution_for_step('on', on_last_step, num_bins)
        off_final_distribution = self.get_accuracy_distribution_for_step('off', off_last_step, num_bins)
        
        # 配色方案（保持不变）
        colors = []
        low_performance_bins = [0]  # 低性能区间定义
        for i in range(num_bins):
            if i in low_performance_bins:
                red_intensity = 0.6 + (i / len(low_performance_bins)) * 0.4
                colors.append(plt.cm.Reds(red_intensity))
            else:
                blue_intensity = 0.4 + ((i - len(low_performance_bins)) / (num_bins - len(low_performance_bins))) * 0.5
                colors.append(plt.cm.Blues(blue_intensity))
        color_palette = {i: colors[i] for i in range(num_bins)}
        
        # 创建1x3子图布局（关键修改：从1x2改为1x3）
        fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'wspace': 0.2})  # 增加间距避免拥挤
        
        # 三个子图的数据和标题（新增Base的分布）
        distributions = [
            (baseline_distribution, 'Base (Step 0)', axes[0]),  # 最左侧：初始状态
            (on_final_distribution, f'GRPO (Step {on_last_step})', axes[1]),
            (off_final_distribution, f'BAPO (Step {off_last_step})', axes[2])
        ]
        
        # 绘制三个分布图
        for distribution, title, ax in distributions:
            x_positions = np.arange(num_bins)
            bottom = np.zeros(num_bins)
            
            # 绘制堆叠柱状图（逻辑不变）
            for original_bin in range(num_bins):
                heights = [distribution[current_bin][original_bin] for current_bin in range(num_bins)]
                if original_bin in low_performance_bins:
                    bars = ax.bar(x_positions, heights, bottom=bottom, 
                        color=color_palette[original_bin], 
                        alpha=0.9, edgecolor='darkred', linewidth=1.5,
                        hatch='///')
                else:
                    bars = ax.bar(x_positions, heights, bottom=bottom, 
                        color=color_palette[original_bin], 
                        alpha=0.7, edgecolor='white', linewidth=0.8)
                bottom += heights
            
            # 在柱子顶部添加总数标签（逻辑不变）
            for i in range(num_bins):
                total_height = bottom[i]
                if total_height > 0:
                    text_color = color_palette[i]
                    font_weight = 'bold' if i in low_performance_bins else 'bold'
                    ax.text(x_positions[i], total_height + max(bottom) * 0.01, 
                        str(int(total_height)), 
                        ha='center', va='bottom', 
                        color=text_color, fontsize=16, fontweight=font_weight)
            
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)  # 设置边框线宽，数值越大边框越粗（如2.0、3.0等）

            # 设置坐标轴和标题（统一样式）
            ax.set_xlabel('Accuracy Bins', fontsize=18, fontweight='normal')
            if ax == axes[0]:  # 仅左侧子图显示y轴标签，避免重复
                ax.set_ylabel('Number of Prompts', fontsize=18, fontweight='normal')
            ax.set_title(title, fontsize=18, fontweight='normal', pad=15)
            ax.set_xticks(x_positions)
            x_labels = [f'{i}/8' for i in range(9)]
            ax.set_xticklabels(x_labels, fontsize=16)
            
            # 低准确率组标签标红
            x_tick_labels = ax.get_xticklabels()
            for i in low_performance_bins:
                x_tick_labels[i].set_color('red')
                x_tick_labels[i].set_weight('normal')
            
            # 增加y轴内部空白（关键修改：扩展y轴上限）
            y_max = ax.get_ylim()[1]
            ax.set_ylim(0, y_max * 1.1)  # 上限扩展为原最大值的1.3倍（可根据需要调整倍数）
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        return fig

    def save_comparison_analysis(self, output_dir: str = None):
        """保存对比分析结果"""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(self.on_policy_dir), "policy_comparison_analysis")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存对比图
        fig = self.plot_policy_comparison(num_bins=9, figsize=(24, 4))
        fig.savefig(os.path.join(output_dir, "grpo_vs_bapo_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Policy comparison analysis saved to: {output_dir}")
        return output_dir


def main():
    """主函数"""
    # 设置两个策略的tracking目录路径

    # on_policy_dir = "/mnt/bn/robotics-rl-lf/vlm_datasets/DeepScaleR-Preview-Dataset/on_sample_on_rollout-deepseek1.5b-train-v2.0-tracking_results"
    # off_policy_dir = "/mnt/bn/robotics-rl-lf/vlm_datasets/DeepScaleR-Preview-Dataset/off_sample_off_rollout-deepseek1.5b-train-v2.0-tracking_results"
    

    # on_policy_dir = "/mnt/bn/robotics-rl-lf/vlm_datasets/countdown/on_sample_on_rollout-qwen2_1.5b-countdown-tracking_results"
    # off_policy_dir = "/mnt/bn/robotics-rl-lf/vlm_datasets/countdown/off_sample_off_rollout-gaussian-qwen2_1.5b-kl-0.1-countdown-tracking_results"

    on_policy_dir = "/mnt/bn/robotics-rl-lf/vlm_datasets/geometry3k/on_sample_on_rollout-qwenvl_3b-geometry3k-train-v3-tracking_results"
    off_policy_dir = "/mnt/bn/robotics-rl-lf/vlm_datasets/geometry3k/off_sample_off_rollout-qwenvl_3b-geometry3k-train-tracking_results"

    # 初始化对比分析器
    analyzer = PolicyComparisonAnalyzer(on_policy_dir, off_policy_dir)
    
    try:
        # 加载数据
        analyzer.load_data()
        
        # 绘制对比分析图
        print("Generating GRPO vs BAPO comparison plot...")
        fig = analyzer.plot_policy_comparison(num_bins=9)
        plt.show()
        
        # 保存结果
        print("Saving comparison analysis...")
        output_dir = analyzer.save_comparison_analysis()
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
