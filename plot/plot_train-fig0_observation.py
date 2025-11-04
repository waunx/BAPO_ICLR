import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple

class GRPOComparisonAnalyzer:
    """
    GRPO前后效果对比分析器
    """
    
    def __init__(self, on_policy_dir: str):
        """
        Args:
            on_policy_dir: on-policy tracking结果保存目录
        """
        self.on_policy_dir = on_policy_dir
        self.baseline_data = None
        self.tracking_data = {}
        
    def load_data(self):
        """加载on-policy的tracking数据"""
        print("Loading GRPO data...")
        
        # 加载baseline
        baseline_file = os.path.join(self.on_policy_dir, "baseline.json")
        if not os.path.exists(baseline_file):
            raise FileNotFoundError(f"Baseline file not found: {baseline_file}")
            
        with open(baseline_file, 'r', encoding='utf-8') as f:
            self.baseline_data = json.load(f)
        
        # 加载tracking步骤数据
        tracking_files = []
        for filename in os.listdir(self.on_policy_dir):
            if filename.startswith("tracking_step_") and filename.endswith(".json"):
                step = int(filename.replace("tracking_step_", "").replace(".json", ""))
                tracking_files.append((step, filename))
        
        tracking_files.sort()
        
        for step, filename in tracking_files:
            filepath = os.path.join(self.on_policy_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                self.tracking_data[step] = json.load(f)
        
        print(f"Loaded tracking steps: {list(self.tracking_data.keys())}")
    
    def categorize_prompts_by_baseline_accuracy(self, num_bins: int = 9):
        """根据baseline准确率将prompts分成9个区间"""
        prompt_hash_to_acc = {}
        for prompt_hash, data in self.baseline_data["data"].items():
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
    
    def get_accuracy_distribution_for_step(self, step, num_bins: int = 9):
        """获取特定步骤的准确率分布"""
        categorized_prompts, bins = self.categorize_prompts_by_baseline_accuracy(num_bins)
        
        if step == "baseline":
            step_distribution = defaultdict(lambda: defaultdict(int))
            for original_bin, prompt_hashes in categorized_prompts.items():
                step_distribution[original_bin][original_bin] = len(prompt_hashes)
            return step_distribution
        
        if step not in self.tracking_data:
            raise ValueError(f"Step {step} not found in tracking data")
        
        step_data = self.tracking_data[step]
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
    
    def plot_grpo_before_after(self, num_bins: int = 9, figsize: Tuple[int, int] = (12, 10)):
        """绘制GRPO前后对比图，2x1布局"""
        if not self.baseline_data or not self.tracking_data:
            raise ValueError("Data must be loaded first")
        
        # 获取最后步骤
        last_step = max(self.tracking_data.keys())
        
        # 获取分布数据
        baseline_distribution = self.get_accuracy_distribution_for_step("baseline", num_bins)
        final_distribution = self.get_accuracy_distribution_for_step(last_step, num_bins)
        
        # 配色方案
        colors = []
        low_performance_bins = [0]  # 低性能区间 (0/8 accuracy)
        for i in range(num_bins):
            if i in low_performance_bins:
                red_intensity = 0.6 + (i / max(1, len(low_performance_bins))) * 0.4
                colors.append(plt.cm.Reds(red_intensity))
            else:
                blue_intensity = 0.4 + ((i - len(low_performance_bins)) / (num_bins - len(low_performance_bins))) * 0.5
                colors.append(plt.cm.Blues(blue_intensity))
        color_palette = {i: colors[i] for i in range(num_bins)}
        
        # 创建2x1子图布局
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'hspace': 0.3})
        
        # 两个子图的数据和标题
        distributions = [
            (baseline_distribution, 'Before GRPO (DeepSeek Distilled 1.5B)', axes[0]),
            (final_distribution, f'After GRPO (3 Epoch)', axes[1])
        ]
        
        # 绘制两个分布图
        for distribution, title, ax in distributions:
            x_positions = np.arange(num_bins)
            bottom = np.zeros(num_bins)
            
            # 绘制堆叠柱状图
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
            
            # 在柱子顶部添加总数标签
            for i in range(num_bins):
                total_height = bottom[i]
                if total_height > 0:
                    text_color = color_palette[i]
                    font_weight = 'bold' if i in low_performance_bins else 'bold'
                    ax.text(x_positions[i], total_height + max(bottom) * 0.01, 
                        str(int(total_height)), 
                        ha='center', va='bottom', 
                        color=text_color, fontsize=22, fontweight=font_weight)
            
            # 设置边框
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            
            # 设置坐标轴和标题
            if ax == axes[1]:  # 只在底部子图显示x轴标签
                ax.set_xlabel('Accuracy Bins', fontsize=22, fontweight='bold')
            ax.set_ylabel('Number of Prompts', fontsize=22, fontweight='bold')
            ax.set_title(title, fontsize=22, fontweight='bold')
            ax.set_xticks(x_positions)
            x_labels = [f'{i}/8' for i in range(9)]
            ax.set_xticklabels(x_labels, fontsize=22)
            
            # 低准确率组标签标红
            x_tick_labels = ax.get_xticklabels()
            for i in low_performance_bins:
                x_tick_labels[i].set_color('red')
                x_tick_labels[i].set_weight('normal')
            
            # 设置y轴范围和网格
            y_max = ax.get_ylim()[1]
            ax.set_ylim(0, y_max * 1.2)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=22)
        
        # 添加红色高亮说明（仅在顶部子图）
        axes[0].text(0.18, 0.85, 'Red bars highlight 0/8 accuracy group', 
                    transform=axes[0].transAxes, fontsize=22, color='red', fontweight='bold',
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.8, edgecolor='red'))
        
        plt.tight_layout()
        return fig
    
    def save_grpo_analysis(self, output_dir: str = None):
        """保存GRPO前后对比分析结果"""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(self.on_policy_dir), "grpo_before_after_analysis")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存对比图
        fig = self.plot_grpo_before_after(num_bins=9, figsize=(14, 10))
        fig.savefig(os.path.join(output_dir, "grpo_before_after_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"GRPO before/after analysis saved to: {output_dir}")
        return output_dir


def main():
    """主函数 - 专门用于GRPO前后对比"""
    # 设置on-policy的tracking目录路径
    on_policy_dir = "/mnt/bn/robotics-rl-lf/vlm_datasets/DeepScaleR-Preview-Dataset/on_sample_on_rollout-deepseek1.5b-train-v2.0-tracking_results"
    
    # 初始化GRPO分析器
    analyzer = GRPOComparisonAnalyzer(on_policy_dir)
    
    try:
        # 加载数据
        analyzer.load_data()
        
        # 绘制GRPO前后对比图
        print("Generating Before/After GRPO comparison plot...")
        fig = analyzer.plot_grpo_before_after(num_bins=9)
        plt.show()
        
        # 保存结果
        print("Saving GRPO before/after analysis...")
        output_dir = analyzer.save_grpo_analysis()
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()