import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_run_data(project, run_id):
    """Get single run data"""
    if run_id == "come from csv":
        # 读取CSV文件，替换 'your_file.csv' 为实际文件名
        df = pd.read_csv('/opt/tiger/e2e_alg/vlm/verl/off-dapo-math-1.5b-v1-2025-8-28_09_54_23.csv')
        return pd.DataFrame({
            'step': df.iloc[:, 0],  # 第一列作为step
            'reward': df.iloc[:, 1]  # 第二列作为reward
        })
    else:
        api = wandb.TrackingApi()
        run = api.run(project=project, run_id=run_id)
        history = run.history(name=['critic/origin_rewards/mean'])
        
        if 'critic/origin_rewards/mean' in history:
            return pd.DataFrame({
                'step': history['step'],
                'reward': history['critic/origin_rewards/mean']
            })
        return None

def create_plot():
    """Create simple comparison plot"""
    # Configuration
    project = "off_policy_grpo"
    run_ids = ["come from csv","run_20250826_8c5cfcc0"] # DAPO
    
    # Get data
    df1 = get_run_data(project, run_ids[0])
    df2 = get_run_data(project, run_ids[1])
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Colors - darker purple, lighter orange
    color1 = '#7d3c98'  # Darker Purple
    color2 = '#f39c12'  # Orange (lighter than before)
    
    # Plot raw data points - fewer, larger points with different shapes
    if df1 is not None:
        # Purple circles - sample every 5th point
        sample_idx1 = range(0, len(df1), 20)
        plt.scatter(df1['step'].iloc[sample_idx1], df1['reward'].iloc[sample_idx1],
                    color=color1, alpha=0.7, s=50, marker='o', label='BAPO')
    if df2 is not None:
        # Orange triangles - sample every 5th point
        sample_idx2 = range(0, len(df2), 20)
        plt.scatter(df2['step'].iloc[sample_idx2], df2['reward'].iloc[sample_idx2],
                    color=color2, alpha=0.7, s=50, marker='^', label='GRPO')
    
    # Plot smoothed lines
    if df1 is not None and len(df1) > 5:
        window = max(3, len(df1) // 10)
        smoothed1 = df1['reward'].rolling(window=window, center=True).mean()
        plt.plot(df1['step'], smoothed1, color=color1, linewidth=3, alpha=0.9)
        
        # Add confidence band
        std1 = df1['reward'].rolling(window=window, center=True).std()
        plt.fill_between(df1['step'],
                         smoothed1 - std1,
                         smoothed1 + std1,
                         color=color1, alpha=0.2)
    
    if df2 is not None and len(df2) > 5:
        window = max(3, len(df2) // 10)
        smoothed2 = df2['reward'].rolling(window=window, center=True).mean()
        plt.plot(df2['step'], smoothed2, color=color2, linewidth=3, alpha=0.9)
        
        # Add confidence band
        std2 = df2['reward'].rolling(window=window, center=True).std()
        plt.fill_between(df2['step'],
                         smoothed2 - std2,
                         smoothed2 + std2,
                         color=color2, alpha=0.2)
    
    # Styling
    plt.xlabel('Training Steps', fontsize=24)
    plt.ylabel('Train Rewards (↑)', fontsize=24)
    plt.title('DeepSeek-Distilled Qwen 1.5B', fontsize=24, pad=10)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # Grid - bolder lines
    plt.grid(True, alpha=0.6, linestyle='-', linewidth=2)
    
    # Set bold border
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    
    # Legend
    plt.legend(fontsize=24)
    
    plt.ylim(0, 0.6)
    # Clean layout
    plt.tight_layout()
    
    # Save
    plt.savefig('training_comparison.png', dpi=300)
    plt.show()
    
    print("Plot saved as 'training_comparison.png'")

if __name__ == "__main__":
    create_plot()