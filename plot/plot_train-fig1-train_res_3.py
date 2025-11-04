import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_run_data(project, run_id):
    """Get single run data"""
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
    project = "off_policy_grpo_debug"
    run_ids = ["run_20250915_11848941", "run_20250919_8a89b775", "run_20250914_b43380c6"] # CountDown
    # Get data
    df1 = get_run_data(project, run_ids[0])
    df2 = get_run_data(project, run_ids[1])
    df3 = get_run_data(project, run_ids[2])
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Colors - darker purple, lighter orange
    # color1 = '#7d3c98'  # Darker Purple
    # color2 = '#f39c12'  # Orange (lighter than before)

    color1 = '#4285F4'  # Blue
    color2 = '#F4B400'  # Yellow
    color3 = '#0F9D58'  # Green
    
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
                   color=color2, alpha=0.7, s=50, marker='^', label='DAPO')
        
    if df3 is not None:
        # Green triangles - sample every 5th point
        sample_idx3 = range(0, len(df3), 20)
        plt.scatter(df3['step'].iloc[sample_idx3], df3['reward'].iloc[sample_idx3], 
                   color=color3, alpha=0.7, s=50, marker='s', label='GRPO')
    
    # Plot smoothed lines
    if df1 is not None and len(df1) > 5:
        window = max(3, len(df1) // 20)
        smoothed1 = df1['reward'].rolling(window=window, center=True).mean()
        plt.plot(df1['step'], smoothed1, color=color1, linewidth=3, alpha=0.9)
        
        # Add confidence band
        std1 = df1['reward'].rolling(window=window, center=True).std()
        plt.fill_between(df1['step'], 
                        smoothed1 - std1, 
                        smoothed1 + std1, 
                        color=color1, alpha=0.2)
    
    if df2 is not None and len(df2) > 5:
        window = max(3, len(df2) // 20)
        smoothed2 = df2['reward'].rolling(window=window, center=True).mean()
        plt.plot(df2['step'], smoothed2, color=color2, linewidth=3, alpha=0.9)
        
        # Add confidence band
        std2 = df2['reward'].rolling(window=window, center=True).std()
        plt.fill_between(df2['step'], 
                        smoothed2 - std2, 
                        smoothed2 + std2, 
                        color=color2, alpha=0.2)
    
    if df3 is not None and len(df3) > 5:
        window = max(3, len(df3) // 20)
        smoothed3 = df3['reward'].rolling(window=window, center=True).mean()
        plt.plot(df3['step'], smoothed3, color=color3, linewidth=3, alpha=0.9)
        
        # Add confidence band
        std3 = df3['reward'].rolling(window=window, center=True).std()
        plt.fill_between(df3['step'], 
                        smoothed3 - std3, 
                        smoothed3 + std3, 
                        color=color3, alpha=0.2)
    # Styling
    plt.xlim(0, 115)
    plt.xlabel('Training Steps', fontsize=24)
    plt.ylabel('Train Rewards (↑)', fontsize=24)
    # plt.title('Planning Task \n (Qwen2.5 Math 1.5B)', fontsize=24, pad=10)
    plt.title('Planning Task', fontsize=24, pad=10)
    # plt.title('Vison Geometry Task \n (Qwen2.5 VL 3B)', fontsize=24, pad=10)
    # plt.title('Visual Geometry Task', fontsize=24, pad=10)
    # plt.title('Mathmetical Task \n (DeepSeek-Distilled Qwen 1.5B)', fontsize=24, pad=20)
    # plt.title('Mathmetical Task', fontsize=24, pad=20)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # Grid - bolder lines
    plt.grid(True, alpha=0.6, linestyle='-', linewidth=2.5, axis='y')
    
    # Set bold border
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    
    # Legend
    plt.legend(fontsize=24)
    
    # Clean layout
    plt.tight_layout()
    
    # Save
    plt.savefig('3_training_comparison.png', dpi=300)
    plt.show()
    
    print("Plot saved as 'training_comparison.png'")

if __name__ == "__main__":
    create_plot()