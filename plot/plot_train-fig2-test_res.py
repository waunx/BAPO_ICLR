import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_run_data(project, run_id):
    """Get single run data"""
    api = wandb.TrackingApi()
    run = api.run(project=project, run_id=run_id)
    history = run.history(name=['val-core/CD-4/reward/mean@8'])
    
    if 'val-core/CD-4/reward/mean@8' in history:
        return pd.DataFrame({
            'step': history['step'],
            'reward': history['val-core/CD-4/reward/mean@8']
        })
    return None
"""
DeepScaler: val-core//reward/mean@8
CountDown: val-core/countdown/reward/mean@8
Geometry3K: val-core/CD-4/reward/mean@8
"""
def create_plot():
    """Create simple comparison plot"""
    # Configuration
    project = "off_policy_grpo_debug"
    # run_ids = ["run_20250823_9695457d", "run_20250817_bb84357e"] # CountDown
    run_ids = ["run_20250902_2cacdb49", "run_20250902_f02acd3b"] # CountDown
    # run_ids = ["run_20250821_ce2976a2", "run_20250820_856cc51e"] # Geometry3K
    # run_ids = ["run_20250818_50b299f4", "run_20250820_856cc51e"] # Geometry3K
    # run_ids = ["run_20250831_c2095fe5", "run_20250831_1663f401"] # Geometry3K
    # run_ids = ["run_20250823_2af07759", "run_20250823_880f2146"] # DeepScalerR
    # run_ids = ["run_20250818_a43ac208", "run_20250817_a4f0e8cc"] # Spatial-Easy  
    # run_ids = ["run_20250821_5d111e23", "run_20250821_137301a5"] # Spatial-Hard  
    
    # Get data
    df1 = get_run_data(project, run_ids[0])
    df2 = get_run_data(project, run_ids[1])
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Colors - darker purple, lighter orange
    color1 = '#7d3c98'  # Darker Purple
    color2 = '#f39c12'  # Orange (lighter than before)
    
    # Plot data points and connect with lines - sample every 20th point
    if df1 is not None:
        # Purple circles - sample every 20th point
        sample_idx1 = range(0, len(df1), 1)
        sampled_steps1 = df1['step'].iloc[sample_idx1]
        sampled_rewards1 = df1['reward'].iloc[sample_idx1]
        
        # Plot scatter points
        plt.scatter(sampled_steps1, sampled_rewards1,
                    color=color1, alpha=0.7, s=150, marker='o', label='BAPO', zorder=3)
        # Connect with solid line
        plt.plot(sampled_steps1, sampled_rewards1, color=color1, linewidth=5, 
                 alpha=0.9, linestyle='-', zorder=2)
        
    if df2 is not None:
        # Orange triangles - sample every 20th point
        sample_idx2 = range(0, len(df2), 1)
        sampled_steps2 = df2['step'].iloc[sample_idx2]
        sampled_rewards2 = df2['reward'].iloc[sample_idx2]
        
        # Plot scatter points
        plt.scatter(sampled_steps2, sampled_rewards2,
                    color=color2, alpha=0.7, s=150, marker='^', label='GRPO', zorder=3)
        # Connect with dashed line
        plt.plot(sampled_steps2, sampled_rewards2, color=color2, linewidth=5, 
                 alpha=0.9, linestyle='--', zorder=2)
    
    # Styling
    plt.xlabel('Training Steps', fontsize=24)
    plt.ylabel('Test Accuracy (↑)', fontsize=24)
    # plt.title('Qwen2.5 Math 1.5B', fontsize=24, pad=10)
    # plt.title('Qwen2.5 VL 3B', fontsize=24, pad=10)
    # plt.title('Geo-3K Test Set', fontsize=24, pad=10)
    plt.title('CD-4 Test Set', fontsize=24, pad=10)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # Grid - bolder lines
    plt.grid(True, alpha=0.6, linestyle='-', linewidth=2)
    # plt.xlim(0, 160)
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
    plt.savefig('test_accuracy.png', dpi=300)
    plt.show()
    
    print("Plot saved as 'test_accuracy.png'")

if __name__ == "__main__":
    create_plot()