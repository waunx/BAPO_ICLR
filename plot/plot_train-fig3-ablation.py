import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_run_data(project, run_id):
    """Get single run data"""
    api = wandb.TrackingApi()
    run = api.run(project=project, run_id=run_id)
    history = run.history(name=['val-core/CD-3to4/reward/mean@8'])
    
    if 'val-core/CD-3to4/reward/mean@8' in history:
        return pd.DataFrame({
            'step': history['step'],
            'reward': history['val-core/CD-3to4/reward/mean@8']
        })
    return None

def create_plot():
    """Create simple comparison plot"""
    # Configuration
    project = "off_policy_grpo_debug"
    # run_ids = ["run_20250823_efc4951a", "run_20250824_bd6590e6", "run_20250817_bb84357e"] # CountDown  
    # run_names = ["BAPO (v=5,m=5)", "BAPO (v=10,m=5)", "BAPO (v=20,m=5)"]
    
    # run_ids = ["run_20250724_be784f98", "run_20250723_7ca53d2d", "run_20250724_11fc3509"] # CountDown  
    # run_names = ["BAPO (kl coef=0)", "BAPO (kl coef=0.01)", "BAPO (kl coef=0.05)"]

    # run_ids = ["run_20250828_61266d6e", "run_20250827_82bee48b", "run_20250828_2e272161"] # CountDown  
    # run_names = ["BAPO (v=5,m=1)", "BAPO (v=5,m=5)", "BAPO (v=5,m=10)"]

    # run_ids = ["run_20250831_1380a67a", "run_20250831_cde69e3a", "run_20250831_35579d56"] # CountDown  
    # run_names = ["BAPO", "BAPO (w/o X2)", "BAPO (w/o X3)"]
    

    run_ids = ["run_20250831_1380a67a", "run_20250831_28aa723e", "run_20250831_95b3cf72"] # CountDown  
    run_names = ["BAPO (adap c2 c3)", "BAPO (c2=4/8, c3=5/8)", "BAPO (c2=1/8, c3=2/8)"]
    

    # run_ids = ["run_20250831_c2095fe5", "run_20250901_8416a32e", "run_20250831_37663daf"]
    # run_names = ["BAPO", "BAPO (w/o X2)", "BAPO (w/o X3)"]


    # Get data
    df1 = get_run_data(project, run_ids[0])
    df2 = get_run_data(project, run_ids[1])
    df3 = get_run_data(project, run_ids[2])
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Colors - darker purple, lighter orange
    color1 = '#4285F4'  # Blue
    color2 = '#F4B400'  # Yellow
    color3 = '#0F9D58'  # Green
    # Plot data points and connect with lines - sample every 20th point
    if df1 is not None:
        # Purple circles - sample every 20th point
        sample_idx1 = range(0, len(df1), 1)
        sampled_steps1 = df1['step'].iloc[sample_idx1]
        sampled_rewards1 = df1['reward'].iloc[sample_idx1]
        
        # Plot scatter points
        plt.scatter(sampled_steps1, sampled_rewards1,
                    color=color1, alpha=0.7, s=150, marker='o', label=run_names[0], zorder=3)
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
                    color=color2, alpha=0.7, s=150, marker='^', label=run_names[1], zorder=3)
        # Connect with dashed line
        plt.plot(sampled_steps2, sampled_rewards2, color=color2, linewidth=5, 
                 alpha=0.9, linestyle='--', zorder=2)
    
    if df3 is not None:
        # Orange triangles - sample every 20th point
        sample_idx3 = range(0, len(df3), 1)
        sampled_steps3 = df3['step'].iloc[sample_idx3]
        sampled_rewards3 = df3['reward'].iloc[sample_idx3]
        
        # Plot scatter points
        plt.scatter(sampled_steps3, sampled_rewards3,
                    color=color3, alpha=0.7, s=150, marker='*', label=run_names[2], zorder=3)
        # Connect with dashed line
        plt.plot(sampled_steps3, sampled_rewards3, color=color3, linewidth=5, 
                 alpha=0.9, linestyle=':', zorder=2)
        
    # Styling
    plt.xlabel('Training Steps', fontsize=24)
    plt.ylabel('Test Accuracy (↑)', fontsize=24)
    # plt.title('Qwen2.5 Math 1.5B', fontsize=24, pad=10)
    # plt.title('Qwen2.5 VL 3B', fontsize=24, pad=10)
    # plt.title('Geo3K Test Set', fontsize=24, pad=10)
    plt.title('CD-34 Test Set', fontsize=24, pad=10)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # plt.xlim(0, 200)
    # Grid - bolder lines
    plt.grid(True, alpha=0.6, linestyle='-', linewidth=2)
    
    # Set bold border
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    
    # Legend
    plt.legend(fontsize=24, loc='lower right')
    
    # Clean layout
    plt.tight_layout()
    
    # Save
    plt.savefig('ablation.png', dpi=300)
    plt.show()
    
    print("Plot saved as 'ablation.png'")

if __name__ == "__main__":
    create_plot()