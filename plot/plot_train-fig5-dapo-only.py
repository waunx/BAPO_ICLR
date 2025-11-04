import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_run_data(project, run_id, metric_name):
    """Get single run data for specific metric"""
    api = wandb.TrackingApi()
    run = api.run(project=project, run_id=run_id)
    history = run.history(name=[metric_name])
    
    if metric_name in history:
        return pd.DataFrame({
            'step': history['step'],
            'value': history[metric_name]
        })
    return None

def create_dapo_comparison_plot():
    """Create comparison plot for DAPO across three tasks"""
    
    # Task configurations with their respective projects
    tasks_config = [
        {
            'project': "off_policy_grpo",
            'run_id': "run_20250915_60878ba6",
            'task_name': "Mathematical",
            'subplot_idx': 0
        },
        {
            'project': "off_policy_grpo_debug",
            'run_id': "run_20250919_8a89b775", 
            'task_name': "Planning",
            'subplot_idx': 1
        },
        {
            'project': "off_policy_grpo_debug",
            'run_id': "run_20250915_51efca58",
            'task_name': "Vision Geometry", 
            'subplot_idx': 2
        }
    ]
    
    metric_name = 'train/num_gen_batches'
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # Color for DAPO
    color = '#F4B400'  # Yellow color from original code
    marker = '^'       # Triangle marker from original code
    
    # Create each subplot
    for task_config in tasks_config:
        ax = axes[task_config['subplot_idx']]
        
        # Get data
        df = get_run_data(task_config['project'], task_config['run_id'], metric_name)
        
        if df is not None and len(df) > 0:
            print(f"Creating plot for {task_config['task_name']} task...")
            
            # Plot raw data points - connect with lines (no smoothing)
            ax.plot(df['step'], df['value'], color=color, linewidth=3, alpha=0.8, marker=marker, 
                   markersize=6, markevery=max(1, len(df)//50), label='DAPO')
            
        else:
            print(f"No data found for {task_config['task_name']} task")
        
        # Styling for each subplot
        ax.set_xlabel('Training Steps', fontsize=20)
        ax.set_ylabel('Rollout Times', fontsize=20)
        ax.set_title(f'{task_config["task_name"]} Task', fontsize=22, pad=10)
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        # Grid - bolder lines
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=2.5, axis='y')
        
        # Set bold border
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
            spine.set_color('black')
        
        # Add legend only to first subplot
        if task_config['subplot_idx'] == 0:
            ax.legend(fontsize=20)
    
    # Overall title
    # fig.suptitle('DAPO - Number of Generation Batches Across Tasks', fontsize=26, y=1.02)
    
    # Clean layout
    plt.tight_layout()
    
    # Save
    filename = 'dapo_three_tasks_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved {filename}")

if __name__ == "__main__":
    create_dapo_comparison_plot()