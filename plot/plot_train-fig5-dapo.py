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

def create_metric_plot(project, run_ids, run_names, metric_name, title, ylabel, filename, show_legend=False):
    """Create plot for a specific metric"""
    # Get data for all runs
    dfs = []
    for run_id in run_ids:
        df = get_run_data(project, run_id, metric_name)
        dfs.append(df)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Colors
    colors = ['#4285F4', '#F4B400', '#0F9D58']  # Blue, Yellow, Green
    markers = ['o', '^', 's']  # Circle, Triangle, Square
    
    # Plot for each run
    for i, (df, run_name, color, marker) in enumerate(zip(dfs, run_names, colors, markers)):
        if df is not None and len(df) > 0:
            # Plot raw data points - sample every 20th point
            sample_idx = range(0, len(df), 20)
            plt.scatter(df['step'].iloc[sample_idx], df['value'].iloc[sample_idx], 
                       color=color, alpha=0.7, s=50, marker=marker, label=run_name)
            
            # Plot smoothed lines
            if len(df) > 5:
                window = max(3, len(df) // 20)
                smoothed = df['value'].rolling(window=window, center=True).mean()
                plt.plot(df['step'], smoothed, color=color, linewidth=4, alpha=0.9)
                
                # Add confidence band
                std = df['value'].rolling(window=window, center=True).std()
                plt.fill_between(df['step'], 
                                smoothed - std, 
                                smoothed + std, 
                                color=color, alpha=0.2)
    
    # Styling
    plt.xlim(0, 115)
    plt.xlabel('Training Steps', fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.title(title, fontsize=24, pad=10)
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
    if show_legend:
        plt.legend(fontsize=24)
    
    # Clean layout
    plt.tight_layout()
    
    # Save
    plt.savefig(filename, dpi=300)
    plt.show()

def create_all_plots():
    """Create all three plots"""
    # Configuration
    project = "off_policy_grpo_debug"
    run_ids = ["run_20250918_7171310c", "run_20250919_8a89b775", "run_20250914_b43380c6"]
    # run_ids = ["run_20250921_a993796a", "run_20250921_6f5b4277", "run_20250914_614e4af1"]
    run_names = ["BAPO", "DAPO", "GRPO"]
    
    # Metric configurations
    metrics_config = [
        {
            'metric_name': 'critic/origin_rewards/mean',
            'title': 'Planing Task - Train Rewards',
            'ylabel': 'Train Rewards',
            'filename': '2_train_rewards.png'
        },
        {
            'metric_name': 'actor/entropy_loss',
            'title': 'Planing Task - Actor Entropy',
            'ylabel': 'Actor Entropy',
            'filename': '2_actor_entropy.png'
        },
        {
            'metric_name': 'response_length/mean',
            'title': 'Planing Task - Response Length',
            'ylabel': 'Response Length',
            'filename': '2_response_length.png'
        }
    ]
    
    # Create each plot
    for i, config in enumerate(metrics_config):
        print(f"Creating plot for {config['metric_name']}...")
        create_metric_plot(
            project=project,
            run_ids=run_ids,
            run_names=run_names,
            metric_name=config['metric_name'],
            title=config['title'],
            ylabel=config['ylabel'],
            filename=config['filename'],
            show_legend=(i == 0)
        )
        print(f"Saved {config['filename']}")

if __name__ == "__main__":
    create_all_plots()