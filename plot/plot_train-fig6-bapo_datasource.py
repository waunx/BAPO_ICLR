import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_data_sources(project, run_id):
    """Get data source metrics from wandb"""
    api = wandb.TrackingApi()
    run = api.run(project=project, run_id=run_id)
    
    # Get all history data
    history = run.history()
    
    # Convert to DataFrame if needed
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history)
    
    return history

def create_normalized_plot(df):
    """Create normalized percentage plot"""
    plt.figure(figsize=(8, 6))
    
    step_col = '_step' if '_step' in df.columns else 'step'
    
    steps = df[step_col]
    current = df['actor/samples_current']
    promoted = df['actor/samples_promoted'] 
    target = df['actor/samples_target']
    
    # Calculate percentages
    total = current + promoted + target
    current_pct = current / total * 100
    promoted_pct = promoted / total * 100
    target_pct = target / total * 100
    
    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']  # current, promoted, target
    labels = ['X1', 'X2', 'X3']
    
    plt.fill_between(steps, 0, current_pct, alpha=0.6, color=colors[0], label=labels[0])
    plt.fill_between(steps, current_pct, current_pct + promoted_pct, alpha=0.6, color=colors[1], label=labels[1])
    plt.fill_between(steps, current_pct + promoted_pct, 100, alpha=0.6, color=colors[2], label=labels[2])
    
    plt.xlabel('Training Steps', fontsize=20)
    plt.ylabel('Data Source Proportion (%)', fontsize=20)
    plt.title('DeepScalerR Train Set', fontsize=20)
    # plt.title('CD-34 Train Set', fontsize=20)
    # plt.title('Geo-3K Train Set', fontsize=20)
    plt.ylim(0, 100)

    plt.xlim(0, 471)
    # plt.xlim(0, 117)
    # plt.xlim(0, 300)
    
    plt.legend(fontsize=20, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    plt.tight_layout()
    plt.savefig('normalized_plot.png', dpi=300)
    plt.show()

def main():
    """Main function"""
    project = "off_policy_grpo" 
    # project = "off_policy_grpo_debug" 
    
    run_id = 'run_20250901_a72d9cc6'
    # run_id = "run_20250918_7171310c"
    # run_id = 'run_20250831_c2095fe5'
    print("Getting data...")
    df = get_data_sources(project, run_id)
    
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    
    # Check if we have the required columns
    required = ['actor/samples_current', 'actor/samples_promoted', 'actor/samples_target']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"Missing columns: {missing}")
        return
    
    print("Creating plots...")

    create_normalized_plot(df)
    print("Done!")

if __name__ == "__main__":
    main()