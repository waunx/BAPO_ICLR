import wandb
import pandas as pd
import numpy as np
from collections import defaultdict

def get_run_data_simple(project, run_id):
    """Get essential rollout data from a run"""
    print(f"Getting data for: {run_id}")
    print("-" * 40)
    
    api = wandb.TrackingApi()
    run = api.run(project=project, run_id=run_id)
    
    # Get history - handle defaultdict case
    try:
        history = run.history()
        if isinstance(history, defaultdict):
            data_dict = dict(history)
            df = pd.DataFrame(data_dict)
        else:
            df = history
    except Exception as e:
        print(f"Error getting history: {e}")
        return None
    
    print(f"Got {len(df)} rows of data")
    
    # Extract the key metrics we need
    required_cols = ['step', 'train/num_gen_batches']
    optional_cols = ['actor/samples_promoted', 'off_policy/promotion_rate']
    
    # Check what we have
    available_cols = []
    for col in required_cols:
        if col in df.columns:
            available_cols.append(col)
        else:
            print(f"Missing required column: {col}")
            return None
    
    for col in optional_cols:
        if col in df.columns:
            available_cols.append(col)
            print(f"Found optional column: {col}")
        else:
            print(f"Missing optional column: {col}")
    
    # Return subset of data
    result_df = df[available_cols].dropna(subset=['step']).copy()
    
    # Sort by step
    result_df = result_df.sort_values('step').reset_index(drop=True)
    
    print(f"Returning {len(result_df)} rows with columns: {list(result_df.columns)}")
    return result_df

def calculate_rollouts_simple(df):
    """Calculate rollouts from the data"""
    
    # Check if we have off-policy data
    has_off_policy = ('actor/samples_promoted' in df.columns and 
                     'off_policy/promotion_rate' in df.columns)
    
    # Base rollouts: sum(train/num_gen_batches) × 256 × 8
    base_sum = df['train/num_gen_batches'].sum()
    base_rollouts = base_sum * 256 * 8
    if has_off_policy:
        base_sum = df['train/num_gen_batches'].sum() / 2
        base_rollouts = base_sum * 256 * 8 
    # Off-policy rollouts: sum(actor/samples_promoted / off_policy/promotion_rate)
    total_off_policy_rollouts = 0
    
    if has_off_policy:
        # Calculate: sum(actor/samples_promoted / off_policy/promotion_rate)
        # Handle division by zero
        promoted = df['actor/samples_promoted'].fillna(0)
        rate = df['off_policy/promotion_rate'].fillna(0)
        
        # Only calculate where rate > 0
        valid_mask = rate > 0
        if valid_mask.any():
            off_policy_per_step = promoted / rate
            off_policy_per_step = off_policy_per_step.fillna(0)
            total_off_policy_rollouts = off_policy_per_step.sum()
        
        print(f"Off-policy calculation:")
        print(f"  - Total promoted samples: {promoted.sum():,.0f}")
        print(f"  - Total promotion rate: {rate.sum():.4f}")
        print(f"  - Total off-policy rollouts: {total_off_policy_rollouts:.1f}")
    
    total_rollouts = base_rollouts + total_off_policy_rollouts
    
    print(f"Rollout calculation:")
    print(f"  - Base: sum(train/num_gen_batches) × 256 × 8 = {base_sum:.1f} × 2048 = {base_rollouts:,.0f}")
    print(f"  - Off-policy rollouts: {total_off_policy_rollouts:.1f}")
    print(f"  - Total rollouts: {total_rollouts:,.0f}")
    
    return {
        'base_sum': base_sum,
        'base_rollouts': base_rollouts,
        'off_policy_rollouts': total_off_policy_rollouts,
        'total_rollouts': total_rollouts,
        'has_off_policy': has_off_policy,
        'max_step': df['step'].max(),
        'num_steps': len(df)
    }

def compare_runs_simple(project, run_ids, run_names=None):
    """Simple comparison between runs"""
    if run_names is None:
        run_names = [f"Run_{i+1}" for i in range(len(run_ids))]
    
    print("SIMPLE ROLLOUT COMPARISON")
    print("=" * 60)
    
    results = {}
    
    # Get data for each run
    for i, run_id in enumerate(run_ids):
        print(f"\n{run_names[i]} ({run_id}):")
        df = get_run_data_simple(project, run_id)
        
        if df is not None:
            result = calculate_rollouts_simple(df)
            results[run_names[i]] = result
        else:
            print(f"Failed to get data for {run_names[i]}")
    
    if len(results) < 2:
        print("Need at least 2 successful runs to compare")
        return results
    
    # Find common step range for fair comparison
    min_max_step = min([r['max_step'] for r in results.values()])
    print(f"\nSTEP ALIGNMENT:")
    print(f"Will compare up to step {min_max_step} (minimum final step)")
    
    # Re-calculate for aligned steps
    aligned_results = {}
    for name, run_id in zip(run_names, run_ids):
        if name in results:
            print(f"\nRe-calculating {name} up to step {min_max_step}:")
            df = get_run_data_simple(project, run_id)
            if df is not None:
                # Filter to aligned steps
                df_aligned = df[df['step'] <= min_max_step].copy()
                aligned_result = calculate_rollouts_simple(df_aligned)
                aligned_results[name] = aligned_result
    
    # Print comparison
    print(f"\nCOMPARISON (aligned to step {min_max_step}):")
    print("=" * 90)
    print(f"{'Method':<15} {'Base Sum':<10} {'Base Rollouts':<15} {'Off-Policy':<12} {'Total Rollouts':<15} {'Per Step':<12} {'Type':<12}")
    print("-" * 90)
    
    efficiency_data = []
    for name, data in aligned_results.items():
        method_type = "Off-Policy" if data['has_off_policy'] else "On-Policy"
        per_step = data['total_rollouts'] / min_max_step
        
        print(f"{name:<15} {data['base_sum']:<10.1f} {data['base_rollouts']:<15,.0f} {data['off_policy_rollouts']:<12.1f} "
              f"{data['total_rollouts']:<15,.0f} {per_step:<12,.0f} {method_type:<12}")
        
        efficiency_data.append({
            'name': name,
            'total_rollouts': data['total_rollouts'],
            'per_step': per_step,
            'type': method_type
        })
    
    # Efficiency analysis
    print(f"\nEFFICIENCY ANALYSIS:")
    print("=" * 50)
    
    efficiency_data.sort(key=lambda x: x['per_step'])
    
    for i, data in enumerate(efficiency_data):
        rank = "Most Efficient" if i == 0 else f"Rank {i+1}"
        print(f"{data['name']:<15} {data['per_step']:<12,.0f} rollouts/step ({rank}, {data['type']})")
    
    if len(efficiency_data) >= 2:
        best = efficiency_data[0]
        worst = efficiency_data[-1]
        ratio = worst['per_step'] / best['per_step']
        diff = worst['total_rollouts'] - best['total_rollouts']
        
        print(f"\nSummary:")
        print(f"- {worst['name']} needs {ratio:.2f}x more rollouts per step than {best['name']}")
        print(f"- To reach step {min_max_step}: {worst['name']} used {diff:,.0f} more total rollouts")
    
    return aligned_results

if __name__ == "__main__":
    project = "off_policy_grpo_debug"
    # run_ids = ["run_20250916_dccb4a7c", "run_20250915_60878ba6"]
    run_ids = ["run_20250918_7171310c", "run_20250919_8a89b775"]
    run_names = ["BAPO", "DAPO"]
    
    results = compare_runs_simple(project, run_ids, run_names)