import pandas as pd
import numpy as np
import os

os.environ.setdefault("MPLCONFIGDIR", "analysis/results/analysis/matplotlib-cache")
import matplotlib.pyplot as plt

def generate_controller_activity_chart(csv_path, output_path="results/analysis/controller_activity_summary.png"):
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    for col in ['submitted_child_jobs', 'completed_child_jobs', 'wait_events']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    # Filter for jobs that had some controller activity (submitted > 0)
    df_controller = df[df['submitted_child_jobs'] > 0].copy()
    
    if df_controller.empty:
        print("No controller activity found in the provided slurm summary.")
        return
        
    # Prepare data
    labels = df_controller['job_name'] + "\n(" + df_controller['job_id'].astype(str) + ")"
    submitted = df_controller['submitted_child_jobs']
    completed = df_controller['completed_child_jobs']
    wait_events = df_controller['wait_events']
    
    x = np.arange(len(labels))
    width = 0.25
    
    plt.figure(figsize=(9, 6))
    
    # Plotting grouped bars
    bars1 = plt.bar(x - width, submitted, width, label='Submitted Child Jobs', color='#4C72B0')
    bars2 = plt.bar(x, completed, width, label='Completed Child Jobs', color='#55A868')
    bars3 = plt.bar(x + width, wait_events, width, label='Wait Events', color='#DD8452')
    
    # Add titles and labels
    plt.title('Controller Activity Summary', fontsize=14, fontweight='bold')
    plt.xlabel('Controller Job', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(x, labels, rotation=0)
    plt.legend()
    
    # Add values on top of bars
    max_val = df_controller[['submitted_child_jobs', 'completed_child_jobs', 'wait_events']].max().max()
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + (max_val * 0.01), int(yval), 
                     ha='center', va='bottom', fontsize=10)

    # Clean up layout
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Saved visualization to {output_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="analysis/results/analysis/slurm_summary.csv"
    )
    parser.add_argument(
        "--output",
        default="analysis/results/analysis/controller_activity_summary.png"
    )
    args = parser.parse_args()

    generate_controller_activity_chart(args.input, args.output)
