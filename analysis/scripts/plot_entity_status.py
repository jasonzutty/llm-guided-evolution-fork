import pandas as pd
import os

os.environ.setdefault("MPLCONFIGDIR", "analysis/results/analysis/matplotlib-cache")
import matplotlib.pyplot as plt

def generate_entity_status_chart(csv_path, output_path="results/analysis/entity_status_summary.png"):
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Fill any missing statuses with 'missing/other' to match the dashboard plan
    status_counts = df['entity_status'].fillna('missing/other').value_counts()

    # Create the bar chart
    plt.figure(figsize=(8, 6))
    
    # Use standard colors for categories
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    bars = plt.bar(status_counts.index, status_counts.values, color=colors[:len(status_counts)])

    # Add titles and labels
    plt.title('Entity Status Summary', fontsize=14, fontweight='bold')
    plt.xlabel('Entity Status', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Clean up the y-axis (we only want integers for counts)
    plt.gca().yaxis.get_major_locator().set_params(integer=True)

    # Add numeric labels to the top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, int(yval), 
                 ha='center', va='bottom', fontsize=11)

    # Remove top and right borders for a cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Ensure the output directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Saved visualization to {output_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="analysis/results/analysis/entity_metrics_summary.csv"
    )
    parser.add_argument(
        "--output",
        default="analysis/results/analysis/entity_status_summary.png"
    )
    args = parser.parse_args()

    generate_entity_status_chart(args.input, args.output)
