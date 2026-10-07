import pandas as pd
import numpy as np
import os

os.environ.setdefault("MPLCONFIGDIR", "analysis/results/analysis/matplotlib-cache")
import matplotlib.pyplot as plt

def generate_fitness_quality_chart(csv_path, output_path="results/analysis/fitness_quality_summary.png"):
    # Load the CSV
    df = pd.read_csv(csv_path)
    if 'fitness_1' not in df.columns:
        print("Missing column: fitness_1")
        return
    df['fitness_1'] = pd.to_numeric(df['fitness_1'], errors='coerce')
    
    # We will evaluate based on fitness_1
    # Missing / Malformed
    missing_count = df['fitness_1'].isna().sum()
    
    # Infinite Fitness
    # Drop missing values first to check for infinity without errors
    valid_fitness = df['fitness_1'].dropna()
    infinite_count = np.isinf(valid_fitness).sum()
    
    # Finite Fitness
    finite_count = (~np.isinf(valid_fitness)).sum()
    
    # Data for the pie chart
    labels = ['Finite Fitness', 'Infinite Fitness', 'Missing/Malformed']
    sizes = [finite_count, infinite_count, missing_count]
    
    # Green for good (finite), Orange for infinite, Red for missing
    colors = ['#55A868', '#DD8452', '#C44E52'] 
    
    # Filter out categories with 0 count to make the chart cleaner
    labels_filtered = [l for l, s in zip(labels, sizes) if s > 0]
    sizes_filtered = [s for s in sizes if s > 0]
    colors_filtered = [c for c, s in zip(colors, sizes) if s > 0]

    # Create the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sizes_filtered, labels=labels_filtered, colors=colors_filtered, 
            autopct='%1.1f%%', startangle=140, 
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    
    # Add title
    plt.title('Entity Fitness Data Quality', fontsize=14, fontweight='bold')
    
    # Add absolute counts as a text box
    total = sum(sizes)
    plt.text(-1.5, -1.2, f"Total Entities: {total}\nFinite: {finite_count}\nInfinite: {infinite_count}\nMissing: {missing_count}", 
             fontsize=11, bbox=dict(facecolor='white', alpha=0.5))

    # Ensure the output directory exists
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
        default="analysis/results/analysis/fitness_quality_summary.png"
    )
    args = parser.parse_args()

    generate_fitness_quality_chart(args.input, args.output)
