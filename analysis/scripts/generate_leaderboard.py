import pandas as pd
import numpy as np
import os

def generate_leaderboard(csv_path, output_path="results/analysis/top_10_leaderboard.csv"):
    df = pd.read_csv(csv_path)

    df['fitness_1'] = pd.to_numeric(df['fitness_1'], errors='coerce')
    df['fitness_2'] = pd.to_numeric(df['fitness_2'], errors='coerce')

    df_finite = df.dropna(subset=['fitness_1', 'fitness_2'])
    df_finite = df_finite[
        ~np.isinf(df_finite['fitness_1']) &
        ~np.isinf(df_finite['fitness_2'])
    ]

    df_sorted = df_finite.sort_values(by=['fitness_1', 'fitness_2'], ascending=[False, False])

    leaderboard = df_sorted[['entity_id', 'fitness_1', 'fitness_2', 'entity_status']].copy()
    leaderboard.rename(columns={'entity_status': 'status'}, inplace=True)

    top_10 = leaderboard.head(10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    top_10.to_csv(output_path, index=False)
    print(f"Saved leaderboard CSV to {output_path}\n")

    print("--- Top Individuals Leaderboard Preview ---")
    print(top_10.to_string(index=False))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="analysis/results/analysis/entity_metrics_summary.csv"
    )
    parser.add_argument(
        "--output",
        default="analysis/results/analysis/top_10_leaderboard.csv"
    )
    args = parser.parse_args()

    generate_leaderboard(args.input, args.output)
