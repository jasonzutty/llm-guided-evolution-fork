import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("analysis/results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

failure_file = OUTPUT_DIR / "failure_counts.csv"
if failure_file.exists():
    df = pd.read_csv(failure_file)
    df.plot(kind="bar", x="failure_reason", y="count", title="Failure Breakdown")
    plt.savefig(OUTPUT_DIR / "failure_breakdown.png")
    plt.clf()

slurm_file = OUTPUT_DIR / "slurm_summary.csv"
if slurm_file.exists():
    df = pd.read_csv(slurm_file)
    if "walltime_used" in df.columns and "memory_used_kb" in df.columns:
        plt.scatter(df["walltime_used"], df["memory_used_kb"])
        plt.xlabel("Walltime")
        plt.ylabel("Memory")
        plt.title("Runtime vs Memory")
        plt.savefig(OUTPUT_DIR / "runtime_vs_memory.png")
        plt.clf()

entity_file = OUTPUT_DIR / "entity_metrics_summary.csv"
if entity_file.exists():
    df = pd.read_csv(entity_file)
    if "fitness_1" in df.columns and "fitness_2" in df.columns:
        df = df.dropna(subset=["fitness_1", "fitness_2"])
        plt.scatter(df["fitness_2"], df["fitness_1"])
        plt.xlabel("fitness_2")
        plt.ylabel("fitness_1")
        plt.title("Pareto Scatter")
        plt.savefig(OUTPUT_DIR / "pareto_plot.png")
        plt.clf()

print("Saved visualization outputs to results/analysis/")