"""Generate per-generation Pareto front plots and turn them into an animated GIF.

This script is based on create_pareto_graphs.py but works generation-by-generation
from `global_gen_*.pkl` files.
"""

import glob
import math
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INVALID_FITNESS = {
    (-math.inf, math.inf, math.inf),
    (-9999999999, 9999999999, 9999999999),
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "sota", "Surrogate", "results")
GIF_DURATION_MS = 500
CREATE_GIF = True
METRICS = ["Kendall_Tau", "MSE", "Runtime"]

KT_MIN = 0.0
KT_MAX = 1.0
MSE_MAX = 100.0
RUNTIME_MAX = 1000.0


def is_valid_fitness(fitness):
    if fitness is None:
        return False
    if not hasattr(fitness, "__iter__"):
        return False
    if len(fitness) < 2:
        return False
    if fitness in INVALID_FITNESS:
        return False
    if any(math.isinf(v) or math.isnan(v) for v in fitness):
        return False
    return True


def extract_fitness_values(dataset):
    values = []
    for attrs in dataset.values():
        fitness = attrs.get("fitness")
        if is_valid_fitness(fitness):
            values.append(
                {
                    "Kendall_Tau": float(fitness[0]),
                    "MSE": float(fitness[1]),
                    "Runtime": float(fitness[2]) if len(fitness) > 2 else 0.0,
                }
            )
    return values


def discover_generations(results_dir):
    pattern = os.path.join(results_dir, "global_gen_*.pkl")
    paths = glob.glob(pattern)
    generations = []
    for path in paths:
        match = re.search(r"global_gen_(\d+)\.pkl$", path)
        if match:
            generations.append((int(match.group(1)), path))
    return sorted(generations, key=lambda item: item[0])


def resolve_results_dir():
    if discover_generations(DEFAULT_RESULTS_DIR):
        return DEFAULT_RESULTS_DIR

    pattern = os.path.join(SCRIPT_DIR, "**", "global_gen_*.pkl")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return DEFAULT_RESULTS_DIR

    parent_dirs = {}
    for path in matches:
        parent_dir = os.path.dirname(path)
        parent_dirs[parent_dir] = parent_dirs.get(parent_dir, 0) + 1

    return max(parent_dirs.items(), key=lambda item: item[1])[0]


def get_pareto_front_2d(x_values, y_values, x_max=True, y_max=False):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    if x_values.shape != y_values.shape:
        raise ValueError("x_values and y_values must have the same shape")

    n = len(x_values)
    is_optimal = np.ones(n, dtype=bool)

    x_score = x_values if x_max else -x_values
    y_score = y_values if y_max else -y_values

    for i in range(n):
        dominates_i = (
            (x_score >= x_score[i])
            & (y_score >= y_score[i])
            & ((x_score > x_score[i]) | (y_score > y_score[i]))
        )
        dominates_i[i] = False
        if np.any(dominates_i):
            is_optimal[i] = False

    return is_optimal


def _axis_limit(values, lower=0.0, upper_default=1.0, pad_fraction=0.08):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return lower, upper_default

    upper = max(float(np.max(values)), upper_default)
    if upper <= lower:
        upper = lower + upper_default

    pad = (upper - lower) * pad_fraction
    return lower, upper + pad


def _plot_front(
    df,
    output_dir,
    x_col,
    y_col,
    x_max,
    y_max,
    filename,
    title,
    x_label,
    y_label,
    color,
    trivial_points=None,
    x_limit=None,
    y_limit=None,
    step_where="post",
):
    if df.empty:
        print(f"Skipping {filename}: no data")
        return

    plot_df = df[[x_col, y_col]].dropna().copy()
    if plot_df.empty:
        print(f"Skipping {filename}: no finite {x_col}/{y_col} data")
        return

    mask = get_pareto_front_2d(
        plot_df[x_col].values,
        plot_df[y_col].values,
        x_max=x_max,
        y_max=y_max,
    )
    pareto_df = plot_df[mask].copy().sort_values(by=x_col)

    plt.figure(figsize=(10, 6))

    if x_limit is None:
        x_limit = _axis_limit(plot_df[x_col].values, lower=0.0, upper_default=1.0)
    if y_limit is None:
        y_limit = _axis_limit(plot_df[y_col].values, lower=0.0, upper_default=1.0)

    plt.xlim(*x_limit)
    plt.ylim(*y_limit)

    plt.scatter(
        plot_df[x_col],
        plot_df[y_col],
        c=color,
        alpha=0.25,
        s=30,
        label="All Experimental Models",
        zorder=1,
    )

    if trivial_points:
        triv_df = pd.DataFrame(trivial_points, columns=[x_col, y_col])
        line_df = pd.concat([pareto_df[[x_col, y_col]], triv_df], ignore_index=True)
        line_mask = get_pareto_front_2d(
            line_df[x_col].values,
            line_df[y_col].values,
            x_max=x_max,
            y_max=y_max,
        )
        line_df = line_df[line_mask].sort_values(by=x_col)
    else:
        line_df = pareto_df[[x_col, y_col]].copy().sort_values(by=x_col)

    if len(line_df) > 0:
        plt.step(
            line_df[x_col],
            line_df[y_col],
            where=step_where,
            color=color,
            lw=2.5,
            label="Experimental Pareto Front",
            zorder=3,
        )

    if len(pareto_df) > 0:
        plt.scatter(
            pareto_df[x_col],
            pareto_df[y_col],
            color=color,
            edgecolor="black",
            s=70,
            label="Pareto Optimal Points",
            zorder=4,
        )

    if trivial_points:
        t_x, t_y = zip(*trivial_points)
        plt.scatter(
            t_x,
            t_y,
            color="black",
            marker="x",
            s=120,
            linewidths=2,
            label="Reference Anchors",
            zorder=5,
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="best", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_pareto_kt_mse(df, output_dir, filename, title, trivial_points=None):
    _plot_front(
        df=df,
        output_dir=output_dir,
        x_col="Kendall_Tau",
        y_col="MSE",
        x_max=True,
        y_max=False,
        filename=filename,
        title=title,
        x_label="Kendall Tau (Higher is Better)",
        y_label="MSE (Lower is Better)",
        color="crimson",
        trivial_points=trivial_points,
        x_limit=(KT_MIN, KT_MAX),
        y_limit=_axis_limit(df["MSE"].values, lower=0.0, upper_default=MSE_MAX),
        step_where="pre",
    )


def plot_pareto_kt_runtime(df, output_dir, filename, title, trivial_points=None):
    _plot_front(
        df=df,
        output_dir=output_dir,
        x_col="Kendall_Tau",
        y_col="Runtime",
        x_max=True,
        y_max=False,
        filename=filename,
        title=title,
        x_label="Kendall Tau (Higher is Better)",
        y_label="Runtime in Seconds (Lower is Better)",
        color="darkorange",
        trivial_points=trivial_points,
        x_limit=(KT_MIN, KT_MAX),
        y_limit=_axis_limit(df["Runtime"].values, lower=0.0, upper_default=RUNTIME_MAX),
        step_where="pre",
    )


def plot_pareto_mse_runtime(df, output_dir, filename, title, trivial_points=None):
    _plot_front(
        df=df,
        output_dir=output_dir,
        x_col="MSE",
        y_col="Runtime",
        x_max=False,
        y_max=False,
        filename=filename,
        title=title,
        x_label="MSE (Lower is Better)",
        y_label="Runtime in Seconds (Lower is Better)",
        color="darkviolet",
        trivial_points=trivial_points,
        x_limit=_axis_limit(df["MSE"].values, lower=0.0, upper_default=MSE_MAX),
        y_limit=_axis_limit(df["Runtime"].values, lower=0.0, upper_default=RUNTIME_MAX),
        step_where="post",
    )


def build_generation_dataframe(current_points, hist_points):
    all_points = current_points + hist_points
    if not all_points:
        return pd.DataFrame(columns=METRICS)
    return pd.DataFrame(all_points, columns=METRICS)


def make_gif(image_dir, output_path, duration=500):
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed. Install it to generate GIFs: pip install Pillow")
        return

    stem = os.path.splitext(os.path.basename(output_path))[0]
    pattern = os.path.join(image_dir, f"{stem}_gen_*.png")
    frames = sorted(
        glob.glob(pattern),
        key=lambda path: int(re.search(r"_gen_(\d+)", os.path.basename(path)).group(1)),
    )
    if not frames:
        print("No generation image files found for GIF creation.")
        return

    images = [Image.open(path) for path in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )
    print(f"GIF created: {output_path} ({len(images)} frames)")


def main():
    results_dir = resolve_results_dir()
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    generations = discover_generations(results_dir)
    if not generations:
        print(f"No generation files found in: {results_dir}")
        return

    print(f"Discovered {len(generations)} generation(s) in {results_dir}")

    for gen, path in generations:
        with open(path, "rb") as handle:
            data = pickle.load(handle)

        current = extract_fitness_values(data.get("GLOBAL_DATA", {}))
        history = extract_fitness_values(data.get("GLOBAL_DATA_HIST", {}))
        df = build_generation_dataframe(current, history)

        if df.empty:
            print(f"Skipping generation {gen}: no valid fitness points")
            continue

        plot_pareto_kt_mse(
            df=df,
            output_dir=output_dir,
            filename=f"pareto_kt_vs_mse_gen_{gen:04d}.png",
            title=f"Pareto Front: Kendall Tau vs. MSE (Generation {gen})",
            trivial_points=[(1.0, 100.0), (0.0, 0.0)],
        )
        plot_pareto_kt_runtime(
            df=df,
            output_dir=output_dir,
            filename=f"pareto_kt_vs_runtime_gen_{gen:04d}.png",
            title=f"Pareto Front: Kendall Tau vs. Runtime (Generation {gen})",
            trivial_points=[(0.0, 0.0), (1.0, 500.0)],
        )
        plot_pareto_mse_runtime(
            df=df,
            output_dir=output_dir,
            filename=f"pareto_mse_vs_runtime_gen_{gen:04d}.png",
            title=f"Pareto Front: MSE vs. Runtime (Generation {gen})",
            trivial_points=[(100.0, 0.0), (0.0, 500.0)],
        )

    if CREATE_GIF:
        make_gif(output_dir, os.path.join(output_dir, "pareto_kt_vs_mse.gif"), duration=GIF_DURATION_MS)
        make_gif(output_dir, os.path.join(output_dir, "pareto_kt_vs_runtime.gif"), duration=GIF_DURATION_MS)
        make_gif(output_dir, os.path.join(output_dir, "pareto_mse_vs_runtime.gif"), duration=GIF_DURATION_MS)

    print(f"Finished. Images and GIF saved in: {output_dir}")


if __name__ == "__main__":
    main()
