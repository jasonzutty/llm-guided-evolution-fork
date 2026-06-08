from __future__ import annotations

import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "analysis"
PARAM_COUNT_PLOT_LIMIT = 200_000  # exclude extreme outliers from Pareto scatter view
DEFAULT_LOGS = [
    ROOT / "slurm-5142024.out",
    ROOT / "slurm-5149662.out",
]

GENERATION_RE = re.compile(r"STARTING GENERATION:\s+(\d+)")
FITNESS_RE = re.compile(
    r"Fitness:\s+\(([^,]+),\s*([^)]+)\), Submission Flag:\s+(True|False)"
)
RUNTIME_RE = re.compile(r"Runtime:\s+(\d+)\s+min,\s+Status:\s+(.+)$")
JOB_RE = re.compile(r"LLM Job-ID:\s+(.+)$")


@dataclass
class SnapshotRecord:
    generation: int
    gene_id: str
    reward: float
    param_count: float
    submission_flag: bool
    runtime_min: int | None
    status: str
    job_id: str
    source_log: str

    @property
    def is_valid(self) -> bool:
        return (
            math.isfinite(self.reward)
            and math.isfinite(self.param_count)
            and self.reward > -1e8
            and self.param_count < 1e8  # excludes 999999999 sentinel
        )

    @property
    def failure_kind(self) -> str:
        if self.is_valid:
            return "valid"
        if not self.submission_flag:
            return "submission_failed"
        if not math.isfinite(self.reward) or not math.isfinite(self.param_count):
            return "runtime_invalid"
        if self.reward <= -1e8 or self.param_count >= 1e8:
            return "hard_failure"
        return "other_failure"


def parse_float(text: str) -> float:
    text = text.strip()
    if text == "inf":
        return math.inf
    if text == "-inf":
        return -math.inf
    return float(text)


def parse_generation_snapshots(log_path: Path) -> list[SnapshotRecord]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    records: list[SnapshotRecord] = []
    i = 0
    while i < len(lines):
        gen_match = GENERATION_RE.search(lines[i])
        if not gen_match:
            i += 1
            continue

        generation = int(gen_match.group(1))
        i += 1
        while i < len(lines) and "Poplutation Info" not in lines[i]:
            i += 1
        if i >= len(lines):
            break

        i += 1
        while i < len(lines):
            line = lines[i]
            if "Invalid Removal" in line:
                break
            if line.startswith("Gene: "):
                gene_id = line.split("Gene: ", 1)[1].strip()
                fitness_line = lines[i + 1] if i + 1 < len(lines) else ""
                runtime_line = lines[i + 2] if i + 2 < len(lines) else ""
                job_line = lines[i + 3] if i + 3 < len(lines) else ""

                fitness_match = FITNESS_RE.search(fitness_line)
                runtime_match = RUNTIME_RE.search(runtime_line)
                job_match = JOB_RE.search(job_line)

                if fitness_match:
                    reward = parse_float(fitness_match.group(1))
                    param_count = parse_float(fitness_match.group(2))
                    submission_flag = fitness_match.group(3) == "True"
                    runtime_min = int(runtime_match.group(1)) if runtime_match else None
                    status = runtime_match.group(2) if runtime_match else "unknown"
                    job_id = job_match.group(1).strip() if job_match else "unknown"
                    records.append(
                        SnapshotRecord(
                            generation=generation,
                            gene_id=gene_id,
                            reward=reward,
                            param_count=param_count,
                            submission_flag=submission_flag,
                            runtime_min=runtime_min,
                            status=status,
                            job_id=job_id,
                            source_log=log_path.name,
                        )
                    )
                    i += 4
                    continue
            i += 1
    return records


def dedupe_individuals(records: list[SnapshotRecord]) -> list[dict]:
    by_gene: dict[str, dict] = {}
    for rec in records:
        existing = by_gene.get(rec.gene_id)
        if existing is None:
            by_gene[rec.gene_id] = {
                "gene_id": rec.gene_id,
                "first_generation": rec.generation,
                "last_generation": rec.generation,
                "appearances": 1,
                "reward": rec.reward,
                "param_count": rec.param_count,
                "runtime_min": rec.runtime_min,
                "submission_flag": rec.submission_flag,
                "status": rec.status,
                "job_id": rec.job_id,
                "source_log": rec.source_log,
                "is_valid": rec.is_valid,
                "failure_kind": rec.failure_kind,
            }
        else:
            existing["last_generation"] = rec.generation
            existing["appearances"] += 1
    return list(by_gene.values())


def is_dominated(candidate: dict, others: list[dict]) -> bool:
    """Return True if any other solution dominates candidate.
    Dominance: maximize reward, minimize param_count.
    """
    for other in others:
        if other["gene_id"] == candidate["gene_id"]:
            continue
        dominates = (
            other["reward"] >= candidate["reward"]
            and other["param_count"] <= candidate["param_count"]
            and (
                other["reward"] > candidate["reward"]
                or other["param_count"] < candidate["param_count"]
            )
        )
        if dominates:
            return True
    return False


def pareto_front(records: list[dict]) -> list[dict]:
    valid = [rec for rec in records if rec["is_valid"]]
    return [rec for rec in valid if not is_dominated(rec, valid)]


def standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0.0] = 1.0
    return (features - mean) / std


def kmeans(features: np.ndarray, k: int, seed: int = 7, steps: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(features), size=k, replace=False)
    centroids = features[idx].copy()

    for _ in range(steps):
        distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        new_centroids = centroids.copy()
        for cluster_idx in range(k):
            members = features[labels == cluster_idx]
            if len(members) == 0:
                new_centroids[cluster_idx] = features[rng.integers(0, len(features))]
            else:
                new_centroids[cluster_idx] = members.mean(axis=0)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
    labels = distances.argmin(axis=1)
    scores = centroids.sum(axis=1)
    order = np.argsort(scores)
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[label] for label in labels])


def cluster_labels(snapshot_records: list[SnapshotRecord]) -> tuple[np.ndarray, list[str]]:
    valid = [rec for rec in snapshot_records if rec.is_valid]
    if not valid:
        return np.array([]), []

    features = np.array(
        [[rec.reward, -rec.param_count] for rec in valid],
        dtype=float,
    )
    features = standardize(features)
    k = min(4, len(valid))
    labels = kmeans(features, k=k)
    names_full = ["Lower", "Developing", "Strong", "Elite"]
    names = names_full[-k:]
    return labels, names


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_generation_summary(records: list[SnapshotRecord]) -> list[dict]:
    by_generation: dict[int, list[SnapshotRecord]] = {}
    for rec in records:
        by_generation.setdefault(rec.generation, []).append(rec)

    summary: list[dict] = []
    for generation in sorted(by_generation):
        bucket = by_generation[generation]
        valid = [rec for rec in bucket if rec.is_valid]
        summary.append(
            {
                "generation": generation,
                "population_size": len(bucket),
                "valid_count": len(valid),
                "invalid_count": len(bucket) - len(valid),
                "mean_reward_valid": (
                    round(float(np.mean([rec.reward for rec in valid])), 6) if valid else ""
                ),
                "max_reward_valid": (
                    round(float(np.max([rec.reward for rec in valid])), 6) if valid else ""
                ),
                "min_param_count_valid": (
                    round(float(np.min([rec.param_count for rec in valid])), 0) if valid else ""
                ),
                "max_param_count_valid": (
                    round(float(np.max([rec.param_count for rec in valid])), 0) if valid else ""
                ),
            }
        )
    return summary


def build_markdown_report(
    logs: list[Path],
    snapshot_records: list[SnapshotRecord],
    unique_records: list[dict],
    frontier: list[dict],
    generation_summary: list[dict],
) -> str:
    valid_unique = [rec for rec in unique_records if rec["is_valid"]]
    invalid_unique = [rec for rec in unique_records if not rec["is_valid"]]
    top_reward = sorted(valid_unique, key=lambda row: row["reward"], reverse=True)[:15]
    top_efficient = sorted(
        valid_unique, key=lambda row: (-row["reward"], row["param_count"])
    )[:15]

    lines: list[str] = []
    lines.append("# Evolution Run Analysis")
    lines.append("")
    lines.append("## Files Analyzed")
    for log in logs:
        lines.append(f"- `{log.name}`")
    lines.append("")
    lines.append("## Run Summary")
    lines.append(f"- Generations covered: `{generation_summary[0]['generation']}` to `{generation_summary[-1]['generation']}`")
    lines.append(f"- Population snapshots parsed: `{len(snapshot_records)}`")
    lines.append(f"- Unique individuals observed: `{len(unique_records)}`")
    lines.append(f"- Unique valid individuals: `{len(valid_unique)}`")
    lines.append(f"- Unique failed / invalid individuals: `{len(invalid_unique)}`")
    lines.append("")
    lines.append("## Top Individuals By Reward")
    lines.append("")
    lines.append("| Rank | Gene ID | First Gen | Reward | Param Count | Appearances |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for idx, row in enumerate(top_reward, start=1):
        lines.append(
            f"| {idx} | `{row['gene_id']}` | {row['first_generation']} | "
            f"{row['reward']:.4f} | {int(row['param_count'])} | {row['appearances']} |"
        )
    lines.append("")
    lines.append("## Top Individuals By Efficiency (High Reward, Low Params)")
    lines.append("")
    lines.append("| Rank | Gene ID | First Gen | Reward | Param Count |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for idx, row in enumerate(top_efficient, start=1):
        lines.append(
            f"| {idx} | `{row['gene_id']}` | {row['first_generation']} | "
            f"{row['reward']:.4f} | {int(row['param_count'])} |"
        )
    lines.append("")
    lines.append("## Overall Pareto Frontier (Maximize Reward, Minimize Param Count)")
    lines.append("")
    lines.append("| Gene ID | First Gen | Reward | Param Count |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in sorted(frontier, key=lambda item: item["reward"], reverse=True):
        lines.append(
            f"| `{row['gene_id']}` | {row['first_generation']} | "
            f"{row['reward']:.4f} | {int(row['param_count'])} |"
        )
    lines.append("")
    lines.append("## Generation Trend")
    lines.append("")
    lines.append("| Gen | Pop Size | Valid | Invalid | Mean Reward | Max Reward | Min Params | Max Params |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in generation_summary:
        lines.append(
            f"| {row['generation']} | {row['population_size']} | {row['valid_count']} | {row['invalid_count']} | "
            f"{row['mean_reward_valid']} | {row['max_reward_valid']} | "
            f"{row['min_param_count_valid']} | {row['max_param_count_valid']} |"
        )
    return "\n".join(lines)


def prepare_plot_data(
    snapshot_records: list[SnapshotRecord],
    unique_records: list[dict],
):
    valid_unique = [rec for rec in unique_records if rec["is_valid"]]
    param_counts = np.array([rec["param_count"] for rec in valid_unique], dtype=float)
    rewards = np.array([rec["reward"] for rec in valid_unique], dtype=float)
    generations = np.array([rec["first_generation"] for rec in valid_unique], dtype=float)
    # Larger dot = fewer params (more efficient)
    param_norm = (param_counts.max() - param_counts) / max(param_counts.max() - param_counts.min(), 1e-9)
    sizes = 15 + 50 * param_norm

    valid_snapshots = [rec for rec in snapshot_records if rec.is_valid]
    labels, cluster_names = cluster_labels(valid_snapshots)
    palette = ["#5E3C99", "#3288BD", "#66C2A5", "#E6AB02"]
    generation_values = sorted({rec.generation for rec in snapshot_records})
    cluster_color = {name: palette[idx] for idx, name in enumerate(cluster_names)}
    grouped_by_generation: dict[int, list[SnapshotRecord]] = {gen: [] for gen in generation_values}
    for rec in valid_snapshots:
        grouped_by_generation[rec.generation].append(rec)

    mean_rewards = []
    max_rewards = []
    for gen in generation_values:
        bucket = grouped_by_generation[gen]
        if bucket:
            mean_rewards.append(np.mean([rec.reward for rec in bucket]))
            max_rewards.append(np.max([rec.reward for rec in bucket]))
        else:
            mean_rewards.append(np.nan)
            max_rewards.append(np.nan)

    return {
        "valid_unique": valid_unique,
        "param_counts": param_counts,
        "rewards": rewards,
        "generations": generations,
        "sizes": sizes,
        "valid_snapshots": valid_snapshots,
        "labels": labels,
        "cluster_names": cluster_names,
        "cluster_color": cluster_color,
        "generation_values": generation_values,
        "mean_rewards": mean_rewards,
        "max_rewards": max_rewards,
    }


def _staircase_pareto(frontier: list[dict]) -> tuple[list[float], list[float]]:
    """Build staircase (step) x/y arrays for the Pareto frontier.

    Sorted by param_count ascending, the frontier is drawn as horizontal
    steps at each reward level with vertical jumps between them.
    """
    points = sorted(frontier, key=lambda row: row["param_count"])
    xs: list[float] = []
    ys: list[float] = []
    for i, pt in enumerate(points):
        if i > 0:
            # vertical jump: same x, previous y -> current y
            xs.append(pt["param_count"])
            ys.append(points[i - 1]["reward"])
        xs.append(pt["param_count"])
        ys.append(pt["reward"])
    return xs, ys


def plot_pareto_figure(
    unique_records: list[dict],
    frontier: list[dict],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    plot_data = prepare_plot_data([], unique_records)

    mask = plot_data["param_counts"] < PARAM_COUNT_PLOT_LIMIT
    scatter = ax.scatter(
        plot_data["param_counts"][mask],
        plot_data["rewards"][mask],
        c=plot_data["generations"][mask],
        s=plot_data["sizes"][mask],
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )

    step_xs, step_ys = _staircase_pareto(frontier)
    ax.plot(step_xs, step_ys, color="black", linewidth=2, label="Pareto frontier")

    frontier_points = sorted(frontier, key=lambda row: row["param_count"])
    ax.scatter(
        [row["param_count"] for row in frontier_points],
        [row["reward"] for row in frontier_points],
        color="gold",
        edgecolors="black",
        s=90,
        zorder=5,
    )

    for row in sorted(frontier_points, key=lambda item: item["reward"], reverse=True)[:6]:
        ax.annotate(
            row["gene_id"][:10],
            (row["param_count"], row["reward"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Pareto Frontier: Reward vs Parameter Count")
    ax.set_xlabel("Parameter count (minimize →)")
    ax.set_ylabel("Mean reward (maximize ↑)")
    ax.set_xlim(right=PARAM_COUNT_PLOT_LIMIT)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("First generation seen")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trend_figure(
    snapshot_records: list[SnapshotRecord],
    unique_records: list[dict],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax_trend = plt.subplots(figsize=(10, 8), constrained_layout=True)
    plot_data = prepare_plot_data(snapshot_records, unique_records)

    random.seed(7)
    for rec, label in zip(plot_data["valid_snapshots"], plot_data["labels"]):
        cluster_name = plot_data["cluster_names"][label]
        jitter = random.uniform(-0.22, 0.22)
        ax_trend.scatter(
            rec.generation + jitter,
            rec.reward,
            color=plot_data["cluster_color"][cluster_name],
            alpha=0.7,
            s=15,
            edgecolors="none",
        )
    ax_trend.plot(
        plot_data["generation_values"],
        plot_data["mean_rewards"],
        color="black",
        linewidth=2,
        label="Mean reward",
    )
    ax_trend.plot(
        plot_data["generation_values"],
        plot_data["max_rewards"],
        color="#D95F02",
        linewidth=2,
        linestyle="--",
        label="Best reward",
    )

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=name,
            markerfacecolor=plot_data["cluster_color"][name], markersize=8
        )
        for name in plot_data["cluster_names"]
    ]
    handles.extend(
        [
            plt.Line2D([0], [0], color="black", linewidth=2, label="Mean reward"),
            plt.Line2D([0], [0], color="#D95F02", linewidth=2, linestyle="--", label="Best reward"),
        ]
    )

    ax_trend.legend(handles=handles, loc="upper left")
    ax_trend.set_title("Generation Trend With Performance Clusters")
    ax_trend.set_xlabel("Generation")
    ax_trend.set_ylabel("Mean reward")
    ax_trend.set_xticks(plot_data["generation_values"])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_combined_figure(
    snapshot_records: list[SnapshotRecord],
    unique_records: list[dict],
    frontier: list[dict],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_pareto, ax_trend) = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    plot_data = prepare_plot_data(snapshot_records, unique_records)

    mask = plot_data["param_counts"] < PARAM_COUNT_PLOT_LIMIT
    scatter = ax_pareto.scatter(
        plot_data["param_counts"][mask],
        plot_data["rewards"][mask],
        c=plot_data["generations"][mask],
        s=plot_data["sizes"][mask],
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )

    step_xs, step_ys = _staircase_pareto(frontier)
    ax_pareto.plot(step_xs, step_ys, color="black", linewidth=2, label="Pareto frontier")

    frontier_points = sorted(frontier, key=lambda row: row["param_count"])
    ax_pareto.scatter(
        [row["param_count"] for row in frontier_points],
        [row["reward"] for row in frontier_points],
        color="gold",
        edgecolors="black",
        s=90,
        zorder=5,
    )

    for row in sorted(frontier_points, key=lambda item: item["reward"], reverse=True)[:6]:
        ax_pareto.annotate(
            row["gene_id"][:10],
            (row["param_count"], row["reward"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax_pareto.set_title("Pareto Frontier: Reward vs Parameter Count")
    ax_pareto.set_xlabel("Parameter count (minimize →)")
    ax_pareto.set_ylabel("Mean reward (maximize ↑)")
    ax_pareto.set_xlim(right=PARAM_COUNT_PLOT_LIMIT)
    ax_pareto.set_ylim(bottom=0)
    ax_pareto.legend(loc="lower right")
    fig.colorbar(scatter, ax=ax_pareto).set_label("First generation seen")

    random.seed(7)
    for rec, label in zip(plot_data["valid_snapshots"], plot_data["labels"]):
        cluster_name = plot_data["cluster_names"][label]
        jitter = random.uniform(-0.22, 0.22)
        ax_trend.scatter(
            rec.generation + jitter,
            rec.reward,
            color=plot_data["cluster_color"][cluster_name],
            alpha=0.7,
            s=15,
            edgecolors="none",
        )
    ax_trend.plot(
        plot_data["generation_values"],
        plot_data["mean_rewards"],
        color="black",
        linewidth=2,
        label="Mean reward",
    )
    ax_trend.plot(
        plot_data["generation_values"],
        plot_data["max_rewards"],
        color="#D95F02",
        linewidth=2,
        linestyle="--",
        label="Best reward",
    )

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=name,
            markerfacecolor=plot_data["cluster_color"][name], markersize=8
        )
        for name in plot_data["cluster_names"]
    ]
    handles.extend(
        [
            plt.Line2D([0], [0], color="black", linewidth=2, label="Mean reward"),
            plt.Line2D([0], [0], color="#D95F02", linewidth=2, linestyle="--", label="Best reward"),
        ]
    )
    ax_trend.legend(handles=handles, loc="upper left")
    ax_trend.set_title("Generation Trend With Performance Clusters")
    ax_trend.set_xlabel("Generation")
    ax_trend.set_ylabel("Mean reward")
    ax_trend.set_xticks(plot_data["generation_values"])

    fig.suptitle("Evolution Analysis", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logs = [path for path in DEFAULT_LOGS if path.exists()]
    records_by_generation: dict[int, list[SnapshotRecord]] = {}
    for log in logs:
        parsed = parse_generation_snapshots(log)
        grouped: dict[int, list[SnapshotRecord]] = {}
        for rec in parsed:
            grouped.setdefault(rec.generation, []).append(rec)
        # Later logs replace earlier generations, handling checkpoint resumes cleanly.
        records_by_generation.update(grouped)

    snapshot_records = [
        rec
        for generation in sorted(records_by_generation)
        for rec in records_by_generation[generation]
    ]

    unique_records = dedupe_individuals(snapshot_records)
    unique_records.sort(key=lambda row: (not row["is_valid"], -row["reward"] if row["is_valid"] else 0, row["gene_id"]))
    frontier = pareto_front(unique_records)
    generation_summary = build_generation_summary(snapshot_records)

    snapshot_csv_rows = [
        {
            "generation": rec.generation,
            "gene_id": rec.gene_id,
            "reward": rec.reward,
            "param_count": rec.param_count,
            "submission_flag": rec.submission_flag,
            "runtime_min": rec.runtime_min,
            "status": rec.status,
            "job_id": rec.job_id,
            "source_log": rec.source_log,
            "is_valid": rec.is_valid,
            "failure_kind": rec.failure_kind,
        }
        for rec in snapshot_records
    ]
    write_csv(
        OUTPUT_DIR / "evolution_population_snapshots.csv",
        snapshot_csv_rows,
        [
            "generation",
            "gene_id",
            "reward",
            "param_count",
            "submission_flag",
            "runtime_min",
            "status",
            "job_id",
            "source_log",
            "is_valid",
            "failure_kind",
        ],
    )
    write_csv(
        OUTPUT_DIR / "evolution_unique_individuals.csv",
        unique_records,
        [
            "gene_id",
            "first_generation",
            "last_generation",
            "appearances",
            "reward",
            "param_count",
            "runtime_min",
            "submission_flag",
            "status",
            "job_id",
            "source_log",
            "is_valid",
            "failure_kind",
        ],
    )
    write_csv(
        OUTPUT_DIR / "generation_summary.csv",
        generation_summary,
        [
            "generation",
            "population_size",
            "valid_count",
            "invalid_count",
            "mean_reward_valid",
            "max_reward_valid",
            "min_param_count_valid",
            "max_param_count_valid",
        ],
    )
    write_csv(
        OUTPUT_DIR / "pareto_frontier.csv",
        sorted(frontier, key=lambda row: row["reward"], reverse=True),
        [
            "gene_id",
            "first_generation",
            "last_generation",
            "appearances",
            "reward",
            "param_count",
            "runtime_min",
            "submission_flag",
            "status",
            "job_id",
            "source_log",
            "is_valid",
            "failure_kind",
        ],
    )

    report_text = build_markdown_report(logs, snapshot_records, unique_records, frontier, generation_summary)
    (OUTPUT_DIR / "evolution_report.md").write_text(report_text, encoding="utf-8")
    plot_pareto_figure(unique_records, frontier, OUTPUT_DIR / "evolution_pareto_plot.png")
    plot_trend_figure(snapshot_records, unique_records, OUTPUT_DIR / "evolution_trend_plot.png")
    plot_combined_figure(snapshot_records, unique_records, frontier, OUTPUT_DIR / "evolution_combined_plot.png")

    print(f"Analyzed {len(snapshot_records)} population snapshot rows across {len(logs)} log files.")
    print(f"Unique individuals: {len(unique_records)} total, {sum(1 for r in unique_records if r['is_valid'])} valid.")
    print(f"Pareto frontier size: {len(frontier)}")
    print(f"Wrote outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
