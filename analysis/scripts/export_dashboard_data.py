# 006_export_dashboard_data.py
# Read-only analysis script that prepares cleaned, dashboard-ready tables from
# intermediate analysis outputs and writes them to results/analysis/.

from pathlib import Path
import argparse
import csv


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize(value: str) -> str:
    return (value or "").strip()


def build_dashboard_runs(slurm_rows: list[dict], comparison_rows: list[dict]) -> list[dict]:
    comparison_by_job = {normalize(r.get("job_id", "")): r for r in comparison_rows}
    out = []

    for row in slurm_rows:
        job_id = normalize(row.get("job_id", ""))
        comp = comparison_by_job.get(job_id, {})

        out.append({
            "job_id": job_id,
            "job_name": normalize(row.get("job_name", "")),
            "status": normalize(row.get("status_guess", "")),
            "failure_reason": normalize(row.get("failure_reason", "")),
            "partition": normalize(row.get("partition", "")),
            "node": normalize(row.get("node", "")),
            "walltime_used": normalize(row.get("walltime_used", "")),
            "memory_used_kb": normalize(row.get("memory_used_kb", "")),
            "score": normalize(comp.get("score", "")),
            "fitness": normalize(comp.get("fitness", "")),
            "accuracy": normalize(comp.get("accuracy", "")),
            "loss": normalize(comp.get("loss", "")),
            "best_score": normalize(comp.get("best_score", "")),
        })

    return out


def build_dashboard_failures(failure_rows: list[dict]) -> list[dict]:
    out = []
    for row in failure_rows:
        out.append({
            "group_type": normalize(row.get("group_type", "")),
            "group_value": normalize(row.get("group_value", "")),
            "count": normalize(row.get("count", "")),
        })
    return out


def build_dashboard_metrics(metric_rows: list[dict]) -> list[dict]:
    out = []
    for row in metric_rows:
        out.append({
            "job_id": normalize(row.get("job_id", "")),
            "metric_name": normalize(row.get("metric_name", "")),
            "metric_value": normalize(row.get("metric_value", "")),
            "source_file": normalize(row.get("source_file", "")),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Export dashboard-ready data tables.")
    parser.add_argument("--slurm-input", default="analysis/results/analysis/slurm_summary.csv")
    parser.add_argument("--comparison-input", default="analysis/results/analysis/run_comparison.csv")
    parser.add_argument("--failure-input", default="analysis/results/analysis/failure_counts.csv")
    parser.add_argument("--metrics-input", default="analysis/results/analysis/run_metrics.csv")
    parser.add_argument("--output-dir", default="analysis/results/analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slurm_rows = read_csv_rows(Path(args.slurm_input))
    comparison_rows = read_csv_rows(Path(args.comparison_input))
    failure_rows = read_csv_rows(Path(args.failure_input))
    metric_rows = read_csv_rows(Path(args.metrics_input))

    dashboard_runs = build_dashboard_runs(slurm_rows, comparison_rows)
    dashboard_failures = build_dashboard_failures(failure_rows)
    dashboard_metrics = build_dashboard_metrics(metric_rows)

    write_csv(
        output_dir / "dashboard_runs.csv",
        ["job_id", "job_name", "status", "failure_reason", "partition", "node", "walltime_used", "memory_used_kb", "score", "fitness", "accuracy", "loss", "best_score"],
        dashboard_runs,
    )
    write_csv(
        output_dir / "dashboard_failures.csv",
        ["group_type", "group_value", "count"],
        dashboard_failures,
    )
    write_csv(
        output_dir / "dashboard_metrics.csv",
        ["job_id", "metric_name", "metric_value", "source_file"],
        dashboard_metrics,
    )

    print(f"Wrote dashboard tables to {output_dir}")


if __name__ == "__main__":
    main()