# compare_runs.py
# Read-only analysis script that combines slurm_summary.csv, run_inventory.csv,
# and run_metrics.csv into a run-level comparison table at
# results/analysis/run_comparison.csv.

from pathlib import Path
import argparse
import csv
from collections import defaultdict, Counter


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def normalize(value: str) -> str:
    return (value or "").strip()


def build_inventory_summary(rows: list[dict]) -> dict:
    summary = defaultdict(int)
    for row in rows:
        category = normalize(row.get("category", ""))
        if category:
            summary[category] += 1
        summary["inventory_total"] += 1
    return dict(summary)


def build_metrics_by_job(rows: list[dict]) -> dict[str, dict]:
    out = defaultdict(dict)
    entity_counts_by_job = defaultdict(set)
    mutate_type_counter_by_job = defaultdict(Counter)

    for row in rows:
        job_id = normalize(row.get("job_id", ""))
        entity_id = normalize(row.get("entity_id", ""))
        metric_name = normalize(row.get("metric_name", ""))
        metric_value = normalize(row.get("metric_value", ""))

        if metric_name == "mutate_type":
            if job_id:
                mutate_type_counter_by_job[job_id][metric_value] += 1
            continue

        if job_id and entity_id:
            entity_counts_by_job[job_id].add(entity_id)

        if not job_id or not metric_name or metric_value == "":
            continue

        # Keep last seen value for job-level comparison
        out[job_id][metric_name] = metric_value

    for job_id, entities in entity_counts_by_job.items():
        out[job_id]["entity_count"] = str(len(entities))

    for job_id, counter in mutate_type_counter_by_job.items():
        if counter:
            most_common_type, most_common_count = counter.most_common(1)[0]
            out[job_id]["most_common_mutate_type"] = most_common_type
            out[job_id]["most_common_mutate_type_count"] = str(most_common_count)

    return out


def main():
    parser = argparse.ArgumentParser(description="Compare runs using analysis outputs.")
    parser.add_argument("--slurm-input", default="analysis/results/analysis/slurm_summary.csv")
    parser.add_argument("--inventory-input", default="analysis/results/analysis/run_inventory.csv")
    parser.add_argument("--metrics-input", default="analysis/results/analysis/run_metrics.csv")
    parser.add_argument("--output", default="analysis/results/analysis/run_comparison.csv")
    args = parser.parse_args()

    slurm_rows = read_csv_rows(Path(args.slurm_input))
    inventory_rows = read_csv_rows(Path(args.inventory_input))
    metrics_rows = read_csv_rows(Path(args.metrics_input))
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    inventory_summary = build_inventory_summary(inventory_rows)
    metrics_by_job = build_metrics_by_job(metrics_rows)

    comparison_rows = []
    for row in slurm_rows:
        job_id = normalize(row.get("job_id", ""))
        metric_map = metrics_by_job.get(job_id, {})

        comparison_rows.append({
            "job_id": job_id,
            "file_name": normalize(row.get("file_name", "")),
            "job_name": normalize(row.get("job_name", "")),
            "partition": normalize(row.get("partition", "")),
            "qos": normalize(row.get("qos", "")),
            "node": normalize(row.get("node", "")),
            "status_guess": normalize(row.get("status_guess", "")),
            "failure_reason": normalize(row.get("failure_reason", "")),
            "walltime_used": normalize(row.get("walltime_used", "")),
            "memory_used_kb": normalize(row.get("memory_used_kb", "")),
            "submitted_child_jobs": normalize(metric_map.get("submitted_child_jobs", row.get("submitted_child_jobs", ""))),
            "completed_child_jobs": normalize(metric_map.get("completed_child_jobs", row.get("completed_child_jobs", ""))),
            "wait_events": normalize(metric_map.get("wait_events", row.get("wait_events", ""))),
            "entity_count": normalize(metric_map.get("entity_count", "")),
            "fitness_1": normalize(metric_map.get("fitness_1", "")),
            "fitness_2": normalize(metric_map.get("fitness_2", "")),
            "generation": normalize(metric_map.get("generation", "")),
            "score": normalize(metric_map.get("score", "")),
            "fitness": normalize(metric_map.get("fitness", "")),
            "accuracy": normalize(metric_map.get("accuracy", "")),
            "loss": normalize(metric_map.get("loss", "")),
            "best_score": normalize(metric_map.get("best_score", "")),
            "runtime_seconds": normalize(metric_map.get("runtime_seconds", "")),
            "most_common_mutate_type": normalize(metric_map.get("most_common_mutate_type", "")),
            "most_common_mutate_type_count": normalize(metric_map.get("most_common_mutate_type_count", "")),
            "inventory_total": inventory_summary.get("inventory_total", 0),
            "inventory_slurm_log_count": inventory_summary.get("slurm_log", 0),
            "inventory_run_config_count": inventory_summary.get("run_config", 0),
            "inventory_result_file_count": inventory_summary.get("result_file", 0),
            "inventory_checkpoint_count": inventory_summary.get("checkpoint", 0),
            "inventory_generated_model_text_count": inventory_summary.get("generated_model_text", 0),
            "inventory_generated_run_script_count": inventory_summary.get("generated_run_script", 0),
            "inventory_state_pickle_count": inventory_summary.get("state_pickle", 0),
        })

    fieldnames = [
        "job_id",
        "file_name",
        "job_name",
        "partition",
        "qos",
        "node",
        "status_guess",
        "failure_reason",
        "walltime_used",
        "memory_used_kb",
        "submitted_child_jobs",
        "completed_child_jobs",
        "wait_events",
        "entity_count",
        "fitness_1",
        "fitness_2",
        "generation",
        "score",
        "fitness",
        "accuracy",
        "loss",
        "best_score",
        "runtime_seconds",
        "most_common_mutate_type",
        "most_common_mutate_type_count",
        "inventory_total",
        "inventory_slurm_log_count",
        "inventory_run_config_count",
        "inventory_result_file_count",
        "inventory_checkpoint_count",
        "inventory_generated_model_text_count",
        "inventory_generated_run_script_count",
        "inventory_state_pickle_count",
    ]

    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Wrote {len(comparison_rows)} rows to {output_file}")


if __name__ == "__main__":
    main()