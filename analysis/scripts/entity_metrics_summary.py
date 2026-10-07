# entity_metrics_summary.py
# Read-only analysis script that converts extracted metric rows into a clean
# per-entity summary table and a readable Markdown report, mainly for
# pickle-derived experiment-state data.

from pathlib import Path
import argparse
import csv
import math
from collections import defaultdict, Counter


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def normalize(value: str) -> str:
    return (value or "").strip()


def parse_float(value: str):
    value = normalize(value).lower()
    if value == "":
        return None
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    if value == "nan":
        return math.nan
    try:
        return float(value)
    except Exception:
        return None


def is_finite_number(value: str) -> bool:
    x = parse_float(value)
    return x is not None and math.isfinite(x)


def build_entities(rows: list[dict]) -> list[dict]:
    entities = defaultdict(lambda: {
        "source_file": "",
        "entity_id": "",
        "job_id": "",
        "entity_status": "",
        "results_job": "",
        "sub_flag": "",
        "start_time": "",
        "fitness_1": "",
        "fitness_2": "",
    })

    mutate_type_counter = defaultdict(Counter)

    for row in rows:
        entity_id = normalize(row.get("entity_id", ""))
        if not entity_id:
            continue

        metric_name = normalize(row.get("metric_name", ""))
        metric_value = normalize(row.get("metric_value", ""))
        source_file = normalize(row.get("source_file", ""))
        job_id = normalize(row.get("job_id", ""))

        entity = entities[entity_id]
        entity["source_file"] = source_file or entity["source_file"]
        entity["entity_id"] = entity_id
        entity["job_id"] = job_id or entity["job_id"]

        if metric_name in entity:
            entity[metric_name] = metric_value

        if metric_name == "mutate_type":
            mutate_type_counter[entity_id][metric_value] += 1

    output_rows = []
    for entity_id, entity in sorted(entities.items()):
        counter = mutate_type_counter.get(entity_id, Counter())
        if counter:
            most_common_type, count = counter.most_common(1)[0]
        else:
            most_common_type, count = "", 0

        output_rows.append({
            "source_file": entity["source_file"],
            "entity_id": entity["entity_id"],
            "job_id": entity["job_id"],
            "entity_status": entity["entity_status"],
            "results_job": entity["results_job"],
            "sub_flag": entity["sub_flag"],
            "start_time": entity["start_time"],
            "fitness_1": entity["fitness_1"],
            "fitness_2": entity["fitness_2"],
            "mutate_type_count": count,
            "most_common_mutate_type": most_common_type,
        })

    return output_rows


def write_csv_output(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "source_file",
        "entity_id",
        "job_id",
        "entity_status",
        "results_job",
        "sub_flag",
        "start_time",
        "fitness_1",
        "fitness_2",
        "mutate_type_count",
        "most_common_mutate_type",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: Path, rows: list[dict]) -> None:
    total_entities = len(rows)
    completed_entities = sum(1 for r in rows if normalize(r.get("entity_status", "")) == "completed")
    missing_status_entities = sum(1 for r in rows if normalize(r.get("entity_status", "")) == "")
    finite_fitness_entities = sum(1 for r in rows if is_finite_number(r.get("fitness_1", "")))
    infinite_fitness_entities = sum(
        1 for r in rows
        if normalize(r.get("fitness_1", "")).lower() in {"inf", "-inf"}
        or normalize(r.get("fitness_2", "")).lower() in {"inf", "-inf"}
    )

    mutate_counter = Counter(
        normalize(r.get("most_common_mutate_type", ""))
        for r in rows
        if normalize(r.get("most_common_mutate_type", ""))
    )

    top_finite = [
        r for r in rows
        if is_finite_number(r.get("fitness_1", ""))
    ]
    top_finite.sort(key=lambda r: parse_float(r.get("fitness_1", "")), reverse=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Entity Metrics Report\n\n")
        f.write("This report summarizes entity-level metrics derived from `run_metrics.csv`.\n\n")

        f.write("## Overall Summary\n\n")
        f.write(f"- Total entities reviewed: {total_entities}\n")
        f.write(f"- Completed entities: {completed_entities}\n")
        f.write(f"- Entities with missing status: {missing_status_entities}\n")
        f.write(f"- Entities with finite fitness_1: {finite_fitness_entities}\n")
        f.write(f"- Entities with infinite fitness values: {infinite_fitness_entities}\n\n")

        f.write("## Mutation Type Summary\n\n")
        if mutate_counter:
            for mutate_type, count in mutate_counter.most_common():
                f.write(f"- {mutate_type}: {count}\n")
        else:
            f.write("- No mutation types found.\n")
        f.write("\n")

        f.write("## Top Entities by fitness_1\n\n")
        if top_finite:
            for row in top_finite[:10]:
                f.write(
                    f"- entity_id: `{normalize(row.get('entity_id', ''))}`, "
                    f"job_id: `{normalize(row.get('job_id', ''))}`, "
                    f"status: `{normalize(row.get('entity_status', ''))}`, "
                    f"fitness_1: `{normalize(row.get('fitness_1', ''))}`, "
                    f"fitness_2: `{normalize(row.get('fitness_2', ''))}`, "
                    f"results_job: `{normalize(row.get('results_job', ''))}`\n"
                )
        else:
            f.write("- No finite fitness values found.\n")


def main():
    parser = argparse.ArgumentParser(description="Build an entity-level summary and Markdown report from run_metrics.csv")
    parser.add_argument("--input", default="analysis/results/analysis/run_metrics.csv")
    parser.add_argument("--output", default="analysis/results/analysis/entity_metrics_summary.csv")
    parser.add_argument("--report-output", default="analysis/results/analysis/entity_metrics_report.md")
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)
    report_output = Path(args.report_output)

    rows = read_csv_rows(input_file)
    entity_rows = build_entities(rows)

    write_csv_output(output_file, entity_rows)
    write_markdown_report(report_output, entity_rows)

    print(f"Wrote {len(entity_rows)} rows to {output_file}")
    print(f"Wrote entity report to {report_output}")


if __name__ == "__main__":
    main()