#!/usr/bin/env python3
from __future__ import annotations

"""
Make:
line or bar chart of generation number vs completion time / duration proxy
or simple generation-completed count
What it shows:
run stability across generations
useful if you want to show one controller run made it through 30 generations

"""

from pathlib import Path
from collections import defaultdict, Counter
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "analysis/results/analysis/matplotlib-cache")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1] / "results" / "analysis"
RUN_METRICS_CSV = BASE_DIR / "run_metrics.csv"
SLURM_SUMMARY_CSV = BASE_DIR / "slurm_summary.csv"
RUN_COMPARISON_CSV = BASE_DIR / "run_comparison.csv"

OUTPUT_REPORT = BASE_DIR / "generation_timeline_report.md"
GRAPHS_DIR = BASE_DIR / "generation_timeline_graphs"


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return []

    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def normalize(value) -> str:
    return str(value or "").strip()


def to_int(value):
    value = normalize(value)
    if value == "":
        return None

    try:
        return int(float(value))
    except Exception:
        return None


def parse_walltime_seconds(value: str):
    value = normalize(value)
    if not value:
        return None

    days = 0
    if "-" in value:
        day_part, value = value.split("-", 1)
        try:
            days = int(day_part)
        except Exception:
            days = 0

    parts = value.split(":")
    try:
        parts = [int(p) for p in parts]
    except Exception:
        return None

    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 1:
        hours = 0
        minutes = 0
        seconds = parts[0]
    else:
        return None

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def safe_div(num, den):
    if den == 0:
        return 0.0
    return num / den


def markdown_table(rows: list[dict], columns: list[str], max_rows: int = 30) -> str:
    if not rows:
        return "_No rows available._\n\n"

    rows = rows[:max_rows]

    header = "| " + " | ".join(columns) + " |\n"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |\n"

    body = ""
    for row in rows:
        values = []
        for col in columns:
            value = normalize(row.get(col, ""))
            value = value.replace("\n", " ").replace("|", "\\|")
            values.append(value)
        body += "| " + " | ".join(values) + " |\n"

    return header + separator + body + "\n"


def find_successful_controller_jobs(slurm_rows: list[dict]) -> list[dict]:
    candidates = []

    for row in slurm_rows:
        submitted = to_int(row.get("submitted_child_jobs", "")) or 0
        completed = to_int(row.get("completed_child_jobs", "")) or 0
        waits = to_int(row.get("wait_events", "")) or 0

        if submitted > 0:
            new = dict(row)
            new["submitted_child_jobs_int"] = submitted
            new["completed_child_jobs_int"] = completed
            new["wait_events_int"] = waits
            new["child_completion_rate"] = safe_div(completed, submitted)
            new["walltime_seconds"] = parse_walltime_seconds(row.get("walltime_used", ""))
            candidates.append(new)

    successful = [
        r for r in candidates
        if normalize(r.get("status_guess", "")) == "completed"
    ]

    if successful:
        successful.sort(
            key=lambda r: (
                r.get("completed_child_jobs_int", 0),
                r.get("submitted_child_jobs_int", 0),
                r.get("wait_events_int", 0),
            ),
            reverse=True,
        )
        return successful

    candidates.sort(
        key=lambda r: (
            r.get("completed_child_jobs_int", 0),
            r.get("submitted_child_jobs_int", 0),
            r.get("wait_events_int", 0),
        ),
        reverse=True,
    )
    return candidates


def select_controller_job(slurm_rows: list[dict]) -> dict | None:
    candidates = find_successful_controller_jobs(slurm_rows)
    if not candidates:
        return None
    return candidates[0]


def extract_generation_events(run_metrics_rows: list[dict]) -> list[dict]:
    events = []

    for row in run_metrics_rows:
        metric_name = normalize(row.get("metric_name", ""))

        if metric_name != "generation":
            continue

        generation = to_int(row.get("metric_value", ""))
        if generation is None:
            continue

        occurrence_index = to_int(row.get("occurrence_index", "")) or 0

        events.append({
            "source_file": normalize(row.get("source_file", "")),
            "job_id": normalize(row.get("job_id", "")),
            "entity_id": normalize(row.get("entity_id", "")),
            "generation": generation,
            "occurrence_index": occurrence_index,
        })

    events.sort(key=lambda r: (r["generation"], r["occurrence_index"]))
    return events


def build_generation_counts(events: list[dict]) -> list[dict]:
    counter = Counter(event["generation"] for event in events)

    rows = []
    cumulative = 0

    for generation in sorted(counter):
        count = counter[generation]
        cumulative += count

        rows.append({
            "generation": str(generation),
            "event_count": str(count),
            "cumulative_events": str(cumulative),
        })

    return rows


def build_generation_proxy_timeline(events: list[dict]) -> list[dict]:
    by_generation = defaultdict(list)

    for event in events:
        by_generation[event["generation"]].append(event.get("occurrence_index", 0))

    rows = []
    for generation in sorted(by_generation):
        indexes = by_generation[generation]
        rows.append({
            "generation": str(generation),
            "first_occurrence_index": str(min(indexes)),
            "last_occurrence_index": str(max(indexes)),
            "event_count": str(len(indexes)),
        })

    return rows


def extract_entity_completion_proxy(run_metrics_rows: list[dict]) -> list[dict]:
    completed_entities = set()

    for row in run_metrics_rows:
        metric_name = normalize(row.get("metric_name", ""))
        metric_value = normalize(row.get("metric_value", ""))
        entity_id = normalize(row.get("entity_id", ""))

        if metric_name == "entity_status" and metric_value == "completed" and entity_id:
            completed_entities.add(entity_id)

    return [{
        "metric": "completed_entities",
        "count": str(len(completed_entities)),
    }]


def save_generation_count_bar(generation_rows: list[dict]) -> Path | None:
    if not generation_rows:
        return None

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    generations = [to_int(row["generation"]) for row in generation_rows]
    counts = [to_int(row["event_count"]) or 0 for row in generation_rows]

    path = GRAPHS_DIR / "generation_completed_count_bar.png"

    plt.figure(figsize=(10, 5))
    plt.bar(generations, counts)
    plt.title("Generation Completed / Activity Count")
    plt.xlabel("Generation")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return path


def save_generation_cumulative_line(generation_rows: list[dict]) -> Path | None:
    if not generation_rows:
        return None

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    generations = [to_int(row["generation"]) for row in generation_rows]
    cumulative = [to_int(row["cumulative_events"]) or 0 for row in generation_rows]

    path = GRAPHS_DIR / "generation_cumulative_timeline.png"

    plt.figure(figsize=(10, 5))
    plt.plot(generations, cumulative, marker="o")
    plt.title("Cumulative Generation Progress")
    plt.xlabel("Generation")
    plt.ylabel("Cumulative Events")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return path


def save_generation_duration_proxy_line(proxy_rows: list[dict]) -> Path | None:
    if not proxy_rows:
        return None

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    generations = [to_int(row["generation"]) for row in proxy_rows]
    last_indexes = [to_int(row["last_occurrence_index"]) or 0 for row in proxy_rows]

    if not any(last_indexes):
        return None

    path = GRAPHS_DIR / "generation_duration_proxy.png"

    plt.figure(figsize=(10, 5))
    plt.plot(generations, last_indexes, marker="o")
    plt.title("Generation Timeline Duration Proxy")
    plt.xlabel("Generation")
    plt.ylabel("Last Occurrence Index")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return path


def graph_md(path: Path) -> str:
    rel = path.relative_to(OUTPUT_REPORT.parent).as_posix()
    return f"![{path.stem}]({rel})\n\n"


def write_report(
    selected_controller: dict | None,
    controller_candidates: list[dict],
    generation_rows: list[dict],
    proxy_rows: list[dict],
    fallback_rows: list[dict],
    graphs: list[Path],
):
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    max_generation = max(
        [to_int(row["generation"]) or 0 for row in generation_rows],
        default=0,
    )
    total_generation_events = sum(
        to_int(row["event_count"]) or 0
        for row in generation_rows
    )

    with OUTPUT_REPORT.open("w") as f:
        f.write("# Generation Timeline Report\n\n")
        f.write(
            "This report summarizes generation-level progress for controller-style evolutionary runs. "
            "It shows whether the controller progressed steadily across generations or failed early.\n\n"
        )

        f.write("## Summary\n\n")
        f.write(f"- Max generation observed: {max_generation}\n")
        f.write(f"- Total generation events observed: {total_generation_events}\n")

        if selected_controller:
            submitted = selected_controller.get("submitted_child_jobs_int", 0)
            completed = selected_controller.get("completed_child_jobs_int", 0)
            waits = selected_controller.get("wait_events_int", 0)
            rate = selected_controller.get("child_completion_rate", 0) * 100

            f.write(f"- Selected controller job: `{normalize(selected_controller.get('job_id', ''))}`\n")
            f.write(f"- Controller status: `{normalize(selected_controller.get('status_guess', ''))}`\n")
            f.write(f"- Submitted child jobs: {submitted}\n")
            f.write(f"- Completed child jobs: {completed}\n")
            f.write(f"- Child-job completion rate: {rate:.1f}%\n")
            f.write(f"- Wait events: {waits}\n")
        else:
            f.write("- Selected controller job: none found\n")

        f.write("\n")

        f.write("## Selected Controller Run\n\n")
        if selected_controller:
            f.write(markdown_table(
                [selected_controller],
                [
                    "job_id",
                    "file_name",
                    "job_name",
                    "partition",
                    "node",
                    "status_guess",
                    "failure_reason",
                    "walltime_used",
                    "submitted_child_jobs",
                    "completed_child_jobs",
                    "wait_events",
                ],
                max_rows=1,
            ))
        else:
            f.write("_No controller run was detected. A controller run is identified as a job with submitted child jobs._\n\n")

        f.write("## Controller Candidates\n\n")
        candidate_rows = []
        for row in controller_candidates:
            candidate_rows.append({
                "job_id": normalize(row.get("job_id", "")),
                "file_name": normalize(row.get("file_name", "")),
                "status_guess": normalize(row.get("status_guess", "")),
                "failure_reason": normalize(row.get("failure_reason", "")),
                "submitted_child_jobs": str(row.get("submitted_child_jobs_int", 0)),
                "completed_child_jobs": str(row.get("completed_child_jobs_int", 0)),
                "completion_rate_percent": f"{row.get('child_completion_rate', 0) * 100:.1f}",
                "wait_events": str(row.get("wait_events_int", 0)),
            })

        f.write(markdown_table(
            candidate_rows,
            [
                "job_id",
                "file_name",
                "status_guess",
                "failure_reason",
                "submitted_child_jobs",
                "completed_child_jobs",
                "completion_rate_percent",
                "wait_events",
            ],
            max_rows=10,
        ))

        f.write("## Generation Completed / Activity Count\n\n")
        f.write(
            "This table counts how many generation events were detected for each generation. "
            "It is a stability proxy: if later generations appear, the controller continued progressing instead of stopping early.\n\n"
        )

        f.write(markdown_table(
            generation_rows,
            ["generation", "event_count", "cumulative_events"],
            max_rows=50,
        ))

        f.write("## Generation Duration Proxy\n\n")
        f.write(
            "If exact timestamps are unavailable, the script uses `occurrence_index` as a simple ordering proxy. "
            "A higher last occurrence index for later generations suggests the run continued to progress through the metric stream.\n\n"
        )

        f.write(markdown_table(
            proxy_rows,
            ["generation", "first_occurrence_index", "last_occurrence_index", "event_count"],
            max_rows=50,
        ))

        f.write("## Visualizations\n\n")
        if graphs:
            for graph in graphs:
                f.write(graph_md(graph))
        else:
            f.write("_No generation graphs were produced because generation metrics were not available._\n\n")

        f.write("## Fallback Entity Completion Proxy\n\n")
        f.write(
            "If generation-level rows are sparse, completed entities can still show that the evaluation pipeline produced valid completed candidates.\n\n"
        )
        f.write(markdown_table(
            fallback_rows,
            ["metric", "count"],
            max_rows=10,
        ))

        f.write("## Interpretation\n\n")
        if max_generation >= 30:
            f.write("- The run reached at least 30 generations, which is strong evidence of controller stability across a long evolutionary process.\n")
        elif max_generation > 0:
            f.write(f"- The run reached generation {max_generation}. This shows some progression, but it may not represent a long stable controller run yet.\n")
        else:
            f.write("- No generation numbers were detected. The logs or metrics may not currently expose generation progress.\n")

        f.write("- A smooth or steadily increasing cumulative line suggests stable progression.\n")
        f.write("- Sudden stops, missing later generations, or very low max generation suggest the controller may have failed or stalled early.\n")
        f.write("- Compare this report with failure and controller activity reports to distinguish algorithmic failure from Slurm/orchestration failure.\n\n")



def main():
    print(f"[INFO] Reading run metrics: {RUN_METRICS_CSV}")
    print(f"[INFO] Reading Slurm summary: {SLURM_SUMMARY_CSV}")
    print(f"[INFO] Reading run comparison: {RUN_COMPARISON_CSV}")
    print(f"[INFO] Writing report: {OUTPUT_REPORT}")
    print(f"[INFO] Writing graphs: {GRAPHS_DIR}")

    run_metrics_rows = read_csv_rows(RUN_METRICS_CSV)
    slurm_rows = read_csv_rows(SLURM_SUMMARY_CSV)
    read_csv_rows(RUN_COMPARISON_CSV)

    if not run_metrics_rows:
        print("[ERROR] No rows found in run_metrics.csv. Cannot build generation timeline.")
        return

    controller_candidates = find_successful_controller_jobs(slurm_rows)
    selected_controller = select_controller_job(slurm_rows)

    generation_events = extract_generation_events(run_metrics_rows)
    generation_rows = build_generation_counts(generation_events)
    proxy_rows = build_generation_proxy_timeline(generation_events)
    fallback_rows = extract_entity_completion_proxy(run_metrics_rows)

    graphs = []

    graph = save_generation_count_bar(generation_rows)
    if graph:
        graphs.append(graph)

    graph = save_generation_cumulative_line(generation_rows)
    if graph:
        graphs.append(graph)

    graph = save_generation_duration_proxy_line(proxy_rows)
    if graph:
        graphs.append(graph)

    write_report(
        selected_controller=selected_controller,
        controller_candidates=controller_candidates,
        generation_rows=generation_rows,
        proxy_rows=proxy_rows,
        fallback_rows=fallback_rows,
        graphs=graphs,
    )

    print("[SUCCESS] Generation timeline report generated.")
    print(f"[SUCCESS] Report: {OUTPUT_REPORT}")
    print(f"[SUCCESS] Graphs: {GRAPHS_DIR}")


if __name__ == "__main__":
    main()
