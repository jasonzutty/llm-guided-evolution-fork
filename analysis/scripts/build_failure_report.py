# 005_build_failure_report.py
# Read-only analysis script that builds a failure counts CSV and Markdown report
# from results/analysis/slurm_summary.csv. It supports both simple single-job
# failures and controller-style logs that submit child jobs.

from pathlib import Path
import argparse
import csv
from collections import Counter, defaultdict


def read_csv_rows(path: Path) -> list[dict]:
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


def to_int(value: str) -> int:
    try:
        return int(normalize(value))
    except Exception:
        return 0


def build_count_rows(rows: list[dict]) -> list[dict]:
    status_counter = Counter()
    reason_counter = Counter()

    for row in rows:
        status = normalize(row.get("status_guess", ""))
        reason = normalize(row.get("failure_reason", ""))

        if status:
            status_counter[status] += 1
        if reason:
            reason_counter[reason] += 1

    out = []
    for status, count in sorted(status_counter.items()):
        out.append({
            "group_type": "status_guess",
            "group_value": status,
            "count": count,
        })

    for reason, count in sorted(reason_counter.items()):
        out.append({
            "group_type": "failure_reason",
            "group_value": reason,
            "count": count,
        })

    return out


def write_markdown_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    total_jobs = len(rows)
    status_counter = Counter(normalize(r.get("status_guess", "")) for r in rows if normalize(r.get("status_guess", "")))
    reason_counter = Counter(normalize(r.get("failure_reason", "")) for r in rows if normalize(r.get("failure_reason", "")))

    total_submitted_child_jobs = sum(to_int(r.get("submitted_child_jobs", "")) for r in rows)
    total_completed_child_jobs = sum(to_int(r.get("completed_child_jobs", "")) for r in rows)
    total_wait_events = sum(to_int(r.get("wait_events", "")) for r in rows)

    examples_by_reason = defaultdict(list)
    for row in rows:
        reason = normalize(row.get("failure_reason", ""))
        if reason:
            examples_by_reason[reason].append(row)

    with path.open("w") as f:
        f.write("# Failure Report\n\n")
        f.write("This report summarizes job outcomes parsed from `slurm_summary.csv`.\n\n")

        f.write("## Overall Summary\n\n")
        f.write(f"- Total jobs reviewed: {total_jobs}\n")
        for status, count in sorted(status_counter.items()):
            pct = (count / total_jobs * 100) if total_jobs else 0
            f.write(f"- {status}: {count} ({pct:.1f}%)\n")
        f.write("\n")

        f.write("## Failure Counts by Reason\n\n")
        if reason_counter:
            for reason, count in sorted(reason_counter.items()):
                pct = (count / total_jobs * 100) if total_jobs else 0
                f.write(f"- {reason}: {count} ({pct:.1f}%)\n")
        else:
            f.write("- No failure reasons detected.\n")
        f.write("\n")

        f.write("## Controller / Child Job Activity\n\n")
        f.write(f"- Total submitted child jobs observed: {total_submitted_child_jobs}\n")
        f.write(f"- Total completed child jobs observed: {total_completed_child_jobs}\n")
        f.write(f"- Total wait events observed: {total_wait_events}\n\n")

        f.write("## Example Failed Jobs by Reason\n\n")
        if not examples_by_reason:
            f.write("No failed jobs with classified reasons were found.\n")
            return

        for reason, reason_rows in sorted(examples_by_reason.items()):
            f.write(f"### {reason}\n\n")
            for row in reason_rows[:5]:
                file_name = normalize(row.get("file_name", ""))
                job_id = normalize(row.get("job_id", ""))
                job_name = normalize(row.get("job_name", ""))
                partition = normalize(row.get("partition", ""))
                node = normalize(row.get("node", ""))
                walltime = normalize(row.get("walltime_used", ""))
                memory = normalize(row.get("memory_used_kb", ""))
                submitted = normalize(row.get("submitted_child_jobs", ""))
                completed = normalize(row.get("completed_child_jobs", ""))
                waits = normalize(row.get("wait_events", ""))

                f.write(
                    f"- file: `{file_name}`, "
                    f"job_id: `{job_id}`, "
                    f"job_name: `{job_name}`, "
                    f"partition: `{partition}`, "
                    f"node: `{node}`, "
                    f"walltime_used: `{walltime}`, "
                    f"memory_used_kb: `{memory}`, "
                    f"submitted_child_jobs: `{submitted}`, "
                    f"completed_child_jobs: `{completed}`, "
                    f"wait_events: `{waits}`\n"
                )
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Build a failure report from slurm_summary.csv")
    parser.add_argument("--input", default="analysis/results/analysis/slurm_summary.csv")
    parser.add_argument("--counts-output", default="analysis/results/analysis/failure_counts.csv")
    parser.add_argument("--report-output", default="analysis/results/analysis/failure_report.md")
    args = parser.parse_args()

    input_file = Path(args.input)
    counts_output = Path(args.counts_output)
    report_output = Path(args.report_output)

    rows = read_csv_rows(input_file)
    count_rows = build_count_rows(rows)

    write_csv(
        counts_output,
        fieldnames=["group_type", "group_value", "count"],
        rows=count_rows,
    )

    write_markdown_report(report_output, rows)

    print(f"Wrote failure counts to {counts_output}")
    print(f"Wrote failure report to {report_output}")


if __name__ == "__main__":
    main()