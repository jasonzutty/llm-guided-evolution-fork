# summarize_slurm.py
# Read-only analysis script for Slurm output logs. It summarizes Slurm log files
# from a specified input directory and writes a structured CSV summary to a
# specified output path. It extracts metadata, resource usage, controller-job
# status, and a few operational counts without modifying raw logs.

from pathlib import Path
import argparse
import csv
import re

# Supports both standard Slurm names and custom report-style names.
JOB_ID_FROM_FILENAME_PATTERNS = [
    re.compile(r"slurm-(\d+)\.out$", re.IGNORECASE),
    re.compile(r".*-(\d+)\.out$", re.IGNORECASE),
]

FIELD_PATTERNS = {
    "job_id": re.compile(r"^Job ID:\s+(.*)$", re.MULTILINE),
    "user_id": re.compile(r"^User ID:\s+(.*)$", re.MULTILINE),
    "account": re.compile(r"^Account:\s+(.*)$", re.MULTILINE),
    "job_name": re.compile(r"^Job name:\s+(.*)$", re.MULTILINE),
    "partition": re.compile(r"^Partition:\s+(.*)$", re.MULTILINE),
    "qos": re.compile(r"^QOS:\s+(.*)$", re.MULTILINE),
    "node": re.compile(r"^Nodes:\s+(.*)$", re.MULTILINE),
    "resources": re.compile(r"^Resources:\s+(.*)$", re.MULTILINE),
    "rsrc_used": re.compile(r"^Rsrc Used:\s+(.*)$", re.MULTILINE),
    "started_on": re.compile(r"^Started on\s+(.*)$", re.MULTILINE),
}

# Ordered from most specific / important to more general.
FAILURE_PATTERNS = [
    ("oom_killed", ["oom_kill event", "oom killed", "out of memory"]),
    ("cancelled", ["cancelled at", "cancelled"]),
    ("time_limit", ["due to time limit", "timed out"]),
    ("traceback", ["traceback (most recent call last):"]),
    ("runtime_error", ["runtimeerror:"]),
    ("slurm_error", ["slurmstepd: error:"]),
]

SUBMITTED_JOB_PATTERN = re.compile(r"Submitted batch job\s+(\d+)")
COMPLETED_JOB_PATTERN = re.compile(r"LLM Job Completed Successfully\.", re.IGNORECASE)
WAITING_PATTERN = re.compile(r"Waiting on check4job_completion", re.IGNORECASE)


def extract_first_match(pattern, text):
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def extract_job_id_from_filename(filename: str) -> str:
    for pattern in JOB_ID_FROM_FILENAME_PATTERNS:
        match = pattern.match(filename)
        if match:
            return match.group(1)
    return ""


def classify_log(text: str):
    lower_text = text.lower()

    for reason, patterns in FAILURE_PATTERNS:
        for pattern in patterns:
            if pattern in lower_text:
                return "failed", reason

    if "Begin Slurm Epilog" in text:
        return "completed", ""

    return "unknown", ""


def extract_memory_used_kb(rsrc_used: str) -> str:
    match = re.search(r"mem=(\d+)K", rsrc_used)
    return match.group(1) if match else ""


def extract_walltime_used(rsrc_used: str) -> str:
    match = re.search(r"walltime=([0-9:]+)", rsrc_used)
    return match.group(1) if match else ""


def extract_hostname_hint(text: str, lines: list[str]) -> str:
    started_on = extract_first_match(FIELD_PATTERNS["started_on"], text)
    if started_on:
        return started_on

    for line in lines[:30]:
        stripped = line.strip()
        if stripped.endswith(".pace.gatech.edu"):
            return stripped
        if stripped.startswith("Started on "):
            return stripped.replace("Started on ", "", 1).strip()

    return ""


def count_submitted_jobs(text: str) -> int:
    return len(SUBMITTED_JOB_PATTERN.findall(text))


def count_completed_jobs(text: str) -> int:
    return len(COMPLETED_JOB_PATTERN.findall(text))


def count_wait_events(text: str) -> int:
    return len(WAITING_PATTERN.findall(text))


def summarize_file(path: Path):
    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    filename_job_id = extract_job_id_from_filename(path.name)
    parsed_job_id = extract_first_match(FIELD_PATTERNS["job_id"], text)

    status_guess, failure_reason = classify_log(text)
    rsrc_used = extract_first_match(FIELD_PATTERNS["rsrc_used"], text)
    hostname_hint = extract_hostname_hint(text, lines)

    return {
        "file_name": path.name,
        "job_id": parsed_job_id or filename_job_id,
        "job_name": extract_first_match(FIELD_PATTERNS["job_name"], text),
        "partition": extract_first_match(FIELD_PATTERNS["partition"], text),
        "qos": extract_first_match(FIELD_PATTERNS["qos"], text),
        "node": extract_first_match(FIELD_PATTERNS["node"], text),
        "line_count": len(lines),
        "status_guess": status_guess,
        "failure_reason": failure_reason,
        "walltime_used": extract_walltime_used(rsrc_used),
        "memory_used_kb": extract_memory_used_kb(rsrc_used),
        "hostname_hint": hostname_hint,
        "submitted_child_jobs": count_submitted_jobs(text),
        "completed_child_jobs": count_completed_jobs(text),
        "wait_events": count_wait_events(text),
        "first_5_lines": " | ".join(lines[:5]),
        "last_5_lines": " | ".join(lines[-5:]) if lines else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize Slurm output logs.")
    parser.add_argument(
        "--input",
        default="analysis/test_data/slurm",
        help="Directory containing Slurm .out files",
    )
    parser.add_argument(
        "--output",
        default="analysis/results/analysis/slurm_summary.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    out_files = sorted(input_dir.rglob("*.out"))
    rows = [summarize_file(path) for path in out_files]

    fieldnames = [
        "file_name",
        "job_id",
        "job_name",
        "partition",
        "qos",
        "node",
        "line_count",
        "status_guess",
        "failure_reason",
        "walltime_used",
        "memory_used_kb",
        "hostname_hint",
        "submitted_child_jobs",
        "completed_child_jobs",
        "wait_events",
        "first_5_lines",
        "last_5_lines",
    ]

    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_file}")


if __name__ == "__main__":
    main()
