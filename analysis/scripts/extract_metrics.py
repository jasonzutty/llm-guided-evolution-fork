# extract_metrics.py
# Read-only analysis script that scans logs and state pickle artifacts for real
# metrics and writes them to results/analysis/run_metrics.csv.

from pathlib import Path
import argparse
import csv
import pickle
import re
import math

JOB_ID_FROM_FILENAME_PATTERNS = [
    re.compile(r"slurm-(\d+)\.out$", re.IGNORECASE),
    re.compile(r".*-(\d+)\.out$", re.IGNORECASE),
]

TEXT_METRIC_PATTERNS = [
    ("generation", re.compile(r"(?:STARTING\s+)?GENERATION[:=\s]+(\d+)", re.IGNORECASE)),    
    ("score", re.compile(r"\bscore\b[:=\s]+(-?\d+(?:\.\d+)?)", re.IGNORECASE)),
    ("fitness", re.compile(r"\bfitness\b[:=\s]+(-?\d+(?:\.\d+)?)", re.IGNORECASE)),
    ("accuracy", re.compile(r"\baccuracy\b[:=\s]+(-?\d+(?:\.\d+)?)", re.IGNORECASE)),
    ("loss", re.compile(r"\bloss\b[:=\s]+(-?\d+(?:\.\d+)?)", re.IGNORECASE)),
    ("best_score", re.compile(r"\bbest(?:[_\s-]?so[_\s-]?far)?[_\s-]?score\b[:=\s]+(-?\d+(?:\.\d+)?)", re.IGNORECASE)),
    ("runtime_seconds", re.compile(r"\bruntime(?:_seconds)?\b[:=\s]+(-?\d+(?:\.\d+)?)", re.IGNORECASE)),
]

SUBMITTED_JOB_PATTERN = re.compile(r"Submitted batch job\s+(\d+)", re.IGNORECASE)
COMPLETED_JOB_PATTERN = re.compile(r"LLM Job Completed Successfully\.", re.IGNORECASE)
WAITING_PATTERN = re.compile(r"Waiting on check4job_completion", re.IGNORECASE)


def extract_job_id_from_filename(filename: str) -> str:
    for pattern in JOB_ID_FROM_FILENAME_PATTERNS:
        match = pattern.match(filename)
        if match:
            return match.group(1)
    return ""


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore")


def is_numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def safe_numeric_to_str(value) -> str:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
    return str(value)


def make_row(source_file: str, job_id: str, metric_name: str, metric_value: str, occurrence_index: int, entity_id: str = "") -> dict:
    return {
        "source_file": source_file,
        "job_id": job_id,
        "entity_id": entity_id,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "occurrence_index": occurrence_index,
    }


def extract_text_metric_rows(path: Path) -> list[dict]:
    text = read_text(path)
    rows = []
    job_id = extract_job_id_from_filename(path.name)
    source_file = str(path.as_posix())

    for metric_name, pattern in TEXT_METRIC_PATTERNS:
        matches = pattern.findall(text)
        if not matches:
            continue

        for idx, value in enumerate(matches, start=1):
            rows.append(make_row(source_file, job_id, metric_name, str(value), idx))

    submitted_jobs = len(SUBMITTED_JOB_PATTERN.findall(text))
    completed_jobs = len(COMPLETED_JOB_PATTERN.findall(text))
    wait_events = len(WAITING_PATTERN.findall(text))

    if submitted_jobs:
        rows.append(make_row(source_file, job_id, "submitted_child_jobs", str(submitted_jobs), 1))
    if completed_jobs:
        rows.append(make_row(source_file, job_id, "completed_child_jobs", str(completed_jobs), 1))
    if wait_events:
        rows.append(make_row(source_file, job_id, "wait_events", str(wait_events), 1))

    return rows


def extract_pickle_metric_rows(path: Path) -> list[dict]:
    rows = []
    source_file = str(path.as_posix())

    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
    except Exception:
        return rows

    if not isinstance(obj, dict):
        return rows

    global_data = obj.get("GLOBAL_DATA", {})
    global_data_hist = obj.get("GLOBAL_DATA_HIST", {})
    global_data_ancestry = obj.get("GLOBAL_DATA_ANCESTRY", {})

    # Aggregate top-level counts
    if isinstance(global_data, dict):
        rows.append(make_row(source_file, "", "global_data_count", str(len(global_data)), 1))
    if isinstance(global_data_hist, dict):
        rows.append(make_row(source_file, "", "global_data_hist_count", str(len(global_data_hist)), 1))
    if isinstance(global_data_ancestry, dict):
        rows.append(make_row(source_file, "", "global_data_ancestry_count", str(len(global_data_ancestry)), 1))

    # Per-entity metrics from GLOBAL_DATA
    if isinstance(global_data, dict):
        for entity_id, payload in global_data.items():
            if not isinstance(payload, dict):
                continue

            job_id = str(payload.get("job_id", "") or "")
            occurrence_index = 1

            status = payload.get("status")
            if status is not None:
                rows.append(make_row(source_file, job_id, "entity_status", str(status), occurrence_index, entity_id))

            results_job = payload.get("results_job")
            if results_job not in (None, ""):
                rows.append(make_row(source_file, job_id, "results_job", str(results_job), occurrence_index, entity_id))

            sub_flag = payload.get("sub_flag")
            if sub_flag is not None:
                rows.append(make_row(source_file, job_id, "sub_flag", str(sub_flag), occurrence_index, entity_id))

            start_time = payload.get("start_time")
            if is_numeric(start_time):
                rows.append(make_row(source_file, job_id, "start_time", safe_numeric_to_str(start_time), occurrence_index, entity_id))

            local_output = payload.get("local_output")
            if local_output not in (None, ""):
                rows.append(make_row(source_file, job_id, "local_output", str(local_output), occurrence_index, entity_id))

            fitness = payload.get("fitness")
            if isinstance(fitness, (list, tuple)):
                for idx, fit_value in enumerate(fitness, start=1):
                    rows.append(make_row(source_file, job_id, f"fitness_{idx}", safe_numeric_to_str(fit_value), 1, entity_id))
            elif fitness is not None:
                rows.append(make_row(source_file, job_id, "fitness", safe_numeric_to_str(fitness), 1, entity_id))

    # Per-entity ancestry / mutation fields
    if isinstance(global_data_ancestry, dict):
        for entity_id, payload in global_data_ancestry.items():
            if not isinstance(payload, dict):
                continue

            genes = payload.get("GENES")
            mutate_type = payload.get("MUTATE_TYPE")

            if isinstance(genes, list):
                rows.append(make_row(source_file, "", "ancestry_gene_count", str(len(genes)), 1, entity_id))

            if isinstance(mutate_type, list):
                rows.append(make_row(source_file, "", "mutate_type_count", str(len(mutate_type)), 1, entity_id))
                for idx, mt in enumerate(mutate_type, start=1):
                    rows.append(make_row(source_file, "", "mutate_type", str(mt), idx, entity_id))

    return rows


def collect_candidate_files(repo_root: Path) -> list[Path]:
    files = []

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(repo_root).as_posix()

        # Slurm logs can live at repo root or in nested run output folders.
        if path.suffix == ".out":
            files.append(path)
            continue

        # Generated run artifacts
        if rel.startswith("0/") and path.suffix in {".txt", ".log", ".json", ".csv", ".pkl"}:
            files.append(path)
            continue

        # Standalone pickle state files anywhere outside results/analysis
        if path.suffix == ".pkl" and not rel.startswith("analysis/results/analysis/"):
            files.append(path)
            continue

    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="Extract metrics from logs and pickle state artifacts.")
    parser.add_argument("--input", default=".", help="Repository root or directory to scan")
    parser.add_argument("--output", default="analysis/results/analysis/run_metrics.csv", help="Output CSV path")
    args = parser.parse_args()

    repo_root = Path(args.input).resolve()
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in collect_candidate_files(repo_root):
        if path.suffix == ".pkl":
            rows.extend(extract_pickle_metric_rows(path))
        else:
            rows.extend(extract_text_metric_rows(path))

    fieldnames = ["source_file", "job_id", "entity_id", "metric_name", "metric_value", "occurrence_index"]

    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_file}")


if __name__ == "__main__":
    main()
