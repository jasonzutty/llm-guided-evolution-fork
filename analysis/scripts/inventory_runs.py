# inventory_runs.py
# Read-only analysis script that inventories run-related artifacts in the repo
# and writes a structured table to results/analysis/run_inventory.csv.

from pathlib import Path
import argparse
import csv
from datetime import datetime

CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
NOISE_FILE_NAMES = {".DS_Store", "Thumbs.db"}
EXCLUDED_TOP_LEVEL_DIRS = {
    ".git",
    ".github",
    ".venv",
    "__pycache__",
    "docs",
    "assets",
    "templates",
    "tests",
    "src",
    "cifar10",
    "cifar-10-batches-py",
}

RUN_RELATED_DIR_PREFIXES = [
    "0",
    "results",
    "analysis/test_data",
]


def is_noise_file(path: Path) -> bool:
    return path.name in NOISE_FILE_NAMES or path.suffix == ".pyc"


def is_run_related(path: Path, repo_root: Path) -> bool:
    if is_noise_file(path):
        return False

    rel = path.relative_to(repo_root).as_posix()

    if path.suffix == ".out":
        return True

    if path.suffix in CHECKPOINT_SUFFIXES:
        return True

    if path.suffix == ".pkl":
        return True

    for prefix in RUN_RELATED_DIR_PREFIXES:
        if rel == prefix or rel.startswith(prefix + "/"):
            return True

    return False


def classify_file(path: Path) -> str:
    rel_parts = path.parts

    if path.suffix == ".out":
        return "slurm_log"
    if path.suffix in CHECKPOINT_SUFFIXES:
        return "checkpoint"
    if path.suffix == ".pkl":
        return "state_pickle"
    if rel_parts and rel_parts[0] == "0" and path.name.endswith("_model.txt"):
        return "generated_model_text"
    if rel_parts and rel_parts[0] == "0" and path.suffix == ".sh":
        return "generated_run_script"
    if "results" in rel_parts and path.suffix in {".csv", ".json", ".txt", ".md"}:
        return "result_file"
    if "analysis" in rel_parts and path.suffix in {".csv", ".json", ".txt", ".md"}:
        return "analysis_output"
    if path.suffix in {".yaml", ".yml", ".toml", ".ini"}:
        return "run_config"
    return "other_run_artifact"


def get_file_type(path: Path) -> str:
    return path.suffix.lstrip(".").lower() if path.suffix else "no_extension"


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def build_record(path: Path, repo_root: Path) -> dict:
    stat = path.stat()
    rel_path = path.relative_to(repo_root).as_posix()
    return {
        "relative_path": rel_path,
        "file_name": path.name,
        "parent_dir": path.parent.relative_to(repo_root).as_posix() if path.parent != repo_root else ".",
        "category": classify_file(path),
        "file_type": get_file_type(path),
        "size_bytes": stat.st_size,
        "modified_at": iso_mtime(path),
    }


def collect_files(repo_root: Path) -> list[Path]:
    files = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue

        rel_parts = path.relative_to(repo_root).parts
        if rel_parts and rel_parts[0] in EXCLUDED_TOP_LEVEL_DIRS:
            continue

        if is_run_related(path, repo_root):
            files.append(path)

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Inventory run-related artifacts in the repository.")
    parser.add_argument("--input", default=".", help="Repository root or directory to scan")
    parser.add_argument("--output", default="analysis/results/analysis/run_inventory.csv", help="Output CSV path")
    args = parser.parse_args()

    repo_root = Path(args.input).resolve()
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files(repo_root)
    rows = [build_record(path, repo_root) for path in files]

    fieldnames = [
        "relative_path",
        "file_name",
        "parent_dir",
        "category",
        "file_type",
        "size_bytes",
        "modified_at",
    ]

    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_file}")


if __name__ == "__main__":
    main()