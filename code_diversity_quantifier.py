#!/usr/bin/env python3
"""
Measure code diversity for each generation of an LLM-guided evolution run.

For every `global_data/global_gen_*.pkl` file produced by the run pipeline, the
script resolves the corresponding generated code (defaulting to
`./sota/ExquisiteNetV2/models/network_<gene_id>.py`), computes AST-based cosine
distances between individuals, and summarizes the diversity statistics. Results
are printed with debug statements and optionally written to a CSV file that
resides inside the run directory.
"""

from __future__ import annotations

import argparse
import ast
import math
import pickle
import random
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class DistanceStats:
    count: int = 0
    sampled: bool = False
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    stdev: float | None = None


@dataclass
class DiversityLeaders:
    most_diverse_gene: str | None = None
    most_diverse_score: float | None = None
    least_diverse_gene: str | None = None
    least_diverse_score: float | None = None


@dataclass
class GenerationSummary:
    generation: int
    file_path: Path
    total_gene_records: int
    considered_genes: int
    missing_files: int
    analyzed_files: int
    distance_stats: DistanceStats
    leaders: DiversityLeaders


# --------------------------------------------------------------------------- #
# Argument Parsing                                                            #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify code diversity per generation using AST cosine distances."
    )
    parser.add_argument(
        "run_directory",
        type=Path,
        help="Run directory that contains `global_data/` from pace_ice_island_controller.",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Root directory that holds generated models (defaults to ./sota/ExquisiteNetV2/models).",
    )
    parser.add_argument(
        "--include-hist",
        action="store_true",
        help="Include entries from GLOBAL_DATA_HIST when gathering gene IDs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of generations processed (ascending order).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap the number of code files sampled per generation.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Cap the number of pairwise comparisons per generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling files or pairs.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="code_diversity_summary.csv",
        help="Name of the CSV file written into the run directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summaries without writing the CSV.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Pickle + Gene Utilities                                                     #
# --------------------------------------------------------------------------- #


def extract_generation(filename: str) -> int:
    """Extract the integer generation number from a filename like global_gen_12.pkl."""
    try:
        return int(filename.split("_")[2].split(".")[0])
    except (IndexError, ValueError):
        raise ValueError(f"Cannot extract generation number from {filename}")


def load_pickle(path: Path) -> dict:
    print(f"[DEBUG] Loading pickle: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def gather_gene_ids(data: Mapping[str, dict], include_hist: bool) -> List[str]:
    gene_ids = set()
    for source_key in ("GLOBAL_DATA", "GLOBAL_DATA_HIST" if include_hist else None):
        if not source_key:
            continue
        source = data.get(source_key, {})
        for gene_id in source.keys():
            if gene_id is not None:
                gene_ids.add(str(gene_id))
    return sorted(gene_ids)


# --------------------------------------------------------------------------- #
# Code Repository                                                             #
# --------------------------------------------------------------------------- #


class CodeRepository:
    """
    Resolve and cache generated network code for gene IDs.
    """

    def __init__(self, models_root: Path) -> None:
        self.models_root = models_root
        self._code_cache: Dict[str, str] = {}
        self._ast_cache: Dict[str, Counter[str]] = {}

    def resolve_path(self, gene_id: str) -> Path:
        filename = f"network_{gene_id}.py"
        return self.models_root / filename

    def load_code(self, gene_id: str) -> Optional[str]:
        if gene_id in self._code_cache:
            return self._code_cache[gene_id]
        code_path = self.resolve_path(gene_id)
        if not code_path.is_file():
            print(f"[DEBUG] Missing code file for gene {gene_id}: {code_path}")
            return None
        try:
            code_text = code_path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"[DEBUG] Failed to read {code_path}: {exc}")
            return None
        self._code_cache[gene_id] = code_text
        return code_text

    def get_ast_counts(self, gene_id: str) -> Optional[Counter[str]]:
        if gene_id in self._ast_cache:
            return self._ast_cache[gene_id]
        code_text = self.load_code(gene_id)
        if code_text is None:
            return None
        try:
            tree = ast.parse(code_text)
        except SyntaxError as exc:
            print(f"[DEBUG] SyntaxError parsing gene {gene_id}: {exc}")
            return None
        counts = Counter(type(node).__name__ for node in ast.walk(tree))
        self._ast_cache[gene_id] = counts
        return counts


# --------------------------------------------------------------------------- #
# Distance + Diversity Calculations                                           #
# --------------------------------------------------------------------------- #


def cosine_distance(counts_a: Counter[str], counts_b: Counter[str]) -> float:
    all_keys = set(counts_a.keys()) | set(counts_b.keys())
    dot_product = sum(counts_a.get(k, 0) * counts_b.get(k, 0) for k in all_keys)
    norm_a = math.sqrt(sum((counts_a.get(k, 0)) ** 2 for k in all_keys))
    norm_b = math.sqrt(sum((counts_b.get(k, 0)) ** 2 for k in all_keys))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot_product / (norm_a * norm_b)
    return max(0.0, min(1.0, 1.0 - similarity))


def compute_pairwise_distances(
    gene_ids: Sequence[str],
    ast_cache: Mapping[str, Counter[str]],
    max_pairs: Optional[int],
    rng: random.Random,
) -> Tuple[List[float], Dict[str, List[float]], bool]:
    total_pairs = len(gene_ids) * (len(gene_ids) - 1) // 2
    sampled = False
    if max_pairs is not None and max_pairs < total_pairs:
        sampled = True
        print(f"[DEBUG] Sampling {max_pairs} of {total_pairs} pairs.")
        selected = set()
        while len(selected) < max_pairs:
            idx_a, idx_b = rng.sample(range(len(gene_ids)), 2)
            i, j = sorted((idx_a, idx_b))
            selected.add((i, j))
        pair_indices = list(selected)
    else:
        pair_indices = [(i, j) for i in range(len(gene_ids)) for j in range(i + 1, len(gene_ids))]

    distances: List[float] = []
    contributions: Dict[str, List[float]] = defaultdict(list)
    for i, j in pair_indices:
        gene_i = gene_ids[i]
        gene_j = gene_ids[j]
        dist = cosine_distance(ast_cache[gene_i], ast_cache[gene_j])
        distances.append(dist)
        contributions[gene_i].append(dist)
        contributions[gene_j].append(dist)
    for gene_id in gene_ids:
        contributions.setdefault(gene_id, [])
    return distances, contributions, sampled


def diversity_leaders(contributions: Mapping[str, Sequence[float]]) -> DiversityLeaders:
    if not contributions:
        return DiversityLeaders()

    averages = {}
    for gene_id, values in contributions.items():
        if not values:
            averages[gene_id] = 0.0
        else:
            averages[gene_id] = mean(values)

    most_gene = max(averages, key=averages.get)
    least_gene = min(averages, key=averages.get)
    return DiversityLeaders(
        most_diverse_gene=most_gene,
        most_diverse_score=averages[most_gene],
        least_diverse_gene=least_gene,
        least_diverse_score=averages[least_gene],
    )


def summarize_distances(distances: Sequence[float]) -> DistanceStats:
    if not distances:
        return DistanceStats(count=0)
    return DistanceStats(
        count=len(distances),
        minimum=min(distances),
        maximum=max(distances),
        mean=safe_statistic(mean, distances),
        median=safe_statistic(median, distances),
        stdev=safe_statistic(stdev, distances),
    )


def safe_statistic(func, values: Sequence[float]) -> float | None:
    if not values:
        return None
    if func is stdev and len(values) < 2:
        return 0.0
    try:
        return func(values)
    except Exception as exc:
        print(f"[DEBUG] Statistic {func.__name__} failed on {values}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Generation Summaries                                                        #
# --------------------------------------------------------------------------- #


def summarize_generation(
    generation: int,
    file_path: Path,
    data: Mapping[str, dict],
    repository: CodeRepository,
    rng: random.Random,
    include_hist: bool,
    max_files: Optional[int],
    max_pairs: Optional[int],
) -> GenerationSummary:
    gene_ids = gather_gene_ids(data, include_hist=include_hist)
    total_gene_records = len(gene_ids)
    print(
        f"[DEBUG] Generation {generation} -> unique gene records: {total_gene_records} "
        f"(source file: {file_path.name})"
    )

    if total_gene_records == 0:
        return GenerationSummary(
            generation=generation,
            file_path=file_path,
            total_gene_records=0,
            considered_genes=0,
            missing_files=0,
            analyzed_files=0,
            distance_stats=DistanceStats(),
            leaders=DiversityLeaders(),
        )

    considered_gene_ids = gene_ids.copy()
    if max_files is not None and len(considered_gene_ids) > max_files:
        considered_gene_ids = rng.sample(considered_gene_ids, max_files)
        print(
            f"[DEBUG] Limiting to {max_files} code files out of {total_gene_records} available "
            f"for generation {generation}."
        )

    missing_files = 0
    ast_vectors: Dict[str, Counter[str]] = {}
    for gene_id in considered_gene_ids:
        counts = repository.get_ast_counts(gene_id)
        if counts is None:
            missing_files += 1
        else:
            ast_vectors[gene_id] = counts

    analyzed_files = len(ast_vectors)
    print(
        f"[DEBUG] Generation {generation} -> considered: {len(considered_gene_ids)} "
        f"missing: {missing_files} analyzed: {analyzed_files}"
    )

    if analyzed_files < 2:
        print(f"[DEBUG] Generation {generation} has fewer than two valid code files; skipping distance calc.")
        return GenerationSummary(
            generation=generation,
            file_path=file_path,
            total_gene_records=total_gene_records,
            considered_genes=len(considered_gene_ids),
            missing_files=missing_files,
            analyzed_files=analyzed_files,
            distance_stats=DistanceStats(),
            leaders=DiversityLeaders(),
        )

    ordered_genes = sorted(ast_vectors.keys())
    distances, contributions, sampled = compute_pairwise_distances(
        ordered_genes, ast_vectors, max_pairs=max_pairs, rng=rng
    )
    distance_stats = summarize_distances(distances)
    distance_stats.sampled = sampled

    leaders = diversity_leaders(contributions)
    return GenerationSummary(
        generation=generation,
        file_path=file_path,
        total_gene_records=total_gene_records,
        considered_genes=len(considered_gene_ids),
        missing_files=missing_files,
        analyzed_files=analyzed_files,
        distance_stats=distance_stats,
        leaders=leaders,
    )


def summarize_run(
    run_directory: Path,
    models_root: Path,
    include_hist: bool,
    limit: Optional[int],
    max_files: Optional[int],
    max_pairs: Optional[int],
    seed: int,
) -> List[GenerationSummary]:
    global_data_dir = run_directory / "global_data"
    if not global_data_dir.is_dir():
        raise FileNotFoundError(f"No global_data/ directory found in {run_directory}")

    generation_files = sorted(
        global_data_dir.glob("global_gen_*.pkl"), key=lambda path: extract_generation(path.name)
    )
    if limit is not None:
        generation_files = generation_files[:limit]

    print(f"[DEBUG] Found {len(generation_files)} generation files (limit={limit}).")

    repository = CodeRepository(models_root)
    rng = random.Random(seed)

    summaries: List[GenerationSummary] = []
    for file_path in generation_files:
        generation = extract_generation(file_path.name)
        data = load_pickle(file_path)
        summary = summarize_generation(
            generation=generation,
            file_path=file_path,
            data=data,
            repository=repository,
            rng=rng,
            include_hist=include_hist,
            max_files=max_files,
            max_pairs=max_pairs,
        )
        summaries.append(summary)
    return summaries


# --------------------------------------------------------------------------- #
# Output Helpers                                                              #
# --------------------------------------------------------------------------- #


def print_summary_table(summaries: Sequence[GenerationSummary]) -> None:
    if not summaries:
        print("[DEBUG] No generation summaries available.")
        return

    header = [
        "Gen",
        "Records",
        "Considered",
        "Missing",
        "Analyzed",
        "Pairs",
        "Sampled",
        "Min",
        "Max",
        "Mean",
        "Median",
        "Std",
        "TopGene",
        "TopScore",
        "LowGene",
        "LowScore",
    ]
    print("[DEBUG] Summary Table:")
    print("\t".join(header))

    for summary in summaries:
        stats = summary.distance_stats
        leaders = summary.leaders
        row = [
            str(summary.generation),
            str(summary.total_gene_records),
            str(summary.considered_genes),
            str(summary.missing_files),
            str(summary.analyzed_files),
            str(stats.count),
            "yes" if stats.sampled else "no",
            format_float(stats.minimum),
            format_float(stats.maximum),
            format_float(stats.mean),
            format_float(stats.median),
            format_float(stats.stdev),
            leaders.most_diverse_gene or "",
            format_float(leaders.most_diverse_score),
            leaders.least_diverse_gene or "",
            format_float(leaders.least_diverse_score),
        ]
        print("\t".join(row))


def write_csv(summaries: Sequence[GenerationSummary], destination: Path) -> None:
    if not summaries:
        print(f"[DEBUG] No summaries to write; skipping CSV at {destination}.")
        return

    fieldnames = [
        "generation",
        "file_path",
        "total_gene_records",
        "considered_genes",
        "missing_files",
        "analyzed_files",
        "pair_count",
        "pairs_sampled",
        "distance_min",
        "distance_max",
        "distance_mean",
        "distance_median",
        "distance_stdev",
        "most_diverse_gene",
        "most_diverse_score",
        "least_diverse_gene",
        "least_diverse_score",
    ]

    print(f"[DEBUG] Writing CSV to {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            stats = summary.distance_stats
            leaders = summary.leaders
            writer.writerow(
                {
                    "generation": summary.generation,
                    "file_path": summary.file_path.name,
                    "total_gene_records": summary.total_gene_records,
                    "considered_genes": summary.considered_genes,
                    "missing_files": summary.missing_files,
                    "analyzed_files": summary.analyzed_files,
                    "pair_count": stats.count,
                    "pairs_sampled": stats.sampled,
                    "distance_min": stats.minimum,
                    "distance_max": stats.maximum,
                    "distance_mean": stats.mean,
                    "distance_median": stats.median,
                    "distance_stdev": stats.stdev,
                    "most_diverse_gene": leaders.most_diverse_gene,
                    "most_diverse_score": leaders.most_diverse_score,
                    "least_diverse_gene": leaders.least_diverse_gene,
                    "least_diverse_score": leaders.least_diverse_score,
                }
            )


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


# --------------------------------------------------------------------------- #
# Entry Point                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    run_directory = args.run_directory.resolve()
    default_models_root = Path(__file__).resolve().parent / "sota" / "ExquisiteNetV2" / "models"
    models_root = (args.models_root or default_models_root).resolve()

    print(f"[DEBUG] Run directory: {run_directory}")
    print(f"[DEBUG] Models root: {models_root}")

    summaries = summarize_run(
        run_directory=run_directory,
        models_root=models_root,
        include_hist=args.include_hist,
        limit=args.limit,
        max_files=args.max_files,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )

    print_summary_table(summaries)

    if not args.dry_run:
        csv_path = run_directory / args.csv_name
        write_csv(summaries, csv_path)
        print(f"[DEBUG] CSV written to {csv_path}")
    else:
        print("[DEBUG] Dry-run mode enabled; CSV not written.")


if __name__ == "__main__":
    main()
