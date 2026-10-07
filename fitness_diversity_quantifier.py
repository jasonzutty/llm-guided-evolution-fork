#!/usr/bin/env python3
"""
Compute fitness diversity statistics for each generation of an LLM-guided
evolution run. The script scans the `global_data/global_gen_*.pkl` files that
are produced by runs launched via `pace_ice_island_controller.sbatch`, extracts
fitness tuples, and emits per-generation statistics along with a CSV summary.

Example:
    python fitness_diversity_quantifier.py qwen_focus_3 --include-hist
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable, List, Sequence, Tuple

from deap import base, creator
from deap.tools import emo

# Default to the project's standard fitness weights. We avoid importing the
# project module directly to keep this script self-contained and to prevent
# environments without those dependencies from failing during import.
FITNESS_WEIGHTS = (1.0, -1.0)


@dataclass
class ObjectiveStats:
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    stdev: float | None = None


@dataclass
class CrowdingStats:
    count: int = 0
    infinite_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    stdev: float | None = None


@dataclass
class GenerationSummary:
    generation: int
    file_path: Path
    total_entries: int
    valid_entries: int
    invalid_entries: int
    objectives: List[ObjectiveStats]
    crowding: CrowdingStats
    pareto_front_area: float | None = None
    dominated_pareto_area: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify fitness diversity per generation.")
    parser.add_argument(
        "run_directory",
        type=Path,
        help="Path to the run folder produced by pace_ice_island_controller (contains global_data/).",
    )
    parser.add_argument(
        "--include-hist",
        action="store_true",
        help="Include entries from GLOBAL_DATA_HIST alongside GLOBAL_DATA.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of generations to process (processed in ascending order).",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="fitness_diversity_summary.csv",
        help="Name of the CSV file to write inside the run directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and print results without writing the CSV file.",
    )
    return parser.parse_args()


def extract_generation(filename: str) -> int:
    """Pull the generation number from a name like global_gen_12.pkl."""
    try:
        return int(filename.split("_")[2].split(".")[0])
    except (IndexError, ValueError):
        raise ValueError(f"Unable to extract generation from filename: {filename}")


def load_pickle(path: Path) -> dict:
    print(f"[DEBUG] Loading pickle: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def iter_fitness_values(
    dataset: dict | None,
) -> Iterable[Tuple[float, ...]]:
    if not dataset:
        return []

    valid_values: List[Tuple[float, ...]] = []
    for gene_id, attributes in dataset.items():
        fitness = attributes.get("fitness")
        if is_valid_fitness_tuple(fitness):
            valid_values.append(tuple(float(x) for x in fitness))
        else:
            print(f"[DEBUG] Skipping invalid fitness for gene {gene_id}: {fitness}")
    return valid_values


def is_valid_fitness_tuple(candidate: object) -> bool:
    if not isinstance(candidate, (tuple, list)):
        return False
    if not candidate:
        return False
    for value in candidate:
        if value is None:
            return False
        if not isinstance(value, (int, float)):
            return False
        if not math.isfinite(value):
            return False
    return True


def compute_objective_stats(values: Sequence[Tuple[float, ...]]) -> List[ObjectiveStats]:
    if not values:
        return []

    num_objectives = len(values[0])
    stats_per_objective: List[ObjectiveStats] = []
    for idx in range(num_objectives):
        column = [row[idx] for row in values]
        stats = ObjectiveStats(
            minimum=min(column) if column else None,
            maximum=max(column) if column else None,
            mean=safe_statistic(mean, column),
            median=safe_statistic(median, column),
            stdev=safe_statistic(stdev, column),
        )
        stats_per_objective.append(stats)
        print(
            "[DEBUG] Objective %d stats -> min: %s max: %s mean: %s median: %s stdev: %s"
            % (
                idx,
                stats.minimum,
                stats.maximum,
                stats.mean,
                stats.median,
                stats.stdev,
            )
        )
    return stats_per_objective


def safe_statistic(func, values: Sequence[float]) -> float | None:
    if not values:
        return None
    if func is stdev and len(values) < 2:
        return 0.0
    try:
        return func(values)
    except Exception as exc:  # Catch rare statistics errors.
        print(f"[DEBUG] Statistic {func.__name__} failed on {values}: {exc}")
        return None


def objective_is_maximized(index: int) -> bool:
    if index < len(FITNESS_WEIGHTS):
        return FITNESS_WEIGHTS[index] >= 0
    return True


def dominates(
    contender: Sequence[float],
    incumbent: Sequence[float],
    maximize_flags: Sequence[bool],
) -> bool:
    better_in_any = False
    for idx, contender_value in enumerate(contender):
        incumbent_value = incumbent[idx]
        maximize = maximize_flags[idx]
        if maximize:
            if contender_value < incumbent_value:
                return False
            if contender_value > incumbent_value:
                better_in_any = True
        else:
            if contender_value > incumbent_value:
                return False
            if contender_value < incumbent_value:
                better_in_any = True
    return better_in_any


def extract_pareto_optimal(values: Sequence[Tuple[float, ...]]) -> List[Tuple[float, ...]]:
    if not values:
        return []
    num_objectives = len(values[0])
    maximize_flags = [objective_is_maximized(idx) for idx in range(num_objectives)]
    pareto_front: List[Tuple[float, ...]] = []
    for candidate in values:
        if len(candidate) != num_objectives:
            continue
        dominated = False
        for other in values:
            if other is candidate:
                continue
            if len(other) != num_objectives:
                continue
            if dominates(other, candidate, maximize_flags):
                dominated = True
                break
        if not dominated:
            pareto_front.append(candidate)
    return pareto_front


def get_clamped_pareto_front(
    values: Sequence[Tuple[float, ...]],
    max_accuracy: float = 1.0,
) -> Tuple[List[Tuple[float, float]], bool, bool]:
    pareto_candidates = extract_pareto_optimal(values)
    pareto_front = compute_two_objective_pareto_front(pareto_candidates)
    maximize_accuracy = not FITNESS_WEIGHTS or FITNESS_WEIGHTS[0] >= 0
    minimize_parameters = len(FITNESS_WEIGHTS) < 2 or FITNESS_WEIGHTS[1] <= 0

    if not pareto_front:
        return [], maximize_accuracy, minimize_parameters

    clamped_front: List[Tuple[float, float]] = []
    for accuracy, parameters in pareto_front:
        if not (math.isfinite(accuracy) and math.isfinite(parameters)):
            continue
        clamped_accuracy = max(0.0, min(max_accuracy, accuracy))
        clamped_front.append((clamped_accuracy, parameters))

    if not clamped_front:
        return [], maximize_accuracy, minimize_parameters

    deduped_front: List[Tuple[float, float]] = []
    for accuracy, parameters in clamped_front:
        if not deduped_front:
            deduped_front.append((accuracy, parameters))
            continue
        prev_accuracy, prev_parameters = deduped_front[-1]
        if math.isclose(accuracy, prev_accuracy):
            better = parameters < prev_parameters if minimize_parameters else parameters > prev_parameters
            if better:
                deduped_front[-1] = (accuracy, parameters)
        else:
            deduped_front.append((accuracy, parameters))

    return deduped_front, maximize_accuracy, minimize_parameters


def compute_two_objective_pareto_front(values: Sequence[Tuple[float, ...]]) -> List[Tuple[float, float]]:
    """
    Compute the Pareto front using the first two objectives (accuracy, parameters).
    The first objective is assumed to be maximized when its weight is non-negative,
    while the second is assumed to be minimized when its weight is non-positive.
    """
    if not values:
        return []

    points: List[Tuple[float, float]] = []
    for candidate in values:
        if len(candidate) < 2:
            continue
        accuracy = float(candidate[0])
        parameters = float(candidate[1])
        if not (math.isfinite(accuracy) and math.isfinite(parameters)):
            continue
        points.append((accuracy, parameters))

    if not points:
        return []

    maximize_accuracy = not FITNESS_WEIGHTS or FITNESS_WEIGHTS[0] >= 0
    minimize_parameters = len(FITNESS_WEIGHTS) < 2 or FITNESS_WEIGHTS[1] <= 0

    sorted_points = sorted(
        points,
        key=lambda pt: (
            -pt[0] if maximize_accuracy else pt[0],
            pt[1] if minimize_parameters else -pt[1],
        ),
    )

    pareto_front: List[Tuple[float, float]] = []
    if minimize_parameters:
        best_param = math.inf
        for accuracy, parameters in sorted_points:
            if parameters < best_param:
                pareto_front.append((accuracy, parameters))
                best_param = parameters
    else:
        best_param = -math.inf
        for accuracy, parameters in sorted_points:
            if parameters > best_param:
                pareto_front.append((accuracy, parameters))
                best_param = parameters

    pareto_front.sort(key=lambda pt: pt[0], reverse=not maximize_accuracy)
    return pareto_front


def compute_pareto_front_area(
    values: Sequence[Tuple[float, ...]],
    max_accuracy: float = 1.0,
) -> float | None:
    """
    Integrate the Pareto front (accuracy vs. parameters) using the trapezoidal rule.
    Returns None when fewer than one valid point exists, or 0.0 for a single point.
    """
    deduped_front, maximize_accuracy, _ = get_clamped_pareto_front(values, max_accuracy)
    if not deduped_front:
        return None

    if maximize_accuracy and deduped_front[-1][0] < max_accuracy:
        deduped_front.append((max_accuracy, deduped_front[-1][1]))

    if len(deduped_front) < 2:
        return 0.0

    area = 0.0
    prev_accuracy, prev_parameters = deduped_front[0]
    for accuracy, parameters in deduped_front[1:]:
        width = accuracy - prev_accuracy
        area += abs(width) * (prev_parameters + parameters) / 2.0
        prev_accuracy, prev_parameters = accuracy, parameters
    return area


def compute_dominated_pareto_area(
    values: Sequence[Tuple[float, ...]],
    max_accuracy: float = 1.0,
) -> float | None:
    deduped_front, maximize_accuracy, minimize_parameters = get_clamped_pareto_front(values, max_accuracy)
    if not deduped_front:
        return None

    if minimize_parameters:
        reference_parameters = max(parameters for _, parameters in deduped_front)
    else:
        reference_parameters = min(parameters for _, parameters in deduped_front)

    best_parameters = reference_parameters
    dominated_area = 0.0

    if maximize_accuracy:
        prev_accuracy = max_accuracy
        for accuracy, parameters in deduped_front:
            width = prev_accuracy - accuracy
            if width > 0:
                height = reference_parameters - best_parameters if minimize_parameters else best_parameters - reference_parameters
                dominated_area += max(0.0, height) * width
            if minimize_parameters:
                best_parameters = min(best_parameters, parameters)
            else:
                best_parameters = max(best_parameters, parameters)
            prev_accuracy = accuracy
        width = prev_accuracy - 0.0
        if width > 0:
            height = reference_parameters - best_parameters if minimize_parameters else best_parameters - reference_parameters
            dominated_area += max(0.0, height) * width
    else:
        prev_accuracy = 0.0
        for accuracy, parameters in deduped_front:
            width = accuracy - prev_accuracy
            if width > 0:
                height = reference_parameters - best_parameters if minimize_parameters else best_parameters - reference_parameters
                dominated_area += max(0.0, height) * width
            if minimize_parameters:
                best_parameters = min(best_parameters, parameters)
            else:
                best_parameters = max(best_parameters, parameters)
            prev_accuracy = accuracy
        width = max_accuracy - prev_accuracy
        if width > 0:
            height = reference_parameters - best_parameters if minimize_parameters else best_parameters - reference_parameters
            dominated_area += max(0.0, height) * width

    return dominated_area


def ensure_creator_classes() -> None:
    # Avoid recreating classes if this module is executed multiple times.
    if not hasattr(creator, "FitnessDiversity"):
        creator.create("FitnessDiversity", base.Fitness, weights=FITNESS_WEIGHTS)
    if not hasattr(creator, "IndividualDiversity"):
        creator.create("IndividualDiversity", list, fitness=creator.FitnessDiversity)


def compute_crowding_stats(values: Sequence[Tuple[float, ...]]) -> CrowdingStats:
    if len(values) < 2:
        return CrowdingStats(count=len(values))

    ensure_creator_classes()

    individuals: List[object] = []
    for fitness_values in values:
        fitness = creator.FitnessDiversity(fitness_values)
        individual = creator.IndividualDiversity([])
        individual.fitness = fitness
        individuals.append(individual)

    emo.assignCrowdingDist(individuals)
    distances = [ind.fitness.crowding_dist for ind in individuals]
    finite_distances = [dist for dist in distances if math.isfinite(dist)]
    inf_count = len(distances) - len(finite_distances)

    stats = CrowdingStats(
        count=len(distances),
        infinite_count=inf_count,
        minimum=min(finite_distances) if finite_distances else None,
        maximum=max(finite_distances) if finite_distances else None,
        mean=safe_statistic(mean, finite_distances),
        median=safe_statistic(median, finite_distances),
        stdev=safe_statistic(stdev, finite_distances),
    )

    print(
        "[DEBUG] Crowding stats -> count: %d finite: %d infinite: %d mean: %s median: %s"
        % (
            stats.count,
            len(finite_distances),
            stats.infinite_count,
            stats.mean,
            stats.median,
        )
    )
    return stats


def summarize_generation(
    generation: int,
    file_path: Path,
    data: dict,
    include_hist: bool,
) -> GenerationSummary:
    global_data = data.get("GLOBAL_DATA", {})
    hist_data = data.get("GLOBAL_DATA_HIST", {}) if include_hist else {}

    global_values = list(iter_fitness_values(global_data))
    hist_values = list(iter_fitness_values(hist_data))
    combined_values = global_values + hist_values

    total_entries = len(global_data) + (len(hist_data) if include_hist else 0)
    valid_entries = len(combined_values)
    invalid_entries = total_entries - valid_entries

    print(
        "[DEBUG] Generation %d -> total entries: %d valid: %d invalid: %d"
        % (generation, total_entries, valid_entries, invalid_entries)
    )

    objective_stats = compute_objective_stats(combined_values)
    crowding_stats = compute_crowding_stats(combined_values)
    pareto_area = compute_pareto_front_area(combined_values)
    if pareto_area is not None:
        print(f"[DEBUG] Generation {generation} Pareto front area: {pareto_area}")
    dominated_area = compute_dominated_pareto_area(combined_values)
    if dominated_area is not None:
        print(f"[DEBUG] Generation {generation} dominated Pareto area: {dominated_area}")
    return GenerationSummary(
        generation=generation,
        file_path=file_path,
        total_entries=total_entries,
        valid_entries=valid_entries,
        invalid_entries=invalid_entries,
        objectives=objective_stats,
        crowding=crowding_stats,
        pareto_front_area=pareto_area,
        dominated_pareto_area=dominated_area,
    )


def summarize_run(run_directory: Path, include_hist: bool, limit: int | None) -> List[GenerationSummary]:
    global_data_dir = run_directory / "global_data"
    if not global_data_dir.is_dir():
        raise FileNotFoundError(f"No global_data/ directory found at {global_data_dir}")

    generation_files = sorted(global_data_dir.glob("global_gen_*.pkl"), key=lambda path: extract_generation(path.name))
    if limit is not None:
        generation_files = generation_files[:limit]

    print(f"[DEBUG] Found {len(generation_files)} generation files (limit={limit}).")

    summaries: List[GenerationSummary] = []
    for file_path in generation_files:
        generation_id = extract_generation(file_path.name)
        data = load_pickle(file_path)
        summary = summarize_generation(generation_id, file_path, data, include_hist)
        summaries.append(summary)
    return summaries


def print_summary_table(summaries: Sequence[GenerationSummary]) -> None:
    if not summaries:
        print("[DEBUG] No generation summaries to display.")
        return

    # Determine the maximum number of objectives across generations.
    max_objectives = max((len(summary.objectives) for summary in summaries), default=0)

    header_cells = [
        "Gen",
        "Total",
        "Valid",
        "Invalid",
        "Pareto_Area",
        "Dom_Pareto_Area",
    ]
    for idx in range(max_objectives):
        header_cells.extend(
            [
                f"Obj{idx}_Min",
                f"Obj{idx}_Max",
                f"Obj{idx}_Mean",
                f"Obj{idx}_Median",
                f"Obj{idx}_Std",
            ]
        )
    header_cells.extend(
        [
            "Crowd_Finite",
            "Crowd_Inf",
            "Crowd_Min",
            "Crowd_Max",
            "Crowd_Mean",
            "Crowd_Median",
            "Crowd_Std",
        ]
    )

    print("[DEBUG] Summary Table:")
    print("\t".join(header_cells))

    for summary in summaries:
        row: List[str] = [
            str(summary.generation),
            str(summary.total_entries),
            str(summary.valid_entries),
            str(summary.invalid_entries),
            format_float(summary.pareto_front_area),
            format_float(summary.dominated_pareto_area),
        ]
        for idx in range(max_objectives):
            stats = summary.objectives[idx] if idx < len(summary.objectives) else ObjectiveStats()
            row.extend(
                [
                    format_float(stats.minimum),
                    format_float(stats.maximum),
                    format_float(stats.mean),
                    format_float(stats.median),
                    format_float(stats.stdev),
                ]
            )
        row.extend(
            [
                str(summary.crowding.count - summary.crowding.infinite_count),
                str(summary.crowding.infinite_count),
                format_float(summary.crowding.minimum),
                format_float(summary.crowding.maximum),
                format_float(summary.crowding.mean),
                format_float(summary.crowding.median),
                format_float(summary.crowding.stdev),
            ]
        )
        print("\t".join(row))


def write_csv(summaries: Sequence[GenerationSummary], destination: Path) -> None:
    if not summaries:
        print(f"[DEBUG] No summaries available; skipping CSV write to {destination}.")
        return

    max_objectives = max((len(summary.objectives) for summary in summaries), default=0)
    fieldnames = [
        "generation",
        "file_path",
        "total_entries",
        "valid_entries",
        "invalid_entries",
        "pareto_front_area",
        "dominated_pareto_area",
    ]
    for idx in range(max_objectives):
        fieldnames.extend(
            [
                f"objective_{idx}_min",
                f"objective_{idx}_max",
                f"objective_{idx}_mean",
                f"objective_{idx}_median",
                f"objective_{idx}_stdev",
            ]
        )
    fieldnames.extend(
        [
            "crowding_total",
            "crowding_infinite",
            "crowding_min",
            "crowding_max",
            "crowding_mean",
            "crowding_median",
            "crowding_stdev",
        ]
    )

    print(f"[DEBUG] Writing CSV to: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = {
                "generation": summary.generation,
                "file_path": summary.file_path.name,
                "total_entries": summary.total_entries,
                "valid_entries": summary.valid_entries,
                "invalid_entries": summary.invalid_entries,
                "pareto_front_area": summary.pareto_front_area,
                "dominated_pareto_area": summary.dominated_pareto_area,
                "crowding_total": summary.crowding.count,
                "crowding_infinite": summary.crowding.infinite_count,
                "crowding_min": summary.crowding.minimum,
                "crowding_max": summary.crowding.maximum,
                "crowding_mean": summary.crowding.mean,
                "crowding_median": summary.crowding.median,
                "crowding_stdev": summary.crowding.stdev,
            }
            for idx in range(max_objectives):
                stats = summary.objectives[idx] if idx < len(summary.objectives) else ObjectiveStats()
                row[f"objective_{idx}_min"] = stats.minimum
                row[f"objective_{idx}_max"] = stats.maximum
                row[f"objective_{idx}_mean"] = stats.mean
                row[f"objective_{idx}_median"] = stats.median
                row[f"objective_{idx}_stdev"] = stats.stdev
            writer.writerow(row)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def main() -> None:
    args = parse_args()
    run_directory = args.run_directory.resolve()
    print(f"[DEBUG] Run directory resolved to: {run_directory}")

    summaries = summarize_run(run_directory, include_hist=args.include_hist, limit=args.limit)
    print_summary_table(summaries)

    if not args.dry_run:
        csv_path = run_directory / args.csv_name
        write_csv(summaries, csv_path)
        print(f"[DEBUG] CSV written to {csv_path}")
    else:
        print("[DEBUG] Dry-run mode enabled; CSV not written.")


if __name__ == "__main__":
    main()
