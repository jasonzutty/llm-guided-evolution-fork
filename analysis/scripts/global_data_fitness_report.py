# global_data_pareto_report.py
# Read-only analysis script that loads GLOBAL_DATA from a pickle file,
# identifies finite-fitness entities, ranks the top 10 by fitness_1,
# computes the Pareto front, and writes a readable Markdown report.

from pathlib import Path
import argparse
import math
import pickle


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_finite_pair(fitness):
    return (
        isinstance(fitness, (tuple, list))
        and len(fitness) >= 2
        and is_number(fitness[0])
        and is_number(fitness[1])
        and math.isfinite(fitness[0])
        and math.isfinite(fitness[1])
    )


def dominates(a, b):
    """
    Assume objective 1 is maximize, objective 2 is minimize.
    a dominates b if:
    - a is at least as good in both objectives
    - and strictly better in at least one
    """
    a1, a2 = a["fitness_1"], a["fitness_2"]
    b1, b2 = b["fitness_1"], b["fitness_2"]

    at_least_as_good = (a1 >= b1) and (a2 <= b2)
    strictly_better = (a1 > b1) or (a2 < b2)

    return at_least_as_good and strictly_better


def pareto_front(rows):
    front = []
    for i, row_i in enumerate(rows):
        dominated = False
        for j, row_j in enumerate(rows):
            if i == j:
                continue
            if dominates(row_j, row_i):
                dominated = True
                break
        if not dominated:
            front.append(row_i)
    return front


def main():
    parser = argparse.ArgumentParser(description="Build a Pareto-style Markdown report from GLOBAL_DATA in a pickle file.")
    parser.add_argument(
        "--input",
        default="analysis/test_data/slurm/global_gen_1.pkl",
        help="Path to pickle file containing GLOBAL_DATA",
    )
    parser.add_argument(
        "--output",
        default="analysis/results/analysis/global_data_pareto_report.md",
        help="Output Markdown report path",
    )
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open("rb") as f:
        data = pickle.load(f)

    global_data = data.get("GLOBAL_DATA", {})

    total_entities = len(global_data)
    completed_entities = 0
    finite_rows = []
    infinite_fitness = 0
    missing_or_malformed = 0

    for entity_id, payload in global_data.items():
        status = payload.get("status")
        fitness = payload.get("fitness")

        if status == "completed":
            completed_entities += 1

        if is_finite_pair(fitness):
            finite_rows.append({
                "entity_id": entity_id,
                "job_id": str(payload.get("job_id", "")),
                "results_job": str(payload.get("results_job", "")),
                "status": str(status or ""),
                "fitness_1": float(fitness[0]),
                "fitness_2": float(fitness[1]),
            })
        elif isinstance(fitness, (tuple, list)) and len(fitness) >= 2:
            infinite_fitness += 1
        else:
            missing_or_malformed += 1

    top_by_f1 = sorted(finite_rows, key=lambda r: r["fitness_1"], reverse=True)[:10]
    front = pareto_front(finite_rows)
    front_sorted = sorted(front, key=lambda r: (-r["fitness_1"], r["fitness_2"]))

    with output_file.open("w") as f:
        f.write("# Global Data Pareto Report\n\n")
        f.write(f"This report summarizes `GLOBAL_DATA` from `{input_file.name}`.\n\n")

        f.write("## Overall Summary\n\n")
        f.write(f"- Total entities: {total_entities}\n")
        f.write(f"- Completed entities: {completed_entities}\n")
        f.write(f"- Finite fitness tuples: {len(finite_rows)}\n")
        f.write(f"- Infinite fitness tuples: {infinite_fitness}\n")
        f.write(f"- Missing or malformed fitness: {missing_or_malformed}\n")
        f.write(f"- Pareto front size: {len(front_sorted)}\n\n")

        f.write("## Top 10 Individuals by fitness_1\n\n")
        if top_by_f1:
            for i, row in enumerate(top_by_f1, start=1):
                f.write(
                    f"{i}. entity_id: `{row['entity_id']}`, "
                    f"job_id: `{row['job_id']}`, "
                    f"results_job: `{row['results_job']}`, "
                    f"status: `{row['status']}`, "
                    f"fitness_1: `{row['fitness_1']}`, "
                    f"fitness_2: `{row['fitness_2']}`\n"
                )
        else:
            f.write("No finite fitness rows found.\n")
        f.write("\n")

        f.write("## Pareto Front Individuals\n\n")
        if front_sorted:
            for i, row in enumerate(front_sorted, start=1):
                f.write(
                    f"{i}. entity_id: `{row['entity_id']}`, "
                    f"job_id: `{row['job_id']}`, "
                    f"results_job: `{row['results_job']}`, "
                    f"status: `{row['status']}`, "
                    f"fitness_1: `{row['fitness_1']}`, "
                    f"fitness_2: `{row['fitness_2']}`\n"
                )
        else:
            f.write("No Pareto front could be computed.\n")
        f.write("\n")

        f.write("## Interpretation\n\n")
        f.write("- Top 10 by fitness_1 shows the strongest individuals on the first objective alone.\n")
        f.write("- Pareto front individuals are the non-dominated tradeoff points when maximizing fitness_1 and minimizing fitness_2.\n")
        f.write("- Only finite-fitness entities are included in the Pareto analysis.\n")

    print(f"Wrote Pareto report to {output_file}")


if __name__ == "__main__":
    main()