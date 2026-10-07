"""
Creates a Pareto KPI summary card from the existing Pareto report.
What it shows:
how many non-dominated solutions the generation produced

"""

from pathlib import Path
import re
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1] / "results" / "analysis"
INPUT_REPORT = BASE_DIR / "global_data_pareto_report.md"
OUTPUT_REPORT = BASE_DIR / "pareto_kpi_summary.md"
GRAPHS_DIR = BASE_DIR / "pareto_kpi_graphs"
OUTPUT_GRAPH = GRAPHS_DIR / "pareto_kpi_bar.png"


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Could not find input Pareto report: {path}")
    return path.read_text(errors="ignore")


def extract_int_metric(text: str, label: str) -> int:
    """
    Extracts Markdown lines like:
        - Total entities: 93
    """
    pattern = rf"[-*]\s*{re.escape(label)}\s*:\s*(\d+)"
    match = re.search(pattern, text, re.IGNORECASE)

    if not match:
        raise ValueError(f"Could not find metric '{label}' in Pareto report.")

    return int(match.group(1))


def extract_pareto_metrics(text: str) -> dict:
    return {
        "Total entities": extract_int_metric(text, "Total entities"),
        "Completed entities": extract_int_metric(text, "Completed entities"),
        "Finite fitness tuples": extract_int_metric(text, "Finite fitness tuples"),
        "Infinite fitness tuples": extract_int_metric(text, "Infinite fitness tuples"),
        "Missing or malformed fitness": extract_int_metric(text, "Missing or malformed fitness"),
        "Pareto front size": extract_int_metric(text, "Pareto front size"),
    }


def safe_percent(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return (part / whole) * 100.0


def build_interpretation(metrics: dict) -> list[str]:
    total = metrics["Total entities"]
    completed = metrics["Completed entities"]
    finite = metrics["Finite fitness tuples"]
    pareto = metrics["Pareto front size"]

    completed_rate = safe_percent(completed, total)
    finite_rate = safe_percent(finite, total)
    pareto_total_rate = safe_percent(pareto, total)
    pareto_finite_rate = safe_percent(pareto, finite)

    lines = [
        f"- {completed_rate:.1f}% of all entities reached completed status.",
        f"- {finite_rate:.1f}% of all entities had finite fitness values and were usable for Pareto analysis.",
        f"- {pareto_total_rate:.1f}% of all entities were non-dominated Pareto-front solutions.",
        f"- Among finite-fitness entities, {pareto_finite_rate:.1f}% were on the Pareto front.",
    ]

    if pareto == 0:
        lines.append(
            "- No Pareto-front solutions were found, which suggests the finite-fitness pool was empty or malformed."
        )
    elif pareto_finite_rate >= 40:
        lines.append(
            "- A large share of finite entities are non-dominated, meaning the generation produced many useful tradeoff candidates."
        )
    elif pareto_finite_rate >= 15:
        lines.append(
            "- The Pareto front has a moderate size, suggesting several useful tradeoff candidates were produced."
        )
    else:
        lines.append(
            "- The Pareto front is relatively small, meaning only a few entities were competitive tradeoff solutions."
        )

    return lines


def make_pareto_kpi_graph(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [
        "Total entities",
        "Finite fitness",
        "Pareto front",
    ]

    values = [
        metrics["Total entities"],
        metrics["Finite fitness tuples"],
        metrics["Pareto front size"],
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title("Pareto KPI Summary")
    plt.ylabel("Count")

    for index, value in enumerate(values):
        plt.text(index, value, str(value), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_markdown_report(metrics: dict, graph_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = metrics["Total entities"]
    finite = metrics["Finite fitness tuples"]
    pareto = metrics["Pareto front size"]

    finite_rate = safe_percent(finite, total)
    pareto_total_rate = safe_percent(pareto, total)
    pareto_finite_rate = safe_percent(pareto, finite)

    graph_relative_path = graph_path.relative_to(output_path.parent).as_posix()

    with output_path.open("w") as f:
        f.write("# Pareto KPI Summary\n\n")
        f.write("This summary converts the Pareto report into a small dashboard-style KPI card.\n\n")

        f.write("## KPI Card\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---:|\n")
        f.write(f"| Total entities | {metrics['Total entities']} |\n")
        f.write(f"| Completed entities | {metrics['Completed entities']} |\n")
        f.write(f"| Finite fitness entities | {metrics['Finite fitness tuples']} |\n")
        f.write(f"| Infinite fitness entities | {metrics['Infinite fitness tuples']} |\n")
        f.write(f"| Missing or malformed fitness | {metrics['Missing or malformed fitness']} |\n")
        f.write(f"| Pareto front size | {metrics['Pareto front size']} |\n")
        f.write(f"| Finite fitness rate | {finite_rate:.1f}% |\n")
        f.write(f"| Pareto front as % of total entities | {pareto_total_rate:.1f}% |\n")
        f.write(f"| Pareto front as % of finite entities | {pareto_finite_rate:.1f}% |\n\n")

        f.write("## Visualization\n\n")
        f.write(f"![Pareto KPI Summary]({graph_relative_path})\n\n")

        f.write("## Interpretation\n\n")
        for line in build_interpretation(metrics):
            f.write(line + "\n")

        f.write("\n## What This Shows\n\n")
        f.write(
            "The Pareto front size shows how many non-dominated solutions the generation produced. "
            "These are the candidates that represent the best tradeoffs between the optimization objectives.\n"
        )


def main():
    print(f"[INFO] Reading Pareto report: {INPUT_REPORT}")
    print(f"[INFO] Writing KPI report: {OUTPUT_REPORT}")
    print(f"[INFO] Writing graph: {OUTPUT_GRAPH}")

    text = read_text(INPUT_REPORT)
    metrics = extract_pareto_metrics(text)

    make_pareto_kpi_graph(metrics, OUTPUT_GRAPH)
    write_markdown_report(metrics, OUTPUT_GRAPH, OUTPUT_REPORT)

    print("[SUCCESS] Pareto KPI summary generated.")
    print(f"[SUCCESS] Report: {OUTPUT_REPORT}")
    print(f"[SUCCESS] Graph: {OUTPUT_GRAPH}")


if __name__ == "__main__":
    main()
