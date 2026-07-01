from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_DIR = Path("analysis/results/analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static HTML analysis dashboard.")
    parser.add_argument("--input-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output", default=str(DEFAULT_RESULTS_DIR / "dashboard.html"))
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def rel_path(path: Path, output_path: Path) -> str:
    try:
        return path.relative_to(output_path.parent).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def table_html(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return '<p class="muted">No data available.</p>'
    shown = df.head(max_rows)
    suffix = ""
    if len(df) > max_rows:
        suffix = f'<p class="muted">Showing first {max_rows} of {len(df)} rows.</p>'
    return shown.to_html(index=False, escape=True, classes="data-table") + suffix


def metric_card(label: str, value: object) -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{html.escape(label)}</div>
      <div class="metric-value">{html.escape(str(value))}</div>
    </div>
    """


def image_block(path: Path, output_path: Path, caption: str) -> str:
    if not path.exists():
        return f'<div class="image-missing">{html.escape(caption)} not generated.</div>'
    src = html.escape(rel_path(path, output_path))
    return f"""
    <figure>
      <img src="{src}" alt="{html.escape(caption)}">
      <figcaption>{html.escape(caption)}</figcaption>
    </figure>
    """


def report_block(path: Path) -> str:
    text = read_text(path)
    if not text:
        return f'<p class="muted">Report not found: {html.escape(path.name)}</p>'
    return f"""
    <details open>
      <summary>{html.escape(path.name)}</summary>
      <pre>{html.escape(text)}</pre>
    </details>
    """


def build_dashboard(input_dir: Path, output_path: Path) -> str:
    slurm_summary = read_csv(input_dir / "slurm_summary.csv")
    failure_counts = read_csv(input_dir / "failure_counts.csv")
    entity_summary = read_csv(input_dir / "entity_metrics_summary.csv")
    run_comparison = read_csv(input_dir / "run_comparison.csv")
    run_inventory = read_csv(input_dir / "run_inventory.csv")
    run_metrics = read_csv(input_dir / "run_metrics.csv")
    leaderboard = read_csv(input_dir / "top_10_leaderboard.csv")

    total_runs = len(slurm_summary)
    failed_runs = 0
    completed_runs = 0
    if not slurm_summary.empty and "status_guess" in slurm_summary.columns:
        failed_runs = int((slurm_summary["status_guess"] == "failed").sum())
        completed_runs = int(
            slurm_summary["status_guess"].isin(["completed", "completed_or_unknown"]).sum()
        )

    finite_fitness = 0
    infinite_fitness = 0
    if not entity_summary.empty and {"fitness_1", "fitness_2"}.issubset(entity_summary.columns):
        f1 = pd.to_numeric(entity_summary["fitness_1"], errors="coerce")
        f2 = pd.to_numeric(entity_summary["fitness_2"], errors="coerce")
        finite_fitness = int((f1.notna() & f2.notna()).sum())
        infinite_fitness = int(
            entity_summary["fitness_1"].astype(str).isin(["inf", "-inf"]).sum()
            + entity_summary["fitness_2"].astype(str).isin(["inf", "-inf"]).sum()
        )

    images = [
        (input_dir / "controller_activity_summary.png", "Controller Activity Summary"),
        (input_dir / "entity_status_summary.png", "Entity Status Summary"),
        (input_dir / "fitness_quality_summary.png", "Fitness Quality Summary"),
        (input_dir / "generation_timeline_graphs" / "generation_completed_count_bar.png", "Generation Completed Count"),
        (input_dir / "generation_timeline_graphs" / "generation_cumulative_timeline.png", "Generation Cumulative Timeline"),
        (input_dir / "generation_timeline_graphs" / "generation_duration_proxy.png", "Generation Duration Proxy"),
        (input_dir / "pareto_kpi_graphs" / "pareto_kpi_bar.png", "Pareto KPI Summary"),
    ]

    reports = [
        input_dir / "failure_report.md",
        input_dir / "entity_metrics_report.md",
        input_dir / "global_data_pareto_report.md",
        input_dir / "pareto_kpi_summary.md",
        input_dir / "generation_timeline_report.md",
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM-GE Analysis Dashboard</title>
  <style>
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
      background: #f7f8fa;
    }}
    header {{
      padding: 28px 36px;
      background: #101820;
      color: white;
    }}
    h1, h2, h3 {{ margin-top: 0; }}
    main {{ padding: 28px 36px 56px; }}
    section {{
      margin-bottom: 30px;
      padding: 22px;
      background: white;
      border: 1px solid #d9dee5;
      border-radius: 8px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 14px;
    }}
    .metric-card {{
      border: 1px solid #d9dee5;
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfd;
    }}
    .metric-label {{ font-size: 13px; color: #667085; }}
    .metric-value {{ margin-top: 6px; font-size: 28px; font-weight: 700; }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    figure {{ margin: 0; }}
    img {{
      width: 100%;
      height: auto;
      border: 1px solid #d9dee5;
      border-radius: 6px;
      background: white;
    }}
    figcaption {{ margin-top: 8px; font-size: 13px; color: #667085; }}
    .image-missing {{
      padding: 18px;
      border: 1px dashed #b8c0cc;
      border-radius: 6px;
      color: #667085;
    }}
    .table-wrap {{ overflow-x: auto; }}
    table.data-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    table.data-table th, table.data-table td {{
      border: 1px solid #d9dee5;
      padding: 7px 9px;
      text-align: left;
      vertical-align: top;
    }}
    table.data-table th {{ background: #eef2f6; }}
    details {{
      border: 1px solid #d9dee5;
      border-radius: 6px;
      padding: 12px;
      margin-bottom: 12px;
      background: #fbfcfd;
    }}
    summary {{ cursor: pointer; font-weight: 700; }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      line-height: 1.45;
    }}
    .muted {{ color: #667085; }}
  </style>
</head>
<body>
  <header>
    <h1>LLM-Guided Evolution Analysis Dashboard</h1>
    <p>Static dashboard generated from <code>{html.escape(str(input_dir))}</code>.</p>
  </header>
  <main>
    <section>
      <h2>Run Health</h2>
      <div class="metrics">
        {metric_card("Total Runs", total_runs)}
        {metric_card("Failed Runs", failed_runs)}
        {metric_card("Completed Runs", completed_runs)}
        {metric_card("Entities", len(entity_summary))}
        {metric_card("Finite Fitness", finite_fitness)}
        {metric_card("Infinite Fitness", infinite_fitness)}
      </div>
    </section>

    <section>
      <h2>Generated Visuals</h2>
      <div class="image-grid">
        {''.join(image_block(path, output_path, caption) for path, caption in images)}
      </div>
    </section>

    <section>
      <h2>Reports</h2>
      {''.join(report_block(path) for path in reports)}
    </section>

    <section>
      <h2>Leaderboard</h2>
      <div class="table-wrap">{table_html(leaderboard)}</div>
    </section>

    <section>
      <h2>Slurm Summary</h2>
      <div class="table-wrap">{table_html(slurm_summary)}</div>
    </section>

    <section>
      <h2>Failure Counts</h2>
      <div class="table-wrap">{table_html(failure_counts)}</div>
    </section>

    <section>
      <h2>Run Comparison</h2>
      <div class="table-wrap">{table_html(run_comparison)}</div>
    </section>

    <section>
      <h2>Run Inventory</h2>
      <div class="table-wrap">{table_html(run_inventory)}</div>
    </section>

    <section>
      <h2>Run Metrics</h2>
      <div class="table-wrap">{table_html(run_metrics)}</div>
    </section>

    <section>
      <h2>Entity Metrics Summary</h2>
      <div class="table-wrap">{table_html(entity_summary)}</div>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_dashboard(input_dir, output_path))
    print(f"Wrote static dashboard: {output_path}")


if __name__ == "__main__":
    main()
